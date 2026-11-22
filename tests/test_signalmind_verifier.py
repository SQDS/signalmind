"""Deterministic ClaimVerifier: each required deficiency, and the positive cases that must pass."""

import pytest

from tsmcp.runtime.claims import AnswerBundle
from tsmcp.runtime.ledger import EvidenceLedger, SourceKind
from tsmcp.runtime.taskspec import Budgets, QuantitySpec, TaskSpec, abp_interval_spec
from tsmcp.runtime.timebase import pair_key
from tsmcp.runtime.verifier import CODES, decimals_of, verify

SBP = "abp.sbp.beat_mean.v1"


@pytest.fixture
def led():
    return EvidenceLedger("r")


def spec(quantities=("sbp_mean",), **kw):
    return abp_interval_spec(
        "t", "mean SBP over 10-40 s", "ABP", (10, 40), quantities=quantities, ecg_channel="ECG", **kw
    )


def computed(led, **over):
    base = dict(
        tool="ts_measure", tool_call_id="call1", value=118.0, unit="mmHg", definition_id=SBP,
        channels_effective=("ABP",), interval_requested=(10, 40), intervals_effective=[(10, 40)],
        counts={"accepted": 35, "rejected": 2}, resolution={"known": True, "value": 0.05, "kind": "sem"},
        timebase={"sync_status": "unknown"},
    )  # fmt: skip
    base.update(over)
    return led.record(SourceKind.COMPUTED, base.pop("quantity", "sbp_mean"), **base)


def claim(**over):
    base = {"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": 118.0,
            "unit": "mmHg", "interval": [10, 40], "evidence_refs": ["ev1"], "definition_id": SBP}  # fmt: skip
    base.update(over)
    return base


def run(led, s, *claims, limitations=(), final=False):
    return verify(
        AnswerBundle.from_dict({"claims": list(claims), "limitations": list(limitations), "final": final}), led, s
    )


def codes(report):
    return [d.code for d in report.deficiencies]


def test_valid_full_window_measurement_is_accepted(led):
    computed(led)
    r = run(led, spec(), claim())
    assert r.ok and r.claims["c1"]["status"] == "validated_measurement" and r.claims["c1"]["evidence"] == ["ev1"]


def test_wrong_window_is_scope_mismatch(led):
    computed(led, interval_requested=(10, 20), intervals_effective=[(10, 20)])  # a 10 s measurement
    assert codes(run(led, spec(), claim())) == ["scope_mismatch"]  # presented as the 30 s mean
    computed(led, tool_call_id="call2")  # ev2 over 10-40
    assert codes(run(led, spec(), claim(evidence_refs=["ev2"], interval=[0, 60]))) == ["scope_mismatch"]
    assert run(led, spec(), claim(evidence_refs=["ev2"])).ok


def test_fabricated_or_mismatched_reference(led):
    computed(led)
    assert codes(run(led, spec(), claim(evidence_refs=["ev99"]))) == ["invalid_reference"]
    assert codes(run(led, spec(), claim(evidence_refs=[]))) == ["inadequate_support"]
    computed(led, tool_call_id="call2", quantity="amplitude_pp", definition_id="generic.amplitude_pp.v1", value=52.0)
    assert codes(run(led, spec(), claim(evidence_refs=["ev2"]))) == ["method_mismatch"]  # whole-window max is not SBP


def test_visual_estimate_cannot_be_exact(led):
    led.record(SourceKind.VISUAL_OBSERVATION, "view", tool="ts_zoom", tool_call_id="call1", definition_id="view.v1",
               channels_effective=("ABP",), interval_requested=(10, 40), intervals_effective=[(10, 40)])  # fmt: skip
    assert codes(run(led, spec(), claim())) == ["inadequate_support"]
    assert codes(run(led, spec(), claim(status="approximate"))) == ["precision_unsupported"]
    assert run(led, spec(), claim(status="approximate", precision_decimals=0, value=120)).ok


def test_units_are_checked_not_inferred(led):
    computed(led)
    assert codes(run(led, spec(), claim(unit=None))) == ["unknown_or_incompatible_unit"]
    assert codes(run(led, spec(), claim(unit="furlong"))) == ["unknown_or_incompatible_unit"]
    computed(led, tool_call_id="c2", unit="?")
    assert codes(run(led, spec(), claim(evidence_refs=["ev2"]))) == ["unknown_or_incompatible_unit"]
    # a different but compatible unit needs a derived conversion record, not verifier arithmetic
    assert codes(run(led, spec(), claim(unit="kPa", value=15.73))) == ["unknown_or_incompatible_unit"]
    k = led.derive("unit_convert", ["ev1"], to="kPa")
    assert run(led, spec(), claim(unit="kPa", value=round(k.value, 2), evidence_refs=[k.evidence_id],
                                  derived={"op": "unit_convert"})).ok  # fmt: skip


def test_timebase_blocks_physiology_but_allows_observed_delay(led):
    s = spec(quantities=("sbp_mean",))
    s.quantities.append(QuantitySpec("delay", "delay.r_to_foot.observed.v1", "ms", required=False))
    s.quantities.append(QuantitySpec("pat", "pat.r_to_foot.v1", "ms", required=False))
    computed(led, quantity="r_to_foot_delay", definition_id="delay.r_to_foot.observed.v1", value=212.0, unit="ms",
             channels_effective=("ECG", "ABP"), timebase={"sync_status": "index_aligned_declared"})  # fmt: skip
    computed(led, quantity="pulse_arrival_time", definition_id="pat.r_to_foot.v1", value=212.0, unit="ms",
             channels_effective=("ECG", "ABP"), timebase={"sync_status": "index_aligned_declared"}, tool_call_id="c2")  # fmt: skip
    obs = claim(claim_id="d", quantity="delay", status="observed_delay", value=212.0, unit="ms",
                definition_id="delay.r_to_foot.observed.v1")  # fmt: skip
    assert run(led, s, obs, limitations=[{"quantity": "sbp_mean", "reason": "not measured"}]).ok
    pat = claim(claim_id="p", quantity="pat", status="physiological_timing", value=212.0, unit="ms",
                evidence_refs=["ev2"], definition_id="pat.r_to_foot.v1")  # fmt: skip
    lim = [{"quantity": "sbp_mean", "reason": "not measured"}]
    assert codes(run(led, s, pat, limitations=lim)) == ["timebase_unknown"]
    s.declared_sync[pair_key("ECG", "ABP")] = "declared_synchronous"
    assert run(led, s, pat, limitations=lim).ok
    # wrong status for the definition
    assert codes(run(led, s, claim(claim_id="x", quantity="pat", value=212.0, unit="ms", evidence_refs=["ev2"],
                                   definition_id="pat.r_to_foot.v1"), limitations=lim)) == ["method_mismatch"]  # fmt: skip


def test_required_channels_without_irrelevant_ones(led):
    computed(led)  # ABP only
    assert run(led, spec(), claim()).ok  # the spec has an ECG role; SBP does not need it
    s = spec()
    s.quantities.append(QuantitySpec("pat", "pat.r_to_foot.v1", "ms", required=False))
    computed(led, tool_call_id="c2", quantity="pulse_arrival_time", definition_id="pat.r_to_foot.v1", value=212.0,
             unit="ms", channels_effective=("ABP",), timebase={"sync_status": "declared_synchronous"})  # fmt: skip
    r = run(led, s, claim(), claim(claim_id="p", quantity="pat", status="physiological_timing", value=212.0, unit="ms",
                                   evidence_refs=["ev2"], definition_id="pat.r_to_foot.v1"))  # fmt: skip
    assert codes(r) == ["missing_required_channel"]


def test_warnings_by_severity(led):
    computed(led, warnings=[{"code": "low_coverage", "message": "48 %", "severity": "warning"}])
    assert codes(run(led, spec(), claim())) == ["unresolved_warning_or_conflict"]
    assert run(led, spec(), claim(acknowledged_warnings=["low_coverage"])).ok
    assert run(led, spec(), claim(status="approximate", precision_decimals=0, value=118)).ok  # approximate tolerates it
    computed(led, tool_call_id="c2", warnings=[{"code": "few_beats", "message": "7", "severity": "info"}])
    assert run(led, spec(), claim(evidence_refs=["ev2"])).ok  # info is informational
    computed(led, tool_call_id="c3", warnings=[{"code": "nonfinite", "message": "NaN", "severity": "error"}])
    assert codes(run(led, spec(), claim(evidence_refs=["ev3"], acknowledged_warnings=["nonfinite"]))) == [
        "inadequate_support"
    ]


def test_conflicting_evidence_must_be_resolved(led):
    computed(led)
    computed(led, tool_call_id="c2", value=125.0)
    assert codes(run(led, spec(), claim(evidence_refs=["ev1", "ev2"], value=125.0))) == [
        "unresolved_warning_or_conflict"
    ]
    computed(led, tool_call_id="c3", value=125.0, supersedes="ev1")
    assert run(led, spec(), claim(evidence_refs=["ev1", "ev3"], value=125.0)).ok


def test_arithmetic_and_precision(led):
    computed(led)
    assert codes(run(led, spec(), claim(value=125.0))) == ["arithmetic_mismatch"]
    assert codes(run(led, spec(), claim(value=118.04))) == ["precision_unsupported"]
    assert run(led, spec(), claim(value=118)).ok
    computed(led, tool_call_id="c2", value=117.96)
    assert run(led, spec(), claim(evidence_refs=["ev2"], value=117.96)).ok
    s = spec()
    s.quantities[0] = QuantitySpec("sbp_mean", SBP, "mmHg", precision_decimals=0)
    assert codes(run(led, s, claim(evidence_refs=["ev2"], value=117.96))) == ["precision_unsupported"]


def test_derived_values_are_traceable(led):
    a = computed(led)
    b = computed(led, tool_call_id="c2", quantity="dbp_mean", definition_id="abp.dbp.beat_mean.v1", value=80.0)
    d = led.derive("subtract", [a.evidence_id, b.evidence_id])
    s = spec()
    s.quantities.append(QuantitySpec("pp", "abp.pp.beat_mean.v1", "mmHg", required=False))
    good = claim(claim_id="pp", quantity="pp", value=38.0, evidence_refs=[d.evidence_id], derived={"op": "subtract"},
                 definition_id=None)  # fmt: skip
    assert run(led, s, claim(), good).ok
    own_math = claim(claim_id="pp", quantity="pp", value=38.0, evidence_refs=["ev1", "ev2"], derived={"op": "subtract"})
    assert codes(run(led, s, claim(), own_math)) == ["inadequate_support"]


def test_limitations_and_partial_bundles(led):
    computed(led)
    s = spec(quantities=("sbp_mean", "dbp_mean"))
    r = run(led, s, claim())
    assert codes(r) == ["inadequate_support"] and r.deficiencies[0].claim_id is None
    assert run(led, s, claim(), limitations=[{"quantity": "dbp_mean", "reason": "foot detection unreliable"}]).ok
    assert run(led, s, claim(), claim(claim_id="d", quantity="dbp_mean", status="insufficient_evidence",
                                      value=None, reason="no accepted beats")).ok  # fmt: skip
    assert codes(run(led, s, claim(), claim(claim_id="d", quantity="dbp_mean", status="insufficient_evidence",
                                            value=None))) == ["inadequate_support"]  # fmt: skip
    lim = computed(led, tool_call_id="c3", value=None, status="limitation", note="no accepted complete beat")
    assert codes(run(led, spec(), claim(evidence_refs=[lim.evidence_id]))) == ["inadequate_support"]
    hyp = led.record(SourceKind.HYPOTHESIS, "sbp_mean", value=118.0, unit="mmHg", definition_id=SBP,
                     channels_effective=("ABP",), interval_requested=(10, 40), intervals_effective=[(10, 40)])  # fmt: skip
    assert codes(run(led, spec(), claim(evidence_refs=[hyp.evidence_id]))) == ["inadequate_support"]
    few = computed(led, tool_call_id="c4", counts={"accepted": 3})
    assert codes(run(led, spec(), claim(evidence_refs=[few.evidence_id]))) == ["inadequate_support"]


def test_malformed_bundles_are_rejected_at_parse():
    for bad in ({"claims": "x"}, {"claims": [{"claim_id": "a", "quantity": "q"}]},
                {"claims": [{"claim_id": "a", "quantity": "q", "status": "truth"}]},
                {"claims": [{"claim_id": "a", "quantity": "q", "status": "qualitative"}] * 2}):  # fmt: skip
        with pytest.raises(ValueError):
            AnswerBundle.from_dict(bad)


def test_helpers_and_code_list():
    assert decimals_of(118.0) == 0 and decimals_of(117.96) == 2 and decimals_of(0.0155) == 4 and decimals_of("x") == 0
    for c in ("invalid_reference", "method_mismatch", "scope_mismatch", "missing_required_channel",
              "unknown_or_incompatible_unit", "timebase_unknown", "inadequate_support",
              "unresolved_warning_or_conflict", "arithmetic_mismatch", "precision_unsupported"):  # fmt: skip
        assert c in CODES
    b = Budgets()
    assert b.max_repair_cycles_per_issue == 2 and TaskSpec.minimal().budgets == b


def test_precision_bounded_by_known_resolution(led):
    """A full-precision derived mean with SEM 0.45 supports one decimal, not two (pilot finding)."""
    computed(led, value=148.4437, resolution={"known": True, "value": 0.45, "kind": "standard_error_of_mean"})
    assert codes(run(led, spec(), claim(value=148.44))) == ["precision_unsupported"]
    assert run(led, spec(), claim(value=148.4)).ok
    assert run(led, spec(), claim(value=148.44, status="approximate", precision_decimals=2)).ok
    computed(led, tool_call_id="c2", value=148.4437, resolution={"known": True, "value": 0.004, "kind": "sem"})
    assert run(led, spec(), claim(evidence_refs=["ev2"], value=148.44)).ok


def test_string_limitations_become_notes():
    b = AnswerBundle.from_dict(
        {"claims": [], "limitations": ["2 boundary beats excluded", {"quantity": "dbp_mean", "reason": "flat"}]}
    )
    assert b.notes == ["2 boundary beats excluded"] and b.limitations[0]["quantity"] == "dbp_mean"
    with pytest.raises(ValueError, match="plain strings are kept as notes"):
        AnswerBundle.from_dict({"claims": [], "limitations": [{"reason": "no quantity"}]})


def test_window_rule_requires_boundary_evidence(led):
    """A claim for 'the 60 s after the dropout' needs the stated length and boundary evidence the agent holds."""
    from tsmcp.runtime.taskspec import QuantitySpec, TaskSpec

    s = TaskSpec.minimal()
    s.quantities = [QuantitySpec("sbp_mean", SBP, "mmHg")]
    s.channel_roles = {"pressure": "ABP"}
    s.conventions = {"window_rule": {"type": "after_dropout", "channel": "ABP", "length_s": 60.0, "tolerance_s": 2.0}}
    computed(led, interval_requested=(200.5, 260.5), intervals_effective=[(200.5, 260.5)])  # ev1: the right window
    c = claim(interval=[200.5, 260.5])
    r = run(led, s, c)
    assert codes(r) == ["inadequate_support"] and "boundary evidence" in r.deficiencies[0].message
    # beat exclusions ending at the start support it
    computed(led, tool_call_id="c2", interval_requested=(81.5, 381.5), intervals_effective=[(81.5, 381.5)],
             intervals_excluded=[(160.5, 200.5)], value=123.0)  # fmt: skip
    assert run(led, s, c).ok
    # wrong length
    computed(
        led, tool_call_id="c3", interval_requested=(200.5, 300.5), intervals_effective=[(200.5, 300.5)], value=121.0
    )
    assert codes(run(led, s, claim(evidence_refs=["ev3"], interval=[200.5, 300.5], value=121.0))) == ["scope_mismatch"]
    # a flat-event detection also counts
    led2 = EvidenceLedger("r2")
    computed(led2, interval_requested=(200.5, 260.5), intervals_effective=[(200.5, 260.5)])
    led2.record(SourceKind.COMPUTED, "events.flat", tool="ts_find_events", tool_call_id="c9", value=1, unit="count",
                definition_id="events.v1", channels_effective=("ABP",), interval_requested=(81.5, 381.5),
                intervals_effective=[(81.5, 381.5)], parameters={"candidates": [{"t": 160.5, "dur": 40.0}]})  # fmt: skip
    assert run(led2, s, c).ok


def test_method_mismatch_names_the_derivation_for_derived_only_definitions(led):
    from tsmcp.runtime.taskspec import QuantitySpec, TaskSpec

    s = TaskSpec.minimal()
    s.quantities = [QuantitySpec("pulse_rate_bpm", "rate.from_period.v1", "bpm")]
    s.channel_roles = {"pulse": "PPG"}
    computed(led, quantity="period", definition_id="generic.period.v1", value=0.726, unit="s", channels_effective=("PPG",),
             interval_requested=(64.5, 124.5), intervals_effective=[(64.5, 124.5)], counts={})  # fmt: skip
    r = run(led, s, claim(quantity="pulse_rate_bpm", value=82.6, unit="bpm", interval=[64.5, 124.5],
                          definition_id="rate.from_period.v1"))  # fmt: skip
    assert codes(r) == ["method_mismatch"] and "ts_derive op=rate_from_period" in r.deficiencies[0].message
    d = led.derive("rate_from_period", ["ev1"])
    assert run(led, s, claim(quantity="pulse_rate_bpm", value=d.value, unit="bpm", interval=[64.5, 124.5],
                             evidence_refs=[d.evidence_id], definition_id="rate.from_period.v1")).ok  # fmt: skip
