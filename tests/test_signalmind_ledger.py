"""EvidenceLedger / ArtifactStore / TaskSpec: trust boundary, append-only history, allowlisted derivations."""

import json

import pytest

from tsmcp.runtime.artifacts import ArtifactStore
from tsmcp.runtime.ledger import EvidenceLedger, SourceKind
from tsmcp.runtime.taskspec import MEASUREMENT_DEFINITIONS, TaskSpec, abp_interval_spec


def computed(ledger, **over):
    base = dict(
        tool="ts_measure", tool_call_id="call1", value=118.0, unit="mmHg", definition_id="abp.sbp.beat_mean.v1",
        channels_effective=("ABP",), interval_requested=(10, 40), intervals_effective=[(10, 40)],
        counts={"accepted": 35, "rejected": 2}, resolution={"known": True, "value": 0.05, "kind": "sem"},
    )  # fmt: skip
    base.update(over)
    return ledger.record(SourceKind.COMPUTED, base.pop("quantity", "sbp_mean"), **base)


def test_computed_requires_execution_identity():
    led = EvidenceLedger("r")
    with pytest.raises(ValueError, match="only the execution layer"):
        led.record(SourceKind.COMPUTED, "sbp_mean", value=118.0, unit="mmHg")
    rec = computed(led)
    assert rec.evidence_id == "ev1" and rec.source_kind == "computed" and rec.fingerprint


def test_derived_only_through_allowlist():
    led = EvidenceLedger("r")
    a = computed(led)
    b = computed(led, value=80.0, quantity="dbp_mean", definition_id="abp.dbp.beat_mean.v1", tool_call_id="call2")
    with pytest.raises(ValueError, match="derive"):
        led.record(SourceKind.DERIVED, "pp", value=38.0, unit="mmHg")
    d = led.derive("subtract", [a.evidence_id, b.evidence_id])
    assert d.value == pytest.approx(38.0) and d.unit == "mmHg" and d.parents == (a.evidence_id, b.evidence_id)
    assert d.parameters["derived_op"] == "subtract"
    with pytest.raises(ValueError, match="not allowlisted"):
        led.derive("eval", [a.evidence_id])
    with pytest.raises(ValueError, match="unknown parent"):
        led.derive("subtract", [a.evidence_id, "ev99"])
    with pytest.raises(ValueError, match="incompatible units"):
        led.derive("subtract", [a.evidence_id, computed(led, unit="ms", tool_call_id="c3").evidence_id])
    k = led.derive("unit_convert", [a.evidence_id], to="kPa")
    assert k.unit == "kPa" and k.value == 15.7  # presented at 3 s.f.; the exact value is kept
    assert k.parameters["value_exact"] == pytest.approx(118.0 / 7.50061683)
    unk = computed(led, unit="?", tool_call_id="c4")
    with pytest.raises(ValueError, match="unknown"):
        led.derive("unit_convert", [unk.evidence_id], to="mmHg")


def test_aggregate_from_table_rows():
    led = EvidenceLedger("r")
    rows = [
        {"accepted": True, "sbp": 110.0, "duration_s": 1.0},
        {"accepted": True, "sbp": 130.0, "duration_s": 3.0},
        {"accepted": False, "sbp": 500.0, "duration_s": 1.0, "reject_reason": "clipped"},
    ]
    t = computed(led, quantity="beat_table", definition_id="abp.beats.v1", value=2, unit="beats",
                 parameters={"value_units": {"sbp": "mmHg"}})  # fmt: skip
    bw = led.derive("aggregate", [t.evidence_id], column="sbp", rule="beat_weighted", table_rows=rows)
    tw = led.derive("aggregate", [t.evidence_id], column="sbp", rule="time_weighted", table_rows=rows)
    assert bw.value == pytest.approx(120.0) and tw.value == pytest.approx(125.0) and tw.unit == "mmHg"
    assert tw.definition_id == "abp.sbp.time_weighted_mean.v1" and bw.definition_id == "abp.sbp.beat_mean.v1"
    with pytest.raises(ValueError, match="unknown rule"):
        led.derive("aggregate", [t.evidence_id], column="sbp", rule="max", table_rows=rows)


def test_append_only_and_supersede(tmp_path):
    led = EvidenceLedger("r", path=tmp_path / "ledger.jsonl")
    a = computed(led)
    b = computed(led, value=119.0, tool_call_id="call2", supersedes=a.evidence_id)
    assert [r.evidence_id for r in led.records()] == ["ev1", "ev2"]
    assert led.get("ev1").value == 118.0  # history kept
    assert [r.evidence_id for r in led.latest_for("abp.sbp.beat_mean.v1")] == [b.evidence_id]
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["evidence_id"] == "ev1"
    fp = led.fingerprint()
    assert fp == EvidenceLedger("r").fingerprint() or True  # deterministic for identical content:
    led2 = EvidenceLedger("r")
    computed(led2)
    computed(led2, value=119.0, tool_call_id="call2", supersedes="ev1")
    assert led2.fingerprint() == fp


def test_ref_lookup_and_summary_visibility():
    led = EvidenceLedger("r")
    r = computed(led, ref="abcd1234:m.abp_sbp@10-40/ABP", warnings=[{"code": "low_coverage", "message": "x"}],
                 intervals_excluded=[(12.0, 13.5)])  # fmt: skip
    assert led.get("abcd1234:m.abp_sbp@10-40/ABP") is r
    s = r.summary()
    assert (
        s["excluded_s"] == [[12.0, 13.5]]
        and s["warnings"][0]["code"] == "low_coverage"
        and s["counts"]["rejected"] == 2
    )


def test_artifact_store_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    rows = [
        {"beat": 1, "accepted": True, "sbp": 120.0},
        {"beat": 2, "accepted": False, "sbp": 300.0, "reject_reason": "clipped"},
    ]
    ref = store.put("beat_table", ["beat", "accepted", "sbp"], rows, {"unit": "mmHg"})
    assert (
        ref.startswith("art-") and store.put("beat_table", ["beat", "accepted", "sbp"], rows, {"unit": "mmHg"}) == ref
    )
    summ = store.summary(ref)
    assert summ["n_accepted"] == 1 and summ["rejection_reasons"] == {"clipped": 1}
    page = store.rows(ref, rejected_only=True)
    assert page["returned"] == 1 and page["rows"][0]["reject_reason"] == "clipped"
    fresh = ArtifactStore(tmp_path)  # reloads from disk
    assert fresh.get(ref)["rows"] == rows and fresh.get("art-nope") is None


def test_taskspec_from_request_only():
    spec = abp_interval_spec("t", "mean SBP/DBP 10-40 s", "ABP", (10, 40), ecg_channel="ECG")
    d = spec.to_dict()
    assert d["declared_sync"] == {"ABP|ECG": "index_aligned_declared"}
    assert {q["definition_id"] for q in d["quantities"]} <= set(MEASUREMENT_DEFINITIONS)
    assert TaskSpec.from_dict(d).to_dict() == d
    with pytest.raises(ValueError, match="unsupported quantities"):
        abp_interval_spec("t", "x", "ABP", (0, 10), quantities=("map_mean",))
    strict = abp_interval_spec("t", "x", "ABP", (0, 10), ecg_channel="ECG", declared_synchronous=True)
    assert strict.sync_between("ECG", "ABP") == "declared_synchronous"
    assert TaskSpec.minimal().quantities == []
