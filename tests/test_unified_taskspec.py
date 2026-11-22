"""Public task constraints: missing outputs, method substitutions and wrong windows remain observable errors."""

import json
from pathlib import Path

import pytest

from eval.unified.tasks import public_answer_fields, task_spec_for_case
from tsmcp.runtime.claims import AnswerBundle
from tsmcp.runtime.ledger import EvidenceLedger, SourceKind
from tsmcp.runtime.taskspec import MEASUREMENT_DEFINITIONS, TaskSpec
from tsmcp.runtime.timebase import SyncStatus
from tsmcp.runtime.verifier import verify


def public_case(question, duration=180):
    return {
        "case_id": "public-case",
        "question": question,
        "visible": {
            "dur_s": duration,
            "channels": [
                {"name": name, "fs": 125, "unit": unit}
                for name, unit in (("ECG", "mV"), ("PPG", "?"), ("ABP", "mmHg"), ("CBFV", "cm/s"))
            ],
        },
    }


def check_claim(spec, definition, quantity, value, unit, window, declared_definition=None):
    ledger = EvidenceLedger("task-test")
    ev = ledger.record(
        SourceKind.COMPUTED,
        quantity,
        tool="ts_measure",
        tool_call_id="c1",
        value=value,
        unit=unit,
        definition_id=definition,
        channels_effective=("ABP", "CBFV"),
        interval_requested=window,
        intervals_effective=[window],
        counts={"accepted": 20},
        resolution={"known": True, "value": 1.0},
    )
    claim = {
        "claim_id": "c",
        "quantity": quantity,
        "value": value,
        "unit": unit,
        "status": "validated_measurement",
        "interval": list(window),
        "evidence_refs": [ev.evidence_id],
    }
    if declared_definition:
        claim["definition_id"] = declared_definition
    return verify(AnswerBundle.from_dict({"claims": [claim]}), ledger, spec)


def test_all_public_level2_questions_have_required_fields_without_reference_reads():
    root = Path(__file__).resolve().parents[1] / "eval" / "cases"
    questions = root / "p9level2.questions.jsonl"
    if not questions.exists():
        pytest.skip("public Level 2 cases not built")
    rows = [json.loads(line) for line in questions.read_text().splitlines()]
    assert len(rows) == 49
    for row in rows:
        # Only the public question file and visible metadata are opened, never the grading manifest or answers.
        row["visible"] = json.loads((root / "p9level2" / row["case_id"] / "visible.json").read_text())
        spec = task_spec_for_case(row)
        assert spec.quantities and all(q.required for q in spec.quantities)
        assert all(q.definition_id is None or q.definition_id in MEASUREMENT_DEFINITIONS for q in spec.quantities)
        assert spec.channel_roles and spec.request_text == row["question"]
        assert TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict() == spec.to_dict()
        missing = verify(AnswerBundle.from_dict({"claims": []}), EvidenceLedger("empty"), spec)
        assert not missing.ok and len(missing.deficiencies) == len(spec.quantities)


def test_spec_ignores_hidden_fields_and_case_identity():
    case = public_case("What is the mean beatwise systolic pressure (mmHg) over this recording? Return sbp_mmhg.")
    before = task_spec_for_case(case).to_dict()
    case.update(
        expected_answer={"sbp_mmhg": -999},
        grounding_rubric={"regions": [[17, 19]]},
        tolerances={"sbp_mmhg": 100},
        family="irrelevant",
        case_id="different-id",
    )
    after = task_spec_for_case(case).to_dict()
    after["task_id"] = before["task_id"]
    assert after == before


def test_response_windows_and_extrema_conventions_come_from_question():
    case = public_case(
        "What is the change in mean beatwise systolic pressure (mmHg) from 8.91–43.91 s to "
        "73.91–103.91 s (later minus baseline)? Return sbp_change_mmhg, response_max_time_s and "
        "response_min_time_s. Report systolic peak times; for tied extrema, report the earliest time."
    )
    spec = task_spec_for_case(case)
    assert spec.quantity("sbp_change_mmhg").interval == (8.91, 103.91)
    assert spec.quantity("response_max_time_s").interval == (73.91, 103.91)
    assert json.loads(spec.conventions["baseline_window_s"]) == [8.91, 43.91]
    assert spec.conventions["extrema_times"] == "systolic_peaks"
    assert spec.quantity("response_min_time_s").definition is None


def test_requested_definition_cannot_be_replaced_by_explicit_claim_definition():
    spec = task_spec_for_case(
        public_case("What is the mean beatwise systolic pressure (mmHg) over this recording? Return sbp_mmhg.")
    )
    report = check_claim(spec, "generic.mean.v1", "sbp_mmhg", 95, "mmHg", (0, 180), "generic.mean.v1")
    assert [d.code for d in report.deficiencies] == ["method_mismatch"]
    assert check_claim(spec, "abp.sbp.beat_mean.v1", "sbp_mmhg", 120, "mmHg", (0, 180)).ok


def test_tfa_whole_record_request_rejects_short_window():
    spec = task_spec_for_case(
        public_case(
            "Across LF (0.07–0.20 Hz), what is the mean transfer-function gain from ABP "
            "to CBFV? The estimand is an equal-frequency-bin mean for the complete "
            "recording's nonoverlapping 1-s mean pressure and flow. Report gain.",
            600,
        )
    )
    report = check_claim(spec, "tfa.gain.lf.v1", "gain", 1, "(cm/s)/mmHg", (0, 120))
    assert [d.code for d in report.deficiencies] == ["scope_mismatch"]
    assert spec.channel_roles["flow"] == "CBFV" and spec.interval == (0, 600)
    assert check_claim(spec, "tfa.gain.lf.v1", "gain", 1, "(cm/s)/mmHg", (0, 600)).ok


def test_search_question_allows_local_evidence_and_does_not_invent_event_bounds():
    spec = task_spec_for_case(
        public_case(
            "At what time does the sustained attenuation of arterial pulsations begin "
            "in ABP? Report onset_s in seconds."
        )
    )
    assert spec.interval is None and spec.quantity("onset_s").interval is None
    assert json.loads(spec.conventions["search_interval_s"]) == [0, 180]
    assert check_claim(spec, "pulse.amplitude_change.v1", "onset_s", 50, "s", (40, 80)).ok


def test_brs_fields_are_required_but_no_method_is_fabricated():
    spec = task_spec_for_case(
        public_case(
            "What are the sequence-method BRS (ms/mmHg) and number of valid maximal "
            "sequences, pairing each systolic pressure with its contemporaneous RR "
            "interval? Return brs_ms_per_mmhg and valid_sequences.",
            600,
        )
    )
    assert [q.name for q in spec.quantities] == ["brs_ms_per_mmhg", "valid_sequences"]
    assert all(q.definition is None and q.interval == (0, 600) for q in spec.quantities)
    limitations = [
        {"quantity": q.name, "status": "insufficient_evidence", "reason": "BRS computation unavailable"}
        for q in spec.quantities
    ]
    assert verify(AnswerBundle.from_dict({"claims": [], "limitations": limitations}), EvidenceLedger("b"), spec).ok


def test_pulse_field_and_channel_roles_follow_visible_metadata_only():
    case = public_case(
        "What is the mean cardiac pulse rate represented by PPG over this record, in bpm? "
        "Check the pulse interpretation against ECG before reporting bpm."
    )
    assert public_answer_fields(case) == ["bpm"]
    spec = task_spec_for_case(case)
    assert spec.quantity("bpm").definition_id is None  # the question allows beats or period-based rate
    assert spec.channel_roles["pulse"] == "PPG"
    assert set(spec.declared_sync.values()) == {SyncStatus.INDEX_ALIGNED_DECLARED.value}
    case["visible"]["channels"] = [{"name": "ECG", "fs": 125, "unit": "mV"}]
    assert task_spec_for_case(case).channel_roles == {"ecg": "ECG"}


def test_public_pulse_question_accepts_existing_period_derivation():
    spec = task_spec_for_case(
        public_case(
            "What is the mean cardiac pulse rate represented by PPG over this record, in bpm? "
            "Check the pulse interpretation against ECG before reporting bpm."
        )
    )
    ledger = EvidenceLedger("pulse")
    period = ledger.record(
        SourceKind.COMPUTED,
        "period",
        tool="ts_measure",
        tool_call_id="period1",
        value=0.8,
        unit="s",
        definition_id="generic.period.v1",
        channels_effective=("PPG",),
        interval_requested=(0, 180),
        intervals_effective=[(0, 180)],
    )
    rate = ledger.derive("rate_from_period", [period.evidence_id])
    bundle = AnswerBundle.from_dict(
        {
            "claims": [
                {
                    "claim_id": "rate",
                    "quantity": "bpm",
                    "definition_id": "rate.from_period.v1",
                    "status": "validated_measurement",
                    "value": 75,
                    "unit": "bpm",
                    "interval": [0, 180],
                    "evidence_refs": [rate.evidence_id],
                }
            ]
        }
    )
    assert verify(bundle, ledger, spec).ok


def test_tfa_phase_keeps_requested_radians_and_public_band():
    spec = task_spec_for_case(
        public_case(
            "Across LF (0.07–0.20 Hz), what is the mean ABP-to-CBFV transfer phase "
            "(radians, positive for CBFV leading ABP) and magnitude-squared coherence? "
            "The estimand is an equal-frequency-bin mean for the complete recording's "
            "nonoverlapping 1-s mean pressure and flow. Report phase, coherence.",
            600,
        )
    )
    assert spec.quantity("phase").definition_id == "tfa.phase.lf.v1"
    assert spec.quantity("phase").unit == "rad" and spec.quantity("coherence").unit == ""
    assert spec.conventions["phase_sign"] == "positive_for_CBFV_leading_ABP"


def test_ecg_shift_accepts_checked_opposite_sign_derivation_but_rejects_wrong_arithmetic():
    spec = task_spec_for_case(
        public_case(
            "A recording channel develops an additional timing offset. Identify the inconsistent channel and its "
            "added delay relative to the initial synchronized period, preserving the physiological ECG-to-pulse "
            "delay. Report channel and signed shift_ms (positive means delayed)."
        )
    )
    assert spec.quantity("shift_ms").definition_id is None and spec.quantity("shift_ms").required
    assert spec.quantity("shift_ms").unit == "ms"
    assert spec.sync_between("ECG", "ABP") == SyncStatus.INDEX_ALIGNED_DECLARED.value
    ledger = EvidenceLedger("ecg-shift")
    changes = [
        ledger.record(
            SourceKind.COMPUTED,
            "r_to_foot_delay_change",
            tool="ts_measure",
            tool_call_id=f"delay-{channel}",
            value=-24,
            unit="ms",
            definition_id="delay.r_to_foot.change.v1",
            channels_effective=("ECG", channel),
            interval_requested=(0, 180),
            intervals_effective=[(0, 180)],
            timebase={"sync_status": "index_aligned_declared"},
        )
        for channel in ("ABP", "PPG")
    ]
    # Both pulse arrivals move earlier relative to ECG; an ECG delay has the opposite sign.
    zero = ledger.derive("subtract", [changes[0].evidence_id, changes[0].evidence_id])
    reversed_shift = ledger.derive("subtract", [zero.evidence_id, changes[0].evidence_id])
    assert reversed_shift.value == 24 and reversed_shift.definition_id is None
    claims = [
        {
            "claim_id": "channel",
            "quantity": "channel",
            "status": "qualitative",
            "value": "ECG",
            "interval": [0, 180],
            "evidence_refs": [r.evidence_id for r in changes],
        },
        {
            "claim_id": "shift",
            "quantity": "shift_ms",
            "status": "approximate",
            "precision_decimals": 0,
            "value": 24,
            "unit": "ms",
            "interval": [0, 180],
            "derived": {"op": "subtract"},
            "evidence_refs": [reversed_shift.evidence_id],
        },
    ]
    assert verify(AnswerBundle.from_dict({"claims": claims}), ledger, spec).ok
    claims[1]["value"] = -24
    report = verify(AnswerBundle.from_dict({"claims": claims}), ledger, spec)
    assert [d.code for d in report.deficiencies] == ["arithmetic_mismatch"]
