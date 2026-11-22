import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.af_episode_metrics import score_record  # noqa: E402
from eval.onset_evaluate import acceptance, candidate_diagnostic, load_allowed, summarize_candidate  # noqa: E402


def baseline_row(name, *, detected=True):
    episodes = [{"t": 100, "dur": 400, "intervals_s": [[100, 500]]}] if detected else []
    return score_record(name, 1000, [[100, 500]], episodes, [[0, 1000]])


def boundary_event(point):
    return {
        "onset_s": point,
        "onset_confident": point is not None,
        "onset_lo_s": point if point is not None else 0,
        "onset_hi_s": point if point is not None else 400,
    }


def test_uncertain_interval_containing_truth_is_not_a_correct_point():
    base = baseline_row("synthetic")
    result = candidate_diagnostic(base, [boundary_event(None)])
    assert result["interval_contains_truth"] is True
    assert result["onset_within_60s"] is False
    assert result["signed_onset_error_s"] is None
    assert result["onset_confident"] is False
    assert result["missing_episode"] is False
    assert result["iou"] == 1  # Event overlap does not establish its onset.


def test_unconfident_flag_cannot_be_bypassed_with_a_non_null_point():
    event = boundary_event(100)
    event["onset_confident"] = False
    result = candidate_diagnostic(baseline_row("synthetic"), [event])
    assert result["interval_contains_truth"] is True
    assert result["onset_within_60s"] is False
    assert result["signed_onset_error_s"] is None
    assert result["onset_s"] is None


@pytest.mark.parametrize(("point", "correct"), [(40, True), (160, True), (160.001, False)])
def test_confident_points_use_unchanged_inclusive_60_second_tolerance(point, correct):
    result = candidate_diagnostic(baseline_row("synthetic"), [boundary_event(point)])
    assert result["signed_onset_error_s"] == point - 100
    assert result["onset_within_60s"] is correct


def test_uncertain_and_missing_episodes_remain_in_fixed_denominator():
    rows, originals = [], []
    for name, point, detected in (
        ("correct", 130, True),
        ("wrong", 200, True),
        ("uncertain", None, True),
        ("missing", None, False),
    ):
        original = baseline_row(name, detected=detected)
        events = [boundary_event(point)] if detected else []
        originals.append(original)
        rows.append(
            {
                "record": name,
                "has_af": True,
                "onset_eligible": True,
                "V_ref": candidate_diagnostic(original, events),
                "events": events,
            }
        )
    result = summarize_candidate(rows, originals)
    for population in ("eligible", "all_af"):
        stats = result[population]
        assert stats["n_records"] == 4
        assert stats["onset_correct"] == 1
        assert stats["missing_point"] == stats["onset_confident_false"] == 2
        assert stats["missing_episode"] == 1
        assert stats["confident_count"] == 2
        assert stats["confident_correct"] == 1
        assert stats["confident_accuracy"] == 0.5
        assert stats["unresolved_interval_contains_truth"] == 1
    assert result["all_events"] == {"count": 3, "onset_confident_false": 1}


def gate_summaries(correct):
    return {
        "eligible": {"onset_correct": correct},
        "burden_overlap_all_records": {
            "burden_mae_window_pp": 8.0,
            "burden_mae_readable_pp": 4.0,
            "pooled_sensitivity": 0.84,
            "pooled_ppv": 0.987,
        },
    }


@pytest.mark.parametrize(("correct", "passes"), [(8, False), (9, True)])
def test_gate_requires_at_least_three_more_eligible_onsets_over_v(correct, passes):
    baseline = {"R": gate_summaries(6), "V": gate_summaries(6)}
    result = acceptance(baseline, gate_summaries(correct), True)
    assert result["onset_gain"] == {"R": correct - 6, "V": correct - 6}
    assert result["pass"] is passes


@pytest.mark.parametrize(
    ("metric", "loss"),
    [
        ("burden_mae_window_pp", 1e-12),
        ("burden_mae_readable_pp", 1e-12),
        ("pooled_sensitivity", -1e-12),
        ("pooled_ppv", -1e-12),
    ],
)
def test_gate_allows_no_burden_sensitivity_or_ppv_loss(metric, loss):
    baseline = {"R": gate_summaries(6), "V": gate_summaries(6)}
    candidate = gate_summaries(9)
    assert acceptance(baseline, candidate, True)["pass"] is True
    candidate["burden_overlap_all_records"][metric] += loss
    result = acceptance(baseline, candidate, True)
    assert result["onset_pass"] is True
    assert result["burden_se_ppv_no_loss"] is False
    assert result["pass"] is False


def test_gate_also_requires_improvement_over_r_and_unchanged_support():
    baseline = {"R": gate_summaries(6), "V": gate_summaries(6)}
    assert acceptance(baseline, gate_summaries(9), False)["pass"] is False
    stronger_r = copy.deepcopy(baseline)
    stronger_r["R"]["eligible"]["onset_correct"] = 9
    assert acceptance(stronger_r, gate_summaries(9), True)["pass"] is False


@pytest.mark.parametrize(
    ("name", "permitted"),
    [
        ("afdb/00", {"ltafdb/00"}),
        ("afdb/00", {"afdb/00"}),
        ("ltafdb/not_a_record", {"ltafdb/not_a_record"}),
        ("ltafdb/../../afdb/00", {"ltafdb/../../afdb/00"}),
        ("ltafdb/05", {"ltafdb/00"}),
    ],
)
def test_load_allowed_rejects_disallowed_records_before_accessing_files(monkeypatch, name, permitted):
    from tsmcp.core.io import wfdb

    def forbidden(*args, **kwargs):
        pytest.fail("Rejected record reached filesystem or waveform reader")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_file", forbidden)
        patch.setattr(wfdb, "read_header", forbidden)
        patch.setattr(wfdb, "load_wfdb", forbidden)
        with pytest.raises(ValueError):
            load_allowed(name, permitted)
