"""Shortcuts use public inputs and the item/F5-pair gates remain distinct."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval import level2_baselines as baselines


def candidate(value, task_type="verified_rate"):
    return {
        "task_type": task_type,
        "family": "F6",
        "expected_answer": {"bpm": value},
        "tolerances": {"bpm": 1},
        "x": np.zeros((1000, 2)),
        "fs": 100,
        "channels": ["ECG", "PPG"],
        "units": ["mV", "a.u."],
    }


@pytest.fixture
def generic_tools():
    return {
        "measures": SimpleNamespace(
            dominant_freq=lambda x, fs, unit, t0: {"value": 2.0, "unit": "Hz"},
            mean=lambda x, fs, unit, t0: {"value": float(np.mean(x)), "unit": unit},
        ),
        "detectors": SimpleNamespace(
            changes=lambda x, fs, unit, t0: [{"type": "change", "t": t0 + 7, "dur": 0.0, "score": 10}],
            flat=lambda x, fs, unit, t0: [{"type": "flat", "t": t0 + 2, "dur": 4.0, "score": 20}],
        ),
        "ecg": SimpleNamespace(
            ecg_hr=lambda x, fs, unit, t0: {"value": 80 + t0, "n_beats": 12},
            ecg_rr_stats=lambda x, fs, unit, t0: {"min_max_s": [0.5, 1.0]},
        ),
        "provenance": {"source_verification": "test double"},
    }


def test_constants_are_fixed_by_public_schema():
    public = {"task_type": "verified_rate", "answer_fields": ["bpm"]}
    assert baselines.constant_answer(public) == {"bpm": 72.0}
    assert baselines.constant_answer({"answer_fields": ["channel", "shift_ms"]}) == {"channel": "PPG", "shift_ms": 0}
    assert baselines.constant_answer({"answer_fields": ["unregistered_quantity"]}) is None
    assert baselines.constant_answer({"answer_fields": []}) is None


def test_leave_one_out_ignores_target_value_and_other_task_types():
    candidates = [candidate(999), candidate(60), candidate(100), candidate(500, "different_task")]
    answer, support = baselines.development_median(candidates, 0)
    assert answer == {"bpm": 80}
    assert support == {"bpm": 2}
    candidates[0]["expected_answer"]["bpm"] = -999
    assert baselines.development_median(candidates, 0)[0] == answer
    assert baselines.development_median(candidates, 3)[0] is None


def test_categorical_median_baseline_uses_mode_with_stable_tie_break():
    candidates = [
        {"task_type": "timing_shift", "expected_answer": {"channel": channel, "shift_ms": shift}}
        for channel, shift in [("ECG", 999), ("PPG", -40), ("ABP", 80)]
    ]
    assert baselines.development_median(candidates, 0)[0] == {"channel": "ABP", "shift_ms": 20}


def test_detector_reads_public_signal_and_records_first_output(generic_tools):
    public = {**candidate(999), "answer_fields": ["bpm"]}
    del public["expected_answer"]
    answer, metadata = baselines.detector_only(public, generic_tools)
    assert answer == {"bpm": 120}
    assert metadata["first_output"] == {
        "operation": "dominant_freq",
        "channel": "PPG",
        "output": {"value": 2, "unit": "Hz"},
    }
    assert len(metadata["channel_outputs"]) == 2
    assert not metadata["unsupported"]


def test_onset_uses_top_generic_change_not_true_onset(generic_tools):
    public = {**candidate(0), "task_type": "damping_onset", "answer_fields": ["onset_s"], "baseline_channel": "PPG"}
    answer, metadata = baselines.detector_only(public, generic_tools)
    assert answer == {"onset_s": 7}
    assert metadata["first_output"]["operation"] == "change"
    assert metadata["first_output"]["channel"] == "PPG"
    assert baselines.detector_only({**public, "answer_fields": ["gain"]}, generic_tools)[0] is None


def test_invalid_detector_result_is_missing_not_abstention(generic_tools):
    generic_tools["measures"].dominant_freq = lambda *args, **kwargs: {"value": float("nan")}
    answer, metadata = baselines.detector_only({**candidate(80), "answer_fields": ["bpm"]}, generic_tools)
    assert answer is None
    assert metadata["unsupported"]
    assert metadata["first_output"]["output"]["value"] is None


def test_manual_artifact_shortcuts_include_duration(generic_tools):
    public = {**candidate(0), "task_type": "manual_abp_artifact", "answer_fields": ["onset_s", "duration_s"]}
    assert baselines.constant_answer(public) == {"onset_s": 80.0, "duration_s": 10.0}
    assert baselines.detector_only(public, generic_tools)[0] == {"onset_s": 7, "duration_s": 0.0}
    generic_tools["detectors"].changes = lambda *args, **kwargs: []
    assert baselines.detector_only(public, generic_tools)[0] == {"onset_s": 2, "duration_s": 4.0}


def test_pressure_shortcut_is_explicitly_the_generic_time_mean(generic_tools):
    public = {**candidate(0), "channels": ["ECG", "ABP"], "answer_fields": ["sbp_mmhg"]}
    public["x"][:, 1] = 91.0
    answer, metadata = baselines.detector_only(public, generic_tools)
    assert answer == {"sbp_mmhg": 91.0}
    assert metadata["first_output"]["operation"] == "mean"


@pytest.mark.parametrize(
    ("field", "channel", "periods", "expected"),
    [("hr_change_bpm", "ECG", [[0, 2], [5, 7]], 5.0), ("rr_ratio", "ECG", [[0, 2]], 2.0)],
)
def test_f1_shortcuts_use_only_question_stated_windows(generic_tools, field, channel, periods, expected):
    public = {
        **candidate(0),
        "task_type": field,
        "answer_fields": [field],
        "baseline_channel": channel,
        "baseline_windows_s": periods,
    }
    answer, metadata = baselines.detector_only(public, generic_tools)
    assert answer == {field: expected}
    assert [row["window_s"] for row in metadata["period_outputs"]] == periods
    del public["baseline_windows_s"]
    assert baselines.detector_only(public, generic_tools)[0] is None


def test_f1_pressure_change_is_difference_of_generic_period_means(generic_tools):
    public = {
        **candidate(0),
        "channels": ["ECG", "ABP"],
        "task_type": "sbp_change_mmhg",
        "answer_fields": ["sbp_change_mmhg"],
        "baseline_channel": "ABP",
        "baseline_windows_s": [[0, 2], [5, 7]],
    }
    public["x"][:200, 1] = 90
    public["x"][500:700, 1] = 100
    assert baselines.detector_only(public, generic_tools)[0] == {"sbp_change_mmhg": 10.0}


@pytest.mark.parametrize(
    ("field", "location_fields", "windows", "locations"),
    [
        ("rr_ratio", ["max_rr_time_s", "min_rr_time_s"], [[0, 2]], [1.5, 0.75]),
        ("hr_change_bpm", ["response_max_time_s", "response_min_time_s"], [[0, 2], [5, 7]], [5.75, 6.5]),
    ],
)
def test_f1_detector_supplies_magnitude_and_mechanical_rr_locations(
    generic_tools, field, location_fields, windows, locations
):
    generic_tools["ecg"].series_rr = lambda x, fs, unit, t0: {
        "t": np.array([t0 + 1, t0 + 2]),
        "y": np.array([0.5, 1]),
    }
    public = {
        **candidate(0),
        "answer_fields": [field, *location_fields],
        "baseline_channel": "ECG",
        "baseline_windows_s": windows,
    }
    assert set(baselines.constant_answer(public)) == set(public["answer_fields"])
    answer, metadata = baselines.detector_only(public, generic_tools)
    assert set(answer) == set(public["answer_fields"])
    assert [answer[name] for name in location_fields] == locations
    assert not metadata["unsupported_fields"]
    assert not metadata["unsupported"]
    assert metadata["localization_output"]["operation"] == "series_rr_extrema_midpoints"


def test_f1_pressure_localization_records_generic_sample_extrema(generic_tools):
    public = {
        **candidate(0),
        "channels": ["ECG", "ABP"],
        "answer_fields": ["sbp_change_mmhg", "response_max_time_s", "response_min_time_s"],
        "baseline_channel": "ABP",
        "baseline_windows_s": [[0, 2], [5, 7]],
    }
    public["x"][550, 1] = 10
    public["x"][560, 1] = -10
    answer, metadata = baselines.detector_only(public, generic_tools)
    assert answer == {"sbp_change_mmhg": 0, "response_max_time_s": 5.5, "response_min_time_s": 5.6}
    assert metadata["localization_output"]["operation"] == "raw_sample_extrema"
    assert not metadata["unsupported"]


def test_f1_unsupported_detector_localization_is_reported(generic_tools):
    public = {
        **candidate(0),
        "answer_fields": ["rr_ratio", "max_rr_time_s", "min_rr_time_s"],
        "baseline_channel": "ECG",
        "baseline_windows_s": [[0, 2]],
    }
    answer, metadata = baselines.detector_only(public, generic_tools)
    assert answer == {"rr_ratio": 2}
    assert metadata["unsupported"]
    assert metadata["unsupported_fields"] == ["max_rr_time_s", "min_rr_time_s"]
    assert metadata["localization_output"]["error"]


@pytest.fixture
def paired_candidates():
    shared = {
        "family": "F5",
        "pair_id": "a" * 64,
        "task_type": "heart_rate",
        "answer_fields": ["bpm"],
    }
    return [
        {**candidate(72), **shared, "pair_role": "control", "abstention_correct": False},
        {
            **candidate(0),
            **shared,
            "pair_role": "abstention",
            "expected_answer": {"insufficient_evidence": True, "reason": "Absent pulse signal."},
            "abstention_correct": True,
        },
    ]


def test_constant_abstention_and_constant_answer_cannot_shortcut_pair(monkeypatch, generic_tools, paired_candidates):
    @contextmanager
    def context(root):
        yield generic_tools

    monkeypatch.setattr(baselines, "frozen_generic_tools", context)
    control, insufficient = baselines.run_baselines(paired_candidates)
    assert "constant" in control["successful_baselines"]
    assert "constant_abstention" in insufficient["successful_baselines"]
    assert insufficient["constant"]["grade"]["fabrication"]
    assert insufficient["constant_abstention"]["grade"]["correct_abstention"]
    assert control["retained"] and insufficient["retained"]
    assert control["rejected_by"] == insufficient["rejected_by"] == []
    assert control["pair_gate"] == insufficient["pair_gate"]
    assert control["pair_gate"]["valid"]
    assert not any(score["pair_correct"] for score in control["pair_gate"]["baseline_scores"].values())


def test_same_baseline_must_pass_both_members_after_all_items_are_evaluated(
    monkeypatch, generic_tools, paired_candidates
):
    @contextmanager
    def context(root):
        yield generic_tools

    calls = []

    def detector(public, tools):
        calls.append(public)
        answer = {"bpm": 72} if np.any(public["x"]) else dict(baselines.CONSTANT_ABSTENTION)
        return answer, {}

    paired_candidates[0]["x"][:] = 1
    monkeypatch.setattr(baselines, "frozen_generic_tools", context)
    monkeypatch.setattr(baselines, "detector_only", detector)
    candidates = [paired_candidates[1], candidate(72), paired_candidates[0]]
    rows = baselines.run_baselines(candidates)
    assert len(calls) == 3
    assert rows[0]["rejected_by"] == rows[2]["rejected_by"] == ["detector_only"]
    assert rows[0]["pair_gate"]["baseline_scores"]["detector_only"]["pair_correct"]
    assert rows[0]["pair_gate"]["member_indices"] == [0, 2]
    assert rows[1]["rejected_by"] == ["constant"]
    assert not any(row["retained"] for row in rows)


@pytest.mark.parametrize("malformation", ["incomplete", "duplicate", "mismatch", "unhashed", "other_family"])
def test_malformed_pairs_fail_closed(monkeypatch, generic_tools, paired_candidates, malformation):
    @contextmanager
    def context(root):
        yield generic_tools

    if malformation == "incomplete":
        paired_candidates.pop()
    elif malformation == "duplicate":
        paired_candidates.append(paired_candidates[0])
    elif malformation == "mismatch":
        paired_candidates[1]["task_type"] = "other_task"
    elif malformation == "unhashed":
        for item in paired_candidates:
            item["pair_id"] = "not-a-hash"
    else:
        paired_candidates[1]["family"] = "F6"
    monkeypatch.setattr(baselines, "frozen_generic_tools", context)
    rows = baselines.run_baselines(paired_candidates)
    assert all(not row["retained"] and not row["pair_gate"]["valid"] for row in rows)
    assert all(row["pair_gate"]["error"] for row in rows)
    assert all(row["rejected_by"] == [] for row in rows)


def test_runner_passes_only_public_projection_to_detectors(monkeypatch, generic_tools):
    @contextmanager
    def context(root):
        yield generic_tools

    def detector(public, tools):
        assert set(public) <= baselines.PUBLIC_KEYS
        assert "expected_answer" not in public
        assert "abstention_correct" not in public
        return {"bpm": 72}, {"first_output": None}

    monkeypatch.setattr(baselines, "frozen_generic_tools", context)
    monkeypatch.setattr(baselines, "detector_only", detector)
    row = baselines.run_baselines([candidate(72)])[0]
    assert set(row["rejected_by"]) == {"constant", "detector_only"}
    assert not row["retained"]


@pytest.mark.skipif(not (baselines.SERVER_ROOT / "src/tsmcp").exists(), reason="frozen R source not installed")
def test_frozen_imports_actual_primitives_and_restores_process_state():
    import tsmcp

    original_path = sys.path[:]
    original_prefix = sys.prefix
    original_package = tsmcp
    fs = 100.0
    t = np.arange(4000) / fs
    public = {
        "task_type": "heart_rate",
        "answer_fields": ["heart_rate_bpm"],
        "x": np.sin(2 * np.pi * 1.5 * t)[:, None],
        "fs": fs,
        "channels": ["PPG"],
        "units": ["a.u."],
    }
    with baselines.frozen_generic_tools() as tools:
        assert Path(tools["measures"].__file__).is_relative_to(baselines.SERVER_ROOT)
        answer, _ = baselines.detector_only(public, tools)
        assert abs(answer["heart_rate_bpm"] - 90) < 0.5
        assert tools["provenance"]["interpreter"] == sys.executable
        assert sys.prefix == original_prefix
    assert sys.modules["tsmcp"] is original_package
    assert sys.path == original_path
    assert sys.prefix == original_prefix
