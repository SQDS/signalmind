"""Integrity gates re-evaluate stored predictions and preserve frozen arm setup."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from eval import audit_checks, common, level2_checks, run_claude_code
from eval.level2_baselines import CONSTANT_ABSTENTION
from eval.level2_grade import grade_level2


def numerical_item():
    item = {
        "family": "F1",
        "expected_answer": {"value": 10.0},
        "answer_fields": ["value"],
        "tolerances": {"value": 1.0},
        "ground_truth": {
            "category": "G4",
            "reference1": {"value": 10.0},
            "reference2": {"value": 10.4},
            "agreement": {"accepted": True},
        },
    }
    predictions = {
        "constant": {"value": 0},
        "development_median": None,
        "detector_only": None,
        "constant_abstention": dict(CONSTANT_ABSTENTION),
    }
    item["baselines"] = {
        name: {"answer": answer, "grade": grade_level2(answer, item)} for name, answer in predictions.items()
    }
    item["baselines"]["rejected_by"] = []
    return item


def test_reference_gate_recomputes_half_tolerance_from_values():
    item = numerical_item()
    level2_checks._validate_reference_agreement(item)
    item["ground_truth"]["reference2"]["value"] = 10.6
    with pytest.raises(ValueError, match="exceed half tolerance"):
        level2_checks._validate_reference_agreement(item)


def test_relative_reference_gate_and_exact_sequence_counts():
    item = numerical_item()
    item["tolerances"]["value"] = {"relative": 0.05}
    item["ground_truth"]["reference2"]["value"] = 10.25
    level2_checks._validate_reference_agreement(item)
    item["ground_truth"]["reference2"]["value"] = 10.251
    with pytest.raises(ValueError, match="exceed half tolerance"):
        level2_checks._validate_reference_agreement(item)
    item["tolerances"]["value"] = {"absolute": 0}
    item["ground_truth"]["reference2"]["value"] = 10
    level2_checks._validate_reference_agreement(item)


def test_g4_missing_reference_and_dishonest_recorded_agreement_fail():
    item = numerical_item()
    del item["ground_truth"]["reference2"]
    with pytest.raises(ValueError, match="values are missing"):
        level2_checks._validate_reference_agreement(item)
    item["ground_truth"] = {
        "category": "G2",
        "agreement": {"accepted": True, "difference_bpm": 2, "half_tolerance": 0.75},
    }
    with pytest.raises(ValueError, match="exceeds half tolerance"):
        level2_checks._validate_reference_agreement(item)


def test_empty_rejection_ledger_cannot_override_passing_or_stale_baseline():
    item = numerical_item()
    level2_checks._validate_baselines(item)
    item["baselines"]["constant"]["answer"] = {"value": 10}
    with pytest.raises(ValueError, match="Stored baseline grade"):
        level2_checks._validate_baselines(item)
    item["baselines"]["constant"]["grade"] = grade_level2({"value": 10}, item)
    with pytest.raises(ValueError, match="passes a preregistered baseline"):
        level2_checks._validate_baselines(item)


def f6_item():
    item = {
        "family": "F6",
        "answer_fields": ["bpm"],
        "expected_answer": {"bpm": 70.0},
        "tolerances": {"bpm": 1.5},
        "duration_s": 180,
        "misleading_output": {"operation": "dominant_freq", "channel": "PPG", "output": {"value": 2.0}},
    }
    predictions = {
        "constant": {"bpm": 72},
        "development_median": None,
        "detector_only": {"bpm": 120},
        "constant_abstention": dict(CONSTANT_ABSTENTION),
    }
    item["baselines"] = {
        name: {"answer": answer, "grade": grade_level2(answer, item)} for name, answer in predictions.items()
    }
    item["baselines"]["rejected_by"] = []
    item["baselines"]["detector_only"]["first_output"] = copy.deepcopy(item["misleading_output"])
    return item


def test_f6_requires_actual_initial_output_to_match_recorded_prediction():
    item = f6_item()
    level2_checks._validate_baselines(item)
    item["misleading_output"]["output"]["value"] = 1.5
    with pytest.raises(ValueError, match="actual shortcut prediction"):
        level2_checks._validate_baselines(item)


@pytest.mark.parametrize("frequency", [None, float("nan"), 0.2, 4.0])
def test_f6_first_rate_must_be_finite_and_plausible(frequency):
    item = f6_item()
    item["misleading_output"]["output"]["value"] = frequency
    with pytest.raises(ValueError, match="initial"):
        level2_checks._initial_prediction(item)


def test_f6_wrong_onset_must_point_to_recorded_transient():
    item = {
        "family": "F6",
        "task_type": "verified_onset",
        "answer_fields": ["onset_s"],
        "duration_s": 180,
        "perturbation": {"distractor_s": 90, "distractor_duration_s": 4},
        "misleading_output": {"operation": "change", "channel": "ABP", "output": {"t": 90}},
    }
    assert level2_checks._initial_prediction(item) == {"onset_s": 90}
    for onset in (84.9, 99.1):
        item["misleading_output"]["output"]["t"] = onset
        with pytest.raises(ValueError, match="recorded misleading transient"):
            level2_checks._initial_prediction(item)
    item["misleading_output"]["output"]["t"] = 180
    with pytest.raises(ValueError, match="outside the supplied recording"):
        level2_checks._initial_prediction(item)


def dry_fixture(tmp_path, monkeypatch, *, break_prepare=False, bad_signal=False, home_pass=True):
    monkeypatch.setattr(level2_checks, "ROOT", tmp_path)
    items, plans = [], []
    for number, names in enumerate((["ECG"], ["ECG", "PPG"])):
        directory = tmp_path / "eval/cases/p9level2" / str(number)
        directory.mkdir(parents=True)
        np.save(directory / "signal.npy", np.zeros((20, len(names))), allow_pickle=False)
        visible = {
            "file": "signal.npy",
            "dur_s": 20,
            "channels": [{"name": name, "fs": 1.0, "unit": "a.u."} for name in names],
        }
        (directory / "visible.json").write_text(json.dumps(visible))
        items.append(
            {
                "case_id": str(number),
                "channels": names,
                "answer_fields": ["bpm"],
                "question": "Mean cardiac rate?",
                "signal_sha256": level2_checks.sha(directory / "signal.npy"),
                "visible_sha256": level2_checks.sha(directory / "visible.json"),
            }
        )

    def prepare(case, arm, *args, **kwargs):
        if break_prepare:
            raise RuntimeError("preparation fixture failure")
        base = tmp_path / "plans" / str(len(plans))
        work = base / "work"
        work.mkdir(parents=True)
        plans.append(base)
        shutil.copyfile(Path(case["dir"]) / "signal.npy", work / "signal.npy")
        if bad_signal and arm == "D":
            (work / "signal.npy").write_bytes(b"wrong public signal")
        return {
            "base": base,
            "cwd": work,
            "command": ["offline-fixture", "--effort", "medium"],
            "stdin": common.case_prompt(case["visible"], case["question"], "level2"),
            "mcp_config": {"mcpServers": {"tsmcp": {}}},
        }

    def probe(command, overrides, work, repo):
        assert (work / "visible.json").is_file()
        return {"status": "pass" if home_pass else "fail", "home": "not-reported"}

    monkeypatch.setattr(run_claude_code, "prepare_case", prepare)
    monkeypatch.setattr(audit_checks, "_probe_home", probe)
    return items, plans


def test_both_dry_arms_receive_identical_public_files_and_registry_restores(tmp_path, monkeypatch):
    items, plans = dry_fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(common.ANSWER_FORMATS, "level2", "existing registry value")
    result = level2_checks.dry_runs(items)
    assert result["model_calls"] == 0
    assert result["configuration_pass"]
    assert len(result["runs"]) == 4
    assert all(row["checks"]["same_visible_bytes"] and row["checks"]["same_signal_bytes"] for row in result["runs"])
    assert common.ANSWER_FORMATS["level2"] == "existing registry value"
    assert all(not path.exists() for path in plans)


def test_preparation_exception_restores_absent_registry(tmp_path, monkeypatch):
    items, _ = dry_fixture(tmp_path, monkeypatch, break_prepare=True)
    monkeypatch.delitem(common.ANSWER_FORMATS, "level2", raising=False)
    with pytest.raises(RuntimeError, match="fixture failure"):
        level2_checks.dry_runs(items)
    assert "level2" not in common.ANSWER_FORMATS


def test_home_probe_failure_cannot_hide_wrong_signal_bytes(tmp_path, monkeypatch):
    items, _ = dry_fixture(tmp_path, monkeypatch, bad_signal=True, home_pass=False)
    result = level2_checks.dry_runs(items)
    assert not result["configuration_pass"]
    assert all(row["status"] == "fail" for row in result["runs"] if row["arm"] == "D")
