"""Synthetic cache-integrity and publication checks; no source dataset is read."""

import copy
import json

import numpy as np
import pytest

from eval import level2_checks
from eval.level2_baselines import CONSTANT_ABSTENTION, apply_pair_gate, development_median
from eval.level2_grade import grade_level2
from eval.tasks import p9level2 as builder


def candidate(index=0, value=10.0):
    return {
        "dataset": "synthetic",
        "subject_key": f"synthetic-{index}",
        "record_key": f"synthetic-{index}",
        "source_hashes": {},
        "start_s": 0,
        "family": "F4",
        "task_type": "fixture_quantity",
        "question": "Estimate the stated test quantity.",
        "x": np.zeros((20, 1), dtype=np.float32),
        "fs": 1.0,
        "channels": ["ECG"],
        "units": ["mV"],
        "expected_answer": {"value": value},
        "tolerances": {"value": 0.5},
        "expected_solvability": "specialist",
        "signal_kind": "real_signal",
        "ground_truth": {
            "category": "G4",
            "agreement": {"accepted": True},
            "reference1": {"value": value},
            "reference2": {"value": value},
        },
    }


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "CACHE", tmp_path / "eval/cache/level2")
    monkeypatch.setattr(builder, "MANIFEST", tmp_path / "eval/tasks/p9level2_manifest.json")
    monkeypatch.setattr(builder, "RESULTS", tmp_path / "eval/results/level2-build")
    monkeypatch.setattr(builder, "cache_binding", lambda part: {"source_version": 1, "part": part})
    monkeypatch.setattr(level2_checks, "ROOT", tmp_path)
    monkeypatch.setattr(level2_checks, "separation", lambda *args, **kwargs: {"fixture_only": True})
    (tmp_path / "eval/tasks").mkdir(parents=True)
    (tmp_path / "eval/tasks/level2_splits.json").write_text("{}")
    (tmp_path / "eval/level2_baselines.py").write_text("# synthetic provenance fixture\n")

    def write(part, rows):
        directory = builder.CACHE / part
        directory.mkdir(parents=True, exist_ok=True)
        metadata = []
        for index, original in enumerate(rows):
            row = copy.deepcopy(original)
            path = directory / f"{index:03d}.npy"
            np.save(path, row.pop("x"), allow_pickle=False)
            row["signal_cache"] = str(path.relative_to(tmp_path))
            row["cache_signal_sha256"] = builder.sha(path)
            metadata.append(row)
        builder.dump(directory / "candidates.json", metadata)
        builder.dump(directory / "exclusions.json", [])
        builder.dump(
            directory / "generation.json",
            {
                "binding": builder.cache_binding(part),
                "metadata_sha256": builder.sha(directory / "candidates.json"),
                "exclusions_sha256": builder.sha(directory / "exclusions.json"),
            },
        )
        return directory

    for part in ("physiology", "perturb", "manual"):
        write(part, [candidate()] if part == "physiology" else [])
    return write


def test_valid_cache_recovers_public_shape_and_schema(cache):
    rows, exclusions = builder.load_candidates()
    assert len(rows) == 1 and exclusions == []
    assert rows[0]["x"].shape == (20, 1)
    assert rows[0]["answer_fields"] == ["value"]
    assert rows[0]["duration_s"] == 20
    assert "column 0 = ECG" in rows[0]["question"]


@pytest.mark.parametrize("tamper", ["missing_stamp", "binding", "metadata", "exclusions"])
def test_generation_binding_rejects_tampering_before_waveform_decode(cache, monkeypatch, tamper):
    directory = builder.CACHE / "physiology"
    if tamper == "missing_stamp":
        (directory / "generation.json").unlink()
    elif tamper == "binding":
        stamp = json.loads((directory / "generation.json").read_text())
        stamp["binding"]["source_version"] = 2
        builder.dump(directory / "generation.json", stamp)
    else:
        filename = "candidates.json" if tamper == "metadata" else "exclusions.json"
        path = directory / filename
        path.write_text(path.read_text() + "\n")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("tampered metadata reached waveform decoding")

    monkeypatch.setattr(builder.np, "load", forbidden_decode)
    with pytest.raises(ValueError, match="binding|changed"):
        builder.load_candidates()


def test_signal_hash_mismatch_fails_before_numpy_decode(cache, monkeypatch):
    path = builder.CACHE / "physiology/000.npy"
    path.write_bytes(b"tampered bytes must not reach a decoder")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("tampered waveform reached numpy")

    monkeypatch.setattr(builder.np, "load", forbidden_decode)
    with pytest.raises(ValueError, match="signal hash changed"):
        builder.load_candidates()


def test_missing_generation_part_fails_closed(cache):
    (builder.CACHE / "manual/candidates.json").unlink()
    with pytest.raises(ValueError, match="generation cache missing"):
        builder.load_candidates()


def synthetic_baselines(candidates):
    rows = []
    for index, item in enumerate(candidates):
        median, support = development_median(candidates, index)
        answers = {
            "constant": {"value": 1000},
            "development_median": median,
            "detector_only": None,
            "constant_abstention": dict(CONSTANT_ABSTENTION),
        }
        result = {name: {"answer": answer, "grade": grade_level2(answer, item)} for name, answer in answers.items()}
        result["development_median"]["peer_support"] = support
        result["rejected_by"] = [name for name in answers if result[name]["grade"]["correct"]]
        result["retained"] = not result["rejected_by"]
        rows.append(result)
    return apply_pair_gate(candidates, rows)


def test_freeze_rechecks_retained_median_and_keeps_answers_outside_case_dirs(cache, monkeypatch, capsys):
    # Initial LOO medians reject both 1s. Removing them makes 2 shortcut-solvable;
    # the next iteration must remove it too, leaving the distinct endpoints.
    cache("physiology", [candidate(i, value) for i, value in enumerate((0, 1, 1, 2, 3))])
    monkeypatch.setattr(builder, "run_baselines", synthetic_baselines)
    manifest = builder.freeze(run_dry=False)
    capsys.readouterr()
    assert manifest["final_counts"] == {"F4": 2}
    assert manifest["retained_pool_median_iterations"] == [
        {"pool_size": 3, "rejections": 1},
        {"pool_size": 2, "rejections": 0},
    ]
    assert [row["expected_answer"]["value"] for row in manifest["questions"]] == [0, 3]
    assert sum(row["stage"] == "shortcut_baseline" for row in manifest["exclusions"]) == 2
    assert sum(row["stage"] == "retained_pool_median" for row in manifest["exclusions"]) == 1
    case_root = builder.ROOT / "eval/cases/p9level2"
    for directory in case_root.iterdir():
        assert sorted(path.name for path in directory.iterdir()) == ["signal.npy", "visible.json"]
        visible = json.loads((directory / "visible.json").read_text())
        assert set(visible) == {"file", "dur_s", "channels"}
    answers = builder.ROOT / "eval/cases/p9level2.answers.jsonl"
    questions = builder.ROOT / "eval/cases/p9level2.questions.jsonl"
    assert not answers.is_relative_to(case_root)
    assert not questions.is_relative_to(case_root)
    answer_rows = [json.loads(line) for line in answers.read_text().splitlines()]
    question_rows = [json.loads(line) for line in questions.read_text().splitlines()]
    assert [row["answer"]["value"] for row in answer_rows] == [0, 3]
    assert all(set(row) == {"case_id", "qtype", "question"} for row in question_rows)
    assert manifest["answers_sha256"] == builder.sha(answers)
    assert manifest["questions_sha256"] == builder.sha(questions)


def paired_candidates(index=0):
    control = candidate(index)
    control.update(
        family="F5",
        task_type="pair_fixture",
        pair_id=f"{index:064x}",
        pair_role="control",
        abstention_correct=False,
        answer_fields=["value"],
    )
    control["x"][:, 0] = np.arange(20)
    abstention = copy.deepcopy(control)
    abstention.update(
        pair_role="abstention",
        abstention_correct=True,
        expected_answer=dict(CONSTANT_ABSTENTION),
        tolerances={},
        expected_solvability="abstention",
        signal_kind="controlled_perturbation",
        ground_truth={
            "category": "GT-abstain",
            "agreement": {"accepted": True},
            "perturbation": {"kind": "entire_channel_hold", "channel": "ECG", "value": 0.0},
        },
    )
    abstention["x"][:] = 0
    return [control, abstention]


def test_pair_freeze_retains_individual_shortcuts_but_requires_pair_correct(cache, monkeypatch, capsys):
    cache("physiology", paired_candidates() + paired_candidates(1))
    monkeypatch.setattr(builder, "run_baselines", synthetic_baselines)
    manifest = builder.freeze(run_dry=False)
    capsys.readouterr()
    assert manifest["final_counts"] == {"F5": 4}
    assert manifest["abstention_pair_count"] == 2
    assert manifest["separation"]["counterfactual_pairs_checked"] == 2
    assert manifest["separation"]["one_subject_per_family_or_counterfactual_pair"]
    assert not manifest["separation"]["one_subject_per_family"]
    assert all(pair["oracle_grade"]["pair_correct"] for pair in manifest["abstention_pairs"])
    for row in manifest["questions"]:
        baseline = "constant_abstention" if row["abstention_correct"] else "development_median"
        assert row["baselines"][baseline]["grade"]["correct"]
        assert not row["baselines"]["rejected_by"]
        assert not row["retained_pool_median"]["pair_grade"]["pair_correct"]


def test_pair_median_rejects_both_members_together(cache, monkeypatch):
    cache("physiology", paired_candidates())
    items, _ = builder.load_candidates()
    monkeypatch.setattr(builder, "development_median", lambda rows, index: (rows[index]["expected_answer"], {}))
    assert builder.retained_median_gate(items) == {row["case_id"] for row in items}


def test_incomplete_pair_fails_before_publication(cache):
    cache("physiology", paired_candidates()[:1])
    with pytest.raises(ValueError, match="pair"):
        builder.load_candidates()


def test_counterfactual_check_detects_unrelated_sample_changes(cache, monkeypatch, capsys):
    rows = paired_candidates()
    for row in rows:
        row["channels"] = ["ECG", "ABP"]
        row["units"] = ["mV", "mmHg"]
        row["x"] = np.column_stack((row["x"], np.ones(20)))
    rows[1]["x"][4, 1] = 2
    cache("physiology", rows)
    monkeypatch.setattr(builder, "run_baselines", synthetic_baselines)
    with pytest.raises(ValueError, match="unrelated channel"):
        builder.freeze(run_dry=False)
    capsys.readouterr()


def test_retry_allocation_uses_next_rank_without_reusing_subject():
    rows = []
    for index, (slot, rank, subject) in enumerate(((6, 1, 0), (6, 2, 1), (7, 1, 0), (7, 2, 2))):
        row = candidate(subject)
        row.update(retry_candidate=True, design_slot=slot, retry_rank=rank, case_id=str(index))
        rows.append(row)
    assert {row["case_id"] for row in builder.select_retry_slots(rows)} == {"0", "3"}
    assert {row["case_id"] for row in builder.select_retry_slots(rows[1:])} == {"1", "2"}
    # The later slot has only one subject; reassign the earlier slot to its
    # alternative instead of leaving an otherwise fillable slot empty.
    assert {row["case_id"] for row in builder.select_retry_slots(rows[:3])} == {"1", "2"}
