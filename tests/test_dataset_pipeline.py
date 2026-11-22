"""A reusable policy must generalize across inputs and retain complete source coverage."""

import asyncio
import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.dataset_tools import DatasetTools  # noqa: E402
from eval.dataset_workflow import build_project, read, run_batch, trial  # noqa: E402
from tsmcp.pipeline import SCHEMA, apply_policy, policy_hash, validate_policy  # noqa: E402


def tone_policy():
    rule = {
        "id": "observed_line",
        "operation": "remove_tone",
        "selector": {"kind": "all"},
        "when": {"line_50_fraction": {"min": 0.05}, "line_50_ratio": {"min": 10}},
        "parameters": {"frequency_hz": 50},
    }
    return {
        "schema_version": SCHEMA,
        "goal": "Remove supported line interference and preserve amplitude",
        "channels": {kind: [copy.deepcopy(rule)] for kind in ("ECG", "PPG", "ABP")},
    }


def test_shared_policy_adapts_to_clean_and_tone_inputs_and_preserves_pressure():
    t = np.arange(5000) / 125
    clean = 90 + 5 * np.sin(2 * np.pi * t)
    noisy = clean + 2 * np.sin(2 * np.pi * 50 * t)
    policy = validate_policy(tone_policy())
    keep, changed, _ = apply_policy(clean, 125, "ABP", "mmHg", policy)
    np.testing.assert_array_equal(keep.values, clean)
    assert not changed.any()
    output, _, diag = apply_policy(noisy, 125, "ABP", "mmHg", policy)
    np.testing.assert_allclose(output.values, clean, atol=1e-10)
    assert abs(diag["common_support_mean_shift"]) < 1e-10
    assert output.valid.all() and len(output.operations) == 1


def test_short_outlier_repair_and_long_outlier_mask_keep_other_waveform_exact():
    x = 90 + 5 * np.sin(2 * np.pi * np.arange(2500) / 125)
    noisy = x.copy()
    noisy[400:402] = -200
    noisy[900:925] = -200
    policy = {
        "schema_version": SCHEMA,
        "goal": "local changes",
        "max_repair_ms": 32,
        "channels": {
            "ABP": [
                {
                    "id": "outliers",
                    "operation": "interpolate",
                    "fallback": "mark_invalid",
                    "selector": {"kind": "outliers", "factor": 2},
                }
            ]
        },
    }
    validate_policy(policy)
    result, changed, diag = apply_policy(noisy, 125, "ABP", "mmHg", policy)
    assert np.flatnonzero(changed).tolist() == [400, 401]
    assert result.valid[400:402].all() and not result.valid[900:925].any()
    np.testing.assert_array_equal(result.values[900:925], noisy[900:925])
    np.testing.assert_array_equal(result.values[~changed], noisy[~changed])
    assert diag["output_invalid_samples"] == 25


def test_policy_requires_explicit_channel_and_preserves_missing_support():
    policy = {"schema_version": SCHEMA, "goal": "preserve", "channels": {"ABP": []}}
    x = np.arange(100, dtype=float)
    x[20:30] = np.nan
    state, changed, diag = apply_policy(x, 125, "ABP", "mmHg", validate_policy(policy))
    np.testing.assert_array_equal(state.values, x)
    assert not changed.any() and diag["input_invalid_samples"] == diag["output_invalid_samples"] == 10
    with pytest.raises(ValueError, match="explicitly cover"):
        apply_policy(x, 125, "ECG", "source_units", policy)


def write_vital(path):
    """Two continuous 40s runs with a real 70s gap; all stored row IDs remain adjacent."""
    with h5py.File(path, "w") as file:
        group = file.create_group("Subj_Wins")
        fields = ("SubjectID", "CaseID", "SegmentID", "T", "ECG_Record", "PPG_Record", "ABP_Raw")
        datasets = {name: group.create_dataset(name, (1, 8), dtype=h5py.ref_dtype) for name in fields}
        for i in range(8):
            t = np.arange(1250) / 125 + 10 * i + (70 if i >= 4 else 0)
            wave = 90 + 5 * np.sin(2 * np.pi * t) + 2 * np.sin(2 * np.pi * 50 * t)
            values = {
                "SubjectID": np.frombuffer(path.stem.encode("utf-16-le"), dtype="<u2"),
                "CaseID": np.frombuffer("full-case-identity".encode("utf-16-le"), dtype="<u2"),
                "SegmentID": [i],
                "T": t,
                "ECG_Record": wave / 100,
                "PPG_Record": wave / 2,
                "ABP_Raw": wave,
            }
            for name, value in values.items():
                datasets[name][0, i] = file.create_dataset(f"{name}_{i}", data=np.asarray(value)[None, :]).ref


@pytest.fixture
def project(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("p1.mat", "p2.mat"):
        write_vital(source / name)
    project = tmp_path / "project"
    build_project(source, project, development_records=1, validation_records=1)
    return project


def test_dataset_tools_hide_validation_and_freeze_only_tried_policies(project):
    tools = DatasetTools(project)
    blocks, error = asyncio.run(tools.call("dataset_overview", {}))
    assert not error
    overview = json.loads(blocks[0]["text"])
    assert overview["inventory"]["units"]["ECG"] == "source_units"
    assert overview["inventory"]["device_saturation_flags_available"] is False
    manifest = read(project / "private/manifest.json")
    assert set(manifest["development_records"]).isdisjoint(manifest["validation_records"])
    samples = read(project / "private/samples.json")
    hidden = next(s["id"] for s in samples if s["role"] == "validation")
    blocks, error = asyncio.run(tools.call("dataset_inspect", {"sample_id": hidden, "channel": "ABP"}))
    assert error
    blocks, error = asyncio.run(tools.call("pipeline_trial", {"policy": tone_policy()}))
    assert not error
    tid = json.loads(blocks[0]["text"])["trial_id"]
    _, error = asyncio.run(tools.call("pipeline_finalize", {"trial_id": tid, "rationale": "fixture only"}))
    assert not error
    assert policy_hash(read(project / "policy.json")) == tid
    assert not read(project / "policy-freeze.json")["validation_seen_during_tuning"]
    _, error = asyncio.run(tools.call("pipeline_trial", {"policy": tone_policy()}))
    assert error


def test_batch_covers_entire_records_preserves_gaps_and_resumes_without_model_calls(project):
    policy = tone_policy()
    report = run_batch(project, policy)
    assert report["completed_records"] == 2 and report["stored_rows_processed"] == 16
    assert report["new_model_calls"] == 0 and not report["failed"]
    assert report["scope"] == "entire_directory"
    folder = project / "batch" / policy_hash(policy)
    for record in report["records"]:
        output = folder / record["record_id"]
        files = sorted(output.glob("run_*.npz"))
        assert len(files) == 2
        times = []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_allclose(np.diff(data["time"]), 0.008, atol=1e-12)
                assert data["values"].shape == (3, 5000)
                clean = 90 + 5 * np.sin(2 * np.pi * data["time"])
                np.testing.assert_allclose(data["values"][2], clean, atol=1e-9)
                times.append(data["time"].copy())
        assert times[1][0] - times[0][-1] == pytest.approx(70.008)
    resumed = run_batch(project, policy)
    assert resumed["records"] == report["records"]
    validation = trial(project, policy, "validation")
    assert validation["summary"]["channel_segments"] == 6
    assert validation["summary"]["quality_verdict"] is None


def test_resume_reports_modified_outputs_instead_of_silently_reusing_them(project):
    policy = tone_policy()
    report = run_batch(project, policy)
    first = report["records"][0]
    folder = project / "batch" / policy_hash(policy) / first["record_id"]
    next(folder.glob("run_*.npz")).write_bytes(b"changed")
    resumed = run_batch(project, policy)
    assert resumed["completed_records"] == 1
    assert len(resumed["failed"]) == 1
    assert "artifact changed" in resumed["failed"][0]["error"]
