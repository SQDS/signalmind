"""Level 2 partition, annotation and strict-continuity gates on synthetic data."""

import json

import h5py
import numpy as np
import pytest

from eval import level2_data as data
from eval.tasks.audit_sources import file_hash, opaque


def partitions(subject="development"):
    parent = {
        "datasets": {
            "pulsedb_mimic": {
                "development": [subject],
                "sealed_confirmation": ["sealed"],
                "candidate_subjects": [subject],
                "records": [],
            }
        }
    }
    level2 = {
        "datasets": {
            "pulsedb_mimic": {"development": [subject], "level2_confirmation": [], "future_reserve": ["sealed"]}
        }
    }
    return parent, level2


def test_level2_sealed_candidate_rejected_before_inventory_or_source_read(monkeypatch):
    parent, level2 = partitions()
    parent["datasets"]["pulsedb_mimic"]["candidate_subjects"].append("sealed")
    monkeypatch.setattr(data, "frozen_splits", lambda: (parent, level2))
    monkeypatch.setattr(data, "inventory", lambda: pytest.fail("inventory accessed before partition gate"))
    with pytest.raises(ValueError, match="sealed or unassigned"):
        next(data.records("pulsedb_mimic", prefer_unused=False))


@pytest.mark.parametrize("dataset", ["uci", "jlu", "pulsedb_vitaldb"])
def test_level2_excluded_sources_never_reach_partition_or_inventory(monkeypatch, dataset):
    monkeypatch.setattr(data, "frozen_splits", lambda: pytest.fail("excluded source accessed"))
    with pytest.raises(ValueError, match="excluded"):
        next(data.records(dataset))


def test_level2_frozen_source_hash_gate_prevents_decoding(monkeypatch, tmp_path):
    parent, level2 = partitions()
    path = tmp_path / "synthetic.mat"
    path.write_bytes(b"changed source")
    parent["datasets"]["pulsedb_mimic"]["records"] = [
        {"subject_key": "development", "record_key": "record", "source_sha256": "frozen"}
    ]
    monkeypatch.setattr(data, "frozen_splits", lambda: (parent, level2))
    monkeypatch.setattr(data, "inventory", lambda: {"pulsedb_mimic": {"development": [path]}})
    monkeypatch.setattr(data, "record_key", lambda *args: "record")
    monkeypatch.setattr(data.h5py, "File", lambda *a, **k: pytest.fail("source decoded before hash gate"))
    with pytest.raises(ValueError, match="changed after split freeze"):
        next(data.records("pulsedb_mimic", prefer_unused=False))


def test_level2_only_matching_development_annotation_row_is_decoded(tmp_path):
    table = tmp_path / "markers.csv"
    table.write_bytes(b'Subject,event\nsealed,\xff\xfe"malformed annotation\ndevelopment,1,2,3\n')
    assert data._selected_marker_row(table, "development") == ["development", "1", "2", "3"]
    assert data._selected_marker_row(table, "absent-development") is None


def synthetic_pulse_file(path, count, gap_at=None):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Subj_Wins")
        fields = ["SubjectID", "CaseID", "SegmentID", "T", "ECG_Record", "PPG_Record", "ABP_Raw"]
        for field in fields:
            references = group.create_dataset(field, (1, count), dtype=h5py.ref_dtype)
            for index in range(count):
                offset = 1 if gap_at is not None and index >= gap_at else 0
                values = {
                    "SubjectID": [ord(c) for c in "synthetic-development"],
                    "CaseID": [ord(c) for c in "synthetic-case"],
                    "SegmentID": [index + offset],
                    "T": np.arange(1250) / 125 + 10 * (index + offset),
                    "ECG_Record": np.full(1250, index),
                    "PPG_Record": np.full(1250, index + 10),
                    "ABP_Raw": np.full(1250, index + 100),
                }
                target = handle.create_dataset(f"values/{field}/{index}", data=values[field])
                references[0, index] = target.ref


@pytest.mark.parametrize("gap_at,expected", [(None, 1), (7, 0)])
def test_level2_long_request_never_falls_back_or_crosses_gap(monkeypatch, tmp_path, gap_at, expected):
    subject = opaque("pulsedb_mimic", "synthetic-development")
    parent, level2 = partitions(subject)
    path = tmp_path / "synthetic.mat"
    synthetic_pulse_file(path, 10, gap_at)
    parent["datasets"]["pulsedb_mimic"]["records"] = [
        {"subject_key": subject, "record_key": "record", "source_sha256": file_hash(path)}
    ]
    monkeypatch.setattr(data, "frozen_splits", lambda: (parent, level2))
    monkeypatch.setattr(data, "inventory", lambda: {"pulsedb_mimic": {subject: [path]}})
    monkeypatch.setattr(data, "record_key", lambda *args: "record")
    found = list(data.records("pulsedb_mimic", duration_s=80, min_duration_s=80, prefer_unused=False))
    assert len(found) == expected
    if expected:
        assert found[0]["duration_s"] == 80
        assert found[0]["x"].shape == (10000, 3)
        assert found[0]["window_indices"] == [0, 7]


def test_level2_wfdb_missing_sentinel_and_compound_units(tmp_path):
    path = tmp_path / "synthetic.hea"
    path.write_text(
        "synthetic 2 100 3\nsynthetic.dat 16 10(0)/mV 16 0 0 0 0 ecg\nsynthetic.dat 16 100(0)/cm/s 16 0 0 0 0 mcar\n"
    )
    np.array([[10, 100], [-32768, 200], [20, 300]], dtype="<i2").tofile(path.with_suffix(".dat"))
    result = data._wfdb_record("cves", path, {"header_sha256": file_hash(path), "protocol": "synthetic"})
    assert result["units"] == ["mV", "cm/s"]
    assert result["channels"] == ["ECG", "CBFV"]
    np.testing.assert_allclose(result["x"][:, 1], [1, 2, 3])
    assert np.isnan(result["x"][1, 0])


def test_level2_prefers_later_long_support_over_first_short_run(monkeypatch, tmp_path):
    subject = opaque("pulsedb_mimic", "synthetic-development")
    parent, level2 = partitions(subject)
    path = tmp_path / "synthetic.mat"
    synthetic_pulse_file(path, 16, gap_at=6)
    parent["datasets"]["pulsedb_mimic"]["records"] = [
        {"subject_key": subject, "record_key": "record", "source_sha256": file_hash(path)}
    ]
    monkeypatch.setattr(data, "frozen_splits", lambda: (parent, level2))
    monkeypatch.setattr(data, "inventory", lambda: {"pulsedb_mimic": {subject: [path]}})
    monkeypatch.setattr(data, "record_key", lambda *args: "record")
    found = next(data.records("pulsedb_mimic", duration_s=100, min_duration_s=60, prefer_unused=False))
    assert found["duration_s"] == 100
    assert found["window_indices"] == [6, 15]


def test_level2_separation_checks_unused_frozen_development_sources(monkeypatch, tmp_path):
    split_path, sealed_path = tmp_path / "split.json", tmp_path / "sealed.json"
    split_path.write_text("synthetic frozen split")
    parent, level2 = partitions()
    parent["datasets"]["pulsedb_mimic"]["records"] = [
        {"subject_key": "development", "record_key": "unused-record", "source_sha256": "colliding-hash"}
    ]
    sealed = {
        "split_sha256": file_hash(split_path),
        "waveform_decoding": False,
        "records": [{"subject_key": "sealed", "source_sha256": "colliding-hash"}],
    }
    sealed_path.write_text(json.dumps(sealed))
    monkeypatch.setattr(data, "PARENT_SPLIT", split_path)
    monkeypatch.setattr(data, "SEALED_HASHES", sealed_path)
    monkeypatch.setattr(data, "frozen_splits", lambda: (parent, level2))
    with pytest.raises(ValueError, match="overlap with sealed"):
        data.separation([], [], sealed_path=sealed_path)


def test_level2_separation_checks_emitted_signal_bytes(monkeypatch, tmp_path):
    split_path, sealed_path = tmp_path / "split.json", tmp_path / "sealed.json"
    split_path.write_text("synthetic frozen split")
    signal = tmp_path / "signal.npy"
    np.save(signal, np.arange(20), allow_pickle=False)
    parent, level2 = partitions()
    parent["datasets"]["pulsedb_mimic"]["records"] = [
        {"subject_key": "development", "record_key": "record", "source_sha256": "different-hash"}
    ]
    sealed_path.write_text(
        json.dumps(
            {
                "split_sha256": file_hash(split_path),
                "waveform_decoding": False,
                "records": [{"subject_key": "sealed", "source_sha256": file_hash(signal)}],
            }
        )
    )
    monkeypatch.setattr(data, "PARENT_SPLIT", split_path)
    monkeypatch.setattr(data, "SEALED_HASHES", sealed_path)
    monkeypatch.setattr(data, "frozen_splits", lambda: (parent, level2))
    with pytest.raises(ValueError, match="overlap with sealed"):
        data.separation([], [signal], sealed_path=sealed_path)
    with pytest.raises(ValueError, match="overlap with sealed"):
        data.separation(
            [],
            [],
            sealed_path=sealed_path,
            ground_truth_sources=[{"development_cleaned_npz_sha256": file_hash(signal)}],
        )
