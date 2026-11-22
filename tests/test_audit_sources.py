"""Audit source invariants using synthetic metadata and waveform fixtures only."""

import csv
import hashlib
import json

import h5py
import numpy as np
import pytest

from eval.tasks.audit_sources import _pulse_runs, _read_wfdb_dev, diabetes_events, file_hash, opaque


def test_sealed_candidate_rejected_before_any_source_hashing(monkeypatch, tmp_path):
    from eval.tasks import audit_sources as sources

    reads = []
    monkeypatch.setattr(sources, "file_hash", lambda path: reads.append(path))
    monkeypatch.setattr(sources, "inventory", lambda: {"pulsedb_mimic": {"dev": [tmp_path / "dev.mat"]}})
    manifest = {
        "datasets": {
            "pulsedb_mimic": {
                "development": ["dev"],
                "sealed_confirmation": ["sealed"],
                "candidate_subjects": ["dev", "sealed"],
                "records": [],
            }
        }
    }
    with pytest.raises(ValueError, match="sealed"):
        next(sources.iter_dev_windows("pulsedb_mimic", manifest))
    assert reads == []


def test_source_hash_failure_prevents_waveform_decode(monkeypatch, tmp_path):
    from eval.tasks import audit_sources as sources

    monkeypatch.setattr(sources, "file_hash", lambda path: "changed")
    monkeypatch.setattr(sources, "record_key", lambda dataset, path: "record")
    monkeypatch.setattr(sources, "inventory", lambda: {"pulsedb_mimic": {"dev": [tmp_path / "dev.mat"]}})
    monkeypatch.setattr(sources.h5py, "File", lambda *args, **kwargs: pytest.fail("waveform opened before hash gate"))
    manifest = {
        "datasets": {
            "pulsedb_mimic": {
                "development": ["dev"],
                "sealed_confirmation": [],
                "candidate_subjects": ["dev"],
                "records": [{"record_key": "record", "source_sha256": "frozen"}],
            }
        }
    }
    with pytest.raises(ValueError, match="changed after split freeze"):
        next(sources.iter_dev_windows("pulsedb_mimic", manifest))


def pulse_metadata(path, windows):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Subj_Wins")
        for field in ["SubjectID", "CaseID", "SegmentID", "T"]:
            references = group.create_dataset(field, (1, len(windows)), dtype=h5py.ref_dtype)
            for index, window in enumerate(windows):
                value = window[field]
                if isinstance(value, str):
                    value = [ord(c) for c in value]
                dataset = handle.create_dataset(f"data/{field}/{index}", data=np.asarray(value))
                references[0, index] = dataset.ref


def windows(count):
    return [
        {
            "SubjectID": "synthetic-subject",
            "CaseID": "synthetic-case",
            "SegmentID": [i],
            "T": np.arange(1250) / 125 + 10 * i,
        }
        for i in range(count)
    ]


@pytest.mark.parametrize("break_kind", ["case", "segment", "boundary", "internal_time"])
def test_pulse_runs_never_join_metadata_or_time_gaps(tmp_path, break_kind):
    metadata = windows(4)
    if break_kind == "case":
        metadata[2]["CaseID"] = "other-synthetic-case"
    elif break_kind == "segment":
        metadata[2]["SegmentID"] = [99]
    elif break_kind == "boundary":
        metadata[2]["T"] = metadata[2]["T"] + 0.016
    else:
        metadata[2]["T"][500] += 0.001
    path = tmp_path / "synthetic.mat"
    pulse_metadata(path, metadata)
    with h5py.File(path) as handle:
        runs = _pulse_runs(handle, opaque("pulsedb_mimic", "synthetic-subject"))
    assert runs[0] == [0, 1]
    assert all(not (1 in run and 2 in run) for run in runs)
    if break_kind == "internal_time":
        assert all(2 not in run for run in runs)


def test_pulse_runs_subject_mismatch_is_fatal_and_private(tmp_path):
    metadata = windows(2)
    metadata[1]["SubjectID"] = "other-synthetic-subject"
    path = tmp_path / "synthetic.mat"
    pulse_metadata(path, metadata)
    with h5py.File(path) as handle, pytest.raises(ValueError, match="subject mismatch") as exc:
        _pulse_runs(handle, opaque("pulsedb_mimic", "synthetic-subject"))
    assert "other-synthetic-subject" not in str(exc.value)


def test_first_qualifying_pulse_run_caps_without_crossing_gap(tmp_path):
    metadata = windows(25)
    metadata[13]["CaseID"] = "other-synthetic-case"
    path = tmp_path / "synthetic.mat"
    pulse_metadata(path, metadata)
    with h5py.File(path) as handle:
        runs = _pulse_runs(handle, opaque("pulsedb_mimic", "synthetic-subject"), first_min_windows=12)
    assert runs == [list(range(13))]


def test_wfdb_reader_preserves_compound_units_and_missing_values(tmp_path):
    path = tmp_path / "synthetic.hea"
    path.write_text(
        "synthetic 2 100 3\nsynthetic.dat 16 10(0)/mV 16 0 0 0 0 ecg\nsynthetic.dat 16 100(0)/cm/s 16 0 0 0 0 mcar\n"
    )
    np.array([[10, 100], [-32768, 200], [20, 300]], dtype="<i2").tofile(path.with_suffix(".dat"))
    metadata = {"header_sha256": file_hash(path), "protocol": "synthetic"}
    result = _read_wfdb_dev("cves", path, metadata)
    assert result["units"] == ["mV", "cm/s"]
    np.testing.assert_allclose(result["x"][:, 1], [1, 2, 3])
    assert np.isnan(result["x"][1, 0])
    with path.with_suffix(".dat").open("ab") as stream:
        stream.write(b"\0\0")
    with pytest.raises(ValueError, match="size differs"):
        _read_wfdb_dev("cves", path, metadata)


def test_diabetes_ordinal_alignment_rejects_unmatched_pulse_count(tmp_path, monkeypatch):
    from eval.tasks import audit_sources

    monkeypatch.setitem(audit_sources.ROOTS, "diabetes", tmp_path)
    description = tmp_path / "Data_Description"
    description.mkdir()
    table = description / "GE-71_Head-up-tilt-Day1_Markers_per_subject.csv"
    with table.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject ID", "Group"] + ["event"] * 11 + ["Notes"])
        writer.writerow(["synthetic", "", "1", "0", "0", "2", "3", "0", "0", "0", "0", "4", "5", ""])
    path = tmp_path / "Head-up-tilt_Day1" / "syntheticDA.hea"
    marker = np.zeros(2000)
    for index in [100, 400, 700, 1000, 1300]:
        marker[index : index + 20] = 1
    diagnostics = {}
    events = diabetes_events(path, marker, 100, diagnostics)
    assert diagnostics["status"] == "accepted"
    assert [(event["kind"], event["t"]) for event in events] == [("valsalva", 4), ("tilt", 10)]
    marker[1600:1620] = 1
    assert diabetes_events(path, marker, 100, diagnostics) == []
    assert diagnostics["status"] == "excluded"
    assert diagnostics["detected_pulses"] == 6


def test_sealed_inventory_hashes_raw_bytes_without_decoding_or_changing_split(tmp_path, monkeypatch):
    from eval.tasks import audit_sealed, audit_sources

    source = tmp_path / "opaque.mat"
    source.write_bytes(b"deliberately not an HDF5 file")
    header = tmp_path / "opaque.hea"
    header.write_bytes(b"deliberately not a WFDB header")
    header.with_suffix(".dat").write_bytes(b"raw waveform file bytes")
    paths = {"pulsedb_mimic": {"sealed": [source]}, "cves": {"sealed": [header]}}
    monkeypatch.setattr(audit_sealed, "inventory", lambda: paths)
    monkeypatch.setattr(audit_sealed, "record_key", lambda dataset, path: "record")
    monkeypatch.setattr(audit_sources.h5py, "File", lambda *a, **k: pytest.fail("sealed HDF5 decoded"))
    monkeypatch.setattr(audit_sources, "read_header", lambda *a, **k: pytest.fail("sealed header parsed"))
    monkeypatch.setattr(audit_sources.np, "load", lambda *a, **k: pytest.fail("sealed NumPy array decoded"))
    splits = {
        "datasets": {
            dataset: {
                "development": ["dev"],
                "sealed_confirmation": ["sealed"],
                "records": [
                    {
                        "subject_key": "sealed",
                        "record_key": "record",
                        **({"header_sha256": file_hash(header)} if dataset == "cves" else {}),
                    }
                ],
            }
            for dataset in paths
        }
    }
    split_path, output = tmp_path / "split.json", tmp_path / "sealed.json"
    split_path.write_text(json.dumps(splits))
    original_split = split_path.read_bytes()
    progress = []
    result = audit_sealed.freeze_sealed_hashes(output, split_path, workers=2, progress=lambda *p: progress.append(p))
    assert split_path.read_bytes() == original_split
    assert result["waveform_decoding"] is False
    assert result["record_count"] == 2 and result["raw_file_count"] == 3
    assert result["split_sha256"] == hashlib.sha256(original_split).hexdigest()
    assert result["records"][0]["source_sha256"] == file_hash(source)
    assert result["records"][1]["source_sha256"] == file_hash(header.with_suffix(".dat"))
    assert progress == [(1, 2), (2, 2)]
    assert str(tmp_path) not in output.read_text()


def test_sealed_inventory_rejects_missing_records_before_hashing_sources(tmp_path, monkeypatch):
    from eval.tasks import audit_sealed

    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "pulsedb_mimic": {
                        "development": [],
                        "sealed_confirmation": ["sealed"],
                        "records": [{"subject_key": "sealed", "record_key": "missing"}],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(audit_sealed, "inventory", lambda: {"pulsedb_mimic": {}})
    reads = []
    monkeypatch.setattr(audit_sealed, "file_hash", lambda path: reads.append(path) or "hash")
    with pytest.raises(ValueError, match="coverage differs"):
        audit_sealed.freeze_sealed_hashes(tmp_path / "sealed.json", split_path)
    assert reads == [split_path]


def test_sealed_inventory_rejects_published_checksum_mismatch(tmp_path, monkeypatch):
    from eval.tasks import audit_sealed

    source = tmp_path / "opaque.mat"
    source.write_bytes(b"synthetic raw source")
    monkeypatch.setattr(audit_sealed, "inventory", lambda: {"pulsedb_mimic": {"sealed": [source]}})
    monkeypatch.setattr(audit_sealed, "record_key", lambda *a: "record")
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "pulsedb_mimic": {
                        "development": [],
                        "sealed_confirmation": ["sealed"],
                        "records": [
                            {"subject_key": "sealed", "record_key": "record", "published_source_sha256": "0" * 64}
                        ],
                    }
                }
            }
        )
    )
    destination = tmp_path / "sealed.json"
    with pytest.raises(ValueError, match="published checksum"):
        audit_sealed.freeze_sealed_hashes(destination, split_path)
    assert not destination.exists()
