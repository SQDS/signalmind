"""Boundary tests for the frozen audit validator and model-free isolation probe."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.audit_checks import _probe_home, validate_suite
from eval.tasks.p8audit import perturb


@pytest.fixture
def physiological_channels():
    fs = 50
    time = np.arange(180 * fs) / fs
    signal = np.column_stack(
        [
            np.sin(2 * np.pi * 1.2 * time),
            np.sin(2 * np.pi * 1.2 * (time - 0.2)),
            90 + 20 * np.sin(2 * np.pi * 1.2 * time),
        ]
    ).astype(np.float32)
    return signal, fs, ["ECG", "PPG", "ABP"]


@pytest.mark.parametrize(
    "qtype", ["audit_motion", "audit_step", "audit_unusable", "audit_damped", "audit_shift", "audit_corrupted"]
)
def test_perturbations_are_deterministic_and_leave_reference_channels_untouched(physiological_channels, qtype):
    signal, fs, names = physiological_channels
    original = signal.copy()
    output, answer, params = perturb(signal, fs, names, qtype, seed=17)
    repeated, repeated_answer, repeated_params = perturb(signal, fs, names, qtype, seed=17)
    np.testing.assert_array_equal(signal, original)
    np.testing.assert_array_equal(output, repeated)
    assert answer == repeated_answer and params == repeated_params
    changed = names.index(params["channel"])
    assert np.any(output[:, changed] != original[:, changed])
    for channel in range(len(names)):
        if channel != changed:
            np.testing.assert_array_equal(output[:, channel], original[:, channel])
    assert np.isfinite(output).all()


def test_inserted_shifts_preserve_baseline_and_define_signed_lag(physiological_channels):
    signal, fs, names = physiological_channels
    signs = set()
    for seed in range(10):
        output, answer, params = perturb(signal, fs, names, "audit_shift", seed=seed)
        channel = names.index(answer["channel"])
        start = round(params["start_s"] * fs)
        shift = params["shift_samples"]
        signs.add(int(np.sign(shift)))
        np.testing.assert_array_equal(output[:start], signal[:start])
        # At a known interior timestamp, a positive imposed delay reads an earlier original sample.
        timestamp = start + 100
        assert output[timestamp, channel] == signal[timestamp - shift, channel]
        assert answer["shift_s"] == shift / fs
    assert signs == {-1, 1}


def test_unusable_truth_is_exact_union_of_inserted_intervals(physiological_channels):
    signal, fs, names = physiological_channels
    output, answer, params = perturb(signal, fs, names, "audit_unusable", seed=3)
    mask = np.zeros(len(signal), dtype=bool)
    channel = names.index(params["channel"])
    for kind, (begin, end) in zip(params["interval_kinds"], params["intervals_s"], strict=True):
        first, last = round(begin * fs), round(end * fs)
        mask[first:last] = True
        values = output[first:last, channel]
        if kind == "flat":
            assert np.ptp(values) == 0
        else:
            assert 0 < np.ptp(values) < 0.05 * np.std(signal[:, channel])
            assert np.mean((values == values.min()) | (values == values.max())) > 0.90
    assert answer["percent"] == pytest.approx(100 * mask.mean())
    np.testing.assert_array_equal(output[~mask], signal[~mask])


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def frozen_suite(tmp_path):
    case_id, subject, sealed, record_key = "a" * 24, "b" * 64, "c" * 64, "d" * 64
    source_hash = "e" * 64
    splits = {
        "datasets": {
            "example": {
                "development": [subject],
                "sealed_confirmation": [sealed],
                "records": [
                    {"subject_key": subject, "record_key": record_key, "source_sha256": source_hash},
                    {"subject_key": sealed, "record_key": "f" * 64, "published_source_sha256": "0" * 64},
                ],
            }
        }
    }
    split_path = tmp_path / "splits.json"
    _write(split_path, splits)
    cases_root = tmp_path / "cases"
    case_dir = cases_root / "p8audit" / case_id
    case_dir.mkdir(parents=True)
    np.save(case_dir / "signal.npy", np.arange(240, dtype=float).reshape(120, 2), allow_pickle=False)
    visible = {
        "file": "signal.npy",
        "dur_s": 120,
        "channels": [{"name": name, "unit": "mV", "fs": 1} for name in ("ECG", "PPG")],
    }
    _write(case_dir / "visible.json", visible)
    question = {
        "id": case_id,
        "case_id": case_id,
        "row": 1,
        "family": "PPG",
        "task_type": "audit_rate",
        "source": {
            "dataset": "example",
            "subject_key": subject,
            "record_key": record_key,
            "source_hashes": {"source_sha256": source_hash},
        },
        "score_group": "core",
        "signal_sha256": hashlib.sha256((case_dir / "signal.npy").read_bytes()).hexdigest(),
    }
    manifest = {
        "suite": "p8audit",
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "questions": [question],
        "rows": [{"row": 1, "actual": 1, "requested": 3}],
        "denominators": {"core": 1, "gap_probe": 0, "total": 1},
    }
    manifest_path = tmp_path / "manifest.json"
    _write(manifest_path, manifest)
    _write(cases_root / "p8audit.questions.jsonl", {"case_id": case_id, "qtype": "audit_rate", "question": "Rate?"})
    _write(cases_root / "p8audit.answers.jsonl", {"case_id": case_id, "qtype": "audit_rate", "answer": {"bpm": 72}})
    return SimpleNamespace(
        manifest_path=manifest_path,
        split_path=split_path,
        cases_root=cases_root,
        case_dir=case_dir,
        manifest=manifest,
        splits=splits,
        visible=visible,
    )


def _check(fixture):
    return validate_suite(fixture.manifest_path, fixture.split_path, fixture.cases_root)


def test_validator_distinguishes_byte_hashes_from_sealed_array_hashes(frozen_suite):
    result = _check(frozen_suite)
    assert result["valid"] is True
    assert result["oracle_correct"] == 1
    assert result["sealed_source_hashes_available"] == 1
    assert result["sealed_source_hash_check"] == "complete"
    assert result["sealed_signal_hash_check"].startswith("incomplete")
    assert result["sealed_waveforms_decoded"] is False


def _bind_raw_inventory(fixture, inventory=None):
    """Freeze synthetic raw-source evidence independently of the split fixture."""
    _write(fixture.split_path, fixture.splits)
    fixture.manifest["split_sha256"] = hashlib.sha256(fixture.split_path.read_bytes()).hexdigest()
    frozen = fixture.splits["datasets"]["example"]["records"][1]
    if inventory is None:
        record = {
            "dataset": "example",
            "subject_key": frozen["subject_key"],
            "record_key": frozen["record_key"],
            "source_sha256": frozen["published_source_sha256"],
            "source_bytes": 100,
        }
        if "header_sha256" in frozen:
            record.update(header_sha256=frozen["header_sha256"], header_bytes=30)
        inventory = {
            "schema_version": 1,
            "split_sha256": fixture.manifest["split_sha256"],
            "method": "sha256_raw_source_bytes",
            "waveform_decoding": False,
            "dataset_counts": {"example": 1},
            "record_count": 1,
            "raw_file_count": 1 + ("header_sha256" in frozen),
            "source_bytes": 100 + (30 if "header_sha256" in frozen else 0),
            "records": [record],
        }
    inventory_path = fixture.manifest_path.parent / "sealed.json"
    _write(inventory_path, inventory)
    fixture.manifest["amendment"] = "P1"
    fixture.manifest["sealed_hash_inventory"] = {
        "path": inventory_path.name,
        "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
    }
    _write(fixture.manifest_path, fixture.manifest)
    return inventory


def test_p1_raw_inventory_completes_both_byte_comparisons_without_array_claim(frozen_suite):
    _bind_raw_inventory(frozen_suite)
    result = _check(frozen_suite)
    assert result["valid"]
    assert result["sealed_source_hash_check"] == result["sealed_signal_hash_check"] == "complete"
    assert result["development_source_hashes_compared"] == result["development_signal_hashes_compared"] == 1
    assert result["sealed_raw_files_hashed"] == 1
    assert result["sealed_source_hash_overlap"] == result["sealed_signal_hash_overlap"] == 0
    assert result["sealed_source_bytes_hashed"] is True
    assert result["sealed_waveforms_decoded"] is False
    assert result["canonical_waveform_hash_comparison"].startswith("not performed")


def test_p1_requires_bound_raw_inventory(frozen_suite):
    frozen_suite.manifest["amendment"] = "P1"
    _write(frozen_suite.manifest_path, frozen_suite.manifest)
    result = _check(frozen_suite)
    assert not result["valid"]
    assert any("P1 requires" in error for error in result["errors"])
    assert result["sealed_source_hash_check"] == "incomplete"
    assert result["sealed_signal_hash_check"].startswith("incomplete")


@pytest.mark.parametrize("binding", [{}, {"path": "../sealed.json", "sha256": "0" * 64}])
def test_p1_rejects_malformed_inventory_binding(frozen_suite, binding):
    frozen_suite.manifest.update(amendment="P1", sealed_hash_inventory=binding)
    _write(frozen_suite.manifest_path, frozen_suite.manifest)
    result = _check(frozen_suite)
    assert not result["valid"]
    assert "Invalid sealed raw hash inventory binding" in result["errors"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_subject", "wrong_split", "decoded"])
def test_p1_rejects_incomplete_or_invalid_raw_inventory(frozen_suite, mutation):
    inventory = _bind_raw_inventory(frozen_suite)
    if mutation == "missing":
        inventory["records"] = []
    elif mutation == "duplicate":
        inventory["records"] *= 2
    elif mutation == "wrong_subject":
        inventory["records"][0]["subject_key"] = "b" * 64
    elif mutation == "wrong_split":
        inventory["split_sha256"] = "0" * 64
    else:
        inventory["waveform_decoding"] = True
    _bind_raw_inventory(frozen_suite, inventory)
    result = _check(frozen_suite)
    assert not result["valid"]
    assert result["sealed_source_hash_check"] == "incomplete"
    assert result["sealed_signal_hash_check"].startswith("incomplete")


def test_p1_rejects_raw_inventory_tampering(frozen_suite):
    _bind_raw_inventory(frozen_suite)
    path = frozen_suite.manifest_path.parent / "sealed.json"
    path.write_text(path.read_text() + "\n")
    result = _check(frozen_suite)
    assert not result["valid"]
    assert "Frozen sealed raw hash inventory file hash mismatch" in result["errors"]


@pytest.mark.parametrize("kind", ["source", "header", "signal", "signal_vs_header"])
def test_p1_rejects_development_sources_and_signals_matching_any_sealed_raw_file(frozen_suite, kind):
    records = frozen_suite.splits["datasets"]["example"]["records"]
    question = frozen_suite.manifest["questions"][0]
    if kind == "source":
        records[1]["published_source_sha256"] = records[0]["source_sha256"]
    elif kind == "header":
        records[0]["header_sha256"] = "1" * 64
        question["source"]["source_hashes"]["header_sha256"] = "1" * 64
        records[1]["published_source_sha256"] = "1" * 64
    elif kind == "signal":
        records[1]["published_source_sha256"] = question["signal_sha256"]
    else:
        records[1]["header_sha256"] = question["signal_sha256"]
    _bind_raw_inventory(frozen_suite)
    result = _check(frozen_suite)
    assert not result["valid"]
    key = "sealed_source_hash_overlap" if kind in {"source", "header"} else "sealed_signal_hash_overlap"
    assert result[key] == 1


def test_validator_rejects_sealed_subject_even_with_a_valid_answer(frozen_suite):
    frozen_suite.manifest["questions"][0]["source"]["subject_key"] = "c" * 64
    _write(frozen_suite.manifest_path, frozen_suite.manifest)
    result = _check(frozen_suite)
    assert not result["valid"]
    assert not result["development_only"]


def test_validator_rejects_private_files_in_case(frozen_suite):
    _write(frozen_suite.case_dir / "answer.json", {"bpm": 72})
    result = _check(frozen_suite)
    assert not result["valid"]
    assert not result["case_files_only_signal_and_visible"]


def test_validator_rejects_private_metadata(frozen_suite):
    frozen_suite.visible["subject_key"] = "b" * 64
    _write(frozen_suite.case_dir / "visible.json", frozen_suite.visible)
    assert any("unexpected fields" in error for error in _check(frozen_suite)["errors"])


def test_validator_rejects_signal_tampering(frozen_suite):
    np.save(frozen_suite.case_dir / "signal.npy", np.zeros((120, 2)), allow_pickle=False)
    assert any("signal hash mismatch" in error for error in _check(frozen_suite)["errors"])


def test_validator_rejects_same_subject_twice_within_family(frozen_suite):
    first = frozen_suite.manifest["questions"][0]
    frozen_suite.manifest["questions"].append({**first})
    _write(frozen_suite.manifest_path, frozen_suite.manifest)
    result = _check(frozen_suite)
    assert not result["valid"]
    assert not result["one_subject_per_family"]


def test_validator_rejects_modified_split_bytes(frozen_suite):
    frozen_suite.split_path.write_text(frozen_suite.split_path.read_text() + "\n")
    assert "Frozen split file hash mismatch" in _check(frozen_suite)["errors"]


def test_validator_rejects_published_sealed_source_overlap(frozen_suite):
    frozen_suite.splits["datasets"]["example"]["records"][1]["published_source_sha256"] = "e" * 64
    _write(frozen_suite.split_path, frozen_suite.splits)
    frozen_suite.manifest["split_sha256"] = hashlib.sha256(frozen_suite.split_path.read_bytes()).hexdigest()
    _write(frozen_suite.manifest_path, frozen_suite.manifest)
    result = _check(frozen_suite)
    assert result["sealed_source_hash_overlap"] == 1
    assert not result["valid"]


def test_validator_does_not_open_opaque_id_path_traversal(frozen_suite):
    frozen_suite.manifest["questions"][0]["case_id"] = "../private"
    _write(frozen_suite.manifest_path, frozen_suite.manifest)
    assert any("opaque" in error for error in _check(frozen_suite)["errors"])


def test_model_free_probe_replaces_model_command(tmp_path, monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert "claude" not in command
        assert command[:4] == ["bwrap", "--tmpfs", str(Path.home()), "--"]
        assert command[4:6] == ["/usr/bin/python3", "-c"]
        return SimpleNamespace(returncode=0, stdout='{"private_file_visible":false,"signal_visible":true}', stderr="")

    monkeypatch.setattr("eval.audit_checks.subprocess.run", run)
    result = _probe_home(["bwrap", "--tmpfs", str(Path.home()), "--", "claude", "-p"], {}, tmp_path, tmp_path)
    assert result["status"] == "pass"
    assert len(commands) == 1


def test_model_free_probe_records_sandbox_denial(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "eval.audit_checks.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="Operation not permitted"),
    )
    result = _probe_home(["bwrap", "--", "claude", "-p"], {}, tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert result["stderr"] == "Operation not permitted"
