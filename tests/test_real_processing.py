"""Real-data preparation must not invent continuity or mistake diagnostics for truth."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.prepare_vital_processing import continuous  # noqa: E402
from eval.real_processing_helper import candidate_peaks, spectral_powers, write_submission  # noqa: E402


def metadata():
    return [
        {
            "subject": "anonymous",
            "case": "record",
            "segment": i,
            "start": i * 10.0,
            "end": i * 10.0 + 9.992,
            "regular": True,
        }
        for i in range(6)
    ]


def test_continuous_storage_rows_are_not_enough():
    rows = metadata()
    assert continuous(rows)
    for row in rows[3:]:
        row["start"] += 70
        row["end"] += 70
    assert not continuous(rows)


@pytest.mark.parametrize("field,value", [("case", "other"), ("subject", "other"), ("segment", 99), ("regular", False)])
def test_case_subject_sequence_and_full_grid_are_required(field, value):
    rows = metadata()
    rows[3][field] = value
    assert not continuous(rows)


def test_candidate_events_do_not_bridge_unavailable_regions():
    fs = 125
    t = np.arange(1250) / fs
    x = np.sin(2 * np.pi * 2 * t)
    valid = np.ones(len(x), bool)
    valid[500:625] = False
    groups = candidate_peaks(x, valid, fs, "PPG", 0.2)
    assert len(groups) == 2
    assert all(np.all(valid[p]) for p in groups)
    assert groups[0][-1] < 500 and groups[1][0] > 625


def test_reference_free_export_preserves_samples_and_labels_diagnostics(tmp_path):
    fs = 125
    t = np.arange(1250) / fs
    raw = (90 + 5 * np.sin(2 * np.pi * t) + 0.8 * np.sin(2 * np.pi * 50 * t))[None, :]
    original = raw.copy()
    task = {
        "case_id": "local",
        "signal": {
            "signal_id": "input_local",
            "channels": [
                {"name": "abp", "kind": "ABP", "unit": "mmHg", "fs_hz": fs, "n_samples": raw.shape[1], "t0_s": 0}
            ],
        },
        "downstream_contract": {"detector_prominence": 2.0},
    }
    result = write_submission(
        tmp_path,
        task,
        raw,
        raw.copy(),
        np.zeros(raw.shape, bool),
        np.zeros(raw.shape, bool),
        np.ones(raw.shape, bool),
        "keep",
        [],
    )
    np.testing.assert_array_equal(raw, original)
    with np.load(tmp_path / "output.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["values"], original)
        assert not data["modified_mask"].any()
    assert "No clean reference" in result["measurement_note"]
    assert result["schema_version"] == "signalmind.processing.vital-real-v0.1"
    assert all(m["evidence_id"] == result["signal"]["new_signal_id"] for m in result["measurements"]["after"])
    assert spectral_powers(raw, np.ones(raw.shape, bool), fs)["line_50hz_power"] > 0.25
