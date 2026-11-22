"""Manual labels require development identity, native equality and unambiguous support."""

import json

import numpy as np
import pytest

from eval import level2_manual as manual


def event(start=50, end=55, **overrides):
    return {
        "confidence": "high",
        "mark_layer": "main",
        "problem_type": "physiocal_small_step_like",
        "applied_action": "interpolate_short_main",
        "start_sec": str(start),
        "end_sec": str(end),
        **overrides,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence": "low"},
        {"confidence": "medium"},
        {"mark_layer": "micro_possible"},
        {"problem_type": "multi_signal_dropout_uncertain"},
        {"problem_type": "terminal_bad_segment"},
        {"applied_action": "micro_possible_not_used_in_strict"},
    ],
)
def test_uncertain_or_non_main_manual_marks_are_never_truth(changes):
    assert manual._eligible(event())
    assert not manual._eligible(event(**changes))


def test_manual_mapping_requires_same_native_samples_not_just_same_shape():
    native = np.arange(100, dtype=float)
    assert manual._native_agreement(native, native * 1.1, 1.1) == 0
    with pytest.raises(ValueError, match="does not match"):
        manual._native_agreement(native, np.roll(native, 1) * 1.1, 1.1)
    with pytest.raises(ValueError, match="support or calibration"):
        manual._native_agreement(native, native[:-1], 1)


def test_manual_crop_rejects_ambiguous_longest_defect():
    target = event(80, 90)
    competing = event(100, 110)
    with pytest.raises(ValueError, match="uniquely longest"):
        manual._choose_crop(target, [target, competing], 200, 100, np.random.default_rng(1))


def test_manual_subject_gate_prevents_waveform_or_labels_read(monkeypatch, tmp_path):
    path = tmp_path / "Head-up-tilt_Day2" / "syntheticDB.hea"
    cleaned = tmp_path / "cleaned.npz"
    # No waveform or annotations are present: partition-linked identity must fail first.
    np.savez(cleaned, subject_key=np.array("different-development-record"))
    monkeypatch.setattr(manual, "frozen_splits", lambda: ({}, {}))
    monkeypatch.setattr(
        manual,
        "_development_candidates",
        lambda *a, **k: [("opaque-development", path, "opaque-record", {}, False)],
    )
    monkeypatch.setattr(manual, "_mapped_cleaned", lambda path: cleaned)
    monkeypatch.setattr(manual, "_wfdb_record", lambda *a: pytest.fail("waveform read before identity gate"))
    candidates, exclusions = manual.build_candidates()
    assert candidates == []
    assert exclusions[0]["reason"] == "Manual cleaned subject differs from frozen development mapping"
    assert "different-development-record" not in json.dumps(exclusions)


def test_manual_cache_saves_native_arrays_outside_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(manual, "ROOT", tmp_path)
    raw = np.arange(12, dtype=np.float32).reshape(6, 2)
    path = manual.persist_candidates([{"x": raw, "family": "F2"}], [])
    metadata = json.loads(path.read_text())["candidates"][0]
    assert "x" not in metadata
    np.testing.assert_array_equal(np.load(tmp_path / metadata["signal_cache"], allow_pickle=False), raw)
    assert json.loads((path.parent / "candidates.json").read_text()) == [metadata]
