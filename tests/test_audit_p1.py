"""P1 crop search preserves source samples, support intervals and sealed boundaries."""

import numpy as np
import pytest

from eval.tasks import p8audit


@pytest.fixture
def tilt_window(tmp_path, monkeypatch):
    monkeypatch.setattr(p8audit, "ROOT", tmp_path)
    cache = tmp_path / "eval/cache/p8audit"
    cache.mkdir(parents=True)
    fs = 100.0
    x = np.arange(180 * int(fs) * 2, dtype=np.float32).reshape(-1, 2)
    path = cache / "synthetic-window.npy"
    np.save(path, x, allow_pickle=False)
    meta = {
        "dataset": "synthetic",
        "subject_key": "development-subject",
        "record_key": "development-record",
        "cache_key": "synthetic-window",
        "cache_sha256": p8audit.digest(path),
        "source_hashes": {"source_sha256": "a" * 64},
        "window_start_s": 1000.0,
        "window_duration_s": 180.0,
        "fs": fs,
        "channels": ["ECG", "ABP"],
        "event": {"kind": "tilt", "t": 1080.0},
        "event_case_s": 80.0,
    }
    splits = {"datasets": {"synthetic": {"development": [meta["subject_key"]], "sealed_confirmation": []}}}
    return meta, splits, cache, x


def test_alternate_tilt_crops_preserve_baseline_response_and_provenance(tilt_window):
    meta, splits, cache, original = tilt_window
    crops = p8audit._extra_tilt_windows([meta], splits)
    assert [(crop["window_duration_s"], crop["event_case_s"]) for crop in crops] == p8audit.TILT_EXTRA_CROPS
    fs = meta["fs"]
    for crop in crops:
        assert crop["source_hashes"] == meta["source_hashes"]
        assert crop["subject_key"] == meta["subject_key"]
        assert crop["window_start_s"] + crop["event_case_s"] == meta["event"]["t"]
        x = np.load(cache / f"{crop['cache_key']}.npy", allow_pickle=False)
        # The actual estimand samples are identical, including interpolation support.
        for lo, hi in [(-45, -10), (0, 61)]:
            original_start = round((meta["event_case_s"] + lo) * fs)
            original_end = round((meta["event_case_s"] + hi) * fs)
            crop_start = round((crop["event_case_s"] + lo) * fs)
            crop_end = round((crop["event_case_s"] + hi) * fs)
            np.testing.assert_array_equal(x[crop_start:crop_end], original[original_start:original_end])
        assert crop["event_case_s"] >= 50
        assert crop["window_duration_s"] - crop["event_case_s"] >= 65
        assert crop["candidate_search"]["parent_cache_sha256"] == meta["cache_sha256"]


@pytest.mark.parametrize("partition", ["sealed", "unassigned"])
def test_tilt_crop_rejects_non_development_subject_before_opening_cache(tilt_window, partition):
    meta, splits, cache, _ = tilt_window
    split = splits["datasets"]["synthetic"]
    split["development"] = []
    if partition == "sealed":
        split["sealed_confirmation"] = [meta["subject_key"]]
    (cache / "synthetic-window.npy").unlink()
    with pytest.raises(ValueError, match="sealed subject"):
        p8audit._extra_tilt_windows([meta], splits)


def test_tilt_search_does_not_extend_beyond_existing_development_window(tilt_window):
    meta, splits, cache, original = tilt_window
    meta["event_case_s"] = 61
    meta["event"]["t"] = meta["window_start_s"] + 61
    crops = p8audit._extra_tilt_windows([meta], splits)
    assert len(crops) == 3
    for crop in crops:
        start = round((crop["window_start_s"] - meta["window_start_s"]) * meta["fs"])
        x = np.load(cache / f"{crop['cache_key']}.npy", allow_pickle=False)
        np.testing.assert_array_equal(x, original[start : start + len(x)])


def test_pat_question_states_minimum_definition_without_tangent_substitution():
    question = p8audit.public_question("audit_pat", ["ECG", "ABP"], 180)
    assert "earliest minimum" in question and "centered quadratic" in question
    assert "immediately preceding R once" in question
    assert "tangent" not in question
