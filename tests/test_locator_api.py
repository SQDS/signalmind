"""Baseline fidelity, boundaries, provenance and no-edit contracts."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tsmcp.cleaning.recipes import diagnose
from tsmcp.core.detect import clip, flat, spikes
from tsmcp.locators.api import METHODS, locate_issues
from tsmcp.locators.legacy import legacy_diagnosis, legacy_rules, spikes_with_bounds
from tsmcp.locators.vendor.time_sphere_original import TimeSphereArtifactDetector


def wave(fs=125, seconds=8):
    t = np.arange(round(fs * seconds)) / fs
    return np.sin(2 * np.pi * 1.25 * t)


def test_original_python_block_bytes_are_unchanged():
    import tsmcp.locators.vendor.time_sphere_original as original
    from tsmcp.cleaning.common import read, sha

    path = Path(original.__file__)
    assert sha(path) == read(path.parent / "original_provenance.json")["python_block_sha256"]


def test_existing_30s_rules_and_thresholds_are_preserved():
    x = wave(seconds=65)
    x[200:400] = 0
    old = diagnose(x, 125)["issues"]
    new = legacy_diagnosis(x, 125)
    assert [e["evidence"] for e in new] == old
    assert [[e["start"], e["end"]] for e in new] == [e["range_samples"] for e in old]


def test_spike_bounds_do_not_start_at_the_peak():
    x = wave()
    x[500:505] += [15, 17, 30, 14, 12]
    old = spikes(x, 125)
    new = spikes_with_bounds(x, 125)
    assert len(old) == len(new) > 0
    assert any(e["start"] < e["peak_sample"] for e in new)
    for before, after in zip(old, new, strict=True):
        assert before["t"] == after["peak_sample"] / 125
        assert before["dur"] == (after["end"] - after["start"]) / 125
        assert before["score"] == after["score"]


def test_flat_clip_thresholds_are_unchanged():
    x = wave(seconds=12)
    x[500:750] = 7
    result = legacy_rules(x, 125)
    for fn, kind in ((flat, "flat"), (clip, "clip")):
        old = fn(x, 125)
        new = [r for r in result if r["kind"] == kind]
        assert len(old) == len(new)
        assert sorted((round(e["t"] * 125), round((e["t"] + e["dur"]) * 125)) for e in old) == sorted(
            (e["start"], e["end"]) for e in new
        )


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_leave_input_and_separate_real_breaks(method):
    x = wave(seconds=10)
    x[450:460] = np.nan
    source = np.ones(len(x), bool)
    source[800:805] = False
    t = np.arange(len(x)) / 125
    t[1100:] += 20
    original = hashlib.sha256(x.tobytes()).hexdigest()
    result = locate_issues(x, 125, method=method, source_support=source, times=t)
    assert hashlib.sha256(x.tobytes()).hexdigest() == original
    assert result["missing_ranges"] == [[450, 460]]
    for e in result["events"]:
        assert not e["start"] < 450 < e["end"]
        assert not e["start"] < 800 < e["end"]
        assert not e["start"] < 1100 < e["end"]
    if method == METHODS[2]:
        assert result["coverage_known"] is False and result["evaluated_ranges"] is None


def test_all_missing_is_unscored_not_clean():
    result = locate_issues(np.full(1000, np.nan), 125, method=METHODS[3])
    assert result["status"] == "partial" and not result["evaluated_ranges"]
    assert result["public_flag_ranges"] == [[0, 1000]]
    assert "overall_sqi" not in result


def test_raw_draft_nan_failure_is_reproduced_not_hidden():
    x = wave()
    x[400] = np.nan
    report = TimeSphereArtifactDetector(fs=125).detect(x, include_mesh=False)
    assert report["overall_sqi"] == 1 and report["recommendation"] == "PASS_DIRECTLY"
    # The raw pathology is archived as a draft limitation, not endorsed as a quality check.


def test_shape_validation_rejects_flattening_two_channels():
    with pytest.raises(ValueError, match="one-dimensional"):
        locate_issues(np.zeros((2, 1000)), 125, method=METHODS[3])
