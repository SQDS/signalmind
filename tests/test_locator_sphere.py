"""Engineering contracts for the supplied sphere method, without fitting labels."""

import numpy as np
import pytest

from tsmcp.locators.sphere import (
    _acf_period,
    _adaptive_phase,
    _feature_scores,
    _postprocess,
    locate_sphere,
)


@pytest.mark.parametrize("fs", [125, 250, 360, 500])
def test_fft_preserves_direct_acf_period_and_phase(fs):
    time = np.arange(fs * 10) / fs
    x = np.sin(2 * np.pi * 1.23 * time) + 0.3 * np.sin(2 * np.pi * 2.46 * time + 0.7)
    low, high = max(2, int(fs * 60 / 165)), min(len(x) // 2, int(fs * 60 / 42))
    assert _acf_period(x, fs, low, high, method="fft") == _acf_period(x, fs, low, high, method="direct")
    fft_phase, fft_period = _adaptive_phase(x, fs, method="fft")
    direct_phase, direct_period = _adaptive_phase(x, fs, method="direct")
    assert fft_period == direct_period
    np.testing.assert_array_equal(fft_phase, direct_phase)


@pytest.mark.parametrize("x", [np.array([np.nan] * 1000), np.r_[np.nan, np.zeros(999)]])
def test_nonfinite_input_is_rejected_instead_of_clean(x):
    with pytest.raises(ValueError, match="nonfinite"):
        locate_sphere(x, 125)


def test_single_channel_and_sampling_contract():
    with pytest.raises(ValueError, match="one-dimensional"):
        locate_sphere(np.zeros((2, 1000)), 125)
    for fs in (0, -125, np.inf, np.nan):
        with pytest.raises(ValueError, match="Sampling rate"):
            locate_sphere(np.zeros(1000), fs)
    with pytest.raises(ValueError, match="sensitivity"):
        locate_sphere(np.zeros(1000), 125, sensitivity="unregistered")


def test_short_segment_is_explicitly_unscored():
    result = locate_sphere(np.zeros(200), 125, include_mesh=True)
    assert result["status"] == "unscored"
    assert not result["evaluated_mask"].any()
    assert np.isnan(result["scores"]).all()
    assert result["events"] == [] and result["mesh"] is None
    assert "overall_sqi" not in result and "recommendation" not in result


def test_partial_last_cycle_not_discarded_or_fabricated():
    fs = 125
    t = np.arange(1000) / fs
    x = np.sin(2 * np.pi * 1.25 * t)
    result = locate_sphere(x, fs, include_mesh=True)
    mesh = result["mesh"]
    phase, _ = _adaptive_phase((x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25)), fs)
    assert mesh["valid_cell"].shape[0] == int(np.floor(phase[-1])) + 1
    assert mesh["sample_positions"][mesh["valid_cell"]].max() > 995
    assert np.isnan(mesh["sample_positions"][~mesh["valid_cell"]]).all()
    assert result["evaluated_mask"][-1]
    assert np.isfinite(result["scores"][-1])


def test_three_second_slow_signal_does_not_invent_four_cycles():
    fs = 125
    t = np.arange(375) / fs
    result = locate_sphere(np.sin(2 * np.pi * 0.7 * t), fs, include_mesh=True)
    mesh = result["mesh"]
    assert mesh["valid_cell"].shape[0] < 4
    assert not mesh["valid_cell"][-1].all()
    assert (mesh["sample_positions"][mesh["valid_cell"]] < len(t)).all()


def test_constant_signal_keeps_first_last_alarm_and_half_open_intervals():
    # At 160 Hz the original fallback period has exactly 128 samples, so all
    # cells in the first and last cycles are observed. Test postprocessing, not
    # a detector's ability to infer unobserved parts of a final partial cycle.
    result = locate_sphere(np.zeros(1280), 160)
    assert result["anomaly_mask"].all()
    assert result["events"][0]["start"] == 0
    assert result["events"][0]["end"] == 1280
    reconstructed = np.zeros(1280, bool)
    for event in result["events"]:
        reconstructed[event["start"] : event["end"]] = True
    np.testing.assert_array_equal(reconstructed, result["anomaly_mask"])


def test_partial_cycle_reports_missing_flat_context_even_when_amplitude_is_available():
    result = locate_sphere(np.zeros(1000), 125, include_mesh=True)
    partial = ~result["mesh"]["valid_cell"][-1].all()
    assert partial
    assert np.isnan(result["mesh"]["component_scores"][2, -1]).all()
    assert result["component_unscored_ranges"]["flatness_and_ridge_loss"][-1][1] == 1000
    assert result["evaluated_mask"][-1]  # The separate amplitude feature is available.
    assert result["component_evaluated_samples"]["flatness_and_ridge_loss"] < 1000


def test_gap_fill_is_not_edge_erosion_and_does_not_bridge_unscored_support():
    mask = np.r_[np.ones(10, bool), np.zeros(3, bool), np.ones(10, bool)]
    all_supported = np.ones(len(mask), bool)
    assert _postprocess(mask, all_supported, 3, 2).all()
    all_supported[11] = False
    np.testing.assert_array_equal(_postprocess(mask, all_supported, 3, 2), mask)
    edges = np.r_[np.zeros(2, bool), np.ones(19, bool), np.zeros(2, bool)]
    np.testing.assert_array_equal(_postprocess(edges, np.ones(len(edges), bool), 3, 2), edges)


def test_short_alarm_removed_from_mask_and_not_only_from_event_list():
    mask = np.zeros(100, bool)
    mask[10:12], mask[40:60] = True, True
    result = _postprocess(mask, np.ones(100, bool), 3, 10)
    assert not result[10:12].any()
    assert result[40:60].all()


def test_temporal_volume_context_never_wraps_last_cycle_to_first():
    grid = np.tile(np.sin(np.arange(128) / 128 * 2 * np.pi), (5, 1))
    grid[-1] += 10
    _, components = _feature_scores(grid, np.ones(grid.shape, bool))
    assert np.isnan(components[1, 0]).all()
    assert np.isnan(components[1, -1]).all()
    assert np.isfinite(components[1, 1:-1]).all()


def test_formula_preserves_known_flat_floor_false_positive():
    # Keep this defect visible: a flat part of a full otherwise nonflat cycle
    # receives 4 even without ridge loss, as in the supplied draft.
    row = np.zeros(128)
    row[30:35] = [0.1, 0.2, 1, 0.2, 0.1]
    grid = np.tile(row, (6, 1))
    _, components = _feature_scores(grid, np.ones(grid.shape, bool))
    assert (components[2, :, 70:90] == 4).all()


def test_display_offset_and_mesh_flag_never_change_detection():
    fs = 125
    t = np.arange(2000) / fs
    x = 0.01 * np.sin(2 * np.pi * t)
    x[::100] = -20
    original = x.copy()
    plain = locate_sphere(x, fs)
    display = locate_sphere(x, fs, include_mesh=True)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(plain["scores"], display["scores"])
    np.testing.assert_array_equal(plain["evaluated_mask"], display["evaluated_mask"])
    np.testing.assert_array_equal(plain["anomaly_mask"], display["anomaly_mask"])
    assert plain["events"] == display["events"]
    assert display["mesh"]["display_radius_offset"] > 0
    assert np.nanmin(display["mesh"]["display_radius"]) > 0
    assert np.nanmin(display["mesh"]["radial_coordinate"]) < 0


@pytest.mark.parametrize("fs", [125, 500, 1000])
def test_resolution_reports_native_grid_spacing(fs):
    t = np.arange(fs * 8) / fs
    x = np.sin(2 * np.pi * 1.25 * t)
    x[(t >= 2) & (t < 3)] = 10
    result = locate_sphere(x, fs)
    assert all(event["resolution_samples"] >= 1 for event in result["events"])
    if fs == 1000 and result["events"]:
        assert max(e["resolution_samples"] for e in result["events"]) > 1
