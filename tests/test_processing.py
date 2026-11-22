"""Signal-fidelity and processing-policy checks on independently constructed inputs."""

import asyncio

import numpy as np
import pytest

from tsmcp.processing import ProcessingSession


def session(x, **kw):
    return ProcessingSession(x, fs=250, channel="signal", unit="mV", **kw)


def test_noop_preserves_float64_values_and_parent_input():
    x = np.array([1.1234567890123, 2.9876543210123, np.nan])
    p = session(x)
    assert p.raw.values.dtype == np.float64
    np.testing.assert_array_equal(p.raw.values, x)
    x[0] = 0
    assert p.raw.values[0] == 1.1234567890123
    assert not p.raw.valid[-1] and p.raw.artifact[-1]


def test_linear_gap_reconstruction_is_local_and_keeps_parent_unchanged():
    clean = 80 + 0.01 * np.arange(100)
    raw = clean.copy()
    raw[43:46] = np.nan
    p = session(raw, max_gap_samples=5)
    result = p.process(p.raw.id, "interpolate", [[43, 46]])
    np.testing.assert_allclose(result.values, clean, atol=1e-12)
    np.testing.assert_array_equal(result.values[:43], raw[:43])
    assert np.isnan(p.raw.values[43:46]).all()
    assert result.repaired.sum() == result.artifact.sum() == 3
    assert result.valid.all() and result.operations[0]["parent_id"] == p.raw.id


@pytest.mark.parametrize("window", [(0, 2), (20, 27), (99, 100)])
def test_declared_gap_and_boundary_constraints(window):
    x = np.linspace(0, 1, 100)
    x[slice(*window)] = np.nan
    p = session(x, max_gap_samples=5)
    with pytest.raises(ValueError):
        p.process(p.raw.id, "interpolate", [list(window)])
    assert len(p.versions) == 1


def test_protected_and_saturated_regions_remain_observed():
    x = np.linspace(0, 1, 100)
    x[30:32] = np.nan
    saturated = np.zeros(100, bool)
    saturated[70:74] = True
    p = session(x, saturated=saturated, protected_ranges=[[28, 35]], max_gap_samples=5)
    for window in [[30, 32], [70, 74]]:
        with pytest.raises(ValueError):
            p.process(p.raw.id, "interpolate", [window])
    marked = p.process(p.raw.id, "mark_invalid", [[70, 74]])
    np.testing.assert_array_equal(marked.values, p.raw.values)
    assert not marked.valid[70:74].any()


def test_notch_reduces_known_tone_preserving_dc_and_time_axis():
    fs = 250
    t = np.arange(fs * 20) / fs
    target = 80 + 2 * np.sin(2 * np.pi * 1.0 * t)
    noisy = target + 0.7 * np.sin(2 * np.pi * 50 * t)
    p = session(noisy)
    result = p.process(p.raw.id, "notch", [[0, len(t)]], frequency_hz=50, quality_factor=30)
    inner = slice(fs * 2, -fs * 2)
    assert np.std(result.values[inner] - target[inner]) < 0.01
    assert abs(result.values[inner].mean() - 80) < 0.001
    assert result.values.shape == noisy.shape and result.valid.all()
    np.testing.assert_array_equal(p.raw.values, noisy)


def test_tone_regression_preserves_baseline_and_unrelated_frequency():
    fs = 250
    t = np.arange(fs * 8) / fs
    target = 95 + 4 * np.sin(2 * np.pi * 1.25 * t)
    noisy = target + 3 * np.sin(2 * np.pi * 60 * t + 0.4)
    p = session(noisy)
    result = p.process(p.raw.id, "remove_tone", [[0, len(t)]], frequency_hz=60)
    np.testing.assert_allclose(result.values, target, atol=1e-10)
    assert not result.repaired.any() and result.artifact.all()


def test_filtering_does_not_bridge_invalid_gap():
    t = np.arange(2500) / 250
    x = 3 + np.sin(2 * np.pi * 50 * t)
    x[1000:1100] = np.nan
    p = session(x)
    result = p.process(p.raw.id, "remove_tone", [[0, len(x)]], frequency_hz=50)
    assert np.isnan(result.values[1000:1100]).all()
    assert not result.valid[1000:1100].any()
    assert result.operations[0]["ranges_samples"] == [[0, 1000], [1100, 2500]]


def test_transient_repair_can_include_a_short_gap_and_contaminated_anchors():
    target = 80 + np.arange(200) * 0.01
    noisy = target.copy()
    noisy[80:100] += 4
    noisy[88:91] = np.nan
    p = session(noisy, max_gap_samples=3, max_transient_samples=20)
    result = p.process(p.raw.id, "interpolate", [[80, 100]])
    np.testing.assert_allclose(result.values, target, atol=1e-12)
    assert result.repaired.sum() == 20
    noisy[80:100] = np.nan
    p = session(noisy, max_gap_samples=3, max_transient_samples=20)
    with pytest.raises(ValueError, match="gap longer"):
        p.process(p.raw.id, "interpolate", [[80, 100]])


def test_local_lowpass_preserves_units_slow_component_and_invalid_gap():
    t = np.arange(6000) / 250
    target = 90 + 5 * np.sin(2 * np.pi * t)
    noisy = target + np.sin(2 * np.pi * 40 * t) + 0.7 * np.sin(2 * np.pi * 70 * t)
    noisy[3000:3050] = np.nan
    p = session(noisy)
    result = p.process(p.raw.id, "lowpass", [[1000, 5000]], frequency_hz=10, filter_order=4)
    for region in (slice(1250, 2750), slice(3300, 4750)):
        assert np.sqrt(np.mean((result.values[region] - target[region]) ** 2)) < 0.005
    assert np.isnan(result.values[3000:3050]).all()
    assert not result.valid[3000:3050].any()
    np.testing.assert_array_equal(result.values[:1000], noisy[:1000])
    np.testing.assert_array_equal(result.values[5000:], noisy[5000:])
    np.testing.assert_array_equal(p.raw.values, noisy)


def test_windowed_tone_fit_tracks_amplitude_without_changing_dc():
    t = np.arange(6000) / 250
    target = 95 + 2 * np.sin(2 * np.pi * t)
    noisy = target + (2 + 0.8 * np.sin(2 * np.pi * 0.125 * t)) * np.sin(2 * np.pi * 50 * t)
    p = session(noisy)
    global_fit = p.process(p.raw.id, "remove_tone", [[0, len(t)]], frequency_hz=50)
    local = p.process(p.raw.id, "remove_tone", [[0, len(t)]], frequency_hz=50, window_s=1)
    inner = slice(250, -250)
    assert np.std(local.values[inner] - target[inner]) < np.std(global_fit.values[inner] - target[inner]) / 5
    assert abs(np.mean(local.values[inner] - target[inner])) < 0.001
    assert not local.repaired.any() and local.valid.all()


def test_spectrum_finds_a_known_line_without_processing():
    t = np.arange(2500) / 250
    x = np.sin(2 * np.pi * 1.0 * t) + 0.4 * np.sin(2 * np.pi * 50 * t)
    p = session(x)
    result = p.spectrum(p.raw.id)
    line = result["valid_runs"][0]["candidate_lines"][0]
    assert line["peak_hz"] == 50 and line["band_power_fraction"] > 0.05
    assert len(p.versions) == 1


def test_real_mcp_tools_return_images_and_export_selected_version():
    from mcp import Client

    from tsmcp.server.processing import build_processing_server

    p = session(np.sin(np.arange(1000) / 30))
    delivered = []

    def export(state, action, rationale, limitations):
        delivered.append((state.id, action))
        return {"id": state.id}

    server = build_processing_server(p, export, lambda state: {"mean": float(state.values.mean())})

    async def exercise():
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"ts_process", "ts_export", "ts_spectrum", "ts_zoom"} <= names
            zoom = await client.call_tool("ts_zoom", {"id": p.raw.id, "start": 0, "end": 500})
            assert not zoom.is_error and zoom.content[0].type == "image"
            result = await client.call_tool("ts_export", {"id": p.raw.id, "action": "keep"})
            assert not result.is_error

    asyncio.run(exercise())
    assert delivered == [(p.raw.id, "keep")]
