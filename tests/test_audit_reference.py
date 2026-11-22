"""Independent-reference correctness and agreement gates, using no dataset files."""

import numpy as np
import pytest

from eval import audit_reference as ref


def _waves(seconds=120, fs=125):
    times = np.arange(seconds * fs) / fs
    phase = (times - 0.1) % 1
    # A narrow QRS and pressure pulses with a unique trough and smooth upstroke.
    ecg = np.exp(-0.5 * (((times + 0.5) % 1 - 0.5) / 0.015) ** 2)
    abp = 80 + 40 * np.sin(np.pi * phase) ** 2
    return ecg, abp, fs


def _requires(*names):
    if any(not ref.REFERENCE_PATHS[name].is_file() for name in names):
        pytest.skip("authorized local reference code is not installed")


def test_half_tolerance_gate_is_stricter_than_grading():
    assert ref.agreement(100, 102.5, 5)["accepted"]
    assert not ref.agreement(100, 102.500001, 5)["accepted"]
    assert not ref.agreement(100, 104, 5)["accepted"]
    assert ref.agreement(10, 11.25, 0.25, relative=True)["accepted"]
    assert not ref.agreement(10, 11.26, 0.25, relative=True)["accepted"]
    assert not ref.agreement(np.nan, 1, 5)["accepted"]


def test_independent_rpeaks_and_arterial_extrema_have_known_values():
    ecg, abp, fs = _waves()
    rpeaks = ref.independent_rpeaks(ecg, fs)
    assert len(rpeaks) == 119
    np.testing.assert_allclose(np.diff(rpeaks) / fs, 1)
    pressure = ref._independent_pressure(abp, fs)
    assert np.mean(pressure["sbp"]) == pytest.approx(120, abs=0.02)
    assert np.mean(pressure["dbp"]) == pytest.approx(80, abs=0.02)
    assert np.mean(pressure["map"]) == pytest.approx(100, abs=0.02)


def test_pressure_qc_allows_quantization_but_rejects_sustained_flatness():
    _, abp, fs = _waves(fs=1000)
    quantized = np.round(abp, 1)
    assert np.mean(np.diff(quantized) == 0) > 0.10
    np.testing.assert_array_equal(ref._pressure(quantized, fs), quantized)
    quantized[10 * fs : 11 * fs] = 80
    with pytest.raises(ValueError, match="plateau"):
        ref._pressure(quantized, fs)


def test_independent_brs_known_slope_and_nonphysiological_rejection():
    pressure = np.array([110, 112, 114, 116, 114, 112, 110, 112, 114, 116], float)
    cycles = np.arange(1, len(pressure) + 1)
    rr = np.r_[790, 800 + 10 * (pressure - 110), 790]
    slopes = ref.independent_brs(pressure, rr, cycles)
    np.testing.assert_allclose(slopes, 10)
    assert len(slopes) == 3
    assert not len(ref.independent_brs(pressure, np.full(len(rr), 800), cycles))
    assert not len(ref.independent_brs(pressure, rr + 2000, cycles))


def test_independent_tfa_known_transfer_and_delay():
    rng = np.random.default_rng(812)
    pressure = 90 + rng.normal(size=300)
    flow = 60 + 0.75 * (pressure - 90)
    frequency, gain = ref.independent_tfa(pressure, flow)
    np.testing.assert_allclose(gain[(frequency >= 0.07) & (frequency < 0.20)], 0.75, atol=1e-12)
    with pytest.raises(ValueError, match="100"):
        ref.independent_tfa(pressure[:99], flow[:99])


def test_tilt_estimand_uses_threshold_sustained_latency_and_null_for_unchanged():
    times = np.arange(180, dtype=float)
    hr = np.where(times < 67, 60, 66)
    assert ref._response(times, hr, 60, 3) == ("increase", 7)
    hr[67:70] = 60
    assert ref._response(times, hr, 60, 3) == ("increase", 10)
    assert ref._response(times, np.full_like(times, 60), 60, 3) == ("unchanged", None)
    with pytest.raises(ValueError, match="support"):
        ref._response(times, hr, 20, 3)


@pytest.mark.parametrize("direction,change", [("increase", 3), ("decrease", -3)])
def test_independent_tilt_response_threshold_and_sustained_boundary(direction, change):
    # The final valid five-second run starts at 56 s, includes 60 s, and must
    # not be lost at the right boundary. Earlier response has isolated gaps.
    times = np.arange(181, dtype=float)
    values = np.full(len(times), 60.0)
    values[80:110] += change
    values[80:110:4] = 60
    values[116:121] += change
    for estimator in (ref._response, ref.independent_response):
        assert estimator(times, values, 60, 3) == (direction, 56)
        assert estimator(times, np.full_like(times, 60), 60, 3) == ("unchanged", None)
        shifted_knots = times + 0.25
        exact = np.where(shifted_knots < 70, 60.0, 60.0 + change)
        assert estimator(shifted_knots, exact, 60, 3) == (direction, 11)
        with pytest.raises(ValueError, match="support"):
            estimator(times, values, 20, 3)


def test_local_and_independent_bp_references_agree_on_known_waveform():
    _requires("rpeaks", "beats")
    ecg, abp, fs = _waves()
    result = ref.compare_bp(ecg, abp, fs)
    assert result["accepted"], result
    for source in ("reference1", "reference2"):
        assert result[source]["sbp"] == pytest.approx(120, abs=0.05)
        assert result[source]["dbp"] == pytest.approx(80, abs=0.05)
        assert result[source]["map"] == pytest.approx(100, abs=0.05)
    assert all(len(digest) == 64 for digest in result["source_hashes"].values())


@pytest.mark.parametrize("fs", [125, 500, 1000])
def test_independent_pat_uses_minimum_foot_without_local_events_or_code(monkeypatch, fs):
    ecg, abp, fs = _waves(fs=fs)

    def forbidden_reference(*args):
        raise AssertionError("independent PAT must not call a local reference")

    monkeypatch.setattr(ref, "_reference", forbidden_reference)
    values = ref.independent_pat(ecg, abp, fs)
    # A sine-squared pulse's trough is 100 ms after R; a tangent intersection
    # would occur much later. This checks the definition, not just agreement.
    np.testing.assert_allclose(values, 100, atol=1000 / fs)
    assert len(values) == 118


def test_pat_same_minimum_definition_agrees_on_known_waveform():
    _requires("rpeaks", "beats")
    ecg, abp, fs = _waves()
    result = ref.compare_pat(ecg, abp, fs)
    assert result["accepted"], result
    assert result["half_tolerance"] == 10
    assert result["reference1"] == pytest.approx(100, abs=8)
    assert result["reference2"] == pytest.approx(100, abs=8)


def test_pat_same_definition_still_rejects_timing_disagreement(monkeypatch):
    _requires("rpeaks", "beats")
    ecg, abp, fs = _waves(fs=1000)
    detected = ref.independent_rpeaks(ecg, fs)
    monkeypatch.setattr(ref, "independent_rpeaks", lambda *_: detected + 20)
    result = ref.compare_pat(ecg, abp, fs)
    assert not result["accepted"]
    assert result["half_tolerance"] == 10
    assert result["difference"] == pytest.approx(20, abs=1)


def test_independent_pat_requires_twenty_beats_and_ninety_percent_coverage():
    ecg, abp, fs = _waves(seconds=20)
    with pytest.raises(ValueError, match="minimum-foot PAT coverage"):
        ref.independent_pat(ecg, abp, fs)
    ecg, abp, fs = _waves()
    abp[20 * fs : 40 * fs] = 80
    with pytest.raises(ValueError, match="minimum-foot PAT coverage"):
        ref.independent_pat(ecg, abp, fs)


def test_ecg_and_g2_reject_flat_or_mismatched_rate():
    _requires("rpeaks")
    ecg, abp, fs = _waves()
    result = ref.g2_pulse_rate(ecg, abp, fs)
    assert result["accepted"], result
    assert result["value"] == pytest.approx(60)
    time = np.arange(len(ecg)) / fs
    mismatch = np.sin(2 * np.pi * 1.4 * time)
    assert not ref.g2_pulse_rate(ecg, mismatch, fs)["accepted"]
    with pytest.raises(ValueError, match="constant"):
        ref.ecg_reference(np.zeros_like(ecg), fs)


def test_tfa_local_matches_independent_fft_and_rejects_no_excitation():
    _requires("tfa")
    fs = 125
    rng = np.random.default_rng(813)
    low_rate = rng.normal(size=300)
    pressure = np.repeat(90 + low_rate, fs) + 10 * np.sin(2 * np.pi * np.arange(300 * fs) / fs)
    flow = 60 + 0.75 * (pressure - 90)
    result = ref.compare_tfa(pressure, flow, fs)
    assert result["accepted"], result
    assert result["reference1"] == pytest.approx(0.75)
    assert result["reference2"] == pytest.approx(0.75)
    pure = 90 + 10 * np.sin(2 * np.pi * np.arange(300 * fs) / fs)
    with pytest.raises(ValueError, match="excitation"):
        ref.compare_tfa(pure, 60 + (pure - 90), fs)
