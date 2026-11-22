"""Native timing confounds are rejected independently of intervention labels."""

import numpy as np
import pytest
from scipy import signal

from eval import level2_alignment as alignment


def synthetic_native(fs=125, duration=100, change_ms=0, jitter_ms=0):
    time = np.arange(round(duration * fs)) / fs
    ecg, abp, ppg = np.zeros_like(time), np.full_like(time, 80.0), np.zeros_like(time)
    for index, beat in enumerate(np.arange(1, duration - 1, 0.9)):
        change = change_ms / 1000 if beat >= 45 else 0
        jitter = jitter_ms / 1000 * (-1 if index % 2 else 1)
        ecg += np.exp(-0.5 * ((time - beat) / 0.012) ** 2)
        abp += 40 * np.exp(-0.5 * ((time - beat - 0.25 - change - jitter) / 0.065) ** 2)
        ppg += 2 * np.exp(-0.5 * ((time - beat - 0.35) / 0.09) ** 2)
    return np.column_stack((ecg, abp, ppg))


@pytest.fixture(autouse=True)
def hermetic_r_events(monkeypatch):
    # These tests exercise the two distinct pulse/timing pipelines. The external
    # authorized R detector has its own tests; do not read external code here.
    def reference(ecg, fs):
        peaks, _ = signal.find_peaks(ecg, height=0.5, distance=round(0.3 * fs))
        return {"accepted": True, "rpeaks": peaks, "independent_rpeaks": peaks, "quality": {"synthetic": True}}

    monkeypatch.setattr(alignment, "ecg_reference", reference)


def test_fractional_native_timing_accepts_stable_lag():
    result = alignment.native_lag_support(synthetic_native(), 125, ["ECG", "ABP", "PPG"], 45)
    assert result["accepted"]
    assert set(result["pairs"]) == {"ECG_to_ABP", "ECG_to_PPG", "PPG_to_ABP"}
    for pair in result["pairs"].values():
        assert pair["native_change_disagreement_ms"] < 0.5
        assert pair["implementation_1"]["before"]["beats"] >= 12


@pytest.mark.parametrize("change_ms", [6.5, -8, 24, -40])
def test_native_physiological_lag_change_cannot_be_mistaken_for_injected_offset(change_ms):
    result = alignment.native_lag_support(synthetic_native(change_ms=change_ms), 125, ["ECG", "ABP", "PPG"], 45)
    assert not result["accepted"]
    pair = result["pairs"]["ECG_to_ABP"]
    assert not pair["implementation_1"]["accepted"]
    assert not pair["implementation_2"]["accepted"]
    assert abs(pair["implementation_1"]["native_change_ms"] - change_ms) < 1
    assert abs(pair["implementation_2"]["native_change_ms"] - change_ms) < 1


def test_drift_uses_native_first_and_last_thirty_seconds():
    result = alignment.native_lag_support(
        synthetic_native(duration=100, change_ms=20), 125, ["ECG", "ABP", "PPG"], 45, drift=True
    )
    assert result["windows_s"] == {"before": [0, 30], "after": [70, 100]}
    assert not result["accepted"]


def test_noisy_native_mean_cancellation_is_not_timing_support():
    result = alignment.native_lag_support(synthetic_native(jitter_ms=40), 125, ["ECG", "ABP", "PPG"], 45)
    assert not result["accepted"]
    assert result["pairs"]["ECG_to_ABP"]["implementation_1"]["change_standard_error_ms"] > 5


def test_independent_methods_disagreeing_within_each_native_gate_are_rejected(monkeypatch):
    times = np.arange(1, 99, 0.9)

    def events(change):
        return {"ECG": times, "ABP": times + 0.25 + np.where(times > 45, change, 0)}

    monkeypatch.setattr(alignment, "_guided_events", lambda *a: events(0.004))
    monkeypatch.setattr(alignment, "_unguided_events", lambda *a: events(-0.004))
    result = alignment.native_lag_support(synthetic_native()[:, :2], 125, ["ECG", "ABP"], 45)
    pair = result["pairs"]["ECG_to_ABP"]
    assert pair["implementation_1"]["accepted"] and pair["implementation_2"]["accepted"]
    assert pair["native_change_disagreement_ms"] == pytest.approx(8)
    assert not result["accepted"]


def test_unsupported_or_overlapping_native_windows_fail_closed():
    result = alignment.native_lag_support(synthetic_native(), 125, ["ECG", "ABP", "PPG"], 85)
    assert not result["accepted"]
    result = alignment.native_lag_support(synthetic_native(), 125, ["ECG", "ABP", "PPG"], 5)
    assert not result["accepted"]


def test_missing_beat_support_or_nonfinite_source_never_passes(monkeypatch):
    native = synthetic_native()
    monkeypatch.setattr(
        alignment,
        "_unguided_events",
        lambda *a: {name: np.array([1.0, 2.0]) for name in ("ECG", "ABP", "PPG")},
    )
    assert not alignment.native_lag_support(native, 125, ["ECG", "ABP", "PPG"], 45)["accepted"]
    native[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite aligned"):
        alignment.native_lag_support(native, 125, ["ECG", "ABP", "PPG"], 45)
