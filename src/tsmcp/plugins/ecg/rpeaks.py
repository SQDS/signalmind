"""R-peak detection, ported from bp2rr_new/data_process/01_rpeak_detector/rpeak_detector.py.

Original (validated on MIT-BIH: Se 0.988 / PPV 0.997 / F1 0.992): zero-phase 5-20 Hz band-pass, derivative
energy with a 0.12 s moving average, block-adaptive thresholds over 30 s cores with 2 s padding, 0.25 s refractory
period, refinement to the local |filtered| maximum within +-0.10 s.
Port changes (numpy-only): 3rd-order Butterworth sosfiltfilt -> 0.5 s Hamming FIR band-pass applied with zero delay;
scipy.find_peaks -> tsmcp.core.peaks.find_peaks (same height/distance/prominence semantics).
Validated against the original with eval/validate_rpeaks.py.
"""

from __future__ import annotations

import numpy as np

from tsmcp.core.dsp import fill_nan
from tsmcp.core.peaks import filter_zero_phase, find_peaks, fir_bandpass


def bandpass_qrs(ecg: np.ndarray, fs: float) -> np.ndarray:
    return filter_zero_phase(np.asarray(ecg, dtype=np.float64), fir_bandpass(5.0, 20.0, fs, dur_s=0.5))


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Zero-based R-peak sample indices."""
    ecg = np.asarray(ecg, dtype=np.float64)
    if ecg.ndim != 1 or ecg.size < int(3 * fs):
        return np.empty(0, dtype=np.int64)
    ecg = fill_nan(ecg)
    filtered = bandpass_qrs(ecg, fs)
    derivative = np.gradient(filtered)
    width = max(3, int(round(0.12 * fs)))
    energy = np.convolve(derivative**2, np.ones(width) / width, mode="same")
    candidates: list[int] = []
    block, padding, refine = int(30 * fs), int(2 * fs), int(round(0.10 * fs))
    refractory = max(1, int(round(0.25 * fs)))
    for core_start in range(0, ecg.size, block):
        core_end = min(ecg.size, core_start + block)
        start, end = max(0, core_start - padding), min(ecg.size, core_end + padding)
        local = energy[start:end]
        if local.size < 3:
            continue
        q50, q99 = np.quantile(local, [0.5, 0.99])
        spread = max(float(q99 - q50), np.finfo(float).eps)
        found = find_peaks(
            local, height=q50 + 0.12 * spread, prominence=max(0.04 * spread, np.finfo(float).eps), distance=refractory
        )
        for point in found + start:
            if core_start <= point < core_end:
                lo, hi = max(0, int(point) - refine), min(ecg.size, int(point) + refine + 1)
                candidates.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
    groups: list[list[int]] = []
    for point in sorted(set(candidates)):
        if not groups or point - groups[-1][-1] >= refractory:
            groups.append([point])
        else:
            groups[-1].append(point)
    peaks = np.asarray([max(g, key=lambda p: abs(float(filtered[p]))) for g in groups], dtype=np.int64)
    margin = int(round(0.16 * fs))
    return peaks[(peaks >= margin) & (peaks < ecg.size - margin)]
