"""Pure-numpy replacements for the scipy pieces peak detectors need: FIR band-pass and find_peaks."""

from __future__ import annotations

import numpy as np


def fir_bandpass(lo_hz: float, hi_hz: float, fs: float, dur_s: float = 1.0) -> np.ndarray:
    """Linear-phase windowed-sinc band-pass (Hamming, ~dur_s long, odd length, unit gain at band center)."""
    n = int(round(dur_s * fs)) | 1
    m = np.arange(n) - (n - 1) / 2
    h = (2 * hi_hz / fs) * np.sinc(2 * hi_hz / fs * m) - (2 * lo_hz / fs) * np.sinc(2 * lo_hz / fs * m)
    h *= np.hamming(n)
    fc = (lo_hz + hi_hz) / 2
    gain = np.abs(np.sum(h * np.exp(-2j * np.pi * fc / fs * np.arange(n))))
    return h / gain


def filter_zero_phase(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a symmetric FIR with zero delay via FFT convolution (edges reflected to limit transients)."""
    k = (h.size - 1) // 2
    pad = min(k, x.size - 1)
    xp = np.concatenate([x[pad:0:-1], x, x[-2 : -pad - 2 : -1]]) if pad > 0 else x
    nfft = 1 << int(np.ceil(np.log2(xp.size + h.size - 1)))
    y = np.fft.irfft(np.fft.rfft(xp, nfft) * np.fft.rfft(h, nfft), nfft)[: xp.size + h.size - 1]
    return y[k + pad : k + pad + x.size]


def local_maxima(x: np.ndarray) -> np.ndarray:
    """Indices of strict local maxima; flat plateaus report their middle sample (as scipy does)."""
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    d = np.diff(x)
    # collapse plateaus: rising edge then (zeros) then falling edge
    nz = np.flatnonzero(d != 0)
    if nz.size < 2:
        return np.empty(0, dtype=np.int64)
    s = np.sign(d[nz])
    rise_then_fall = np.flatnonzero((s[:-1] > 0) & (s[1:] < 0))
    left = nz[rise_then_fall] + 1
    right = nz[rise_then_fall + 1]
    return ((left + right) // 2).astype(np.int64)


def prominences(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Topographic prominence (same definition as scipy.signal.peak_prominences, wlen=None).

    For each peak: walk left/right until a strictly higher sample (or the edge); the base on each side is the
    minimum on that stretch; prominence = height - max(left base, right base). Searches are vectorized per peak.
    """
    out = np.empty(peaks.size)
    for j, p in enumerate(peaks):
        h = x[p]
        higher_left = np.flatnonzero(x[:p] > h)
        lo = higher_left[-1] + 1 if higher_left.size else 0
        higher_right = np.flatnonzero(x[p + 1 :] > h)
        hi = p + 1 + higher_right[0] if higher_right.size else x.size
        out[j] = h - max(x[lo : p + 1].min(), x[p:hi].min())
    return out


def find_peaks(
    x: np.ndarray, height: float | None = None, prominence: float | None = None, distance: int = 1
) -> np.ndarray:
    """Subset of scipy.signal.find_peaks: height -> distance (keep highest) -> prominence, in scipy's order."""
    x = np.asarray(x, dtype=np.float64)
    p = local_maxima(x)
    if height is not None:
        p = p[x[p] >= height]
    if distance > 1 and p.size > 1:
        keep = np.ones(p.size, dtype=bool)
        order = np.argsort(x[p])[::-1]
        for idx in order:
            if not keep[idx]:
                continue
            j = idx - 1
            while j >= 0 and p[idx] - p[j] < distance:
                keep[j] = False
                j -= 1
            j = idx + 1
            while j < p.size and p[j] - p[idx] < distance:
                keep[j] = False
                j += 1
        p = p[keep]
    if prominence is not None and p.size:
        p = p[prominences(x, p) >= prominence]
    return p
