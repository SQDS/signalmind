"""Pure-numpy signal processing primitives."""

from __future__ import annotations

import numpy as np


def finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def fill_nan(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs (edges held)."""
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    if bad.all():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    out = x.copy()
    out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return out


def detrend_linear(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n < 3:
        return x - x.mean()
    t = np.arange(n, dtype=np.float64)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation r[0..max_lag] via FFT (biased estimator, r[0]=1)."""
    x = x - x.mean()
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    r = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    return r / r[0] if r[0] > 0 else np.zeros(max_lag + 1)


def parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak location by parabolic interpolation around index i."""
    if i <= 0 or i >= y.size - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    d = a - 2 * b + c
    return float(i) if d == 0 else i + 0.5 * (a - c) / d


def welch_psd(x: np.ndarray, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with Hann window, 50% overlap. Returns (freqs, psd)."""
    x = x - x.mean()
    n = x.size
    nperseg = int(min(n, nperseg or max(256, min(n, int(8 * fs)))))
    step = max(1, nperseg // 2)
    win = np.hanning(nperseg)
    scale = fs * (win**2).sum()
    starts = range(0, n - nperseg + 1, step)
    acc = None
    k = 0
    for s in starts:
        seg = x[s : s + nperseg]
        seg = (seg - seg.mean()) * win
        p = np.abs(np.fft.rfft(seg)) ** 2 / scale
        acc = p if acc is None else acc + p
        k += 1
    psd = acc / max(k, 1)
    psd[1:-1] *= 2
    return np.fft.rfftfreq(nperseg, 1 / fs), psd


def run_fraction(mask: np.ndarray, min_len: int) -> float:
    """Fraction of samples that lie in runs of True of length >= min_len."""
    if mask.size == 0 or not mask.any():
        return 0.0
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lens = ends - starts
    return float(lens[lens >= min_len].sum() / mask.size)


def minmax_envelope(x: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min/max of x over consecutive blocks of `per = ceil(n/cols)` samples.

    Returns (center_index, lo, hi); center_index is the fractional sample index of each block's center,
    so callers map blocks to time exactly (only the last block may be shorter; nothing is padded).
    """
    n = x.size
    per = max(1, int(np.ceil(n / cols)))
    nb = int(np.ceil(n / per))
    full = (n // per) * per
    lo = np.empty(nb)
    hi = np.empty(nb)
    if full:
        blocks = x[:full].reshape(-1, per)
        lo[: full // per] = np.nanmin(blocks, axis=1)
        hi[: full // per] = np.nanmax(blocks, axis=1)
    if full < n:
        lo[-1], hi[-1] = np.nanmin(x[full:]), np.nanmax(x[full:])
    starts = np.arange(nb) * per
    ends = np.minimum(starts + per, n)
    return (starts + ends - 1) / 2.0, lo, hi
