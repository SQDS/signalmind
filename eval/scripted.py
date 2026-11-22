"""Scripted baseline: tsmcp.core algorithms + fixed rules, no LLM.

Separates "how good are the algorithms" from "what does the model add". Free, deterministic — runs in CI.
Unsupported question types return None (scored as no_answer, not as a negative).
"""

from __future__ import annotations

import numpy as np

from tsmcp.core.dsp import run_fraction
from tsmcp.core.measure import MEASURES
from tsmcp.core.views.describe import crossing_hz

WIN_S = 5.0


def _features(x: np.ndarray, fs: float) -> np.ndarray:
    w = int(WIN_S * fs)
    nb = x.size // w
    f = np.array([[np.mean(s), np.std(s), crossing_hz(s, fs) or 0.0] for s in x[: nb * w].reshape(nb, w)])
    return f


def change_time(x: np.ndarray, fs: float) -> float | None:
    """Best single step in any windowed feature (mean / std / crossing freq), MAD-normalized, with a minimum effect."""
    f = _features(x, fs)
    best, best_t = 0.0, None
    for j in range(f.shape[1]):
        y = f[:, j]
        scale = np.median(np.abs(np.diff(y))) * 1.4826 + 1e-12  # robust noise of the feature series
        ref = np.median(np.abs(y)) + 1e-12
        n = y.size
        cs = np.cumsum(y)
        for k in range(6, n - 6):
            d = abs(cs[k - 1] / k - (cs[-1] - cs[k - 1]) / (n - k))
            z = d / scale
            if z > best and d / ref > 0.15:  # absolute minimum effect: 15% of the feature's typical size
                best, best_t = z, k * WIN_S
    if best < 8:
        return None
    # oscillation check: a slow periodic modulation gives a large "step" but its two halves overlap heavily
    return best_t


def spike_count(x: np.ndarray, fs: float) -> int:
    w = max(3, int(0.2 * fs) | 1)
    med = np.convolve(x, np.ones(w) / w, mode="same")
    r = np.abs(x - med)
    thr = 8 * np.median(r) * 1.4826 + 1e-12
    thr = max(thr, 3 * np.std(x))
    idx = np.flatnonzero(r > thr)
    if idx.size == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(idx) > int(0.1 * fs)))


def artifact(x: np.ndarray, fs: float) -> tuple[bool, float | None]:
    flat = np.concatenate([[False], np.abs(np.diff(x)) < 1e-9])
    lo, hi = np.min(x), np.max(x)
    tol = 1e-3 * (hi - lo)
    rail = (np.abs(x - lo) <= tol) | (np.abs(x - hi) <= tol)
    for mask in (flat, rail):
        if run_fraction(mask, int(1.0 * fs)) > 0:
            m = np.concatenate(([False], mask, [False]))
            d = np.diff(m.astype(np.int8))
            s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            long = s[(e - s) >= int(1.0 * fs)]
            return True, float(long[0] / fs)
    return False, None


def answer(qtype: str, question: str, x: np.ndarray, fs: float) -> dict | None:
    if qtype == "change_time":
        return {"t": change_time(x, fs)}
    if qtype == "period":
        import re

        a, b = [float(v) for v in re.findall(r"from ([\d.]+) s to ([\d.]+) s", question)[0]]
        v = MEASURES["period"](x[int(a * fs) : int(b * fs)].astype(np.float64), fs, "?")["value"]
        return {"period_s": v}
    if qtype == "artifact":
        p, t = artifact(x, fs)
        return {"present": p, "t": t}
    if qtype == "spike_count":
        return {"count": spike_count(x, fs)}
    if qtype == "mean_hr":
        from tsmcp.plugins.ecg import ecg_hr

        return {"bpm": ecg_hr(x, fs, "mV")["value"]}
    return None
