"""Exact measurements. Every result carries value, unit, method and a self-describing ref."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tsmcp.core.compact import sig
from tsmcp.core.dsp import (
    autocorr,
    detrend_linear,
    fill_nan,
    finite,
    parabolic_peak,
    welch_psd,
)
from tsmcp.core.errors import SignalError
from tsmcp.core.evidence import ref
from tsmcp.core.signal import Channel

MAX_PERIOD_FRACTION = 0.5  # a period must repeat at least twice inside the window


def _prep(x: np.ndarray) -> np.ndarray:
    if x.size < 8:
        raise SignalError("window has fewer than 8 samples; widen it")
    if np.mean(~np.isfinite(x)) > 0.5:
        raise SignalError("window is more than 50% NaN; choose another window")
    return fill_nan(x)


def period(x: np.ndarray, fs: float, unit: str, t0: float = 0.0, **_) -> dict:
    """Fundamental period from the first autocorrelation peak after the first zero crossing."""
    y = detrend_linear(_prep(x))
    max_lag = int(y.size * MAX_PERIOD_FRACTION)
    if max_lag < 3:
        raise SignalError("window too short to estimate a period")
    r = autocorr(y, max_lag)
    neg = np.flatnonzero(r < 0)
    if neg.size == 0:
        return {
            "value": None,
            "unit": "s",
            "method": "acf",
            "note": "no periodicity (autocorrelation never crosses zero)",
        }
    seg = r[neg[0] :]
    i = int(np.argmax(seg)) + neg[0]
    strength = float(r[i])
    if strength < 0.2:
        return {
            "value": None,
            "unit": "s",
            "method": "acf",
            "acf_peak": sig(strength),
            "note": "no clear periodicity (acf peak < 0.2)",
        }
    lag = parabolic_peak(r, i)
    return {
        "value": sig(lag / fs, 4),
        "unit": "s",
        "method": "acf",
        "acf_peak": sig(strength),
    }


def dominant_freq(x: np.ndarray, fs: float, unit: str, t0: float = 0.0, **_) -> dict:
    """Frequency of the largest Welch-PSD peak above DC."""
    f, p = welch_psd(detrend_linear(_prep(x)), fs)
    if f.size < 3:
        raise SignalError("window too short for a spectrum")
    k = int(np.argmax(p[1:])) + 1
    kk = parabolic_peak(np.log(p + 1e-300), k)
    df = f[1] - f[0]
    frac = float(p[k] / p[1:].sum()) if p[1:].sum() > 0 else 0.0
    return {
        "value": sig(kk * df, 4),
        "unit": "Hz",
        "method": "welch",
        "resolution_hz": sig(df),
        "power_frac": sig(frac),
    }


def amplitude_pp(x: np.ndarray, fs: float, unit: str, t0: float = 0.0, **_) -> dict:
    """Robust peak-to-peak: 99.5th minus 0.5th percentile (ignores isolated spikes)."""
    v = finite(_prep(x))
    lo, hi = np.percentile(v, [0.5, 99.5])
    return {
        "value": sig(hi - lo),
        "unit": unit,
        "method": "p99.5-p0.5",
        "max_minus_min": sig(v.max() - v.min()),
    }


def rms(x: np.ndarray, fs: float, unit: str, t0: float = 0.0, **_) -> dict:
    v = finite(x)
    return {
        "value": sig(float(np.sqrt(np.mean(v**2)))),
        "unit": unit,
        "method": "rms",
        "ac_rms": sig(float(v.std())),
    }


def mean(x: np.ndarray, fs: float, unit: str, t0: float = 0.0, **_) -> dict:
    return {"value": sig(float(np.mean(finite(x)))), "unit": unit, "method": "mean"}


# A measure is f(x, fs, unit, t0, ch=...) -> dict with at least "value" and "unit"; t0 = time of x[0] (s).
# `ch` is the full Channel, so a measure may cache whole-channel work (e.g. beat detection) and slice by window.
MeasureFn = Callable[..., dict]

MEASURES: dict[str, MeasureFn] = {
    "period": period,
    "dominant_freq": dominant_freq,
    "amplitude_pp": amplitude_pp,
    "rms": rms,
    "mean": mean,
}


def measure(
    sid: str,
    ch: Channel,
    x: np.ndarray,
    t0: float,
    t1: float,
    what: list[str],
    tdec: int,
    registry: dict[str, MeasureFn] | None = None,
) -> list[dict]:
    reg = MEASURES if registry is None else registry
    unknown = [w for w in what if w not in reg]
    if unknown:
        raise SignalError(f"unknown measure(s) {unknown}; available: {', '.join(reg)}")
    out = []
    win = f"{round(t0, tdec):g}-{round(t1, tdec):g}"
    for w in what:
        res = {"name": w, "ch": ch.name, **reg[w](x, ch.fs, ch.unit, t0, ch=ch)}
        res["ref"] = ref(sid, f"m.{w}", win + (f"/{ch.name}" if ch.name else ""))
        out.append(res)
    return out
