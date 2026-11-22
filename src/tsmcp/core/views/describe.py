"""Overview of a signal in a few hundred tokens. Only defined quantities — no pseudo-scores.

Block profile per equal time block: mean, std, and cross_hz = median-crossing rate with hysteresis / 2,
i.e. the dominant oscillation frequency for roughly sinusoidal signals (a rhythm proxy, not a spectrum).
"""

from __future__ import annotations

import numpy as np

from tsmcp.core.compact import sig, tround
from tsmcp.core.dsp import finite, run_fraction
from tsmcp.core.measure import period
from tsmcp.core.signal import Channel, Signal, time_decimals

MAX_DESCRIBE_CHANNELS = 8
N_BLOCKS = {"normal": 12, "high": 24}
PERIOD_PROBE_S = 120.0  # period is estimated on a leading probe window; stated in the output


def _stats_sample(x: np.ndarray, limit: int = 2_000_000) -> np.ndarray:
    """Strided sample for percentile stats on very long channels (exact for shorter ones)."""
    step = max(1, x.size // limit)
    return np.asarray(x[::step], dtype=np.float64)


def crossing_hz(x: np.ndarray, fs: float) -> float | None:
    """Oscillation frequency from median crossings with hysteresis (+-0.25 std band; robust to noise chatter)."""
    v = x[np.isfinite(x)]
    if v.size < 16:
        return None
    d = v - np.median(v)
    h = 0.25 * d.std()
    if h == 0:
        return 0.0
    s = np.sign(d[np.abs(d) > h])
    n_cross = int(np.count_nonzero(np.diff(s)))
    return n_cross / 2.0 / (v.size / fs)


def _quant(v: float | None, q: float) -> float | None:
    return None if v is None else sig(round(v / q) * q)


def quality(ch: Channel) -> dict:
    x = np.asarray(ch.data, dtype=np.float64)
    nan = ~np.isfinite(x)
    v = x[~nan]
    if v.size == 0:
        return {"nan_frac": 1.0}
    flat = np.concatenate([[False], np.diff(x) == 0])
    lo, hi = v.min(), v.max()
    tol = 1e-3 * (hi - lo) if hi > lo else 0
    at_rail = (np.abs(x - lo) <= tol) | (np.abs(x - hi) <= tol)
    return {
        "nan_frac": sig(nan.mean()),
        "flat_frac": sig(run_fraction(flat, max(2, int(0.1 * ch.fs)))),
        "clip_frac": sig(run_fraction(at_rail & ~nan, 3)) if hi > lo else 1.0,
    }


def channel_summary(ch: Channel, detail: str, tdec: int) -> dict:
    s = _stats_sample(np.asarray(ch.data))
    v = finite(s)
    out: dict = {"name": ch.name, "fs": sig(ch.fs, 6), "unit": ch.unit, "n": ch.n}
    if v.size:
        p1, p50, p99 = np.percentile(v, [1, 50, 99])
        out["stats"] = {
            "mean": sig(v.mean()),
            "std": sig(v.std()),
            "p1": sig(p1),
            "median": sig(p50),
            "p99": sig(p99),
        }
    out["quality"] = quality(ch)
    probe_n = int(min(ch.n, PERIOD_PROBE_S * ch.fs))
    try:
        per = period(np.asarray(ch.data[:probe_n], dtype=np.float64), ch.fs, ch.unit)
        out["period_probe"] = {
            "window": [0, tround(probe_n / ch.fs, tdec)],
            "period_s": per.get("value"),
            "acf_peak": per.get("acf_peak"),
        }
    except Exception:
        pass
    # Coarse block profile: lets the model see where level/energy changes without an image.
    nb = min(N_BLOCKS.get(detail, 12), max(1, ch.n // 16))
    edges = np.linspace(0, ch.n, nb + 1).astype(int)
    scale = float(v.std()) if v.size and v.std() > 0 else 1.0
    q = 10.0 ** np.floor(np.log10(scale * 1e-2))  # resolution: ~1% of the channel's std
    means, stds, freqs = [], [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        seg = finite(np.asarray(ch.data[a:b], dtype=np.float64))
        means.append(_quant(float(seg.mean()), q) if seg.size else None)
        stds.append(_quant(float(seg.std()), q) if seg.size else None)
        fz = crossing_hz(seg, ch.fs) if seg.size else None
        freqs.append(sig(fz) if fz is not None else None)
    out["blocks"] = {
        "dt_s": tround(ch.duration / nb, tdec),
        "mean": means,
        "std": stds,
        "cross_hz": freqs,
    }
    return out


def describe(s: Signal, detail: str = "normal") -> dict:
    tdec = time_decimals(max(c.fs for c in s.channels))
    chans = s.channels
    omitted = 0
    if len(chans) > MAX_DESCRIBE_CHANNELS:
        ranked = sorted(
            chans,
            key=lambda c: -float(np.nanstd(_stats_sample(np.asarray(c.data), 200_000))),
        )
        chans, omitted = (
            ranked[:MAX_DESCRIBE_CHANNELS],
            len(chans) - MAX_DESCRIBE_CHANNELS,
        )
    out = {
        "id": s.id,
        "dur_s": tround(s.duration, tdec),
        "start": s.start,
        "n_channels": len(s.channels),
        "channels": [channel_summary(c, detail, tdec) for c in chans],
    }
    if omitted:
        out["omitted_channels"] = omitted
        out["all_channel_names"] = [c.name for c in s.channels]
    return out
