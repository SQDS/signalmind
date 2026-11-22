"""Generic event detectors. Each returns a list of events {type, t, dur, score, detail} with times in absolute seconds.

Design rule learned from the prototype: never detect "changes" with a mean-shift model on raw samples — oscillatory
signals (e.g. respiratory modulation) fool it. Changes are found on windowed *feature* series (mean, RMS,
oscillation frequency) with a robust (MAD) noise scale AND an absolute minimum effect size.
"""

from __future__ import annotations

import numpy as np

from tsmcp.core.dsp import fill_nan


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) >= min_len
    return list(zip(s[keep].tolist(), e[keep].tolist(), strict=True))


def spikes(x: np.ndarray, fs: float, unit: str = "?", t0: float = 0.0, **_) -> list[dict]:
    """Short transients (<~0.2 s) far outside the local signal range (robust z > 8 against a 0.2 s moving average)."""
    x = fill_nan(np.asarray(x, dtype=np.float64))
    w = max(3, int(0.2 * fs) | 1)
    base = np.convolve(x, np.ones(w) / w, mode="same")
    r = np.abs(x - base)
    scale = 1.4826 * np.median(r) + 1e-12
    thr = max(8 * scale, 3 * float(np.std(x)))
    idx = np.flatnonzero(r > thr)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > int(0.1 * fs)) + 1)
    out = []
    for g in groups:
        k = g[np.argmax(r[g])]
        out.append(
            {
                "type": "spike",
                "t": t0 + k / fs,
                "dur": (g[-1] - g[0] + 1) / fs,
                "score": float(r[k] / scale),
                "caveat": "sharp normal beats or edges can look like spikes; verify with view='verify'",
            }
        )
    return out


def flat(x: np.ndarray, fs: float, unit: str = "?", t0: float = 0.0, **_) -> list[dict]:
    """Sensor stuck: >= 1 s with zero sample-to-sample change."""
    x = np.asarray(x, dtype=np.float64)
    mask = np.concatenate([[False], np.abs(np.diff(x)) < 1e-12])
    return [
        {"type": "flat", "t": t0 + a / fs, "dur": (b - a) / fs, "score": (b - a) / fs} for a, b in _runs(mask, int(fs))
    ]


def clip(x: np.ndarray, fs: float, unit: str = "?", t0: float = 0.0, **_) -> list[dict]:
    """Saturation: >= 0.5 s of samples within 0.1% of the record's min or max (rail)."""
    x = np.asarray(x, dtype=np.float64)
    v = x[np.isfinite(x)]
    if v.size == 0:
        return []
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return []
    tol = 1e-3 * (hi - lo)
    out = []
    for level, mask in (("high", np.abs(x - hi) <= tol), ("low", np.abs(x - lo) <= tol)):
        for a, b in _runs(mask, max(3, int(0.5 * fs))):
            out.append({"type": "clip", "t": t0 + a / fs, "dur": (b - a) / fs, "score": (b - a) / fs, "detail": level})
    return out


def _crossing_hz(seg: np.ndarray, fs: float) -> float:
    """Oscillation frequency from median crossings with hysteresis, timed between the first and last crossing
    (not crossings / window length), so short windows are not quantized to whole half-cycles."""
    d = seg - np.median(seg)
    h = 0.25 * d.std()
    if h == 0:
        return 0.0
    idx = np.flatnonzero(np.abs(d) > h)
    s = np.sign(d[idx])
    cross = idx[1:][np.diff(s) != 0]  # first sample past the band after each sign change
    if cross.size < 2:
        return 0.0
    return float((cross.size - 1) / 2.0 / ((cross[-1] - cross[0]) / fs))


def feature_series(x: np.ndarray, fs: float, win_s: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-window mean, RMS (std) and oscillation frequency. Returns (window start offsets in s, features)."""
    w = max(8, int(win_s * fs))
    nb = x.size // w
    if nb == 0:
        return np.empty(0), {}
    blocks = fill_nan(np.asarray(x[: nb * w], dtype=np.float64)).reshape(nb, w)
    feats = {
        "level": blocks.mean(axis=1),
        "rms": blocks.std(axis=1),
        "freq": np.array([_crossing_hz(b, fs) for b in blocks]),
    }
    return np.arange(nb) * (w / fs), feats


def _robust_sd(v: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else 0.0


def _best_split(y: np.ndarray, min_seg: int) -> tuple[int, float, float, float]:
    """Best single mean-shift split of a feature series.

    Returns (index, cusum z, relative step, effect size). The effect size compares the step with the spread
    *inside* each side (robust SD): a slow oscillation has a large spread relative to any split, a true step does not.
    """
    n = y.size
    if n < 2 * min_seg:
        return -1, 0.0, 0.0, 0.0
    scale = 1.4826 * np.median(np.abs(np.diff(y))) / np.sqrt(2) + 1e-12
    ref = np.median(np.abs(y)) + 1e-12
    cs = np.concatenate([[0], np.cumsum(y)])
    k = np.arange(min_seg, n - min_seg + 1)
    left = cs[k] / k
    right = (cs[n] - cs[k]) / (n - k)
    d = np.abs(left - right)
    z = d / scale * np.sqrt(k * (n - k) / n)  # CUSUM-style statistic (locates the split)
    i = int(np.argmax(z))
    kk = int(k[i])
    spread = np.sqrt((_robust_sd(y[:kk]) ** 2 * kk + _robust_sd(y[kk:]) ** 2 * (n - kk)) / n) + 1e-12
    return kk, float(z[i]), float(d[i] / ref), float(d[i] / spread)


def changes(x: np.ndarray, fs: float, unit: str = "?", t0: float = 0.0, max_changes: int = 10, **_) -> list[dict]:
    """Abrupt, persistent changes in level, amplitude (RMS) or oscillation frequency (binary segmentation on
    windowed features). Gradual/periodic modulation is ignored because both a robust z >= 8 and a >= 15%
    relative step are required, and each side must last >= 6 windows."""
    dur = x.size / fs
    win = float(np.clip(dur / 400, 1.0, 60.0))
    starts, feats = feature_series(x, fs, win)
    if starts.size < 12:
        return []
    out: list[dict] = []
    segments = [(0, starts.size)]
    while segments and len(out) < max_changes:
        a, b = segments.pop(0)
        best = None
        for name, y in feats.items():
            k, z, rel, eff = _best_split(y[a:b], 6)
            if k > 0 and z >= 8 and rel >= 0.15 and eff >= 3.0 and (best is None or z > best[1]):
                best = (a + k, z, rel, name, y[a : a + k].mean(), y[a + k : b].mean())
        if best is None:
            continue
        k, z, rel, name, before, after = best
        # refine the boundary to sample resolution inside the two windows around it
        t_change = t0 + starts[k]
        out.append(
            {
                "type": "change",
                "t": t_change,
                "dur": 0.0,
                "score": z,
                "detail": f"{name}: {before:.3g} -> {after:.3g}",
                "resolution_s": win,
                "caveat": "frequent premature beats can look like a frequency step; verify with view='verify'",
            }
        )
        segments += [(a, k), (k, b)]
    return sorted(out, key=lambda e: e["t"])


def _period_samples(x: np.ndarray, max_lag: int) -> int | None:
    """Dominant period in samples from the first autocorrelation peak after the first zero crossing."""
    from tsmcp.core.dsp import autocorr

    y = x - x.mean()
    if y.std() == 0:
        return None
    r = autocorr(y, max_lag)
    neg = np.flatnonzero(r < 0)
    if neg.size == 0:
        return None
    i = int(np.argmax(r[neg[0] :])) + neg[0]
    return i if r[i] > 0.2 else None


DISCORD_POINTS = 64  # each candidate window is resampled to this many points before comparison
DISCORD_MAX_WINDOWS = 6000


def discords(x: np.ndarray, fs: float, unit: str = "?", t0: float = 0.0, k: int = 5, **_) -> list[dict]:
    """Most unusual stretches: windows (one dominant period long) whose nearest non-overlapping neighbour is
    farthest away, after z-normalisation (matrix-profile discords). Score = that distance / median distance."""
    x = fill_nan(np.asarray(x, dtype=np.float64))
    n = x.size
    if n < 8 or k <= 0 or np.ptp(x) == 0:
        return []
    probe = x[: min(n, 200_000)]
    m = _period_samples(probe, min(probe.size // 3, 20_000))
    if m is None:
        m = int(np.clip(n // 200, 16, 2000))
    m = int(np.clip(m, 8, max(8, n // 8)))
    step = max(1, m // 4)
    while (n - m) // step + 1 > DISCORD_MAX_WINDOWS:
        step *= 2
    starts = np.arange(0, n - m + 1, step)
    if starts.size < 8:
        return []
    # pool each window to DISCORD_POINTS samples, then z-normalise
    idx = starts[:, None] + (np.arange(DISCORD_POINTS) * m // DISCORD_POINTS)[None, :]
    span = max(1, m // DISCORD_POINTS)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    w = (cs[np.minimum(idx + span, n)] - cs[idx]) / span
    mu, sd = w.mean(axis=1, keepdims=True), w.std(axis=1, keepdims=True)
    flat_rows = sd[:, 0] < 1e-12
    z = ((w - mu) / np.where(sd < 1e-12, 1.0, sd)).astype(np.float32)
    nn = np.full(starts.size, np.inf, dtype=np.float32)
    excl = int(np.ceil(m / step))  # windows closer than one period overlap: trivial matches
    for a in range(0, starts.size, 1024):
        corr = z[a : a + 1024] @ z.T / DISCORD_POINTS
        d = np.sqrt(np.maximum(0.0, 2 * DISCORD_POINTS * (1 - corr)))
        rows = np.arange(a, min(a + 1024, starts.size))
        for off in range(-excl, excl + 1):
            cols = rows + off
            ok = (cols >= 0) & (cols < starts.size)
            d[np.flatnonzero(ok), cols[ok]] = np.inf
        nn[rows] = d.min(axis=1)
    nn[flat_rows] = 0.0  # constant stretches are the flat detector's job
    med = float(np.median(nn[np.isfinite(nn)])) or 1e-12
    out = []
    score = nn.copy()
    for _ in range(k):
        i = int(np.argmax(score))
        if not np.isfinite(score[i]) or score[i] <= 0:
            break
        out.append(
            {
                "type": "discord",
                "t": t0 + starts[i] / fs,
                "dur": m / fs,
                "score": float(score[i] / med),
                "detail": f"window = 1 dominant period ({m} samples)",
                "caveat": "relative shape outlier, not proof of an anomaly; compare candidates with view='verify'",
            }
        )
        lo, hi = max(0, i - 2 * excl), min(score.size, i + 2 * excl + 1)
        score[lo:hi] = -np.inf
    return out


DETECTORS = {"spike": spikes, "flat": flat, "clip": clip, "change": changes, "discord": discords}
