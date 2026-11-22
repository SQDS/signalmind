"""Frozen single-scale discord baseline from 550f594, for paired dev validation only."""

import numpy as np

from tsmcp.core.detect import _period_samples
from tsmcp.core.dsp import fill_nan

DISCORD_POINTS = 64
DISCORD_MAX_WINDOWS = 6000


def single_scale_discords(x: np.ndarray, fs: float, unit: str = "?", t0: float = 0.0, k: int = 5, **_) -> list[dict]:
    """Most unusual stretches: windows (one dominant period long) whose nearest non-overlapping neighbour is
    farthest away, after z-normalisation (matrix-profile discords). Score = that distance / median distance."""
    x = fill_nan(np.asarray(x, dtype=np.float64))
    n = x.size
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
        out.append({"type": "discord", "t": t0 + starts[i] / fs, "dur": m / fs, "score": float(score[i] / med),
                    "detail": f"window = 1 dominant period ({m} samples)"})  # fmt: skip
        lo, hi = max(0, i - 2 * excl), min(score.size, i + 2 * excl + 1)
        score[lo:hi] = -np.inf
    return out
