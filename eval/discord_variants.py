"""Dev-only ranking experiments; no labels, ids, or dataset-specific parameters enter a scorer.

RRF uses k=10, one-based ranks and 20 non-overlapping candidates per source. All sources have equal weight.
Correspondence uses centres within the larger window; each source contributes only its best matching rank.
"""

from dataclasses import dataclass

import numpy as np

from eval.discord_baseline import single_scale_discords
from tsmcp.core.detect import _period_samples, feature_series
from tsmcp.core.dsp import fill_nan

POINTS = 64
MAX_WINDOWS = 6000
RANK_DEPTH = 20
RRF_K = 10


@dataclass(frozen=True)
class Candidate:
    start: int
    width: int
    score: float
    source: str

    @property
    def centre(self):
        return self.start + self.width / 2


def profile(x, width, cs):
    """Same grid and float32 distances as the original, reducing correlation before sqrt."""
    step = max(1, width // 4)
    while (x.size - width) // step + 1 > MAX_WINDOWS:
        step *= 2
    starts = np.arange(0, x.size - width + 1, step)
    if starts.size < 8:
        return starts, np.zeros(starts.size), step
    idx = starts[:, None] + (np.arange(POINTS) * width // POINTS)[None, :]
    span = max(1, width // POINTS)
    w = (cs[np.minimum(idx + span, x.size)] - cs[idx]) / span
    sd = w.std(axis=1, keepdims=True)
    z = ((w - w.mean(axis=1, keepdims=True)) / np.where(sd < 1e-12, 1.0, sd)).astype(np.float32)
    nn = np.full(starts.size, np.inf, dtype=np.float32)
    excl = int(np.ceil(width / step))
    for a in range(0, starts.size, 1024):
        corr = z[a : a + 1024] @ z.T
        rows = np.arange(a, min(a + 1024, starts.size))
        for off in range(-excl, excl + 1):
            cols = rows + off
            ok = (cols >= 0) & (cols < starts.size)
            corr[np.flatnonzero(ok), cols[ok]] = -np.inf
        nn[rows] = np.sqrt(np.maximum(0.0, 2 * POINTS * (1 - corr.max(axis=1) / POINTS)))
    nn[sd[:, 0] < 1e-12] = 0.0
    return starts, nn, step


def ranked(starts, scores, width, source, radius, depth=RANK_DEPTH):
    """Greedy ranking with original-style two-window suppression within a source."""
    remaining = np.where(np.isfinite(scores), scores, -np.inf).copy()
    out = []
    for _ in range(depth):
        if not remaining.size:
            break
        i = int(np.argmax(remaining))
        if remaining[i] <= 0:
            break
        out.append(Candidate(int(starts[i]), width, float(remaining[i]), source))
        remaining[np.abs(starts - starts[i]) <= radius] = -np.inf
    return out


def duplicate(a, b):
    return abs(a.centre - b.centre) <= max(a.width, b.width)


def merge(candidates, k):
    selected = []
    for candidate in candidates:
        if any(duplicate(candidate, other) for other in selected):
            continue
        selected.append(candidate)
        if len(selected) >= k:
            break
    return selected


def rrf(lists, k):
    candidates = []
    for ranking in lists:
        for candidate in ranking:
            score = 0.0
            for source in lists:
                for rank, other in enumerate(source, 1):
                    if duplicate(candidate, other):
                        score += 1 / (RRF_K + rank)
                        break
            candidates.append(Candidate(candidate.start, candidate.width, score, candidate.source))
    return merge(sorted(candidates, key=lambda c: -c.score), k)


def robust_z(values):
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    med = np.median(values[finite])
    mad = np.median(np.abs(values[finite] - med))
    return np.where(finite, (values - med) / max(1.4826 * float(mad), 1e-12), 0)


def cheap_rankings(x, period):
    # Period-difference: trailing lag P, mean absolute difference over a P-sample window.
    width = period
    delta = np.abs(x[period:] - x[:-period])
    cs = np.r_[0.0, np.cumsum(delta)]
    starts = np.arange(period, x.size - width + 1, max(1, width // 4))
    scores = (cs[starts - period + width] - cs[starts - period]) / width
    difference = ranked(starts, scores, width, "period-difference", 2 * width)

    # Adjacent rolling feature-window changes, capped at 2,000 feature windows for cheap frequency estimation.
    # feature_series's 'rms' is std, not root-mean-square about zero; retain its existing definition.
    width = max(period, int(np.ceil(x.size / 2000)))
    offsets, feats = feature_series(x, 1.0, float(width))
    if offsets.size < 2:
        return difference, []
    changes = [np.abs(np.diff(feats[name])) for name in ("rms", "freq")]
    score = sum(np.maximum(0, robust_z(change)) for change in changes)
    # Centre the reported candidate on the boundary between the two compared feature windows.
    starts = offsets[1:].astype(int) - width // 2
    features = ranked(starts, score, width, "rms/frequency-change", 2 * width)
    return difference, features


def variant_discords(x, fs, unit="?", t0=0.0, k=5, *, variant, **_):
    x = fill_nan(np.asarray(x, dtype=np.float64))
    n = x.size
    if n < 64 or k <= 0 or np.ptp(x) == 0:
        return []
    probe = x[: min(n, 200_000)]
    period = _period_samples(probe, min(probe.size // 3, 20_000))
    primary_width = int(np.clip(period if period is not None else np.clip(n // 200, 16, 2000), 8, n // 8))
    scales = (
        [(period, "1x"), (period * 0.5, "0.5x"), (period * 2, "2x")]
        if period is not None
        else [(n / 100, "n/100"), (n / 400, "n/400"), (n / 25, "n/25")]
    )
    cs = np.r_[0.0, np.cumsum(x)]
    lists, candidates, seen = [], [], set()
    for length, label in scales[1:] if variant == "append" else scales:
        width = int(np.clip(length, 8, n // 8))
        if width in seen:
            continue
        seen.add(width)
        starts, nn, step = profile(x, width, cs)
        scores = robust_z(nn) if variant == "robust_z" else nn / (float(np.median(nn[np.isfinite(nn)])) or 1e-12)
        if variant == "robust_z":
            candidates.extend(
                Candidate(int(starts[i]), width, float(scores[i]), label)
                for i in np.flatnonzero(np.isfinite(scores) & (scores > 0))
            )
        else:
            lists.append(ranked(starts, scores, width, label, 2 * int(np.ceil(width / step)) * step))
    if variant == "append":
        # Preserve the complete original five-candidate list, including scores and ordering.
        primary = single_scale_discords(x, fs, unit, t0, k=5)
        selected = [Candidate(round((e["t"] - t0) * fs), round(e["dur"] * fs), e["score"], "1x") for e in primary]
        side = sorted((c for source in lists for c in source), key=lambda c: -c.score)
        for candidate in side:
            if not any(duplicate(candidate, other) for other in selected):
                selected.append(candidate)
        extras = selected[len(primary) : k]
        # Append scores stay below the primary scores so score-sorting clients preserve the prefix too.
        floor = min((e["score"] for e in primary), default=1.0)
        return primary[:k] + [
            event(Candidate(c.start, c.width, floor / (i + 2), c.source), fs, t0) for i, c in enumerate(extras)
        ]
    if variant == "robust_z":
        selected = merge(sorted(candidates, key=lambda c: -c.score), k)
    elif variant in ("rrf", "independent"):
        if variant == "independent":
            lists.extend(cheap_rankings(x, primary_width))
        selected = rrf(lists, k)
    else:
        raise ValueError(f"unknown variant: {variant}")
    return [event(c, fs, t0) for c in selected]


def event(candidate, fs, t0):
    return {
        "type": "discord",
        "t": t0 + candidate.start / fs,
        "dur": candidate.width / fs,
        "score": candidate.score,
        "detail": f"window = {candidate.source} ({candidate.width} samples)",
        "caveat": "relative outlier, not proof of an anomaly; compare candidates with view='verify'",
    }
