"""Frozen native probes. Never bridge invalid/estimated support to create intervals."""

from __future__ import annotations

import numpy as np

from tsmcp.inspection.support import native_support
from tsmcp.plugins.ecg.rpeaks import detect_r_peaks
from tsmcp.plugins.pressure.pulses import detect_pulses_unguided
from tsmcp.processing import intervals

from .common import digest

DEFINITIONS = {
    "ECG": "QRS local absolute extremum of 5-20 Hz FIR analysis copy; not a human R-apex reference",
    "PPG": "native smoothed systolic peak; intervals between adjacent peaks in the same support",
    "ABP": "native foot-to-next-foot; half-open cycle min/max; trapezoidal pressure integral including valid endpoint",
}


def probe(values, time, fs, valid, estimated, task):
    x, t = np.asarray(values, float), np.asarray(time, float)
    support = np.asarray(valid, bool) & ~np.asarray(estimated, bool) & np.isfinite(x)
    events, periods, pressures = [], [], []
    for support_id, (start, stop) in enumerate(intervals(support)):
        # A malformed caller cannot make the detector bridge a physical time gap.
        breaks = np.flatnonzero(np.abs(np.diff(t[start:stop]) - 1 / fs) > 1e-6) + start + 1
        if len(breaks):
            raise ValueError("Probe input crosses an unexplained time gap")
        if task == "ECG":
            positions = detect_r_peaks(x[start:stop], fs) + start if fs > 40 else np.empty(0, dtype=int)
            feet = None
        else:
            f, p, _ = detect_pulses_unguided(x[start:stop], fs)
            positions, feet = p + start, f + start
        offset = len(events)
        for p in positions:
            events.append({"event_id": len(events), "sample": int(p), "time_s": float(t[p]), "support_id": support_id})
        if task in ("ECG", "PPG"):
            for j, (a, b) in enumerate(zip(positions[:-1], positions[1:], strict=True)):
                periods.append(
                    {
                        "left_event_id": offset + j,
                        "right_event_id": offset + j + 1,
                        "time_s": float((t[a] + t[b]) / 2),
                        "interval_s": float(t[b] - t[a]),
                        "support_id": support_id,
                    }
                )
        else:
            for a, b in zip(feet[:-1], feet[1:], strict=True):
                if b <= a or not support[a : b + 1].all():
                    continue
                y, grid = x[a : b + 1], t[a : b + 1]
                mean = float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(grid)) / (grid[-1] - grid[0]))
                pressures.append(
                    {
                        "start_sample": int(a),
                        "stop_sample": int(b),
                        "start_s": float(t[a]),
                        "stop_s": float(t[b]),
                        "sbp": float(np.max(x[a:b])),
                        "dbp": float(np.min(x[a:b])),
                        "mean_pressure": mean,
                        "unit": "mmHg",
                        "support_id": support_id,
                    }
                )
    covered = np.zeros(len(x), bool)
    for interval in periods:
        a, b = events[interval["left_event_id"]]["sample"], events[interval["right_event_id"]]["sample"]
        covered[a:b] = True
    for cycle in pressures:
        covered[cycle["start_sample"] : cycle["stop_sample"]] = True
    return {
        "task": task,
        "definition": DEFINITIONS[task],
        "definition_hash": digest(DEFINITIONS[task]),
        "events": events,
        "intervals": periods,
        "pressures": pressures,
        "eligible_support": intervals(support),
        "detector_support": native_support(support, fs, task, events),
        "measurement_support": intervals(covered),
        "measured_samples": int(covered.sum()),
        "estimated_samples_excluded": int(np.sum(estimated)),
        "reference_accuracy": None,
    }


def match_events(predicted, reference, tolerance):
    """Ordered one-to-one association: maximize matches, then minimize total error."""
    p, r = np.asarray(predicted), np.asarray(reference)
    n, m = len(p), len(r)
    scores = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    steps = np.zeros((n + 1, m + 1), np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, direction = scores[i - 1][j], 1
            if scores[i][j - 1] > best:
                best, direction = scores[i][j - 1], 2
            error = abs(float(p[i - 1] - r[j - 1]))
            if error <= tolerance:
                prior = scores[i - 1][j - 1]
                candidate = (prior[0] + 1, prior[1] - error)
                if candidate >= best:
                    best, direction = candidate, 3
            scores[i][j], steps[i, j] = best, direction
    pairs, i, j = [], n, m
    while i and j:
        direction = steps[i, j]
        if direction == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def score_events(
    predicted, reference, *, fs, released=True, complete=True, association_ms=40, peak_ms=8, interval_ms=16
):
    if not complete:
        return {"correct": None, "reason": "incomplete_reference"}
    p, r = np.asarray(predicted), np.asarray(reference)
    pairs = match_events(p, r, fs * association_ms / 1000)
    intervals_correct = sum(
        b == a + 1 and d == c + 1 and abs(float((p[b] - p[a]) - (r[d] - r[c]))) <= fs * interval_ms / 1000
        for (a, c), (b, d) in zip(pairs[:-1], pairs[1:], strict=True)
    )
    located = sum(abs(float(p[a] - r[b])) <= fs * peak_ms / 1000 for a, b in pairs)
    correct = len(p) == len(r) == len(pairs) == located and len(r) >= 2 and intervals_correct == len(r) - 1
    return {
        "false_positive": len(p) - len(pairs),
        "false_negative": len(r) - len(pairs),
        "located": located,
        "correct_intervals": int(intervals_correct),
        "content_correct": bool(correct),
        "correct_delivery": bool(correct and released),
        "released": bool(released),
    }


def fixed_denominator_metrics(rows):
    total = sum(r["duration_s"] for r in rows)
    if not total or any(r.get("content_correct") is None for r in rows):
        return {"Y": None, "C": None, "Rbad": None, "goodloss": None, "reason": "incomplete_reference_coverage"}
    released = sum(r["duration_s"] for r in rows if r["released"])
    good = sum(r["duration_s"] for r in rows if r["released"] and r["content_correct"])
    baseline_good = sum(r["duration_s"] for r in rows if r.get("baseline_correct"))
    lost = sum(
        r["duration_s"] for r in rows if r.get("baseline_correct") and not (r["released"] and r["content_correct"])
    )
    return {
        "Y": good / total,
        "C": released / total,
        "Rbad": (released - good) / released if released else None,
        "goodloss": lost / baseline_good if baseline_good else None,
    }
