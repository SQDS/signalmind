"""Output adapters for the existing, numerically unchanged diagnostic rules."""

import numpy as np

from tsmcp.cleaning.recipes import diagnose
from tsmcp.core.detect import changes, clip, discords, flat


def spikes_with_bounds(x, fs):
    """Same numerical rules as core.detect.spikes; retain the original group's endpoints."""
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
                "start": int(g[0]),
                "end": int(g[-1]) + 1,
                "peak_sample": int(k),
                "kind": "spike",
                "score": float(r[k] / scale),
                "metric": "legacy_spike_residual",
                "evidence": {"threshold": float(thr), "robust_scale": float(scale)},
                "resolution_samples": 1,
            }
        )
    return out


def legacy_diagnosis(values, fs):
    report = diagnose(values, fs)
    events = []
    for issue in report["issues"]:
        a, b = issue["range_samples"]
        events.append(
            {
                "start": int(a),
                "end": int(b),
                "kind": issue["kind"],
                "score": float(issue.get("contrast", issue.get("value", 0))),
                "metric": "legacy_30s_diagnosis",
                "evidence": issue,
                "resolution_samples": int(b - a),
            }
        )
    return events


def legacy_rules(values, fs):
    # The caller supplies finite continuous runs, so no interpolation is introduced.
    events = spikes_with_bounds(values, fs)
    for fn in (flat, clip):
        for event in fn(values, fs):
            a = round(event["t"] * fs)
            b = a + round(event["dur"] * fs)
            events.append(
                {
                    "start": a,
                    "end": b,
                    "kind": event["type"],
                    "score": float(event["score"]),
                    "metric": "legacy_" + event["type"],
                    "evidence": event,
                    "resolution_samples": 1,
                }
            )
    return sorted(events, key=lambda e: (e["start"], e["end"], e["kind"]))


def legacy_auxiliary(values, fs):
    points, events = [], []
    for event in changes(values, fs):
        points.append(
            {
                "sample": round(event["t"] * fs),
                "kind": "change",
                "score": event["score"],
                "resolution_samples": max(1, round(event["resolution_s"] * fs)),
                "evidence": event,
            }
        )
    for event in discords(values, fs):
        a = round(event["t"] * fs)
        events.append(
            {
                "start": a,
                "end": a + round(event["dur"] * fs),
                "kind": "relative_shape_outlier",
                "score": event["score"],
                "resolution_samples": 1,
                "evidence": event,
                "metric": "legacy_discord_nearest_neighbor",
            }
        )
    return events, points
