"""Parameterized, read-only detector selected by the reviewer. Ranges are half-open."""

import numpy as np
from scipy.ndimage import median_filter

from tsmcp.processing import intervals

METRICS = ["step", "median_residual", "second_difference"]
LIMITS = {
    "scale_window_s": (2, 60),
    "baseline_window_s": (0.024, 2),
    "seed_ratio": (0.05, 20),
    "tail_ratio": (0.01, 20),
    "quiet_s": (0, 0.3),
    "max_seed_s": (0.008, 3),
}


def context_range(n, fs, target, flank_s=5):
    a, b = target
    if not 0 <= a < b <= n or flank_s not in (5, 10, 20, 40):
        raise ValueError("Use a valid target interval and a flank of 5, 10, 20 or 40 seconds")
    return max(0, a - round(flank_s * fs)), min(n, b + round(flank_s * fs))


def pattern_schema():
    fields = {"metric": {"type": "string", "enum": METRICS}}
    for name, (lo, hi) in LIMITS.items():
        fields[name] = {"type": "number", "minimum": lo, "maximum": hi}
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def validate_rule(rule):
    if set(rule) != {"metric", *LIMITS} or rule["metric"] not in METRICS:
        raise ValueError("Unknown pattern or missing parameters")
    for name, (lo, hi) in LIMITS.items():
        value = rule[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
            raise ValueError(f"{name} outside supported range")
    if rule["tail_ratio"] > rule["seed_ratio"]:
        raise ValueError("Tail ratio must not exceed seed ratio")


def detect(values, usable, fs, rule, *, evaluated=None):
    validate_rule(rule)
    found = []
    for a, b in intervals(usable & np.isfinite(values)):
        x = values[a:b]
        if len(x) < max(4, round(0.25 * fs)):
            continue
        width = max(3, round(rule["baseline_window_s"] * fs) | 1)
        if rule["metric"] == "median_residual":
            residual = np.abs(
                x - median_filter(x, size=min(width, len(x) if len(x) % 2 else len(x) - 1), mode="nearest")
            )
        elif rule["metric"] == "step":
            d = np.abs(np.diff(x))
            residual = np.maximum(np.r_[0, d], np.r_[d, 0])
        else:
            residual = np.r_[0, np.abs(np.diff(x, n=2)), 0]
        # Estimate local pulse span with robust quantiles. Interpolated scales avoid block-edge steps.
        chunk = max(4, round(rule["scale_window_s"] * fs))
        spans = []
        centers = []
        for lo in range(0, len(x), chunk):
            hi = min(len(x), lo + chunk)
            window = x[max(0, hi - chunk) : hi]
            spans.append(np.ptp(np.percentile(window, [5, 95])))
            centers.append((lo + hi - 1) / 2)
        scale = np.interp(np.arange(len(x)), centers, spans)
        supported = scale > np.finfo(float).eps * np.maximum(1, np.abs(x))
        if evaluated is not None:
            evaluated[a:b] = supported
        score = np.divide(residual, scale, out=np.zeros(len(x)), where=supported)
        seeds = intervals((score >= rule["seed_ratio"]) & supported)
        seeds = [(s, e) for s, e in seeds if e - s <= max(1, round(rule["max_seed_s"] * fs))]
        if not seeds:
            continue
        seed = np.zeros(len(x), bool)
        for s, e in seeds:
            seed[s:e] = True
        quiet = round(rule["quiet_s"] * fs)
        tails = intervals((score >= rule["tail_ratio"]) & supported)
        merged = []
        for s, e in tails:
            if merged and s - merged[-1][1] <= quiet and supported[merged[-1][1] : s].all():
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            if seed[s:e].any():
                found.append([a + s, a + e])
    return found
