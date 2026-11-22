"""Data-only R2 evidence and accounting, derived exclusively from tool replies."""

from __future__ import annotations

import math
from fractions import Fraction

from eval import b_workflow as bw


def midpoint_paper_window(start: float, end: float, duration: float, fs: float) -> list[float] | None:
    """Thirty seconds around the returned midpoint, with earlier sample ties."""
    rate = Fraction(str(fs))
    middle = (Fraction(str(start)) + Fraction(str(end))) / 2
    # Select the earlier boundary sample; the full length stays fixed before clipping.
    first = math.floor((middle - 15) * rate)
    count = math.floor(30 * rate + Fraction(1, 2))
    total = round(Fraction(str(duration)) * rate)
    lo, hi = max(0, first), min(total, first + count)
    if hi <= lo:
        return None
    return [float(Fraction(lo) / rate), float(Fraction(hi) / rate)]


def burden_plan(sid: str, index: dict, noisy: dict, duration: float, fs: float) -> list[dict]:
    """Preserve v0 selection and gaps; change only unknown-region view arguments."""
    stretches = sorted(noisy["events"], key=lambda e: (-e["dur"], e["t"]))[:3]
    plan = bw.burden_plan(sid, index, noisy, duration)
    replaced = []
    for item in plan:
        if item["key"].startswith("unknown."):
            event = stretches[int(item["key"].split(".")[1]) - 1]
            win = midpoint_paper_window(event["t"], event["t"] + event["dur"], duration, fs)
            if win is None:
                continue
            item = {**item, "arguments": {"id": sid, "window": win, "view": "ecg_paper"}}
        replaced.append(item)
    return replaced


def _union(intervals) -> list[tuple[float, float]]:
    merged = []
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _seconds(intervals: list[tuple[float, float]], start: float, end: float) -> float:
    return math.fsum(max(0.0, min(b, end) - max(a, start)) for a, b in intervals)


def _compact(value: float) -> int | float:
    value = round(value, 6)
    return int(value) if value.is_integer() else value


def burden_arithmetic(index: dict, burden: dict, index_ref: str, burden_ref: str) -> list[dict]:
    """Keep v0 blocks verbatim and append hourly/whole-record interval arithmetic."""
    original = bw.burden_arithmetic(index, burden, index_ref, burden_ref)
    duration = burden["window_s"]
    supported = _union(interval for ep in index["episodes"] for interval in ep["intervals_s"])
    spans = _union((ep["t"], ep["t"] + ep["span_s"]) for ep in index["episodes"])
    supported_within = _union((max(a, c), min(b, d)) for a, b in supported for c, d in spans if max(a, c) < min(b, d))

    def row(start, end):
        af_s = _seconds(supported, start, end)
        span_s = _seconds(spans, start, end)
        within_s = max(0.0, span_s - _seconds(supported_within, start, end))
        outside_s = max(0.0, end - start - span_s)
        return [_compact(float(v)) for v in (start, end, af_s, span_s, within_s, outside_s)]

    return [
        *original,
        {
            "arithmetic": "support_span_accounting",
            "source_refs": [index_ref, burden_ref],
            "columns": [
                "start_s",
                "end_s",
                "supported_af_s",
                "episode_span_s",
                "within_span_non_support_unclassified_s",
                "outside_span_unclassified_s",
            ],
            "hours": [row(*hour["window"]) for hour in original[0]["hours"]],
            "whole_record": row(0, duration),
            "denominator_s": duration,
            "pct_of_window": burden["pct_of_window"],
        },
    ]
