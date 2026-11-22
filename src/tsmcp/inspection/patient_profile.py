"""Patient-relative amplitude and beat-rate review signals, fitted on observed continuous support."""

import numpy as np

from tsmcp.processing import intervals


def beat_pairs(probe):
    peaks = probe["events"]
    return [
        (a["sample"], b["sample"])
        for a, b in zip(peaks, peaks[1:], strict=False)
        if a["support_id"] == b["support_id"] and a["sample"] < b["sample"]
    ]


def robust_center(values):
    v = np.asarray(values, float)
    if not v.size or not np.isfinite(v).all():
        raise ValueError("No usable patient reference; supply finite observed beat measurements")
    median = float(np.median(v))
    return median, float(1.4826 * np.median(np.abs(v - median)))


def amplitude_reference(troughs, peaks, minimum, maximum, *, lower_limit=None, upper_limit=None):
    low, low_mad = robust_center(troughs)
    high, high_mad = robust_center(peaks)
    span = float(np.median(np.asarray(peaks) - np.asarray(troughs)))
    lower = low - max(6 * low_mad, 0.3 * span) if lower_limit is None else float(lower_limit)
    upper = high + max(6 * high_mad, 0.3 * span) if upper_limit is None else float(upper_limit)
    if (
        not np.isfinite([lower, upper]).all()
        or lower > upper
        or (lower == upper and (lower_limit is not None or upper_limit is not None))
    ):
        raise ValueError("Amplitude limits must be finite and ordered for this channel and unit")
    minimum, maximum = min(lower, minimum), max(upper, maximum)
    pad = max(0.05 * (maximum - minimum), np.spacing(max(1, abs(maximum), abs(minimum))))
    return dict(
        low=lower,
        high=upper,
        beat_min_median=low,
        beat_max_median=high,
        typical_swing=span,
        display=[float(minimum - pad), float(maximum + pad)],
        beats=len(peaks),
    )


def rate_reference(bpms):
    median, mad = robust_center(bpms)
    margin = max(10.0, 0.2 * median, 3 * mad)
    return dict(median=median, scaled_mad=mad, low=median - margin, high=median + margin, intervals=len(bpms))


def patient_flags(x, observed, fs, pairs, amp, rate):
    """Yield exact amplitude crossings and individual outlying beat intervals. All ranges are half-open."""
    for kind, mask, limit in (("y_high", x > amp["high"], amp["high"]), ("y_low", x < amp["low"], amp["low"])):
        for a, b in intervals(observed & mask):
            value = float(np.max(x[a:b]) if kind == "y_high" else np.min(x[a:b]))
            yield a, b, kind, value, float(limit)
    for a, b in pairs:
        if not observed[a : b + 1].all():
            continue
        bpm = 60.0 * fs / (b - a)
        if bpm > rate["high"]:
            yield a, b, "rate_high", bpm, rate["high"]
        elif bpm < rate["low"]:
            yield a, b, "rate_low", bpm, rate["low"]
