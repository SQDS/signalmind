"""Arterial pulse detection and per-beat pressure features (numpy only).

Guided search ported from the user's `hemoworld/beats.py::detect_abp_pulses_guided` (R-peak guided foot/systolic
search, PAT gate 32-600 ms); Savitzky-Golay smoothing re-implemented as a least-squares FIR. The unguided mode is a
prominence-based systolic peak search with the pre-systolic minimum as the foot. Both modes are explicit methods;
neither falls back to the other.

A beat is the half-open span [foot_i, foot_{i+1}). SBP = maximum raw sample in the beat, DBP = minimum raw sample
within +-20 ms of the foot, PP = SBP - DBP. Every beat is kept in the table with an accepted flag and a reason.
"""

from __future__ import annotations

import numpy as np

from tsmcp.core.dsp import autocorr, detrend_linear, fill_nan, parabolic_peak
from tsmcp.core.peaks import find_peaks

METHOD_VERSION = "pressure.pulses.v1"
MIN_BEAT_S, MAX_BEAT_S = 0.2, 2.5  # absolute sanity bounds only (300-24 bpm); rhythm rules below do the work
MIN_PP_MMHG = 8.0  # as in the source implementation
MIN_PP_RELATIVE = 0.2  # when the unit is unknown: fraction of the median pulse pressure
FLAT_S = 0.1
CLIP_RUN = 3
SPIKE_FRACTION = 0.5  # one-sample jump larger than this fraction of the channel's robust range is an artefact
PROMINENCE_FRACTION = 0.15  # unguided systolic peaks: prominence relative to the smoothed signal's p1-p99 range
BLOCK_S = 60.0  # unguided detection runs per block with its own period estimate (heart rate drifts)
MIN_DISTANCE_S, MAX_DISTANCE_S = 0.2, 0.75
MERGED_FACTOR, SPLIT_FACTOR = 1.6, 0.55  # beat duration vs the reference period (ACF period or local RR)
REJECT_ORDER = (
    "incomplete_boundary",
    "nonfinite",
    "spike",
    "clipped",
    "flat",
    "implausible_duration",
    "missed_beats",
    "split_beat",
    "low_pulse_pressure",
    "out_of_range",
)


def savgol_coeffs(window: int, polyorder: int = 2) -> np.ndarray:
    """Least-squares FIR for the centre sample (identical to scipy.signal.savgol_coeffs for odd windows)."""
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and larger than polyorder")
    half = window // 2
    m = np.arange(-half, half + 1, dtype=np.float64)
    a = np.vander(m, polyorder + 1, increasing=True)
    return np.linalg.pinv(a)[0][::-1]


def smooth(x: np.ndarray, fs: float) -> np.ndarray:
    w = max(5, int(round(0.08 * fs)) | 1)
    if x.size < w:
        return x.copy()
    h = savgol_coeffs(w, 2)
    k = w // 2
    xp = np.concatenate([x[k:0:-1], x, x[-2 : -k - 2 : -1]])
    return np.convolve(xp, h, mode="valid")


def detect_pulses_guided(abp: np.ndarray, rpeaks: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """At most one foot and systolic peak per R-R cycle (port of hemoworld.beats.detect_abp_pulses_guided)."""
    signal = fill_nan(np.asarray(abp, dtype=np.float64))
    sm = smooth(signal, fs)
    feet: list[int] = []
    systolic: list[int] = []
    ref: list[float] = []
    for index, r_value in enumerate(rpeaks):
        rpeak = int(r_value)
        next_r = int(rpeaks[index + 1]) if index + 1 < len(rpeaks) else len(signal) - 1
        peak_lo = rpeak + int(round(0.10 * fs))
        peak_hi = min(len(signal) - 1, rpeak + int(round(0.75 * fs)), next_r + int(round(0.30 * fs)))
        if peak_hi <= peak_lo:
            continue
        peak = peak_lo + int(np.argmax(sm[peak_lo : peak_hi + 1]))
        foot_lo = max(0, rpeak + int(round(0.032 * fs)))
        foot_hi = peak - int(round(0.040 * fs))
        if foot_hi <= foot_lo:
            continue
        foot = foot_lo + int(np.argmin(sm[foot_lo : foot_hi + 1]))
        pat = (foot - rpeak) / fs
        if not (0.032 <= pat <= 0.60):
            continue
        if feet and foot <= feet[-1]:
            continue
        feet.append(foot)
        systolic.append(peak)
        r_local = rpeaks[max(0, index - 20) : index + 21]
        rr = np.diff(r_local) / fs
        ref.append(float(np.median(rr)) if rr.size >= 3 else float("nan"))
    return np.asarray(feet, np.int64), np.asarray(systolic, np.int64), np.asarray(ref, dtype=float)


def dominant_period(x: np.ndarray, fs: float) -> float | None:
    """ACF period of a block (same estimator as the generic `period` measure); None without clear periodicity."""
    y = detrend_linear(np.asarray(x, dtype=np.float64))
    max_lag = int(y.size * 0.5)
    if max_lag < 3 or y.size < 8 or not np.isfinite(y).all() or y.std() == 0:
        return None
    r = autocorr(y, max_lag)
    neg = np.flatnonzero(r < 0)
    if neg.size == 0:
        return None
    i = int(np.argmax(r[neg[0] :])) + neg[0]
    if r[i] < 0.2:
        return None
    lag = parabolic_peak(r, i) / fs
    return float(lag) if MIN_BEAT_S <= lag <= MAX_BEAT_S else None  # the same physiological bounds as a beat


def detect_pulses_unguided(abp: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Systolic peaks by prominence with a refractory distance of 0.6 x the block's ACF period (so a dicrotic notch
    is not a second pulse); foot = minimum between the previous peak and this one. Returns (feet, peaks, ref_period)
    where ref_period[i] is the period estimate the beat was detected under (NaN when none)."""
    signal = fill_nan(np.asarray(abp, dtype=np.float64))
    if signal.size < int(2 * fs):
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    sm = smooth(signal, fs)
    block, pad = int(BLOCK_S * fs), int(2 * fs)
    peaks_all: list[int] = []
    period_all: list[float] = []
    for core_start in range(0, sm.size, block):
        core_end = min(sm.size, core_start + block)
        a, b = max(0, core_start - pad), min(sm.size, core_end + pad)
        seg = sm[a:b]
        lo, hi = np.percentile(seg, [1, 99])
        prom = PROMINENCE_FRACTION * float(hi - lo)
        if prom <= 0:
            continue
        t = dominant_period(seg, fs)
        dist = 0.6 * t if t else MIN_DISTANCE_S  # without a period estimate, only the physiological floor applies
        dist = min(MAX_DISTANCE_S, max(MIN_DISTANCE_S, dist))
        found = find_peaks(seg, distance=max(1, int(round(dist * fs))), prominence=prom) + a
        for p in found:
            if core_start <= p < core_end:
                peaks_all.append(int(p))
                period_all.append(t if t else float("nan"))
    peaks = np.asarray(peaks_all, np.int64)
    if peaks.size < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    feet = np.empty(peaks.size, np.int64)
    back = int(round(0.6 * fs))
    for i, p in enumerate(peaks):
        start = peaks[i - 1] if i else max(0, p - back)
        feet[i] = start + int(np.argmin(sm[start : p + 1]))
    return feet, peaks, np.asarray(period_all, dtype=float)


def _flat_run(seg: np.ndarray, fs: float) -> bool:
    d = np.diff(seg) == 0
    if not d.any():
        return False
    need = max(1, int(round(FLAT_S * fs)))
    run = best = 0
    for v in d:
        run = run + 1 if v else 0
        best = max(best, run)
    return best >= need


def _clipped(seg: np.ndarray, lo: float, hi: float) -> bool:
    span = (hi - lo) * 1e-3
    at = (seg >= hi - span) | (seg <= lo + span)
    if at.sum() < CLIP_RUN:
        return False
    run = 0
    for v in at:
        run = run + 1 if v else 0
        if run >= CLIP_RUN:
            return True
    return False


def beat_table(
    x: np.ndarray,
    fs: float,
    t0: float,
    unit: str,
    feet: np.ndarray,
    peaks: np.ndarray,
    i_start: int,
    i_end: int,
    channel_range: tuple[float, float],
    ref_period: np.ndarray | None = None,
) -> list[dict]:
    """Rows for every detected beat that touches [i_start, i_end) of x (indices into x; t0 = time of x[0]).

    Beats whose span is not entirely inside the window are kept and rejected as `incomplete_boundary`.
    """
    pressure = unit in ("mmHg", "kPa", "cmH2O")
    finite_all = x[np.isfinite(x)]
    robust_range = float(np.percentile(finite_all, 99) - np.percentile(finite_all, 1)) if finite_all.size else 0.0
    rows: list[dict] = []
    pp_all = []
    half = max(1, int(round(0.02 * fs)))
    for i, foot in enumerate(feet):
        end = int(feet[i + 1]) if i + 1 < len(feet) else None
        if end is not None and end <= i_start:
            continue
        if foot >= i_end:
            break
        seg_end = end if end is not None else min(x.size, foot + int(MAX_BEAT_S * fs))
        seg = x[foot:seg_end]
        if seg.size == 0:
            continue
        sbp = float(np.nanmax(seg)) if np.isfinite(seg).any() else float("nan")
        dbp_seg = x[max(0, foot - half) : foot + half + 1]
        dbp = float(np.nanmin(dbp_seg)) if np.isfinite(dbp_seg).any() else float("nan")
        pp = sbp - dbp
        reasons = []
        if end is None or foot < i_start or end > i_end:
            reasons.append("incomplete_boundary")
        if not np.isfinite(seg).all() or not np.isfinite(dbp):
            reasons.append("nonfinite")
        else:
            if _flat_run(seg, fs):
                reasons.append("flat")
            if _clipped(seg, *channel_range):
                reasons.append("clipped")
            if seg.size > 1 and np.abs(np.diff(seg)).max() > SPIKE_FRACTION * robust_range:
                reasons.append("spike")
        dur = (end - foot) / fs if end is not None else None
        if dur is not None and not (MIN_BEAT_S <= dur <= MAX_BEAT_S):
            reasons.append("implausible_duration")
        if pressure and np.isfinite(pp) and pp < MIN_PP_MMHG:
            reasons.append("low_pulse_pressure")
        if pressure and np.isfinite(pp) and (dbp < 10 or sbp > 300 or sbp < 20):
            reasons.append("out_of_range")
        row = {
            "_ref_period": float(ref_period[i]) if ref_period is not None and i < len(ref_period) else float("nan"),
            "beat": len(rows) + 1,
            "t_foot_s": round(float(t0 + foot / fs), 4),
            "t_peak_s": round(float(t0 + int(peaks[i]) / fs), 4) if i < len(peaks) else None,
            "t_end_s": round(float(t0 + end / fs), 4) if end is not None else None,
            "duration_s": round(float(dur), 4) if dur is not None else None,
            "sbp": round(sbp, 3) if np.isfinite(sbp) else None,
            "dbp": round(dbp, 3) if np.isfinite(dbp) else None,
            "pp": round(pp, 3) if np.isfinite(pp) else None,
            "accepted": not reasons,
            "reject_reason": None,
            "_reasons": reasons,
        }
        rows.append(row)
        if np.isfinite(pp) and not reasons:
            pp_all.append(pp)
    if not pressure and pp_all:
        med = float(np.median(pp_all))
        for r in rows:
            if r["pp"] is not None and r["pp"] < MIN_PP_RELATIVE * med:
                r["_reasons"].append("low_pulse_pressure")
                r["accepted"] = False
    # rhythm-relative duration rules against an independent reference period (ACF period of the block, or the local
    # R-R median in guided mode): a beat much longer than the period hides missed beats (its maximum would stand in
    # for several beats); a much shorter one is a split beat (e.g. a dicrotic notch taken for a pulse)
    durs = np.array([r["duration_s"] if r["duration_s"] is not None else np.nan for r in rows], dtype=float)
    for i, r in enumerate(rows):
        if r["duration_s"] is None:
            continue
        ref = r["_ref_period"]
        if not np.isfinite(ref):
            local = durs[max(0, i - 20) : i + 21]
            local = local[np.isfinite(local)]
            if local.size < 5:
                continue
            ref = float(np.median(local))
        if r["duration_s"] > MERGED_FACTOR * ref:
            r["_reasons"].append("missed_beats")
        elif r["duration_s"] < SPLIT_FACTOR * ref:
            r["_reasons"].append("split_beat")
    for r in rows:
        r.pop("_ref_period")
        rs = r.pop("_reasons")
        r["reject_reason"] = next((k for k in REJECT_ORDER if k in rs), None) if rs else None
        r["accepted"] = not rs
    return rows


def summarize(rows: list[dict], t0: float, t1: float, unit: str) -> dict:
    acc = [r for r in rows if r["accepted"]]
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            reasons[r["reject_reason"]] = reasons.get(r["reject_reason"], 0) + 1
    out: dict = {
        "n_beats": len(rows),
        "counts": {"accepted": len(acc), "rejected": len(rows) - len(acc), "rejection_reasons": reasons},
    }
    dur = sum(r["duration_s"] for r in acc)
    out["coverage"] = round(float(dur) / (t1 - t0), 4) if t1 > t0 else None
    excluded = [(r["t_foot_s"], r["t_end_s"] or r["t_foot_s"]) for r in rows if not r["accepted"]]
    out["intervals_excluded"] = _merge(excluded)
    for col in ("sbp", "dbp", "pp"):
        v = np.array([r[col] for r in acc], dtype=float)
        if v.size:
            out[f"{col}_mean"] = float(v.mean())
            out[f"{col}_sem"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
    warnings = []
    pressure = unit in ("mmHg", "kPa", "cmH2O")
    if not pressure:
        warnings.append(
            {
                "code": "unit_unknown",
                "severity": "warning",
                "message": f"channel unit is {unit!r}: values are source amplitude; mmHg calibration unverified",
            }
        )
    if rows and len(acc) / len(rows) < 0.8:
        warnings.append(
            {
                "code": "many_rejected_beats",
                "severity": "warning",
                "message": f"{len(rows) - len(acc)} of {len(rows)} beats rejected: {reasons}",
            }
        )
    if out["coverage"] is not None and out["coverage"] < 0.6:
        warnings.append(
            {
                "code": "low_coverage",
                "severity": "warning",
                "message": f"accepted beats cover {out['coverage']:.0%} of the window",
            }
        )
    if "clipped" in reasons:
        warnings.append(
            {"code": "clipping_present", "severity": "warning", "message": f"{reasons['clipped']} clipped beats"}
        )
    if "missed_beats" in reasons or "split_beat" in reasons:
        n_rhythm = reasons.get("missed_beats", 0) + reasons.get("split_beat", 0)
        warnings.append(
            {
                "code": "beat_detection_irregular",
                "severity": "warning",
                "message": f"{n_rhythm} beats rejected as merged or split relative to the local rhythm",
            }
        )
    if "spike" in reasons:
        warnings.append(
            {"code": "spikes_present", "severity": "warning", "message": f"{reasons['spike']} beats with spikes"}
        )
    if "flat" in reasons:
        warnings.append(
            {"code": "flat_segments", "severity": "warning", "message": f"{reasons['flat']} beats contain flat runs"}
        )
    if 0 < len(acc) < 10:
        warnings.append({"code": "few_beats", "severity": "info", "message": f"only {len(acc)} accepted beats"})
    out["warnings"] = warnings
    return out


def _merge(spans: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
