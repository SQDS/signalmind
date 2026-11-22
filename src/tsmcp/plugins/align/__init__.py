"""Alignment plugin (opt-in): ECG-to-pulse timing, beat matching and polarity, with explicit timebase status.

Pulse feet use the intersecting-tangent definition (tangent at the maximum upstroke slope meets the horizontal through
the preceding minimum; sub-sample), detected on the pulse channel itself (unguided, period-aware); R-peaks come from
the toolkit's validated detector on the record's ECG. Each foot is paired with the closest preceding R within
30-800 ms, one-to-one. Delays are *observed* sample-index delays: channels of one record are index-aligned by
declaration, not verified, so these are not pulse-arrival-time claims unless synchrony is declared.

Measures (on the pulse channel): pulse_delay (median R-to-foot delay, ms, per-beat table), pulse_delay_change
(largest sustained step in the delay series: change in ms and onset), beats_missing_in_pulse / beats_missing_in_ecg
(R-R intervals without a pulse / pulses without a preceding R), and on any channel polarity_change (persistent sign
change of the waveform skewness: onset and before/after skewness).
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from tsmcp.core.errors import SignalError
from tsmcp.plugins import context
from tsmcp.plugins.base import DomainPlugin
from tsmcp.plugins.pressure.pulses import detect_pulses_unguided, smooth
from tsmcp.plugins.pulse import change_point
from tsmcp.runtime.precision import present

METHOD_VERSION = "align.v1"
PAIR_MIN_S, PAIR_MAX_S = 0.03, 0.8
POLARITY_WIN_S = 2.0
POLARITY_PERSIST = 3
_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def tangent_feet(x: np.ndarray, fs: float, feet: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    sm = smooth(np.nan_to_num(np.asarray(x, dtype=np.float64), nan=float(np.nanmedian(x))), fs)
    d = np.gradient(sm)
    out = np.full(len(feet), np.nan)
    for i, (a, p) in enumerate(zip(feet, peaks, strict=True)):
        if p <= a + 1:
            continue
        j = a + int(np.argmax(d[a:p]))
        if d[j] <= 0:
            continue
        t = j - (sm[j] - sm[a]) / d[j]
        out[i] = min(max(t, a), j) / fs
    return out


def _pairs(ch):
    ecg = context.find(ch, "ecg")
    if ecg is None:
        raise SignalError("alignment measures need an ECG channel in the same record (name containing 'ECG')")
    with _LOCK:
        hit = _CACHE.get(ch)
    if hit is not None:
        return hit
    from tsmcp.plugins.ecg import _channel_beats  # the toolkit's validated R-peak detector, cached per channel

    rp, good = _channel_beats(ecg)
    r_t = rp[good] / ecg.fs
    feet, peaks, _ = detect_pulses_unguided(np.asarray(ch.data, dtype=np.float64), ch.fs)
    f_t = tangent_feet(ch.data, ch.fs, feet, peaks)
    f_t = f_t[np.isfinite(f_t)]
    pairs, used = [], set()
    for ft in f_t:
        k = int(np.searchsorted(r_t, ft, side="right")) - 1
        if k < 0 or k in used:
            continue
        lag = ft - r_t[k]
        if PAIR_MIN_S <= lag <= PAIR_MAX_S:
            pairs.append((float(r_t[k]), float(ft), float(lag)))
            used.add(k)
    hit = {"ecg": ecg.name, "r_t": r_t, "f_t": f_t, "pairs": pairs, "used": used}
    with _LOCK:
        _CACHE[ch] = hit
    return hit


def _base(ch, info, t0, t1, definition, method):
    return {
        "method": method,
        "method_version": METHOD_VERSION,
        "definition_id": definition,
        "channels_effective": [ch.name, info["ecg"]],
        "intervals_effective": [[t0, t1]],
        "timebase": {
            "identity": "same record",
            "sync_status": "index_aligned_declared",
            "channels": [info["ecg"], ch.name],
            "basis": "ECG and pulse are columns of one record; alignment declared by the file, not verified",
        },
        "conventions": {
            "foot": "intersecting tangent",
            "pairing": f"closest preceding R within {PAIR_MIN_S}-{PAIR_MAX_S} s",
            "delay_kind": "observed sample-index delay (not pulse arrival time unless synchrony is declared)",
        },
    }


def _window_pairs(ch, t0, t1):
    info = _pairs(ch)
    sel = [p for p in info["pairs"] if t0 <= p[0] and p[1] < t1]
    return info, sel


def pulse_delay(x, fs, unit, t0=0.0, ch=None, **_):
    if ch is None:
        raise SignalError("alignment measures need the loaded channel")
    t1 = t0 + x.size / fs
    info, sel = _window_pairs(ch, t0, t1)
    out = _base(
        ch, info, t0, t1, "delay.r_to_foot.observed.v1", "median ECG R-peak to pulse foot (intersecting tangent) delay"
    )
    if len(sel) < 5:
        return {
            **out,
            "value": None,
            "unit": "ms",
            "status": "limitation",
            "note": f"{len(sel)} paired beats (< 5)",
            "counts": {"accepted": len(sel), "rejected": 0},
        }
    lag = np.array([p[2] for p in sel]) * 1000
    iqr = np.percentile(lag, [25, 75])
    se = 1.2533 * lag.std(ddof=1) / np.sqrt(lag.size)  # standard error of the median (normal approximation)
    res = {"known": True, "value": round(max(se, 1000 / fs), 3), "kind": "standard_error_of_median_ms"}
    rows = [{"t_r_s": round(r, 4), "t_foot_s": round(f, 4), "delay_ms": round(lg * 1000, 2)} for r, f, lg in sel]
    return {
        **out,
        "value": present(float(np.median(lag)), res),
        "unit": "ms",
        "resolution": res,
        "iqr_ms": [round(float(iqr[0]), 1), round(float(iqr[1]), 1)],
        "counts": {"accepted": len(sel), "rejected": 0, "rejection_reasons": {}},
        "_artifact": {
            "kind": "delay_table",
            "columns": ["t_r_s", "t_foot_s", "delay_ms"],
            "rows": rows,
            "meta": {"channel": ch.name, "ecg": info["ecg"], "method_version": METHOD_VERSION},
        },
    }


def pulse_delay_change(x, fs, unit, t0=0.0, ch=None, **_):
    if ch is None:
        raise SignalError("alignment measures need the loaded channel")
    t1 = t0 + x.size / fs
    info, sel = _window_pairs(ch, t0, t1)
    out = _base(ch, info, t0, t1, "delay.r_to_foot.change.v1", "largest sustained step in the R-to-foot delay series")
    t = np.array([p[1] for p in sel])
    v = np.array([p[2] for p in sel]) * 1000
    cp = change_point(t, v)
    if cp is None:
        return {**out, "value": None, "unit": "ms", "status": "limitation", "note": f"{v.size} paired beats (< 16)"}
    warnings = []
    if cp["score"] < 5:
        warnings.append({"code": "weak_change", "severity": "warning", "message": f"step score {cp['score']} (< 5)"})
    if not cp["persistent"]:
        warnings.append({"code": "not_persistent", "severity": "warning", "message": "the delay does not stay changed"})
    sd = float(np.std(np.diff(v)) / np.sqrt(2)) if v.size > 2 else None
    res = {"known": bool(sd), "value": round(sd / np.sqrt(15), 3) if sd else None, "kind": "approx_se_of_step_ms"}
    return {
        **out,
        "value": present(cp["step"], res),
        "unit": "ms",
        "onset_s": cp["onset_s"],
        "delay_before_ms": present(cp["before"]),
        "delay_after_ms": present(cp["after"]),
        "score": cp["score"],
        "persistent": cp["persistent"],
        "transients": cp["transients"],
        "warnings": warnings,
        "resolution": res,
        "counts": {"accepted": int(v.size), "rejected": 0, "rejection_reasons": {}},
    }


def _missing(kind, x, fs, unit, t0, ch):
    if ch is None:
        raise SignalError("alignment measures need the loaded channel")
    t1 = t0 + x.size / fs
    info = _pairs(ch)
    r_t, f_t = info["r_t"], info["f_t"]
    r_w = r_t[(r_t >= t0) & (r_t < t1)]
    f_w = f_t[(f_t >= t0) & (f_t < t1)]
    if kind == "pulse":  # R-R intervals without a pulse foot in (R + 30 ms, next R + 30 ms]
        gaps = [
            float(a)
            for a, b in zip(r_w[:-1], r_w[1:], strict=True)
            if not np.any((f_t > a + PAIR_MIN_S) & (f_t <= b + PAIR_MIN_S))
        ]
        did, what = "beats.missing_in_pulse.v1", f"R-R intervals in {info['ecg']} with no {ch.name} pulse"
    else:  # pulses with no R within the preceding 800 ms
        gaps = [float(f) for f in f_w if not np.any((r_t < f - PAIR_MIN_S) & (r_t >= f - PAIR_MAX_S))]
        did, what = "beats.missing_in_ecg.v1", f"{ch.name} pulses with no preceding {info['ecg']} R-peak"
    out = _base(ch, info, t0, t1, did, what)
    return {
        **out,
        "value": len(gaps),
        "unit": "beats",
        "times_s": [round(g, 3) for g in gaps[:50]],
        "n_r_peaks": int(r_w.size),
        "n_pulses": int(f_w.size),
        "counts": {"accepted": int(min(r_w.size, f_w.size)), "rejected": 0, "rejection_reasons": {}},
        "resolution": {"known": True, "value": 1, "kind": "count"},
    }


def beats_missing_in_pulse(x, fs, unit, t0=0.0, ch=None, **_):
    return _missing("pulse", x, fs, unit, t0, ch)


def beats_missing_in_ecg(x, fs, unit, t0=0.0, ch=None, **_):
    return _missing("ecg", x, fs, unit, t0, ch)


def polarity_change(x, fs, unit, t0=0.0, ch=None, **_):
    if x.size < int(6 * POLARITY_WIN_S * fs):
        raise SignalError("window too short for a polarity profile (need >= 12 s)")
    w = int(POLARITY_WIN_S * fs)
    skews, times = [], []
    for s in range(0, x.size - w + 1, w):
        seg = x[s : s + w]
        seg = seg[np.isfinite(seg)]
        if seg.size < w // 2 or seg.std() == 0:
            skews.append(np.nan)
        else:
            z = (seg - seg.mean()) / seg.std()
            skews.append(float(np.mean(z**3)))
        times.append(t0 + s / fs)
    sk = np.array(skews)
    sign = np.sign(np.nan_to_num(sk))
    base_sign = np.sign(np.nanmedian(sk[: max(3, sk.size // 4)]))
    onset = None
    for i in range(sk.size - POLARITY_PERSIST + 1):
        if base_sign != 0 and np.all(sign[i : i + POLARITY_PERSIST] == -base_sign):
            onset = i
            break
    out = {
        "method": "persistent sign change of 2-s window skewness",
        "method_version": METHOD_VERSION,
        "definition_id": "polarity.change.v1",
        "intervals_effective": [[t0, t0 + x.size / fs]],
        "timebase": {"identity": "single channel", "sync_status": "unknown", "channels": [ch.name if ch else ""]},
        "resolution": {"known": True, "value": POLARITY_WIN_S, "kind": "window_s"},
    }
    if onset is None:
        return {
            **out,
            "value": None,
            "unit": "s",
            "status": "limitation",
            "note": "no persistent polarity change",
            "skew_median": round(float(np.nanmedian(sk)), 2),
        }
    before, after = float(np.nanmedian(sk[:onset])) if onset else float("nan"), float(np.nanmedian(sk[onset:]))
    return {
        **out,
        "value": round(times[onset], 2),
        "unit": "s",
        "skew_before": round(before, 2),
        "skew_after": round(after, 2),
    }


PLUGIN = DomainPlugin(
    "align",
    measures={
        "pulse_delay": pulse_delay,
        "pulse_delay_change": pulse_delay_change,
        "beats_missing_in_pulse": beats_missing_in_pulse,
        "beats_missing_in_ecg": beats_missing_in_ecg,
        "polarity_change": polarity_change,
    },
    summary="on a pulse channel (PPG/ABP) of a record with ECG: pulse_delay (median R-to-foot, ms; observed index "
    "delay, not PAT unless synchrony is declared), pulse_delay_change (sustained step in that delay: ms + onset), "
    "beats_missing_in_pulse / beats_missing_in_ecg (counts + times); polarity_change on any channel (onset of a "
    "persistent waveform polarity flip)",
)
