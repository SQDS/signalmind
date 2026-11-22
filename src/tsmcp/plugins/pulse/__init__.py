"""Pulse plugin (opt-in): beat-level features and quality for any pulsatile channel (PPG, ABP, CBFV).

Beats come from the period-aware unguided detector of `plugins.pressure.pulses` (systolic peaks by prominence with a
refractory distance of 0.6 x the block's autocorrelation period; foot = minimum before the peak). Per beat: foot/peak
times, duration, amplitude (peak - foot), mean level, accepted flag with reason, and a template-correlation quality
index (each beat resampled to 64 points and correlated with the median template; port of the hemoworld SQI).

Measures: pulse_beats (table), pulse_rate (60 / mean beat duration over accepted complete beats), pulse_quality
(fraction of accepted beats with template correlation >= 0.9, plus low-quality intervals), pulse_amplitude_change and
pulse_level_change (largest sustained step in per-beat amplitude / mean level, with a persistence check and the
transient excursions that were not counted). Values keep the channel's unit; an unknown unit stays '?'.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from tsmcp.core.errors import SignalError
from tsmcp.plugins.base import DomainPlugin
from tsmcp.plugins.pressure.pulses import beat_table, detect_pulses_unguided
from tsmcp.runtime.precision import present

METHOD_VERSION = "pulse.v1"
SQI_POINTS = 64
SQI_GOOD = 0.9
MIN_SIDE = 8  # beats on each side of a change-point
TRANSIENT_MAX_BEATS = 10
_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def template_sqi(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    tokens = []
    for a, b in zip(starts, ends, strict=True):
        seg = np.asarray(x[a:b], dtype=np.float64)
        if seg.size < 4 or not np.isfinite(seg).all():
            tokens.append(np.full(SQI_POINTS, np.nan))
            continue
        tokens.append(np.interp(np.linspace(0, seg.size - 1, SQI_POINTS), np.arange(seg.size), seg))
    if not tokens:
        return np.empty(0)
    t = np.asarray(tokens)
    centered = t - np.nanmean(t, axis=1, keepdims=True)
    norm = centered / np.maximum(np.sqrt(np.nansum(centered**2, axis=1, keepdims=True)), 1e-8)
    template = np.nanmedian(norm, axis=0)
    template -= np.nanmean(template)
    template /= max(float(np.sqrt(np.nansum(template**2))), 1e-8)
    return np.nansum(norm * template[None, :], axis=1)


def _detect(ch):
    with _LOCK:
        hit = _CACHE.get(ch)
    if hit is None:
        hit = detect_pulses_unguided(np.asarray(ch.data, dtype=np.float64), ch.fs)
        with _LOCK:
            _CACHE[ch] = hit
    return hit


def _table(x, fs, unit, t0, ch):
    if ch is None or ch.fs != fs:
        raise SignalError("pulse measures need the loaded channel")
    if x.size < int(3 * fs):
        raise SignalError("window shorter than 3 s; pulse measures need at least 3 s")
    feet, peaks, ref = _detect(ch)
    data = np.asarray(ch.data, dtype=np.float64)
    fin = data[np.isfinite(data)]
    rng = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 0.0)
    i0 = int(round(t0 * fs))
    rows = beat_table(data, fs, 0.0, unit, feet, peaks, i0, i0 + x.size, rng, ref)
    t1 = t0 + x.size / fs
    # amplitude, level and template SQI for every row with a complete span
    idx = [(int(round(r["t_foot_s"] * fs)), int(round(r["t_end_s"] * fs))) for r in rows if r["t_end_s"] is not None]
    sqi = (
        template_sqi(data, np.array([a for a, _ in idx], dtype=int), np.array([b for _, b in idx], dtype=int))
        if idx
        else []
    )
    k = 0
    for r in rows:
        if r["t_end_s"] is None:
            r.update(amplitude=None, level=None, sqi=None)
            continue
        a, b = idx[k]
        seg = data[a:b]
        r["amplitude"] = round(float(np.nanmax(seg) - data[a]), 4) if seg.size and np.isfinite(seg).any() else None
        r["level"] = round(float(np.nanmean(seg)), 4) if seg.size and np.isfinite(seg).any() else None
        r["sqi"] = round(float(sqi[k]), 3) if np.isfinite(sqi[k]) else None
        k += 1
    return rows, t1


def _common(rows, t0, t1, ch, unit, definition, method):
    acc = [r for r in rows if r["accepted"]]
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            reasons[r["reject_reason"]] = reasons.get(r["reject_reason"], 0) + 1
    excluded, cur = [], None
    for r in rows:
        if not r["accepted"] and r["t_end_s"] is not None:
            span = [r["t_foot_s"], r["t_end_s"]]
            if cur and span[0] <= cur[1]:
                cur[1] = max(cur[1], span[1])
            else:
                cur = list(span)
                excluded.append(cur)
    dur = sum(r["duration_s"] for r in acc)
    warnings = []
    if rows and len(acc) / len(rows) < 0.8:
        warnings.append(
            {
                "code": "many_rejected_beats",
                "severity": "warning",
                "message": f"{len(rows) - len(acc)} of {len(rows)} beats rejected: {reasons}",
            }
        )
    if t1 > t0 and dur / (t1 - t0) < 0.6:
        warnings.append(
            {"code": "low_coverage", "severity": "warning", "message": f"accepted beats cover {dur / (t1 - t0):.0%}"}
        )
    return {
        "method": method,
        "method_version": METHOD_VERSION,
        "definition_id": definition,
        "counts": {"accepted": len(acc), "rejected": len(rows) - len(acc), "rejection_reasons": reasons},
        "coverage": round(dur / (t1 - t0), 4) if t1 > t0 else None,
        "intervals_effective": [[t0, t1]],
        "intervals_excluded": excluded,
        "warnings": warnings,
        "timebase": {
            "identity": "single channel",
            "sync_status": "unknown",
            "basis": "single channel",
            "channels": [ch.name],
        },
        "_artifact": {
            "kind": "pulse_beat_table",
            "rows": rows,
            "columns": [
                "beat",
                "t_foot_s",
                "t_peak_s",
                "t_end_s",
                "duration_s",
                "amplitude",
                "level",
                "sqi",
                "accepted",
                "reject_reason",
            ],
            "meta": {"channel": ch.name, "unit": unit, "window_s": [t0, t1], "method_version": METHOD_VERSION},
        },
    }


def pulse_beats(x, fs, unit, t0=0.0, ch=None, **_):
    rows, t1 = _table(x, fs, unit, t0, ch)
    out = _common(
        rows, t0, t1, ch, unit, "pulse.beats.v1", "pulse beat table (unguided, period-aware); amplitude = peak - foot"
    )
    listed = [{k: v for k, v in r.items() if k != "beat"} for r in rows[:120]]
    return {
        **out,
        "value": out["counts"]["accepted"],
        "unit": "beats",
        "n_beats": len(rows),
        "beats": listed,
        "truncated": len(rows) > 120,
        "resolution": {"known": True, "value": 1 / fs, "kind": "beat_time_resolution_s"},
    }


def pulse_rate(x, fs, unit, t0=0.0, ch=None, **_):
    rows, t1 = _table(x, fs, unit, t0, ch)
    out = _common(rows, t0, t1, ch, unit, "pulse.rate.beats.v1", "60 / mean duration of accepted complete beats")
    d = np.array([r["duration_s"] for r in rows if r["accepted"]], dtype=float)
    if d.size < 5:
        return {**out, "value": None, "unit": "bpm", "status": "limitation", "note": f"{d.size} accepted beats (< 5)"}
    rate = 60.0 / d.mean()
    inst = 60.0 / d
    sem = float(inst.std(ddof=1) / np.sqrt(inst.size))
    res = {"known": True, "value": round(sem, 4), "kind": "standard_error_of_beat_rate"}
    return {**out, "value": present(rate, res), "unit": "bpm", "resolution": res, "n_beats": int(d.size)}


def pulse_quality(x, fs, unit, t0=0.0, ch=None, **_):
    rows, t1 = _table(x, fs, unit, t0, ch)
    out = _common(
        rows,
        t0,
        t1,
        ch,
        unit,
        "pulse.quality.v1",
        f"fraction of beats accepted with template correlation >= {SQI_GOOD}",
    )
    if not rows:
        return {**out, "value": None, "unit": "", "status": "limitation", "note": "no pulses detected in this window"}
    good = [r for r in rows if r["accepted"] and (r.get("sqi") or 0) >= SQI_GOOD]
    low = [
        [r["t_foot_s"], r["t_end_s"] or r["t_foot_s"]]
        for r in rows
        if not (r["accepted"] and (r.get("sqi") or 0) >= SQI_GOOD)
    ]
    merged: list[list[float]] = []
    for a, b in low:
        if merged and a <= merged[-1][1] + 0.01:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    sq = [r["sqi"] for r in rows if r.get("sqi") is not None]
    return {
        **out,
        "value": round(len(good) / len(rows), 3),
        "unit": "",
        "low_quality_intervals": merged[:60],
        "sqi_median": round(float(np.median(sq)), 3) if sq else None,
        "n_beats": len(rows),
        "resolution": {"known": True, "value": round(1 / len(rows), 4), "kind": "one_beat_fraction"},
    }


def change_point(t: np.ndarray, v: np.ndarray) -> dict | None:
    """Largest sustained step (two-segment mean model, t-like score); persistence = the 15 beats after the step stay
    on the new side by at least half the step. Transients = short excursions that return within 10 beats."""
    n = v.size
    if n < 2 * MIN_SIDE:
        return None
    best = None
    for k in range(MIN_SIDE, n - MIN_SIDE + 1):
        a, b = v[:k], v[k:]
        sd = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (n - 2))
        score = abs(b.mean() - a.mean()) / max(sd, 1e-9) * np.sqrt(k * (n - k) / n)
        if best is None or score > best[1]:
            best = (k, score)
    k, score = best
    before, after = float(np.median(v[max(0, k - 15) : k])), float(np.median(v[k : k + 15]))
    step = float(v[k:].mean() - v[:k].mean())
    persistent = (after - before) * step > 0 and abs(after - before) >= 0.5 * abs(step)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826 or 1e-9
    out_mask = np.abs(v - med) > 3 * mad
    transients, i = [], 0
    while i < n:
        if out_mask[i]:
            j = i
            while j < n and out_mask[j]:
                j += 1
            if j - i <= TRANSIENT_MAX_BEATS and j < n:
                transients.append(
                    {"t_start_s": round(float(t[i]), 3), "t_end_s": round(float(t[j - 1]), 3), "beats": j - i}
                )
            i = j
        else:
            i += 1
    return {
        "onset_s": round(float(t[k]), 3),
        "score": round(float(score), 2),
        "step": step,
        "before": before,
        "after": after,
        "persistent": bool(persistent),
        "transients": transients[:20],
    }


def _change(field: str, x, fs, unit, t0, ch) -> dict:
    rows, t1 = _table(x, fs, unit, t0, ch)
    did = f"pulse.{field}_change.v1"
    out = _common(rows, t0, t1, ch, unit, did, f"largest sustained step in per-beat {field} (two-segment mean model)")
    acc = [r for r in rows if r["accepted"] and r.get(field) is not None]
    t = np.array([r["t_foot_s"] for r in acc], dtype=float)
    v = np.array([r[field] for r in acc], dtype=float)
    cp = change_point(t, v)
    if cp is None:
        return {
            **out,
            "value": None,
            "unit": "s",
            "status": "limitation",
            "note": f"{v.size} accepted beats; need {2 * MIN_SIDE}",
        }
    warnings = list(out["warnings"])
    if cp["score"] < 5:
        warnings.append(
            {
                "code": "weak_change",
                "severity": "warning",
                "message": f"step score {cp['score']} (< 5): the change may not be real",
            }
        )
    if not cp["persistent"]:
        warnings.append(
            {
                "code": "not_persistent",
                "severity": "warning",
                "message": "the level after the candidate does not stay on the new side: possibly transient",
            }
        )
    beat_s = float(np.median(np.diff(t))) if t.size > 1 else None
    return {
        **out,
        "warnings": warnings,
        "value": cp["onset_s"],
        "unit": "s",
        "step": present(cp["step"]),
        "step_unit": unit if unit not in ("", None) else "?",
        "before": present(cp["before"]),
        "after": present(cp["after"]),
        "relative_change": round(cp["step"] / cp["before"], 3) if cp["before"] else None,
        "score": cp["score"],
        "persistent": cp["persistent"],
        "transients": cp["transients"],
        "resolution": {"known": bool(beat_s), "value": round(beat_s, 3) if beat_s else None, "kind": "one_beat_s"},
    }


def pulse_amplitude_change(x, fs, unit, t0=0.0, ch=None, **_):
    return _change("amplitude", x, fs, unit, t0, ch)


def pulse_level_change(x, fs, unit, t0=0.0, ch=None, **_):
    return _change("level", x, fs, unit, t0, ch)


PLUGIN = DomainPlugin(
    "pulse",
    measures={
        "pulse_beats": pulse_beats,
        "pulse_rate": pulse_rate,
        "pulse_quality": pulse_quality,
        "pulse_amplitude_change": pulse_amplitude_change,
        "pulse_level_change": pulse_level_change,
    },
    summary="any pulsatile channel (PPG, ABP, CBFV): pulse_beats (table: foot/peak, amplitude, level, template SQI, "
    "accepted + reason), pulse_rate (bpm from beat durations), pulse_quality (fraction of good beats + low-quality "
    "intervals), pulse_amplitude_change / pulse_level_change (largest sustained step: onset, before/after, "
    "persistence, transients not counted)",
)
