"""Pressure plugin (opt-in): per-beat ABP features with an auditable beat table.

Measures (pulse-only detection): abp_beats, abp_sbp, abp_dbp, abp_pp; ECG-guided variants: abp_beats_ecg, abp_sbp_ecg,
abp_dbp_ecg, abp_pp_ecg. Each result carries definition_id, counts, coverage, excluded intervals, warnings, resolution
(standard error of the beat mean) and the full beat table as an artifact; the text reply lists the first beats only.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from tsmcp.core.compact import sig as sig3
from tsmcp.core.errors import SignalError
from tsmcp.plugins.base import DomainPlugin
from tsmcp.plugins.pressure.pulses import (
    METHOD_VERSION,
    beat_table,
    detect_pulses_guided,
    detect_pulses_unguided,
    summarize,
)
from tsmcp.runtime.precision import present

MAX_BEATS_LISTED = 150
DEFINITIONS = {"sbp": "abp.sbp.beat_mean.v1", "dbp": "abp.dbp.beat_mean.v1", "pp": "abp.pp.beat_mean.v1"}

_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # Channel -> {guidance: (feet, peaks, meta)}
_SIGNALS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # Channel -> Signal it belongs to
_LOCK = threading.Lock()


def register_signal(sig_obj) -> None:
    """Called by the server / executor before measuring, so ECG guidance can find the record's ECG channel."""
    with _LOCK:
        for c in sig_obj.channels:
            _SIGNALS[c] = sig_obj


def _ecg_channel(sig_obj, ch):
    if sig_obj is None:
        with _LOCK:
            sig_obj = _SIGNALS.get(ch)
    if sig_obj is None:
        return None
    for c in sig_obj.channels:
        if c is not ch and ("ecg" in c.name.lower() or "ekg" in c.name.lower()):
            return c
    return None


def _pulses(ch, guidance: str, sig_obj) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with _LOCK:
        hit = _CACHE.get(ch, {}).get(guidance)
    if hit is not None:
        return hit
    if guidance == "ecg":
        ecg = _ecg_channel(sig_obj, ch)
        if ecg is None:
            raise SignalError(
                "ECG-guided pressure measures need an ECG channel in the same record (name containing 'ECG'); "
                "use the pulse-only measures (abp_sbp, abp_dbp, abp_pp, abp_beats) instead"
            )
        from tsmcp.plugins.ecg import _channel_beats  # the toolkit's own R-peak detector, cached per channel

        rp, good = _channel_beats(ecg)
        rp = rp[good]
        if ecg.fs != ch.fs:
            rp = np.round(rp * (ch.fs / ecg.fs)).astype(np.int64)
        feet, peaks, ref = detect_pulses_guided(ch.data, rp, ch.fs)
        meta = {
            "guidance": "ecg",
            "ecg_channel": ecg.name,
            "n_r_peaks": int(rp.size),
            "timebase": {
                "identity": "same record",
                "sync_status": "index_aligned_declared",
                "basis": "ECG and pressure are columns of one record; alignment declared by the file, not verified",
                "channels": [ecg.name, ch.name],
            },
        }
    else:
        feet, peaks, ref = detect_pulses_unguided(np.asarray(ch.data, dtype=np.float64), ch.fs)
        meta = {
            "guidance": "pulse",
            "timebase": {
                "identity": "single channel",
                "sync_status": "unknown",
                "basis": "single channel",
                "channels": [ch.name],
            },
        }
    hit = (feet, peaks, ref, meta)
    with _LOCK:
        _CACHE.setdefault(ch, {})[guidance] = hit
    return hit


def _table(x, fs, unit, t0, ch, sig_obj, guidance):
    if x.size < int(2 * fs):
        raise SignalError("window shorter than 2 s; per-beat pressure measures need at least 2 s")
    if ch is None or ch.fs != fs:
        raise SignalError("pressure measures need the loaded channel (ts_measure provides it)")
    feet, peaks, ref, meta = _pulses(ch, guidance, sig_obj)
    i0 = int(round(t0 * fs))
    i1 = i0 + x.size
    data = np.asarray(ch.data, dtype=np.float64)
    finite = data[np.isfinite(data)]
    rng = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    rows = beat_table(data, fs, 0.0, unit, feet, peaks, i0, i1, rng, ref)
    t1 = t0 + x.size / fs
    return rows, summarize(rows, t0, t1, unit), meta, t1


def _artifact(rows, meta, unit, t0, t1, ch):
    return {
        "kind": "beat_table",
        "columns": [
            "beat",
            "t_foot_s",
            "t_peak_s",
            "t_end_s",
            "duration_s",
            "sbp",
            "dbp",
            "pp",
            "accepted",
            "reject_reason",
        ],
        "rows": rows,
        "meta": {"channel": ch.name, "unit": unit, "window_s": [t0, t1], "method_version": METHOD_VERSION, **meta},
    }


def _common(summary, meta, unit, t0, t1, ch, rows):
    pressure = unit in ("mmHg", "kPa", "cmH2O")
    return {
        "unit": unit if pressure else "?",
        "method_version": METHOD_VERSION,
        "guidance": meta["guidance"],
        "counts": summary["counts"],
        "coverage": summary["coverage"],
        "intervals_effective": [[t0, t1]],
        "intervals_excluded": summary["intervals_excluded"],
        "warnings": summary["warnings"],
        "timebase": meta["timebase"],
        "channels_effective": [ch.name] + ([meta["ecg_channel"]] if meta.get("ecg_channel") else []),
        "_artifact": _artifact(rows, meta, unit, t0, t1, ch),
    }


def _beats(x, fs, unit, t0=0.0, ch=None, sig=None, guidance="pulse") -> dict:
    rows, summary, meta, t1 = _table(x, fs, unit, t0, ch, sig, guidance)
    base = _common(summary, meta, unit, t0, t1, ch, rows)
    listed = [{k: v for k, v in r.items() if k != "beat"} for r in rows[:MAX_BEATS_LISTED]]
    return {
        "value": summary["counts"]["accepted"],
        **base,
        "unit": "beats",
        "method": f"beat table ({meta['guidance']}-detected feet; SBP = beat max, DBP = min at foot)",
        "definition_id": "abp.beats.v1",
        "value_units": {"sbp": base["unit"], "dbp": base["unit"], "pp": base["unit"]},
        "n_beats": summary["n_beats"],
        "aggregation": None,
        "resolution": {"known": True, "value": 1 / fs, "kind": "beat_time_resolution_s"},
        "beats": listed,
        "truncated": len(rows) > MAX_BEATS_LISTED,
        **({"status": "limitation", "note": "no pulses detected in this window"} if not rows else {}),
    }


def _stat(col: str, x, fs, unit, t0, ch, sig, guidance) -> dict:
    rows, summary, meta, t1 = _table(x, fs, unit, t0, ch, sig, guidance)
    base = _common(summary, meta, unit, t0, t1, ch, rows)
    mean = summary.get(f"{col}_mean")
    sem = summary.get(f"{col}_sem")
    out = {
        **base,
        "method": f"beat-weighted mean of per-beat {col.upper()} over accepted complete beats ({meta['guidance']})",
        "definition_id": DEFINITIONS[col],
        "aggregation": "beat_weighted",
        "n_beats": summary["counts"]["accepted"],
    }
    if mean is None:
        return {
            **out,
            "value": None,
            "status": "limitation",
            "note": "no accepted complete beat in this window: " + str(summary["counts"]["rejection_reasons"]),
        }
    resolution = {
        "known": sem is not None,
        "value": sig3(sem) if sem is not None else None,
        "kind": "standard_error_of_beat_mean",
    }
    return {
        **out,
        "value": present(mean, resolution),
        "resolution": resolution,
        "sd": sig3(float(np.std([r[col] for r in rows if r["accepted"]], ddof=1)))
        if summary["counts"]["accepted"] > 1
        else None,
    }


def abp_beats(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _beats(x, fs, unit, t0, ch, sig, "pulse")


def abp_sbp(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _stat("sbp", x, fs, unit, t0, ch, sig, "pulse")


def abp_dbp(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _stat("dbp", x, fs, unit, t0, ch, sig, "pulse")


def abp_pp(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _stat("pp", x, fs, unit, t0, ch, sig, "pulse")


def abp_beats_ecg(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _beats(x, fs, unit, t0, ch, sig, "ecg")


def abp_sbp_ecg(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _stat("sbp", x, fs, unit, t0, ch, sig, "ecg")


def abp_dbp_ecg(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _stat("dbp", x, fs, unit, t0, ch, sig, "ecg")


def abp_pp_ecg(x, fs, unit, t0=0.0, ch=None, sig=None, **_):
    return _stat("pp", x, fs, unit, t0, ch, sig, "ecg")


PLUGIN = DomainPlugin(
    "pressure",
    measures={
        "abp_beats": abp_beats,
        "abp_sbp": abp_sbp,
        "abp_dbp": abp_dbp,
        "abp_pp": abp_pp,
        "abp_beats_ecg": abp_beats_ecg,
        "abp_sbp_ecg": abp_sbp_ecg,
        "abp_dbp_ecg": abp_dbp_ecg,
        "abp_pp_ecg": abp_pp_ecg,
    },
    summary="per-beat arterial pressure on a pressure channel: abp_beats (beat table: foot/peak times, SBP, DBP, PP, "
    "accepted flag + reason), abp_sbp / abp_dbp / abp_pp (beat-weighted means over accepted complete beats, with "
    "counts, coverage, excluded intervals, warnings); *_ecg variants use the record's ECG R-peaks to guide the beat "
    "search (explicit method, needs an ECG channel). Values keep the channel's unit; an unknown unit stays '?'",
)
