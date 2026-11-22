"""Gap-aware RR AF statistics, readable-time accounting and episode confidence.

COSEn uses m=1 sample entropy with an adaptive tolerance giving at least five
m=2 matches: SampEn + log(2*r) - log(mean RR), with RR and r in seconds
(Lake & Moorman, 2011). Embeddings never cross masked RR intervals.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from tsmcp.core.compact import sig, tround
from tsmcp.core.errors import SignalError
from tsmcp.core.signal import time_decimals
from tsmcp.plugins.ecg.morphology import beat_classes, channel_classes, lead_support, stretches

# Round 2 freeze: DS1 + LTAFDB 07 and eight allowlisted dev records only.
# Fusion on: LTAFDB MAE 1.049684 pp; DS1+07 MAE 1.594709 pp; median record
# Se/PPV (>=5% AF) 0.853878/0.991007; AF-free mean/max 0.167485/1.889474%.
# Fusion off: MAE 2.356531/1.924655 pp, Se/PPV 0.776797/0.996341.
# Public readable-time accounting is unchanged from Round 1. No detection
# thresholds are fitted during episode confidence calibration.
AF_WINDOW_RR = 32
AF_STRIDE_RR = 8
AF_MIN_S = 10.0
AF_MERGE_GAP_S = 30.0
RR_MIN_S = 0.25
RR_MAX_S = 3.0
RR_MAX_JUMP = 3.0
AF_COSE_N_MIN = -1.4
AF_TPR_MIN = 0.50
AF_NRMSSD_MIN = 0.12
# Sparse large RR jumps inflate RMSSD: accepted false windows in DS1/108 had
# median ndiff 0.067 versus 0.160 in LTAFDB/00. This preserves all original RR
# adjacencies and requires sustained irregularity; it never removes premature RR.
AF_NDIFF_MIN = 0.08
AF_MIN_VALID_FRAC = 0.50
PERIODIC_LAG_REL = 0.50
COSE_MIN_TOL_S = 0.03
COSE_MIN_MATCHES = 5

_CACHE = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([float(a), float(b)])
    return out


def clip_intervals(intervals, a, b):
    return union((max(a, x), min(b, y)) for x, y in intervals if y > a and x < b)


def rr_quality(rr, good, pause_confirmed=None):
    """Plausible consecutive intervals; never invent adjacency across missing beats.

    An explicitly morphology-confirmed pause may bypass the jump gate, but not
    the 0.25–3 s range. No pause is inferred solely from a large RR.
    """
    readable = good[:-1] & good[1:]
    plausible = np.isfinite(rr) & (rr >= RR_MIN_S) & (rr <= RR_MAX_S)
    valid = readable & plausible
    jump = np.zeros(len(rr), bool)
    if len(rr) > 1:
        adjacent = valid[:-1] & valid[1:]
        ratio = np.maximum(rr[1:], rr[:-1]) / np.maximum(np.minimum(rr[1:], rr[:-1]), 1e-12)
        positions = np.flatnonzero(adjacent & (ratio > RR_MAX_JUMP))
        jump[positions] = True
        jump[positions + 1] = True
    if pause_confirmed is not None:
        jump &= ~pause_confirmed
    reasons = np.where(
        ~readable, "unreadable_beats", np.where(~plausible, "implausible_rr", np.where(jump, "rr_jump", "valid"))
    )
    return valid & ~jump, reasons


def af_statistics(rr, valid=None):
    """COSEn, turning-point ratio, normalized RMSSD on a masked, ordered RR window."""
    rr = np.asarray(rr, float)
    valid = np.ones(len(rr), bool) if valid is None else np.asarray(valid, bool)
    valid &= np.isfinite(rr) & (rr > 0)
    r = rr[valid]
    pair_ok = valid[:-1] & valid[1:]
    triple_ok = valid[:-2] & valid[1:-1] & valid[2:]
    if len(r) < 4 or pair_ok.sum() < 3 or triple_ok.sum() < 2:
        return {"cosen": -10.0, "tpr": 0.0, "nrmssd": 0.0, "ndiff": 0.0, "n_valid": int(len(r))}
    differences = np.diff(rr)
    rmssd = np.sqrt(np.mean(differences[pair_ok] ** 2)) / r.mean()
    turns = differences[:-1] * differences[1:] < 0
    tpr = np.mean(turns[triple_ok])
    # Embeddings for m=2 are formed ONLY where the two original RRs are valid.
    pairs = np.lib.stride_tricks.sliding_window_view(rr, 2)[pair_ok]
    d1 = np.abs(r[:, None] - r[None, :])[np.triu_indices(len(r), 1)]
    d2 = np.max(np.abs(pairs[:, None, :] - pairs[None, :, :]), axis=2)[np.triu_indices(len(pairs), 1)]
    kth = min(COSE_MIN_MATCHES - 1, len(d2) - 1)
    tolerance = max(COSE_MIN_TOL_S, float(np.partition(d2, kth)[kth]) + 1e-9)
    b = np.mean(d1 <= tolerance)
    a = np.mean(d2 <= tolerance)
    cosen = -np.log(max(a, 1e-12) / max(b, 1e-12)) + np.log(2 * tolerance) - np.log(r.mean())
    ndiff = np.median(np.abs(differences[pair_ok])) / np.median(r)
    return {
        "cosen": float(cosen),
        "tpr": float(tpr),
        "nrmssd": float(rmssd),
        "ndiff": float(ndiff),
        "n_valid": int(len(r)),
    }


def _analyse(beats, duration, offset=0.0):
    from tsmcp.plugins.ecg import classify_rr

    times = beats["t_s"]
    rr = np.diff(times)
    good = beats["good"]
    base, reasons = rr_quality(rr, good)
    # Supraventricular prematurity stays: in AF short RR is evidence, not a reason
    # to discard a beat. Mask BOTH intervals touching ventricular/unknown beats.
    excluded = np.isin(beats["label"], ["ventricular", "unknown"])
    valid = base & ~excluded[:-1] & ~excluded[1:]
    score_sum, score_n = np.zeros(len(rr)), np.zeros(len(rr))
    score_max = np.zeros(len(rr))
    stat_sum = np.zeros(len(rr))
    states = np.full(len(rr), "unknown", dtype="U8")
    windows = []
    starts = list(range(0, max(0, len(rr) - AF_WINDOW_RR + 1), AF_STRIDE_RR))
    if len(rr) >= AF_WINDOW_RR and (not starts or starts[-1] != len(rr) - AF_WINDOW_RR):
        starts.append(len(rr) - AF_WINDOW_RR)
    if 4 <= len(rr) < AF_WINDOW_RR:
        starts = [0]
    for i in starts:
        end = min(i + AF_WINDOW_RR, len(rr))
        r, v, raw_valid = rr[i:end], valid[i:end], base[i:end]
        # Readability is a signal-quality decision. A readable ventricular rhythm
        # remains readable even if too few non-ventricular RR remain to test AF.
        state = "clean" if raw_valid.all() else "degraded" if raw_valid.mean() >= AF_MIN_VALID_FRAC else "unknown"
        eligible = state != "unknown" and v.mean() >= AF_MIN_VALID_FRAC
        stats = af_statistics(r, v)
        reason = "accepted"
        score = 0.0
        if not eligible:
            reason = "unknown_quality" if state == "unknown" else "insufficient_nonventricular_rr"
        elif len(r) < AF_WINDOW_RR:
            # The legacy fallback is retained on contiguous valid segments only;
            # prematurity-only pair replacement is disabled by Addendum B4.
            votes = [classify_rr(r[a:b], remove_pairs=False) == "irregular" for a, b in stretches(v) if b - a >= 4]
            score = 1.01 if votes and np.mean(votes) >= 0.5 else 0.0
            reason = "short_rr_fallback"
        else:
            score = min(
                np.exp(stats["cosen"] - AF_COSE_N_MIN),
                stats["tpr"] / AF_TPR_MIN,
                stats["nrmssd"] / AF_NRMSSD_MIN,
                stats["ndiff"] / AF_NDIFF_MIN,
            )
            # m=1 entropy alone can miss period-three structure. Check recurrence
            # only on contiguous valid RR, without removing premature beats.
            lag = []
            for k in (1, 2, 3):
                contiguous = np.lib.stride_tricks.sliding_window_view(v, k + 1).all(axis=1)
                lag.append(float(np.median(np.abs(r[k:] - r[:-k])[contiguous])) if contiguous.any() else np.nan)
            periodic = lag[0] > 0 and min(lag[1:]) < PERIODIC_LAG_REL * lag[0]
            if periodic:
                score = 0.0
            if score < 1:
                reason = min(
                    (
                        ("low_cosen", np.exp(stats["cosen"] - AF_COSE_N_MIN)),
                        ("low_tpr", stats["tpr"] / AF_TPR_MIN),
                        ("low_nrmssd", stats["nrmssd"] / AF_NRMSSD_MIN),
                        ("low_ndiff", stats["ndiff"] / AF_NDIFF_MIN),
                    ),
                    key=lambda z: z[1],
                )[0]
                if periodic:
                    reason = "periodic_rr"
        if eligible:
            score_sum[i:end] += score
            score_max[i:end] = np.maximum(score_max[i:end], score)
            stat_sum[i:end] += stats["nrmssd"]
            score_n[i:end] += 1
        states[i:end] = np.where(raw_valid, state, "unknown")
        windows.append(
            {
                "start": float(times[i]),
                "end": float(times[end]),
                "state": state,
                "reason": reason,
                "score": float(score),
                **stats,
            }
        )
    # Refine the supporting 32-RR windows using a local eight-RR RMSSD at each
    # beat. This stops a window from painting its quiet leading/trailing beats AF.
    pair_valid = valid[:-1] & valid[1:]
    energy = np.r_[0.0, np.where(pair_valid, np.diff(rr) ** 2, 0.0)]
    count = np.r_[0.0, pair_valid.astype(float)]
    kernel = np.ones(min(8, max(1, len(rr))))
    if len(rr):
        local_n = np.convolve(count, kernel, "same")
        local_energy = np.convolve(energy, kernel, "same")
        local_rr = np.convolve(np.where(valid, rr, 0), kernel, "same")
        local_rr /= np.maximum(np.convolve(valid.astype(float), kernel, "same"), 1)
        local_stat = np.sqrt(local_energy / np.maximum(local_n, 1)) / np.maximum(local_rr, 1e-12)
        per_beat = np.minimum(score_max, local_stat / AF_NRMSSD_MIN)
    else:
        per_beat = score_max
    statistic = np.divide(stat_sum, score_n, out=np.zeros_like(stat_sum), where=score_n > 0)
    readable = base
    # Boundary refinement: the accumulated per-beat statistic crosses 1 at a
    # specific RR boundary, rather than emitting overlapping 32-RR window spans.
    core_score = np.divide(score_sum, score_n, out=np.zeros_like(score_sum), where=score_n > 0)
    # Excluding ectopic RR from a statistic does not make the underlying atrial
    # rhythm cease during those readable beats. Require independent local valid
    # pairs and positive window evidence; never bridge unknown-quality time.
    per_beat = np.maximum(per_beat, core_score)
    known = beats["label"] != "unknown"
    support = (per_beat >= 1.0) & readable & known[:-1] & known[1:] & (local_n >= 3) if len(rr) else np.zeros(0, bool)
    # Refine boundaries of supported episodes; a single marginal window must
    # not manufacture an episode with no positive overlapping-window consensus.
    seed = (core_score >= 1.0) & valid
    for a, b in stretches(support):
        if not seed[a:b].any():
            support[a:b] = False
    supported = [[float(times[a]), float(times[b])] for a, b in stretches(support)]
    # Establish AF support once, before the caller's presentation merge gap.
    # Regrouping episodes must never change the numerator of AF burden.
    groups = []
    for interval in supported:
        if groups and interval[0] - groups[-1][-1][1] < AF_MERGE_GAP_S:
            groups[-1].append(interval)
        else:
            groups.append([interval])
    supported = [interval for group in groups if sum(b - a for a, b in group) >= AF_MIN_S for interval in group]
    readable_intervals = [[float(times[a]), float(times[b])] for a, b in stretches(readable)]
    noisy = []
    previous = offset
    for a, b in readable_intervals:
        if a > previous:
            noisy.append([previous, a])
        previous = b
    if previous < offset + duration:
        noisy.append([previous, offset + duration])
    return {
        "beats": beats,
        "rr": rr,
        "valid": valid,
        "quality_valid": base,
        "quality_reasons": reasons,
        "state": states,
        "score": per_beat,
        "statistic": statistic,
        "supported": supported,
        "readable": readable_intervals,
        "noisy": noisy,
        "windows": windows,
    }


def af_analysis(x, fs, t0=0.0, ch=None):
    if ch is None or ch.fs != fs:
        return _analyse(beat_classes(x, fs, t0), len(x) / fs, t0)
    source = channel_classes(ch)
    with _LOCK:
        hit = _CACHE.get(ch)
    if hit is None or hit[0] is not source:
        beats = {**source, "t_s": source["p"] / fs}
        result = _analyse(beats, ch.duration)
        with _LOCK:
            _CACHE[ch] = (source, result)
        return result
    return hit[1]


def probability(score):
    from tsmcp.plugins.ecg.calibration import CALIBRATION_X, CALIBRATION_Y

    return float(np.interp(score, CALIBRATION_X, CALIBRATION_Y))


def _episode_rows(analysis, t0, t1, merge_gap_s):
    if not np.isfinite(merge_gap_s) or merge_gap_s < 0:
        raise SignalError("merge_gap_s must be finite and >= 0")
    groups = []
    for a, b in clip_intervals(analysis["supported"], t0, t1):
        if groups and a - groups[-1][-1][1] < merge_gap_s:
            groups[-1].append([a, b])
        else:
            groups.append([[a, b]])
    out = []
    beats = analysis["beats"]
    for intervals in groups:
        af_s = sum(b - a for a, b in intervals)
        start, end = intervals[0][0], intervals[-1][1]
        mask = (beats["t_s"][:-1] >= start) & (beats["t_s"][1:] <= end) & analysis["valid"]
        stat = float(np.mean(analysis["statistic"][mask])) if mask.any() else 0.0
        score = (end - start) * stat
        out.append(
            {
                "type": "ecg_irregular",
                "t": start,
                "start": start,
                "dur": end - start,
                "span_s": end - start,
                "af_s": af_s,
                "intervals_s": intervals,
                "mean_hr": float(60 / analysis["rr"][mask].mean()),
                "mean_irregularity": stat,
                "score": score,
                "p_true": probability(score),
                "leads": lead_support(beats, np.r_[mask, False]),
            }
        )
    return sorted(out, key=lambda e: -e["span_s"])


def ecg_af_episodes(x, fs, unit, t0=0.0, ch=None, merge_gap_s=AF_MERGE_GAP_S):
    analysis = af_analysis(x, fs, t0, ch)
    episodes = _episode_rows(analysis, t0, t0 + len(x) / fs, merge_gap_s)
    return {
        "value": len(episodes),
        "unit": "episodes",
        "method": "AF support grouped across short gaps",
        "merge_gap_s": merge_gap_s,
        "episodes": episodes,
    }


def ecg_irregular(x, fs, unit, t0=0.0, ch=None):
    return sorted(ecg_af_episodes(x, fs, unit, t0, ch)["episodes"], key=lambda e: e["t"])


def ecg_noisy(x, fs, unit, t0=0.0, ch=None):
    analysis = af_analysis(x, fs, t0, ch)
    return [
        {
            "type": "ecg_noisy",
            "t": a,
            "dur": b - a,
            "score": b - a,
            "detail": "unknown RR quality",
            "leads": lead_support(analysis["beats"]),
        }
        for a, b in clip_intervals(analysis["noisy"], t0, t0 + len(x) / fs)
    ]


def ecg_af_burden(x, fs, unit, t0=0.0, ch=None, merge_gap_s=AF_MERGE_GAP_S):
    analysis = af_analysis(x, fs, t0, ch)
    episodes = _episode_rows(analysis, t0, t0 + len(x) / fs, merge_gap_s)
    readable = sum(b - a for a, b in clip_intervals(analysis["readable"], t0, t0 + len(x) / fs))
    af_s = sum(e["af_s"] for e in episodes)
    if af_s > readable + 1e-7:
        raise RuntimeError(f"AF support ({af_s}) exceeds readable time ({readable})")
    window = len(x) / fs
    tdec = time_decimals(fs)
    pct_readable = sig(100 * af_s / readable) if readable else None
    return {
        "value": pct_readable,
        "unit": "%",
        "method": "AF support: pct_of_window = 100*af_s/window_s; pct_of_readable = value = 100*af_s/readable_s",
        "pct_of_window": sig(100 * af_s / window),
        "pct_of_readable": pct_readable,
        "af_s": tround(af_s, tdec),
        "window_s": tround(window, tdec),
        "irregular_s": tround(af_s, tdec),
        "readable_s": tround(readable, tdec),
        "unreadable_s": tround(window - readable, tdec),
        "n_episodes": len(episodes),
    }
