"""ECG plugin: beat-level measures built on the ported R-peak detector.

All times are absolute seconds (window start t0 + offset). RR intervals are between consecutive detected beats.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from tsmcp.core.compact import sig
from tsmcp.core.errors import SignalError
from tsmcp.plugins.base import DomainPlugin
from tsmcp.plugins.ecg.af import ecg_af_burden, ecg_af_episodes, ecg_irregular, ecg_noisy
from tsmcp.plugins.ecg.rpeaks import bandpass_qrs, detect_r_peaks

MAX_BEATS_LISTED = 150
PROFILE_BLOCK_S = 10.0
MAX_PROFILE_BLOCKS = 60
LOW_AMP_GLOBAL = 0.25  # beat amplitude below this x record-typical amplitude -> unreadable (noise / lead off)
MAX_RR_S = 5.0  # longest plausible RR interval (asystole pauses beyond this are reported as missing data)
LOW_AMP_LOCAL = 0.35  # ... or below this x the local (+-50 beats) typical amplitude


_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # Channel -> (peaks, good) for the whole channel
_LOCK = threading.Lock()


def _quality(filtered: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """A beat is 'good' if its QRS amplitude is not far below the record-typical and the local-typical amplitude.
    Lead-off / flat stretches make the adaptive detector pick up noise peaks; those are tiny and get excluded."""
    if peaks.size == 0:
        return np.zeros(0, bool)
    amp = np.abs(filtered[peaks])
    return _amplitude_quality(amp)


def _amplitude_quality(amp):
    if not len(amp):
        return np.zeros(0, bool)
    glob = np.median(amp)
    k = 50
    pad = np.pad(amp, k, mode="edge")
    local = (
        np.array([np.median(pad[i : i + 2 * k + 1]) for i in range(amp.size)])
        if amp.size > 2 * k
        else np.full(amp.size, glob)
    )
    return (amp >= LOW_AMP_GLOBAL * glob) & (amp >= LOW_AMP_LOCAL * local)


def _detect_q(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if len(x) > 600 * fs:
        # 30 s padding preserves the detector's global 30 s adaptive block grid.
        # Avoid full-day FFT temporaries; only amplitudes and peaks are retained.
        peaks, amplitudes = [], []
        for start in range(0, len(x), round(300 * fs)):
            end = min(len(x), start + round(300 * fs))
            a, b = max(0, start - round(30 * fs)), min(len(x), end + round(30 * fs))
            segment = np.asarray(x[a:b], dtype=float)
            p = detect_r_peaks(segment, fs)
            p = p[(p + a >= start) & (p + a < end)]
            peaks.append(p + a)
            amplitudes.append(np.abs(bandpass_qrs(segment, fs)[p]))
        p, amp = np.concatenate(peaks), np.concatenate(amplitudes)
        return p, _amplitude_quality(amp)
    x = np.asarray(x, dtype=np.float64)
    p = detect_r_peaks(x, fs)
    return p, _quality(bandpass_qrs(x, fs), p) if p.size else np.zeros(0, bool)


def _channel_beats(ch) -> tuple[np.ndarray, np.ndarray]:
    with _LOCK:
        hit = _CACHE.get(ch)
    if hit is None:
        hit = _detect_q(ch.data, ch.fs)
        with _LOCK:
            _CACHE[ch] = hit
    return hit


def _beats_q(x: np.ndarray, fs: float, t0: float = 0.0, ch=None) -> tuple[np.ndarray, np.ndarray]:
    """(R-peak indices into x, good-quality flags). With the full channel available, beats are detected once per
    channel (cached) and sliced: faster for repeated zooming and more accurate at window edges."""
    if x.size < int(3 * fs):
        raise SignalError("window shorter than 3 s; ECG beat measures need at least 3 s")
    if ch is not None and ch.fs == fs:
        i0 = int(round(t0 * fs))
        allp, allg = _channel_beats(ch)
        m = (allp >= i0) & (allp < i0 + x.size)
        p, g = allp[m] - i0, allg[m]
    else:
        p, g = _detect_q(x, fs)
    if g.sum() < 3:
        raise SignalError("fewer than 3 readable beats in this window (no ECG, or signal quality too low)")
    return p, g


def _beats(x: np.ndarray, fs: float, t0: float = 0.0, ch=None) -> np.ndarray:
    """Good-quality R peaks only."""
    p, g = _beats_q(x, fs, t0, ch)
    return p[g]


def _rr(x, fs, t0, ch) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """RR intervals between consecutive detected beats, their end times (absolute s), a validity mask (both ends
    readable) and the number of excluded beats."""
    p, g = _beats_q(x, fs, t0, ch)
    rr = np.diff(p) / fs
    valid = g[1:] & g[:-1] & (rr <= MAX_RR_S)  # a > 5 s gap is missing data (ecg_noisy), not an interval
    return rr, t0 + p[1:] / fs, valid, int((~g).sum())


def _valid_pairs_diff(rr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ok = valid[1:] & valid[:-1]
    return np.diff(rr)[ok]


def ecg_hr(x, fs, unit, t0=0.0, ch=None) -> dict:
    """Average heart rate = 60 / mean RR over readable beats (equals 60*(beats-1)/span when all are readable)."""
    rr, _, v, excl = _rr(x, fs, t0, ch)
    r = rr[v]
    if r.size < 2:
        raise SignalError("too few readable RR intervals in this window")
    inst = 60 / r
    out = {
        "value": sig(60 / r.mean()),
        "unit": "bpm",
        "method": "r-peaks",
        "n_beats": int(r.size + 1),
        "median_bpm": sig(np.median(inst)),
        "p5_p95_bpm": [sig(np.percentile(inst, 5)), sig(np.percentile(inst, 95))],
    }
    if excl:
        out["excluded_beats"] = excl
    return out


def ecg_rr_stats(x, fs, unit, t0=0.0, ch=None) -> dict:
    """RR-interval statistics (time-domain HRV) over readable beats."""
    rr, _, v, excl = _rr(x, fs, t0, ch)
    r = rr[v]
    if r.size < 2:
        raise SignalError("too few readable RR intervals in this window")
    d = _valid_pairs_diff(rr, v)
    out = {
        "value": sig(r.mean()), "unit": "s", "method": "r-peaks", "n_rr": int(r.size), "sdnn_s": sig(r.std()),
        "rmssd_s": sig(np.sqrt(np.mean(d**2))) if d.size else None,
        "pnn50": sig(np.mean(np.abs(d) > 0.05)) if d.size else None,
        "cv": sig(r.std() / r.mean()), "min_max_s": [sig(r.min()), sig(r.max())],
    }  # fmt: skip
    if excl:
        out["excluded_beats"] = excl
    return out


def ecg_rr_profile(x, fs, unit, t0=0.0, ch=None) -> dict:
    """Rhythm over time: per block, mean HR, RR coefficient of variation, and short-long alternation.

    rr_cv high and alt low suggests irregular rhythm (e.g. atrial fibrillation); alt high (successive RR differences
    alternating in sign with >10% change) suggests bigeminy-like patterns; a jump in hr marks a rate change.
    Blocks with no readable beats are null.
    """
    rr_all, tr_all, v, _ = _rr(x, fs, t0, ch)
    rr, tr = rr_all[v], tr_all[v] - t0
    dur = x.size / fs
    nb = int(min(MAX_PROFILE_BLOCKS, max(1, round(dur / PROFILE_BLOCK_S))))
    edges = np.linspace(0, dur, nb + 1)
    hr, cv, alt, n = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        r = rr[(tr >= a) & (tr < b)]
        n.append(int(r.size))
        if r.size < 2:
            hr.append(None)
            cv.append(None)
            alt.append(None)
            continue
        hr.append(sig(60 / r.mean()))
        cv.append(sig(r.std() / r.mean(), 2))
        d = np.diff(r)
        big = np.abs(d) > 0.1 * r.mean()
        flips = (np.sign(d[1:]) != np.sign(d[:-1])) & big[1:] & big[:-1] if d.size > 1 else np.zeros(0, bool)
        alt.append(sig(flips.mean(), 2) if flips.size else 0.0)
    return {"value": None, "unit": "per block", "method": "r-peaks", "t0": sig(t0, 6), "dt_s": sig(dur / nb),
            "hr_bpm": hr, "rr_cv": cv, "alt": alt, "n_rr": n}  # fmt: skip


def ecg_beats(x, fs, unit, t0=0.0, ch=None) -> dict:
    """Individual readable beats in the window: R-peak times and the RR interval preceding each (max 150 beats)."""
    p = _beats(x, fs, t0, ch)
    if p.size > MAX_BEATS_LISTED:
        raise SignalError(f"{p.size} beats in window; ecg_beats lists at most {MAX_BEATS_LISTED}: use a shorter window")
    t = t0 + p / fs
    return {"value": int(p.size), "unit": "beats", "method": "r-peaks", "t_s": [round(float(v), 3) for v in t],
            "rr_s": [round(float(v), 3) for v in np.diff(p) / fs]}  # fmt: skip


# ---------------------------------------------------------------- rhythm events
EPISODE_WIN_BEATS = 16
EPISODE_STEP_BEATS = 8
MIN_EPISODE_S = 10.0
PAUSE_S = 2.0
MERGE_GAP_S = 15.0  # same-kind episodes this close are one episode (a single odd-looking window must not split it)


IRREGULAR_MEDIAN_REL = 0.12  # tuned on the dev set only (MIT-BIH DS1 + LTAFDB 07); see eval/validate_af.py


def remove_ectopic_pairs(rr: np.ndarray) -> np.ndarray:
    """Replace premature-beat + compensatory-pause pairs (short RR then long RR, summing to ~2 normal RRs)
    by two average intervals. Isolated ectopics then no longer look like irregular rhythm; AF stays irregular."""
    rr = rr.astype(float).copy()
    med = np.median(rr)
    i = 0
    while i < rr.size - 1:
        a, b = rr[i], rr[i + 1]
        if a < 0.85 * med and b > 1.1 * med and abs(a + b - 2 * med) < 0.2 * med:
            rr[i] = rr[i + 1] = (a + b) / 2
            i += 2
        else:
            i += 1
    return rr


def _mdiff(rr: np.ndarray, k: int) -> float:
    return float(np.median(np.abs(rr[k:] - rr[:-k])) / np.median(rr))


def classify_rr(rr: np.ndarray, t_irr: float | None = None, remove_pairs: bool = True) -> str:
    """Rhythm class of a short RR sequence from the structure of its successive differences.

    MEDIANS of |RR[i]-RR[i-k]| relative to the median RR. Order matters:
    1. lag-1 small                          -> regular
    2. lag-2 small (period 2 on raw RR)     -> alternating (bigeminy-like)
    3. lag-3 small (period 3 on raw RR)     -> period3 (trigeminy-like)
    4. after removing ectopic pairs, lag-1 small -> regular (sinus with isolated ectopics)
    5. otherwise                            -> irregular (AF-like)
    """
    t_irr = IRREGULAR_MEDIAN_REL if t_irr is None else t_irr
    d1 = _mdiff(rr, 1)
    if d1 < t_irr:
        return "regular"
    if _mdiff(rr, 2) < 0.5 * d1:
        return "alternating"
    if _mdiff(rr, 3) < 0.5 * d1:
        return "period3"
    if remove_pairs and _mdiff(remove_ectopic_pairs(rr), 1) < t_irr:
        return "regular"
    return "irregular"


def _rr_windows(x, fs, t0, ch):
    """(beat times of each RR's start, RR, valid) — valid = both beats readable."""
    rr, t_end, v, _ = _rr(x, fs, t0, ch)
    return t_end - rr, rr, v


def _episodes(labels: list[str], starts: list[float], ends: list[float], want: str, kind: str) -> list[dict]:
    out, cur = [], None
    for lab, a, b in zip(labels, starts, ends, strict=True):
        if lab == want:
            cur = [a, b] if cur is None else [cur[0], b]
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    merged: list[list[float]] = []
    for a, b in out:
        if merged and a - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [{"type": kind, "t": a, "dur": b - a, "score": b - a} for a, b in merged if b - a >= MIN_EPISODE_S]


def _rhythm_labels(x, fs, t0, ch):
    t, rr, v = _rr_windows(x, fs, t0, ch)
    labels, starts, ends = [], [], []
    for i in range(0, max(0, rr.size - EPISODE_WIN_BEATS + 1), EPISODE_STEP_BEATS):
        w = slice(i, i + EPISODE_WIN_BEATS)
        labels.append(classify_rr(rr[w]) if v[w].all() else "noisy")
        starts.append(float(t[i]))
        ends.append(float(t[i + EPISODE_WIN_BEATS - 1] + rr[i + EPISODE_WIN_BEATS - 1]))
    return labels, starts, ends


def _rhythm(x, fs, unit, t0=0.0, ch=None, want="irregular", kind="ecg_irregular") -> list[dict]:
    return _episodes(*_rhythm_labels(x, fs, t0, ch), want, kind)


def ecg_alternating(x, fs, unit, t0=0.0, ch=None) -> list[dict]:
    return _rhythm(x, fs, unit, t0, ch, "alternating", "ecg_alternating")


def ecg_period3(x, fs, unit, t0=0.0, ch=None) -> list[dict]:
    return _rhythm(x, fs, unit, t0, ch, "period3", "ecg_period3")


def ecg_pause(x, fs, unit, t0=0.0, ch=None) -> list[dict]:
    t, rr, v = _rr_windows(x, fs, t0, ch)
    rr = np.where(v, rr, 0.0)  # invalid (unreadable or > MAX_RR_S) intervals are ecg_noisy, not pauses
    return [
        {"type": "ecg_pause", "t": float(t[i]), "dur": float(rr[i]), "score": float(rr[i])}
        for i in np.flatnonzero(rr > PAUSE_S)
    ]


def _rate_episodes(x, fs, unit, t0, ch, test, kind, min_s=30.0) -> list[dict]:
    t, rr, v = _rr_windows(x, fs, t0, ch)
    hr = np.where(v, 60 / rr, np.nan)
    k = 8  # median over 8 beats: robust to single ectopics
    if hr.size < k:
        return []
    med = np.array(
        [
            np.nanmedian(w) if np.isfinite(w).sum() >= 5 else np.nan
            for w in (hr[i : i + k] for i in range(hr.size - k + 1))
        ]
    )
    mask = np.nan_to_num(test(np.nan_to_num(med, nan=75.0)), nan=False)
    out, i = [], 0
    while i < mask.size:
        if mask[i]:
            j = i
            while j + 1 < mask.size and mask[j + 1]:
                j += 1
            a, b = float(t[i]), float(t[min(j + k, t.size - 1)])
            if b - a >= min_s:
                out.append(
                    {
                        "type": kind,
                        "t": a,
                        "dur": b - a,
                        "score": float(np.median(med[i : j + 1])),
                        "detail": "median bpm",
                    }
                )
            i = j + 1
        else:
            i += 1
    return out


def ecg_tachy(x, fs, unit, t0=0.0, ch=None) -> list[dict]:
    return _rate_episodes(x, fs, unit, t0, ch, lambda h: h > 100, "ecg_tachy")


def ecg_brady(x, fs, unit, t0=0.0, ch=None) -> list[dict]:
    return _rate_episodes(x, fs, unit, t0, ch, lambda h: h < 50, "ecg_brady")


# ---------------------------------------------------------------- derived series
def series_rr(x, fs, unit, t0=0.0, ch=None) -> dict:
    rr, t_end, v, _ = _rr(x, fs, t0, ch)
    return {"t": t_end[v], "y": rr[v], "unit": "s", "label": "RR interval (readable beats)"}


def series_hr(x, fs, unit, t0=0.0, ch=None) -> dict:
    rr, t_end, v, _ = _rr(x, fs, t0, ch)
    return {"t": t_end[v], "y": 60 / rr[v], "unit": "bpm", "label": "instantaneous HR (readable beats)"}


# ---------------------------------------------------------------- ECG paper view
MAX_PAPER_S = 60.0
ROW_S = 10.0


def view_ecg_paper(ch, t0: float, t1: float, detail: str = "normal", title: str | None = None, events=None) -> bytes:
    """Standard ECG paper: 25 mm/s, 10 mm/mV, 1 mm / 5 mm grid, 10 s per row, R peaks marked with RR in ms."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from tsmcp.core.signal import slice_channel
    from tsmcp.core.views.render import _LOCK, _png

    if t1 - t0 > MAX_PAPER_S + 1e-9:
        raise SignalError(f"ecg_paper shows at most {MAX_PAPER_S:g} s; use a shorter window or view='strip'")
    rows = max(1, int(np.ceil((t1 - t0) / ROW_S - 1e-9)))
    first = max(0, int(np.floor(t0 * ch.fs))) / ch.fs
    x = slice_channel(ch, t0, t1)
    peaks = _beats(x, ch.fs, first, ch) if x.size >= 3 * ch.fs else np.empty(0, int)
    lo, hi = np.nanpercentile(x, [0.1, 99.9]) if x.size else (-1, 1)
    span = float(np.clip(hi - lo + 0.4, 2.0, 5.0))
    mid = (hi + lo) / 2
    px_per_mm = 4.8 if detail == "normal" else 8.0
    w = int(ROW_S * 25 * px_per_mm) + 60
    row_h = int(span * 10 * px_per_mm)
    h = min(2576, rows * row_h + 60)
    with _LOCK:
        fig = Figure(figsize=(w / 100, h / 100), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(rows, 1, squeeze=False)[:, 0]
        tt = first + np.arange(x.size) / ch.fs
        pk_t = first + peaks / ch.fs
        for r, ax in enumerate(axes):
            a, b = t0 + r * ROW_S, t0 + (r + 1) * ROW_S
            ax.set_xlim(a, b)
            ax.set_ylim(mid - span / 2, mid + span / 2)
            ax.set_xticks(np.arange(np.ceil(a / 0.2) * 0.2, b, 0.2), minor=False)
            ax.set_xticks(np.arange(np.ceil(a / 0.04) * 0.04, b, 0.04), minor=True)
            ax.set_yticks(np.arange(np.ceil((mid - span / 2) / 0.5) * 0.5, mid + span / 2, 0.5))
            ax.set_yticks(np.arange(np.ceil((mid - span / 2) / 0.1) * 0.1, mid + span / 2, 0.1), minor=True)
            ax.grid(True, which="major", color="#f08080", linewidth=0.6)
            ax.grid(True, which="minor", color="#fbd5d5", linewidth=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            m = (tt >= a) & (tt < b)
            ax.plot(tt[m], x[m], color="black", linewidth=0.7)
            ax.text(a, mid + span / 2, f" {a:g} s", fontsize=7, va="top", color="#444")
            pm = (pk_t >= a) & (pk_t < b)
            for k in np.flatnonzero(pm):
                ax.plot([pk_t[k]], [mid + span / 2 * 0.85], "v", color="#1f4e79", markersize=3)
                if k > 0:
                    ax.text((pk_t[k] + pk_t[k - 1]) / 2, mid + span / 2 * 0.9, f"{1000 * (pk_t[k] - pk_t[k - 1]):.0f}",
                            fontsize=6, ha="center", va="top", color="#1f4e79")  # fmt: skip
        axes[0].set_title(
            (title + "  " if title else "") + f"{ch.name}: 25 mm/s, 10 mm/mV, RR in ms", fontsize=8, loc="left"
        )
        return _png(fig)


PLUGIN = DomainPlugin(
    "ecg",
    measures={
        "ecg_hr": ecg_hr,
        "ecg_rr_stats": ecg_rr_stats,
        "ecg_rr_profile": ecg_rr_profile,
        "ecg_beats": ecg_beats,
        "ecg_af_burden": ecg_af_burden,
        "ecg_af_episodes": ecg_af_episodes,
    },
    summary="ECG only: ecg_hr (bpm), ecg_rr_stats (RR mean/SDNN/RMSSD/pNN50/CV), "
    "ecg_rr_profile (per ~10 s block: hr, rr_cv, alt = short-long alternation — use it to find rhythm changes), "
    "ecg_beats (R-peak times + RR list, windows <= ~2 min), "
    "ecg_af_episodes (AF-like episodes ranked by span; af_s excludes gaps), "
    "ecg_af_burden (AF-like rhythm: pct_of_window = 100*af_s/window_s; "
    "pct_of_readable = value = 100*af_s/readable_s; af_s, window_s, readable_s, unreadable_s in seconds; "
    "use this instead of adding up episodes yourself)",
    detectors={
        "ecg_irregular": ecg_irregular,
        "ecg_alternating": ecg_alternating,
        "ecg_period3": ecg_period3,
        "ecg_pause": ecg_pause,
        "ecg_tachy": ecg_tachy,
        "ecg_brady": ecg_brady,
        "ecg_noisy": ecg_noisy,
    },  # fmt: skip
    detectors_summary="ECG only: ecg_irregular (irregularly irregular RR, AF-like episodes >= 10 s), ecg_alternating "
    "(long-short RR alternation, bigeminy-like), ecg_period3 (trigeminy-like), ecg_pause (RR > 2 s), "
    "ecg_tachy (HR > 100 for >= 30 s), ecg_brady (HR < 50 for >= 30 s), ecg_noisy (unreadable stretches: "
    "lead off / noise; rhythm detectors skip them)",
    series={"ecg_rr": series_rr, "ecg_hr": series_hr},
    views={"ecg_paper": view_ecg_paper},
    views_summary="ECG only: ecg_rr / ecg_hr (RR-interval tachogram / instantaneous HR over time — the best way to "
    "see rhythm changes and AF episodes over hours), ecg_paper (standard 25 mm/s ECG paper with RR labels, <= 60 s)",
)
