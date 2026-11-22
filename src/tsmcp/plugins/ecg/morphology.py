"""Cached, sample-rate independent QRS morphology and optional two-lead fusion.

Templates are learned from the complete channel, never from a requested view.
All filtering is bounded to five-minute blocks with padding for the symmetric FIR.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from tsmcp.core.dsp import fill_nan
from tsmcp.core.peaks import filter_zero_phase, fir_bandpass

# Round 2 freeze, DS1 only: V/E Se 0.857181, PPV 0.863334; brief VT 26/35.
# R-peak F1: MLII 0.992959, fusion 0.991839. No held-out data were used.
# Strict per-beat prematurity/pause retained only 2417/3788 V/E beats, so
# coherent ectopic clusters and very different broad complexes retain evidence
# after the local RR median adapts to runs or ventricular escape rhythms.
BLOCK_S = 300.0
GRID_FS = 200.0
GRID = np.arange(-32, 33) / GRID_FS
CLUSTER_CORR = 0.90
FAR_CORR = 0.85
MIN_CLUSTER_CORR = 0.90
WIDE_MS = 120.0
PREMATURE_REL = 0.80
MAX_CLUSTERS = 4
CLUSTER_REPLACE_MIN = 8
FUSION_TOL_S = 0.050
REFRACTORY_S = 0.25
ALIGN_S = 0.040
WIDTH_AMPLITUDE_REL = 0.06
WIDTH_MIN_MS = 15.0
WIDTH_MAX_MS = 300.0
MIN_AMPLITUDE = 1e-8
TEMPLATE_MEMORY = 0.25
COUNT_MEMORY = 0.80
DOMINANT_MIN_CONSECUTIVE = 20
PAUSE_REL = 1.10
VENTRICULAR_CLUSTER_TIMING_FRAC = 0.25
RELATIVE_WIDTH = 1.25
WIDE_SHAPE_CORR = 0.95
FUSION_QUALITY_MARGIN = 0.10
FUSION_GAP_REL = 1.50
VERY_FAR_CORR = 0.60

_CACHE = weakref.WeakKeyDictionary()
_FUSED = weakref.WeakKeyDictionary()
_LEADS = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()


def attach_leads(channels):
    """Opt in to fusion of the first two synchronous ECG leads, without retaining channels.

    The server calls this only for ECG operations; library clients can call it explicitly.
    Passing a single channel disables fusion and retains the single-lead path.
    """
    channels = list(channels[:2])
    with _LOCK:
        for ch in channels:
            peers = tuple(weakref.ref(c) for c in channels if c is not ch and c.fs == ch.fs and c.n == ch.n)
            if _LEADS.get(ch) != peers:
                _LEADS[ch] = peers
                _FUSED.pop(ch, None)


def _normalise(waves):
    waves = waves - waves.mean(axis=-1, keepdims=True)
    return waves / np.maximum(np.linalg.norm(waves, axis=-1, keepdims=True), 1e-12)


def _correlate(waves, templates):
    """Normalized cross-correlation with alignment constrained to +/-40 ms."""
    templates = np.asarray(templates)
    best = np.full((len(waves), len(templates)), -1.0)
    aligned = np.zeros((len(waves), len(templates)), int)
    padded = np.pad(waves, ((0, 0), (8, 8)), mode="edge")
    for shift in range(-8, 9):
        shifted = _normalise(padded[:, 8 + shift : 8 + shift + waves.shape[1]])
        corr = shifted @ templates.T
        update = corr > best
        best[update], aligned[update] = corr[update], shift
    return best, aligned


def qrs_features(x, fs, peaks):
    """Aligned 0.5–40 Hz QRS snippets, widths in ms, amplitude; for one padded block.

    Baseline subtraction uses the outer 25 ms on each side. Width includes connected
    deflections above 6% amplitude, allowing gaps up to 20 ms within the QRS.
    The 200 Hz interpolation grid makes width/correlation independent of input fs.
    """
    y = filter_zero_phase(fill_nan(np.asarray(x, float)), fir_bandpass(0.5, min(40.0, fs * 0.45), fs, 2.0))
    if not len(peaks):
        return np.empty((0, GRID.size)), np.empty(0), np.empty(0)
    offsets = np.arange(-round(ALIGN_S * fs), round(ALIGN_S * fs) + 1)
    ix = np.clip(peaks[:, None] + offsets, 0, y.size - 1)
    aligned = ix[np.arange(len(peaks)), np.argmax(np.abs(y[ix]), axis=1)]
    positions = aligned[:, None] + GRID[None, :] * fs
    waves = np.interp(positions.ravel(), np.arange(y.size), y).reshape(len(peaks), -1)
    left, right = np.median(waves[:, :5], axis=1), np.median(waves[:, -5:], axis=1)
    waves -= left[:, None] + (right - left)[:, None] * np.linspace(0, 1, GRID.size)
    amp = np.max(np.abs(waves), axis=1)
    widths = np.empty(len(peaks))
    for i, wave in enumerate(waves):
        active = np.flatnonzero(np.abs(wave) >= WIDTH_AMPLITUDE_REL * amp[i])
        if not active.size or amp[i] <= 1e-10:
            widths[i] = 0.0
            continue
        groups = np.split(active, np.flatnonzero(np.diff(active) > 4) + 1)
        group = min(groups, key=lambda g: np.min(np.abs(g - 32)))
        widths[i] = 1000 * (group[-1] - group[0] + 1) / GRID_FS
    return _normalise(waves), widths, amp


def _classify(x, fs, peaks, good):
    n = peaks.size
    out = {
        "p": peaks,
        "good": good.copy(),
        "label": np.full(n, "unknown", dtype="U16"),
        "cluster": np.full(n, -1, int),
        "correlation": np.zeros(n),
        "width_ms": np.zeros(n),
        "amplitude": np.zeros(n),
        "wide": np.zeros(n, bool),
        "support": good.astype(int),
        "n_leads": np.ones(n, int),
        "premature": np.zeros(n, bool),
    }
    if not n:
        return out
    rr = np.r_[np.nan, np.diff(peaks) / fs]
    pad = np.pad(np.nan_to_num(rr, nan=np.nanmedian(rr)), 16, mode="edge")
    local = np.median(np.lib.stride_tricks.sliding_window_view(pad, 33), axis=1)
    premature = rr < PREMATURE_REL * local
    pause = np.r_[rr[1:], np.nan] > PAUSE_REL * local
    timing = premature | pause
    out["premature"] = premature
    templates, counts, cluster_ids = [], [], []
    next_cluster_id = 0
    dominant_ids = set()
    for start in range(0, len(x), max(1, round(BLOCK_S * fs))):
        end = min(len(x), start + round(BLOCK_S * fs))
        ids = np.flatnonzero((peaks >= start) & (peaks < end))
        if not ids.size:
            continue
        a, b = max(0, start - round(2 * fs)), min(len(x), end + round(2 * fs))
        waves, widths, amp = qrs_features(x[a:b], fs, peaks[ids] - a)
        quality = good[ids] & (amp > MIN_AMPLITUDE) & (widths >= WIDTH_MIN_MS) & (widths <= WIDTH_MAX_MS)
        out["good"][ids] = quality
        out["support"][ids] = quality.astype(int)
        out["width_ms"][ids], out["amplitude"][ids] = widths, amp
        out["wide"][ids] = widths > WIDE_MS
        if not quality.any():
            continue
        if not templates:
            seed = quality & ~premature[ids]
            seed = seed if seed.any() else quality
            templates.append(_normalise(np.median(waves[seed], axis=0)))
            counts.append(1.0)
            cluster_ids.append(next_cluster_id)
            next_cluster_id += 1
        # Up to four persistent clusters. New clusters are seeded by the most frequent
        # unexplained shape, not the first noisy outlier in a block.
        for _ in range(MAX_CLUSTERS):
            corr, _ = _correlate(waves, templates)
            unexplained = quality & (corr.max(axis=1) < CLUSTER_CORR)
            if unexplained.sum() < 2:
                break
            candidates = np.flatnonzero(unexplained)[:: max(1, int(unexplained.sum()) // 64)]
            neighbours = (waves[candidates] @ waves[unexplained].T > CLUSTER_CORR).sum(axis=1)
            template = waves[candidates[np.argmax(neighbours)]].copy()
            if len(templates) < MAX_CLUSTERS:
                templates.append(template)
                counts.append(0.0)
                cluster_ids.append(next_cluster_id)
            else:
                membership = np.sum((corr >= CLUSTER_CORR) & quality[:, None], axis=0)
                stale = int(np.argmin(membership))
                if membership[stale] >= CLUSTER_REPLACE_MIN or neighbours.max() < CLUSTER_REPLACE_MIN:
                    break
                templates[stale], counts[stale], cluster_ids[stale] = template, 0.0, next_cluster_id
            next_cluster_id += 1
        corr, shifts = _correlate(waves, templates)
        quality &= corr.max(axis=1) >= MIN_CLUSTER_CORR
        out["good"][ids] = quality
        out["support"][ids] = quality.astype(int)
        cluster = np.argmax(corr, axis=1)
        for k in range(len(templates)):
            members = quality & (cluster == k)
            counts[k] *= COUNT_MEMORY
            if members.any():
                padded = np.pad(waves[members], ((0, 0), (8, 8)), mode="edge")
                indexes = np.arange(waves.shape[1])[None, :] + 8 + shifts[members, k, None]
                aligned = np.take_along_axis(padded, indexes, axis=1)
                template = _normalise(np.median(aligned, axis=0))
                templates[k] = _normalise(TEMPLATE_MEMORY * templates[k] + (1 - TEMPLATE_MEMORY) * template)
                counts[k] += members.sum()
        dominant = int(np.argmax(counts))
        dominant_ids.add(cluster_ids[dominant])
        for k in range(len(templates)):
            members = quality & (cluster == k)
            if (
                max((b - a for a, b in stretches(members)), default=0) >= DOMINANT_MIN_CONSECUTIVE
                and _correlate(np.asarray([templates[k]]), [templates[dominant]])[0][0, 0] >= 0
            ):
                dominant_ids.add(cluster_ids[k])
        dominant_slots = [k for k, cid in enumerate(cluster_ids) if cid in dominant_ids]
        domcorr = _correlate(waves, [templates[k] for k in dominant_slots])[0].max(axis=1)
        # A chronically wide dominant complex (e.g. bundle branch block) is still
        # dominant. Width supports a ventricular label when the shape also changes.
        far = domcorr < FAR_CORR
        domwidth = np.median(widths[quality & (cluster == dominant)])
        nondominant = ~np.isin(cluster, dominant_slots)
        broad = (widths > WIDE_MS) & (
            ((domcorr < CLUSTER_CORR) & nondominant)
            | ((domwidth <= WIDE_MS) & (domcorr < WIDE_SHAPE_CORR) & (widths > RELATIVE_WIDTH * domwidth))
        )
        # Recurrent ectopic shapes retain their identity in a run after the local
        # RR median adapts to the faster rhythm. Timing alone never labels a V.
        shape_timing = timing[ids].copy()
        for k in range(len(templates)):
            members = quality & (cluster == k)
            if members.sum() >= 3 and np.mean(timing[ids][members]) >= VENTRICULAR_CLUSTER_TIMING_FRAC:
                shape_timing[members] = True
        ventricular = (
            (broad & (shape_timing | (domcorr < VERY_FAR_CORR)))
            | (far & premature[ids])
            | (far & shape_timing & nondominant)
        )
        labels = np.where(ventricular, "ventricular", np.where(premature[ids], "supraventricular", "dominant"))
        labels[~quality] = "unknown"
        out["label"][ids], out["cluster"][ids], out["correlation"][ids] = (
            labels,
            np.asarray(cluster_ids)[cluster],
            domcorr,
        )
    return out


def _channel_classes(ch):
    from tsmcp.plugins.ecg import _channel_beats

    with _LOCK:
        hit = _CACHE.get(ch)
    if hit is None:
        p, g = _channel_beats(ch)
        hit = _classify(ch.data, ch.fs, p, g)
        with _LOCK:
            _CACHE[ch] = hit
    return hit


def _fuse(first, second, fs):
    """One-to-one 50 ms alignment; quality first, then QRS amplitude.

    Only candidates with quality support enter the union. Long stretches with
    no supported beats remain visible as invalid RR gaps; an unreadable second
    lead must not insert noise peaks between readable first-lead beats.
    """
    p, q = first["p"], second["p"]
    pairs, j = [], 0
    for i in range(len(p)):
        while j < len(q) and q[j] < p[i] - FUSION_TOL_S * fs:
            pairs.append((1, j, int(second["good"][j])))
            j += 1
        if j < len(q) and abs(q[j] - p[i]) <= FUSION_TOL_S * fs:
            choices = [(int(first["good"][i]), first["amplitude"][i]), (int(second["good"][j]), second["amplitude"][j])]
            lead = int(choices[1] > choices[0])
            pairs.append((lead, j if lead else i, int(first["good"][i]) + int(second["good"][j])))
            j += 1
        else:
            pairs.append((0, i, int(first["good"][i])))
    pairs.extend((1, k, int(second["good"][k])) for k in range(j, len(q)))
    sources = (first, second)
    local_quality = [
        np.convolve(np.pad(s["good"].astype(float), 16, mode="edge"), np.ones(33) / 33, "valid")
        if len(s["p"])
        else np.empty(0)
        for s in sources
    ]
    primary_rr = np.diff(p) / fs
    typical_rr = np.median(primary_rr) if len(primary_rr) else 0.0
    pairs.sort(key=lambda z: sources[z[0]]["p"][z[1]])
    keep = []
    for row in pairs:
        lead, i, support = row
        if not support:
            continue
        if lead == 1 and support == 1 and len(p) > 1:
            right = int(np.searchsorted(p, q[i]))
            if 0 < right < len(p) and first["good"][right - 1 : right + 1].all():
                # A second-lead candidate inside a complete readable primary
                # cycle is commonly a T wave. Rescue actual primary gaps only.
                near = min(q[i] - p[right - 1], p[right] - q[i])
                if near > FUSION_TOL_S * fs and (p[right] - p[right - 1]) / fs < FUSION_GAP_REL * typical_rr:
                    continue
        if keep and sources[lead]["p"][i] - sources[keep[-1][0]]["p"][keep[-1][1]] < REFRACTORY_S * fs:
            oldlead, oldi, oldsupport = keep[-1]
            if (support > 0, sources[lead]["amplitude"][i]) > (oldsupport > 0, sources[oldlead]["amplitude"][oldi]):
                keep[-1] = row
        else:
            keep.append(row)
    out = {
        key: np.array([sources[lead][key][i] for lead, i, _ in keep], dtype=value.dtype) for key, value in first.items()
    }
    out["support"] = np.array([s for _, _, s in keep], int)
    out["n_leads"] = np.full(len(keep), 2, int)
    # Cluster identity includes the lead, since its templates use a different axis.
    out["cluster"] = 2 * out["cluster"] + np.array([lead for lead, _, _ in keep], dtype=int)
    # Amplitude chooses the peak location, not a different morphology reference
    # on every beat. Prefer the requested lead's morphology; the peer rescues
    # unavailable beats or a locally less readable primary morphology stream.
    if len(p) and len(out["p"]):
        right = np.minimum(np.searchsorted(p, out["p"]), len(p) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(abs(p[left] - out["p"]) < abs(p[right] - out["p"]), left, right)
        use = (abs(p[nearest] - out["p"]) <= FUSION_TOL_S * fs) & first["good"][nearest]
        if len(q):
            right2 = np.minimum(np.searchsorted(q, out["p"]), len(q) - 1)
            left2 = np.maximum(right2 - 1, 0)
            nearest2 = np.where(abs(q[left2] - out["p"]) < abs(q[right2] - out["p"]), left2, right2)
            peer = (abs(q[nearest2] - out["p"]) <= FUSION_TOL_S * fs) & second["good"][nearest2]
            peer &= ~use | (local_quality[1][nearest2] > local_quality[0][nearest] + FUSION_QUALITY_MARGIN)
            use &= ~peer
            for key in ("label", "correlation", "width_ms", "wide", "premature"):
                out[key][peer] = second[key][nearest2[peer]]
            out["cluster"][peer] = 2 * second["cluster"][nearest2[peer]] + 1
        for key in ("label", "correlation", "width_ms", "wide", "premature"):
            out[key][use] = first[key][nearest[use]]
        out["cluster"][use] = 2 * first["cluster"][nearest[use]]
    return out


def channel_classes(ch):
    with _LOCK:
        peers = [r() for r in _LEADS.get(ch, ()) if r() is not None]
        hit = _FUSED.get(ch)
    if not peers:
        return _channel_classes(ch)
    if hit is None:
        hit = _fuse(_channel_classes(ch), _channel_classes(peers[0]), ch.fs)
        with _LOCK:
            _FUSED[ch] = hit
    return hit


def fused_peaks(ch):
    """Use the original single-lead detection path unless a second lead was attached."""
    from tsmcp.plugins.ecg import _channel_beats

    with _LOCK:
        paired = any(r() is not None for r in _LEADS.get(ch, ()))
    if not paired:
        return _channel_beats(ch)
    beats = channel_classes(ch)
    return beats["p"], beats["good"]


def beat_classes(x, fs, t0=0.0, ch=None):
    """Per-beat arrays: times, labels, cluster IDs, dominant correlation, ms widths, quality and lead support."""
    from tsmcp.plugins.ecg import _detect_q

    if ch is not None and ch.fs == fs:
        result = channel_classes(ch)
        i0 = round(t0 * fs)
        mask = (result["p"] >= i0) & (result["p"] < i0 + len(x))
        result = {k: v[mask].copy() for k, v in result.items()}
        result["p"] -= i0
    else:
        p, g = _detect_q(x, fs)
        result = _classify(x, fs, p, g)
    result["t_s"] = t0 + result["p"] / fs
    return result


def lead_support(beats, mask=None):
    if not len(beats["p"]):
        return "1/1"
    n = int(beats["n_leads"][0])
    support = beats["support"] if mask is None else beats["support"][mask]
    return f"{2 if n == 2 and len(support) and np.mean(support == 2) >= 0.5 else 1}/{n}"


def stretches(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
