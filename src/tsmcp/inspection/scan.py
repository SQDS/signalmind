"""Candidate discovery on source-supported samples. All outputs use block-local half-open indices."""

from collections import defaultdict

import numpy as np

from tsmcp.inspection.profiles import profile
from tsmcp.plugins.ecg.rpeaks import detect_r_peaks
from tsmcp.plugins.pressure.pulses import detect_pulses_unguided
from tsmcp.processing import intervals

# Engineering starting points, fixed before evaluating the 80-case development set.
PRESETS = {
    "quick": {"sensitivity": 0.25, "factor": 1.5},
    "balanced": {"sensitivity": 0.5, "factor": 1.0},
    "deep": {"sensitivity": 0.75, "factor": 0.65},
    "thorough": {"sensitivity": 0.75, "factor": 0.65},
}
SCALES = (5, 10, 20, 60, 100)
LINE_FRACTION = 0.012
HIGH_FRACTION = {"ECG": 0.10, "PPG": 0.05, "ABP": 0.05}
HIGH_BAND_HZ = {"ECG": 45, "PPG": 18, "ABP": 20}


def windows(n, size, stride):
    """Include a full final window anchored at the end; one partial window for short runs."""
    if n <= size:
        yield 0, n
        return
    for start in range(0, n - size + 1, stride):
        yield start, start + size
    if (n - size) % stride:
        yield n - size, n


def abp_pulse_episodes(x, fs, factor, *, evaluated=None):
    """Review transient changes in pressure cycles; physiological changes can meet these criteria."""
    feet, _, _ = detect_pulses_unguided(x, fs)
    flank, guard = 6, 6  # complete pressure cycles on each side, outside the central beats
    margin = flank + guard
    if len(feet) - 1 <= 2 * margin:
        return
    durations = np.diff(feet)
    samples = x[feet[0] : feet[-1] + 1]
    starts = feet[:-1] - feet[0]
    sbp = np.maximum.reduceat(samples[:-1], starts)
    dbp = np.minimum.reduceat(samples[:-1], starts)
    mean = np.add.reduceat(0.5 * (samples[:-1] + samples[1:]), starts) / durations
    features = np.column_stack((sbp, dbp, mean, sbp - dbp, durations / fs))
    bands = np.lib.stride_tricks.sliding_window_view(features, 2 * margin + 1, axis=0)
    left, right = bands[:, :, :flank], bands[:, :, -flank:]
    before, after = np.median(left, axis=2), np.median(right, axis=2)
    jitter = 1.4826 * np.maximum(
        np.median(abs(left - before[:, :, None]), axis=2), np.median(abs(right - after[:, :, None]), axis=2)
    )
    beats = features[margin:-margin]
    # Compare with both flanks, scaling pressure changes by local pulse pressure and variability.
    pp = np.maximum(before[:, 3], after[:, 3])
    scale = np.maximum(0.18 * pp[:, None], 4 * jitter[:, :3])
    drop = (np.minimum(before[:, :3], after[:, :3]) - beats[:, :3]) / np.maximum(scale, np.finfo(float).eps)
    pressure = np.minimum(drop[:, 2], np.maximum(drop[:, 0], drop[:, 1]))
    slow = np.maximum(before[:, 4], after[:, 4])
    timing = (beats[:, 4] - slow) / np.maximum(0.55 * slow, 4 * jitter[:, 4])
    long_low = np.minimum(timing, np.maximum(drop[:, 1], drop[:, 2]))
    paired = pressure >= factor
    sustained = paired & (np.r_[False, paired[:-1]] | np.r_[paired[1:], False])
    core = (pp > 0) & (sustained | (long_low >= factor))
    if evaluated is not None:
        for a, b in intervals(pp > 0):
            evaluated[feet[margin + a] : feet[margin + b]] = True
    tails = (pp > 0) & ((pressure >= 0.55 * factor) | (long_low >= 0.55 * factor))
    episodes = []
    for a, b in intervals(tails):
        if not core[a:b].any():
            continue
        if episodes and a - episodes[-1][1] <= 2:
            episodes[-1][1] = b
        else:
            episodes.append([a, b])
    for a, b in episodes:
        score = np.maximum(np.where(sustained[a:b], pressure[a:b], 0), long_low[a:b])
        yield int(feet[margin + a]), int(feet[margin + b]), float(np.max(score)), factor


def ecg_excursions(x, fs, scales, factor):
    """Locate bounded amplitude excursions with sustained displacement and observed recovery."""
    history = max(4, round(1.5 * fs))
    recovery = max(2, round(0.12 * fs))
    minimum = max(recovery, round(0.20 * fs))
    max_length = round(8 * fs)
    last_return = 0
    next_seed = history
    changes = np.abs(np.diff(x))
    if not len(changes):
        return
    edge_threshold = np.copy(scales)
    for left in range(0, len(changes), history):
        right = min(len(changes), left + history)
        refs = np.r_[changes[max(0, left - 2 * history) : left], changes[right : right + 2 * history]]
        if refs.size:
            edge_threshold[left:right] = np.minimum(scales[left:right], 8 * np.percentile(refs, 90))
    for seed in np.flatnonzero(changes > edge_threshold) + 1:
        if seed < next_seed or seed + minimum + recovery > len(x):
            continue
        anchor = x[max(seed - history, last_return) : seed]
        center = float(np.median(anchor))
        width = max(float(np.ptp(np.percentile(anchor, [5, 95]))), scales[seed - 1] * 0.05)
        limit = 1.5 * width
        left = int(seed)
        while left > seed - recovery and abs(x[left - 1] - center) > limit:
            left -= 1
        residual = np.abs(x[left : min(len(x), left + max_length + recovery)] - center)
        outside = residual > limit
        # End at the start of the first sustained return; the confirmation samples are context.
        returns = [(a, b) for a, b in intervals(~outside) if b - a >= recovery]
        if not returns:
            continue
        end = returns[0][0]
        if not minimum <= end <= max_length or outside[:end].mean() < 0.75:
            continue
        post = x[left + end : min(left + end + history, len(x))]
        if abs(float(np.median(post)) - center) > width * 0.5:
            continue
        last_return = left + end
        next_seed = last_return + recovery
        strength = float(np.max(residual[:end]) / width)
        if strength >= 3 * factor:
            yield left, left + end, strength, 3 * factor


def ecg_baseline_shifts(x, fs, factor, *, evaluated=None):
    """Find short, bounded baseline displacements even when superimposed QRS slopes still match."""
    hop = max(1, round(0.08 * fs))
    width = max(3, round(0.32 * fs)) | 1
    flank = max(2, round(2 * fs))
    spacing = max(1, round(fs))
    offset = round(2.5 * fs)
    if len(x) < 2 * offset + flank + width:
        return
    windows_view = np.lib.stride_tricks.sliding_window_view
    level = np.median(windows_view(x, width)[::hop], axis=1)
    centers = np.arange(len(level)) * hop + width // 2
    # Evaluate the two reference flanks separately so a gain change or ramp is not a bounded excursion.
    q05, median, q95 = np.percentile(windows_view(x, flank)[::spacing], [5, 50, 95], axis=1)
    knots = np.arange(len(median)) * spacing + (flank - 1) / 2
    pre, post = centers - offset, centers + offset
    before, after = np.interp(pre, knots, median), np.interp(post, knots, median)
    span = np.maximum(np.interp(pre, knots, q95 - q05), np.interp(post, knots, q95 - q05))
    span = np.maximum(span, np.finfo(float).eps * max(1.0, np.max(np.abs(x))))
    supported = (pre >= knots[0]) & (post <= knots[-1]) & (abs(after - before) <= 0.75 * span)
    if evaluated is not None:
        for left, right in intervals(supported):
            start = centers[left] - hop // 2
            evaluated[start : start + (right - left) * hop] = True
    threshold = 1.5 * factor
    for sign in (1, -1):
        strength = np.minimum(sign * (level - before), sign * (level - after)) / span
        for a, b in intervals(supported & (strength >= threshold)):
            if (b - a) * hop < 0.30 * fs:
                continue
            peak = a + int(np.argmax(strength[a:b]))
            tail = 0.30 * span[peak]
            left, right = a, b
            while left > 0 and (right - left) * hop <= 4 * fs and sign * (level[left - 1] - before[peak]) > tail:
                left -= 1
            while right < len(level) and (right - left) * hop <= 4 * fs and sign * (level[right] - after[peak]) > tail:
                right += 1
            if left == 0 or right == len(level) or (right - left) * hop > 4 * fs:
                continue
            yield int(centers[left] - hop // 2), int(centers[right] - hop // 2), float(strength[peak]), threshold


def shifted_qrs_references(references, width):
    shifts = np.lib.stride_tricks.sliding_window_view(references, width, axis=1)
    bank = np.concatenate((shifts, (shifts[:, 1:] + shifts[:, :-1]) * 0.5), axis=1)
    bank -= np.median(bank, axis=2, keepdims=True)
    return bank, np.ptp(bank, axis=2), np.max(abs(np.diff(bank, axis=2)), axis=2), np.sum(bank**2, axis=2)


def ecg_repeated_qrs_edges(x, fs, edges):
    """Explain steep edges only when several other raw QRS complexes support the same shape."""
    explained = np.zeros(len(edges), bool)
    if not len(edges) or fs <= 40:
        return explained
    peaks = detect_r_peaks(x, fs)
    radius = max(2, round(0.10 * fs))
    # Compare original samples; shift only the references, including half-sample offsets.
    lag = max(1, round(0.012 * fs))
    margin = radius + lag
    peaks = peaks[(peaks >= margin) & (peaks + margin < len(x))]
    if len(peaks) < 6:
        return explained
    offsets = np.arange(-radius, radius + 1)
    references = np.asarray(x[peaks[:, None] + np.arange(-margin, margin + 1)], dtype=float)
    raw = references[:, lag : lag + len(offsets)]
    centered = raw - np.median(raw, axis=1, keepdims=True)
    broad = np.sum(abs(centered) > 0.5 * np.max(abs(centered), axis=1, keepdims=True), axis=1) >= max(
        2, round(0.016 * fs)
    )
    # Try missed beats only at a locally supported cadence. They are never added to an event table.
    mid = edges + 0.5
    after = np.searchsorted(peaks, mid).clip(0, len(peaks) - 1)
    before = np.searchsorted(peaks, mid, side="right").clip(1, len(peaks)) - 1
    nearest_distance = np.minimum(abs(mid - peaks[after]), abs(mid - peaks[before]))
    orphan = edges[nearest_distance > 0.08 * fs]
    extra = []
    for group in np.split(orphan, np.flatnonzero(np.diff(orphan) > 0.08 * fs) + 1):
        if not len(group):
            continue
        seed = int(group[np.argmax(abs(x[group + 1] - x[group]))])
        if seed < radius or seed + radius >= len(x):
            continue
        anchor = x[seed + offsets]
        center = seed + offsets[np.argmax(abs(anchor - np.median(anchor)))]
        j = np.searchsorted(peaks, center)
        adjacent = peaks[max(0, j - 4) : j + 4]
        if len(adjacent) < 4:
            continue
        period = np.median(np.diff(adjacent))
        distance = np.min(abs(adjacent - center))
        if abs(distance - period) <= 0.20 * period:
            extra.append(center)

    ref_batch = -1
    for center in np.union1d(peaks, np.asarray(extra, dtype=int)):
        lo, hi = np.searchsorted(mid, [center - 0.08 * fs, center + 0.08 * fs])
        if lo == hi or center < radius or center + radius >= len(x):
            continue
        j = np.searchsorted(peaks, center)
        ids = np.arange(max(0, j - 6), min(len(peaks), j + 7))
        distance = abs(peaks[ids] - center)
        ids = ids[broad[ids] & (distance > 0.20 * fs) & (distance <= 10 * fs)]
        if len(ids) < 3:
            continue
        # Bound the expanded reference bank on long continuous recordings.
        if j // 128 != ref_batch:
            ref_batch = j // 128
            first = max(0, ref_batch * 128 - 6)
            last = min(len(peaks), (ref_batch + 1) * 128 + 6)
            bank, spans, slopes, energy = shifted_qrs_references(references[first:last], len(offsets))
        ids = ids - first
        beat = x[center + offsets]
        beat = beat - np.median(beat)
        refs = bank[ids]
        projection = np.einsum("ijk,k->ij", refs, beat)
        ref_energy = np.maximum(energy[ids], np.finfo(float).tiny)
        gain = projection / ref_energy
        residual = beat - refs * gain[:, :, None]
        # Limit amplitude fitting and inspect both the waveform and its fastest raw edge.
        # An impulse on top of a similar QRS must still be visible in the slope residual.
        plausible_gain = (gain >= 0.75) & (gain <= 1.35)
        amplitude_ok = abs(residual) <= (0.23 * spans[ids] * gain)[:, :, None]
        slope_ok = abs(np.diff(residual, axis=2)) <= (0.42 * slopes[ids] * gain)[:, :, None]
        fits = plausible_gain & np.all(amplitude_ok, axis=2) & np.all(slope_ok, axis=2)
        if np.count_nonzero(fits.any(axis=1)) >= 3:
            explained[lo:hi] = True
            continue
        # A difference in the flank should not put an alarm on a matching central upstroke.
        # Require similarity of the whole complex and raw agreement around the actual edge.
        similar = plausible_gain & (projection >= 0.92 * np.sqrt(ref_energy * np.dot(beat, beat)))
        flank = max(1, round(0.016 * fs))
        for k in range(lo, hi):
            edge = edges[k] - center + radius
            left, right = max(0, edge - flank), min(len(beat), edge + 2 + flank)
            local = np.all(amplitude_ok[:, :, left:right], axis=2) & np.all(slope_ok[:, :, left : right - 1], axis=2)
            if np.count_nonzero((similar & local).any(axis=1)) >= 3:
                explained[k] = True
    return explained


def local_step_scale(x, fs):
    """Estimate the background pulse swing from flanks, excluding the tested tile and its two edges."""
    out = np.full(max(0, len(x) - 1), np.nan)
    hop = max(1, min(round(fs / 4), len(x) // 4))
    flank = max(4, round(2 * fs))
    for a in range(0, len(out), hop):
        b = min(a + hop, len(out))
        sides = [x[max(0, a - flank) : a], x[b + 1 : b + 1 + flank]]
        refs = np.concatenate(sides)
        if len(refs) >= 4:
            lo, hi = np.percentile(refs, [5, 95])
            out[a:b] = max(hi - lo, np.finfo(float).eps * max(1.0, abs(lo), abs(hi)))
    return out


def local_qrs_matches(x, fs, edges):
    # Bound both the detector thresholds and the reference beat bank used to explain an edge.
    matched = np.zeros(len(edges), bool)
    width, flank = max(1, round(8 * fs)), round(10 * fs)
    for start in range(0, len(x), width):
        i, j = np.searchsorted(edges, [start, start + width])
        if i < j:
            left = max(0, start - flank)
            matched[i:j] = ecg_repeated_qrs_edges(x[left : start + width + flank], fs, edges[i:j] - left)
    return matched


def transient_cores(x, fs, scale, factor):
    """Complete 1–4 sample excursions with quiet observed flanks, allowing both residual polarities."""
    d = np.abs(np.diff(x))
    edges = np.flatnonzero(d > 0.40 * factor * scale) + 1
    last = 0
    for a in edges:
        if a < max(4, last) or a + 4 >= len(x):
            continue
        for width in range(1, min(4, max(1, round(0.04 * fs))) + 1):
            b = a + width
            if b + 4 > len(x):
                break
            swing = scale[a - 1]
            if d[b - 1] <= 0.4 * factor * swing or abs(x[a - 1] - x[b]) > 0.25 * swing:
                continue
            flank = max(np.abs(np.diff(x[a - 4 : a])).max(), np.abs(np.diff(x[b : b + 4])).max())
            residual = x[a:b] - np.linspace(x[a - 1], x[b], width + 2)[1:-1]
            if flank > 0.18 * swing or np.abs(residual).min() <= 0.2 * factor * swing:
                continue
            bipolar = residual.min() < 0 < residual.max()
            yield (
                int(a),
                int(b),
                "biphasic_core" if bipolar else "impulsive_core",
                float(np.abs(residual).max() / swing),
            )
            last = b + 1
            break


def scan(values, usable, causes, fs, modality, preset):
    factor = PRESETS[preset]["factor"]
    cfg = profile(preset)
    high_band = HIGH_BAND_HZ.get(modality)
    signals = []
    evidence = []
    coverage = defaultdict(int)
    valid_runs = intervals(usable)
    assessed = {
        k: np.zeros(len(values), bool)
        for k in (
            "large_step",
            "flat_run",
            "spectrum",
            "impulsive_core",
            "biphasic_core",
            "ecg_excursion",
            "ecg_baseline_shift",
            "abp_pulse_episode",
            *causes,
        )
    }
    spectral_frames = []
    context_values = defaultdict(list)

    for cause, mask in causes.items():
        assessed[cause][:] = True
        for a, b in intervals(mask):
            signals.append((a, b, cause, b - a, 1, "samples"))
    for a, b in valid_runs:
        x = values[a:b]
        n = len(x)
        assessed["flat_run"][a:b] = n >= max(3, round(fs * 0.5))
        # Constant stretches and short abrupt changes survive averaging in the long views.
        for start, stop in intervals(np.diff(x) == 0):
            length = stop - start + 1
            if length >= max(3, round(fs * 0.5)):
                signals.append((a + start, a + stop + 1, "flat_run", length / fs, 0.5, "s"))
        span = float(np.ptp(np.percentile(x, [5, 95]))) or float(np.ptp(x))
        delta = np.abs(np.diff(x))
        if modality == "ABP" and span > 0:
            for start, stop, value, threshold in abp_pulse_episodes(
                x, fs, factor, evaluated=assessed["abp_pulse_episode"][a:b]
            ):
                signals.append((a + start, a + stop, "abp_pulse_episode", value, threshold, "local_cycle_ratio"))
        scales = local_step_scale(x, fs)
        if n > 8:
            for kind in ("impulsive_core", "biphasic_core"):
                assessed[kind][a + 4 : b - 4] = np.isfinite(scales[3 : n - 5])
        if modality == "ECG" and n > 3 * fs:
            assessed["ecg_excursion"][a + round(1.5 * fs) : b - round(0.32 * fs)] = True
        step_limit = {"ECG": 0.55, "PPG": 0.60, "ABP": 0.65, "RESP": 0.80}[modality] * factor
        assessed["large_step"][a : b - 1] = np.isfinite(scales)
        edges = np.flatnonzero(delta > step_limit * scales)
        if modality == "ECG" and edges.size:
            repeated = local_qrs_matches(x, fs, edges)
            coverage["ecg_step_edges"] += len(edges)
            coverage["ecg_step_edges_matching_repeated_qrs"] += int(repeated.sum())
            edges = edges[~repeated]
        for i in edges:
            signals.append(
                (a + int(i), a + int(i) + 2, "large_step", float(delta[i] / scales[i]), step_limit, "local_swing_ratio")
            )
        for left, right, kind, strength in transient_cores(x, fs, scales, factor):
            signals.append((a + left, a + right, kind, strength, 0.40 * factor, "local_swing_ratio"))
        if span > 0:
            if modality == "ECG":
                for start, stop, value, threshold in ecg_excursions(x, fs, scales, factor):
                    signals.append((a + start, a + stop, "ecg_excursion", value, threshold, "local_span_ratio"))
                for start, stop, value, threshold in ecg_baseline_shifts(
                    x, fs, factor, evaluated=assessed["ecg_baseline_shift"][a:b]
                ):
                    signals.append((a + start, a + stop, "ecg_baseline_shift", value, threshold, "flank_span_ratio"))
        # 2-s short frames, 1-s stride. Power fractions are dimensionless.
        frame = max(2, round(cfg["sentinel"]["window_seconds"] * fs))
        for left, right in windows(n, frame, max(1, round(fs * cfg["sentinel"]["hop_seconds"]))):
            y = x[left:right]
            if len(y) < max(16, round(fs)):
                coverage["spectral_insufficient_samples"] += len(y)
                continue
            fft = np.fft.rfft((y - y.mean()) * np.hanning(len(y)))
            power = np.abs(fft) ** 2
            freq = np.fft.rfftfreq(len(y), 1 / fs)
            denominator = float(power[freq >= 0.5].sum())
            if denominator <= 0:
                continue
            assessed["spectrum"][a + left : a + right] = True
            if high_band is not None and fs / 2 > high_band:
                spectral_frames.append((a + left, a + right, float(power[freq >= high_band].sum() / denominator)))
            for hz in (50, 60):
                if hz + 1 >= fs / 2:
                    continue
                band = np.abs(freq - hz) <= 1
                near = (np.abs(freq - hz) >= 2) & (np.abs(freq - hz) <= 6)
                if not near.any() or not band.any():
                    continue
                v = float(power[band].sum() / denominator)
                contrast = power[band].max() / max(float(np.median(power[near])), 1e-30)
                if v >= LINE_FRACTION * factor and contrast >= 5:
                    signals.append((a + left, a + right, f"line_{hz}", v, LINE_FRACTION * factor, "power_fraction"))
            if high_band is not None and fs / 2 > high_band:
                v = float(power[freq >= HIGH_BAND_HZ[modality]].sum() / denominator)
                threshold = HIGH_FRACTION[modality] * factor
                if v >= threshold:
                    signals.append((a + left, a + right, "high_frequency", v, threshold, "power_fraction"))
        # Compute context summaries only from this support; coarse windows never gate short-frame findings.
        for scale in cfg["multiscale"]["windows_seconds"]:
            size = max(1, round(scale * fs))
            for left, right in windows(n, size, max(1, round(size * cfg["multiscale"]["hop_ratio"]))):
                if right - left < size:
                    coverage[f"{scale}s_short_runs"] += 1
                    continue
                segment = x[left:right]
                coverage[f"{scale}s_windows"] += 1
                if span > 0:
                    # A slow trend can be physiologic; retained as evidence only.
                    edge = max(1, len(segment) // 5)
                    drift = float((segment[-edge:].mean() - segment[:edge].mean()) / span)
                    context_values[scale].append((a + left, a + right, float(abs(drift))))
                    if abs(drift) >= 0.5:
                        evidence.append(
                            {"range": [a + left, a + right], "window_s": scale, "level_change_over_span": drift}
                        )

    # Merge intervals within a single evidence family only; retain the strongest
    # observation and the count. Overlapping families remain separately inspectable.
    groups = defaultdict(list)
    for start, end, kind, value, threshold, unit in signals:
        groups[kind].append((start, end, value, threshold, unit))
    events = []
    for kind, hits in groups.items():
        previous = None
        for start, end, value, threshold, unit in sorted(hits):
            if previous and start < previous["end"]:
                previous["end"] = max(end, previous["end"])
                previous["value"] = max(value, previous["value"])
                previous["trigger_count"] += 1
                previous["observations"].append([start, end, float(value), float(threshold)])
            else:
                previous = dict(
                    start=start,
                    end=end,
                    kind=kind,
                    value=value,
                    threshold=threshold,
                    unit=unit,
                    trigger_count=1,
                    observations=[[start, end, float(value), float(threshold)]],
                )
                events.append(previous)
    coverage["tool_results"] = {}
    for tool, mask in assessed.items():
        alerts = [
            [e["start"], e["end"]]
            for e in events
            if e["kind"] == tool or (tool == "spectrum" and e["kind"] in ("line_50", "line_60", "high_frequency"))
        ]
        coverage["tool_results"][tool] = dict(
            status=("suspected_artifact" if alerts else "evaluated_no_alert") if mask.any() else "insufficient_support",
            evaluated_ranges=intervals(mask),
            unsupported_ranges=intervals(~mask),
            alert_ranges=alerts,
        )
    for kind, applicable in (("ecg_excursion", "ECG"), ("ecg_baseline_shift", "ECG"), ("abp_pulse_episode", "ABP")):
        if modality != applicable:
            coverage["tool_results"][kind]["status"] = "not_applicable"
    for hz in (50, 60):
        k = f"line_{hz}"
        coverage["tool_results"][k] = dict(
            coverage["tool_results"]["spectrum"],
            alert_ranges=[[e["start"], e["end"]] for e in events if e["kind"] == k],
        )
        row = coverage["tool_results"][k]
        if hz + 1 >= fs / 2:
            row.update(status="not_applicable", evaluated_ranges=[], unsupported_ranges=[[0, len(values)]])
        else:
            row["status"] = (
                "suspected_artifact"
                if row["alert_ranges"]
                else "evaluated_no_alert"
                if row["evaluated_ranges"]
                else "insufficient_support"
            )
    coverage["multiscale"] = {}
    for scale in cfg["multiscale"]["windows_seconds"]:
        records = context_values[scale]
        support = np.zeros(len(values), bool)
        bad = np.zeros(len(values), bool)
        for left, right, score in records:
            support[left:right] = True
            if score >= 0.5:
                bad[left:right] = True
        scores = [row[2] for row in records]
        coverage["multiscale"][str(scale)] = dict(
            window_seconds=scale,
            hop_seconds=scale * cfg["multiscale"]["hop_ratio"],
            metric="absolute_level_change_over_swing",
            evaluated_ranges=intervals(support),
            unsupported_ranges=intervals(~support),
            status="suspected_artifact" if bad.any() else "evaluated_no_alert" if scores else "insufficient_support",
            median=float(np.median(scores)) if scores else None,
            mean=float(np.mean(scores)) if scores else None,
            low_quantile=float(np.percentile(scores, 5)) if scores else None,
            worst=max(scores, default=None),
            support_samples=int(support.sum()),
            bad_time_fraction=float(bad.sum() / support.sum()) if scores else None,
            longest_bad_seconds=max((b - a for a, b in intervals(bad)), default=0) / fs,
            event_count=len(intervals(bad)),
        )
    coverage["tool_results"]["high_frequency"] = dict(
        coverage["tool_results"]["spectrum"],
        alert_ranges=[[e["start"], e["end"]] for e in events if e["kind"] == "high_frequency"],
    )
    high = coverage["tool_results"]["high_frequency"]
    if high_band is None or fs / 2 <= high_band:
        high.update(status="not_applicable", evaluated_ranges=[], unsupported_ranges=[[0, len(values)]])
    else:
        high["status"] = (
            "suspected_artifact"
            if high["alert_ranges"]
            else "evaluated_no_alert"
            if high["evaluated_ranges"]
            else "insufficient_support"
        )
    if spectral_frames:
        powers = np.array([f[2] for f in spectral_frames])
        bad = np.zeros(len(values), bool)
        for left, r, v in spectral_frames:
            if v >= HIGH_FRACTION[modality] * factor:
                bad[left:r] = True
        coverage["high_frequency_summary"] = dict(
            mean=float(powers.mean()),
            median=float(np.median(powers)),
            low_quantile=float(np.percentile(powers, 5)),
            worst=float(powers.max()),
            event_count=len(intervals(bad)),
            bad_time_fraction=float(bad.sum() / assessed["spectrum"].sum()),
            longest_bad_seconds=max((r - left for left, r in intervals(bad)), default=0) / fs,
            support_samples=int(assessed["spectrum"].sum()),
        )
    return sorted(events, key=lambda e: (e["start"], e["end"], e["kind"])), dict(coverage), evidence
