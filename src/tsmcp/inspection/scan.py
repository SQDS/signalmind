"""Candidate discovery on source-supported samples. All outputs use block-local half-open indices."""

from collections import defaultdict

import numpy as np

from tsmcp.plugins.ecg.rpeaks import detect_r_peaks
from tsmcp.plugins.pressure.pulses import detect_pulses_unguided
from tsmcp.processing import intervals

# Engineering starting points, fixed before evaluating the 80-case development set.
PRESETS = {
    "quick": {"sensitivity": 0.25, "factor": 1.5},
    "balanced": {"sensitivity": 0.5, "factor": 1.0},
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


def ecg_excursions(x, fs, span, factor):
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
    edge_threshold = min(span, 8 * np.percentile(changes, 90))
    for seed in np.flatnonzero(changes > edge_threshold) + 1:
        if seed < next_seed or seed + minimum + recovery > len(x):
            continue
        anchor = x[max(seed - history, last_return) : seed]
        center = float(np.median(anchor))
        width = max(float(np.ptp(np.percentile(anchor, [5, 95]))), span * 0.05)
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


def scan(values, usable, causes, fs, modality, preset):
    factor = PRESETS[preset]["factor"]
    signals = []
    evidence = []
    coverage = defaultdict(int)
    valid_runs = intervals(usable)
    for cause, mask in causes.items():
        for a, b in intervals(mask):
            signals.append((a, b, cause, b - a, 1, "samples"))
    for a, b in valid_runs:
        x = values[a:b]
        n = len(x)
        # Constant stretches and short abrupt changes survive averaging in the long views.
        for start, stop in intervals(np.diff(x) == 0):
            length = stop - start + 1
            if length >= max(3, round(fs * 0.5)):
                signals.append((a + start, a + stop + 1, "flat_run", length / fs, 0.5, "s"))
        span = float(np.ptp(np.percentile(x, [5, 95]))) or float(np.ptp(x))
        delta = np.abs(np.diff(x))
        if modality == "ABP" and span > 0:
            for start, stop, value, threshold in abp_pulse_episodes(x, fs, factor):
                signals.append((a + start, a + stop, "abp_pulse_episode", value, threshold, "local_cycle_ratio"))
        if span > 0:
            edges = np.flatnonzero(delta > 0.55 * span * factor)
            if modality == "ECG" and edges.size:
                repeated = ecg_repeated_qrs_edges(x, fs, edges)
                coverage["ecg_step_edges"] += len(edges)
                coverage["ecg_step_edges_matching_repeated_qrs"] += int(repeated.sum())
                edges = edges[~repeated]
            for i in edges:
                signals.append(
                    (a + int(i), a + int(i) + 2, "large_step", float(delta[i] / span), 0.55 * factor, "ratio")
                )
            if modality == "ECG":
                for start, stop, value, threshold in ecg_excursions(x, fs, span, factor):
                    signals.append((a + start, a + stop, "ecg_excursion", value, threshold, "local_span_ratio"))
                for start, stop, value, threshold in ecg_baseline_shifts(x, fs, factor):
                    signals.append((a + start, a + stop, "ecg_baseline_shift", value, threshold, "flank_span_ratio"))
        # 2-s short frames, 1-s stride. Power fractions are dimensionless.
        frame = max(2, round(2 * fs))
        for left, right in windows(n, frame, max(1, round(fs))):
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
            if fs / 2 > HIGH_BAND_HZ[modality]:
                v = float(power[freq >= HIGH_BAND_HZ[modality]].sum() / denominator)
                threshold = HIGH_FRACTION[modality] * factor
                if v >= threshold:
                    signals.append((a + left, a + right, "high_frequency", v, threshold, "power_fraction"))
        # Compute context summaries only from this support; coarse windows never gate short-frame findings.
        for scale in SCALES:
            size = max(1, round(scale * fs))
            for left, right in windows(n, size, max(1, size // 2)):
                if right - left < size:
                    coverage[f"{scale}s_short_runs"] += 1
                    continue
                segment = x[left:right]
                coverage[f"{scale}s_windows"] += 1
                if span > 0:
                    # A slow trend can be physiologic; retained as evidence only.
                    edge = max(1, len(segment) // 5)
                    drift = float((segment[-edge:].mean() - segment[:edge].mean()) / span)
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
            else:
                previous = dict(
                    start=start, end=end, kind=kind, value=value, threshold=threshold, unit=unit, trigger_count=1
                )
                events.append(previous)
    return sorted(events, key=lambda e: (e["start"], e["end"], e["kind"])), dict(coverage), evidence
