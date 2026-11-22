"""Engineering-corrected period-grid locator from the supplied time-sphere draft.

The four heuristic feature formulae and thresholds are deliberately retained.
In particular, a flat region without ridge loss can still receive the original
flat score of 4, and ordinary sharp waveforms can exceed the amplitude rule.
Those are method limitations to measure, not implementation tests to tune away.
Spherical coordinates are a display mapping and do not enter detection.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate

THRESHOLDS = {"strict": 2.4, "balanced": 3.2, "loose": 4.2}
PHASE_BINS = 128
R_BASE = 1.0
RADIAL_SCALE = 0.25
PERIOD_RANGE_BPM = (42.0, 165.0)
FEATURES = (
    ("suspected_cross_cycle_irregularity", "cross_cycle_roughness"),
    ("suspected_radial_cubic_shift", "radial_cubic_deviation"),
    ("suspected_low_activity", "flatness_and_ridge_loss"),
    ("suspected_extreme_amplitude", "radial_envelope_exceedance"),
)


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _acf_period(segment, fs, min_lag, max_lag, *, method="fft"):
    """Same centering, normalization, search interval and fallback as the draft."""
    centered = np.asarray(segment, dtype=np.float64) - np.mean(segment)
    if np.std(centered) < 1e-6 or len(centered) <= max_lag:
        return fs * 0.8
    if method == "direct":
        acf = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    elif method == "fft":
        acf = correlate(centered, centered, mode="full", method="fft")[len(centered) - 1 :]
    else:
        raise ValueError("Unknown autocorrelation implementation")
    acf = acf / (acf[0] + 1e-8)
    search = acf[min_lag:max_lag]
    return float(min_lag + np.argmax(search)) if search.size else fs * 0.8


def _adaptive_phase(values, fs, *, method="fft"):
    """Preserve the draft's 6 s / 3 s / 2 s adaptive-period calculation."""
    n = len(values)
    min_lag = max(2, int(fs * 60.0 / PERIOD_RANGE_BPM[1]))
    max_lag = min(n // 2, int(fs * 60.0 / PERIOD_RANGE_BPM[0]))
    global_period = _acf_period(values, fs, min_lag, max_lag, method=method)
    win, hop = max(1, int(fs * 6.0)), max(1, int(fs * 3.0))
    centers, periods = [], []
    if n > win * 1.5:
        for start in range(0, n - win + 1, hop):
            period = _acf_period(values[start : start + win], fs, min_lag, max_lag, method=method)
            if 0.65 * global_period <= period <= 1.45 * global_period:
                centers.append(start + win / 2.0)
                periods.append(period)
    if len(centers) >= 2:
        inst_period = np.interp(np.arange(n), centers, periods)
        inst_period = uniform_filter1d(inst_period, size=max(1, int(fs * 2.0)))
    else:
        inst_period = np.full(n, global_period)
    phase = np.cumsum(1.0 / np.maximum(inst_period, 1.0))
    phase -= phase[0]
    return phase, global_period


def _mad_normalize(values):
    """Original positive MAD score, using only actually supported grid cells."""
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if finite.any():
        observed = values[finite]
        median = np.median(observed)
        mad = np.median(np.abs(observed - median)) + 1e-8
        result[finite] = np.maximum(0.0, (observed - median) / (1.4826 * mad))
    return result


def _feature_scores(grid_z, valid):
    """Four unchanged heuristics; missing temporal context produces NaN, not 0."""
    radius = R_BASE + RADIAL_SCALE * grid_z
    k, m = radius.shape
    d2 = np.full_like(radius, np.nan)
    if k >= 2:
        d2[0] = np.abs(radius[1] - radius[0])
        d2[-1] = np.abs(radius[-1] - radius[-2])
    if k >= 3:
        d2[1:-1] = np.abs(radius[2:] - 2.0 * radius[1:-1] + radius[:-2])
    meridian_median = np.nanmedian(radius, axis=0, keepdims=True)
    cross = 0.6 * d2 + 0.4 * np.abs(radius - meridian_median)

    # Only longitude wraps. There is no artificial connection between the first
    # and last temporal cycles. A partial final cycle remains partially missing.
    cubic = radius**3 - R_BASE**3
    padded = np.pad(cubic, ((1, 1), (0, 0)), mode="constant", constant_values=np.nan)
    padded = np.pad(padded, ((0, 0), (8, 8)), mode="wrap")
    volume_proxy = np.zeros_like(radius)
    for dk in range(3):
        for dm in range(17):
            volume_proxy += padded[dk : dk + k, dm : dm + m]
    volume_proxy = np.abs(volume_proxy / 51.0)

    d1 = np.abs(np.diff(radius, axis=1, prepend=radius[:, -1:]))
    finite_d1 = np.isfinite(d1)
    width = max(4, m // 8)
    rough = uniform_filter1d(np.where(finite_d1, d1, 0.0), size=width, axis=1, mode="wrap")
    context = uniform_filter1d(finite_d1.astype(float), size=width, axis=1, mode="wrap")
    rough[context < 1.0 - 1e-12] = np.nan
    ridge_height = float(np.percentile(meridian_median - R_BASE, 92))
    full_cycle = valid.all(axis=1, keepdims=True)
    peak_height = np.max(np.where(valid, radius - R_BASE, -np.inf), axis=1, keepdims=True)
    ridge_loss = np.maximum(0.0, 1.0 - peak_height / (ridge_height + 1e-6))
    flat = np.where(rough < 0.004 * RADIAL_SCALE, 1.0 + 3.0 * ridge_loss, 0.0)
    flat = np.where(full_cycle & np.isfinite(rough), flat * 4.0, np.nan)
    spike = np.where(valid, np.maximum(0.0, np.abs(grid_z) - 4.5) * 2.5, np.nan)
    return radius, np.stack([_mad_normalize(cross), 0.85 * _mad_normalize(volume_proxy), flat, spike])


def _project_scores(component_grid, phase, positions, valid):
    """Piecewise linear score interpolation within observed phase cells.

    The final observed cell ends at the native record boundary. Holding its
    score within that partial cell does not invent a waveform or another cycle.
    An unsupported cell is never bridged to a later supported cell.
    """
    flat = component_grid.reshape(4, -1)
    grid_position = phase * PHASE_BINS
    lower = np.minimum(np.floor(grid_position).astype(int), flat.shape[1] - 1)
    upper = np.minimum(lower + 1, flat.shape[1] - 1)
    fraction = np.clip(grid_position - lower, 0, 1)
    before, after = flat[:, lower], flat[:, upper]
    scores = before.copy()
    interpolate = np.isfinite(before) & np.isfinite(after)
    scores[interpolate] = (before + (after - before) * fraction[None, :])[interpolate]

    observed_positions = positions.ravel()[valid.ravel()]
    widths = np.diff(np.r_[observed_positions, len(phase)])
    # The source sample grid is the maximum possible native time precision.
    widths = np.maximum(1, np.ceil(widths)).astype(int)
    resolution = widths[np.minimum(lower, len(widths) - 1)]
    return scores, resolution


def _postprocess(mask, evaluated, gap_samples, minimum_samples):
    result = mask.copy()
    for start, end in _runs(~result):
        if start > 0 and end < len(result) and end - start <= gap_samples and evaluated[start:end].all():
            result[start:end] = True
    for start, end in _runs(result):
        if end - start < minimum_samples:
            result[start:end] = False
    return result


def locate_sphere(values, fs, *, sensitivity="balanced", include_mesh=False):
    """Locate suspicious intervals in one finite continuous channel, without edits.

    ``evaluated_mask`` means at least one feature was available. Per-feature
    coverage is reported separately; a zero score is not a verified clean label.
    The caller must split source-invalid/nonfinite regions and genuine time gaps.
    """
    x = np.asarray(values, dtype=np.float64)
    fs = float(fs)
    if x.ndim != 1:
        raise ValueError("Time sphere requires a one-dimensional single channel")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if not np.isfinite(x).all():
        raise ValueError("Split nonfinite/source-invalid intervals before time-sphere analysis")
    if sensitivity not in THRESHOLDS:
        raise ValueError("Unknown time-sphere sensitivity")
    n = len(x)
    threshold = THRESHOLDS[sensitivity]
    minimum_samples = max(2, int(fs * 0.08))
    gap_samples = max(1, int(fs * 0.35))
    parameters = {
        "algorithm": "time_sphere_engineering_v0.1",
        "fs": fs,
        "sensitivity": sensitivity,
        "threshold": threshold,
        "phase_bins": PHASE_BINS,
        "r_base": R_BASE,
        "radial_scale": RADIAL_SCALE,
        "period_range_bpm": list(PERIOD_RANGE_BPM),
        "acf": "FFT, original centered biased normalization and exclusive lag upper bound",
        "phase_window_s": 6.0,
        "phase_hop_s": 3.0,
        "period_smoothing_s": 2.0,
        "global_period_limits": [0.65, 1.45],
        "maximum_fill_gap_samples": gap_samples,
        "minimum_event_samples": minimum_samples,
        "formula_changes": False,
        "scores_are_probabilities": False,
    }
    if n < max(2, int(fs * 3)):
        return {
            "status": "unscored",
            "events": [],
            "evaluated_mask": np.zeros(n, bool),
            "scores": np.full(n, np.nan),
            "anomaly_mask": np.zeros(n, bool),
            "coverage_notes": ["Fewer than the draft's minimum three seconds; no synthetic cycles were created."],
            "component_evaluated_samples": {name: 0 for _, name in FEATURES},
            "component_unscored_ranges": {name: [[0, n]] if n else [] for _, name in FEATURES},
            "parameters": parameters,
            **({"mesh": None} if include_mesh else {}),
        }

    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    scale = float(q75 - q25) if q75 - q25 > 1e-8 else float(np.std(x) + 1e-8)
    z = (x - q50) / scale
    phase, global_period = _adaptive_phase(z, fs)
    cycles = max(1, int(np.floor(phase[-1])) + 1)
    grid_k, grid_m = np.meshgrid(np.arange(cycles), np.arange(PHASE_BINS), indexing="ij")
    grid_phase = grid_k + grid_m / float(PHASE_BINS)
    valid = grid_phase <= phase[-1]
    grid_z, positions = np.full(grid_phase.shape, np.nan), np.full(grid_phase.shape, np.nan)
    grid_z[valid] = np.interp(grid_phase[valid], phase, z)
    positions[valid] = np.interp(grid_phase[valid], phase, np.arange(n))
    radius, component_grid = _feature_scores(grid_z, valid)
    components, resolution = _project_scores(component_grid, phase, positions, valid)
    finite = np.isfinite(components)
    evaluated = finite.any(axis=0)
    usable_scores = np.where(finite, components, -np.inf)
    total = np.where(evaluated, usable_scores.max(axis=0), np.nan)
    dominant = usable_scores.argmax(axis=0)
    raw_mask = evaluated & (total > threshold)
    mask = _postprocess(raw_mask, evaluated, gap_samples, minimum_samples)
    events = []
    for start, end in _runs(mask):
        votes = dominant[start:end][raw_mask[start:end]]
        feature_index = int(np.bincount(votes, minlength=4).argmax())
        maximum = float(np.nanmax(total[start:end]))
        kind, metric = FEATURES[feature_index]
        events.append(
            {
                "start": int(start),
                "end": int(end),
                "kind": kind,
                "score": maximum,
                "metric": metric,
                "resolution_samples": int(resolution[start:end].max()),
                "evidence": {
                    "dominant_feature": metric,
                    "threshold": threshold,
                    "max_feature_score": maximum,
                    "heuristic_severity": round(min(1.0, maximum / (threshold * 2.2)), 3),
                    "interpretation": "Candidate from a retained heuristic; neither noise truth nor a clinical cause.",
                },
            }
        )

    parameters.update(
        normalization={"median": float(q50), "scale": scale, "rule": "IQR, or std + 1e-8 for tiny IQR"},
        global_period_samples=float(global_period),
        latitude_cycles=cycles,
        coverage_definition="At least one feature available; per-feature counts retained separately",
    )
    result = {
        "status": "evaluated" if evaluated.all() else "partial",
        "events": events,
        "evaluated_mask": evaluated,
        "scores": total,
        "anomaly_mask": mask,
        "component_evaluated_samples": {name: int(finite[i].sum()) for i, (_, name) in enumerate(FEATURES)},
        "component_unscored_ranges": {
            name: [[int(a), int(b)] for a, b in _runs(~finite[i])] for i, (_, name) in enumerate(FEATURES)
        },
        "coverage_notes": [
            "No waveform samples were modified; geometry uses interpolated phase-grid observations.",
            "Missing temporal neighbors and incomplete-cycle contexts are unscored for the affected feature.",
            "The final partial phase cell is mapped only onto its existing native time support.",
            "All four draft formulae and weights are retained, including known normal-waveform false positives.",
            "The 128-bin phase grid can miss native narrow spikes; no raw-rate compensation was added.",
            "Spherical display does not participate in scoring; radial cubic deviation is not physical volume.",
        ],
        "parameters": parameters,
    }
    if include_mesh:
        minimum_radius = float(radius[valid].min())
        positive_margin = max(1e-6, float(np.spacing(abs(minimum_radius))))
        offset = max(0.0, positive_margin - minimum_radius)
        display_radius = radius + offset
        mesh_scores = np.max(np.where(np.isfinite(component_grid), component_grid, -np.inf), axis=0)
        mesh_scores[~valid] = np.nan
        result["mesh"] = {
            "fs": fs,
            "threshold": threshold,
            "phase_bins": PHASE_BINS,
            "normalization": parameters["normalization"],
            "phase": grid_phase,
            "phase_unit": "cumulative_cycles",
            "cycle_index": grid_k,
            "phase_fraction": grid_m / float(PHASE_BINS),
            "theta": np.linspace(0.12, np.pi - 0.12, cycles)[:, None],
            "phi": np.linspace(0, 2 * np.pi, PHASE_BINS, endpoint=False)[None, :],
            "angle_unit": "radians",
            "valid_cell": valid,
            "sample_positions": positions,
            "z": grid_z,
            "radial_coordinate": radius,
            "display_radius": display_radius,
            "display_radius_offset": offset,
            "scores": mesh_scores,
            "component_scores": component_grid,
            "component_available": np.isfinite(component_grid),
        }
    return result
