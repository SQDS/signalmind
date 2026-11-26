"""Public observed evidence for the registered short-core candidate comparison."""

import numpy as np

from tsmcp.processing import PCHIP_ANCHORS

from .spike_evidence import phase_neighbours


def comparison_context(case, before, a, b, other_cores, policy):
    t, x, valid, estimated, pr = before
    fs = case["fs"]
    good = valid & ~estimated & np.isfinite(x) & np.isfinite(t)
    broken = np.abs(np.diff(t) - 1 / fs) > np.maximum(1e-6, 32 * np.spacing(np.abs(t[1:])))
    good[1:] &= ~broken
    for start, stop in case.get("source_time_conflicts", []):
        good[start:stop] = False
    for start, stop in other_cores:
        if (start, stop) != (a, b):
            good[start:stop] = False
    left, right = a - PCHIP_ANCHORS, b + PCHIP_ANCHORS
    evidence = dict(anchor_ranges=[[left, a], [b, right]])
    if left < 0 or right > len(x) or not good[left:right].all():
        return dict(evidence, reason="unsupported_or_overlapping_flanks"), None
    radius = round(fs * policy["flank_radius_s"])
    differences = []
    for lo, hi in [(max(0, a - radius), a), (b, min(len(x), b + radius))]:
        # A break ends this flank. Samples beyond it cannot set the local noise scale.
        gaps = np.flatnonzero(~good[lo:hi]) + lo
        if len(gaps):
            if hi == a:
                lo = gaps[-1] + 1
            else:
                hi = gaps[0]
        differences.extend(np.abs(np.diff(x[lo:hi])))
    scale = max(np.quantile(differences, policy["step_quantile"]), np.spacing(max(1.0, np.max(abs(x[left:right])))))
    limit = float(scale * policy["flank_multiplier"])
    baseline = np.linspace(x[a - 1], x[b], b - a + 2)[1:-1]
    excursion = float(np.max(np.abs(x[a:b] - baseline)))
    edge = max(abs(np.diff(x[left:a], n=2)[0]), abs(np.diff(x[b:right], n=2)[0]))
    evidence.update(flank_limit=limit, anchor_curvature=float(edge), core_excursion=excursion)
    if edge > limit:
        return dict(evidence, reason="suspected_contamination_outside_core"), None
    if excursion <= limit:
        return dict(evidence, reason="no_compact_excursion_above_flank_variation"), None
    indices, refs, cycles, why = phase_neighbours(x, good, pr["events"], fs, a, b)
    evidence.update(phase_cycles=cycles, phase_unavailable_reason=why)
    if why:
        return evidence, None
    ref = np.median(refs[:, (indices >= a) & (indices < b)], axis=0)
    return evidence, ref


def phase_error(core, reference):
    return float(np.mean(np.abs(core - reference))) if reference is not None else None
