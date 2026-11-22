"""Local second-pass review of a saved batch result, using original observations only."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from tsmcp.peak_policy import observe, outside_rr

VERSION = "signalmind.peak-refinement.v0.1"


@dataclass(frozen=True)
class RefinementConfig:
    support_energy_fraction: float = 0.5
    detector_distance_ms: float = 120
    context_ms: float = 120


def align_candidates(ecg, original, mode, radius):
    """Anchor every search to its original marker to avoid iterative coordinate drift."""
    aligned = original.copy()
    if mode == "none":
        return aligned
    for i, point in enumerate(original):
        a, b = max(0, point - radius), min(len(ecg), point + radius + 1)
        local = ecg[a:b]
        if not np.isfinite(local).any():
            continue
        if mode == "positive":
            offset = np.nanargmax(local)
        elif mode == "negative":
            offset = np.nanargmin(local)
        elif mode == "absolute":
            offset = np.nanargmax(np.abs(local - np.nanmedian(local)))
        else:
            raise ValueError("Unknown alignment mode")
        aligned[i] = a + int(offset)
    return aligned


def refine(ecg, original_peaks, batch_result, batch_policy, fs=125, config=RefinementConfig()):
    """Reconcile edits and emit a separate review queue; the queue is not an exclusion verdict.

    Deletion needs both weak aligned QRS-band evidence and no nearby detector candidate.
    Template correlation alone never deletes a marker. Uncertain deleted candidates are
    restored as candidates and flagged, without claiming that they are confirmed beats.
    """
    x, original = np.asarray(ecg, float), np.asarray(original_peaks, int)
    before = np.asarray(batch_result["new_peak_samples"], int)
    if before.shape != original.shape or batch_result["old_peak_samples"] != original.tolist():
        raise ValueError("The batch result must use the same original marker identities")
    mode = batch_policy.get("alignment", "none")
    radius = round(batch_policy.get("radius_ms", 80) * fs / 1000)
    aligned = align_candidates(x, original, mode, radius)
    evidence = observe(x, aligned, fs)
    energy, distance = evidence["energy_fraction"], evidence["detector_distance_ms"]
    context = round(config.context_ms * fs / 1000)
    complete = np.array(
        [
            p >= context and p + context < len(x) and np.isfinite(x[p - context : p + context + 1]).all()
            for p in aligned
        ],
        bool,
    )
    supported = complete & (energy >= config.support_energy_fraction) & (distance <= config.detector_distance_ms)
    weak_unmatched = complete & (energy < config.support_energy_fraction) & (distance > config.detector_distance_ms)
    uncertain = ~(supported | weak_unmatched)
    after = before.copy()
    reasons = {}
    queue = []
    for i in range(len(original)):
        if weak_unmatched[i]:
            after[i] = -1
            reasons[i] = "weak_aligned_qrs_and_detector_disagreement"
        elif supported[i]:
            after[i] = aligned[i]
            reasons[i] = "restore_supported_candidate" if before[i] < 0 else "align_supported_candidate"
        else:
            if before[i] < 0:
                after[i] = original[i]
                reasons[i] = "restore_unresolved_candidate"
            queue.append(
                {"marker_index0": i, "reason": "incomplete_context" if not complete[i] else "evidence_disagreement"}
            )
    # One final coordinate cannot denote two different markers. Keep the strongest
    # candidate, breaking equal-evidence ties by distance to the original coordinate.
    occupied = {}
    for i in sorted(
        np.flatnonzero(after >= 0),
        key=lambda j: (-int(supported[j]), -energy[j], abs(int(after[j] - original[j])), int(j)),
    ):
        point = int(after[i])
        if point in occupied:
            after[i] = -1
            reasons[i] = "same_coordinate_duplicate"
        else:
            occupied[point] = int(i)
    # Distinct nearby candidates need further review; closeness alone does not delete them.
    ids = sorted(np.flatnonzero(after >= 0), key=lambda i: after[i])
    refractory = round(batch_policy.get("refractory_ms", 0) * fs / 1000)
    nearby = set()
    for a, b in zip(ids[:-1], ids[1:], strict=True):
        if after[b] - after[a] < refractory:
            nearby.update((int(a), int(b)))
    queue.extend({"marker_index0": i, "reason": "nearby_distinct_candidates"} for i in sorted(nearby))
    changes = []
    for i in np.flatnonzero(before != after):
        changes.append(
            {
                "marker_index0": int(i),
                "old_R_id": f"R{i + 1:03d}",
                "batch_sample0": int(before[i]),
                "refined_sample0": int(after[i]),
                "operation": "restore" if before[i] < 0 else "delete" if after[i] < 0 else "move",
                "reason": reasons[i],
                "aligned_energy_fraction": float(energy[i]),
                "detector_distance_ms": float(distance[i]),
            }
        )
    kept = after[after >= 0]
    return {
        **batch_result,
        "new_peak_samples": after.tolist(),
        "deleted_count": int((after < 0).sum()),
        "moved_count": int(((after >= 0) & (after != original)).sum()),
        "features": {
            **batch_result["features"],
            "removed_fraction": float(np.mean(after < 0)) if len(after) else 0,
            "kept_peak_count": len(kept),
            "kept_rr_outside_fraction": outside_rr(kept, fs),
        },
        "refinement_version": VERSION,
        "refinement_config": asdict(config),
        "refinement_operations": changes,
        "review_required": bool(queue),
        "review_queue": queue,
        "evidence_summary": {
            "supported": int(supported.sum()),
            "weak_unmatched": int(weak_unmatched.sum()),
            "uncertain": int(uncertain.sum()),
        },
        "window_decision_revised": False,
        "waveform_values_modified": False,
    }
