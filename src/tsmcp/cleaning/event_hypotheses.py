"""Evidence for a protected ECG event. No event or waveform writes are available here."""

import numpy as np

from .dataset_evidence import acquisition
from .quality_gate import ACCEPTANCE


def protected_event(case, events, a, b):
    if case["task"] != "ECG":
        return None
    margin = round(ACCEPTANCE["protected_ecg_s"] * case["fs"])
    near = (z["sample"] for z in events if a - margin <= z["sample"] < b + margin)
    return min(near, key=lambda q: abs(q - (a + b - 1) / 2), default=None)


def observed_window(case, data, a, b):
    t, x, valid, est, _ = data
    fs = case["fs"]
    lo, hi = max(0, a - round(8 * fs)), min(len(t), b + round(8 * fs))
    ok = valid[lo:hi] & ~est[lo:hi] & np.isfinite(x[lo:hi]) & np.isfinite(t[lo:hi])
    ok[1:] &= np.abs(np.diff(t[lo:hi]) - 1 / fs) <= 1e-6
    for start, stop in case.get("source_time_conflicts", []):
        if start < hi and lo < stop:
            ok[max(lo, start) - lo : min(hi, stop) - lo] = False
    if not ok[a - lo : b - lo].all():
        return a, a
    left = np.flatnonzero(~ok[: a - lo])
    right = np.flatnonzero(~ok[b - lo :])
    return (lo + int(left[-1]) + 1 if len(left) else lo, b + int(right[0]) if len(right) else hi)


def pulse_evidence(case, peer, data, at, times):
    shared = case.get("clock_id") and case.get("clock_id") == peer.get("clock_id")
    if case.get("clock_id") or peer.get("clock_id"):
        same_clock = bool(shared)
    else:
        same_clock = bool(
            case.get("source_sha256")
            and case.get("source_sha256") == peer.get("source_sha256")
            and case.get("rows")
            and acquisition(case) == acquisition(peer)
        )
    if not (same_clock and case.get("has_source_time") and peer.get("has_source_time")):
        return dict(status="unavailable", reason="shared_clock_unverified")
    t, _, _, _, pr = data
    i = int(np.searchsorted(t, at))
    if i >= len(t) or abs(t[i] - at) > 1 / peer["fs"]:
        return dict(status="unavailable", reason="outside_peer_acquisition")
    lo, hi = observed_window(peer, data, i, i + 1)
    pulses = np.asarray([t[z["sample"]] for z in pr["events"] if lo <= z["sample"] < hi])
    offsets = []
    target = None
    for left, right in zip(times[:-1], times[1:], strict=True):
        if lo == hi or left < t[lo] or right > t[hi - 1]:
            continue
        # Require an unambiguous native pulse inside this observed cardiac cycle.
        cycle = pulses[(pulses > left) & (pulses < right)]
        if len(cycle) != 1:
            continue
        lag = float(cycle[0] - left)
        if left == at:
            target = lag
        elif right != at:
            offsets.append(lag)
    report = dict(neighbour_cycles=len(offsets), target_lag_s=target)
    if len(offsets) < 4:
        return dict(report, status="unavailable", reason="insufficient_neighbour_lags")
    lag = float(np.median(offsets))
    mad = float(np.median(np.abs(np.asarray(offsets) - lag)))
    report.update(neighbour_median_lag_s=lag, neighbour_mad_s=mad)
    if mad > ACCEPTANCE["protected_ecg_s"]:
        return dict(report, status="unavailable", reason="unstable_neighbour_lags")
    if target is None:
        return dict(report, status="indeterminate", reason="no_unique_target_pulse")
    tolerance = max(2 / peer["fs"], 3 * mad)
    return dict(report, status="consistent" if abs(target - lag) <= tolerance else "discordant", tolerance_s=tolerance)


def conflict_evidence(case, before, a, b, q, peers):
    """Describe competing explanations on the current observed revision, without assigning a truth label."""
    t, x, _, _, pr = before
    fs = case["fs"]
    lo, hi = observed_window(case, before, min(a, q), max(b, q + 1))
    marks = [z["sample"] for z in pr["events"] if lo <= z["sample"] < hi]
    radius = max(2, round(0.08 * fs))
    neighbours = [
        p for p in marks if p != q and lo <= p - radius and p + radius < hi and (p + radius < a or p - radius >= b)
    ]
    shape = dict(neighbour_samples=neighbours, core_samples=[a, b])
    if lo <= q - radius and q + radius < hi:
        y = x[q - radius : q + radius + 1]
        shape.update(raw_window=[q - radius, q + radius + 1], raw_values=y.tolist(), peak_to_peak=float(np.ptp(y)))
        if len(neighbours) >= 3:
            beats = np.array([x[p - radius : p + radius + 1] for p in neighbours])
            shape["neighbour_median_swing"] = float(np.median(np.ptp(beats, axis=1)))
            template = np.median(beats - beats.mean(axis=1, keepdims=True), axis=0)
            center = y - y.mean()
            norm = np.linalg.norm(center) * np.linalg.norm(template)
            if norm:
                shape["template_correlation"] = float(np.dot(center, template) / norm)
    independent = dict(detector="wfdb.xqrs", read_range=[lo, hi], status="unavailable", reason="insufficient_context")
    if hi - lo >= 4 * fs and min(q - lo, hi - 1 - q) >= fs:
        try:
            from wfdb.processing import xqrs_detect

            points = np.asarray(xqrs_detect(x[lo:hi], fs=fs, verbose=False), dtype=int) + lo
            near = points[np.abs(points - q) <= radius].tolist()
            independent = dict(
                detector="wfdb.xqrs",
                read_range=[lo, hi],
                matching_samples=near,
                status="agreement" if near else "disagreement",
            )
        except (ImportError, ValueError, IndexError) as exc:
            independent["reason"] = type(exc).__name__
    peer_results = {peer["task"]: pulse_evidence(case, peer, data, t[q], t[marks]) for peer, data in peers}
    supporting = ["native_protected_landmark"]
    conflicting = []
    if independent["status"] == "agreement":
        supporting.append("independent_detector_agrees")
    elif independent["status"] == "disagreement":
        conflicting.append("independent_detector_disagrees")
    if any(z["status"] == "consistent" for z in peer_results.values()):
        supporting.append("neighbour_lag_consistent_pulse")
    if any(z["status"] == "discordant" for z in peer_results.values()):
        conflicting.append("neighbour_lag_discordant_pulse")
    return dict(
        version="qrs-conflict.1",
        sample=q,
        action="preserve",
        review_status="need_evidence",
        event_amended=False,
        independent=independent,
        morphology=shape,
        peers=peer_results,
        disagreements=conflicting,
        hypotheses={
            "H0": dict(explanation="true QRS or physiological variant", supporting=supporting, opposing=conflicting),
            "H1": dict(
                explanation="interbeat artifact detected as a beat", supporting=conflicting, opposing=supporting
            ),
            "H2": dict(
                explanation="artifact superimposed on a real beat",
                observations=["suspected_core_near_native_event"],
                missing=["independent_clean_core_morphology"],
            ),
            "H3": dict(explanation="unresolved", observations=["no_reference_event_label"]),
        },
        next_evidence=["inspect_observed_neighbour_morphology", "review_independent_event_annotation"],
    )
