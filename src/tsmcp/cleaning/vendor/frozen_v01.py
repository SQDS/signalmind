"""Frozen public-input-only engineering baselines for this synthetic pilot.

No function in this module opens reference arrays, answer keys, generator
manifests, or another case. These are illustrative classical DSP rules, not
clinical algorithms and not a promise that every case will pass.
"""
from __future__ import annotations

import numpy as np
import scipy
from scipy.signal import iirnotch, sosfiltfilt, tf2sos, welch

from .common import change_mask, mask_ranges


BASELINE_VERSION = "public-fixed-0.1"

# Frozen engineering parameters. They were chosen without reading private
# reference values or grader outcomes. All applied parameters are logged.
PARAMETERS = {
    "anchor_samples_each_side": 4,
    "gap_anchor_fit_error_fraction": 0.08,
    "gap_anchor_slope_difference_fraction": 0.12,
    "spike_max_width_samples": 4,
    "spike_jump_fraction": 0.25,
    "spike_peak_residual_fraction": 0.50,
    "spike_edge_residual_fraction": 0.25,
    "spike_anchor_step_fraction": 0.12,
    "spike_anchor_level_difference_fraction": 0.20,
    "line_min_segment_s": 4.0,
    "line_welch_window_s": 4.0,
    "line_search_half_width_hz": 0.65,
    "line_neighbor_inner_hz": 2.0,
    "line_neighbor_outer_hz": 6.0,
    "line_peak_to_neighbor_ratio_min": 20.0,
    "line_power_fraction_min": 0.005,
    "notch_Q": 35.0,
    "notch_pad_s": 1.0,
}


def _operation(op, intervals, parameters):
    return {
        "op": op,
        "ranges_samples": [[int(s), int(e)] for s, e in intervals],
        "parameters": parameters,
        "implementation": "baselines." + BASELINE_VERSION,
        "version": "numpy=" + np.__version__ + ";scipy=" + scipy.__version__,
    }


def _scale(values, valid):
    a = values[valid & np.isfinite(values)]
    if len(a) < 2:
        return 0.0
    return float(np.percentile(a, 99) - np.percentile(a, 1))


def _protected(task, n):
    result = np.zeros(n, dtype=bool)
    policy = task["public_policy"]
    windows = policy.get("protected_event_windows", [])
    windows = windows + policy.get("protected_event_windows_optional", [])
    for item in windows:
        interval = item.get("range_samples") if isinstance(item, dict) else item
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("Malformed public protected-event interval")
        s, e = int(interval[0]), int(interval[1])
        if not 0 <= s < e <= n:
            raise ValueError("Public protected-event interval is out of bounds")
        result[s:e] = True
    return result


def _initial_arrays(data):
    x = np.asarray(data["values"], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError("This baseline expects exactly one C x N channel")
    missing = np.asarray(data["observed_missing_mask"], dtype=bool)
    clipped = np.asarray(data["device_saturation_flag"], dtype=bool)
    if x.shape != missing.shape or x.shape != clipped.shape:
        raise ValueError("Public observation masks have inconsistent shapes")
    # Explicit acquisition flags are used by both baselines; this does not
    # modify the observed waveform or infer any hidden reference label.
    valid = np.isfinite(x) & ~missing & ~clipped
    artifact = missing | clipped | ~np.isfinite(x)
    return x.copy(), valid.copy(), artifact.copy(), np.zeros_like(valid)


def identity(task, data):
    """Keep waveform values; honor public missing/saturation evidence."""
    x, valid, artifact, repaired = _initial_arrays(data)
    return {
        "values": x,
        "artifact_mask": artifact,
        "repaired_mask": repaired,
        "valid_mask": valid,
        "action": "keep",
        "operations": [],
        "rationale": "保留输入波形；仅继承公开缺失和设备饱和标志所要求的有效性。",
        "limitations": ["这是原样波形基线，不进行修补或去噪。"],
    }


def _eligible_gap(x, valid, protected, s, e, scale):
    k = PARAMETERS["anchor_samples_each_side"]
    if s < k or e + k > len(x) or protected[s:e].any():
        return False
    if not valid[s-k:s].all() or not valid[e:e+k].all() or scale <= 0:
        return False
    left, right = x[s-k:s], x[e:e+k]
    axis = np.arange(k, dtype=float)
    a = np.polyfit(axis, left, 1)
    b = np.polyfit(axis, right, 1)
    residual = max(float(np.max(np.abs(left-np.polyval(a, axis)))),
                   float(np.max(np.abs(right-np.polyval(b, axis)))))
    return (
        residual <= PARAMETERS["gap_anchor_fit_error_fraction"] * scale
        and abs(a[0]-b[0]) <= PARAMETERS["gap_anchor_slope_difference_fraction"] * scale
    )


def _repair_missing(x, valid, artifact, repaired, protected, task):
    limit = task["public_policy"].get("max_gap_samples")
    if limit is None or int(limit) <= 0:
        return []
    scale = _scale(x, valid)
    intervals = []
    for s, e in mask_ranges(~np.isfinite(x)):
        if e-s <= int(limit) and _eligible_gap(x, valid, protected, s, e, scale):
            x[s:e] = np.interp(np.arange(s, e), [s-1, e], [x[s-1], x[e]])
            valid[s:e] = True
            artifact[s:e] = True
            repaired[s:e] = True
            intervals.append([s, e])
    return intervals


def _repair_spikes(x, valid, artifact, repaired, protected):
    """Repair short discontinuous excursions with quiet two-sided context.

    Sharp but continuously sampled QRS complexes usually fail the quiet-anchor
    and edge-jump conditions. Public event protection always takes precedence.
    This conservative rule is intentionally allowed to miss some artifacts.
    """
    scale = _scale(x, valid)
    if scale <= 0:
        return []
    k = PARAMETERS["anchor_samples_each_side"]
    jumps = np.flatnonzero(
        valid[:-1] & valid[1:]
        & (np.abs(np.diff(x)) > PARAMETERS["spike_jump_fraction"] * scale)
    ) + 1
    candidates = []
    for s in jumps:
        for width in range(1, PARAMETERS["spike_max_width_samples"] + 1):
            e = int(s + width)
            if s < k or e + k > len(x):
                continue
            if not valid[s-k:e+k].all() or protected[s:e].any() or repaired[s:e].any():
                continue
            left, right = x[s-k:s], x[e:e+k]
            if max(np.max(np.abs(np.diff(left))), np.max(np.abs(np.diff(right)))) > PARAMETERS["spike_anchor_step_fraction"] * scale:
                continue
            if abs(x[e]-x[s-1]) > PARAMETERS["spike_anchor_level_difference_fraction"] * scale:
                continue
            replacement = np.interp(np.arange(s, e), [s-1, e], [x[s-1], x[e]])
            residual = x[s:e] - replacement
            same_sign = np.all(residual > 0) or np.all(residual < 0)
            if not same_sign:
                continue
            if np.max(np.abs(residual)) < PARAMETERS["spike_peak_residual_fraction"] * scale:
                continue
            if min(abs(residual[0]), abs(residual[-1])) < PARAMETERS["spike_edge_residual_fraction"] * scale:
                continue
            candidates.append((float(np.max(np.abs(residual))), int(s), e, replacement))
    intervals = []
    occupied = np.zeros(len(x), dtype=bool)
    for _, s, e, replacement in sorted(candidates, key=lambda a: (-a[0], a[1], a[2])):
        if occupied[max(0, s-1):min(len(x), e+1)].any():
            continue
        x[s:e] = replacement
        artifact[s:e] = True
        repaired[s:e] = True
        occupied[s:e] = True
        intervals.append([s, e])
    return sorted(intervals)


def _line_evidence(segment, fs, candidates):
    nperseg = min(len(segment), max(16, int(round(PARAMETERS["line_welch_window_s"] * fs))))
    frequencies, power = welch(segment, fs=fs, window="hann", nperseg=nperseg,
                               noverlap=nperseg//2, detrend="constant",
                               scaling="density")
    total = float(np.sum(power))
    if total <= 0:
        return None
    observed = []
    for target in candidates:
        if not 0 < target < fs/2 - 2.0:
            continue
        line = np.abs(frequencies-target) <= PARAMETERS["line_search_half_width_hz"]
        distance = np.abs(frequencies-target)
        neighbor = (distance >= PARAMETERS["line_neighbor_inner_hz"]) & (distance <= PARAMETERS["line_neighbor_outer_hz"])
        if not line.any() or not neighbor.any():
            continue
        peak_index = np.flatnonzero(line)[np.argmax(power[line])]
        denominator = max(float(np.median(power[neighbor])), np.finfo(float).tiny)
        contrast = float(power[peak_index]/denominator)
        fraction = float(np.sum(power[line])/total)
        if contrast >= PARAMETERS["line_peak_to_neighbor_ratio_min"] and fraction >= PARAMETERS["line_power_fraction_min"]:
            observed.append({
                "candidate_hz":float(target),
                "estimated_hz":float(frequencies[peak_index]),
                "peak_to_neighbor_ratio":contrast,
                "line_power_fraction":fraction,
                "nperseg":int(nperseg),
            })
    if not observed:
        return None
    # One strongly evidenced frequency per continuous segment in this edition.
    return max(observed, key=lambda p: p["line_power_fraction"])


def _suppress_line(x, valid, artifact, task):
    fs = float(task["signal"]["channels"][0]["fs_hz"])
    candidates = task["public_policy"].get("candidate_line_hz", [50.0, 60.0])
    operations = []
    for s, e in mask_ranges(valid):
        if e-s < int(round(PARAMETERS["line_min_segment_s"] * fs)):
            continue
        evidence = _line_evidence(x[s:e], fs, candidates)
        if evidence is None:
            continue
        b, a = iirnotch(evidence["estimated_hz"], Q=PARAMETERS["notch_Q"], fs=fs)
        sos = tf2sos(b, a)
        padlen = min(int(round(PARAMETERS["notch_pad_s"] * fs)), e-s-2)
        if padlen < 1:
            continue
        x[s:e] = sosfiltfilt(sos, x[s:e], padtype="odd", padlen=padlen)
        artifact[s:e] = True
        operations.append(_operation("notch", [[s, e]], {
            "f0_hz":evidence["estimated_hz"], "Q":PARAMETERS["notch_Q"],
            "fs_hz":fs, "phase":"forward_backward_offline",
            "padtype":"odd", "padlen":int(padlen),
            "sos":sos.tolist(), "public_psd_evidence":evidence,
            "selection_rule":{key:value for key,value in PARAMETERS.items() if key.startswith("line_")},
        }))
    return operations


def fixed(task, data):
    """Apply the frozen rules using only this case's public observation."""
    values, valid_mask, artifact_mask, repaired_mask = _initial_arrays(data)
    x, valid = values[0], valid_mask[0]
    artifact, repaired = artifact_mask[0], repaired_mask[0]
    protected = _protected(task, len(x))
    operations = []

    gaps = _repair_missing(x, valid, artifact, repaired, protected, task)
    if gaps:
        operations.append(_operation("repair", gaps, {
            "method":"linear", "source":"observed_missing",
            "max_gap_samples":task["public_policy"].get("max_gap_samples"),
            "extrapolate":False,
            "anchor_samples_each_side":PARAMETERS["anchor_samples_each_side"],
            "anchor_fit_error_fraction":PARAMETERS["gap_anchor_fit_error_fraction"],
            "anchor_slope_difference_fraction":PARAMETERS["gap_anchor_slope_difference_fraction"],
            "public_protected_windows_respected":True,
        }))

    spikes = _repair_spikes(x, valid, artifact, repaired, protected)
    if spikes:
        operations.append(_operation("repair", spikes, {
            "method":"linear", "source":"public_discontinuous_excursion_rule",
            "extrapolate":False,
            **{key:value for key,value in PARAMETERS.items() if key.startswith("spike_")},
            "anchor_samples_each_side":PARAMETERS["anchor_samples_each_side"],
            "public_protected_windows_respected":True,
        }))

    line_ops = _suppress_line(x, valid, artifact, task)
    operations.extend(line_ops)
    invalid_intervals = mask_ranges(~valid)
    if invalid_intervals:
        operations.append(_operation("mask_invalid", invalid_intervals, {
            "source":"public_missing_saturation_or_unrepaired_gap",
            "long_gap_policy":"leave_missing_values_unchanged",
            "saturation_policy":"leave_observed_clipped_values_unchanged",
        }))

    changed = bool(change_mask(np.asarray(data["values"]), values).any())
    if changed and invalid_intervals:
        action = "partial_repair_and_mask"
    elif repaired.any():
        action = "repair"
    elif line_ops:
        action = "denoise"
    elif invalid_intervals:
        action = "mask_only"
    else:
        action = "keep"

    return {
        "values":values, "artifact_mask":artifact_mask,
        "repaired_mask":repaired_mask, "valid_mask":valid_mask,
        "action":action, "operations":operations,
        "rationale":"按公开缺失、设备标志和保护规则局部修补；只有满足固定谱峰证据条件才进行窄带处理。",
        "limitations":[
            "固定经典基线仅用于本合成开发试跑；未读取参考答案，也不保证所有可修补污染均被识别。",
            "插补值是估计；有效测量应继续遵守公开支持范围规则。",
        ],
    }


def run_baseline(task, data, mode):
    if mode == "identity":
        return identity(task, data)
    if mode == "fixed":
        return fixed(task, data)
    raise ValueError("Unknown baseline mode: " + str(mode))
