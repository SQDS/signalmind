"""Non-overridable write checks on the actual parent, not on a model's description or peak count."""

import numpy as np
from scipy.signal import find_peaks

from tsmcp.inspection.support import native_support
from tsmcp.processing import PCHIP_ANCHORS, PCHIP_TOOL, PCHIP_VERSION, intervals

from .common import changed, digest, sha

ACCEPTANCE = {
    "version": "physiology-gate.1",
    "spike_max_samples": 4,
    "protected_ecg_s": 0.08,
    "read_context_s": 8,
    "min_neighbour_beats": 3,
    "template_correlation": 0.90,
    "template_gain": [0.65, 1.5],
    "cadence_fraction": 0.20,
    "max_refinement_rounds": 1,
}
SPEC_ID = digest([ACCEPTANCE, sha(__file__)])
OPERATIONS = {
    "interpolate",
    PCHIP_TOOL,
    "repair",
    "mark_invalid",
    "mask_invalid",
    "notch",
    "lowpass",
    "remove_tone",
    "local_detrend",
}


def same_beat_evidence(x, good, fs, a, b):
    """Inspect raw extrema, their shape and cadence independently of the QRS detector."""
    radius = max(3, round(0.045 * fs))
    width = round(ACCEPTANCE["read_context_s"] * fs)
    lo = max(0, a - width)
    hi = min(len(x), b + width)
    # Use only the continuous observed run around this particular core.
    left = np.flatnonzero(~good[lo:a])
    right = np.flatnonzero(~good[b:hi])
    if len(left):
        lo += int(left[-1]) + 1
    if len(right):
        hi = b + int(right[0])
    if hi - lo < 3 * fs or a - radius < lo or b + radius > hi:
        return {"status": "insufficient_support", "reason": "morphology_context", "read_range": [lo, hi]}
    y = x[lo:hi]
    center = float(np.median(y))
    peak = a + int(np.argmax(np.abs(x[a:b] - center)))
    signed = (y - center) * np.sign(x[peak] - center)
    scale = float(np.ptp(np.percentile(y, [2, 98])))
    if scale <= np.finfo(float).eps:
        return {"status": "insufficient_support", "reason": "no_neighbour_swing", "read_range": [lo, hi]}
    refs = find_peaks(signed, prominence=0.25 * scale, distance=max(1, round(0.30 * fs)))[0] + lo
    refs = refs[(abs(refs - peak) > 0.20 * fs) & (refs >= lo + radius) & (refs + radius < hi)]
    if len(refs) < 3:
        return {"status": "insufficient_support", "reason": "too_few_independent_extrema", "read_range": [lo, hi]}
    periods = np.diff(refs)
    cadence = float(np.median(periods))
    near = float(np.min(np.abs(refs - peak)))
    offsets = np.arange(-radius, radius + 1)
    target = x[peak + offsets]
    target = target - target.mean()
    windows = x[refs[:, None] + offsets]
    windows = windows - windows.mean(axis=1, keepdims=True)
    amp = np.ptp(windows, axis=1)
    norms = np.linalg.norm(windows, axis=1) * np.linalg.norm(target)
    similarity = np.divide(windows @ target, norms, out=np.zeros(len(refs)), where=norms > 0)
    gain = np.divide(np.ptp(target), amp, out=np.zeros(len(refs)), where=amp > 0)
    matches = (similarity >= ACCEPTANCE["template_correlation"]) & (gain >= 0.65) & (gain <= 1.5)
    physiological = matches.sum() >= 3 and abs(near - cadence) <= 0.20 * cadence
    return dict(
        status="possible_physiology" if physiological else "evaluated_no_alert",
        read_range=[lo, hi],
        peak=peak,
        neighbour_count=len(refs),
        matching_shapes=int(matches.sum()),
        nearest_s=near / fs,
        median_period_s=cadence / fs,
        best_correlation=float(similarity.max()),
    )


def spike_check(case, x, valid, estimated, pr, a, b):
    fs = case["fs"]
    good = valid & ~estimated & np.isfinite(x)
    if (
        not (0 < a < b < len(x))
        or b - a > min(ACCEPTANCE["spike_max_samples"], 5 * fs)
        or not good[a - 1 : b + 1].all()
    ):
        return dict(status="insufficient_support", reason="core_or_anchor_unsupported")
    metadata = pr.get("detector_support")
    if metadata is None:
        metadata = native_support(good, fs, case["task"], pr["events"])
    if not any(left <= a - 1 and b + 1 <= r for left, r in metadata["evaluated_ranges"]) or any(
        left < b and a < r for left, r in metadata["empty_event_ranges"]
    ):
        return dict(status="insufficient_support", reason="detector_boundary_or_empty")
    evidence = same_beat_evidence(x, good, fs, a, b)
    if evidence["status"] != "evaluated_no_alert":
        return evidence
    if case["task"] == "ECG":
        margin = round(0.08 * fs)
        if any(a - margin <= ev["sample"] < b + margin for ev in pr["events"]):
            return dict(evidence, status="possible_physiology", reason="near_native_qrs")
    return evidence


def waveform_gate(case, before, after, operations, declared=None, *, filter_policy=None):
    """Use declared operator ranges for authorization and independently diff numerical and support writes."""
    t, x, v, e, pr = before
    t2, y, v2, e2, _ = after
    n = len(x)
    checks = {}
    checks["grid"] = np.array_equal(t, t2) and y.shape == x.shape
    if not checks["grid"]:
        return dict(status="rejected", checks=checks)
    allowed = np.zeros(n, bool)
    numeric = np.zeros(n, bool)
    interp = np.zeros(n, bool)
    disable = np.zeros(n, bool)
    registered = bool(operations)
    for op in operations:
        name = op["op"]
        if name not in OPERATIONS:
            registered = False
            continue
        for a, b in op["ranges_samples"]:
            if not 0 <= a < b <= n:
                registered = False
                continue
            allowed[a:b] = True
            if name in ("mark_invalid", "mask_invalid"):
                disable[a:b] = True
            if name not in ("mark_invalid", "mask_invalid"):
                numeric[a:b] = True
            if name in ("repair", "interpolate", PCHIP_TOOL):
                interp[a:b] = True
    if declared is not None:
        permit = np.zeros(n, bool)
        for a, b in declared:
            if not 0 <= a < b <= n:
                registered = False
            else:
                permit[a:b] = True
        allowed &= permit
    delta = changed(x, y)
    mask_delta = (v != v2) | (e != e2)
    checks.update(
        registered_tools=registered,
        write_ranges=not np.any((delta | mask_delta) & ~allowed),
        value_scope=not np.any(delta & ~numeric),
        support_scope=not np.any(((e != e2) & ~interp) | ((v != v2) & ~(interp | disable))),
        observed_provenance=not np.any((v2 & ~v & ~e2) | (e & ~e2) | (delta & interp & ~e2)),
        finite=bool(np.isfinite(y[v2]).all()),
    )
    details = []
    for a, b in intervals((e2 & ~e) | (delta & interp)):
        d = spike_check(case, x, v, e, pr, a, b)
        details.append(dict(a=a, b=b, **d))
    checks["interpolation_support"] = all(d["status"] == "evaluated_no_alert" for d in details)
    # Different operations have different required outcomes. A support mark cannot pass as a numerical repair.
    for op in operations:
        name = op["op"]
        if name in ("mark_invalid", "mask_invalid"):
            checks["mark_invalid"] = checks.get("mark_invalid", True) and all(
                not v2[a:b].any() for a, b in op["ranges_samples"]
            )
        elif name == PCHIP_TOOL:
            from scipy.interpolate import PchipInterpolator

            params = op["parameters"]
            k = PCHIP_ANCHORS
            regions = op["ranges_samples"]
            ok = (
                op.get("version") == PCHIP_VERSION
                and params.get("method") == "pchip"
                and params.get("anchors_per_side") == k
                and params.get("extrapolate") is False
                and params.get("anchor_ranges") == [[[a - k, a], [b, b + k]] for a, b in regions]
            )
            for a, b in regions:
                if not (k <= a < b <= n - k and b - a <= ACCEPTANCE["spike_max_samples"]):
                    ok = False
                    continue
                ix = np.r_[a - k : a, b : b + k]
                grid = t[a - k : b + k]
                supported = (v[ix] & ~e[ix] & ~allowed[ix] & np.isfinite(x[ix])).all()
                continuous = np.all(
                    np.abs(np.diff(grid) - 1 / case["fs"]) <= np.maximum(1e-6, 32 * np.spacing(abs(grid[1:])))
                )
                if not (supported and continuous and (v2[a:b] & e2[a:b]).all()) or any(
                    lo < b + k and a - k < hi for lo, hi in case.get("source_time_conflicts", [])
                ):
                    ok = False
                    continue
                expected = PchipInterpolator(ix, x[ix], extrapolate=False)(np.arange(a, b))
                ok = ok and np.array_equal(y[a:b], expected)
            checks["registered_pchip"] = bool(ok and checks.get("registered_pchip", True))
        elif name in ("notch", "remove_tone", "lowpass", "local_detrend"):
            flags = []
            for a, b in op["ranges_samples"]:
                if not (v[a:b] & ~e[a:b]).all() or b - a < 10:
                    flags.append(False)
                    continue
                p = op["parameters"]
                fs = case["fs"]
                if name in ("notch", "remove_tone"):
                    hz = p.get("frequency_hz", p.get("f0_hz"))
                    if hz is None or not 0 < hz < fs / 2:
                        flags.append(False)
                        continue
                    weights = np.exp(-2j * np.pi * hz * np.arange(b - a) / fs) * np.hanning(b - a)
                    attenuation = abs((y[a:b] - y[a:b].mean()) @ weights) < abs((x[a:b] - x[a:b].mean()) @ weights)
                elif name == "lowpass":
                    attenuation = np.mean(np.diff(y[a:b]) ** 2) < np.mean(np.diff(x[a:b]) ** 2)
                else:
                    attenuation = case["task"] == "ECG" and np.ptp(y[a:b]) <= np.ptp(x[a:b])
                swing = max(float(np.ptp(np.percentile(x[a:b], [5, 95]))), np.finfo(float).eps)
                dc_ok = case["task"] != "ABP" or abs(np.mean(y[a:b] - x[a:b])) <= 0.05 * swing
                # Compare the low-frequency physiological envelope instead of judging the change only by smoothness.
                duration = b - a
                width = min(duration // 2, max(1, round(fs * (0.04 if case["task"] == "ECG" else 0.08))))
                kernel = np.ones(max(1, width)) / max(1, width)
                pre = np.convolve(x[a:b], kernel, mode="valid")
                post = np.convolve(y[a:b], kernel, mode="valid")
                pre_swing = float(np.ptp(np.percentile(pre, [3, 97])))
                post_swing = float(np.ptp(np.percentile(post, [3, 97])))
                preserved = pre_swing > 0 and 0.7 * pre_swing <= post_swing <= 1.3 * pre_swing
                if name == "local_detrend":
                    # Baselines are expected to move; preserve measured QRS excursions against their adjacent flanks.
                    radius = max(1, round(0.08 * fs))
                    peaks = [ev["sample"] for ev in pr["events"] if a + radius <= ev["sample"] < b - radius]
                    preserved = bool(peaks) and all(
                        0.8 * np.ptp(x[q - radius : q + radius + 1])
                        <= np.ptp(y[q - radius : q + radius + 1])
                        <= 1.2 * np.ptp(x[q - radius : q + radius + 1])
                        for q in peaks
                    )
                flags.append(bool(attenuation and dc_ok and preserved))
            checks[name] = checks.get(name, True) and all(flags)
    filter_details = []
    for op in operations:
        if op["op"] in ("notch", "remove_tone", "lowpass", "local_detrend") and (
            filter_policy or "context_samples" in op["parameters"]
        ):
            from .filter_retest import filter_retest

            if not filter_policy:
                checks["filter_profile"] = False
                continue
            specific, rows = filter_retest(case, before, after, op, filter_policy)
            for key, value in specific.items():
                checks[key] = bool(value and checks.get(key, True))
            filter_details.extend(rows)
    return dict(
        status="passed" if all(checks.values()) else "rejected",
        checks=checks,
        interpolations=details,
        **({"filter_retest": filter_details} if filter_policy else {}),
    )


def registered_parameters(tool, trigger, case, proposed):
    """Registered DSP recipes only. Validate parameters before calling the existing numerical operator."""
    fs = case["fs"]
    if tool in ("remove_tone", "notch") and trigger in ("line_50", "line_60"):
        hz = float(trigger[5:])
        params = {"frequency_hz": hz}
        if hz >= fs / 2:
            return None
    elif tool == "lowpass" and trigger == "high_frequency":
        hz = proposed.get("frequency_hz", {"ECG": 40, "PPG": 12, "ABP": 18}[case["task"]])
        if not 1 <= hz < fs / 2:
            return None
        params = {"frequency_hz": hz, "filter_order": 4}
    elif tool == "local_detrend" and trigger == "ecg_baseline_shift" and case["task"] == "ECG":
        window = proposed.get("window_s", 0.4)
        if not 0.18 <= window <= 0.8:
            return None
        params = {"window_s": window}
    else:
        return None
    if any(k not in params or params[k] != value for k, value in proposed.items()):
        return None
    return params
