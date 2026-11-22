"""Reusable peak-annotation policies. Original ECG/ABP samples are never modified."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tsmcp.core.dsp import fill_nan
from tsmcp.plugins.ecg.rpeaks import bandpass_qrs, detect_r_peaks
from tsmcp.processing import intervals

SCHEMA = "signalmind.peak-policy.v0.1"
FEATURE_NAMES = {
    "missing_fraction",
    "flat_fraction",
    "old_peak_count",
    "old_rr_outside_fraction",
    "detector_disagreement_fraction",
    "median_template_correlation",
    "removed_fraction",
    "kept_peak_count",
    "kept_rr_outside_fraction",
}


def digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate(policy):
    if policy.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    if policy.get("alignment", "none") not in {"none", "positive", "negative", "absolute"}:
        raise ValueError("alignment is none, positive, negative or absolute")
    bounds = {
        "radius_ms": (0, 200),
        "min_energy_fraction": (0, 2),
        "min_template_correlation": (-1, 1),
        "max_detector_distance_ms": (0, 500),
        "refractory_ms": (0, 500),
    }
    for name, (low, high) in bounds.items():
        if name in policy and (not np.isfinite(policy[name]) or not low <= policy[name] <= high):
            raise ValueError(f"{name} must be between {low} and {high}")
    for rule in policy.get("exclude_if", []):
        if (
            rule.get("feature") not in FEATURE_NAMES
            or rule.get("op") not in {"lt", "gt"}
            or not np.isfinite(rule["value"])
        ):
            raise ValueError("exclude_if uses documented features, lt/gt and finite numeric values")
    digest(policy)
    return policy


def outside_rr(peaks, fs):
    rr = np.diff(np.sort(peaks)) / fs
    return float(np.mean((rr < 0.3) | (rr > 2))) if rr.size else 1.0


def observe(ecg, old_peaks, fs=125):
    x, peaks = np.asarray(ecg, float), np.asarray(old_peaks, int)
    finite = np.isfinite(x)
    temporary = fill_nan(x)
    filtered = bandpass_qrs(temporary, fs)
    detected = detect_r_peaks(temporary, fs)
    radius = max(1, round(0.12 * fs))
    ix = np.clip(peaks[:, None] + np.arange(-radius, radius + 1), 0, len(x) - 1)
    waves = filtered[ix]
    energy = np.sqrt(np.mean(waves**2, axis=1))
    median_energy = max(float(np.median(energy)) if energy.size else 0, 1e-12)
    normalized = waves - np.mean(waves, axis=1, keepdims=True)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    template = np.median(normalized, axis=0) if len(normalized) else np.zeros(2 * radius + 1)
    template /= max(float(np.linalg.norm(template)), 1e-12)
    correlation = normalized @ template
    distance = (
        np.min(np.abs(peaks[:, None] - detected[None, :]), axis=1) / fs * 1000
        if len(detected)
        else np.full(len(peaks), 1e9)
    )
    flat = np.zeros(len(x), bool)
    for a, b in intervals(finite[1:] & finite[:-1] & (np.diff(x) == 0)):
        if b - a + 1 >= 0.5 * fs:
            flat[a : b + 1] = True
    features = {
        "missing_fraction": float((~finite).mean()),
        "flat_fraction": float(flat.mean()),
        "old_peak_count": len(peaks),
        "old_rr_outside_fraction": outside_rr(peaks, fs),
        "detector_disagreement_fraction": float(np.mean(distance > 120)) if len(peaks) else 1.0,
        "median_template_correlation": float(np.median(correlation)) if len(peaks) else 0.0,
    }
    return {
        "features": features,
        "energy_fraction": energy / median_energy,
        "template_correlation": correlation,
        "detector_distance_ms": distance,
        "detected_peaks": detected,
    }


def apply(ecg, old_peaks, policy, fs=125, observed=None):
    """Return one coordinate per ORIGINAL marker, -1 for deletion, plus a window decision."""
    x, old = np.asarray(ecg, float), np.asarray(old_peaks, int)
    observed = observe(x, old, fs) if observed is None else observed
    new = old.copy()
    remove = np.zeros(len(old), bool)
    for parameter, statistic, comparison in (
        ("min_energy_fraction", "energy_fraction", np.less),
        ("min_template_correlation", "template_correlation", np.less),
        ("max_detector_distance_ms", "detector_distance_ms", np.greater),
    ):
        if parameter in policy:
            remove |= comparison(observed[statistic], policy[parameter])
    new[remove] = -1
    radius = round(policy.get("radius_ms", 80) * fs / 1000)
    mode = policy.get("alignment", "none")
    if mode != "none":
        for i in np.flatnonzero(~remove):
            a, b = max(0, old[i] - radius), min(len(x), old[i] + radius + 1)
            values = x[a:b]
            if not np.isfinite(values).any():
                continue
            if mode == "positive":
                offset = np.nanargmax(values)
            elif mode == "negative":
                offset = np.nanargmin(values)
            else:
                offset = np.nanargmax(np.abs(values - np.nanmedian(values)))
            new[i] = a + int(offset)
    # Explicit refractory selection; exact collisions retain the stronger candidate even at zero refractory.
    refractory = max(1, round(policy.get("refractory_ms", 0) * fs / 1000))
    selected = []
    for i in sorted(np.flatnonzero(new >= 0), key=lambda i: (-observed["energy_fraction"][i], int(i))):
        if any(abs(new[i] - new[j]) < refractory for j in selected):
            new[i] = -1
        else:
            selected.append(i)
    kept = new[new >= 0]
    features = {
        **observed["features"],
        "removed_fraction": float(np.mean(new < 0)) if len(old) else 0.0,
        "kept_peak_count": len(kept),
        "kept_rr_outside_fraction": outside_rr(kept, fs),
    }
    reasons = [
        rule
        for rule in policy.get("exclude_if", [])
        if (
            features[rule["feature"]] < rule["value"]
            if rule["op"] == "lt"
            else features[rule["feature"]] > rule["value"]
        )
    ]
    return {
        "exclude": bool(reasons),
        "exclude_reasons": reasons,
        "old_peak_samples": old.tolist(),
        "new_peak_samples": new.tolist(),
        "deleted_count": int((new < 0).sum()),
        "moved_count": int(((new >= 0) & (new != old)).sum()),
        "features": features,
        "waveform_values_modified": False,
    }
