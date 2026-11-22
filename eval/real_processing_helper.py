"""Public display measurements and exports for reference-free real-waveform trials.

Candidate events and spectral powers are diagnostics, not clinical annotations or
clean-signal ground truth. Measurement filtering never changes the exported values.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

PROTOCOL = "signalmind.processing.vital-real-v0.1"


def save_npz(path, **arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            entry = zipfile.ZipInfo(key + ".npy", (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, buffer.getvalue())


def mask_ranges(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool).reshape(-1), False].astype(np.int8)))
    return edges.reshape(-1, 2).tolist()


def candidate_peaks(values, valid, fs, kind, prominence):
    """Fixed observable-data detectors. Results are candidate events, never truth."""
    values = np.asarray(values, float).reshape(-1)
    groups = []
    for start, end in mask_ranges(np.asarray(valid, bool).reshape(-1) & np.isfinite(values)):
        if end - start < 3 * fs:
            continue
        x = values[start:end]
        if kind == "ECG":
            filtered = sosfiltfilt(butter(3, [5, 20], fs=fs, btype="bandpass", output="sos"), x)
            width = max(3, round(0.12 * fs))
            energy = np.convolve(np.gradient(filtered) ** 2, np.ones(width) / width, mode="same")
            found = []
            for left in range(0, len(x), round(30 * fs)):
                right = min(len(x), left + round(30 * fs))
                a, b = max(0, left - round(2 * fs)), min(len(x), right + round(2 * fs))
                q50, q99 = np.quantile(energy[a:b], [0.5, 0.99])
                spread = max(q99 - q50, np.finfo(float).eps)
                peaks, _ = find_peaks(
                    energy[a:b], height=q50 + 0.12 * spread, prominence=0.04 * spread, distance=max(1, round(0.25 * fs))
                )
                for peak in peaks + a:
                    if left <= peak < right:
                        lo, hi = max(0, peak - round(0.1 * fs)), min(len(x), peak + round(0.1 * fs) + 1)
                        found.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
            peaks = []
            for point in sorted(set(found)):
                if not peaks or point - peaks[-1] >= round(0.25 * fs):
                    peaks.append(point)
                elif abs(filtered[point]) > abs(filtered[peaks[-1]]):
                    peaks[-1] = point
            peaks = np.asarray(peaks, int)
        else:
            filtered = sosfiltfilt(butter(3, 12, fs=fs, btype="lowpass", output="sos"), x)
            peaks, _ = find_peaks(filtered, distance=max(1, round(0.3 * fs)), prominence=prominence)
        margin = round(0.16 * fs)
        peaks = peaks[(peaks >= margin) & (peaks < len(x) - margin)]
        groups.append(peaks + start)
    return groups


def spectral_powers(values, valid, fs):
    """Length-weighted Welch band powers on contiguous finite support."""
    bands = {
        "line_50hz_power": (49, 51),
        "line_60hz_power": (59, 61),
        "high_35hz_power": (35, fs / 2),
        "total_ac_power": (0.5, fs / 2),
    }
    totals = dict.fromkeys(bands, 0.0)
    support = 0
    x = np.asarray(values, float).reshape(-1)
    for start, end in mask_ranges(np.asarray(valid, bool).reshape(-1) & np.isfinite(x)):
        if end - start < 64:
            continue
        f, power = welch(x[start:end], fs=fs, nperseg=min(end - start, round(8 * fs)), detrend="constant")
        df = f[1] - f[0]
        for name, (low, high) in bands.items():
            totals[name] += float(power[(f >= low) & (f <= high)].sum() * df) * (end - start)
        support += end - start
    return {name: value / support if support else None for name, value in totals.items()}


def measure_signal(values, valid, task):
    x = np.asarray(values, np.float64)
    v = np.asarray(valid, bool) & np.isfinite(x)
    meta = task["signal"]["channels"][0]
    fs, unit, n = meta["fs_hz"], meta["unit"], meta["n_samples"]
    rows = []

    def add(name, value, metric_unit):
        rows.append(
            {
                "name": name,
                "value": value,
                "unit": metric_unit,
                "status": "ok" if value is not None else "insufficient_evidence",
                "range_samples": [0, n],
                "evidence_id": task["signal"]["signal_id"],
            }
        )

    add("valid_samples", int(v.sum()), "sample")
    add("valid_fraction", float(v.mean()), "1")
    finite = x[v]
    for name, function in (("sample_mean", np.mean), ("sample_min", np.min), ("sample_max", np.max)):
        add(name, float(function(finite)) if finite.size else None, unit)
    groups = candidate_peaks(x, v, fs, meta["kind"], task["downstream_contract"]["detector_prominence"])
    add("candidate_events", sum(len(p) for p in groups), "candidate")
    rr = [np.diff(p) / fs for p in groups if len(p) >= 2]
    add("candidate_rate_bpm", float(60 / np.mean(np.concatenate(rr))) if rr else None, "bpm")
    for name, value in spectral_powers(x, v, fs).items():
        add(name, value, unit + "^2")
    return rows


def write_submission(
    outdir,
    task,
    input_values,
    values,
    artifact_mask,
    repaired_mask,
    valid_mask,
    action,
    operations,
    rationale="",
    limitations=None,
):
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, np.float64)
    shape = (1, task["signal"]["channels"][0]["n_samples"])
    arrays = {
        "values": values,
        "artifact_mask": np.asarray(artifact_mask, bool),
        "modified_mask": ~np.isclose(input_values, values, atol=1e-12, rtol=1e-10, equal_nan=True),
        "repaired_mask": np.asarray(repaired_mask, bool),
        "valid_mask": np.asarray(valid_mask, bool),
    }
    if any(a.shape != shape for a in arrays.values()):
        raise ValueError("Output arrays must retain the original one-channel sample shape")
    save_npz(folder / "output.npz", **arrays)
    digest = hashlib.sha256((folder / "output.npz").read_bytes()).hexdigest()
    new_id = "derived_" + digest[:20]
    before = measure_signal(input_values, np.isfinite(input_values), task)
    after = measure_signal(values, arrays["valid_mask"], task)
    for row in after:
        row["evidence_id"] = new_id
    result = {
        "schema_version": PROTOCOL,
        "case_id": task["case_id"],
        "action": action,
        "signal": {
            "new_signal_id": new_id,
            "parent_signal_id": task["signal"]["signal_id"],
            "artifact_uri": "output.npz",
            "sha256": digest,
            "channels": task["signal"]["channels"],
            "analysis_role": "morphology_and_amplitude",
        },
        "array_layout": {
            k: {"storage_key": k, "dtype": str(a.dtype), "shape": list(a.shape)} for k, a in arrays.items()
        },
        "operations": operations,
        "measurements": {"before": before, "after": after},
        "rationale": rationale,
        "limitations": limitations or [],
        "measurement_note": "Candidate detectors and spectral powers are diagnostics. Support changes do not establish accuracy gains. No clean reference is available.",
    }
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result
