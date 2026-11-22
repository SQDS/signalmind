"""Shared artifact helpers. These functions never read private references."""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

PROTOCOL = "signalmind.synthetic.processing.v0.3"

def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n", encoding="utf-8")

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def save_npz(path, **arrays):
    """Deterministic compressed NPZ. No object arrays or pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for key in sorted(arrays):
            a = np.asarray(arrays[key])
            if a.dtype.hasobject:
                raise TypeError("Object arrays are forbidden")
            b = io.BytesIO()
            np.lib.format.write_array(b, a, allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, b.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                       compresslevel=6)

def change_mask(a, b, atol=1e-12, rtol=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    return ~np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)

def mask_ranges(mask):
    """Sorted, disjoint half-open ranges for one channel."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [[int(a), int(b)] for a, b in edges.reshape(-1, 2)]

def ranges_mask(ranges, n):
    result = np.zeros(n, dtype=bool)
    for start, end in ranges:
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Sample ranges require integer bounds")
        if not 0 <= start < end <= n:
            raise ValueError("Invalid half-open interval")
        result[start:end] = True
    return result

def detected_peaks(values, valid, fs, prominence, min_distance_s=0.30):
    """Independent, public fixed detector for the stylized synthetic signals.
    Never connects events across an invalid region. Not a clinical detector.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(x)
    runs = []
    distance = max(1, int(round(min_distance_s * fs)))
    for s, e in mask_ranges(valid):
        if e - s < 3:
            continue
        p, _ = find_peaks(x[s:e], prominence=prominence, distance=distance)
        runs.append(p.astype(int) + s)
    return runs

def measure_signal(values, valid, task):
    """Public display metrics; not the private fixed-domain quality scores.
    Comparison must account for support differences. No clinical claims.
    """
    x = np.asarray(values, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.isfinite(x)
    if x.ndim != 2 or x.shape[0] != 1 or x.shape != v.shape:
        raise ValueError("v0.1 expects one channel and matching C x N arrays")
    meta = task["signal"]["channels"][0]
    fs, n, unit = meta["fs_hz"], meta["n_samples"], meta["unit"]
    out = []
    def add(name, value, unit, status="ok"):
        out.append({"name":name, "value":value, "unit":unit,
                    "status":status, "range_samples":[0, n],
                    "evidence_id":task["signal"]["signal_id"]})
    finite_values = x[v]
    add("valid_samples", int(v.sum()), "sample")
    add("valid_fraction", float(v.mean()), "1")
    for name, fun in [("sample_mean", np.mean),("sample_min",np.min),("sample_max",np.max)]:
        if len(finite_values):
            add(name, float(fun(finite_values)), unit)
        else:
            add(name, None, unit, "insufficient_evidence")
    dc = task["downstream_contract"]
    peaks = detected_peaks(x, v, fs, dc["detector_prominence"], dc["min_distance_s"])
    add("detected_beats", int(sum(len(p) for p in peaks)), "beat")
    rr = [np.diff(p) / fs for p in peaks if len(p) >= 2]
    if rr and sum(len(a) for a in rr):
        add("mean_hr_bpm", float(60 / np.mean(np.concatenate(rr))), "bpm")
    else:
        add("mean_hr_bpm", None, "bpm", "insufficient_evidence")
    return out

def write_submission(outdir, task, input_values, values, artifact_mask,
                     repaired_mask, valid_mask, action, operations,
                     rationale="", limitations=None):
    """Create result.json + output.npz from a single public case.
    Never supplies processing choices or reference values.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float64)
    expected_shape = (len(task["signal"]["channels"]), task["signal"]["channels"][0]["n_samples"])
    arrays = {
        "values": values,
        "artifact_mask": np.asarray(artifact_mask, dtype=bool),
        "modified_mask": change_mask(input_values, values),
        "repaired_mask": np.asarray(repaired_mask, dtype=bool),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
    }
    if any(a.shape != expected_shape for a in arrays.values()):
        raise ValueError("All arrays must have shape C x N")
    source_folder = task.get("_case_dir")
    if source_folder:
        with np.load(Path(source_folder) / "input.npz", allow_pickle=False) as raw:
            before_valid = np.isfinite(input_values) & ~raw["device_saturation_flag"]
    else:
        before_valid = np.isfinite(input_values)
    datafile = outdir / "output.npz"
    save_npz(datafile, **arrays)
    digest = sha256_file(datafile)
    new_id = "derived_" + digest[:20]
    before = measure_signal(input_values, before_valid, task)
    after = measure_signal(values, arrays["valid_mask"], task)
    for m in after:
        m["evidence_id"] = new_id
    result = {
        "schema_version":PROTOCOL, "case_id":task["case_id"], "action":action,
        "signal":{
            "new_signal_id":new_id,
            "parent_signal_id":task["signal"]["signal_id"],
            "artifact_uri":"output.npz", "sha256":digest,
            "channels":task["signal"]["channels"],
            "analysis_role":("beat_timing_only" if task["objective"].get("evaluation_track") == "beat_timing" else "morphology_and_amplitude"),
        },
        "array_layout": {k:{"storage_key":k,"dtype":str(a.dtype),"shape":list(a.shape)}
                         for k,a in arrays.items()},
        "operations":operations,
        "measurements":{"before":before,"after":after},
        "rationale":rationale, "limitations":limitations or [],
    }
    write_json(outdir / "result.json", result)
    return result
