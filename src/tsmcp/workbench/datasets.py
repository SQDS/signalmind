"""CSV/NPZ adapters with explicit time, channel and unit mappings."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import TASKS, digest, read, sha, write
from tsmcp.cleaning.provenance import check_stop, emit


def files_for(dataset):
    root = Path(dataset["path"]).resolve()
    suffixes = {"vital": {".mat"}, "csv": {".csv"}, "npz": {".npz"}, "auto": {".csv", ".npz"}}[dataset["format"]]
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes)
    return [p for p in files if p.suffix.lower() in suffixes]


def fields(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("CSV has no sample rows")
        return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def preview(dataset):
    files = files_for(dataset)
    result = {"file_count": len(files), "files": [p.name for p in files[:20]], "fields": []}
    if files and dataset["format"] != "vital":
        path = files[0]
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                sample = [r for _, r in zip(range(3), reader, strict=False)]
            result["fields"] = [
                {"key": k, "shape": [None], "sample": [r[k] for r in sample]} for k in reader.fieldnames or []
            ]
        else:
            for key, value in fields(path).items():
                result["fields"].append(
                    {
                        "key": key,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sample": [str(v) for v in value.ravel()[:3]],
                    }
                )
    return result


def vector(data, spec, key="key"):
    value = np.asarray(data[spec[key]])
    if value.ndim == 2:
        if "channel" not in spec or "sample_axis" not in spec:
            raise ValueError("2-D input requires channel index and sample_axis")
        if spec["sample_axis"] not in (0, 1):
            raise ValueError("sample_axis must be 0 or 1")
        value = np.take(value, int(spec["channel"]), axis=1 - spec["sample_axis"])
    if value.ndim != 1:
        raise ValueError("Expected a one-dimensional channel after mapping")
    if value.dtype.kind in "US":
        value = np.char.strip(value.astype(str))
        value = np.where(value == "", "nan", value)
    return value.astype(np.float64)


def time_grid(data, channel, mapping, n):
    key = channel.get("time_key", mapping.get("time_key"))
    fs = channel.get("fs", mapping.get("fs"))
    if key:
        t = np.asarray(data[key], dtype=float).reshape(-1)
        scale = {"s": 1, "ms": 0.001}[mapping.get("time_unit", "s")]
        t = t * scale
        if len(t) != n or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Time must be finite, strictly increasing, and match the samples")
        if n < 2:
            raise ValueError("At least two samples are required")
        fs = float(fs) if fs else 1 / float(np.median(np.diff(t)))
        ticks = np.diff(t) * fs
        tolerance = max(1e-4, float(np.max(np.abs(t))) * np.finfo(float).eps * fs * 32)
        if (
            not np.isfinite(fs)
            or fs <= 0
            or np.any(np.abs(ticks - np.rint(ticks)) > tolerance)
            or np.any(ticks < 1 - tolerance)
        ):
            raise ValueError("Irregular sampling: resampling is not performed by this importer")
        cuts = np.flatnonzero(ticks > 1 + tolerance) + 1
    else:
        if not fs or not np.isfinite(float(fs)) or float(fs) <= 0:
            raise ValueError("Sampling rate is required when no source time axis exists")
        fs = float(fs)
        t, cuts = np.arange(n) / fs, np.array([], int)
    return t, fs, np.r_[0, cuts, n].astype(int), bool(key)


def mask_vector(data, spec, key, n, default):
    if not spec.get(key):
        return np.full(n, default, bool)
    value = vector(data, spec, key)
    if len(value) != n or not np.isin(value, [0, 1]).all():
        raise ValueError(f"{key} must contain one boolean/0/1 value per sample")
    return value.astype(bool)


def import_dataset(dataset, mapping, output, tasks, selected_files=None):
    output = Path(output)
    cases, sources, missing, failures, patients = [], [], [], [], set()
    files = files_for(dataset)
    if selected_files:
        files = [p for p in files if p.name in selected_files]
        if len(files) != len(set(selected_files)):
            raise ValueError("Selected file names must identify exactly one file each")
    if not files:
        raise ValueError("No supported input files")
    scope_path = output / "audit_file_scope.json"
    if scope_path.exists():
        files = [Path(p) for p in read(scope_path)]
    else:
        write(scope_path, [str(p) for p in files])
    known_patient_for_all = True
    channels = mapping.get("channels", [])
    observed_path = output / "audit_sources.json"
    observed = read(observed_path) if observed_path.exists() else {}
    if not channels or any(c["task"] not in TASKS for c in channels):
        raise ValueError("Map at least one ECG, PPG or ABP channel")
    if len({c["task"] for c in channels}) != len(channels):
        raise ValueError("Map one native channel per task in this first version")
    for path in files:
        check_stop(output)
        source_hash, rid = sha(path), "r_" + digest(str(path.resolve()))[:12]
        if str(path) in observed and observed[str(path)] != source_hash:
            raise ValueError("Previously audited source file changed; create a new task")
        observed[str(path)] = source_hash
        write(observed_path, observed)
        try:
            data = fields(path)
        except (ValueError, OSError) as exc:
            failures.append({"record_id": rid, "task": None, "reason": str(exc)})
            continue
        patient = mapping.get("patient_id")
        if mapping.get("patient_key"):
            ids = np.unique(np.asarray(data[mapping["patient_key"]]).astype(str))
            if len(ids) != 1:
                raise ValueError("Each file must identify one patient; split multi-patient files first")
            patient = str(ids[0])
        if patient:
            patients.add("import::" + str(patient))
        else:
            known_patient_for_all = False
        sources.append(
            {
                "record_id": rid,
                "source_sha256": source_hash,
                "source_name": path.name,
                "patient_key": "import::" + str(patient) if patient else None,
            }
        )
        for task in tasks:
            channel = next((c for c in channels if c["task"] == task), None)
            if not channel or channel["key"] not in data:
                missing.append(
                    {"record_id": rid, "task": task, "status": "unusable_for_task", "reason": "missing_channel"}
                )
                continue
            try:
                unit = channel.get("unit", "source_units")
                if task == "ABP" and unit != "mmHg":
                    raise ValueError("ABP pressure requires explicit mmHg units")
                x = vector(data, channel)
                t, fs, cuts, has_time = time_grid(data, channel, mapping, len(x))
                valid = mask_vector(data, channel, "valid_key", len(x), True) & np.isfinite(x)
                estimated = mask_vector(data, channel, "estimated_key", len(x), False)
                for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
                    cid = f"{rid}_{task}_{j:04d}"
                    folder = output / "public" / cid
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        folder / "input.npz", values=x[a:b], time=t[a:b], valid=valid[a:b], estimated=estimated[a:b]
                    )
                    case = {
                        "id": cid,
                        "record_id": rid,
                        "task": task,
                        "task_id": TASKS[task]["task_id"],
                        "field": channel["key"],
                        "unit": unit,
                        "fs": fs,
                        "n": int(b - a),
                        "duration_s": float((b - a) / fs),
                        "source_sha256": source_hash,
                        "source_range_samples": [int(a), int(b)],
                        "has_source_time": has_time,
                        "source_time_conflicts": [],
                        "mapping_hash": digest(mapping),
                        "input_sha256": sha(folder / "input.npz"),
                    }
                    write(folder / "source.json", case)
                    cases.append(case)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                failures.append({"record_id": rid, "task": task, "reason": str(exc)})
        emit(output, "import_record_completed", record_id=rid, completed_records=len(sources), total_records=len(files))
    manifest = {
        "schema": "signalmind.cleaning.dataset.v2",
        "patients": len(patients) if known_patient_for_all else None,
        "patient_keys": sorted(patients),
        "records": sources,
        "cases": cases,
        "unavailable_tasks": missing,
        "audit_failures": failures,
        "scope_hash": digest(cases),
        "source_inputs_unchanged": True,
        "reference_manifest": None,
        "task_config": {"tasks": tasks, "purpose": mapping.get("purpose", "Native waveform tasks")},
        "import_mapping": mapping,
        "dataset_id": dataset["id"],
    }
    write(output / "dataset_manifest.json", manifest)
    write(
        output / "data_audit.json", {"files": len(files), "cases": len(cases), "failures": failures, "missing": missing}
    )
    return manifest
