"""Read-only PulseDB adaptation; full records, independently supported channels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .common import TASKS, digest, read, sha, write
from .provenance import check_stop


class VitalReader:
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        self.group = self.file["Subj_Wins"]
        self.references = {}
        sizes = [d.size for d in self.group.values() if isinstance(d, h5py.Dataset) and h5py.check_dtype(ref=d.dtype)]
        self.n = max(sizes, default=1)

    def close(self):
        self.file.close()

    def array(self, name, row):
        field = self.group[name]
        if h5py.check_dtype(ref=field.dtype):
            if name not in self.references:
                self.references[name] = np.asarray(field).ravel()
            field = self.file[self.references[name][row]]
        return np.asarray(field).reshape(-1)

    def text(self, name, row, default):
        if name not in self.group:
            return default
        a = self.array(name, row)
        return a.astype("<u2").tobytes().decode("utf-16-le").rstrip("\0")

    def channels(self):
        result = {}
        for kind, task in TASKS.items():
            field = task["field"]
            if field not in self.group:
                field = kind + "_Raw" if kind != "ABP" else None
            result[kind] = field if field is not None and field in self.group else None
        return result

    def metadata(self, row, field, subject):
        n = len(self.array(field, row))
        has_time = "T" in self.group
        t = self.array("T", row).astype(float) if has_time else np.arange(n) / 125
        if len(t) != n or n < 2 or not np.isfinite(t).all():
            raise ValueError("Invalid time/sample dimensions")
        dt = np.diff(t)
        fs = 1 / float(np.median(dt)) if has_time and np.all(dt > 0) else 125.0
        tolerance = max(1e-9, 32 * np.spacing(max(1.0, float(np.max(np.abs(t))))))
        # This file family declares 125 Hz; verify against the complete actual grid.
        if has_time and (abs(fs - 125) > 0.001 or np.max(np.abs(dt - 1 / 125)) > max(tolerance, 1e-6)):
            raise ValueError("Unexpected or irregular sampling grid; resampling is outside v1")
        return {
            "row": row,
            "n": n,
            "start": float(t[0]),
            "end": float(t[-1]),
            "fs": 125.0,
            "has_time": has_time,
            "time_tolerance": max(tolerance, 1e-6),
            "subject": self.text("SubjectID", row, subject),
            "case": self.text("CaseID", row, "unknown"),
            "segment": int(self.array("SegmentID", row)[0]) if "SegmentID" in self.group else None,
        }


def continuous(a, b, normalized=False):
    return bool(
        not normalized
        and a["has_time"]
        and b["has_time"]
        and a["subject"] == b["subject"]
        and a["case"] == b["case"]
        and a["case"] != "unknown"
        and a["fs"] == b["fs"]
        and (a["segment"] is None or b["segment"] == a["segment"] + 1)
        and abs(b["start"] - a["end"] - 1 / a["fs"]) <= max(a["time_tolerance"], b["time_tolerance"])
    )


def matlab_indices(values, n):
    a = np.asarray(values)
    if not np.isfinite(a).all() or not np.equal(a, np.round(a)).all() or np.any((a < 1) | (a > n)):
        raise ValueError("Expected one-based MATLAB indices")
    return a.astype(np.int64) - 1


def mark_time_conflicts(metadata):
    """Preserve all observations; quarantine ambiguous timestamps on BOTH source spans."""
    groups, conflicts = {}, []
    for row in metadata:
        if row and row["has_time"] and row["case"] != "unknown":
            groups.setdefault((row["subject"], row["case"]), []).append(row)
    for rows in groups.values():
        active = []
        for row in sorted(rows, key=lambda r: (r["start"], r["row"])):
            tolerance = row["time_tolerance"]
            active = [old for old in active if old["end"] >= row["start"] - tolerance]
            for old in active:
                a, b = max(old["start"], row["start"]), min(old["end"], row["end"])
                if a <= b + tolerance:
                    conflicts.append({"rows": [old["row"], row["row"]], "time_s": [a, b]})
                    for item in (old, row):
                        left = max(0, round((a - item["start"]) * item["fs"]))
                        right = min(item["n"], round((b - item["start"]) * item["fs"]) + 1)
                        item.setdefault("time_conflicts", []).append([left, right])
            active.append(row)
    return conflicts


def historical_patients(repo):
    """All inspected pilot candidates, not only chosen success cases, are excluded."""
    used = set()
    for p in (repo / "eval/runs").glob("*/private/manifest.json"):
        x = read(p)
        if not isinstance(x, dict) or "records" not in x:
            continue
        selected = set(x.get("development_records", []) + x.get("validation_records", []))
        used.update("vital::" + r["subject"] for r in x["records"] if r["id"] in selected)
    selection = repo / "eval/cases/vital-real-20261105/private/selection.json"
    if selection.exists():
        for row in read(selection).get("candidates", []):
            used.add("vital::" + Path(row["source_file"]).stem)
    reference = repo / "eval/runs/vital-coding-data-20261107/private/truth.json"
    if reference.exists():
        used.update(r["subject_key"] for r in read(reference))
    for selection in (repo / "eval/runs").glob("*/private/selection.json"):
        used.update(read(selection).get("patients", []))
    return used


def audit(
    dataset,
    inventory,
    output,
    *,
    count=10,
    seed=111101,
    repo=None,
    config=None,
    patient_manifest=None,
    reuse_development=False,
    excluded_patients=(),
    resume=False,
):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if (output / "dataset_manifest.json").exists():
        raise ValueError("Audit already exists; reuse that manifest or choose a new output")
    repo = Path(repo) if repo else Path(__file__).resolve().parents[3]
    used = historical_patients(repo) | set(excluded_patients)
    candidates = [
        r
        for r in read(inventory)
        if Path(r["source"]["path"]).parent == dataset
        and r["split"] == "train"
        and (r["subject_key"] not in used or (reuse_development and patient_manifest))
    ]
    if resume and (output / "private/selection.json").exists():
        selection = read(output / "private/selection.json")
        patients, chosen = selection["patients"], selection["records"]
    else:
        patients = sorted({r["subject_key"] for r in candidates}, key=lambda s: digest([seed, s]))[:count]
        if patient_manifest:
            requested = read(patient_manifest)
            patients = requested["patients"] if isinstance(requested, dict) else requested
            eligible = {r["subject_key"] for r in candidates}
            if len(patients) != len(set(patients)) or not set(patients).issubset(eligible):
                raise ValueError("Patient manifest includes duplicates, non-training or previously inspected patients")
        if len(patients) != count:
            raise ValueError("Not enough previously unused training patients")
        chosen = [r for r in candidates if r["subject_key"] in patients]
        selection = {
            "seed": seed,
            "patients": patients,
            "excluded_patients": sorted(used),
            "source_inventory": str(Path(inventory).resolve()),
            "reuse_existing_development_cohort": reuse_development,
            "records": chosen,
        }
    output.mkdir(parents=True, exist_ok=True)
    write(
        output / "private/selection.json",
        selection,
    )
    (output / "private").chmod(0o700)
    cases, sources, missing, failures = [], [], [], []
    for record in chosen:
        check_stop(output)
        path = Path(record["source"]["path"])
        source_hash = sha(path)
        if source_hash != record["source"]["sha256"]:
            raise ValueError("Selected source changed since the upstream inventory")
        rid = "r_" + digest(record["subject_key"])[:12]
        mat = VitalReader(path)
        info = {
            "record_id": rid,
            "source_sha256": source_hash,
            "rows": mat.n,
            "fields": mat.channels(),
            "sampling_rate_hz": 125,
            "sample_index_base": 0,
            "intervals": "half_open",
        }
        try:
            for kind, field in mat.channels().items():
                if config and kind not in config["tasks"]:
                    continue
                if field is None:
                    missing.append(
                        {"record_id": rid, "task": kind, "status": "unusable_for_task", "reason": "missing_channel"}
                    )
                    continue
                normalized = kind != "ABP" and field == kind + "_Raw"
                metadata = []
                for i in range(mat.n):
                    try:
                        m = mat.metadata(i, field, path.stem)
                        if "vital::" + m["subject"] != record["subject_key"]:
                            raise ValueError("Source patient identity disagrees with inventory")
                        metadata.append(m)
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        metadata.append(None)
                        failures.append({"record_id": rid, "task": kind, "row": i, "reason": str(exc)})
                info.setdefault("time_conflicts", {})[kind] = mark_time_conflicts(metadata)
                groups = []
                for m in metadata:
                    if m is None:
                        continue
                    if (
                        not groups
                        or m["row"] != groups[-1][-1]["row"] + 1
                        or not continuous(groups[-1][-1], m, normalized)
                    ):
                        groups.append([])
                    groups[-1].append(m)
                for number, group in enumerate(groups):
                    cid = f"{rid}_{kind}_{number:04d}"
                    x = np.concatenate([mat.array(field, r["row"]) for r in group]).astype(np.float64)
                    t = (
                        np.concatenate([mat.array("T", r["row"]) for r in group]).astype(np.float64)
                        if group[0]["has_time"]
                        else np.arange(len(x)) / 125
                    )
                    folder = output / "public" / cid
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / "input.npz", values=x, time=t)
                    case = {
                        "id": cid,
                        "record_id": rid,
                        "task": kind,
                        "task_id": TASKS[kind]["task_id"],
                        "field": field,
                        "unit": TASKS[kind]["unit"],
                        "fs": 125.0,
                        "n": len(x),
                        "duration_s": len(x) / 125,
                        "source_sha256": source_hash,
                        "rows": [r["row"] for r in group],
                        "row_lengths": [r["n"] for r in group],
                        "normalized_windows": normalized,
                        "has_source_time": group[0]["has_time"],
                        "time_tolerance_s": max(r["time_tolerance"] for r in group),
                        "input_sha256": sha(folder / "input.npz"),
                    }
                    offset, time_conflicts = 0, []
                    for row in group:
                        time_conflicts.extend([[a + offset, b + offset] for a, b in row.get("time_conflicts", [])])
                        offset += row["n"]
                    case["source_time_conflicts"] = time_conflicts
                    write(folder / "source.json", case)
                    cases.append(case)
        finally:
            mat.close()
        sources.append(info)
        write(output / "audit-progress.json", {"records_audited": len(sources), "cases": len(cases)})
    manifest = {
        "schema": "signalmind.cleaning.dataset.v1",
        "seed": seed,
        "patients": len(patients),
        "records": sources,
        "cases": cases,
        "unavailable_tasks": missing,
        "audit_failures": failures,
        "scope_hash": digest(cases),
        "source_inputs_unchanged": True,
        "reference_manifest": None,
        "task_config": config,
    }
    write(output / "dataset_manifest.json", manifest)
    write(
        output / "data_audit.json",
        {
            "patients": len(patients),
            "files": len(sources),
            "cases": len(cases),
            "per_task_cases": dict(Counter(r["task"] for r in cases)),
            "duration_s_by_task": {k: sum(r["duration_s"] for r in cases if r["task"] == k) for k in TASKS},
            "missing": missing,
            "failures": failures,
            "reference_metrics": None,
            "time_conflict_samples_by_task": {
                k: sum(b - a for c in cases if c["task"] == k for a, b in c["source_time_conflicts"]) for k in TASKS
            },
        },
    )
    return manifest
