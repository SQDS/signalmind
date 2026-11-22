"""Dataset workspaces: inventory, record-disjoint examples, and local full-record execution.

The first adapter reads PulseDB Vital MAT files. No provider is imported here.
Private source identities stay in the workspace; model tools expose anonymous examples.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from eval.prepare_vital_processing import FIELDS, FS, PulseMat
from eval.real_processing_helper import save_npz
from tsmcp.pipeline import apply_policy, features, policy_hash, validate_policy


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    paths = (
        "src/tsmcp/processing.py",
        "src/tsmcp/pipeline.py",
        "eval/dataset_workflow.py",
        "eval/prepare_vital_processing.py",
        "eval/real_processing_helper.py",
    )
    return hashlib.sha256("".join(sha(root / p) for p in paths).encode()).hexdigest()


def rank(value):
    return hashlib.sha256(("dataset-workflow-v1:" + value).encode()).hexdigest()


def metadata(record):
    path = Path(record["metadata_path"])
    if sha(path) != record["metadata_sha256"]:
        raise ValueError("Source time metadata changed")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def spans(record):
    ids = metadata(record)["run_id"]
    edges = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def check_source(record):
    stat = Path(record["source"]).stat()
    if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
        raise ValueError("Source size or mtime differs from the dataset inventory")


def load_span(mat, record, start, end):
    rows = [mat.metadata(i) for i in range(start, end)]
    if not rows or not all(row["regular"] and row["subject"] == record["subject"] for row in rows):
        raise ValueError("Source identity or full sample time grid does not match")
    for left, right in zip(rows[:-1], rows[1:], strict=True):
        if (
            left["case"] != right["case"]
            or right["segment"] != left["segment"] + 1
            or abs(right["start"] - left["end"] - 1 / FS) > 1e-6
        ):
            raise ValueError("Requested span crosses an actual case/segment/time discontinuity")
    values = np.vstack([np.concatenate([mat.array(field, i) for i in range(start, end)]) for field in FIELDS.values()])
    time_axis = np.concatenate([mat.array("T", i) for i in range(start, end)])
    if values.shape != (3, len(time_axis)):
        raise ValueError("Source channel shapes do not match the time grid")
    return {"values": values.astype(np.float64), "time": time_axis.astype(np.float64)}


def record_runs(record):
    check_source(record)
    mat = PulseMat(record["source"])
    try:
        for start, end in spans(record):
            yield start, end, load_span(mat, record, start, end)
    finally:
        mat.file.close()


def build_project(source, project, inventory=None, development_records=4, validation_records=4):
    source, project = Path(source).resolve(), Path(project).resolve()
    project.mkdir(parents=True, exist_ok=False)
    private = project / "private"
    private.mkdir(mode=0o700)
    (project / "samples").mkdir()
    records = []
    if inventory:
        original = Path(inventory).resolve()
        for row in read(original):
            path = Path(row["source"]["path"]).resolve()
            if path.parent != source:
                continue
            stat = row["source"]["stat"]
            meta = original.parent / "inventory" / row["subject_key"].replace("::", "__") / "metadata.npz"
            record = {
                "id": "record_" + rank(path.name)[:12],
                "source": str(path),
                "subject": path.stem,
                "split": row["split"],
                "size": stat["size"],
                "mtime_ns": stat["mtime_ns"],
                "source_sha256": row["source"]["sha256"],
                "windows": row["windows"],
                "runs": row["runs"],
                "metadata_path": str(meta),
                "metadata_sha256": next(a["sha256"] for a in row["artifacts"] if a["path"] == "metadata.npz"),
            }
            check_source(record)
            lengths = [b - a for a, b in spans(record)]
            if sum(lengths) != record["windows"] or len(lengths) != record["runs"]:
                raise ValueError("Inventory counts differ from time metadata")
            records.append(record)
        if {r["source"] for r in records} != {str(p.resolve()) for p in source.glob("*.mat")}:
            raise ValueError("Inventory does not exactly cover this source directory")
    else:
        # Portable first scan: inspect source metadata once, retaining full-grid checks.
        meta_dir = private / "metadata"
        meta_dir.mkdir()
        for path in sorted(source.glob("*.mat")):
            mat = PulseMat(path)
            try:
                rows = [mat.metadata(i) for i in range(mat.n)]
            finally:
                mat.file.close()
            if not all(r["regular"] and r["subject"] == path.stem for r in rows):
                raise ValueError("Source has an irregular time grid or inconsistent identity")
            breaks = [True] + [
                a["case"] != b["case"] or b["segment"] != a["segment"] + 1 or abs(b["start"] - a["end"] - 1 / FS) > 1e-6
                for a, b in zip(rows[:-1], rows[1:], strict=True)
            ]
            rid = "record_" + rank(path.name)[:12]
            meta = meta_dir / (rid + ".npz")
            save_npz(meta, run_id=np.cumsum(breaks, dtype=np.int32) - 1)
            stat = path.stat()
            records.append(
                {
                    "id": rid,
                    "source": str(path),
                    "subject": path.stem,
                    "split": "unspecified",
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "source_sha256": sha(path),
                    "windows": len(rows),
                    "runs": sum(breaks),
                    "metadata_path": str(meta),
                    "metadata_sha256": sha(meta),
                }
            )
    eligible = sorted([r for r in records if r["split"] in {"train", "unspecified"}], key=lambda r: r["id"])
    if len(eligible) < development_records + validation_records:
        raise ValueError("Not enough separate development/validation records in the allowed pool")
    dev, val = eligible[:development_records], eligible[development_records : development_records + validation_records]
    manifest = {
        "schema_version": "signalmind.dataset-workspace.v0.1",
        "adapter": "pulsedb_vital",
        "source": str(source),
        "records": records,
        "development_records": [r["id"] for r in dev],
        "validation_records": [r["id"] for r in val],
        "channels": list(FIELDS),
        "fs_hz": FS,
        "units": ["source_units", "source_units", "mmHg"],
    }
    write(private / "manifest.json", manifest)
    samples = []
    for role, selected in (("development", dev), ("validation", val)):
        for record in selected:
            ranges = spans(record)
            # Examine evenly spaced positions in stored data, then keep their real continuous neighborhood.
            selected_ranges = set()
            for pos in (0, record["windows"] // 2, record["windows"] - 1):
                a, b = next((a, b) for a, b in ranges if a <= pos < b)
                start = max(a, min(pos - 3, b - 6))
                selected_ranges.add((start, min(b, start + 6)))
            mat = PulseMat(record["source"])
            try:
                for start, end in sorted(selected_ranges):
                    sid = f"{record['id']}_sample_{start:06d}"
                    values = load_span(mat, record, start, end)
                    path = project / "samples" / (sid + ".npz")
                    save_npz(path, **values)
                    samples.append(
                        {
                            "id": sid,
                            "record_id": record["id"],
                            "role": role,
                            "rows": [start, end],
                            "samples": len(values["time"]),
                            "sha256": sha(path),
                            "features": {
                                kind: features(x, FS) for kind, x in zip(FIELDS, values["values"], strict=True)
                            },
                        }
                    )
            finally:
                mat.file.close()
    write(private / "samples.json", samples)
    summary = {
        "files": len(records),
        "stored_10s_windows": sum(r["windows"] for r in records),
        "stored_hours": sum(r["windows"] for r in records) / 360,
        "continuous_runs": sum(r["runs"] for r in records),
        "source_bytes": sum(r["size"] for r in records),
        "channel_kinds": list(FIELDS),
        "units": {"ECG": "source_units", "PPG": "source_units", "ABP": "mmHg"},
        "unit_note": "ECG/PPG physical units are not confirmed by the source MAT fields. Do not assume mV or cross-record amplitude calibration.",
        "device_saturation_flags_available": False,
        "fs_hz": FS,
        "sample_roles": dict(Counter(s["role"] for s in samples)),
        "development_records": len(dev),
        "validation_records": len(val),
        "selection": "Deterministic record-disjoint selection from training/unspecified pool; three time positions per record.",
        "source_waveforms_fully_scanned": False,
        "new_model_calls": 0,
        "scope_note": "Whole-directory metadata inventory; waveform examples are a subset. Batch execution is separate.",
    }
    write(project / "inventory-summary.json", summary)
    return summary


def process_arrays(data, policy, source_id):
    rows, diagnostics = [], {}
    for kind, x in zip(FIELDS, data["values"], strict=True):
        state, changed, diag = apply_policy(x, FS, kind, "mmHg" if kind == "ABP" else "source_units", policy, source_id)
        rows.append((state.values, state.valid, state.artifact, state.repaired, changed))
        diagnostics[kind] = diag
    arrays = {
        name: np.stack([row[i] for row in rows])
        for i, name in enumerate(("values", "valid_mask", "artifact_mask", "repaired_mask", "modified_mask"))
    }
    arrays["time"] = data["time"]
    return arrays, diagnostics


def summarize(rows):
    counts = Counter()
    examples = []
    for row in rows:
        for kind, diag in row["channels"].items():
            counts["channel_segments"] += 1
            for field in (
                "samples",
                "changed_samples",
                "input_invalid_samples",
                "output_invalid_samples",
                "repaired_samples",
            ):
                counts[field] += diag[field]
            counts["unchanged_channel_segments"] += diag["changed_samples"] == 0
            examples.append(
                {
                    "id": row["id"],
                    "channel": kind,
                    "changed_fraction": diag["changed_samples"] / diag["samples"],
                    "valid_fraction": diag["valid_fraction"],
                    "mean_shift": diag["common_support_mean_shift"],
                    "skipped_ranges": sum(len(r["skipped"]) for r in diag["rule_trace"]),
                }
            )
    examples.sort(key=lambda r: 1 - r["valid_fraction"] + r["changed_fraction"], reverse=True)
    return {
        **dict(counts),
        "review_examples": examples[:8],
        "quality_verdict": None,
        "note": "Diagnostics and coverage only; no clean-reference or clinical accuracy score.",
    }


def trial(project, policy, role="development"):
    project = Path(project)
    validate_policy(policy)
    pid = policy_hash(policy)
    folder = project / "trials" / pid / role
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in read(project / "private/samples.json"):
        if sample["role"] != role:
            continue
        path = project / "samples" / (sample["id"] + ".npz")
        if sha(path) != sample["sha256"]:
            raise ValueError("Sample content changed")
        with np.load(path, allow_pickle=False) as data:
            arrays, diag = process_arrays(data, policy, sample["id"])
        save_npz(folder / (sample["id"] + ".npz"), **arrays)
        rows.append({"id": sample["id"], "record_id": sample["record_id"], "channels": diag})
    result = {
        "policy_sha256": pid,
        "implementation_sha256": implementation_hash(),
        "role": role,
        "summary": summarize(rows),
        "cases": rows,
    }
    write(folder / "report.json", result)
    write(folder.parent / "policy.json", policy)
    return result


def run_batch(project, policy, record_ids=None, output_name="batch"):
    """Process every stored row in selected records, one continuous run at a time; resume exact outputs."""
    project = Path(project)
    validate_policy(policy)
    pid = policy_hash(policy)
    code_hash = implementation_hash()
    freeze = project / "policy-freeze.json"
    if freeze.exists() and read(freeze)["policy_sha256"] == pid:
        if read(freeze)["implementation_sha256"] != code_hash:
            raise ValueError("Processing implementation changed after policy trial/freeze")
    manifest = read(project / "private/manifest.json")
    chosen = [r for r in manifest["records"] if record_ids is None or r["id"] in record_ids]
    if record_ids is not None and {r["id"] for r in chosen} != set(record_ids):
        raise ValueError("Unknown record id in batch selection")
    folder = project / output_name / pid
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "policy.json", policy)
    records, failed = [], []
    started = time.monotonic()
    for record in chosen:
        target = folder / record["id"]
        target.mkdir(exist_ok=True)
        complete = target / "complete.json"
        try:
            check_source(record)
            if complete.exists():
                result = read(complete)
                if (
                    result["policy_sha256"] != pid
                    or result["source_sha256"] != record["source_sha256"]
                    or result["implementation_sha256"] != code_hash
                ):
                    raise ValueError("Completed record provenance differs")
                if not all(sha(target / name) == digest for name, digest in result["artifacts"].items()):
                    raise ValueError("Saved batch artifact changed")
                records.append(result)
                continue
            if sha(record["source"]) != record["source_sha256"]:
                raise ValueError("Source content changed")
            rows, artifacts, seen = [], {}, 0
            for start, end, data in record_runs(record):
                sid = f"run_{start:06d}_{end:06d}"
                arrays, diag = process_arrays(data, policy, record["id"] + "_" + sid)
                path = target / (sid + ".npz")
                save_npz(path, **arrays)
                artifacts[path.name] = sha(path)
                rows.append({"id": record["id"] + "_" + sid, "rows": [start, end], "channels": diag})
                seen += end - start
            if seen != record["windows"] or sha(record["source"]) != record["source_sha256"]:
                raise ValueError("Batch coverage incomplete or source changed during execution")
            write(target / "operations.json", rows)
            artifacts["operations.json"] = sha(target / "operations.json")
            result = {
                "record_id": record["id"],
                "policy_sha256": pid,
                "implementation_sha256": code_hash,
                "source_sha256": record["source_sha256"],
                "stored_rows_processed": seen,
                "continuous_runs": len(rows),
                "summary": summarize(rows),
                "artifacts": artifacts,
            }
            write(complete, result)
            records.append(result)
        except (ValueError, OSError) as exc:
            failed.append({"record_id": record["id"], "error": str(exc)})
        print(
            json.dumps(
                {"completed_records": len(records), "failed_records": len(failed), "selected_records": len(chosen)}
            ),
            flush=True,
        )
    result = {
        "policy_sha256": pid,
        "implementation_sha256": code_hash,
        "selected_records": len(chosen),
        "completed_records": len(records),
        "failed": failed,
        "stored_rows_processed": sum(r["stored_rows_processed"] for r in records),
        "elapsed_s": round(time.monotonic() - started, 3),
        "new_model_calls": 0,
        "records": records,
        "source_directory_records": len(manifest["records"]),
        "scope": "entire_directory" if len(chosen) == len(manifest["records"]) else "selected_complete_records",
        "quality_verdict": None,
    }
    write(folder / "report.json", result)
    return result
