"""Append-only execution receipts and exact, sparse operator snapshots."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .common import array_hash, changed, digest, read, sha, write


class StopRequested(RuntimeError):
    """A resumable stop at a model/tool/record boundary."""


def check_stop(workspace):
    if (Path(workspace) / "stop.requested").exists():
        raise StopRequested("Stopped at a safe boundary; checkpoint retained")


def freeze_implementation(workspace):
    """Retain the implementation used by future runs, once, alongside its hashes."""
    workspace = Path(workspace)
    target = workspace / "implementation_manifest.json"
    if target.exists():
        return read(target)
    package = Path(__file__).resolve().parents[1]
    files = {}
    for path in package.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json"}:
            relative = path.relative_to(package)
            destination = workspace / "private/frozen_source/tsmcp" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            files[str(relative)] = sha(path)
    manifest = {
        "files": files,
        "python": sys.version,
        "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
    }
    write(target, manifest)
    (workspace / "private").chmod(0o700)
    return manifest


def emit(workspace, event_type, **fields):
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(UTC).isoformat(),
        "type": event_type,
        "actor": fields.pop("actor", "executor"),
        **fields,
    }
    # One journal shared by the parent and numerical worker processes.
    with (root / "events.jsonl").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def local_stats(values, valid, estimated, indices):
    x = np.asarray(values)[indices]
    finite = x[np.isfinite(x)]
    return {
        "samples": len(indices),
        "valid_samples": int(np.asarray(valid)[indices].sum()),
        "eligible_samples": int((np.asarray(valid)[indices] & ~np.asarray(estimated)[indices] & np.isfinite(x)).sum()),
        "estimated_samples": int(np.asarray(estimated)[indices].sum()),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
    }


class StepRecorder:
    def __init__(self, workspace, case, recipe, implementation, parent):
        self.workspace = Path(workspace)
        self.case = case
        self.id = digest([case["input_sha256"], recipe["hash"], implementation, parent, uuid.uuid4().hex])
        self.folder = self.workspace / "steps" / self.id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "schema": "signalmind.operator-trace.v1",
            "trace_id": self.id,
            "case_id": case["id"],
            "recipe_hash": recipe["hash"],
            "implementation_hash": implementation,
            "parent_artifact_id": parent,
            "source_sha256": case["input_sha256"],
            "steps": [],
        }
        write(self.folder / "trace.json", self.manifest)

    def record(self, before, after, operation, ranges, parameters, explanation):
        index = len(self.manifest["steps"])
        altered = (
            (before.values.view(np.uint64) != after.values.view(np.uint64))
            | (before.valid != after.valid)
            | (before.repaired != after.repaired)
        )
        indices = np.flatnonzero(altered)
        path = self.folder / f"{index:04d}.npz"
        # Replacement values, never arithmetic deltas (which can lose precision).
        np.savez_compressed(
            path,
            indices=indices,
            values=after.values[indices],
            valid=after.valid[indices],
            estimated=after.repaired[indices],
        )
        addressed = np.zeros(len(after.values), bool)
        for a, b in ranges:
            addressed[a:b] = True
        selected = np.flatnonzero(addressed)
        step = {
            "index": index,
            "operation": operation,
            "ranges_samples": ranges,
            "parameters": parameters,
            "explanation": explanation,
            "explanation_origin": "recipe_provided" if explanation else "not_provided",
            "patch": path.name,
            "patch_sha256": sha(path),
            "before_hash": array_hash(before.values, before.valid, before.repaired),
            "after_hash": array_hash(after.values, after.valid, after.repaired),
            "changed_samples": int(changed(before.values, after.values).sum()),
            "support_changes": int((before.valid != after.valid).sum()),
            "retest": {
                "before": local_stats(before.values, before.valid, before.repaired, selected),
                "after": local_stats(after.values, after.valid, after.repaired, selected),
                "basis": "executor_computed_local_statistics",
                "reference_verified": False,
            },
            "probe_status": "not_run_at_intermediate_step",
        }
        self.manifest["steps"].append(step)
        write(self.folder / "trace.json", self.manifest)
        emit(
            self.workspace,
            "operator_completed",
            case_id=self.case["id"],
            trace_id=self.id,
            recipe_hash=self.manifest["recipe_hash"],
            step=index,
            operation=operation,
        )

    @property
    def reference(self):
        path = self.folder / "trace.json"
        return {
            "path": str(path.relative_to(self.workspace)),
            "sha256": sha(path),
            "steps": len(self.manifest["steps"]),
        }


def restore_steps(root, trace_reference, values, valid, estimated, through):
    """Reconstruct an actually recorded operator boundary, checking its dependencies."""
    path = Path(root) / trace_reference["path"]
    if sha(path) != trace_reference["sha256"]:
        raise ValueError("Operator trace changed")
    trace = read(path)
    if not 0 <= through < len(trace["steps"]):
        raise ValueError("Unknown operator step")
    values, valid, estimated = values.copy(), valid.copy(), estimated.copy()
    for step in trace["steps"][: through + 1]:
        if array_hash(values, valid, estimated) != step["before_hash"]:
            raise ValueError("Operator input dependency differs")
        patch = path.parent / step["patch"]
        if sha(patch) != step["patch_sha256"]:
            raise ValueError("Operator snapshot changed")
        with np.load(patch, allow_pickle=False) as data:
            ids = data["indices"]
            values[ids], valid[ids], estimated[ids] = data["values"], data["valid"], data["estimated"]
        if array_hash(values, valid, estimated) != step["after_hash"]:
            raise ValueError("Operator output dependency differs")
    return values, valid, estimated
