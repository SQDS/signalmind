"""Small shared artifact contracts. Numerical quality is never inferred from file validity."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

TASKS = {
    "ECG": {"task_id": "ecg_waveform_for_rr", "unit": "source_units", "field": "ECG_Record"},
    "PPG": {"task_id": "ppg_waveform_for_pulse", "unit": "source_units", "field": "PPG_Record"},
    "ABP": {"task_id": "abp_waveform_for_pressure", "unit": "mmHg", "field": "ABP_Raw"},
}


@contextmanager
def locked(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recipe_input_key(case):
    if "patient_reference" in case or "screen_cores" in case:
        return digest([case["input_sha256"], case.get("patient_reference"), case.get("screen_cores")])
    return case["input_sha256"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def changed(before, after):
    return ~((before == after) | (np.isnan(before) & np.isnan(after)))


def time_support(case):
    mask = np.ones(case["n"], bool)
    for a, b in case.get("source_time_conflicts", []):
        mask[a:b] = False
    return mask


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def no_reference():
    return {
        "Y": None,
        "C": None,
        "Rbad": None,
        "goodloss": None,
        "evidence_basis": "unlabeled_diagnostics",
        "reason": "No independent complete task reference provided",
        "reference_verified": False,
    }


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "cleaning").rglob("*.py"))
    files += sorted((root / "cleaning" / "vendor").glob("*.json"))
    files += sorted((root / "followup").glob("*.py"))
    files += [
        root / "processing.py",
        root / "pipeline.py",
        root / "workbench" / "datasets.py",
        root / "workbench" / "runner.py",
    ]
    files += sorted((root / "inspection").glob("*.yaml")) + sorted((root / "inspection").glob("*.py"))
    files += sorted((root / "inspection" / "schemas").glob("*.json"))
    files += sorted((root / "plugins").rglob("*.py"))
    files += sorted((root / "core").rglob("*.py"))
    return digest({str(p.relative_to(root)): sha(p) for p in files})
