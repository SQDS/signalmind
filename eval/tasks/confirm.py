"""Sealed, annotation-only SHDB-AF / IRIDIA-AF confirmation suite.

Freeze eligibility with ``.venv/bin/python -m eval.tasks.confirm`` and commit the
result before running ``eval/build_cases.py --suite p6confirm --confirm``.
No detector or model is imported or called here.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.tasks import confirm_iridia, confirm_shdb

ROOT = Path(__file__).resolve().parents[2]
ELIGIBILITY = Path(__file__).with_name("confirm_eligibility.json")
ENDPOINTS = ("longest_af", "af_burden")
FS = 200
MIN_SEGMENT_SAMPLES = 72000 * FS
QUESTIONS = {
    "longest_af": "When does the longest episode of atrial fibrillation or flutter begin? Count AF or flutter separated "
    "by less than 30 seconds of other rhythm as one continuous episode. Answer in seconds from the start.",
    "af_burden": "For what percentage of the total recording time is the heart in atrial fibrillation or flutter?",
}


def annotation_facts(intervals: list, n_samples: int, fs: float = FS) -> dict:
    """Use sample coordinates, union raw AF, then merge only gaps <30 seconds."""
    if n_samples <= 0 or fs <= 0 or not np.isfinite(fs):
        raise ValueError("Invalid annotation duration or sampling rate")
    raw = []
    for start, end in sorted(intervals):
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= n_samples):
            raise ValueError("AF interval outside segment or invalid sample coordinates")
        if raw and start <= raw[-1][1]:
            raw[-1][1] = max(raw[-1][1], end)
        else:
            raw.append([start, end])
    merged = []
    for start, end in raw:
        if merged and start - merged[-1][1] < 30 * fs:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    longest = max(merged, key=lambda pair: pair[1] - pair[0]) if merged else None
    af_samples = sum(end - start for start, end in raw)
    return {
        "dur_s": n_samples / fs,
        "af_samples": af_samples,
        "af_s": af_samples / fs,
        "af_percent": 100 * af_samples / n_samples,
        "n_af_segments": len(raw),
        "n_merged_af_episodes": len(merged),
        "longest_af": None
        if longest is None
        else {"start": longest[0] / fs, "end": longest[1] / fs, "dur_s": (longest[1] - longest[0]) / fs},
    }


def endpoint_reasons(row: dict) -> dict:
    """Each endpoint is independent; record every failed frozen criterion."""
    common = []
    if not row["annotation_available"]:
        common.append("no_manual_rhythm_annotations")
    if row["n_samples"] < MIN_SEGMENT_SAMPLES:
        common.append("shorter_than_20_hours")
    if common:
        return {kind: list(common) for kind in ENDPOINTS}
    reasons = {kind: [] for kind in ENDPOINTS}
    longest, burden = row["longest_af"], row["af_percent"]
    if longest is None:
        reasons["longest_af"].append("no_annotated_af")
    else:
        if longest["dur_s"] <= 300:
            reasons["longest_af"].append("longest_duration_not_above_300_seconds")
        if longest["start"] < 120:
            reasons["longest_af"].append("longest_onset_before_120_seconds")
    if burden >= 90:
        reasons["longest_af"].append("burden_not_below_90_percent")
    if burden <= 5:
        reasons["af_burden"].append("burden_not_above_5_percent")
    if burden >= 95:
        reasons["af_burden"].append("burden_not_below_95_percent")
    return reasons


def assign_patients(segments: list[dict], case_namespace: uuid.UUID = uuid.NAMESPACE_URL) -> list[dict]:
    """First eligible record id / numeric file index, independently per endpoint."""
    groups = defaultdict(list)
    for segment in segments:
        row = dict(segment)
        if row["fs"] != FS:
            raise ValueError("Confirmation signals must be sampled at 200 Hz")
        row.update(annotation_facts(row["af_intervals"], row["n_samples"], row["fs"]))
        row["exclusion_reasons"] = endpoint_reasons(row)
        row["eligible"] = {kind: not row["exclusion_reasons"][kind] for kind in ENDPOINTS}
        groups[(row["database"], row["patient_id"])].append(row)
    patients = []
    for (database, patient_id), rows in sorted(groups.items()):
        rows.sort(key=lambda row: (row["record"], row["file_index"]))
        keys = [row["segment"] for row in rows]
        order = [(row["record"], row["file_index"]) for row in rows]
        if len(keys) != len(set(keys)) or len(order) != len(set(order)):
            raise ValueError("Duplicate patient segment or record/file index")
        selected = {kind: next((row["segment"] for row in rows if row["eligible"][kind]), None) for kind in ENDPOINTS}
        cases = {
            kind: uuid.uuid5(case_namespace, f"p6confirm/{database}/{patient_id}/{kind}").hex
            for kind in ENDPOINTS
            if selected[kind] is not None
        }
        patients.append(
            {
                "database": database,
                "patient_id": patient_id,
                "segments": rows,
                "selected": selected,
                "case_ids": cases,
                "eligible": {kind: selected[kind] is not None for kind in ENDPOINTS},
                "exclusion_reasons": {
                    kind: sorted({reason for row in rows for reason in row["exclusion_reasons"][kind]})
                    if selected[kind] is None
                    else []
                    for kind in ENDPOINTS
                },
            }
        )
    return patients


def summarize(patients: list[dict]) -> dict:
    counts = {}
    for database in ("shdb-af", "iridia-af", "total"):
        subset = [p for p in patients if database == "total" or p["database"] == database]
        counts[database] = {"patients": len(subset)}
        for kind in ENDPOINTS:
            eligible = sum(p["eligible"][kind] for p in subset)
            counts[database][kind] = {
                "eligible": eligible,
                "excluded": len(subset) - eligible,
                "exclusion_reasons_nonexclusive": dict(
                    sorted(Counter(reason for p in subset for reason in p["exclusion_reasons"][kind]).items())
                ),
            }
            if database == "total":
                counts[database][kind].update(target=60, meets_target=eligible >= 60)
    return counts


def collect_eligibility(data_root: Path = ROOT / "data", *, case_namespace: str | None = None) -> dict:
    """Verify bytes before reading metadata/annotations; never decode ECG values."""
    hashes = {
        "shdb-af": confirm_shdb.verify_shdb(data_root / "shdb-af"),
        "iridia-af": confirm_iridia.verify_iridia(data_root / "iridia-af"),
    }
    confirm_iridia.extract_iridia_archive(
        data_root / "iridia-af" / confirm_iridia.ZIP_NAME, data_root / "iridia-af" / "records"
    )
    shdb = confirm_shdb.inspect_shdb(data_root / "shdb-af")
    iridia = confirm_iridia.inspect_iridia(data_root / "iridia-af")
    # The namespace is private along with eligibility, so IDs cannot be enumerated
    # from the public patient metadata. Reuse it when rechecking frozen facts.
    namespace = uuid.UUID(case_namespace) if case_namespace is not None else uuid.uuid4()
    patients = assign_patients(shdb["segments"] + iridia["segments"], namespace)
    return {
        "schema_version": 1,
        "suite": "p6confirm",
        "case_namespace": str(namespace),
        "frozen_spec_commit": "a48abca",
        "amendment": "C1",
        "amendment_commit": "89b83fb",
        "rules": {
            "unit": "one whole distributed file per patient per endpoint; duration >= 72000 s; no concatenation or cutting",
            "minimum_segment_duration_s": MIN_SEGMENT_SAMPLES / FS,
            "scan_order": "record id, then numeric file index; first eligible separately per endpoint",
            "af_labels": {"shdb-af": ["(AFIB", "(AFL"], "iridia-af": "manually annotated AF episodes"},
            "merge_gap_strictly_below_s": 30,
            "burden": "raw annotated AF time / whole segment duration * 100; merged gaps excluded",
            "tie": "earliest onset",
            "longest_af": "merged duration > 300 s, onset >= 120 s, burden < 90%",
            "af_burden": "5% < burden < 95%",
            "leads": {"shdb-af": "channel 0", "iridia-af": "lead II (column 1)"},
        },
        "questions": QUESTIONS,
        "data_hashes": hashes,
        "formats": {"shdb-af": shdb["formats"], "iridia-af": iridia["formats"]},
        "counts": summarize(patients),
        "patients": patients,
        "no_detector_or_model_used": True,
    }


def require_committed(path: Path) -> None:
    """Check HEAD, not the potentially read-only worktree index."""
    rel = path.resolve().relative_to(ROOT).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=False)
    if result.returncode or result.stdout != path.read_bytes():
        raise ValueError("Commit confirm_eligibility.json unchanged before building cases")


def prepare_confirm(data_root: Path = ROOT / "data", eligibility_path: Path = ELIGIBILITY) -> dict:
    """Fail before changing case directories if integrity or frozen truth differs."""
    require_committed(eligibility_path)
    frozen = json.loads(eligibility_path.read_text())
    if collect_eligibility(data_root, case_namespace=frozen["case_namespace"]) != frozen:
        raise ValueError("Confirmation data or annotation facts differ from the frozen eligibility JSON")
    return frozen


def read_signal(row: dict) -> np.ndarray:
    if row["database"] == "shdb-af":
        return confirm_shdb.read_signal(ROOT / "data" / "shdb-af", row)
    if row["database"] == "iridia-af":
        return confirm_iridia.read_lead_ii(ROOT / "data" / "iridia-af", row)
    raise ValueError("Unknown confirmation database")


def build_confirm(frozen: dict):
    """Decode only frozen selections, emitting separate sessions per endpoint."""
    for patient in frozen["patients"]:
        for row in patient["segments"]:
            kinds = [kind for kind in ENDPOINTS if patient["selected"][kind] == row["segment"]]
            if not kinds:
                continue
            x = read_signal(row)
            if x.shape != (row["n_samples"],) or x.dtype != np.float32:
                raise ValueError("Waveform layout differs from frozen segment metadata")
            meta = {
                "database": row["database"],
                "patient_id": row["patient_id"],
                "record": row["record"],
                "segment": row["segment"],
                "file_index": row["file_index"],
                "start_sample": row["start_sample"],
                "dur_s": row["dur_s"],
            }
            for kind in kinds:
                answer = (
                    {"t": row["longest_af"]["start"], "longest_dur_s": row["longest_af"]["dur_s"], **meta}
                    if kind == "longest_af"
                    else {"percent": row["af_percent"], **meta}
                )
                yield patient["case_ids"][kind], kind, "-", x, FS, answer, QUESTIONS[kind]


if __name__ == "__main__":
    evidence = collect_eligibility()
    ELIGIBILITY.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence["counts"], indent=2))
