"""Opened LTAFDB development records, with the frozen Part 1 + C1 definitions.

Run with --opened-cases pointing at the existing p2long/p4long JSONL files, commit
eligibility, then build with eval/build_cases.py --suite p7dev --p7dev.
Only explicitly allowed LTAFDB inputs are opened; no detector/model is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.tasks.confirm import annotation_facts as interval_facts
from eval.tasks.confirm_shdb import rhythm_intervals

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ltafdb"
ELIGIBILITY = Path(__file__).with_name("p7dev_eligibility.json")
ENDPOINTS = ("longest_af", "af_burden")
QUESTIONS = {
    "longest_af": "When does the longest episode of atrial fibrillation or flutter begin? Count AF or flutter separated "
    "by less than 30 seconds of other rhythm as one continuous episode. Answer in seconds from the start.",
    "af_burden": "For what percentage of the total recording time is the heart in atrial fibrillation or flutter?",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def opened_sources(opened_cases: Path) -> dict:
    """Read only the two opened suites, the dev split and former-fresh inventory."""
    split = ROOT / "eval/tasks/ltafdb_split.json"
    inventory = ROOT / "eval/results/af-continuity/fresh-inventory.json"
    paths = {"dev": split, "former_fresh": inventory}
    sources = {
        "dev": json.loads(split.read_text())["dev"],
        "former_fresh": [row["record"] for row in json.loads(inventory.read_text())["records"]],
    }
    repeated_rows = {}
    for suite in ("p2long", "p4long"):
        path = opened_cases / f"{suite}.answers.jsonl"
        paths[suite] = path
        # Only these suites' ltaf-prefixed cases are LTAFDB (the rest are synthetic).
        records = [
            str(row["answer"]["record"])
            for line in path.read_text().splitlines()
            if line.strip() and (row := json.loads(line))["case_id"].startswith("ltaf")
        ]
        if not records:
            raise ValueError(f"No LTAFDB records in opened suite {suite}")
        sources[suite] = sorted(set(records))
        repeated_rows[suite] = sorted(rec for rec, count in Counter(records).items() if count > 1)
    for source, records in sources.items():
        if len(records) != len(set(records)) or any(
            not isinstance(r, str) or not r.isascii() or not r.isdigit() for r in records
        ):
            raise ValueError(f"Invalid or duplicate record ids in {source}")
        sources[source] = sorted(records)
    if len(sources["dev"]) != 8 or len(sources["former_fresh"]) != 38:
        raise ValueError("Opened dev/former-fresh inventory differs from the task")
    membership = defaultdict(list)
    for source, records in sources.items():
        for rec in records:
            membership[rec].append(source)
    return {
        "opened_cases": str(opened_cases.resolve()),
        "records_by_source": sources,
        "input_sha256": {source: sha256(path) for source, path in paths.items()},
        "membership": dict(sorted(membership.items())),
        "repeated_record_rows_within_opened_suites": repeated_rows,
        "records_in_multiple_sources": {rec: labels for rec, labels in sorted(membership.items()) if len(labels) > 1},
    }


def annotation_facts(header, annotation) -> dict:
    """Whole native 128 Hz file; expert ATR rhythm changes are the only truth."""
    if header.fs != 128 or header.n_sig != 2 or header.fmt != ["16", "16"] or header.sig_len <= 0:
        raise ValueError("Unexpected LTAFDB sampling rate/channel layout/format")
    intervals = rhythm_intervals(annotation.sample, annotation.aux_note, header.sig_len)
    facts = interval_facts(intervals, header.sig_len, header.fs)
    reasons = {kind: [] for kind in ENDPOINTS}
    if facts["dur_s"] < 72000:
        reasons = {kind: ["shorter_than_20_hours"] for kind in ENDPOINTS}
    else:
        longest, burden = facts["longest_af"], facts["af_percent"]
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
    return {
        "fs": header.fs,
        "n_samples": header.sig_len,
        **facts,
        "eligible": {kind: not reasons[kind] for kind in ENDPOINTS},
        "exclusion_reasons": reasons,
    }


def collect_eligibility(
    data_dir: Path = DATA, *, opened_cases: Path = ROOT / "eval/cases", case_namespace: str | None = None
) -> dict:
    import wfdb

    sources = opened_sources(opened_cases)
    namespace = uuid.UUID(case_namespace) if case_namespace else uuid.uuid4()
    records, hashes = [], {}
    for rec, labels in sources["membership"].items():
        # No glob over data: unopened records and other databases are never read.
        for ext in ("hea", "atr", "dat"):
            path = data_dir / f"{rec}.{ext}"
            hashes[path.name] = sha256(path)
        header = wfdb.rdheader(str(data_dir / rec))
        if header.file_name != [f"{rec}.dat"] * 2 or header.units != ["mV"] * 2:
            raise ValueError(f"Unexpected LTAFDB signal file/units: {rec}")
        if (data_dir / f"{rec}.dat").stat().st_size != header.sig_len * 4:
            raise ValueError(f"Incomplete LTAFDB waveform: {rec}")
        annotation = wfdb.rdann(str(data_dir / rec), "atr")
        facts = annotation_facts(header, annotation)
        records.append(
            {
                "database": "ltafdb",
                "record": rec,
                "patient_id": rec,
                "sources": labels,
                **facts,
                "case_ids": {
                    kind: uuid.uuid5(namespace, f"p7dev/ltafdb/{rec}/{kind}").hex
                    for kind in ENDPOINTS
                    if facts["eligible"][kind]
                },
            }
        )
    dat_groups = defaultdict(list)
    for row in records:
        dat_groups[hashes[f"{row['record']}.dat"]].append(row["record"])
    duplicates = [recs for recs in dat_groups.values() if len(recs) > 1]
    if duplicates:
        raise ValueError(f"Duplicate source waveform records: {duplicates}")
    counts = {}
    for source in (*sources["records_by_source"], "total"):
        subset = [row for row in records if source == "total" or source in row["sources"]]
        counts[source] = {
            "records": len(subset),
            "selected_records": sum(any(row["eligible"].values()) for row in subset),
            **{
                kind: {
                    "eligible": sum(row["eligible"][kind] for row in subset),
                    "excluded": sum(not row["eligible"][kind] for row in subset),
                    "exclusion_reasons_nonexclusive": dict(
                        sorted(Counter(reason for row in subset for reason in row["exclusion_reasons"][kind]).items())
                    ),
                }
                for kind in ENDPOINTS
            },
        }
    return {
        "schema_version": 1,
        "suite": "p7dev",
        "case_namespace": str(namespace),
        "specification": "confirmation-set.md frozen Part 1 + Amendment C1; p7dev.md allowed opened LTAFDB only",
        "rules": {
            "truth": ".atr only; AF=(AFIB or (AFL; initial unknown time is not AF",
            "segment": "one whole distributed file >=72000 s; no resampling, cutting or concatenation",
            "lead": "channel 0, native 128 Hz, mV",
            "merge_gap_strictly_below_s": 30,
            "tie": "earliest onset",
            "burden": "100 * raw annotated AF samples / whole-file sample count; merged gaps excluded",
            "longest_af": "merged duration >300 s, onset >=120 s, burden <90%",
            "af_burden": "5% < burden <95%",
            "grading": {"longest_af_tolerance_s": 60, "af_burden_tolerance_pp": 5},
            "independence": "one record per LTAFDB subject; one separate case/session per endpoint",
        },
        "sources": sources,
        "identity_check": {
            "unique_records": len(records),
            "unique_patients": len({row["patient_id"] for row in records}),
            "duplicate_source_waveforms": duplicates,
        },
        "data_hashes": {"algorithm": "sha256", "files": hashes},
        "questions": QUESTIONS,
        "counts": counts,
        "records": records,
        "no_detector_or_model_used": True,
    }


def require_committed(path: Path) -> None:
    rel = path.resolve().relative_to(ROOT).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=False)
    if result.returncode or result.stdout != path.read_bytes():
        raise ValueError("Commit p7dev_eligibility.json unchanged before building cases")


def prepare_p7dev(data_dir: Path = DATA, eligibility_path: Path = ELIGIBILITY) -> dict:
    require_committed(eligibility_path)
    frozen = json.loads(eligibility_path.read_text())
    actual = collect_eligibility(
        data_dir,
        opened_cases=Path(frozen["sources"]["opened_cases"]),
        case_namespace=frozen["case_namespace"],
    )
    if actual != frozen:
        raise ValueError("P7 development inputs or annotation facts differ from committed eligibility")
    return frozen


def read_signal(data_dir: Path, row: dict) -> np.ndarray:
    """Copy the first lead in bounded chunks, preserving the complete file."""
    import wfdb

    x = np.empty(row["n_samples"], dtype=np.float32)
    for start in range(0, x.size, 1_000_000):
        end = min(start + 1_000_000, x.size)
        chunk = wfdb.rdrecord(str(data_dir / row["record"]), sampfrom=start, sampto=end, channels=[0])
        if chunk.fs != row["fs"] or chunk.p_signal.shape != (end - start, 1):
            raise ValueError("Waveform layout differs from frozen header")
        x[start:end] = chunk.p_signal[:, 0]
    return x


def build_p7dev(frozen: dict, data_dir: Path = DATA):
    for row in frozen["records"]:
        kinds = [kind for kind in ENDPOINTS if row["eligible"][kind]]
        if not kinds:
            continue
        x = read_signal(data_dir, row)
        if x.shape != (row["n_samples"],) or x.dtype != np.float32:
            raise ValueError("Waveform layout differs from frozen segment metadata")
        meta = {key: row[key] for key in ("database", "patient_id", "record", "dur_s", "sources")}
        for kind in kinds:
            answer = (
                {"t": row["longest_af"]["start"], "longest_dur_s": row["longest_af"]["dur_s"], **meta}
                if kind == "longest_af"
                else {"percent": row["af_percent"], **meta}
            )
            yield row["case_ids"][kind], kind, "-", x, row["fs"], answer, QUESTIONS[kind]


def write_manifest(suite_dir: Path, frozen: dict, questions: list[dict], answers: list[dict]) -> dict:
    """Audit public/private separation and hash every generated case for overlap checks."""
    expected = {cid: (row, kind) for row in frozen["records"] for kind, cid in row["case_ids"].items()}
    qmap = {row["case_id"]: row for row in questions}
    amap = {row["case_id"]: row for row in answers}
    if (
        len(qmap) != len(questions)
        or len(amap) != len(answers)
        or set(expected) != set(qmap)
        or set(expected) != set(amap)
        or {p.name for p in suite_dir.iterdir()} != set(expected)
    ):
        raise ValueError("Case ids differ from committed eligibility or are duplicated")
    cases = []
    for cid, (row, kind) in expected.items():
        directory = suite_dir / cid
        if {p.name for p in directory.iterdir()} != {"signal.npy", "visible.json"}:
            raise ValueError("Case directory must contain only signal.npy and visible.json")
        signal = directory / "signal.npy"
        x = np.load(signal, mmap_mode="r", allow_pickle=False)
        if x.shape != (row["n_samples"],) or x.dtype != np.float32:
            raise ValueError("Case does not preserve the complete native first lead")
        visible = json.loads((directory / "visible.json").read_text())
        if visible != {
            "file": "signal.npy",
            "dur_s": round(row["dur_s"], 3),
            "channels": [{"name": "ch0", "unit": "mV", "fs": row["fs"]}],
        } or qmap[cid] != {"case_id": cid, "qtype": kind, "question": QUESTIONS[kind], "source": "ecg-long"}:
            raise ValueError("Public metadata differs from frozen question/layout")
        meta = {key: row[key] for key in ("database", "patient_id", "record", "dur_s", "sources")}
        truth = (
            {"t": row["longest_af"]["start"], "longest_dur_s": row["longest_af"]["dur_s"], **meta}
            if kind == "longest_af"
            else {"percent": row["af_percent"], **meta}
        )
        if amap[cid] != {"case_id": cid, "qtype": kind, "variant": "-", "answer": truth}:
            raise ValueError("Answer differs from committed annotation truth")
        cases.append(
            {
                "case_id": cid,
                "qtype": kind,
                **meta,
                "n_samples": row["n_samples"],
                "fs": row["fs"],
                "signal_sha256": sha256(signal),
                "signal_bytes": signal.stat().st_size,
                "visible_sha256": sha256(directory / "visible.json"),
            }
        )
    for kind in ENDPOINTS:
        patients = [row["patient_id"] for row in cases if row["qtype"] == kind]
        if len(patients) != len(set(patients)):
            raise ValueError("Repeated patient within an endpoint")
    manifest = {
        "schema_version": 1,
        "suite": "p7dev",
        "eligibility_sha256": sha256(ELIGIBILITY),
        "eligibility_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "artifacts": {
            suffix: sha256(suite_dir.parent / f"p7dev.{suffix}.jsonl") for suffix in ("questions", "answers")
        },
        "counts": {kind: sum(row["qtype"] == kind for row in cases) for kind in ENDPOINTS},
        "case_layout_verified": True,
        "unique_patient_per_endpoint": True,
        "no_detector_or_model_used": True,
        "cases": cases,
    }
    (suite_dir.parent / "p7dev.manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opened-cases", type=Path, default=ROOT / "eval/cases")
    args = parser.parse_args()
    namespace = json.loads(ELIGIBILITY.read_text())["case_namespace"] if ELIGIBILITY.exists() else None
    evidence = collect_eligibility(opened_cases=args.opened_cases, case_namespace=namespace)
    ELIGIBILITY.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence["counts"], indent=2))
