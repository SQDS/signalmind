"""Compare development cases with confirmation identity/hash metadata only.

This module never opens confirmation questions, answers, case directories, or
signals. A supplied signal manifest has the shape ``{"suite": "p6confirm",
"cases": [{"case_id": "...", "signal_sha256": "..."}, ...]}``. Source-container
checksums in eligibility are deliberately not treated as emitted signal hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def _digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid signal SHA256")
    return value


def load_confirmation_index(eligibility_path: Path, manifest_path: Path | None = None) -> dict:
    """Retain only identities and hashes from the authorized confirmation files.

    Missing signal hashes remain explicit. They must be supplied by an existing
    manifest; this function cannot obtain them by reading confirmation signals.
    """
    eligibility_bytes = Path(eligibility_path).read_bytes()
    eligibility = json.loads(eligibility_bytes)
    if eligibility.get("suite") != "p6confirm":
        raise ValueError("Expected p6confirm eligibility")
    records = set()
    cases = {}
    for patient in eligibility["patients"]:
        database = patient["database"]
        segments = {row["segment"]: row for row in patient["segments"]}
        if len(segments) != len(patient["segments"]):
            raise ValueError("Duplicate confirmation segment identity")
        records.update((database, row["record"]) for row in segments.values())
        for endpoint, case_id in patient["case_ids"].items():
            if case_id in cases:
                raise ValueError("Duplicate confirmation case identity")
            row = segments[patient["selected"][endpoint]]
            if row["database"] != database:
                raise ValueError("Confirmation database identity disagrees")
            cases[case_id] = {
                "case_id": case_id,
                "database": database,
                "record": row["record"],
                "signal_sha256": None,
            }
    manifest_sha256 = None
    if manifest_path is not None:
        manifest_bytes = Path(manifest_path).read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("suite") != "p6confirm":
            raise ValueError("Expected p6confirm signal manifest")
        seen = set()
        for row in manifest["cases"]:
            case_id = row["case_id"]
            if case_id in seen or case_id not in cases:
                raise ValueError("Duplicate or unknown confirmation manifest case identity")
            seen.add(case_id)
            for key in ("database", "record"):
                if key in row and row[key] != cases[case_id][key]:
                    raise ValueError("Confirmation manifest identity disagrees with eligibility")
            cases[case_id]["signal_sha256"] = _digest(row["signal_sha256"])
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return {
        "suite": "p6confirm",
        "eligibility_sha256": hashlib.sha256(eligibility_bytes).hexdigest(),
        "signal_manifest_sha256": manifest_sha256,
        "source_records": [{"database": database, "record": record} for database, record in sorted(records)],
        "cases": [cases[case_id] for case_id in sorted(cases)],
    }


def compare_overlap(p7_cases: list[dict], confirmation: dict) -> dict:
    """Return deterministic identity and signal-hash comparisons with coverage.

    Every development case must supply ``case_id``, ``database``, ``record`` and
    ``signal_sha256``. A zero-overlap claim requires all confirmation case hashes.
    Repeated hashes for the same development record across endpoints are expected;
    repeated hashes for distinct records are separately reported as duplicates.
    """
    p7_ids = [row["case_id"] for row in p7_cases]
    if len(p7_ids) != len(set(p7_ids)):
        raise ValueError("Duplicate development case identity")
    p7_by_record, p6_by_record = defaultdict(list), defaultdict(list)
    p7_by_hash, p6_by_hash = defaultdict(list), defaultdict(list)
    for row in p7_cases:
        p7_by_record[(row["database"], row["record"])].append(row["case_id"])
        p7_by_hash[_digest(row["signal_sha256"])].append(row)
    for row in confirmation["cases"]:
        p6_by_record[(row["database"], row["record"])].append(row["case_id"])
        if row["signal_sha256"] is not None:
            p6_by_hash[_digest(row["signal_sha256"])].append(row["case_id"])
    confirmation_records = {(row["database"], row["record"]) for row in confirmation["source_records"]}
    identities = [
        {
            "database": database,
            "record": record,
            "p7dev_case_ids": sorted(p7_by_record[(database, record)]),
            "p6confirm_case_ids": sorted(p6_by_record[(database, record)]),
        }
        for database, record in sorted(p7_by_record.keys() & confirmation_records)
    ]
    hashes = [
        {
            "signal_sha256": digest,
            "p7dev_case_ids": sorted(row["case_id"] for row in p7_by_hash[digest]),
            "p6confirm_case_ids": sorted(p6_by_hash[digest]),
        }
        for digest in sorted(p7_by_hash.keys() & p6_by_hash.keys())
    ]
    duplicates = []
    for digest, rows in sorted(p7_by_hash.items()):
        records = {(row["database"], row["record"]) for row in rows}
        if len(records) > 1:
            duplicates.append(
                {
                    "signal_sha256": digest,
                    "case_ids": sorted(row["case_id"] for row in rows),
                    "records": [{"database": database, "record": record} for database, record in sorted(records)],
                }
            )
    missing = sorted(row["case_id"] for row in confirmation["cases"] if row["signal_sha256"] is None)
    complete = bool(confirmation["cases"]) and not missing
    status = "failed" if identities or hashes else "passed" if complete else "incomplete"
    return {
        "status": status,
        "zero_overlap": False if identities or hashes else True if complete else None,
        "p7dev_case_count": len(p7_cases),
        "p7dev_record_count": len(p7_by_record),
        "confirmation_eligibility_sha256": confirmation["eligibility_sha256"],
        "confirmation_signal_manifest_sha256": confirmation["signal_manifest_sha256"],
        "identity_check": {
            "status": "failed" if identities else "passed",
            "confirmation_record_count": len(confirmation_records),
            "overlaps": identities,
        },
        "signal_hash_check": {
            "status": "failed" if hashes else "passed" if complete else "incomplete",
            "confirmation_case_count": len(confirmation["cases"]),
            "confirmation_hashed_case_count": len(confirmation["cases"]) - len(missing),
            "confirmation_missing_hash_case_ids": missing,
            "overlaps": hashes,
            "limitation": None
            if complete
            else "Emitted p6confirm signal.npy SHA256 hashes are unavailable for some or all cases; source-container "
            "checksums do not establish emitted-signal overlap. Confirmation signals were not opened.",
        },
        "p7dev_duplicate_signals_across_records": duplicates,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, default=root / "eval/cases/p7dev.manifest.json")
    parser.add_argument("--confirmation-eligibility", type=Path, default=root / "eval/tasks/confirm_eligibility.json")
    parser.add_argument("--confirmation-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=root / "eval/results/p7dev/overlap.json")
    args = parser.parse_args()
    manifest_bytes = args.development_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("suite") != "p7dev":
        parser.error("Expected p7dev development manifest")
    confirmation = load_confirmation_index(args.confirmation_eligibility, args.confirmation_manifest)
    report = compare_overlap(manifest["cases"], confirmation)
    report["development_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "identity": report["identity_check"]["status"],
                "signal_hash": report["signal_hash_check"]["status"],
                "output": str(args.output),
            }
        )
    )
    sys.exit(0 if report["status"] == "passed" else 1 if report["status"] == "failed" else 2)


if __name__ == "__main__":
    main()
