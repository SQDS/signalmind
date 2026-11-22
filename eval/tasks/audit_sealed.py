"""Hash sealed audit source files as raw bytes, without parsing their contents.

Frozen subject assignments are read only. Native paths stay in memory; the output
contains opaque keys and SHA256 digests only. WFDB headers are hashed as raw bytes
alongside their waveform files, and no HDF5 or waveform reader is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.tasks.audit_sources import SPLIT_PATH, file_hash, inventory, load_manifest, record_key

SEALED_PATH = Path(__file__).with_name("audit_sealed_hashes.json")
METHOD = "sha256_raw_source_bytes"


def freeze_sealed_hashes(
    destination: Path = SEALED_PATH, splits_path: Path = SPLIT_PATH, workers: int = 4, progress=None
) -> dict:
    """Compute a complete raw-byte inventory bound to unchanged frozen splits."""
    split_hash = file_hash(splits_path)
    splits = load_manifest(splits_path)
    local = inventory()
    candidates = []
    for dataset_name, dataset in splits["datasets"].items():
        sealed = set(dataset["sealed_confirmation"])
        if sealed & set(dataset["development"]):
            raise ValueError("Development and sealed subject partitions overlap")
        expected = {record["record_key"]: record for record in dataset["records"] if record["subject_key"] in sealed}
        resolved = {}
        for subject in sealed:
            for path in local.get(dataset_name, {}).get(subject, []):
                key = record_key(dataset_name, path)
                if key in resolved:
                    raise ValueError("Sealed local record mapping is duplicated")
                resolved[key] = (subject, path)
        if set(expected) != set(resolved):
            raise ValueError("Sealed local record coverage differs from frozen split")
        for key, record in sorted(expected.items()):
            subject, path = resolved[key]
            if subject != record["subject_key"]:
                raise ValueError("Sealed record subject differs from frozen split")
            candidates.append((dataset_name, subject, key, path, record))

    def hash_record(candidate):
        dataset_name, subject, key, path, frozen = candidate
        source = path if dataset_name == "pulsedb_mimic" else path.with_suffix(".dat")
        result = {"dataset": dataset_name, "subject_key": subject, "record_key": key}
        files = [("source", source)]
        if dataset_name != "pulsedb_mimic":
            files.append(("header", path))
        try:
            for kind, source_path in files:
                before = source_path.stat()
                digest = file_hash(source_path)
                after = source_path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Sealed source changed while raw bytes were hashed")
                result[f"{kind}_sha256"] = digest
                result[f"{kind}_bytes"] = after.st_size
            for field in ("source_sha256", "header_sha256"):
                if field in frozen and frozen[field] != result.get(field):
                    raise ValueError("Sealed raw source disagrees with frozen checksum")
            published = frozen.get("published_source_sha256")
            if published and published != result["source_sha256"]:
                raise ValueError("Sealed raw source disagrees with published checksum")
        except OSError:
            raise ValueError("Sealed raw source could not be hashed; path suppressed") from None
        return result

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(hash_record, candidates):
            records.append(record)
            if progress is not None:
                progress(len(records), len(candidates))
    if file_hash(splits_path) != split_hash:
        raise ValueError("Frozen split changed while sealed inventory was generated")
    result = {
        "schema_version": 1,
        "split_sha256": split_hash,
        "method": METHOD,
        "waveform_decoding": False,
        "scope": "Raw source files and WFDB headers; no canonical waveform or window comparison",
        "dataset_counts": dict(sorted(Counter(record["dataset"] for record in records).items())),
        "record_count": len(records),
        "raw_file_count": sum(1 + ("header_sha256" in record) for record in records),
        "source_bytes": sum(record["source_bytes"] + record.get("header_bytes", 0) for record in records),
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SEALED_PATH)
    parser.add_argument("--splits", type=Path, default=SPLIT_PATH)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(json.dumps({"sealed_records_hashed": done, "sealed_records_total": total}), flush=True)

    result = freeze_sealed_hashes(args.output, args.splits, args.workers, progress)
    print(
        json.dumps({key: result[key] for key in ("dataset_counts", "record_count", "raw_file_count", "source_bytes")})
    )


if __name__ == "__main__":
    main()
