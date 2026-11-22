"""Metadata-only Level 2 reservation; never open a waveform or annotation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261026


def freeze() -> dict:
    parent_path = ROOT / "eval/tasks/audit_splits.json"
    hashes_path = ROOT / "eval/tasks/audit_sealed_hashes.json"
    parent = json.loads(parent_path.read_text())
    inventory = json.loads(hashes_path.read_text())

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    if inventory["split_sha256"] != digest(parent_path):
        raise ValueError("Sealed byte inventory does not bind the parent split")
    result = {
        "schema_version": 1,
        "seed": SEED,
        "parent_split_sha256": digest(parent_path),
        "source_hash_inventory_sha256": digest(hashes_path),
        "assignment": "Sort SHA256(20261026:level2:dataset:opaque_subject); first floor(n/2) confirmation, rest future reserve",
        "waveforms_decoded": False,
        "datasets": {},
    }
    for name, entry in parent["datasets"].items():
        sealed = entry["sealed_confirmation"]
        ranked = sorted(sealed, key=lambda s: hashlib.sha256(f"{SEED}:level2:{name}:{s}".encode()).hexdigest())
        confirmation, reserve = ranked[: len(ranked) // 2], ranked[len(ranked) // 2 :]
        assert not (set(entry["development"]) & set(sealed))
        result["datasets"][name] = {
            "development": entry["development"],
            "level2_confirmation": confirmation,
            "future_reserve": reserve,
            "source_hashes": [
                {k: v for k, v in r.items() if k in {"subject_key", "record_key", "source_sha256", "header_sha256"}}
                for r in inventory["records"]
                if r["dataset"] == name
            ],
        }
    return result


if __name__ == "__main__":
    result = freeze()
    (ROOT / "eval/tasks/level2_splits.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                name: {k: len(v[k]) for k in ("development", "level2_confirmation", "future_reserve")}
                for name, v in result["datasets"].items()
            }
        )
    )
