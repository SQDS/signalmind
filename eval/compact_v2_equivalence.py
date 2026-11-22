"""Deterministic v1/v2 MCP equivalence on all 69 opened AF development records.

No models, annotations, or paid services are used. Each record runs in its own
process, both interfaces share the same SignalStore, and every episode is fetched
through the public detail request. This bounds memory without replacing the
frozen detector or bypassing MCP serialization.

  .venv/bin/python eval/compact_v2_equivalence.py \
      --data-root data --output eval/results/compact-v2/equivalence.json

The first channel and whole-record window match the frozen AF diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASELINE = "baseline-R-2026-10-12"
FROZEN_PATHS = (
    "src/tsmcp/plugins/ecg/af.py",
    "src/tsmcp/plugins/ecg/morphology.py",
    "src/tsmcp/plugins/ecg/calibration.py",
    "src/tsmcp/plugins/ecg/rpeaks.py",
    "src/tsmcp/plugins/ecg/__init__.py",
    "src/tsmcp/core/measure.py",
    "src/tsmcp/core/detect.py",
)
AFDB_RECORDS = tuple(
    "04015 04043 04048 04126 04746 04908 04936 05091 05121 05261 06426 06453 "
    "06995 07162 07859 07879 07910 08215 08219 08378 08405 08434 08455".split()
)
BURDEN_FIELDS = (
    "value",
    "unit",
    "pct_of_window",
    "pct_of_readable",
    "af_s",
    "window_s",
    "irregular_s",
    "readable_s",
    "unreadable_s",
    "n_episodes",
)


def file_sha256(path: Path) -> str:
    """Hash waveforms in bounded chunks rather than loading a second copy."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def frozen_source_hashes() -> dict:
    out = {}
    for name in FROZEN_PATHS:
        original = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT)
        current = (ROOT / name).read_bytes()
        if current != original:
            raise ValueError(f"Frozen detector/measurement source changed: {name}")
        out[name] = hashlib.sha256(current).hexdigest()
    return out


def population() -> tuple[list[dict], dict]:
    """Validate all opened records, without consulting question answers."""
    paths = {
        "ltafdb_split": ROOT / "eval/tasks/ltafdb_split.json",
        "former_fresh_inventory": ROOT / "eval/results/af-continuity/fresh-inventory.json",
        "afdb_eligibility": ROOT / "eval/tasks/afdb_eligibility.json",
    }
    split = json.loads(paths["ltafdb_split"].read_text())
    fresh = [r["record"] for r in json.loads(paths["former_fresh_inventory"].read_text())["records"]]
    afdb = json.loads(paths["afdb_eligibility"].read_text())["records"]
    if len(split["dev"]) != 8 or len(fresh) != 38 or set(fresh) != set(split["unused"]):
        raise ValueError("Expected all 8 LTAFDB dev and 38 former-fresh records")
    if set(afdb) != set(AFDB_RECORDS):
        raise ValueError("Expected all 23 opened AFDB waveform records")
    records = [
        *({"record": f"ltafdb/{rec}", "cohort": "ltafdb_dev"} for rec in sorted(split["dev"])),
        *({"record": f"ltafdb/{rec}", "cohort": "ltafdb_former_fresh"} for rec in sorted(fresh)),
        *({"record": f"afdb/{rec}", "cohort": "afdb_opened"} for rec in AFDB_RECORDS),
    ]
    if len(records) != 69 or len({r["record"] for r in records}) != 69:
        raise ValueError("Record population has omissions or duplicates")
    return records, {key: file_sha256(path) for key, path in paths.items()}


def clipped(intervals, start, end):
    """Independent half-open clipping, preserving the frozen interval order."""
    return [[max(start, a), min(end, b)] for a, b in intervals if min(end, b) > max(start, a)]


def check_equal(actual, expected, context):
    if actual != expected:
        raise AssertionError(f"{context}: expected {expected!r}, got {actual!r}")


async def call(client, name, args):
    result = await client.call_tool(name, args)
    texts = [block.text for block in result.content if block.type == "text"]
    if result.is_error:
        return {"is_error": True, "text": "\n".join(texts)}
    if len(texts) != 1:
        raise AssertionError(f"{name}: expected one compact JSON text block, got {len(texts)}")
    return {"is_error": False, "payload": json.loads(texts[0]), "chars": len(texts[0])}


async def compare_record(data_root: Path, record: str) -> dict:
    from mcp import Client

    from tsmcp.core.compact import dumps, tround
    from tsmcp.core.io.wfdb import read_header
    from tsmcp.core.signal import time_decimals
    from tsmcp.core.store import SignalStore
    from tsmcp.plugins.ecg.af import af_analysis
    from tsmcp.server.app import build_server

    started = time.monotonic()
    source = data_root / f"{record}.hea"
    header = read_header(source)
    waveforms = sorted({source.parent / item["file"] for item in header["signals"]})
    input_hashes = {str(path.relative_to(data_root)): file_sha256(path) for path in [source, *waveforms]}
    store = SignalStore(allow_roots=[data_root])
    # Load once without ts_describe's unrelated overview computations. The AF
    # measurements, index and every detail are still real public MCP requests.
    signal = store.load(source=str(source))
    channel = signal.channels[0]
    tdec = time_decimals(channel.fs)
    request = {"id": signal.id, "channels": [channel.name], "what": ["ecg_af_episodes", "ecg_af_burden"]}
    base = {
        "record": record,
        "channel_index": 0,
        "channel": channel.name,
        "fs": channel.fs,
        "n_samples": len(channel.data),
        "window": [0, channel.duration],
        "input_sha256": input_hashes,
    }
    async with Client(build_server(store, images=False, interface="v1")) as v1:
        first = await call(v1, "ts_measure", request)
    async with Client(build_server(store, images=False, interface="v2")) as v2:
        second = await call(v2, "ts_measure", request)
        if first["is_error"] or second["is_error"]:
            return {
                **base,
                "status": "matching_baseline_error" if first == second else "mismatch",
                "v1": first,
                "v2": second,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        old, new = first["payload"], second["payload"]
        check_equal(new["interface_version"], "v2", "interface version")
        if "interface_version" in old:
            raise AssertionError("v1 output unexpectedly changed")
        old_measures = {row["name"]: row for row in old["measurements"]}
        new_measures = {row["name"]: row for row in new["measurements"]}
        legacy = old_measures["ecg_af_episodes"]["episodes"]
        ranked = sorted(legacy, key=lambda row: (-row["span_s"], row["start"]))
        check_equal(legacy, ranked, "unrounded v1 ordering")
        index = new_measures["ecg_af_episodes"]
        check_equal(index["count"], len(legacy), "complete count")
        check_equal(index["channel"], channel.name, "channel")
        check_equal(index["units"], "s", "units")
        check_equal(index["window"], [0, tround(channel.duration, tdec)], "analysed window")
        check_equal(index["ranking"], "span_s descending, start ascending (unrounded)", "ranking rule")
        check_equal(len(index["episodes"]), len(legacy), "complete episode index")
        ids = [row["id"] for row in index["episodes"]]
        check_equal(len(set(ids)), len(ids), "unique stable IDs")
        analysis = af_analysis(channel.data, channel.fs, ch=channel)
        expected_index, reconstructed, readability = [], [], []
        detail_chars = 0
        for number, (raw, row) in enumerate(zip(ranked, index["episodes"], strict=True)):
            start, end = raw["start"], raw["intervals_s"][-1][1]
            unknown = clipped(analysis["noisy"], start, end)
            readable = clipped(analysis["readable"], start, end)
            expected = {
                "id": row["id"],
                "start": tround(start, tdec),
                "end": tround(end, tdec),
                "span_s": tround(raw["span_s"], tdec),
                "af_s": tround(raw["af_s"], tdec),
                "unknown_s": tround(sum(b - a for a, b in unknown), tdec),
                "merged_from": len(raw["intervals_s"]),
            }
            check_equal(row, expected, f"episode {number} precision/index")
            expected_index.append(json.loads(dumps(expected)))
            result = await call(
                v2,
                "ts_measure",
                {"id": signal.id, "what": ["ecg_af_episode_detail"], "episode": row["id"]},
            )
            if result["is_error"]:
                raise AssertionError(f"episode {number} detail failed: {result['text']}")
            detail = result["payload"]["measurements"][0]
            check_equal(detail["episode"], row["id"], f"episode {number} ID resolution")
            check_equal(detail["channel"], channel.name, f"episode {number} detail channel")
            check_equal(detail["window"], index["window"], f"episode {number} detail window")
            check_equal(detail["legacy_episode"], raw, f"episode {number} exact v1 evidence")
            # Match exactly the JSON-cleaned frozen readable/unknown intervals.
            check_equal(detail["readable_intervals_s"], json.loads(dumps(readable)), f"episode {number} readable")
            check_equal(detail["unknown_intervals_s"], json.loads(dumps(unknown)), f"episode {number} unknown")
            reconstructed.append(detail["legacy_episode"])
            readability.append({"readable": readable, "unknown": unknown})
            detail_chars += result["chars"]
        before_burden = {key: old_measures["ecg_af_burden"].get(key) for key in BURDEN_FIELDS}
        after_burden = {key: new_measures["ecg_af_burden"].get(key) for key in BURDEN_FIELDS}
        check_equal(after_burden, before_burden, "every burden value/denominator")
        check_equal(reconstructed, legacy, "all exact episodes reconstructed in rank order")
        # Compare the exact longest raw episode, not merely its rounded duration.
        check_equal(reconstructed[:1], ranked[:1], "longest-episode identity")
    # A brand new server has no presentation cache. Its index must recreate all
    # IDs, and its first detail is requested before any index on that server.
    async with Client(build_server(store, images=False, interface="v2")) as restarted:
        if ids:
            result = await call(
                restarted,
                "ts_measure",
                {"id": signal.id, "what": ["ecg_af_episode_detail"], "episode": ids[0]},
            )
            check_equal(result["is_error"], False, "fresh-server stable ID resolves without index")
            check_equal(result["payload"]["measurements"][0]["legacy_episode"], ranked[0], "fresh-server longest")
        result = await call(restarted, "ts_measure", {**request, "what": ["ecg_af_episodes"]})
        check_equal(result["is_error"], False, "fresh-server index")
        repeated_index = result["payload"]["measurements"][0]
        check_equal(repeated_index, index, "stable IDs/index across server lifetimes")
    out = {
        **base,
        "status": "pass",
        "episode_count": len(legacy),
        "support_interval_count": sum(len(row["intervals_s"]) for row in legacy),
        "detail_calls": len(legacy),
        "checks": {
            "complete_index": True,
            "all_index_fields_after_time_precision": True,
            "all_detail_evidence_exact_v1": True,
            "all_readability_exact_frozen_analysis": True,
            "all_burden_fields_exact_v1": True,
            "unrounded_rank_order": True,
            "longest_episode_identity": True,
            "stable_ids_fresh_server": True,
        },
        "hashes": {
            "v1_episodes": object_sha256(legacy),
            "v2_details_reconstructed": object_sha256(reconstructed),
            "expected_index": object_sha256(expected_index),
            "actual_index": object_sha256(index["episodes"]),
            "readability": object_sha256(readability),
            "v1_burden": object_sha256(before_burden),
            "v2_burden": object_sha256(after_burden),
        },
        "burden": before_burden,
        "longest_episode": index["episodes"][0] if ids else None,
        "response_characters": {
            "v1_index_and_burden": first["chars"],
            "v2_index_and_burden": second["chars"],
            "v2_all_details": detail_chars,
        },
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    store.forget(signal.id)
    del signal, channel, analysis, store
    gc.collect()
    return out


def source_manifest():
    files = [Path(__file__), *sorted((ROOT / "src").rglob("*.py"))]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in files}


def save(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", help="Run one allowlisted record for smoke verification")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    records, manifests = population()
    if args.record:
        records = [row for row in records if row["record"] == args.record]
        if not records:
            parser.error("--record must be one of the 69 opened development records")
    if args.worker:
        if len(records) != 1:
            parser.error("--worker requires --record")
        try:
            with tempfile.TemporaryDirectory(prefix="compact-v2-handles-") as cache:
                os.environ["TSMCP_CACHE_DIR"] = cache
                result = asyncio.run(compare_record(args.data_root, args.record))
        except Exception as error:
            result = {
                "record": args.record,
                "status": "failure",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        save(args.output, result)
        return
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": "deterministic MCP interface replay; no models or paid calls",
        "baseline": BASELINE,
        "detector_sha256": frozen_source_hashes(),
        "source_sha256": source_manifest(),
        "population_manifest_sha256": manifests,
        "scope": {
            "expected_records": len(records),
            "full_population": len(records) == 69,
            "channel_index": 0,
            "window": "whole record",
        },
        "precision": "time coordinates and second-valued quantities: clamp(ceil(log10(fs)),1,4) decimals; other measurements: 3 significant figures; full detail retains exact v1 JSON; rank/integrate before rounding",
        "method": "Separate process per record; shared SignalStore/frozen AF cache for real v1/v2 MCP calls; complete index plus one explicit MCP detail call per episode; fresh v2 server verifies stable IDs and first detail before index. No annotation or question-answer reads.",
        "records": [],
    }
    with tempfile.TemporaryDirectory(prefix="compact-v2-workers-") as scratch:
        for item in records:
            worker_output = Path(scratch) / "record.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--data-root",
                str(args.data_root),
                "--record",
                item["record"],
                "--output",
                str(worker_output),
                "--worker",
            ]
            proc = subprocess.run(command, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and worker_output.exists():
                row = json.loads(worker_output.read_text())
                worker_output.unlink()
            else:
                row = {
                    "record": item["record"],
                    "status": "failure",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[-4000:],
                }
            result["records"].append({**item, **row})
            result["summary"] = {
                "attempted_records": len(result["records"]),
                "passed_records": sum(row["status"] == "pass" for row in result["records"]),
                "matching_baseline_errors": sum(
                    row["status"] == "matching_baseline_error" for row in result["records"]
                ),
                "failed_records": sum(
                    row["status"] not in {"pass", "matching_baseline_error"} for row in result["records"]
                ),
                "episodes_verified": sum(row.get("episode_count", 0) for row in result["records"]),
                "support_intervals_verified": sum(row.get("support_interval_count", 0) for row in result["records"]),
                "complete": len(result["records"]) == len(records),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            save(args.output, result)
            print(
                json.dumps(
                    {
                        "record": item["record"],
                        "status": row["status"],
                        "episodes": row.get("episode_count"),
                        "elapsed_s": row.get("elapsed_s"),
                        "error": row.get("error"),
                    }
                ),
                flush=True,
            )
    result["source_sha256_at_completion"] = source_manifest()
    result["source_unchanged_during_run"] = result["source_sha256"] == result["source_sha256_at_completion"]
    save(args.output, result)
    summary = result["summary"]
    if summary["failed_records"] or summary["matching_baseline_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
