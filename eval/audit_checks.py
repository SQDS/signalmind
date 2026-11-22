"""Offline capability-audit integrity checks and exact C/D command dry runs.

No model process is launched. Runtime isolation probes replace the model command
with a small Python program; the Bash network restriction is configuration only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from eval.common import read_jsonl
from eval.grade import grade, oracle

ROOT = Path(__file__).resolve().parent.parent
HEX = re.compile(r"[0-9a-f]{64}")
CASE_ID = re.compile(r"[0-9a-f]{24}")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sealed_inventory(manifest: dict, manifest_path: Path, splits: dict, splits_path: Path, errors: list) -> dict:
    """Validate the separately frozen raw-byte inventory without reopening data."""
    binding = manifest.get("sealed_hash_inventory")
    result = {"hashes": set(), "records": 0, "raw_files": 0, "complete": False}
    if binding is None:
        if manifest.get("amendment") == "P1":
            errors.append("P1 requires a frozen sealed raw hash inventory")
        return result
    before = len(errors)
    if (
        not isinstance(binding, dict)
        or set(binding) != {"path", "sha256"}
        or not isinstance(binding["path"], str)
        or Path(binding["path"]).name != binding["path"]
        or not isinstance(binding["sha256"], str)
        or not HEX.fullmatch(binding["sha256"])
    ):
        errors.append("Invalid sealed raw hash inventory binding")
        return result
    path = manifest_path.parent / binding["path"]
    if not path.is_file() or path.is_symlink() or _sha(path) != binding["sha256"]:
        errors.append("Frozen sealed raw hash inventory file hash mismatch")
        return result
    try:
        inventory = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        errors.append("Sealed raw hash inventory cannot be read as JSON")
        return result
    if not isinstance(inventory, dict) or not isinstance(inventory.get("records"), list):
        errors.append("Sealed raw hash inventory has an invalid structure")
        return result
    if (
        inventory.get("schema_version") != 1
        or inventory.get("method") != "sha256_raw_source_bytes"
        or inventory.get("waveform_decoding") is not False
        or inventory.get("split_sha256") != _sha(splits_path)
    ):
        errors.append("Sealed raw hash inventory method or frozen split binding mismatch")
    expected = {
        (name, record["record_key"]): record
        for name, dataset in splits["datasets"].items()
        for record in dataset["records"]
        if record["subject_key"] in dataset["sealed_confirmation"]
    }
    seen = set()
    counts = Counter()
    total_bytes = 0
    for record in inventory.get("records", []):
        if not isinstance(record, dict):
            errors.append("Sealed raw hash inventory has an invalid record")
            continue
        key = (record.get("dataset"), record.get("record_key"))
        if any(not isinstance(value, str) for value in key):
            errors.append("Sealed raw hash inventory has an invalid record key")
            continue
        frozen = expected.get(key)
        if key in seen or frozen is None or record.get("subject_key") != frozen["subject_key"]:
            errors.append("Sealed raw hash inventory has a duplicate or unregistered record")
            continue
        seen.add(key)
        counts[key[0]] += 1
        kinds = ["source", "header"] if "header_sha256" in frozen else ["source"]
        for kind in kinds:
            value, size = record.get(f"{kind}_sha256"), record.get(f"{kind}_bytes")
            if not isinstance(value, str) or not HEX.fullmatch(value):
                errors.append("Sealed raw hash inventory has a missing or invalid file hash")
                continue
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                errors.append("Sealed raw hash inventory has an invalid file size")
                continue
            checksum = frozen.get(f"{kind}_sha256")
            if kind == "source":
                checksum = checksum or frozen.get("published_source_sha256")
            if checksum and checksum != value:
                errors.append("Sealed raw hash inventory disagrees with frozen checksum metadata")
            result["hashes"].add(value)
            result["raw_files"] += 1
            total_bytes += size
        result["records"] += 1
    if seen != set(expected):
        errors.append("Sealed raw hash inventory does not cover every frozen sealed record")
    if (
        inventory.get("dataset_counts") != dict(counts)
        or inventory.get("record_count") != result["records"]
        or inventory.get("raw_file_count") != result["raw_files"]
        or inventory.get("source_bytes") != total_bytes
    ):
        errors.append("Sealed raw hash inventory aggregate counts mismatch")
    result["complete"] = before == len(errors)
    return result


def validate_suite(manifest_path: Path, splits_path: Path, cases_root: Path) -> dict:
    """Recheck frozen files and privacy boundaries, without reading any source waveform."""
    manifest = json.loads(manifest_path.read_text())
    splits = json.loads(splits_path.read_text())
    suite = manifest["suite"]
    errors: list[str] = []
    questions = manifest["questions"]
    question_rows = read_jsonl(cases_root / f"{suite}.questions.jsonl")
    answer_rows = read_jsonl(cases_root / f"{suite}.answers.jsonl")
    public = {row["case_id"]: row for row in question_rows}
    answers = {row["case_id"]: row for row in answer_rows}
    if len(public) != len(question_rows) or len(answers) != len(answer_rows):
        errors.append("Duplicate external question or answer ids")
    if manifest["split_sha256"] != _sha(splits_path):
        errors.append("Frozen split file hash mismatch")
    for suffix in ("questions", "answers"):
        expected = manifest.get(f"{suffix}_sha256")
        if expected and expected != _sha(cases_root / f"{suite}.{suffix}.jsonl"):
            errors.append(f"Frozen external {suffix} file hash mismatch")
    split_overlap = 0
    sealed_source_hashes: set[str] = set()
    sealed_signal_hashes: set[str] = set()
    sealed_records, hashed_sealed_records = 0, 0
    for dataset in splits["datasets"].values():
        development, sealed = set(dataset["development"]), set(dataset["sealed_confirmation"])
        split_overlap += len(development & sealed)
        if any(not HEX.fullmatch(key) for key in development | sealed):
            errors.append("A split subject key is not an opaque SHA256")
        for record in dataset["records"]:
            if record["subject_key"] in sealed:
                sealed_records += 1
                sealed_source_hash = record.get("source_sha256") or record.get("published_source_sha256")
                if sealed_source_hash:
                    hashed_sealed_records += 1
                    sealed_source_hashes.add(sealed_source_hash)
                if record.get("signal_sha256"):
                    sealed_signal_hashes.add(record["signal_sha256"])
    raw_inventory = _sealed_inventory(manifest, manifest_path, splits, splits_path, errors)
    raw_required = manifest.get("amendment") == "P1" or manifest.get("sealed_hash_inventory") is not None
    if raw_required:
        sealed_source_hashes = raw_inventory["hashes"]
        hashed_sealed_records = raw_inventory["records"]
        # P1 compares the generated NPY file bytes with the sealed raw source
        # bytes. This is deliberately not a canonical waveform/window check.
        sealed_signal_hashes = sealed_source_hashes
    if split_overlap:
        errors.append("Development and sealed subject partitions overlap")
    seen_ids: set[str] = set()
    seen_families: set[tuple[str, str, str]] = set()
    source_overlaps, signal_overlaps, oracle_count = 0, 0, 0
    source_hashes_checked, signal_hashes_checked = 0, 0
    groups: Counter = Counter()
    row_counts: Counter = Counter()
    for question in questions:
        case_id = question["case_id"]
        if not CASE_ID.fullmatch(case_id) or question["id"] != case_id:
            errors.append("A case id is not the specified opaque 24-character hex id")
            continue  # Also prevents path traversal in malformed manifests.
        if case_id in seen_ids:
            errors.append(f"{case_id}: repeated case id")
        seen_ids.add(case_id)
        source = question["source"]
        dataset_name, subject = source["dataset"], source["subject_key"]
        dataset = splits["datasets"].get(dataset_name)
        if dataset is None:
            errors.append(f"{case_id}: unregistered dataset")
            continue
        if subject not in dataset["development"] or subject in dataset["sealed_confirmation"]:
            errors.append(f"{case_id}: subject is not exclusively development")
        family_key = (dataset_name, subject, question["family"])
        if family_key in seen_families:
            errors.append(f"{case_id}: repeated subject within a family")
        seen_families.add(family_key)
        records = {record["record_key"]: record for record in dataset["records"]}
        record = records.get(source["record_key"], {})
        if record.get("subject_key") != subject:
            errors.append(f"{case_id}: record is not registered to the frozen subject")
        source_hashes = source["source_hashes"]
        if not source_hashes.get("source_sha256"):
            errors.append(f"{case_id}: source waveform hash is missing")
        for key, value in source_hashes.items():
            if not isinstance(value, str) or not HEX.fullmatch(value) or record.get(key) != value:
                errors.append(f"{case_id}: source provenance does not match frozen hashes")
        source_overlaps += int(
            any(isinstance(value, str) and value in sealed_source_hashes for value in source_hashes.values())
        )
        source_hashes_checked += len(source_hashes)
        case_dir = cases_root / suite / case_id
        if not case_dir.is_dir() or {path.name for path in case_dir.iterdir()} != {"signal.npy", "visible.json"}:
            errors.append(f"{case_id}: case directory must contain only signal.npy and visible.json")
            continue
        if any(path.is_symlink() for path in case_dir.iterdir()):
            errors.append(f"{case_id}: public case files must not be symlinks")
            continue
        visible = json.loads((case_dir / "visible.json").read_text())
        if set(visible) != {"file", "dur_s", "channels"} or visible.get("file") != "signal.npy":
            errors.append(f"{case_id}: visible metadata contains unexpected fields or filename")
        channels = visible["channels"]
        names = [channel["name"] for channel in channels]
        if not names or len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
            errors.append(f"{case_id}: invalid channel names")
        if any(set(channel) != {"name", "unit", "fs"} for channel in channels):
            errors.append(f"{case_id}: unexpected public channel metadata")
        rates = [channel["fs"] for channel in channels]
        if not rates or any(isinstance(rate, bool) or not isinstance(rate, (int, float)) for rate in rates):
            errors.append(f"{case_id}: invalid sampling rate")
            continue
        if any(not math.isfinite(rate) or rate <= 0 or rate != rates[0] for rate in rates):
            errors.append(f"{case_id}: channels do not share one positive sampling rate")
            continue
        signal_path = case_dir / "signal.npy"
        signal = np.load(signal_path, allow_pickle=False, mmap_mode="r")
        shape_ok = (signal.ndim == 1 and len(channels) == 1) or (signal.ndim == 2 and signal.shape[1] == len(channels))
        if not shape_ok or not math.isclose(len(signal) / rates[0], visible["dur_s"], abs_tol=1 / rates[0]):
            errors.append(f"{case_id}: signal shape/duration disagrees with public channel metadata")
        signal_hash = _sha(signal_path)
        if signal_hash != question["signal_sha256"]:
            errors.append(f"{case_id}: emitted signal hash mismatch")
        signal_overlaps += int(signal_hash in sealed_signal_hashes)
        signal_hashes_checked += 1
        qtype = question["task_type"]
        if case_id not in public or public[case_id]["qtype"] != qtype:
            errors.append(f"{case_id}: external question missing or qtype mismatch")
        if case_id not in answers or answers[case_id]["qtype"] != qtype:
            errors.append(f"{case_id}: external answer missing or qtype mismatch")
        else:
            answer = answers[case_id]["answer"]
            if not grade(qtype, answer, oracle(qtype, answer))["correct"]:
                errors.append(f"{case_id}: oracle grading failed")
            else:
                oracle_count += 1
        groups[question["score_group"]] += 1
        row_counts[question["row"]] += 1
    expected_ids = set(public) | set(answers)
    if expected_ids != seen_ids or set(public) != set(answers):
        errors.append("Manifest/question/answer id sets differ")
    suite_dir = cases_root / suite
    if {path.name for path in suite_dir.iterdir()} != seen_ids:
        errors.append("Suite contains unregistered case directories or files")
    denominators = manifest["denominators"]
    if (
        groups["core"] != denominators["core"]
        or groups["gap-probe"] != denominators["gap_probe"]
        or len(questions) != denominators["total"]
        or sum(groups.values()) != groups["core"] + groups["gap-probe"]
    ):
        errors.append("Score denominators disagree with accepted questions")
    for row in manifest["rows"]:
        if row_counts[row["row"]] != row["actual"] or row["actual"] > row["requested"]:
            errors.append(f"Row {row['row']}: count mismatch")
    if source_overlaps or signal_overlaps:
        errors.append("Development and sealed waveform hashes overlap")
    return {
        "valid": not errors,
        "errors": errors,
        "question_count": len(questions),
        "oracle_correct": oracle_count,
        "case_files_only_signal_and_visible": not any("directory" in error for error in errors),
        "one_subject_per_family": not any("within a family" in error for error in errors),
        "development_only": not any("exclusively development" in error for error in errors),
        "subject_partition_overlap": split_overlap,
        "sealed_source_hashes_available": hashed_sealed_records,
        "sealed_source_records": sealed_records,
        "sealed_raw_files_hashed": raw_inventory["raw_files"],
        "development_source_hashes_compared": source_hashes_checked,
        "development_signal_hashes_compared": signal_hashes_checked,
        "sealed_source_hash_overlap": source_overlaps,
        "sealed_source_hash_check": (
            "complete"
            if raw_inventory["complete"] or (not raw_required and hashed_sealed_records == sealed_records)
            else "incomplete"
        ),
        "sealed_signal_hash_overlap": signal_overlaps,
        "sealed_signal_hash_check": (
            "complete"
            if raw_inventory["complete"] or (not raw_required and len(sealed_signal_hashes) == sealed_records)
            else (
                "incomplete: sealed raw source inventory is missing or invalid"
                if raw_required
                else "incomplete: no full sealed array hash inventory"
            )
        ),
        "sealed_signal_hash_comparison_basis": (
            "Development generated signal.npy SHA256 versus every sealed raw source/header SHA256"
            if raw_required
            else "Development generated signal.npy SHA256 versus sealed signal-array SHA256 where available"
        ),
        "canonical_waveform_hash_comparison": "not performed; sealed waveforms were not decoded",
        "sealed_source_bytes_hashed": raw_inventory["complete"],
        "sealed_waveforms_decoded": False,
        "denominators": denominators,
    }


def _probe_home(command: list[str], overrides: dict, work: Path, repo: Path) -> dict:
    """Reuse the exact outer bwrap prefix, replacing its model command with Python."""
    if not command or command[0] != "bwrap" or "--" not in command:
        return {"status": "unavailable", "reason": "no outer bwrap prefix"}
    with tempfile.NamedTemporaryFile(prefix=".audit-isolation-", dir=repo) as sentinel:
        payload = (
            "import json, os; from pathlib import Path; "
            f"print(json.dumps({{'private_file_visible':Path({sentinel.name!r}).exists(),"
            f"'signal_visible':Path({str(work / 'signal.npy')!r}).is_file(),"
            "'home':os.environ.get('HOME')}))"
        )
        prefix = command[: command.index("--") + 1]
        probe = [*prefix, "/usr/bin/python3", "-c", payload]
        env = {**os.environ, **{key: value for key, value in overrides.items() if value != "<redacted>"}}
        try:
            result = subprocess.run(probe, cwd=work, env=env, capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"status": "unavailable", "reason": type(error).__name__}
    if result.returncode:
        return {"status": "unavailable", "returncode": result.returncode, "stderr": result.stderr.strip()[:500]}
    details = json.loads(result.stdout)
    return {
        "status": "pass" if not details["private_file_visible"] and details["signal_visible"] else "fail",
        **details,
    }


def dry_runs(manifest_path: Path, cases_root: Path, output_dir: Path, repo: Path = ROOT) -> dict:
    """Materialize one PPG + one multimodal case for C/D, never execute a model."""
    manifest = json.loads(manifest_path.read_text())
    suite = manifest["suite"]
    chosen = []
    for family in ("PPG", "Multi"):
        matching = [question for question in manifest["questions"] if question["family"] == family]
        if not matching:
            raise ValueError(f"No {family} item available for dry run")
        chosen.append(matching[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for question in chosen:
        case_id = question["case_id"]
        visible = json.loads((cases_root / suite / case_id / "visible.json").read_text())
        names = [channel["name"] for channel in visible["channels"]]
        for arm in ("C", "D"):
            invocation = [
                str(repo / ".venv/bin/python"),
                str(repo / "eval/run_claude_code.py"),
                "--suite",
                suite,
                "--arm",
                arm,
                "--only",
                case_id,
                "--dry-run",
            ]
            process = subprocess.run(invocation, cwd=repo, capture_output=True, text=True, timeout=30, check=False)
            record = {"arm": arm, "case_id": case_id, "family": question["family"], "returncode": process.returncode}
            if process.returncode:
                record.update(status="fail", stderr=process.stderr.strip()[:500])
                results.append(record)
                continue
            plan = json.loads(process.stdout)
            (output_dir / f"dry-run-{arm}-{case_id}.json").write_text(json.dumps(plan, indent=2) + "\n")
            prompt = plan["stdin"]
            labels = [f"{channel['name']} ({channel['unit']}, {channel['fs']} Hz)" for channel in visible["channels"]]
            ordered_channels = ", ".join(labels) in prompt
            column_mapping = all(f"column {index} = {name}" in prompt for index, name in enumerate(names))
            loading_instruction = "assign these channel names in this order" in prompt
            work = Path(plan["cwd"])
            work_files = sorted(path.name for path in work.iterdir())
            copied_signal_matches = _sha(work / "signal.npy") == _sha(cases_root / suite / case_id / "signal.npy")
            record.update(
                status=(
                    "pass"
                    if ordered_channels
                    and column_mapping
                    and loading_instruction
                    and work_files == ["signal.npy"]
                    and copied_signal_matches
                    else "fail"
                ),
                channel_order=names,
                channels_in_prompt_order=ordered_channels,
                column_mapping_in_prompt=column_mapping,
                channel_assignment_instruction_in_prompt=loading_instruction,
                one_fs=len({channel["fs"] for channel in visible["channels"]}) == 1,
                public_work_files=work_files,
                copied_signal_matches=copied_signal_matches,
                model_invoked=False,
            )
            if arm == "C":
                record["mcp_tsmcp_configured"] = "tsmcp" in plan["mcp_config"]["mcpServers"]
            else:
                command = plan["command"]
                settings = json.loads(command[command.index("--settings") + 1])["sandbox"]
                home_tmpfs = any(
                    command[index : index + 2] == ["--tmpfs", str(Path.home())] for index in range(len(command) - 1)
                )
                network_blocked = (
                    settings["enabled"]
                    and settings["failIfUnavailable"]
                    and not settings["allowUnsandboxedCommands"]
                    and settings["network"]["strictAllowlist"]
                    and settings["network"]["allowedDomains"] == []
                )
                record.update(
                    home_hidden_configuration=home_tmpfs,
                    bash_network_blocked_configuration=network_blocked,
                    bash_network_runtime="not exercised: Claude's Bash sandbox requires a session",
                    home_runtime=_probe_home(command, plan["environment_overrides"], work, repo),
                )
                if not home_tmpfs or not network_blocked or record["home_runtime"]["status"] == "fail":
                    record["status"] = "fail"
            results.append(record)
    return {"model_calls": 0, "runs": results, "all_dry_runs_pass": all(row["status"] == "pass" for row in results)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval/tasks/p8audit_manifest.json")
    parser.add_argument("--splits", type=Path, default=ROOT / "eval/tasks/audit_splits.json")
    parser.add_argument("--cases-root", type=Path, default=ROOT / "eval/cases")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/results/audit-prep/checks.json")
    parser.add_argument("--dry-runs", action="store_true")
    args = parser.parse_args()
    result = {"integrity": validate_suite(args.manifest, args.splits, args.cases_root)}
    if args.dry_runs:
        result["dry_runs"] = dry_runs(args.manifest, args.cases_root, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(
        0 if result["integrity"]["valid"] and result.get("dry_runs", {}).get("all_dry_runs_pass", True) else 1
    )


if __name__ == "__main__":
    main()
