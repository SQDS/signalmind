"""Review one frozen trigger, then search accepted waveforms for the same morphology."""

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from tsmcp.cleaning.agent import BudgetClient, vision_preflight
from tsmcp.cleaning.common import digest, read, sha, write
from tsmcp.cleaning.contracts import published_run
from tsmcp.followup.reviewer import review
from tsmcp.inspection.run import load_channel, version_id


def load_seed(workspace, inspection, candidate_id):
    cand_path = inspection / "candidates.jsonl"
    current = (inspection / "release.json").is_file()
    if current:
        release = read(inspection / "release.json")
        revision = release["freeze"]
        expected = release["sha256"]
    else:
        frozen = read(inspection / "candidate_manifest.json")
        revision = frozen["run_id"]
        expected = frozen["files"]["candidates.jsonl"]
    if sha(cand_path) != expected:
        raise ValueError("Frozen candidate manifest mismatch")
    with cand_path.open() as stream:
        c = next((row for row in map(json.loads, stream) if row["candidate_id"] == candidate_id), None)
    if not c or not current and c["run_id"] != revision:
        raise ValueError("Unknown candidate")
    region = c["region"] if current else c["range"]
    case = next(v for v in read(workspace / "dataset_manifest.json")["cases"] if v["id"] == region["block_id"])
    path = workspace / "public" / case["id"] / "input.npz"
    if sha(path) != case["input_sha256"]:
        raise ValueError("Source changed")
    source = c["source_version_id"]
    if current:
        from tsmcp.cleaning.common import array_hash
        from tsmcp.cleaning.engine import load_arrays

        prepared = read(workspace / "records" / case["record_id"] / "discovery.json")["parents"][case["id"]]
        released = read(workspace / "agent_state.json")["accepted"][case["id"]]["artifact_id"]
        ids = [released, prepared]
        for f in (workspace / "records" / case["record_id"] / "patches").glob("*.json"):
            doc = read(f)
            if doc["plan"]["write_ranges"][0]["block_id"] == case["id"]:
                ids.append(doc["candidate_version_id"])
        for aid in dict.fromkeys(ids):
            art = read(workspace / "artifacts" / aid / "artifact.json")
            if art["content_hash"] == source:
                _, t, x, v, e = load_arrays(workspace, case, art)
                if array_hash(x, t, v, e) != source:
                    raise ValueError("Seed version content changed")
                return case, region, x, v & ~e, source, revision
        raise ValueError("Candidate waveform version unavailable")
    if version_id(case) != source:
        raise ValueError("Original waveform differs from frozen trigger")
    x, masks = load_channel(path, case)
    return case, region, x, ~np.logical_or.reduce(list(masks.values())), source, revision


def investigate(workspace, inspection, candidate_id, out, flank, *, client_factory=BudgetClient):
    case, region, x, usable, source, revision = load_seed(workspace, inspection, candidate_id)
    if not usable[region["start_sample"] : region["end_sample"]].any():
        raise ValueError("Seed has no observed support for this morphology detector")
    out.mkdir(parents=True, exist_ok=True)
    # All follow-ups to this finalized workspace share one ledger, even in different output folders.
    # During active cleaning the controller uses the live run's existing BudgetClient.
    if not published_run(workspace):
        raise ValueError("Use the cleaning controller for an active workspace")
    cfg = read(workspace / "run.json").get("config", {}) if (workspace / "run.json").exists() else {}
    budget_root = (
        Path(cfg["budget_root"]).resolve()
        if cfg.get("budget_root")
        else workspace.resolve().parent / ".pattern-review-budgets" / workspace.resolve().name
    )
    budget_root.mkdir(parents=True, exist_ok=True)
    with (budget_root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(
            candidate_id=candidate_id,
            source=case["input_sha256"],
            inspection=revision,
            seed_version=source,
            accepted_version=read(workspace / "agent_state.json")["accepted_tasks"][case["task"]],
            flank_s=flank,
            reviewer_hash=sha(Path(__file__).with_name("reviewer.py")),
            detector_hash=sha(Path(__file__).with_name("pattern.py")),
        )
        item = out / "reviews" / digest(binding)
        item.mkdir(parents=True, exist_ok=True)
        if (item / "decision.json").exists():
            return read(item / "decision.json")
        (budget_root / "private").mkdir(mode=0o700, exist_ok=True)
        if not (budget_root / "api_budget.json").exists():
            if not (workspace / "api_budget.json").is_file():
                raise ValueError("Live follow-up requires an authorized model budget")
            ledger = read(workspace / "api_budget.json")
            prior = out / "api_budget.json"
            if prior.exists():
                previous = read(prior)
                if (
                    previous["calls"][: len(ledger["calls"])] != ledger["calls"]
                    or previous["cap_usd"] != ledger["cap_usd"]
                ):
                    raise ValueError("Incompatible budget history")
                ledger = previous
            if ledger["cap_usd"] > 5:
                raise ValueError("Expected the authorized shared cleaning budget")
            write(budget_root / "api_budget.json", ledger)
        client = client_factory(budget_root, cap=read(budget_root / "api_budget.json")["cap_usd"])
        try:
            vision_preflight(client)
            report = review(
                client, item, case, x, usable, source, [region["start_sample"], region["end_sample"]], flank
            )
        finally:
            client.close()
            write(out / "api_budget.json", read(budget_root / "api_budget.json"))
        report.update(binding=binding, decision_dir=str(item.resolve()))
        (item / "reviewer_source.py").write_bytes(Path(__file__).with_name("reviewer.py").read_bytes())
        write(item / "decision.json", report)
        (item / "detector_source.py").write_bytes(Path(__file__).with_name("pattern.py").read_bytes())
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--decision", type=Path, help="Replay a registered decision against the accepted batch offline")
    p.add_argument("--inspection", type=Path)
    p.add_argument("--candidate")
    p.add_argument("--flank", type=int, choices=(5, 10), default=5)
    args = p.parse_args()
    if args.live:
        if not args.inspection or not args.candidate:
            p.error("--live requires --inspection and --candidate")
        decision = investigate(args.workspace, args.inspection, args.candidate, args.out, args.flank)
    else:
        decision = read(args.decision)
    if decision["status"] != "registered":
        print(json.dumps(decision, ensure_ascii=False))
        raise SystemExit(2)
    from tsmcp.followup.rescan import scan_accepted

    version = read(args.workspace / "agent_state.json")["accepted_tasks"][decision["task"]]
    destination = args.out / "rescans" / digest(decision)[:24] / version
    report = scan_accepted(args.workspace, decision, destination)
    print(
        json.dumps(
            {
                "decision": decision.get("decision_dir"),
                "rescan": str(destination),
                "scanned": report["scanned"],
                "counts": report["counts"],
            },
            ensure_ascii=False,
        )
    )
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
