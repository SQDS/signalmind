"""Review one frozen trigger, then search accepted waveforms for the same morphology."""

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from tsmcp.cleaning.agent import BudgetClient, vision_preflight
from tsmcp.cleaning.common import digest, read, sha, write
from tsmcp.followup.reviewer import review
from tsmcp.inspection.run import load_channel, version_id


def investigate(workspace, inspection, candidate_id, out, flank, *, client_factory=BudgetClient):
    frozen = read(inspection / "candidate_manifest.json")
    cand_path = inspection / "candidates.jsonl"
    if sha(cand_path) != frozen["files"]["candidates.jsonl"]:
        raise ValueError("Frozen candidate manifest mismatch")
    with cand_path.open() as stream:
        candidate = next(
            (
                json.loads(line)
                for line in stream
                if candidate_id in line and json.loads(line)["candidate_id"] == candidate_id
            ),
            None,
        )
    if candidate is None or candidate["run_id"] != frozen["run_id"]:
        raise ValueError("Unknown candidate")
    region = candidate["range"]
    case = next(c for c in read(workspace / "dataset_manifest.json")["cases"] if c["id"] == region["block_id"])
    path = workspace / "public" / case["id"] / "input.npz"
    if version_id(case) != candidate["source_version_id"] or sha(path) != case["input_sha256"]:
        raise ValueError("Original waveform differs from frozen trigger")
    x, masks = load_channel(path, case)
    usable = ~np.logical_or.reduce(list(masks.values()))
    if not usable[region["start_sample"] : region["end_sample"]].any():
        raise ValueError("Seed has no observed support for this morphology detector")
    out.mkdir(parents=True, exist_ok=True)
    # All follow-ups to this finalized workspace share one ledger, even in different output folders.
    # During active cleaning the controller uses the live run's existing BudgetClient.
    state = read(workspace / "agent_state.json")
    if not state.get("finalized", state.get("complete", False)):
        raise ValueError("Use the cleaning controller for an active workspace")
    budget_root = workspace.resolve().parent / ".pattern-review-budgets" / workspace.resolve().name
    budget_root.mkdir(parents=True, exist_ok=True)
    with (budget_root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(
            candidate_id=candidate_id,
            source=case["input_sha256"],
            inspection=frozen["run_id"],
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
                client, item, case, x, usable, version_id(case), [region["start_sample"], region["end_sample"]], flank
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
