"""Dataset-level CLI. prepare/batch/audit are entirely local; develop makes one model session."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dataset_tools import DatasetTools  # noqa: E402
from eval.dataset_workflow import build_project, read, run_batch, sha, trial, write  # noqa: E402
from eval.unified.drivers import cost_of, make_driver  # noqa: E402
from eval.unified.loop import Limits, run_session  # noqa: E402

SYSTEM = """You are SignalMind's dataset-level workflow developer. Your deliverable is one reusable executable
processing policy for a whole physiological waveform dataset. Use dataset_overview, inspect representative
development examples including ordinary observations, try policies locally, compare diagnostics, and finalize
one tested policy. Never create per-example or per-record special cases. Empty rules preserve a channel.
The local executor applies the same policy to complete continuous runs without any model calls. Trials are
short sampled runs; consider how your choices generalize to longer runs. Input/units/time are preserved.
Do not equate smoother signals, changed candidate metrics or masked support with restored physiological truth.
Keep genuine pulse/QRS structure and slow pressure trends. Narrowband and outlier features are evidence to
inspect, not ground truth. No clean reference or manual clinical labels are available. Do not use indiscriminate
filtering merely to show action. ECG/PPG physical units are unconfirmed source_units: do not invent mV or
apply an absolute amplitude threshold as a cross-record saturation diagnosis. Device saturation flags are
unavailable. Band power alone is not proof of line interference; consider the neighboring spectrum too.
Do not claim validation data have been reviewed: they are hidden until after
policy freeze. You have 24 tool calls; reserve a call for pipeline_finalize and finish after it succeeds.
"""


async def develop(args):
    project = args.project.resolve()
    if (project / "development-session.json").exists() or (project / "policy.json").exists():
        raise ValueError("This project already has a development attempt; preserve it and use a new workspace")
    if not args.live or args.cap <= 0.05:
        raise ValueError("Development requires --live and a positive --cap above the $0.05 next-request reserve")
    tools = DatasetTools(project)
    driver = make_driver("deepseek-flash", max_tokens=8192)
    sources = [
        "src/tsmcp/pipeline.py",
        "src/tsmcp/processing.py",
        "eval/dataset_workflow.py",
        "eval/dataset_tools.py",
        "eval/run_dataset.py",
        "eval/unified/drivers.py",
        "eval/unified/loop.py",
    ]
    provenance = {
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "model": "deepseek-flash",
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "cap_usd": args.cap,
        "billing_basis": "returned-usage peak-price estimate",
        "goal": args.goal,
    }
    write(project / "development-provenance.json", provenance)
    try:
        result = await run_session(
            driver,
            tools,
            SYSTEM,
            args.goal,
            Limits(max_turns=30, max_tool_calls=24, max_seconds=900, max_output_tokens_total=40000),
            project / "private/development-transcript.jsonl",
            provenance,
            budget_exhausted=lambda usage: cost_of("deepseek-flash", usage) + 0.05 >= args.cap,
        )
    finally:
        driver.close()
    result["estimated_peak_usd"] = cost_of("deepseek-flash", result["usage"])
    result["policy_finalized"] = (project / "policy.json").exists()
    result["actual_billed_usd"] = None
    write(project / "development-session.json", result)
    if result["policy_finalized"]:
        validation = trial(project, read(project / "policy.json"), "validation")
        write(
            project / "validation-summary.json",
            {
                "policy_sha256": validation["policy_sha256"],
                **validation["summary"],
                "purpose": "Post-freeze unseen-record diagnostics; no clean-reference accuracy score.",
            },
        )
    return {k: v for k, v in result.items() if k not in {"text", "usage"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Inventory an entire PulseDB-Vital directory and select disjoint examples")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument("--inventory", type=Path)
    prepare.add_argument("--development-records", type=int, default=4)
    prepare.add_argument("--validation-records", type=int, default=4)
    dev = sub.add_parser("develop", help="One DS tool session develops a reusable policy; validation follows freeze")
    dev.add_argument("--project", type=Path, required=True)
    dev.add_argument("--goal", required=True)
    dev.add_argument("--cap", type=float, required=True)
    dev.add_argument("--live", action="store_true")
    batch = sub.add_parser("batch", help="Run a frozen policy locally; no API requests")
    batch.add_argument("--project", type=Path, required=True)
    batch.add_argument("--policy", type=Path)
    batch.add_argument("--scope", choices=["development", "validation", "all"], default="validation")
    batch.add_argument("--limit", type=int)
    audit = sub.add_parser("audit", help="Read saved inventory, development cost, validation and batch reports")
    audit.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_project(
            args.source, args.project, args.inventory, args.development_records, args.validation_records
        )
    elif args.command == "develop":
        result = asyncio.run(develop(args))
    elif args.command == "batch":
        manifest = read(args.project / "private/manifest.json")
        ids = [r["id"] for r in manifest["records"]] if args.scope == "all" else manifest[args.scope + "_records"]
        if args.limit is not None:
            ids = ids[: args.limit]
        result = run_batch(args.project, read(args.policy or args.project / "policy.json"), ids)
        result = {k: v for k, v in result.items() if k != "records"}
    else:
        result = {
            p.name: read(p)
            for p in [
                args.project / "inventory-summary.json",
                args.project / "development-session.json",
                args.project / "validation-summary.json",
            ]
            if p.exists()
        }
        result["batches"] = [
            {k: v for k, v in read(p).items() if k != "records"} for p in (args.project / "batch").glob("*/report.json")
        ]
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
