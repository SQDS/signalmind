"""CLI for audit, local baselines, actual Agent execution, evaluation and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read, write
from .config import task_config
from .data import audit
from .engine import execute_version, register_recipe


def baselines(workspace):
    workspace = Path(workspace).resolve()
    manifest = read(workspace / "dataset_manifest.json")
    versions = {}
    for method in ("identity", "fixed_dsp", "rule_adaptive_dsp"):
        recipe = register_recipe(workspace, kind=method)
        version = execute_version(workspace, recipe, manifest["cases"])
        versions[method] = version["version_id"]
        print(json.dumps({"method": method, "version": version["version_id"], "report": version["report"]}), flush=True)
        write(workspace / "baselines.json", versions)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pipeline = sub.add_parser("dataset", help="run the dataset-level grouping and review pipeline")
    pipeline.add_argument("--spec", type=Path, required=True)
    pipeline.add_argument("--workspace", type=Path, required=True)
    pipeline.add_argument("--live", action="store_true")
    pipeline.add_argument("--resume", action="store_true")
    a = sub.add_parser("audit")
    a.add_argument("--dataset", type=Path, required=True)
    a.add_argument("--inventory", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--patients", type=int, default=10)
    a.add_argument("--seed", type=int, default=111101)
    a.add_argument("--task-config", type=Path)
    a.add_argument("--purpose")
    a.add_argument("--patient-manifest", type=Path)
    a.add_argument("--reuse-development-cohort", action="store_true")
    for name in ("profile", "run", "evaluate", "export"):
        p = sub.add_parser(name)
        p.add_argument("--workspace", type=Path, required=True)
        if name == "run":
            p.add_argument("--agent", action="store_true")
            p.add_argument("--live", action="store_true")
            p.add_argument("--api-cap", type=float, default=5)
            p.add_argument("--model", default="deepseek-flash")
        if name == "evaluate":
            p.add_argument("--version")
            p.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.command == "dataset":
        from .dataset_entry import dataset_cli

        print(json.dumps(dataset_cli(args), ensure_ascii=False, indent=2))
        return
    if args.command == "audit":
        result = audit(
            args.dataset,
            args.inventory,
            args.out,
            count=args.patients,
            seed=args.seed,
            config=task_config(args.task_config, args.purpose),
            patient_manifest=args.patient_manifest,
            reuse_development=args.reuse_development_cohort,
        )
        result = {"patients": result["patients"], "cases": len(result["cases"])}
    elif args.command in {"profile", "run"} and not getattr(args, "agent", False):
        result = baselines(args.workspace)
    elif args.command == "run":
        if not args.live or not 0.35 < args.api_cap <= 5:
            raise ValueError("This run requires --live and an explicit API cap no greater than the authorized $5")
        from .agent import run

        result = run(args.workspace, cap=args.api_cap, model=args.model)
    elif args.command == "evaluate":
        if args.version:
            from .evaluation import evaluate_version

            result = evaluate_version(args.workspace, args.version, args.reference)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        result = {
            method: read(args.workspace / "versions" / vid / "version.json")["report"]
            for method, vid in read(args.workspace / "baselines.json").items()
        }
        if (args.workspace / "export/evaluation_report.json").exists():
            result["agent"] = read(args.workspace / "export/evaluation_report.json")
    else:
        # Export never creates a new model decision.
        path = args.workspace / "export/evaluation_report.json"
        if not path.exists():
            raise ValueError("No model-finalized output exists yet")
        result = read(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
