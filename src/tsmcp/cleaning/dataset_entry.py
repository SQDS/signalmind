"""Create or resume a reproducible workbench-compatible dataset job from a single specification."""

from __future__ import annotations

import fcntl
from pathlib import Path

from tsmcp.inspection.scan import PRESETS

from .common import digest, implementation_hash, read, sha, write
from .context import ContextBuilder
from .provenance import freeze_implementation
from .repairability import domain_for

METHODS = {"dataset_rules", "dataset_agent", "inspect_only"}


def settings(config):
    c = dict(config)
    if c.get("inspection_preset", "balanced") not in PRESETS:
        raise ValueError("Unknown inspection preset")
    c.setdefault("inspection_preset", "balanced")
    c.setdefault("purpose", "Native ECG RR, PPG PPI and beat pressure")
    c.setdefault("tasks", ["ECG", "PPG", "ABP"])
    modalities = {"ECG", "PPG", "ABP", "RESP"} if c["method"] == "inspect_only" else {"ECG", "PPG", "ABP"}
    if not c["tasks"] or not set(c["tasks"]) <= modalities:
        raise ValueError("Select supported TaskPacks")
    c.setdefault("routing", "reason")
    if c["routing"] not in ("reason", "jev"):
        raise ValueError("routing must be reason or jev")
    c.setdefault("model", "deepseek-flash")
    if c["model"] != "deepseek-flash":
        raise ValueError("Only deepseek-flash is configured")
    c.setdefault("api_cap", 5 if c["method"] == "dataset_agent" else 0)
    if c["method"] == "dataset_agent" and not 0.35 < float(c["api_cap"]) <= 5:
        raise ValueError("Expected DS budget over 0.35 and up to 5 USD")
    if c["method"] in ("dataset_rules", "inspect_only") and (c["routing"] == "jev" or c["api_cap"] != 0):
        raise ValueError("Local rules require zero API budget and reason routing")
    for key, default in [("joint_views", 8), ("write_views", 50), ("jev_token_cap", 0)]:
        c.setdefault(key, default)
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f"Invalid {key}")
    if c["routing"] == "jev" and c["jev_token_cap"] <= 0:
        raise ValueError("Set jev_token_cap for JEV routing")
    domain_for(c)
    ContextBuilder(c)
    if type(c.get("inspection_wall_seconds", 3600)) is not int or c.get("inspection_wall_seconds", 3600) <= 0:
        raise ValueError("Invalid inspection wall budget")
    return c


def dataset_cli(args):
    from tsmcp.workbench.runner import execute_run

    spec_path = args.spec.resolve()
    spec = read(spec_path)
    dataset = dict(spec["dataset"])
    for key in ("path", "inventory"):
        if key in dataset:
            dataset[key] = str((spec_path.parent / dataset[key]).resolve())
    dataset.setdefault("id", "dataset-" + digest([dataset["format"], dataset["path"]])[:16])
    if dataset["format"] == "vital" and "inventory" not in dataset:
        raise ValueError("Vital requires its file inventory")
    config = settings(spec["config"])
    if "RESP" in config["tasks"] and dataset["format"] == "vital":
        raise ValueError("Map the respiration channel in CSV or NPZ for inspection")
    if config["method"] not in METHODS:
        raise ValueError("Use dataset_agent or dataset_rules")
    if config.get("patient_manifest"):
        value = read(spec_path.parent / config.pop("patient_manifest"))
        config["patient_keys"] = value["patients"] if isinstance(value, dict) else value
    for key in ("replay_from", "budget_root", "receipt_seed", "knowledge_root"):
        if config.get(key):
            config[key] = str((spec_path.parent / config[key]).resolve())
    if config["method"] == "dataset_agent" and not config.get("replay_from") and not args.live:
        raise ValueError("Pass --live for paid model calls")
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "run.json"
        signature = digest([dataset, config, sha(Path(dataset["inventory"])) if "inventory" in dataset else None])
        if path.exists():
            if read(path).get("spec_hash") != signature:
                raise ValueError("Workspace already belongs to another run specification")
        else:
            if any(f.name != ".setup.lock" for f in root.iterdir()):
                raise ValueError("Choose an empty workspace for a new run")
            freeze_implementation(root)
            write(
                path,
                dict(
                    id=root.name,
                    name=root.name,
                    config=config,
                    dataset=dataset,
                    status="queued",
                    stage="queued",
                    attempts=0,
                    implementation_hash=implementation_hash(),
                    spec_hash=signature,
                ),
            )
    info = execute_run(root, resume=args.resume)
    if info["status"] not in ("completed", "partial"):
        raise RuntimeError(info.get("error") or info["status"])
    return read(root / "export/evaluation_report.json")
