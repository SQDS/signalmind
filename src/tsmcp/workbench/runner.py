"""Execution subprocess used by the local queue; no web-server state is required."""

from __future__ import annotations

import fcntl
import sys
from pathlib import Path

from tsmcp.cleaning.common import implementation_hash, read, write
from tsmcp.cleaning.provenance import StopRequested, check_stop, emit


def execute_run(root, agent_runner=None, *, resume=False):
    root = Path(root)
    with (root / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            (root / "stop.requested").unlink(missing_ok=True)
        return _execute_run(root, agent_runner)


def _execute_run(root, agent_runner=None):
    from tsmcp.cleaning.cli import baselines
    from tsmcp.cleaning.data import audit

    from .datasets import import_dataset

    root = Path(root)
    info = read(root / "run.json")
    config, dataset = info["config"], info["dataset"]

    def stage(value):
        info.update(stage=value, error=None)
        # Preserve a simultaneous stop request instead of resetting it to running.
        info["status"] = "stopping" if (root / "stop.requested").exists() else "running"
        write(root / "run.json", info)
        emit(root, "stage_started", stage=value)

    try:
        if info["implementation_hash"] != implementation_hash():
            raise ValueError("Implementation changed after task creation")
        check_stop(root)
        if not (root / "dataset_manifest.json").exists():
            stage("audit")
            if dataset["format"] == "vital":
                previously_used = set()
                for selection in root.parent.glob("*/private/selection.json"):
                    if selection.parent.parent != root:
                        previously_used.update(read(selection).get("patients", []))
                patient_manifest = None
                if config.get("patient_keys"):
                    patient_manifest = root / "requested-patients.json"
                    write(patient_manifest, config["patient_keys"])
                audit(
                    dataset["path"],
                    dataset["inventory"],
                    root,
                    count=len(config["patient_keys"])
                    if config.get("patient_keys")
                    else int(config.get("patients", 10)),
                    seed=int(config.get("seed", 111101)),
                    patient_manifest=patient_manifest,
                    reuse_development=bool(config.get("reuse_development", False)),
                    excluded_patients=previously_used,
                    resume=True,
                    config={"tasks": config["tasks"], "purpose": config.get("purpose", "Native waveform tasks")},
                )
            else:
                import_dataset(
                    dataset,
                    {
                        **dataset.get("mapping", {}),
                        **config.get("mapping", {}),
                        "purpose": config.get("purpose", "Native waveform tasks"),
                    },
                    root,
                    config["tasks"],
                    config.get("files"),
                )
        if not read(root / "dataset_manifest.json")["cases"]:
            raise ValueError("Audit produced no runnable channels; inspect data_audit.json")
        if config["method"] == "inspect_only":
            from tsmcp.inspection.run import inspect

            stage("inspection")
            result = inspect(root, root / "inspection", config.get("inspection_preset", "balanced"))
            status = "completed" if result["inspection_complete"] else "partial"
            write(root / "export/evaluation_report.json", {**result, "status": status})
            info.update(status=status, stage=status)
            write(root / "run.json", info)
            emit(root, "run_finished", status=status)
            return info
        stage("baselines")
        baselines(root)
        check_stop(root)
        if config["method"] in {"dataset_agent", "dataset_rules"}:
            from tsmcp.cleaning.dataset_pipeline import run_dataset

            result = run_dataset(root, config, stage)
            info.update(status=result["status"], error=None)
        elif config["method"] == "agent":
            stage("agent")
            if agent_runner is None:
                from tsmcp.cleaning.agent import run

                agent_runner = run
            result = agent_runner(root, cap=config["api_cap"], model=config.get("model", "deepseek-flash"))
            status = (
                result["stop"]
                if result["stop"] in {"stopped", "interrupted"}
                else "completed"
                if result["workflow_complete"]
                else "partial"
            )
            info.update(status=status, error=result.get("error"))
        else:
            manifest = read(root / "dataset_manifest.json")
            methods = read(root / "baselines.json")
            reports = {name: read(root / "versions" / vid / "version.json")["report"] for name, vid in methods.items()}
            write(root / "export/baseline_reports.json", reports)
            info["status"] = (
                "partial" if manifest["audit_failures"] or any(r["errors"] for r in reports.values()) else "completed"
            )
        info["stage"] = info["status"]
    except StopRequested as exc:
        info.update(status="stopped", error=str(exc))
    except Exception as exc:
        info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write(root / "run.json", info)
    emit(root, "run_finished", status=info["status"], error=info.get("error"))
    return info


if __name__ == "__main__":
    execute_run(Path(sys.argv[1]))
