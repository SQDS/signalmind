"""Resolve the three supported purposes without inventing unsupported clinical tasks."""

from pathlib import Path

from .common import TASKS, digest, read


def task_config(path=None, purpose=None):
    document = {}
    if path:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            document = yaml.safe_load(path.read_text())
        else:
            document = read(path)
    known = {v["task_id"]: k for k, v in TASKS.items()}
    if "taskpacks" in document:
        tasks = [known[r["task_id"]] for r in document["taskpacks"] if r.get("enabled", True)]
    elif document.get("task", {}).get("signal_kind"):
        tasks = [document["task"]["signal_kind"]]
    else:
        tasks = list(TASKS)
    if not tasks or any(k not in TASKS for k in tasks):
        raise ValueError("v1 supports ECG/RR, native PPG/PPI and ABP beat pressure TaskPacks")
    return {
        "tasks": tasks,
        "purpose": purpose or document.get("task", {}).get("purpose", "Three native waveform tasks"),
        "config_hash": digest(document),
        "status": "engineering_unlabeled",
        "reference_metrics": None,
    }
