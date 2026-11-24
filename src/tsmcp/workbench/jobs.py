"""One queued Agent process, durable checkpoints, and explicit stop/resume."""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from tsmcp.cleaning.common import implementation_hash, read, sha, write
from tsmcp.cleaning.provenance import emit, freeze_implementation


class Jobs:
    def __init__(self, store):
        self.store = store
        self.closing = threading.Event()
        self.lock = threading.Lock()
        self.process = None
        self.active_root = None
        self.thread = None

    def start(self):
        for row in self.store.runs():
            root = self.store.run_path(row["id"])
            if (root / "run.json").exists():
                info = read(root / "run.json")
                if info["status"] in {"running", "stopping"}:
                    if self.runner_alive(root):
                        continue
                    info["status"] = "interrupted"
                    info["error"] = "Service restarted; resume from the saved checkpoint"
                    write(root / "run.json", info)
        self.thread = threading.Thread(target=self.loop, name="signalmind-job-queue", daemon=True)
        self.thread.start()

    @staticmethod
    def runner_alive(root):
        path = root / "runner.pid"
        if not path.exists():
            return False
        proc = Path("/proc") / path.read_text().strip() / "cmdline"
        try:
            command = proc.read_bytes().split(b"\0")
            return b"tsmcp.workbench.runner" in command and str(root).encode() in command
        except FileNotFoundError:
            return False

    def create(self, config):
        dataset = self.store.dataset(config["dataset_id"])
        tasks = config.get("tasks", ["ECG", "PPG", "ABP"])
        method = config.get("method", "agent")
        modalities = (
            {"ECG", "PPG", "ABP", "RESP"}
            if method == "inspect_only" and dataset["format"] != "vital"
            else {"ECG", "PPG", "ABP"}
        )
        if not tasks or not set(tasks) <= modalities:
            raise ValueError("Select supported tasks")
        cap = float(config.get("api_cap", 5))
        if method == "inspect_only":
            from tsmcp.inspection.scan import PRESETS

            if config.get("inspection_preset", "balanced") not in PRESETS:
                raise ValueError("Unknown inspection preset")
            cap = 0.0
        if method in {"dataset_rules", "dataset_agent"}:
            from tsmcp.cleaning.dataset_entry import settings

            if any(k in config for k in ("budget_root", "replay_from", "receipt_seed", "knowledge_root")):
                raise ValueError("External ledgers and receipt directories are CLI options")
            config = settings({**config, "api_cap": 0 if method == "dataset_rules" else cap})
            cap = config["api_cap"]
        if method not in {"agent", "baselines", "inspect_only", "dataset_rules", "dataset_agent"} or (
            method == "agent" and not 0.35 < cap <= 5
        ):
            raise ValueError("Agent budget must be > $0.35 and <= $5; choose agent, baselines or inspect_only")
        if config.get("model", "deepseek-flash") != "deepseek-flash":
            raise ValueError("The validated visual runner uses deepseek-flash")
        rid = "clean-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        root = self.store.root / "runs" / rid
        root.mkdir(parents=True)
        freeze_implementation(root)
        info = {
            "id": rid,
            "name": config.get("name") or rid,
            "status": "queued",
            "stage": "queued",
            "created_at": datetime.now(UTC).isoformat(),
            "config": {**config, "tasks": tasks, "method": method, "api_cap": cap},
            "dataset": dataset,
            "implementation_hash": implementation_hash(),
            "attempts": 0,
        }
        write(root / "run.json", info)
        self.store.add_run(root)
        emit(root, "run_queued", actor="user", config=info["config"])
        return info

    def stop(self, rid):
        with self.lock:
            root = self.store.run_path(rid)
            info = read(root / "run.json")
            if info["status"] not in {"queued", "running", "stopping"}:
                raise ValueError("This task is not active")
            (root / "stop.requested").touch()
            if info["status"] == "queued":
                info["status"] = "stopped"
                write(root / "run.json", info)
            else:
                # The runner owns its state file while active; the flag is the command.
                info["status"] = "stopping"
            emit(root, "stop_requested", actor="user")
        return info

    def resume(self, rid):
        with self.lock:
            root = self.store.run_path(rid)
            info = read(root / "run.json")
            if info["status"] not in {"stopped", "interrupted", "failed"}:
                raise ValueError("Only a stopped/interrupted/failed task can resume")
            if self.runner_alive(root):
                raise ValueError("The previous process is still reaching its checkpoint")
            if info["implementation_hash"] != implementation_hash():
                raise ValueError("Implementation changed; create a new linked run instead")
            manifest_path = root / "dataset_manifest.json"
            if manifest_path.exists():
                for c in read(manifest_path)["cases"]:
                    if sha(root / "public" / c["id"] / "input.npz") != c["input_sha256"]:
                        raise ValueError("Frozen task input changed")
            if (root / "agent_state.json").exists() and read(root / "agent_state.json").get("finalized"):
                raise ValueError("This Agent session was finalized; create a new linked run")
            (root / "stop.requested").unlink(missing_ok=True)
            info.update(status="queued", error=None)
            write(root / "run.json", info)
            emit(root, "resume_requested", actor="user")
        return info

    def loop(self):
        while not self.closing.wait(0.5):
            with self.lock:
                queued = []
                external_active = False
                for row in self.store.runs():
                    root = self.store.run_path(row["id"])
                    if (root / "run.json").exists():
                        info = read(root / "run.json")
                        external_active |= self.runner_alive(root)
                        if info["status"] == "queued":
                            queued.append((info["created_at"], root, info))
                if not queued or external_active:
                    continue
                _, root, info = min(queued, key=lambda x: x[0])
                info.update(status="running", attempts=info["attempts"] + 1)
                write(root / "run.json", info)
                self.active_root = root
                logs = root / "runner_attempts"
                logs.mkdir(exist_ok=True)
                stream = (logs / f"{info['attempts']:03d}.log").open("w")
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "tsmcp.workbench.runner", str(root)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    cwd=self.store.repo,
                )
                (root / "runner.pid").write_text(str(self.process.pid))
            code = self.process.wait()
            stream.close()
            info = read(root / "run.json")
            if info["status"] in {"running", "stopping"}:
                info.update(status="interrupted", error=f"Runner exited {code} before a terminal receipt")
                write(root / "run.json", info)
            self.process, self.active_root = None, None

    def close(self):
        self.closing.set()
        if self.active_root:
            (self.active_root / "stop.requested").touch()
        if self.thread:
            # Current record/model call is allowed to reach its checkpoint.
            self.thread.join(timeout=1)
