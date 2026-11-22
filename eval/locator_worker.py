"""Reference-blind persistent worker, with a fresh resource-limited child for each case."""

from __future__ import annotations

import importlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/impl/src")

import numpy as np

from tsmcp.cleaning.common import read, sha, write
from tsmcp.locators.api import locate_issues

# Warm numerical imports once, before single-threaded fork; no data or references loaded.
for module in ("tsmcp.locators.legacy", "tsmcp.locators.sphere", "tsmcp.locators.vendor.time_sphere_original"):
    importlib.import_module(module)


def one_channel(array):
    if array.ndim == 1:
        return array.copy()
    if array.ndim == 2 and array.shape[0] == 1:
        return array[0].copy()
    raise ValueError("Only explicitly single-channel input is accepted")


def execute(job, folder):
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    started = time.perf_counter()
    path = Path("/inputs") / job["input_relative"]
    if sha(path) != job["input_sha256"]:
        raise ValueError("Input checksum differs from frozen manifest")
    with np.load(path, allow_pickle=False) as z:
        values = one_channel(z["values"]).astype(np.float64)
        times = one_channel(z["time"]) if "time" in z else np.arange(len(values)) / job["fs"]
        flags = np.zeros(len(values), bool)
        for key in ("observed_missing_mask", "device_saturation_flag"):
            if key in z:
                flags |= one_channel(z[key]).astype(bool)
    source = np.ones(len(values), bool)
    for a, b in job.get("source_exclusions", []):
        source[a:b] = False
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    result = locate_issues(
        values,
        job["fs"],
        method=job["method"],
        source_support=source,
        times=times,
        device_flags=flags,
        include_mesh=job.get("include_mesh", False),
    )
    algorithm_s = time.perf_counter() - started
    scores = result.pop("sample_scores")
    if np.isfinite(scores).any():
        np.savez_compressed(folder / "scores.npz", scores=scores)
        result["scores_file"] = "scores.npz"
    if result.get("meshes"):
        payloads = result.pop("meshes")
        result["mesh_files"] = []
        for i, payload in enumerate(payloads):
            mesh = payload["mesh"]
            arrays = {k: v for k, v in mesh.items() if isinstance(v, np.ndarray)}
            metadata = {k: v for k, v in mesh.items() if not isinstance(v, np.ndarray)}
            np.savez_compressed(folder / f"mesh-{i}.npz", **arrays)
            write(folder / f"mesh-{i}.json", metadata)
            result["mesh_files"].append(
                {"range": payload["range"], "arrays": f"mesh-{i}.npz", "metadata": f"mesh-{i}.json"}
            )
    result.update(
        case_id=job["case_id"],
        stage=job["stage"],
        input_sha256=job["input_sha256"],
        implementation_hash=job["implementation_hash"],
        runtime={
            "load_s": load_s,
            "algorithm_s": algorithm_s,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )
    write(folder / "result.json", result)


def process(job):
    folder = Path("/output") / job["output_relative"]
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:
        try:
            log = (folder / "stderr.txt").open("w")
            os.dup2(log.fileno(), 2)
            execute(job, folder)
        except Exception as exc:
            traceback.print_exc()
            write(
                folder / "result.json",
                {
                    "method": job["method"],
                    "case_id": job["case_id"],
                    "stage": job["stage"],
                    "status": "failed",
                    "events": [],
                    "coverage_known": True,
                    "evaluated_ranges": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime": {"algorithm_s": None, "load_s": None},
                },
            )
            os._exit(1)
        os._exit(0)
    timed_out = False
    while True:
        done, status, usage = os.wait4(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() - started > job["timeout_s"]:
            os.kill(pid, signal.SIGKILL)
            _, status, usage = os.wait4(pid, 0)
            timed_out = True
            break
        time.sleep(0.01)
    if timed_out or not (folder / "result.json").exists():
        write(
            folder / "result.json",
            {
                "method": job["method"],
                "case_id": job["case_id"],
                "stage": job["stage"],
                "status": "timeout" if timed_out else "failed",
                "events": [],
                "coverage_known": True,
                "evaluated_ranges": [],
                "error": "hard_60s_timeout" if timed_out else f"worker_exit_{status}",
                "runtime": {"algorithm_s": None, "load_s": None},
            },
        )
    result = read(folder / "result.json")
    result["runtime"]["peak_rss_mb"] = usage.ru_maxrss / 1024
    result["runtime"]["job_wall_s"] = time.monotonic() - started
    result["input_sha256"] = job["input_sha256"]
    result["implementation_hash"] = job["implementation_hash"]
    result["isolation"] = {
        "network": "unshared",
        "references_mounted": False,
        "threads": 1,
        "memory_limit_gib": 4,
        "timeout_s": job["timeout_s"],
        "automatic_retries": 0,
    }
    write(folder / "result.json", result)
    return {"status": result["status"], "output_relative": job["output_relative"], "runtime": result["runtime"]}


def main():
    print(
        json.dumps(
            {
                "ready": True,
                "pid": os.getpid(),
                "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "home_credentials_visible": Path("/home/yanghaoxiang24/.config/tsmcp-eval").exists(),
                "reference_visible": Path(
                    "/home/yanghaoxiang24/signal_agent/eval/cases/processing-v0.3/SignalMind_Test_Pack_v0.3/private"
                ).exists(),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        job = json.loads(line)
        print(json.dumps(process(job)), flush=True)


if __name__ == "__main__":
    main()
