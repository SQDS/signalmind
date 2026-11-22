"""Reuse the established Bubblewrap isolation, with candidate-only writable artifacts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import digest, read, recipe_input_key, write
from .provenance import check_stop, emit


def run_job(workspace, cases, recipe, implementation, parents):
    check_stop(workspace)
    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from eval.processing_tools import PYTHON, sandbox_command

    inputs = [(c["id"], recipe_input_key(c)) for c in cases]
    job_id = digest([recipe, inputs, implementation, {k: v["artifact_id"] for k, v in parents.items()}])[:24]
    output = workspace / "jobs" / job_id
    output.mkdir(parents=True, exist_ok=True)
    if (output / "verified.json").exists():
        from .common import sha

        cached = read(output / "verified.json")
        for item in cached["results"].values():
            if item["arrays"] and sha(workspace / item["arrays"]) != item["arrays_sha256"]:
                raise ValueError("Worker checkpoint changed")
            if sha(workspace / item["probe_path"]) != item["probe_sha256"]:
                raise ValueError("Worker probe checkpoint changed")
        emit(workspace, "worker_cache_hit", job_id=job_id, cases=len(cases))
        return cached["results"], cached["errors"]
    public_link = output / "public"
    if not public_link.is_symlink():
        public_link.symlink_to("/case", target_is_directory=True)
    adapted = {}
    for key, value in parents.items():
        adapted[key] = dict(value)
        if value["arrays"]:
            adapted[key]["arrays"] = "/parents/" + str(Path(value["arrays"]).relative_to("artifacts"))
    write(output / "job.json", {"cases": cases, "recipe": recipe, "implementation": implementation, "parents": adapted})
    shutil.copyfile(workspace / "recipes" / recipe["hash"] / "program.py", output / "program.py")
    cache = output / "mpl-cache"
    cache.mkdir(exist_ok=True)
    from matplotlib import get_cachedir

    for font_cache in Path(get_cachedir()).glob("fontlist*.json"):
        shutil.copyfile(font_cache, cache / font_cache.name)
    command = sandbox_command(workspace / "public", output, agent=True)[:-1]
    (workspace / "artifacts").mkdir(exist_ok=True)
    command += [
        "--ro-bind",
        str(workspace / "artifacts"),
        "/parents",
        "--setenv",
        "MPLCONFIGDIR",
        "/output/mpl-cache",
        "--",
        str(PYTHON),
        "-I",
        "/opt/signalmind/src/tsmcp/cleaning/worker.py",
    ]
    for attempt in range(2):
        check_stop(workspace)
        attempt_id = uuid.uuid4().hex
        attempt_path = output / "attempts" / attempt_id
        attempt_path.mkdir(parents=True)
        started = time.monotonic()
        emit(
            workspace,
            "worker_attempt_started",
            job_id=job_id,
            attempt_id=attempt_id,
            attempt=attempt,
            recipe_hash=recipe["hash"],
            case_count=len(cases),
        )
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
            (attempt_path / "stdout.txt").write_text(completed.stdout)
            (attempt_path / "stderr.txt").write_text(completed.stderr)
            receipt = {
                "exit_code": completed.returncode,
                "elapsed_s": time.monotonic() - started,
                "attempt": attempt,
                "attempt_id": attempt_id,
            }
            write(attempt_path / "execution.json", receipt)
            emit(workspace, "worker_attempt_completed", job_id=job_id, **receipt)
            break
        except subprocess.TimeoutExpired as exc:
            receipt = {
                "status": "timeout",
                "attempt": attempt,
                "attempt_id": attempt_id,
                "elapsed_s": time.monotonic() - started,
            }
            write(attempt_path / "execution.json", receipt)
            (attempt_path / "stdout.txt").write_bytes(exc.stdout or b"")
            (attempt_path / "stderr.txt").write_bytes(exc.stderr or b"")
            emit(workspace, "worker_attempt_failed", job_id=job_id, **receipt)
            if attempt == 1:
                return {}, [
                    {"case_id": c["id"], "task": c["task"], "status": "execution_failed", "reason": "worker_timeout"}
                    for c in cases
                ]
    write(
        output / "execution.json",
        {
            "exit_code": completed.returncode,
            "technical_retries": attempt,
            "stderr": completed.stderr[-3000:],
            "network": "isolated",
            "references_mounted": False,
        },
    )
    if not (output / "result.json").exists() or completed.returncode:
        return {}, [
            {
                "case_id": c["id"],
                "task": c["task"],
                "status": "execution_failed",
                "reason": (completed.stderr or completed.stdout)[-1800:],
            }
            for c in cases
        ]
    result = read(output / "result.json")
    from .engine import evaluate_output, load_arrays
    from .recipes import STANDARD_OPERATIONS, RecipeContext

    verified, errors = {}, result["errors"]
    for case in cases:
        if case["id"] not in result["results"]:
            continue
        candidate = result["results"][case["id"]]
        try:
            raw, times, previous, valid, estimated = load_arrays(workspace, case, parents.get(case["id"]))
            if candidate["arrays"]:
                path = (output / candidate["arrays"]).resolve()
                if not path.is_relative_to(output.resolve()):
                    raise ValueError("Candidate path escapes its output workspace")
                with np.load(path, allow_pickle=False) as z:
                    y, y_valid, y_estimated = (z[k].copy() for k in ("values", "valid", "estimated"))
            else:
                y, y_valid, y_estimated = previous.copy(), valid.copy(), estimated.copy()
            if recipe["category"] == "standard_operators":
                replay = RecipeContext(previous, valid, estimated, case)
                for op in candidate["operations"]:
                    if op["op"] not in STANDARD_OPERATIONS:
                        raise ValueError("Unregistered numerical operation in standard recipe")
                    parameters = {
                        k: v
                        for k, v in op["parameters"].items()
                        if k in {"frequency_hz", "quality_factor", "filter_order", "window_s"}
                    }
                    replay.apply(op["op"], op["ranges_samples"], **parameters)
                if not (
                    np.allclose(y, replay.current.values, rtol=1e-10, atol=1e-12, equal_nan=True)
                    and np.array_equal(y_valid, replay.current.valid)
                    and np.array_equal(y_estimated, replay.current.repaired)
                ):
                    raise ValueError("Candidate does not reproduce its declared SignalMind operations")
            verified[case["id"]] = evaluate_output(
                workspace,
                case,
                recipe,
                implementation,
                parents.get(case["id"]),
                raw,
                times,
                previous,
                y,
                y_valid,
                y_estimated,
                candidate["operations"],
                time.monotonic(),
            )
            verified[case["id"]]["processing_environment"] = candidate.get("evaluation_environment")
            if candidate.get("operator_trace"):
                trace = candidate["operator_trace"]
                verified[case["id"]]["operator_trace"] = {
                    **trace,
                    "path": str((output / trace["path"]).relative_to(workspace)),
                }
            write(workspace / "artifacts" / verified[case["id"]]["artifact_id"] / "artifact.json", verified[case["id"]])
        except (ValueError, KeyError, OSError) as exc:
            errors.append(
                {"case_id": case["id"], "task": case["task"], "status": "execution_failed", "reason": str(exc)}
            )
    if not errors:
        write(output / "verified.json", {"results": verified, "errors": errors})
    emit(
        workspace,
        "record_completed",
        job_id=job_id,
        completed_cases=len(verified),
        failed_cases=len(errors),
        recipe_hash=recipe["hash"],
    )
    return verified, errors


def run_recipe(workspace, cases, recipe, implementation, parents):
    groups = {}
    for case in cases:
        groups.setdefault(case["record_id"], []).append(case)
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                run_job,
                Path(workspace),
                group,
                recipe,
                implementation,
                {c["id"]: parents[c["id"]] for c in group if c["id"] in parents},
            )
            for group in groups.values()
        ]
        for future in futures:
            part, failed = future.result()
            results.update(part)
            errors.extend(failed)
    return results, errors
