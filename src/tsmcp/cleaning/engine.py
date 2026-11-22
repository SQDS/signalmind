"""Immutable waveform artifacts and local execution shared by baselines and Agent."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tsmcp.processing import intervals

from .common import (
    TASKS,
    array_hash,
    changed,
    digest,
    implementation_hash,
    no_reference,
    read,
    recipe_input_key,
    sha,
    time_support,
    write,
)
from .provenance import StepRecorder, StopRequested, check_stop, emit
from .recipes import diagnose, execute
from .tasks import probe


def load_arrays(workspace, case, artifact=None):
    workspace = Path(workspace)
    raw_path = workspace / "public" / case["id"] / "input.npz"
    with np.load(raw_path, allow_pickle=False) as z:
        raw, times = z["values"].copy(), z["time"].copy()
        source_valid = z["valid"].astype(bool) if "valid" in z else np.isfinite(raw)
        source_estimated = z["estimated"].astype(bool) if "estimated" in z else np.zeros(len(raw), bool)
    if artifact and artifact.get("arrays"):
        with np.load(workspace / artifact["arrays"], allow_pickle=False) as z:
            values, valid, estimated = (z[k].copy() for k in ("values", "valid", "estimated"))
    else:
        values, valid, estimated = raw.copy(), source_valid, source_estimated
    valid &= time_support(case)
    return raw, times, values, valid, estimated


def process_one(workspace, case, recipe, implementation, parent=None, module=None):
    workspace = Path(workspace)
    check_stop(workspace)
    checkpoint = (
        workspace
        / "case_checkpoints"
        / (
            digest(
                [recipe_input_key(case), case["id"], recipe, implementation, parent["artifact_id"] if parent else None]
            )
            + ".json"
        )
    )
    if checkpoint.exists():
        saved = read(checkpoint)
        if (not saved["arrays"] or sha(workspace / saved["arrays"]) == saved["arrays_sha256"]) and sha(
            workspace / saved["probe_path"]
        ) == saved["probe_sha256"]:
            emit(workspace, "case_cache_hit", case_id=case["id"], artifact_id=saved["artifact_id"])
            return saved
        raise ValueError("Checkpoint artifact changed")
    started = time.monotonic()
    emit(workspace, "case_started", case_id=case["id"], recipe_hash=recipe["hash"])
    raw, times, previous, valid, estimated = load_arrays(workspace, case, parent)
    recorder = (
        StepRecorder(workspace, case, recipe, implementation, parent["artifact_id"] if parent else None)
        if module
        else None
    )
    values, valid, estimated, operations = execute(previous, valid, estimated, case, recipe, module, recorder)
    result = evaluate_output(
        workspace,
        case,
        recipe,
        implementation,
        parent,
        raw,
        times,
        previous,
        values,
        valid,
        estimated,
        operations,
        started,
    )
    if recorder:
        result["operator_trace"] = recorder.reference
        write(workspace / "artifacts" / result["artifact_id"] / "artifact.json", result)
    write(checkpoint, result)
    emit(
        workspace,
        "case_completed",
        case_id=case["id"],
        artifact_id=result["artifact_id"],
        recipe_hash=recipe["hash"],
        elapsed_s=time.monotonic() - started,
    )
    return result


def evaluate_output(
    workspace, case, recipe, implementation, parent, raw, times, previous, values, valid, estimated, operations, started
):
    """Trusted postprocessing runs independently of model-authored Python."""
    workspace = Path(workspace)
    values, valid, estimated = np.asarray(values, float), np.asarray(valid, bool), np.asarray(estimated, bool)
    if any(a.shape != raw.shape for a in (values, valid, estimated)) or np.any(valid & ~np.isfinite(values)):
        raise ValueError("Invalid output sample grid or valid nonfinite samples")
    if np.any(valid & ~time_support(case)):
        raise ValueError("Source time-conflict support cannot be declared usable")
    content = array_hash(values, times, valid, estimated)
    aid = digest(
        {
            "content": content,
            "case": case,
            "recipe": recipe["hash"],
            "implementation": implementation,
            "parent": parent["artifact_id"] if parent else None,
        }
    )
    folder = workspace / "artifacts" / aid
    folder.mkdir(parents=True, exist_ok=True)
    delta = changed(raw, values)
    local_changed = changed(previous, values)
    with np.load(workspace / "public" / case["id"] / "input.npz", allow_pickle=False) as source:
        initial_valid = source["valid"].astype(bool) if "valid" in source else np.isfinite(raw)
        initial_estimated = source["estimated"].astype(bool) if "estimated" in source else np.zeros(len(raw), bool)
    support_changed = (valid != (initial_valid & time_support(case))) | (estimated != initial_estimated)
    focus = None
    finite_difference = np.where(np.isfinite(raw) & np.isfinite(values), np.abs(values - raw), 0)
    if finite_difference.any():
        focus = int(np.argmax(finite_difference))
    elif (delta | support_changed).any():
        left, right = max(intervals(delta | support_changed), key=lambda r: r[1] - r[0])
        focus = (left + right) // 2
    arrays = None
    if delta.any() or not np.array_equal(valid, np.isfinite(raw)) or estimated.any():
        np.savez_compressed(
            folder / "waveform.npz",
            values=values,
            valid=valid,
            estimated=estimated,
            observed=np.isfinite(raw),
            modified=delta,
        )
        arrays = str((folder / "waveform.npz").relative_to(workspace))
    cache = workspace / "probe_cache" / digest([case["id"], content, implementation])
    if (cache / "probe.json").exists():
        measures = read(cache / "probe.json")
        diagnostics = read(cache / "diagnosis.json")
    else:
        measures = probe(values, times, case["fs"], valid, estimated, case["task"])
        diagnostics = diagnose(values, case["fs"], valid & ~estimated)
        write(cache / "probe.json", measures)
        write(cache / "diagnosis.json", diagnostics)
    eligible = valid & ~estimated & np.isfinite(values)
    common = np.isfinite(raw) & np.isfinite(values)
    shift = float(np.mean(values[common] - raw[common])) if common.any() else None
    stats = {
        "n": len(raw),
        "duration_s": len(raw) / case["fs"],
        "modified_samples": int(delta.sum()),
        "support_changed_samples": int(support_changed.sum()),
        "parent_modified_samples": int(local_changed.sum()),
        "estimated_samples": int(estimated.sum()),
        "eligible_samples": int(eligible.sum()),
        "measured_samples": measures["measured_samples"],
        "unusable_samples": int((~valid).sum()),
        "events": len(measures["events"]),
        "intervals": len(measures["intervals"]),
        "pressure_cycles": len(measures["pressures"]),
        "remaining_issue_count": len(diagnostics["issues"]),
        "mean_shift": shift,
        "max_abs_change": float(np.max(np.abs(values[common] - raw[common]))) if common.any() else None,
    }
    result = {
        "artifact_id": aid,
        "case_id": case["id"],
        "task": case["task"],
        "arrays": arrays,
        "arrays_sha256": sha(workspace / arrays) if arrays else None,
        "source_sha256": case["input_sha256"],
        "content_hash": content,
        "change_focus_sample": focus,
        "support_changed_ranges": intervals(support_changed),
        "probe_path": str((cache / "probe.json").relative_to(workspace)),
        "probe_sha256": sha(cache / "probe.json"),
        "diagnosis": diagnostics,
        "stats": stats,
        "operations": operations,
        "recipe_hash": recipe["hash"],
        "implementation_hash": implementation,
        "parent_artifact_id": parent["artifact_id"] if parent else None,
        "program_checks": {"passed": True, "time_and_units_preserved": True, "fresh_probe": True},
        "reference_metrics": no_reference(),
        "elapsed_s": time.monotonic() - started,
    }
    import scipy

    result["evaluation_environment"] = {"numpy": np.__version__, "scipy": scipy.__version__}
    write(folder / "artifact.json", result)
    return result


def process_record(job):
    results, errors = {}, []
    workspace, cases, recipe, implementation, parents = job
    check_stop(workspace)
    for case in cases:
        check_stop(workspace)
        try:
            results[case["id"]] = process_one(workspace, case, recipe, implementation, parents.get(case["id"]))
        except StopRequested:
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            errors.append(
                {"case_id": case["id"], "task": case["task"], "reason": str(exc), "status": "execution_failed"}
            )
    return results, errors


def aggregate(cases, results, errors):
    tasks = {}
    for task in TASKS:
        scope = [c for c in cases if c["task"] == task]
        selected = [results[c["id"]] for c in scope if c["id"] in results]
        if not scope:
            continue
        totals = {
            key: sum(a["stats"][key] for a in selected)
            for key in (
                "n",
                "duration_s",
                "modified_samples",
                "support_changed_samples",
                "estimated_samples",
                "eligible_samples",
                "measured_samples",
                "unusable_samples",
                "events",
                "intervals",
                "pressure_cycles",
                "remaining_issue_count",
            )
        }
        total_n = sum(c["n"] for c in scope)
        tasks[task] = {
            "scope_cases": len(scope),
            "executed_cases": len(selected),
            "totals": totals,
            "rule_eligible_fraction": totals["eligible_samples"] / total_n if total_n else None,
            "reference_metrics": no_reference(),
        }
    return {
        "tasks": tasks,
        "errors": errors,
        "program_checks_passed": not errors and len(results) == len(cases),
        "status": "executed" if not errors and len(results) == len(cases) else "partial",
        "quality_claim": "No independent real-waveform accuracy measured",
    }


def register_recipe(workspace, kind=None, source=None, category="standard_operators"):
    workspace = Path(workspace)
    if category not in {"standard_operators", "custom_extension"}:
        raise ValueError("Unknown recipe attribution category")
    if source is not None:
        info = {"kind": "python", "category": category, "source_sha256": digest(source), "entrypoint": "run"}
    else:
        if kind not in {"identity", "fixed_dsp", "rule_adaptive_dsp"}:
            raise ValueError("Unknown registered recipe")
        info = {"kind": kind, "category": "baseline", "source_sha256": implementation_hash()}
    info["hash"] = digest(info)
    folder = workspace / "recipes" / info["hash"]
    folder.mkdir(parents=True, exist_ok=True)
    if source is not None:
        (folder / "program.py").write_text(source)
    write(folder / "recipe.json", info)
    emit(workspace, "recipe_submitted", recipe_hash=info["hash"], category=category)
    if source is not None:
        try:
            compile(source, "agent_recipe.py", "exec")
        except SyntaxError as exc:
            emit(workspace, "recipe_compile_failed", recipe_hash=info["hash"], error=str(exc))
            raise
    emit(workspace, "recipe_registered", recipe_hash=info["hash"])
    return info


def execute_version(workspace, recipe, cases, *, stage="baseline", parent=None, workers=2):
    workspace = Path(workspace)
    check_stop(workspace)
    implementation = implementation_hash()
    vid = "v_" + digest([recipe, [recipe_input_key(c) for c in cases], stage, parent, implementation])[:20]
    target = workspace / "versions" / vid
    for c in cases:
        if sha(workspace / "public" / c["id"] / "input.npz") != c["input_sha256"]:
            raise ValueError("Input content changed")
    if (target / "version.json").exists():
        old = read(target / "version.json")
        for a in old["results"].values():
            if a["arrays"] and sha(workspace / a["arrays"]) != a["arrays_sha256"]:
                raise ValueError("Cached waveform artifact was modified")
            if sha(workspace / a["probe_path"]) != a["probe_sha256"]:
                raise ValueError("Cached probe was modified")
        emit(workspace, "version_cache_hit", version_id=vid, recipe_hash=recipe["hash"], stage=stage)
        return old
    parents = read(workspace / "versions" / parent / "version.json")["results"] if parent else {}
    started = time.monotonic()
    emit(
        workspace,
        "version_started",
        version_id=vid,
        recipe_hash=recipe["hash"],
        stage=stage,
        parent_version=parent,
        case_count=len(cases),
    )
    if recipe["kind"] == "python":
        from .sandbox import run_recipe

        results, errors = run_recipe(workspace, cases, recipe, implementation, parents)
    else:
        groups = {}
        for c in cases:
            groups.setdefault(c["record_id"], []).append(c)
        jobs = [(str(workspace), group, recipe, implementation, parents) for group in groups.values()]
        results, errors = {}, []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part, failed in pool.map(process_record, jobs):
                results.update(part)
                errors.extend(failed)
    result = {
        "version_id": vid,
        "parent_version": parent,
        "recipe": recipe,
        "stage": stage,
        "case_ids": [c["id"] for c in cases],
        "results": results,
        "implementation_hash": implementation,
        "report": aggregate(cases, results, errors),
        "elapsed_s": time.monotonic() - started,
        "self_review": "not_applicable" if stage == "baseline" else None,
    }
    write(target / "version.json", result)
    emit(
        workspace,
        "version_completed",
        version_id=vid,
        recipe_hash=recipe["hash"],
        stage=stage,
        completed_cases=len(results),
        errors=len(errors),
        elapsed_s=result["elapsed_s"],
    )
    return result
