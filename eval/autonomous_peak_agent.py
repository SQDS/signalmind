"""A real dataset agent loop: the model requests batch, audit, inspection, refinement and delivery."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dataset_tools import tool  # noqa: E402
from eval.dataset_workflow import read, sha, write  # noqa: E402
from eval.run_peak_review import GUIDE, ReviewTools  # noqa: E402
from eval.unified.drivers import cost_of, make_driver  # noqa: E402
from eval.unified.loop import Limits, _redact_images  # noqa: E402
from tsmcp.peak_policy import apply, observe, validate  # noqa: E402
from tsmcp.peak_refinement import RefinementConfig, align_candidates, refine  # noqa: E402

SYSTEM = """You are the operational SignalMind dataset agent. Own this whole task: choose and run a batch
processing policy, audit the actual batch output, inspect representative problem and ordinary examples,
call a refinement program with your chosen scope/parameters, audit and inspect its actual output, then
select and deliver the version you judge most useful. The host will not perform these steps for you.
Use the tools to do the work, not just describe a plan. Read dataset_overview first.

This task concerns ECG R-marker review across the supplied dataset. Original R markers are fallible;
preserve original waveform samples. ECG physical units are unknown source_units; ABP is mmHg. There is no
verified cross-channel alignment. Different/negative QRS morphology is not proof of a false beat.
Programs and audit indicators are fallible. They do not expose private review labels or accuracy scores.
Inspect traces when indicators disagree. A lower diagnostic count does not guarantee better physiological
results, and a review queue is not automatically a confirmed exclusion list.

You can call the existing peak_refinement_v1 program; you do not have to write new Python. Its description
and parameters are in the catalog. You decide when to call it, its scope and parameters, and whether to
accept the result or retain an audited parent. Use aggregate reports and representative inspections;
there is no need for an LLM session per window. Finish only after dataset_finalize succeeds.
Your tool responses include remaining calls. Reserve calls for a post-refinement audit, inspection and
finalization. Do not claim to have performed a stage without its successful tool result.
"""


def timestamp():
    return datetime.now(UTC).isoformat()


class AutonomousPeakTools:
    def __init__(self, public, output, max_calls=40):
        self.public, self.output = Path(public).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.index = read(self.public / "index.json")
        self.ids = [c["id"] for c in self.index["cases"]]
        self.viewer = ReviewTools(self.public, self.output / "views")
        self.viewer.visible = set(self.ids)  # Entire public dataset, without any reference/identity files.
        self.versions, self.audits, self.inspections, self.evidence_cache = {}, {}, [], {}
        self.calls, self.max_calls, self.completed, self.delivery = 0, max_calls, False, None
        self.schemas = [
            tool(
                "dataset_overview",
                "Read dataset metadata, available batch policy/program contracts, and unlabelled representative examples.",
                {},
            ),
            tool(
                "dataset_run_batch",
                "Run YOUR supplied policy on the entire public dataset locally and create an immutable batch version. Next request dataset_audit.",
                {"policy": {"type": "object"}, "rationale": {"type": "string"}},
                ("policy", "rationale"),
            ),
            tool(
                "dataset_audit",
                "Audit a real saved version, returning aggregate changes, diagnostic conflicts, review groups and representative case ids. No private labels or accuracy scores.",
                {"version_id": {"type": "string"}},
                ("version_id",),
            ),
            tool(
                "dataset_inspect",
                "Inspect raw ECG/ABP and this version's R markers; for a refinement also report parent/current marker changes. Any public case id is allowed.",
                {
                    "version_id": {"type": "string"},
                    "case_id": {"type": "string"},
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                },
                ("version_id", "case_id"),
            ),
            tool(
                "dataset_run_refinement",
                "Call an existing local program on your selected group from an audited parent. Creates a new version; does not change the parent. You choose scope and parameters.",
                {
                    "parent_version": {"type": "string"},
                    "program": {"type": "string", "enum": ["peak_refinement_v1"]},
                    "scope": {
                        "type": "string",
                        "enum": ["all", "conflicting_deletions", "high_deletion", "review_queue", "nearby_candidates"],
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "support_energy_fraction": {"type": "number", "minimum": 0},
                            "detector_distance_ms": {"type": "number", "minimum": 0},
                            "context_ms": {"type": "number", "minimum": 0},
                        },
                    },
                    "rationale": {"type": "string"},
                },
                ("parent_version", "program", "scope", "parameters", "rationale"),
            ),
            tool(
                "dataset_finalize",
                "Select an audited version after you have audited and inspected a refinement result. Explain acceptance/rollback and remaining limitations. This is the terminal delivery action.",
                {
                    "version_id": {"type": "string"},
                    "rationale": {"type": "string"},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                },
                ("version_id", "rationale", "limitations"),
            ),
        ]

    def load(self, uid):
        return self.viewer.load(uid)

    def aligned_evidence(self, uid, policy):
        mode, radius = policy.get("alignment", "none"), policy.get("radius_ms", 80)
        key = (uid, mode, radius)
        if key not in self.evidence_cache:
            data = self.load(uid)
            positions = align_candidates(data["ecg"], data["old_peaks"], mode, round(radius * 125 / 1000))
            self.evidence_cache[key] = observe(data["ecg"], positions)
        return self.evidence_cache[key]

    def save_version(self, kind, predictions, policy, parent=None, execution=None):
        vid = f"v{len(self.versions) + 1:03d}"
        folder = self.output / "versions" / vid
        folder.mkdir(parents=True)
        write(folder / "predictions.json", predictions)
        metadata = {
            "version_id": vid,
            "kind": kind,
            "parent": parent,
            "policy": policy,
            "created_at": timestamp(),
            "execution": execution,
            "predictions_sha256": sha(folder / "predictions.json"),
        }
        write(folder / "version.json", metadata)
        self.versions[vid] = {**metadata, "predictions": predictions}
        self.viewer.trials[vid] = {"policy": policy, "results": predictions}
        response = {k: v for k, v in metadata.items() if k != "policy"}
        response["execution"] = {k: v for k, v in execution.items() if k != "case_ids"}
        return response

    def overview(self):
        sample_ids = self.index.get("development_examples", self.ids[:12])
        guide = dict(GUIDE)
        guide["max_detector_distance_ms"] = guide["max_detector_distance_ms"].replace("independent", "alternative")
        return {
            "dataset_windows": len(self.ids),
            "fs_hz": 125,
            "units": self.index["units"],
            "batch_policy_contract": guide,
            "program_catalog": {
                "peak_refinement_v1": {
                    "callable": "tsmcp.peak_refinement.refine",
                    "parameters": asdict(RefinementConfig()),
                    "description": "Revisit saved marker edits using original signal. Align candidates before checking QRS-band energy and detector distance; weak energy AND large detector distance support deletion. Negative template correlation alone does not delete. Restores supported/uncertain candidates, records changes and a review queue. Can restore unwanted extra candidates too; compare and inspect. Preserves parent window-exclusion decisions.",
                }
            },
            "examples": [{"id": uid, **self.load(uid)["observed"]["features"]} for uid in sample_ids],
            "scope": "Public observations only. Dataset distribution and all records are available, reference annotations are not.",
        }

    def batch(self, policy, rationale):
        validate(policy)
        started = time.monotonic()
        predictions = {}
        for uid in self.ids:
            data = self.load(uid)
            predictions[uid] = apply(data["ecg"], data["old_peaks"], policy, observed=data["observed"])
        return self.save_version(
            "batch",
            predictions,
            policy,
            execution={
                "cases_executed": len(self.ids),
                "elapsed_s": time.monotonic() - started,
                "rationale": rationale,
                "numerical_model_calls": 0,
            },
        )

    def audit(self, vid):
        version = self.versions[vid]
        rows, counts = [], Counter()
        groups = {name: [] for name in ("conflicting_deletions", "high_deletion", "review_queue", "nearby_candidates")}
        for uid, prediction in version["predictions"].items():
            data = self.load(uid)
            new, old = np.asarray(prediction["new_peak_samples"]), data["old_peaks"]
            aligned = self.aligned_evidence(uid, version["policy"])
            support = (aligned["energy_fraction"] >= 0.5) & (aligned["detector_distance_ms"] <= 120)
            conflicts = int(((new < 0) & support).sum())
            kept = np.sort(new[new >= 0])
            nearby = int(np.sum(np.diff(kept) / 125 < 0.3))
            deleted = int((new < 0).sum())
            pending = len(prediction.get("review_queue", []))
            row = {
                "case_id": uid,
                "old_markers": len(old),
                "deleted": deleted,
                "moved": int(((new >= 0) & (new != old)).sum()),
                "conflicting_deletions": conflicts,
                "nearby_pairs_under300ms": nearby,
                "review_items": pending,
                "excluded": prediction["exclude"],
            }
            if version["parent"]:
                parent = np.asarray(self.versions[version["parent"]]["predictions"][uid]["new_peak_samples"])
                row.update(
                    restored_from_parent=int(((parent < 0) & (new >= 0)).sum()),
                    newly_deleted_from_parent=int(((parent >= 0) & (new < 0)).sum()),
                )
            rows.append(row)
            for key in (
                "old_markers",
                "deleted",
                "moved",
                "conflicting_deletions",
                "nearby_pairs_under300ms",
                "review_items",
            ):
                counts[key] += row[key]
            counts["excluded_windows"] += prediction["exclude"]
            if conflicts:
                groups["conflicting_deletions"].append(uid)
            if deleted > 0.1 * max(len(old), 1):
                groups["high_deletion"].append(uid)
            if pending:
                groups["review_queue"].append(uid)
            if nearby:
                groups["nearby_candidates"].append(uid)
        ranked = sorted(rows, key=lambda r: (r["conflicting_deletions"], r["review_items"], r["deleted"]), reverse=True)
        ordinary = sorted(rows, key=lambda r: hashlib.sha256(r["case_id"].encode()).hexdigest())[:3]
        report = {
            "version_id": vid,
            "parent_version": version["parent"],
            "windows": len(rows),
            "totals": dict(counts),
            "groups": {k: {"count": len(v), "examples": v[:5]} for k, v in groups.items()},
            "priority_examples": ranked[:6],
            "random_comparison_examples": ordinary,
            "meaning": "Observable consistency checks, NOT true error counts. The QRS detector and energy features are related. Fewer conflicts or review flags need not mean more correct peaks.",
        }
        if version["parent"] in self.audits:
            previous = self.audits[version["parent"]]["report"]["totals"]
            report["change_from_parent"] = {key: counts[key] - previous.get(key, 0) for key in counts}
        write(self.output / "versions" / vid / "audit.json", {**report, "all_cases": rows})
        self.audits[vid] = {"report": report, "groups": groups}
        return report

    def inspect(self, version_id, case_id, start_s=0, end_s=10):
        version = self.versions[version_id]
        blocks = self.viewer.inspect(case_id, start_s, end_s, version_id)
        data = self.load(case_id)
        current = version["predictions"][case_id]
        parent = self.versions[version["parent"]]["predictions"][case_id] if version["parent"] else None
        selected = np.flatnonzero((data["old_peaks"] >= start_s * 125) & (data["old_peaks"] < end_s * 125))
        comparison = [
            {
                "R_id": f"R{i + 1:03d}",
                "original": int(data["old_peaks"][i]),
                "parent": parent["new_peak_samples"][i] if parent else int(data["old_peaks"][i]),
                "current": current["new_peak_samples"][i],
            }
            for i in selected
        ]
        blocks.insert(
            0,
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "version_id": version_id,
                        "case_id": case_id,
                        "parent_current_comparison": comparison,
                        "review_queue": current.get("review_queue", []),
                    }
                ),
            },
        )
        self.inspections.append({"version_id": version_id, "case_id": case_id, "start_s": start_s, "end_s": end_s})
        write(self.output / "inspections.json", self.inspections)
        return blocks

    def refinement(self, parent_version, program, scope, parameters, rationale):
        if parent_version not in self.audits:
            raise ValueError("Audit the parent output before selecting a refinement program")
        if program != "peak_refinement_v1":
            raise ValueError("Unknown refinement program")
        config = RefinementConfig(**parameters)
        if any(not np.isfinite(v) or v < 0 for v in asdict(config).values()):
            raise ValueError("Refinement parameters must be finite nonnegative numbers")
        selected = self.ids if scope == "all" else self.audits[parent_version]["groups"][scope]
        if not selected:
            raise ValueError("This audit group is empty; choose a nonempty scope")
        parent = self.versions[parent_version]
        predictions = dict(parent["predictions"])
        started = time.monotonic()
        for uid in selected:
            data = self.load(uid)
            predictions[uid] = refine(
                data["ecg"], data["old_peaks"], parent["predictions"][uid], parent["policy"], config=config
            )
        return self.save_version(
            "refinement",
            predictions,
            parent["policy"],
            parent_version,
            {
                "program": program,
                "callable": "tsmcp.peak_refinement.refine",
                "parameters": asdict(config),
                "scope": scope,
                "case_ids": selected,
                "cases_executed": len(selected),
                "elapsed_s": time.monotonic() - started,
                "rationale": rationale,
                "numerical_model_calls": 0,
            },
        )

    def finalize(self, version_id, rationale, limitations):
        if version_id not in self.audits:
            raise ValueError("Audit the selected version before delivery")
        root = version_id
        while self.versions[root]["parent"]:
            root = self.versions[root]["parent"]
        candidates = []
        for vid, version in self.versions.items():
            ancestor = vid
            while self.versions[ancestor]["parent"]:
                ancestor = self.versions[ancestor]["parent"]
            if (
                ancestor == root
                and version["kind"] == "refinement"
                and vid in self.audits
                and any(i["version_id"] == vid for i in self.inspections)
            ):
                candidates.append(vid)
        if not candidates:
            raise ValueError(
                "Run a refinement, then audit and inspect its output before finalizing; you may explain a rollback"
            )
        selected = self.versions[version_id]
        write(self.output / "final_predictions.json", selected["predictions"])
        self.delivery = {
            "status": "completed",
            "selected_version": version_id,
            "initial_batch_version": root,
            "reviewed_refinement_versions": candidates,
            "selected_refined_result": selected["kind"] == "refinement",
            "rationale": rationale,
            "limitations": limitations,
            "decision_owner": "model",
            "final_predictions_sha256": sha(self.output / "final_predictions.json"),
            "private_labels_accessed": False,
            "tool_calls": self.calls,
        }
        write(self.output / "delivery.json", self.delivery)
        self.completed = True
        return self.delivery

    async def call(self, name, args):
        self.calls += 1
        if self.completed:
            return [{"type": "text", "text": "Dataset task is already delivered"}], True
        try:
            if name == "dataset_overview":
                result = self.overview()
            elif name == "dataset_run_batch":
                result = self.batch(**args)
            elif name == "dataset_audit":
                result = self.audit(args["version_id"])
            elif name == "dataset_inspect":
                blocks = self.inspect(**args)
                blocks.append({"type": "text", "text": f"Remaining tool calls: {self.max_calls - self.calls}"})
                return blocks, False
            elif name == "dataset_run_refinement":
                result = self.refinement(**args)
            elif name == "dataset_finalize":
                result = self.finalize(**args)
            else:
                raise ValueError("Unknown dataset tool")
            response = {**result, "remaining_tool_calls": self.max_calls - self.calls}
            return [{"type": "text", "text": json.dumps(response, ensure_ascii=False, allow_nan=False)}], False
        except (ValueError, TypeError, KeyError) as exc:
            return [{"type": "text", "text": str(exc)}], True


async def run_agent(
    driver,
    tools,
    goal,
    out,
    cap=0.50,
    limits=Limits(max_turns=40, max_tool_calls=40, max_seconds=1200, max_output_tokens_total=50000),
):
    """The controller only routes actual model tool calls; it never performs hidden repair steps."""
    out = Path(out)
    (out / "private").mkdir(exist_ok=True, mode=0o700)
    messages, usage, activity = [{"role": "user", "content": goal}], Counter(), []
    started = time.monotonic()
    stop, error, turns, continuations, model_names = "max_turns", None, 0, 0, set()
    with (out / "private/transcript.jsonl").open("w") as transcript, (out / "activity.jsonl").open("w") as public_log:

        def emit(event):
            transcript.write(json.dumps(event, ensure_ascii=False) + "\n")
            transcript.flush()

        try:
            while turns < limits.max_turns:
                if cost_of("deepseek-flash", usage) + 0.05 >= cap:
                    stop = "estimated_cost_budget"
                    break
                if time.monotonic() - started > limits.max_seconds:
                    stop = "timeout"
                    break
                if tools.calls >= limits.max_tool_calls or usage["output_tokens"] >= limits.max_output_tokens_total:
                    stop = "tool_or_token_budget"
                    break
                response = await asyncio.to_thread(driver.step, SYSTEM, messages, tools.schemas)
                turns += 1
                model_names.add(response.model)
                usage.update(response.usage)
                emit(
                    {
                        "type": "assistant",
                        "request_id": response.id,
                        "model": response.model,
                        "content": _redact_images(response.content),
                        "usage": response.usage,
                    }
                )
                messages.append({"role": "assistant", "content": response.content})
                calls = [b for b in response.content if b.get("type") == "tool_use"]
                if not calls:
                    feedback = "The dataset task is not delivered. Continue with the missing batch/audit/inspect/refinement/recheck steps, then call dataset_finalize. No host-side repair will be substituted."
                    messages.append({"role": "user", "content": feedback})
                    continuations += 1
                    emit({"type": "controller_feedback", "content": feedback})
                    continue
                results = []
                for call in calls:
                    if tools.calls >= limits.max_tool_calls:
                        blocks, failed = [{"type": "text", "text": "Tool budget exhausted"}], True
                    else:
                        blocks, failed = await tools.call(call["name"], call.get("input") or {})
                    event = {
                        "sequence": len(activity) + 1,
                        "at": timestamp(),
                        "origin": "model_tool_call",
                        "model": response.model,
                        "request_id": response.id,
                        "tool_use_id": call["id"],
                        "tool": call["name"],
                        "arguments": call.get("input") or {},
                        "is_error": failed,
                        "result": _redact_images(blocks),
                    }
                    activity.append(event)
                    public_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    public_log.flush()
                    print(
                        json.dumps(
                            {
                                "tool": call["name"],
                                "sequence": len(activity),
                                "error": failed,
                                "versions": list(tools.versions),
                                "completed": tools.completed,
                            }
                        ),
                        flush=True,
                    )
                    results.append(
                        {"type": "tool_result", "tool_use_id": call["id"], "content": blocks, "is_error": failed}
                    )
                    if tools.completed:
                        break
                emit({"type": "tool_results", "content": _redact_images(results)})
                messages.append({"role": "user", "content": results})
                if tools.completed:
                    stop = "workflow_complete"
                    break
        except Exception as exc:
            error, stop = f"{type(exc).__name__}: {str(exc)[:400]}", "error"
    result = {
        "stop": stop,
        "error": error,
        "workflow_complete": tools.completed,
        "turns": turns,
        "tool_calls": tools.calls,
        "controller_continuations": continuations,
        "served_models": sorted(m for m in model_names if m),
        "usage": dict(usage),
        "estimated_peak_usd": cost_of("deepseek-flash", usage),
        "actual_billed_usd": None,
        "latency_s": time.monotonic() - started,
        "host_injected_processing_steps": 0,
        "processing_trigger": "actual_model_tool_calls_only",
        "delivery": tools.delivery,
    }
    write(out / "session.json", result)
    return result


async def main_run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "private").mkdir(mode=0o700)
    sources = [
        "eval/autonomous_peak_agent.py",
        "src/tsmcp/peak_policy.py",
        "src/tsmcp/peak_refinement.py",
        "src/tsmcp/plugins/ecg/rpeaks.py",
        "src/tsmcp/core/dsp.py",
        "src/tsmcp/core/peaks.py",
        "src/tsmcp/processing.py",
        "eval/run_peak_review.py",
        "eval/dataset_tools.py",
        "eval/dataset_workflow.py",
        "eval/real_processing_helper.py",
        "eval/unified/drivers.py",
        "eval/unified/loop.py",
    ]
    provenance = {
        "date": "2026-11-06",
        "model": "deepseek-flash",
        "cap_usd": args.cap,
        "public_input_directory": str(args.data.resolve()),
        "public_index_sha256": sha(args.data / "index.json"),
        "source_sha256": {p: sha(ROOT / p) for p in sources},
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "goal": args.goal,
        "run_type": "Live model-controlled batch-audit-refine-recheck workflow",
        "private_reference_data_available_to_tools": False,
    }
    for p in sources:
        target = args.out / "private/source" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, target)
    write(args.out / "provenance.json", provenance)
    tools = AutonomousPeakTools(args.data, args.out)
    driver = make_driver("deepseek-flash", max_tokens=8192)
    try:
        return await run_agent(driver, tools, args.goal, args.out, args.cap)
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Only the public dataset directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--cap", type=float, default=0.50)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live or args.cap <= 0.05:
        raise ValueError("Live model execution needs --live and cap above the next-request reserve")
    result = asyncio.run(main_run(args))
    print(json.dumps({k: v for k, v in result.items() if k not in {"usage", "delivery"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
