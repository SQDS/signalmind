"""DS-controlled batch/check/code/refine loop with labelled development and isolated evaluation labels."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.autonomous_peak_agent import AutonomousPeakTools, run_agent  # noqa: E402
from eval.dataset_tools import tool  # noqa: E402
from eval.dataset_workflow import read, sha, write  # noqa: E402
from eval.grade_peak_review import grade_case, summarize  # noqa: E402
from eval.peak_program_worker import normalize  # noqa: E402
from eval.processing_tools import PYTHON, sandbox_command  # noqa: E402
from eval.unified.drivers import make_driver  # noqa: E402
from eval.unified.loop import Limits  # noqa: E402

SYSTEM = """You are the operational SignalMind coding agent. Own batch execution, checking DEVELOPMENT
reference feedback, writing a reusable refinement program, running/debugging it, rechecking, and delivery.
The host will not write the numerical algorithm or repair your results for you. Start with dataset_overview.
Run a batch, inspect development feedback and representative traces, create your program through
dataset_write_program, and test it. You may revise code by creating another immutable program version.
After choosing a program, run it on scope=all and examine the result before dataset_finalize.

There is a fixed patient-disjoint development/evaluation split. Development references are intentionally
available for calibration and fitting; evaluation annotations and scores are inaccessible until after delivery.
Do not encode case-id/patient-id exceptions or memorize exact waveforms. fit() may learn general parameters
from development examples only. Original samples, marker identities and time axes must remain intact.
ECG physical units are unconfirmed source_units; ABP is mmHg. Cross-channel timing is not verified.

This task matches the author's R-fiducial convention shown by DEVELOPMENT target coordinates. A large
absolute negative excursion can be an S trough and is not interchangeable with an R top. Negative QRS
morphology is not by itself a false beat. Test localization choices against the development references.
Keep/delete and coordinate refinement are separate decisions. Do not remove waveform T waves when dropping
mistaken R annotations. Excluded development windows have NO per-peak truth: do not label all their peaks false.
Window exclusion means unsuitable or unresolved for the complete clean-RR analysis window, not a diagnosis.

Primary development checks are complete retained-window passes, extra-marker deletion precision/recall,
wrong deletions, localization and balanced window decisions. Do not optimize a proxy count or overall
accuracy by discarding everything. Do not deliver an obvious development regression without inspecting it
and considering a better tested version. Unresolved review queues are not successful exclusions.

Use available Python numpy/scipy/tsmcp tools for a general program. sklearn is NOT installed. The program
runs in a read-only input, no-network sandbox with a120s wall limit and4GiB address-space limit. You have
48 tool calls; keep calls for feedback, full execution, audit/inspection and finalization.
"""

PROGRAM_CONTRACT = {
    "fit": "Optional fit(examples, fs) -> arbitrary in-memory state, called once per execution. Each example has ecg, abp, old_peaks arrays, anonymous record_id and reference dict. reference.target_peak_samples is one position per original marker or -1; None for excluded windows. Do not learn per-peak labels from excluded windows.",
    "process": "Required process(ecg, abp, old_peaks, fs, previous, state) -> dict with new_peak_samples (same length as old_peaks; integer coordinates or -1) and exclude(bool). Optional review_queue(list), review_required(bool), exclude_reasons(list). Retained coordinates must be unique and in range. Original old_R identity is the array index, not sorted output ordinal.",
    "libraries": ["numpy", "scipy", "tsmcp"],
    "deployment": "When changing a successful program from development to all, use the same parent version as its development test unless you deliberately intend and validate a repeated refinement.",
    "not_installed": ["sklearn", "torch"],
    "constraints": "No waveform outputs or new/missing-beat additions. Program may analyze temporary filtered copies. No case ids are passed to process. Seed stochastic methods explicitly. Return numeric predictions, not prose.",
    "empty_scaffold": "def fit(examples, fs):\n    return None\n\ndef process(ecg, abp, old_peaks, fs, previous, state):\n    return {'new_peak_samples': previous['new_peak_samples'], 'exclude': previous['exclude']}\n",
}


def prepare(reference, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "private").mkdir(mode=0o700)
    index, truth = read(reference / "public/index.json"), read(reference / "private/truth.json")
    profiles = defaultdict(lambda: {"move": False, "delete": False, "retain": False})
    for row in truth:
        profile = profiles[row["subject_key"]]
        if row["target_peak_samples"] is None:
            continue
        with np.load(reference / "public" / row["id"] / "input.npz", allow_pickle=False) as data:
            old = data["old_peaks"]
        target = np.asarray(row["target_peak_samples"])
        profile["retain"] = True
        profile["move"] |= bool(((target >= 0) & (target != old)).any())
        profile["delete"] |= bool((target < 0).any())
    strata = defaultdict(list)
    for subject, p in profiles.items():
        strata[("M" if p["move"] else "") + ("D" if p["delete"] else "") + ("R" if p["retain"] else "X")].append(
            subject
        )
    development, split_counts = set(), {}
    for key, subjects in strata.items():
        ordered = sorted(
            subjects, key=lambda s: hashlib.sha256(("vital-coding-split-20261107:" + s).encode()).hexdigest()
        )
        n = min(len(ordered) - 1, max(1, round(len(ordered) / 3))) if len(ordered) > 1 else 1
        development.update(ordered[:n])
        split_counts[key] = {"subjects": len(ordered), "development": n, "evaluation": len(ordered) - n}
    public = out / "public"
    shutil.copytree(reference / "public", public)
    public_labels, mapping = {}, []
    for row in truth:
        role = "development" if row["subject_key"] in development else "evaluation"
        mapping.append({**row, "role": role})
        if role == "development":
            public_labels[row["id"]] = {
                k: v for k, v in row.items() if k in {"id", "disposition", "exclusion_subtype", "target_peak_samples"}
            }
            public_labels[row["id"]]["subject_key"] = (
                "record_" + hashlib.sha256(row["subject_key"].encode()).hexdigest()[:12]
            )
    # A small representative list for the overview, with edit-type coverage in development only.
    representatives = []
    for _key, subjects in sorted(strata.items()):
        selected = [r for r in mapping if r["subject_key"] in set(subjects) & development]
        representatives.extend(r["id"] for r in sorted(selected, key=lambda r: r["id"])[:2])
    index["development_examples"] = list(dict.fromkeys(representatives))
    index["development_case_ids"] = sorted(public_labels)
    index["evaluation_case_ids"] = sorted(r["id"] for r in mapping if r["role"] == "evaluation")
    write(public / "index.json", index)
    write(public / "development_labels.json", public_labels)
    write(out / "private/truth.json", mapping)
    summary = {
        "date": "2026-11-07",
        "seed": "vital-coding-split-20261107",
        "strata": split_counts,
        "development_subjects": len(development),
        "evaluation_subjects": len(profiles) - len(development),
        "development_windows": len(public_labels),
        "evaluation_windows": len(mapping) - len(public_labels),
        "patient_overlap": 0,
        "evaluation_labels_in_public": False,
        "prior_annotations_seen_by_developer": True,
        "split_selected_using_prior_model_scores": False,
    }
    write(out / "split.json", summary)
    return summary


class CodingPeakTools(AutonomousPeakTools):
    def __init__(self, public, output, max_calls=48):
        super().__init__(public, output, max_calls)
        self.development = read(self.public / "development_labels.json")
        self.programs, self.feedbacks, self.program_attempts = {}, {}, 0
        self.schemas += [
            tool(
                "dataset_development_feedback",
                "Score ONLY the labelled development patients, with per-error examples. Evaluation answers are never read. Request this after a candidate version, before choosing a program/version.",
                {"version_id": {"type": "string"}},
                ("version_id",),
            ),
            tool(
                "dataset_write_program",
                "Save YOUR Python source as an immutable refinement program. Code is not executed on the host. See fit/process contract in dataset_overview.",
                {"source": {"type": "string"}, "description": {"type": "string"}},
                ("source", "description"),
            ),
            tool(
                "dataset_run_program",
                "Execute your program in isolation on development or all windows. fit sees development references only. Merge valid results into a new refinement version; report code errors instead of repairing them on the host.",
                {
                    "parent_version": {"type": "string"},
                    "program_id": {"type": "string"},
                    "scope": {"type": "string", "enum": ["development", "all"]},
                    "rationale": {"type": "string"},
                },
                ("parent_version", "program_id", "scope", "rationale"),
            ),
        ]

    def overview(self):
        result = super().overview()
        result["development_references"] = {
            "windows": len(self.development),
            "subjects": len({r["subject_key"] for r in self.development.values()}),
            "public": True,
            "evaluation_references": "unavailable",
        }
        result["programming_contract"] = PROGRAM_CONTRACT
        result["task_convention"] = (
            "Use the supplied development reference coordinates to establish the R fiducial. Absolute-amplitude extrema can choose S troughs. Never train per-peak deletion labels on excluded windows, whose targets are None. Match window-level conservative clean-RR inclusion separately."
        )
        return result

    def feedback(self, vid):
        version = self.versions[vid]
        rows, errors = [], []
        for uid, target in self.development.items():
            data, prediction = self.load(uid), version["predictions"][uid]
            counts = grade_case(target, data["old_peaks"], prediction)
            rows.append({**target, "counts": counts})
            if not counts["window_agreement"] or (
                target["target_peak_samples"] is not None and not counts["retained_windows_end_to_end_8ms"]
            ):
                details = []
                if target["target_peak_samples"] is not None:
                    expected, actual = (
                        np.asarray(target["target_peak_samples"]),
                        np.asarray(prediction["new_peak_samples"]),
                    )
                    bad = ((expected < 0) != (actual < 0)) | (
                        (expected >= 0) & (actual >= 0) & (np.abs(actual - expected) > 1)
                    )
                    for i in np.flatnonzero(bad)[:6]:
                        details.append(
                            {
                                "R_id": f"R{i + 1:03d}",
                                "original": int(data["old_peaks"][i]),
                                "predicted": int(actual[i]),
                                "reference": int(expected[i]),
                            }
                        )
                errors.append(
                    {
                        "case_id": uid,
                        "reference_action": target["disposition"],
                        "predicted_exclude": prediction["exclude"],
                        "wrong_deletions": counts.get("delete_fp", 0),
                        "missed_deletions": counts.get("delete_fn", 0),
                        "peak_examples": details,
                    }
                )
        result = {
            "version_id": vid,
            "scope": "development_only",
            "metrics": summarize(rows),
            "error_examples": sorted(
                errors,
                key=lambda r: (r["wrong_deletions"], r["missed_deletions"], len(r["peak_examples"])),
                reverse=True,
            )[:10],
            "evaluation_labels_or_scores": "unavailable",
        }
        self.feedbacks[vid] = result
        write(self.output / "versions" / vid / "development_feedback.json", result)
        return result

    def inspect(self, version_id, case_id, start_s=0, end_s=10):
        blocks = super().inspect(version_id, case_id, start_s, end_s)
        if case_id in self.development:
            target = self.development[case_id]
            points = self.load(case_id)["old_peaks"]
            selected = np.flatnonzero((points >= start_s * 125) & (points < end_s * 125))
            supplied = {
                "partition": "labelled_development",
                "reference_action": target["disposition"],
                "target_markers": [
                    {"R_id": f"R{i + 1:03d}", "sample0": target["target_peak_samples"][i]} for i in selected
                ]
                if target["target_peak_samples"] is not None
                else None,
            }
        else:
            supplied = {"partition": "evaluation", "reference": "not available"}
        blocks.insert(0, {"type": "text", "text": json.dumps(supplied)})
        return blocks

    def write_program(self, source, description):
        ast.parse(source)
        pid = "p_" + hashlib.sha256(source.encode()).hexdigest()[:16]
        folder = self.output / "programs" / pid
        if not folder.exists():
            folder.mkdir(parents=True)
            (folder / "program.py").write_text(source)
            write(
                folder / "metadata.json",
                {
                    "program_id": pid,
                    "description": description,
                    "source_sha256": sha(folder / "program.py"),
                    "source_origin": "dataset_write_program_argument",
                },
            )
        self.programs[pid] = folder
        return {"program_id": pid, "source_sha256": sha(folder / "program.py"), "status": "saved_not_executed"}

    def run_program(self, parent_version, program_id, scope, rationale):
        if parent_version not in self.audits or parent_version not in self.feedbacks:
            raise ValueError("Audit the parent and request its development feedback before running a new program")
        if scope not in {"all", "development"}:
            raise ValueError("scope must be development or all")
        self.program_attempts += 1
        folder = self.output / "program_runs" / f"attempt_{self.program_attempts:03d}"
        folder.mkdir(parents=True)
        parent = self.versions[parent_version]
        chosen = self.ids if scope == "all" else sorted(self.development)
        shutil.copyfile(self.programs[program_id] / "program.py", folder / "program.py")
        write(
            folder / "job.json",
            {"fs_hz": 125, "case_ids": chosen, "previous": {uid: parent["predictions"][uid] for uid in chosen}},
        )
        command = sandbox_command(self.public, folder, agent=True)
        command = command[:-1] + [
            "--ro-bind",
            str(ROOT / "eval/peak_program_worker.py"),
            "/opt/program_worker.py",
            "--ro-bind",
            str(self.programs[program_id] / "program.py"),
            "/output/program.py",
            "--",
            str(PYTHON),
            "-I",
            "/opt/program_worker.py",
        ]
        started = time.monotonic()
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            write(folder / "execution.json", {"status": "timeout", "program_id": program_id, "scope": scope})
            return {
                "status": "timeout",
                "program_id": program_id,
                "message": "120s wall limit exceeded; revise your code",
            }
        result = (
            read(folder / "result.json") if (folder / "result.json").exists() else {"predictions": {}, "errors": []}
        )
        execution = {
            "program": "model_generated_python",
            "program_id": program_id,
            "program_sha256": sha(folder / "program.py"),
            "scope": scope,
            "case_ids": chosen,
            "cases_executed": len(result["predictions"]),
            "elapsed_s": time.monotonic() - started,
            "returncode": process.returncode,
            "rationale": rationale,
            "stdout": process.stdout[-8000:],
            "stderr": process.stderr[-8000:],
            "errors": result["errors"],
            "sandbox": "bubblewrap: read-only public data; no network/home/credentials; development labels only",
        }
        write(folder / "execution.json", execution)
        if process.returncode or result["errors"] or set(result["predictions"]) != set(chosen):
            return {
                "status": "program_failed",
                "program_id": program_id,
                "completed": len(result["predictions"]),
                "errors": result["errors"][:3],
                "stdout": execution["stdout"],
                "stderr": execution["stderr"],
            }
        validated = {}
        for uid, candidate in result["predictions"].items():
            data = self.load(uid)
            validated[uid] = normalize(candidate, parent["predictions"][uid], data["old_peaks"], len(data["ecg"]), 125)
        predictions = {**parent["predictions"], **validated}
        return self.save_version("refinement", predictions, parent["policy"], parent_version, execution)

    def finalize(self, version_id, rationale, limitations):
        if version_id not in self.feedbacks:
            raise ValueError("Read development feedback for the selected version before finalizing")
        custom_runs = [v for v in self.versions.values() if v["execution"].get("program") == "model_generated_python"]
        if not custom_runs:
            raise ValueError("Write and test a refinement program before delivery")
        selected = self.versions[version_id]
        if selected["execution"].get("program") == "model_generated_python" and selected["execution"]["scope"] != "all":
            raise ValueError("Run your selected program on scope=all before delivering it")
        return super().finalize(version_id, rationale, limitations)

    async def call(self, name, args):
        if name not in {"dataset_development_feedback", "dataset_write_program", "dataset_run_program"}:
            return await super().call(name, args)
        self.calls += 1
        if self.completed:
            return [{"type": "text", "text": "Dataset already delivered"}], True
        try:
            if name == "dataset_development_feedback":
                result = self.feedback(args["version_id"])
            elif name == "dataset_write_program":
                result = self.write_program(**args)
            else:
                result = self.run_program(**args)
            return [
                {
                    "type": "text",
                    "text": json.dumps(
                        {**result, "remaining_tool_calls": self.max_calls - self.calls},
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                }
            ], result.get("status") in {"program_failed", "timeout"}
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            return [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], True


class PromptedDriver:
    """Keep the existing controller unchanged while giving this task its explicit coding contract."""

    def __init__(self, driver):
        self.driver = driver

    def step(self, system, messages, tools):
        return self.driver.step(SYSTEM, messages, tools)


async def live(args):
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "private").mkdir(mode=0o700)
    sources = [
        "eval/coding_peak_agent.py",
        "eval/peak_program_worker.py",
        "eval/autonomous_peak_agent.py",
        "eval/processing_tools.py",
        "eval/grade_peak_review.py",
        "src/tsmcp/peak_policy.py",
        "src/tsmcp/peak_refinement.py",
        "eval/unified/drivers.py",
        "eval/unified/loop.py",
        "eval/run_peak_review.py",
        "eval/dataset_tools.py",
        "eval/dataset_workflow.py",
        "eval/real_processing_helper.py",
        "src/tsmcp/plugins/ecg/rpeaks.py",
        "src/tsmcp/core/peaks.py",
        "src/tsmcp/core/dsp.py",
        "src/tsmcp/processing.py",
        "docs/signalmind/coding-agent-protocol-20261107.md",
    ]
    provenance = {
        "date": "2026-11-07",
        "model": "deepseek-flash",
        "cap_usd": args.cap,
        "data": str(args.data.resolve()),
        "public_index_sha256": sha(args.data / "index.json"),
        "development_reference_sha256": sha(args.data / "development_labels.json"),
        "split_sha256": sha(args.data.parent / "split.json"),
        "program_environment": json.loads(
            subprocess.check_output(
                [
                    str(PYTHON),
                    "-c",
                    "import sys,json,numpy,scipy; print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__}))",
                ],
                text=True,
            )
        ),
        "source_sha256": {p: sha(ROOT / p) for p in sources},
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "evaluation_labels_accessible": False,
        "goal": args.goal,
    }
    for name in sources:
        path = args.out / "private/source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    write(args.out / "provenance.json", provenance)
    tools = CodingPeakTools(args.data, args.out)
    driver = make_driver("deepseek-flash", max_tokens=8192)
    try:
        result = await run_agent(
            PromptedDriver(driver),
            tools,
            args.goal,
            args.out,
            args.cap,
            Limits(max_turns=48, max_tool_calls=48, max_seconds=1500, max_output_tokens_total=60000),
        )
    finally:
        driver.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal", default="自主完成批量、开发反馈检查、编写并测试细修程序、全量执行和复查交付。")
    parser.add_argument("--cap", type=float, default=0.50)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.reference, args.out)
    else:
        if not args.live or args.cap <= 0.05:
            raise ValueError("Use --live with an explicit positive budget")
        result = asyncio.run(live(args))
    print(json.dumps({k: v for k, v in result.items() if k not in {"usage", "delivery"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
