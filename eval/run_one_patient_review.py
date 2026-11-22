"""Rerun a single complete record using candidate discovery, new visual reviews, batch cleaning and rescans."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path

import numpy as np

from tsmcp.cleaning.agent import BudgetClient, run, vision_preflight
from tsmcp.cleaning.cli import baselines
from tsmcp.cleaning.common import digest, read, sha, write
from tsmcp.cleaning.controller import CleaningTools
from tsmcp.followup.reviewer import review
from tsmcp.inspection.run import inspect, load_channel, version_id


def prepare(source, record, out):
    original = read(source / "dataset_manifest.json")
    records = [r for r in original["records"] if r["record_id"] == record]
    cases = [c for c in original["cases"] if c["record_id"] == record]
    if len(records) != 1 or not cases:
        raise ValueError("Record missing in the audited input")
    for task in ("ECG", "PPG", "ABP"):
        seen = [row for c in cases if c["task"] == task for row in c["rows"]]
        if sorted(seen) != list(range(records[0]["rows"])):
            raise ValueError("Incomplete source window inventory for " + task)
    for case in cases:
        origin = source / "public" / case["id"] / "input.npz"
        dest = out / "public" / case["id"] / "input.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            try:
                os.link(origin.resolve(), dest)
            except OSError:
                shutil.copyfile(origin, dest)
        if sha(dest) != case["input_sha256"]:
            raise ValueError("Source hash mismatch")
    m = {
        **original,
        "cases": cases,
        "records": records,
        "patients": 1,
        "scope_hash": digest([(c["id"], c["input_sha256"]) for c in cases]),
        "unavailable_tasks": [],
        "audit_failures": [],
    }
    if (out / "dataset_manifest.json").exists() and read(out / "dataset_manifest.json") != m:
        raise ValueError("Select a new output directory for a changed scope")
    write(out / "dataset_manifest.json", m)
    write(
        out / "data_audit.json",
        dict(
            record_id=record,
            source_windows=records[0]["rows"],
            source_manifest_sha256=sha(source / "dataset_manifest.json"),
            complete_rows_verified=True,
            inputs_verified=len(cases),
        ),
    )
    return m


def exemplars(out, previous):
    groups = {}
    ranks = {"ecg_excursion": 4, "large_step": 3, "high_frequency": 2, "line_50": 1, "line_60": 1}
    with (out / "inspection/evidence.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["kind"]
            if kind not in ranks:
                continue
            task = row["range"]["channel_id"]
            seed = previous["seed_samples"]
            same = previous["seed_case"] == row["range"]["block_id"] and [row["start"], row["end"]] == seed
            rank = (same, ranks[kind], row["value"] / row["threshold"])
            if task not in groups or rank > groups[task][0]:
                groups[task] = (rank, row)
    return {task: row for task, (_, row) in groups.items()}


def execute(source, record, out, known, budget_root):
    m = prepare(source, record, out)
    if not inspect(out, out / "inspection", "balanced")["inspection_complete"]:
        raise ValueError("Incomplete inspection")
    if not (out / "baselines.json").exists() or len(read(out / "baselines.json")) != 3:
        baselines(out)
    previous = read(known)
    if previous["rule_id"] != digest(
        [previous["task"], previous["rule"], sha(Path(__file__).resolve().parents[1] / "src/tsmcp/followup/pattern.py")]
    ):
        raise ValueError("Prior detector implementation differs")
    selections = exemplars(out, previous)
    write(
        out / "selection.json",
        {
            "policy": "one per signal: discussed PPG trigger when present, then excursion, step, spectrum, largest threshold ratio",
            "seeds": selections,
        },
    )
    cases = {c["id"]: c for c in m["cases"]}
    billing = budget_root / "api_budget.json"
    with (budget_root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = read(billing)
        own = out / "api_budget.json"
        if own.exists():
            local = read(own)
            n = min(len(parent["calls"]), len(local["calls"]))
            if parent["calls"][:n] != local["calls"][:n]:
                raise ValueError("Cost ledger history diverged")
            if len(local["calls"]) < len(parent["calls"]):
                write(own, parent)
        else:
            write(own, parent)
            write(
                out / "initial_budget.json",
                {"estimated_peak_usd": parent["estimated_peak_usd"], "calls": len(parent["calls"])},
            )
        (out / "private").mkdir(mode=0o700, exist_ok=True)
        try:
            client = BudgetClient(out, cap=parent["cap_usd"])
            brief = []
            saved = []
            try:
                vision_preflight(client)
                for task, e in selections.items():
                    c = cases[e["range"]["block_id"]]
                    folder = out / "visual-reviews" / task
                    dest = folder / "decision.json"
                    if not dest.exists():
                        x, masks = load_channel(out / "public" / c["id"] / "input.npz", c)
                        decision = review(
                            client,
                            folder,
                            c,
                            x,
                            ~np.logical_or.reduce(list(masks.values())),
                            version_id(c),
                            [e["start"], e["end"]],
                            5,
                        )
                        decision.update(
                            source_candidate_id=e["candidate_id"],
                            seed_case=c["id"],
                            seed_samples=[e["start"], e["end"]],
                        )
                        write(dest, decision)
                    decision = read(dest)
                    if decision["status"] == "registered":
                        saved.append(decision)
                    brief.append(
                        {
                            k: decision[k]
                            for k in (
                                "status",
                                "seed_case",
                                "seed_samples",
                                "observed",
                                "suspected_range_s",
                                "suspected_cause",
                                "alternative_or_limitation",
                                "method_explanation",
                                "rule",
                                "reason",
                            )
                            if k in decision
                        }
                    )
                    print("visual review", task, decision["status"], flush=True)
            finally:
                client.close()
            ctl = CleaningTools(out)
            if not ctl.state.get("finalized"):
                rules = ctl.state.setdefault("registered_patterns", {})
                for d in [previous, *saved]:
                    if d["seed_case"] in cases:
                        rules[d["rule_id"]] = {
                            **d,
                            "origin": "persisted_model_decision" if d is previous else "new_visual_reviewer",
                        }
                ctl.save()
                msg = out / "private/messages.json"
                if not msg.exists():
                    write(
                        msg,
                        [
                            {
                                "role": "user",
                                "content": (
                                    "对这一名患者的 ECG、PPG、ABP 全范围重新执行 SignalMind 试跑、批处理、实际看图和复查。"
                                    "本次已有独立视觉审查意见和冻结检测规则，批处理接受后会自动全范围回扫。请使用这些证据，"
                                    "对无法可靠恢复的短暂掉线可标记不可用；普通QRS、低压和生理变化需保护。自主决定处理和范围。"
                                    "如有可疑同类形态，用工具查看或记入未决项。不得硬编码患者、记录或时刻。"
                                    "下列为模型在本次原始信号上形成的观察与不确定项："
                                    + json.dumps(brief, ensure_ascii=False)
                                ),
                            }
                        ],
                    )
                result = run(out, cap=parent["cap_usd"])
                print("cleaning", result.get("stop"), flush=True)
        finally:
            if own.exists():
                write(billing, read(own))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ["source", "out", "known", "budget-root"]:
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--live", required=True, action="store_true")
    a = p.parse_args()
    execute(a.source, a.record, a.out, a.known, a.budget_root)
