"""Headroom map: accuracy per arm x question type, with tokens/cost/turns. Recomputed from raw rows.

  uv run --group eval python eval/report.py --suite p1 [--runs p1-A-... p1-C-...]
Always includes the free rows: scripted (algorithms, no LLM), null (constant 'nothing'), oracle.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import scripted  # noqa: E402
from eval.common import CASES, RUNS, load_suite, read_jsonl  # noqa: E402
from eval.grade import grade, null, oracle  # noqa: E402


def free_rows(suite: str, answers: dict) -> dict[str, list[dict]]:
    out = {"oracle": [], "null": [], "scripted": []}
    for c in load_suite(suite):
        a = answers[c["case_id"]]
        x = np.load(Path(c["dir"]) / c["visible"]["file"])
        fs = c["visible"]["channels"][0]["fs"]
        for name, pred in (
            ("oracle", oracle(c["qtype"], a)),
            ("null", null(c["qtype"])),
            ("scripted", scripted.answer(c["qtype"], c["question"], x, fs)),
        ):
            out[name].append(
                {
                    "case_id": c["case_id"],
                    "qtype": c["qtype"],
                    "pred": pred,
                    "status": "ok",
                    "usage": {},
                    "cost_usd": 0,
                    "turns": 0,
                }
            )
    return out


def summarize(rows: list[dict], answers: dict) -> dict:
    per = defaultdict(lambda: {"n": 0, "correct": 0, "no_answer": 0, "excluded": 0})
    tok = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
    cost, turns = 0.0, []
    for r in rows:
        q = per[r["qtype"]]
        q["n"] += 1
        if r.get("status") in ("contaminated", "error"):
            q["excluded"] += 1
            continue
        g = grade(r["qtype"], answers[r["case_id"]], r.get("pred"))
        if g["status"] == "no_answer":
            q["no_answer"] += 1
        elif g["correct"]:
            q["correct"] += 1
        u = r.get("usage") or {}
        tok["in"] += u.get("input_tokens") or 0
        tok["out"] += u.get("output_tokens") or 0
        tok["cache_read"] += u.get("cache_read_input_tokens") or 0
        tok["cache_write"] += u.get("cache_creation_input_tokens") or 0
        cost += r.get("cost_usd") or 0
        if r.get("turns"):
            turns.append(r["turns"])
    n = sum(q["n"] - q["excluded"] for q in per.values())
    correct = sum(q["correct"] for q in per.values())
    return {
        "per": dict(per),
        "n": n,
        "acc": correct / n if n else 0,
        "tok": tok,
        "cost": cost,
        "turns": float(np.mean(turns)) if turns else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    answers = {r["case_id"]: r["answer"] for r in read_jsonl(CASES / f"{args.suite}.answers.jsonl")}
    arms = free_rows(args.suite, answers)
    run_names = args.runs or sorted(p.name for p in RUNS.glob(f"{args.suite}-*") if (p / "results.jsonl").exists())
    for rn in run_names:
        arms[rn.removeprefix(args.suite + "-")] = read_jsonl(RUNS / rn / "results.jsonl")
    qtypes = sorted({r["qtype"] for r in load_suite(args.suite)})
    head = (
        "| arm | acc | " + " | ".join(qtypes) + " | no-ans | excl | in+cache tok/case | out tok/case | $/case | turns |"
    )
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for name, rows in arms.items():
        s = summarize(rows, answers)
        cells = []
        for q in qtypes:
            p = s["per"].get(q)
            cells.append(f"{p['correct']}/{p['n'] - p['excluded']}" if p else "-")
        n = max(len(rows), 1)
        na = sum(p["no_answer"] for p in s["per"].values())
        ex = sum(p["excluded"] for p in s["per"].values())
        tin = (s["tok"]["in"] + s["tok"]["cache_read"] + s["tok"]["cache_write"]) / n
        print(
            f"| {name} | {s['acc']:.0%} | "
            + " | ".join(cells)
            + f" | {na} | {ex} | {tin:,.0f} | {s['tok']['out'] / n:,.0f} | {s['cost'] / n:.3f} | {s['turns']:.1f} |"
        )
    print(
        "\nrows marked excl = contaminated/error (not scored). no-ans = missing/unparseable ANSWER line (counted wrong in acc)."
    )


if __name__ == "__main__":
    main()
