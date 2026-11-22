"""Registered Level 2 C/D analysis (docs/tasks/level2-analysis-plan.md §3, §6, §7).

  .venv/bin/python eval/level2_analysis.py --tag l2 --coding eval/results/p9level2/coding \
      [--attribution eval/results/p9level2/attribution.json] --out eval/results/p9level2/analysis.json

Grades every session with the frozen `eval/level2_grade.py`, using evidence records from the frozen blinded coding
(`<session_key>.json`, keys from `eval/level2_traces.py`). Reports every metric per capability category (generic,
specialist, abstention), family, signal kind and GT category, for all 49 items and for the registered sensitivity sets
(low-confidence GT excluded; complete case excluding infra).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from eval.level2_grade import grade_level2, grade_level2_pair
from eval.level2_traces import session_key

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"
MANIFEST = ROOT / "eval" / "tasks" / "p9level2_manifest.json"
ARMS = ("C", "D")
INFRA = {"api_error", "server_provenance"}
CLASSES = ("grounded_correct", "correct_ungrounded", "incorrect", "correct_abstention", "fabrication",
           "unnecessary_abstention", "no_answer", "timeout", "infra")  # fmt: skip


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def client_version(transcript: Path | None) -> str:
    """Claude Code version from the transcript's init event (incident E1: 2.1.293 or 2.1.311)."""
    if transcript is None or not transcript.exists():
        return "none"
    m = re.search(r'"claude_code_version":"([^"]+)"', transcript.read_text(errors="replace")[:20000])
    return m.group(1) if m else "unknown"


def tool_counts(transcript: Path) -> Counter:
    calls: Counter = Counter()
    if not transcript.exists():
        return calls
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "assistant":
            for c in e.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    calls[c.get("name", "?").removeprefix("mcp__tsmcp__")] += 1
    return calls


def sessions(tag: str) -> dict[tuple[str, str], dict]:
    """(arm, case_id) -> final session record: the result row if any, else the last error class."""
    out: dict[tuple[str, str], dict] = {}
    for d in sorted(RUNS.glob(f"p9level2-[CD]-*-{tag}-w*")):
        arm = d.name.split("-")[1]
        errs = defaultdict(list)
        for e in read(d / "errors.jsonl"):
            errs[e["case_id"]].append(e.get("class"))
        for cid, classes in errs.items():
            out[(arm, cid)] = {"error_classes": classes, "dir": d}
        for r in read(d / "results.jsonl"):
            out[(arm, r["case_id"])] = {**r, "error_classes": errs.get(r["case_id"], []), "dir": d}
    return out


def status_of(s: dict | None) -> str:
    if s is None:
        return "not_run"
    if "pred" in s:
        return "answered" if s["pred"] is not None else "no_answer"
    last = s["error_classes"][-1] if s["error_classes"] else "unknown"
    return "infra" if last in INFRA else "timeout" if last == "timeout" else "no_answer"


def result_class(status: str, g: dict | None) -> str:
    if status in ("infra", "timeout", "not_run"):
        return "infra" if status != "timeout" else "timeout"
    if status == "no_answer" or g is None or g["outcome"] == "no_answer":
        return "no_answer"
    if g["outcome"] in ("correct_abstention", "fabrication", "unnecessary_abstention"):
        return g["outcome"]
    if g["correct"]:
        return "grounded_correct" if g["grounded"] else "correct_ungrounded"
    return "incorrect"


def grade_all(tag: str, coding: Path) -> list[dict]:
    items = json.loads(MANIFEST.read_text())["questions"]
    runs = sessions(tag)
    rows = []
    for item in items:
        for arm in ARMS:
            s = runs.get((arm, item["case_id"]))
            status = status_of(s)
            key = session_key(arm, item["case_id"])
            coded = coding / f"{key}.json"
            evidence = json.loads(coded.read_text())["records"] if coded.exists() else []
            g = grade_level2(s.get("pred"), item, evidence) if status == "answered" else None
            if g is None and status == "no_answer":
                g = grade_level2(None, item, evidence)
            transcript = None
            if s is not None:
                t = s["dir"] / "transcripts" / f"{item['case_id']}.jsonl"
                transcript = t if t.exists() else s["dir"] / "transcripts" / f"{item['case_id']}.partial.jsonl"
            usage = (s or {}).get("usage") or {}
            rows.append({
                "arm": arm, "case_id": item["case_id"], "session_key": key, "family": item["family"],
                "category": item["expected_solvability"], "signal_kind": item["signal_kind"],
                "gt_category": item["ground_truth"]["category"], "gt_confidence": item["ground_truth"]["confidence"],
                "task_type": item["task_type"], "pair_id": item.get("pair_id"), "pair_role": item.get("pair_role"),
                "status": status, "error_classes": (s or {}).get("error_classes", []), "coded": coded.exists(),
                "grade": g, "class": result_class(status, g),
                "cost_usd": max((s or {}).get("cost_usd") or 0, (s or {}).get("cc_cost_usd") or 0) if s else 0.0,
                "latency_s": (s or {}).get("latency_s"), "turns": (s or {}).get("turns"),
                "tokens": {k: usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                          "cache_read_input_tokens", "output_tokens")},
                "tools": dict(tool_counts(transcript)) if transcript else {},
                "client_version": client_version(transcript),
            })  # fmt: skip
    return rows


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    classes = Counter(r["class"] for r in rows)
    correct = classes["grounded_correct"] + classes["correct_ungrounded"] + classes["correct_abstention"]
    graded = [r["grade"] for r in rows if r["grade"] is not None]
    rate = lambda key: sum(bool(g[key]) for g in graded)  # noqa: E731
    return {
        "n": n, "correct": correct, "correct_ci95": wilson(correct, n),
        "grounded_correct": classes["grounded_correct"] + sum(
            r["class"] == "correct_abstention" and r["grade"]["grounded"] for r in rows),
        "classes": {c: classes[c] for c in CLASSES},
        "evidence_validity": rate("evidence_validity"), "decisive_regions_supported": rate("decisive_regions_supported"),
        "required_computations_supported": rate("required_computations_supported"),
        "verification_performed": rate("verification_performed"), "graded": len(graded),
    }  # fmt: skip


def paired(rows: list[dict]) -> dict:
    by = defaultdict(dict)
    for r in rows:
        by[r["case_id"]][r["arm"]] = r["class"] in ("grounded_correct", "correct_ungrounded", "correct_abstention")
    by = {k: v for k, v in by.items() if len(v) == 2}  # only items with both arms in this subset
    b = sum(v.get("C", False) and not v.get("D", False) for v in by.values())
    c = sum(v.get("D", False) and not v.get("C", False) for v in by.values())
    both = sum(v.get("C", False) and v.get("D", False) for v in by.values())
    return {
        "items": len(by),
        "both_correct": both,
        "C_only": b,
        "D_only": c,
        "exact_mcnemar_p_descriptive": round(mcnemar_exact(b, c), 4),
    }


def grouped(rows: list[dict], field: str) -> dict:
    out = {}
    for value in sorted({r[field] for r in rows}):
        sub = [r for r in rows if r[field] == value]
        out[value] = {arm: summary([r for r in sub if r["arm"] == arm]) for arm in ARMS} | {"paired": paired(sub)}
    return out


def errors_by_task(rows: list[dict]) -> dict:
    out = {}
    for task in sorted({r["task_type"] for r in rows}):
        for arm in ARMS:
            fields = defaultdict(lambda: {"abs": [], "rel": []})
            for r in rows:
                if r["task_type"] == task and r["arm"] == arm and r["grade"]:
                    for f, e in r["grade"]["field_errors"].items():
                        if e.get("absolute_error") is not None:
                            fields[f]["abs"].append(e["absolute_error"])
                            if e.get("relative_error") is not None:
                                fields[f]["rel"].append(e["relative_error"])
            for f, v in fields.items():
                q = lambda xs: [round(x, 4) for x in statistics.quantiles(xs, n=4)] if len(xs) > 1 else xs  # noqa: E731
                out.setdefault(task, {}).setdefault(arm, {})[f] = {
                    "n": len(v["abs"]), "median_abs": round(statistics.median(v["abs"]), 4),
                    "iqr_abs": q(v["abs"]), "median_rel": round(statistics.median(v["rel"]), 4) if v["rel"] else None,
                }  # fmt: skip
    return out


def abstention(rows: list[dict], items: list[dict], tag: str, coding: Path, exclude: set[str]) -> dict:
    runs = sessions(tag)
    pairs = defaultdict(list)
    for item in items:
        if item.get("pair_id"):
            pairs[item["pair_id"]].append(item)
    out = {}
    for arm in ARMS:
        scores = []
        for members in pairs.values():
            if any(m["case_id"] in exclude for m in members):
                continue
            answers, evidence = [], []
            for m in members:
                s = runs.get((arm, m["case_id"]))
                answers.append(s.get("pred") if s and "pred" in s else None)
                coded = coding / f"{session_key(arm, m['case_id'])}.json"
                evidence.append(json.loads(coded.read_text())["records"] if coded.exists() else [])
            p = grade_level2_pair(answers, members, evidence)
            scores.append(
                {k: p[k] for k in ("correct_abstention", "fabrication", "unnecessary_abstention", "pair_correct")}
            )
        answerable = [r for r in rows if r["arm"] == arm and r["category"] != "abstention"]
        out[arm] = {
            "pairs": len(scores), **{k: sum(s[k] for s in scores) for k in ("correct_abstention", "fabrication",
                                                                            "unnecessary_abstention", "pair_correct")},
            "unnecessary_abstention_all_answerable": [sum(r["class"] == "unnecessary_abstention" for r in answerable),
                                                      len(answerable)],
        }  # fmt: skip
    return out


def behaviour(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm and r["status"] != "not_run"]
        tools: Counter = Counter()
        for r in sub:
            tools.update(r["tools"])
        med = lambda xs: round(statistics.median(xs), 3) if xs else None  # noqa: E731
        out[arm] = {
            "sessions": len(sub), "cost_usd_total": round(sum(r["cost_usd"] for r in sub), 4),
            "cost_usd_median": med([r["cost_usd"] for r in sub if r["cost_usd"]]),
            "latency_s_median": med([r["latency_s"] for r in sub if r["latency_s"] is not None]),
            "turns_median": med([r["turns"] for r in sub if r["turns"] is not None]),
            "tool_calls_median": med([sum(r["tools"].values()) for r in sub]),
            "tool_calls_by_tool": dict(tools.most_common()),
            "tokens_total": {k: sum(r["tokens"][k] for r in sub) for k in sub[0]["tokens"]} if sub else {},
            "failures_by_class": dict(Counter(c for r in sub for c in r["error_classes"])),
        }  # fmt: skip
    return out


def f6(rows: list[dict]) -> dict:
    return {arm: {"n": len(s := [r for r in rows if r["arm"] == arm and r["family"] == "F6" and r["grade"]]),
                  "verification_performed": sum(r["grade"]["verification_performed"] for r in s),
                  "misleading_output_detected": sum(r["grade"]["misleading_output_detected"] for r in s)}
            for arm in ARMS}  # fmt: skip


def analyse(rows: list[dict], items: list[dict], tag: str, coding: Path, exclude: set[str]) -> dict:
    rows = [r for r in rows if r["case_id"] not in exclude]
    return {
        "items": len({r["case_id"] for r in rows}),
        "overall": {arm: summary([r for r in rows if r["arm"] == arm]) for arm in ARMS} | {"paired": paired(rows)},
        **{f"by_{f}": grouped(rows, f) for f in ("category", "family", "signal_kind", "gt_category", "client_version")},
        "numerical_errors": errors_by_task(rows),
        "abstention": abstention(rows, items, tag, coding, exclude),
        "f6": f6(rows),
        "behaviour": behaviour(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--coding", type=Path, required=True)
    ap.add_argument("--attribution", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    items = json.loads(MANIFEST.read_text())["questions"]
    rows = grade_all(args.tag, args.coding)
    low = {i["case_id"] for i in items if i["ground_truth"]["confidence"] == "low"}
    infra = {r["case_id"] for r in rows if r["class"] == "infra"}
    result = {
        "plan": "docs/tasks/level2-analysis-plan.md",
        "uncoded_sessions": sum(not r["coded"] and r["status"] != "not_run" for r in rows),
        "complete": analyse(rows, items, args.tag, args.coding, set()),
        "sensitivity_low_confidence_excluded": analyse(rows, items, args.tag, args.coding, low),
        "sensitivity_complete_case": analyse(rows, items, args.tag, args.coding, infra),
        "sessions": [{k: v for k, v in r.items() if k != "grade"} | {
            "grade": {k: v for k, v in (r["grade"] or {}).items()}} for r in rows],
    }  # fmt: skip
    if args.attribution and args.attribution.exists():
        att = json.loads(args.attribution.read_text())
        result["attribution"] = att
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    c = result["complete"]
    print("uncoded sessions:", result["uncoded_sessions"])
    for cat, v in c["by_category"].items():
        print(cat, {a: (v[a]["correct"], v[a]["n"], v[a]["grounded_correct"]) for a in ARMS}, v["paired"])


if __name__ == "__main__":
    main()
