"""Paired comparison of Claude arms C-R, D and E on p5afdb (diagnostic; protocol docs/tasks/p5afdb-python-arms.md).

  uv run --group eval python eval/compare_p5afdb_arms.py [--json out.json]

Rows that are missing, contaminated or failed count as wrong (reported separately). No model calls.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import CASES, RUNS  # noqa: E402
from eval.grade import grade  # noqa: E402

ARMS = {
    "C-R": "p5afdb-C-claude-opus-5-5-medium-R",
    "D": "p5afdb-D-claude-opus-5-5-medium",
    "E": "p5afdb-E-claude-opus-5-5-medium-R",
}
PAIRS = [("C-R", "D"), ("C-R", "E"), ("E", "D")]
# Leakage audit over D/E tool inputs (lower-cased). Network patterns: blocked by the Bash sandbox, logged as attempts.
LEAK_FILE = (r"/home\b", r"~/", r"\.atr\b", r"\.hea\b", r"\.qrs\b", r"\bafdb\b", r"\bltafdb\b", r"answers\.jsonl",
             r"\bcases/", r"\bdata/", r"rdann", r"pn_dir")  # fmt: skip
LEAK_NET = (r"physionet", r"\bcurl\b", r"\bwget\b", r"urllib", r"\brequests\.", r"http[s]?://", r"socket",
            r"pip install")  # fmt: skip
DENIED = re.compile(r"operation not permitted|network is unreachable|name or service not known|"
                    r"temporary failure in name resolution|connection refused|permission denied", re.I)  # fmt: skip


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Difference p1 - p2 of paired proportions and Newcombe's (1998, method 10) 95 % CI.
    a = both right, b = only arm 1 right, c = only arm 2 right, d = both wrong."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / den if den else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    hi = delta + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return delta, lo, hi


def load_arm(run: str, answers: dict) -> dict:
    rows = {}
    path = RUNS / run / "results.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[r["case_id"]] = r
    out = {}
    for cid, a in answers.items():
        r = rows.get(cid)
        if r is None:
            out[cid] = {"qtype": a["qtype"], "ok": False, "status": "missing", "pred": None, "err": None}
            continue
        g = grade(a["qtype"], a["answer"], r.get("pred")) if r["status"] == "ok" else {"correct": False}
        pred = r.get("pred") or {}
        key = "t" if a["qtype"] == "longest_af" else "percent"
        p, truth = pred.get(key), a["answer"].get(key)
        err = abs(float(p) - truth) if p is not None and truth is not None else None
        u = r.get("usage") or {}
        out[cid] = {
            "qtype": a["qtype"], "ok": bool(g.get("correct")), "status": r["status"], "pred": p, "truth": truth,
            "err": err, "turns": r.get("turns"), "tool_calls": len(r.get("tool_calls") or []),
            "input_tokens": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                         "cache_read_input_tokens")),
            "output_tokens": u.get("output_tokens") or 0, "cost": r.get("cost_usd") or 0.0,
            "latency": r.get("latency_s"), "flags": r.get("contamination_flags") or [],
            "provenance": {k: r.get(k) for k in ("server_git_sha", "e_git_sha", "e_package_path", "e_plugin_hint")},
        }  # fmt: skip
    return out


def leakage_audit(run: str) -> dict:
    """Scan every tool input of every transcript; also count tool results that look like sandbox denials."""
    report = {}
    for t in sorted((RUNS / run / "transcripts").glob("*.jsonl")):
        inputs, denials = [], 0
        for line in t.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = e.get("message", {}).get("content") if isinstance(e.get("message"), dict) else None
            for b in content if isinstance(content, list) else []:
                if b.get("type") == "tool_use":
                    inputs.append(json.dumps(b.get("input")))
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    denials += bool(DENIED.search(c if isinstance(c, str) else json.dumps(c)))
        blob = re.sub(r"/tmp/tsmcp-[a-z]-[\w]+(/home)?", "<work>", "\n".join(inputs).lower())
        report[t.stem] = {
            "file_hits": sorted({p for p in LEAK_FILE if re.search(p, blob)}),
            "network_hits": sorted({p for p in LEAK_NET if re.search(p, blob)}),
            "denied_results": denials,
        }
    return report


def summarise(arms: dict, ids: list[str]) -> dict:
    res = {"n": len(ids), "arms": {}, "pairs": {}}
    for name, D in arms.items():
        k = sum(D[i]["ok"] for i in ids)
        errs = [D[i]["err"] for i in ids if D[i]["err"] is not None]
        res["arms"][name] = {"correct": k, "wilson95": wilson(k, len(ids)),
                             "median_abs_err": st.median(errs) if errs else None}  # fmt: skip
    for x, y in PAIRS:
        X, Y = arms[x], arms[y]
        a = sum(X[i]["ok"] and Y[i]["ok"] for i in ids)
        b = sum(X[i]["ok"] and not Y[i]["ok"] for i in ids)
        c = sum(Y[i]["ok"] and not X[i]["ok"] for i in ids)
        d = len(ids) - a - b - c
        delta, lo, hi = newcombe_paired(a, b, c, d)
        res["pairs"][f"{x} vs {y}"] = {"both": a, f"only_{x}": b, f"only_{y}": c, "neither": d,
                                       "mcnemar_p": binomtest(b, b + c).pvalue if b + c else 1.0,
                                       "diff": delta, "newcombe95": (lo, hi)}  # fmt: skip
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    answers = {
        json.loads(x)["case_id"]: json.loads(x) for x in (CASES / "p5afdb.answers.jsonl").read_text().splitlines()
    }
    arms = {name: load_arm(run, answers) for name, run in ARMS.items()}
    out = {"diagnostic_only": True, "endpoints": {}, "behaviour": {}, "failures": {}, "leakage": {}, "cases": []}
    for q in ("longest_af", "af_burden", None):
        ids = sorted(i for i, a in answers.items() if q is None or a["qtype"] == q)
        out["endpoints"][q or "pooled_28_descriptive"] = summarise(arms, ids)
    for name, D in arms.items():
        v = list(D.values())
        out["behaviour"][name] = {
            "input_tokens_per_case": st.mean(r.get("input_tokens", 0) for r in v),
            "output_tokens_per_case": st.mean(r.get("output_tokens", 0) for r in v),
            "cost_total": sum(r.get("cost", 0) for r in v), "cost_per_case": st.mean(r.get("cost", 0) for r in v),
            "latency_median_s": st.median(r["latency"] for r in v if r.get("latency") is not None),
            "turns_per_case": st.mean(r.get("turns") or 0 for r in v),
            "tool_calls_per_case": st.mean(r.get("tool_calls", 0) for r in v),
        }  # fmt: skip
        out["failures"][name] = {s: sum(r["status"] == s for r in v) for s in ("missing", "contaminated", "error")}
        out["failures"][name]["flagged_ids"] = [i for i, r in D.items() if r.get("flags")]
    for name in ("D", "E"):
        out["leakage"][name] = leakage_audit(ARMS[name])
    for i in sorted(answers):
        oks = {n: arms[n][i]["ok"] for n in arms}
        if len(set(oks.values())) > 1:
            out["cases"].append({"case_id": i, "qtype": answers[i]["qtype"], "truth": arms["D"][i].get("truth"),
                                 **{n: {"ok": arms[n][i]["ok"], "pred": arms[n][i]["pred"], "err": arms[n][i]["err"]}
                                    for n in arms}})  # fmt: skip
    text = json.dumps(out, indent=1, default=float)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
