"""Paired v1 vs v2 (compact interface) diagnostic on p5afdb; protocol docs/tasks/compact-v2-diagnostic.md.

.venv/bin/python eval/compare_v2_diagnostic.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import CASES, RUNS  # noqa: E402
from eval.compare_p5afdb_arms import load_arm, newcombe_paired, wilson  # noqa: E402

MODELS = {"claude": "claude-opus-5-5-medium", "deepseek": "deepseek-flash_deepseek-flash-high"}


def tool_stats(run: str, cid: str) -> dict:
    """Tool-result characters, tool calls by name and detail requests, from the transcript."""
    chars, names, details = 0, [], 0
    for line in (RUNS / run / "transcripts" / f"{cid}.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = e.get("message") if isinstance(e.get("message"), dict) else {}
        for b in m.get("content", []) if isinstance(m.get("content"), list) else []:
            if b.get("type") == "tool_use":
                names.append(b.get("name"))
                details += "ecg_af_episode_detail" in json.dumps(b.get("input"))
            elif b.get("type") == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    chars += sum(len(x.get("text", "")) for x in c if isinstance(x, dict))
                elif isinstance(c, str):
                    chars += len(c)
    return {"tool_chars": chars, "calls": len(names), "detail_calls": details,
            "episode_index_calls": sum(1 for n in names if n == "mcp__tsmcp__ts_measure")}  # fmt: skip


def usage_split(run: str) -> dict:
    out = {}
    for line in (RUNS / run / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        u = r.get("usage") or {}
        out[r["case_id"]] = {k: u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                       "cache_read_input_tokens", "output_tokens")}  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    answers = {
        json.loads(x)["case_id"]: json.loads(x) for x in (CASES / "p5afdb.answers.jsonl").read_text().splitlines()
    }
    out = {"diagnostic_only": True, "models": {}}
    for name, tag in MODELS.items():
        runs = {"v1": f"p5afdb-C-{tag}-R", "v2": f"p5afdb-C-{tag}-R-v2"}
        arms = {k: load_arm(r, answers) for k, r in runs.items()}
        tools = {k: {c: tool_stats(r, c) for c in answers} for k, r in runs.items()}
        use = {k: usage_split(r) for k, r in runs.items()}
        res = {"endpoints": {}, "per_case_means": {}, "changes": []}
        for q in ("longest_af", "af_burden", None):
            ids = sorted(i for i, a in answers.items() if q is None or a["qtype"] == q)
            X, Y = arms["v1"], arms["v2"]
            a = sum(X[i]["ok"] and Y[i]["ok"] for i in ids)
            b = sum(X[i]["ok"] and not Y[i]["ok"] for i in ids)
            c = sum(Y[i]["ok"] and not X[i]["ok"] for i in ids)
            d, lo, hi = newcombe_paired(a, b, c, len(ids) - a - b - c)
            errs = {k: [arms[k][i]["err"] for i in ids if arms[k][i]["err"] is not None] for k in arms}
            res["endpoints"][q or "pooled_descriptive"] = {
                "n": len(ids), "v1": a + b, "v2": a + c, "v1_only": b, "v2_only": c,
                "mcnemar_p": binomtest(b, b + c).pvalue if b + c else 1.0, "diff_v2_minus_v1": -d,
                "newcombe95": (-hi, -lo), "wilson95": {k: wilson(sum(arms[k][i]["ok"] for i in ids), len(ids)) for k in arms},
                "median_abs_err": {k: st.median(v) if v else None for k, v in errs.items()},
            }  # fmt: skip
        for k in runs:
            v = arms[k]
            res["per_case_means"][k] = {
                "uncached_input": st.mean(use[k][c]["input_tokens"] for c in answers),
                "cache_write": st.mean(use[k][c]["cache_creation_input_tokens"] for c in answers),
                "cache_read": st.mean(use[k][c]["cache_read_input_tokens"] for c in answers),
                "output": st.mean(use[k][c]["output_tokens"] for c in answers),
                "total_input": st.mean(v[c]["input_tokens"] for c in answers),
                "cost": st.mean(v[c]["cost"] for c in answers), "cost_total": sum(v[c]["cost"] for c in answers),
                "latency_median": st.median(v[c]["latency"] for c in answers),
                "turns": st.mean(v[c]["turns"] or 0 for c in answers),
                "tool_calls": st.mean(tools[k][c]["calls"] for c in answers),
                "tool_result_chars": st.mean(tools[k][c]["tool_chars"] for c in answers),
                "detail_calls_total": sum(tools[k][c]["detail_calls"] for c in answers),
                "failures": sum(v[c]["status"] != "ok" for c in answers),
            }  # fmt: skip
        for c in sorted(answers):
            if arms["v1"][c]["ok"] != arms["v2"][c]["ok"]:
                res["changes"].append({"case_id": c, "qtype": answers[c]["qtype"], "truth": arms["v1"][c]["truth"],
                                       "v1": arms["v1"][c]["pred"], "v2": arms["v2"][c]["pred"],
                                       "v2_correct": arms["v2"][c]["ok"]})  # fmt: skip
        out["models"][name] = res
    text = json.dumps(out, indent=1, default=float)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
