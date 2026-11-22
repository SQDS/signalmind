"""Registered Part 2 analysis (docs/tasks/verification-experiment.md): A0 vs B vs C, per endpoint.

  .venv/bin/python eval/confirm_analysis.py --suite p6confirm --a0 A0.predictions.json \
      --c runC1+runC2+... --b runB1+runB2+... [--json out.json]

Primary: C vs B per endpoint, exact two-sided McNemar, Holm over the two endpoints (family alpha 0.05), Newcombe 95 %
CI. Success: C > B with Holm-adjusted p < 0.05 on at least one endpoint. Safety guardrail (descriptive): Newcombe
lower bound of C - B above -0.05 on the other endpoint. Secondary family (Holm): C vs A0 and B vs A0 per endpoint.
Missing, failed or contaminated rows count as wrong. Run dirs joined with '+' are merged (one worker per dir).
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
from eval.compare_p5afdb_arms import newcombe_paired, wilson  # noqa: E402
from eval.grade import grade  # noqa: E402

ENDPOINTS = ("longest_af", "af_burden")
B_CACHE = Path(__file__).resolve().parent / "cache" / "b-workflow" / "objects"
ALPHA, GUARDRAIL = 0.05, -0.05


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def key_of(qtype: str) -> str:
    return "t" if qtype == "longest_af" else "percent"


def rows_of(runs: str) -> tuple[dict, list]:
    rows, errors = {}, []
    for part in runs.split("+"):
        d = RUNS / part
        if (d / "results.jsonl").exists():
            for line in (d / "results.jsonl").read_text().splitlines():
                r = json.loads(line)
                rows[r["case_id"]] = r
        if (d / "errors.jsonl").exists():
            errors += [json.loads(x) for x in (d / "errors.jsonl").read_text().splitlines() if x.strip()]
    return rows, errors


def score(answers: dict, preds: dict, rows: dict | None = None) -> dict:
    out = {}
    for cid, a in answers.items():
        r = rows.get(cid) if rows is not None else None
        status = "ok" if rows is None else ("missing" if r is None else r.get("status"))
        pred = preds.get(cid) if rows is None else (r.get("pred") if r and status == "ok" else None)
        ok, err = False, None
        if pred is not None:
            ok = bool(grade(a["qtype"], a["answer"], pred).get("correct"))
            p, t = pred.get(key_of(a["qtype"])), a["answer"].get(key_of(a["qtype"]))
            err = abs(float(p) - float(t)) if p is not None and t is not None else None
        out[cid] = {"ok": ok, "err": err, "status": status, "row": r, "db": a["answer"].get("database")}
    return out


def paired(x: dict, y: dict, ids: list[str]) -> dict:
    a = sum(x[i]["ok"] and y[i]["ok"] for i in ids)
    b = sum(x[i]["ok"] and not y[i]["ok"] for i in ids)
    c = sum(y[i]["ok"] and not x[i]["ok"] for i in ids)
    d, lo, hi = newcombe_paired(a, b, c, len(ids) - a - b - c) if ids else (0.0, 0.0, 0.0)
    p = binomtest(b, b + c).pvalue if b + c else 1.0
    return {"n": len(ids), "only_first": b, "only_second": c, "both": a, "mcnemar_p": p,
            "diff_first_minus_second": d, "newcombe95": [lo, hi]}  # fmt: skip


def err_summary(arm: dict, ids: list[str]) -> dict:
    e = sorted(arm[i]["err"] for i in ids if arm[i]["err"] is not None)
    if not e:
        return {"n": 0}
    q = lambda p: e[min(len(e) - 1, int(p * (len(e) - 1)))]  # noqa: E731
    return {"n": len(e), "median": st.median(e), "p75": q(0.75), "p90": q(0.9), "max": e[-1]}


def b_onset_coverage(arm: dict, answers: dict, ids: list[str]) -> dict:
    """Reference onset inside any < 1 h ts_zoom window of the bundle the model actually received."""
    exact = tol = n = 0
    for i in ids:
        r = arm[i]["row"]
        sha = r.get("bundle_sha256") if r else None
        path = B_CACHE / f"{sha}.json" if sha else None
        if not path or not path.exists():
            continue
        items = json.loads(path.read_text()).get("items", [])
        ws = [it.get("arguments", {}).get("window") for it in items if it.get("tool") == "ts_zoom"]
        ws = [w for w in ws if isinstance(w, list) and len(w) == 2 and w[1] - w[0] < 3600]
        t = answers[i]["answer"].get("t")
        n += 1
        exact += any(w[0] <= t <= w[1] for w in ws)
        tol += any(w[0] - 60 <= t <= w[1] + 60 for w in ws)
    return {"bundles_found": n, "onset_inside_strip": exact, "onset_within_60s_of_strip": tol}


def behaviour(arm: dict, errors: list) -> dict:
    rows = [v["row"] for v in arm.values() if v["row"]]
    if not rows:
        return {"sessions": 0}
    u = [r.get("usage") or {} for r in rows]
    cost = [max(r.get("cost_usd") or 0, r.get("cc_cost_usd") or 0) for r in rows]
    classes: dict[str, int] = {}
    for e in errors:
        classes[e.get("class", "unknown")] = classes.get(e.get("class", "unknown"), 0) + 1
    return {
        "sessions": len(rows),
        "uncached_input_mean": st.mean(x.get("input_tokens") or 0 for x in u),
        "cache_write_mean": st.mean(x.get("cache_creation_input_tokens") or 0 for x in u),
        "cache_read_mean": st.mean(x.get("cache_read_input_tokens") or 0 for x in u),
        "output_tokens_mean": st.mean(x.get("output_tokens") or 0 for x in u),
        "cost_total": sum(cost), "cost_per_session": st.mean(cost),
        "latency_median_s": st.median(r.get("latency_s") or 0 for r in rows),
        "turns_mean": st.mean(r.get("turns") or 0 for r in rows),
        "tool_calls_mean": st.mean(len(r.get("tool_calls") or []) for r in rows),
        "final_status": {s: sum(v["status"] == s for v in arm.values()) for s in {v["status"] for v in arm.values()}},
        "attempt_errors_by_class": classes,
    }  # fmt: skip


def analyse(suite: str, a0_path: Path, c_runs: str, b_runs: str) -> dict:
    answers = {
        json.loads(x)["case_id"]: json.loads(x) for x in (CASES / f"{suite}.answers.jsonl").read_text().splitlines()
    }
    a0_preds = {r["case_id"]: r.get("pred") for r in json.loads(a0_path.read_text())["rows"]}
    c_rows, c_err = rows_of(c_runs)
    b_rows, b_err = rows_of(b_runs)
    arms = {"A0": score(answers, a0_preds), "B": score(answers, {}, b_rows), "C": score(answers, {}, c_rows)}
    out = {"suite": suite, "endpoints": {}, "primary": {}, "secondary": {}, "behaviour": {}, "subgroups": {}}
    primary_p, secondary_p = {}, {}
    for q in ENDPOINTS:
        ids = sorted(i for i, a in answers.items() if a["qtype"] == q)
        ep = {"n": len(ids)}
        for name, arm in arms.items():
            k = sum(arm[i]["ok"] for i in ids)
            ep[name] = {"correct": k, "wilson95": list(wilson(k, len(ids))), "abs_error": err_summary(arm, ids)}
            if name != "A0":
                ep[name]["vs_A0"] = {
                    "corrected_detector_errors": sum(not arms["A0"][i]["ok"] and arm[i]["ok"] for i in ids),
                    "introduced_errors": sum(arms["A0"][i]["ok"] and not arm[i]["ok"] for i in ids),
                }  # fmt: skip
        if q == "longest_af":
            ep["B"]["onset_coverage"] = b_onset_coverage(arms["B"], answers, ids)
        ep["C_vs_B"] = paired(arms["C"], arms["B"], ids)
        ep["C_vs_A0"] = paired(arms["C"], arms["A0"], ids)
        ep["B_vs_A0"] = paired(arms["B"], arms["A0"], ids)
        primary_p[q] = ep["C_vs_B"]["mcnemar_p"]
        secondary_p[f"C_vs_A0:{q}"] = ep["C_vs_A0"]["mcnemar_p"]
        secondary_p[f"B_vs_A0:{q}"] = ep["B_vs_A0"]["mcnemar_p"]
        dbs = sorted({arms["C"][i]["db"] for i in ids if arms["C"][i]["db"]})
        out["subgroups"][q] = {db: {"n": len(s := [i for i in ids if arms["C"][i]["db"] == db]),
                                    **{n: sum(arms[n][i]["ok"] for i in s) for n in arms},
                                    "C_vs_B": paired(arms["C"], arms["B"], s)} for db in dbs}  # fmt: skip
        out["endpoints"][q] = ep
    adj = holm(primary_p)
    sig = {q: bool(adj[q] < ALPHA) for q in ENDPOINTS}
    c_better = {
        q: out["endpoints"][q]["C_vs_B"]["only_first"] > out["endpoints"][q]["C_vs_B"]["only_second"] for q in ENDPOINTS
    }
    success = any(sig[q] and c_better[q] for q in ENDPOINTS)
    harm = [q for q in ENDPOINTS if sig[q] and not c_better[q]]
    guard = {}
    for q in ENDPOINTS:
        other = [x for x in ENDPOINTS if x != q][0]
        guard[q] = {"other_endpoint": other, "lower_bound": out["endpoints"][other]["C_vs_B"]["newcombe95"][0],
                    "above_minus_5pp": bool(out["endpoints"][other]["C_vs_B"]["newcombe95"][0] > GUARDRAIL)}  # fmt: skip
    beh = {n: behaviour(arms[n], e) for n, e in (("B", b_err), ("C", c_err))}
    ratio = (
        beh["C"].get("cost_per_session", 0) / beh["B"]["cost_per_session"] if beh["B"].get("cost_per_session") else None
    )
    out["primary"] = {
        "raw_p": primary_p, "holm_p": adj, "significant": sig, "C_better": c_better, "success": success,
        "guardrail_descriptive": {q: guard[q] for q in ENDPOINTS if sig[q] and c_better[q]},
        "failure_a_C_worse_significant": harm, "failure_b_neither_significant": not any(sig.values()),
        "cost_ratio_C_over_B": ratio, "failure_c_flag": bool(success and ratio and ratio > 2),
    }  # fmt: skip
    out["secondary"] = {"raw_p": secondary_p, "holm_p": holm(secondary_p)}
    out["behaviour"] = beh
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--a0", type=Path, required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = analyse(args.suite, args.a0, args.c, args.b)
    text = json.dumps(out, indent=1, default=float)
    if args.json:
        args.json.write_text(text + "\n")
    print(json.dumps({"primary": out["primary"], "endpoints": {q: {k: v for k, v in e.items()} for q, e in out["endpoints"].items()}},
                     indent=1, default=float))  # fmt: skip


if __name__ == "__main__":
    main()
