"""Part 2 development analysis on p7dev: A0 vs B (each version) vs C (docs/tasks/verification-experiment.md).

  .venv/bin/python eval/part2_dev_analysis.py --a0 eval/results/part2-dev/a0-R.predictions.json \
      --c p7dev-C-claude-opus-5-5-medium-dev --b v0=p7dev-B-claude-opus-5-5-medium-v0 [--b v1=...] [--json out.json]

Development data only; never point it at p6confirm. Missing/failed rows count as wrong.
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

SUITE = "p7dev"
MANIFESTS = Path(__file__).resolve().parent / "results" / "b-workflow" / SUITE / "manifests"
TOL = {"longest_af": 60.0}


def key_of(qtype: str) -> str:
    return "t" if qtype == "longest_af" else "percent"


def graded(qtype: str, answer: dict, pred: dict | None) -> tuple[bool, float | None]:
    if pred is None:
        return False, None
    g = grade(qtype, answer, pred)
    k = key_of(qtype)
    p, t = pred.get(k), answer.get(k)
    err = abs(float(p) - float(t)) if p is not None and t is not None else None
    return bool(g.get("correct")), err


def load_run(run: str, answers: dict) -> dict:
    """`run` may join several run dirs with '+' (e.g. the endpoint-specific runs of the final B)."""
    rows = {}
    for part in run.split("+"):
        path = RUNS / part / "results.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rows[r["case_id"]] = r
    out = {}
    for cid, a in answers.items():
        r = rows.get(cid)
        if r is None or r.get("status") != "ok":
            out[cid] = {"ok": False, "err": None, "status": "missing" if r is None else r.get("status"), "row": r}
            continue
        ok, err = graded(a["qtype"], a["answer"], r.get("pred"))
        out[cid] = {"ok": ok, "err": err, "status": "ok", "row": r}
    return out


def load_a0(path: Path, answers: dict) -> dict:
    data = json.loads(path.read_text())
    preds = {r["case_id"]: r.get("pred") for r in data["rows"]}
    out = {}
    for cid, a in answers.items():
        ok, err = graded(a["qtype"], a["answer"], preds.get(cid))
        out[cid] = {"ok": ok, "err": err, "status": "ok" if cid in preds else "missing", "row": None}
    return out


def b_coverage(cid: str, answer: dict) -> dict:
    """Did B's fixed bundle contain evidence near the reference onset / the reference episode?"""
    m = json.loads((MANIFESTS / f"{cid}.json").read_text())
    t = answer.get("t")
    windows = [it["arguments"].get("window") for it in m.get("items", []) if it.get("tool") == "ts_zoom"]
    windows = [w for w in windows if isinstance(w, list) and len(w) == 2]
    strip_near_onset = t is not None and any(w[0] - TOL["longest_af"] <= t <= w[1] + TOL["longest_af"] for w in windows
                                             if (w[1] - w[0]) < 3600)  # fmt: skip
    dropped = m.get("dropped") or []
    return {"strip_near_reference_onset": strip_near_onset, "zoom_items": len(windows),
            "dropped_items": len(dropped), "core_text_chars": m["sizes"]["core_text_chars"],
            "images": m["sizes"]["images"], "calls": m["sizes"]["calls"]}  # fmt: skip


def paired(x: dict, y: dict, ids: list[str]) -> dict:
    a = sum(x[i]["ok"] and y[i]["ok"] for i in ids)
    b = sum(x[i]["ok"] and not y[i]["ok"] for i in ids)
    c = sum(y[i]["ok"] and not x[i]["ok"] for i in ids)
    d, lo, hi = newcombe_paired(a, b, c, len(ids) - a - b - c)
    return {"only_first": b, "only_second": c, "mcnemar_p": binomtest(b, b + c).pvalue if b + c else 1.0,
            "diff_first_minus_second": d, "newcombe95": (lo, hi)}  # fmt: skip


def behaviour(arm: dict) -> dict:
    rows = [v["row"] for v in arm.values() if v["row"]]
    if not rows:
        return {}
    u = [r.get("usage") or {} for r in rows]
    return {
        "sessions": len(rows),
        "input_tokens_mean": st.mean(sum(x.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                                   "cache_read_input_tokens")) for x in u),
        "output_tokens_mean": st.mean(x.get("output_tokens") or 0 for x in u),
        "cost_total": sum(max(r.get("cost_usd") or 0, r.get("cc_cost_usd") or 0) for r in rows),
        "latency_median_s": st.median(r.get("latency_s") or 0 for r in rows),
        "turns_mean": st.mean(r.get("turns") or 0 for r in rows),
        "tool_calls_mean": st.mean(len(r.get("tool_calls") or []) for r in rows),
        "failures": sum(v["status"] != "ok" for v in arm.values()),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", type=Path, required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--b", action="append", default=[], help="label=run_dir")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    answers = {
        json.loads(x)["case_id"]: json.loads(x) for x in (CASES / f"{SUITE}.answers.jsonl").read_text().splitlines()
    }
    arms = {"A0": load_a0(args.a0, answers), "C": load_run(args.c, answers)}
    for spec in args.b:
        label, run = spec.split("=", 1)
        arms[f"B-{label}"] = load_run(run, answers)
    out = {"suite": SUITE, "development_only": True, "endpoints": {}, "behaviour": {}, "b_coverage": {}, "cases": []}
    for q in ("longest_af", "af_burden"):
        ids = sorted(i for i, a in answers.items() if a["qtype"] == q)
        ep = {"n": len(ids), "correct": {}, "wilson95": {}, "median_abs_err": {}, "vs_A0": {}, "C_vs_B": {}}
        for name, arm in arms.items():
            k = sum(arm[i]["ok"] for i in ids)
            errs = [arm[i]["err"] for i in ids if arm[i]["err"] is not None]
            ep["correct"][name] = k
            ep["wilson95"][name] = wilson(k, len(ids))
            ep["median_abs_err"][name] = st.median(errs) if errs else None
            if name != "A0":
                ep["vs_A0"][name] = {
                    "corrected_detector_errors": sum(arms["A0"][i]["ok"] is False and arm[i]["ok"] for i in ids),
                    "introduced_errors": sum(arms["A0"][i]["ok"] and not arm[i]["ok"] for i in ids),
                }  # fmt: skip
        for name in arms:
            if name.startswith("B-"):
                ep["C_vs_B"][name] = paired(arms["C"], arms[name], ids)
        out["endpoints"][q] = ep
    for name, arm in arms.items():
        if name != "A0":
            out["behaviour"][name] = behaviour(arm)
    cov = {cid: b_coverage(cid, a["answer"]) for cid, a in answers.items() if (MANIFESTS / f"{cid}.json").exists()}
    on = [cov[c] for c in cov if answers[c]["qtype"] == "longest_af"]
    out["b_coverage"] = {
        "longest_af_strip_near_reference_onset": sum(c["strip_near_reference_onset"] for c in on),
        "longest_af_cases": len(on),
        "cases_with_dropped_items": sum(c["dropped_items"] > 0 for c in cov.values()),
        "core_text_chars_max": max(c["core_text_chars"] for c in cov.values()),
        "per_case": cov,
    }  # fmt: skip
    for cid, a in sorted(answers.items()):
        row = {"case_id": cid, "qtype": a["qtype"], "truth": a["answer"].get(key_of(a["qtype"]))}
        for name, arm in arms.items():
            pred = arm[cid]["row"]["pred"] if arm[cid]["row"] else None
            row[name] = {"ok": arm[cid]["ok"], "err": arm[cid]["err"], "pred": pred}
        out["cases"].append(row)
    text = json.dumps(out, indent=1, default=float)
    if args.json:
        args.json.write_text(text + "\n")
    print(json.dumps({"endpoints": out["endpoints"], "behaviour": out["behaviour"],
                      "b_coverage": {k: v for k, v in out["b_coverage"].items() if k != "per_case"}},
                     indent=1, default=float))  # fmt: skip


if __name__ == "__main__":
    main()
