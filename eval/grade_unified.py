"""Grade matched-arm runs on the frozen Level 2 suite with the frozen grader (docs/signalmind/phase2-protocol.md §5).

  .venv/bin/python eval/grade_unified.py --tag dev --arms DS-Base DS-Agent H-Base H-Agent --out eval/results/phase2/dev.json
                                          [--coding DIR]   # blinded grounding coding (same protocol as Level 2)

Use --primary-output answer_line for a shared final-answer capability metric across arms. The default application
mode preserves historical scoring: Base = ANSWER line, Agent = accepted claims, Tools = recorded claims. Both output
dimensions, submission states and grounding are always reported separately; a missing output is never backfilled.
Grounding is pending until the same blinded coding is supplied for all arms. Scoring uses `eval/level2_grade.py`,
unchanged. Paired statistics are
within model (Base, Tools, Agent); cross-model rows are
descriptive and name the exact comparison model. Historical Level 2 arms (Claude Opus 5.5 C and D) are included as
named external references, never as "Claude Haiku".

Use --traces-out DIR --trace-text-limit 0 for a new complete-text export (the historical default remains 4000).
Use the existing level2_coding.py prepare/collect workflow. Give coders only
the prepared batches, not this grader's output or the trace index. Ledger capability counts are diagnostics, never
substitutes for independently coded evidence. Missing billed cost stays unknown; token-price estimates are separate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.compare_p5afdb_arms import newcombe_paired, wilson  # noqa: E402
from eval.level2_coding import validate as validate_coding  # noqa: E402
from eval.level2_grade import grade_level2, grade_level2_pair  # noqa: E402
from eval.level2_traces import render, session_key  # noqa: E402

RUNS = ROOT / "eval" / "runs"
MANIFEST = ROOT / "eval" / "tasks" / "p9level2_manifest.json"
# Capability diagnostic only: these prefixes do not establish grounding or correct task completion.
CAPABILITY_DEFS = {
    "TFA": ("tfa.",),
    "beat_segmentation": ("abp.", "pulse.rate", "pulse.beats"),
    "beat_envelope_and_cross_channel_quality": ("pulse.amplitude_change", "pulse.quality", "abp."),
    "beat_matching": ("beats.missing",),
    "sub_beat_alignment": ("delay.",),
    "BRS": ("brs.",),
}
SCORED_STATUSES = {"ok", "execution_failure"}
GROUNDING_FIELDS = (
    "decisive_regions_supported",
    "required_computations_supported",
    "evidence_validity",
    "verification_performed",
    "misleading_output_detected",
    "grounded",
    "grounded_correct",
    "correct_but_ungrounded",
)


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def ledger_defs(run_dir: Path, case_id: str) -> list[dict]:
    f = run_dir / "runtime" / case_id / "ledger.jsonl"
    return read(f)


def session_rows(tag: str, arm: str) -> tuple[Path | None, dict]:
    d = RUNS / f"p9level2-U-{arm}-{tag}"
    return (d if d.exists() else None), {r["case_id"]: r for r in read(d / "results.jsonl")}


def evidence_for(coding: Path | None, arm: str, case_id: str) -> list[dict] | None:
    if coding is None:
        return None
    key = session_key(arm, case_id)
    f = coding / f"{key}.json"
    if not f.exists():
        return None
    doc = json.loads(f.read_text())
    problems = validate_coding(doc, key)
    if problems:
        raise ValueError(f"Invalid grounding coding {f}: {'; '.join(problems)}")
    return doc["records"]


def grade_answer(answer, item: dict, evidence: list[dict] | None) -> dict:
    result = grade_level2(answer, item, evidence)
    result["grounding_status"] = "pending" if evidence is None else "coded"
    if evidence is None:
        result.update(dict.fromkeys(GROUNDING_FIELDS))
    return result


def primary_source(arm: str, primary_output: str) -> str:
    return "application" if primary_output == "application" and arm.endswith(("-Agent", "-Tools")) else "answer_line"


def grade_arm(tag, arm, items, coding, primary_output="application"):
    run_dir, rows = session_rows(tag, arm)
    out = {}
    for cid, item in items.items():
        r = rows.get(cid)
        if r is None:
            out[cid] = {"status": "not_run"}
            continue
        if r.get("dry_run"):
            out[cid] = {"status": "dry_run"}
            continue
        ev = evidence_for(coding, arm, cid)
        has_application = arm.endswith(("-Agent", "-Tools"))
        source = primary_source(arm, primary_output)
        answers = {"answer_line": r.get("pred"), "application": r.get("application_answer")}
        grades = {
            "answer_line": grade_answer(answers["answer_line"], item, ev),
            "application": grade_answer(answers["application"], item, ev) if has_application else None,
        }
        cap = item.get("specialist_capability")
        prefixes = CAPABILITY_DEFS.get(cap, ()) if cap else ()
        recs = ledger_defs(run_dir, cid) if run_dir else []
        ok_recs = [x for x in recs if x.get("source_kind") in ("computed", "derived") and x.get("status", "ok") == "ok"]
        out[cid] = {
            "status": "ok" if not r.get("error") else "execution_failure", "stop": r.get("stop"),
            "requested_model": r.get("model"), "served_models": r.get("served_models") or [],
            "source_commit": r.get("git_commit"), "worker_tag": r.get("worker_tag"),
            "primary": grades[source], **grades, "pred": answers[source], "primary_source": source,
            "answer_line_pred": answers["answer_line"], "application_pred": answers["application"] if has_application else None,
            "application_applicable": has_application,
            "specialist_tool_success": (any(str(x.get("definition_id", "")).startswith(prefixes) for x in ok_recs)
                                        if prefixes else None),
            "tools_used": Counter(str(x.get("tool")) for x in recs),
            "cost_usd_actual": r.get("cost_usd_actual"),
            "cost_usd_estimated": r.get("cost_usd_estimated", r.get("cost_usd")),
            "cost_usd_conservative": r.get("cost_usd_conservative"),
            "latency_s": r.get("latency_s"), "turns": r.get("turns"), "tool_calls": r.get("tool_calls"),
            "usage": r.get("usage") or {}, "finalization": r.get("finalization"),
        }  # fmt: skip
    return out


def summary(graded: dict, items: dict, ids: list[str], which: str = "primary") -> dict:
    attempted = [graded[i] for i in ids if graded[i].get("status") in SCORED_STATUSES]
    rows = [r for r in attempted if r.get(which) is not None]
    n = len(rows)
    correct = sum(bool(r[which]["correct"]) for r in rows)
    coded = [r for r in rows if r[which]["grounding_status"] == "coded"]
    grounded = sum(bool(r[which]["grounded_correct"]) for r in coded)
    fab = sum(bool(r[which].get("fabrication")) for r in rows)
    unnec = sum(bool(r[which].get("unnecessary_abstention")) for r in rows)
    no_ans = sum(r[which]["outcome"] in ("no_answer", "invalid") for r in rows)
    spec = [r for r in rows if r.get("specialist_tool_success") is not None]
    errs = [e["relative_error"] for r in rows for e in r[which]["field_errors"].values()
            if isinstance(e, dict) and e.get("relative_error") is not None]  # fmt: skip
    costs = {}
    for kind in ("actual", "estimated", "conservative"):
        values = [r[f"cost_usd_{kind}"] for r in rows if r[f"cost_usd_{kind}"] is not None]
        subtotal = round(sum(values), 6) if values else None
        costs.update(
            {
                f"cost_usd_{kind}_total": subtotal if len(values) == n else None,
                f"cost_usd_{kind}_known_subtotal": subtotal,
                f"cost_usd_{kind}_known_n": len(values),
                f"cost_usd_{kind}_missing_n": n - len(values),
            }
        )
    return {
        "output": which, "n": n, "n_requested": len(ids), "not_applicable": len(attempted) - n,
        "not_run": sum(graded[i]["status"] == "not_run" for i in ids),
        "dry_run_excluded": sum(graded[i]["status"] == "dry_run" for i in ids),
        "correct": None if attempted and not rows else correct, "correct_rate": correct / n if n else None,
        "correct_ci95": wilson(correct, n) if n else None,
        "grounding_status": "complete" if n and len(coded) == n else "partial" if coded else "pending",
        "grounding_coded_n": len(coded), "grounding_pending_n": n - len(coded),
        "grounded_correct": grounded if n and len(coded) == n else None,
        "grounded_correct_coded_subset": grounded if coded else None,
        "fabrication": fab, "unnecessary_abstention": unnec, "no_or_invalid_answer": no_ans,
        "execution_failures": sum(r["status"] == "execution_failure" for r in rows),
        "specialist_tool_success": [sum(bool(r["specialist_tool_success"]) for r in spec), len(spec)],
        "median_relative_error": round(statistics.median(errs), 4) if errs else None,
        **costs,
        "latency_s_median": median_present(rows, "latency_s"),
        "tool_calls_median": median_present(rows, "tool_calls"),
        "tokens": {k: sum(r["usage"].get(k, 0) for r in rows) for k in ("input_tokens", "output_tokens",
                                                                       "cache_creation_input_tokens", "cache_read_input_tokens")},
    }  # fmt: skip


def median_present(rows: list[dict], key: str):
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.median(values) if values else None


def paired(a: dict, b: dict, ids: list[str], which="primary", metric="correct") -> dict:
    requested_n = len(ids)
    ids = [
        i
        for i in ids
        if a[i].get("status") in SCORED_STATUSES
        and b[i].get("status") in SCORED_STATUSES
        and a[i].get(which) is not None
        and b[i].get(which) is not None
        and a[i][which].get(metric) is not None
        and b[i][which].get(metric) is not None
    ]
    x = {i: bool(a[i][which][metric]) for i in ids}
    y = {i: bool(b[i][which][metric]) for i in ids}
    both = sum(x[i] and y[i] for i in ids)
    only_a = sum(x[i] and not y[i] for i in ids)
    only_b = sum(y[i] and not x[i] for i in ids)
    d, lo, hi = newcombe_paired(both, only_a, only_b, len(ids) - both - only_a - only_b) if ids else (None, None, None)
    p = (binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0) if ids else None
    return {"metric": metric, "n": len(ids), "n_requested": requested_n, "excluded_n": requested_n - len(ids),
            "first_correct": sum(x.values()), "second_correct": sum(y.values()),
            "both": both, "neither": len(ids) - both - only_a - only_b,
            "only_first": only_a, "only_second": only_b, "diff_first_minus_second": d,
            "newcombe95": [lo, hi] if ids else None, "exact_mcnemar_p": p}  # fmt: skip


def comparison(graded: dict, first: str, second: str, groups: dict) -> dict:
    return {
        "first": first,
        "second": second,
        "numeric_correctness": {g: paired(graded[first], graded[second], ids) for g, ids in groups.items()},
        "grounded_correct_secondary": {
            g: paired(graded[first], graded[second], ids, metric="grounded_correct") for g, ids in groups.items()
        },
    }


def f5_summary(graded: dict, items: dict) -> dict:
    pairs = defaultdict(list)
    for item in items.values():
        if item.get("pair_id"):
            pairs[item["pair_id"]].append(item)
    scores, incomplete, not_run = [], 0, 0
    for members in pairs.values():
        completed = [graded[m["case_id"]]["status"] in SCORED_STATUSES for m in members]
        if not any(completed):
            not_run += 1
        elif not all(completed):
            incomplete += 1
        else:
            scores.append(grade_level2_pair([graded[m["case_id"]]["pred"] for m in members], members))
    return {
        "n": len(scores),
        "n_requested": len(pairs),
        "not_run": not_run,
        "incomplete": incomplete,
        **{
            k: sum(s[k] for s in scores)
            for k in ("correct_abstention", "fabrication", "unnecessary_abstention", "pair_correct")
        },
    }


def export_traces(tag: str, arms: list[str], out: Path, text_limit: int = 4000) -> dict:
    """Use the same trace renderer and opaque keys for Base, Tools and Agent coding."""
    cases = ROOT / "eval" / "cases"
    questions = {q["case_id"]: q["question"] for q in read(cases / "p9level2.questions.jsonl")}
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for arm in arms:
        run_dir, rows = session_rows(tag, arm)
        for cid, row in rows.items():
            if row.get("dry_run"):
                continue
            transcript = run_dir / "transcripts" / f"{cid}.jsonl"
            events = read(transcript)
            if not events:
                raise ValueError(f"Cannot code a missing or empty transcript: {transcript}")
            visible = json.loads((cases / "p9level2" / cid / "visible.json").read_text())
            key = session_key(arm, cid)
            # Temporary paths and run ids contain arm labels; hide those as well as the index.
            trace = render(events, questions[cid], visible, text_limit=text_limit).replace(arm, "session")
            (out / f"{key}.md").write_text(trace)
            index[key] = {"arm": arm, "case_id": cid, "transcript": str(transcript)}
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    return {"sessions": len(index), "directory": str(out), "text_limit": text_limit}


def main() -> None:
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--coding", type=Path)
    ap.add_argument(
        "--compare",
        nargs=2,
        action="append",
        default=[],
        metavar=("FIRST", "SECOND"),
        help="explicit paired comparison, including across models; may be repeated",
    )
    ap.add_argument(
        "--primary-output",
        choices=("answer_line", "application"),
        default="application",
        help="answer_line: shared final-answer metric; application: historical Base/Agent selection",
    )
    ap.add_argument(
        "--runs-root", type=Path, default=RUNS, help="read runs from this directory without copying or modifying them"
    )
    ap.add_argument("--traces-out", type=Path, help="export blinded traces for level2_coding.py prepare/collect")
    ap.add_argument("--trace-text-limit", type=int, default=4000, help="characters per trace text block; 0 = unlimited")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.trace_text_limit < 0:
        ap.error("--trace-text-limit must be nonnegative")
    if any(arm not in args.arms for pair in args.compare for arm in pair):
        ap.error("each --compare arm must also be listed in --arms")
    RUNS = args.runs_root
    man = json.loads(MANIFEST.read_text())["questions"]
    items = {q["case_id"]: q for q in man}
    graded = {arm: grade_arm(args.tag, arm, items, args.coding, args.primary_output) for arm in args.arms}
    groups = {
        "all": list(items), "answerable": [i for i, q in items.items() if not q["abstention_correct"]],
        **{f"category:{c}": [i for i, q in items.items() if q["expected_solvability"] == c] for c in ("generic", "specialist", "abstention")},
        **{f"signal:{k}": [i for i, q in items.items() if q["signal_kind"] == k] for k in ("real_signal", "controlled_perturbation")},
        **{f"family:{f}": [i for i, q in items.items() if q["family"] == f] for f in sorted({q["family"] for q in man})},
    }  # fmt: skip
    groups["answerable_without_low_confidence"] = [
        i for i in groups["answerable"] if items[i].get("ground_truth", {}).get("confidence") != "low"
    ]
    for capability in ("BRS", "TFA"):
        groups[f"capability:{capability}"] = [
            i for i in groups["answerable"] if items[i].get("specialist_capability") == capability
        ]
    result = {
        "tag": args.tag,
        "arms": args.arms,
        "primary_metric": "numeric_correctness",
        "primary_output": args.primary_output,
        "primary_sources": {arm: primary_source(arm, args.primary_output) for arm in args.arms},
        "runs_root": str(RUNS),
        "model_provenance": {},
        "grounding_method": "Level 2 blinded trace coding; pending until supplied for each session",
        "ledger_diagnostics": "tool success is not evidence of grounded correctness",
        "summaries": {},
        "output_summaries": {},
        "submission_states": {},
        "paired_within_model": {},
        "paired_requested": {},
        "pairs_f5": {},
    }
    if args.traces_out:
        result["coding_traces"] = export_traces(args.tag, args.arms, args.traces_out, args.trace_text_limit)
    for arm in args.arms:
        result["summaries"][arm] = {g: summary(graded[arm], items, ids) for g, ids in groups.items()}
        if args.primary_output == "application":
            result["summaries"][arm]["answer_line_secondary"] = summary(
                graded[arm], items, groups["answerable"], "answer_line"
            )
        result["output_summaries"][arm] = {
            which: {g: summary(graded[arm], items, ids, which) for g, ids in groups.items()}
            for which in ("answer_line", "application")
        }
        attempted = [r for r in graded[arm].values() if r["status"] in SCORED_STATUSES]
        result["model_provenance"][arm] = {
            "requested_models": sorted({r["requested_model"] for r in attempted if r["requested_model"]}),
            "served_models": sorted({m for r in attempted for m in r["served_models"]}),
            "source_commits": sorted({r["source_commit"] for r in attempted if r["source_commit"]}),
        }
        result["submission_states"][arm] = {
            "applicable": arm.endswith(("-Agent", "-Tools")),
            "n": len(attempted),
            "finalization": dict(Counter((r.get("finalization") or "absent") for r in attempted)),
        }
        result["pairs_f5"][arm] = f5_summary(graded[arm], items)
    for m in sorted({arm.rsplit("-", 1)[0] for arm in args.arms}):
        comparisons = {}
        for first, second in (("Agent", "Base"), ("Tools", "Base"), ("Agent", "Tools")):
            first, second = f"{m}-{first}", f"{m}-{second}"
            if first in graded and second in graded:
                comparisons[f"{first}_vs_{second}"] = comparison(graded, first, second, groups)
        if comparisons:
            result["paired_within_model"][m] = comparisons
    for first, second in args.compare:
        result["paired_requested"][f"{first}_vs_{second}"] = comparison(graded, first, second, groups)
    hist = ROOT / "eval/results/p9level2/analysis.json"
    if hist.exists():
        h = json.loads(hist.read_text())["complete"]["by_category"]
        result["historical_level2_claude_opus_5_5"] = {cat: {arm: (v[arm]["correct"], v[arm]["n"]) for arm in ("C", "D")}
                                                       for cat, v in h.items()}  # fmt: skip
    result["sessions"] = {arm: {cid: {k: v for k, v in g.items() if k != "tools_used"} for cid, g in graded[arm].items()}
                          for arm in args.arms}  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    for arm in args.arms:
        s = result["summaries"][arm]
        print(
            arm,
            {
                g: (s[g]["correct"], s[g]["n"])
                for g in ("answerable", "category:generic", "category:specialist", "category:abstention")
            },
        )
    print(json.dumps(result["paired_within_model"], default=str)[:600])


if __name__ == "__main__":
    main()
