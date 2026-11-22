"""Independent grader for the C_tools vs C_full evaluation (docs/signalmind/evaluation-freeze.md).

  .venv/bin/python eval/grade_signalmind.py --suite p10abp --tag main --out eval/results/signalmind/p10abp-analysis.json

Reads, per session, the harness result row and the runtime directory copied into the run
(`runtime/<case_id>/{answer.json, bundles.json, ledger.jsonl}`). Correctness is judged against the frozen answers file;
support is judged by this module's own replay of the ledger (NOT by the runtime verifier's decisions): a cited record
must exist, be computed/derived, carry the field's definition, cover the asked interval and agree with the claimed
value. "Accepted by the runtime" is never used as evidence of correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"
CASES = ROOT / "eval" / "cases"
ARMS = ("C_tools", "C_full")
FIELD_DEFS = {
    "sbp_mean": {"abp.sbp.beat_mean.v1", "abp.sbp.time_weighted_mean.v1"},
    "dbp_mean": {"abp.dbp.beat_mean.v1"},
    "sbp_change": {"abp.sbp.beat_mean_change.v1"},
    "pulse_rate_bpm": {"rate.from_period.v1"},
}
FIELD_UNITS = {"sbp_mean": "mmHg", "dbp_mean": "mmHg", "sbp_change": "mmHg", "pulse_rate_bpm": "bpm"}
FIELD_CHANNEL = {"pulse_rate_bpm": "PPG"}  # the quantity must come from this channel's evidence


def decimals(v) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return 0
    return max(0, -Decimal(repr(float(v))).normalize().as_tuple().exponent)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def load_ledger(p: Path) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(p):
        out[r["evidence_id"]] = r
        if r.get("ref"):
            out.setdefault(r["ref"], r)
    return out


# ---- independent support replay ---------------------------------------------------------------------------------------
def effective_interval(rec: dict | None, ledger: dict, depth: int = 0) -> tuple | None:
    """A record's requested interval; for derived records the unique interval of their ancestors (recursive)."""
    if rec is None or depth > 6:
        return None
    if rec.get("interval_requested"):
        return tuple(rec["interval_requested"])
    ivs = {effective_interval(ledger.get(p), ledger, depth + 1) for p in rec.get("parents", [])}
    ivs.discard(None)
    return ivs.pop() if len(ivs) == 1 else None


def supported(
    claim: dict, field: str, ledger: dict, intervals: list[list[float]], iv_tol: float = 1e-3
) -> tuple[bool, str]:
    refs = claim.get("evidence") or claim.get("evidence_refs") or []
    recs = [ledger.get(r) for r in refs]
    if not refs or any(r is None for r in recs):
        return False, "missing_or_unknown_evidence"
    want = FIELD_DEFS[field]
    cand = [r for r in recs if r.get("definition_id") in want and r.get("source_kind") in ("computed", "derived")
            and r.get("status", "ok") == "ok" and isinstance(r.get("value"), (int, float))]  # fmt: skip
    if not cand:
        return False, "no_computed_record_of_the_right_definition"
    rec = cand[-1]
    unit = FIELD_UNITS[field]
    if rec.get("unit") != unit or claim.get("unit") != unit:
        return False, "unit"
    if field in FIELD_CHANNEL:
        chans = set(rec.get("channels_effective") or [])
        for p in rec.get("parents", []):
            chans |= set((ledger.get(p) or {}).get("channels_effective") or [])
        if FIELD_CHANNEL[field] not in chans:
            return False, f"evidence_not_from_{FIELD_CHANNEL[field]}"
    tol = max(0.5 * 10 ** (-decimals(rec["value"])), 0.5 * 10 ** (-decimals(claim.get("value"))),
              float((rec.get("resolution") or {}).get("value") or 0.0))  # fmt: skip
    if (
        not isinstance(claim.get("value"), (int, float))
        or abs(float(claim["value"]) - float(rec["value"])) > tol + 1e-9
    ):
        return False, "value_differs_from_evidence"
    if field == "sbp_change":
        parents = [ledger.get(p) for p in rec.get("parents", [])]
        if len(parents) != 2 or any(p is None for p in parents):
            return False, "change_without_two_parents"
        ivs = [effective_interval(p, ledger) for p in parents]
        if None in ivs or sorted(ivs) != sorted(tuple(i) for i in intervals):
            return False, "parent_intervals_differ_from_question"
        return True, "ok"
    iv = effective_interval(rec, ledger)
    if iv is None or abs(iv[0] - intervals[0][0]) > iv_tol or abs(iv[1] - intervals[0][1]) > iv_tol:
        return False, "interval_differs_from_question"
    return True, "ok"


def overprecise(claim: dict, field: str, ledger: dict) -> bool:
    """The claimed decimals exceed what the cited evidence's known resolution supports (first significant digit)."""
    refs = claim.get("evidence") or claim.get("evidence_refs") or []
    recs = [ledger.get(r) for r in refs if ledger.get(r)]
    cand = [r for r in recs if r.get("definition_id") in FIELD_DEFS[field] and isinstance(r.get("value"), (int, float))]
    if not cand or not isinstance(claim.get("value"), (int, float)):
        return False
    res = cand[-1].get("resolution") or {}
    if not (res.get("known") and res.get("value") and float(res["value"]) > 0):
        return decimals(claim["value"]) > decimals(cand[-1]["value"])
    allowed = max(0, -math.floor(math.log10(float(res["value"]))))
    return decimals(claim["value"]) > min(allowed, max(decimals(cand[-1]["value"]), allowed))


def tool_call_order(transcript: Path) -> list[str]:
    """Names of the model's tool calls in execution order, from the harness transcript. Calls whose result was an
    error are listed as `<name>!` (they produce no evidence, so they must not shift evidence positions)."""
    calls: list[tuple[str, str]] = []
    errored: set[str] = set()
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        content = e.get("message", {}).get("content", []) if isinstance(e.get("message"), dict) else []
        if e.get("type") == "assistant":
            for c in content or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    calls.append((c.get("id", ""), c.get("name", "?").removeprefix("mcp__tsmcp__")))
        elif e.get("type") == "user":
            for c in content or []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                    errored.add(c.get("tool_use_id", ""))
    return [name + ("!" if cid in errored else "") for cid, name in calls]


EVIDENCE_TOOLS = ("ts_measure", "ts_zoom", "ts_find_events")


def evidence_position(rec: dict | None, ledger: dict, order: list[str]) -> int | None:
    """Transcript position at which a record was created: computed records via their k-th evidence-producing call,
    derived records via the k-th ts_derive call (derived records are created in call order)."""
    if rec is None:
        return None
    if rec.get("source_kind") == "derived":
        derived_seqs = sorted(
            r["seq"]
            for r in {v["evidence_id"]: v for v in ledger.values()}.values()
            if r.get("source_kind") == "derived"
        )
        k = derived_seqs.index(rec["seq"])
        derive_calls = [i for i, n in enumerate(order) if n == "ts_derive"]
        return derive_calls[k] if k < len(derive_calls) else None
    cid = rec.get("tool_call_id") or ""
    if not cid.startswith("call"):
        return None
    k = int(cid[4:]) - 1
    producing = [i for i, n in enumerate(order) if n in EVIDENCE_TOOLS]
    return producing[k] if k < len(producing) else None


ANSWERED = {"accepted", "partial", "recorded", "accepted_partial", "accepted_then_new_evidence", "revision_rejected"}
NO_SUBMISSION = {"none", "incomplete", None, "absent"}


def boundary_evidence(ledger: dict, item: dict) -> bool:
    """Did the session hold evidence of the dropout boundary (flat events, beat exclusions, no-beat measurement) within
    tolerance of the true end? Reported as behaviour; not part of correctness."""
    end = item.get("true_dropout_end_s")
    tol = item.get("interval_tolerance_s", 2.0)
    channel = (item.get("control") or {}).get("channel")
    for r in {v["evidence_id"]: v for v in ledger.values()}.values():
        if channel and channel not in (r.get("channels_effective") or []):
            continue
        ends = [b for _, b in (r.get("intervals_excluded") or [])]
        if str(r.get("quantity", "")).startswith("events."):
            ends += [float(e["t"]) + float(e.get("dur") or 0) for e in (r.get("parameters") or {}).get("candidates", [])
                     if e.get("t") is not None]  # fmt: skip
        if r.get("status") == "limitation" and r.get("interval_requested"):
            ends.append(float(r["interval_requested"][1]))
        if any(abs(e - end) <= tol for e in ends):
            return True
    return False


def claims_of(answer: dict | None, mode: str) -> list[dict]:
    if not answer:
        return []
    if mode == "log":  # the model's own status is `submitted_status`; `status` is the renderer's unverified label
        return [
            {**c, "status": c.get("submitted_status") or c.get("status")} for c in answer.get("submitted_claims", [])
        ]
    return list(answer.get("accepted_claims", [])) + list(answer.get("unresolved", []))


def bundle_claims(bundle: dict) -> list[dict]:
    return [{**c, "evidence": c.get("evidence_refs", [])} for c in bundle.get("claims", [])]


def grade_claims(claims: list[dict], limitations: list, item: dict, ledger: dict) -> dict:
    expected, tol = item["expected"], item["tolerance"]
    limitations = [lim for lim in (limitations or []) if isinstance(lim, dict)]
    fields = list(expected)
    per = {}
    for f in fields:
        c = next((c for c in claims if c.get("quantity") == f), None)
        lim = next((lim for lim in limitations if lim.get("quantity") == f), None)
        numeric = c is not None and isinstance(c.get("value"), (int, float)) and not isinstance(c.get("value"), bool)
        abstained = (c is not None and not numeric and c.get("status") in ("insufficient_evidence", "unresolved_conflict",
                                                                            "incomplete_execution")) or lim is not None  # fmt: skip
        if expected[f] is None:  # control: the right answer is an explicit limitation, not a number
            correct = abstained and not numeric
            sup, why = (True, "control") if correct else (False, "number_on_control")
            err = None
        else:
            err = abs(float(c["value"]) - expected[f]) if numeric else None
            correct = numeric and err <= tol[f]
            sup, why = (supported(c, f, ledger, item["intervals"], item.get("interval_tolerance_s", 1e-3))
                        if numeric else (False, "no_number"))  # fmt: skip
            if sup and overprecise(c, f, ledger):
                if c.get("status") == "approximate" and c.get("precision_decimals") is not None:
                    why = "approximate_labelled_due_to_precision"  # qualified, not an unsupported precise claim
                else:
                    sup, why = False, "overprecise_for_evidence_resolution"
        per[f] = {"claimed": c.get("value") if c else None, "status": (c or {}).get("status") or (lim or {}).get("status"),
                  "numeric": numeric, "abstained": abstained, "correct": bool(correct), "supported": bool(sup), "why": why,
                  "abs_error": err, "unsupported_precise": numeric and not sup,
                  "decimals": decimals(c.get("value")) if numeric else None}  # fmt: skip
    extra: dict = {}
    if item.get("true_dropout_end_s") is not None:  # gap items: window correctness and boundary evidence, separately
        claimed_iv = next((c["interval"] for c in claims
                           if isinstance(c.get("interval"), (list, tuple)) and len(c["interval"]) == 2), None)  # fmt: skip
        tol = item.get("interval_tolerance_s", 2.0)
        extra["window_claimed"] = claimed_iv
        extra["window_correct"] = bool(claimed_iv and abs(claimed_iv[0] - item["intervals"][0][0]) <= tol
                                       and abs(claimed_iv[1] - item["intervals"][0][1]) <= tol)  # fmt: skip
        extra["boundary_evidence"] = boundary_evidence(ledger, item)
    return {**extra, "fields": per, "correct": all(p["correct"] for p in per.values()),
            "grounded_correct": all(p["correct"] and p["supported"] for p in per.values()),
            "coverage": sum(p["supported"] and p["correct"] for p in per.values()) / len(fields),
            "unsupported_precise": any(p["unsupported_precise"] for p in per.values()),
            "incorrect_abstention": any(p["abstained"] and expected[f] is not None for f, p in per.items()),
            "max_abs_error": max((p["abs_error"] for p in per.values() if p["abs_error"] is not None), default=None)}  # fmt: skip


def grade_session(row: dict | None, rt: Path, item: dict, arm: str, transcript: Path | None = None) -> dict:
    mode = "log" if arm == "C_tools" else "verify"
    state = json.loads((rt / "answer.json").read_text()) if (rt / "answer.json").exists() else None
    bundles = json.loads((rt / "bundles.json").read_text()) if (rt / "bundles.json").exists() else []
    ledger = load_ledger(rt / "ledger.jsonl")
    out = {"arm": arm, "case_id": item["case_id"], "kind": item["kind"], "category": item["category"],
           "status": "ok" if row else "no_result", "finalization": (state or {}).get("finalization", "absent"),
           "submissions": len(bundles), "n_evidence": len({r["evidence_id"] for r in ledger.values()}),
           "cost_usd": max((row or {}).get("cost_usd") or 0, (row or {}).get("cc_cost_usd") or 0),
           "latency_s": (row or {}).get("latency_s"), "turns": (row or {}).get("turns"),
           "tool_calls": (len(tc) if isinstance(tc := (row or {}).get("tool_calls"), list) else tc),
        "tools_used": dict(Counter(t.removeprefix("mcp__tsmcp__") for t in tc)) if isinstance(tc, list) else {}, "usage": (row or {}).get("usage") or {}}  # fmt: skip
    if state is None or state.get("finalization") in NO_SUBMISSION or state.get("answer") is None:
        out.update(final=None, incomplete_execution=True, grounded_correct=False, correct=False)
        return out
    out["incomplete_execution"] = False
    final = grade_claims(claims_of(state["answer"], mode), state["answer"].get("limitations", []), item, ledger)
    out["final"] = final
    out["correct"], out["grounded_correct"] = final["correct"], final["grounded_correct"]
    first = next((b for b in bundles if isinstance(b.get("bundle"), dict) and "claims" in b["bundle"]), None)
    if first:
        out["initial"] = grade_claims(
            bundle_claims(first["bundle"]), first["bundle"].get("limitations", []), item, ledger
        )
    if mode == "verify":
        reports = [b["report"] for b in bundles]
        rejected = [
            b
            for b in bundles
            if b["report"].get("ok") is False
            and any(d["code"] != "malformed_bundle" for d in b["report"].get("deficiencies", []))
        ]
        out["syntax_rejections"] = sum(1 for b in bundles if b["report"].get("ok") is False) - len(rejected)
        out["runtime_rejections"] = len(rejected)
        out["runtime_deficiency_codes"] = dict(
            Counter(d["code"] for b in rejected for d in b["report"]["deficiencies"])
        )
        # verifier false rejection: a rejected bundle whose claims this grader finds supported and correct
        out["verifier_false_rejections"] = sum(
            1 for b in rejected if isinstance(b.get("bundle"), dict) and "claims" in b["bundle"]
            and grade_claims(bundle_claims(b["bundle"]), b["bundle"].get("limitations", []), item, ledger)["grounded_correct"]
        )  # fmt: skip
        # repair with new evidence: first submission semantically rejected, final accepted bundle cites evidence
        # created after that submission (positions from the transcript's tool-call order, not from the runtime)
        accepted_idx = [i for i, b in enumerate(bundles) if b["report"].get("ok") is True]
        first_semantic = next((i for i, b in enumerate(bundles) if b in rejected), None)
        out["repaired_with_new_evidence"] = False
        out["blocked_without_repair"] = (
            bool(rejected) and not accepted_idx and state.get("finalization") not in ANSWERED
        )
        out["repair_new_evidence"] = out["repair_recalculation"] = out["repair_reporting_only"] = False
        if first_semantic is not None and accepted_idx and accepted_idx[-1] > first_semantic:
            kinds = set()
            if transcript is not None:
                order = tool_call_order(transcript)
                submits = [i for i, name in enumerate(order) if name == "ts_submit"]
                if len(submits) > first_semantic:
                    cut = submits[first_semantic]
                    cited = {
                        ref for c in bundles[accepted_idx[-1]]["bundle"]["claims"] for ref in c.get("evidence_refs", [])
                    }
                    for ref in cited:
                        rec = ledger.get(ref)
                        pos = evidence_position(rec, ledger, order)
                        if rec is None or pos is None or pos <= cut:
                            continue
                        kinds.add("new_evidence" if rec.get("source_kind") == "computed" else "recalculation")
            out["repair_new_evidence"] = "new_evidence" in kinds
            out["repair_recalculation"] = "recalculation" in kinds and "new_evidence" not in kinds
            out["repair_reporting_only"] = not kinds
            out["repair_new_window_or_channel"] = False
            if out["repair_new_evidence"] and transcript is not None:
                before = {(tuple(ledger[r].get("interval_requested") or ()), tuple(ledger[r].get("channels_effective") or ()))
                          for c in bundles[0]["bundle"].get("claims", []) for r in c.get("evidence_refs", []) if r in ledger}  # fmt: skip
                for ref in cited:
                    rec = ledger.get(ref)
                    pos = evidence_position(rec, ledger, order)
                    if rec and rec.get("source_kind") == "computed" and pos is not None and pos > cut:
                        key = (tuple(rec.get("interval_requested") or ()), tuple(rec.get("channels_effective") or ()))
                        if key not in before:
                            out["repair_new_window_or_channel"] = True
            out["repaired_with_new_evidence"] = out["repair_new_evidence"]  # v0.1 key
        # requalification (v0.1 key): any repair without new signal evidence
        out["repaired_by_requalification"] = bool(
            first_semantic is not None
            and accepted_idx
            and accepted_idx[-1] > first_semantic
            and not out["repair_new_evidence"]
        )
        out["deterministic_output_corrections"] = 0  # the verifier never alters values (by construction)
        out["runtime_reports"] = len(reports)
    return out


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def summarize(sessions: list[dict], items: dict) -> dict:
    def block(rows):
        n = len(rows)
        answerable = [
            r
            for r in rows
            if items[r["case_id"]]["expected"] and any(v is not None for v in items[r["case_id"]]["expected"].values())
        ]
        controls = [r for r in rows if r not in answerable]
        gc = sum(r["grounded_correct"] for r in answerable)
        errs = [
            r["final"]["max_abs_error"]
            for r in answerable
            if r.get("final") and r["final"]["max_abs_error"] is not None
        ]
        return {
            "sessions": n, "answerable": len(answerable), "controls": len(controls),
            "grounded_correct": gc, "grounded_correct_ci95": wilson(gc, len(answerable)),
            "correct_any_support": sum(r["correct"] for r in answerable),
            "controls_correct": sum(r["correct"] for r in controls),
            "unsupported_precise_claims": sum(bool(r.get("final") and r["final"]["unsupported_precise"]) for r in rows),
            "incorrect_abstentions": sum(bool(r.get("final") and r["final"]["incorrect_abstention"]) for r in answerable),
            "incomplete_execution": sum(r["incomplete_execution"] for r in rows),
            "gap_window_correct": sum(bool(r.get("final") and r["final"].get("window_correct")) for r in rows),
            "gap_boundary_evidence": sum(bool(r.get("final") and r["final"].get("boundary_evidence")) for r in rows),
            "partial_coverage_mean": round(statistics.mean([r["final"]["coverage"] for r in answerable if r.get("final")] or [0]), 3),
            "abs_error_median": round(statistics.median(errs), 3) if errs else None,
            "abs_error_p90": round(float(sorted(errs)[int(0.9 * (len(errs) - 1))]), 3) if errs else None,
            "initial_grounded_correct": sum(bool(r.get("initial") and r["initial"]["grounded_correct"]) for r in answerable),
            "initial_correct": sum(bool(r.get("initial") and r["initial"]["correct"]) for r in answerable),
            "verifier_false_rejections": sum(r.get("verifier_false_rejections", 0) for r in rows),
            "runtime_rejections": sum(r.get("runtime_rejections", 0) for r in rows),
            "repaired_with_new_evidence": sum(bool(r.get("repaired_with_new_evidence")) for r in rows),
            "repaired_by_requalification": sum(bool(r.get("repaired_by_requalification")) for r in rows),
            "repair_new_evidence": sum(bool(r.get("repair_new_evidence")) for r in rows),
            "repair_new_window_or_channel": sum(bool(r.get("repair_new_window_or_channel")) for r in rows),
            "repair_recalculation": sum(bool(r.get("repair_recalculation")) for r in rows),
            "repair_reporting_only": sum(bool(r.get("repair_reporting_only")) for r in rows),
            "blocked_without_repair": sum(bool(r.get("blocked_without_repair")) for r in rows),
            "deterministic_output_corrections": 0,
            "finalization": dict(Counter(r["finalization"] for r in rows)),
            "deficiency_codes": dict(sum((Counter(r.get("runtime_deficiency_codes", {})) for r in rows), Counter())),
            "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 4),
            "cost_usd_median": round(statistics.median([r["cost_usd"] for r in rows if r["cost_usd"]] or [0]), 4),
            "latency_s_median": round(statistics.median([r["latency_s"] for r in rows if r["latency_s"]] or [0]), 1),
            "tool_calls_median": statistics.median([r["tool_calls"] for r in rows if r["tool_calls"] is not None] or [0]),
            "tokens": {k: sum((r["usage"].get(k) or 0) for r in rows) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")},
        }  # fmt: skip

    out = {"by_arm": {arm: block([s for s in sessions if s["arm"] == arm]) for arm in ARMS}}
    for cat in sorted({items[s["case_id"]]["category"] for s in sessions}):
        out.setdefault("by_category", {})[cat] = {
            arm: block([s for s in sessions if s["arm"] == arm and items[s["case_id"]]["category"] == cat])
            for arm in ARMS
        }
    by_case = defaultdict(dict)
    for s in sessions:
        by_case[s["case_id"]][s["arm"]] = s
    pairs = [v for v in by_case.values() if len(v) == 2]
    out["paired"] = {
        "items": len(pairs),
        "grounded_correct_tools_only": sum(v["C_tools"]["grounded_correct"] and not v["C_full"]["grounded_correct"] for v in pairs),
        "grounded_correct_full_only": sum(v["C_full"]["grounded_correct"] and not v["C_tools"]["grounded_correct"] for v in pairs),
        "newly_introduced_errors_in_full": sum(v["C_tools"]["correct"] and not v["C_full"]["correct"] for v in pairs),
        "errors_removed_in_full": sum(v["C_full"]["correct"] and not v["C_tools"]["correct"] for v in pairs),
    }  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", required=True, nargs="+", help="one or more run tags (a resumed run has several)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    answers = {a["case_id"]: a for a in read_jsonl(CASES / f"{args.suite}.answers.jsonl")}
    manifest = {
        m["case_id"]: m for m in json.loads((ROOT / "eval/tasks" / f"{args.suite}_manifest.json").read_text())["items"]
    }
    items = {cid: {**manifest[cid], **answers[cid]} for cid in answers}
    sessions = []
    chosen: dict[tuple[str, str], dict] = {}
    for arm in ARMS:
        for d in sorted(d for t in args.tag for d in RUNS.glob(f"{args.suite}-C-*-{t}-{arm}-w*")):
            rows = {r["case_id"]: r for r in read_jsonl(d / "results.jsonl")}
            for rt in sorted((d / "runtime").glob("*")) if (d / "runtime").exists() else []:
                cid = rt.name
                if cid not in items or not (rt / "answer.json").exists():
                    continue  # never started (e.g. refused before the model was reached): no session to grade
                t = d / "transcripts" / f"{cid}.jsonl"
                graded = grade_session(rows.get(cid), rt, items[cid], arm, t if t.exists() else None)
                graded["run_dir"] = d.name
                key = (arm, cid)
                if key in chosen and chosen[key]["status"] == "ok" and graded["status"] != "ok":
                    continue  # keep the session that produced a result row
                chosen[key] = graded
    sessions = list(chosen.values())
    result = {"suite": args.suite, "tag": args.tag, "summary": summarize(sessions, items), "sessions": sessions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["summary"]["by_arm"], indent=1))
    print(json.dumps(result["summary"]["paired"], indent=1))


if __name__ == "__main__":
    main()
