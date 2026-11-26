"""PR-01: observational repair funnel for dataset runs. Never calls a planner, scorer or numerical operator."""

import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from .common import array_hash, changed, digest, locked, read, sha, write
from .contracts import CandidateRepairOutcome, RepairAttempt
from .engine import load_arrays

FILES = ("attempts.jsonl", "candidates.jsonl", "api_calls.jsonl")


def optional(path, default):
    return read(path) if path.exists() else default


def json_rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def precheck_codes(pre):
    status, reason = pre.get("status"), pre.get("reason")
    if reason == "stale_parent_context":
        return {"stale_evidence"}
    if reason == "near_native_qrs" or status == "possible_physiology":
        return {"protected_landmark_conflict"}
    if reason == "core_or_anchor_unsupported":
        return {"anchor_unsupported"}
    if reason == "detector_boundary_or_empty":
        return {"detector_unsupported"}
    if status == "insufficient_support":
        return {"need_evidence"}
    return set()


def model_calls(root):
    """Charge each observed transport attempt once. Batch costs are not divisible into per-hit costs."""
    calls = {}
    for event in json_rows(root / "events.jsonl"):
        if event["type"] not in ("model_call_started", "model_call_failed", "model_call_completed"):
            continue
        num = event["call_number"]
        item = calls.setdefault(
            num,
            dict(call_number=num, request_id=None, status="started", elapsed_s=None, usage=None, estimated_usd=None),
        )
        if event["type"] == "model_call_completed":
            item.update(
                status="returned",
                request_id=event["request_id"],
                elapsed_s=event.get("elapsed_s"),
                usage=event.get("usage"),
                estimated_usd=event.get("cost_usd"),
            )
        elif event["type"] == "model_call_failed":
            item.update(status="failed", elapsed_s=event.get("elapsed_s"), error_type=event.get("error_type"))
    return calls


def attempt(root, folder, trial, ids, phase, round_no, patches, receipts):
    plan_id = trial.get("plan_id")
    plan = read(folder / "plans" / (plan_id + ".json")) if plan_id else {}
    patch = patches.get(plan_id, {})
    aid = trial.get("candidate") or patch.get("candidate_version_id")
    artifact = read(root / "artifacts" / aid / "artifact.json") if aid else {}
    generated = bool(aid or trial.get("candidate_hash"))
    hard = trial.get("hard_gate", {})
    reports = list(trial.get("ranges", {}).values())
    refs = {d["receipt"] for d in reports if d.get("receipt")}
    refs.update(r for r in (trial.get("receipt"), trial.get("plan_receipt")) if r and r in receipts)
    refs.update(patch.get("receipt", {}).get("evidence_refs", []))
    refs.update(d["prior_receipt"] for d in reports if d.get("prior_receipt"))
    for ref in list(refs):
        for _, data, _ in receipts.get(ref, {}).get("evidence", []):
            for field in ("blind_observation", "diagnostic_receipt"):
                if data.get(field, {}).get("receipt"):
                    refs.add(data[field]["receipt"])
    why = trial.get("reason")
    codes = (
        set(trial.get("reason_codes", []))
        | precheck_codes(trial.get("precheck", {}))
        | precheck_codes(trial.get("initial_precheck") or {})
        | set(trial.get("repairability", {}).get("reason_codes", []))
    )
    verdicts = {d.get("decision") for d in reports}
    has_review = bool(trial.get("receipt") or verdicts - {"pending", None})
    gate = hard.get("status", "unknown") if generated else "not_run"
    review = "not_run"
    if verdicts & {"need_evidence", "pending"}:
        codes.add("need_evidence")
    if "rollback" in verdicts:
        codes.add("review_rejected")
    for evidence in hard.get("interpolations", []):
        codes.update(precheck_codes(evidence))
    if patch:
        review = "accepted"
    elif (
        "need_evidence" in verdicts
        or why == "need_evidence"
        or trial.get("review_decision") == "need_evidence"
        or (trial.get("status") == "pending" and trial.get("receipt"))
    ):
        review = "need_evidence"
        codes.add("need_evidence")
    elif has_review and (trial.get("status") == "rollback" or "rollback" in verdicts):
        review = "rejected"
        codes.add("review_rejected")
    elif has_review or reports:
        review = "incomplete"
        codes.add("need_evidence")
    if gate == "rejected":
        codes.add("hard_gate_failed")
    if why == "write_view_budget" or trial.get("view_budget_skipped"):
        codes.add("budget_exhausted")
    if why == "need_evidence":
        codes.add("need_evidence")
    if trial.get("status") == "rollback" and not generated:
        codes.add("candidate_numerical_failure")
    if why == "write_review_unavailable":
        codes.add("need_evidence")
    if trial.get("status") == "ineligible" and not codes:
        codes.add(
            "legacy_preconditions_undifferentiated"
            if why == "unsupported_parameters_support_or_stale_parent"
            else "method_not_applicable"
        )
    op = trial.get("method") or plan.get("tool", {}).get("tool_id", "unknown")
    is_mask = op in ("mask_invalid", "mark_invalid")
    numerical_changes = artifact.get("stats", {}).get("parent_modified_samples")
    if patch and numerical_changes == 0 and not is_mask:
        is_mask = any(
            z["op"] in ("mask_invalid", "mark_invalid") for z in plan["tool"]["parameters"].get("operations", [])
        )
    outcome = (
        ("mark_invalid" if is_mask else "waveform_repair" if numerical_changes else "preserve")
        if patch
        else ("waveform_trial" if generated and not is_mask else "preserve")
    )
    region = plan.get("write_ranges", [])
    if not region and "a" in trial:
        region = [dict(block_id=trial["case"], start_sample=trial["a"], end_sample=trial["b"])]
    return RepairAttempt(
        attempt_id=digest([folder.name, phase, round_no, trial["case"], plan_id, ids, op]),
        record_id=folder.name,
        block_id=trial["case"],
        candidate_ids=ids,
        phase=phase,
        round=round_no,
        operation=op,
        parent_version_id=trial.get("parent") or plan.get("parent_version_id"),
        candidate_version_id=aid,
        candidate_content_hash=trial.get("candidate_hash") or artifact.get("content_hash"),
        candidate_revision=plan.get("candidate_revision") or trial.get("candidate_revision"),
        plan_id=plan_id,
        parameters=plan.get("tool", {}).get("parameters", trial.get("parameters", {})),
        read_ranges=plan.get("read_ranges", []),
        write_ranges=region,
        receipt_ids=sorted(refs),
        request_ids=sorted({receipts[k]["request_id"] for k in refs if k in receipts}),
        patch_id=patch.get("idempotency_key"),
        applicability=trial.get("precheck", {}).get(
            "status", "evaluated" if generated else trial.get("status", "unknown")
        ),
        review_evidence=reports,
        operator_invoked=generated or "candidate_numerical_failure" in codes,
        candidate_generated=generated,
        hard_gate=gate,
        review=review,
        commit="committed" if patch else "not_committed",
        outcome=outcome,
        reason_codes=sorted(codes),
        failed_checks=sorted(k for k, v in hard.get("checks", {}).items() if not v),
        detail=why,
        numerical_elapsed_s=artifact.get("elapsed_s", trial.get("numerical_elapsed_s")),
        candidate_changed_samples=artifact.get("stats", {}).get("parent_modified_samples"),
    )


def record_rows(root, rid, cases, waves, receipts, frozen, budget_pending):
    folder = root / "records" / rid
    discovery = read(folder / "discovery.json")
    rescan = read(folder / "rescan.json")
    patches = {p["receipt"]["plan_id"]: p for f in (folder / "patches").glob("*.json") for p in [read(f)]}
    initial = {h["candidate_id"]: h for h in discovery["candidates"]}
    candidates = dict(initial)
    mid = optional(folder / "candidate-revision-2.json", {})
    for rows in (mid.get("candidates", []), rescan["candidates"]):
        candidates.update((h["candidate_id"], h) for h in rows)
    open_items = read(folder / "report.json")["unresolved"]
    unresolved = {r["hit"] for r in open_items if r.get("hit")}
    pending_pages = {r["group"] for r in open_items if r.get("group")}
    changes = {}
    for revision in rescan["history"]:
        for state, ids in revision["changes"].items():
            changes.update((i, state) for i in ids)
    by_hit = defaultdict(list)
    decisions = defaultdict(list)
    total_attempts = []
    for nr, suffix in enumerate(("", "-rescan"), 1):
        revision = mid.get("id") if suffix else frozen
        info = {h["id"]: h for h in read(folder / ("release-dispositions.json" if suffix else "hit-dispositions.json"))}
        actual = optional(folder / ("decision-trace" + suffix + ".json"), {})
        plans = optional(folder / ("repair-plans" + suffix + ".json"), {})
        for group in optional(folder / ("groups" + suffix + ".json"), []):
            for page in group["pages"]:
                pid = page["id"]
                policy = plans.get(pid, {})
                for hid in group["member_ids"][page["start"] : page["end"]]:
                    h = actual.get(hid, info.get(hid, {}))
                    if h.get("policy_id") != pid:
                        h = {}
                    if pid in pending_pages:
                        unresolved.add(hid)
                    codes = precheck_codes(h.get("applicability", {}))
                    if not policy and (
                        pid in budget_pending or group["id"] in budget_pending or group["route"] == "unrouted"
                    ):
                        codes.add("budget_exhausted")
                    if h.get("reason") == "protected_landmark":
                        codes.add("protected_landmark_conflict")
                    if h.get("support_status") == "insufficient_support":
                        codes.add("need_evidence")
                    if not policy and h.get("decision_source") != "local_rule":
                        codes.add("plan_unavailable")
                    if policy.get("wave") == "preserve":
                        codes.add("policy_preserve")
                    if (
                        policy.get("wave") not in ("preserve", None)
                        and h.get("wave_requested") == "preserve"
                        and not codes
                        and "decision_reasons" not in h
                    ):
                        codes.add("policy_condition_or_support")
                    codes.update(h.get("decision_reasons", []))
                    decisions[hid].append(
                        dict(
                            round=nr,
                            revision=revision,
                            policy_id=pid,
                            policy_action=policy.get("wave"),
                            selected_action=h.get("wave_requested"),
                            applicability=h.get("applicability", {}).get("status"),
                            repairability=h.get("repairability"),
                            parameters=policy.get("parameters"),
                            conditions=policy.get("wave_if"),
                            reason_codes=sorted(codes),
                            reason=policy.get("reason"),
                            receipt=policy.get("receipt"),
                            diagnosis_receipt=policy.get("diagnosis_receipt"),
                            request_ids=sorted(
                                {
                                    receipts[k]["request_id"]
                                    for k in (policy.get("receipt"), policy.get("diagnosis_receipt"))
                                    if k in receipts
                                }
                            ),
                        )
                    )
        operations = []
        for kind in ("spike", "filter"):
            for trial in optional(folder / (kind + "-trials" + suffix + ".json"), []):
                branches = trial.get("comparison", {}).get("alternatives") or [trial]
                operations.extend((branch, [trial["hit"]], "fine") for branch in branches)
        for gate_name in ["coarse-gate", "fine-gate"] if nr == 1 else ["fine-gate-rescan"]:
            for cid, entry in optional(folder / (gate_name + ".json"), {}).items():
                plan = read(folder / "plans" / (entry["plan_id"] + ".json"))
                ids = sorted(
                    {
                        i
                        for op in plan["tool"]["parameters"]["operations"]
                        for i in op.get("parameters", {}).get("member_hits", [])
                    }
                )
                operations.append((dict(entry, case=cid), ids, "coarse" if gate_name == "coarse-gate" else "fine"))
        for trial, ids, phase in operations:
            row = attempt(root, folder, trial, ids, phase, nr, patches, receipts)
            total_attempts.append(row)
            for hid in ids:
                by_hit[hid].append(row)
    for row in total_attempts:
        yield "attempts", row.model_dump()
    # Read each block once. Neighbouring hits can overlap a commit; observed deltas alone never imply causal credit.
    per_block = defaultdict(list)
    for hid, c in candidates.items():
        per_block[c["region"]["block_id"]].append((hid, c))
    for cid, members in per_block.items():
        w = waves[cid]
        case = cases[cid]
        _, _, _, initial_valid, _ = load_arrays(root, case)
        art = read(root / "artifacts" / w["artifact_id"] / "artifact.json")
        raw, t, y, valid, estimated = load_arrays(root, case, art)
        if (
            sha(root / "public" / cid / "input.npz") != case["input_sha256"]
            or array_hash(y, t, valid, estimated) != w["content_hash"]
        ):
            raise ValueError(f"Released or source waveform changed for {cid}")
        modified = changed(raw, y)
        disabled = initial_valid & ~valid
        for hid, c in members:
            attempts = by_hit[hid]
            decision = decisions[hid]
            codes = {code for z in attempts for code in z.reason_codes} | {
                code for z in decision for code in z["reason_codes"]
            }
            rescan_status = changes.get(hid, "not_recorded")
            if rescan["status"] == "pending":
                rescan_status = "pending"
                codes.add("need_evidence")
            if any(r.get("hit") == hid and r["stage"] == "final_rescan" for r in open_items):
                codes.add("rescan_unresolved")
            if not attempts and not codes:
                codes.add("no_numerical_action" if decision else "not_planned")
            region = c["region"]
            a, b = region["start_sample"], region["end_sample"]
            yield (
                "candidates",
                CandidateRepairOutcome(
                    candidate_id=hid,
                    record_id=rid,
                    region=region,
                    detector=c["detector"],
                    in_initial_freeze=hid in initial,
                    decisions=decision,
                    attempts=[z.attempt_id for z in attempts],
                    committed_actions=sorted({z.outcome for z in attempts if z.patch_id}),
                    stages=dict(
                        applicability=[d["applicability"] or "not_recorded" for d in decision],
                        plan="available" if any(d["policy_action"] for d in decision) else "not_recorded",
                        candidate_generated=any(z.candidate_generated for z in attempts),
                        hard_gate=sorted({z.hard_gate for z in attempts}) or ["not_run"],
                        review=sorted({z.review for z in attempts}) or ["not_run"],
                        commit=any(z.patch_id for z in attempts),
                        rescan=rescan_status,
                    ),
                    rescan=rescan_status,
                    unresolved=hid in unresolved,
                    reason_codes=sorted(codes),
                    source_version_id=c["source_version_id"],
                    released_version_id=w["artifact_id"],
                    released_changed_samples_in_region=int(modified[a:b].sum()),
                    newly_disabled_samples_in_region=int(disabled[a:b].sum()),
                ).model_dump(),
            )


def export_outcomes(root, out):
    """Rebuild from public execution evidence; no private evaluator inputs or model credentials required."""
    root, out = Path(root), Path(out)
    with locked(out):
        marker = out / "summary.json"
        if marker.exists() and read(marker).get("report_type") != "repair_outcomes":
            raise ValueError("Output directory contains another report")
        if not marker.exists() and any((out / f).exists() for f in FILES):
            raise ValueError("Choose a separate directory for repair outcomes")
        return _export(root, out)


def _export(root, out):
    freeze = read(root / "ledger/freeze.json")
    waves = read(root / "export/waveform_manifest.json")
    cases = {c["id"]: c for c in read(root / "dataset_manifest.json")["cases"]}
    receipts = {f.stem: read(f) for f in (root / "dataset-receipts").glob("*.json")}
    calls = model_calls(root)
    budget_pending = set()
    for event in json_rows(root / "events.jsonl"):
        if event["type"] == "decision_pending" and "budget" in event.get("reason", ""):
            budget_pending.update(event.get("ids", []))
            if "id" in event:
                budget_pending.add(event["id"])
    counts = Counter(
        dict.fromkeys(
            (
                "initial_candidates",
                "candidates",
                "attempts",
                "generated",
                "committed",
                "initial_candidates_with_fine_commit",
            ),
            0,
        )
    )
    actions = Counter(
        dict.fromkeys(("waveform_trial", "waveform_repair", "preserve", "mark_invalid", "event_amendment"), 0)
    )
    reasons = Counter()
    out.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=out) as tmp, ExitStack() as stack:
        staging = Path(tmp)
        streams = {f: stack.enter_context((staging / f).open("w")) for f in FILES}
        for item in calls.values():
            streams["api_calls.jsonl"].write(json.dumps(item, allow_nan=False) + "\n")
        for rid in sorted(freeze["records"]):
            for kind, row in record_rows(root, rid, cases, waves, receipts, freeze["revision_id"], budget_pending):
                streams[kind + ".jsonl"].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                counts[kind] += 1
                if kind == "attempts":
                    actions[row["outcome"]] += 1
                    counts["generated"] += row["candidate_generated"]
                    counts["committed"] += row["commit"] == "committed"
                else:
                    counts["initial_candidates"] += row["in_initial_freeze"]
                    counts["initial_candidates_with_fine_commit"] += row["in_initial_freeze"] and bool(
                        row["committed_actions"]
                    )
                    reasons.update(row["reason_codes"])
        stack.close()
        if counts["initial_candidates"] != freeze["candidate_count"]:
            raise ValueError("Frozen candidate count differs")
        for f in FILES:
            (staging / f).replace(out / f)
    summary = dict(
        report_type="repair_outcomes",
        schema_version="1.1",
        revision=freeze["revision_id"],
        counts=dict(counts),
        attempt_outcomes=dict(actions),
        candidates_by_reason=dict(reasons),
        waveform_delivery=dict(
            blocks=len(waves),
            modified_samples=sum(z["modified_samples"] for z in waves.values()),
            estimated_samples=sum(b - a for z in waves.values() for a, b in z["estimated_ranges"]),
        ),
        api_transport_attempts=len(calls),
        api_returned_usd=(
            sum(c["estimated_usd"] for c in calls.values() if c["status"] == "returned")
            if all(c["estimated_usd"] is not None for c in calls.values() if c["status"] == "returned")
            else None
        ),
        api_unconfirmed_calls=sum(c["status"] != "returned" for c in calls.values()),
        actual_billed_usd=None,
        files={f: sha(out / f) for f in FILES},
    )
    write(out / "summary.json", summary)
    return summary
