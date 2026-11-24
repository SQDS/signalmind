"""Immutable discovery revisions and per-block publication receipts. The dataset run owns the writer lock."""

from collections import defaultdict

from .common import digest, locked, read, sha, write
from .contracts import Candidate, CandidateState, Patch, SampleRange, Support


def region(cid, a, b):
    return SampleRange(block_id=cid, start_sample=a, end_sample=b)


def source_row(hit, case, version, t, revision):
    a, b = hit["a"], hit["b"]
    state = hit.get("support_status", Support.ALERT)
    key = digest(
        [
            hit["reason"],
            a,
            b,
            hit.get("threshold"),
            hit["value"],
            hit.get("features"),
            hit.get("peer_bpm"),
            hit.get("anchor_values"),
            hit.get("context_digest"),
            hit.get("peer_context_digest"),
            state,
        ]
    )
    return Candidate(
        candidate_id=hit["id"],
        record_id=case["record_id"],
        channel_id=case["task"],
        region=region(case["id"], a, b),
        source_time_s=(float(t[a]), float(t[b - 1] + 1 / case["fs"])),
        source_version_id=version,
        revision=revision,
        detector=hit["reason"],
        threshold=hit.get("threshold"),
        value=hit["value"],
        support_status=state,
        evidence_digest=key,
        state=CandidateState.DISCOVERED,
    ).model_dump(mode="json")


def freeze(root, records, scope, inspection):
    complete = inspection["inspection_complete"] and all(r.prepared_ok for r in records)
    sources = {r.rid: sha(r.path / "discovery.json") for r in records}
    key = digest([scope, sources, inspection["run_id"]])
    obj = dict(
        scope=scope,
        revision_id=key,
        revision=1,
        status="candidate_frozen" if complete else "pending",
        inspection_run=inspection["run_id"],
        records=sources,
        candidate_count=sum(len(read(r.path / "discovery.json")["candidates"]) for r in records),
    )
    path = root / "ledger/freeze.json"
    if path.exists() and read(path)["status"] == "candidate_frozen" and read(path) != obj:
        raise ValueError("Frozen candidate revision changed")
    write(path, obj)
    return obj


def difference(previous, current):
    old = {h["candidate_id"]: h for h in previous}
    now = {h["candidate_id"]: h for h in current}
    return dict(
        new=sorted(now.keys() - old.keys()),
        resolved=sorted(old.keys() - now.keys()),
        persistent=sorted(i for i in old.keys() & now.keys() if old[i]["evidence_digest"] == now[i]["evidence_digest"]),
        invalidated=sorted(
            i for i in old.keys() & now.keys() if old[i]["evidence_digest"] != now[i]["evidence_digest"]
        ),
    )


def commit_patch(directory, patch: Patch, current_parent):
    """Compare-and-swap within the run lock. A matching receipt is idempotent on resume."""
    with locked(directory / "patches"):
        path = directory / "patches" / (patch.idempotency_key + ".json")
        entry = patch.model_dump(mode="json")
        if path.exists():
            if read(path) != entry:
                raise ValueError("Conflicting evidence for idempotency key")
            return
        if patch.plan.parent_version_id != current_parent or patch.receipt.review_decision not in (
            "accept",
            "mark_unrecoverable",
        ):
            raise ValueError("Patch is stale or lacks release approval")
        folder = directory / "patches"
        # One immutable parent has exactly one committed successor in this run.
        head = folder / "parents" / (digest(current_parent) + ".json")
        if head.exists() and read(head)["key"] != patch.idempotency_key:
            raise ValueError("Conflicting patches on the same parent")
        write(head, dict(key=patch.idempotency_key))
        write(path, entry)


def disposition(h, trials, gates):
    if h.get("decision_source") == "unreviewed":
        return CandidateState.PENDING
    if h["id"] in trials:
        trial = trials[h["id"]]
        if trial.get("hard_gate", {}).get("status") == "rejected":
            return CandidateState.HARD_REJECT
        return {
            "accepted": CandidateState.ACCEPTED,
            "rollback": CandidateState.REJECTED,
            "pending": CandidateState.PENDING,
            "ineligible": CandidateState.UNKNOWN,
        }.get(trial["status"], CandidateState.UNKNOWN)
    if h["wave_requested"] == "mark_invalid":
        g = gates.get(h["case"], {})
        if h.get("final_fine_excluded"):
            return CandidateState.BAD
        if g.get("hard_gate", {}).get("status") == "rejected":
            return CandidateState.HARD_REJECT
        if any(d.get("decision") == "pending" for d in g.get("ranges", {}).values()):
            return CandidateState.PENDING
        if g and not g["accepted"]:
            return CandidateState.REJECTED
    if h["measurement"] == "review" or h.get("support_status") in ("insufficient_support", "failed", "pending"):
        return CandidateState.UNKNOWN
    return CandidateState.PRESERVED


def release_index(root, records):
    """One queryable final ledger, including resolved discoveries and new findings from the final waveform."""
    import json

    target = root / "ledger" / "candidates.jsonl"
    tmp = target.with_suffix(".tmp")
    count = 0
    with tmp.open("w") as f:
        for r in records:
            doc = read(r.path / "discovery.json")
            rescan = read(r.path / "rescan.json")
            base = {c["candidate_id"]: c for c in doc["candidates"]}
            final = {c["candidate_id"]: c for c in rescan["candidates"]}
            initial_states = {h["id"]: h for h in read(r.path / "hit-dispositions.json")}
            states = {h["id"]: h for h in read(r.path / "release-dispositions.json")}
            patches = defaultdict(list)
            block_patches = defaultdict(list)
            for path in (r.path / "patches").glob("*.json"):
                patch = Patch.model_validate(read(path))
                patches[patch.plan.candidate_id].append(patch.idempotency_key)
                if patch.plan.candidate_id.startswith(("coarse-gate-", "fine-gate-")):
                    for region in patch.plan.write_ranges:
                        block_patches[region.block_id].append(
                            (region.start_sample, region.end_sample, patch.idempotency_key)
                        )
            for hid in sorted(base.keys() | final.keys() | initial_states.keys() | states.keys()):
                raw = final.get(hid) or base.get(hid)
                if raw is None:
                    revpath = r.path / "candidate-revision-2.json"
                    raw = next(c for c in read(revpath)["candidates"] if c["candidate_id"] == hid)
                cand = Candidate.model_validate(raw)
                info = states.get(hid, initial_states.get(hid, {}))
                cand.state = CandidateState(info.get("state", "need_evidence"))
                observed = (
                    "resolved"
                    if hid not in final
                    else "new"
                    if hid not in base
                    else "invalidated"
                    if base[hid]["evidence_digest"] != final[hid]["evidence_digest"]
                    else "persistent"
                )
                row = dict(
                    cand.model_dump(mode="json"),
                    change=observed,
                    group=info.get("group"),
                    diagnosis=info.get("diagnosis"),
                    decision_ref=info.get("decision_source"),
                    patches=sorted(
                        set(patches.get(hid, []))
                        | {
                            key
                            for a, b, key in block_patches[cand.region.block_id]
                            if a < cand.region.end_sample and cand.region.start_sample < b
                        }
                    ),
                    final_measurement_policy=info.get("measurement"),
                    prior_version_id=(base.get(hid) or {}).get("source_version_id"),
                )
                f.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
                count += 1
    tmp.replace(target)
    report = dict(candidates=count, sha256=sha(target), freeze=read(root / "ledger/freeze.json")["revision_id"])
    write(root / "ledger/release.json", report)
    return report
