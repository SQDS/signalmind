"""Whole-dataset orchestration shared by the workbench and CLI. Artifacts stay in the run workspace."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from tsmcp.processing import ProcessingSession, intervals

from . import ledger
from .common import array_hash, changed, digest, implementation_hash, read, sha, write
from .context import ContextBuilder
from .contracts import Diagnosis, Patch, RepairPlan, ReviewReceipt, Run, Tool, outcome
from .dataset_evidence import (
    acquisition,
    change_ranges,
    discover,
    enrich,
    hit_location,
    jev_routes,
    locations,
    native_rows,
    patient_profiles,
    view,
)
from .dataset_review import FIELDS, QUALITIES, Reviewer, applies
from .engine import aggregate, evaluate_output, load_arrays
from .knowledge import Knowledge
from .provenance import check_stop, emit
from .quality_gate import ACCEPTANCE, SPEC_ID, registered_parameters, spike_check, waveform_gate
from .tasks import probe


def csv_table(path, rows, fields):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    temporary.replace(path)


class Record:
    def __init__(self, root, cases, bank, config, reviewer, implementation):
        self.root = root
        self.cases = {c["id"]: c for c in cases}
        self.rid = cases[0]["record_id"]
        self.path = root / "records" / self.rid
        self.path.mkdir(parents=True, exist_ok=True)
        self.bank = bank
        self.config = config
        self.reviewer = reviewer
        self.implementation = implementation
        self.raw = {cid: bank["identity"][cid] for cid in self.cases}
        self.unresolved = []
        self.pending_ranges = defaultdict(list)
        self.spent_gate_views = 0
        self.revision = digest(
            ["coarse", self.implementation, sorted((k, a["content_hash"]) for k, a in self.raw.items())]
        )
        self.context = ContextBuilder(config)
        self.round = 1
        self.blocked = False
        self.set_stage = None
        self.prepared_ok = True

    def phase(self, name):
        if self.set_stage:
            self.set_stage(name)
            emit(self.root, "quality_phase", stage=name, record=self.rid, revision=self.revision)

    def loader(self, arts):
        @lru_cache(maxsize=12)
        def get(cid):
            a = arts[cid]
            _, t, x, v, e = load_arrays(self.root, self.cases[cid], a)
            if array_hash(x, t, v, e) != a["content_hash"] or sha(self.root / a["probe_path"]) != a["probe_sha256"]:
                raise ValueError(f"Artifact or native probe changed for {cid}")
            return t, x, v, e, read(self.root / a["probe_path"])

        return get

    def picture(self, key, loc, before, after=None):
        path = self.path / "views" / (key + ".png")
        metrics = view(path, loc, self.cases, before, after)
        return path, metrics

    def case_span(self, cid, a, b, data):
        return hit_location(dict(case=cid, a=a, b=b), self.cases, data)

    def plan(self, cid, parent, target, ranges, operations, hit=None):
        case = self.cases[cid]
        diagnosis = (hit or {}).get("diagnosis")
        if hit and hit.get("wave_requested") == "interpolate_spike" and hit.get("trial_diagnosis"):
            diagnosis = hit["trial_diagnosis"]
        diagnosis = diagnosis or dict(label="unknown", evidence_refs=[parent["content_hash"]])
        protected = []
        if any(op["op"] in ("interpolate", "repair") for op in operations):
            meta = read(self.root / parent["probe_path"])
            bounds = meta["detector_support"]
            restricted = [
                pair
                for k in ("unsupported_ranges", "boundary_uncertain_ranges", "empty_event_ranges")
                for pair in bounds[k]
            ]
            if case["task"] == "ECG":
                m = round(0.08 * case["fs"])
                restricted.extend(
                    (max(0, z["sample"] - m), min(case["n"], z["sample"] + m + 1)) for z in meta["events"]
                )
            protected = [
                ledger.region(cid, lo, hi) for lo, hi in restricted if any(a < hi and lo < b for a, b in ranges)
            ]
        plan = RepairPlan(
            candidate_id=target,
            candidate_revision=self.revision,
            parent_version_id=parent["artifact_id"],
            diagnosis=Diagnosis.model_validate(diagnosis),
            tool=Tool(
                tool_id=operations[0]["op"] if len(operations) == 1 else "registered_recipe",
                tool_version=self.implementation,
                parameters={"operations": operations},
            ),
            read_ranges=[ledger.region(cid, 0, case["n"])],
            write_ranges=[ledger.region(cid, a, b) for a, b in ranges],
            protected_ranges=protected,
            expected_effects=[op["op"] for op in operations],
            acceptance_spec_id=SPEC_ID,
            stop_conditions=["unsupported", "protected_physiology", "scope_violation", "failed_retest"],
        )
        write(self.path / "plans" / (plan.plan_id + ".json"), plan.model_dump(mode="json"))
        return plan

    def publish(self, plan, parent, candidate, check, evidence, action):
        self.phase("patch_committing")
        receipt = ReviewReceipt(
            candidate_id=plan.candidate_id,
            candidate_revision=self.revision,
            parent_version_id=parent["artifact_id"],
            candidate_version_id=candidate["artifact_id"],
            plan_id=plan.plan_id,
            acceptance_spec_id=SPEC_ID,
            hard_gate=check,
            review_decision=action,
            evidence_refs=evidence,
            reviewer_version=self.implementation,
        )
        patch = Patch(
            plan=plan,
            receipt=receipt,
            candidate_version_id=candidate["artifact_id"],
            idempotency_key=digest([plan.plan_id, candidate["artifact_id"]]),
        )
        ledger.commit_patch(self.path, patch, parent["artifact_id"])

    def isolated_spikes(self, parents, alerts=()):
        """Recover individual bounded excursions from a coarse recipe that was declined for the whole block."""
        current = self.loader(parents)
        original = self.loader(self.raw)
        hits = []
        cores = defaultdict(set)
        for h in alerts:
            if h["reason"] in ("biphasic_core", "impulsive_core") and h["b"] - h["a"] <= 4:
                cores[h["case"]].add((h["a"], h["b"]))
        for cid, c in self.cases.items():
            for op in self.bank["fixed_dsp"][cid]["operations"]:
                if op.get("parameters", {}).get("source") == "public_discontinuous_excursion_rule":
                    cores[cid].update((a, b) for a, b in op["ranges_samples"])
            if not cores[cid]:
                continue
            _, x, v, e, pr = current(cid)
            raw = original(cid)[1]
            for a, b in sorted(cores[cid]):
                if a < 4 or b + 4 > len(x) or b - a > min(4, 5 * c["fs"]):
                    continue
                if not (v[a - 4 : b + 4] & ~e[a - 4 : b + 4]).all() or not np.array_equal(
                    raw[a - 4 : b + 4], x[a - 4 : b + 4]
                ):
                    continue
                check = spike_check(c, x, v, e, pr, a, b)
                # Preserve the candidate and its reason even when numerical repair is inapplicable.
                reason = "isolated_spike_candidate"
                hits.append(
                    dict(
                        id=digest([cid, a, b, reason])[:24],
                        case=cid,
                        task=c["task"],
                        a=a,
                        b=b,
                        reason=reason,
                        support_status=check["status"]
                        if check["status"] != "evaluated_no_alert"
                        else "suspected_artifact",
                        applicability=check,
                        value=float(np.ptp(x[a - 1 : b + 1])),
                        anchor_values=[float(z) for z in (x[a - 1], x[a:b].min(), x[a:b].max(), x[b])],
                    )
                )
        return hits

    def spike_trials(self, parents, hits):
        accepted = parents.copy()
        log = []
        for h in hits:
            if h["reason"] != "isolated_spike_candidate" or h["wave_requested"] != "interpolate_spike":
                continue
            check_stop(self.root)
            cid, a, b = h["case"], h["a"], h["b"]
            c, par = self.cases[cid], accepted[cid]
            raw, t, x, v, e = load_arrays(self.root, c, par)
            rec = dict(
                hit=h["id"],
                case=cid,
                a=a,
                b=b,
                method="interpolate",
                group=h["group"],
                plan_receipt=h["decision_source"],
                parent=par["artifact_id"],
                status="ineligible",
            )
            log.append(rec)
            old_pr = read(self.root / par["probe_path"])
            pre = spike_check(c, x, v, e, old_pr, a, b)
            op = dict(
                op="interpolate",
                ranges_samples=[[a, b]],
                parameters={"max_transient_samples": min(4, int(5 * c["fs"]))},
            )
            plan = self.plan(cid, par, h["id"], [[a, b]], [op], h)
            left, r = max(0, a - round(5 * c["fs"])), min(len(x), b + round(5 * c["fs"]))
            if h.get("context_digest", array_hash(x[left:r], v[left:r], e[left:r])) != array_hash(
                x[left:r], v[left:r], e[left:r]
            ):
                pre = {"status": "insufficient_support", "reason": "stale_parent_context"}
            rec.update(plan_id=plan.plan_id, candidate_revision=self.revision, precheck=pre)
            if pre["status"] != "evaluated_no_alert":
                rec.update(
                    reason=pre.get("reason", pre["status"]),
                    hard_gate={"status": "rejected", "checks": {"support_and_morphology": False}},
                )
                if h["measurement"] == "keep":
                    h["measurement"] = "review"
                continue
            started = time.monotonic()
            session = ProcessingSession(
                x,
                fs=c["fs"],
                channel=c["task"],
                unit=c["unit"],
                source_id=par["content_hash"],
                missing=~v | e,
                max_transient_samples=min(4, int(5 * c["fs"])),
            )
            state = session.process(session.raw.id, "interpolate", [[a, b]])
            candidate = evaluate_output(
                self.root,
                c,
                {"hash": digest(["isolated_spike", cid, a, b])},
                self.implementation,
                par,
                raw,
                t,
                x,
                state.values,
                v,
                e | state.repaired,
                state.operations,
                started,
            )
            rec["candidate"] = candidate["artifact_id"]
            # Use the same observed support in the diagnostic comparison, then exclude estimates in the release probe.
            old_pr = read(self.root / par["probe_path"])
            new_pr = probe(state.values, t, c["fs"], v, e, c["task"])
            old, new = [
                [z["sample"] for z in p["events"] if a - c["fs"] <= z["sample"] < b + c["fs"]] for p in (old_pr, new_pr)
            ]
            before_step, after_step = [float(np.max(np.abs(np.diff(y[a - 1 : b + 1])))) for y in (x, state.values)]
            rec["retest"] = dict(
                before_step=before_step, after_step=after_step, before_landmarks=old, after_landmarks=new
            )
            passed = (
                len(old) == len(new)
                and all(abs(i - j) <= 1 for i, j in zip(old, new, strict=True))
                and after_step < before_step
            )
            hard = waveform_gate(
                c, (t, x, v, e, old_pr), (t, state.values, v, e | state.repaired, new_pr), state.operations, [[a, b]]
            )
            hard["checks"]["native_timing_and_step"] = bool(passed)
            hard["status"] = "passed" if all(hard["checks"].values()) else "rejected"
            rec["hard_gate"] = hard
            if hard["status"] != "passed":
                rec.update(status="rollback", reason="local_waveform_or_landmark_retest")
            elif self.spent_gate_views >= self.config["write_views"]:
                rec.update(status="pending", reason="write_view_budget")
            else:
                key = "spike-" + h["id"] + ("" if self.round == 1 else "-retry")
                before = self.loader(accepted)
                after = self.loader(accepted | {cid: candidate})
                loc = self.case_span(cid, a, b, before)
                loc.update(context_s=2.0, core_samples=[a, b], spike_retest=rec["retest"])
                pic, metrics = self.picture(key, loc, before, after)
                self.spent_gate_views += 1
                rec["image"] = str(pic.relative_to(self.path))
                rec["review_evidence"] = metrics
                d = self.review_writes(
                    [
                        dict(
                            id=key,
                            images=[pic],
                            data=dict(
                                task=c["task"],
                                changed_ranges=[[a, b]],
                                parent=par["content_hash"],
                                candidate=candidate["content_hash"],
                                native=metrics,
                                diagnostic=rec["retest"],
                                spike_review_guide=(
                                    "Assess the numerical samples against the measured anchors, the timed "
                                    "landmarks and (if supported) the same-phase neighbours. These are observational "
                                    "comparisons. A displaced or removed physiological feature is distinct "
                                    "from omission of an interval across estimated support in the release table. "
                                    "Cite the local timing or morphology behind your decision."
                                ),
                                revision=self.revision,
                                plan_id=plan.plan_id,
                                acceptance_spec=SPEC_ID,
                            ),
                        )
                    ],
                    before,
                    after,
                    {key: loc},
                ).get(key)
                if d:
                    rec.update(
                        status={"accept": "accepted", "rollback": "rollback", "need_evidence": "pending"}[
                            d["decision"]
                        ],
                        reason=d["reason"],
                        receipt=d["receipt"],
                    )
                    if d["decision"] == "accept":
                        self.publish(plan, par, candidate, hard, [d["receipt"]], "accept")
                        accepted[cid] = candidate
                else:
                    rec.update(status="pending", reason="write_review_unavailable")
            if rec["status"] == "pending":
                self.unresolved.append(dict(stage="spike_trial", case=cid, hit=h["id"], reason=rec["reason"]))
            if rec["status"] != "accepted" and h["measurement"] == "keep":
                h["measurement"] = "review"
        write(self.path / ("spike-trials.json" if self.round == 1 else "spike-trials-rescan.json"), log)
        return accepted

    def review_writes(self, rows, before, after, locations):
        decisions = self.reviewer.decide("gate", rows)
        for row in rows:
            key = row["id"]
            first = decisions.get(key)
            if not first or first["decision"] != "need_evidence":
                continue
            if self.spent_gate_views >= self.config["write_views"]:
                continue
            loc = dict(locations[key], context_s=10)
            pic, metrics = self.picture(key + "-expanded", loc, before, after)
            self.spent_gate_views += 1
            second = self.reviewer.decide(
                "gate",
                [
                    dict(
                        id=key,
                        images=[pic],
                        data=dict(row["data"], native=metrics, prior_review=first["receipt"], request=first["reason"]),
                    )
                ],
            ).get(key)
            if second:
                decisions[key] = dict(second, prior_receipt=first["receipt"])
        return decisions

    def gate(self, label, parents, candidates, authorized_plans=None):
        before = self.loader(parents)
        after = self.loader(candidates)
        rows = []
        needed = defaultdict(list)
        result = parents.copy()
        log = {}
        gates = {}
        locations = {}
        plans = {}
        for cid in self.cases:
            if parents[cid]["content_hash"] == candidates[cid]["content_hash"]:
                continue
            check_stop(self.root)
            ops = candidates[cid]["operations"]
            authorized = [(a, b) for op in ops for a, b in op["ranges_samples"]]
            plan = (
                authorized_plans[cid]
                if authorized_plans is not None
                else self.plan(cid, parents[cid], label + "-" + cid, authorized, ops)
            )
            if plan.parent_version_id != parents[cid]["artifact_id"] or plan.candidate_revision != self.revision:
                raise ValueError("Write approval is bound to another parent or candidate revision")
            authorized = [(z.start_sample, z.end_sample) for z in plan.write_ranges]
            hard = waveform_gate(self.cases[cid], before(cid), after(cid), ops, authorized)
            plans[cid] = plan
            gates[cid] = hard
            if hard["status"] != "passed":
                log[cid] = dict(
                    accepted=False,
                    plan_id=plan.plan_id,
                    hard_gate=hard,
                    ranges={},
                    parent_hash=parents[cid]["content_hash"],
                    candidate_hash=candidates[cid]["content_hash"],
                )
                self.pending_ranges[cid].extend(change_ranges(before(cid), after(cid)))
                continue
            width = max(1, round(10 * self.cases[cid]["fs"]))
            tiles = []
            for lo, hi in change_ranges(before(cid), after(cid)):
                for a in range(lo, hi, width):
                    b = min(hi, a + width)
                    if tiles and b - tiles[-1][0][0] <= width:
                        tiles[-1].append((a, b))
                    else:
                        tiles.append([(a, b)])
            for spans in tiles:
                a, b = spans[0][0], spans[-1][1]
                key = f"{label}-{cid}-{a}-{b}"
                needed[cid].append(key)
                if self.spent_gate_views >= self.config["write_views"]:
                    continue
                loc = self.case_span(cid, a, b, before)
                locations[key] = loc
                image, metrics = self.picture(key, loc, before, after)
                rows.append(
                    dict(
                        id=key,
                        images=[image],
                        data=dict(
                            task=self.cases[cid]["task"],
                            changed_ranges=spans,
                            hard_gate=hard,
                            revision=self.revision,
                            plan_id=plan.plan_id,
                            acceptance_spec=SPEC_ID,
                            changed_seconds_relative_to_view=[
                                [(l0 - a) / self.cases[cid]["fs"], (r0 - a) / self.cases[cid]["fs"]] for l0, r0 in spans
                            ],
                            native=metrics,
                            parent=parents[cid]["content_hash"],
                            candidate=candidates[cid]["content_hash"],
                        ),
                    )
                )
                self.spent_gate_views += 1
        self.phase("independent_review")
        decisions = self.review_writes(rows, before, after, locations)
        for cid, keys in needed.items():
            accepted = all(decisions.get(k, {}).get("decision") == "accept" for k in keys)
            if accepted:
                self.publish(
                    plans[cid],
                    parents[cid],
                    candidates[cid],
                    gates[cid],
                    [decisions[k]["receipt"] for k in keys],
                    "mark_unrecoverable" if label.startswith("fine-gate") else "accept",
                )
                result[cid] = candidates[cid]
            if any(decisions.get(k, {}).get("decision") in (None, "need_evidence") for k in keys):
                self.pending_ranges[cid].extend(change_ranges(before(cid), after(cid)))
                self.unresolved.append(dict(stage=label, case=cid, reason="write_review_pending"))
            log[cid] = dict(
                parent_hash=parents[cid]["content_hash"],
                candidate_hash=candidates[cid]["content_hash"],
                accepted=accepted,
                plan_id=plans[cid].plan_id,
                hard_gate=gates[cid],
                ranges={k: decisions.get(k, {"decision": "pending"}) for k in keys},
            )
        write(self.path / (label + ".json"), log)
        return result

    def coarse(self):
        self.phase("coarse_planning")
        if not self.reviewer.enabled:
            chosen = self.raw.copy()
            records = {}
            before = self.loader(self.raw)
            after = self.loader(self.bank["rule_adaptive_dsp"])
            for cid, c in self.cases.items():
                art = self.bank["rule_adaptive_dsp"][cid]
                if art["content_hash"] != self.raw[cid]["content_hash"]:
                    hard = waveform_gate(c, before(cid), after(cid), art["operations"])
                    records[cid] = {"hard_gate": hard, "review": "not_applicable_local_baseline"}
                    if hard["status"] == "passed":
                        chosen[cid] = art
            write(self.path / "local-dsp-checks.json", records)
            return chosen
        before = self.loader(self.raw)
        by_type = defaultdict(list)
        protected = []
        for cid, c in self.cases.items():
            available = {"identity": self.raw[cid]["content_hash"]}
            operations = []
            for name in ("fixed_dsp", "rule_adaptive_dsp"):
                art = self.bank[name][cid]
                if art["content_hash"] in available.values():
                    continue
                # A whole-block DSP candidate can include a transient interpolation on a plausible QRS.
                # Route that evidence back to the reviewer before offering a numerical replacement.
                if c["task"] == "ECG" and art["stats"]["estimated_samples"]:
                    _, _, _, est, probe = before(cid)
                    _, _, _, e2, _ = self.loader(self.bank[name])(cid)
                    peaks = np.array([r["sample"] for r in probe["events"]], int)
                    veto = False
                    for left, right in intervals(e2 & ~est):
                        margin = round(0.08 * c["fs"])
                        if np.any((peaks >= left - margin) & (peaks < right + margin)):
                            protected.append(dict(case=cid, a=left, b=right, method=name))
                            veto = True
                    if veto:
                        continue
                available[name] = art["content_hash"]
                operations.extend((name, o["op"]) for o in art["operations"])
            if len(available) > 1:
                by_type[c["task"], tuple(sorted(available)), tuple(sorted(set(operations)))].append(cid)
        write(self.path / "coarse-protected.json", protected)
        requests = []
        ids = {}
        for (task, methods, operations), members in sorted(by_type.items()):
            key = self.rid + "-coarse-" + digest([task, methods, operations])[:12]
            members.sort(
                key=lambda cid: sum(
                    self.bank[n][cid]["stats"]["modified_samples"]
                    + self.bank[n][cid]["stats"]["support_changed_samples"]
                    for n in methods
                )
            )
            example_ids = list(dict.fromkeys([members[len(members) // 2], members[-1]]))
            pics = []
            examples = []
            for cid in example_ids:
                for name in methods:
                    if name == "identity":
                        continue
                    after = self.loader(self.bank[name])
                    ranges = change_ranges(before(cid), after(cid))
                    a, b = max(ranges, key=lambda pair: pair[1] - pair[0])
                    width = max(1, round(self.cases[cid]["fs"] * 10))
                    mid = (a + b) // 2
                    loc = self.case_span(cid, max(a, mid - width // 2), min(b, mid + width // 2 + 1), before)
                    pic, metrics = self.picture("coarse-" + name + "-" + cid, loc, before, after)
                    pics.append(pic)
                    examples.append(dict(method=name, stats=self.bank[name][cid]["stats"], native=metrics))
            ids[key] = members
            requests.append(
                dict(
                    id=key,
                    images=pics,
                    data=dict(
                        available=methods,
                        operations=operations,
                        blocks=len(members),
                        scope=digest([[cid, *[self.bank[n][cid]["content_hash"] for n in methods]] for cid in members]),
                        image_order=examples,
                        task=task,
                        purpose=self.config["purpose"],
                    ),
                )
            )
        choices = self.reviewer.decide("coarse", requests)
        write(self.path / "coarse-selection.json", choices)
        write(self.path / "coarse-groups.json", ids)
        candidates = self.raw.copy()
        for key, members in ids.items():
            for cid in members:
                if key in choices:
                    candidates[cid] = self.bank[choices[key]["method"]][cid]
                else:
                    self.pending_ranges[cid].append((0, self.cases[cid]["n"]))
                    self.unresolved.append(dict(stage="coarse", case=cid, reason="unreviewed"))
        return self.gate("coarse-gate", self.raw, candidates)

    def groups(self, hits, data):
        by = defaultdict(list)
        scores = (
            jev_routes(self.root, self.config, hits) if self.config["routing"] == "jev" and not self.blocked else {}
        )
        for h in hits:
            score = scores.get(h["id"])
            h["jev_score"] = score
            route = ("joint" if score <= 0.5 else "local") if score is not None else "reason"
            if self.config["routing"] == "jev" and score is None:
                route = "unrouted"
            by[h["task"], h["reason"], route].append(h)
        requests = []
        rosters = []
        for (task, reason, route), members in sorted(by.items()):
            gid = f"{self.rid}-{task}-{reason}-{route}" + ("" if self.round == 1 else "-rescan")
            for h in members:
                h["group"] = gid
            values = np.array(
                [[np.nan if h["features"][k] is None else h["features"][k] for k in FIELDS] for h in members]
            )
            stats = {}
            for j, k in enumerate(FIELDS):
                known = values[:, j][np.isfinite(values[:, j])]
                if len(known):
                    lo, center, hi = np.percentile(known, [0, 50, 100])
                    stats[k] = [float(lo), float(center), float(hi)]
                    values[:, j] = (np.nan_to_num(values[:, j], nan=center) - center) / max(hi - lo, 1e-8)
                else:
                    values[:, j] = 0
            distance = np.abs(values).mean(axis=1)
            typical = int(distance.argmin())
            reps = [
                typical,
                max((j for j in range(len(members)) if j != typical), key=distance.__getitem__, default=typical),
            ]
            roles = ["typical", "detail" if len(members) == 1 else "boundary"]
            candidate_group = reason == "isolated_spike_candidate"
            eligibility = np.array([h.get("applicability", {}).get("status") == "evaluated_no_alert" for h in members])
            if candidate_group and eligibility.any() and not eligibility.all():
                reps = []
                for mask in (eligibility, ~eligibility):
                    ids = np.flatnonzero(mask)
                    deviations = np.abs(values[ids] - np.median(values[ids], axis=0)).mean(axis=1)
                    reps.append(int(ids[deviations.argmin()]))
                roles = ["trial_eligible", "protected_comparison"]
            examples = [members[j] for j in reps]
            images = []
            for i, h in enumerate(examples):
                loc = hit_location(h, self.cases, data)
                if self.round > 1:
                    loc["context_s"] = 10
                if reason == "isolated_spike_candidate":
                    loc["show_detail"] = True
                if i == 1 and len(members) == 1:
                    loc["context_s"] = 0.4
                # Keep representative plots readable for sustained alerts.
                if loc["end"] - loc["start"] > 15:
                    loc["end"] = loc["start"] + 15
                pic, _ = self.picture(f"{gid}-{i}", loc, data)
                images.append(pic)
            case_ids = sorted({h["case"] for h in members})
            case_index = {cid: j for j, cid in enumerate(case_ids)}
            columns = ["case_index", "a", "b", "trigger_value", *FIELDS, "ECG_bpm", "PPG_bpm", "ABP_bpm", "jev_score"]
            if reason == "isolated_spike_candidate":
                columns.extend(
                    ["left_anchor", "core_min", "core_max", "right_anchor", "trial_eligible", "precheck_status"]
                )
            table = [
                [
                    case_index[h["case"]],
                    h["a"],
                    h["b"],
                    h["value"],
                    *[h["features"][k] for k in FIELDS],
                    *[h["peer_bpm"].get(k) for k in ("ECG", "PPG", "ABP")],
                    h["jev_score"],
                    *h.get("anchor_values", []),
                    *(
                        [int(eligibility[i]), h.get("applicability", {}).get("status", "insufficient_support")]
                        if candidate_group
                        else []
                    ),
                ]
                for i, h in enumerate(members)
            ]
            requests.append(
                dict(
                    id=gid,
                    images=images,
                    data=dict(
                        task=task,
                        revision=self.revision,
                        acceptance_spec=SPEC_ID,
                        trigger=reason,
                        evidence_ids=["view_1", "view_2", "members"],
                        route=route,
                        scope=len(members),
                        summary=stats,
                        membership=digest(members),
                        cases=[
                            dict(id=cid, fs=self.cases[cid]["fs"], unit=self.cases[cid]["unit"]) for cid in case_ids
                        ],
                        columns=columns,
                        feedback=[h["feedback"] for h in examples if h.get("feedback")],
                        member_rows=table,
                        image_rows=reps,
                        image_roles=roles,
                        purpose=self.config["purpose"],
                    ),
                )
            )
            if reason == "isolated_spike_candidate":
                evidence = requests[-1]["data"]
                evidence["core_samples_by_member"] = [data(h["case"])[1][h["a"] : h["b"]].tolist() for h in members]
                evidence["trial_scope"] = dict(
                    eligible=int(eligibility.sum()),
                    protected=int((~eligibility).sum()),
                    image_eligible=[bool(eligibility[i]) for i in reps],
                    interpretation="Eligibility only permits a reversible trial; it does not diagnose an artifact.",
                )
                if eligibility.any() and not eligibility.all():
                    evidence["trial_scope"]["instruction"] = (
                        "View 1 represents eligible cores; view 2 represents protected members. "
                        "Give both a full-group diagnosis and trial_diagnosis for eligible members, "
                        "citing the matching view and measurements."
                    )
                evidence["plot_key"] = (
                    "Both columns show observations. Left: beat context. Right: sampled points at the shaded core "
                    "and its immediate anchors. Compare the core shape against neighbouring beats. "
                    "A near-constant core can be far from both anchors even when its internal span is small. "
                    "Null means unavailable; reason routing has no JEV score."
                )
            request = requests.pop()
            pages = self.context.group_pages(request)
            for row, start, stop in pages:
                requests.append(row)
                for h in members[start:stop]:
                    h["policy_id"] = row["id"]
            rosters.append(
                dict(
                    id=gid,
                    task=task,
                    reason=reason,
                    route=route,
                    count=len(members),
                    examples=[h["id"] for h in examples],
                    member_ids=[h["id"] for h in members],
                    member_rows_sha256=digest(table),
                    pages=[dict(id=r["id"], start=a, end=b) for r, a, b in pages],
                )
            )
        suffix = "" if self.round == 1 else "-rescan"
        write(self.path / ("groups" + suffix + ".json"), rosters)
        if not self.blocked:
            self.phase("diagnosis_running")
        plans = self.reviewer.decide("groups", requests) if not self.blocked else {}
        if not self.blocked:
            self.phase("repair_planning")
        write(self.path / ("repair-plans" + suffix + ".json"), plans)
        for h in hits:
            plan = plans.get(h["policy_id"])
            if plan is None:
                h.update(
                    wave_requested="preserve",
                    measurement="review",
                    decision_source="unreviewed" if self.reviewer.enabled else "local_rule",
                )
            else:
                h["diagnosis"] = plan.get("diagnosis", dict(label="unknown", evidence_refs=[plan["receipt"]]))
                f = h["features"]
                h["parameters"] = plan.get("parameters", {})
                h.update(
                    wave_requested=plan["wave"] if applies(plan["wave_if"], f) else "preserve",
                    measurement=plan["measure"] if applies(plan["measure_if"], f) else plan["otherwise"],
                    decision_source=plan["receipt"],
                )
                if h["reason"] == "isolated_spike_candidate" and h["wave_requested"] == "interpolate_spike":
                    eligible = h.get("applicability", {}).get("status", "insufficient_support") == "evaluated_no_alert"
                    h["trial_diagnosis"] = plan.get("trial_diagnosis", h["diagnosis"])
                    if not eligible or h["trial_diagnosis"]["label"] != "artifact":
                        h["wave_requested"] = "preserve"
                uncertain = h.get("support_status") in ("failed", "pending", "insufficient_support") or plan.get(
                    "diagnosis", {}
                ).get("label") in ("mixed", "unknown")
                if uncertain:
                    if h["wave_requested"] not in ("mark_invalid", "interpolate_spike"):
                        h["wave_requested"] = "preserve"
                    if h["measurement"] == "keep":
                        h["measurement"] = "review"
        if self.reviewer.enabled:
            self.unresolved.extend(
                dict(stage="groups", group=r["id"], reason="unreviewed") for r in requests if r["id"] not in plans
            )
        if any(g["route"] == "unrouted" for g in rosters):
            self.unresolved.append(dict(stage="routing", reason="JEV quota exhausted"))
        return rosters

    def filter_trials(self, parents, hits):
        selected = parents.copy()
        log = []
        for h in hits:
            op = h["wave_requested"]
            if op not in ("remove_tone", "lowpass", "local_detrend"):
                continue
            cid, a, b = h["case"], h["a"], h["b"]
            c = self.cases[cid]
            par = selected[cid]
            raw, t, x, v, e = load_arrays(self.root, c, par)
            left, r = max(0, a - round(5 * c["fs"])), min(len(x), b + round(5 * c["fs"]))
            rec = dict(hit=h["id"], case=cid, a=a, b=b, method=op, parent=par["artifact_id"])
            log.append(rec)
            params = registered_parameters(op, h["reason"], c, h.get("parameters", {}))
            if (
                params is None
                or not (v[a:b] & ~e[a:b]).all()
                or h["context_digest"] != array_hash(x[left:r], v[left:r], e[left:r])
            ):
                rec.update(status="ineligible", reason="unsupported_parameters_support_or_stale_parent")
                h["measurement"] = "review"
                continue
            operation = dict(op=op, ranges_samples=[[a, b]], parameters=params)
            plan = self.plan(cid, par, h["id"], [[a, b]], [operation], h)
            key = "filter-" + h["id"] + "-" + str(self.round)
            session = ProcessingSession(
                x, fs=c["fs"], channel=c["task"], unit=c["unit"], source_id=par["content_hash"], missing=~v | e
            )
            started = time.monotonic()
            try:
                state = session.process(session.raw.id, op, [[a, b]], **params)
            except ValueError as exc:
                rec.update(status="ineligible", reason=str(exc), plan_id=plan.plan_id)
                h["measurement"] = "review"
                continue
            candidate = evaluate_output(
                self.root,
                c,
                {"hash": plan.plan_id},
                self.implementation,
                par,
                raw,
                t,
                x,
                state.values,
                v,
                e,
                state.operations,
                started,
            )
            before = self.loader(selected)
            after = self.loader(selected | {cid: candidate})
            hard = waveform_gate(c, before(cid), after(cid), state.operations, [[a, b]])
            rec.update(plan_id=plan.plan_id, candidate=candidate["artifact_id"], hard_gate=hard)
            if hard["status"] != "passed":
                rec.update(status="rollback", reason="hard_gate")
            elif self.spent_gate_views >= self.config["write_views"]:
                rec.update(status="pending", reason="write_view_budget")
            else:
                loc = self.case_span(cid, a, b, before)
                pic, metrics = self.picture(key, loc, before, after)
                self.spent_gate_views += 1
                d = self.review_writes(
                    [
                        dict(
                            id=key,
                            images=[pic],
                            data=dict(
                                task=c["task"],
                                revision=self.revision,
                                parent=par["content_hash"],
                                candidate=candidate["content_hash"],
                                plan_id=plan.plan_id,
                                hard_gate=hard,
                                native=metrics,
                            ),
                        )
                    ],
                    before,
                    after,
                    {key: loc},
                ).get(key)
                if d and d["decision"] == "accept":
                    self.publish(plan, par, candidate, hard, [d["receipt"]], "accept")
                    selected[cid] = candidate
                    rec.update(status="accepted", reason=d["reason"], receipt=d["receipt"])
                elif d and d["decision"] == "rollback":
                    rec.update(status="rollback", reason=d["reason"], receipt=d["receipt"])
                else:
                    rec.update(status="pending", reason="need_evidence")
            if rec["status"] != "accepted":
                h["measurement"] = "review"
            if rec["status"] == "pending":
                self.unresolved.append(dict(stage="filter_trial", hit=h["id"], reason=rec["reason"]))
        write(self.path / ("filter-trials.json" if self.round == 1 else "filter-trials-rescan.json"), log)
        return selected

    def fine(self, parents, hits):
        label = "fine-gate" if self.round == 1 else "fine-gate-rescan"
        authorized_plans = {}
        after = parents.copy()
        requested = defaultdict(list)
        for h in hits:
            if h["wave_requested"] == "mark_invalid" and h["reason"] != "protected_landmark":
                requested[h["case"]].append((h["a"], h["b"], h["id"]))
        for cid, rows in requested.items():
            check_stop(self.root)
            started = time.monotonic()
            c = self.cases[cid]
            par = parents[cid]
            raw, t, x, v, e = load_arrays(self.root, c, par)
            mask = np.zeros(len(x), bool)
            for a, b, _ in rows:
                mask[a:b] = True
            rr = intervals(mask & v & ~e)
            if not rr:
                continue
            authorized_plans[cid] = self.plan(
                cid,
                par,
                label + "-" + cid,
                rr,
                [dict(op="mark_invalid", ranges_samples=rr, parameters={"member_hits": [hid for _, _, hid in rows]})],
            )
            session = ProcessingSession(
                x, fs=c["fs"], channel=c["task"], unit=c["unit"], source_id=par["content_hash"], missing=~v | e
            )
            state = session.process(session.raw.id, "mark_invalid", rr)
            recipe = {"hash": digest(["dataset.mark_invalid", rows])}
            after[cid] = evaluate_output(
                self.root,
                c,
                recipe,
                self.implementation,
                par,
                raw,
                t,
                x,
                state.values,
                np.where(mask & v & ~e, state.valid, v),
                e.copy(),
                state.operations,
                started,
            )
        return self.gate(label, parents, after, authorized_plans) if requested else parents.copy()

    def scan_version(self, versions):
        data = self.loader(versions)
        cases = list(self.cases.values())
        self.scan_coverage = {}
        hits = discover(
            self.root,
            cases,
            data,
            self.profile,
            self.config["inspection_preset"],
            self.config.get("secondary_probe", False),
            self.scan_coverage,
        )
        if self.reviewer.enabled and not self.blocked:
            protected = self.path / "coarse-protected.json"
            seen = set()
            for z in read(protected) if protected.exists() else []:
                cid, a, b = z["case"], z["a"], z["b"]
                if (cid, a, b) in seen:
                    continue
                seen.add((cid, a, b))
                _, _, v, e, _ = data(cid)
                if (v[a:b] & ~e[a:b]).all():
                    hits.append(
                        dict(
                            id=digest([cid, a, b, "protected_landmark"])[:24],
                            case=cid,
                            task=self.cases[cid]["task"],
                            a=a,
                            b=b,
                            reason="protected_landmark",
                            support_status="possible_physiology",
                            value=None,
                        )
                    )
            hits.extend(self.isolated_spikes(versions, hits))
        enrich(self.cases, data, self.profile, hits)
        candidates = []
        peers = defaultdict(list)
        for c in self.cases.values():
            peers[acquisition(c)].append(c["id"])
        for h in hits:
            cid = h["case"]
            c = self.cases[cid]
            t, x, v, e, pr = data(cid)
            a, b = h["a"], h["b"]
            left, r = max(0, a - round(5 * c["fs"])), min(len(t), b + round(5 * c["fs"]))
            h["context_digest"] = array_hash(x[left:r], v[left:r], e[left:r])
            peer_context = {}
            for peer in peers[acquisition(c)]:
                if peer == cid:
                    continue
                pt, px, pv, pe, _ = data(peer)
                left, right = np.searchsorted(pt, [t[a] - 5, t[b - 1] + 1 / c["fs"] + 5])
                peer_context[peer] = array_hash(pt[left:right], px[left:right], pv[left:right], pe[left:right])
            h["peer_context_digest"] = digest(peer_context)
            support = pr["detector_support"]
            if h.get("support_status") is None:
                uncertain = any(
                    lo < b and a < hi
                    for k in ("unsupported_ranges", "boundary_uncertain_ranges", "empty_event_ranges")
                    for lo, hi in support[k]
                )
                h["support_status"] = "insufficient_support" if uncertain else "suspected_artifact"
            candidates.append(ledger.source_row(h, c, versions[cid]["content_hash"], t, self.round))
        return hits, candidates

    def prepare(self, scope):
        path = self.path / "discovery.json"
        stamp = self.path / "discovery.sha256"
        if path.exists() and stamp.exists():
            if sha(path) != stamp.read_text():
                raise ValueError("Changed patient discovery checkpoint")
            doc = read(path)
            if doc.get("profile_sha256") != sha(self.path / "patient_profile.json"):
                raise ValueError("Changed patient profile checkpoint")
            if doc["scope"] != scope:
                raise ValueError("Changed discovery scope")
            if doc["complete"]:
                self.prepared_ok = True
                return
        self.profile = patient_profiles(list(self.cases.values()), self.loader(self.raw))
        write(self.path / "patient_profile.json", self.profile)
        parent = self.raw if self.blocked else self.coarse()
        # Failures from a requested detector stop the *global* barrier; identity remains available.
        try:
            hits, candidates = self.scan_version(parent)
            baseline = (
                discover(
                    self.root,
                    list(self.cases.values()),
                    self.loader(self.raw),
                    self.profile,
                    self.config["inspection_preset"],
                    self.config.get("secondary_probe", False),
                )
                if any(parent[k]["content_hash"] != self.raw[k]["content_hash"] for k in parent)
                else [h for h in hits if h["reason"] not in ("protected_landmark", "isolated_spike_candidate")]
            )

        except (ValueError, RuntimeError, ImportError) as exc:
            from .provenance import StopRequested

            if isinstance(exc, StopRequested):
                raise
            self.prepared_ok = False
            self.unresolved.append(dict(stage="inspection", reason=str(exc), record=self.rid))
            hits, candidates, baseline = [], [], []
        coverage = {}
        data = self.loader(parent)
        for cid in self.cases:
            _, _, _, _, pr = data(cid)
            coverage[cid] = dict(
                version=parent[cid]["content_hash"],
                native=pr["detector_support"],
                tools=getattr(self, "scan_coverage", {}).get(cid, {}),
                status="scanned" if self.prepared_ok else "failed",
            )
        write(
            path,
            dict(
                scope=scope,
                parents={cid: a["artifact_id"] for cid, a in parent.items()},
                complete=self.prepared_ok,
                profile_sha256=sha(self.path / "patient_profile.json"),
                coverage=coverage,
                baseline_alerts=dict(Counter(h["reason"] for h in baseline)),
                hits=hits,
                candidates=candidates,
                unresolved=self.unresolved,
                pending=dict(self.pending_ranges),
                spent_views=self.spent_gate_views,
            ),
        )
        stamp.write_text(sha(path))

    def run(self):
        cases = list(self.cases.values())
        doc = read(self.path / "discovery.json")
        self.profile = read(self.path / "patient_profile.json")
        self.pending_ranges = defaultdict(list, doc["pending"])
        self.spent_gate_views = doc["spent_views"]
        self.unresolved = list(doc["unresolved"])
        parent = {cid: read(self.root / "artifacts" / aid / "artifact.json") for cid, aid in doc["parents"].items()}
        data = self.loader(parent)
        hits = doc["hits"]
        if self.blocked:
            self.unresolved.append(dict(stage="candidate_freeze_pending", reason="incomplete_required_inspection"))
            for cid, c in self.cases.items():
                self.pending_ranges[cid].append((0, c["n"]))
        groups = self.groups(hits, data)
        if not self.blocked:
            self.phase("repair_running")
        final = self.spike_trials(self.filter_trials(self.fine(parent, hits), hits), hits)
        after = self.loader(final)
        self.phase("final_rescan")
        self.round = 2
        residual, current = self.scan_version(final) if not self.blocked else (hits, doc["candidates"])
        diff = ledger.difference(doc["candidates"], current)
        reopened = set(diff["new"] + diff["invalidated"])
        first_trials = [r for name in ("spike-trials", "filter-trials") for r in read(self.path / (name + ".json"))]
        rejected = {r["hit"]: r for r in first_trials if r["status"] == "rollback"}
        first_gate = self.path / "fine-gate.json"
        if first_gate.is_file():
            mask_gates = read(first_gate)
            for h in hits:
                g = mask_gates.get(h["case"])
                if h["wave_requested"] == "mark_invalid" and g and not g["accepted"]:
                    rejected[h["id"]] = {"reason": "mask_review_or_gate_declined"}
        reopened.update(rejected)
        release = {h["id"]: h for h in hits if h["id"] not in set(diff["resolved"])}
        todo = [h for h in residual if h["id"] in reopened]
        for h in todo:
            if h["id"] in rejected:
                h["feedback"] = rejected[h["id"]].get("reason")
        history = [dict(revision=2, changes=diff)]
        if todo and not self.blocked:
            self.revision = digest(
                [self.revision, [(c["candidate_id"], c["evidence_digest"], c["source_version_id"]) for c in current]]
            )
            write(
                self.path / "candidate-revision-2.json",
                dict(id=self.revision, candidates=current, targets=[h["id"] for h in todo]),
            )
            self.groups(todo, after)
            revised = self.spike_trials(self.filter_trials(self.fine(final, todo), todo), todo)
            release.update((h["id"], h) for h in todo)
            if any(revised[cid]["artifact_id"] != final[cid]["artifact_id"] for cid in final):
                final = revised
                after = self.loader(final)
                self.round = 3
                residual, next_rows = self.scan_version(final)
                change = ledger.difference(current, next_rows)
                history.append(dict(revision=3, changes=change))
                for hid in change["resolved"]:
                    release.pop(hid, None)
                stale = set(change["new"] + change["invalidated"])
                for h in residual:
                    if h["id"] in stale:
                        self.pending_ranges[h["case"]].append((h["a"], h["b"]))
                        h.update(
                            group="final-rescan",
                            wave_requested="preserve",
                            measurement="review",
                            decision_source="unreviewed",
                            state="need_evidence",
                        )
                        self.unresolved.append(dict(stage="final_rescan", hit=h["id"], reason="rework_limit"))
                        release[h["id"]] = h
                current = next_rows
        for cid in self.cases:
            support = after(cid)[4]["detector_support"]
            for k in ("unsupported_ranges", "boundary_uncertain_ranges", "empty_event_ranges"):
                self.pending_ranges[cid].extend(support[k])
        release_hits = list(release.values())
        write(
            self.path / "rescan.json",
            dict(
                revision=self.round,
                revision_id=digest(current),
                changes=diff,
                candidates=current,
                history=history,
                coverage=getattr(self, "scan_coverage", {}),
                reassessed=[h["id"] for h in todo],
                status="pending" if self.blocked else "executed",
            ),
        )
        joints, controls = locations(cases, after, release_hits, uncertain_only=True)
        remaining = []
        first_pass = []
        for index, loc in enumerate(joints):
            onset, stop = loc["episode_start"], loc["episode_end"]
            slots = [
                dict(loc, start=float(a), end=min(float(a) + 10, stop), event=index)
                for a in np.arange(onset, stop, 10.0)
            ]
            focal = min((loc["start"] + loc["end"]) / 2, stop - 1e-8)
            best = next((i for i, tile in enumerate(slots) if tile["start"] <= focal < tile["end"]), 0)
            first_pass.append(slots.pop(best))
            remaining.extend(slots)
        tiles = first_pass + remaining
        selected = tiles[: self.config["joint_views"]]
        chosen = {loc["event"] for loc in selected}
        places = [("joint", x) for x in selected] + [("control", x) for x in controls]
        ranks = {cid: np.zeros(c["n"], np.int8) for cid, c in self.cases.items()}
        for h in release_hits:
            rank = ranks[h["case"]][h["a"] : h["b"]]
            np.maximum(rank, QUALITIES.index(h["measurement"]), out=rank)
            cid = h["case"]
            a, b = h["a"], h["b"]
            _, x, v, e, _ = data(cid)
            _, y, v2, e2, _ = after(cid)
            h.update(
                final_fine_modified=int(changed(x[a:b], y[a:b]).sum()),
                final_fine_excluded=int(((v[a:b] & ~e[a:b]) & ~(v2[a:b] & ~e2[a:b])).sum()),
            )
        requests = []
        locs = {}
        for i, (kind, loc) in enumerate(places):
            # Tile sustained events and merge overlapping decisions at the native measurement key.
            first = loc["start"]
            last = loc["end"]
            for j, t0 in enumerate(np.arange(first, last, 10.0)):
                tile = dict(loc, start=float(t0), end=min(last, float(t0) + 10))
                key = f"{self.rid}-{kind}-{i:03}-{j:02}"
                pic, metrics = self.picture(key, tile, data, after)
                measurements = [z for k, v in metrics.items() if k.startswith("after_") for z in v]
                requests.append(
                    dict(
                        id=key,
                        images=[pic],
                        data=dict(
                            keys=[z["key"] for z in measurements],
                            native=metrics,
                            scope=kind,
                            version={
                                cid: a["content_hash"]
                                for cid, a in final.items()
                                if acquisition(self.cases[cid]) == acquisition(self.cases[loc["case"]])
                            },
                        ),
                    )
                )
                locs[key] = tile
        direct = self.reviewer.decide("joint", requests) if not self.blocked else {}
        overrides = defaultdict(list)
        for r in requests:
            key = r["id"]
            d = direct.get(key)
            if not self.reviewer.enabled:
                continue
            if d is None:
                self.unresolved.append(dict(stage="joint", id=key, reason="unreviewed"))
                continue
            exceptions = {z["key"]: z["quality"] for z in d["exceptions"]} if d else {}
            for k in r["data"]["keys"]:
                overrides[k].append((key, exceptions.get(k, d["default"] if d else "review")))
        write(self.path / "direct-review.json", dict(locations=locs, decisions=direct, ranked_joint_episodes=joints))
        trials = {r["hit"]: r for name in ("spike-trials", "filter-trials") for r in read(self.path / (name + ".json"))}
        gate_path = self.path / "fine-gate.json"
        gates = read(gate_path) if gate_path.exists() else {}
        for h in hits:
            if "final_fine_excluded" not in h:
                cid, a, b = h["case"], h["a"], h["b"]
                _, x, v, e, _ = data(cid)
                _, y, v2, e2, _ = after(cid)
                h.update(
                    final_fine_modified=int(changed(x[a:b], y[a:b]).sum()),
                    final_fine_excluded=int(((v[a:b] & ~e[a:b]) & ~(v2[a:b] & ~e2[a:b])).sum()),
                )
            h["state"] = ledger.disposition(h, trials, gates).value
        revised_trials = {r["hit"]: r for f in self.path.glob("*-trials-rescan.json") for r in read(f)}
        last_gate = self.path / "fine-gate-rescan.json"
        for h in release_hits:
            if "state" not in h:
                h["state"] = ledger.disposition(h, revised_trials, read(last_gate) if last_gate.exists() else {}).value
        write(self.path / "release-dispositions.json", release_hits)
        write(self.path / "hit-dispositions.json", hits)
        related = defaultdict(lambda: defaultdict(list))
        for h in release_hits:
            related[h["case"]][h["group"]].append((h["a"], h["b"]))
        unviewed = defaultdict(list)
        unchecked_tiles = tiles[len(selected) :] + [locs[k] for k in locs if "-joint-" in k and k not in direct]
        for loc in unchecked_tiles:
            for c in cases:
                if acquisition(c) == acquisition(self.cases[loc["case"]]):
                    t = data(c["id"])[0]
                    a, b = t.searchsorted([loc["start"] - 1e-7, loc["end"] - 1e-7])
                    unviewed[c["id"]].append((a, b))
        measures = []
        waves = {}
        for cid, c in self.cases.items():
            artifact = final[cid]
            t, y, v, e, pr = after(cid)
            members = []
            covers = {}
            for gid, ranges in related[cid].items():
                m = np.zeros(len(t), bool)
                for lo, hi in ranges:
                    m[lo:hi] = True
                covers[gid] = m
            for a, b, row in native_rows(pr):
                key = f"{cid}:{a}:{b}"
                ds = overrides.get(key, [])
                level = max(QUALITIES.index(st) for _, st in ds) if ds else int(ranks[cid][a:b].max())
                pending = any(lo < b and a < hi for lo, hi in self.pending_ranges[cid])
                if pending:
                    level = max(level, 1)
                if not (v[a : b + 1] & ~e[a : b + 1]).all():
                    raise ValueError("Native measurement crosses unavailable support")
                record = dict(
                    case=cid,
                    channel=c["task"],
                    a=a,
                    b=b,
                    start_s=float(t[a]),
                    end_s=float(t[b]),
                    quality=QUALITIES[level],
                    groups=" ".join(g for g, m in covers.items() if m[a:b].any()),
                    direct_ids=" ".join(i for i, _ in ds),
                    waveform_version=artifact["content_hash"],
                )
                record.update(
                    row,
                    pending_evidence=pending,
                    joint_view_unchecked=any(lo < b and a < hi for lo, hi in unviewed[cid]),
                )
                members.append(record)
            measures.extend(members)
            waves[cid] = dict(
                record=c["record_id"],
                channel=c["task"],
                artifact_id=artifact["artifact_id"],
                source_sha256=c["input_sha256"],
                waveform=artifact["arrays"] or f"public/{cid}/input.npz",
                probe=artifact["probe_path"],
                probe_sha256=artifact["probe_sha256"],
                content_hash=artifact["content_hash"],
                valid_ranges=intervals(v),
                estimated_ranges=intervals(e),
                trusted_support=intervals(v & ~e),
                bounds=[0, c["n"]],
                measurements=dict(Counter(row["quality"] for row in members)),
                modified_samples=artifact["stats"]["modified_samples"],
            )
        report = dict(
            record=self.rid,
            baseline_alerts=doc["baseline_alerts"],
            blocks=len(cases),
            hits=len(hits),
            groups=len(groups),
            rescan_diff={k: len(v) for k, v in diff.items()},
            joint_episodes=len(joints),
            joint_selected=len(chosen),
            joint_view_count=len(selected),
            joint_views_pending=len(tiles) - len(selected),
            unselected_joint=len(joints) - len(chosen),
            direct_views=len(requests),
            unresolved=self.unresolved,
            reference_accuracy=None,
            recheck_by_reason=dict(Counter(r["reason"] for r in residual)),
            measurement_states=dict(Counter(r["quality"] for r in measures)),
        )
        write(self.path / "report.json", report)
        return final, waves, measures, report


def export_atlas(target, cases, raw, final, measures):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    per_case = defaultdict(list)
    for r in measures:
        if r["quality"] != "keep":
            per_case[r["case"]].append(r)
    blocks = defaultdict(list)
    limits = {}
    for c in cases:
        cid = c["id"]
        blocks[acquisition(c)].append(c)
        _, x, _, _, _ = raw(cid)
        _, y, _, _, _ = final(cid)
        vals = np.r_[x[np.isfinite(x)], y[np.isfinite(y)]]
        if len(vals):
            lo, hi = limits.get(c["task"], (np.inf, -np.inf))
            limits[c["task"]] = (min(lo, float(vals.min())), max(hi, float(vals.max())))
    sample_count = Counter()
    pages = 0
    tmp = target.with_suffix(".tmp")
    with PdfPages(tmp) as pdf:
        for block in blocks.values():
            block.sort(key=lambda c: ("ECG", "PPG", "ABP").index(c["task"]))
            lo = min(float(raw(c["id"])[0][0]) for c in block)
            end = max(float(raw(c["id"])[0][-1]) + 1 / c["fs"] for c in block)
            for start in np.arange(lo, end - 1e-7, 30):
                fig = Figure(figsize=(12, 2.0 * len(block) + 1.25), layout="constrained")
                FigureCanvasAgg(fig)
                axes = fig.subplots(len(block), 2, sharex=True, sharey="row", squeeze=False)
                for i, c in enumerate(block):
                    cid = c["id"]
                    fs = c["fs"]
                    for j, get in enumerate((raw, final)):
                        t, x, v, e, _ = get(cid)
                        a, b = t.searchsorted([start - 1e-7, start + 30 - 1e-7])
                        ax = axes[i, j]
                        ax.plot(t[a:b] - start, x[a:b], lw=0.68, color="#286381" if j == 0 else "#087469")
                        if j:
                            sample_count[cid] += int(b - a)
                            for r in per_case[cid]:
                                if r["a"] < b and a < r["b"]:
                                    ax.axvspan(
                                        max(t[r["a"]] - start, 0),
                                        min(30, t[r["b"]] - start),
                                        ymin=0.88,
                                        ymax=0.99,
                                        color="#dba436" if r["quality"] == "review" else "#bc4632",
                                        lw=0,
                                    )
                        for left, h in intervals(~v[a:b] | e[a:b]):
                            ax.axvspan(
                                t[a + left] - start,
                                t[a + h - 1] + 1 / fs - start,
                                facecolor="#84919a22",
                                edgecolor="#71818d",
                                hatch="///",
                                lw=0.2,
                            )
                        ymin, ymax = limits.get(c["task"], (-1, 1))
                        pad = max((ymax - ymin) * 0.05, 1e-6)
                        ax.set(xlim=(0, 30), ylim=(ymin - pad, ymax + pad))
                        if j == 0:
                            ax.set_ylabel(c["task"] + "\n" + c["unit"])
                        if i == 0:
                            ax.set_title("Source" if j == 0 else "Released")
                fig.suptitle(
                    f"{block[0]['record_id']}   |   source time {start:.3f} s  |  "
                    "gold: review  red: withheld  hatch: unavailable"
                )
                axes[-1, 0].set_xlabel("Seconds within page")
                pdf.savefig(fig)
                fig.clear()
                pages += 1
    if any(sample_count[c["id"]] != c["n"] for c in cases):
        raise ValueError("Incomplete atlas coverage")
    tmp.replace(target)
    return pages


def run_dataset(root, config, stage):
    started = time.monotonic()
    root = Path(root).resolve()
    manifest = read(root / "dataset_manifest.json")
    implementation = implementation_hash()
    bank = {}
    for name, vid in read(root / "baselines.json").items():
        version = read(root / "versions" / vid / "version.json")
        if not version["report"]["program_checks_passed"]:
            raise ValueError(f"Incomplete {name} baseline")
        bank[name] = version["results"]
    records = defaultdict(list)
    for c in manifest["cases"]:
        records[c["record_id"]].append(c)
    knowledge = Knowledge(
        config.get("knowledge_root")
        or (root.parent / ".signalmind-knowledge" if config["method"] == "dataset_agent" else root / "knowledge")
    )
    snapshot = knowledge.freeze(root, config.get("replay_from"))
    scope = digest(
        [
            implementation,
            snapshot["id"],
            SPEC_ID,
            manifest,
            {
                k: v
                for k, v in config.items()
                if k not in ("budget_root", "replay_from", "receipt_seed", "knowledge_root")
            },
        ]
    )
    spec = root / "dataset-stage.json"
    if spec.exists() and read(spec)["scope"] != scope:
        raise ValueError("Dataset configuration changed; use a fresh workspace")
    write(spec, dict(scope=scope))
    write(root / "acceptance-spec.json", dict(id=SPEC_ID, checks=ACCEPTANCE, implementation=implementation))
    results = {}
    waves = {}
    record_reports = []
    complete_rows = []
    from contextlib import nullcontext

    from tsmcp.inspection.run import inspect

    stage("inspection_running")
    inspection = inspect(
        root, root / "inspection", config["inspection_preset"], config.get("inspection_wall_seconds", 3600)
    )
    rev = Reviewer(root, config)
    rev.knowledge = knowledge
    rev.snapshot = snapshot
    stage("candidate_freeze_pending")
    with rev.session() if inspection["inspection_complete"] else nullcontext(rev):
        stages = []
        for rid, cases in sorted(records.items()):
            check_stop(root)
            r = Record(root, cases, bank, config, rev, implementation)
            r.set_stage = stage
            r.blocked = not inspection["inspection_complete"]
            rev.record = rid
            r.prepare(scope)
            stages.append(r)
        frozen = ledger.freeze(root, stages, scope, inspection)
        stage("candidate_frozen" if frozen["status"] == "candidate_frozen" else "candidate_freeze_pending")
        for r in stages:
            rid = r.rid
            cases = list(r.cases.values())
            rev.record = rid
            r.revision = frozen["revision_id"]
            r.blocked = frozen["status"] != "candidate_frozen"
            check_stop(root)
            stage("dataset_record")
            checkpoint = r.path / "checkpoint.json"
            if checkpoint.exists():
                cp = read(checkpoint)
                if cp["scope"] != scope or any(sha(r.path / file) != h for file, h in cp["files"].items()):
                    raise ValueError("Changed patient checkpoint")
                if any(sha(root / "dataset-receipts" / (k + ".json")) != v for k, v in cp["receipts"].items()):
                    raise ValueError("Patient decision receipt changed")
                final = {cid: read(root / "artifacts" / a / "artifact.json") for cid, a in cp["artifacts"].items()}
                verified = r.loader(final)
                for cid in final:
                    verified(cid)
                wm = read(r.path / "waveforms.json")
                meas = read(r.path / "measurements.json")
                rep = read(r.path / "report.json")
            else:
                final, wm, meas, rep = r.run()
                if config.get("atlas", True):
                    rep["atlas_pages"] = export_atlas(
                        r.path / "atlas-30s.pdf", cases, r.loader(r.raw), r.loader(final), meas
                    )
                write(r.path / "waveforms.json", wm)
                write(r.path / "measurements.json", meas)
                write(r.path / "report.json", rep)
                files = {
                    str(p.relative_to(r.path)): sha(p)
                    for p in sorted(r.path.rglob("*"))
                    if p.is_file() and p != checkpoint
                }
                used_receipts = sorted(
                    {
                        d["receipt"]
                        for fn in ("coarse-selection", "repair-plans", "repair-plans-rescan")
                        if (r.path / (fn + ".json")).exists()
                        for d in read(r.path / (fn + ".json")).values()
                    }
                )
                for name in ("coarse-gate", "fine-gate", "fine-gate-rescan"):
                    if (r.path / (name + ".json")).exists():
                        used_receipts.extend(
                            d["receipt"]
                            for item in read(r.path / (name + ".json")).values()
                            for d in item["ranges"].values()
                            if "receipt" in d
                        )
                used_receipts.extend(d["receipt"] for d in read(r.path / "direct-review.json")["decisions"].values())
                used_receipts.extend(d["receipt"] for d in read(r.path / "spike-trials.json") if "receipt" in d)
                used_receipts.extend(
                    d["receipt"] for f in r.path.glob("*-trials*.json") for d in read(f) if "receipt" in d
                )
                # Include upstream blind observations, expanded-view and rework receipts in the checkpoint.
                for key in list(set(used_receipts)):
                    saved = read(root / "dataset-receipts" / (key + ".json"))
                    for item in saved.get("evidence", []):
                        for field in ("blind_observation", "diagnostic_receipt"):
                            upstream = item[1].get(field, {}).get("receipt")
                            if upstream:
                                used_receipts.append(upstream)
                        if item[1].get("prior_review"):
                            used_receipts.append(item[1]["prior_review"])
                receipt_hashes = {k: sha(root / "dataset-receipts" / (k + ".json")) for k in sorted(set(used_receipts))}
                if not rep["unresolved"]:
                    write(
                        checkpoint,
                        dict(
                            scope=scope,
                            artifacts={cid: a["artifact_id"] for cid, a in final.items()},
                            files=files,
                            receipts=receipt_hashes,
                        ),
                    )
            results.update(final)
            waves.update(wm)
            complete_rows.extend(meas)
            record_reports.append(rep)
            emit(root, "dataset_record_completed", record=rid, blocks=len(final), groups=rep["groups"])
    export = root / "export"
    export.mkdir(exist_ok=True)
    assert set(results) == {c["id"] for c in manifest["cases"]}
    report = aggregate(manifest["cases"], results, [])
    vid = "v_" + digest(["dataset_released", scope, {k: a["artifact_id"] for k, a in results.items()}])[:20]
    recipe = dict(kind="dataset_release", hash=scope, category="dataset")
    write(root / "recipes" / scope / "recipe.json", recipe)
    write(
        root / "versions" / vid / "version.json",
        dict(
            version_id=vid,
            parent_version=None,
            recipe=recipe,
            stage="released",
            case_ids=list(results),
            results=results,
            implementation_hash=implementation,
            report=report,
            elapsed_s=None,
        ),
    )
    write(
        root / "agent_state.json",
        dict(
            accepted={k: dict(version_id=vid, artifact_id=a["artifact_id"]) for k, a in results.items()},
            accepted_tasks={t: vid for t in {c["task"] for c in manifest["cases"]}},
            reviews={},
            actions=[],
            evidence={},
        ),
    )
    write(export / "waveform_manifest.json", waves)
    fields = [
        "case",
        "channel",
        "a",
        "b",
        "start_s",
        "end_s",
        "quality",
        "groups",
        "pending_evidence",
        "joint_view_unchecked",
        "direct_ids",
        "waveform_version",
        "interval_s",
        "sbp",
        "dbp",
        "mean_pressure",
        "unit",
        "support_id",
    ]
    csv_table(export / "measurements.csv", complete_rows, fields)
    csv_table(export / "usable_measurements.csv", (r for r in complete_rows if r["quality"] == "keep"), fields)
    ids = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    calls = {r["request_id"]: r for r in ids if r["type"] == "model_call_completed"}
    status = (
        "partial"
        if manifest.get("audit_failures")
        or manifest.get("unavailable_tasks")
        or any(r["unresolved"] for r in record_reports)
        else "completed"
    )
    summary = dict(
        mode=config["method"],
        status=status,
        scope=scope,
        records=record_reports,
        scope_cases=len(results),
        tasks=report["tasks"],
        purpose=config["purpose"],
        unavailable_tasks=manifest.get("unavailable_tasks", []),
        audit_failures=manifest.get("audit_failures", []),
        model_calls=len(calls),
        estimated_ds_usd=sum(r["cost_usd"] for r in calls.values()),
        replayed=bool(config.get("replay_from")),
        jev_usage=read(root / "jev_usage.json") if (root / "jev_usage.json").exists() else None,
        reference_accuracy=None,
    )
    if config.get("budget_root") and config["method"] == "dataset_agent" and not config.get("replay_from"):
        external = read(Path(config["budget_root"]) / "api_budget.json")
        numbers = {e["call_number"] for e in ids if e["type"] == "model_call_started"}
        own = [c for c in external["calls"] if c["number"] in numbers]
        write(
            root / "api_budget.json",
            dict(
                model=config["model"],
                cap_usd=config["api_cap"],
                estimated_peak_usd=summary["estimated_ds_usd"],
                unknown_charge_reserve_usd=sum(c["reserve_usd"] for c in own if c["status"] != "returned"),
                actual_billed_usd=None,
                calls=own,
            ),
        )
    summary.update(
        outcome=outcome(status),
        candidate_revision=frozen["revision_id"],
        discovery_status=frozen["status"],
        acceptance_spec_id=SPEC_ID,
    )
    state = read(root / "agent_state.json")
    state.update(outcome(status), outcome=outcome(status))
    write(root / "agent_state.json", state)
    knowledge.record(
        "shared",
        dict(
            scope=scope,
            release=vid,
            acceptance_spec=SPEC_ID,
            coverage=summary["discovery_status"],
            reference_accuracy=None,
            evidence_ref=digest(waves),
            kind="run_outcome",
        ),
    )
    summary["candidate_ledger"] = ledger.release_index(root, stages)
    release = Run(
        scope=scope,
        implementation=implementation,
        skill_snapshot=snapshot["id"],
        acceptance_spec_id=SPEC_ID,
        candidate_revision=frozen["revision_id"],
        records=sorted(records),
        outcome=outcome(status),
    )
    write(root / "ledger/run.json", release.model_dump())
    from .provenance import dataset_telemetry

    write(export / "telemetry.json", dataset_telemetry(manifest, ids, calls, time.monotonic() - started, summary))
    write(export / "evaluation_report.json", summary)
    return summary
