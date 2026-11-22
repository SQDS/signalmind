"""Whole-dataset orchestration shared by the workbench and CLI. Artifacts stay in the run workspace."""

from __future__ import annotations

import csv
import json
import operator
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from tsmcp.processing import ProcessingSession, intervals

from .common import array_hash, changed, digest, implementation_hash, read, sha, write
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
from .dataset_review import FIELDS, QUALITIES, Reviewer
from .engine import aggregate, evaluate_output, load_arrays
from .provenance import check_stop, emit
from .tasks import probe

OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


def applies(conditions, features):
    return all(features[c["field"]] is not None and OPS[c["op"]](features[c["field"]], c["value"]) for c in conditions)


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

    def isolated_spikes(self, parents):
        """Recover individual bounded excursions from a coarse recipe that was declined for the whole block."""
        current = self.loader(parents)
        original = self.loader(self.raw)
        hits = []
        for cid, c in self.cases.items():
            for op in self.bank["fixed_dsp"][cid]["operations"]:
                if op.get("parameters", {}).get("source") != "public_discontinuous_excursion_rule":
                    continue
                _, x, v, e, pr = current(cid)
                raw = original(cid)[1]
                peaks = [r["sample"] for r in pr["events"]]
                peaks.extend(r["start_sample"] for r in pr["pressures"])
                if pr["pressures"]:
                    peaks.append(pr["pressures"][-1]["stop_sample"])
                peaks = np.asarray(peaks)
                margin = round(0.08 * c["fs"])
                for a, b in op["ranges_samples"]:
                    # The frozen excursion detector uses four quiet anchor samples on either side.
                    if a < 4 or b + 4 > len(x) or b - a > min(4, 5 * c["fs"]):
                        continue
                    win = slice(a - 4, b + 4)
                    if not (v[win] & ~e[win]).all() or not np.array_equal(raw[win], x[win]):
                        continue
                    if ((peaks >= a - margin) & (peaks < b + margin)).any():
                        continue
                    reason = "isolated_spike_candidate"
                    hits.append(
                        dict(
                            id=digest([cid, a, b, reason])[:24],
                            case=cid,
                            task=c["task"],
                            a=a,
                            b=b,
                            reason=reason,
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
            if not (v[a - 1 : b + 1] & ~e[a - 1 : b + 1]).all():
                rec["reason"] = "core_or_anchor_no_longer_observed"
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
            if not passed:
                rec.update(status="rollback", reason="local_waveform_or_landmark_retest")
            elif self.spent_gate_views >= self.config["write_views"]:
                rec.update(status="pending", reason="write_view_budget")
            else:
                key = "spike-" + h["id"]
                before = self.loader(accepted)
                after = self.loader(accepted | {cid: candidate})
                loc = self.case_span(cid, a, b, before)
                loc["context_s"] = 0.45
                pic, metrics = self.picture(key, loc, before, after)
                self.spent_gate_views += 1
                rec["image"] = str(pic.relative_to(self.path))
                d = self.reviewer.decide(
                    "gate",
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
                            ),
                        )
                    ],
                ).get(key)
                if d:
                    rec.update(
                        status="accepted" if d["decision"] == "accept" else "rollback",
                        reason=d["reason"],
                        receipt=d["receipt"],
                    )
                    if d["decision"] == "accept":
                        accepted[cid] = candidate
                else:
                    rec.update(status="pending", reason="write_review_unavailable")
            if rec["status"] == "pending":
                self.unresolved.append(dict(stage="spike_trial", case=cid, hit=h["id"], reason=rec["reason"]))
            if rec["status"] != "accepted" and h["measurement"] == "keep":
                h["measurement"] = "review"
        write(self.path / "spike-trials.json", log)
        return accepted

    def gate(self, label, parents, candidates):
        before = self.loader(parents)
        after = self.loader(candidates)
        rows = []
        needed = defaultdict(list)
        result = parents.copy()
        log = {}
        for cid in self.cases:
            if parents[cid]["content_hash"] == candidates[cid]["content_hash"]:
                continue
            check_stop(self.root)
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
                image, metrics = self.picture(key, loc, before, after)
                rows.append(
                    dict(
                        id=key,
                        images=[image],
                        data=dict(
                            task=self.cases[cid]["task"],
                            changed_ranges=spans,
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
        decisions = self.reviewer.decide("gate", rows)
        for cid, keys in needed.items():
            accepted = all(decisions.get(k, {}).get("decision") == "accept" for k in keys)
            if accepted:
                result[cid] = candidates[cid]
            if any(k not in decisions for k in keys):
                self.pending_ranges[cid].extend(change_ranges(before(cid), after(cid)))
                self.unresolved.append(dict(stage=label, case=cid, reason="write_review_pending"))
            log[cid] = dict(
                parent_hash=parents[cid]["content_hash"],
                candidate_hash=candidates[cid]["content_hash"],
                accepted=accepted,
                ranges={k: decisions.get(k, {"decision": "pending"}) for k in keys},
            )
        write(self.path / (label + ".json"), log)
        return result

    def coarse(self):
        if not self.reviewer.enabled:
            return {cid: self.bank["rule_adaptive_dsp"][cid] for cid in self.cases}
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
        scores = jev_routes(self.root, self.config, hits) if self.config["routing"] == "jev" else {}
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
            gid = f"{self.rid}-{task}-{reason}-{route}"
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
            examples = [members[j] for j in reps]
            images = []
            for i, h in enumerate(examples):
                loc = hit_location(h, self.cases, data)
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
                columns.extend(["left_anchor", "core_min", "core_max", "right_anchor"])
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
                ]
                for h in members
            ]
            requests.append(
                dict(
                    id=gid,
                    images=images,
                    data=dict(
                        task=task,
                        trigger=reason,
                        route=route,
                        scope=len(members),
                        summary=stats,
                        membership=digest(members),
                        cases=[
                            dict(id=cid, fs=self.cases[cid]["fs"], unit=self.cases[cid]["unit"]) for cid in case_ids
                        ],
                        columns=columns,
                        member_rows=table,
                        image_rows=reps,
                        image_roles=["typical", "detail" if len(members) == 1 else "boundary"],
                        purpose=self.config["purpose"],
                    ),
                )
            )
            if reason == "isolated_spike_candidate":
                evidence = requests[-1]["data"]
                evidence["core_samples_by_member"] = [data(h["case"])[1][h["a"] : h["b"]].tolist() for h in members]
                evidence["plot_key"] = (
                    "Both columns show observations. Left: beat context. Right: sampled points at the shaded core "
                    "and its immediate anchors. Compare the core shape against neighbouring beats. "
                    "Null means unavailable; reason routing has no JEV score."
                )
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
                )
            )
        write(self.path / "groups.json", rosters)
        plans = self.reviewer.decide("groups", requests)
        write(self.path / "repair-plans.json", plans)
        for h in hits:
            plan = plans.get(h["group"])
            if plan is None:
                h.update(
                    wave_requested="preserve",
                    measurement="review",
                    decision_source="unreviewed" if self.reviewer.enabled else "local_rule",
                )
            else:
                f = h["features"]
                h.update(
                    wave_requested=plan["wave"] if applies(plan["wave_if"], f) else "preserve",
                    measurement=plan["measure"] if applies(plan["measure_if"], f) else plan["otherwise"],
                    decision_source=plan["receipt"],
                )
        if self.reviewer.enabled:
            self.unresolved.extend(
                dict(stage="groups", group=r["id"], reason="unreviewed") for r in requests if r["id"] not in plans
            )
        if any(g["route"] == "unrouted" for g in rosters):
            self.unresolved.append(dict(stage="routing", reason="JEV quota exhausted"))
        return rosters

    def fine(self, parents, hits):
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
        return self.gate("fine-gate", parents, after) if requested else parents.copy()

    def run(self):
        cases = list(self.cases.values())
        self.profile = patient_profiles(cases, self.loader(self.raw))
        write(self.path / "patient_profile.json", self.profile)
        baseline_hits = discover(
            self.root,
            cases,
            self.loader(self.raw),
            self.profile,
            self.config["inspection_preset"],
            self.config.get("secondary_probe", False),
        )
        parent = self.coarse()
        data = self.loader(parent)
        hits = (
            discover(
                self.root,
                cases,
                data,
                self.profile,
                self.config["inspection_preset"],
                self.config.get("secondary_probe", False),
            )
            if any(parent[cid]["content_hash"] != self.raw[cid]["content_hash"] for cid in parent)
            else baseline_hits
        )
        if self.reviewer.enabled:
            seen = set()
            for z in read(self.path / "coarse-protected.json"):
                cid = z["case"]
                a, b = z["a"], z["b"]
                if (cid, a, b) in seen:
                    continue
                seen.add((cid, a, b))
                hits.append(
                    dict(
                        id=digest([cid, a, b, "protected_landmark"])[:24],
                        case=cid,
                        task=self.cases[cid]["task"],
                        a=a,
                        b=b,
                        reason="protected_landmark",
                        value=None,
                    )
                )
            hits.extend(self.isolated_spikes(parent))
        enrich(self.cases, data, self.profile, hits)
        groups = self.groups(hits, data)
        final = self.spike_trials(self.fine(parent, hits), hits)
        after = self.loader(final)
        joints, controls = locations(cases, data, hits, uncertain_only=True)
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
        for h in hits:
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
        direct = self.reviewer.decide("joint", requests)
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
        write(self.path / "hit-dispositions.json", hits)
        related = defaultdict(lambda: defaultdict(list))
        for h in hits:
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
        residual = (
            discover(
                self.root,
                cases,
                after,
                self.profile,
                self.config["inspection_preset"],
                self.config.get("secondary_probe", False),
            )
            if any(final[k]["content_hash"] != parent[k]["content_hash"] for k in final)
            else hits
        )
        report = dict(
            record=self.rid,
            baseline_alerts=dict(Counter(h["reason"] for h in baseline_hits)),
            blocks=len(cases),
            hits=len(hits),
            groups=len(groups),
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
    scope = digest(
        [
            implementation,
            manifest,
            {k: v for k, v in config.items() if k not in ("budget_root", "replay_from", "receipt_seed")},
        ]
    )
    spec = root / "dataset-stage.json"
    if spec.exists() and read(spec)["scope"] != scope:
        raise ValueError("Dataset configuration changed; use a fresh workspace")
    write(spec, dict(scope=scope))
    results = {}
    waves = {}
    record_reports = []
    complete_rows = []
    with Reviewer(root, config).session() as rev:
        for rid, cases in sorted(records.items()):
            check_stop(root)
            stage("dataset_record")
            r = Record(root, cases, bank, config, rev, implementation)
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
                        for fn in ("coarse-selection", "repair-plans")
                        if (r.path / (fn + ".json")).exists()
                        for d in read(r.path / (fn + ".json")).values()
                    }
                )
                for name in ("coarse-gate", "fine-gate"):
                    if (r.path / (name + ".json")).exists():
                        used_receipts.extend(
                            d["receipt"]
                            for item in read(r.path / (name + ".json")).values()
                            for d in item["ranges"].values()
                            if "receipt" in d
                        )
                used_receipts.extend(d["receipt"] for d in read(r.path / "direct-review.json")["decisions"].values())
                used_receipts.extend(d["receipt"] for d in read(r.path / "spike-trials.json") if "receipt" in d)
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
    write(export / "evaluation_report.json", summary)
    return summary
