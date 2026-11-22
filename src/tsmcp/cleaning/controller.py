"""Cleaning and follow-up tools and candidate-bound model review; no hidden numerical repair."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from tsmcp.followup.pattern import detect, pattern_schema
from tsmcp.followup.rescan import scan_version

from .common import TASKS, digest, implementation_hash, no_reference, read, sha, write
from .engine import execute_version, load_arrays, register_recipe
from .views import render, select_cases


def tool(name, description, properties=None, required=()):
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties or {}, "required": list(required)},
    }


STRING = {"type": "string"}
TASK = {"type": "string", "enum": list(TASKS)}


class CleaningTools:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.manifest = read(self.workspace / "dataset_manifest.json")
        self.cases = {c["id"]: c for c in self.manifest["cases"]}
        state_path = self.workspace / "agent_state.json"
        self.state = (
            read(state_path)
            if state_path.exists()
            else {
                "reviews": {},
                "evidence": {},
                "reports_seen": [],
                "accepted": {},
                "accepted_tasks": {},
                "candidate_signatures": [],
                "candidate_counts": {},
                "actions": [],
                "complete": False,
            }
        )
        self.schemas = [
            tool("inspect_dataset", "Read full-scope diagnostics, task progress and public cases. No references."),
            tool("list_capabilities", "Read operator and Python recipe contracts plus existing recipes."),
            tool(
                "inspect_cases",
                (
                    "Return actual candidate images and numeric evidence. Omit case_id"
                    " to inspect the required batch categories."
                ),
                {
                    "version_id": STRING,
                    "case_id": STRING,
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "target_samples": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "flank_s": {"type": "integer", "enum": [5, 10, 20, 40]},
                },
                ("version_id",),
            ),
            tool(
                "register_pattern",
                (
                    "Register a recurring morphology seen on the image for automatic scanning after a batch. "
                    "Metric options: adjacent step, running-median residual or second difference. "
                    "Ratios use local p95-p05. quiet_s joins tails; max_seed_s caps contiguous seed duration."
                ),
                {
                    "evidence_id": STRING,
                    "signal_view": {"type": "string", "enum": ["raw", "candidate"]},
                    "seed_samples": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "observed": STRING,
                    "suspected_cause": STRING,
                    "alternative_or_limitation": STRING,
                    "method_explanation": STRING,
                    "rule": pattern_schema(),
                },
                (
                    "evidence_id",
                    "signal_view",
                    "seed_samples",
                    "observed",
                    "suspected_cause",
                    "alternative_or_limitation",
                    "method_explanation",
                    "rule",
                ),
            ),
            tool(
                "get_pattern_matches",
                "Page through residual same-pattern matches for the reviewed batch.",
                {"version_id": STRING, "rule_id": STRING, "offset": {"type": "integer", "minimum": 0}},
                ("version_id", "rule_id"),
            ),
            tool(
                "run_candidate",
                (
                    "Register/select a recipe and execute pilot or batch. A batch requ"
                    "ires an accepted model-reviewed pilot of the same recipe/task."
                ),
                {
                    "task": TASK,
                    "stage": {"type": "string", "enum": ["pilot", "batch"]},
                    "recipe_id": STRING,
                    "source": STRING,
                    "parent_version": STRING,
                    "baseline": {"type": "string", "enum": ["identity"]},
                    "category": {"type": "string", "enum": ["standard_operators", "custom_extension"]},
                    "rationale": STRING,
                },
                ("task", "stage", "rationale"),
            ),
            tool(
                "get_report",
                (
                    "Read actual candidate checks, diagnostics, coverage, unresolved i"
                    "ssues and aggregate; not hidden accuracy."
                ),
                {"version_id": STRING},
                ("version_id",),
            ),
            tool(
                "submit_self_review",
                (
                    "Evaluate THIS candidate's postprocessing evidence; recommend insp"
                    "ect/revise/rollback/accept/stop. Only accepted batches update out"
                    "put pointers."
                ),
                {
                    "version_id": STRING,
                    "action": {"type": "string", "enum": ["inspect", "revise", "rollback", "accept", "stop"]},
                    "evidence_ids": {"type": "array", "items": STRING},
                    "observations": STRING,
                    "remaining_concerns": {"type": "array", "items": STRING},
                },
                ("version_id", "action", "evidence_ids", "observations", "remaining_concerns"),
            ),
            tool(
                "finalize_run",
                (
                    "Export actual accepted waveform references, fresh task outputs an"
                    "d complete accounting. No accepted output is fabricated."
                ),
                {"rationale": STRING},
                ("rationale",),
            ),
        ]

    def save(self):
        write(self.workspace / "agent_state.json", self.state)

    def pending_reviews(self, task=None):
        pending = []
        for action in self.state["actions"]:
            if action["action"] != "run_candidate" or (task and action["task"] != task):
                continue
            vid = action["version_id"]
            review = self.state["reviews"].get(vid)
            if review and review["action"] in {"accept", "revise", "rollback", "stop"}:
                continue
            if self.version(vid)["report"]["program_checks_passed"]:
                pending.append(vid)
        return pending

    def version(self, vid):
        path = self.workspace / "versions" / vid / "version.json"
        if not vid.startswith("v_") or "/" in vid or not path.exists():
            raise ValueError("Unknown version")
        version = read(path)
        if version["implementation_hash"] != implementation_hash():
            raise ValueError("Implementation changed: rerun under the current frozen implementation")
        return version

    def inspect_dataset(self):
        baselines = read(self.workspace / "baselines.json")
        identity = self.version(baselines["identity"])
        summaries = {}
        for task in TASKS:
            cases = [c for c in self.cases.values() if c["task"] == task]
            selected = select_cases(identity, cases)
            summaries[task] = {
                "cases": len(cases),
                "duration_s": sum(c["duration_s"] for c in cases),
                "diagnosis": identity["report"]["tasks"].get(task),
                "representative_cases": selected,
                "accepted_version": self.state["accepted_tasks"].get(task),
            }
        return {
            "patients": self.manifest["patients"],
            "enabled_tasks": [task for task in TASKS if any(c["task"] == task for c in self.cases.values())],
            "task_config": self.manifest.get("task_config"),
            "tasks": summaries,
            "scope_hash": self.manifest["scope_hash"],
            "baseline_versions": baselines,
            "reference_metrics": no_reference(),
            "workflow": (
                "Per task: run pilot -> get_report -> inspect_cases -> submit_self"
                "_review -> run batch -> repeat review -> finalize_run"
            ),
            "missing_channels": self.manifest["unavailable_tasks"],
        }

    def list_capabilities(self):
        return {
            "operations": ["notch", "remove_tone", "lowpass", "mark_invalid"],
            "local_repair": (
                "category=custom_extension adds interpolate (estimated samples excluded from probes) and ECG-only "
                "local_detrend (window_s=0.18 to 0.8, observed samples, bilateral anchors, cosine-tapered offset). "
                "Retest QRS timing, morphology, and localized baseline after an actual trial."
            ),
            "known_lowpass_candidates_hz": {"ECG": 40, "PPG": 12, "ABP": 18},
            "parameters": (
                "notch/remove_tone/lowpass: frequency_hz; notch: quality_factor; l"
                "owpass: filter_order; remove_tone: window_s"
            ),
            "public_api": {
                "ctx": "ctx.task, ctx.fs, ctx.metadata, ctx.values (read-only parent), ctx.current, ctx.diagnosis",
                "select": (
                    "ctx.select('flat', min_duration_s=1.0, pad_samples=2); ctx.select"
                    "('outliers', factor=2, window_s=10); ctx.select('missing'); "
                    "ctx.select('review_rule', method=method, threshold=threshold) reproduces a visual review rule; "
                    "ctx.select('screen_family', family=family) recomputes grouped observed ranges, including "
                    "ecg_baseline_shift, ecg_excursion, step_near_qrs, step_away_from_qrs and abp_pulse_episode. "
                    "These are hypotheses. Check shape and preservation conditions for each member."
                ),
                "apply": (
                    "ctx.apply(operation, [[start_sample, stop_sample], ...], **parame"
                    "ters); indices are zero-based half-open"
                ),
                "run": "def run(ctx): ...; return ctx.current",
                "applies": "Optional def applies(metadata, diagnosis): return bool",
            },
            "example": (
                "def run(ctx):\n    for issue in ctx.diagnosis['issues']:\n        i"
                "f issue['kind'] == 'narrowband' and issue['contrast'] >= 20:\n    "
                "        ctx.apply('remove_tone', [issue['range_samples']], freque"
                "ncy_hz=issue['frequency_hz'], window_s=10)\n    return ctx.current"
                "\n"
            ),
            "diagnostic_note": (
                "Large steps or a low SQI are observations, not proof of noise. EC"
                "G source units are uncalibrated. Preserve ABP DC and real slow tr"
                "ends."
            ),
            "standard_recipe_rule": (
                "Use ctx.apply for numerical waveform writes; no custom replacemen"
                "t arrays or hand-edited events. QRS analysis filtering is not a d"
                "eliverable waveform."
            ),
            "limits": (
                "identity + up to 2 initial new candidates per task; at most two a"
                "dditional evidence-based revisions. No repetitive signatures. API"
                " budget shared globally."
            ),
            "baseline_identity": "Use baseline='identity' if justified by inspection; no forced modifications",
        }

    def run_candidate(
        self,
        task,
        stage,
        rationale,
        recipe_id=None,
        source=None,
        baseline=None,
        category="standard_operators",
        parent_version=None,
    ):
        if task not in TASKS or stage not in {"pilot", "batch"}:
            raise ValueError("Unknown task or stage")
        pending = self.pending_reviews(task)
        if pending:
            raise ValueError(f"Review pending pilot/batch candidates before another execution: {pending}")
        if sum(v is not None for v in (recipe_id, source, baseline)) != 1:
            raise ValueError("Supply exactly one recipe_id, source, or baseline=identity")
        if source is not None:
            recipe = register_recipe(self.workspace, source=source, category=category)
        elif baseline == "identity":
            recipe = register_recipe(self.workspace, kind="identity")
        elif recipe_id:
            if "/" in recipe_id:
                raise ValueError("Invalid recipe id")
            recipe = read(self.workspace / "recipes" / recipe_id / "recipe.json")
        else:
            raise ValueError("Only identity is available as an Agent baseline shortcut")
        selected = [c for c in self.cases.values() if c["task"] == task]
        if not selected:
            raise ValueError("This task has no eligible input channel in the frozen scope")
        if parent_version and self.state["accepted_tasks"].get(task) != parent_version:
            raise ValueError("Parent must be this task's currently accepted batch version")
        signature = digest([task, stage, recipe["hash"], parent_version])
        if signature in self.state["candidate_signatures"]:
            raise ValueError("This exact candidate already ran; inspect its existing version instead")
        if stage == "pilot":
            count = self.state["candidate_counts"].get(task, 0)
            if recipe["kind"] != "identity" and count >= 4:
                raise ValueError("Candidate/revision budget exhausted for this task")
            if count >= 2 and not rationale.strip():
                raise ValueError("Further revision requires new evidence")
            identity = self.version(read(self.workspace / "baselines.json")["identity"])
            ids = {r["case_id"] for r in select_cases(identity, selected)}
            ids.update(self.state.get("reviewer_pilot_cases", {}).get(task, ()))
            selected = [c for c in selected if c["id"] in ids]
            if recipe["kind"] != "identity":
                self.state["candidate_counts"][task] = count + 1
        else:
            matches = [
                r
                for r in self.state["reviews"].values()
                if r["task"] == task
                and r["recipe_hash"] == recipe["hash"]
                and r["stage"] == "pilot"
                and r["action"] == "accept"
                and r.get("parent_version") == parent_version
            ]
            if not matches:
                raise ValueError("First inspect and accept a pilot of the same recipe/task")
        version = execute_version(self.workspace, recipe, selected, stage=stage, parent=parent_version)
        self.state["candidate_signatures"].append(signature)
        self.state["actions"].append(
            {
                "action": "run_candidate",
                "version_id": version["version_id"],
                "recipe_hash": recipe["hash"],
                "task": task,
                "stage": stage,
                "rationale": rationale,
            }
        )
        self.save()
        return {
            "version_id": version["version_id"],
            "recipe_id": recipe["hash"],
            "stage": stage,
            "cases_executed": len(version["results"]),
            "errors": version["report"]["errors"][:5],
            "next": "get_report then inspect_cases then submit_self_review",
        }

    def get_report(self, version_id):
        version = self.version(version_id)
        self.state["reports_seen"] = sorted(set(self.state["reports_seen"] + [version_id]))
        self.save()
        scope = [self.cases[c] for c in version["case_ids"]]
        return {
            "version_id": version_id,
            "recipe": version["recipe"],
            "stage": version["stage"],
            **version["report"],
            "review_cases": select_cases(version, scope),
            "recurring_patterns": self.state.get("pattern_rescans", {}).get(version_id, {}),
            "self_review": self.state["reviews"].get(version_id),
            "evidence_basis": "Actual waveform, frozen probes, unlabelled observations. No true accuracy scores.",
        }

    def inspect_cases(self, version_id, case_id=None, start_s=None, end_s=None, target_samples=None, flank_s=5):
        if target_samples is not None and not case_id:
            raise ValueError("A sample interval belongs to one case")
        version = self.version(version_id)
        selected = (
            [{"case_id": case_id, "reasons": ["agent_requested"]}]
            if case_id
            else select_cases(version, [self.cases[c] for c in version["case_ids"]])
        )
        blocks, evidence = [], []
        for item in selected:
            if item["case_id"] not in version["results"]:
                raise ValueError("Case is not part of this candidate")
            e, picture = render(
                self.workspace,
                version,
                self.cases[item["case_id"]],
                item["reasons"],
                start_s,
                end_s,
                item.get("focus_sample"),
                target_samples,
                flank_s,
            )
            self.state["evidence"][e["evidence_id"]] = e
            evidence.append(e)
            blocks.extend([{"type": "text", "text": json.dumps(e, ensure_ascii=False)}, picture])
        self.save()
        return [
            {
                "type": "text",
                "text": json.dumps({"version_id": version_id, "evidence_ids": [e["evidence_id"] for e in evidence]}),
            }
        ] + blocks

    def register_pattern(
        self,
        evidence_id,
        signal_view,
        seed_samples,
        observed,
        suspected_cause,
        alternative_or_limitation,
        method_explanation,
        rule,
    ):
        e = self.state["evidence"][evidence_id]
        c = self.cases[e["case_id"]]
        v = self.version(e["version_id"])
        if signal_view not in {"raw", "candidate"} or any(
            not t.strip() for t in (observed, suspected_cause, alternative_or_limitation, method_explanation)
        ):
            raise ValueError("Specify the observed signal, suspected mechanism, alternative and executable method")
        if sha(self.workspace / e["image"]) != e["image_sha256"]:
            raise ValueError("Review image has changed")
        a, b = seed_samples
        if not e["range_samples"][0] <= a < b <= e["range_samples"][1]:
            raise ValueError("Inspect the seed before registering a rule")
        artifact = v["results"][c["id"]] if signal_view == "candidate" else None
        if sha(self.workspace / "public" / c["id"] / "input.npz") != c["input_sha256"] or (
            artifact and artifact["arrays"] and sha(self.workspace / artifact["arrays"]) != artifact["arrays_sha256"]
        ):
            raise ValueError("Seed waveform has changed")
        _, _, x, valid, est = load_arrays(self.workspace, c, artifact)
        hits = [r for r in detect(x, valid & ~est, c["fs"], rule) if r[0] < b and a < r[1]]
        if not hits:
            raise ValueError("Rule did not reproduce the observed seed")
        rule_id = digest([c["task"], rule, sha(Path(__file__).parents[1] / "followup/pattern.py")])
        decision = dict(
            rule_id=rule_id,
            task=c["task"],
            seed_case=c["id"],
            seed_version="raw-" + c["input_sha256"],
            observed_version_id=e["version_id"] if artifact else "raw-" + c["input_sha256"],
            seed_samples=seed_samples,
            seed_method_hits=hits,
            evidence_id=evidence_id,
            observed=observed,
            suspected_cause=suspected_cause,
            alternative_or_limitation=alternative_or_limitation,
            method_explanation=method_explanation,
            rule=rule,
        )
        self.state.setdefault("registered_patterns", {})[rule_id] = decision
        active = self.state["accepted_tasks"].get(c["task"])
        if active:
            self.state.setdefault("pattern_rescans", {})[active] = self.pattern_sweep(c["task"], self.version(active))
        self.save()
        return decision

    def pattern_sweep(self, task, version):
        reports = {}
        for rid, decision in self.state.get("registered_patterns", {}).items():
            if decision["task"] != task:
                continue
            path = Path("pattern_rescans") / version["version_id"] / rid
            result = scan_version(self.workspace, decision, version, self.workspace / path)
            if not result["complete"]:
                raise ValueError(f"Pattern rescan incomplete: {path}/summary.json")
            reports[rid] = dict(
                report=str(path / "summary.json"),
                findings=str(path / "findings.json"),
                counts=result["counts"],
                scanned=result["scanned"],
            )
        return reports

    def get_pattern_matches(self, version_id, rule_id, offset=0):
        ref = self.state["pattern_rescans"][version_id][rule_id]
        path = self.workspace / ref["findings"]
        if sha(path) != read(self.workspace / ref["report"])["findings_sha256"]:
            raise ValueError("Rescan results have changed")
        items = [
            {"case_id": row["case_id"], "target_samples": r, "fs": row["fs"]}
            for row in read(path)
            for r in row["after_eligible"]
        ]
        return {"total": len(items), "matches": items[offset : offset + 50]}

    def submit_self_review(self, version_id, action, evidence_ids, observations, remaining_concerns):
        version = self.version(version_id)
        if action not in {"inspect", "revise", "rollback", "accept", "stop"}:
            raise ValueError("Unknown self-review action")
        if version_id not in self.state["reports_seen"] or not observations.strip() or not evidence_ids:
            raise ValueError("Read the actual report and candidate images, then describe observed changes")
        evidence = []
        for eid in evidence_ids:
            e = self.state["evidence"][eid]
            if e["version_id"] != version_id or sha(self.workspace / e["image"]) != e["image_sha256"]:
                raise ValueError("Stale or modified image evidence")
            a = version["results"][e["case_id"]]
            if a["content_hash"] != e["content_hash"] or sha(self.workspace / a["probe_path"]) != e["probe_sha256"]:
                raise ValueError("Stale measurement evidence")
            evidence.append(e)
        task = self.cases[version["case_ids"][0]]["task"]
        review = {
            "version_id": version_id,
            "task": task,
            "stage": version["stage"],
            "action": action,
            "parent_version": version["parent_version"],
            "recipe_hash": version["recipe"]["hash"],
            "evidence_ids": evidence_ids,
            "observations": observations,
            "remaining_concerns": remaining_concerns,
            "reference_truth": False,
        }
        old_review = self.state["reviews"].get(version_id)
        if (
            action == "accept"
            and old_review
            and old_review["action"] == "inspect"
            and set(evidence_ids).issubset(old_review["evidence_ids"])
        ):
            raise ValueError("Pending inspection: acquire the additional evidence before accepting")
        if action == "accept":
            if not version["report"]["program_checks_passed"]:
                raise ValueError("Cannot accept failed numerical/contracts checks")
            expected = select_cases(version, [self.cases[c] for c in version["case_ids"]])
            required_views = (
                expected if version["stage"] == "batch" else [p for p in expected if "modified_cases" in p["reasons"]]
            )
            for view in required_views:
                if not any(
                    e["case_id"] == view["case_id"]
                    and e["range_samples"][0] <= view["focus_sample"] < e["range_samples"][1]
                    for e in evidence
                ):
                    raise ValueError(
                        f"Inspect required changed/control location: {view}; support-only edits also require review"
                    )
            if version["stage"] == "batch":
                required = {c["id"] for c in self.cases.values() if c["task"] == task}
                if set(version["case_ids"]) != required:
                    raise ValueError("Batch does not cover this task's complete scope")
                scans = self.pattern_sweep(task, version)
                self.state.setdefault("pattern_rescans", {})[version_id] = scans
                previous = self.state["accepted_tasks"].get(task)
                if previous != version_id:
                    self.state.setdefault("accepted_history", {}).setdefault(task, []).append(previous)
                self.state["accepted"].update(
                    {
                        cid: {"version_id": version_id, "artifact_id": a["artifact_id"]}
                        for cid, a in version["results"].items()
                    }
                )
                self.state["accepted_tasks"][task] = version_id
        if action == "rollback" and self.state["accepted_tasks"].get(task) == version_id:
            history = self.state.setdefault("accepted_history", {}).setdefault(task, [])
            previous = history.pop() if history else None
            self.state["accepted"] = {
                cid: p for cid, p in self.state["accepted"].items() if self.cases[cid]["task"] != task
            }
            self.state["accepted_tasks"].pop(task, None)
            if previous:
                old_version = self.version(previous)
                self.state["accepted"].update(
                    {
                        cid: {"version_id": previous, "artifact_id": a["artifact_id"]}
                        for cid, a in old_version["results"].items()
                    }
                )
                self.state["accepted_tasks"][task] = previous
        self.state["reviews"][version_id] = review
        self.state["actions"].append({**review, "action": "submit_self_review", "review_action": action})
        self.save()
        return {
            **review,
            "accepted_task_versions": self.state["accepted_tasks"],
            "recurring_patterns": self.state.get("pattern_rescans", {}).get(version_id, {}),
            "next": "run same recipe as batch"
            if action == "accept" and version["stage"] == "pilot"
            else "continue or finalize",
        }

    def finalize_run(self, rationale, *, decision_owner="model"):
        if decision_owner == "model":
            pending = self.pending_reviews()
            if pending:
                raise ValueError(
                    f"Every successful candidate needs an explicit SelfReview, including rejected ones: {pending}"
                )
            if any(self.state["reviews"][vid]["action"] != "accept" for vid in self.state["accepted_tasks"].values()):
                raise ValueError("Selected task versions require a final accept review")
        accepted, accounting, unresolved = {}, [], []
        versions = {vid: self.version(vid) for vid in set(self.state["accepted_tasks"].values())}
        residuals = defaultdict(list)
        for task, vid in self.state["accepted_tasks"].items():
            methods = {k for k, d in self.state.get("registered_patterns", {}).items() if d["task"] == task}
            reports = self.state.get("pattern_rescans", {}).get(vid, {})
            if methods - set(reports) and decision_owner == "model":
                reports = self.pattern_sweep(task, versions[vid])
                self.state.setdefault("pattern_rescans", {})[vid] = reports
            for rule_id, ref in reports.items():
                path = self.workspace / ref["findings"]
                if sha(path) != read(self.workspace / ref["report"])["findings_sha256"]:
                    raise ValueError("Pattern findings changed after review")
                for row in read(path):
                    residuals[row["case_id"]].extend(
                        dict(kind="recurring_pattern", rule_id=rule_id, range_samples=span, version_id=vid)
                        for span in row["after_eligible"]
                    )
        for case in self.cases.values():
            pointer = self.state["accepted"].get(case["id"])
            if not pointer:
                accounting.append({"case_id": case["id"], "task": case["task"], "status": "unprocessed"})
                continue
            version = versions[pointer["version_id"]]
            a = version["results"][case["id"]]
            if a["artifact_id"] != pointer["artifact_id"] or sha(self.workspace / a["probe_path"]) != a["probe_sha256"]:
                raise ValueError("Accepted waveform/probe dependency no longer matches")
            if a["arrays"] and sha(self.workspace / a["arrays"]) != a["arrays_sha256"]:
                raise ValueError("Accepted waveform content changed")
            accepted[case["id"]] = a
            accounting.append(
                {
                    "case_id": case["id"],
                    "task": case["task"],
                    "status": "accepted_by_policy" if a["stats"]["eligible_samples"] else "unusable_for_task",
                    "artifact_id": a["artifact_id"],
                    "eligible_samples": a["stats"]["eligible_samples"],
                    "measured_samples": a["stats"]["measured_samples"],
                    "estimated_samples": a["stats"]["estimated_samples"],
                    "unusable_samples": a["stats"]["unusable_samples"],
                }
            )
            issues = a["diagnosis"]["issues"] + residuals[case["id"]]
            if issues:
                unresolved.append({"case_id": case["id"], "task": case["task"], "issues": issues})
        complete = (
            len(accepted) == len(self.cases) and not self.manifest["audit_failures"] and decision_owner == "model"
        )
        output = self.workspace / "export"
        write(output / "waveform_manifest.json", accepted)
        audit_failed = [{**r, "status": "execution_failed"} for r in self.manifest["audit_failures"]]
        write(output / "task_support.json", accounting + self.manifest["unavailable_tasks"] + audit_failed)
        write(output / "unresolved_issues.json", unresolved)
        write(output / "self_reviews.json", self.state["reviews"])
        write(
            output / "pattern_followups.json",
            {
                "methods": self.state.get("registered_patterns", {}),
                "accepted": {
                    k: self.state.get("pattern_rescans", {}).get(v, {}) for k, v in self.state["accepted_tasks"].items()
                },
            },
        )
        write(output / "dataset_manifest.json", self.manifest)
        report = {
            "status": "FINALIZED_COMPLETE" if complete else "FINALIZED_PARTIAL",
            "decision_owner": decision_owner,
            "rationale": rationale,
            "patients": self.manifest["patients"],
            "scope_cases": len(self.cases),
            "accepted_cases": len(accepted),
            "unprocessed_cases": len(self.cases) - len(accepted),
            "task_versions": self.state["accepted_tasks"],
            "tasks": {
                task: versions[vid]["report"]["tasks"][task] for task, vid in self.state["accepted_tasks"].items()
            },
            "unresolved_cases": len(unresolved),
            "reference_metrics": no_reference(),
            "meaning": "Engineering completion and policy decisions, not proven recovery",
            "implementation_hash": implementation_hash(),
            "scope_hash": self.manifest["scope_hash"],
        }
        write(output / "evaluation_report.json", report)
        # Derived files are references to immutable, version-bound probe artifacts; no table rewriting.
        write(
            output / "downstream_manifest.json",
            [
                {
                    "case_id": cid,
                    "task": a["task"],
                    "waveform_artifact": a["artifact_id"],
                    "probe_path": a["probe_path"],
                    "probe_sha256": a["probe_sha256"],
                }
                for cid, a in accepted.items()
            ],
        )
        columns = {
            "events": ["event_id", "sample", "time_s", "support_id"],
            "intervals": ["left_event_id", "right_event_id", "time_s", "interval_s", "support_id"],
            "pressures": [
                "start_sample",
                "stop_sample",
                "start_s",
                "stop_s",
                "sbp",
                "dbp",
                "mean_pressure",
                "unit",
                "support_id",
            ],
        }
        for name, fields in columns.items():
            with (output / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["case_id", "task", "waveform_version", *fields])
                writer.writeheader()
                for cid, artifact in accepted.items():
                    measures = read(self.workspace / artifact["probe_path"])
                    writer.writerows(
                        {"case_id": cid, "task": artifact["task"], "waveform_version": artifact["artifact_id"], **row}
                        for row in measures[name]
                    )
        self.state["complete"] = complete
        self.state["finalized"] = decision_owner == "model"
        self.state["delivery"] = report
        self.save()
        return report

    def call(self, name, args):
        if name not in {t["name"] for t in self.schemas}:
            raise ValueError("Unknown tool")
        result = getattr(self, name)(**args)
        return (
            result if isinstance(result, list) else [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
        )
