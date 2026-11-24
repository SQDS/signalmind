"""Scoped, independent visual decisions with content-addressed, replayable receipts."""

from __future__ import annotations

import base64
import fcntl
import json
import operator
import time
from contextlib import contextmanager
from pathlib import Path

from jsonschema import ValidationError, validate

from tsmcp.followup.reviewer import tool

from .agent import BudgetClient, vision_preflight
from .common import digest, read, sha, write
from .context import ContextBuilder
from .provenance import check_stop, emit

FIELDS = ("seconds", "step_norm", "span_norm", "beat_distance_ms", "bpm_min", "bpm_max", "peer_count", "peer_alerts")
QUALITIES = ("keep", "review", "withhold")


OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


def applies(conditions, features):
    return all(features[c["field"]] is not None and OPS[c["op"]](features[c["field"]], c["value"]) for c in conditions)


def enum(*values):
    return {"type": "string", "enum": list(values)}


def obj(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def arr(item):
    return {"type": "array", "items": item}


TEXT = {"type": "string", "minLength": 1}
condition = arr(obj(field=enum(*FIELDS), op=enum("ge", "le", "gt", "lt"), value={"type": "number"}))
DIAGNOSIS = obj(
    label=enum("artifact", "physiology", "mixed", "unknown"),
    confidence={"type": "number", "minimum": 0, "maximum": 1},
    evidence_refs=arr(TEXT),
    alternative_hypotheses=arr(TEXT),
)
SCHEMAS = {
    "diagnosis": obj(id=TEXT, diagnosis=DIAGNOSIS, reason=TEXT),
    "coarse": obj(id=TEXT, method=enum("identity", "fixed_dsp", "rule_adaptive_dsp"), reason=TEXT),
    "groups": obj(
        id=TEXT,
        wave=enum("preserve", "mark_invalid", "interpolate_spike", "remove_tone", "lowpass", "local_detrend"),
        wave_if=condition,
        measure=enum(*QUALITIES),
        measure_if=condition,
        otherwise=enum(*QUALITIES),
        reason=TEXT,
    ),
    "observation": obj(id=TEXT, reason=TEXT),
    "gate": obj(
        id=TEXT,
        decision=enum("accept", "rollback", "reject", "need_evidence", "preserve", "mark_unrecoverable"),
        reason=TEXT,
    ),
    "joint": obj(
        id=TEXT, default=enum(*QUALITIES), exceptions=arr(obj(key=TEXT, quality=enum(*QUALITIES))), reason=TEXT
    ),
}
SCHEMAS["groups"]["properties"]["parameters"] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "frequency_hz": {"type": "number", "exclusiveMinimum": 0},
        "window_s": {"type": "number", "exclusiveMinimum": 0},
        "filter_order": {"const": 4},
    },
}
PROMPTS = {
    "coarse": (
        "You are the repair agent. Choose one available recipe for this signal and change type after inspecting "
        "its plots and measurements. Protect native beats and absolute pressure. Identity is "
        "acceptable."
    ),
    "diagnosis": (
        "Classify the observed group as artifact, physiology, mixed or unknown. Use its representative "
        "waveforms and member values, cite only the supplied evidence IDs, and record competing hypotheses. "
        "Describe uncertainty without proposing a repair."
    ),
    "groups": (
        "You are the repair planner. Use the supplied immutable diagnostic receipt and numeric member evidence. "
        "The diagnosis role has already read both views. Do not change its diagnosis. "
        "Numerical repair requires an artifact diagnosis. Use only actions registered for this trigger. "
        "Give one conditional policy per item. The two images show a typical member and a boundary member "
        "(for a singleton, context and detail of the same event). The member table lists every hit; "
        "case_index indexes cases, a:b is the half-open sample core. Inspect the trigger value and member features. "
        "For isolated spikes, compare the observed core with both measured anchors. Span within a flat-topped "
        "core may be small despite a large jump. Where provided, trial_diagnosis applies only to eligible members. "
        "Choose an interpolation trial only for artifact members with trial_eligible=1; conditions must match "
        "at least one actual eligible row on the page. Preserve when a trial is not justified. This trials "
        "SignalMind linear interpolation between the adjacent observed anchors separately at each eligible core. "
        "Choose a trial if the anchors and transient morphology make this plausible; an independent reviewer "
        "will inspect the actual before/after, and failed trials revert. Protected landmarks cannot be interpolated. "
        "Overlapping peer alerts alone do not require review. "
        "wave=mark_invalid only for clear technical unusability. wave_if and measure_if are AND "
        "conditions; an empty list matches all. "
        "Measurements use keep, review (quarantine uncertainty) or withhold (clearly unsuitable). "
        "Choose measure on match and otherwise on nonmatch. "
        "seconds is core duration; step_norm and span_norm are scaled by personal typical beat "
        "swing; beat_distance_ms is to nearest native landmark; bpm_min/max are local overlapping "
        "interval rates, peer_count and peer_alerts count other channels. "
        "JEV labels if present route evidence only. A large step can be a genuine beat. Cite "
        "concrete morphology and limitations."
    ),
    "observation": (
        "Independently describe plausible physiology, distortion and ambiguity in the before/after image "
        "before seeing the proposed action or numerical checks."
    ),
    "gate": (
        "You are the independent release reviewer. Inspect parent and candidate plots, proposed "
        "modified or disabled samples, and frozen native retest. Reject removal of plausible true "
        "beats, collateral changes, poor recovery, or insufficient evidence."
    ),
    "joint": (
        "Independently adjudicate the supplied native measurements against synchronized final "
        "waveforms. For EACH item choose default quality for its listed measurements and give "
        "exact key exceptions. "
        "Inspect each stated native interval/cycle, separating doubtful intervals from ordinary "
        "adjacent beats. No invented peaks. keep means usable by this check, review means "
        "pending, withhold means unsuitable."
    ),
}


def eligible_features(data):
    columns = data["columns"]
    pos = columns.index("trial_eligible")
    return [dict(zip(columns, row, strict=True)) for row in data["member_rows"] if row[pos]]


def mixed_spikes(row):
    scope = row["data"].get("trial_scope", {})
    return bool(scope.get("eligible") and scope.get("protected"))


def actions_for(trigger, task):
    extra = {
        "isolated_spike_candidate": "interpolate_spike",
        "line_50": "remove_tone",
        "line_60": "remove_tone",
        "high_frequency": "lowpass",
        "ecg_baseline_shift": "local_detrend",
    }
    actions = ["preserve", "mark_invalid"]
    if trigger in extra and not (extra[trigger] == "local_detrend" and task != "ECG"):
        actions.append(extra[trigger])
    return actions


class Reviewer:
    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        self.replay = Path(config["replay_from"]) / "dataset-receipts" if config.get("replay_from") else None
        self.seed = Path(config["receipt_seed"]) / "dataset-receipts" if config.get("receipt_seed") else None
        self.enabled = config["method"] == "dataset_agent"
        self.client = None
        self.exhausted = False
        self.context = ContextBuilder(config)
        self.record = "setup"
        self.account = self.root / "context-usage.json"
        self.history = read(self.account) if self.account.exists() else {}
        self.deadline = None

    def debit(self):
        count = self.history.get(self.record, 0)
        if count >= self.context.calls or time.monotonic() >= self.deadline:
            raise RuntimeError("context_budget_exhausted")
        self.history[self.record] = count + 1
        write(self.account, self.history)

    def bounded(self, stage, rows, system, schema, size):
        batch = []
        for row in rows:
            chunk = batch + [row]
            if batch and (len(chunk) > size or not self.context.fits(self.context.usage(chunk, system, schema))):
                yield batch
                batch = []
            if self.context.fits(self.context.usage([row], system, schema)):
                batch.append(row)
            else:
                emit(self.root, "decision_pending", stage=stage, id=row["id"], reason="context_budget")
        if batch:
            yield batch

    @contextmanager
    def session(self):
        if not self.enabled or self.replay:
            yield self
            return
        budget = Path(self.config.get("budget_root") or self.root)
        budget.mkdir(parents=True, exist_ok=True)
        with (budget / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.client = BudgetClient(self.root, cap=self.config["api_cap"], budget_root=budget, compact_tools=True)
            try:
                (self.root / "private").mkdir(exist_ok=True, mode=0o700)
                try:
                    self.deadline = time.monotonic() + self.context.seconds
                    self.client.before_attempt = self.debit
                    self.client.output_limit = self.context.output
                    self.client.deadline = self.deadline
                    if self.context.calls:
                        vision_preflight(self.client)
                    else:
                        self.exhausted = True
                except RuntimeError as exc:
                    if str(exc) != "api_budget_exhausted":
                        raise
                    self.exhausted = True
                yield self
            finally:
                self.client.close()

    def decide(self, stage, rows):
        if stage == "groups" and self.enabled and rows and "diagnostic_receipt" not in rows[0]["data"]:
            plans = {}
            pages = {}
            for row in rows:
                group = row["data"].get("visual_group_id", row["id"])
                pages.setdefault(group, []).append(row)
            visuals = []
            for group, parts in pages.items():
                first = parts[0]
                if len(parts) == 1:
                    visuals.append(dict(first, id=group))
                    continue
                # Diagnose each logical group once. Per-member tables move to the planner in bounded pages.
                data = {
                    k: v
                    for k, v in first["data"].items()
                    if k not in ("member_rows", "member_offset", "cases", "core_samples_by_member")
                }
                data.update(
                    scope=data["group_size"], evidence_ids=["view_1", "view_2", "summary", "representative_members"]
                )
                visuals.append(dict(id=group, images=first["images"], data=data))
            for batch in self.bounded("diagnosis", visuals, PROMPTS["diagnosis"], SCHEMAS["diagnosis"], 4):
                assessments = self.decide("diagnosis", batch)
                bound = []
                for group, assessment in assessments.items():
                    for r in pages[group]:
                        bound.append(
                            dict(
                                r,
                                images=[],
                                data=dict(
                                    r["data"], diagnostic_receipt=assessment, view_hashes=[sha(p) for p in r["images"]]
                                ),
                            )
                        )
                if bound:
                    plans.update(self.decide("groups", bound))
            return plans
        if stage == "gate" and self.enabled and rows:
            # The first pass sees only the picture and modality. Its response is version-bound in the second pass.
            observed = self.decide(
                "observation",
                [
                    dict(
                        id=r["id"],
                        images=r["images"],
                        data=dict(
                            task=r["data"]["task"],
                            revision=r["data"].get("revision"),
                            parent=r["data"].get("parent"),
                            candidate=r["data"].get("candidate"),
                        ),
                    )
                    for r in rows
                ],
            )
            rows = [
                dict(r, data=dict(r["data"], blind_observation=observed[r["id"]])) for r in rows if r["id"] in observed
            ]
        role = "review" if stage in ("gate", "observation", "joint") else "repair"
        if stage != "diagnosis" and getattr(self, "snapshot", None):
            rows = [
                dict(
                    r,
                    data=dict(
                        r["data"],
                        role_context=self.knowledge.context(
                            self.snapshot, role, r["data"].get("task"), r["data"].get("trigger", stage)
                        ),
                    ),
                )
                for r in rows
            ]
        decisions = {}
        schema = tool("submit", "Return each requested ID exactly once.", {"items": arr(SCHEMAS[stage])})
        system = (
            PROMPTS[stage] + " Use submit. Give a short concrete reason in Chinese. No reference gold is available."
        )
        # A group carries all member measurements; keep its image pair and complete table in a single request.
        size = 1 if stage == "groups" else 4
        for batch in self.bounded(stage, rows, system, schema, size):
            check_stop(self.root)
            if stage == "diagnosis":
                item = SCHEMAS[stage]
                if any(mixed_spikes(r) for r in batch):
                    item = dict(item, properties=dict(item["properties"], trial_diagnosis=DIAGNOSIS))
                schema = tool("submit", "Return each requested ID exactly once.", {"items": arr(item)})
            if stage == "groups":
                action_list = actions_for(batch[0]["data"]["trigger"], batch[0]["data"]["task"])
                data = batch[0]["data"]
                if "trial_scope" in data and not eligible_features(data):
                    action_list = [v for v in action_list if v != "interpolate_spike"]
                fields = dict(SCHEMAS[stage]["properties"], wave=enum(*action_list))
                schema = tool("submit", "Return each requested ID exactly once.", {"items": arr(obj(**fields))})
            meta = [(r["id"], r["data"], [sha(p) for p in r["images"]]) for r in batch]
            binding = digest([self.config["model"], "compact_tools.v1", system, schema, meta])
            path = self.root / "dataset-receipts" / (binding + ".json")
            expected = {r["id"] for r in batch}

            def checked(result, expected=expected, batch=batch, schema=schema):
                validate(result, schema["input_schema"])
                if sorted(r["id"] for r in result["items"]) != sorted(expected):
                    raise ValueError("One decision required for every supplied id")
                for d, r in zip(
                    sorted(result["items"], key=lambda d: d["id"]), sorted(batch, key=lambda r: r["id"]), strict=True
                ):
                    if stage == "coarse" and d["method"] not in r["data"]["available"]:
                        raise ValueError("Unavailable method")
                    if stage == "groups" and d["wave"] not in actions_for(r["data"]["trigger"], r["data"]["task"]):
                        raise ValueError("Operation outside registered trigger scope")
                    if stage == "groups" and d["wave"] not in ("preserve", "mark_invalid"):
                        assessment = r["data"]["diagnostic_receipt"]
                        diag = assessment["diagnosis"]
                        if d["wave"] == "interpolate_spike" and "trial_scope" in r["data"]:
                            diag = assessment.get("trial_diagnosis", diag)
                            if not any(applies(d["wave_if"], f) for f in eligible_features(r["data"])):
                                raise ValueError(
                                    "No eligible member on this page satisfies wave_if. Use conditions "
                                    "supported by eligible rows, an unconditional trial, or explicitly preserve."
                                )
                        if diag["label"] != "artifact":
                            raise ValueError("Numerical repair requires an artifact diagnosis for the target members")
                    if stage == "diagnosis":
                        diagnoses = [d["diagnosis"]]
                        if mixed_spikes(r):
                            if not d.get("trial_diagnosis"):
                                raise ValueError("Provide trial_diagnosis for eligible members separately")
                            diagnoses.append(d["trial_diagnosis"])
                        elif "trial_diagnosis" in d:
                            raise ValueError("Only mixed-eligibility short-core groups have a separate trial diagnosis")
                        for diagnosis in diagnoses:
                            refs = diagnosis["evidence_refs"]
                            if (not refs and diagnosis["label"] != "unknown") or not set(refs) <= set(
                                r["data"]["evidence_ids"]
                            ):
                                raise ValueError("Use the supplied evidence IDs for this diagnosis")
                    if (
                        stage == "gate"
                        and d["decision"] == "mark_unrecoverable"
                        and not r["data"].get("hard_gate", {}).get("checks", {}).get("mark_invalid")
                    ):
                        raise ValueError("Only a verified support mask can be released as unrecoverable")
                    if stage == "joint":
                        keys = [v["key"] for v in d["exceptions"]]
                        if len(keys) != len(set(keys)) or not set(keys) <= set(r["data"]["keys"]):
                            raise ValueError("Invalid native measurement key")

            saved = self.replay / path.name if self.replay else self.seed / path.name if self.seed else path
            split = path.parent / "splits" / path.name
            saved_split = saved.parent / "splits" / path.name
            if split.exists() or saved_split.exists():
                receipt = read(split if split.exists() else saved_split)
                if receipt != [r["id"] for r in batch] or len(batch) < 2:
                    raise ValueError("Invalid batch split")
                if not split.exists():
                    write(split, receipt)
                midpoint = len(batch) // 2
                decisions.update(self.decide(stage, batch[:midpoint]))
                decisions.update(self.decide(stage, batch[midpoint:]))
                continue
            if path.exists() or self.replay or saved.is_file():
                rec = read(path if path.exists() else saved)
                if (
                    rec["binding"] != binding
                    or rec["stage"] != stage
                    or rec.get("decision_sha256") != digest([binding, rec.get("request_id"), rec.get("decision")])
                    or rec["evidence"] != json.loads(json.dumps(meta))
                ):
                    raise ValueError("Review evidence changed")
                checked(rec["decision"])
                if not path.exists():
                    write(path, rec)
                    emit(
                        self.root,
                        "dataset_receipt_reused",
                        request_id=rec["request_id"],
                        binding=binding,
                        source=str(saved),
                    )
            elif self.exhausted or not self.enabled:
                continue
            else:
                content = []
                for r in batch:
                    content.append(
                        dict(
                            type="text",
                            text=json.dumps(
                                {"id": r["id"], "data": r["data"]},
                                ensure_ascii=False,
                                separators=(",", ":") if stage == "groups" else None,
                            ),
                        )
                    )
                    for p in r["images"]:
                        content.append(
                            dict(
                                type="image",
                                source=dict(
                                    type="base64",
                                    media_type="image/png",
                                    data=base64.b64encode(p.read_bytes()).decode(),
                                ),
                            )
                        )
                messages = [dict(role="user", content=content)]
                rec = None
                correction = ""
                for attempt in range(2):
                    request_schema = schema
                    if attempt:
                        fields = {k: v for k, v in SCHEMAS[stage]["properties"].items() if k != "id"}
                        if stage == "groups":
                            fields["wave"] = enum(*action_list)
                        request_schema = tool(
                            "submit",
                            "One object for each image item in its original order.",
                            {
                                f"row{n + 1}": obj(
                                    **(
                                        dict(fields, trial_diagnosis=DIAGNOSIS)
                                        if stage == "diagnosis" and mixed_spikes(r)
                                        else fields
                                    )
                                )
                                for n, r in enumerate(batch)
                            },
                        )
                        mapping = {f"row{n + 1}": r["id"] for n, r in enumerate(batch)}
                        instruction = "Submit using exactly these row keys: " + json.dumps(mapping)
                        if correction:
                            instruction += ". Correct the prior submission: " + correction
                        if stage == "groups":
                            instruction += ". Allowed wave actions here: " + ", ".join(fields["wave"]["enum"])
                        messages = [
                            dict(
                                role="user",
                                content=[
                                    *content,
                                    dict(type="text", text=instruction),
                                ],
                            )
                        ]
                    try:
                        usage = self.context.usage(batch, system, request_schema)
                        if attempt:
                            encoded_instruction = len(instruction.encode())
                            usage["text_bytes"] += encoded_instruction
                        if attempt:
                            usage["reserved_tokens"] += encoded_instruction
                        if not self.context.fits(usage):
                            break
                        self.deadline = time.monotonic() + self.context.seconds
                        self.client.deadline = self.deadline
                        if not isinstance(self.client, BudgetClient):
                            self.debit()
                        res = self.client.request(messages, [request_schema], system=system)
                        if time.monotonic() > self.deadline:
                            raise RuntimeError("context_budget_exhausted")
                    except RuntimeError as exc:
                        if str(exc) == "context_budget_exhausted":
                            emit(
                                self.root,
                                "decision_pending",
                                stage=stage,
                                ids=sorted(expected),
                                reason="record_or_wall_budget",
                            )
                            break
                        if str(exc) != "api_budget_exhausted":
                            raise
                        self.exhausted = True
                        break
                    blocks = [b.model_dump(exclude_none=True) for b in res.content]
                    calls = [b for b in blocks if b["type"] == "tool_use"]
                    try:
                        if len(calls) != 1 or calls[0]["name"] != "submit":
                            raise ValueError("Use submit exactly once")
                        result = calls[0]["input"]
                        if attempt:
                            validate(result, request_schema["input_schema"])
                            result = {"items": [dict(id=id_, **result[row]) for row, id_ in mapping.items()]}
                        checked(result)
                    except (ValueError, ValidationError) as exc:
                        correction = exc.message[:240] if isinstance(exc, ValidationError) else str(exc)[:240]
                        emit(
                            self.root,
                            "dataset_submission_rejected",
                            stage=stage,
                            request_id=res.id,
                            reason=str(exc)[:240],
                        )
                        if attempt and len(batch) > 1:
                            write(split, [r["id"] for r in batch])
                            midpoint = len(batch) // 2
                            decisions.update(self.decide(stage, batch[:midpoint]))
                            decisions.update(self.decide(stage, batch[midpoint:]))
                        continue
                    rec = dict(binding=binding, stage=stage, evidence=meta, decision=result, request_id=res.id)
                    rec["submission_format"] = "ordered_keys" if attempt else "ids"
                    rec["decision_sha256"] = digest([binding, res.id, result])
                    write(path, rec)
                    if stage != "diagnosis" and getattr(self, "snapshot", None):
                        for item in result["items"]:
                            data = next(row["data"] for row in batch if row["id"] == item["id"])
                            self.knowledge.record(
                                role,
                                dict(
                                    modality=data.get("task"),
                                    trigger=data.get("trigger", stage),
                                    stage=stage,
                                    source=binding,
                                    version=self.snapshot["id"],
                                    confidence=item.get("diagnosis", {}).get("confidence"),
                                    note=item["reason"][:420],
                                    outcome=item.get("decision") or item.get("wave"),
                                    scope=self.record,
                                ),
                            )
                    emit(self.root, "dataset_review", stage=stage, binding=binding, request_id=res.id)
                    break
                if rec is None:
                    continue
            for d in rec["decision"]["items"]:
                decision = dict(d, receipt=binding, request_id=rec["request_id"])
                if stage == "gate":
                    decision["review_decision"] = d["decision"]
                    decision["decision"] = {
                        "mark_unrecoverable": "accept",
                        "reject": "rollback",
                        "preserve": "rollback",
                    }.get(d["decision"], d["decision"])
                if stage == "groups":
                    assessment = next(r["data"]["diagnostic_receipt"] for r in batch if r["id"] == d["id"])
                    decision.update(diagnosis=assessment["diagnosis"], diagnosis_receipt=assessment["receipt"])
                    if "trial_diagnosis" in assessment:
                        decision["trial_diagnosis"] = assessment["trial_diagnosis"]
                decisions[d["id"]] = decision
        return decisions
