"""Scoped, independent visual decisions with content-addressed, replayable receipts."""

from __future__ import annotations

import base64
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from jsonschema import ValidationError, validate

from tsmcp.followup.reviewer import tool

from .agent import BudgetClient, vision_preflight
from .common import digest, read, sha, write
from .provenance import check_stop, emit

FIELDS = ("seconds", "step_norm", "span_norm", "beat_distance_ms", "bpm_min", "bpm_max", "peer_count", "peer_alerts")
QUALITIES = ("keep", "review", "withhold")


def enum(*values):
    return {"type": "string", "enum": list(values)}


def obj(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def arr(item):
    return {"type": "array", "items": item}


TEXT = {"type": "string", "minLength": 1}
condition = arr(obj(field=enum(*FIELDS), op=enum("ge", "le", "gt", "lt"), value={"type": "number"}))
SCHEMAS = {
    "coarse": obj(id=TEXT, method=enum("identity", "fixed_dsp", "rule_adaptive_dsp"), reason=TEXT),
    "groups": obj(
        id=TEXT,
        wave=enum("preserve", "mark_invalid", "interpolate_spike"),
        wave_if=condition,
        measure=enum(*QUALITIES),
        measure_if=condition,
        otherwise=enum(*QUALITIES),
        reason=TEXT,
    ),
    "gate": obj(id=TEXT, decision=enum("accept", "rollback"), reason=TEXT),
    "joint": obj(
        id=TEXT, default=enum(*QUALITIES), exceptions=arr(obj(key=TEXT, quality=enum(*QUALITIES))), reason=TEXT
    ),
}
PROMPTS = {
    "coarse": (
        "You are the repair agent. Choose one available recipe for this signal and change type after inspecting "
        "its plots and measurements. Protect native beats and absolute pressure. Identity is "
        "acceptable."
    ),
    "groups": (
        "You are the repair agent. Inspect the representative raw signals and numeric evidence. "
        "Give one executable conditional policy per group. Two images show a typical member and a boundary member "
        "(for a singleton, context and detail of the same event). The member table lists every hit; "
        "case_index indexes cases, a:b is the half-open sample core. Inspect the trigger value and member features. "
        "Only for isolated_spike_candidate you may choose interpolate_spike with wave_if thresholds; this trials "
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


class Reviewer:
    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        self.replay = Path(config["replay_from"]) / "dataset-receipts" if config.get("replay_from") else None
        self.seed = Path(config["receipt_seed"]) / "dataset-receipts" if config.get("receipt_seed") else None
        self.enabled = config["method"] == "dataset_agent"
        self.client = None
        self.exhausted = False

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
                    vision_preflight(self.client)
                except RuntimeError as exc:
                    if str(exc) != "api_budget_exhausted":
                        raise
                    self.exhausted = True
                yield self
            finally:
                self.client.close()

    def decide(self, stage, rows):
        decisions = {}
        schema = tool("submit", "Return each requested ID exactly once.", {"items": arr(SCHEMAS[stage])})
        system = (
            PROMPTS[stage] + " Use submit. Give a short concrete reason in Chinese. No reference gold is available."
        )
        # A group carries all member measurements; keep its image pair and complete table in a single request.
        size = 1 if stage == "groups" else 4
        for i in range(0, len(rows), size):
            check_stop(self.root)
            batch = rows[i : i + size]
            meta = [(r["id"], r["data"], [sha(p) for p in r["images"]]) for r in batch]
            binding = digest([self.config["model"], "compact_tools.v1", system, schema, meta])
            path = self.root / "dataset-receipts" / (binding + ".json")
            expected = {r["id"] for r in batch}

            def checked(result, expected=expected, batch=batch):
                validate(result, schema["input_schema"])
                if sorted(r["id"] for r in result["items"]) != sorted(expected):
                    raise ValueError("One decision required for every supplied id")
                for d, r in zip(
                    sorted(result["items"], key=lambda d: d["id"]), sorted(batch, key=lambda r: r["id"]), strict=True
                ):
                    if stage == "coarse" and d["method"] not in r["data"]["available"]:
                        raise ValueError("Unavailable method")
                    if (
                        stage == "groups"
                        and d["wave"] == "interpolate_spike"
                        and r["data"]["trigger"] != "isolated_spike_candidate"
                    ):
                        raise ValueError("Interpolation requires a bounded isolated spike candidate")
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
                for attempt in range(2):
                    request_schema = schema
                    if attempt:
                        fields = {k: v for k, v in SCHEMAS[stage]["properties"].items() if k != "id"}
                        if stage == "groups" and batch[0]["data"]["trigger"] != "isolated_spike_candidate":
                            fields["wave"] = enum("preserve", "mark_invalid")
                        request_schema = tool(
                            "submit",
                            "One object for each image item in its original order.",
                            {f"row{n + 1}": obj(**fields) for n in range(len(batch))},
                        )
                        mapping = {f"row{n + 1}": r["id"] for n, r in enumerate(batch)}
                        instruction = "Submit using exactly these row keys: " + json.dumps(mapping)
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
                        res = self.client.request(messages, [request_schema], system=system)
                    except RuntimeError as exc:
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
                    emit(self.root, "dataset_review", stage=stage, binding=binding, request_id=res.id)
                    break
                if rec is None:
                    continue
            for d in rec["decision"]["items"]:
                decisions[d["id"]] = dict(d, receipt=binding, request_id=rec["request_id"])
        return decisions
