"""Versioned contracts for a dataset run. Coordinates are block-local and half-open."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import digest


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Support(StrEnum):
    CLEAR = "evaluated_no_alert"
    ALERT = "suspected_artifact"
    PHYSIOLOGY = "possible_physiology"
    UNKNOWN = "insufficient_support"
    NA = "not_applicable"
    FAILED = "failed"
    PENDING = "pending"


class CandidateState(StrEnum):
    DISCOVERED = "discovered"
    DIAGNOSING = "awaiting_diagnosis"
    DIAGNOSED = "diagnosed"
    PLANNED = "repair_planned"
    TRIAL = "trial_running"
    HARD_REJECT = "hard_gate_rejected"
    IN_REVIEW = "awaiting_review"
    REJECTED = "review_rejected"
    UNKNOWN = "need_evidence"
    ACCEPTED = "accepted"
    PRESERVED = "preserved"
    BAD = "unrecoverable"
    PENDING = "pending_budget"


class SampleRange(Contract):
    block_id: str
    start_sample: int = Field(ge=0)
    end_sample: int

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sample <= self.start_sample:
            raise ValueError("Empty or reversed sample range")
        return self


class ToolResult(Contract):
    tool_id: str
    status: Support
    evaluated_ranges: list[SampleRange]
    unsupported_ranges: list[SampleRange]
    evidence_refs: list[str]


class Candidate(Contract):
    candidate_id: str
    record_id: str
    channel_id: str
    region: SampleRange
    source_time_s: tuple[float, float]
    source_version_id: str
    revision: int = Field(ge=1)
    detector: str
    threshold: float | None
    value: float | None
    support_status: Support
    evidence_digest: str
    state: CandidateState


class Diagnosis(Contract):
    label: Literal["artifact", "physiology", "mixed", "unknown"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: list[str]
    alternative_hypotheses: list[str] = Field(default_factory=list)


class Tool(Contract):
    tool_id: str
    tool_version: str
    parameters: dict


class RepairPlan(Contract):
    candidate_id: str
    candidate_revision: str
    parent_version_id: str
    diagnosis: Diagnosis
    tool: Tool
    read_ranges: list[SampleRange]
    write_ranges: list[SampleRange]
    protected_ranges: list[SampleRange]
    expected_effects: list[str]
    acceptance_spec_id: str
    stop_conditions: list[str]

    @property
    def plan_id(self):
        return digest(self.model_dump())


class ReviewReceipt(Contract):
    candidate_id: str
    candidate_revision: str
    parent_version_id: str
    candidate_version_id: str
    plan_id: str
    acceptance_spec_id: str
    hard_gate: dict
    review_decision: Literal["accept", "reject", "need_evidence", "preserve", "mark_unrecoverable"]
    evidence_refs: list[str]
    reviewer_version: str

    @model_validator(mode="after")
    def acceptance_has_evidence(self):
        if self.review_decision in ("accept", "mark_unrecoverable") and (
            self.hard_gate.get("status") != "passed"
            or not isinstance(self.hard_gate.get("checks"), dict)
            or not self.hard_gate["checks"]
            or not {
                "grid",
                "registered_tools",
                "write_ranges",
                "support_scope",
                "observed_provenance",
                "finite",
                "value_scope",
                "interpolation_support",
            }
            <= self.hard_gate["checks"].keys()
            or not all(v is True for v in self.hard_gate["checks"].values())
            or not self.evidence_refs
        ):
            raise ValueError("Acceptance requires a passed hard gate and independent evidence")
        return self


class Patch(Contract):
    plan: RepairPlan
    candidate_version_id: str
    idempotency_key: str
    receipt: ReviewReceipt

    @model_validator(mode="after")
    def exact_binding(self):
        r, p = self.receipt, self.plan
        if (
            r.candidate_id,
            r.candidate_revision,
            r.parent_version_id,
            r.plan_id,
            r.acceptance_spec_id,
            r.candidate_version_id,
        ) != (
            p.candidate_id,
            p.candidate_revision,
            p.parent_version_id,
            p.plan_id,
            p.acceptance_spec_id,
            self.candidate_version_id,
        ) or self.idempotency_key != digest([p.plan_id, self.candidate_version_id]):
            raise ValueError("Patch or review refers to a different version, scope or revision")
        return self


class RunOutcome(Contract):
    status: Literal["completed", "partial", "failed", "cancelled", "running"]
    finalized: bool
    complete: bool

    @model_validator(mode="after")
    def consistent(self):
        if self.complete != (self.status == "completed") or self.finalized != (self.status in ("completed", "partial")):
            raise ValueError("Inconsistent run outcome")
        return self


class Run(Contract):
    schema_version: Literal["1.0"] = "1.0"
    scope: str
    implementation: str
    skill_snapshot: str
    acceptance_spec_id: str
    candidate_revision: str
    records: list[str]
    outcome: RunOutcome


def outcome(status):
    return RunOutcome(
        status=status, finalized=status in ("completed", "partial"), complete=status == "completed"
    ).model_dump()


def published_run(root):
    """Accept current and historical genuine releases, never a preview or a stopped checkpoint."""
    from .common import read

    state = read(root / "agent_state.json")
    run = read(root / "run.json") if (root / "run.json").exists() else {}
    if run.get("status") in ("queued", "running", "stopping", "stopped", "failed", "cancelled"):
        return False
    if "outcome" in state:
        return RunOutcome.model_validate(state["outcome"]).finalized
    if "finalized" in state or "complete" in state:
        return state.get("finalized", state.get("complete", False))
    report = root / "export/evaluation_report.json"
    return bool(
        state.get("accepted_tasks") and report.exists() and read(report).get("status") in ("completed", "partial")
    )


class RepairAttempt(Contract):
    """One operation evaluation, including requests which stopped before numerical execution."""

    attempt_id: str
    record_id: str
    block_id: str
    candidate_ids: list[str]
    phase: Literal["coarse", "fine"]
    round: int
    operation: str
    parent_version_id: str | None
    candidate_version_id: str | None
    candidate_content_hash: str | None
    candidate_revision: str | None
    plan_id: str | None
    parameters: dict
    read_ranges: list[SampleRange]
    write_ranges: list[SampleRange]
    receipt_ids: list[str]
    request_ids: list[str]
    patch_id: str | None
    applicability: str
    review_evidence: list[dict]
    operator_invoked: bool | None
    candidate_generated: bool
    hard_gate: str
    review: str
    commit: Literal["committed", "not_committed"]
    outcome: Literal["waveform_repair", "waveform_trial", "mark_invalid", "preserve", "event_amendment"]
    reason_codes: list[str]
    failed_checks: list[str]
    detail: str | None
    numerical_elapsed_s: float | None
    candidate_changed_samples: int | None


class CandidateRepairOutcome(Contract):
    candidate_id: str
    record_id: str
    region: SampleRange
    detector: str
    in_initial_freeze: bool
    stages: dict
    decisions: list[dict]
    attempts: list[str]
    committed_actions: list[str]
    rescan: str
    unresolved: bool
    reason_codes: list[str]
    source_version_id: str
    released_version_id: str
    released_changed_samples_in_region: int
    newly_disabled_samples_in_region: int
    waveform_mae: float | None = None
    waveform_gain: float | None = None
    reference_status: Literal["independent_reference_not_supplied"] = "independent_reference_not_supplied"


class Repairability(Contract):
    status: Literal["eligible", "need_evidence", "preserve_physiology", "unsupported", "unrecoverable_for_goal"]
    profile_id: str | None
    profile_version: str | None
    calibration: str
    species: str | None
    device: str | None
    modality: str
    fs_hz: float
    unit: str | None
    goal: str | None
    acceptance_spec_id: str
    reason_codes: list[str]
    operation: str
    duration_s: float
    local_period_s: float | None
    period_fraction: float | None
    observed_anchors: bool
    time_supported: bool
    event_evidence: dict | None = None
