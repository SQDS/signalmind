"""Claims and answer bundles: the structured channel through which conclusions are submitted and rendered."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class ClaimStatus(StrEnum):
    # statuses a claim may request
    VALIDATED_MEASUREMENT = "validated_measurement"  # exact value supported by computed/derived evidence
    APPROXIMATE = "approximate"  # a value with stated coarse precision (e.g. localisation from a view)
    QUALITATIVE = "qualitative"  # a categorical statement (e.g. "ABP is flat from 120 s")
    OBSERVED_DELAY = "observed_delay"  # an inter-channel delay in sample time, no physiological meaning claimed
    PHYSIOLOGICAL_TIMING = "physiological_timing"  # e.g. pulse arrival time; needs declared synchrony
    # statuses for unresolved components (limitations)
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    INCOMPLETE_EXECUTION = "incomplete_execution"  # infrastructure, not a statement about the physiology


SUPPORTED = {
    ClaimStatus.VALIDATED_MEASUREMENT,
    ClaimStatus.APPROXIMATE,
    ClaimStatus.QUALITATIVE,
    ClaimStatus.OBSERVED_DELAY,
    ClaimStatus.PHYSIOLOGICAL_TIMING,
}
UNRESOLVED = {ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.UNRESOLVED_CONFLICT, ClaimStatus.INCOMPLETE_EXECUTION}


@dataclass(frozen=True)
class Claim:
    claim_id: str
    quantity: str
    status: str
    value: float | int | str | None = None
    unit: str | None = None
    interval: tuple[float, float] | None = None
    channels: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    definition_id: str | None = None
    derived: dict | None = None  # {"op": ..., "inputs": [...]} when the value is a derivation the model proposes
    acknowledged_warnings: tuple[str, ...] = ()
    precision_decimals: int | None = None
    reason: str | None = None  # required for unresolved statuses
    intervals: tuple[tuple[float, float], ...] = ()  # the individual windows when a claim spans several

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        if not isinstance(d, dict):
            raise ValueError("claim must be an object")
        for k in ("claim_id", "quantity", "status"):
            if not isinstance(d.get(k), str) or not d[k]:
                raise ValueError(f"claim needs a nonempty string {k!r}")
        try:
            ClaimStatus(d["status"])
        except ValueError:
            raise ValueError(
                f"claim {d['claim_id']}: unknown status {d['status']!r}; allowed: {[s.value for s in ClaimStatus]}"
            ) from None
        interval = d.get("interval")
        intervals: tuple = ()
        if interval is not None:
            if not isinstance(interval, (list, tuple)) or len(interval) == 0:
                raise ValueError(f"claim {d['claim_id']}: interval must be [t0, t1] or a list of [t0, t1] pairs")
            if all(isinstance(x, (list, tuple)) for x in interval):  # several windows: keep them, span = overall
                try:
                    intervals = tuple((float(a), float(b)) for a, b in interval)
                except (TypeError, ValueError):
                    raise ValueError(f"claim {d['claim_id']}: each interval must be [t0, t1]") from None
                interval = (min(a for a, _ in intervals), max(b for _, b in intervals))
            else:
                if len(interval) != 2:
                    raise ValueError(f"claim {d['claim_id']}: interval must be [t0, t1]")
                try:
                    interval = (float(interval[0]), float(interval[1]))
                except (TypeError, ValueError):
                    raise ValueError(f"claim {d['claim_id']}: interval must be [t0, t1] with numeric bounds") from None
        value = d.get("value")
        if isinstance(value, bool) or (value is not None and not isinstance(value, (int, float, str))):
            raise ValueError(f"claim {d['claim_id']}: value must be a number, string or null")
        return cls(
            claim_id=d["claim_id"],
            quantity=d["quantity"],
            status=d["status"],
            value=value,
            unit=d.get("unit"),
            interval=interval,
            channels=tuple(d.get("channels", ()) or ()),
            evidence_refs=tuple(d.get("evidence_refs", ()) or ()),
            definition_id=d.get("definition_id"),
            derived=d.get("derived"),
            acknowledged_warnings=tuple(d.get("acknowledged_warnings", ()) or ()),
            precision_decimals=d.get("precision_decimals"),
            reason=d.get("reason"),
            intervals=intervals,
        )


@dataclass
class AnswerBundle:
    claims: list[Claim]
    limitations: list[dict] = field(default_factory=list)  # {"quantity", "status", "reason", "evidence_refs"}
    final: bool = False  # recorded for the trace only; has no effect on verification or termination
    accept_partial: bool = False  # after reviewing reported deficiencies, accept a partial answer
    text: str | None = None  # free text; never the source of rendered numbers
    notes: list[str] = field(default_factory=list)  # free-text remarks (e.g. plain strings given as limitations)

    def to_dict(self) -> dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "limitations": self.limitations,
            "final": self.final,
            "accept_partial": self.accept_partial,
            "text": self.text,
            "notes": self.notes,
        }

    def fingerprint(self) -> str:
        d = self.to_dict()
        d.pop("text", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> AnswerBundle:
        if not isinstance(d, dict) or not isinstance(d.get("claims"), list):
            raise ValueError("bundle must be an object with a 'claims' list")
        claims = [Claim.from_dict(c) for c in d["claims"]]
        ids = [c.claim_id for c in claims]
        if len(set(ids)) != len(ids):
            raise ValueError("claim_id values must be unique within a bundle")
        raw = d.get("limitations") or []
        if not isinstance(raw, list):
            raise ValueError("limitations must be a list")
        notes = [str(n) for n in (d.get("notes") or []) if isinstance(n, (str, int, float))]
        lims = []
        for lim in raw:
            if isinstance(lim, str):  # a remark, not a structured limitation: kept verbatim as a note
                notes.append(lim)
                continue
            if not isinstance(lim, dict) or not lim.get("quantity") or not lim.get("reason"):
                raise ValueError(
                    'each limitation needs {"quantity": ..., "reason": ...} (plain strings are kept as notes)'
                )
            lims.append(lim)
            st = lim.get("status", ClaimStatus.INSUFFICIENT_EVIDENCE.value)
            if ClaimStatus(st) not in UNRESOLVED:
                raise ValueError(f"limitation status must be one of {[s.value for s in UNRESOLVED]}")
            lim["status"] = st
        supported = {c.quantity for c in claims if ClaimStatus(c.status) in SUPPORTED}
        unresolved = {c.quantity for c in claims if ClaimStatus(c.status) in UNRESOLVED} | {
            lim["quantity"] for lim in lims
        }
        conflict = supported & unresolved
        if conflict:
            raise ValueError(
                f"quantities {sorted(conflict)} cannot have both a supported claim "
                "and a limitation or unresolved claim. "
                "Put caveats for a retained value in notes, or withdraw its supported claim to abstain."
            )
        return cls(claims=claims, limitations=lims, final=bool(d.get("final", False)), text=d.get("text"), notes=notes,
                   accept_partial=bool(d.get("accept_partial", False)))  # fmt: skip
