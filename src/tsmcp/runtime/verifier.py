"""Deterministic claim verification: does the cited evidence support the submitted claim?

Checks provenance (references, source kind), method (definition), scope (interval, channels), units, timebase,
adequacy (counts, limitations), warnings/conflicts, arithmetic and precision. It is not an oracle of physiological or
clinical truth and it is independent of any benchmark grader.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal

from tsmcp.runtime.claims import SUPPORTED, UNRESOLVED, AnswerBundle, Claim, ClaimStatus
from tsmcp.runtime.ledger import EvidenceLedger, EvidenceRecord, SourceKind
from tsmcp.runtime.taskspec import DEPENDENCY_RULES, MEASUREMENT_DEFINITIONS, TaskSpec
from tsmcp.runtime.timebase import REQUIRED_SYNC, SyncStatus, supports
from tsmcp.runtime.units import compatible, convert, family

CODES = (
    "invalid_reference",
    "method_mismatch",
    "scope_mismatch",
    "missing_required_channel",
    "unknown_or_incompatible_unit",
    "timebase_unknown",
    "inadequate_support",
    "unresolved_warning_or_conflict",
    "arithmetic_mismatch",
    "precision_unsupported",
    "malformed_bundle",
)
SCOPE_TOL_S = 1e-3
EXACT_STATUSES = {ClaimStatus.VALIDATED_MEASUREMENT, ClaimStatus.OBSERVED_DELAY, ClaimStatus.PHYSIOLOGICAL_TIMING}
_FALLBACK_DEFINITION = {"kind": "measurement", "required_roles": (), "unit_family": None}


@dataclass(frozen=True)
class Deficiency:
    code: str
    claim_id: str | None
    message: str
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationReport:
    ok: bool
    deficiencies: list[Deficiency]
    claims: dict[str, dict] = field(default_factory=dict)  # claim_id -> {"status", "evidence", "deficiencies"}
    bundle_fingerprint: str = ""

    def for_claim(self, claim_id: str) -> list[Deficiency]:
        return [d for d in self.deficiencies if d.claim_id == claim_id]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "deficiencies": [d.to_dict() for d in self.deficiencies],
            "claims": self.claims,
            "bundle_fingerprint": self.bundle_fingerprint,
        }


def decimals_of(value) -> int:
    """Number of decimals in the shortest repr of a float (3 sig-fig values like 118.0 -> 1, 0.0155 -> 4)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0
    d = Decimal(repr(float(value))).normalize()
    return max(0, -d.as_tuple().exponent)


def _half_ulp(value) -> float:
    return 0.5 * 10.0 ** (-decimals_of(value))


def _tolerance(rec: EvidenceRecord) -> float:
    res = rec.resolution or {}
    r = float(res.get("value") or 0.0) if res.get("known") else 0.0
    return max(r, _half_ulp(rec.value))


def _union_covers(intervals, target, tol=SCOPE_TOL_S) -> bool:
    t0, t1 = target
    cursor = t0
    for a, b in sorted(intervals):
        if b <= cursor + tol:
            continue
        if a > cursor + tol:
            return False
        cursor = max(cursor, b)
        if cursor >= t1 - tol:
            return True
    return cursor >= t1 - tol


def verify(bundle: AnswerBundle, ledger: EvidenceLedger, spec: TaskSpec) -> VerificationReport:
    defs: list[Deficiency] = []
    claims: dict[str, dict] = {}
    for c in bundle.claims:
        found, used = _check_claim(c, ledger, spec)
        defs += found
        claims[c.claim_id] = {
            "status": c.status if not found else "rejected",
            "evidence": [r.evidence_id for r in used],
            "deficiencies": [d.code for d in found],
        }
    covered = (
        {c.quantity for c in bundle.claims if ClaimStatus(c.status) in SUPPORTED}
        | {lim["quantity"] for lim in bundle.limitations}
        | {c.quantity for c in bundle.claims if ClaimStatus(c.status) in UNRESOLVED}
    )
    for q in spec.quantities:
        if q.required and q.name not in covered:
            defs.append(
                Deficiency(
                    "inadequate_support",
                    None,
                    f"required quantity {q.name!r} has neither a claim nor a stated limitation",
                )
            )
    return VerificationReport(ok=not defs, deficiencies=defs, claims=claims, bundle_fingerprint=bundle.fingerprint())


def _check_claim(c: Claim, ledger: EvidenceLedger, spec: TaskSpec) -> tuple[list[Deficiency], list[EvidenceRecord]]:
    out: list[Deficiency] = []
    status = ClaimStatus(c.status)

    def bad(code, msg, *recs):
        out.append(Deficiency(code, c.claim_id, msg, tuple(r.evidence_id for r in recs)))

    if status in UNRESOLVED:
        if not c.reason:
            bad("inadequate_support", f"status {status.value} needs a reason")
        return out, []

    # 1. provenance: every reference must resolve
    recs: list[EvidenceRecord] = []
    for ref in c.evidence_refs:
        r = ledger.get(ref)
        if r is None:
            bad("invalid_reference", f"evidence {ref!r} does not exist in this run's ledger")
        else:
            recs.append(r)
    if out:
        return out, recs
    if not recs:
        bad(
            "inadequate_support", "no evidence cited; cite the evidence_id (or ref) of the measurement that supports it"
        )
        return out, recs

    # 2. method: which definition does this claim assert?
    qspec = spec.quantity(c.quantity)
    def_id = c.definition_id or (qspec.definition_id if qspec else None)
    if def_id is None:
        cited = {r.definition_id for r in recs if r.definition_id and r.numeric is not None}
        if len(cited) == 1:
            def_id = cited.pop()
        elif status in EXACT_STATUSES:
            bad("method_mismatch", "state definition_id: the cited evidence carries several or no definitions")
            return out, recs
    definition = MEASUREMENT_DEFINITIONS.get(def_id, _FALLBACK_DEFINITION) if def_id else _FALLBACK_DEFINITION
    kind = definition["kind"]
    if status is ClaimStatus.VALIDATED_MEASUREMENT and kind in ("observed_delay", "physiological_timing"):
        bad("method_mismatch", f"{def_id} is a {kind} quantity: submit it with status {kind}")
    if status is ClaimStatus.OBSERVED_DELAY and kind != "observed_delay":
        bad("method_mismatch", f"status observed_delay needs an observed-delay definition, not {def_id}")
    if status is ClaimStatus.PHYSIOLOGICAL_TIMING and kind != "physiological_timing":
        bad("method_mismatch", f"status physiological_timing needs a physiological-timing definition, not {def_id}")
    if out:
        return out, recs

    if c.derived:
        op = c.derived.get("op") if isinstance(c.derived, dict) else None
        supporting = [
            r for r in recs if r.source_kind == SourceKind.DERIVED.value and r.parameters.get("derived_op") == op
        ]
        if not supporting:
            bad(
                "inadequate_support",
                f"a derived value must cite a derived evidence record created by the {op!r} "
                "operation; the model's own arithmetic is not evidence",
            )
            return out, recs
    else:
        supporting = [r for r in recs if def_id is None or r.definition_id == def_id]

    if qspec and qspec.definition_id and c.definition_id and c.definition_id != qspec.definition_id:
        bad("method_mismatch", f"the task asks for {qspec.definition_id}; the claim declares {c.definition_id}")
        return out, recs

    visual = [r for r in recs if r.source_kind == SourceKind.VISUAL_OBSERVATION.value]
    if not supporting:
        if visual and len(visual) == len(recs):
            if status in (ClaimStatus.APPROXIMATE, ClaimStatus.QUALITATIVE):
                supporting = visual
            else:
                bad(
                    "inadequate_support",
                    "only visual observations are cited: a value read off an image cannot be a validated "
                    "measurement; measure it, or submit status approximate with precision_decimals",
                    *visual,
                )
                return out, recs
        else:
            what = sorted({r.definition_id or r.quantity for r in recs})
            how = f"cite a {def_id} measurement or fix definition_id"
            if definition.get("derive_with"):
                how = (
                    f"{def_id} is derived: call ts_derive op={definition['derive_with']} on a "
                    f"{definition.get('derived_from', 'suitable')} record and cite the derived evidence"
                )
            bad("method_mismatch", f"the cited evidence measures {what}, not {def_id}; {how}", *recs)
            return out, recs

    # 3. source kind: an exact measurement needs computed/derived evidence
    allowed = set(DEPENDENCY_RULES.get(kind, DEPENDENCY_RULES["measurement"])["source_kinds"])
    if status in EXACT_STATUSES:
        if all(r.source_kind == SourceKind.VISUAL_OBSERVATION.value for r in supporting):
            bad(
                "inadequate_support",
                "only visual observations are cited: a value read off an image cannot be a "
                "validated measurement; measure it, or submit status approximate with "
                "precision_decimals",
                *supporting,
            )
            return out, supporting
        if any(r.source_kind == SourceKind.HYPOTHESIS.value for r in supporting):
            bad("inadequate_support", "a hypothesis record cannot support a measurement", *supporting)
            return out, supporting
        supporting = [r for r in supporting if r.source_kind in allowed]
        if not supporting:
            bad("inadequate_support", f"{status.value} needs evidence of kind {sorted(allowed)}")
            return out, recs

    # 4. limitations reported by the method itself
    ok_recs = [r for r in supporting if r.status == "ok"]
    if not ok_recs:
        notes = "; ".join(f"{r.evidence_id}: {r.note}" for r in supporting if r.note)
        bad(
            "inadequate_support",
            f"the cited method reported it could not produce a defensible value ({notes})",
            *supporting,
        )
        return out, supporting

    # 5. required channels (what the measurement depends on, not every channel of the record)
    effective = {ch for r in ok_recs for ch in r.channels_effective}
    for role in definition.get("required_roles", ()):
        name = spec.channel_roles.get(role)
        if name is None:
            if spec.quantities:  # a real spec that lacks the role: the task cannot be fulfilled as asked
                bad("missing_required_channel", f"the task declares no channel for role {role!r}, which {def_id} needs")
            continue
        if name not in effective:
            bad(
                "missing_required_channel",
                f"{def_id} needs role {role!r} = channel {name!r}; the cited evidence used {sorted(effective)}",
                *ok_recs,
            )
    if c.channels and not set(c.channels) <= effective:
        bad(
            "scope_mismatch",
            f"claim names channels {list(c.channels)} but the evidence used {sorted(effective)}",
            *ok_recs,
        )

    numeric = c.value is not None and not isinstance(c.value, str)
    # 6. units: the claim must be in the evidence's unit (conversions are derived records, not verifier arithmetic)
    if numeric and status is not ClaimStatus.QUALITATIVE:
        if c.unit is None or family(c.unit) is None:
            bad("unknown_or_incompatible_unit", f"claim unit {c.unit!r} is missing or not a known unit")
        else:
            for r in ok_recs:
                if r.numeric is None:
                    continue
                if r.unit is None or family(r.unit) is None:
                    bad(
                        "unknown_or_incompatible_unit",
                        f"{r.evidence_id} has unit {r.unit!r} (unknown / calibration unverified): it cannot "
                        f"support a value in {c.unit}",
                        r,
                    )
                elif r.unit != c.unit:
                    how = "cite a unit_convert derived record" if compatible(r.unit, c.unit) else "incompatible"
                    bad(
                        "unknown_or_incompatible_unit",
                        f"{r.evidence_id} is in {r.unit}, the claim in {c.unit}: {how}",
                        r,
                    )
            if qspec and not compatible(c.unit, qspec.unit):
                bad(
                    "unknown_or_incompatible_unit", f"the task asks for {qspec.unit}; {c.unit} is not convertible to it"
                )
    if out:
        return out, ok_recs

    # 7. scope: the evidence must have been computed over the claimed interval, not a sub- or super-window; and an
    #    exact claim must be for the interval the task asks for (a revised task re-checks this)
    asked = (qspec.interval if qspec else None) or spec.interval
    if (
        asked is not None
        and c.interval is not None
        and status in EXACT_STATUSES
        and (abs(c.interval[0] - asked[0]) > SCOPE_TOL_S or abs(c.interval[1] - asked[1]) > SCOPE_TOL_S)
    ):
        bad("scope_mismatch", f"the task asks for [{asked[0]:g}, {asked[1]:g}); the claim is for "
                              f"[{c.interval[0]:g}, {c.interval[1]:g})")  # fmt: skip
        return out, ok_recs
    target = c.interval or asked
    if target is not None:
        for r in ok_recs:
            if r.source_kind == SourceKind.DERIVED.value:
                ints = sorted(set(r.intervals_effective))
                span = (min(i[0] for i in ints), max(i[1] for i in ints)) if ints else None
                claimed = sorted(set(c.intervals)) if c.intervals else None
                same_set = (
                    claimed is not None
                    and len(claimed) == len(ints)
                    and all(
                        abs(a[0] - b[0]) <= SCOPE_TOL_S and abs(a[1] - b[1]) <= SCOPE_TOL_S
                        for a, b in zip(claimed, ints, strict=True)
                    )
                )
                spans = (
                    span is not None
                    and abs(span[0] - target[0]) <= SCOPE_TOL_S
                    and abs(span[1] - target[1]) <= SCOPE_TOL_S
                )
                if not ints or not (same_set or spans):
                    bad(
                        "scope_mismatch",
                        f"{r.evidence_id} covers {ints}; give the same windows (a list of [t0, t1]) or their overall "
                        f"span, not {list(target)}",
                        r,
                    )
                continue
            req = r.interval_requested or (r.intervals_effective[0] if r.intervals_effective else None)
            if req is None:
                bad("scope_mismatch", f"{r.evidence_id} has no recorded interval; cannot support an interval claim", r)
            elif status in EXACT_STATUSES and (
                abs(req[0] - target[0]) > SCOPE_TOL_S or abs(req[1] - target[1]) > SCOPE_TOL_S
            ):
                bad(
                    "scope_mismatch",
                    f"{r.evidence_id} was measured over [{req[0]:g}, {req[1]:g}) but the claim is for "
                    f"[{target[0]:g}, {target[1]:g}); measure the claimed interval",
                    r,
                )
            elif status not in EXACT_STATUSES and not _union_covers([req], target):
                bad(
                    "scope_mismatch",
                    f"{r.evidence_id} ([{req[0]:g}, {req[1]:g})) does not cover the claimed "
                    f"[{target[0]:g}, {target[1]:g})",
                    r,
                )
    if out:
        return out, ok_recs

    # 7b. window rule (e.g. "the 60 s after the dropout ends"): the claimed window must have the stated length and its
    #     start must be supported by boundary evidence the agent actually has (a flat/dropout detection, beat
    #     exclusions or a method limitation ending there). The rule knows nothing about where the boundary truly is.
    rule = spec.conventions.get("window_rule") if isinstance(spec.conventions, dict) else None
    if isinstance(rule, dict) and status in EXACT_STATUSES and c.interval is not None:
        length = float(rule.get("length_s", 0) or 0)
        tol = float(rule.get("tolerance_s", 2.0) or 2.0)
        if length and abs((c.interval[1] - c.interval[0]) - length) > 1.0:
            bad("scope_mismatch", f"the task asks for a {length:g} s window; the claim spans "
                                  f"{c.interval[1] - c.interval[0]:g} s")  # fmt: skip
        if rule.get("type") == "after_dropout":
            channel = rule.get("channel")
            if not _boundary_supported(ledger, channel, c.interval[0], tol):
                bad(
                    "inadequate_support",
                    f"the window start {c.interval[0]:g} s is not supported by boundary evidence on {channel}: detect "
                    "the dropout (flat events, beat exclusions or a measurement that reports no beats) ending at that "
                    "time, then measure from there",
                )
        if out:
            return out, ok_recs

    # 8. timebase: physiological timing needs declared synchrony; an observed delay needs a common index
    if kind in REQUIRED_SYNC:
        needed = REQUIRED_SYNC[kind]
        statuses = [r.timebase.get("sync_status", SyncStatus.UNKNOWN.value) for r in ok_recs]
        chans = sorted(effective)
        if len(chans) >= 2:
            statuses.append(spec.sync_between(chans[0], chans[-1]))
        if not any(supports(s, needed) for s in statuses):
            bad(
                "timebase_unknown",
                f"{def_id} needs channel synchrony of at least {needed.value}; the evidence and the "
                f"task declare {sorted(set(statuses))}. Submit it as an observed delay "
                "(if a common sample index exists) or state the declared offset",
                *ok_recs,
            )
            return out, ok_recs

    # 9. adequacy: counts and error-level warnings
    min_beats = definition.get("min_accepted_beats")
    for r in ok_recs:
        acc = r.counts.get("accepted")
        if min_beats and acc is not None and acc < min_beats:
            bad(
                "inadequate_support",
                f"{r.evidence_id} has {acc} accepted beats; {def_id} needs at least {min_beats}",
                r,
            )
        if r.warning_codes("error"):
            bad("inadequate_support", f"{r.evidence_id} carries error-level warnings {r.warning_codes('error')}", r)
    if out:
        return out, ok_recs

    # 10. unresolved warnings (exact statuses must acknowledge them) and conflicts between cited values
    if status in EXACT_STATUSES:
        pending = sorted({w for r in ok_recs for w in r.warning_codes("warning")} - set(c.acknowledged_warnings))
        if pending:
            msgs = {w.code: w.message for r in ok_recs for w in r.warnings if w.code in pending}
            bad(
                "unresolved_warning_or_conflict",
                "warnings not addressed: "
                + "; ".join(f"{k}: {v}" for k, v in msgs.items())
                + ". Address them (other interval/method) or list the codes in acknowledged_warnings and qualify",
                *ok_recs,
            )
    nums = [r for r in ok_recs if r.numeric is not None]
    if numeric and len(nums) >= 2:
        base = nums[-1]
        for r in nums[:-1]:
            if r.evidence_id == base.supersedes or compatible(r.unit, base.unit) is False:
                continue
            v = (
                convert(r.numeric, r.unit, base.unit)
                if r.unit != base.unit and compatible(r.unit, base.unit)
                else r.numeric
            )
            if abs(v - base.numeric) > max(_tolerance(r), _tolerance(base)):
                bad(
                    "unresolved_warning_or_conflict",
                    f"cited values disagree: {r.evidence_id}={r.value} {r.unit} vs "
                    f"{base.evidence_id}={base.value} {base.unit}; resolve or cite one",
                    r,
                    base,
                )
    if out:
        return out, ok_recs

    # 11. arithmetic and 12. precision
    if numeric and status in EXACT_STATUSES | {ClaimStatus.APPROXIMATE}:
        base = nums[-1] if nums else None
        if base is None:
            if status is ClaimStatus.APPROXIMATE and all(
                r.source_kind == SourceKind.VISUAL_OBSERVATION.value for r in ok_recs
            ):
                if c.precision_decimals is None:
                    bad("precision_unsupported", "approximate claims must state precision_decimals")
                return out, ok_recs  # a labelled, coarse observation; nothing numeric to check it against
            bad("inadequate_support", "no numeric evidence among the cited records")
            return out, ok_recs
        claimed = float(c.value)
        tol = max(_tolerance(base), _half_ulp(c.value))  # a claim may be rounded coarser than its evidence
        if status is ClaimStatus.APPROXIMATE:
            if c.precision_decimals is None:
                bad("precision_unsupported", "approximate claims must state precision_decimals")
            else:
                tol = max(tol, 0.5 * 10.0 ** (-c.precision_decimals))
        if abs(claimed - base.numeric) > tol + 1e-12:
            bad(
                "arithmetic_mismatch",
                f"claim value {c.value} {c.unit} differs from {base.evidence_id} = {base.value} "
                f"{base.unit} by more than its resolution ({tol:g})",
                base,
            )
        if status in EXACT_STATUSES:
            lim = decimals_of(base.value)
            res = base.resolution or {}
            if res.get("known") and res.get("value") and float(res["value"]) > 0:
                # a value is reported no finer than the first significant digit of its uncertainty
                lim = min(lim, max(0, -math.floor(math.log10(float(res["value"])))))
            if qspec and qspec.precision_decimals is not None:
                lim = min(lim, qspec.precision_decimals)
            if decimals_of(c.value) > lim:
                bad(
                    "precision_unsupported",
                    f"claim reports {decimals_of(c.value)} decimals; the evidence supports at most {lim} "
                    f"(resolution {base.resolution}); round the value or submit status approximate "
                    f"({base.evidence_id} = {base.value} {base.unit})",
                    base,
                )
    return out, ok_recs


def _boundary_supported(ledger: EvidenceLedger, channel: str | None, start: float, tol: float) -> bool:
    """Is there evidence in the ledger of a non-pulsatile stretch on `channel` ending within `tol` s of `start`?"""
    for r in ledger.records():
        if channel and channel not in r.channels_effective:
            continue
        ends: list[float] = [b for _, b in r.intervals_excluded]  # beat-table exclusions (flat / nonfinite / ...)
        if r.quantity.startswith("events."):  # detector candidates: t + dur
            for e in r.parameters.get("candidates", []):
                t, d = e.get("t"), e.get("dur") or 0.0
                if t is not None:
                    ends.append(float(t) + float(d))
        if r.status == "limitation" and r.interval_requested is not None:  # a measurement that found no beats
            ends.append(float(r.interval_requested[1]))
        if any(abs(e - start) <= tol for e in ends):
            return True
    return False
