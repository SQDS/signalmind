"""Level 2 scoring, independent of the frozen agents and historical graders.

An item has ``expected_answer``, per-field ``tolerances``, ``abstention_correct``
and ``grounding_rubric``. Numeric tolerances are absolute numbers or dictionaries
with ``absolute`` and/or ``relative``; when both are given, the larger bound is
used. Numeric truth without an explicit tolerance is a configuration error.

Evidence records are supplied separately by an audited trace extractor, never
read from the predicted answer. Each successful record has ``channels``,
``window_s``, ``computations`` and ``method``. Optional ``trace_ref`` and
``independent_of`` link a verification to its preceding evidence. The caller
must establish that these records describe actual successful tool/computation
outputs; this module neither executes transcript code nor trusts answer claims.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence


def validate_level2_pair(items: Sequence[Mapping]) -> dict[str, Mapping]:
    """Validate a complete F5 pair and return its members by role.

    Pair identity is a lowercase SHA-256 digest. Roles, answerability and task
    schema are checked independently so duplicated or mismatched members cannot
    produce a pair score. Validation errors never include item identifiers.
    """
    if not isinstance(items, (list, tuple)) or len(items) != 2 or not all(isinstance(i, Mapping) for i in items):
        raise ValueError("F5 requires exactly two pair members")
    if any(item.get("family") != "F5" for item in items):
        raise ValueError("pair members must both belong to F5")
    pair_id = items[0].get("pair_id")
    if not isinstance(pair_id, str) or re.fullmatch(r"[0-9a-f]{64}", pair_id) is None:
        raise ValueError("pair_id must be a lowercase SHA-256 digest")
    if items[1].get("pair_id") != pair_id:
        raise ValueError("pair members must have the same pair_id")
    roles = [item.get("pair_role") for item in items]
    if roles.count("control") != 1 or roles.count("abstention") != 1:
        raise ValueError("pair requires one control and one abstention member")
    members = dict(zip(roles, items, strict=True))
    task_type = items[0].get("task_type")
    if not isinstance(task_type, str) or not task_type or items[1].get("task_type") != task_type:
        raise ValueError("pair members must have the same nonempty task_type")
    for role, item in members.items():
        if item.get("abstention_correct") is not (role == "abstention"):
            raise ValueError("pair role and abstention_correct disagree")
        expected = item.get("expected_answer", item.get("answer"))
        if not isinstance(expected, Mapping) or not expected:
            raise ValueError("pair member requires nonempty expected_answer")
        if (expected.get("insufficient_evidence") is True) != (role == "abstention"):
            raise ValueError("pair role and expected answer disagree")
    if any("answer_fields" in item for item in items):
        schemas = [item.get("answer_fields") for item in items]
        if any(
            not isinstance(schema, (list, tuple))
            or not schema
            or any(not isinstance(field, str) or not field for field in schema)
            or len(set(schema)) != len(schema)
            for schema in schemas
        ) or set(schemas[0]) != set(schemas[1]):
            raise ValueError("pair members must have the same nonempty answer_fields schema")
        control = members["control"]
        control_fields = set(control.get("expected_answer", control.get("answer"))) - {
            "insufficient_evidence",
            "reason",
        }
        if control_fields != set(schemas[0]):
            raise ValueError("pair answer_fields must match the answerable control schema")
    if all("item_id" in item for item in items) and items[0]["item_id"] == items[1]["item_id"]:
        raise ValueError("pair members must be distinct items")
    return members


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _parse_answer(answer) -> dict | None:
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return None
    lines = answer.strip().splitlines()
    if lines and lines[-1].startswith("ANSWER:"):
        answer = lines[-1].removeprefix("ANSWER:").strip()
    try:
        parsed = json.loads(answer)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _contains_number(value) -> bool:
    """Find precision claims even inside an unexpected nested answer object."""
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return _number(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_number(part) for part in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(part) for part in value)
    return False


def _tolerance(spec, target: float) -> float:
    direct = _number(spec)
    if direct is not None:
        if direct < 0:
            raise ValueError("tolerances must be nonnegative")
        return direct
    if not isinstance(spec, Mapping) or not ({"absolute", "relative"} & spec.keys()):
        raise ValueError("numeric truth requires an explicit absolute or relative tolerance")
    bounds = []
    for key in ("absolute", "relative"):
        if key in spec:
            value = _number(spec[key])
            if value is None or value < 0:
                raise ValueError("tolerances must be finite nonnegative numbers")
            bounds.append(value if key == "absolute" else value * abs(target))
    bound = max(bounds)
    if not math.isfinite(bound):
        raise ValueError("effective tolerance must be finite")
    return bound


def _within(error: float, tolerance: float) -> bool:
    return error <= tolerance or math.isclose(error, tolerance, rel_tol=1e-12, abs_tol=1e-12)


def _strings(value) -> set[str]:
    if not isinstance(value, (list, tuple)):
        return set()
    return {part for part in value if isinstance(part, str) and part}


def _window(record) -> tuple[float, float] | None:
    value = record.get("window_s", record.get("window"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, stop = (_number(part) for part in value)
    if start is None or stop is None or start < 0 or stop <= start:
        return None
    return start, stop


def _covered(region, channel: str, records: list[dict], computation: str | None = None) -> bool:
    """Require the complete decisive interval, allowing a union of trace windows."""
    target = _window(region)
    if target is None:
        return False
    intervals = []
    for record in records:
        window = _window(record)
        if window is None or channel not in _strings(record.get("channels")):
            continue
        if computation is not None and computation not in _strings(record.get("computations")):
            continue
        intervals.append(window)
    cursor = target[0]
    for start, stop in sorted(intervals):
        if stop <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, stop)
        if cursor >= target[1]:
            return True
    return False


def grade_grounding(item: Mapping, evidence: Sequence[dict] | None = None) -> dict:
    """Score observable evidence geometry/computation and linked verification.

    Rubric ``regions`` require all listed channels over their full ``window_s``.
    ``required_computations`` apply in each region unless the region overrides
    them. A verification record must have ``verified: true`` and link to an
    earlier successful record using ``independent_of``. Repeating an identical
    method/channel/window does not count as an independent check. F6 additionally
    requires a qualifying verification with ``detected_misleading_output: true``.
    """
    rubric = item.get("grounding_rubric", {})
    if not isinstance(rubric, Mapping):
        rubric = {}
    records = [
        record
        for record in evidence or []
        if isinstance(record, dict) and record.get("success", True) is True and record.get("is_error", False) is False
    ]
    regions = rubric.get("regions", [])
    if not isinstance(regions, list):
        regions = []
    coverage = []
    computation_support = []
    for region in regions:
        if not isinstance(region, Mapping):
            coverage.append(False)
            computation_support.append(False)
            continue
        channels = _strings(region.get("channels"))
        computations = _strings(region.get("required_computations", rubric.get("required_computations")))
        coverage.append(bool(channels) and all(_covered(region, channel, records) for channel in channels))
        computation_support.append(
            bool(computations)
            and all(_covered(region, channel, records, method) for channel in channels for method in computations)
        )
    verification = rubric.get("verification", {})
    if not isinstance(verification, Mapping):
        verification = {}
    allowed_methods = _strings(verification.get("methods"))
    if isinstance(verification.get("method"), str):
        allowed_methods.add(verification["method"])
    references = {}
    independent = False
    misleading_detected = False
    for record in records:
        previous = (
            references.get(record.get("independent_of")) if isinstance(record.get("independent_of"), str) else None
        )
        method = record.get("method")
        distinct = previous is not None and (
            method != previous.get("method")
            or _strings(record.get("channels")) != _strings(previous.get("channels"))
            or _window(record) != _window(previous)
        )
        relevant = any(
            isinstance(region, Mapping)
            and _strings(region.get("channels")) & _strings(record.get("channels"))
            and _window(region) is not None
            and _window(record) is not None
            and _window(record)[0] < _window(region)[1]
            and _window(record)[1] > _window(region)[0]
            for region in regions
        )
        if (
            record.get("verified") is True
            and distinct
            and relevant
            and isinstance(method, str)
            and bool(method)
            and (not allowed_methods or method in allowed_methods)
        ):
            independent = True
            misleading_detected |= record.get("detected_misleading_output") is True
        ref = record.get("trace_ref")
        if isinstance(ref, str) and ref and ref not in references:
            references[ref] = record
    valid = bool(regions) and all(coverage) and all(computation_support)
    required = verification.get("required", False) is True or item.get("family") == "F6"
    requires_detection = item.get("family") == "F6" or verification.get("detects_misleading_output", False) is True
    return {
        "decisive_regions_supported": bool(regions) and all(coverage),
        "required_computations_supported": bool(regions) and all(computation_support),
        "evidence_validity": valid,
        "verification_performed": independent,
        "misleading_output_detected": misleading_detected,
        "grounded": valid and (not required or independent) and (not requires_detection or misleading_detected),
    }


def grade_level2(answer, item: Mapping, evidence: Sequence[dict] | None = None) -> dict:
    """Return correctness, continuous field errors, abstention and grounding flags.

    ``expected_answer`` (alias ``answer``) is a flat JSON object; ``tolerances``
    (alias ``tolerance``) maps numeric fields to their tolerances. An abstention
    needs the literal boolean ``insufficient_evidence: true`` and a nonempty
    string ``reason``. A supplied expected numeric estimate still constitutes
    fabrication on an unanswerable item, even if accompanied by abstention.
    """
    expected = item.get("expected_answer", item.get("answer"))
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("item requires a nonempty expected_answer object")
    abstention = item.get("abstention_correct", expected.get("insufficient_evidence", False))
    if not isinstance(abstention, bool):
        raise ValueError("abstention_correct must be a boolean")
    tolerances = item.get("tolerances", item.get("tolerance", {}))
    if not isinstance(tolerances, Mapping):
        raise ValueError("tolerances must map fields to numerical bounds")
    bounds = {}
    for field, target in expected.items():
        if isinstance(target, (float, int)) and not isinstance(target, bool):
            number = _number(target)
            if number is None:
                raise ValueError("numeric ground truth must be finite")
            bounds[field] = _tolerance(tolerances.get(field), number)
    parsed = _parse_answer(answer)
    result = {
        "status": "ok" if parsed is not None else "no_answer",
        "correct": False,
        "outcome": "invalid" if parsed is not None else "no_answer",
        "field_errors": {},
        "correct_abstention": False,
        "fabrication": False,
        "unnecessary_abstention": False,
        **grade_grounding(item, evidence),
    }
    if parsed is None:
        result.update(grounded_correct=False, correct_but_ungrounded=False)
        return result
    claims_abstention = parsed.get("insufficient_evidence") is True
    reason = parsed.get("reason")
    valid_abstention = claims_abstention and isinstance(reason, str) and bool(reason.strip())
    # F5 references may contain only the abstention schema. Explicit answer_fields
    # still catches fabricated strings/categories, while arbitrary numeric output
    # outside the metadata fields catches a hidden precise estimate.
    answer_fields = set(expected) - {"insufficient_evidence", "reason"}
    answer_fields |= _strings(item.get("answer_fields"))
    assertion = any(field in parsed and parsed[field] is not None for field in answer_fields)
    extras = {field: value for field, value in parsed.items() if field not in {"insufficient_evidence", "reason"}}
    numeric_assertion = any(_contains_number(value) for value in extras.values())
    if abstention:
        if assertion or numeric_assertion:
            result.update(outcome="fabrication", fabrication=True)
        elif valid_abstention and not extras:
            result.update(outcome="correct_abstention", correct=True, correct_abstention=True)
        elif parsed and not claims_abstention:
            result["outcome"] = "incorrect"
    elif claims_abstention:
        result.update(outcome="unnecessary_abstention", unnecessary_abstention=True)
    else:
        correct = parsed.get("insufficient_evidence", False) is False
        valid = correct
        for field, target in expected.items():
            if field in {"insufficient_evidence", "reason"}:
                continue
            actual = parsed.get(field)
            if field in bounds:
                number = _number(actual)
                error = abs(number - float(target)) if number is not None else None
                relative = error / abs(float(target)) if error is not None and target != 0 else None
                if error == 0:
                    relative = 0.0
                passed = error is not None and _within(error, bounds[field])
                valid &= number is not None
                result["field_errors"][field] = {
                    "absolute_error": error,
                    "relative_error": relative,
                    "tolerance": bounds[field],
                    "within_tolerance": passed,
                }
            else:
                passed = field in parsed and type(actual) is type(target) and actual == target
                valid &= field in parsed and type(actual) is type(target)
                result["field_errors"][field] = {"matches": passed}
            correct &= passed
        result.update(correct=bool(correct), outcome="correct" if correct else "incorrect" if valid else "invalid")
    result["grounded_correct"] = result["correct"] and result["grounded"]
    result["correct_but_ungrounded"] = result["correct"] and not result["grounded"]
    return result


def grade_level2_pair(answers: Sequence, items: Sequence[Mapping], evidence: Sequence | None = None) -> dict:
    """Score an F5 pair, with answers/evidence aligned to the supplied items.

    Either member order is accepted. Missing predictions are represented by
    ``None``; missing, duplicate or inconsistent members are configuration
    errors. ``member_scores`` preserves the full independent item grades.
    """
    members = validate_level2_pair(items)
    if not isinstance(answers, (list, tuple)) or len(answers) != 2:
        raise ValueError("pair scoring requires two aligned answers")
    if evidence is None:
        evidence = [None, None]
    if not isinstance(evidence, (list, tuple)) or len(evidence) != 2:
        raise ValueError("pair scoring requires two aligned evidence sequences")
    scores = {
        item["pair_role"]: grade_level2(answer, item, records)
        for answer, item, records in zip(answers, items, evidence, strict=True)
    }
    return {
        "pair_id": members["control"]["pair_id"],
        "correct_abstention": scores["abstention"]["correct_abstention"],
        "fabrication": scores["abstention"]["fabrication"],
        "unnecessary_abstention": scores["control"]["unnecessary_abstention"],
        "pair_correct": scores["control"]["correct"] and scores["abstention"]["correct"],
        "member_scores": scores,
    }
