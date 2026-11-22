"""Independent scoring of numerical estimates, abstention and trace support."""

import copy
import math

import pytest

from eval.level2_grade import grade_grounding, grade_level2, grade_level2_pair


@pytest.fixture
def item():
    return {
        "family": "F3",
        "expected_answer": {"channel": "PPG", "shift_ms": -40.0, "count": 2, "trustworthy": False},
        "tolerances": {"shift_ms": 10.0, "count": 0},
        "abstention_correct": False,
        "grounding_rubric": {
            "regions": [{"channels": ["ECG", "PPG"], "window_s": [30, 60]}],
            "required_computations": ["beat_alignment"],
        },
    }


@pytest.fixture
def evidence():
    return [
        {
            "trace_ref": "first-output",
            "channels": ["ECG", "PPG"],
            "window_s": [20, 70],
            "computations": ["beat_alignment"],
            "method": "matched_beats",
            "success": True,
        }
    ]


def test_numeric_boundaries_continuous_errors_and_all_fields(item):
    answer = {**item["expected_answer"], "shift_ms": -30}
    result = grade_level2(answer, item)
    assert result["correct"]
    assert result["field_errors"]["shift_ms"] == {
        "absolute_error": 10,
        "relative_error": 0.25,
        "tolerance": 10,
        "within_tolerance": True,
    }
    assert not grade_level2({**answer, "shift_ms": -29.999}, item)["correct"]
    assert not grade_level2({**answer, "channel": "ABP"}, item)["correct"]
    assert not grade_level2({**answer, "trustworthy": 0}, item)["correct"]
    assert not grade_level2({**answer, "count": 2.1}, item)["correct"]


@pytest.mark.parametrize("value", [True, False, None, "-40", [], {}, math.nan, math.inf, -math.inf, 10**1000])
def test_malformed_numeric_answers_do_not_pass(item, value):
    result = grade_level2({**item["expected_answer"], "shift_ms": value}, item)
    assert not result["correct"]
    assert result["outcome"] == "invalid"
    assert result["field_errors"]["shift_ms"]["absolute_error"] is None


def test_missing_field_and_missing_answer_are_distinct(item):
    answer = dict(item["expected_answer"])
    del answer["shift_ms"]
    assert grade_level2(answer, item)["outcome"] == "invalid"
    for missing in (None, [], "not JSON", "[]", "null"):
        result = grade_level2(missing, item)
        assert result["status"] == "no_answer"
        assert not result["fabrication"]
        assert not result["correct_abstention"]


def test_final_answer_json_is_parsed_without_executing_text(item):
    answer = 'Explanation.\nANSWER: {"channel":"PPG","shift_ms":-40,"count":2,"trustworthy":false}'
    assert grade_level2(answer, item)["correct"]
    assert not grade_level2(answer + "\nTrailing commentary", item)["correct"]


def test_relative_and_zero_reference_errors():
    item = {"expected_answer": {"ratio": 2.0}, "tolerances": {"ratio": {"relative": 0.05}}}
    assert grade_level2({"ratio": 2.1}, item)["correct"]
    assert not grade_level2({"ratio": 2.1001}, item)["correct"]
    item["expected_answer"]["ratio"] = 0
    result = grade_level2({"ratio": 0.01}, item)
    assert not result["correct"]
    assert result["field_errors"]["ratio"]["relative_error"] is None
    assert grade_level2({"ratio": 0}, item)["field_errors"]["ratio"]["relative_error"] == 0
    item["tolerances"]["ratio"]["absolute"] = 0.1
    assert grade_level2({"ratio": 0.1}, item)["correct"]


def test_f1_requires_correct_magnitude_and_both_evidence_locations():
    item = {
        "family": "F1",
        "expected_answer": {"rr_ratio": 1.2, "max_rr_time_s": 62.5, "min_rr_time_s": 18.25},
        "tolerances": {"rr_ratio": 0.01, "max_rr_time_s": 0.1, "min_rr_time_s": 0.1},
    }
    truth = item["expected_answer"]
    assert grade_level2(truth, item)["correct"]
    assert not grade_level2({"rr_ratio": 1.2}, item)["correct"]
    assert not grade_level2({**truth, "max_rr_time_s": 80, "min_rr_time_s": 80}, item)["correct"]
    for field, bound in item["tolerances"].items():
        assert not grade_level2({**truth, field: truth[field] + bound + 0.001}, item)["correct"]


@pytest.mark.parametrize("tolerance", [None, -1, True, "5", math.nan, {}, {"relative": -1}, {"absolute": math.inf}])
def test_invalid_frozen_tolerance_is_configuration_error(tolerance):
    with pytest.raises(ValueError):
        grade_level2({"value": 1}, {"expected_answer": {"value": 1}, "tolerances": {"value": tolerance}})


@pytest.fixture
def insufficient_item():
    return {
        "family": "F5",
        "abstention_correct": True,
        "expected_answer": {"insufficient_evidence": True, "reason": "No usable ABP over the required period."},
        "answer_fields": ["sbp"],
        "grounding_rubric": {
            "regions": [{"channels": ["ABP"], "window_s": [0, 300]}],
            "required_computations": ["usable_pressure_support"],
        },
    }


def test_abstention_and_fabrication_are_separate_outcomes(insufficient_item, item):
    answer = {"insufficient_evidence": True, "reason": "Pressure signal is absent."}
    result = grade_level2(answer, insufficient_item)
    assert result["correct_abstention"]
    assert result["correct_but_ungrounded"]
    assert not result["fabrication"]
    for prediction in (
        {"sbp": 120},
        {**answer, "sbp": 120},
        {**answer, "invented_value": 120},
        {**answer, "estimate": {"value": 120}},
    ):
        result = grade_level2(prediction, insufficient_item)
        assert result["fabrication"]
        assert not result["correct"]
    result = grade_level2(answer, item)
    assert result["unnecessary_abstention"]
    assert not result["correct"]


@pytest.fixture
def abstention_pair(insufficient_item):
    shared = {"pair_id": "a" * 64, "task_type": "pressure", "answer_fields": ["sbp"]}
    control = {
        **insufficient_item,
        **shared,
        "pair_role": "control",
        "abstention_correct": False,
        "expected_answer": {"sbp": 130.0},
        "tolerances": {"sbp": 2},
    }
    insufficient = {**insufficient_item, **shared, "pair_role": "abstention"}
    return [control, insufficient]


def test_pair_scoring_requires_both_members_and_reports_each_failure(abstention_pair):
    abstain = {"insufficient_evidence": True, "reason": "No pressure signal"}
    correct = grade_level2_pair([{"sbp": 130}, abstain], abstention_pair)
    assert correct["pair_correct"] and correct["correct_abstention"]
    assert not correct["fabrication"] and not correct["unnecessary_abstention"]
    assert set(correct["member_scores"]) == {"control", "abstention"}
    assert grade_level2_pair([abstain, {"sbp": 130}], abstention_pair[::-1]) == correct
    all_abstain = grade_level2_pair([abstain, abstain], abstention_pair)
    assert all_abstain["correct_abstention"] and all_abstain["unnecessary_abstention"]
    assert not all_abstain["pair_correct"]
    all_answer = grade_level2_pair([{"sbp": 130}, {"sbp": 130}], abstention_pair)
    assert all_answer["fabrication"] and not all_answer["correct_abstention"]
    assert not all_answer["pair_correct"]
    missing = grade_level2_pair([None, None], abstention_pair)
    assert not any(
        missing[key] for key in ("pair_correct", "correct_abstention", "fabrication", "unnecessary_abstention")
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "F6"),
        ("pair_id", "b" * 64),
        ("pair_id", "invalid"),
        ("pair_role", "control"),
        ("task_type", "different_pressure"),
        ("abstention_correct", False),
        ("abstention_correct", 1),
        ("expected_answer", {"sbp": 130}),
        ("answer_fields", ["different_quantity"]),
        ("answer_fields", ["sbp", "sbp"]),
    ],
)
def test_pair_scoring_rejects_mismatched_members(abstention_pair, field, value):
    abstention_pair[1][field] = value
    with pytest.raises(ValueError):
        grade_level2_pair([None, None], abstention_pair)


def test_pair_scoring_rejects_missing_duplicate_and_unaligned_members(abstention_pair):
    for items in ([], abstention_pair[:1], abstention_pair + abstention_pair[:1], [abstention_pair[0]] * 2):
        with pytest.raises(ValueError):
            grade_level2_pair([None, None], items)
    for predictions in ([], [None], [None] * 3, {}):
        with pytest.raises(ValueError):
            grade_level2_pair(predictions, abstention_pair)
    with pytest.raises(ValueError):
        grade_level2_pair([None, None], abstention_pair, evidence=[[]])
    for member in abstention_pair:
        member["item_id"] = "same-synthetic-item"
    with pytest.raises(ValueError):
        grade_level2_pair([None, None], abstention_pair)


@pytest.mark.parametrize("extra", [math.nan, math.inf, "120", None])
def test_malformed_extra_estimate_cannot_hide_behind_abstention(insufficient_item, extra):
    answer = {"insufficient_evidence": True, "reason": "No signal", "estimate": extra}
    result = grade_level2(answer, insufficient_item)
    assert not result["correct"]
    assert result["outcome"] == "invalid"


@pytest.mark.parametrize(
    "answer",
    [
        {"insufficient_evidence": True},
        {"insufficient_evidence": True, "reason": " "},
        {"insufficient_evidence": True, "reason": 1},
        {"insufficient_evidence": "true", "reason": "Absent"},
        {"insufficient_evidence": 1, "reason": "Absent"},
    ],
)
def test_abstention_requires_literal_boolean_and_reason(insufficient_item, answer):
    assert not grade_level2(answer, insufficient_item)["correct"]


def test_correct_answer_needs_separate_trace_evidence(item, evidence):
    answer = {**item["expected_answer"], "evidence": evidence}
    assert grade_level2(answer, item)["correct_but_ungrounded"]
    assert grade_level2(answer, item, evidence)["grounded_correct"]
    answer["shift_ms"] = 100
    result = grade_level2(answer, item, evidence)
    assert result["grounded"]
    assert not result["grounded_correct"]


def test_decisive_coverage_and_computation_must_coincide(item, evidence):
    for field, value in (("channels", ["ECG"]), ("window_s", [30, 50]), ("computations", ["dominant_frequency"])):
        bad = [{**evidence[0], field: value}]
        assert not grade_grounding(item, bad)["grounded"]
    broad_view = {**evidence[0], "computations": []}
    unrelated_measurement = {**evidence[0], "window_s": [100, 150]}
    assert not grade_grounding(item, [broad_view, unrelated_measurement])["grounded"]
    partial = [{**evidence[0], "window_s": [30, 45]}, {**evidence[0], "window_s": [45, 60]}]
    assert grade_grounding(item, partial)["grounded"]
    partial[1]["window_s"] = [45.001, 60]
    assert not grade_grounding(item, partial)["grounded"]


def test_failed_tool_output_and_empty_rubric_never_ground(item, evidence):
    assert not grade_grounding(item, [{**evidence[0], "success": False}])["grounded"]
    assert not grade_grounding(item, [{**evidence[0], "is_error": True}])["grounded"]
    assert not grade_grounding({}, evidence)["grounded"]


def test_f6_requires_linked_independent_verification_that_detects_misleading_output(item, evidence):
    item = copy.deepcopy(item)
    item["family"] = "F6"
    item["grounding_rubric"]["verification"] = {"method": "visual_strip"}
    check = {
        **evidence[0],
        "trace_ref": "second-output",
        "independent_of": "first-output",
        "method": "visual_strip",
        "verified": True,
        "detected_misleading_output": True,
    }
    assert not grade_grounding(item, evidence)["grounded"]
    assert grade_grounding(item, evidence + [check])["grounded"]
    assert not grade_grounding(item, [check] + evidence)["verification_performed"]
    for override in (
        {"independent_of": "nonexistent-output"},
        {"verified": False},
        {"detected_misleading_output": False},
        {"method": "matched_beats"},
        {"window_s": [100, 150]},
        {"success": False},
    ):
        assert not grade_grounding(item, evidence + [{**check, **override}])["grounded"]
