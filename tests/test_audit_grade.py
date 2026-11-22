"""Audit scoring contracts: row-specific tolerances and multicomponent answers."""

import ast
import math
from pathlib import Path

import pytest

from eval import common
from eval.audit_formats import AUDIT_ANSWER_FORMATS
from eval.common import ANSWER_FORMATS, case_prompt
from eval.grade import AUDIT_FIELDS, grade, null, oracle

ANSWERS = {
    "audit_rate": {"bpm": 72.0},
    "audit_motion": {"t": 40.0},
    "audit_step": {"t": 50.0},
    "audit_unusable": {"percent": 30.0},
    "audit_bp": {"sbp": 120.0, "dbp": 80.0, "map": 92.0},
    "audit_valsalva": {"t": 90.0},
    "audit_damped": {"t": 70.0},
    "audit_tilt": {"t": 100.0},
    "audit_shift": {"channel": "PPG", "shift_s": -0.30},
    "audit_pat": {"pat_ms": 220.0},
    "audit_corrupted": {"channel": "ABP"},
    "audit_response": {
        "hr_direction": "increase",
        "sbp_direction": "decrease",
        "hr_latency_s": 12.0,
        "sbp_latency_s": 30.0,
    },
    "audit_brs": {"brs_ms_mmhg": 8.0},
    "audit_cbfv_tilt": {"t": 110.0},
    "audit_tfa": {"gain": 0.8},
}


@pytest.mark.parametrize("qtype", ANSWERS)
def test_audit_oracle_and_missing(qtype):
    answer = {**ANSWERS[qtype], "provenance": "hidden"}
    assert qtype in ANSWER_FORMATS
    predicted = oracle(qtype, answer)
    assert set(predicted) == set(AUDIT_FIELDS[qtype])
    assert grade(qtype, answer, predicted)["correct"] is True
    assert grade(qtype, answer, null(qtype))["correct"] is False
    assert grade(qtype, answer, {})["correct"] is False
    assert grade(qtype, answer, None) == {"status": "no_answer", "correct": None, "err": None}
    assert grade(qtype, answer, [1])["status"] == "no_answer"


@pytest.mark.parametrize(
    ("qtype", "field", "tolerance"),
    [
        ("audit_rate", "bpm", 3),
        ("audit_motion", "t", 2),
        ("audit_step", "t", 5),
        ("audit_unusable", "percent", 10),
        ("audit_bp", "sbp", 5),
        ("audit_bp", "dbp", 5),
        ("audit_bp", "map", 5),
        ("audit_valsalva", "t", 10),
        ("audit_damped", "t", 5),
        ("audit_tilt", "t", 15),
        ("audit_shift", "shift_s", 0.05),
        ("audit_pat", "pat_ms", 20),
        ("audit_response", "hr_latency_s", 10),
        ("audit_response", "sbp_latency_s", 10),
        ("audit_brs", "brs_ms_mmhg", 2),
        ("audit_cbfv_tilt", "t", 15),
        ("audit_tfa", "gain", 0.16),
    ],
)
def test_audit_numeric_boundaries(qtype, field, tolerance):
    answer = ANSWERS[qtype]
    for sign in (-1, 1):
        boundary = {**answer, field: answer[field] + sign * tolerance}
        outside = {**answer, field: answer[field] + sign * (tolerance + 1e-6)}
        assert grade(qtype, answer, boundary)["correct"] is True
        assert grade(qtype, answer, outside)["correct"] is False
    for invalid in (True, False, None, "72", math.nan, math.inf, -math.inf, [], {}, 10**1000):
        assert grade(qtype, answer, {**answer, field: invalid})["correct"] is False


@pytest.mark.parametrize("qtype", ["audit_shift", "audit_corrupted"])
@pytest.mark.parametrize("channel", [None, True, 0, [], {}, "", "ppg", "unknown"])
def test_audit_channel_is_exact_name(qtype, channel):
    answer = ANSWERS[qtype]
    assert grade(qtype, answer, {**answer, "channel": channel})["correct"] is False


def test_shift_sign_and_domain_constraints():
    answer = ANSWERS["audit_shift"]
    assert grade("audit_shift", answer, {**answer, "shift_s": 0.30})["correct"] is False
    assert grade("audit_motion", {"t": 0}, {"t": -1})["correct"] is False
    assert grade("audit_unusable", {"percent": 95}, {"percent": 101})["correct"] is False
    assert grade("audit_unusable", {"percent": 5}, {"percent": -1})["correct"] is False


@pytest.mark.parametrize("field", ["hr_direction", "sbp_direction"])
@pytest.mark.parametrize("value", [None, True, [], {}, "up", "Increase", "unchanged"])
def test_response_requires_both_exact_directions(field, value):
    answer = ANSWERS["audit_response"]
    assert grade("audit_response", answer, {**answer, field: value})["correct"] is False


def test_response_unchanged_has_no_latency():
    answer = {**ANSWERS["audit_response"], "hr_direction": "unchanged", "hr_latency_s": None}
    assert grade("audit_response", answer, oracle("audit_response", answer))["correct"] is True
    for latency in (0, False, "null"):
        assert grade("audit_response", answer, {**answer, "hr_latency_s": latency})["correct"] is False
    incomplete = {key: value for key, value in answer.items() if key != "hr_latency_s"}
    assert grade("audit_response", answer, incomplete)["correct"] is False


@pytest.mark.parametrize(("qtype", "field"), [("audit_brs", "brs_ms_mmhg"), ("audit_tfa", "gain")])
def test_relative_zero_reference(qtype, field):
    assert grade(qtype, {field: 0}, {field: 0})["correct"] is True
    assert grade(qtype, {field: 0}, {field: 0.01})["correct"] is False


def test_existing_hr_scoring_stays_relative():
    assert grade("mean_hr", {"bpm": 100}, {"bpm": 104})["correct"] is True
    assert grade("audit_rate", {"bpm": 100}, {"bpm": 104})["correct"] is False


def test_format_registration_preserves_every_original_prompt():
    # Read the unchanged historical registry literal: package initialization may
    # extend it, but must not overwrite any existing suite's answer format.
    source = ast.parse(Path(common.__file__).read_text())
    original = next(
        ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ANSWER_FORMATS" for target in node.targets)
    )
    assert not original.keys() & AUDIT_ANSWER_FORMATS.keys()
    assert set(AUDIT_ANSWER_FORMATS) == set(AUDIT_FIELDS)
    visible = {"dur_s": 60, "channels": [{"name": "ECG", "unit": "mV", "fs": 125}]}
    for qtype, format_text in original.items():
        assert ANSWER_FORMATS[qtype] == format_text
        assert case_prompt(visible, "Question?", qtype) == (
            "You are given a recorded signal: duration 60 s, channels: ECG (mV, 125 Hz).\n"
            "Question: Question?\n"
            "Answer as accurately as you can. End your reply with exactly one final line of the form\n"
            f"ANSWER: {format_text}"
        )
