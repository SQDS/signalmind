"""Eval self-tests that need no data files and no API: grader sanity + scripted-baseline regression guard."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import scripted  # noqa: E402
from eval.common import parse_answer  # noqa: E402
from eval.grade import grade, null, oracle  # noqa: E402
from eval.tasks import synthetic  # noqa: E402


@pytest.fixture(scope="module")
def dev_cases():
    return list(synthetic.build(20260930, "dev"))


def test_oracle_perfect_null_poor(dev_cases):
    o = [grade(q, a, oracle(q, a))["correct"] for _, q, _, _, a, _ in dev_cases]
    n = [grade(q, a, null(q))["correct"] for _, q, _, _, a, _ in dev_cases]
    assert all(o) and sum(n) / len(n) < 0.5


def test_no_answer_is_not_negative():
    assert grade("change_time", {"t": None}, None)["status"] == "no_answer"
    assert parse_answer('blah\nANSWER: {"t": null}') == {"t": None}
    assert parse_answer("ANSWER: not json") is None


def test_scripted_baseline_regression(dev_cases):
    ok = [
        grade(q, a, scripted.answer(q, question, x, synthetic.FS))["correct"] for _, q, _, x, a, question in dev_cases
    ]
    assert sum(ok) == len(ok)  # algorithms saturate the dev family; any drop is a regression
