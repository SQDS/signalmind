"""Selection and unchanged-grader contract for the deterministic A0 baseline."""

import copy
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.a0_baseline import MODELS, answer_from_payload, grade_rows, grade_saved, select_longest  # noqa: E402


def test_longest_uses_span_not_support_score_or_input_order():
    episodes = [
        {"t": 200.0, "dur": 100.0, "span_s": 100.0, "af_s": 99.0, "score": 9.0},
        {"t": 900.0, "dur": 110.0, "span_s": 110.0, "af_s": 40.0, "score": 1.0},
        {"t": 500.0, "dur": 110.0, "span_s": 110.0, "af_s": 39.0, "score": 0.5},
    ]
    original = copy.deepcopy(episodes)
    assert answer_from_payload("longest_af", {"episodes": episodes}) == {"t": 500.0}
    assert select_longest(list(reversed(episodes))) == episodes[2]
    assert episodes == original


def test_m_native_spans_are_not_regrouped():
    episodes = [{"t": 0, "dur": 40}, {"t": 60, "dur": 40}, {"t": 200, "dur": 80}]
    # The 20 s gap must not be newly bridged, even though R uses a 30 s grouping.
    assert answer_from_payload("longest_af", {"episodes": episodes}) == {"t": 200.0}
    assert answer_from_payload("longest_af", {"episodes": []}) == {"t": None}


def test_burden_uses_whole_window_percentage_without_recalculation():
    payload = {"value": 80.0, "pct_of_readable": 80.0, "pct_of_window": 12.3, "af_s": 123.456}
    assert answer_from_payload("af_burden", payload) == {"percent": 12.3}
    with pytest.raises(KeyError):
        answer_from_payload("af_burden", {"value": 80.0})


@pytest.mark.parametrize(
    ("qtype", "truth", "prediction", "correct", "error"),
    [
        ("longest_af", {"t": 100.0}, {"t": 160.0}, True, 60.0),
        ("longest_af", {"t": 100.0}, {"t": 160.001}, False, 60.001),
        ("longest_af", {"t": None}, {"t": None}, True, None),
        ("longest_af", {"t": 100.0}, {"t": None}, False, None),
        ("af_burden", {"percent": 20.0}, {"percent": 25.0}, True, 5.0),
        ("af_burden", {"percent": 20.0}, {"percent": 25.001}, False, 5.001),
        ("af_burden", {"percent": 20.0}, None, None, None),
    ],
)
def test_grading_keeps_frozen_tolerance_and_no_answer_semantics(qtype, truth, prediction, correct, error):
    from eval.grade import grade

    source = [{"case_id": "synthetic", "qtype": qtype, "pred": prediction}]
    row = grade_rows(source, {"synthetic": truth})[0]
    assert row["grade"] == grade(qtype, truth, prediction)
    assert row["grade"]["correct"] is correct
    if error is None:
        assert row["absolute_error"] is None
    else:
        assert row["absolute_error"] == pytest.approx(error)
    assert "grade" not in source[0]


def test_saved_predictions_are_joined_by_case_and_graded_separately(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "baseline": "A0",
                "rows": [
                    {"case_id": "onset", "qtype": "longest_af", "pred": {"t": 100}},
                    {"case_id": "burden", "qtype": "af_burden", "pred": {"percent": 20}},
                ],
            }
        )
    )
    saved = predictions.read_bytes()
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        json.dumps({"case_id": "burden", "answer": {"percent": 26}})
        + "\n"
        + json.dumps({"case_id": "onset", "answer": {"t": 160}})
        + "\n"
    )
    # Reverse the case order and let the agent correct burden but lose onset.
    for model in MODELS:
        for arm in ("M", "R"):
            directory = tmp_path / "runs" / f"p5afdb-C-{model}-{arm}"
            directory.mkdir(parents=True)
            (directory / "results.jsonl").write_text(
                json.dumps({"case_id": "burden", "qtype": "af_burden", "pred": {"percent": 26}})
                + "\n"
                + json.dumps({"case_id": "onset", "qtype": "longest_af", "pred": None})
                + "\n"
            )
    output = tmp_path / "graded.json"
    grade_saved(Namespace(predictions=predictions, answers=answers, runs_dir=tmp_path / "runs", output=output))
    result = json.loads(output.read_text())
    assert predictions.read_bytes() == saved
    assert result["summary"] == {
        "longest_af": {"correct": 1, "total": 1},
        "af_burden": {"correct": 0, "total": 1},
    }
    for comparison in result["agent_comparisons"].values():
        assert comparison["paired"] == {"both_correct": 0, "a0_only": 1, "agent_only": 1, "both_wrong": 0}
