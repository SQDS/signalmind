"""Synthetic accounting/geometry tests; no external runs, signals, or model calls."""

import pytest

from eval.part2_dev_audit import (
    annotation_intervals,
    episode_geometry,
    keyed,
    parse_transcript,
    score,
    union,
)


def assistant(blocks, message_id="m1"):
    return {"type": "assistant", "message": {"id": message_id, "content": blocks}}


def tool_use(tool_id="t1"):
    return {"type": "tool_use", "id": tool_id, "name": "mcp__tsmcp__ts_measure", "input": {"what": ["ecg_af_episodes"]}}


def tool_result(content, tool_id="t1", error=False):
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": error}]
        },
    }


def test_streamed_tools_join_by_id_and_keep_lines_without_sdk_or_thinking():
    use = tool_use()
    response = tool_result([{"type": "text", "text": '{"measurements":[]}'}, {"type": "image", "data": "ignored"}])
    events = [
        assistant([{"type": "thinking", "thinking": "never exported"}, use]),
        assistant([use]),
        {**response, "tool_use_result": {"duplicate": "not a second response"}},
        response,
        assistant([{"type": "text", "text": "Decision."}], "m2"),
        assistant([{"type": "text", "text": "Decision."}], "m2"),
        {"type": "result", "subtype": "success", "num_turns": 2, "result": "Decision."},
    ]
    result = parse_transcript(events)
    assert len(result["calls"]) == 1
    call = result["calls"][0]
    assert call["call_line"] == 1 and call["result_line"] == 3
    assert call["evidence"]["images"] == 1
    assert call["evidence"]["payloads"] == [{"measurements": []}]
    assert result["public_statements"] == [{"line": 5, "text": "Decision."}]
    assert result["session_result"]["num_turns"] == 2
    assert "never exported" not in str(result)


def test_oversized_payload_rejection_stays_error_not_available_index():
    events = [
        assistant([tool_use()]),
        tool_result("Tool response exceeds maximum allowed tokens", error=True),
    ]
    call = parse_transcript(events)["calls"][0]
    assert call["is_error"]
    assert not call["evidence"]["payloads"]
    assert "exceeds" in call["evidence"]["non_json_text"][0]
    assert not call["evidence_usable"]
    unflagged = parse_transcript(
        [
            assistant([tool_use()]),
            tool_result("Error: result (90,101 characters) exceeds maximum allowed tokens"),
        ]
    )["calls"][0]
    assert not unflagged["is_error"]  # original flag is preserved, not silently rewritten
    assert unflagged["presentation_error"]
    assert not unflagged["evidence_usable"]


def test_orphan_result_and_conflicting_stream_duplicates_fail_closed():
    with pytest.raises(ValueError, match="orphan"):
        parse_transcript([tool_result("error")])
    changed = {**tool_use(), "input": {"what": ["ecg_af_burden"]}}
    with pytest.raises(ValueError, match="inconsistent tool-use"):
        parse_transcript([assistant([tool_use()]), assistant([changed])])
    with pytest.raises(ValueError, match="inconsistent repeated"):
        parse_transcript([assistant([tool_use()]), tool_result("first"), tool_result("second")])


def test_out_of_order_parallel_results_are_joined_not_zipped():
    events = [
        assistant([tool_use("a"), tool_use("b")]),
        tool_result('{"value":2}', "b"),
        tool_result('{"value":1}', "a"),
    ]
    calls = parse_transcript(events)["calls"]
    assert [c["evidence"]["payloads"][0]["value"] for c in calls] == [1, 2]
    assert [c["result_line"] for c in calls] == [3, 2]


def test_annotation_persistence_initial_unknown_and_flutter():
    raw = annotation_intervals([0, 5, 6, 10, 20, 25], ["", "(AFIB\x00", "", "(N", "(AFL", "(N"], 1, 30)
    assert raw == [[5, 10], [20, 25]]
    assert union(raw, gap=30) == [[5.0, 25]]
    assert union([[0, 10], [40, 50]], gap=30) == [[0.0, 10.0], [40.0, 50.0]]
    assert union([[0, 10], [39.999, 50]], gap=30) == [[0.0, 50]]


def test_geometry_distinguishes_overlap_from_onset_and_detects_overmerge():
    episodes = [
        {"start": 100, "dur": 100, "span_s": 100, "af_s": 80, "intervals_s": [[100, 180]]},
        {"start": 250, "dur": 70, "span_s": 70, "af_s": 70, "intervals_s": [[250, 320]]},
    ]
    result = episode_geometry(episodes, [[0, 210], [240, 280]], [0, 210])
    assert result["fragmented"] is False
    assert result["candidate_overlaps_longest"]
    assert not result["candidate_start_within_60s"]
    assert result["episodes"][0]["reference_episodes_overlapped"] == [1]
    merged = [{"start": 100, "dur": 200, "af_s": 100, "intervals_s": [[100, 200]]}]
    assert episode_geometry(merged, [[0, 210], [240, 280]], [0, 210])["overmerged_across_reference_episodes"]
    assert episode_geometry([], [[0, 210]], [0, 210])["missing"]


def test_rejected_answer_remains_execution_failure_and_retry_is_phase_specific():
    answer = {"case_id": "dev", "qtype": "longest_af", "answer": {"t": 100}}
    error = {"class": "B_single_turn_contract", "pred": {"t": 100}}
    assert score(answer, None, error)["failure_class"] == "execution_contract"
    assert not score(answer, None, error)["correct"]
    gateway = {"class": "api_error"}
    assert score(answer, None, gateway)["failure_class"] == "infrastructure"
    assert score(answer, {"status": "ok", "pred": {"t": 100}}, gateway)["correct"]
    with pytest.raises(ValueError, match="unclassified"):
        score(answer, None)


def test_grading_edges_and_duplicate_case_guard():
    answer = {"case_id": "dev", "qtype": "longest_af", "answer": {"t": 100}}
    assert score(answer, {"pred": {"t": 160}})["correct"]
    assert not score(answer, {"pred": {"t": 160.001}})["correct"]
    burden = {"case_id": "dev", "qtype": "af_burden", "answer": {"percent": 30}}
    assert score(burden, {"pred": {"percent": 35}})["correct"]
    assert not score(burden, {"pred": {"percent": 35.001}})["correct"]
    with pytest.raises(ValueError, match="duplicate"):
        keyed([answer, answer])
