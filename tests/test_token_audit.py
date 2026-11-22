"""Synthetic accounting regression tests; no data, detector, or paid calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.token_audit import audit_case, match_cases  # noqa: E402


def assistant(message_id, blocks, *, input_tokens=10, output_tokens=1):
    return {
        "type": "assistant",
        "message": {
            "id": message_id,
            "content": blocks,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    }


def call(tool_id, handle="abc"):
    return {
        "type": "tool_use",
        "id": tool_id,
        "name": "mcp__tsmcp__ts_measure",
        "input": {"id": handle, "what": ["ecg_af_burden"]},
    }


def response(tool_id, text, *, is_error=False, image=False):
    content = [{"type": "text", "text": text}]
    if image:
        content.append({"type": "image", "source": {"data": "not-counted-as-text"}})
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": is_error}]
        },
        "tool_use_result": {"content": content},
    }


def run_audit(tmp_path, events, *, requests):
    usage = {"input_tokens": requests * 10, "output_tokens": requests * 2}
    events.append({"type": "result", "usage": usage})
    transcripts = tmp_path / "run-R" / "transcripts"
    transcripts.mkdir(parents=True)
    (transcripts / "example.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    row = {"case_id": "example", "qtype": "af_burden", "model": "synthetic", "turns": requests, "usage": usage}
    return audit_case(transcripts.parent, row, 100)


def test_streamed_message_and_envelope_dedup_and_history(tmp_path):
    events = [
        assistant("a", [call("t1")]),
        assistant("a", [{"type": "text", "text": "same request"}], output_tokens=2),
        response("t1", "éabc", image=True),
        assistant("b", [call("t2")], output_tokens=2),
        response("t2", "xyz"),
        assistant("c", [{"type": "text", "text": "final"}], output_tokens=2),
    ]
    result = run_audit(tmp_path, events, requests=3)
    assert result["requests"] == 3
    assert len(result["calls"]) == 2
    first, second = result["calls"]
    assert (first["chars"], first["utf8_bytes"], first["images"]) == (4, 5, 1)
    assert (first["following_requests"], first["history_exposure_chars"], first["resend_chars"]) == (2, 8, 4)
    assert (second["following_requests"], second["history_exposure_chars"], second["resend_chars"]) == (1, 3, 0)
    assert not first["repeated_query"]
    assert second["repeated_query"] and not second["repeat_after_error"]
    assert all(value == 0 for value in result["usage_minus_request_usage"].values())


def test_corrected_handle_repeat_and_retry_are_separate(tmp_path):
    events = [
        assistant("a", [call("bad", handle="signal")], output_tokens=2),
        response("bad", "unknown handle", is_error=True),
        {"type": "system", "subtype": "api_retry", "attempt": 1, "error_status": 503},
        assistant("b", [call("good", handle="deadbeef")], output_tokens=2),
        response("good", "{}"),
        assistant("c", [{"type": "text", "text": "final"}], output_tokens=2),
    ]
    result = run_audit(tmp_path, events, requests=3)
    assert len(result["api_retries"]) == 1
    assert result["calls"][0]["is_error"]
    assert result["calls"][1]["repeated_query"]
    assert result["calls"][1]["repeat_after_error"]
    assert result["requests"] == 3


def test_matching_never_pairs_error_with_success(tmp_path):
    events = [
        assistant("a", [call("bad")], output_tokens=2),
        response("bad", "error", is_error=True),
        assistant("b", [call("good")], output_tokens=2),
        response("good", "123456"),
        assistant("c", [{"type": "text", "text": "final"}], output_tokens=2),
    ]
    r = run_audit(tmp_path, events, requests=3)
    m = {**r, "arm": "M", "calls": [{**r["calls"][1], "chars": 4}]}
    pairs = match_cases([m, r])
    assert len(pairs[0]["identical_queries"]) == 1
    assert pairs[0]["identical_queries"][0]["delta_chars"] == 2
