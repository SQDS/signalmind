"""Offline synthetic parsing checks; no source data, sockets or models."""

import json

import pytest

from eval.audit_analysis import c_geometry, keyed, numeric_errors, parse_transcript, read_jsonl, reconcile


def use(key="private-call", **args):
    return {"type": "tool_use", "id": key, "name": "mcp__tsmcp__ts_zoom", "input": args}


def assistant(*blocks):
    return {"type": "assistant", "message": {"id": "private-request", "content": list(blocks)}}


def response(key="private-call", text="{}", error=False):
    return {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": key, "content": text, "is_error": error}]},
    }


def final(**kwargs):
    return {"type": "result", "subtype": "success", "num_turns": 2, "result": "done", **kwargs}


def test_stream_join_dedup_and_private_data_not_exported():
    call = use(id="private-handle", source="/private/source", window=[1, 5], channels=["ABP"])
    events = [
        assistant({"type": "thinking", "thinking": "private-thought"}, call),
        assistant(call),
        {**response(text='{"value":4}'), "tool_use_result": {"secret": "private-copy"}},
        response(text='{"value":4}'),
        final(),
    ]
    trace = parse_transcript(events)
    assert len(trace["calls"]) == 1
    assert trace["calls"][0]["call_line"] == 1
    assert trace["calls"][0]["result_line"] == 3
    exported = json.dumps({k: v for k, v in trace.items() if k != "terminal"})
    assert "private" not in exported


def test_parallel_results_and_missing_result_fail_closed():
    events = [assistant(use("a"), use("b")), response("b"), response("a"), final()]
    assert [c["result_line"] for c in parse_transcript(events)["calls"]] == [3, 2]
    with pytest.raises(ValueError, match="missing result"):
        parse_transcript([assistant(use()), final()])
    with pytest.raises(ValueError, match="orphan"):
        parse_transcript([response(), final()])
    with pytest.raises(ValueError, match="inconsistent tool-use") as error:
        parse_transcript([assistant(use(window=[1, 2])), assistant(use(window=[2, 3])), final()])
    assert "private" not in str(error.value)


def test_conflicting_duplicate_and_terminal_validation():
    with pytest.raises(ValueError, match="inconsistent repeated"):
        parse_transcript([assistant(use()), response(text="one"), response(text="two"), final()])
    with pytest.raises(ValueError, match="one terminal"):
        parse_transcript([final(), final()])


def test_channel_warning_overrides_misleading_channels_and_geometry_is_half_open():
    payload = '{"window":[0,180],"channels":["ABP","CBFV"],"warn":"view level uses one channel: ABP"}'
    trace = parse_transcript(
        [
            assistant(use(view="level")),
            response(text=payload),
            assistant(use("next", window=[45, 70])),
            response("next"),
            final(),
        ]
    )
    visible = {"dur_s": 180, "channels": [{"name": "ABP"}, {"name": "CBFV"}]}
    truth = {"answer": {"t": 72}, "perturbation": {"start_s": 70, "channel": "CBFV"}}
    geometry = c_geometry(trace["calls"], visible, truth, "audit_motion")
    assert geometry["successful_windows"][0]["channels"] == ["ABP"]
    assert geometry["onset_covered_by_lines"] == []


def test_rejected_evidence_is_not_geometric_support():
    trace = parse_transcript([assistant(use()), response(text="Error: exceeds maximum allowed tokens"), final()])
    assert trace["unusable_tool_results"] == 1
    geometry = c_geometry(
        trace["calls"], {"dur_s": 10, "channels": [{"name": "ABP"}]}, {"answer": {"t": 4}}, "audit_tilt"
    )
    assert geometry["onset_covered_by_lines"] == []


def test_exact_saved_usage_reconciles_without_substituting_sdk_model_usage():
    row = {
        "status": "ok",
        "usage": {"input_tokens": 8},
        "turns": 2,
        "text": "done",
        "cc_cost_usd": 0.3,
        "tool_calls": [],
    }
    trace = parse_transcript(
        [final(usage={"input_tokens": 8}, total_cost_usd=0.3, modelUsage={"x": {"inputTokens": 1000}})]
    )
    reconcile(row, trace)
    with pytest.raises(ValueError, match="usage mismatch"):
        reconcile({**row, "usage": {"input_tokens": 1000}}, trace)


def test_field_errors_keep_units_zero_and_categorical_separate():
    errors = numeric_errors({"sbp": 100, "t": 0, "channel": "ABP"}, {"sbp": 105, "t": 2, "channel": "PPG"})
    assert errors == {
        "sbp": {"signed": 5, "absolute": 5, "relative": 0.05},
        "t": {"signed": 2, "absolute": 2, "relative": None},
    }
    assert numeric_errors({"t": 1}, {"t": True})["t"]["absolute"] is None
    with pytest.raises(ValueError, match="duplicate") as error:
        keyed([{"case_id": "private"}, {"case_id": "private"}])
    assert "private" not in str(error.value)


def test_blank_lines_cannot_silently_shift_physical_trace_citations(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{}\n\n{"type":"result"}\n')
    with pytest.raises(ValueError, match="physical line citations"):
        read_jsonl(path)
