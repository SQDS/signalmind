"""Replay accounting and true MCP dispatch, including errors and state isolation."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from mcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.compact_v2_replay import (  # noqa: E402
    dispatch,
    parse_transcript,
    remap_arguments,
    replay_transcript,
    result_stats,
    summary,
)
from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.server.app import build_server  # noqa: E402


def assistant(request, blocks):
    return {"type": "assistant", "message": {"id": request, "content": blocks}}


def tool(call_id, name="ts_measure", arguments=None):
    return {"type": "tool_use", "id": call_id, "name": "mcp__tsmcp__" + name, "input": arguments or {"id": "bad"}}


def response(call_id, content, error=False):
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": error}]
        },
        "tool_use_result": {"content": content},
    }


def test_transcript_parser_keeps_errors_unicode_and_distinct_later_requests(tmp_path):
    call = tool("t1", arguments={"id": "bad", "what": ["z", "a"], "unrecognized": 5})
    events = [
        assistant("a", [call]),
        assistant("a", [call]),
        response("t1", [{"type": "text", "text": "é错"}, {"type": "image", "data": "not-text"}], True),
        assistant("a", [{"type": "text", "text": "stream fragment"}]),
        assistant("b", [tool("t2")]),
        response("t2", "yes"),
        assistant("c", [{"type": "text", "text": "final"}]),
        {"type": "result"},
    ]
    path = tmp_path / "case.jsonl"
    path.write_text("\n".join(map(json.dumps, events)))
    parsed = parse_transcript(path)
    assert parsed["requests"] == 3
    first, second = parsed["calls"]
    assert first["arguments"] == call["input"]
    assert first["following_requests"] == 2
    assert (first["original"]["chars"], first["original"]["utf8_bytes"], first["original"]["images"]) == (2, 5, 1)
    assert first["original"]["is_error"] and first["original"]["history_exposure_chars"] == 4
    assert second["following_requests"] == 1


def test_remap_changes_only_real_sources_and_previously_returned_handles(tmp_path):
    original = {"id": "guessed", "source": "/tmp/tsmcp-C-test/work/signal.npy", "what": ["z", "a"], "window": [-1, 2]}
    changed = remap_arguments(original, tmp_path / "signal.npy", {"known": "new"})
    assert changed == {**original, "source": str(tmp_path / "signal.npy")}
    assert remap_arguments({"id": "known", "extra": 7}, tmp_path, {"known": "new"}) == {"id": "new", "extra": 7}
    assert original["source"] == "/tmp/tsmcp-C-test/work/signal.npy"


def test_direct_request_handler_matches_in_process_client_errors(tmp_path):
    async def compare():
        server = build_server(SignalStore([tmp_path]), interface="v2")
        args = {"id": "坏句柄", "what": ["ecg_af_burden"]}
        direct = await dispatch(server, "ts_measure", args)
        async with Client(build_server(SignalStore([tmp_path]), interface="v2")) as client:
            wire = await client.call_tool("ts_measure", args)
        assert direct.content == wire.content
        assert direct.is_error == wire.is_error
        assert direct.is_error

    asyncio.run(compare())


def test_replay_separate_forget_state_and_original_order(tmp_path, monkeypatch):
    monkeypatch.setenv("TSMCP_CACHE_DIR", str(tmp_path / "cache"))
    signal = tmp_path / "signal.npy"
    np.save(signal, np.sin(np.arange(400) / 10).astype(np.float32), allow_pickle=False)
    specs = [
        ("ts_load", {"source": "/tmp/tsmcp-C-test/work/signal.npy", "fs": 100}, "old"),
        ("ts_forget", {"id": "old"}, None),
        ("ts_describe", {"id": "old"}, None),
        ("ts_load", {"data": [0, 1], "fs": "bad-rate", "id": "probe"}, None),
    ]
    calls = [
        {
            "tool_use_id": str(i),
            "tool": name,
            "arguments": args,
            "original_returned_handle": old,
            "following_requests": 1,
            "original": result_stats(["old"], 0, i > 1, 1),
        }
        for i, (name, args, old) in enumerate(specs)
    ]
    result = asyncio.run(replay_transcript({"calls": calls}, signal, {}, tmp_path / "isolated"))
    assert [call["tool"] for call in result["calls"]] == [name for name, _, _ in specs]
    for version in ("v1", "v2"):
        assert result["calls"][1][version]["is_error"] is False
        assert result["calls"][2][version]["is_error"] is True
        assert result["calls"][3][version]["is_error"] is True
        assert "fs" in result["calls"][3][version]["error_text"]
    summed = summary(result["calls"])
    assert summed["calls"] == 4 and summed["error_status_mismatches"] == 0
    assert summed["v1"]["errors"] == summed["v2"]["errors"] == 2


def test_spawn_workers_match_serial_case_replay(tmp_path):
    """The parallel eval path preserves identical serialized counts and forget state."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from eval.compact_v2_replay import replay_case_job

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    signal = case_dir / "signal.npy"
    np.save(signal, np.sin(np.arange(400) / 10).astype(np.float32), allow_pickle=False)
    transcript = tmp_path / "run-R" / "transcripts" / "case.jsonl"
    transcript.parent.mkdir(parents=True)
    events = [
        assistant("a", [tool("t1", "ts_load", {"source": "/tmp/tsmcp-C-test/work/signal.npy", "fs": 100})]),
        response("t1", '{"id":"old"}'),
        assistant("b", [tool("t2", "ts_forget", {"id": "old"})]),
        response("t2", '{"id":"old","forgotten":true}'),
        assistant("c", [tool("t3", "ts_describe", {"id": "old"})]),
        response("t3", "unknown", True),
        assistant("d", [{"type": "text", "text": "final"}]),
        {"type": "result"},
    ]
    transcript.write_text("\n".join(map(json.dumps, events)))
    serial = replay_case_job(([transcript], signal, tmp_path / "serial"))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        groups = list(pool.map(replay_case_job, [([transcript], signal, tmp_path / f"worker{i}") for i in range(2)]))
    assert groups == [serial, serial]
