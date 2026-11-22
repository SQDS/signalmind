"""Responses adapter integration with fake HTTP responses only: no key reads or model requests."""

import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.unified import drivers  # noqa: E402
from eval.unified.loop import Limits, run_session  # noqa: E402
from eval.unified.responses import OpenAIResponsesDriver  # noqa: E402


@pytest.fixture(autouse=True)
def no_credentials_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("mock adapter tests must not read credentials or send requests")

    monkeypatch.setattr(drivers, "read_key", forbidden)
    monkeypatch.setattr(drivers.OpenAIChatDriver, "_post", forbidden)


def test_two_turn_tools_preserve_native_reasoning_phase_and_image_without_trace_leak(tmp_path):
    first_output = [
        {
            "id": "rs1",
            "type": "reasoning",
            "encrypted_content": "opaque-reasoning-one",
            "summary": [{"type": "summary_text", "text": "private-summary-marker"}],
        },
        {
            "id": "msg1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "I will inspect the waveform.", "annotations": []}],
        },
        {
            "id": "fc1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call-image",
            "name": "inspect",
            "arguments": '{"source":"signal.npy"}',
        },
        {
            "id": "fc2",
            "type": "function_call",
            "status": "completed",
            "call_id": "call-error",
            "name": "inspect",
            "arguments": '{"source":"missing.npy"}',
        },
    ]
    final_output = [
        {"id": "rs2", "type": "reasoning", "encrypted_content": "opaque-reasoning-two", "summary": []},
        {
            "id": "msg2",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": 'ANSWER: {"bpm": 72}', "annotations": []}],
        },
    ]
    replies = [
        {
            "id": "resp1",
            "model": "gpt-6-astra",
            "status": "completed",
            "output": first_output,
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 20},
                "output_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 25},
            },
        },
        {
            "id": "resp2",
            "model": "gpt-6-astra",
            "status": "completed",
            "output": final_output,
            "usage": {
                "input_tokens": 110,
                "input_tokens_details": {"cached_tokens": 70, "cache_write_tokens": 0},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        },
    ]
    sent = []

    def transport(body):
        sent.append(deepcopy(body))
        return replies.pop(0)

    class Tools:
        schemas = [
            {
                "name": "inspect",
                "description": "Inspect a signal.",
                "input_schema": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "window": {"type": "array"}},
                    "required": ["source"],
                },
            }
        ]

        async def call(self, name, args):
            assert name == "inspect"
            if args["source"] == "missing.npy":
                return [{"type": "text", "text": "File not found."}], True
            return [
                {"type": "text", "text": "Waveform view."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
            ], False

    driver = OpenAIResponsesDriver({"model": "gpt-6-astra", "vision": True}, max_tokens=16384, transport=transport)
    trace = tmp_path / "trace.jsonl"
    result = asyncio.run(
        run_session(driver, Tools(), "Measure from evidence.", "What is the pulse rate?", Limits(), trace, {})
    )

    assert result["error"] is None and result["turns"] == 2 and result["tool_calls"] == 2
    assert result["text"] == 'ANSWER: {"bpm": 72}' and result["stop"] == "end_turn"
    assert result["usage"] == {
        "input_tokens": 60,
        "output_tokens": 42,
        "cache_read_input_tokens": 130,
        "cache_creation_input_tokens": 20,
    }
    first, second = sent
    assert first["instructions"] == "Measure from evidence." and first["model"] == "gpt-6-astra"
    assert first["max_output_tokens"] == 16384 and first["reasoning"] == {"effort": "medium"}
    assert first["store"] is False and "include" not in first and "previous_response_id" not in first
    assert first["tools"][0]["strict"] is False
    assert first["tools"][0]["parameters"] == Tools.schemas[0]["input_schema"]
    assert first["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "What is the pulse rate?"}]}]
    assert second["input"][:5] == first["input"] + first_output  # native state exactly once, not assistant projection
    image_result, error_result = second["input"][5:]
    assert image_result == {
        "type": "function_call_output",
        "call_id": "call-image",
        "output": [
            {"type": "input_text", "text": "Waveform view."},
            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        ],
    }
    assert error_result["call_id"] == "call-error"
    assert error_result["output"] == [
        {"type": "input_text", "text": "Tool error (is_error=true):"},
        {"type": "input_text", "text": "File not found."},
    ]
    text = trace.read_text()
    assert "opaque-reasoning" not in text and "private-summary-marker" not in text and "encrypted_content" not in text
    assert "AA==" not in text  # loop still redacts images independently of the private Responses history
    events = [json.loads(line) for line in text.splitlines()]
    assistant = [event for event in events if event["type"] == "assistant"]
    assert [e["message"]["usage"]["reasoning_tokens"] for e in assistant] == [25, 5]

    # A subsequent user turn replays both native outputs and appends only the new user message.
    replies.append({"id": "resp3", "model": "gpt-6-astra", "status": "completed", "output": [], "usage": {}})
    previous_projection = [{"role": "user", "content": "What is the pulse rate?"}] + [
        {"role": event["type"], "content": event["message"]["content"]}
        for event in events
        if event["type"] in ("user", "assistant")
    ]
    driver.step(
        "Measure from evidence.",
        previous_projection + [{"role": "user", "content": "Explain the units."}],
        Tools.schemas,
    )
    assert sent[2]["input"] == second["input"] + final_output + [
        {"role": "user", "content": [{"type": "input_text", "text": "Explain the units."}]},
    ]


def test_incomplete_reasoning_response_maps_output_limit_and_keeps_token_usage():
    sent = []

    def transport(body):
        sent.append(body)
        return {
            "id": "limited",
            "model": "gpt-6-astra",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "id": "rs", "encrypted_content": "opaque", "summary": []}],
            "usage": {"input_tokens": 10, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 50}},
        }

    driver = OpenAIResponsesDriver({"model": "gpt-6-astra", "reasoning_effort": "high"}, transport=transport)
    step = driver.step("sys", [{"role": "user", "content": "q"}], [])
    assert sent[0]["reasoning"] == {"effort": "high"}
    assert step.stop_reason == "max_tokens" and step.content == []
    assert step.usage["output_tokens"] == step.usage["reasoning_tokens"] == 50
    assert step.usage["input_tokens"] == 10 and step.usage["cache_read_input_tokens"] == 0
    assert "opaque" not in json.dumps(step.__dict__)
