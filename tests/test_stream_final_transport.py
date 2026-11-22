import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.stream_final_transport import stream_final_post, stream_final_post_with_reconnect  # noqa: E402


class FakeStream:
    is_success = True

    def __init__(self, events):
        self.events = events
        self.sent = None

    def stream(self, method, url, **kw):
        self.sent = (method, url, kw)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self):
        yield ": keepalive"
        for event in self.events:
            yield "data: " + json.dumps(event)


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_completed_native_items_replace_partial_terminal_snapshots(status):
    reasoning = {"type": "reasoning", "id": "rs1", "encrypted_content": "opaque-private"}
    call = {"type": "function_call", "call_id": "call1", "name": "measure", "arguments": "{}", "status": "completed"}
    stream = FakeStream(
        [
            {"type": "response.output_text.delta", "delta": "intermediate"},
            {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
            {"type": "response.output_item.done", "output_index": 1, "item": call},
            {
                "type": "response." + status,
                "response": {
                    "id": "response1",
                    "status": status,
                    "output": [{"type": "reasoning", "id": "rs1"}],
                    "usage": {"input_tokens": 12, "output_tokens": 34},
                },
            },
        ]
    )
    driver = SimpleNamespace(_http=stream, url="https://api.openai.com/v1/responses", key="test-only")
    body = {"model": "gpt-6-astra", "store": False, "max_output_tokens": 16384, "reasoning": {"effort": "medium"}}
    result = stream_final_post(driver, body)
    assert result["output"] == [reasoning, call]
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 34} and result["status"] == status
    assert stream.sent[2]["json"] == {**body, "stream": True} and "stream" not in body


def test_nonterminal_disconnect_does_not_invent_a_completed_response():
    stream = FakeStream([{"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}])
    driver = SimpleNamespace(_http=stream, url="mock://responses", key="test-only")
    with pytest.raises(RuntimeError, match="without a terminal"):
        stream_final_post(driver, {})


def test_disconnect_reconnects_without_changing_request_settings(monkeypatch):
    import httpx2

    from eval import stream_final_transport as transport

    class Disconnected:
        closed = False

        def stream(self, *args, **kwargs):
            raise httpx2.RemoteProtocolError("disconnected")

        def close(self):
            self.closed = True

    initial = Disconnected()
    fresh = FakeStream(
        [
            {
                "type": "response.completed",
                "response": {"id": "r2", "status": "completed", "output": [], "usage": {"output_tokens": 1}},
            }
        ]
    )
    monkeypatch.setattr(transport.httpx2, "Client", lambda **kwargs: fresh)
    monkeypatch.setattr(transport.time, "sleep", lambda seconds: None)
    driver = SimpleNamespace(_http=initial, url="mock://responses", key="test-only")
    body = {"model": "gpt-6-astra", "reasoning": {"effort": "medium"}, "max_output_tokens": 16384}
    result = stream_final_post_with_reconnect(driver, body)
    assert result["id"] == "r2" and initial.closed and driver._network_disconnects == 1
    assert fresh.sent[2]["json"] == {**body, "stream": True}


def test_enabling_streaming_replaces_the_request_callable_used_by_step():
    from eval.run_processing import enable_response_streaming
    from eval.unified.responses import OpenAIResponsesDriver

    def ordinary_transport(body):
        raise AssertionError("ordinary transport must not remain bound after streaming is enabled")

    driver = OpenAIResponsesDriver({"model": "gpt-6-astra", "vision": True}, transport=ordinary_transport)
    driver.key = "test-only"
    driver._http = FakeStream(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "streamed",
                    "model": "gpt-6-astra",
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            }
        ]
    )
    enable_response_streaming(driver)
    result = driver.step("system", [{"role": "user", "content": "hello"}], [])
    assert result.content == [{"type": "text", "text": "done"}]
    assert driver._http.sent[2]["json"]["stream"] is True
