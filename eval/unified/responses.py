"""Responses API adapter for one append-only unified-loop session.

Keep native output items, including encrypted reasoning and assistant phase, in the private request history.
Only ordinary assistant text and function calls enter Step.content and the user-facing transcript.
See https://developers.openai.com/api/docs/guides/reasoning and /guides/function-calling.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy

from eval.unified.drivers import OpenAIChatDriver, Step


class OpenAIResponsesDriver(OpenAIChatDriver):
    """Reuse the existing authenticated HTTP transport, with Responses request/state semantics."""

    def __init__(self, cfg: dict, max_tokens: int = 4096, transport=None):
        super().__init__(cfg, max_tokens=max_tokens, transport=transport)
        self.url = (
            self.url.removesuffix("/chat/completions") + "/responses" if transport is None else "mock://responses"
        )
        self._items: list[dict] = []
        self._seen_messages = 0
        self._http = None
        if transport is None:
            import httpx2

            self._http = httpx2.Client(
                timeout=300.0,
                transport=httpx2.HTTPTransport(retries=2, limits=httpx2.Limits(keepalive_expiry=180.0)),
            )

    def _post(self, body: dict) -> dict:
        response = self._http.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"})
        if not response.is_success:
            detail = response.text.replace(self.key, "[redacted]")[:300]
            raise RuntimeError(f"HTTP {response.status_code} from the gateway: {detail}")
        return response.json()

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def _parts(self, content) -> list[dict]:
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        parts = []
        for block in blocks:
            if block["type"] == "text":
                parts.append({"type": "input_text", "text": block["text"]})
            elif block["type"] == "image" and self.cfg.get("vision"):
                source = block["source"]
                url = (
                    f"data:{source['media_type']};base64,{source['data']}"
                    if source["type"] == "base64"
                    else source["url"]
                )
                parts.append({"type": "input_image", "image_url": url})
        return parts

    def _new_items(self, messages: list[dict]) -> list[dict]:
        items = []
        for message in messages:
            if message["role"] == "assistant":
                continue  # native output already stored; the loop's projection omits reasoning and phase
            blocks = message["content"]
            blocks = blocks if isinstance(blocks, list) else [{"type": "text", "text": blocks}]
            pending = []
            for block in blocks:
                if block["type"] != "tool_result":
                    pending.extend(self._parts([block]))
                    continue
                if pending:
                    items.append({"role": message["role"], "content": pending})
                    pending = []
                output = self._parts(block.get("content", ""))
                if block.get("is_error"):
                    output.insert(0, {"type": "input_text", "text": "Tool error (is_error=true):"})
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block["tool_use_id"],
                        "output": output or [{"type": "input_text", "text": "(empty tool result)"}],
                    }
                )
            if pending:
                items.append({"role": message["role"], "content": pending})
        return items

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Step:
        inputs = self._items + self._new_items(messages[self._seen_messages :])
        body = {
            "model": self.cfg["model"],
            "instructions": system,
            "input": inputs,
            "store": False,
            "max_output_tokens": self.max_tokens,
            "reasoning": {"effort": self.cfg.get("reasoning_effort", "medium")},
            "tools": [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                    "strict": False,
                }
                for tool in tools
            ],
        }
        started = time.monotonic()
        response = self.transport(body)
        if response.get("error") or response.get("status") == "failed":
            raise RuntimeError(f"Responses failed: {(response.get('error') or {}).get('code', 'unknown')}")
        output = response.get("output") or []
        self._items = inputs + deepcopy(output)
        self._seen_messages = len(messages)
        content = []
        for item in output:
            if item["type"] == "message":
                for part in item.get("content", []):
                    if part["type"] == "output_text":
                        content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "refusal":
                        content.append({"type": "text", "text": part["refusal"]})
            elif item["type"] == "function_call" and item.get("status", "completed") == "completed":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except ValueError:
                    arguments = {"__unparseable_arguments__": item.get("arguments")}
                content.append({"type": "tool_use", "id": item["call_id"], "name": item["name"], "input": arguments})
        raw_usage = response.get("usage") or {}
        input_details = raw_usage.get("input_tokens_details") or {}
        cached, written = input_details.get("cached_tokens") or 0, input_details.get("cache_write_tokens") or 0
        usage = {
            "input_tokens": (raw_usage.get("input_tokens") or 0) - cached - written,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": written,
            "output_tokens": raw_usage.get("output_tokens") or 0,  # already includes reasoning tokens
            "reasoning_tokens": (raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0,
        }
        reason = (response.get("incomplete_details") or {}).get("reason")
        stop = "tool_use" if any(c["type"] == "tool_use" for c in content) else "end_turn"
        if response.get("status") == "incomplete":
            stop = "max_tokens" if reason == "max_output_tokens" else reason or "incomplete"
        return Step(
            content,
            usage,
            stop,
            response.get("model"),
            response.get("id"),
            round((time.monotonic() - started) * 1000, 1),
            {"response_status": response.get("status"), "incomplete_reason": reason},
        )
