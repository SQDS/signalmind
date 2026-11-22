"""Model drivers: one `step(system, messages, tools)` per API family, returning Anthropic-style content blocks.

Security: a credentialed request is only sent to an https:// endpoint, or to localhost (an authenticated tunnel).
Keys come from a 0600 key file or an environment variable; they are never logged, printed or written.
"""

from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# Model registry. Prices in USD per million tokens; None = not verified. The runner records unknown billing costs
# as null and uses a separately labelled reference-price estimate for budget checks. Endpoint IDs need a live smoke.
MODELS: dict[str, dict] = {
    "deepseek-flash": {
        "api": "anthropic", "base_url": "https://api.deepseek.com/anthropic",
        "key_file": "~/.config/tsmcp-eval/deepseek.key", "model": "deepseek-flash", "vision": True,
        "balance_url": "https://api.deepseek.com/user/balance",
        # Peak USD / 1M tokens, verified 2026-10-29; off-peak billing is lower. Cache writes use uncached input.
        # https://api-docs.deepseek.com/quick_start/pricing/
        "prices": {"in": 0.3, "out": 1.2, "cache_read": 0.006, "cache_write": 0.3},
    },
    "claude-haiku": {
        "api": "anthropic", "base_url_env": "ANTHROPIC_BASE_URL", "key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": "claude-haiku-5-5", "vision": True, "prices": None,
    },
    "gpt-6-terra": {
        "api": "openai", "base_url_env": "GPT6_TERRA_BASE_URL", "key_env": "GPT6_TERRA_API_KEY",
        "model": "gpt-6-terra", "vision": None, "prices": None,
    },
    "gpt-6-astra": {
        "api": "responses", "base_url": "https://api.openai.com/v1",
        "key_file": "~/.config/tsmcp-eval/astra-test.key", "model": "gpt-6-astra",
        "vision": True, "reasoning_effort": "medium",
        # Standard long-context rates are a conservative per-session budget estimate, not a bill.
        # https://developers.openai.com/api/docs/models/gpt-6-astra (checked 2026-10-30)
        "prices": {"in": 20.0, "out": 75.0, "cache_read": 2.0, "cache_write": 25.0},
        "short_context_prices": {"in": 10.0, "out": 50.0, "cache_read": 1.0, "cache_write": 12.5},
        "long_context_threshold": 272000, "price_basis": "standard_long_context_upper_bound",
    },
}  # fmt: skip


class InsecureEndpoint(RuntimeError):
    pass


def secure_url(url: str) -> str:
    u = urlparse(url or "")
    if u.scheme == "https":
        return url.rstrip("/")
    if u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1"):
        return url.rstrip("/")  # an authenticated local tunnel terminates here
    raise InsecureEndpoint(f"refusing to send credentials to {u.scheme or '?'}://{u.hostname or '?'}: need https:// or a "
                           "localhost tunnel")  # fmt: skip


def read_key(cfg: dict) -> str:
    if cfg.get("key_file"):
        p = Path(cfg["key_file"]).expanduser()
        if not p.exists():
            raise RuntimeError(f"key file {p} missing")
        if p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise RuntimeError(f"key file {p} must be mode 0600")
        return p.read_text().strip()
    key = os.environ.get(cfg.get("key_env", ""), "")
    if not key:
        raise RuntimeError(f"environment variable {cfg.get('key_env')} is not set")
    return key


def base_url(cfg: dict) -> str:
    return secure_url(cfg.get("base_url") or os.environ.get(cfg.get("base_url_env", ""), ""))


@dataclass
class Step:
    content: list[dict]
    usage: dict
    stop_reason: str | None
    model: str | None
    id: str | None = None
    raw_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class AnthropicDriver:
    """Messages API with native tool use (Claude; DeepSeek's Anthropic-compatible endpoint)."""

    def __init__(self, cfg: dict, max_tokens: int = 4096, prompt_cache: bool = True, client=None):
        self.cfg, self.max_tokens, self.prompt_cache = cfg, max_tokens, prompt_cache
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=read_key(cfg), base_url=base_url(cfg), max_retries=2, timeout=300)
        self.client = client

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Step:
        msgs = messages
        if self.prompt_cache and msgs:  # incremental prompt caching: mark the newest block (identical in all arms)
            msgs = [dict(m) for m in messages]
            last = msgs[-1]
            content = (
                last["content"] if isinstance(last["content"], list) else [{"type": "text", "text": last["content"]}]
            )
            content = [dict(c) for c in content]
            content[-1]["cache_control"] = {"type": "ephemeral"}
            msgs[-1] = {**last, "content": content}
        t = time.monotonic()
        r = self.client.messages.create(model=self.cfg["model"], system=system, messages=msgs, tools=tools,
                                        max_tokens=self.max_tokens)  # fmt: skip
        content = [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b) for b in r.content]
        u = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
        usage = {
            k: u.get(k) or 0
            for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        }
        return Step(content, usage, r.stop_reason, getattr(r, "model", None), getattr(r, "id", None),
                    round((time.monotonic() - t) * 1000, 1))  # fmt: skip

    def close(self) -> None:
        self.client.close()


class OpenAIChatDriver:
    """OpenAI-compatible chat completions with function tools (for an OpenAI-compatible gateway)."""

    def __init__(self, cfg: dict, max_tokens: int = 4096, transport=None):
        self.cfg, self.max_tokens = cfg, max_tokens
        self.url = base_url(cfg) + "/chat/completions" if transport is None else "mock://chat"
        self.key = read_key(cfg) if transport is None else "mock"
        self.transport = transport or self._post

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})  # fmt: skip
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:  # never echo the request (it carries the key header)
                detail = e.read().decode(errors="replace").replace(self.key, "[redacted]")[:300]
                raise RuntimeError(f"HTTP {e.code} from the gateway: {detail}") from None
            except urllib.error.URLError as e:
                if not isinstance(e.reason, ConnectionResetError) or attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))

    @staticmethod
    def to_openai(system: str, messages: list[dict], vision: bool) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            if m["role"] == "assistant":
                text = "".join(c.get("text", "") for c in content if c["type"] == "text")
                calls = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
                         for c in content if c["type"] == "tool_use"]  # fmt: skip
                out.append({"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})})
                continue
            images = []
            for c in content:
                if c["type"] == "tool_result":
                    parts = (
                        c["content"]
                        if isinstance(c["content"], list)
                        else [{"type": "text", "text": str(c["content"])}]
                    )
                    text = "\n".join(p.get("text", "") for p in parts if p["type"] == "text")
                    images += [p for p in parts if p["type"] == "image"]
                    out.append({"role": "tool", "tool_call_id": c["tool_use_id"], "content": text or "(image)"})
                elif c["type"] == "text":
                    out.append({"role": "user", "content": c["text"]})
            if images and vision:
                out.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{i['source']['media_type']};base64,{i['source']['data']}"}}
                    for i in images]})  # fmt: skip
        return out

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> Step:
        body = {"model": self.cfg["model"], "max_tokens": self.max_tokens,
                "messages": self.to_openai(system, messages, bool(self.cfg.get("vision"))),
                "tools": [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                            "parameters": t["input_schema"]}} for t in tools]}  # fmt: skip
        t = time.monotonic()
        r = self.transport(body)
        msg = r["choices"][0]["message"]
        content = []
        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})
        for call in msg.get("tool_calls") or []:
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except ValueError:
                args = {"__unparseable_arguments__": call["function"].get("arguments")}
            content.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": args})
        u = r.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        usage = {"input_tokens": (u.get("prompt_tokens") or 0) - cached, "output_tokens": u.get("completion_tokens") or 0,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}  # fmt: skip
        stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}.get(
            r["choices"][0].get("finish_reason")
        )
        return Step(content, usage, stop, r.get("model"), r.get("id"), round((time.monotonic() - t) * 1000, 1))


def make_driver(name: str, **kw):
    cfg = MODELS[name]
    if cfg["api"] == "responses":
        from eval.unified.responses import OpenAIResponsesDriver

        return OpenAIResponsesDriver(cfg, **kw)
    return AnthropicDriver(cfg, **kw) if cfg["api"] == "anthropic" else OpenAIChatDriver(cfg, **kw)


def cost_of(name: str, usage: dict) -> float | None:
    p = MODELS[name].get("prices")
    if not p:
        return None
    return (usage.get("input_tokens", 0) * p["in"] + usage.get("output_tokens", 0) * p["out"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]) / 1e6  # fmt: skip
