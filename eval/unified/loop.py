"""The model-independent tool loop. Transcripts use the same event shapes as Claude Code's stream-json
(`assistant` / `user` messages with `tool_use` / `tool_result` blocks, then `result`), so the existing trace sanitiser
and the grader's transcript readers work unchanged. Images are stored as placeholders with a hash."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Limits:
    max_turns: int = 40
    max_tool_calls: int = 30
    max_seconds: float = 900.0
    max_output_tokens_total: int = 60000


def _redact_images(blocks: list[dict]) -> list[dict]:
    out = []
    for b in blocks:
        if b.get("type") == "image":
            data = b.get("source", {}).get("data", "")
            out.append({"type": "image", "sha256": hashlib.sha256(data.encode()).hexdigest()[:16], "bytes": len(data)})
        elif b.get("type") == "tool_result" and isinstance(b.get("content"), list):
            out.append({**b, "content": _redact_images(b["content"])})
        else:
            out.append(b)
    return out


async def run_session(
    driver, tools, system: str, prompt: str, limits: Limits, transcript: Path, meta: dict, budget_exhausted=None
) -> dict:
    transcript.parent.mkdir(parents=True, exist_ok=True)
    f = open(transcript, "w")  # noqa: SIM115 - closed in finally
    t0 = time.monotonic()
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    messages: list[dict] = [{"role": "user", "content": prompt}]
    turns = tool_calls = 0
    final_text, stop, error, models = "", "max_turns", None, set()

    def emit(obj):
        f.write(json.dumps(obj) + "\n")
        f.flush()

    emit({"type": "system", "subtype": "init", "unified_runner": True, **meta})
    try:
        while turns < limits.max_turns:
            if budget_exhausted is not None and budget_exhausted(usage):
                stop = "estimated_cost_budget"
                break
            if time.monotonic() - t0 > limits.max_seconds:
                stop = "timeout"
                break
            if usage["output_tokens"] > limits.max_output_tokens_total:
                stop = "output_token_budget"
                break
            step = await asyncio.to_thread(driver.step, system, messages, tools.schemas)
            turns += 1
            models.add(step.model)
            for k in usage:
                usage[k] += step.usage.get(k, 0)
            emit({"type": "assistant", "message": {"id": step.id, "model": step.model, "content": _redact_images(step.content),
                                                   "usage": step.usage, "stop_reason": step.stop_reason}})  # fmt: skip
            messages.append({"role": "assistant", "content": step.content})
            uses = [b for b in step.content if b.get("type") == "tool_use"]
            texts = [b.get("text", "") for b in step.content if b.get("type") == "text"]
            if not uses:
                final_text, stop = "\n".join(texts), step.stop_reason or "end_turn"
                break
            results = []
            for u in uses:
                if tool_calls >= limits.max_tool_calls:
                    blocks, is_err = (
                        [{"type": "text", "text": "tool-call budget exhausted; give your final answer now"}],
                        True,
                    )
                else:
                    tool_calls += 1
                    blocks, is_err = await tools.call(u["name"], u.get("input") or {})
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": blocks, "is_error": is_err})
            emit({"type": "user", "message": {"role": "user", "content": _redact_images(results)}})
            messages.append({"role": "user", "content": results})
    except Exception as e:  # noqa: BLE001 - recorded as an execution failure, never raised past the session
        error, stop = f"{type(e).__name__}: {str(e)[:300]}", "error"
    finally:
        wall = round(time.monotonic() - t0, 1)
        emit({"type": "result", "subtype": stop, "is_error": error is not None, "result": final_text, "usage": usage,
              "num_turns": turns, "duration_ms": int(wall * 1000), "error": error})  # fmt: skip
        f.close()
    return {"text": final_text, "stop": stop, "error": error, "usage": usage, "turns": turns, "tool_calls": tool_calls,
            "latency_s": wall, "served_models": sorted(m for m in models if m)}  # fmt: skip
