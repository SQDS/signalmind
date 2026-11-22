"""P0 relay smoke test: does the configured Anthropic endpoint behave like the real API?

Checks: served model == requested; usage has cache fields and caching works; count_tokens works;
real image token cost (usage diff with/without a PNG); 2-turn tool loop with thinking; Batches reachable.
Cost: well under $1. Run: uv run --group eval python eval/smoke_relay.py
"""

from __future__ import annotations

import base64
import io
import json
import sys

import anthropic
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-5-5"
client = anthropic.Anthropic()
report: dict = {"model_requested": MODEL}


def png(width_px: int, height_px: int) -> str:
    fig = Figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    t = np.linspace(0, 10, 2500)
    ax.plot(
        t,
        np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.default_rng(0).standard_normal(t.size),
        lw=0.6,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def usage_dict(u) -> dict:
    return {
        k: getattr(u, k, None)
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    }


def call(**kw):
    kw.setdefault("model", MODEL)
    kw.setdefault("max_tokens", 2000)
    kw.setdefault("output_config", {"effort": "low"})
    return client.messages.create(**kw)


def step(name, fn):
    try:
        report[name] = fn()
    except Exception as e:  # record, keep going
        report[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    print(name, json.dumps(report[name], ensure_ascii=False)[:400], flush=True)


# 1. served model
def served():
    r = call(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=200,
    )
    return {
        "served": r.model,
        "match": r.model == MODEL,
        "usage": usage_dict(r.usage),
        "stop": r.stop_reason,
    }


step("served_model", served)

# 2. caching: ~2k-token stable system prompt, two identical calls
SYSTEM = "You are a signal analysis assistant. " + " ".join(
    f"Rule {i}: be precise about units and time." for i in range(250)
)


def caching():
    out = []
    for _ in range(2):
        r = call(
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "Reply: ok"}],
            max_tokens=200,
        )
        out.append(usage_dict(r.usage))
    return {
        "first": out[0],
        "second": out[1],
        "cache_works": bool(out[1]["cache_read_input_tokens"]),
    }


step("caching", caching)


# 3. count_tokens
def count():
    r = client.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": "hello world"}])
    return {"input_tokens": r.input_tokens}


step("count_tokens", count)


# 4. image token cost via usage diff (and count_tokens if available)
def image_cost():
    res = {}
    for w, h in [(1200, 400), (2400, 800)]:
        img = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": png(w, h)},
        }
        with_img = call(
            messages=[
                {
                    "role": "user",
                    "content": [img, {"type": "text", "text": "Reply: ok"}],
                }
            ],
            max_tokens=200,
        )
        res[f"{w}x{h}"] = with_img.usage.input_tokens
    base = call(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Reply: ok"}]}],
        max_tokens=200,
    )
    res["text_only"] = base.usage.input_tokens
    res["img_tokens"] = {k: v - res["text_only"] for k, v in res.items() if "x" in k}
    return res


step("image_tokens", image_cost)

# 5. two-turn tool loop with thinking, history append-only
TOOLS = [
    {
        "name": "ts_measure",
        "description": "Measure a quantity of the loaded signal. Returns compact JSON.",
        "input_schema": {
            "type": "object",
            "properties": {"what": {"type": "string"}},
            "required": ["what"],
            "additionalProperties": False,
        },
    }
]


def tool_loop():
    msgs = [
        {
            "role": "user",
            "content": "Use ts_measure to get the 'period' of the signal, then tell me the period.",
        }
    ]
    turns = []
    for _ in range(3):
        r = call(messages=msgs, tools=TOOLS, output_config={"effort": "medium"})
        turns.append(
            {
                "stop": r.stop_reason,
                "blocks": [b.type for b in r.content],
                "usage": usage_dict(r.usage),
            }
        )
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            break
        results = [
            {
                "type": "tool_result",
                "tool_use_id": b.id,
                "content": '{"period":0.82,"unit":"s"}',
            }
            for b in r.content
            if b.type == "tool_use"
        ]
        msgs.append({"role": "user", "content": results})
    final = next((b.text for b in r.content if b.type == "text"), "")
    return {"turns": turns, "final": final[:120]}


step("tool_loop", tool_loop)


# 6. batches reachable (list only, no cost)
def batches():
    page = client.messages.batches.list(limit=1)
    return {"reachable": True, "n": len(page.data)}


step("batches", batches)

with open("eval/smoke_relay_result.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print("saved eval/smoke_relay_result.json")
