"""Single-shot baselines through the Messages API: A (numbers in prompt), B (one strip image).

  uv run --group eval python eval/run_api.py --suite p1 --arm A [--limit 5] [--effort medium]

Per row: served model (asserted), usage (all four token fields), cost at official prices, latency, stop_reason.
Infra failures go to errors.jsonl, never into results. Responses are cached on disk by request hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.arms.inputs import numeric_text, strip_png  # noqa: E402
from eval.common import (  # noqa: E402
    ROOT,
    RUNS,
    append_jsonl,
    balance,
    case_prompt,
    cost_usd,
    load_provider,
    load_suite,
    parse_answer,
    read_jsonl,
)

CACHE = ROOT / "cache" / "api"


def build_request(case: dict, arm: str, model: str, effort: str | None) -> dict:
    v = case["visible"]
    x = np.load(Path(case["dir"]) / v["file"])
    fs = v["channels"][0]["fs"]
    prompt = case_prompt(v, case["question"], case["qtype"])
    if arm == "A":
        content = [{"type": "text", "text": numeric_text(x, fs) + "\n\n" + prompt}]
    elif arm == "B":
        b64, note = strip_png(x, fs)
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": note + "\n\n" + prompt},
        ]
    else:
        raise ValueError(arm)
    req = {"model": model, "max_tokens": 64000, "messages": [{"role": "user", "content": content}]}
    if effort:
        req["output_config"] = {"effort": effort}
    return req


def call(client, req: dict) -> tuple[dict, bool]:
    key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:24]
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text()), True
    t = time.monotonic()
    with client.messages.stream(**req) as s:
        msg = s.get_final_message()
    out = {
        "served_model": msg.model,
        "stop_reason": msg.stop_reason,
        "text": "".join(b.text for b in msg.content if b.type == "text"),
        "usage": {
            k: getattr(msg.usage, k, None)
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        },
        "latency_s": round(time.monotonic() - t, 2),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out, False


def main() -> None:
    import anthropic

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument(
        "--provider", default="anthropic", help="anthropic (default endpoint) or a key of eval/providers.local.json"
    )
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="case ids")
    ap.add_argument("--run", default=None, help="run name (default: <suite>-<arm>-<model>-<effort>)")
    args = ap.parse_args()

    prov = load_provider(args.provider)
    model = prov["model"]
    effort = prov["effort"] if isinstance(prov["effort"], str) else (args.effort if prov["effort"] else None)
    bal0 = balance(prov)
    if bal0:
        print("balance before:", bal0, flush=True)
    if args.arm == "B" and not prov["vision"]:
        raise SystemExit(f"{args.provider} has no vision; arm B needs images")
    tag = model if args.provider == "anthropic" else f"{args.provider}_{model}"
    run = RUNS / (args.run or f"{args.suite}-{args.arm}-{tag}-{effort or 'default'}")
    done = {r["case_id"] for r in read_jsonl(run / "results.jsonl")}
    kw = {"base_url": prov["base_url"], "api_key": prov["api_key"]} if prov["base_url"] else {}
    client = anthropic.Anthropic(max_retries=4, **kw)
    cases = [c for c in load_suite(args.suite) if not args.only or c["case_id"] in args.only][: args.limit]
    spent = 0.0
    for c in cases:
        if c["case_id"] in done:
            continue
        req = build_request(c, args.arm, model, effort)
        try:
            out, cached = call(client, req)
        except Exception as e:  # infra failure: recorded, not scored
            append_jsonl(
                run / "errors.jsonl", {"case_id": c["case_id"], "class": type(e).__name__, "msg": str(e)[:500]}
            )
            print(c["case_id"], "ERROR", type(e).__name__, str(e)[:120], flush=True)
            continue
        if out["served_model"] != model:
            if args.provider == "anthropic":  # strict on our main endpoint; other providers report names loosely
                append_jsonl(
                    run / "errors.jsonl",
                    {"case_id": c["case_id"], "class": "served_model_mismatch", "msg": out["served_model"]},
                )
                print(c["case_id"], "MODEL MISMATCH", out["served_model"], flush=True)
                continue
            print(c["case_id"], "note: provider reports model", out["served_model"], flush=True)
        cost = cost_usd(model, out["usage"], prov.get("prices"))
        spent += 0 if cached else cost
        row = {
            "case_id": c["case_id"],
            "arm": args.arm,
            "qtype": c["qtype"],
            "model": model,
            "provider": args.provider,
            "served_model": out["served_model"],
            "effort": effort,
            "pred": parse_answer(out["text"]),
            "stop_reason": out["stop_reason"],
            "status": "truncated"
            if out["stop_reason"] == "max_tokens"
            else ("refusal" if out["stop_reason"] == "refusal" else "ok"),
            "usage": out["usage"],
            "cost_usd": round(cost, 5),
            "latency_s": out["latency_s"],
            "turns": 1,
            "text": out["text"][-3000:],
        }
        append_jsonl(run / "results.jsonl", row)
        print(c["case_id"], c["qtype"], row["pred"], f"${cost:.4f}", "(cached)" if cached else "", flush=True)
    print(f"new spend this invocation: ${spent:.3f}  -> {run}")
    if bal0:
        print("balance after:", balance(prov), "(before:", bal0 + ")", flush=True)


if __name__ == "__main__":
    main()
