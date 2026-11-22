"""Shared eval plumbing: case layout, prompts, answer parsing, pricing, JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = (
    ROOT / "cases"
)  # eval/cases/<suite>/<case_id>/{signal.npy,visible.json}; answers in eval/cases/<suite>.answers.jsonl
RUNS = ROOT / "runs"

MODEL = "claude-opus-5-5"
# $/MTok, official first-party prices (relay prices may differ; we report official for comparability)
PRICES = {
    "claude-opus-5-5": {"in": 4.0, "out": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-sonnet-5-5": {"in": 2.0, "out": 10.0, "cache_read": 0.20, "cache_write": 2.5},
}

ANSWER_FORMATS = {
    "change_time": '{"t": <seconds from record start, or null if nothing changes>}',
    "period": '{"period_s": <seconds>}',
    "artifact": '{"present": <true|false>, "t": <start time in seconds, or null>}',
    "spike_count": '{"count": <integer>}',
    "mean_hr": '{"bpm": <beats per minute>}',
    "af_burden": '{"percent": <percentage of recording time, 0-100>}',
    "longest_af": '{"t": <seconds from record start, or null if there is no atrial fibrillation or flutter>}',
    "episode_start": '{"t": <seconds from record start>}',
    "long_artifact": '{"t": <seconds from record start, or null if there is no such segment>}',
    "anomaly_time": '{"t": <start of the anomaly, in samples>}',
    "rhythm_change": '{"t": <seconds from record start of the first rhythm change, or null if none>}',
}


def case_prompt(visible: dict, question: str, qtype: str) -> str:
    chans = ", ".join(f"{c['name']} ({c['unit']}, {c['fs']} Hz)" for c in visible["channels"])
    return (
        f"You are given a recorded signal: duration {visible['dur_s']} s, channels: {chans}.\n"
        f"Question: {question}\n"
        "Answer as accurately as you can. End your reply with exactly one final line of the form\n"
        f"ANSWER: {ANSWER_FORMATS[qtype]}"
    )


_ANS = re.compile(r"ANSWER:\s*(\{.*\})\s*$", re.MULTILINE)


def parse_answer(text: str) -> dict | None:
    """Last `ANSWER: {...}` line; None if missing or not valid JSON (== no answer, not a wrong answer)."""
    m = _ANS.findall(text or "")
    if not m:
        return None
    try:
        v = json.loads(m[-1])
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def load_provider(name: str) -> dict:
    """'anthropic' = the default endpoint from the environment (Claude); others from eval/providers.local.json."""
    import os

    if name == "anthropic":
        return {"name": "anthropic", "model": MODEL, "vision": True, "effort": True, "base_url": None, "api_key": None}
    cfg_path = ROOT / "providers.local.json"
    if not cfg_path.exists():
        raise SystemExit("missing eval/providers.local.json (copy eval/providers.example.json and fill it in)")
    cfg = json.loads(cfg_path.read_text()).get(name)
    if not cfg:
        raise SystemExit(f"provider {name!r} not in eval/providers.local.json")
    key = os.environ.get(cfg.get("api_key_env", ""), "")
    if not key and cfg.get("api_key_file"):
        kf = Path(cfg["api_key_file"]).expanduser()
        key = kf.read_text().strip() if kf.exists() else ""
    if not key:
        raise SystemExit(f"no API key for {name}: set {cfg.get('api_key_env')} or api_key_file")
    eff = cfg.get("effort", False)  # False: omit; True: use --effort; "high": fixed level for this provider
    return {"name": name, "model": cfg["model"], "vision": bool(cfg.get("vision", False)), "effort": eff,
            "balance_url": cfg.get("balance_url"),
            "base_url": cfg["base_url"], "api_key": key,
            "prices": {"in": cfg.get("price_in", 0.0), "out": cfg.get("price_out", 0.0),
                       "cache_read": cfg.get("price_cache_read", cfg.get("price_in", 0.0)),
                       "cache_write": cfg.get("price_cache_write", cfg.get("price_in", 0.0))}}  # fmt: skip


def balance(prov: dict) -> str | None:
    """Account balance (for providers that expose it) — real spend = balance before - after a run."""
    if not prov.get("balance_url"):
        return None
    import urllib.request

    req = urllib.request.Request(prov["balance_url"], headers={"Authorization": f"Bearer {prov['api_key']}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            info = json.loads(r.read())["balance_infos"][0]
        return f"{info['total_balance']} {info['currency']}"
    except Exception as e:  # informational only
        return f"unavailable ({type(e).__name__})"


def cost_usd(model: str, usage: dict, prices: dict | None = None) -> float:
    p = prices or PRICES.get(model, PRICES[MODEL])
    return (
        (usage.get("input_tokens") or 0) * p["in"]
        + (usage.get("output_tokens") or 0) * p["out"]
        + (usage.get("cache_read_input_tokens") or 0) * p["cache_read"]
        + (usage.get("cache_creation_input_tokens") or 0) * p["cache_write"]
    ) / 1e6


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_jsonl(p: Path, row: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_suite(suite: str) -> list[dict]:
    """Cases with their public part only (question, visible meta, signal path). Answers are loaded by the grader."""
    rows = read_jsonl(CASES / f"{suite}.questions.jsonl")
    for r in rows:
        r["dir"] = str(CASES / suite / r["case_id"])
        r["visible"] = json.loads((Path(r["dir"]) / "visible.json").read_text())
    return rows
