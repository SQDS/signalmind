"""Deterministic sanitised traces for blinded grounding coding (docs/tasks/level2-analysis-plan.md §4).

  .venv/bin/python eval/level2_traces.py --tag l2 --out eval/results/p9level2/traces

Per session (arm, item) one markdown file `<session_key>.md` with: the question, the public channel layout and
duration, every assistant text block, every tool call with its arguments, and every tool result (error status kept,
images replaced by a placeholder naming the call, text truncated at 4,000 characters by default; set
--trace-text-limit 0 for complete text in a new export). Nothing from the manifest's
reference, tolerance, rubric or grade is read or written. `index.json` maps session keys to (arm, case_id) and is not
given to the coder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"
CASES = ROOT / "eval" / "cases"
LIMIT = 4000


def clip(text: str, text_limit: int = LIMIT) -> str:
    return (
        text
        if text_limit == 0 or len(text) <= text_limit
        else text[:text_limit] + f"\n[... truncated, {len(text) - text_limit} more characters]"
    )


def session_key(arm: str, case_id: str) -> str:
    return hashlib.sha256(f"level2-coding:{arm}:{case_id}".encode()).hexdigest()[:12]


def render(events: list[dict], question: str, visible: dict, text_limit: int = LIMIT) -> str:
    if text_limit < 0:
        raise ValueError("text_limit must be nonnegative (0 means unlimited)")
    chans = ", ".join(
        f"column {k} = {c['name']} ({c['unit']}, {c['fs']:g} Hz)" for k, c in enumerate(visible["channels"])
    )
    out = [f"# Question\n{question}\n", f"Signal: duration {visible['dur_s']} s; {chans}.\n", "# Trace\n"]
    calls: dict[str, str] = {}
    step = 0
    for e in events:
        if e.get("type") not in ("assistant", "user"):
            continue
        content = e.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            kind = c.get("type")
            if kind == "text" and c.get("text", "").strip():
                out.append(f"## [assistant text]\n{clip(c['text'], text_limit)}\n")
            elif kind == "tool_use":
                step += 1
                ref = f"s{step}"
                calls[c.get("id", ref)] = ref
                name = c.get("name", "?").removeprefix("mcp__tsmcp__")
                args = c.get("input", {})
                if name == "Bash" and isinstance(args.get("command"), str):
                    body = f"```\n{clip(args['command'], text_limit)}\n```"
                else:
                    body = "```json\n" + clip(json.dumps(args, ensure_ascii=False), text_limit) + "\n```"
                out.append(f"## [{ref}] tool call: {name}\n{body}\n")
            elif kind == "tool_result":
                ref = calls.get(c.get("tool_use_id"), "s?")
                parts = c.get("content")
                if isinstance(parts, str):
                    parts = [{"type": "text", "text": parts}]
                texts = []
                for p in parts or []:
                    if p.get("type") == "image":
                        texts.append(f"[image returned by {ref}]")
                    elif p.get("type") == "text":
                        texts.append(p.get("text", ""))
                status = "ERROR" if c.get("is_error") else "ok"
                out.append(f"## [{ref}] result ({status})\n```\n{clip(chr(10).join(texts), text_limit)}\n```\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="p9level2")
    ap.add_argument("--arm-glob", default="[CD]-*", help="run-dir arm segment, e.g. 'C-*-main-C_full'")
    ap.add_argument("--trace-text-limit", type=int, default=LIMIT, help="characters per text block; 0 = unlimited")
    args = ap.parse_args()
    if args.trace_text_limit < 0:
        ap.error("--trace-text-limit must be nonnegative")
    questions = {
        json.loads(x)["case_id"]: json.loads(x)
        for x in (CASES / f"{args.suite}.questions.jsonl").read_text().splitlines()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    index = {}
    for d in sorted(
        RUNS.glob(f"{args.suite}-{args.arm_glob}-w*" if args.suite != "p9level2" else f"p9level2-[CD]-*-{args.tag}-w*")
    ):
        arm = d.name.split("-")[1] if args.suite == "p9level2" else d.name.rsplit("-", 2)[-2]
        for t in sorted((d / "transcripts").glob("*.jsonl")):
            case_id = t.name.split(".")[0]
            events = [json.loads(x) for x in t.read_text(errors="replace").splitlines() if x.strip().startswith("{")]
            visible = json.loads((CASES / args.suite / case_id / "visible.json").read_text())
            key = session_key(arm, case_id)
            text = render(events, questions[case_id]["question"], visible, text_limit=args.trace_text_limit)
            if t.name.endswith(".partial.jsonl"):
                text += "\n# Session ended without a final result (timeout or interruption)\n"
            (args.out / f"{key}.md").write_text(text)
            index[key] = {"arm": arm, "case_id": case_id, "transcript": str(t.relative_to(ROOT))}
    (args.out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    print(len(index), "sessions")


if __name__ == "__main__":
    main()
