"""Single entry for the coding Agent, including one automatic context-preserving token-budget continuation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.coding_peak_agent import live  # noqa: E402
from eval.dataset_workflow import read, write  # noqa: E402
from eval.resume_coding_peak_agent import resume  # noqa: E402


async def execute(args):
    result = await live(args)
    if (
        not result["workflow_complete"]
        and result["stop"] == "tool_or_token_budget"
        and result["tool_calls"] < 48
        and result["turns"] < 48
        and result["estimated_peak_usd"] + 0.05 < args.cap
    ):
        await resume(args.out, args.cap)
        result = read(args.out / "session-final.json")
    segments = [read(args.out / "session.json")] + [
        read(p) for p in sorted((args.out / "continuations").glob("segment_*/session.json"))
    ]
    summary = {
        **result,
        "active_execution_total_s": sum(s["latency_s"] for s in segments),
        "execution_segments": len(segments),
        "single_entry_workflow": True,
    }
    write(args.out / "workflow-result.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--cap", type=float, default=0.5)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live or args.cap <= 0.05:
        raise ValueError("Use --live and a budget above the next-request reserve")
    result = asyncio.run(execute(args))
    print(json.dumps({k: v for k, v in result.items() if k not in {"usage", "delivery"}}, ensure_ascii=False, indent=2))
