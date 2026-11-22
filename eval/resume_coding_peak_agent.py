"""Resume the same DS coding conversation/state after a controller token allowance pause."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.autonomous_peak_agent import run_agent  # noqa: E402
from eval.coding_peak_agent import SYSTEM, CodingPeakTools  # noqa: E402
from eval.dataset_workflow import read, sha, write  # noqa: E402
from eval.unified.drivers import make_driver  # noqa: E402
from eval.unified.loop import Limits  # noqa: E402


def restore_tools(data, output, calls):
    tools = CodingPeakTools(data, output)
    for path in sorted((output / "versions").glob("*/version.json")):
        meta = read(path)
        vid = meta["version_id"]
        predictions = read(path.parent / "predictions.json")
        assert sha(path.parent / "predictions.json") == meta["predictions_sha256"]
        tools.versions[vid] = {**meta, "predictions": predictions}
        tools.viewer.trials[vid] = {"policy": meta["policy"], "results": predictions}
        audit_path = path.parent / "audit.json"
        if audit_path.exists():
            saved = read(audit_path)
            rows = saved.pop("all_cases")
            groups = {
                "conflicting_deletions": [r["case_id"] for r in rows if r["conflicting_deletions"]],
                "high_deletion": [r["case_id"] for r in rows if r["deleted"] > 0.1 * max(r["old_markers"], 1)],
                "review_queue": [r["case_id"] for r in rows if r["review_items"]],
                "nearby_candidates": [r["case_id"] for r in rows if r["nearby_pairs_under300ms"]],
            }
            tools.audits[vid] = {"report": saved, "groups": groups}
        feedback = path.parent / "development_feedback.json"
        if feedback.exists():
            tools.feedbacks[vid] = read(feedback)
    for path in (output / "programs").glob("*/metadata.json"):
        meta = read(path)
        assert sha(path.parent / "program.py") == meta["source_sha256"]
        tools.programs[meta["program_id"]] = path.parent
    tools.inspections = read(output / "inspections.json") if (output / "inspections.json").exists() else []
    tools.program_attempts = max(
        [int(p.name.split("_")[-1]) for p in (output / "program_runs").glob("attempt_*")], default=0
    )
    tools.calls = calls
    return tools


def recover_messages(transcript, goal, tools):
    """Recover exact tool images from saved inputs/versions; hashes must match the original API log."""
    messages = [{"role": "user", "content": goal}]
    calls, image_count = {}, 0
    for event in map(json.loads, transcript.read_text().splitlines()):
        if event["type"] == "assistant":
            for block in event["content"]:
                if block["type"] == "tool_use":
                    calls[block["id"]] = block
            messages.append({"role": "assistant", "content": event["content"]})
        elif event["type"] == "tool_results":
            content = copy.deepcopy(event["content"])
            for result in content:
                for i, block in enumerate(result["content"]):
                    if block["type"] == "image":
                        call = calls[result["tool_use_id"]]
                        assert call["name"] == "dataset_inspect"
                        args = call["input"]
                        rendered = tools.viewer.inspect(
                            args["case_id"], args.get("start_s", 0), args.get("end_s", 10), args["version_id"]
                        )
                        picture = next(b for b in rendered if b["type"] == "image")
                        encoded = picture["source"]["data"]
                        assert hashlib.sha256(encoded.encode()).hexdigest()[:16] == block["sha256"]
                        assert len(encoded) == block["bytes"]
                        result["content"][i] = picture
                        image_count += 1
            messages.append({"role": "user", "content": content})
        elif event["type"] == "controller_feedback":
            messages.append({"role": "user", "content": event["content"]})
    return messages, image_count


class HistoryDriver:
    def __init__(self, driver, history):
        self.driver, self.history = driver, history

    def step(self, system, messages, tools):
        return self.driver.step(SYSTEM, [*self.history, *messages], tools)


async def resume(output, total_cap):
    original = read(output / "session.json")
    assert not original["workflow_complete"] and not (output / "delivery.json").exists()
    provenance = read(output / "provenance.json")
    for name, expected in provenance["source_sha256"].items():
        assert sha(ROOT / name) == expected, "Original numerical/tool implementation changed"
    existing = sorted((output / "continuations").glob("segment_*/session.json"))
    previous = [original] + [read(p) for p in existing]
    spent = sum(s["estimated_peak_usd"] for s in previous)
    total_calls = previous[-1]["tool_calls"]
    total_turns = sum(s["turns"] for s in previous)
    if spent + 0.05 >= total_cap or total_calls >= 48 or total_turns >= 48:
        raise ValueError("No remaining authorized cost/call/turn allocation")
    tools = restore_tools(Path(provenance["data"]), output, total_calls)
    history, pictures = recover_messages(output / "private/transcript.jsonl", provenance["goal"], tools)
    for path in existing:
        meta = read(path.parent / "resume.json")
        extra, n = recover_messages(path.parent / "private/transcript.jsonl", meta["continuation_goal"], tools)
        history.extend(extra)
        pictures += n
    goal = (
        "Continue the SAME dataset task after a controller output-token allowance pause. Numerical programs and "
        "previous outputs have not been edited. Your saved versions are "
        + ", ".join(tools.versions)
        + "; your saved program IDs are "
        + ", ".join(sorted(tools.programs))
        + f". You have {48 - total_calls} remaining tool calls and an additional60000 output-token allowance within "
        "the ORIGINAL total API cost cap. Per-response allowance is now16384 tokens to support code arguments. "
        "Continue your own testing/revision and full execution, then audit/inspect, read development feedback "
        "and finalize the version you choose. Evaluation reference labels remain unavailable."
    )
    segment = output / "continuations" / f"segment_{len(existing) + 1:03d}"
    segment.mkdir(parents=True, exist_ok=False)
    write(
        segment / "resume.json",
        {
            "date": "2026-11-07",
            "reason": "Original controller reached60000 cumulative output tokens before delivery",
            "prior_api_estimate_usd": spent,
            "original_total_cap_usd": total_cap,
            "remaining_cap_usd": total_cap - spent,
            "restored_versions": list(tools.versions),
            "restored_programs": sorted(tools.programs),
            "reconstructed_images_hash_matched": pictures,
            "history_message_count": len(history),
            "history_sha256": hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest(),
            "original_transcript_sha256": sha(output / "private/transcript.jsonl"),
            "continuation_source_sha256": sha(Path(__file__)),
            "max_response_tokens": 16384,
            "additional_output_token_allowance": 60000,
            "continuation_goal": goal,
            "algorithms_or_evaluation_labels_changed": False,
        },
    )
    driver = make_driver("deepseek-flash", max_tokens=16384)
    try:
        result = await run_agent(
            HistoryDriver(driver, history),
            tools,
            goal,
            segment,
            total_cap - spent,
            Limits(max_turns=48 - total_turns, max_tool_calls=48, max_seconds=1500, max_output_tokens_total=60000),
        )
    finally:
        driver.close()
    combined = {
        **result,
        "turns": total_turns + result["turns"],
        "estimated_peak_usd": spent + result["estimated_peak_usd"],
        "controller_continuations": sum(p["controller_continuations"] for p in previous)
        + result["controller_continuations"],
        "resume_segments": len(existing) + 1,
        "original_budget_preserved": total_cap,
        "initial_incomplete_attempt_preserved": True,
    }
    write(output / "session-final.json", combined)
    print(
        json.dumps({k: v for k, v in combined.items() if k not in {"usage", "delivery"}}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-cap", type=float, default=0.5)
    args = parser.parse_args()
    asyncio.run(resume(args.out, args.total_cap))
