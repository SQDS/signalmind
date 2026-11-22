"""Controller recovery preserves total budget and does not retry model/program errors."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import run_coding_workflow as workflow  # noqa: E402
from eval.dataset_workflow import write  # noqa: E402


def test_output_allowance_pause_continues_without_restarting_development(tmp_path, monkeypatch):
    actions = []
    args = argparse.Namespace(out=tmp_path, cap=0.5)

    async def first(args):
        actions.append("initial")
        result = {
            "workflow_complete": False,
            "stop": "tool_or_token_budget",
            "tool_calls": 23,
            "turns": 15,
            "estimated_peak_usd": 0.08,
            "latency_s": 400,
        }
        write(args.out / "session.json", result)
        return result

    async def continuation(out, cap):
        actions.append("resume")
        assert cap == 0.5
        result = {
            "workflow_complete": True,
            "stop": "workflow_complete",
            "tool_calls": 39,
            "turns": 29,
            "estimated_peak_usd": 0.12,
            "latency_s": 140,
        }
        folder = out / "continuations/segment_001"
        folder.mkdir(parents=True)
        write(folder / "session.json", {**result, "estimated_peak_usd": 0.04})
        write(out / "session-final.json", result)

    monkeypatch.setattr(workflow, "live", first)
    monkeypatch.setattr(workflow, "resume", continuation)
    result = asyncio.run(workflow.execute(args))
    assert actions == ["initial", "resume"]
    assert result["workflow_complete"] and result["active_execution_total_s"] == 540
    assert result["estimated_peak_usd"] == 0.12


def test_model_error_is_not_automatically_retried(tmp_path, monkeypatch):
    args = argparse.Namespace(out=tmp_path, cap=0.5)

    async def first(args):
        result = {
            "workflow_complete": False,
            "stop": "error",
            "tool_calls": 1,
            "turns": 1,
            "estimated_peak_usd": 0.01,
            "latency_s": 5,
        }
        write(args.out / "session.json", result)
        return result

    async def unexpected(*args):
        raise AssertionError("Do not silently retry errors")

    monkeypatch.setattr(workflow, "live", first)
    monkeypatch.setattr(workflow, "resume", unexpected)
    result = asyncio.run(workflow.execute(args))
    assert not result["workflow_complete"] and result["execution_segments"] == 1
