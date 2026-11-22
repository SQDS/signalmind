"""The model must request every workflow stage; no host-side repair substitutes for autonomy."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.autonomous_peak_agent import AutonomousPeakTools, run_agent  # noqa: E402
from eval.dataset_workflow import read, sha, write  # noqa: E402
from eval.real_processing_helper import save_npz  # noqa: E402
from eval.unified.drivers import Step  # noqa: E402
from tsmcp.peak_policy import SCHEMA  # noqa: E402

POLICY = {"schema_version": SCHEMA, "alignment": "positive", "min_template_correlation": 0, "refractory_ms": 200}


@pytest.fixture
def workspace(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    t = np.arange(2500)
    centers = np.arange(150, 2400, 125)
    x = sum(np.exp(-0.5 * ((t - p) / 2) ** 2) - 1.3 * np.exp(-0.5 * ((t - p - 8) / 2) ** 2) for p in centers)
    old = centers.copy()
    old[::3] += 8
    cases = []
    for uid in ("c1", "c2"):
        folder = public / uid
        folder.mkdir()
        save_npz(folder / "input.npz", ecg=x, abp=90 + 5 * np.sin(2 * np.pi * t / 125), old_peaks=old)
        cases.append({"id": uid, "sha256": sha(folder / "input.npz")})
    write(
        public / "index.json",
        {"cases": cases, "development_examples": ["c1"], "units": {"ecg": "source_units", "abp": "mmHg"}},
    )
    return public, tmp_path / "agent"


def invoke(tools, name, args):
    blocks, error = asyncio.run(tools.call(name, args))
    return (json.loads(blocks[0]["text"]) if not error and name != "dataset_inspect" else blocks), error


def test_no_hidden_refinement_and_completion_requires_model_review(workspace):
    public, out = workspace
    tools = AutonomousPeakTools(public, out)
    first, error = invoke(tools, "dataset_run_batch", {"policy": POLICY, "rationale": "fixture"})
    assert not error and first["version_id"] == "v001"
    assert len(tools.versions) == 1
    _, error = invoke(
        tools,
        "dataset_run_refinement",
        {
            "parent_version": "v001",
            "program": "peak_refinement_v1",
            "scope": "all",
            "parameters": {},
            "rationale": "fixture",
        },
    )
    assert error and len(tools.versions) == 1
    audit, error = invoke(tools, "dataset_audit", {"version_id": "v001"})
    assert not error and audit["totals"]["conflicting_deletions"] > 0
    assert len(tools.versions) == 1  # Even when audit finds issues, the host does not run refinement.
    _, error = invoke(tools, "dataset_finalize", {"version_id": "v001", "rationale": "premature", "limitations": []})
    assert error and not tools.completed
    before = sha(out / "versions/v001/predictions.json")
    second, error = invoke(
        tools,
        "dataset_run_refinement",
        {
            "parent_version": "v001",
            "program": "peak_refinement_v1",
            "scope": "conflicting_deletions",
            "parameters": {},
            "rationale": "fixture",
        },
    )
    assert not error and second["parent"] == "v001"
    assert sha(out / "versions/v001/predictions.json") == before
    invoke(tools, "dataset_audit", {"version_id": "v002"})
    _, error = invoke(
        tools, "dataset_finalize", {"version_id": "v002", "rationale": "not inspected", "limitations": []}
    )
    assert error
    _, error = invoke(tools, "dataset_inspect", {"version_id": "v002", "case_id": "c2", "start_s": 0, "end_s": 5})
    assert not error
    final, error = invoke(tools, "dataset_finalize", {"version_id": "v002", "rationale": "checked", "limitations": []})
    assert not error and tools.completed and final["decision_owner"] == "model"
    assert read(out / "delivery.json")["selected_refined_result"]


def test_tools_cannot_inspect_private_reference_paths(workspace):
    public, out = workspace
    (public.parent / "private").mkdir()
    write(public.parent / "private/truth.json", {"secret": "GOLDEN_REFERENCE_SENTINEL"})
    tools = AutonomousPeakTools(public, out)
    overview, error = invoke(tools, "dataset_overview", {})
    assert not error and "GOLDEN_REFERENCE_SENTINEL" not in json.dumps(overview)
    invoke(tools, "dataset_run_batch", {"policy": POLICY, "rationale": "fixture"})
    _, error = invoke(tools, "dataset_inspect", {"version_id": "v001", "case_id": "../private/truth.json"})
    assert error


class ScriptedDriver:
    def __init__(self, tools):
        self.tools = tools
        self.index = 0
        self.script = [
            None,
            ("dataset_overview", {}),
            ("dataset_run_batch", {"policy": POLICY, "rationale": "scripted model choice"}),
            ("dataset_audit", {"version_id": "v001"}),
            ("dataset_inspect", {"version_id": "v001", "case_id": "c1", "start_s": 0, "end_s": 5}),
            (
                "dataset_run_refinement",
                {
                    "parent_version": "v001",
                    "program": "peak_refinement_v1",
                    "scope": "all",
                    "parameters": {},
                    "rationale": "scripted choice after audit",
                },
            ),
            ("dataset_audit", {"version_id": "v002"}),
            ("dataset_inspect", {"version_id": "v002", "case_id": "c1", "start_s": 0, "end_s": 5}),
            (
                "dataset_finalize",
                {"version_id": "v002", "rationale": "scripted review complete", "limitations": ["fixture"]},
            ),
        ]

    def step(self, system, messages, schemas):
        current = self.script[self.index]
        self.index += 1
        if current is None:
            return Step([{"type": "text", "text": "Done"}], {}, "end_turn", "scripted_test")
        if self.index == 2:
            assert not self.tools.versions
            assert "not delivered" in messages[-1]["content"]
        name, args = current
        return Step(
            [{"type": "tool_use", "id": f"call-{self.index}", "name": name, "input": args}],
            {"input_tokens": 10, "output_tokens": 5},
            "tool_use",
            "scripted_test",
            f"response-{self.index}",
        )


def test_live_loop_routes_only_actual_model_calls_and_continues_incomplete_task(workspace):
    public, out = workspace
    tools = AutonomousPeakTools(public, out)
    result = asyncio.run(run_agent(ScriptedDriver(tools), tools, "fixture goal", out))
    assert result["workflow_complete"] and result["stop"] == "workflow_complete"
    assert result["controller_continuations"] == 1
    assert result["host_injected_processing_steps"] == 0
    events = [json.loads(line) for line in (out / "activity.jsonl").read_text().splitlines()]
    assert len(events) == 8 and all(e["origin"] == "model_tool_call" for e in events)
    assert [e["tool"] for e in events] == [entry[0] for entry in ScriptedDriver(tools).script[1:]]
    assert all(e["request_id"].startswith("response-") for e in events)
