"""Generated programs see development labels only and execute through the Agent's explicit tools."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.coding_peak_agent import CodingPeakTools  # noqa: E402
from eval.dataset_workflow import read, sha, write  # noqa: E402
from eval.peak_program_worker import normalize  # noqa: E402
from eval.real_processing_helper import save_npz  # noqa: E402
from tsmcp.peak_policy import SCHEMA  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    t = np.arange(2500)
    centers = np.arange(150, 2400, 125)
    wave = sum(np.exp(-0.5 * ((t - p) / 2) ** 2) for p in centers)
    cases = []
    for uid in ("development_case", "evaluation_case"):
        folder = public / uid
        folder.mkdir()
        save_npz(folder / "input.npz", ecg=wave, abp=90 + np.sin(t / 125), old_peaks=centers + 3)
        cases.append({"id": uid, "sha256": sha(folder / "input.npz")})
    write(
        public / "index.json",
        {
            "cases": cases,
            "development_examples": ["development_case"],
            "development_case_ids": ["development_case"],
            "evaluation_case_ids": ["evaluation_case"],
            "units": {"ecg": "source_units", "abp": "mmHg"},
        },
    )
    reference = {
        "id": "development_case",
        "subject_key": "dev_record",
        "disposition": "CORRECTED",
        "exclusion_subtype": "",
        "target_peak_samples": centers.tolist(),
    }
    write(public / "development_labels.json", {"development_case": reference})
    (tmp_path / "private").mkdir()
    write(tmp_path / "private/truth.json", {"evaluation_case": "EVALUATION_SECRET_SENTINEL"})
    return public, tmp_path / "output", tmp_path / "private/truth.json"


def call(tools, name, args):
    blocks, failed = asyncio.run(tools.call(name, args))
    return json.loads(blocks[0]["text"]) if not failed and name != "dataset_inspect" else blocks, failed


def start(tools):
    result, failed = call(
        tools, "dataset_run_batch", {"policy": {"schema_version": SCHEMA, "alignment": "none"}, "rationale": "fixture"}
    )
    assert not failed and result["version_id"] == "v001"
    call(tools, "dataset_audit", {"version_id": "v001"})
    return call(tools, "dataset_development_feedback", {"version_id": "v001"})[0]


def test_review_items_cannot_be_hidden_by_a_false_pending_flag():
    result = normalize(
        {"new_peak_samples": [10], "exclude": False, "review_queue": ["uncertain"], "review_required": False},
        {"features": {}, "old_peak_samples": [10]},
        np.array([10]),
        100,
        125,
    )
    assert result["review_required"]


def test_feedback_contains_only_development_targets(workspace):
    public, out, _ = workspace
    tools = CodingPeakTools(public, out)
    result = start(tools)
    assert result["metrics"]["counts"]["windows"] == 1
    assert "evaluation_case" not in json.dumps(result)
    assert "EVALUATION_SECRET_SENTINEL" not in json.dumps(result)
    assert all(r["case_id"] == "development_case" for r in result["error_examples"])


def test_model_program_executes_in_sandbox_with_only_development_references(workspace):
    public, out, secret = workspace
    tools = CodingPeakTools(public, out)
    start(tools)
    source = f"""from pathlib import Path
import json
import numpy as np
from scipy.signal import find_peaks
def fit(examples, fs):
    assert len(examples) == 1
    assert examples[0]['reference']['id'] == 'development_case'
    assert set(json.loads(Path('/case/development_labels.json').read_text())) == {{'development_case'}}
    assert not Path({str(secret)!r}).exists()
    assert not Path('/case/../private/truth.json').exists()
    assert 'eth0' not in Path('/proc/net/dev').read_text()
    return None
def process(ecg, abp, old_peaks, fs, previous, state):
    new = [max(0,int(p)-6)+int(np.argmax(ecg[max(0,int(p)-6):min(len(ecg),int(p)+7)])) for p in old_peaks]
    return {{'new_peak_samples':new, 'exclude':False}}
"""
    program, failed = call(
        tools, "dataset_write_program", {"source": source, "description": "test fixture, not a model run"}
    )
    assert not failed
    result, failed = call(
        tools,
        "dataset_run_program",
        {"parent_version": "v001", "program_id": program["program_id"], "scope": "all", "rationale": "fixture"},
    )
    assert not failed, result
    assert result["version_id"] == "v002" and result["execution"]["cases_executed"] == 2
    assert (out / "programs" / program["program_id"] / "program.py").read_text() == source
    feedback, failed = call(tools, "dataset_development_feedback", {"version_id": "v002"})
    assert not failed and feedback["metrics"]["counts"]["retained_windows_end_to_end_8ms"] == 1
    assert read(out / "program_runs/attempt_001/execution.json")["returncode"] == 0


def test_malformed_program_outputs_are_errors_not_host_repaired_results(workspace):
    public, out, _ = workspace
    tools = CodingPeakTools(public, out)
    start(tools)
    p, _ = call(
        tools,
        "dataset_write_program",
        {
            "source": "def process(ecg, abp, old_peaks, fs, previous, state):\n    return {'new_peak_samples':[], 'exclude':False}\n",
            "description": "bad fixture",
        },
    )
    before = sha(out / "versions/v001/predictions.json")
    blocks, failed = call(
        tools,
        "dataset_run_program",
        {"parent_version": "v001", "program_id": p["program_id"], "scope": "development", "rationale": "fixture"},
    )
    assert failed and "program_failed" in blocks[0]["text"]
    assert len(tools.versions) == 1
    assert sha(out / "versions/v001/predictions.json") == before
