"""Version selection and unchanged model instructions; all sessions are local fakes."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import b_revisions as br  # noqa: E402
from eval import b_workflow as bw  # noqa: E402
from eval import run_claude_code as harness  # noqa: E402


@pytest.fixture
def case(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    (directory / "signal.npy").write_bytes(b"synthetic harness fixture; never loaded")
    return {
        "case_id": "synthetic",
        "qtype": "longest_af",
        "question": "When does the longest AF begin?",
        "dir": str(directory),
        "visible": {
            "file": "signal.npy",
            "dur_s": 1000,
            "channels": [{"name": "ch0", "unit": "mV", "fs": 250}],
        },
    }


@pytest.fixture
def provider():
    return {"model": "test-model", "vision": True, "effort": True, "base_url": None, "api_key": None}


def provenance(root):
    return {
        "server_root": str(root),
        "server_git_sha": bw.FROZEN_R,
        "server_package_path": str(root / "src/tsmcp/__init__.py"),
        "interface": "v1",
    }


@pytest.fixture
def workers(monkeypatch):
    calls = []

    def fake_worker(case, server_root, workflow="v0"):
        calls.append(workflow)
        bundle = {"status": "ready", "content": [bw.text_block(bw.canonical({"evidence_version": workflow}))]}
        meta = {
            "status": "ready",
            "identity": {"provenance": provenance(server_root), "workflow_sha256": br.workflow_hash(workflow)},
            "bundle_sha256": "a" * 64,
            "request_sha256": "b" * 64,
            "sizes": {"calls": 4, "images": 0, "text_chars": 100},
            "dropped": [],
        }
        return bundle, meta

    monkeypatch.setattr(bw, "build_in_worker", lambda case, root: fake_worker(case, root))
    monkeypatch.setattr(br, "build_in_worker", fake_worker)
    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")
    return calls


@pytest.mark.parametrize("qtype", ["longest_af", "af_burden"])
def test_prompt_instructions_and_model_configuration_are_byte_identical(case, provider, workers, qtype):
    case["qtype"] = qtype
    prompts, commands = [], []
    for selected in ("v0", "r1", "r2", "final"):
        plan = harness.prepare_case(case, "B", provider, "medium", True, b_workflow=selected)
        try:
            message = json.loads(plan["stdin"])["message"]["content"]
            prompts.append(message[0]["text"].encode())
            commands.append(plan["command"])
            # Everything following the original prompt comes directly from the bundle.
            assert message[1:] == [bw.text_block(bw.canonical({"evidence_version": selected}))]
            expected = br.effective_workflow(selected, qtype)
            for metadata in (plan["provenance"], plan["bundle_manifest"]):
                assert metadata["b_workflow"] == selected
                assert metadata["b_workflow_effective"] == expected
            assert plan["provenance"]["workflow_sha256"] == br.workflow_hash(selected)
        finally:
            shutil.rmtree(plan["base"])
    assert len(set(prompts)) == 1
    assert all(command == commands[0] for command in commands)
    assert workers == ["v0", "r1", "r2", "final"]


@pytest.mark.parametrize("selected", [None, "v0", "r1", "r2", "final"])
def test_cli_dry_run_records_selection_without_model_or_balance_calls(
    case, provider, workers, monkeypatch, capsys, selected
):
    monkeypatch.setattr(bw, "public_cases", lambda *args: [case])
    monkeypatch.setattr(harness, "load_provider", lambda name: provider)

    def forbidden(*args, **kwargs):
        pytest.fail("dry run attempted model execution or balance access")

    monkeypatch.setattr(harness, "run_isolated_session", forbidden)
    monkeypatch.setattr(harness, "balance", forbidden)
    args = ["run_claude_code.py", "--suite", "p7dev", "--arm", "B", "--dry-run"]
    if selected is not None:
        args += ["--b-workflow", selected]
    monkeypatch.setattr(sys, "argv", args)
    harness.main()
    printed = json.loads(capsys.readouterr().out)
    try:
        chosen = selected or "v0"
        meta = printed["bundle_manifest"]
        assert meta["b_workflow"] == chosen
        assert meta["b_workflow_effective"] == br.effective_workflow(chosen, case["qtype"])
        assert workers == [chosen]
        assert Path(printed["stdin_path"]).read_text() == printed["stdin"]
    finally:
        shutil.rmtree(Path(printed["cwd"]).parent)


@pytest.mark.parametrize("qtype", ["longest_af", "af_burden"])
@pytest.mark.parametrize("selected", ["v0", "r1", "r2", "final"])
def test_cli_result_rows_record_selected_and_effective_versions(
    case, provider, workers, tmp_path, monkeypatch, qtype, selected
):
    case["qtype"] = qtype
    monkeypatch.setattr(bw, "public_cases", lambda *args: [case])
    monkeypatch.setattr(harness, "load_provider", lambda name: provider)
    monkeypatch.setattr(harness, "check_server", provenance)
    monkeypatch.setattr(harness, "balance", lambda prov: None)
    monkeypatch.setattr(harness, "RUNS", tmp_path / "runs")

    def fake_session(*args, **kwargs):
        return (
            json.dumps(
                {"type": "result", "result": 'ANSWER: {"t": 100}', "num_turns": 1, "usage": {}, "is_error": False}
            ),
            "",
            0,
        )

    monkeypatch.setattr(harness, "run_isolated_session", fake_session)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_claude_code.py",
            "--suite",
            "p7dev",
            "--arm",
            "B",
            "--b-workflow",
            selected,
            "--run",
            "test",
        ],
    )
    harness.main()
    row = json.loads((tmp_path / "runs/test/results.jsonl").read_text())
    assert row["b_workflow"] == selected
    assert row["b_workflow_effective"] == br.effective_workflow(selected, qtype)
    assert row["workflow_sha256"] == br.workflow_hash(selected)
    # A valid resume skips the already completed row without rebuilding evidence.
    harness.main()
    assert workers == [selected]


def test_resume_rejects_different_selected_or_effective_workflow(case, provider, tmp_path, monkeypatch):
    monkeypatch.setattr(bw, "public_cases", lambda *args: [case])
    monkeypatch.setattr(harness, "load_provider", lambda name: provider)
    monkeypatch.setattr(harness, "check_server", provenance)
    monkeypatch.setattr(harness, "RUNS", tmp_path)
    root = bw.SERVER_ROOT
    row = {
        **provenance(root),
        "case_id": case["case_id"],
        "qtype": case["qtype"],
        "workflow_sha256": br.workflow_hash("final"),
        "b_workflow": "final",
        "b_workflow_effective": "r2",
    }
    monkeypatch.setattr(harness, "read_jsonl", lambda path: [row])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_claude_code.py",
            "--suite",
            "p7dev",
            "--arm",
            "B",
            "--b-workflow",
            "final",
        ],
    )
    with pytest.raises(SystemExit, match="B workflow provenance"):
        harness.main()
    row["b_workflow"] = "r1"
    with pytest.raises(SystemExit, match="server provenance"):
        harness.main()


@pytest.mark.parametrize("failure", ["api_error", "served_model_mismatch"])
def test_b_error_rows_keep_version_metadata(case, provider, workers, tmp_path, monkeypatch, failure):
    monkeypatch.setattr(bw, "public_cases", lambda *args: [case])
    monkeypatch.setattr(harness, "load_provider", lambda name: provider)
    monkeypatch.setattr(harness, "check_server", provenance)
    monkeypatch.setattr(harness, "balance", lambda prov: None)
    monkeypatch.setattr(harness, "RUNS", tmp_path / "runs")
    events = [
        {
            "type": "assistant",
            "message": {
                "model": "other-model" if failure == "served_model_mismatch" else provider["model"],
                "content": [],
            },
        },
        {"type": "result", "result": "API Error: simulated", "num_turns": 1, "usage": {}, "is_error": True},
    ]
    monkeypatch.setattr(
        harness, "run_isolated_session", lambda *args, **kwargs: ("\n".join(map(json.dumps, events)), "", 0)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_claude_code.py",
            "--suite",
            "p7dev",
            "--arm",
            "B",
            "--b-workflow",
            "final",
            "--run",
            "test",
        ],
    )
    harness.main()
    row = json.loads((tmp_path / "runs/test/errors.jsonl").read_text())
    assert row["class"] == failure
    assert row["b_workflow"] == "final"
    assert row["b_workflow_effective"] == "r1"


def test_b_version_option_has_no_effect_on_arm_c(case, provider, workers, tmp_path, monkeypatch):
    base = tmp_path / "same-work-path"

    def same_directory(**kwargs):
        base.mkdir()
        return str(base)

    monkeypatch.setattr(harness.tempfile, "mkdtemp", same_directory)
    plans = []
    for selected in ("v0", "r1", "r2", "final"):
        plan = harness.prepare_case(case, "C", provider, "medium", True, b_workflow=selected)
        plans.append({key: plan[key] for key in ("command", "stdin", "env", "mcp_config", "provenance")})
        shutil.rmtree(base)
    assert all(plan == plans[0] for plan in plans)
    assert workers == []
