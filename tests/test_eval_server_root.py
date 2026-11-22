"""Harness isolation/provenance tests: fake entrypoints only, never a model or detector."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import run_claude_code as harness  # noqa: E402


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "case"
    source.mkdir()
    (source / "signal.npy").write_bytes(b"not opened by the harness")
    return {
        "case_id": "test-case",
        "dir": str(source),
        "qtype": "mean_hr",
        "question": "Mean heart rate?",
        "visible": {"file": "signal.npy", "dur_s": 10, "channels": [{"name": "ECG", "unit": "mV", "fs": 250}]},
    }


@pytest.fixture
def provider():
    return {"model": "test-model", "vision": True, "effort": True, "base_url": None, "api_key": None}


@pytest.fixture
def fake_server(tmp_path):
    root = tmp_path / "server checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)  # fmt: skip
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (venv / "pyvenv.cfg").write_text(
        f"home = {Path(sys._base_executable).parent}\ninclude-system-site-packages = false\n"
    )
    package = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "tsmcp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (venv / "bin" / "tsmcp").write_text(
        "import json, os, sys, tsmcp\n"
        "print(json.dumps({'package': tsmcp.__file__, 'args': sys.argv[1:], 'pid': os.getpid()}))\n"
    )
    return root


def test_selected_server_process_reports_real_import_and_executes_entrypoint(fake_server, tmp_path):
    report = tmp_path / "report.json"
    p = subprocess.run(
        harness.server_command(fake_server) + ["--provenance", str(report), "--", "--allow-root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    entrypoint = json.loads(p.stdout)
    info = json.loads(report.read_text())
    assert info["server_root"] == str(fake_server)
    assert info["server_package_path"] == entrypoint["package"]
    assert (
        info["server_git_sha"]
        == subprocess.check_output(["git", "-C", str(fake_server), "rev-parse", "HEAD"], text=True).strip()
    )
    assert entrypoint["args"] == ["--allow-root", str(tmp_path)]
    assert harness.check_server(fake_server) == info


def test_server_rejects_package_symlink_outside_checkout(fake_server, tmp_path):
    package = next((fake_server / ".venv" / "lib").glob("python*/site-packages/tsmcp"))
    outside = tmp_path / "other-tsmcp"
    package.rename(outside)
    package.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SystemExit, match="outside server root"):
        harness.check_server(fake_server)


def test_config_uses_selected_venv_and_disables_images(case, provider, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")
    provider["vision"] = False
    root = tmp_path / "alternate checkout"
    plan = harness.prepare_case(case, "C", provider, "medium", True, root)
    try:
        cfg = plan["mcp_config"]["mcpServers"]["tsmcp"]
        assert cfg["command"] == str(root / ".venv" / "bin" / "python")
        assert cfg["args"][:4] == ["-I", str(harness.SERVER_LAUNCHER), "--server-root", str(root)]
        assert cfg["args"][-3:] == ["--allow-root", str(plan["cwd"]), "--no-images"]
        assert plan["command"][-4:] == ["--tools", "", "--allowedTools", harness.ALLOW_C]
        assert plan["stdin"].endswith(harness.NO_VISION_NOTE)
    finally:
        harness.shutil.rmtree(plan["base"])


def test_dry_run_one_case_no_process_or_balance(case, provider, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run must not launch a process, model or balance request")

    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")
    monkeypatch.setattr(harness, "load_suite", lambda suite: [case, {**case, "case_id": "second"}])
    monkeypatch.setattr(harness, "load_provider", lambda name: provider)
    monkeypatch.setattr(harness, "balance", forbidden)
    monkeypatch.setattr(harness.subprocess, "run", forbidden)
    monkeypatch.setattr(harness, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_claude_code.py",
            "--suite",
            "test",
            "--arm",
            "C",
            "--dry-run",
            "--server-root",
            str(tmp_path / "R"),
            "--run",
            "named",
            "--label",
            "R",
        ],
    )
    harness.main()
    out = json.loads(capsys.readouterr().out)
    try:
        assert out["dry_run"] and out["case_id"] == "test-case"
        assert out["run_directory"] == str(tmp_path / "runs" / "named-R")
        assert out["mcp_config"]["mcpServers"]["tsmcp"]["command"] == str(tmp_path / "R/.venv/bin/python")
        cfg_index = out["command"].index("--mcp-config") + 1
        assert json.loads(Path(out["command"][cfg_index]).read_text()) == out["mcp_config"]
        assert Path(out["stdin_path"]).read_text() == out["stdin"]
        assert not (tmp_path / "runs").exists()
    finally:
        harness.shutil.rmtree(Path(out["cwd"]).parent)


@pytest.mark.parametrize("mode", ["good", "missing", "outside", "changed"])
def test_result_requires_actual_server_report(case, provider, tmp_path, monkeypatch, mode):
    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")
    root = tmp_path / "R"
    info = {
        "server_root": str(root),
        "server_git_sha": "a" * 40,
        "server_package_path": str(root / "src/tsmcp/__init__.py"),
    }
    work = None

    def fake_claude(cmd, cwd, env, stdin, timeout_s, stream_to=None):
        nonlocal work
        work = cwd
        config = json.loads(Path(cmd[cmd.index("--mcp-config") + 1]).read_text())["mcpServers"]["tsmcp"]
        report = Path(config["args"][config["args"].index("--provenance") + 1])
        observed = dict(info)
        if mode == "outside":
            observed["server_package_path"] = str(tmp_path / "R-other/tsmcp/__init__.py")
        if mode == "changed":
            observed["server_git_sha"] = "b" * 40
        if mode != "missing":
            report.write_text(json.dumps(observed))
        event = {"type": "result", "result": 'ANSWER: {"bpm": 70}', "usage": {}, "num_turns": 1}
        return json.dumps(event), "", 0

    monkeypatch.setattr(harness, "run_isolated_session", fake_claude)
    out = harness.run_case(
        case, "C", provider, "medium", tmp_path / "run", True, server_root=root, expected_server=info
    )
    assert not work.exists()
    if mode == "good":
        assert {k: out[k] for k in info} == info
    else:
        assert out["error"] == "server_provenance"


def test_label_suffix_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "RUNS", tmp_path)
    assert harness.run_directory("run", "M") != harness.run_directory("run", "R")
    assert harness.run_directory("run", None) == tmp_path / "run"
    with pytest.raises(SystemExit, match="--label"):
        harness.run_directory("run", "../R")


def test_main_persists_provenance_and_rejects_mixed_resume(case, provider, tmp_path, monkeypatch):
    root = tmp_path / "R"
    info = {
        "server_root": str(root),
        "server_git_sha": "a" * 40,
        "server_package_path": str(root / "src/tsmcp/__init__.py"),
    }
    result = {
        **info,
        "served_models": [],
        "text": 'ANSWER: {"bpm": 70}',
        "is_error": False,
        "contamination_flags": [],
        "usage": {},
        "cc_cost_usd": 0,
        "wall_s": 0,
        "turns": 1,
        "tool_calls": [],
    }
    monkeypatch.setattr(harness, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(harness, "load_suite", lambda suite: [case])
    monkeypatch.setattr(harness, "load_provider", lambda name: provider)
    monkeypatch.setattr(harness, "check_server", lambda selected: dict(info))
    monkeypatch.setattr(harness, "balance", lambda prov: None)
    monkeypatch.setattr(harness, "run_case", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_claude_code.py",
            "--suite",
            "test",
            "--arm",
            "C",
            "--server-root",
            str(root),
            "--run",
            "named",
            "--label",
            "R",
        ],
    )
    harness.main()
    rows = harness.read_jsonl(tmp_path / "runs/named-R/results.jsonl")
    assert len(rows) == 1 and {k: rows[0][k] for k in info} == info
    info["server_git_sha"] = "b" * 40
    with pytest.raises(SystemExit, match="choose a new --label"):
        harness.main()


def test_dry_run_redacts_authentication(case, provider, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")
    provider.update(base_url="https://example.invalid", api_key="secret-test-value")
    plan = harness.prepare_case(case, "C", provider, "medium", True)
    try:
        harness.print_dry_run(plan, case["case_id"], tmp_path)
        raw = capsys.readouterr().out
        assert "secret-test-value" not in raw
        assert json.loads(raw)["environment_overrides"]["ANTHROPIC_AUTH_TOKEN"] == "<redacted>"
        assert os.environ.get("ANTHROPIC_AUTH_TOKEN") != "secret-test-value"
    finally:
        harness.shutil.rmtree(plan["base"])
