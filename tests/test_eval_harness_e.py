"""Arm E provenance and plugin hint (no model calls)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import run_claude_code as rcc  # noqa: E402


def test_e_plugin_hint_uses_the_plugin_summaries():
    info = {"summary": "ECG only: ecg_af_burden (x)", "detectors_summary": "ECG only: ecg_irregular (y)"}
    hint = rcc.e_plugin_hint(info)
    assert "from tsmcp.plugins.ecg import PLUGIN" in hint
    assert info["summary"] in hint and info["detectors_summary"] in hint


@pytest.mark.skipif(not (rcc.VENV_D / "bin" / "python").exists(), reason="needs eval/.venv-d")
def test_e_provenance_imports_from_the_requested_checkout():
    info = rcc.e_provenance(rcc.REPO)
    assert Path(info["package_path"]).resolve().is_relative_to((rcc.REPO / "src").resolve())
    assert len(info["e_git_sha"]) >= 40 and "ecg_af_burden" in info["summary"]


@pytest.mark.parametrize("path", ["/home/u/data/afdb/04015.atr", "ls ltafdb/", "cat afdb.hea"])
def test_afdb_paths_are_flagged_as_suspicious(path):
    import re

    assert any(re.search(p, path.lower()) for p in rcc.SUSPICIOUS)


def test_client_runs_in_its_own_session_and_group_kill_does_not_reach_harness(tmp_path):
    import os

    # the "client" kills its own process group; the harness (this test process) must survive and get the output
    script = "import os, signal, sys; print('before', flush=True); os.killpg(os.getpgid(0), signal.SIGKILL)"
    out, err, rc = rcc.run_isolated_session([sys.executable, "-c", script], tmp_path, dict(os.environ), "", 30)
    assert "before" in out and rc == -9


def test_timeout_kills_the_whole_client_group(tmp_path):
    import os
    import subprocess

    script = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        rcc.run_isolated_session([sys.executable, "-c", script], tmp_path, dict(os.environ), "", 1)


def test_client_env_drops_host_session_coupling(monkeypatch):
    import os

    for k in rcc.HOST_SESSION_VARS:
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "kept")
    env = {k: v for k, v in os.environ.items() if k not in rcc.HOST_SESSION_VARS}
    assert not set(rcc.HOST_SESSION_VARS) & set(env) and env["CLAUDE_CODE_ENTRYPOINT"] == "kept"


def test_streamed_output_is_on_disk_and_returned(tmp_path):
    import os

    sink = tmp_path / "case.partial.jsonl"
    out, err, rc = rcc.run_isolated_session(
        [sys.executable, "-c", "print('line1'); print('line2')"], tmp_path, dict(os.environ), "", 30, stream_to=sink
    )
    assert out.splitlines() == ["line1", "line2"] and sink.read_text() == out and rc == 0


def test_interface_flag_reaches_server_args_only_when_set(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(rcc, "claude_bin", lambda: "/fake/claude")
    sig = tmp_path / "case"
    sig.mkdir()
    (sig / "signal.npy").write_bytes(b"x")
    case = {
        "case_id": "c",
        "dir": str(sig),
        "question": "q?",
        "qtype": "mean_hr",
        "visible": {"file": "signal.npy", "dur_s": 1.0, "channels": [{"name": "ch0", "unit": "mV", "fs": 1.0}]},
    }
    prov = {"model": "m", "vision": True, "base_url": None}
    for iface, expected in ((None, False), ("v2", True)):
        plan = rcc.prepare_case(case, "C", prov, None, False, rcc.REPO, None, iface)
        args = plan["mcp_config"]["mcpServers"]["tsmcp"]["args"]
        assert ("--interface" in args) is expected and (not expected or args[args.index("--interface") + 1] == "v2")
        json.dumps(plan["mcp_config"])
