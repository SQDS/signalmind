"""B contracts with scripted MCP replies and synthetic signals; never a model."""

import asyncio
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import b_workflow as bw  # noqa: E402
from eval import run_claude_code as harness  # noqa: E402


@pytest.fixture
def case(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    np.save(directory / "signal.npy", np.sin(np.arange(1000)), allow_pickle=False)
    return {
        "case_id": "synthetic",
        "qtype": "longest_af",
        "question": "When does the longest AF begin?",
        "dir": str(directory),
        "visible": {"file": "signal.npy", "dur_s": 10000, "channels": [{"name": "ch0", "unit": "mV", "fs": 250}]},
    }


def episode(t, span):
    return {"t": t, "span_s": span, "af_s": span - 20, "intervals_s": [[t, t + 30], [t + 50, t + span]]}


class ScriptedClient:
    def __init__(self, episodes=None, padding=0, core_padding=None, verification_padding=0):
        self.episodes = (
            episodes if episodes is not None else [episode(100 + i * 1200, 1000 - i * 100) for i in range(7)]
        )
        self.padding = padding
        self.core_padding = core_padding or {}
        self.verification_padding = verification_padding
        self.calls = []
        self.replies = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, copy.deepcopy(arguments)))
        data = {"id": "12345678"}
        if name in ("ts_load", "ts_describe"):
            data.update(dur_s=10000)
        elif name == "ts_find_events":
            data["events"] = [{"t": 9000, "dur": 70}, {"t": 9100, "dur": 90}, {"t": 9400, "dur": 80}]
        elif name == "ts_measure":
            what = arguments["what"][0]
            row = {"name": what, "ref": f"12345678:m.{what}@0-10000", "ch": "ch0"}
            if what == "ecg_af_episodes":
                row.update(episodes=self.episodes, padding="x" * self.padding, merge_gap_s=30)
            elif what == "ecg_af_burden":
                row.update(pct_of_window=12.3, af_s=1230, window_s=10000, readable_s=9000)
            else:
                row.update(cv=0.123, n=80)
            data["measurements"] = [row]
        core_key = {
            "ts_load": "load",
            "ts_describe": "describe",
            "ecg_af_burden": "burden",
            "ecg_af_episodes": "episode_index",
        }.get(arguments.get("what", [name])[0])
        if core_key in self.core_padding:
            data["padding"] = core_key + ":" + "x" * self.core_padding[core_key]
        elif core_key is None and self.verification_padding:
            data["padding"] = "v" * self.verification_padding
        # Whitespace and key order are intentional: must survive verbatim.
        raw = json.dumps(data, indent=1)
        blocks = [SimpleNamespace(type="text", text=raw)]
        if name == "ts_zoom":
            blocks.append(SimpleNamespace(type="image", mime_type="image/png", data="cG5n"))
        self.replies.append(raw)
        return SimpleNamespace(content=blocks, is_error=False)


def test_deterministic_verbatim_and_call_priority(case):
    first, second = ScriptedClient(), ScriptedClient()
    a = asyncio.run(bw.collect(case, first))
    b = asyncio.run(bw.collect(case, second))
    assert bw.canonical(a) == bw.canonical(b)
    assert bw.digest(bw.canonical(a).encode()) == bw.digest(bw.canonical(b).encode())
    assert a["status"] == "ready" and a["sizes"]["calls"] == len(first.calls) == 20
    assert a["sizes"]["images"] <= 8 and a["sizes"]["text_chars"] <= 32000
    assert a["verification_text_budget"] == 32000 - a["sizes"]["core_text_chars"]
    assert a["sizes"]["verification_text_chars"] <= a["verification_text_budget"]
    assert [i["key"] for i in a["items"][:5]] == ["load", "describe", "burden", "episode_index", "whole_rr"]
    assert all(i["content"][0]["text"] == first.replies[i["call_number"] - 1] for i in a["items"])
    assert len([i for i in a["items"] if i["key"].startswith("onset.")]) == 9
    assert any(d["reason"] == "call_budget" for d in a["dropped"])
    assert all("content" not in d for d in a["dropped"])
    assert a["arithmetic"] == []


def test_candidate_union_ranking_and_strict_gap_boundary():
    eps = [episode(i * 1500, span) for i, span in enumerate([1000, 900, 800, 700, 600, 500, 499])]
    assert bw.candidates({"episodes": list(reversed(eps))}) == eps[:6]
    tied = [episode(300, 100), episode(0, 100)]
    assert bw.candidates({"episodes": tied})[0]["t"] == 0
    eps = [episode(0, 100), episode(399, 100), episode(799, 100)]
    plan, skipped = bw.longest_plan("id", {"episodes": eps}, 1000)
    gap_windows = [p["arguments"]["window"] for p in plan if p["key"].startswith("gap.")]
    assert [100, 399] in gap_windows and [499, 799] not in gap_windows
    assert skipped[0]["key"] == "onset.1.before"
    assert all(p["arguments"]["window"][0] >= 0 for p in plan)


@pytest.mark.parametrize(
    "core_chars,limit",
    [(0, 32000), (15999, 16001), (16000, 16000), (16001, 16000), (32000, 16000), (64000, 16000)],
)
def test_verification_budget_residual_and_floor(core_chars, limit):
    assert bw.BUDGET.text_chars == 32000 and bw.BUDGET.verification_floor_chars == 16000
    assert bw.BUDGET.verification_limit(core_chars) == limit


def test_text_drops_verification_lowest_priority_first_and_counts_omitted_calls(case):
    original = asyncio.run(bw.collect(case, ScriptedClient()))
    retained = [i for i in original["items"] if not i["required"] and i["priority"] <= 10]
    allowance = bw.sizes(bw.evidence_content(retained, []))["text_chars"]
    cap = original["sizes"]["core_text_chars"] + allowance
    bundle = asyncio.run(bw.collect(case, ScriptedClient(), bw.Budget(text_chars=cap, verification_floor_chars=0)))
    assert bundle["verification_text_budget"] == allowance
    assert bundle["sizes"]["verification_text_chars"] == allowance and bundle["sizes"]["calls"] == 20
    assert bundle["sizes"]["retained_calls"] < bundle["sizes"]["calls"]
    removed = [d for d in bundle["dropped"] if d["reason"] == "text_budget"]
    assert [(d["priority"], d["call_number"]) for d in removed] == sorted(
        [(d["priority"], d["call_number"]) for d in removed], reverse=True
    )
    core_keys = {"load", "describe", "burden", "episode_index"}
    assert {i["key"] for i in bundle["items"] if i["required"]} == core_keys
    assert not core_keys.intersection(d["key"] for d in bundle["dropped"])
    assert {i["key"] for i in bundle["items"] if not i["required"]} == {i["key"] for i in retained}
    assert all(not d["required"] for d in removed)
    assert {f"onset.{i}.strip" for i in (1, 2, 3)} <= {i["key"] for i in bundle["items"]}
    assert all(d["executed"] and d["response_sha256"] for d in removed)


@pytest.mark.parametrize("qtype,expected_calls", [("longest_af", 20), ("af_burden", 12)])
def test_all_four_core_outputs_remain_complete_above_cap(case, qtype, expected_calls):
    case["qtype"] = qtype
    core_keys = {"load", "describe", "burden", "episode_index"}
    client = ScriptedClient(core_padding=dict.fromkeys(core_keys, 33000))
    bundle = asyncio.run(bw.collect(case, client))
    assert bundle["status"] == "ready" and bundle["sizes"]["calls"] == expected_calls
    assert bundle["sizes"]["core_text_chars"] > 132000
    assert bundle["sizes"]["text_chars"] > 32000
    assert bundle["verification_text_budget"] == 16000
    assert 0 < bundle["sizes"]["verification_text_chars"] <= 16000
    assert {i["key"] for i in bundle["items"] if i["required"]} == core_keys
    for item in bundle["items"][:4]:
        assert item["content"][0]["text"] == client.replies[item["call_number"] - 1]
        assert bw.payload(item)["padding"] == item["key"] + ":" + "x" * 33000
    assert not core_keys.intersection(d["key"] for d in bundle["dropped"])
    assert any(i["tool"] == "ts_zoom" and i["arguments"].get("view") == "strip" for i in bundle["items"])
    message = json.loads(bw.user_message("question", bundle))["message"]["content"]
    assert message[1:] == bundle["content"]


def test_default_floor_drops_verification_with_oversized_core(case):
    core_keys = {"load", "describe", "burden", "episode_index"}
    client = ScriptedClient(core_padding=dict.fromkeys(core_keys, 33000), verification_padding=1500)
    bundle = asyncio.run(bw.collect(case, client))
    assert bundle["status"] == "ready"
    assert bundle["sizes"]["core_text_chars"] > 132000
    assert bundle["verification_text_budget"] == 16000
    assert sum(len(reply) for reply in client.replies[4:]) > 16000
    assert 0 < bundle["sizes"]["verification_text_chars"] <= 16000
    assert bundle["sizes"]["calls"] == len(client.calls) == 20
    assert 4 < bundle["sizes"]["retained_calls"] < 20
    assert {i["key"] for i in bundle["items"] if i["required"]} == core_keys
    assert all(i["content"][0]["text"] == client.replies[i["call_number"] - 1] for i in bundle["items"])
    text_drops = [d for d in bundle["dropped"] if d["reason"] == "text_budget"]
    assert text_drops and all(d["executed"] and not d["required"] for d in text_drops)
    assert not core_keys.intersection(d["key"] for d in bundle["dropped"])


def test_images_drop_before_execution(case):
    bundle = asyncio.run(bw.collect(case, ScriptedClient(), bw.Budget(images=2)))
    assert bundle["sizes"]["images"] == 2
    assert any(d["reason"] == "image_budget" and not d["executed"] for d in bundle["dropped"])


def test_hourly_support_not_span_and_numeric_restating():
    ep = {"t": 3590, "span_s": 50, "af_s": 30, "intervals_s": [[3590, 3605], [3625, 3640]]}
    burden = {"pct_of_window": 0.417, "af_s": 30, "window_s": 7200, "readable_s": 7000}
    rows = bw.burden_arithmetic({"episodes": [ep]}, burden, "index-ref", "burden-ref")
    assert rows[0]["hours"] == [{"window": [0, 3600], "af_s": 10}, {"window": [3600, 7200], "af_s": 20}]
    assert {k: rows[1][k] for k in burden} == burden
    assert rows[1]["source_ref"] == "burden-ref"


def test_burden_unknown_strips_and_gap_selection(case):
    case["qtype"] = "af_burden"
    bundle = asyncio.run(bw.collect(case, ScriptedClient()))
    assert bundle["sizes"]["calls"] == 12 and bundle["sizes"]["images"] == 7
    assert bundle["items"][5]["arguments"] == {"id": "12345678", "types": ["ecg_noisy"], "top_k": 3}
    strips = bundle["items"][6:]
    assert [i["arguments"]["window"][0] for i in strips[:3]] == [9100, 9400, 9000]
    assert len(bundle["arithmetic"]) == 2
    assert {i["key"]: i["priority"] for i in bundle["items"][4:6]} == {"whole_rr": 5, "unknown_index": 5}


@pytest.mark.parametrize("qtype", ["longest_af", "af_burden"])
def test_core_and_verification_sizes_include_framing_and_arithmetic(case, qtype):
    case["qtype"] = qtype
    bundle = asyncio.run(bw.collect(case, ScriptedClient()))
    core_chars = verification_chars = tool_chars = 0
    for item in bundle["items"]:
        raw_chars = sum(len(block["text"]) for block in item["content"] if block["type"] == "text")
        tool_chars += raw_chars
        framed_chars = raw_chars + len(bw.canonical({"tool": item["tool"], "arguments": item["arguments"]})) + 1
        if item["required"]:
            core_chars += framed_chars
        else:
            verification_chars += framed_chars
    verification_chars += sum(len(bw.canonical(row)) for row in bundle["arithmetic"])
    assert bundle["sizes"]["core_text_chars"] == core_chars
    assert bundle["sizes"]["verification_text_chars"] == verification_chars
    assert bundle["sizes"]["text_chars"] == core_chars + verification_chars
    assert bundle["sizes"]["text_chars"] == sum(len(b["text"]) for b in bundle["content"] if b["type"] == "text")
    assert bundle["sizes"]["tool_text_chars"] == tool_chars


@pytest.mark.parametrize("keep_arithmetic", [True, False])
def test_arithmetic_uses_verification_budget_and_is_dropped_last(case, keep_arithmetic):
    case["qtype"] = "af_burden"
    original = asyncio.run(bw.collect(case, ScriptedClient()))
    arithmetic_chars = sum(len(bw.canonical(row)) for row in original["arithmetic"])
    allowance = arithmetic_chars if keep_arithmetic else 0
    budget = bw.Budget(text_chars=0, verification_floor_chars=allowance)
    bundle = asyncio.run(bw.collect(case, ScriptedClient(), budget))
    assert bundle["status"] == "ready"
    assert bundle["verification_text_budget"] == allowance
    assert bundle["sizes"]["verification_text_chars"] == allowance
    assert bundle["sizes"]["calls"] == original["sizes"]["calls"] == 12
    assert bundle["sizes"]["retained_calls"] == 4 and bundle["sizes"]["images"] == 0
    assert all(i["required"] for i in bundle["items"])
    assert bundle["arithmetic"] == (original["arithmetic"] if keep_arithmetic else [])
    removed = [d for d in bundle["dropped"] if d["reason"] == "text_budget"]
    assert [d["priority"] for d in removed] == sorted([d["priority"] for d in removed], reverse=True)
    tool_keys = {i["key"] for i in original["items"] if not i["required"]}
    assert {d["key"] for d in removed if d["key"] in tool_keys} == tool_keys
    arithmetic_removed = [d for d in removed if d["key"] not in tool_keys]
    assert len(arithmetic_removed) == (0 if keep_arithmetic else 2)
    assert all(d["priority"] == 0 for d in arithmetic_removed)


def test_empty_episodes_and_tool_error_preserved(case):
    class ErrorClient(ScriptedClient):
        async def call_tool(self, name, arguments):
            result = await super().call_tool(name, arguments)
            if arguments.get("what") == ["ecg_rr_stats"]:
                result.content = [SimpleNamespace(type="text", text="too few readable RR intervals in this window")]
                result.is_error = True
            return result

    empty = asyncio.run(bw.collect(case, ScriptedClient(episodes=[])))
    assert empty["sizes"]["calls"] == 5 and not empty["dropped"]
    errors = asyncio.run(bw.collect(case, ErrorClient()))
    assert any(i["is_error"] and "too few readable" in i["content"][0]["text"] for i in errors["items"])


def test_unknown_suite_rejected_before_io(tmp_path, monkeypatch):
    # Freeze amendment F1 (2026-10-22) allows the frozen p6confirm run; any other suite is still refused before I/O.
    def forbidden(*args, **kwargs):
        pytest.fail("unlisted suite must be rejected before any read")

    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(ValueError, match="restricted"):
        bw.public_cases(tmp_path, "p9unlisted")


def test_cache_integrity_and_identity(case, tmp_path):
    bundle = asyncio.run(bw.collect(case, ScriptedClient()))
    identity = bw.request_identity(case, {"server_git_sha": bw.FROZEN_R})
    key = bw.digest(bw.canonical(identity).encode())
    sha = bw.digest((bw.canonical(bundle) + "\n").encode())
    meta = bw.manifest(bundle, sha, identity, key)
    assert meta["sizes"] == bundle["sizes"]
    assert meta["verification_text_budget"] == bundle["verification_text_budget"]
    assert meta["budget"]["verification_floor_chars"] == 16000
    assert "required_text_chars" not in meta and "required_text_chars" not in bundle
    bw.write_json(tmp_path / "objects" / f"{sha}.json", bundle)
    bw.write_json(tmp_path / "requests" / f"{key}.json", meta)
    assert bw.load_cached(tmp_path, key) == (bundle, meta)
    changed = bw.request_identity({**case, "question": "different?"}, {"server_git_sha": bw.FROZEN_R})
    assert identity != changed
    (tmp_path / "objects" / f"{sha}.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        bw.load_cached(tmp_path, key)


def test_real_mcp_bundle_byte_identical_on_synthetic_signal(case, tmp_path):
    from mcp import Client

    from tsmcp.core.store import SignalStore
    from tsmcp.server.app import build_server

    # No clinical data and no expensive detector session: a short regular waveform.
    fs = 100
    t = np.arange(0, 30, 1 / fs)
    np.save(Path(case["dir"]) / "signal.npy", np.sin(2 * np.pi * 1.2 * t), allow_pickle=False)
    case["visible"]["channels"][0]["fs"] = fs

    async def once():
        async with Client(build_server(SignalStore(allow_roots=[case["dir"]]))) as client:
            return await bw.collect(case, client)

    a, b = asyncio.run(once()), asyncio.run(once())
    assert bw.canonical(a) == bw.canonical(b)
    assert all(block["source"]["data"].startswith("iVBOR") for block in a["content"] if block["type"] == "image")


def fake_worker(case, server_root):
    bundle = asyncio.run(bw.collect(case, ScriptedClient()))
    provenance = {
        "server_root": str(server_root),
        "server_git_sha": bw.FROZEN_R,
        "server_package_path": str(server_root / "src/tsmcp/__init__.py"),
        "interface": "v1",
    }
    identity = bw.request_identity(case, provenance)
    return bundle, bw.manifest(bundle, "a" * 64, identity, "b" * 64)


def test_b_dry_run_exact_prompt_no_mcp_one_turn(case, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bw, "build_in_worker", fake_worker)
    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")
    prov = {"model": "same-as-C", "vision": True, "base_url": None}
    plan = harness.prepare_case(case, "B", prov, "medium", True)
    try:
        command = plan["command"]
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--max-turns") + 1] == "1"
        assert command[command.index("--model") + 1] == "same-as-C"
        assert "--mcp-config" not in command and plan["mcp_config"] is None
        content = json.loads(plan["stdin"])["message"]["content"]
        assert content[0]["text"].startswith(harness.case_prompt(case["visible"], case["question"], case["qtype"]))
        harness.print_dry_run(plan, case["case_id"], tmp_path)
        printed = json.loads(capsys.readouterr().out)
        assert Path(printed["stdin_path"]).read_text() == plan["stdin"]
        summary = json.loads(printed["stdin"])["message"]["content"]
        assert [b for b in content if b["type"] == "text"] == [b for b in summary if b["type"] == "text"]
        assert all("data" not in b["source"] for b in summary if b["type"] == "image")
    finally:
        shutil.rmtree(plan["base"])


@pytest.mark.parametrize("tool_calls,turns,valid", [(False, 1, True), (True, 1, False), (False, 2, False)])
def test_b_result_provenance_usage_and_single_turn_guard(case, tmp_path, monkeypatch, tool_calls, turns, valid):
    monkeypatch.setattr(bw, "build_in_worker", fake_worker)
    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")

    def local_result(command, cwd, env, stdin, timeout_s, stream_to=None):
        assert timeout_s == 900
        content = [{"type": "tool_use", "name": "Read", "input": {}}] if tool_calls else []
        events = [
            {"type": "assistant", "message": {"model": "same-as-C", "content": content}},
            {
                "type": "result",
                "result": 'ANSWER: {"t": 100}',
                "num_turns": turns,
                "usage": {"input_tokens": 100},
                "total_cost_usd": 0.01,
            },
        ]
        return "\n".join(json.dumps(e) for e in events), "", 0

    monkeypatch.setattr(harness, "run_isolated_session", local_result)
    result = harness.run_case(
        case, "B", {"model": "same-as-C", "vision": True, "base_url": None}, "medium", tmp_path, True
    )
    assert result["server_git_sha"] == bw.FROZEN_R and result["bundle_sha256"] == "a" * 64
    if valid:
        assert result["usage"]["input_tokens"] == 100 and result["turns"] == 1
        assert result["tool_calls"] == [] and result["cc_cost_usd"] == 0.01
    else:
        assert result["error"] == "B_single_turn_contract"


def test_worker_uses_r_python_and_content_addressed_result(case, tmp_path, monkeypatch):
    bundle, meta = fake_worker(case, tmp_path / "R")
    seen = []

    def worker(command, **kwargs):
        seen.append(command)
        assert json.loads(kwargs["input"]) == case
        return SimpleNamespace(stdout=json.dumps({"request_sha256": "b" * 64}))

    monkeypatch.setattr(bw.subprocess, "run", worker)
    monkeypatch.setattr(bw, "load_cached", lambda cache, key: (bundle, meta))
    assert bw.build_in_worker(case, tmp_path / "R", tmp_path / "cache") == (bundle, meta)
    assert seen[0][0] == str(tmp_path / "R/.venv/bin/python")
    assert "--worker" in seen[0] and "--server-root" in seen[0]


def test_source_verification_checks_venv_import_sha_and_bytes(tmp_path, monkeypatch):
    import tsmcp

    root = tmp_path / "R"
    package = root / "src/tsmcp/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"# frozen\n")
    monkeypatch.setattr(tsmcp, "__file__", str(package))
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))

    def git(command, **kwargs):
        if "rev-parse" in command:
            return bw.FROZEN_R + "\n"
        if "ls-tree" in command:
            return "src/tsmcp/__init__.py\n"
        return b"# frozen\n"

    monkeypatch.setattr(bw.subprocess, "check_output", git)
    assert bw.verify_server(root)["server_package_path"] == str(package)
    package.write_bytes(b"# modified\n")
    with pytest.raises(ValueError, match="modified frozen"):
        bw.verify_server(root)
    package.write_bytes(b"# frozen\n")
    (package.parent / "extra.py").write_text("")
    with pytest.raises(ValueError, match="extra frozen"):
        bw.verify_server(root)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "wrong"))
    with pytest.raises(ValueError, match="R venv"):
        bw.verify_server(root)


def test_b_main_dry_run_never_requests_balance_or_launches_model(case, monkeypatch, capsys):
    monkeypatch.setattr(bw, "public_cases", lambda *args: [case])
    monkeypatch.setattr(bw, "build_in_worker", fake_worker)
    monkeypatch.setattr(harness, "claude_bin", lambda: "/fake/claude")

    def forbidden(*args, **kwargs):
        pytest.fail("dry run cannot call a model or a balance endpoint")

    monkeypatch.setattr(harness, "balance", forbidden)
    monkeypatch.setattr(harness, "run_isolated_session", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_claude_code.py", "--suite", "p5afdb", "--arm", "B", "--dry-run"])
    harness.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["case_id"] == case["case_id"] and printed["bundle_manifest"]["status"] == "ready"
    shutil.rmtree(Path(printed["cwd"]).parent)
