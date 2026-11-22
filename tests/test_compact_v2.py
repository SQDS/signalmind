"""Synthetic presentation equivalence and frozen-baseline contract checks."""

import asyncio
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from tsmcp.core.compact import dumps, tround
from tsmcp.core.signal import time_decimals
from tsmcp.core.store import SignalStore
from tsmcp.plugins.ecg import af
from tsmcp.server.app import build_server, main
from tsmcp.server.compact_v2 import AF_CONVENTIONS, RANKING

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "baseline-R-2026-10-12"


def test_detector_and_all_measurement_helpers_are_byte_identical_to_baseline():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE, "src/tsmcp/core", "src/tsmcp/plugins"], cwd=ROOT, text=True
    ).splitlines()
    assert "src/tsmcp/plugins/ecg/af.py" in paths and "src/tsmcp/core/measure.py" in paths
    for path in paths:
        baseline = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)
        assert hashlib.sha256((ROOT / path).read_bytes()).digest() == hashlib.sha256(baseline).digest(), path


def fixture_analysis(supported=None, duration=400):
    times = np.arange(duration + 1, dtype=float)
    valid = np.ones(duration, bool)
    valid[20:30] = False
    return {
        "supported": [[10.0001, 20], [40, 60], [100, 110], [140, 150]] if supported is None else supported,
        "readable": [[0, 20], [30, duration]],
        "noisy": [[20, 30]],
        "beats": {
            "p": np.arange(duration + 1),
            "t_s": times,
            "n_leads": np.ones(duration + 1, int),
            "support": np.ones(duration + 1, int),
        },
        "rr": np.ones(duration),
        "valid": valid,
        "statistic": np.full(duration, 0.2123456),
    }


async def call(server, name, **args):
    result = await server.call_tool(name, args)
    assert not result.is_error
    return json.loads(result.content[0].text)


def make_signal(channels=1, fs=250, duration=400):
    store = SignalStore()
    signal = store.load(
        data=np.zeros((int(duration * fs), channels)), fs=fs, channels=[f"lead{i}" for i in range(channels)]
    )
    return store, signal


def test_default_and_explicit_v1_keep_baseline_schemas_and_tool_text(monkeypatch):
    monkeypatch.setattr(af, "af_analysis", lambda *a, **k: fixture_analysis())
    module = types.ModuleType("_compact_v2_baseline_app")
    source = subprocess.check_output(["git", "show", f"{BASELINE}:src/tsmcp/server/app.py"], cwd=ROOT)
    exec(compile(source, "baseline_app.py", "exec"), module.__dict__)

    async def capture(factory):
        server = factory(SignalStore(), images=False)
        tools = await server.list_tools()
        loaded = await server.call_tool("ts_load", {"data": [0.0] * 400, "fs": 1})
        sid = json.loads(loaded.content[0].text)["id"]
        calls = [
            ("ts_describe", {}),
            ("ts_zoom", {"window": [-1, 20]}),
            ("ts_measure", {"what": ["mean", "ecg_af_episodes", "ecg_af_burden"]}),
            ("ts_find_events", {"types": ["ecg_irregular", "ecg_noisy"], "top_k": 2}),
            *[("ts_measure", {"what": ["mean"]})] * 17,
            ("ts_forget", {}),
        ]
        outputs = [loaded.content[0].text]
        for name, args in calls:
            outputs.append((await server.call_tool(name, {"id": sid, **args})).content[0].text)
        return tools, outputs

    async def check():
        frozen = await capture(module.build_server)
        assert await capture(build_server) == frozen
        assert await capture(lambda store, **kw: build_server(store, interface="v1", **kw)) == frozen

    asyncio.run(check())


def test_complete_index_burden_detail_quality_and_restart(monkeypatch):
    analysis = fixture_analysis()
    monkeypatch.setattr(af, "af_analysis", lambda *a, **k: analysis)
    store, signal = make_signal(channels=2, fs=100)

    async def check():
        v1, v2 = build_server(store), build_server(store, interface="v2")
        args = {
            "id": signal.id,
            "what": ["ecg_af_episodes", "ecg_af_burden"],
            "window": [5.005, 180.003],
            "channels": ["lead1"],
        }
        old, new = await call(v1, "ts_measure", **args), await call(v2, "ts_measure", **args)
        index = new["measurements"][0]
        originals = old["measurements"][0]["episodes"]
        assert new["interface_version"] == "v2" and new["af_conventions"] == AF_CONVENTIONS
        assert index["count"] == len(index["episodes"]) == len(originals) == 3
        assert index["channel"] == "lead1" and index["units"] == "s" and index["window"] == [5, 180.01]
        assert index["ranking"] == RANKING and "intervals_s" not in dumps(index)
        for key, value in old["measurements"][1].items():
            if key != "method":
                assert new["measurements"][1][key] == value
        rows = index["episodes"]
        assert rows[0]["merged_from"] == 2 and rows[0]["unknown_s"] == 10
        assert rows[0]["span_s"] - rows[0]["af_s"] == 20  # readable non-AF gap is not unknown
        assert rows[1]["start"] == 100 and rows[2]["start"] == 140  # exact 30-second gap does not merge
        assert rows[1]["span_s"] == rows[2]["span_s"]  # earliest start breaks ties
        restarted = build_server(SignalStore(), interface="v2")
        for row, original in zip(rows, originals, strict=True):
            assert set(row) == {"id", "start", "end", "span_s", "af_s", "unknown_s", "merged_from"}
            detail_args = {"id": signal.id, "what": ["ecg_af_episode_detail"], "episode": row["id"]}
            detail = (await call(v2, "ts_measure", **detail_args))["measurements"][0]
            assert detail == (await call(restarted, "ts_measure", **detail_args))["measurements"][0]
            assert detail["legacy_episode"] == original
            a, b = original["start"], original["intervals_s"][-1][1]
            assert detail["readable_intervals_s"] == af.clip_intervals(analysis["readable"], a, b)
            assert detail["unknown_intervals_s"] == af.clip_intervals(analysis["noisy"], a, b)
        assert (await call(restarted, "ts_measure", **args))["measurements"][0] == index

    asyncio.run(check())


def test_raw_span_ranking_precedes_rounding_and_index_never_truncates(monkeypatch):
    tied_starts = [121 + i * 50 for i in range(18)]
    supported = [[1, 11.0001], [61, 71.0002], *[[start, start + 10] for start in tied_starts]]
    monkeypatch.setattr(af, "af_analysis", lambda *a, **k: fixture_analysis(supported, duration=1000))
    store, signal = make_signal(fs=100, duration=1000)

    async def check():
        response = await call(build_server(store, interface="v2"), "ts_measure", id=signal.id, what=["ecg_af_episodes"])
        index = response["measurements"][0]
        assert index["count"] == len(index["episodes"]) == len(supported) == 20
        assert [e["start"] for e in index["episodes"]] == [61, 1, *tied_starts]
        assert {e["span_s"] for e in index["episodes"]} == {10}

    asyncio.run(check())


@pytest.mark.parametrize("readable", [True, False])
def test_empty_index_and_no_readable_burden(monkeypatch, readable):
    analysis = fixture_analysis([])
    if not readable:
        analysis["readable"], analysis["noisy"] = [], [[0, 400]]
    monkeypatch.setattr(af, "af_analysis", lambda *a, **k: analysis)
    store, signal = make_signal(fs=1)

    async def check():
        args = {"id": signal.id, "what": ["ecg_af_episodes", "ecg_af_burden"]}
        new = await call(build_server(store, interface="v2"), "ts_measure", **args)
        old = await call(build_server(store), "ts_measure", **args)
        assert new["measurements"][0]["count"] == 0 and new["measurements"][0]["episodes"] == []
        new_burden, old_burden = new["measurements"][1], old["measurements"][1]
        new_burden.pop("method")
        old_burden.pop("method")
        assert new_burden == old_burden
        assert ("pct_of_readable" in new_burden) is readable

    asyncio.run(check())


def test_v2_real_mcp_all_tools_versioned_and_descriptions_complete(monkeypatch):
    monkeypatch.setattr(af, "af_analysis", lambda *a, **k: fixture_analysis())

    async def check():
        async with Client(build_server(SignalStore(), images=False, interface="v2")) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert all(len(t.description) < 2000 for t in tools.values())
            for name in ("ts_measure", "ts_find_events"):
                assert "strictly <30 s" in tools[name].description
                assert "whole-record burden" in tools[name].description
                assert "Unknown time is not evidence of absence of AF" in tools[name].description
            assert "episode" in tools["ts_measure"].input_schema["properties"]
            loaded = json.loads((await client.call_tool("ts_load", {"data": [0.0] * 400, "fs": 1})).content[0].text)
            assert loaded["interface_version"] == "v2"
            for name, args in [
                ("ts_describe", {}),
                ("ts_zoom", {"window": [0, 100]}),
                ("ts_measure", {"what": ["ecg_af_episodes", "ecg_af_burden"]}),
                ("ts_find_events", {"types": ["ecg_irregular"]}),
                ("ts_forget", {}),
            ]:
                out = await client.call_tool(name, {"id": loaded["id"], **args})
                assert not out.is_error
                assert json.loads(out.content[0].text)["interface_version"] == "v2"
                assert out.content[0].text.count('"af_conventions"') <= 1

    asyncio.run(check())


def test_episode_detail_rejects_ambiguous_or_invalid_requests(monkeypatch):
    monkeypatch.setattr(af, "af_analysis", lambda *a, **k: fixture_analysis())
    store, signal = make_signal(fs=1)

    async def check():
        server = build_server(store, interface="v2")
        index = (await call(server, "ts_measure", id=signal.id, what=["ecg_af_episodes"]))["measurements"][0]
        episode = index["episodes"][0]["id"]
        for args in [
            {"what": ["ecg_af_episode_detail"]},
            {"what": ["ecg_af_episode_detail"], "episode": "bad"},
            {"what": ["ecg_af_episode_detail"], "episode": episode[:-1] + ("0" if episode[-1] != "0" else "1")},
            {"what": ["ecg_af_episode_detail"], "episode": episode, "window": [0, 200]},
            {"what": ["ecg_af_burden"], "episode": episode},
        ]:
            with pytest.raises(ToolError):
                await server.call_tool("ts_measure", {"id": signal.id, **args})

    asyncio.run(check())


def test_synthetic_ecg_actual_detector_equivalence():
    from test_ecg import synth_ecg

    x, _ = synth_ecg(np.random.default_rng(8).uniform(0.45, 1.1, 120), fs=100)
    store = SignalStore()
    signal = store.load(data=x, fs=100)

    async def check():
        old_server, new_server = build_server(store), build_server(store, interface="v2")
        args = {"id": signal.id, "what": ["ecg_af_episodes", "ecg_af_burden"]}
        old, new = await call(old_server, "ts_measure", **args), await call(new_server, "ts_measure", **args)
        originals = old["measurements"][0]["episodes"]
        assert originals
        index = new["measurements"][0]
        assert index["count"] == len(originals)
        for row, original in zip(index["episodes"], originals, strict=True):
            detail = await call(
                new_server, "ts_measure", id=signal.id, what=["ecg_af_episode_detail"], episode=row["id"]
            )
            assert detail["measurements"][0]["legacy_episode"] == original
            for key in ("start", "span_s", "af_s"):
                assert row[key] == tround(original[key], time_decimals(100))
        assert {k: v for k, v in new["measurements"][1].items() if k != "method"} == {
            k: v for k, v in old["measurements"][1].items() if k != "method"
        }

    asyncio.run(check())


@pytest.mark.parametrize("args,version", [([], "v1"), (["--interface", "v2"], "v2")])
def test_cli_version_selection(monkeypatch, args, version):
    captured = {}

    def factory(store, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(run=lambda transport: captured.update(transport=transport))

    monkeypatch.setattr("tsmcp.server.app.build_server", factory)
    main(args)
    assert captured == {"images": True, "interface": version, "transport": "stdio"}
