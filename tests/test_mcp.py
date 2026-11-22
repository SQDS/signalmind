"""Through the real MCP layer, in-process: what we test is what clients get."""

import asyncio
import json

import pytest
from mcp import Client

from tsmcp.core.store import SignalStore
from tsmcp.server.app import build_server


def run(coro):
    return asyncio.run(coro)


async def _session(store, images=True, calls=()):
    out = []
    async with Client(build_server(store, images=images)) as c:
        out.append(await c.list_tools())
        for name, args in calls:
            out.append(await c.call_tool(name, args))
    return out


def test_tool_list_static_and_annotated():
    tools = run(_session(SignalStore()))[0].tools
    names = sorted(t.name for t in tools)
    assert names == ["ts_describe", "ts_find_events", "ts_forget", "ts_load", "ts_measure", "ts_zoom"]
    for t in tools:
        assert "." not in t.name
        assert t.annotations.open_world_hint is False
        assert len(t.description) < 2000  # clients truncate long descriptions
    assert run(_session(SignalStore()))[0].tools == tools


def test_end_to_end(sine_npy):
    p, fs = sine_npy
    st = SignalStore(allow_roots=[p.parent])
    lst, loaded = run(_session(st, calls=[("ts_load", {"source": str(p), "fs": fs})]))
    assert not loaded.is_error
    sid = json.loads(loaded.content[0].text)["id"]
    _, zoom, meas, bad = run(
        _session(
            st,
            calls=[
                ("ts_zoom", {"id": sid, "window": [290, 310]}),
                (
                    "ts_measure",
                    {
                        "id": sid,
                        "window": [0, 290],
                        "what": ["period", "dominant_freq"],
                    },
                ),
                ("ts_zoom", {"id": sid, "window": [900, 1000]}),
            ],
        )
    )
    assert [b.type for b in zoom.content] == ["text", "image"]
    m = json.loads(meas.content[0].text)["measurements"]
    assert m[0]["value"] == pytest.approx(1 / 1.2, rel=0.01) and m[0]["ref"].startswith(sid + ":m.period@")
    assert bad.is_error and "outside the signal" in bad.content[0].text


def test_unknown_id_and_no_images(sine_npy):
    p, fs = sine_npy
    st = SignalStore(allow_roots=[p.parent])
    _, r = run(_session(st, calls=[("ts_describe", {"id": "deadbeef"})]))
    assert r.is_error and "ts_load" in r.content[0].text
    sid = st.load(source=str(p), fs=fs).id
    _, z = run(_session(st, images=False, calls=[("ts_zoom", {"id": sid, "window": [0, 10]})]))
    assert [b.type for b in z.content] == ["text"]


def test_stdio_entrypoint(sine_npy):
    """The real `tsmcp` console script over stdio, as a client would launch it."""
    import os
    import sys

    from mcp import StdioServerParameters

    p, fs = sine_npy
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from tsmcp import main; main()", "--allow-root", str(p.parent)],
        env={**os.environ},
    )

    async def go():
        async with Client(params) as c:
            names = sorted(t.name for t in (await c.list_tools()).tools)
            r = await c.call_tool("ts_load", {"source": str(p), "fs": fs})
            return names, r

    names, r = run(go())
    assert "ts_load" in names and not r.is_error
