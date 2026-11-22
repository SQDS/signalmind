"""Default-mode behavioural regression of server/app.py against the baseline commit's source (not just the tool-list hash).

The baseline `app.py` is loaded from git (`d7e97cc`, the commit before SignalMind), both servers run the same scripted
multi-channel session in-process, and every text reply and every image is compared byte for byte, with and without
images, for interface v1 and v2.
"""

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.server import app as current  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "d7e97cc"


def baseline_module():
    src = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:src/tsmcp/server/app.py"], cwd=ROOT)
    mod = types.ModuleType("_baseline_app")
    exec(compile(src, "baseline_app.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    d = tmp_path_factory.mktemp("rec")
    abp, ecg, _ = synth_abp(fs=125, duration_s=180, hrv=0.05, seed=4, flat=(100, 104), spikes=(140.2,))
    ppg = np.roll(abp, 30) / 40.0
    p = d / "rec.npy"
    np.save(p, np.stack([ecg, abp, ppg], 1).astype(np.float32))
    return p


def script(sid):
    return [
        ("ts_describe", {"id": sid}),
        ("ts_describe", {"id": sid, "detail": "high"}),
        ("ts_zoom", {"id": sid, "window": [10, 20]}),
        ("ts_zoom", {"id": sid, "window": [0, 180], "view": "strip", "channels": ["ABP"], "rows": 3}),
        ("ts_zoom", {"id": sid, "window": [95, 110], "channels": ["ABP"], "view": "level", "detail": "high"}),
        ("ts_zoom", {"id": sid, "window": [98, 106], "channels": ["ABP"], "view": "verify"}),
        ("ts_zoom", {"id": sid, "window": [0, 60], "channels": ["ECG"], "view": "ecg_rr"}),
        ("ts_zoom", {"id": sid, "window": [-5, 10]}),
        ("ts_zoom", {"id": sid, "window": [500, 600]}),
        ("ts_measure", {"id": sid, "what": ["mean", "rms", "amplitude_pp", "period", "dominant_freq"]}),
        (
            "ts_measure",
            {"id": sid, "what": ["ecg_hr", "ecg_rr_stats", "ecg_beats"], "window": [10, 40], "channels": ["ECG"]},
        ),
        ("ts_measure", {"id": sid, "what": ["mean"], "window": [100, 104], "channels": ["ABP", "PPG"]}),
        ("ts_measure", {"id": sid, "what": ["abp_sbp"], "channels": ["ABP"]}),
        ("ts_measure", {"id": sid, "what": ["nope"]}),
        ("ts_find_events", {"id": sid, "types": ["change", "flat", "spike"], "channels": ["ABP"], "snippets": True}),
        ("ts_find_events", {"id": sid, "types": ["ecg_irregular", "ecg_pause"], "channels": ["ECG"], "top_k": 3}),
        ("ts_find_events", {"id": sid, "types": ["bogus"]}),
        ("ts_zoom", {"id": sid, "window": [138, 143], "channels": ["ABP"]}),
        ("ts_measure", {"id": "zzzz", "what": ["mean"]}),
        ("ts_forget", {"id": sid}),
        ("ts_describe", {"id": sid}),
    ]


async def run_session(factory, store, path, images, interface):
    server = factory(store, images=images, interface=interface)
    out = []
    async with Client(server) as c:
        tools = (await c.list_tools()).tools
        out.append([(t.name, t.description, t.input_schema) for t in tools])
        loaded = await c.call_tool("ts_load", {"source": str(path), "fs": 125, "channels": ["ECG", "ABP", "PPG"],
                                               "units": ["mV", "mmHg", "?"]})  # fmt: skip
        out.append([x.text for x in loaded.content])
        sid = json.loads(loaded.content[0].text)["id"]
        for name, args in script(sid):
            r = await c.call_tool(name, args)
            out.append((name, r.is_error, [x.text if x.type == "text" else ("image", x.data) for x in r.content]))
    return out


@pytest.mark.parametrize("images", [True, False])
@pytest.mark.parametrize("interface", ["v1", "v2"])
def test_default_behaviour_identical_to_baseline(record, monkeypatch, tmp_path, images, interface):
    monkeypatch.setenv("TSMCP_CACHE_DIR", str(tmp_path / "cache"))
    base = baseline_module()
    a = asyncio.run(run_session(base.build_server, SignalStore(allow_roots=[record.parent]), record, images, interface))
    monkeypatch.setenv("TSMCP_CACHE_DIR", str(tmp_path / "cache2"))
    b = asyncio.run(
        run_session(current.build_server, SignalStore(allow_roots=[record.parent]), record, images, interface)
    )
    assert a[0] == b[0]  # tool list, descriptions, schemas
    assert a[1] == b[1]  # load reply
    for (na, ea, ca), (nb, eb, cb) in zip(a[2:], b[2:], strict=True):
        assert (na, ea) == (nb, eb)
        assert ca == cb, na
    assert sum(1 for step in a[2:] if step[1]) == (6 if images else 7)  # the scripted error cases really errored
    assert sum(1 for step in a[2:] for c in step[2] if isinstance(c, tuple)) == (8 if images else 0)
