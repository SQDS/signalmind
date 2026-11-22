import json

import numpy as np
import pytest

from tsmcp.core.peaks import find_peaks
from tsmcp.plugins import PLUGINS, all_measures
from tsmcp.plugins.ecg import ecg_beats, ecg_hr, ecg_rr_profile
from tsmcp.plugins.ecg.rpeaks import detect_r_peaks

FS = 360.0


def synth_ecg(rr: np.ndarray, fs: float = FS, noise: float = 0.02, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """QRS-like narrow biphasic pulses + P/T bumps + baseline wander at the given RR sequence."""
    t_beats = 1.0 + np.concatenate([[0], np.cumsum(rr)])
    n = int((t_beats[-1] + 1.0) * fs)
    t = np.arange(n) / fs
    x = 0.15 * np.sin(2 * np.pi * 0.25 * t)
    for tb in t_beats:
        x += 1.0 * np.exp(-(((t - tb) / 0.012) ** 2)) - 0.25 * np.exp(-(((t - tb - 0.03) / 0.015) ** 2))
        x += 0.15 * np.exp(-(((t - tb - 0.25) / 0.05) ** 2)) + 0.1 * np.exp(-(((t - tb + 0.16) / 0.03) ** 2))
    x += noise * np.random.default_rng(seed).standard_normal(n)
    return x, t_beats


def test_find_peaks_matches_scipy():
    from scipy.signal import find_peaks as sp_find_peaks

    rng = np.random.default_rng(0)
    for _ in range(100):
        x = np.convolve(rng.standard_normal(3000), np.ones(9) / 9, "same")
        h, pr, d = np.quantile(x, rng.uniform(0.3, 0.9)), rng.uniform(0, 0.5), int(rng.integers(1, 50))
        assert np.array_equal(
            sp_find_peaks(x, height=h, prominence=pr, distance=d)[0], find_peaks(x, height=h, prominence=pr, distance=d)
        )


def test_rpeaks_on_synthetic_ecg():
    rr = 0.8 + 0.05 * np.sin(np.arange(120) / 5)
    x, tb = synth_ecg(rr)
    p = detect_r_peaks(x, FS) / FS
    inner = tb[(tb > 0.2) & (tb < x.size / FS - 0.2)]
    hits = [np.min(np.abs(p - b)) < 0.05 for b in inner]
    assert np.mean(hits) > 0.99 and abs(p.size - inner.size) <= 1


def test_hr_and_beats():
    x, tb = synth_ecg(np.full(100, 0.75))
    hr = ecg_hr(x, FS, "mV")
    assert hr["value"] == pytest.approx(80, rel=0.01)
    b = ecg_beats(x[: int(30 * FS)], FS, "mV", t0=100.0)
    assert b["t_s"][0] == pytest.approx(101.0, abs=0.02)  # absolute times honor t0


def test_rr_profile_flags_bigeminy_and_af():
    regular = np.full(80, 0.8)
    bigem = np.tile([0.5, 1.1], 40)  # short-long alternation
    af = np.random.default_rng(1).uniform(0.45, 1.1, 80)  # irregularly irregular
    x, tb = synth_ecg(np.concatenate([regular, bigem, af]))
    prof = ecg_rr_profile(x, FS, "mV")
    dt = prof["dt_s"]
    blk = lambda t: int(t // dt)  # noqa: E731
    t_bigem, t_af = 1 + regular.sum(), 1 + regular.sum() + bigem.sum()
    reg_alt = prof["alt"][blk(10)]
    big_alt = prof["alt"][blk(t_bigem + 20)]
    af_cv, af_alt = prof["rr_cv"][blk(t_af + 20)], prof["alt"][blk(t_af + 20)]
    assert reg_alt < 0.2 and big_alt > 0.8
    assert af_cv > 0.15 and af_alt < big_alt


def test_registry_and_tool_description():
    reg = all_measures()
    assert {"period", "ecg_hr", "ecg_rr_profile"} <= set(reg)
    with pytest.raises(ValueError):
        all_measures(PLUGINS + [PLUGINS[1]])


def test_ts_measure_ecg_via_mcp(tmp_path):
    import asyncio

    from mcp import Client

    from tsmcp.core.store import SignalStore
    from tsmcp.server.app import build_server

    x, _ = synth_ecg(np.full(100, 0.75))
    p = tmp_path / "ecg.npy"
    np.save(p, x.astype(np.float32))

    async def go():
        async with Client(build_server(SignalStore(allow_roots=[tmp_path]))) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            sid = json.loads((await c.call_tool("ts_load", {"source": str(p), "fs": FS})).content[0].text)["id"]
            r = await c.call_tool(
                "ts_measure", {"id": sid, "what": ["ecg_hr", "ecg_rr_stats", "ecg_af_burden"], "window": [10, 60]}
            )
            return tools["ts_measure"].description, json.loads(r.content[0].text)

    desc, out = asyncio.run(go())
    assert "ecg_rr_profile" in desc and len(desc) < 2000
    assert out["measurements"][0]["value"] == pytest.approx(80, rel=0.02)
    assert out["measurements"][0]["ref"].endswith("10-60/ch0")
    burden = out["measurements"][2]
    for formula in ("pct_of_window = 100*af_s/window_s", "pct_of_readable = value = 100*af_s/readable_s"):
        assert formula in desc and formula in burden["method"]
    assert burden["value"] == burden["pct_of_readable"] == burden["pct_of_window"] == 0
    assert burden["af_s"] == burden["irregular_s"] == burden["unreadable_s"] == 0
    assert burden["window_s"] == burden["readable_s"] == 50
    assert burden["n_episodes"] == 0 and burden["unit"] == "%"
    assert burden["ref"].endswith("10-60/ch0")
