"""Detectors, derived series and views (the 'reading' layer)."""

import asyncio
import json

import numpy as np
import pytest
from mcp import Client

from tsmcp.core.detect import changes, clip, discords, flat, spikes
from tsmcp.core.signal import Channel
from tsmcp.core.store import SignalStore
from tsmcp.core.views.render import render_series, render_snippets, render_strip, render_verify
from tsmcp.plugins.ecg import classify_rr, ecg_irregular, ecg_pause, series_rr, view_ecg_paper
from tsmcp.server.app import build_server

FS = 250.0


def sine(dur=600, f=1.2, noise=0.05, seed=0):
    t = np.arange(int(dur * FS)) / FS
    return np.sin(2 * np.pi * f * t) + noise * np.random.default_rng(seed).standard_normal(t.size), t


def test_spike_flat_clip():
    x, _ = sine(120)
    x[int(30 * FS)] += 8
    x[int(80 * FS) : int(83 * FS)] = x[int(80 * FS)]
    y, _ = sine(120)
    y[int(50 * FS) : int(52 * FS)] = y.max()
    assert [round(e["t"]) for e in spikes(x, FS)] == [30]
    assert "verify" in spikes(x, FS)[0]["caveat"]
    assert [round(e["t"]) for e in flat(x, FS)] == [80]
    assert any(abs(e["t"] - 50) < 0.1 for e in clip(y, FS))


@pytest.mark.parametrize("kind", ["freq", "amp", "level"])
def test_change_found(kind):
    x, t = sine(600)
    m = t >= 250
    if kind == "freq":
        x = np.sin(2 * np.pi * np.cumsum(np.where(m, 0.8, 1.2)) / FS) + 0.05 * np.random.default_rng(1).standard_normal(
            t.size
        )
    elif kind == "amp":
        x[m] *= 2
    else:
        x[m] += 1.5
    ev = changes(x, FS, t0=1000.0)
    assert ev and abs(ev[0]["t"] - 1250) <= 2 * ev[0]["resolution_s"]
    assert "verify" in ev[0]["caveat"]


def test_slow_modulation_is_not_a_change():
    x, t = sine(600)
    x *= 1 + 0.3 * np.sin(2 * np.pi * 0.03 * t)  # respiratory-like AM: the prototype's false-positive case
    assert changes(x, FS) == []


def test_classify_rr():
    rng = np.random.default_rng(0)
    assert classify_rr(np.full(16, 0.8) + 0.01 * rng.standard_normal(16)) == "regular"
    assert classify_rr(np.tile([0.5, 1.1], 8)) == "alternating"
    assert classify_rr(np.tile([0.8, 0.8, 0.45], 6)[:16] + 0.005 * rng.standard_normal(16)) in (
        "period3",
        "alternating",
    )
    assert classify_rr(rng.uniform(0.45, 1.1, 16)) == "irregular"


def _ecg(rr, fs=360.0):
    tb = 1.0 + np.concatenate([[0], np.cumsum(rr)])
    t = np.arange(int((tb[-1] + 1) * fs)) / fs
    x = np.zeros_like(t)
    for b in tb:
        x += np.exp(-(((t - b) / 0.012) ** 2)) - 0.25 * np.exp(-(((t - b - 0.03) / 0.015) ** 2))
    return x + 0.01 * np.random.default_rng(2).standard_normal(t.size), tb


def test_ecg_irregular_episode_and_pause():
    rng = np.random.default_rng(3)
    rr = np.concatenate([np.full(60, 0.8), rng.uniform(0.45, 1.1, 80), np.full(60, 0.8)])
    x, tb = _ecg(rr)
    ep = ecg_irregular(x, 360.0, "mV")
    start, end = tb[60], tb[140]
    assert len(ep) == 1 and abs(ep[0]["t"] - start) < 12 and abs(ep[0]["t"] + ep[0]["dur"] - end) < 12
    x2, tb2 = _ecg(np.concatenate([np.full(20, 0.8), [2.6], np.full(20, 0.8)]))
    p = ecg_pause(x2, 360.0, "mV")
    assert len(p) == 1 and abs(p[0]["dur"] - 2.6) < 0.05


def test_views_render():
    x, _ = _ecg(np.full(200, 0.8))
    from tsmcp.core.signal import Channel

    ch = Channel("ECG", 360.0, x.astype(np.float32), "mV")
    for png in (render_strip(ch, 0, ch.duration), render_series([series_rr(x, 360.0, "mV")], 0, ch.duration),
                view_ecg_paper(ch, 10, 40)):  # fmt: skip
        assert png[:4] == b"\x89PNG"
    with pytest.raises(Exception, match="at most"):
        view_ecg_paper(ch, 0, 120)


def test_find_events_and_views_via_mcp(tmp_path):
    x, _ = sine(600)
    x[int(300 * FS)] += 8
    p = tmp_path / "s.npy"
    np.save(p, x.astype(np.float32))

    async def go(images):
        async with Client(build_server(SignalStore(allow_roots=[tmp_path]), images=images)) as c:
            names = {t.name: t for t in (await c.list_tools()).tools}
            sid = json.loads((await c.call_tool("ts_load", {"source": str(p), "fs": FS})).content[0].text)["id"]
            ev = await c.call_tool("ts_find_events", {"id": sid, "types": ["spike", "flat"]})
            strip = await c.call_tool("ts_zoom", {"id": sid, "window": [0, 600], "view": "strip"})
            ser = await c.call_tool("ts_zoom", {"id": sid, "window": [0, 600], "view": "freq"})
            bad = await c.call_tool("ts_zoom", {"id": sid, "window": [0, 600], "view": "nope"})
            return names, json.loads(ev.content[0].text), strip, ser, bad

    names, ev, strip, ser, bad = asyncio.run(go(True))
    assert all(len(t.description) < 2000 for t in names.values())
    assert (
        ev["counts"] == {"spike": 1}
        and abs(ev["events"][0]["t"] - 300) < 0.05
        and ev["events"][0]["ref"].endswith("@300")
    )
    assert [b.type for b in strip.content] == ["text", "image"] and [b.type for b in ser.content] == ["text", "image"]
    assert bad.is_error and "available" in bad.content[0].text
    _, _, strip_t, ser_t, _ = asyncio.run(go(False))
    assert [b.type for b in ser_t.content] == ["text"]
    binned = json.loads(ser_t.content[0].text)["binned"]
    assert len(binned["median"]) == 120 and abs(binned["median"][5] - 1.2) < 0.1


def test_lead_off_is_noisy_not_af():
    from tsmcp.plugins.ecg import ecg_hr, ecg_noisy

    x, tb = _ecg(np.full(150, 0.8))
    fs = 360.0
    a, b = int(40 * fs), int(80 * fs)
    x[a:b] = 0.01 * np.random.default_rng(5).standard_normal(b - a)  # lead off: only small noise
    assert ecg_irregular(x, fs, "mV") == []
    noisy = ecg_noisy(x, fs, "mV")
    assert noisy and any(e["t"] < 60 < e["t"] + e["dur"] for e in noisy)
    assert ecg_hr(x, fs, "mV")["value"] == pytest.approx(75, rel=0.02)  # HR from readable beats only


def test_every_advertised_name_is_registered():
    """Tool descriptions are generated from plugin summaries; a name there that is not registered makes the model
    call something that does not exist (happened once with ecg_noisy)."""
    import re

    from tsmcp.plugins import PLUGINS, all_detectors, all_measures, all_series, all_views

    known = set(all_detectors()) | set(all_measures()) | set(all_series()) | set(all_views()) | {"wave", "strip"}
    for p in PLUGINS:
        for text in (p.summary, p.detectors_summary, p.views_summary):
            for name in re.findall(
                r"\b(?:ecg_[a-z0-9_]+|spike|flat|clip|change|level|rms|freq|period|dominant_freq|amplitude_pp|mean)\b",
                text,
            ):
                assert name in known, f"{p.name} advertises {name!r} but it is not registered"


def test_discord_finds_the_odd_cycle():
    from tsmcp.core.detect import discords

    fs = 100.0
    t = np.arange(int(300 * fs)) / fs
    x = (
        np.sin(2 * np.pi * 1.0 * t)
        + 0.3 * np.sin(2 * np.pi * 3.0 * t)
        + 0.02 * np.random.default_rng(0).standard_normal(t.size)
    )
    a = int(200.0 * fs)
    x[a : a + int(fs)] = -x[a : a + int(fs)]  # one inverted cycle: same amplitude, same energy, wrong shape
    ev = discords(x, fs, t0=1000.0)
    assert ev and abs(ev[0]["t"] + ev[0]["dur"] / 2 - 1200.5) < 1.5 and ev[0]["score"] > 2


def test_strip_size_and_call_budget_note(tmp_path):
    from PIL import Image

    x, _ = sine(3600)
    p = tmp_path / "s.npy"
    np.save(p, x.astype(np.float32))

    async def go():
        async with Client(build_server(SignalStore(allow_roots=[tmp_path]))) as c:
            sid = json.loads((await c.call_tool("ts_load", {"source": str(p), "fs": FS})).content[0].text)["id"]
            strip = await c.call_tool("ts_zoom", {"id": sid, "window": [0, 3600], "view": "strip", "rows": 20})
            last = None
            for _ in range(20):
                last = await c.call_tool("ts_measure", {"id": sid, "what": ["rms"], "window": [0, 10]})
            return strip, json.loads(last.content[0].text)

    strip, last = asyncio.run(go())
    import base64
    import io

    w, h = Image.open(io.BytesIO(base64.b64decode(strip.content[1].data))).size
    assert w <= 1200 and h <= 720  # rows capped at 6
    assert "budget_note" in last


@pytest.mark.parametrize("unreadable", [False, True])
def test_ecg_af_burden_measure(unreadable):
    from tsmcp.plugins.ecg import ecg_af_burden

    rng = np.random.default_rng(7)
    rr = np.concatenate([np.full(100, 0.8), rng.uniform(0.45, 1.1, 100), np.full(100, 0.8)])
    x, tb = _ecg(rr)
    if unreadable:
        a, b = int(30 * 360), int(55 * 360)
        x[a:b] = 0.01 * np.random.default_rng(5).standard_normal(b - a)  # lead off in the regular stretch
    true = 100 * (tb[200] - tb[100]) / (x.size / 360.0)
    m = ecg_af_burden(x, 360.0, "mV")
    assert {
        "value",
        "unit",
        "method",
        "irregular_s",
        "readable_s",
        "n_episodes",
        "pct_of_window",
        "pct_of_readable",
        "af_s",
        "window_s",
        "unreadable_s",
    } <= m.keys()  # legacy and explicit-denominator fields
    assert abs(m["pct_of_window"] - true) < 6 and m["unit"] == "%"
    assert 0 < m["pct_of_window"] <= m["pct_of_readable"] == m["value"] <= 100
    assert m["pct_of_window"] == pytest.approx(100 * m["af_s"] / m["window_s"], abs=0.05)
    assert m["pct_of_readable"] == pytest.approx(100 * m["af_s"] / m["readable_s"], abs=0.05)
    assert m["window_s"] == pytest.approx(x.size / 360.0, abs=0.0005)
    assert m["readable_s"] + m["unreadable_s"] == pytest.approx(m["window_s"], abs=0.001)
    assert m["n_episodes"] > 0 and 0 < m["af_s"] <= m["readable_s"]
    if unreadable:
        assert m["unreadable_s"] > 0 and m["pct_of_window"] < m["pct_of_readable"]
    else:
        # the AF detector treats a few seconds at the window edges as unreadable even on clean signal
        assert m["unreadable_s"] <= 0.05 * m["window_s"]
        assert m["pct_of_readable"] - m["pct_of_window"] <= 0.05 * m["pct_of_readable"]


@pytest.mark.parametrize("periods", [0.3, 3.0])
def test_discord_anomaly_lengths(periods):
    fs = 100.0
    t = np.arange(30_000) / fs
    x = np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)
    x += 0.02 * np.random.default_rng(0).standard_normal(t.size)
    a, b = 20_000, 20_000 + int(periods * fs)
    local = np.arange(b - a) / fs
    x[a:b] = np.sin(2 * np.pi * (2 * local + local**2))
    events = discords(x, fs, t0=1000, k=3)
    assert any(e["t"] < 1000 + b / fs and e["t"] + e["dur"] > 1000 + a / fs for e in events)
    assert all("1 dominant period" in e["detail"] and "verify" in e["caveat"] for e in events)
    assert [e["score"] for e in events] == sorted((e["score"] for e in events), reverse=True)
    for i, e in enumerate(events):
        for other in events[:i]:
            assert abs(e["t"] + e["dur"] / 2 - other["t"] - other["dur"] / 2) > max(e["dur"], other["dur"])


@pytest.mark.parametrize("x", [np.array([]), np.ones(1000), np.full(1000, np.nan), np.arange(16.0)])
def test_discord_uninformative_signals(x):
    assert discords(x, 100) == []


def test_discord_compares_both_anomaly_lengths():
    fs = 100.0
    t = np.arange(30_000) / fs
    x = np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)
    x += 0.02 * np.random.default_rng(0).standard_normal(t.size)
    for start, duration in ((100, 0.3), (200, 3.0)):
        a, b = round(start * fs), round((start + duration) * fs)
        local = np.arange(b - a) / fs
        x[a:b] = np.sin(2 * np.pi * (2 * local + local**2))
    events = discords(x, fs, k=3)
    for start, end in ((100, 100.3), (200, 203)):
        assert any(e["t"] < end and e["t"] + e["dur"] > start for e in events)
    assert len({e["dur"] for e in events}) == 1


def test_discord_original_aperiodic_window(monkeypatch):
    import tsmcp.core.detect as detect

    # Original fallback is n/200 clipped to [16, 2000], then to the record's window bound.
    x = np.random.default_rng(7).normal(size=2000)
    x[1200:1220] += np.linspace(-5, 5, 20)
    monkeypatch.setattr(detect, "_period_samples", lambda *args: None)
    events = discords(x, 10, k=5)
    assert events
    for e in events:
        m = round(e["dur"] * 10)
        assert m == 16
        assert e["detail"] == "window = 1 dominant period (16 samples)"


def test_snippets_rank_context_channels_and_dimensions(monkeypatch):
    import io

    from PIL import Image

    import tsmcp.core.views.render as rendering

    channels = [Channel("one", 100, np.zeros(10_000)), Channel("two", 100, np.ones(10_000))]
    events = [
        {"type": "discord", "t": 20.0 + i * 10, "dur": 2.0, "score": i + 1, "ch": channels[i % 2].name}
        for i in range(5)
    ]
    captured = []
    original = rendering._png

    def capture(fig):
        captured.append(fig)
        return original(fig)

    monkeypatch.setattr(rendering, "_png", capture)
    png = render_snippets(channels, events, top_n=20)
    w, h = Image.open(io.BytesIO(png)).size
    assert w <= 1200 and h <= 700
    assert len(captured[0].axes) == 4
    for rank, ax in enumerate(captured[0].axes, 1):
        event = events[-rank]
        assert ax.get_title(loc="left").startswith(f"#{rank}  discord")
        assert f"score={event['score']}" in ax.get_title(loc="left")
        assert ax.get_xlim() == pytest.approx((event["t"] - 4, event["t"] + 6))
        assert event["ch"] in ax.get_ylabel()
        assert ax.patches  # shaded candidate span


@pytest.mark.parametrize("window", [(0, 2), (10.123, 12.123), (58, 60), (0, 0.001)])
def test_verify_focus_context_and_dimensions(window, monkeypatch):
    import io

    from PIL import Image

    import tsmcp.core.views.render as rendering

    x, _ = sine(60, noise=0)
    ch = Channel("sensor", FS, x, "mV")
    captured = []
    original = rendering._png

    def capture(fig):
        captured.append(fig)
        return original(fig)

    monkeypatch.setattr(rendering, "_png", capture)
    png = render_verify(ch, *window, detail="high")
    w, h = Image.open(io.BytesIO(png)).size
    assert w <= 1200 and h <= 900
    wave, freq, rms = captured[0].axes
    a, b = window
    context = (max(0, a - 10 * (b - a)), min(60, b + 10 * (b - a)))
    assert wave.get_xlim() == pytest.approx(window)
    assert freq.get_xlim() == rms.get_xlim() == pytest.approx(context)
    assert all(ax.patches for ax in (wave, freq, rms))
    if b - a >= 2:
        assert np.median(freq.lines[0].get_ydata()) == pytest.approx(1.2, rel=0.02)
        assert np.median(rms.lines[0].get_ydata()) == pytest.approx(1 / np.sqrt(2), rel=0.06)
        assert min(freq.lines[0].get_xdata()) >= context[0]
    else:
        assert "Insufficient samples" in freq.texts[0].get_text()


@pytest.mark.parametrize("images", [True, False])
def test_snippets_verify_and_caveats_via_mcp(tmp_path, images):
    x, _ = sine(120)
    x[int(30 * FS)] += 8
    path = tmp_path / "snippets.npy"
    np.save(path, x.astype(np.float32))

    async def go():
        async with Client(build_server(SignalStore(allow_roots=[tmp_path]), images=images)) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert tools == {"ts_load", "ts_describe", "ts_zoom", "ts_measure", "ts_find_events", "ts_forget"}
            sid = json.loads((await client.call_tool("ts_load", {"source": str(path), "fs": FS})).content[0].text)["id"]
            params = {"id": sid, "types": ["spike"]}
            plain = await client.call_tool("ts_find_events", params)
            snippets = await client.call_tool("ts_find_events", {**params, "snippets": True})
            empty = await client.call_tool("ts_find_events", {"id": sid, "types": ["flat"], "snippets": True})
            verify = await client.call_tool("ts_zoom", {"id": sid, "window": [29, 31], "view": "verify"})
            return plain, snippets, empty, verify

    plain, snippets, empty, verify = asyncio.run(go())
    assert [b.type for b in plain.content] == ["text"]
    assert [b.type for b in snippets.content] == (["text", "image"] if images else ["text"])
    assert plain.content[0].text == snippets.content[0].text
    assert "verify" in json.loads(plain.content[0].text)["events"][0]["caveat"]
    assert [b.type for b in empty.content] == ["text"]
    if images:
        assert not verify.is_error and [b.type for b in verify.content] == ["text", "image"]
    else:
        assert verify.is_error and "images are disabled" in verify.content[0].text
