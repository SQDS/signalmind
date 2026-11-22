"""PNG rendering. Figure + FigureCanvasAgg under a lock (tools run concurrently; pyplot is not thread-safe).

Views:
- render:        stacked channels, shared time axis (per-pixel min/max envelope for long windows)
- render_strip:  one channel wrapped into rows, like a Holter strip — long windows stay readable in one image
- render_series: derived series (e.g. RR intervals, rolling RMS) — hours of data as one slow curve
All views can overlay events (vertical markers + short labels) so image and event list share one reference frame.
"""

from __future__ import annotations

import io
import math
import threading

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.core.detect import feature_series
from tsmcp.core.dsp import minmax_envelope
from tsmcp.core.errors import SignalError
from tsmcp.core.signal import Channel, slice_channel

WIDTH_PX = {"normal": 1200, "high": 2400}
ROW_PX = {"normal": 220, "high": 300}
MAX_ROWS = 8
MAX_EVENT_LABELS = 40
COLOR = "#1f4e79"
EVENT_COLOR = "#c0392b"
_LOCK = threading.Lock()


def _t_first(ch: Channel, t0: float) -> float:
    """Time of the first sample returned by slice_channel for this window."""
    return max(0, int(np.floor(t0 * ch.fs))) / ch.fs


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.4)
    ax.tick_params(labelsize=8)


def _wave(ax, ch: Channel, a: float, b: float, cols: int) -> None:
    x = slice_channel(ch, a, b)
    if x.size == 0:
        return
    if x.size > 2 * cols:
        ci, lo, hi = minmax_envelope(x, cols)
        ax.fill_between(_t_first(ch, a) + ci / ch.fs, lo, hi, color=COLOR, linewidth=0.6, step="mid")
    else:
        ax.plot(_t_first(ch, a) + np.arange(x.size) / ch.fs, x, color=COLOR, linewidth=0.8)


def _events(ax, events: list[dict] | None, a: float, b: float, label: bool = True) -> None:
    if not events:
        return
    shown = [e for e in events if a <= e["t"] < b][:MAX_EVENT_LABELS]
    for e in shown:
        ax.axvline(e["t"], color=EVENT_COLOR, linewidth=0.8, alpha=0.8)
        if e.get("dur", 0) > 0:
            ax.axvspan(e["t"], min(b, e["t"] + e["dur"]), color=EVENT_COLOR, alpha=0.08)
        if label:
            ax.text(e["t"], 1.0, f" {e.get('label', e['type'])}", transform=ax.get_xaxis_transform(),
                    fontsize=7, color=EVENT_COLOR, va="top", ha="left", clip_on=True)  # fmt: skip


def _png(fig) -> bytes:
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def render(chans: list[Channel], t0: float, t1: float, detail: str = "normal", title: str | None = None,
           events: list[dict] | None = None) -> bytes:  # fmt: skip
    chans = chans[:MAX_ROWS]
    w = WIDTH_PX.get(detail, 1200)
    h = min(2576, 60 + ROW_PX.get(detail, 220) * len(chans))
    with _LOCK:
        fig = Figure(figsize=(w / 100, h / 100), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(chans), 1, sharex=True, squeeze=False)[:, 0]
        for ax, ch in zip(axes, chans, strict=True):
            _wave(ax, ch, t0, t1, w - 120)
            _events(ax, events, t0, t1, label=ax is axes[0])
            ax.set_ylabel(f"{ch.name} [{ch.unit}]", fontsize=9)
            _style(ax)
        axes[-1].set_xlabel("time [s]", fontsize=9)
        axes[-1].set_xlim(t0, t1)
        if title:
            axes[0].set_title(title, fontsize=9, loc="left")
        return _png(fig)


STRIP_MAX_ROWS = 6
STRIP_ROW_PX = {"normal": 110, "high": 160}
STRIP_WIDTH_PX = {"normal": 1200, "high": 1600}


def strip_rows(t0: float, t1: float, rows: int | None) -> int:
    n = int(rows) if rows else math.ceil((t1 - t0) / 100)
    return int(min(STRIP_MAX_ROWS, max(2, n)))


def render_strip(ch: Channel, t0: float, t1: float, detail: str = "normal", title: str | None = None,
                 events: list[dict] | None = None, rows: int | None = None) -> bytes:  # fmt: skip
    n = strip_rows(t0, t1, rows)
    row_s = (t1 - t0) / n
    w = STRIP_WIDTH_PX.get(detail, 1200)
    h = 50 + STRIP_ROW_PX.get(detail, 110) * n  # 6 rows at normal detail: 1200 x 710 px, ~1.1k image tokens
    with _LOCK:
        fig = Figure(figsize=(w / 100, h / 100), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(n, 1, squeeze=False)[:, 0]
        x_all = slice_channel(ch, t0, t1)
        lo_all, hi_all = np.nanpercentile(x_all, [0.05, 99.95]) if x_all.size else (0, 1)
        pad = 0.1 * (hi_all - lo_all + 1e-12)
        for r, ax in enumerate(axes):
            a, b = t0 + r * row_s, t0 + (r + 1) * row_s
            _wave(ax, ch, a, b, w - 100)
            _events(ax, events, a, b)
            ax.set_xlim(a, b)
            ax.set_ylim(lo_all - pad, hi_all + pad)
            _style(ax)
        axes[0].set_title(
            (title + "  " if title else "") + f"{ch.name} [{ch.unit}], {n} rows x {row_s:g} s", fontsize=9, loc="left"
        )
        axes[-1].set_xlabel("time [s] (each row has its own time axis)", fontsize=9)
        return _png(fig)


def _binned_envelope(t: np.ndarray, y: np.ndarray, t0: float, t1: float, cols: int):
    """Per-pixel-column 10th/90th percentile of an irregularly sampled series (t sorted). Percentiles, not
    min/max: a few outliers per column must not make a regular rhythm look scattered."""
    edges = np.linspace(t0, t1, cols + 1)
    idx = np.searchsorted(t, edges)
    lo, hi, tc = [], [], []
    for c in range(cols):
        a, b = idx[c], idx[c + 1]
        if b > a:
            seg = y[a:b]
            lo.append(np.nanpercentile(seg, 10))
            hi.append(np.nanpercentile(seg, 90))
            tc.append((edges[c] + edges[c + 1]) / 2)
    return np.array(tc), np.array(lo), np.array(hi)


def render_series(series: list[dict], t0: float, t1: float, detail: str = "normal", title: str | None = None,
                  events: list[dict] | None = None) -> bytes:  # fmt: skip
    """One panel per derived series. Dense series: per-column p10-p90 band plus every point (faint)."""
    w = WIDTH_PX.get(detail, 1200)
    h = min(2576, 60 + ROW_PX.get(detail, 220) * len(series))
    with _LOCK:
        fig = Figure(figsize=(w / 100, h / 100), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(series), 1, sharex=True, squeeze=False)[:, 0]
        for ax, s in zip(axes, series, strict=True):
            t, y = np.asarray(s["t"], float), np.asarray(s["y"], float)
            m = (t >= t0) & (t < t1) & np.isfinite(y)
            t, y = t[m], y[m]
            cols = w - 120
            if t.size > 2 * cols:
                tc, lo, hi = _binned_envelope(t, y, t0, t1, cols)
                ax.fill_between(tc, lo, hi, color=COLOR, alpha=0.35, linewidth=0, step="mid")
                ax.plot(t, y, ",", color=COLOR, alpha=0.25)
            elif t.size:
                ax.plot(t, y, "o-", color=COLOR, markersize=2, linewidth=0.6)
            if t.size:
                lo_q, hi_q = np.percentile(y, [0.5, 99.5])
                pad = 0.1 * (hi_q - lo_q + 1e-12)
                ax.set_ylim(lo_q - pad, hi_q + pad)  # robust limits: rare outliers must not flatten the curve
            _events(ax, events, t0, t1, label=ax is axes[0])
            ax.set_ylabel(f"{s['label']} [{s['unit']}]", fontsize=8)
            _style(ax)
        axes[-1].set_xlabel("time [s]", fontsize=9)
        axes[-1].set_xlim(t0, t1)
        if title:
            axes[0].set_title(title, fontsize=9, loc="left")
        return _png(fig)


def render_snippets(ch: Channel | list[Channel], events: list[dict], top_n: int = 4) -> bytes:
    """One score-ranked panel per candidate, at most four, with two window lengths of context on each side.

    Point events use their resolution (or one second) as a centred display window. A channel list lets events
    from a multi-channel search share one image without drawing a candidate against the wrong waveform.
    """
    chans = [ch] if isinstance(ch, Channel) else ch
    by_name = {c.name: c for c in chans}
    selected = sorted(events, key=lambda e: -e.get("score", 0))[: max(1, min(4, top_n))]
    if not selected or not chans:
        raise SignalError("snippets need at least one event and channel")
    with _LOCK:
        fig = Figure(figsize=(12, (40 + 160 * len(selected)) / 100), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(selected), 1, squeeze=False)[:, 0]
        for rank, (ax, event) in enumerate(zip(axes, selected, strict=True), 1):
            c = by_name[event["ch"]] if "ch" in event else chans[0]
            start, duration = event["t"], event.get("dur", 0)
            if duration <= 0:
                duration = max(1 / c.fs, event.get("resolution_s", 1.0))
                start -= duration / 2
            a, b = max(0, start - 2 * duration), min(c.duration, start + 3 * duration)
            _wave(ax, c, a, b, 1080)
            ax.axvspan(max(a, start), min(b, start + duration), color=EVENT_COLOR, alpha=0.15)
            ax.axvline(event["t"], color=EVENT_COLOR, linewidth=0.7)
            ax.set_xlim(a, b)
            ax.set_title(
                f"#{rank}  {event['type']}  t={event['t']:.6g} s  score={event.get('score', 0):.3g}",
                fontsize=9,
                loc="left",
            )
            ax.set_ylabel(f"{c.name} [{c.unit}]", fontsize=8)
            _style(ax)
        axes[-1].set_xlabel("time [s] (each panel has its own time axis)", fontsize=9)
        return _png(fig)


def render_verify(ch: Channel, t0: float, t1: float, detail: str = "normal", title: str | None = None,
                  events: list[dict] | None = None) -> bytes:  # fmt: skip
    """Focused waveform plus frequency and RMS in +/- ten focus-window lengths of context (1200 x 860).

    Feature bins are one focus window long (at least eight samples). RMS means standard deviation, consistently
    with the generic series view. The focused waveform retains per-pixel extrema when it exceeds image resolution.
    """
    t0, t1 = max(0.0, t0), min(ch.duration, t1)
    if t1 <= t0:
        raise SignalError(f"verify window is outside channel {ch.name!r} or empty")
    width = t1 - t0
    a, b = max(0.0, t0 - 10 * width), min(ch.duration, t1 + 10 * width)
    win_samples = max(8, int(width * ch.fs))
    win = win_samples / ch.fs
    starts, features = feature_series(slice_channel(ch, a, b), ch.fs, win)
    times = _t_first(ch, a) + starts + win / 2
    with _LOCK:
        fig = Figure(figsize=(12, 8.6), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 1)
        _wave(axes[0], ch, t0, t1, 1080)
        _events(axes[0], events, t0, t1)
        axes[0].set_xlim(t0, t1)
        axes[0].set_ylabel(f"{ch.name} [{ch.unit}]", fontsize=9)
        axes[0].set_title((title + " — " if title else "") + "verify: focused waveform", fontsize=10, loc="left")
        axes[0].set_xlabel("time [s] — focus", fontsize=9)
        for ax, name, label in zip(
            axes[1:], ("freq", "rms"), ("frequency [Hz]", f"RMS (std) [{ch.unit}]"), strict=True
        ):
            if starts.size:
                ax.plot(times, features[name], "o-", color=COLOR, markersize=3, linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "Insufficient samples for derived series", transform=ax.transAxes, ha="center")
            ax.set_xlim(a, b)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"Context: [{a:g}, {b:g}) s; {win:g} s feature windows", fontsize=9, loc="left")
        for ax in axes:
            ax.axvspan(t0, t1, color=EVENT_COLOR, alpha=0.12)
            _style(ax)
        axes[-1].set_xlabel("time [s] — context; shaded = focus window", fontsize=9)
        return _png(fig)
