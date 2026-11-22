"""Inputs for the single-shot baselines: A (numbers in the prompt) and B (one strip image)."""

from __future__ import annotations

import base64
import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.core.dsp import minmax_envelope

A_VALUES = 2000  # ~8k tokens: 1000 blocks x (min, max)


def numeric_text(x: np.ndarray, fs: float, n_values: int = A_VALUES) -> str:
    """Appendix-B style: per-block min/max, scaled to integers in [-999, 999], absolute (not diff) for readability."""
    cols = n_values // 2
    ci, lo, hi = minmax_envelope(np.asarray(x, dtype=np.float64), cols)
    vmin, vmax = float(np.nanmin(lo)), float(np.nanmax(hi))
    mid, half = (vmax + vmin) / 2, max((vmax - vmin) / 2, 1e-12)
    q = lambda v: np.round((v - mid) / half * 999).astype(int)  # noqa: E731
    per = int(np.ceil(x.size / cols))
    pairs = ";".join(f"{a},{b}" for a, b in zip(q(lo), q(hi), strict=True))
    return (
        f"Signal data: the recording ({x.size} samples at {fs:g} Hz) is split into {len(lo)} consecutive blocks of "
        f"{per} samples ({per / fs:g} s each; block k covers [k*{per / fs:g}, (k+1)*{per / fs:g}) s). For each block the "
        f"minimum and maximum are given as 'min,max', blocks separated by ';'. Values are integers v; physical value = "
        f"{mid!r} + v * {half!r} / 999.\n{pairs}"
    )


def strip_png(x: np.ndarray, fs: float, width: int = 2576, rows: int | None = None) -> tuple[str, str]:
    """B: the whole record wrapped into rows (like a Holter strip), per-pixel envelope, max width. Returns (b64, note)."""
    dur = x.size / fs
    rows = rows or int(min(8, max(1, np.ceil(dur / 100))))
    row_s = dur / rows
    height = min(2576, 40 + 250 * rows)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, 1, squeeze=False)[:, 0]
    lo_all, hi_all = np.nanpercentile(x, [0.05, 99.95])
    pad = 0.1 * (hi_all - lo_all + 1e-12)
    for r, ax in enumerate(axes):
        a, b = int(r * row_s * fs), int(min(x.size, (r + 1) * row_s * fs))
        seg = np.asarray(x[a:b], dtype=np.float64)
        ci, lo, hi = minmax_envelope(seg, width - 150)
        tt = a / fs + ci / fs
        ax.fill_between(tt, lo, hi, color="#1f4e79", linewidth=0.5, step="mid")
        ax.set_xlim(r * row_s, (r + 1) * row_s)
        ax.set_ylim(min(lo_all, np.nanmin(lo)) - pad, max(hi_all, np.nanmax(hi)) + pad)
        ax.set_axisbelow(True)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.4)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("time [s]", fontsize=9)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    note = f"The image shows the whole recording wrapped into {rows} rows of {row_s:g} s each (read each row's own time axis)."
    return base64.b64encode(buf.getvalue()).decode(), note
