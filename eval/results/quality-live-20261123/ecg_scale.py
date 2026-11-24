"""Render the archived 30 s annotations with a fixed heartbeat-scale axis for each patient."""

from collections import defaultdict

import matplotlib as mpl
import numpy as np
from annotation_view import OUT, PEOPLE, PREVIEWS, ROOT, TAG, compact_time, final_measurements
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from tsmcp.cleaning.common import array_hash, read
from tsmcp.cleaning.dataset_evidence import acquisition
from tsmcp.cleaning.engine import load_arrays
from tsmcp.processing import intervals

ROWS = ("ECG", "full", "PPG", "ABP")
HEIGHTS = (2.15, 0.82, 0.95, 0.95)


def scales(cases):
    bounds = defaultdict(dict)
    extrema = defaultdict(list)
    for c in cases:
        rid, ch = c["record_id"], c["task"]
        with np.load(ROOT / "public" / c["id"] / "input.npz") as z:
            y = z["values"]
            low, high = bounds[rid].get(ch, (np.inf, -np.inf))
            bounds[rid][ch] = (min(low, np.nanmin(y)), max(high, np.nanmax(y)))
            if ch == "ECG":
                width = round(2 * c["fs"])
                chunks = y[: (len(y) // width) * width].reshape(-1, width)
                chunks = chunks[np.isfinite(chunks).all(axis=1)]
                if len(chunks):
                    extrema[rid].append(np.column_stack((chunks.min(axis=1), chunks.max(axis=1))))
    full = np.ceil(1.05 * max(abs(v) for channels in bounds.values() for v in channels["ECG"]))
    for rid, channels in bounds.items():
        # Two-second extrema preserve narrow R peaks. Occasional high-amplitude episodes do not set the detail axis.
        data = np.vstack(extrema[rid])
        low, high = np.percentile(data[:, 0], 10), np.percentile(data[:, 1], 90)
        span = high - low
        step = 0.5 * 10 ** np.floor(np.log10(span))
        channels["ECG"] = (np.floor((low - 0.08 * span) / step) * step, np.ceil((high + 0.15 * span) / step) * step)
        channels["full"] = (-full, full)
        for ch in ("PPG", "ABP"):
            low, high = channels[ch]
            pad = 0.05 * (high - low)
            channels[ch] = (low - pad, high + pad)
    return bounds


def load_block(cases, wm):
    block = {}
    for c in cases:
        cid = c["id"]
        doc = wm[cid]
        artifact = read(ROOT / "artifacts" / doc["artifact_id"] / "artifact.json")
        raw, t, y, valid, estimated = load_arrays(ROOT, c, artifact)
        assert array_hash(y, t, valid, estimated) == doc["content_hash"]
        assert np.array_equal(raw, y, equal_nan=True) and not estimated.any()
        block[c["task"]] = (c, t, y, valid)
    return block


def draw_column(axes, block, start, bounds, bycase, title, *, preview):
    samples = {}
    for row, ax in zip(ROWS, axes, strict=True):
        ch = "ECG" if row == "full" else row
        c, t, y, valid = block[ch]
        a, b = t.searchsorted([start - 1e-7, start + 30 - 1e-7])
        u = t[a:b] - start
        ax.plot(u, y[a:b], color="#244c60", lw=0.85 if row != "full" else 0.65)
        if row in ("full", "ECG") or not preview:
            ax.set_ylim(*bounds[row])
        else:
            ax.margins(y=0.24)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2 if row == "full" else 5, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(axis="y", color="#d6e1e4", alpha=0.6, lw=0.48)
        for lo, hi in intervals(~valid[a:b]):
            ax.axvspan(
                u[lo], u[hi - 1] + 1 / c["fs"], facecolor="#e3e7e9", hatch="////", edgecolor="#647681", lw=0.4, zorder=0
            )
        if row != "full":
            samples[c["id"]] = int(b - a)
            spans = defaultdict(list)
            for left, right, state in bycase[c["id"]]:
                if left < start + 30 and start < right:
                    lpos = max(0, left - start)
                    rpos = min(30, right - start)
                    spans[state].append((lpos, rpos - lpos))
            for state, bands in spans.items():
                ax.broken_barh(
                    bands,
                    (1.008, 0.052),
                    transform=ax.get_xaxis_transform(),
                    facecolors=TAG[state][1],
                    edgecolors="none",
                    clip_on=False,
                    zorder=3,
                )
        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.tick_params(axis="both", labelsize=9, length=3)
        if row == "ECG":
            lo, hi = bounds["ECG"]
            ax.set_title(f"{title}　｜　心搏轴 {lo:g} 至 {hi:g}", loc="left", fontsize=10.5, pad=13)
            ax.set_ylabel("ECG 心搏放大\n原始单位")
        elif row == "full":
            ax.set_ylabel("ECG 全幅\n原始单位", fontsize=9)
        else:
            ax.set_ylabel(ch + "\n" + ("mmHg" if ch == "ABP" else "原始单位"))
        if row == "ABP":
            ax.set_xlabel("本窗时间（秒）", fontsize=10)
        else:
            ax.tick_params(axis="x", labelbottom=False)
    return samples


def layout(columns, *, preview):
    fig = Figure(figsize=(14, 8.7) if preview else (10.8, 7.8), layout="constrained")
    grid = fig.add_gridspec(5, columns, height_ratios=(*HEIGHTS, 0.24))
    axes = []
    for col in range(columns):
        panels = []
        for i, row in enumerate(ROWS):
            share = axes[0][i] if col and row != "ECG" else None
            panels.append(fig.add_subplot(grid[i, col], sharey=share))
        axes.append(panels)
    legend = fig.add_subplot(grid[-1, :])
    legend.axis("off")
    handles = [Patch(facecolor=clr, edgecolor="none", label="逐拍" + label) for label, clr in TAG.values()]
    handles.append(Patch(facecolor="#e3e7e9", edgecolor="#647681", hatch="////", label="波形停用"))
    legend.legend(handles=handles, loc="center", ncols=4, frameon=False, fontsize=10)
    return fig, axes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bycase, _, wm = final_measurements()
    cases = read(ROOT / "dataset_manifest.json")["cases"]
    index = {c["id"]: c for c in cases}
    blocks = defaultdict(list)
    for c in cases:
        blocks[acquisition(c)].append(c)
    bounds = scales(cases)
    fig, axes = layout(2, preview=True)
    fig.suptitle("DS 逐拍标注　·　ECG 上行放大心搏，下行显示完整幅值", fontsize=16, weight="bold", color="#193947")
    for col, (example, start, subtitle) in enumerate(PREVIEWS):
        c = index[example]
        rid = c["record_id"]
        block = load_block(blocks[acquisition(c)], wm)
        label = f"{PEOPLE[rid]}　{compact_time(start)}–{compact_time(start + 30)}　{subtitle}"
        samples = draw_column(axes[col], block, start, bounds[rid], bycase, label, preview=True)
        assert samples == {case["id"]: round(30 * case["fs"]) for case in blocks[acquisition(c)]}
        print(rid, "ECG detail", bounds[rid]["ECG"], "overview", bounds[rid]["full"], flush=True)
    fig.savefig(OUT / "ecg-zoom-30s.png", dpi=165)
    fig.clear()

    with mpl.rc_context({"font.family": ["DejaVu Sans", "Droid Sans Fallback"]}):
        for rid, person in PEOPLE.items():
            path = OUT / f"{rid}-ecg-zoom-30s.pdf"
            tmp = path.with_suffix(".tmp")
            totals = defaultdict(int)
            with PdfPages(tmp, metadata={"Title": person + "　ECG 心搏放大与全幅 · 30 秒图册"}) as pdf:
                for group in blocks.values():
                    if group[0]["record_id"] != rid:
                        continue
                    block = load_block(group, wm)
                    onset = min(t[0] for _, t, _, _ in block.values())
                    end = max(t[-1] + 1 / c["fs"] for c, t, _, _ in block.values())
                    for start in np.arange(onset, end - 1e-7, 30):
                        fig, axes = layout(1, preview=False)
                        label = f"{person}　源时间 {start:.3f}–{min(start + 30, end):.3f} 秒"
                        counts = draw_column(axes[0], block, start, bounds[rid], bycase, label, preview=False)
                        for key, n in counts.items():
                            totals[key] += n
                        pdf.savefig(fig)
                        fig.clear()
                pages = pdf.get_pagecount()
            assert totals == {c["id"]: c["n"] for c in cases if c["record_id"] == rid}
            recorded_pages = read(ROOT / "records" / rid / "report.json").get("atlas_pages")
            if recorded_pages is not None:
                assert pages == recorded_pages
            tmp.replace(path)
            print(person, "pages", pages, "all channel samples accounted for", flush=True)


if __name__ == "__main__":
    main()
