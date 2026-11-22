"""Render consecutive 60-second pages on the existing patient gallery's acquisition clock."""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

TASKS = ("ECG", "PPG", "ABP")


def clock(seconds):
    seconds = round(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


@lru_cache(maxsize=4)
def load_block(folder, block_id):
    payload = (folder / "blocks" / f"{block_id:03d}.js").read_text()
    block = json.loads(payload.removeprefix("receiveBlock(").removesuffix(");\n"))
    if block["i"] != block_id:
        raise ValueError("Unexpected block id")
    for d in block["signals"].values():
        for name in ("raw", "output"):
            if d[name] is not None:
                d[name] = np.frombuffer(base64.b64decode(d[name]), dtype="<f8")
    return block["signals"]


def intersect(regions, base, left, right):
    for a, b, *_ in regions:
        if base + a < right and base + b > left:
            yield max(left, base + a) - left, min(right, base + b) - left


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return edges.reshape(-1, 2).tolist()


def draw(job):
    src, dst, no, fs, start, stop, blocks, total = job
    rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "path.simplify": False})
    fig = Figure(figsize=(13.2, 5.5), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    present = np.zeros(stop - start, bool)
    samples = 0
    for b, base in blocks:
        lo, hi = max(start, base), min(stop, base + b["n"])
        present[lo - start : hi - start] = True
        samples += hi - lo
    gaps = runs(~present)
    stats = {}
    for task, ax in zip(TASKS, axes, strict=True):
        bands = {key: [] for key in ("triggers", "ai", "stops", "modified", "rescan")}
        for b, base in blocks:
            ch = load_block(src, b["i"])[task]
            lo, hi = max(start - base, 0), min(stop - base, b["n"])
            grid = (np.arange(lo, hi) + base - start) / fs
            ax.plot(grid, ch["raw"][lo:hi], color="#22577a", lw=0.78, zorder=3)
            if ch["output"] is not None and list(intersect(ch["modified"], base, start, stop)):
                ax.plot(grid, ch["output"][lo:hi], color="#138b74", lw=0.85, ls="--", zorder=4)
            for key in bands:
                regs = [e["span"] for e in ch["ai"]] if key == "ai" else ch[key]
                bands[key].extend(intersect(regs, base, start, stop))
        for lft, rgt in gaps:
            ax.axvspan(lft / fs, rgt / fs, color="#ebeff2", lw=0, zorder=0)
        for lft, rgt in bands["ai"]:
            ax.axvspan(lft / fs, rgt / fs, color="#f8cb59", alpha=0.45, lw=0, zorder=1)
            ax.plot((lft + rgt) / (2 * fs), 0.97, transform=ax.get_xaxis_transform(), marker="v", color="#c88e00", ms=5)
        for key, y, color in [
            ("triggers", 0.055, "#ce7828"),
            ("stops", 0.105, "#656d77"),
            ("modified", 0.155, "#804cad"),
            ("rescan", 0.205, "#b33859"),
        ]:
            if not bands[key]:
                continue
            segs = [[(lft / fs, y), (rgt / fs, y)] for lft, rgt in bands[key]]
            trans = ax.get_xaxis_transform()
            ax.add_collection(LineCollection(segs, transform=trans, colors=color, linewidths=1.35), autolim=False)
            mids = [(lft + rgt) / (2 * fs) for lft, rgt in bands[key]]
            ax.plot(mids, [y] * len(mids), ls="", marker=".", ms=2, color=color, transform=trans)
        stats[task] = {k: len(v) for k, v in bands.items()}
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(task + ("\nmmHg" if task == "ABP" else "\n源单位"), fontsize=10)
        ax.margins(y=0.14)
        ax.grid(axis="x", color="#dfe7ec", alpha=0.6, linewidth=0.5)
        if not blocks:
            ax.set_yticks([])
            ax.text(0.5, 0.52, "此时段没有采集数据", ha="center", transform=ax.transAxes, color="#667986")
    duration = (stop - start) / fs
    axes[-1].set(xlim=(0, duration), xticks=np.arange(0, duration + 0.001, 5), xlabel="本页内时间（秒）")
    fig.suptitle(f"{clock(start / fs)} – {clock(stop / fs)}", ha="left", x=0.077, y=0.982, fontsize=18)
    fig.text(
        0.975,
        0.94,
        f"第 {no} / {total} 页   ·   实际覆盖 {present.sum() / fs:g} / {duration:g} 秒",
        ha="right",
        fontsize=10,
    )
    fig.text(
        0.077,
        0.893,
        "蓝线  原始波形     灰底  采集缺口     橙线  初筛触发     黄区/三角  AI 圈选     灰线  任务停用     紫线  数值修改     红线  回扫命中",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.077, right=0.985, top=0.845, bottom=0.11, hspace=0.22)
    image = f"minutes/minute-{no:03d}.png"
    fig.savefig(dst / image, dpi=120)
    return dict(
        page=no,
        start_s=start / fs,
        end_s=stop / fs,
        covered_samples=int(present.sum()),
        stored_samples=samples,
        missing_ranges_s=[[a / fs, b / fs] for a, b in gaps],
        counts=stats,
        source_blocks=[b["i"] for b, _ in blocks],
        image=image,
    )


def build(src, dst):
    src, dst = Path(src), Path(dst)
    m = json.loads((src / "summary.json").read_text())
    rates = {b["fs"] for b in m["blocks"]}
    if len(rates) != 1:
        raise ValueError("Expected a uniform sampling rate across this synchronized three-channel gallery")
    fs = rates.pop()
    tick = int(round(60 * fs))
    entries = [(b, round(b["start"] * fs)) for b in m["blocks"]]
    if any(abs(b["start"] * fs - base) > 1e-4 for b, base in entries):
        raise ValueError("Blocks do not share the patient's sample-time grid")
    n = max(base + b["n"] for b, base in entries)
    pages = math.ceil(n / tick)
    (dst / "minutes").mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, start in enumerate(range(0, n, tick), 1):
        stop = min(n, start + tick)
        blocks = [(b, base) for b, base in entries if base < stop and start < base + b["n"]]
        jobs.append((src, dst, index, fs, start, stop, blocks, pages))
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(draw, jobs, chunksize=4))
    assert sum(r["stored_samples"] for r in rows) == sum(b["n"] for b in m["blocks"])
    ai_pages = [r["page"] for r in rows if any(x["ai"] for x in r["counts"].values())]
    result = dict(
        record=m["record"],
        fs=fs,
        elapsed_s=n / fs,
        stored_s=m["stored_s"],
        pages=pages,
        ai_pages=ai_pages,
        empty=sum(not r["covered_samples"] for r in rows),
        windows=rows,
    )
    (dst / "windows.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (dst / "pages.js").write_text(
        "window.preview=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + ";"
    )
    shutil.copyfile(Path(__file__).with_name("minutes.html"), dst / "index.html")
    if (src / "reviewed-event.png").exists():
        shutil.copyfile(src / "reviewed-event.png", dst / "reviewed-event.png")
    zip_path = dst.parent / (dst.name + ".zip")
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=2) as z:
        for path in sorted(dst.rglob("*")):
            if path.is_file():
                z.write(path, str(path.relative_to(dst.parent)))
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args.gallery, args.out)
