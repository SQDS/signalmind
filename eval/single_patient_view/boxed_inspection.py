"""Render frozen screening intervals as boxes on the original single-patient waveform."""

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tsmcp.cleaning.common import read, sha

TASKS = ("ECG", "PPG", "ABP")
STEP, SPECTRUM, SOURCE, REVIEW = "#bb4b14", "#73459a", "#555b63", "#087a71"
BASELINE = "#a03160"
PULSE = "#286ba3"


def color(kind):
    if kind == "abp_pulse_episode":
        return PULSE
    if kind == "historical_review":
        return REVIEW
    if kind == "ecg_baseline_shift":
        return BASELINE
    if kind.startswith("line_") or kind == "high_frequency":
        return SPECTRUM
    if kind in {"large_step", "ecg_excursion", "flat_run"}:
        return STEP
    return SOURCE


def boxed_ranges(rows):
    """Union touching/overlapping hits within one block and rule for display only."""
    result = []
    for kind, ranges in rows.items():
        spans = []
        for a, b, cid in sorted(ranges):
            if spans and a <= spans[-1][1]:
                spans[-1][1] = max(b, spans[-1][1])
                spans[-1][2].append(cid)
            else:
                spans.append([a, b, [cid]])
        result.extend((a, b, kind, ids) for a, b, ids in spans)
    return result


def render(workspace, inspection, out, detail_page, detail_second):
    out.mkdir(parents=True, exist_ok=True)
    meta = read(workspace / "gallery/summary.json")
    pages = read(workspace / "preview-60s/windows.json")
    cases = {c["id"]: c for c in read(workspace / "dataset_manifest.json")["cases"]}
    blocks = {b["i"]: b for b in meta["blocks"]}
    fs = pages["fs"]
    manifest = read(inspection / "candidate_manifest.json")
    if sha(inspection / "evidence.jsonl") != manifest["files"]["evidence.jsonl"]:
        raise ValueError("Inspection evidence changed")
    by_kind = defaultdict(lambda: defaultdict(list))
    all_ids = set()
    with (inspection / "evidence.jsonl").open() as stream:
        for row in map(json.loads, stream):
            cid = row["range"]["block_id"]
            if row["observed_version_id"] != "raw-" + cases[cid]["input_sha256"]:
                raise ValueError("Original and inspection versions differ")
            by_kind[cid][row["kind"]].append((row["start"], row["end"], row["candidate_id"]))
            all_ids.add(row["candidate_id"])
    boxes = {cid: boxed_ranges(groups) for cid, groups in by_kind.items()}

    @lru_cache(maxsize=9)
    def values(cid):
        path = workspace / "public" / cid / "input.npz"
        if sha(path) != cases[cid]["input_sha256"]:
            raise ValueError("Waveform hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            return data["values"].reshape(-1)

    def draw(ax, page, task, left, right, *, reviewed=(), show_spectrum=True):
        origin = round(page["start_s"] * fs)
        visible = set()
        ranges = []
        for index in page["source_blocks"]:
            b = blocks[index]
            cid = b["ids"][task]
            base = round(b["start"] * fs) - origin
            start = max(0, left - base)
            stop = min(b["n"], right - base)
            if start >= stop:
                continue
            x = values(cid)
            ax.plot((base + np.arange(start, stop)) / fs, x[start:stop], lw=0.9, color="#224f6a", zorder=3)
            layers = [*boxes.get(cid, ()), *reviewed.get(cid, ())] if reviewed else boxes.get(cid, ())
            for a, z, kind, ids in layers:
                if a >= stop or z <= start or (color(kind) == SPECTRUM and not show_spectrum):
                    continue
                for uid in ids:
                    visible.add(uid)
                region = x[max(start, a) : min(stop, z)]
                supported = region[np.isfinite(region)]
                ranges.append((base + a, base + z, kind, supported))
        for a, z in page["missing_ranges_s"]:
            ax.axvspan(a, z, color="#e8ebee", lw=0, zorder=0)
        if not page["source_blocks"]:
            ax.set_yticks([])
            ax.text(0.5, 0.5, "无采集数据", ha="center", va="center", transform=ax.transAxes, color=SOURCE)
        lo, hi = ax.get_ylim()
        dy = hi - lo
        ax.set_ylim(lo - 0.10 * dy, hi + 0.10 * dy)
        for a, z, kind, supported in ranges:
            ink = color(kind)
            pad = 0.08 if ink == REVIEW else 0.045
            low, high = (
                (float(supported.min()) - dy * pad, float(supported.max()) + dy * pad) if len(supported) else (lo, hi)
            )
            ax.add_patch(
                Rectangle(
                    (a / fs, low),
                    (z - a) / fs,
                    high - low,
                    edgecolor=ink,
                    facecolor=to_rgba(ink, 0.07),
                    linewidth=1.3,
                    linestyle="--" if ink == REVIEW else "-",
                    zorder=4 if ink == STEP else 2,
                )
            )
        ax.set_xlim(left / fs, right / fs)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("ABP（mmHg）" if task == "ABP" else f"{task}（源单位）")
        return visible

    def minute(page):
        fig = Figure(figsize=(12.4, 5.8))
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 1, sharex=True)
        visible = set()
        width = round((page["end_s"] - page["start_s"]) * fs)
        for task, ax in zip(TASKS, axes, strict=True):
            ids = draw(ax, page, task, 0, width)
            visible.update(ids)
            ax.set_title(f"初筛命中 {len(ids)} 条", loc="right", fontsize=9, pad=1)
        axes[-1].set(xlabel="本页时间（秒）", xticks=np.arange(0, width / fs + 0.01, 5))
        fig.suptitle(f"全程初筛框选   ·   第 {page['page']} / {pages['pages']} 分钟", x=0.082, ha="left", fontsize=17)
        fig.text(
            0.082,
            0.902,
            "红框  ECG 基线偏移     蓝框  ABP 逐搏待查     橙框  其他波形触发     紫框  频谱检测窗     灰框  来源标志     灰底  无采集",
        )
        fig.text(0.082, 0.855, "框的左右边对应命中样本的起止时间。同一规则首尾相接的提示合画一框。")
        fig.subplots_adjust(left=0.082, right=0.985, top=0.80, bottom=0.10, hspace=0.34)
        return fig, visible

    page = pages["windows"][detail_page - 1]
    target = round(detail_second * fs)
    offset = round(page["start_s"] * fs)
    seed = next(
        b
        for i in page["source_blocks"]
        if (b := blocks[i])["start"] <= page["start_s"] + detail_second < b["start"] + b["duration"]
    )
    cid = seed["ids"]["ECG"]
    base = round(seed["start"] * fs) - offset
    a, z, kind, _ = next(
        box
        for box in boxes[cid]
        if box[2] in {"ecg_baseline_shift", "large_step"} and base + box[0] <= target < base + box[1]
    )
    label = "基线偏移" if kind == "ecg_baseline_shift" else "尖刺"
    left, right = base + a, base + z
    reviews = {}
    for task in TASKS:
        path = workspace / "visual-reviews" / task / "decision.json"
        if path.is_file():
            review = read(path)
            reviewed_case = review["seed_case"]
            if (
                review["status"] == "registered"
                and reviewed_case == seed["ids"][task]
                and review["seed_version"] == "raw-" + cases[reviewed_case]["input_sha256"]
            ):
                u, v = review["suspected_samples"]
                if u < z and v > a:
                    reviews[reviewed_case] = [(u, v, "historical_review", [])]

    with rc_context({"font.family": "Noto Sans CJK SC", "pdf.fonttype": 42, "axes.unicode_minus": False}):
        fig = Figure(figsize=(12.4, 6.2))
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 2, sharex="col")
        for row, task in enumerate(TASKS):
            draw(axes[row, 0], page, task, left - round(5 * fs), right + round(5 * fs))
            draw(
                axes[row, 1],
                page,
                task,
                left - round(0.23 * fs),
                right + round(0.23 * fs),
                reviewed=reviews,
                show_spectrum=False,
            )
        axes[0, 0].set_title("初筛命中前后各 5 秒", fontsize=12)
        axes[0, 1].set_title("同一位置放大" + ("，叠加上次 AI 复查框" if reviews else ""), fontsize=12)
        for ax in axes[-1]:
            ax.set_xlabel(f"第 {detail_page} 分钟内的时间（秒）")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        fig.suptitle(
            f"{label}的初筛范围   ·   本页 {left / fs:.3f}–{right / fs:.3f} 秒", x=0.08, ha="left", fontsize=17
        )
        fig.text(
            0.08,
            0.896,
            "红框  ECG 基线偏移     蓝框  ABP 逐搏待查     橙框  陡变样本     紫框  高频能量检测窗"
            + ("     青色虚框  上次 AI 复查范围" if reviews else ""),
        )
        fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.11, wspace=0.22, hspace=0.26)
        fig.savefig(out / f"{label}局部框选.png", dpi=145)
        fig.clear()
        seen = set()
        with PdfPages(out / "全程初筛框选-每页60秒.pdf") as pdf:
            for win in pages["windows"]:
                fig, ids = minute(win)
                seen.update(ids)
                if win["page"] == detail_page:
                    fig.savefig(out / f"第{detail_page}分钟-框选.png", dpi=140)
                pdf.savefig(fig)
                fig.clear()
    if seen != all_ids or len(all_ids) != manifest["candidate_count"]:
        raise ValueError("Full gallery does not cover all source candidates")
    print(f"{len(seen)} candidates, {pages['pages']} pages, detail {left / fs:.3f}–{right / fs:.3f} s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--inspection", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--detail-page", required=True, type=int)
    p.add_argument("--detail-second", required=True, type=float)
    args = p.parse_args()
    render(args.workspace, args.inspection, args.out, args.detail_page, args.detail_second)
