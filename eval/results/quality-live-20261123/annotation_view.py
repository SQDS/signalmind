"""Display the final Vital measurement labels alongside the original grouped DS evidence."""

import csv
import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from tsmcp.cleaning.common import array_hash, read, sha
from tsmcp.cleaning.dataset_evidence import acquisition
from tsmcp.cleaning.engine import load_arrays
from tsmcp.processing import intervals

HERE = Path(__file__).resolve().parent
DISPLAY = read(Path(os.environ["SIGNALMIND_DISPLAY_CONFIG"]))
ROOT = Path(DISPLAY["workspace"]).resolve()
OUT = Path(DISPLAY["output"]).resolve()
CHANNELS = ("ECG", "PPG", "ABP")
PEOPLE = DISPLAY["people"]
PREVIEWS = DISPLAY["previews"]
LABEL = {"artifact": "伪迹", "physiology": "生理变化", "mixed": "混合", "unknown": "原因未定"}
TAG = {"keep": ("可用", "#9bc7bd"), "review": ("待复查", "#e2a435"), "withhold": ("排除", "#c23f37")}
ACTIONS = {
    "keep": "可用",
    "review": "待复查",
    "withhold": "排除",
    "preserve": "保留原值",
    "mark_invalid": "提议停用",
    "interpolate_spike": "提议插值",
    "lowpass": "提议低通",
    "remove_tone": "提议去窄带干扰",
    "local_detrend": "提议局部去趋势",
}
REASON = {
    "abp_pulse_episode": "压力脉搏形态异常",
    "high_frequency": "高频成分偏强",
    "impulsive_core": "短脉冲",
    "biphasic_core": "双相短脉冲",
    "isolated_spike_candidate": "孤立尖峰候选",
    "probe_disagreement": "检测器结果不一致",
    "rate_high": "速率偏高",
    "rate_low": "速率偏低",
    "ecg_baseline_shift": "ECG 基线偏移",
    "ecg_excursion": "ECG 局部偏移",
    "large_step": "相邻采样点突变",
    "line_50": "50 Hz 成分偏强",
    "line_60": "60 Hz 成分偏强",
    "protected_landmark": "候选触及心搏保护区",
    "y_high": "高于个人幅值阈值",
    "y_low": "低于个人幅值阈值",
    "unavailable": "当前版本无可用观测支持",
}
mpl.rcParams.update(
    {
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def counts_text(counts):
    return "，".join(f"{ACTIONS.get(k, k)} {n}" for k, n in counts.most_common())


def compact_time(t):
    return f"{int(t) // 60:02d}分{int(t) % 60:02d}秒"


def group_gallery():
    index = []
    with PdfPages(OUT / "ds-groups.pdf", metadata={"Title": "DS 分组原图与归因"}) as pdf:
        for suffix, stage in [("", "首轮"), ("-rescan", "复扫后重新判读")]:
            for rid, person in PEOPLE.items():
                folder = ROOT / "records" / rid
                f = folder / ("groups" + suffix + ".json")
                if not f.exists():
                    continue
                groups = sorted(read(f), key=lambda g: (CHANNELS.index(g["task"]), g["reason"]))
                plans = read(folder / ("repair-plans" + suffix + ".json"))
                for g in groups:
                    gid = g["id"]
                    pages = [plans[z["id"]] for z in g["pages"]]
                    assert sum(z["end"] - z["start"] for z in g["pages"]) == g["count"]
                    (binding,) = {p["diagnosis_receipt"] for p in pages}
                    rec = read(ROOT / "dataset-receipts" / (binding + ".json"))
                    assert rec["binding"] == binding and rec["stage"] == "diagnosis"
                    (ev,) = [z for z in rec["evidence"] if z[0] == gid]
                    (decision,) = [z for z in rec["decision"]["items"] if z["id"] == gid]
                    assert all(p["diagnosis"] == decision["diagnosis"] for p in pages)
                    pics = [folder / "views" / f"{gid}-{i}.png" for i in range(2)]
                    assert [sha(p) for p in pics] == ev[2]
                    group_label = LABEL[decision["diagnosis"]["label"]]
                    title = REASON.get(g["reason"], g["reason"])
                    line = decision["reason"]
                    page = len(index) + 1
                    img_roles = ev[1]["image_roles"]
                    assert len(img_roles) == 2
                    index.append(
                        dict(
                            页=page,
                            轮次=stage,
                            患者=person,
                            通道=g["task"],
                            触发原因=title,
                            DS归因=group_label,
                            组内提示=g["count"],
                            数值分片=len(pages),
                            模型判断原文=line,
                            各分片方案原文=json.dumps(
                                [
                                    {
                                        "成员范围": [z["start"], z["end"]],
                                        "wave": p["wave"],
                                        "wave_if": p["wave_if"],
                                        "measure": p["measure"],
                                        "measure_if": p["measure_if"],
                                        "otherwise": p["otherwise"],
                                        "原因": p["reason"],
                                    }
                                    for z, p in zip(g["pages"], pages, strict=True)
                                ],
                                ensure_ascii=False,
                            ),
                            分组编号=gid,
                            诊断回执=binding,
                        )
                    )
                    fig = Figure(figsize=(12.5, 8.1), facecolor="white")
                    fig.text(
                        0.045, 0.948, f"{person}　{g['task']}　·　{title}", size=18, weight="bold", color="#203b48"
                    )
                    fig.text(
                        0.045,
                        0.895,
                        f"{stage}　｜　DS 分组归因：{group_label}　｜　{g['count']} 条提示，{len(pages)} 个规划分片",
                        size=12,
                        color="#274d5c",
                    )
                    for i, pic in enumerate(pics):
                        box = fig.add_axes([0.035 + i * 0.49, 0.365, 0.455, 0.445])
                        with Image.open(pic) as im:
                            box.imshow(np.asarray(im.convert("RGB")))
                        box.set_axis_off()
                        name = {"typical": "典型成员", "boundary": "边界成员", "detail": "同一成员的局部放大"}[
                            img_roles[i]
                        ]
                        box.set_title(f"送审图 {i + 1}　{name}", fontsize=10)
                    fig.text(0.045, 0.305, "DS 诊断原文", size=12, weight="bold", color="#203b48")
                    fig.text(0.045, 0.275, "\n".join(textwrap.wrap(line, 82)), size=11, va="top", linespacing=1.5)
                    wave = Counter(p["wave"] for p in pages)
                    measure = Counter(p["measure"] for p in pages)
                    fig.text(0.045, 0.107, f"规划分片的波形动作　{counts_text(wave)}", size=10)
                    fig.text(
                        0.045,
                        0.070,
                        f"命中分片条件时的逐拍策略　{counts_text(measure)}　·　逐拍最终状态见 30 秒交付图册。",
                        size=10,
                        color="#425d66",
                    )
                    fig.text(0.96, 0.045, str(page), ha="right", size=10)
                    if page == 1:
                        fig.savefig(OUT / "ds-first-group.png", dpi=145)
                    pdf.savefig(fig)
                    fig.clear()
    with (OUT / "ds-group-index.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    return index


def final_measurements():
    # Verify the structured rows used for the preview against the full delivery.
    with (ROOT / "export/measurements.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    bycase = defaultdict(list)
    for m in rows:
        bycase[m["case"]].append((float(m["start_s"]), float(m["end_s"]), m["quality"]))
    counts = {ch: dict(Counter(r["quality"] for r in rows if r["channel"] == ch)) for ch in CHANNELS}
    reports = {k: Counter() for k in CHANNELS}
    waves = read(ROOT / "export/waveform_manifest.json")
    for w in waves.values():
        reports[w["channel"]].update(w["measurements"])
    assert all(Counter(counts[ch]) == reports[ch] for ch in CHANNELS)
    return bycase, counts, waves


def overview():
    bycase, counts, wm = final_measurements()
    cases = {c["id"]: c for c in read(ROOT / "dataset_manifest.json")["cases"]}
    selects = PREVIEWS
    fig = Figure(figsize=(13.5, 7.8), layout="constrained")
    gs = fig.add_gridspec(5, 2, height_ratios=[0.72, 1, 1, 1, 0.32])
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0.003, 0.83, "DS 标注落在哪一拍？", fontsize=20, weight="bold", color="#193947")
    head.text(
        0.003,
        0.43,
        "顶端色带标逐拍结果；斜线标停用的波形采样范围。每列为一段连续 30 秒，同通道左右共用纵轴。",
        fontsize=11,
        color="#345360",
    )
    axes = {}
    for i, ch in enumerate(CHANNELS):
        axes[ch, 0] = fig.add_subplot(gs[i + 1, 0])
        axes[ch, 1] = fig.add_subplot(gs[i + 1, 1], sharey=axes[ch, 0])
    notes = []
    evidence = []
    for j, (example, start, subtitle) in enumerate(selects):
        focus = cases[example]
        end = start + 30
        peers = [c for c in cases.values() if acquisition(c) == acquisition(focus)]
        assert {c["task"] for c in peers} == set(CHANNELS)
        span_notes = {}
        for c in peers:
            cid = c["id"]
            ch = c["task"]
            ax = axes[ch, j]
            entry = wm[cid]
            art = read(ROOT / "artifacts" / entry["artifact_id"] / "artifact.json")
            raw, t, y, v, e = load_arrays(ROOT, c, art)
            assert array_hash(y, t, v, e) == entry["content_hash"]
            assert np.array_equal(raw, y, equal_nan=True) and not e.any()
            _, _, _, v0, e0 = load_arrays(ROOT, c)
            lo, hi = t.searchsorted([start - 1e-7, end - 1e-7])
            assert hi - lo == round(30 * c["fs"])
            u = t[lo:hi] - start
            ax.plot(u, y[lo:hi], lw=0.9, c="#254d62")
            ax.margins(y=0.25)
            for a, b in intervals((v0 & ~e0 & ~v)[lo:hi]):
                ax.axvspan(
                    u[a],
                    u[b - 1] + 1 / c["fs"],
                    facecolor="#dee1e3",
                    edgecolor="#536575",
                    alpha=0.75,
                    hatch="////",
                    lw=0.45,
                    zorder=0,
                )
            tally = Counter()
            for a, b, s in bycase[cid]:
                if a < end and start < b:
                    ax.axvspan(
                        max(0, a - start),
                        min(30, b - start),
                        ymin=0.91,
                        ymax=0.985,
                        facecolor=TAG[s][1],
                        edgecolor="none",
                        zorder=4,
                    )
                    tally[s] += 1
            span_notes[ch] = dict(tally)
            if j == 0:
                ax.set_ylabel(f"{ch}\n" + ("mmHg" if ch == "ABP" else "原始单位"))
            if ch == "ECG":
                ax.set_title(
                    f"{PEOPLE[c['record_id']]}　{compact_time(start)}–{compact_time(end)}　｜　{subtitle}",
                    loc="left",
                    fontsize=11,
                    pad=6,
                    color="#1d3946",
                )
            ax.set(xlim=(0, 30), xticks=np.arange(0, 31, 5))
            ax.spines["bottom"].set_color("#a7b7bd")
            ax.spines["left"].set_color("#a7b7bd")
            if ch != "ABP":
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("本窗时间（秒）")
            if ch == focus["task"]:
                spans = [
                    [float(t[lo + a]), float(t[lo + b - 1] + 1 / c["fs"])] for a, b in intervals((v0 & ~e0 & ~v)[lo:hi])
                ]
                evidence.append(
                    dict(
                        case=cid,
                        start=start,
                        end=end,
                        disabled_intervals=spans,
                        withheld_intervals=[
                            [a, b] for a, b, s in bycase[cid] if s == "withhold" and a < end and start < b
                        ],
                    )
                )
        notes.append(dict(case=example, start_s=start, end_s=end, overlapping_measurements=span_notes))
    leg = fig.add_subplot(gs[-1, :])
    leg.axis("off")
    handles = [Line2D([], [], lw=1.4, c="#254d62", label="实测波形")]
    handles += [Patch(facecolor=k[1], edgecolor="none", label="逐拍" + k[0]) for k in TAG.values()]
    handles.append(Patch(facecolor="#dee1e3", edgecolor="#536575", hatch="////", label="波形停用"))
    leg.legend(handles=handles, ncols=5, loc="center", frameon=False, fontsize=11)
    fig.savefig(OUT / "final-labels-30s.png", dpi=160)
    return counts, notes, evidence


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    counts, windows, example_marks = overview()
    with mpl.rc_context({"font.family": ["DejaVu Sans", "Droid Sans Fallback"]}):
        groups = group_gallery()
    print("Measurement labels", counts)
    print(
        "Group labels by pass",
        {s: dict(Counter(g["DS归因"] for g in groups if g["轮次"] == s)) for s in ("首轮", "复扫后重新判读")},
    )
    print("Examples", example_marks)
