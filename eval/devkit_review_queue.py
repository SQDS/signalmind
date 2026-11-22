"""Select representative images after the run-wide scan and audit QRS proximity."""

import argparse
import csv
import heapq
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.cleaning.common import digest, read, sha, write
from tsmcp.followup.pattern import context_range
from tsmcp.inspection.run import load_channel
from tsmcp.plugins.ecg.rpeaks import detect_r_peaks
from tsmcp.processing import intervals


def summarize(workspace, run, out, flank_s=5):
    out.mkdir(parents=True, exist_ok=True)
    frozen = read(run / "candidate_manifest.json")
    for key in ("intervals.csv", "negative_audit.jsonl", "candidates.jsonl", "evidence.jsonl"):
        if sha(run / key) != frozen["files"][key]:
            raise ValueError(f"Modified inspection artifact: {key}")
    cases = {c["id"]: c for c in read(workspace / "dataset_manifest.json")["cases"]}
    groups = defaultdict(list)
    excursions = defaultdict(list)
    with (run / "intervals.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            groups[(cases[row["case_id"]]["task"], row["trigger"])].append(row)
            if row["trigger"] == "ecg_excursion":
                excursions[row["case_id"]].append((float(row["start_s"]), float(row["end_s"])))
    requests = []
    counts = {}
    for (task, kind), rows in groups.items():
        by_score = sorted(rows, key=lambda r: float(r["maximum_value"]))
        counts[f"{task}.{kind}"] = len(rows)
        for tag, row in [("maximum", by_score[-1]), ("median", by_score[(len(rows) - 1) // 2])]:
            request = (row["case_id"], float(row["start_s"]), float(row["end_s"]), kind)
            if not any(p[:4] == request for p in requests):
                requests.append((*request, tag))
    negative = defaultdict(list)
    with (run / "negative_audit.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)["range"]
            negative[r["channel_id"]].append(r)
    for _, regions in sorted(negative.items()):
        for r in heapq.nsmallest(2, regions, key=lambda region: digest([111101, region])):
            c = cases[r["block_id"]]
            requests.append(
                (c["id"], r["start_sample"] / c["fs"], r["end_sample"] / c["fs"], "no_alert", "random_audit_example")
            )
    # The fixed ECG probe is supporting timing evidence, never a corruption label.
    hits = defaultdict(list)
    for r in groups.get(("ECG", "large_step"), []):
        hits[r["case_id"]].append((float(r["start_s"]) + float(r["end_s"])) / 2)
    result = Counter()
    for cid, event_times in hits.items():
        c = cases[cid]
        path = workspace / "public" / cid / "input.npz"
        if sha(path) != c["input_sha256"]:
            raise ValueError("Changed inspection input")
        x, masks = load_channel(path, c)
        mid = np.asarray(event_times)
        result["total"] += len(mid)
        for a, b in intervals(~np.logical_or.reduce(list(masks.values()))):
            where = mid[(mid >= a / c["fs"]) & (mid < b / c["fs"])]
            if not len(where) or b - a < c["fs"]:
                continue
            peaks = (a + detect_r_peaks(x[a:b], c["fs"])) / c["fs"]
            if not len(peaks):
                continue
            result["with_probe_events"] += len(where)
            pos = np.searchsorted(peaks, where)
            distance = np.minimum(
                abs(where - peaks[np.maximum(pos - 1, 0)]), abs(where - peaks[np.minimum(pos, len(peaks) - 1)])
            )
            result["within_80ms"] += int(np.sum(distance <= 0.080))
    cards = []
    # Keep the established sample numbers for triggers and random controls.
    for i, (cid, s, e, kind, reason) in enumerate(sorted(requests, key=lambda r: r[3] == "ecg_excursion")):
        c = cases[cid]
        fs = c["fs"]
        path = workspace / "public" / cid / "input.npz"
        if sha(path) != c["input_sha256"]:
            raise ValueError("Changed inspection input")
        x, masks = load_channel(path, c)
        episode = (
            next(((lo, hi) for lo, hi in excursions[cid] if lo < e and s < hi), None)
            if kind in {"large_step", "ecg_excursion"}
            else None
        )
        focus_start, focus_end = episode or (s, e)
        fig = Figure(figsize=(8, 2.8), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        target = (round(focus_start * fs), round(focus_end * fs))
        a, b = context_range(len(x), fs, target, flank_s)
        ax.plot(np.arange(a, b) / fs, x[a:b], lw=1, color="#215f84")
        left, right = max(s, a / fs), min(e, b / fs)
        if episode:
            ax.axvspan(*episode, facecolor="#e99b19", alpha=0.23, label="Suspected episode")
            if kind == "large_step":
                ax.axvspan(left, right, facecolor="#a43e18", alpha=0.7, label="Step trigger")
            title = f"{c['task']} / Suspected excursion ({episode[0]:.3f}–{episode[1]:.3f} s)"
            ax.legend(loc="upper right", fontsize=8)
        elif kind == "no_alert":
            ax.axvspan(left, right, facecolor="none", edgecolor="#64748b", linestyle="--", linewidth=1.4)
            title = f"{c['task']} / Random audit window (no alert)"
        else:
            ax.axvspan(left, right, alpha=0.20, color="#e99b19")
            title = f"{c['task']} / Flagged interval ({kind})"
        ax.set(xlabel="Block time (s)", ylabel=c["unit"], title=title)
        name = f"sample-{i:02d}.png"
        fig.savefig(out / name, dpi=120)
        card = dict(case_id=cid, task=c["task"], start_s=s, end_s=e, kind=kind, selection=reason, image=name)
        card.update(view_s=[a / fs, b / fs], flank_s=flank_s)
        if episode:
            card["suspected_episode_s"] = list(episode)
        cards.append(card)
    report = dict(
        run_id=frozen["run_id"],
        candidate_counts=counts,
        candidates=sum(counts.values()),
        qrs_proximity=result,
        proximity_basis="ECG large-step centers within 80 ms of native detected QRS; association only",
        cards=cards,
        actual_model_views=0,
    )
    write(out / "queue.json", report)
    percentages = 100 * result["within_80ms"] / result["total"] if result["total"] else 0
    tags = {"maximum": "指标最高", "median": "指标中位", "random_audit_example": "未报警随机样本"}
    content = "".join(
        f'<figure><img loading="lazy" src="{r["image"]}" alt="{html.escape(r["kind"])}"><figcaption>'
        f"{html.escape(r['case_id'])} · {tags[r['selection']]} · {r['start_s']:.3f}–{r['end_s']:.3f} s"
        + (
            f" · 持续异常候选 {r['suspected_episode_s'][0]:.3f}–{r['suspected_episode_s'][1]:.3f} s"
            if "suspected_episode_s" in r
            else ""
        )
        + "</figcaption></figure>"
        for r in cards
    )
    (
        out / "index.html"
    ).write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vital 质量探查样图</title><style>body{{max-width:900px;margin:auto;padding:20px;font:17px/1.6 system-ui;color:#193848}}figure{{margin:20px 0;padding:12px;border:1px solid #ccd6da}}img{{width:100%}}</style><h1>Vital 质量探查样图</h1>
<p>共 {sum(counts.values()):,} 条规则触发记录，选出 {len(cards)} 张代表图。浅橙底纹显示候选区间；若有持续偏移，深橙细条标出所选陡变。灰色虚线框标出未报警随机抽查区间。候选边界待复核。</p>
<p>ECG 陡变中有 {percentages:.1f}% 的区间中点距程序检测的 QRS 不超过 80 毫秒。此数值用于判断复查优先级。</p>
<a href="queue.json">查看取样依据</a>{content}</html>""")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "run", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--flank", type=int, choices=(5, 10), default=5)
    main_args = p.parse_args()
    summarize(main_args.workspace, main_args.run, main_args.out, main_args.flank)
