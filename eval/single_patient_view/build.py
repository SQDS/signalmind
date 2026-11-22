"""Offline overview and native-sample browser for one completed three-channel run."""

import argparse
import base64
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from tsmcp.cleaning.common import array_hash, changed, read, sha, time_support, write
from tsmcp.cleaning.engine import load_arrays
from tsmcp.processing import intervals

TASKS = ("ECG", "PPG", "ABP")


def js(path, prefix, data):
    path.write_text(
        prefix + "(" + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ");\n"
    )


def encoded(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f8").tobytes()).decode("ascii")


def span_union(spans, n):
    mask = np.zeros(n, bool)
    for a, b in spans:
        mask[a:b] = True
    return intervals(mask)


def make_overview(out, meta, overview_data):
    with rc_context({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "font.size": 10}):
        fig = Figure(figsize=(14, 7), facecolor="white")
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 1, sharex=True)
        for ax, task in zip(axes, TASKS, strict=True):
            info = overview_data[task]
            ax.add_collection(LineCollection(info["bars"], colors="#23628a", linewidths=0.8))
            all_y = [y for segment in info["bars"] for _, y in segment]
            lo = min(all_y)
            hi = max(all_y)
            span = max(1, hi - lo)
            ax.set_ylim(lo - 0.21 * span, hi + 0.13 * span)
            ax.set_ylabel(task + "\n" + ("mmHg" if task == "ABP" else "源单位"))
            ax.grid(axis="y", alpha=0.13)
            ax.spines[["top", "right"]].set_visible(False)
            for name, color, lane in [
                ("trigger", "#d58939", -0.055),
                ("excluded", "#707480", -0.11),
                ("modified", "#7956ab", -0.16),
            ]:
                segs = [[(a, lo + lane * span), (b, lo + lane * span)] for a, b in info[name]]
                ax.add_collection(LineCollection(segs, colors=color, linewidths=2, capstyle="round", alpha=0.85))
                # A 2-sample alert is far below one pixel at full-record scale.
                # Put at most one dot per display bin to mark its presence at that zoom.
                width_hours = meta["elapsed_s"] / 3600
                dots = sorted({min(1799, int(1800 * (a + b) / (2 * width_hours))) for a, b in info[name]})
                if dots:
                    ax.scatter(
                        [(col + 0.5) * width_hours / 1800 for col in dots],
                        [lo + lane * span] * len(dots),
                        s=3,
                        color=color,
                        lw=0,
                    )
            if info["ai"]:
                ax.scatter(
                    info["ai"],
                    [hi + 0.08 * span] * len(info["ai"]),
                    marker="v",
                    s=47,
                    facecolors="#faca58",
                    edgecolors="#805000",
                    linewidths=0.8,
                    zorder=3,
                )
        axes[-1].set(xlabel="距首个观测的小时数 · 空白表示记录缺口", xlim=(0, meta["elapsed_s"] / 3600))
        fig.suptitle("单患者完整观测  |  ECG · PPG · ABP", x=0.09, ha="left", y=0.99, fontsize=17)
        fig.legend(
            handles=[
                Line2D([], [], color="#23628a", label="原始幅度范围"),
                Line2D([], [], color="#d58939", label="程序触发位置"),
                Line2D([], [], color="#f1b735", marker="v", ls="", label="本次看图圈选"),
                Line2D([], [], color="#707480", label="新增停用"),
                Line2D([], [], color="#7956ab", label="数值修改"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.083, 0.94),
            frameon=False,
            ncol=5,
        )
        fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.09, hspace=0.24)
        fig.savefig(out / "overview.png", dpi=130)


def reviewed_event(out, group, data, reviews):
    example = next((r for r in reviews if r["task"] == "PPG" and r["case"] == group["PPG"]["id"]), None)
    if not example:
        return
    fs = group["PPG"]["fs"]
    anchor = example["span"] or example["seed"]
    with rc_context({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "font.size": 9}):
        fig = Figure(figsize=(11, 6), facecolor="white")
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 2, gridspec_kw={"width_ratios": [1.8, 1]})
        for row, task in enumerate(TASKS):
            payload = data[task]
            raw = np.frombuffer(base64.b64decode(payload["raw"]), dtype="<f8")
            output = np.frombuffer(base64.b64decode(payload["output"]), dtype="<f8") if payload["output"] else raw
            for col, padding in enumerate((5, 0.45)):
                ax = axes[row, col]
                a = max(0, anchor[0] - round(padding * fs))
                b = min(len(raw), anchor[1] + round(padding * fs))
                t = np.arange(a, b) / fs
                ax.plot(t, raw[a:b], color="#245d80", lw=1.1)
                ax.plot(t, output[a:b], color="#168874", ls="--", lw=0.8)
                for entry in payload["ai"]:
                    left, h = entry["span"]
                    if a < h and left < b:
                        ax.axvspan(max(left, a) / fs, min(h, b) / fs, color="#f6c961", alpha=0.38)
                for left, h in payload["excluded"]:
                    if a < h and left < b:
                        ax.axvspan(
                            max(left, a) / fs,
                            min(h, b) / fs,
                            facecolor="none",
                            hatch="///",
                            edgecolor="#899298",
                            lw=0.5,
                        )
                ax.set_xlim(a / fs, b / fs)
                ax.ticklabel_format(axis="x", useOffset=False)
                if col == 0:
                    ax.set_ylabel(task + " / " + ("mmHg" if task == "ABP" else "源单位"))
                if row == 0:
                    ax.set_title("疑点前后各 5 秒" if col == 0 else "采样点局部放大")
                if row == 2:
                    ax.set_xlabel("连续段内时间（秒）")
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("同一源时间位置的三通道复查", x=0.085, ha="left", fontsize=16)
        fig.text(
            0.085,
            0.92,
            "黄色为本次视觉复查圈选，灰色斜线为实际新增停用范围。蓝线为原始值，绿色虚线为交付值。",
            fontsize=10,
        )
        fig.subplots_adjust(top=0.825, bottom=0.1, left=0.09, right=0.98, hspace=0.38, wspace=0.16)
        fig.savefig(out / "reviewed-event.png", dpi=140)


def build(root, out):
    manifest = read(root / "dataset_manifest.json")
    state = read(root / "agent_state.json")
    if not state["complete"] or set(state["accepted_tasks"]) != set(TASKS):
        raise ValueError("Awaiting the three accepted batch versions")
    if len(manifest["records"]) != 1:
        raise ValueError("Expected one patient record")
    cases = manifest["cases"]
    versions = {k: read(root / "versions" / vid / "version.json") for k, vid in state["accepted_tasks"].items()}
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    (out / "reviews").mkdir(exist_ok=True)
    candidates = defaultdict(list)
    with (root / "inspection/candidates.jsonl").open() as stream:
        for line in stream:
            c = json.loads(line)
            rr = c["range"]
            candidates[rr["block_id"]].append([rr["start_sample"], rr["end_sample"], c["observations"][0]["tool_id"]])
    grouped = defaultdict(dict)
    for c in cases:
        grouped[tuple(c["rows"])][c["task"]] = c
    reviews = []
    for p in sorted((root / "visual-reviews").glob("*/decision.json")):
        d = read(p)
        images = []
        for v in d["views"]:
            src = p.parent / "views" / v["image"]
            if sha(src) != v["sha256"]:
                raise ValueError("Reviewer image hash mismatch")
            name = p.parent.name + "-" + src.name
            shutil.copyfile(src, out / "reviews" / name)
            images.append({"image": "reviews/" + name, "range": v["view_samples"], "flank": v["flank_s"]})
        reviews.append(
            {
                "task": p.parent.name,
                "case": d["seed_case"],
                "status": d["status"],
                "seed": d["seed_samples"],
                "span": d.get("suspected_samples"),
                "obs": d.get("observed", d.get("reason", "")),
                "cause": d.get("suspected_cause", ""),
                "limitations": d.get("alternative_or_limitation", ""),
                "images": images,
                "rule": d.get("rule"),
                "request": d.get("request_id"),
            }
        )
    rescans = []
    residuals = defaultdict(list)
    for task, vid in state["accepted_tasks"].items():
        for rid, ref in state.get("pattern_rescans", {}).get(vid, {}).items():
            report = read(root / ref["report"])
            if (
                not report["complete"]
                or report["waveform_version"] != vid
                or sha(root / ref["findings"]) != report["findings_sha256"]
            ):
                raise ValueError("Incomplete/unbound postrepair search")
            rescans.append(
                {
                    "task": task,
                    "rule_id": rid,
                    "counts": report["counts"],
                    "cases": report["scanned"],
                    "hypothesis": report["cause_hypothesis"],
                    "version": vid,
                }
            )
            for row in read(root / ref["findings"]):
                residuals[row["case_id"]].extend([a, b, rid] for a, b in row["after_eligible"])
    batch_reviews = []
    for task, vid in state["accepted_tasks"].items():
        rev = state["reviews"][vid]
        pictures = []
        for eid in rev["evidence_ids"]:
            ev = state["evidence"][eid]
            src = root / ev["image"]
            if ev["version_id"] != vid or sha(src) != ev["image_sha256"]:
                raise ValueError("Self-review evidence mismatch")
            name = f"{task}-batch-{eid}.png"
            shutil.copyfile(src, out / "reviews" / name)
            pictures.append(dict(image="reviews/" + name, case=ev["case_id"], span=ev["range_samples"]))
        batch_reviews.append(
            dict(
                task=task,
                version=vid,
                observation=rev["observations"],
                concerns=rev["remaining_concerns"],
                pictures=pictures,
            )
        )
    overview_data = {k: defaultdict(list) for k in TASKS}
    stats = {k: Counter() for k in TASKS}
    blocks = []
    origin = None
    for i, group in enumerate(sorted(grouped.values(), key=lambda g: g["ECG"]["rows"][0])):
        data = {}
        group_times = None
        for task in TASKS:
            c = group[task]
            cid = c["id"]
            fs = c["fs"]
            v = versions[task]
            a = v["results"][cid]
            path = root / "public" / cid / "input.npz"
            if sha(path) != c["input_sha256"] or (a["arrays"] and sha(root / a["arrays"]) != a["arrays_sha256"]):
                raise ValueError("Waveform file changed")
            raw, t, x, valid, est = load_arrays(root, c, a)
            if (
                array_hash(x, t, valid, est) != a["content_hash"]
                or state["accepted"][cid]["artifact_id"] != a["artifact_id"]
            ):
                raise ValueError("Accepted content mismatch")
            if group_times is not None and not np.array_equal(t, group_times):
                raise ValueError("Cannot show these channels on a shared source grid")
            group_times = t
            if origin is None:
                origin = float(t[0])
            start = float(t[0] - origin)
            native = np.isfinite(raw) & time_support(c)
            bad = intervals(native & ~valid)
            edited = intervals(changed(raw, x))
            ai = [{"span": r["span"], "tag": r["task"]} for r in reviews if r["case"] == cid and r["span"]]
            masked = intervals(~valid | est)
            operations = a["operations"]
            if operations and a.get("operator_trace"):
                trace = a["operator_trace"]
                if sha(root / trace["path"]) != trace["sha256"]:
                    raise ValueError("Operator trace mismatch")
                steps = read(root / trace["path"])["steps"]
                operations = [
                    {**op, "explanation": step["explanation"], "retest": step["retest"]}
                    for op, step in zip(operations, steps, strict=True)
                ]
            item = dict(
                id=cid,
                unit=c["unit"],
                raw=encoded(raw),
                output=encoded(x) if edited else None,
                stops=masked,
                excluded=bad,
                source_bad=intervals(~native),
                modified=edited,
                ai=ai,
                estimated=intervals(est),
                triggers=candidates[cid],
                rescan=residuals[cid],
                operations=operations,
            )
            data[task] = item
            info = overview_data[task]
            # Per-block bins retain true extrema and do not interpolate across source gaps.
            step = max(1, round(fs * 6))
            for j in range(0, len(raw), step):
                part = raw[j : j + step]
                good = part[np.isfinite(part)]
                if len(good):
                    coord = (start + (j + len(part) / 2) / fs) / 3600
                    info["bars"].append([(coord, float(good.min())), (coord, float(good.max()))])
            for name, spans in (
                ("excluded", bad),
                ("modified", edited),
                ("trigger", span_union([r[:2] for r in candidates[cid]], len(x))),
            ):
                info[name].extend(((start + lo / fs) / 3600, (start + hi / fs) / 3600) for lo, hi in spans)
            info["ai"].extend((start + (r["span"][0] + r["span"][1]) / 2 / fs) / 3600 for r in ai)
            stats[task].update(
                blocks=1,
                raw_samples=len(x),
                candidate_triggers=len(candidates[cid]),
                new_invalid=sum(b - a for a, b in bad),
                modified=sum(b - a for a, b in edited),
                estimated=int(est.sum()),
                scored_samples=a["stats"]["measured_samples"],
                events=a["stats"]["events"],
                intervals=a["stats"]["intervals"],
                pressure_cycles=a["stats"]["pressure_cycles"],
            )
        blocks.append(
            dict(
                i=i,
                start=start,
                duration=len(t) / fs,
                n=len(t),
                fs=fs,
                ids={k: g["id"] for k, g in group.items()},
                source_start=float(t[0]),
                counts={
                    k: {name: len(data[k][name]) for name in ["triggers", "ai", "modified", "excluded", "rescan"]}
                    for k in TASKS
                },
            )
        )
        reviewed_event(out, group, data, reviews)
        js(out / "blocks" / f"{i:03d}.js", "receiveBlock", {"i": i, "signals": data})
    budget = read(root / "api_budget.json")
    meta = dict(
        record=manifest["records"][0]["record_id"],
        origin=origin,
        blocks=blocks,
        elapsed_s=max(b["start"] + b["duration"] for b in blocks),
        stored_s=sum(b["duration"] for b in blocks),
        stats=stats,
        reviews=reviews,
        batch_reviews=batch_reviews,
        rescans=rescans,
        accepted=state["accepted_tasks"],
        inspected=read(root / "inspection/inspection_summary.json"),
        api_increment=budget["estimated_peak_usd"] - read(root / "initial_budget.json")["estimated_peak_usd"],
        api_cumulative=budget["estimated_peak_usd"],
        api_executions=len(budget["calls"]) - read(root / "initial_budget.json")["calls"],
    )
    write(out / "summary.json", meta)
    js(out / "data.js", "receiveMeta", meta)
    make_overview(out, meta, overview_data)
    for name in ("index.html", "viewer.js"):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    archive = out.parent / "single-patient-gallery.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=4) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, "patient-gallery/" + str(f.relative_to(out)))
    print(archive, archive.stat().st_size)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build(a.workspace, a.out)
