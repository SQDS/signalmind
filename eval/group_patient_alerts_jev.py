"""Group a complete Vital screening run by channel and original reason, with per-hit JEV review routing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from tsmcp.cleaning.common import read, sha
from tsmcp.cleaning.engine import load_arrays
from tsmcp.inspection.patient_profile import beat_pairs

MODEL = "jev-1.14.0"
URL = "https://api.wulusai.com/v1/evaluate"
CHANNELS = ("ECG", "PPG", "ABP")
RATE = {"rate_high", "rate_low", "probe_disagreement"}
WHY = {
    "large_step": "相邻样本突跳",
    "y_low": "幅值偏低",
    "y_high": "幅值偏高",
    "rate_high": "速率偏快",
    "rate_low": "速率偏慢",
    "probe_disagreement": "检测器分歧",
    "ecg_excursion": "ECG 异常偏移",
    "ecg_baseline_shift": "ECG 基线偏移",
    "abp_pulse_episode": "动脉压力脉搏形态变化",
}
COMMON = """Group individual screening alerts for subsequent review. Each question contains exactly one numeric row.
All answers describe review routes, not clinical diagnoses or permission to discard data. Use only that row.
Columns in order: alert duration (s), trigger value, maximum single-sample step / typical patient beat swing in core incl. both boundary edges, core signal range / typical swing,
distance from the core midpoint to the nearest native beat landmark (ms), own minimum/maximum native rates (bpm) overlapping the core,
peer A overlapping native rate median, peer B overlapping native rate median, number of peers with a rate or waveform alert overlapping the core (0 to 2),
number of peers with detector disagreement overlapping it. Null means unknown. Waveform amplitudes and step sizes are channel-specific. An alert is a hypothesis.
Consider peers, temporal extent and landmark proximity. Do not assume an unusual rate is an artifact."""
LOCAL_RATE = """True if the evidence favors an isolated single-channel event-detection/timing mismatch: discrepant native rate or detector disagreement while both peer channels provide coherent rates, with no compelling simultaneous pulse change. False if a shared rhythm/pulse change is plausible, peers are also disrupted, or evidence does not distinguish these. Classify where to review next."""
LOCAL_WAVE = """True if the evidence favors a local waveform shape issue away from ordinary beat landmarks, such as an abrupt isolated edge, clipping, lost pulsatility, or a local baseline shift with adequately supported stable peers. False if the deviation is plausibly beat-related or synchronous across channels, or the numeric evidence is inconclusive. Classify where to review next."""
MAX_TOKENS = 4_000_000
BATCH = 24


def dump(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


def sig(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def route_names(reason):
    if reason in RATE:
        return ("本通道搏动定位复测", "联合节律或证据不足复查")
    return ("本通道局部波形复查", "搏动相关或证据不足联合复查")


def native_rates(d, a, b):
    lo = d["right"].searchsorted(a, side="right")
    hi = d["left"].searchsorted(b, side="left")
    return d["bpm"][lo:hi]


def hits_by_group(source):
    w = source / "candidate"
    manifest = read(w / "dataset_manifest.json")
    profile = read(source / "patient-profile.json")
    base = read(w / "versions" / read(w / "baselines.json")["identity"] / "version.json")["results"]
    origin = read(source / "discovery.json")["origin"]
    by_record = defaultdict(dict)
    for c in manifest["cases"]:
        by_record[tuple(c["rows"])][c["task"]] = c
    out = defaultdict(list)
    for members in by_record.values():
        signals = {}
        for task, c in members.items():
            cid = c["id"]
            assert sha(w / "public" / cid / "input.npz") == c["input_sha256"]
            _, t, x, valid, est = load_arrays(w, c)
            probe_path = w / base[cid]["probe_path"]
            assert sha(probe_path) == base[cid]["probe_sha256"]
            probe = read(probe_path)
            pairs = np.array(beat_pairs(probe), dtype=int).reshape(-1, 2)
            left, right = pairs.T
            flags = np.zeros((2, len(x)), bool)
            for h in c["screen_cores"]:
                flags[int(h["kind"] == "probe_disagreement"), h["a"] : h["b"]] = True
            signals[task] = dict(
                t=t,
                x=x,
                support=valid & ~est,
                flags=flags,
                left=left,
                right=right,
                bpm=60.0 * c["fs"] / (right - left),
                peaks=np.array([z["sample"] for z in probe["events"]], int),
            )
        for task, c in members.items():
            d = signals[task]
            peers = [k for k in CHANNELS if k != task]
            fs = c["fs"]
            for h in c["screen_cores"]:
                a, b = h["a"], h["b"]
                t = d["t"]
                start = t[a]
                end = t[b - 1] + 1 / fs
                own = native_rates(d, a, b)
                x = d["x"]
                support = d["support"]
                left_edge = max(0, a - 1)
                right_edge = min(b + 1, len(x))
                neighbor_ok = support[left_edge : right_edge - 1] & support[left_edge + 1 : right_edge]
                steps = np.abs(np.diff(x[left_edge:right_edge]))[neighbor_ok]
                vals = x[a:b][support[a:b]]
                scale = profile["amplitude"][task]["typical_swing"]
                peaks = d["peaks"]
                mid = (a + b - 1) * 0.5
                j = int(peaks.searchsorted(mid))
                near = peaks[max(j - 1, 0) : j + 1]
                pm = []
                concurrence = [0, 0]
                for peer in peers:
                    if peer not in signals:
                        pm.append(None)
                        continue
                    other = signals[peer]
                    # Match by source time and the same acquisition, retaining true gaps.
                    g0, g1 = other["t"].searchsorted([start - 1e-7, end - 1e-7])
                    vals2 = native_rates(other, g0, g1) if g0 < g1 else []
                    pm.append(round(float(np.median(vals2)), 1) if len(vals2) else None)
                    for k in range(2):
                        concurrence[k] += int(other["flags"][k, g0:g1].any())
                feat = [
                    round((b - a) / fs, 3),
                    round(h["value"], 3) if h["value"] is not None else None,
                    round(float(np.max(steps)) / scale, 3) if len(steps) else None,
                    round(float(np.ptp(vals)) / scale, 3) if len(vals) else None,
                    round(float(np.min(np.abs(near - mid))) * 1000 / fs) if len(near) else None,
                    round(float(own.min()), 1) if len(own) else None,
                    round(float(own.max()), 1) if len(own) else None,
                    *pm,
                    *concurrence,
                ]
                out[task, h["kind"]].append(
                    dict(id=h["id"], case=c["id"], a=a, b=b, t=round(float(start - origin), 4), fs=fs, features=feat)
                )
    ledger = read(source / "member-ledger.json")
    expected = {r["source"] for r in ledger}
    ids = [r["id"] for grp in out.values() for r in grp]
    assert len(ids) == len(set(ids)) and set(ids) == expected
    return out


def query_batch(job):
    name, state, members, path = job
    # Compact question-specific measurements avoid asking the model to find rows in a shared long table.
    questions = {
        f"r{i:02d}": dict(
            type="boolean",
            instructions="Does this row support the local route? Measurements "
            + json.dumps(r["features"], separators=(",", ":")),
        )
        for i, r in enumerate(members)
    }
    body = dict(model=MODEL, state=state, questions=questions)
    binding = sig([body, [r["id"] for r in members]])
    if path.exists():
        receipt = read(path)
        if receipt["binding"] != binding:
            raise ValueError(f"Input changed for {name}")
        return receipt
    token = os.environ["WULUSAI_API_KEY"]
    for attempt in range(2):
        try:
            resp = requests.post(
                URL, headers={"Authorization": "Bearer " + token}, json=body, timeout=(8, 60), allow_redirects=False
            )
            if resp.status_code in (429, 502, 503, 504) and attempt == 0:
                time.sleep(2)
                continue
            resp.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt:
                raise
    data = resp.json()
    assert data["model"] == MODEL and set(data["answers"]) == set(questions)
    scores = {}
    for key, m in zip(questions, members, strict=True):
        prob = data["answers"][key]["probability"]
        assert isinstance(prob, (float, int)) and 0 <= prob <= 1
        scores[m["id"]] = prob
    receipt = dict(
        batch=name,
        binding=binding,
        model=data["model"],
        request_id=resp.headers.get("x-request-id"),
        usage=data["usage"],
        attempts=attempt + 1,
        scores=scores,
    )
    dump(path, receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--render-only", action="store_true", help="Render the already classified groups without contacting JEV."
    )
    cfg = ap.parse_args()
    source = cfg.source
    out = cfg.out
    if cfg.render_only:
        render(source, out)
        return
    receipts = out / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    profile = read(source / "patient-profile.json")
    groups = hits_by_group(source)
    batches = []
    for (task, reason), members in sorted(groups.items()):
        peers = [k for k in CHANNELS if k != task]
        amp = profile["amplitude"][task]
        state = (
            f"{COMMON}\nChannel={task}; trigger={reason}; peers A/B={peers[0]}/{peers[1]}; personal rate reference {profile['hr']['median']:.1f} and alert limits {profile['hr']['low']:.1f}–{profile['hr']['high']:.1f} bpm. Channel amplitude alert limits {amp['low']:.2f}–{amp['high']:.2f}.\n"
            + (LOCAL_RATE if reason in RATE else LOCAL_WAVE)
        )
        members.sort(key=lambda v: v["id"])
        for i in range(0, len(members), BATCH):
            name = f"{task}-{reason}-{i // BATCH:03}"
            batches.append((name, state, members[i : i + BATCH], receipts / (name + ".json")))
    print("primary", len(groups), "batches", len(batches), "items", sum(map(len, groups.values())), flush=True)
    usage = Counter()
    scores = {}
    attempts = 0
    cached_tokens = sum(
        sum(read(f)["usage"].get(k, 0) for k in ("input_tokens", "output_tokens")) for f in receipts.glob("*.json")
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(batches), 4):
            pending = sum(not j[3].exists() for j in batches[i : i + 4])
            if cached_tokens + pending * 24000 > MAX_TOKENS:
                raise RuntimeError("JEV token stop reached")
            for result in pool.map(query_batch, batches[i : i + 4]):
                usage.update(result["usage"])
                scores.update(result["scores"])
                attempts += result["attempts"]
            cached_tokens = sum(
                sum(read(f)["usage"].get(k, 0) for k in ("input_tokens", "output_tokens"))
                for f in receipts.glob("*.json")
            )
            if i % 24 == 0:
                print("verified", len(scores), "tokens", cached_tokens, flush=True)
    assert sum(map(len, groups.values())) == len(scores)
    export(source, out, groups, scores, usage, attempts, len(batches))
    render(source, out)


def export(source, out, groups, scores, usage, attempts, n_batches):
    with (source / "delivery/未逐框复核的跨通道重合位置.csv").open(encoding="utf-8-sig") as f:
        priorities = list(csv.DictReader(f))
    old_ids = defaultdict(set)
    for r in read(source / "member-ledger.json"):
        old_ids[r["source"]].add(r["packet"])
    leaf = defaultdict(list)
    for (task, reason), members in sorted(groups.items()):
        for m in members:
            p = scores[m["id"]]
            route = 0 if p > 0.5 else 1
            key = (task, reason, route)
            end = m["t"] + (m["b"] - m["a"]) / m["fs"]
            priority = []
            for i, v in enumerate(priorities, 1):
                t0 = (int(v["分钟"]) - 1) * 60
                if (
                    m["case"] in v["记录块"].split("/")
                    and m["t"] < t0 + float(v["分钟内止秒_不含"]) - 1e-7
                    and t0 + float(v["分钟内起秒"]) < end - 1e-7
                ):
                    priority.append(f"X{i:02}")
            row = dict(
                m,
                channel=task,
                reason=reason,
                p=p,
                route=route_names(reason)[route],
                priority=priority,
                previous=sorted(old_ids[m["id"]]),
            )
            leaf[key].append(row)
    tasks = []
    detail = []
    for num, ((task, reason, route), rows) in enumerate(sorted(leaf.items()), 1):
        gid = f"G{num:02}"
        data = np.array([[np.nan if v is None else v for v in x["features"]] for x in rows], float)
        for c in range(data.shape[1]):
            col = data[:, c]
            mask = np.isfinite(col)
            col[~mask] = np.median(col[mask]) if mask.any() else 0.0
        scale = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        z = (data - np.median(data, axis=0)) / np.where(scale > 1e-6, scale, 1.0)
        distance = np.mean(np.abs(np.clip(z, -8, 8)), axis=1)
        typical = rows[int(distance.argmin())]
        pics = [("组内典型", typical)]
        unusual = sorted(
            range(len(rows)), key=lambda i: (not bool(rows[i]["priority"]), abs(rows[i]["p"] - 0.5), -distance[i])
        )
        for i in unusual:
            if rows[i]["id"] != typical["id"]:
                pics.append(("联合复查" if rows[i]["priority"] else "分流边界", rows[i]))
                break
        all_priority = sorted({v for r in rows for v in r["priority"]})
        tasks.append(
            dict(
                group=gid,
                channel=task,
                reason=reason,
                label=route_names(reason)[route],
                count=len(rows),
                near_boundary=sum(abs(r["p"] - 0.5) <= 0.101 for r in rows),
                joint_windows=all_priority,
                case_count=len({r["case"] for r in rows}),
                examples=[dict(role=k, **r) for k, r in pics],
                agent_review="待复查",
            )
        )
        for r in sorted(rows, key=lambda z: z["t"]):
            detail.append(
                dict(
                    任务组=gid,
                    通道=task,
                    触发原因=reason,
                    JEV细分=r["route"],
                    原始提示=r["id"],
                    记录块=r["case"],
                    起始样本=r["a"],
                    结束样本_不含=r["b"],
                    记录内起秒=r["t"],
                    记录内止秒_不含=round(r["t"] + (r["b"] - r["a"]) / r["fs"], 4),
                    JEV本通道路线分数=r["p"],
                    原分钟组=" ".join(r["previous"]),
                    联合复查编号=" ".join(r["priority"]),
                )
            )

    def csvout(name, rows):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    csvout("每条提示的JEV分组.csv", detail)
    csvout(
        "Agent逐组复查清单.csv",
        [
            dict(
                任务组=t["group"],
                通道=t["channel"],
                原始原因=WHY[t["reason"]],
                JEV细分=t["label"],
                提示条数=t["count"],
                连续块数=t["case_count"],
                分数靠近分界条数=t["near_boundary"],
                图片页=int(t["group"][1:]),
                联合复查编号=" ".join(t["joint_windows"]),
                状态=t["agent_review"],
            )
            for t in tasks
        ],
    )
    cross = defaultdict(set)
    for task in tasks:
        for ix in task["joint_windows"]:
            cross[ix].add(task["group"])
    assert set(cross) == {f"X{i:02}" for i in range(1, len(priorities) + 1)}
    csvout(
        "跨通道位置与任务组.csv",
        [
            dict(联合复查编号=f"X{i:02}", 候选任务组=" ".join(sorted(cross[f"X{i:02}"])), **r)
            for i, r in enumerate(priorities, 1)
        ],
    )
    assert len(detail) == len({r["原始提示"] for r in detail}) == sum(t["count"] for t in tasks)
    dump(out / "group_review.json", tasks)
    summary = dict(
        model=MODEL,
        patient=read(source / "patient-profile.json")["patient"],
        source_packets=len(read(source / "packets.json")),
        first=len(groups),
        second=len(tasks),
        hits=len(detail),
        batches=n_batches,
        attempts=attempts,
        usage=usage,
        by_channel={
            k: dict(
                first=sum(t == k for t, _ in groups),
                second=sum(t["channel"] == k for t in tasks),
                hits=sum(t["count"] for t in tasks if t["channel"] == k),
            )
            for k in CHANNELS
        },
        uncertain_score_count=sum(t["near_boundary"] for t in tasks),
        threshold=0.5,
        returned_classes="P(local)>0.5 selects local review; all other scores select joint review.",
        grouping_keys=["channel", "original_trigger", "JEV_route"],
        time_limits=False,
        priority_windows=len(cross),
        agent_reviewed=0,
    )
    dump(out / "分组汇总.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def render(source, out):
    from functools import lru_cache

    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    w = source / "candidate"
    cases = read(w / "dataset_manifest.json")["cases"]
    blocks = {c["id"]: tuple(c["rows"]) for c in cases}
    peers = {(tuple(c["rows"]), c["task"]): c for c in cases}
    profile = read(source / "patient-profile.json")
    jobs = read(out / "group_review.json")
    origin = read(source / "discovery.json")["origin"]

    @lru_cache(maxsize=12)
    def trace(block, task):
        case = peers[(block, task)]
        _, time_s, x, valid, est = load_arrays(w, case)
        return time_s, np.where(valid & ~est, x, np.nan)

    with (
        rc_context({"font.family": "Noto Sans CJK SC", "pdf.fonttype": 42, "axes.unicode_minus": False}),
        PdfPages(out / "Agent逐组波形图.pdf") as pdf,
    ):
        for job in jobs:
            examples = job["examples"]
            fig = Figure(figsize=(12, 6.9))
            FigureCanvasAgg(fig)
            axes = fig.subplots(3, len(examples), sharex="col", sharey="row", squeeze=False)
            fig.suptitle(
                f"{job['group']}　{job['channel']} · {WHY[job['reason']]}　｜　{job['label']}",
                x=0.085,
                y=0.975,
                ha="left",
                fontsize=17,
            )
            fig.text(
                0.085,
                0.89,
                f"{job['count']:,} 条提示，来自 {job['case_count']} 个连续记录块。橙底标出本条提示的时间，各通道共用时间轴。",
                fontsize=10,
            )
            for col, row in enumerate(examples):
                block = blocks[row["case"]]
                start = origin + row["t"]
                length = (row["b"] - row["a"]) / row["fs"]
                page = 1 + int(row["t"] // 60)
                axes[0, col].set_title(
                    f"{row['role']}　·　第 {page} 分钟 {row['t'] % 60:.2f} 秒　·　本通道路线分数 {row['p']:.2f}",
                    loc="left",
                    fontsize=10,
                )
                for i, t in enumerate(CHANNELS):
                    ax = axes[i, col]
                    if (block, t) not in peers:
                        ax.text(0.5, 0.5, "无该通道", transform=ax.transAxes, ha="center")
                        continue
                    time_s, x = trace(block, t)
                    view = (time_s >= start - 5) & (time_s < start + length + 5)
                    ax.axvspan(0, length, fc="#f5c170", alpha=0.48 if t == row["channel"] else 0.20, lw=0)
                    ax.plot(
                        time_s[view] - start, x[view], lw=1.0, color="#116782" if t == row["channel"] else "#40606d"
                    )
                    ax.set_xlim(-5, length + 5)
                    ax.set_ylim(*profile["amplitude"][t]["display"])
                    ax.spines[["top", "right"]].set_visible(False)
                    if col == 0:
                        ax.set_ylabel(t + "\n" + ("mmHg" if t == "ABP" else "原始单位"))
                axes[-1, col].set_xlabel("相对触发起点（秒）")
            fig.subplots_adjust(left=0.085, right=0.98, top=0.79, bottom=0.10, wspace=0.08, hspace=0.20)
            pdf.savefig(fig)
            fig.clear()


if __name__ == "__main__":
    main()
