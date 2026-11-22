"""Prepare reference-masked Vital gaps and prior waveform exclusions, then evaluate offline TimesFM candidates."""

import argparse
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from eval.timesfm_gap_forecast import bridge, file_hash
from tsmcp.cleaning.common import array_hash, read, write
from tsmcp.cleaning.engine import load_arrays
from tsmcp.cleaning.tasks import match_events, probe
from tsmcp.processing import intervals

CHANNELS = ("ECG", "PPG", "ABP")
FS = 125
CONTEXT = 625
MIN_SIDE = 250
LENGTHS = (1, 6, 25, 125, 625)
SEED = 111101
N_BLOCKS = 24


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Release:
    def __init__(self, delivery):
        self.delivery = delivery
        self.manifest = read(delivery / "波形与探针清单.json")
        self.work = Path(next(iter(self.manifest.values()))["parent"]).parents[2]
        self.cases = {c["id"]: c for c in read(self.work / "dataset_manifest.json")["cases"]}
        self.signal = lru_cache(maxsize=8)(self._load_signal)

    def _load_signal(self, cid):
        entry = self.manifest[cid]
        case = self.cases[cid]
        raw, t, y, v, e = load_arrays(self.work, case, read(entry["parent"]))
        if entry["arrays_path"]:
            with np.load(entry["arrays_path"], allow_pickle=False) as data:
                y, v, e = (data[k].copy() for k in ("values", "valid", "estimated"))
        if (
            array_hash(y, t, v, e) != entry["hash"]
            or case["fs"] != FS
            or not np.allclose(np.diff(t), 1 / FS, atol=1e-6, rtol=0)
        ):
            raise ValueError(f"Release hash or sampling grid mismatch for {cid}")
        return t, raw, y, v, e


def cluster_gaps(spans):
    """Bridge nearby exclusions in one inference call, retaining observed islands in the exported result."""
    clusters = []
    for a, b in spans:
        if clusters and a - clusters[-1][-1][1] < MIN_SIDE:
            clusters[-1].append((a, b))
        else:
            clusters.append([(a, b)])
    return clusters


def prepare(delivery, out, joint_id=None):
    release = Release(delivery)
    rng = np.random.default_rng(SEED)
    kept = defaultdict(list)
    for table in ("RR_PPI全部含状态.csv", "逐拍压力全部含状态.csv"):
        for r in read_csv(delivery / table):
            if r["quality"] == "keep":
                kept[r["case"]].append((int(r["a"]), int(r["b"])))
    flanks, requests, truth = {}, {}, {}
    examples, actual = [], []
    eligible = defaultdict(dict)
    source_origin = min(release.signal(c)[0][0] for c in release.cases)

    def register(values, horizon):
        key = f"F{len(requests) + 1:04d}"
        flanks[key] = values.copy()
        requests[key] = horizon
        return key

    for cid, c in release.cases.items():
        t, raw, y, v, e = release.signal(cid)
        good = v & ~e & np.isfinite(y)
        block = tuple(c["rows"])
        public = good & (raw == y)
        keep = np.zeros(len(y), bool)
        for a, b in kept[cid]:
            keep[a:b] = True
        # All target values and all context samples of the benchmark come from observed retained support.
        zones = [
            (a + CONTEXT + max(LENGTHS) // 2, b - CONTEXT - (max(LENGTHS) - max(LENGTHS) // 2))
            for a, b in intervals(keep & public)
            if b - a >= 2 * CONTEXT + max(LENGTHS)
        ]
        if zones and not release.manifest[cid]["block_hold"]:
            native = read(release.manifest[cid]["probe_path"])["events"]
            peaks = [r["sample"] for r in native if any(start <= r["sample"] <= end for start, end in zones)]
            if peaks:
                eligible[c["task"]][block] = (cid, zones, peaks)
        # A source acquisition discontinuity has no samples to synthesize; work only within this recorded block.
        omitted = intervals(~good)
        clusters = cluster_gaps(omitted)
        for i, spans in enumerate(clusters):
            a, b = spans[0][0], spans[-1][1]
            left = max(0, a - CONTEXT, clusters[i - 1][-1][1] if i else 0)
            right = min(len(y), b + CONTEXT, clusters[i + 1][0][0] if i + 1 < len(clusters) else len(y))
            assert good[left:a].all() and good[b:right].all()
            ql = (
                register(y[left:a], b - a) if a - left >= MIN_SIDE and not release.manifest[cid]["block_hold"] else None
            )
            qr = (
                register(y[b:right][::-1], b - a)
                if right - b >= MIN_SIDE and not release.manifest[cid]["block_hold"]
                else None
            )
            actual.append(
                dict(
                    id=f"A{len(actual) + 1:02d}",
                    case=cid,
                    channel=c["task"],
                    a=a,
                    b=b,
                    spans=spans,
                    ql=ql,
                    qr=qr,
                    left_available=a - left,
                    right_available=right - b,
                    seconds_from_origin=round(float(t[a] - source_origin), 4),
                )
            )
    shared = sorted(set.intersection(*(set(eligible[k]) for k in CHANNELS)))
    if len(shared) < N_BLOCKS:
        raise ValueError(f"Need {N_BLOCKS} common retained blocks, found {len(shared)}")
    selected = rng.choice(len(shared), size=N_BLOCKS, replace=False)
    for task in CHANNELS:
        for i, idx in enumerate(selected):
            cid, zones, peaks = eligible[task][shared[idx]]
            if i % 2 == 0:
                mid = int(rng.choice(peaks))
                selection = "native_peak_center"
            else:
                positions = np.concatenate([np.arange(start, end + 1) for start, end in zones])
                mid = int(rng.choice(positions))
                selection = "uniform"
            _, raw, y, v, e = release.signal(cid)
            for length in LENGTHS:
                a = mid - length // 2
                b = a + length
                assert (v & ~e)[a - CONTEXT : b + CONTEXT].all() and np.array_equal(y[a:b], raw[a:b])
                name = f"B{len(examples) + 1:04d}"
                ql = register(y[a - CONTEXT : a], length)
                qr = register(y[b : b + CONTEXT][::-1], length)
                truth[name] = raw[a:b].copy()
                examples.append(dict(id=name, case=cid, channel=task, a=a, b=b, ql=ql, qr=qr, selection=selection))
    joint = []
    if joint_id:
        loc = read(delivery.parent / "post-locations.json")[joint_id]
        for cid, c in release.cases.items():
            if c["rows"] != release.cases[loc["case"]]["rows"]:
                continue
            t, _, y, v, e = release.signal(cid)
            a, b = map(int, t.searchsorted(source_origin + np.array([loc["start"], loc["end"]]) - 1e-6))
            span = next(((lo, hi) for lo, hi in intervals(v & ~e) if lo <= a < b <= hi), None)
            if not span:
                continue
            left = max(span[0], a - CONTEXT)
            right = min(span[1], b + CONTEXT)
            ql = register(y[left:a], b - a) if a - left >= MIN_SIDE else None
            qr = register(y[b:right][::-1], b - a) if right - b >= MIN_SIDE else None
            if ql or qr:
                joint.append(
                    dict(id=joint_id, case=cid, channel=c["task"], a=a, b=b, ql=ql, qr=qr, record_start_s=loc["start"])
                )
    inp = out / "model_input"
    inp.mkdir(parents=True, exist_ok=True)
    private = out / "scoring"
    private.mkdir(exist_ok=True)
    np.savez_compressed(inp / "flanks.npz", **flanks)
    write(inp / "requests.json", requests)
    np.savez_compressed(private / "reference.npz", **truth)
    write(
        private / "locations.json",
        dict(
            benchmark=examples,
            previous_exclusions=actual,
            illustrative_joint_event=joint,
            seed=SEED,
            context=CONTEXT,
            min_side=MIN_SIDE,
            fs=FS,
            lengths=LENGTHS,
            shared_blocks_eligible=len(shared),
            selected_blocks=N_BLOCKS,
            grouping="Nearby unavailable spans share external context; only original unavailable samples are replaced.",
        ),
    )
    print(
        "evaluation gaps",
        len(examples),
        "actual spans",
        sum(len(r["spans"]) for r in actual),
        "actual clusters",
        len(actual),
        "model inputs",
        len(requests),
        flush=True,
    )


def repeat_cycle(context, length):
    """Repeat the local cycle estimated solely from the observed flank. Use level hold for a flat/unmatched flank."""
    lags = np.arange(round(0.38 * FS), min(round(1.5 * FS), len(context) // 2) + 1)
    size = min(len(context) - lags[-1], 3 * FS)
    tail = context[-size:] - context[-size:].mean()
    norm = np.linalg.norm(tail)
    correlations = []
    for lag in lags:
        other = context[-size - lag : -lag]
        other = other - other.mean()
        denom = norm * np.linalg.norm(other)
        correlations.append(float(other @ tail / denom) if denom > 0 else 0.0)
    scores = np.array(correlations)
    if scores.max() < 0.3:
        return np.full(length, context[-1])
    maxima = np.flatnonzero((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
    plausible = maxima[scores[maxima] >= 0.90 * scores.max()]
    lag = lags[plausible[0] if len(plausible) else np.argmax(scores)]
    return np.resize(context[-lag:], length)


def basic_bridge(left, right, length):
    if left is None:
        return np.full(length, right[-1])
    if right is None:
        return np.full(length, left[-1])
    return np.linspace(left[-1], right[-1], length + 2)[1:-1]


def inspect_probe(task, left, middle, right):
    before = np.array([]) if left is None else left
    after = np.array([]) if right is None else right[::-1]
    x = np.concatenate([before, middle, after])
    p = probe(x, np.arange(len(x)) / FS, FS, np.ones(len(x), bool), np.zeros(len(x), bool), task)
    a = len(before)
    peaks = [z["sample"] - a for z in p["events"] if a <= z["sample"] < a + len(middle)]
    # Same reference feet are used to isolate absolute-pressure distortion from timing error.
    cycles = [
        (z["start_sample"] - a, z["stop_sample"] - a)
        for z in p["pressures"]
        if a <= z["start_sample"] < z["stop_sample"] < a + len(middle)
    ]
    return peaks, cycles


def pressure_stats(x, a, b):
    seg = x[a:b]
    mean = (x[a:b] + x[a + 1 : b + 1]).mean() / 2
    return np.array([seg.max(), seg.min(), mean])


def evaluate(examples, truth, contexts, predictions):
    rows = []
    for z in examples:
        length = z["b"] - z["a"]
        y = truth[z["id"]]
        L, R = contexts[z["ql"]], contexts[z["qr"]]
        fl, fr = predictions[z["ql"]], predictions[z["qr"]][::-1]
        cl, cr = repeat_cycle(L, length), repeat_cycle(R, length)[::-1]
        for side in ("both", "left_only", "right_only"):
            lctx = L if side != "right_only" else None
            rctx = R if side != "left_only" else None
            models = {
                "linear_or_hold": basic_bridge(lctx, rctx, length),
                "repeat_cycle": bridge(cl if lctx is not None else None, cr if rctx is not None else None),
                "timesfm": bridge(fl if lctx is not None else None, fr if rctx is not None else None),
            }
            if length >= FS:
                gold, cycles = inspect_probe(z["channel"], lctx, y, rctx)
            for name, pred in models.items():
                err = pred - y
                row = dict(
                    id=z["id"],
                    case=z["case"],
                    channel=z["channel"],
                    selection=z["selection"],
                    side=side,
                    samples=length,
                    seconds=length / FS,
                    method=name,
                    mae=float(abs(err).mean()),
                    rmse=float(np.sqrt(np.mean(err**2))),
                    max_abs_error=float(abs(err).max()),
                    tp=None,
                    fp=None,
                    fn=None,
                    abp_cycles=0,
                    sbp_abs_sum=0.0,
                    dbp_abs_sum=0.0,
                    map_abs_sum=0.0,
                )
                if length >= FS:
                    peaks, _ = inspect_probe(z["channel"], lctx, pred, rctx)
                    pairs = match_events(peaks, gold, 5 if z["channel"] == "ECG" else 10)
                    row.update(tp=len(pairs), fp=len(peaks) - len(pairs), fn=len(gold) - len(pairs))
                    if cycles:
                        sums = sum(abs(pressure_stats(pred, a, b) - pressure_stats(y, a, b)) for a, b in cycles)
                        row.update(
                            abp_cycles=len(cycles),
                            sbp_abs_sum=float(sums[0]),
                            dbp_abs_sum=float(sums[1]),
                            map_abs_sum=float(sums[2]),
                        )
                rows.append(row)
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["channel"], r["samples"], r["side"]].append(r)
    summary = []
    for (task, n, side), group in sorted(groups.items()):
        sub = {
            k: sorted([r for r in group if r["method"] == k], key=lambda r: r["id"])
            for k in ("linear_or_hold", "repeat_cycle", "timesfm")
        }
        ordered = [r["id"] for r in sub["timesfm"]]
        assert all([r["id"] for r in v] == ordered for v in sub.values())
        maes = {k: np.array([r["mae"] for r in vals]) for k, vals in sub.items()}
        for k, part in sub.items():
            tp = sum(r["tp"] or 0 for r in part)
            fp = sum(r["fp"] or 0 for r in part)
            fn = sum(r["fn"] or 0 for r in part)
            cycles = sum(r["abp_cycles"] for r in part)
            summary.append(
                dict(
                    channel=task,
                    samples=n,
                    seconds=n / FS,
                    side=side,
                    n_gaps=len(part),
                    method=k,
                    median_gap_mae=float(np.median(maes[k])),
                    mean_gap_mae=float(np.mean(maes[k])),
                    p90_gap_mae=float(np.percentile(maes[k], 90)),
                    wins_vs_linear_or_hold=int(sum(maes[k] < maes["linear_or_hold"])),
                    wins_vs_cycle=int(sum(maes[k] < maes["repeat_cycle"])),
                    probe_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                    tp=tp if n >= FS else None,
                    fp=fp if n >= FS else None,
                    fn=fn if n >= FS else None,
                    abp_cycles=cycles,
                    sbp_mae=sum(r["sbp_abs_sum"] for r in part) / cycles if cycles else None,
                    dbp_mae=sum(r["dbp_abs_sum"] for r in part) / cycles if cycles else None,
                    map_mae=sum(r["map_abs_sum"] for r in part) / cycles if cycles else None,
                )
            )
    return summary


def materialize(release, clusters, pred, out):
    by_case = defaultdict(list)
    for z in clusters:
        by_case[z["case"]].append(z)
    folder = out / "候选波形"
    folder.mkdir(exist_ok=True)
    inventory = {}
    ranges = []
    for cid, entry in release.manifest.items():
        item = dict(
            previous_release_hash=entry["hash"],
            previous_manifest=str((release.delivery / "波形与探针清单.json").resolve()),
            candidate=None,
            predicted_samples=0,
            unfilled_samples=0,
            clinical_reference=None,
        )
        inventory[cid] = item
        if cid not in by_case:
            continue
        t, raw, y, v, e = release.signal(cid)
        result = y.copy()
        filled = np.zeros(len(y), bool)
        eligible = v & ~e & np.isfinite(y)
        for z in by_case[cid]:
            ql, qr = z["ql"], z["qr"]
            candidate = bridge(pred[ql] if ql else None, pred[qr][::-1] if qr else None) if ql or qr else None
            for a, b in z["spans"]:
                if candidate is not None:
                    result[a:b] = candidate[a - z["a"] : b - z["a"]]
                    filled[a:b] = True
                ranges.append(
                    dict(
                        cluster=z["id"],
                        case=cid,
                        channel=z["channel"],
                        a=a,
                        b=b,
                        samples=b - a,
                        record_start_s=round(z["seconds_from_origin"] + (a - z["a"]) / FS, 4),
                        context_left_samples=z["left_available"] if ql else 0,
                        context_right_samples=z["right_available"] if qr else 0,
                        filled=candidate is not None,
                        estimated_only=True,
                        reference_available=False,
                    )
                )
        estimate = e | filled
        valid = v | filled
        task_support = valid & ~estimate & np.isfinite(result)
        assert np.array_equal(result[~filled], y[~filled], equal_nan=True) and not np.any(filled & eligible)
        assert np.array_equal(task_support, eligible)
        file = folder / (cid + ".npz")
        np.savez_compressed(
            file,
            time=t,
            raw_values=raw,
            previous_values=y,
            candidate_values=result,
            timesfm_filled=filled,
            valid=valid,
            estimated=estimate,
            task_support=task_support,
            previous_valid=v,
            previous_estimated=e,
        )
        item.update(
            candidate=str(file.relative_to(out)),
            sha256=file_hash(file),
            predicted_samples=int(filled.sum()),
            unfilled_samples=int((~eligible & ~filled).sum()),
        )
    if ranges:
        save_csv(out / "实际缺口逐段结果.csv", ranges)
    write(out / "候选波形清单.json", inventory)
    return ranges


def draw_figures(release, source, truth, contexts, pred, out):
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    units = {"ECG": "原始单位", "PPG": "原始单位", "ABP": "mmHg"}
    with rc_context({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "pdf.fonttype": 42}):
        for length in (FS, 5 * FS):
            fig = Figure(figsize=(12.3, 7.6))
            FigureCanvasAgg(fig)
            axes = fig.subplots(3, 2, sharex=True, sharey="row")
            for i, channel in enumerate(CHANNELS):
                z = next(
                    r
                    for r in source["benchmark"]
                    if r["channel"] == channel and r["selection"] == "native_peak_center" and r["b"] - r["a"] == length
                )
                y = truth[z["id"]]
                L, R = contexts[z["ql"]], contexts[z["qr"]]
                xx = np.arange(len(y)) / FS
                for j, side in enumerate(("both", "left_only")):
                    ax = axes[i, j]
                    right = R if side == "both" else None
                    fl = pred[z["ql"]]
                    point = bridge(fl, pred[z["qr"]][::-1] if side == "both" else None)
                    flank = FS
                    ax.axvspan(0, length / FS, color="#ffdf9c", alpha=0.36, lw=0)
                    ax.plot(np.arange(-flank, 0) / FS, L[-flank:], c="#384e60", lw=0.9)
                    if right is not None:
                        ax.plot((len(y) + np.arange(flank)) / FS, R[::-1][:flank], c="#384e60", lw=0.9)
                    ax.plot(xx, y, color="#344e60", lw=1.35, label="被遮住的原始观测")
                    ax.plot(xx, basic_bridge(L, right, length), color="#9f839d", lw=0.9, ls="--", label="直线或末值")
                    ax.plot(xx, point, color="#ce6229", lw=1.05, label="TimesFM 估计")
                    ax.spines[["right", "top"]].set_visible(False)
                    if i == 0:
                        ax.set_title("两侧都有观测" if j == 0 else "仅用左侧观测", loc="left", fontsize=11)
                    if j == 0:
                        ax.set_ylabel(channel + "\n" + units[channel])
            fig.suptitle(f"藏起 {length / FS:g} 秒原始波形，再与修补结果对照", x=0.08, y=0.965, ha="left", fontsize=16)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.073, 0.91), ncol=3, frameon=False)
            fig.text(0.08, 0.82, "每路展示按固定种子抽到的首个峰中心样例。橙色背景是模型未看的区间。", fontsize=9)
            for ax in axes[-1]:
                ax.set_xlabel("相对缺口起点（秒）")
            fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.75, wspace=0.07, hspace=0.22)
            fig.savefig(out / f"遮挡对照_{length // FS}秒.png", dpi=140)
            fig.clear()
        peerids = {(tuple(c["rows"]), c["task"]): cid for cid, c in release.cases.items()}
        if not source["previous_exclusions"]:
            return
        largest = max(source["previous_exclusions"], key=lambda z: max(b - a for a, b in z["spans"]))
        with PdfPages(out / "既有缺口补值对照.pdf") as pdf:
            for z in source["previous_exclusions"]:
                cid = z["case"]
                a, b = z["a"], z["b"]
                c = release.cases[cid]
                t, raw, prev, _, _ = release.signal(cid)
                with np.load(out / "候选波形" / (cid + ".npz"), allow_pickle=False) as block:
                    y = block["candidate_values"]
                    mask = block["timesfm_filled"]
                lo = max(0, a - 3 * FS)
                hi = min(len(t), b + 3 * FS)
                xx = t[lo:hi] - t[a]
                fig = Figure(figsize=(11.0, 7.6))
                FigureCanvasAgg(fig)
                axes = fig.subplots(4, 1, sharex=True)
                channel = z["channel"]
                for j, (values, txt) in enumerate(((raw, "原始 " + channel), (y, "候选 " + channel))):
                    ax = axes[j]
                    ax.plot(xx, values[lo:hi], color="#36576f", lw=1)
                    for start, end in z["spans"]:
                        ax.axvspan((start - a) / FS, (end - a) / FS, color="#fad99c", alpha=0.55, lw=0)
                        if j == 1 and mask[start:end].all():
                            il, ih = max(0, start - 1), min(len(y), end + 1)
                            ax.plot(t[il:ih] - t[a], y[il:ih], color="#c35b25", lw=1.45)
                            if end - start <= 3:
                                ax.plot(t[start:end] - t[a], y[start:end], ".", color="#c35b25", ms=4)
                    ax.set_ylabel(txt + "\n" + units[channel])
                overlay = np.r_[raw[lo:hi], prev[lo:hi], y[lo:hi]]
                f = overlay[np.isfinite(overlay)]
                pad = max(0.1 * np.ptp(f), 1e-6)
                for ax in axes[:2]:
                    ax.set_ylim(f.min() - pad, f.max() + pad)
                for ax, kind in zip(axes[2:], (k for k in CHANNELS if k != channel), strict=True):
                    peer = peerids.get((tuple(c["rows"]), kind))
                    if peer:
                        pt, _, py, _, _ = release.signal(peer)
                        on = (pt >= t[lo]) & (pt <= t[hi - 1])
                        ax.plot(pt[on] - t[a], py[on], color="#637d81", lw=0.85)
                    ax.set_ylabel(kind + "\n" + units[kind])
                for ax in axes:
                    ax.spines[["right", "top"]].set_visible(False)
                    ax.set_xlim(xx[0], xx[-1])
                minute = int(z["seconds_from_origin"] // 60) + 1
                sec = z["seconds_from_origin"] % 60
                side = (
                    "双侧融合"
                    if z["ql"] and z["qr"]
                    else "仅左侧前推"
                    if z["ql"]
                    else "仅右侧反推"
                    if z["qr"]
                    else "上下文不足"
                )
                fig.suptitle(
                    f"第 {minute} 分钟 {sec:.3f} 秒　·　{z['id']}　·　{side}", ha="left", x=0.1, y=0.98, fontsize=16
                )
                fig.text(
                    0.10,
                    0.90,
                    "黄色为原先停用或估计的样本，橙线为本地模型估计。下方两个通道供读图对照。原始缺口内缺少干净参照。",
                    fontsize=9,
                )
                axes[-1].set_xlabel("相对本组缺口起点（秒）")
                fig.subplots_adjust(left=0.1, right=0.98, bottom=0.09, top=0.85, hspace=0.27)
                pdf.savefig(fig)
                if z["id"] == largest["id"]:
                    fig.savefig(out / "最长实际缺口_候选对照.png", dpi=140)
                fig.clear()


def joint_preview(release, events, pred, out):
    """Illustrate what filling a disputed measurement would do, outside the actual repair inventory."""
    if not events:
        return
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def measurements(p, a, b):
        if p["task"] == "ABP":
            return [
                dict(samples=[r["start_sample"], r["stop_sample"]], sbp=r["sbp"], dbp=r["dbp"], map=r["mean_pressure"])
                for r in p["pressures"]
                if r["start_sample"] < b and a < r["stop_sample"]
            ]
        peaks = p["events"]
        return [
            round(r["interval_s"], 3)
            for r in p["intervals"]
            if peaks[r["left_event_id"]]["sample"] < b and a < peaks[r["right_event_id"]]["sample"]
        ]

    summary = {}
    with rc_context({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False}):
        fig = Figure(figsize=(10.8, 6.6))
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(events), 1, sharex=True, squeeze=False)
        for (ax,), z in zip(axes, events, strict=True):
            a, b = z["a"], z["b"]
            cid = z["case"]
            task = z["channel"]
            key = z["id"]
            t, _, y, v, e = release.signal(cid)
            left = pred[z["ql"]] if z["ql"] else None
            right = pred[z["qr"]][::-1] if z["qr"] else None
            copy = y.copy()
            copy[a:b] = bridge(left, right)
            pr = probe(copy, t, FS, v, e, task)
            prior = read(release.manifest[cid]["probe_path"])
            summary[task] = dict(
                id=key,
                a=a,
                b=b,
                previous=measurements(prior, a, b),
                if_estimates_were_accepted=measurements(pr, a, b),
                previous_landmarks=[r["sample"] for r in prior["events"] if a <= r["sample"] < b],
                estimated_landmarks=[r["sample"] for r in pr["events"] if a <= r["sample"] < b],
            )
            lo = max(0, a - 2 * FS)
            hi = min(len(y), b + 2 * FS)
            offset = z["record_start_s"] % 60
            grid = t - t[a] + offset
            ax.axvspan(grid[a], grid[b - 1] + 1 / FS, fc="#ffdf9c", alpha=0.44, lw=0)
            ax.plot(grid[lo:hi], y[lo:hi], color="#284f66", lw=1.35, label="上版保留的实际波形")
            ax.plot(grid[a:b], copy[a:b], color="#c35b25", lw=1.25, label="假设挖掉后由 TimesFM 填入")
            ax.set_ylabel(task + "\n" + ("mmHg" if task == "ABP" else "原始单位"))
            ax.spines[["right", "top"]].set_visible(False)
        axes[-1, 0].set_xlabel("本分钟内时间（秒）")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0.095, 0.92))
        minute = int(events[0]["record_start_s"] // 60) + 1
        fig.suptitle(
            f"第 {minute} 分钟　{key}　｜　有争议的波形被替换时会怎样", x=0.10, ha="left", y=0.975, fontsize=16
        )
        fig.text(
            0.10, 0.81, "黄色是人为遮挡区。每路模型只看本通道两侧，蓝线在遮挡区内的观测仅用于事后对照。", fontsize=9
        )
        fig.subplots_adjust(left=0.1, right=0.98, top=0.76, bottom=0.10, hspace=0.22)
        fig.savefig(out / f"{key}_如果直接补波.png", dpi=145)
    write(out / "联合疑点假设补波的探针变化.json", summary)


def report(delivery, out):
    release = Release(delivery)
    source = read(out / "scoring/locations.json")
    model = read(out / "predictions/model.json")
    assert model["input_sha256"] == file_hash(out / "model_input/flanks.npz")
    assert model["lengths_sha256"] == file_hash(out / "model_input/requests.json")
    assert model["outputs_sha256"] == file_hash(out / "predictions/forecasts.npz")
    with (
        np.load(out / "predictions/forecasts.npz", allow_pickle=False) as fp,
        np.load(out / "model_input/flanks.npz", allow_pickle=False) as cx,
        np.load(out / "scoring/reference.npz", allow_pickle=False) as ref,
    ):
        pred = dict(fp)
        contexts = dict(cx)
        truth = dict(ref)
    assert set(pred) == set(contexts)
    rows = evaluate(source["benchmark"], truth, contexts, pred)
    stats = aggregate(rows)
    ranges = materialize(release, source["previous_exclusions"], pred, out)
    save_csv(out / "遮挡测试逐窗.csv", rows)
    save_csv(out / "遮挡测试按通道和长度.csv", stats)
    write(
        out / "汇总.json",
        dict(
            benchmark_gaps=len(truth),
            geometry_cases=len(rows) // 3,
            actual_ranges=len(ranges),
            actual_samples=sum(z["samples"] for z in ranges),
            candidate_samples=sum(z["samples"] for z in ranges if z["filled"]),
            model=model,
            metrics=stats,
        ),
    )
    draw_figures(release, source, truth, contexts, pred, out)
    joint_preview(release, source.get("illustrative_joint_event", []), pred, out)
    print("verified candidates, references and scored gaps", len(truth), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "report"))
    ap.add_argument("delivery", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--inspect-joint", help="Additionally simulate masking one named joint-review event in a separate preview."
    )
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.delivery, args.out, args.inspect_joint)
    else:
        report(args.delivery, args.out)


if __name__ == "__main__":
    main()
