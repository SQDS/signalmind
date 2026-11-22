"""Compare the exported MIMIC checkpoint to TimesFM 3.0 using the public forecasting API."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from timesfm3 import TimesFM3Forecaster

from eval.timesfm3_mimic_finetune import (
    BASE,
    BASE_SHA,
    CHANNELS,
    CONTEXT,
    FS,
    HORIZONS,
    crop_values,
    eval_schedule,
    load_bank,
    write_json,
)
from eval.timesfm_gap_forecast import file_hash
from eval.timesfm_gap_trial import save_csv
from tsmcp.cleaning.tasks import match_events
from tsmcp.plugins.ecg.rpeaks import detect_r_peaks


def predictions(forecaster, waves):
    contexts, peers = [], []
    for sample in waves:
        for channel in range(3):
            others = [c for c in range(3) if c != channel]
            contexts.append(sample[channel, :CONTEXT].copy())
            peers.append(sample[others].copy())
    horizon = waves.shape[-1] - CONTEXT
    with torch.inference_mode():
        predictions = list(
            forecaster.predict_batch(contexts, horizon=horizon, past_future_covariates=peers, use_znorm=True)
        )
    result = np.stack([p.forecast for p in predictions]).reshape(len(waves), 3, horizon)
    if not np.isfinite(result).all():
        raise ArithmeticError("Nonfinite public-API prediction")
    return result


def scores(context, truth, estimate):
    error = float(np.abs(estimate - truth).mean())
    tail = min(len(truth), 5 * FS)
    swing = np.ptp(np.percentile(truth[-tail:], [1, 99]))
    corr = np.corrcoef(truth, estimate)[0, 1] if min(truth.std(), estimate.std()) > 0 else None
    return {
        "mae": error,
        "normalized_mae": error / max(float(context.std()), 1e-7),
        "trace_r": float(corr) if corr is not None else None,
        "tail_amplitude_ratio": float(np.ptp(np.percentile(estimate[-tail:], [1, 99])) / swing) if swing > 0 else None,
    }


def rpeak_counts(context, estimate, peaks, start):
    # The detector excludes its last 0.16 seconds, so score both lists on the same support.
    stop = len(estimate) - round(0.16 * FS)
    gold = peaks - (start + CONTEXT)
    gold = gold[(gold >= 0) & (gold < stop)]
    found = detect_r_peaks(np.concatenate((context, estimate)), FS) - CONTEXT
    found = found[(found >= 0) & (found < stop)]
    tp = len(match_events(found, gold, tolerance=5))
    return {"peak_tp": tp, "peak_fp": int(len(found) - tp), "peak_fn": int(len(gold) - tp)}


def summarize(rows):
    result = []
    for horizon in HORIZONS:
        for channel in CHANNELS:
            group = [r for r in rows if r["seconds"] == horizon // FS and r["channel"] == channel]
            record = {"seconds": horizon // FS, "channel": channel, "patients": len(group)}
            for arm in ("base", "tuned"):
                for measure in ("mae", "normalized_mae", "trace_r", "tail_amplitude_ratio"):
                    key = f"{arm}_{measure}"
                    values = [r[key] for r in group if r[key] is not None]
                    record[key] = float(np.median(values)) if values else None
            record["improved_patients"] = sum(r["tuned_mae"] < r["base_mae"] for r in group)
            record["median_mae_change_pct"] = (record["tuned_mae"] / record["base_mae"] - 1) * 100
            record["base_peak_f1"] = record["tuned_peak_f1"] = record["observed_peak_f1"] = None
            if channel == "ECG":
                for arm in ("base", "tuned", "observed"):
                    tp, fp, fn = (sum(r[f"{arm}_peak_{m}"] for r in group) for m in ("tp", "fp", "fn"))
                    record[f"{arm}_peak_f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
            result.append(record)
    return result


def plot_examples(examples, out, *, curves=None, panels=None, share_y=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 2, figsize=(13.5, 7.5), layout="constrained", sharey="row" if share_y else False)
    if curves is None:
        curves = (
            ("Observed", "#303438", 1.9, "-"),
            ("TimesFM 3.0", "#cb7527", 1.4, "--"),
            ("MIMIC fine-tuned", "#007b91", 1.3, "-"),
        )
    if panels is None:
        panels = ((5 * FS, 0, 5 * FS, "5 s gap"), (60 * FS, 55 * FS, 60 * FS, "Last 5 s of a 60 s gap"))
    for col, (h, start, stop, title) in enumerate(panels):
        data = examples[h]
        indices = np.arange(start, stop)
        for i, channel in enumerate(CHANNELS):
            panel = ax[i, col]
            for key, color, width, style in curves:
                panel.plot(
                    indices / FS, data[key][i, indices], color=color, linewidth=width, linestyle=style, label=key
                )
            panel.set_ylabel("ABP (mmHg)" if channel == "ABP" else f"{channel} (record units)")
            panel.grid(axis="y", alpha=0.22)
            panel.spines[["right", "top"]].set_visible(False)
            if i == 0:
                panel.set_title(title, loc="left", weight="bold")
            if i == 2:
                panel.set_xlabel("Time from start of gap (s)")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside upper center", ncols=len(curves), frameon=False)
    fig.savefig(out, dpi=145)
    plt.close(fig)


def model_card(root, summary):
    protocol = json.loads((root / "protocol.json").read_text())
    export = json.loads((root / "export.json").read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    train = [p for p in manifest["patients"] if p["split"] == "train" and p["clips"]]
    table = [
        f"| {r['channel']} | {r['patients']} | {r['base_mae']:.4g} | {r['tuned_mae']:.4g} |"
        for r in summary
        if r["seconds"] == 5
    ]
    lines = [
        "---",
        "base_model: google/timesfm-3.0-pytorch",
        "license: other",
        "license_name: timesfm-non-commercial-license-v1.0",
        "---",
        "",
        "# TimesFM 3.0 · PulseDB MIMIC 三通道修补",
        "",
        "此目录是已合并的 PyTorch 检查点，可由官方 TimesFM 3.0 推理接口直接加载。"
        "每次遮住一个目标通道，使用该路缺口前的上下文和另外两路覆盖缺口的同步实测波形。训练和评估的采样率均为 125 Hz。",
        "",
        "## 加载和使用",
        "",
        "在本机的 timesfm Conda 环境中，将固定版本的 timesfm3 源码目录加入 Python 搜索路径后即可运行。",
        "",
        "```python",
        "import sys",
        "import numpy as np",
        'sys.path.insert(0, "/home/yanghaoxiang24/.cache/signalmind/timesfm3-331c6d3/src")',
        "from timesfm3 import TimesFM3Forecaster",
        "",
        f"model = TimesFM3Forecaster.from_pretrained({str((root / 'checkpoint').resolve())!r},",
        '    device="cpu", local_files_only=True)',
        "",
        "def repair_gap(target_context, peer_a, peer_b):",
        "    context = np.asarray(target_context, dtype=np.float32)",
        "    peers = np.asarray([peer_a, peer_b], dtype=np.float32)",
        "    horizon = peers.shape[1] - context.size",
        "    return model.predict(context, horizon=horizon, past_future_covariates=peers, use_znorm=True).forecast",
        "```",
        "",
        "使用 512 个上下文点可复现本次输入设置。修补 ECG 时传入 PPG、ABP，修补 PPG 时传入 ECG、ABP，"
        "修补 ABP 时传入 ECG、PPG。两路邻道均从目标上下文的起点开始，到缺口末端结束。",
        "",
        "## 训练和评估",
        "",
        f"从现有 PulseDB MIMIC 训练分区的 {len(train):,} 名患者中抽取了 {sum(len(p['clips']) for p in train):,} 段实测波形。"
        "原始字段为 ECG_Record、PPG_Record、ABP_Raw。训练随机选择正向或反向的单侧上下文，轮流遮住三路目标并保留另外两路。"
        "缺口长度为 1、5、10、30 或 60 秒。",
        f"在最后四个 Transformer 层的时间及跨通道注意力上加入 rank-8 LoRA，同时训练输出层，共更新 {protocol['trainable_parameters']:,} 个参数。"
        f"完成 {protocol['epochs']} 轮训练，依据独立验证患者的归一化 MAE 选择第 {export['best_epoch']} 轮并合并权重。",
        "",
        "下表为独立测试患者单侧重建 5 秒缺口的结果。先在每个缺口计算平均绝对误差，再在患者间取中位数。"
        "ECG、PPG 使用原记录单位，ABP 使用 mmHg。",
        "",
        "| 通道 | 测试患者 | 原版 MAE | 微调 MAE |",
        "|---|---:|---:|---:|",
        *table,
        "",
        "测试在实测参考波形上人为遮挡。R 峰指标采用 PulseDB 提供的峰位置，允许 ±40 ms 偏差。"
        "实际丢失片段的逐点精度，以及其他医院数据上的迁移效果，仍需独立验证。"
        " [完整指标](../test-summary.csv) · [波形对照](../test-waveforms.png) · [训练协议](../protocol.json)",
        "",
        "本地权重及微调版本适用 TimesFM 非商业许可。 [许可全文](LICENSE)",
        "",
    ]
    (root / "checkpoint/README.md").write_text("\n".join(lines))


def evaluate(args):
    root = args.run
    export = json.loads((root / "export.json").read_text())
    if (
        file_hash(root / "checkpoint/model.safetensors") != export["sha256"]
        or file_hash(args.base / "model.safetensors") != BASE_SHA
    ):
        raise ValueError("Unexpected base or evaluation checkpoint")
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("high")
    test = load_bank(root / "data", "test")
    batches = eval_schedule(test, "test")
    plan = [
        dict(patient=p, clip=int(slot), crop=int(start), seconds=h // FS)
        for h, jobs in batches
        for p, slot, start, _ in jobs
    ]
    write_json(root / "test-gaps.json", plan)
    models = {
        arm: TimesFM3Forecaster.from_pretrained(
            str(path.resolve()), local_files_only=True, device=args.device, per_core_batch_size=36
        )
        for arm, path in (("base", args.base), ("tuned", root / "checkpoint"))
    }
    rows, examples = evaluate_batches(test, batches, models)
    save_csv(root / "test-paired.csv", rows)
    summary = summarize(rows)
    save_csv(root / "test-summary.csv", summary)
    write_json(
        root / "test-summary.json",
        dict(
            checkpoint_sha256=export["sha256"],
            patients=len({r["patient"] for r in rows}),
            target_gaps=len(rows),
            groups=summary,
        ),
    )
    plot_examples(examples, root / "test-waveforms.png")
    model_card(root, summary)
    for row in summary:
        print(row["seconds"], row["channel"], row["patients"], row["base_mae"], row["tuned_mae"], flush=True)


def evaluate_batches(bank, batches, models, *, display_names=None):
    if display_names is None:
        display_names = {"base": "TimesFM 3.0", "tuned": "MIMIC fine-tuned"}
    rows, examples = [], {}
    for i, (h, jobs) in enumerate(batches, 1):
        values = crop_values(bank, jobs, h)
        outputs = {arm: predictions(model, values) for arm, model in models.items()}
        if h in (5 * FS, 60 * FS) and h not in examples:
            examples[h] = {
                "Observed": values[0, :, CONTEXT:],
                **{name: outputs[arm][0] for arm, name in display_names.items()},
            }
        for j, (patient, slot, start, _) in enumerate(jobs):
            for c, channel in enumerate(CHANNELS):
                truth = values[j, c, CONTEXT:]
                context = values[j, c, :CONTEXT]
                row = {"patient": patient, "clip": slot, "crop": start, "seconds": h // FS, "channel": channel}
                for arm, pred in outputs.items():
                    measured = scores(context, truth, pred[j, c])
                    if c == 0:
                        measured.update(rpeak_counts(context, pred[j, c], bank[patient][slot][1], start))
                    row.update({f"{arm}_{k}": v for k, v in measured.items()})
                if c == 0:
                    counts = rpeak_counts(context, truth, bank[patient][slot][1], start)
                    row.update({f"observed_{k}": v for k, v in counts.items()})
                rows.append(row)
        if i % 20 == 0 or i == len(batches):
            print(f"Compared {i}/{len(batches)} batches", flush=True)
    return rows, examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--device", default="cuda:0")
    evaluate(parser.parse_args())
