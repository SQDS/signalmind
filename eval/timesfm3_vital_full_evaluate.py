"""Lock Vital test gaps and compare stock, MIMIC-only and full Vital-only TimesFM checkpoints."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from timesfm3 import TimesFM3Forecaster

from eval.timesfm3_mimic_data import FIELDS, INVENTORY, prepare_patient, seed_for
from eval.timesfm3_mimic_evaluate import evaluate_batches, plot_examples, predictions
from eval.timesfm3_mimic_finetune import (
    BASE,
    BASE_SHA,
    CHANNELS,
    CONTEXT,
    FS,
    HORIZONS,
    adapters,
    choose_crop,
    eval_schedule,
    load_bank,
    make_batches,
)
from eval.timesfm3_vital_full_batches import cache_path
from eval.timesfm3_vital_full_data import RAW, VITAL, atomic_json, json_file
from eval.timesfm_gap_forecast import file_hash
from eval.timesfm_gap_trial import save_csv

MIMIC_RUN = Path("eval/runs/timesfm3-mimic-three-channel-20261121")
LABELS = {"base": "TimesFM 3.0", "mimic": "MIMIC fine-tuned", "vital": "Vital fine-tuned"}
CURVES = (
    ("Observed", "#283034", 2.0, "-"),
    (LABELS["base"], "#ca772c", 1.35, "--"),
    (LABELS["mimic"], "#874b92", 1.4, ":"),
    (LABELS["vital"], "#007d88", 1.35, "-"),
)


def plot_mimic_transfer(root, examples, batches):
    short, long = 5 * FS, 60 * FS
    short_patient = next(jobs[0][0] for h, jobs in batches if h == short)
    long_patient = next(jobs[0][0] for h, jobs in batches if h == long)
    plot_examples(
        examples,
        root / "mimic-transfer-waveforms.png",
        curves=CURVES,
        panels=(
            (short, 0, short, f"MIMIC {short_patient} · 5 s gap"),
            (long, long - short, long, f"MIMIC {long_patient} · Last 5 s of a 60 s gap"),
        ),
    )
    plot_examples(
        examples,
        root / "mimic-transfer-60s.png",
        curves=CURVES,
        share_y=True,
        panels=(
            (long, 0, 30 * FS, f"MIMIC {long_patient} · First 30 s of the gap"),
            (long, 30 * FS, long, f"MIMIC {long_patient} · Last 30 s of the same gap"),
        ),
    )


def prepare(root):
    folder = root / "test-data"
    (folder / "test").mkdir(parents=True, exist_ok=True)
    test_ids = sorted(json_file(VITAL / "test/subject_indices.json"))
    training = {r["patient"] for r in json_file(root / "train-manifest.json")["patients"]}
    if training.intersection(test_ids) or set(test_ids).intersection(json_file(VITAL / "val/subject_indices.json")):
        raise ValueError("Vital test patient contamination")
    with ProcessPoolExecutor(max_workers=8) as pool:
        entries = list(
            pool.map(
                partial(prepare_patient, source="vital"),
                [(s, "test", RAW, INVENTORY, folder) for s in test_ids],
                chunksize=2,
            )
        )
    atomic_json(folder / "manifest.json", dict(source=str(RAW), fields=FIELDS, fs=FS, patients=entries))
    bank = load_bank(folder, "test")
    plan = []
    for h in HORIZONS:
        eligible = [s for s, clips in bank.items() if max(x.shape[-1] for x, _ in clips) >= CONTEXT + h]
        eligible.sort(key=lambda s: seed_for(f"vital-full/test/order/{h}/{s}"))
        jobs = [
            choose_crop(bank, s, h, np.random.default_rng(seed_for(f"vital-full/test/crop/{h}/{s}")), False)
            for s in eligible
        ]
        plan.extend({"h": n, "jobs": j} for n, j in make_batches(h, jobs))
    atomic_json(root / "test-gaps.json", plan)
    atomic_json(
        root / "test-protocol.json",
        dict(
            gaps_sha256=file_hash(root / "test-gaps.json"),
            selection="One deterministic gap per eligible test patient and duration. Evaluate the three target channels separately.",
            fs=FS,
            context=CONTEXT,
            horizons=HORIZONS,
            patients=len(bank),
            sources=list(LABELS),
            gap_counts={str(h // FS): sum(len(b["jobs"]) for b in plan if b["h"] == h) for h in HORIZONS},
        ),
    )


def summarize_three(rows):
    result = []
    for h in HORIZONS:
        for channel in CHANNELS:
            group = [r for r in rows if r["seconds"] == h // FS and r["channel"] == channel]
            if not group:
                raise ValueError("Empty test stratum")
            record = dict(seconds=h // FS, channel=channel, patients=len(group))
            for arm in LABELS:
                for measure in ("mae", "normalized_mae", "trace_r", "tail_amplitude_ratio"):
                    key = f"{arm}_{measure}"
                    values = [r[key] for r in group if r[key] is not None]
                    record[key] = float(np.median(values)) if values else None
            for comparator in ("base", "mimic"):
                record[f"mae_change_pct_vs_{comparator}"] = 100 * (
                    record["vital_mae"] / record[comparator + "_mae"] - 1
                )
                record[f"improved_patients_vs_{comparator}"] = sum(
                    r["vital_mae"] < r[comparator + "_mae"] for r in group
                )
                rng = np.random.default_rng(seed_for(f"vital/test/bootstrap/{h}/{channel}"))
                draws = rng.integers(len(group), size=(2000, len(group)))
                new = np.asarray([r["vital_mae"] for r in group])
                previous = np.asarray([r[comparator + "_mae"] for r in group])
                changes = 100 * (np.median(new[draws], axis=1) / np.median(previous[draws], axis=1) - 1)
                low, high = np.percentile(changes, [2.5, 97.5])
                record[f"paired_ci95_low_vs_{comparator}"] = float(low)
                record[f"paired_ci95_high_vs_{comparator}"] = float(high)
            for arm in (*LABELS, "observed"):
                f1 = None
                if channel == "ECG":
                    tp, fp, fn = (sum(r[f"{arm}_peak_{k}"] for r in group) for k in ("tp", "fp", "fn"))
                    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
                record[arm + "_peak_f1"] = f1
            result.append(record)
    return result


def write_card(root, groups, export):
    source = json_file(root / "train-manifest.json")
    actual = source["totals"]
    lines = [
        "# TimesFM 3.0 · PulseDB Vital 全量三通道修补",
        "",
        f"从 Google TimesFM 3.0 初始化，使用 Vital 训练分区的 {source['split_patients']['train']:,} 名患者。"
        f"原始 {actual['raw']:,} 个 10 秒窗中，{actual['eligible']:,} 个通过三路波形和时间网格检查。"
        f"每轮实际覆盖全部合格窗，共完成 {export['epochs_completed']} 轮，按 Vital 验证集选中第 {export['best_epoch']} 轮。",
        "",
        "三个通道为 ECG_Record、PPG_Record、ABP_Raw，采样率 125 Hz。"
        "每个训练缺口选一个通道为目标，提供该路的 512 个上下文点及另外两路覆盖缺口的实测数据。"
        "每个源窗每轮有至少 64 个采样点参与目标损失。训练更新最后四层的 rank-8 LoRA 及输出层，共 1,065,536 个参数。"
        "三轮按顺序轮换目标通道。",
        "",
        "## 本地调用",
        "",
        "在 timesfm Conda 环境中可用以下方式补 ECG。对于其他目标，按 ECG、PPG、ABP 的原顺序传入剩余两路邻道。",
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
        "def fill_ecg(ecg_before, ppg, abp):",
        "    horizon = len(ppg) - len(ecg_before)",
        "    return model.predict(np.asarray(ecg_before), horizon=horizon,",
        "                         past_future_covariates=np.stack([ppg, abp]), use_znorm=True).forecast",
        "```",
        "",
        "## 独立 Vital 测试",
        "",
        "使用同一固定缺口进行三模型配对。每名患者每个时长最多一个缺口，各通道分别遮挡。"
        "MAE 为缺口内平均绝对误差的患者中位数，ECG、PPG 采用原记录单位，ABP 采用 mmHg。",
        "",
        "| 缺口 | 目标 | 患者 | 原版 MAE | MIMIC 版 MAE | Vital 版 MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in groups:
        if r["seconds"] in (5, 60):
            lines.append(
                f"| {r['seconds']} 秒 | {r['channel']} | {r['patients']} | {r['base_mae']:.4g} | "
                f"{r['mimic_mae']:.4g} | {r['vital_mae']:.4g} |"
            )
    lines += [
        "",
        "这些为有实测参照的遮挡结果，真实掉线段的波形属于模型估计。"
        " [完整指标](../test-summary.csv) · [波形图](../test-waveforms.png) · [训练覆盖核验](../coverage-receipt.json)",
        "",
        "此本地微调权重适用 TimesFM 非商业许可。 [许可全文](LICENSE)",
        "",
    ]
    lines += [
        "MIMIC 留出集使用之前固定的缺口作同条件复核。"
        " [5 秒与 60 秒末段的波形](../mimic-transfer-waveforms.png) · [完整 60 秒](../mimic-transfer-60s.png) · "
        "[MIMIC 指标](../mimic-transfer-summary.csv)",
        "",
    ]
    (root / "checkpoint/README.md").write_text("\n".join(lines))


def verify_export(root, export):
    manifest = json_file(root / "train-manifest.json")
    state = torch.load(root / "training-state.pt", map_location="cpu", weights_only=True)
    with np.load(root / "coverage-plan.npz", allow_pickle=False) as plan:
        visits = state["seen"].numpy()
        if len(visits) != len(plan["counts"]) or not np.all(visits == export["epochs_completed"]):
            raise ValueError("Training receipts do not cover all scheduled examples")
        exposures = int(np.dot(plan["counts"].astype(np.int64), visits.astype(np.int64)))
        if exposures != manifest["totals"]["eligible"] * export["epochs_completed"]:
            raise ValueError("Optimizer row coverage differs from Vital training split")
    epochs = [v for v in state["history"] if v["epoch"]]
    by_channel = np.sum([e["rows_by_target"] for e in epochs], axis=0)
    selected_rows = np.sum([e["rows_by_target"] for e in epochs if e["epoch"] <= export["best_epoch"]], axis=0)
    atomic_json(
        root / "coverage-receipt.json",
        dict(
            train_patients=manifest["split_patients"]["train"],
            raw_rows=manifest["totals"]["raw"],
            eligible_rows=manifest["totals"]["eligible"],
            rows_excluded=manifest["totals"]["excluded"],
            committed_training_groups=int(visits.sum()),
            committed_source_row_visits=exposures,
            channels=CHANNELS,
            all_epochs_supervised_source_rows_by_channel=by_channel.tolist(),
            selected_checkpoint_supervised_source_rows_by_channel=selected_rows.tolist(),
            completed_optimizer_steps=state["steps"],
            selected_epoch=export["best_epoch"],
        ),
    )
    del state
    sample = np.load(cache_path(root), mmap_mode="r")[:, : CONTEXT + 625].copy()[None]
    base = TimesFM3Forecaster.from_pretrained(str(BASE), device="cpu", local_files_only=True, per_core_batch_size=3)
    original = predictions(base, sample)
    adapters(base.model)
    trained = load_file(root / f"epoch-{export['best_epoch']}.safetensors")
    incompatible = base.model.load_state_dict(trained, strict=False)
    if incompatible.unexpected_keys or not all(torch.isfinite(p).all() for p in trained.values()):
        raise ValueError("Invalid best-epoch adaptation")
    expected = predictions(base, sample)
    del base
    merged = TimesFM3Forecaster.from_pretrained(
        str((root / "checkpoint").resolve()), device="cpu", local_files_only=True, per_core_batch_size=3
    )
    actual = predictions(merged, sample)
    scale = np.maximum(sample[:, :, :CONTEXT].std(-1), 1e-7)
    delta = np.abs(actual - expected).max(-1) / scale
    change = np.abs(actual - original).mean(-1) / scale
    if not ((delta < 1e-3).all() and (change > 1e-6).all()):
        raise ValueError(f"Checkpoint merge verification failed: {delta} / {change}")
    atomic_json(
        root / "export-audit.json",
        dict(
            channels=CHANNELS,
            merged_vs_unmerged_max_error_in_context_sd=delta[0].tolist(),
            merged_vs_original_mean_difference_in_context_sd=change[0].tolist(),
        ),
    )


def evaluate(root, device):
    export = json_file(root / "export.json")
    if file_hash(root / "test-gaps.json") != json_file(root / "test-protocol.json")["gaps_sha256"]:
        raise ValueError("Held-out gap plan changed")
    paths = {"base": BASE, "mimic": MIMIC_RUN / "checkpoint", "vital": root / "checkpoint"}
    hashes = {k: file_hash(v / "model.safetensors") for k, v in paths.items()}
    if hashes != dict(base=BASE_SHA, mimic=json_file(MIMIC_RUN / "export.json")["sha256"], vital=export["sha256"]):
        raise ValueError("Checkpoint identity mismatch")
    torch.set_num_threads(3)
    verify_export(root, export)
    torch.set_float32_matmul_precision("high")
    models = {
        k: TimesFM3Forecaster.from_pretrained(
            str(p.resolve()), device=device, local_files_only=True, per_core_batch_size=36
        )
        for k, p in paths.items()
    }
    bank = load_bank(root / "test-data", "test")
    batches = [(b["h"], b["jobs"]) for b in json_file(root / "test-gaps.json")]
    rows, examples = evaluate_batches(bank, batches, models, display_names=LABELS)
    if len({(r["patient"], r["seconds"], r["channel"]) for r in rows}) != len(rows):
        raise ValueError("Repeated test target")
    save_csv(root / "test-paired.csv", rows)
    groups = summarize_three(rows)
    save_csv(root / "test-summary.csv", groups)
    atomic_json(
        root / "test-summary.json",
        dict(
            patients=len({r["patient"] for r in rows}), target_gaps=len(rows), checkpoint_sha256=hashes, groups=groups
        ),
    )
    plot_examples(examples, root / "test-waveforms.png", curves=CURVES)
    write_card(root, groups, export)
    print("Finished three-way public-API test", len(rows), "target gaps", flush=True)
    mimic_bank = load_bank(MIMIC_RUN / "data", "test")
    mimic_batches = eval_schedule(mimic_bank, "test")
    frozen_mimic = [
        dict(patient=p, clip=int(slot), crop=int(start), seconds=h // FS)
        for h, jobs in mimic_batches
        for p, slot, start, _ in jobs
    ]
    if frozen_mimic != json_file(MIMIC_RUN / "test-gaps.json"):
        raise ValueError("MIMIC transfer cases differ from the original held-out comparison")
    cross, cross_examples = evaluate_batches(mimic_bank, mimic_batches, models, display_names=LABELS)
    plot_mimic_transfer(root, cross_examples, mimic_batches)
    cross_groups = summarize_three(cross)
    save_csv(root / "mimic-transfer-paired.csv", cross)
    save_csv(root / "mimic-transfer-summary.csv", cross_groups)
    atomic_json(
        root / "mimic-transfer.json",
        dict(
            patients=len({r["patient"] for r in cross}),
            target_gaps=len(cross),
            gaps_sha256=file_hash(MIMIC_RUN / "test-gaps.json"),
            checkpoint_sha256=hashes,
            groups=cross_groups,
        ),
    )
    print("Finished frozen MIMIC transfer comparison", len(cross), "target gaps", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "evaluate"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.run)
    else:
        evaluate(args.run, args.device)
