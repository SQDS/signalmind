"""Post-run descriptive analysis; never changes detectors, thresholds or frozen receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsmcp.cleaning.common import read, sha, write
from tsmcp.locators.api import METHODS
from tsmcp.locators.evaluation import aggregate_rows


def analyze(root):
    repo = Path(__file__).resolve().parents[1]
    rows = read(root / "scores-synthetic.json")
    controls = [r for r in rows if r["family"] == "preservation_control"]
    pack = read(repo / "eval/cases/processing-v0.3/SignalMind_Test_Pack_v0.3/private/manifest.json")
    sharp_ids = {
        c["case_id"]
        for c in pack["cases"]
        if "sharp" in c.get("construction_features", []) and c["family"] == "preservation_control"
    }
    sharp = [r for r in rows if r["case_id"] in sharp_ids]
    normal = aggregate_rows(controls)
    protected_sharp = aggregate_rows(sharp)
    inputs = read(root / "manifest.json")
    protocol = read(root / "freeze.json")
    verified = 0
    for row in inputs["records"]:
        for path in (Path(row["input_path"]), root / "inputs" / row["input_relative"]):
            if sha(path) != row["input_sha256"]:
                raise ValueError(f"Input changed: {path}")
        verified += 1
    for name, expected in protocol["files"].items():
        if sha(root / "snapshots" / protocol["implementation_hash"] / name) != expected:
            raise ValueError("Frozen snapshot differs from recorded implementation")
    counts, methods = {}, {}
    for stage in ("dev", "synthetic", "vital"):
        receipt = read(root / f"stage-{stage}.json")
        counts[stage] = {k: receipt[k] for k in ("cases", "completed_jobs", "elapsed_s", "scope_complete")}
        for case in [r for r in inputs["records"] if r["stage"] == stage]:
            for method in receipt["methods"]:
                output = read(root / "worker_outputs" / stage / case["case_id"] / method / "result.json")
                assert output["input_sha256"] == case["input_sha256"]
                assert output["implementation_hash"] == protocol["implementation_hash"]
                methods.setdefault(stage, {}).setdefault(method, {}).setdefault(output["status"], 0)
                methods[stage][method][output["status"]] += 1
    analysis = {
        "normal_controls": normal,
        "sharp_controls": protected_sharp,
        "sharp_case_ids": sorted(sharp_ids),
        "verification": {
            "source_inputs_verified": verified,
            "staged_inputs_verified": verified,
            "source_snapshot_files_verified": len(protocol["files"]),
            "stages": counts,
            "receipt_statuses": methods,
            "model_api_calls": 0,
        },
        "scope_notes": [
            "pure controls are the eight preservation_control cases, not all cases with an empty primary scoring target",
            "sharp subset is the two explicitly sharp preservation-control cases; E&C denotes their target morphology windows",
            "60 s limit is wall time per method and entire input case, shared by all finite subsegments, including loading/output",
            "algorithm_s is the locate_issues wall time after prewarming imports; failed/timeout rows have null timing",
            "D-C combines coverage, boundary, valid-MAD support, projection and FFT adjustments; not a causal FFT ablation",
        ],
    }
    write(root / "additional-analysis.json", analysis)
    report_dir = repo / "eval/results/locator-comparison-20261115"
    report_dir.mkdir(parents=True, exist_ok=True)
    main = read(root / "summary-synthetic.json")
    vital = read(root / "summary-vital.json")
    text = [
        "# 时间球定位器 v0.1：离线对照结果",
        "",
        "2026-11-15。已完成 26 题开发检查、80 题冻结开发集对照及既有 10 名 Vital 患者的 4,098 个任务片段。没有调用模型，没有修改波形，也没有提取 RR。",
        "",
        "**本轮不支持把这版时间球评分替换为 Agent 的默认问题定位器。** 修正版解决了数据处理和执行问题，正常波形上的误报仍然很高。球面渲染不参与当前检测；本结果不评价 DS 是否更容易理解球形图。",
        "",
        "## 80题主结果",
        "",
        "主域为固定评分域中有限且未被公开缺失/设备标志直接指出的观测。下面按实际时间累计，不是题目通过率，也不是临床准确率。",
        "",
        "| 方法 | 污染时间召回 | 精确率 | 时间F1 | 正常区误报比例 | 检测中位耗时 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = main["by_method"][method]
        metrics = row["tracks"]["unknown_observed"]["micro"]
        text.append(
            f"| {method} | {metrics['recall']:.2%} | {metrics['precision']:.2%} | {metrics['f1']:.3f} | {metrics['normal_fp_ratio']:.2%} | {row['runtime']['algorithm_median_s'] * 1000:.2f} ms |"
        )
    text += [
        "",
        "A 是30秒粗定位，正常区误报仍高；B只针对尖峰、平坦和削顶，整体低召回不能解释为它对所有目标都差。C/D没有改善本轮整体定位效果。",
        "",
        "## 正常控制与尖锐形态",
        "",
        "纯正常控制为8题；另对其中明确标为 sharp 的2题统计 E∩C 正常目标形态窗口。主域无剩余污染的18题还包含公开不可用标志/缺口题，不能称作18条纯正常波形。",
        "",
        "| 方法 | 8题正常区告警比例 | 2题sharp正常目标窗口告警比例 |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        n = normal["by_method"][method]["tracks"]["all_problems"]["micro"]
        s = protected_sharp["by_method"][method]["tracks"]["all_problems"]["micro"]
        text.append(f"| {method} | {n['normal_fp_ratio']:.2%} | {s['normal_event_fp_ratio']:.2%} |")
    text += [
        "",
        "误报的已知机制：原稿平坦项即使没有山脊丢失也可给4分，超过3.2阈值；径向三次偏差相对固定球半径，会响应正常脉冲；稀疏ECG的小IQR也可将正常尖峰放大到异常阈值以上。这些公式本轮保持不变，没有按80题分数调参。",
        "",
        "## Vital完整范围扫描",
        "",
        "10人，每通道73,590秒（约20.44小时），三个通道共4,098个连续任务片段。这里是已有PulseDB-Vital派生观测，不是新增临床标注数据。所有真实恢复/定位准确率字段均为null。",
        "",
        "| 方法 | 告警时间比例 | 累计定位用时 | 单片段中位耗时 | 执行回执 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = vital["by_method"][method]
        t = row["tracks"]["unknown_observed"]["micro"]
        text.append(
            f"| {method} | {t['alarm_s'] / t['domain_s']:.2%} | {row['runtime']['algorithm_total_s']:.2f} s | {row['runtime']['algorithm_median_s'] * 1000:.2f} ms | {row['runtime']['algorithm_measured_rows']}/4098 |"
        )
    text += [
        "",
        "告警比例不是污染率，不能据此判断真实数据谁更准确。D累计定位用时低于C，但短片段的中位耗时并不一定更低。累计定位用时是各次locate_issues调用墙钟用时之和，整批运行还包括fork、加载、输出及辅助候选。",
        "",
        "## 工程验收与保真",
        "",
        f"- {verified}份原始输入及对应公开输入副本的SHA256全部一致；冻结实现快照{len(protocol['files'])}个文件校验通过。",
        "- 原稿Python代码块逐字节保留。所有方法共同按有限观测、来源支持和真实断点分段，不插值跨缺口；这是算法对照，不是完整Agent对照。",
        "- 26/80/Vital合计21,020份执行回执，包括4组主方法和1组辅助候选；状态计数见 additional-analysis.json。",
        "- worker只能读取公开波形和冻结实现，不能读取私有参考或模型凭据。独立复核手算了5题×4方法×2域的40项结果，未发现改变排序的评分错误。",
        "- 限额是每方法×整个输入case共60秒墙钟时间（包括有效子段、加载与输出），每子进程4 GiB；两个隔离worker、单线程数值计算、无自动重试。",
        "- C的局部评分覆盖未知，不能声称100%。D的总覆盖表示至少一项可算，缺失的分量上下文仍在逐项覆盖中记录。",
        "- 生成网格的重放与原D结果逐事件一致；网格降采样仅用于绘图，数值评分不变。",
        "- D−C是FFT、尾圈、边界、有效MAD样本和分数回映的综合变化，不能归因于单个改动。",
        "",
        "## 查看与复现",
        "",
        "- [全部对照图册](../../runs/locator-comparison-20261115/index.html)",
        "- [80题逐条CSV](../../runs/locator-comparison-20261115/scores-synthetic.csv)",
        "- [Vital逐条CSV](../../runs/locator-comparison-20261115/scores-vital.csv)",
        "- [正常控制与工程校验](../../runs/locator-comparison-20261115/additional-analysis.json)",
        "- [冻结配置](../../runs/locator-comparison-20261115/freeze.json)",
        "- [接口和运行说明](../../../docs/signalmind/locator-comparison-v1.md)",
        "",
        "下一步若继续开发，应另外编号修改评分方法，并重新冻结验证；优先区分正常重复形态与异常、补充相位可信度和原采样率短瞬变定位。本轮结果不能支持直接自动排除告警区间。",
        "",
    ]
    (report_dir / "README.md").write_text("\n".join(text))
    write(
        report_dir / "summary.json",
        {
            "synthetic": main["by_method"],
            "vital": vital["by_method"],
            "verification": analysis["verification"],
            "experimental_claim": "time-sphere v0.1 did not improve this exposed synthetic development comparison",
        },
    )
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "eval/runs/locator-comparison-20261115"
    )
    args = parser.parse_args()
    result = analyze(args.out)
    print(json.dumps(result["verification"], ensure_ascii=False))
