# SignalMind

SignalMind 是一个面向整库生理波形的处理 Agent，支持 ECG 心搏和 RR、PPG 脉搏和 PPI、ABP 逐拍压力。它按患者、信号通道和触发原因组织初筛结果，由修补角色制定组内条件，本地执行 SignalMind 算子，再由独立上下文的复查角色检查候选。交付包含原始观测支持、实际修改、估计范围和逐拍测量的用途判定。

## 从哪里开始读

| 想弄清的问题 | 源码 |
|---|---|
| 数据目录、任务、预算如何进入整库运行？ | [数据集入口](src/tsmcp/cleaning/dataset_entry.py) · [CLI](src/tsmcp/cleaning/cli.py) |
| 三路信号用哪些规则筛查，如何定位及生成同步图？ | [初筛规则](src/tsmcp/inspection/scan.py) · [患者基线](src/tsmcp/inspection/patient_profile.py) · [证据与绘图](src/tsmcp/cleaning/dataset_evidence.py) |
| 每组两张图和全部成员数值怎样送给 DS？模型可以提哪些条件？ | [分组和试修](src/tsmcp/cleaning/dataset_pipeline.py) · [模型协议和回执](src/tsmcp/cleaning/dataset_review.py) |
| 插值怎样落到采样点，怎样保护心搏及复测、回退？ | [SignalMind 算子](src/tsmcp/processing.py) · [冻结探针](src/tsmcp/cleaning/tasks.py) · [执行及版本记录](src/tsmcp/cleaning/engine.py) |
| 怎样核对“没有单独看图”和“逐拍质量可疑”是两个独立状态？ | [整库回归测试](tests/test_dataset_entry.py) · [输入与输出规范](docs/signalmind/dataset-entry.md) |

## 可独立运行的检查

在仓库根目录执行以下命令。测试用临时目录构造输入，无需患者波形或模型凭据。

```bash
uv sync --group dev --group eval --extra cleaning --extra workbench
.venv/bin/python -m pytest -q tests/test_dataset_entry.py tests/test_processing.py tests/test_devkit_inspection.py tests/test_workbench.py
.venv/bin/tsmcp-clean --help
```

如需运行自己的数据，按输入格式配置数据路径、通道和时间轴。命令行和本地工作台使用同一调度器。[整库运行说明](docs/signalmind/dataset-entry.md) · [本地工作台](docs/signalmind/workbench.md) · [示例配置](configs/cleaning/dataset-entry.example.json)

## 这次开放审查的问题

在两名 Vital 患者的开发回放中，模型对 7 处 ECG 短尖峰候选执行了线性插值试验。复查退回 6 处、接受 1 处。接受位置的峰形和节律与邻近心搏高度一致，事后检查列为疑似误修。另一次配对检验在同一段 ECG 的四个位置分别加入三种两采样点尖峰。三种形态各有 4/4 触发报警，均为 0/4 实际修回。这些数据是开发集上的工程结果。独立真实波形恢复准确率仍无参考值。

建议审查时沿一条提示依次追踪分组输入、试修选择、保护区、算子写入、复测和最终用途状态，特别留意靠近记录块边缘的心搏以及双相尖峰的边界定位。[核验脚本](eval/results/group-repair-v2-20261122/audit.py) · [注入对照](eval/results/group-repair-v2-20261122/injected_control.py) · [邻拍对照](eval/results/group-repair-v2-20261122/accepted_peak.py)

这个公开快照提供源码、配置示例和测试。历史评测脚本的全量重放需要另行准备其数据、归档和隔离运行环境。患者波形、逐拍表、模型调用回执和密钥由数据持有方本地管理。开源许可见 [LICENSE](LICENSE)。
