# SignalMind

SignalMind 是一个面向整库生理波形的处理 Agent，支持 ECG 心搏和 RR、PPG 脉搏和 PPI、ABP 逐拍压力。它按患者、信号通道和触发原因组织初筛结果，由修补角色制定组内条件，本地执行 SignalMind 算子，再由独立上下文的复查角色检查候选。交付包含原始观测支持、实际修改、估计范围和逐拍测量的用途判定。

本版接入全库候选冻结、检测器支持边界、独立复查与补丁回滚。短尖峰试修会逐成员核对适用条件，复查可以同时查看采样点、估计值和邻近心搏的同相位证据。[质量流程与发布规范](docs/signalmind/quality-plan-20261123.md) · [短尖峰修补对照](eval/results/spike-repair-20261124/README.md)

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
.venv/bin/python -m pytest -q tests/test_dataset_entry.py tests/test_processing.py tests/test_devkit_inspection.py tests/test_workbench.py tests/test_planned_quality.py tests/test_spike_repair_evidence.py -k "not skill_promotion_needs_sandbox_and_independent_paired_references"
.venv/bin/tsmcp-clean --help
```

如需运行自己的数据，按输入格式配置数据路径、通道和时间轴。命令行和本地工作台使用同一调度器。[整库运行说明](docs/signalmind/dataset-entry.md) · [本地工作台](docs/signalmind/workbench.md) · [示例配置](configs/cleaning/dataset-entry.example.json)

## 本次修补实测

在同一名 Vital 开发患者的四个固定位置分别加入三种短尖峰，位置和评分参照固定。两版 DS 对 12 个条件均完成定位，最终修回从 1/12 增至 6/12。强双相 4/4、中等单向 2/4、强单向 0/4。三组在注入区外改写的 ECG 样本数均为 0。强单向尖峰仍受 QRS 保护，强双相的复扫达到返工上限，运行状态为部分完成。独立患者上的恢复准确率仍待验证。[汇总结果与对照口径](eval/results/spike-repair-20261124/README.md) · [心搏保护逻辑](src/tsmcp/cleaning/quality_gate.py) · [试修复查图的构造](src/tsmcp/cleaning/spike_evidence.py)

开发工作区全套 1,302 项测试通过。上面的公开测试集使用生成的数据。技能晋升的隔离 worker 集成测试还需要可创建独立命名空间的 Linux、bubblewrap 及评测解释器，历史全量评测另需归档。[隔离 worker](src/tsmcp/cleaning/sandbox.py) · [评测环境设置](eval/processing_tools.py)

## 查看自己的标注

工作台提供按候选状态查询的账本。对于三通道波形以原值交付、带有标注的工作区，绘图脚本还可输出每页 30 秒的标注，以及按患者固定纵轴的 ECG 放大图。运行脚本时，将环境变量 `SIGNALMIND_DISPLAY_CONFIG` 指向自己的 JSON 文件，其中 `workspace` 为运行目录，`output` 为输出目录，`people` 将记录 ID 映射到显示名，`previews` 包含两个示例窗口，每项为 `[通道块ID, 起始源时间秒, 图标题]`。每个窗口需要 30 秒连续观测。[标注绘图](eval/results/quality-live-20261123/annotation_view.py) · [ECG 纵轴与完整图册](eval/results/quality-live-20261123/ecg_scale.py)

公开仓库提供源码、配置示例、测试和汇总结果。患者数据、逐拍明细、模型回执和凭据由数据持有方本地管理。[许可证](LICENSE)
