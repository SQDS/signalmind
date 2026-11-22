# SignalMind 本地工作台

工作台读取现有运行档案，并可启动新的 Vital、CSV、NPZ 清洗任务。历史浏览不重新运行 DSP，不调用模型。

## 启动

在仓库根目录：

```bash
uv pip install --python .venv/bin/python -e '.[workbench]'
.venv/bin/tsmcp-ui serve
```

浏览器打开 <http://127.0.0.1:8787>。可用 `--port`、`--repo`、`--state-dir` 指定端口、仓库位置与工作台数据目录。远程服务器使用 SSH 转发，例如 `ssh -L 8787:127.0.0.1:8787 用户@服务器`，然后在自己电脑上打开上述地址。

默认数据目录为 `.signalmind-workbench/`，已加入 Git 忽略列表。程序不会把患者数据或 API 密钥写入代码仓库的版本控制。

## 五个页面

- **运行总览**：当前任务范围、各通道支持和测量覆盖、诊断提示、历史去重患者数和 API 用量估算。四轮 Vital 实验是同一批 10 人；累计估算 $0.392523648，实际账单未知。
- **数据与新任务**：登记本地路径或上传 CSV/NPZ，确认映射，选择任务、患者/文件范围、运行方法与预算。选择“仅三基线”不会调用模型。
- **Agent 处理过程**：全部宏工具记录、失败、调整、视觉自检和回退。版本关系中的父版本表示波形依赖；试跑到批处理的方案复用单独展示。
- **波形检查**：筛选完整连续段；缩放、比较版本、查看实际中间算子状态；点击标记查看原因、参数、复测和方案源码。
- **方案与交付档案**：所有保存的候选源码及差异、报告下载、完整过程包。包含编译失败的源码草稿。

深链接示例：

```text
/#signals?run=RUN_ID&task=PPG&case=RECORD_PPG_0000
/#process?run=RUN_ID
```

## 导入契约

Vital 继续使用原 MAT 适配器、训练分区与患者排除规则。CSV 每行一个采样点；NPZ 支持独立的一维数值数组，或显式指定采样轴及通道索引的二维数组。对象数组不启用 pickle。

每个文件独立成记录，不按文件名拼接。缺少真实时间字段时必须填写采样率，输出标为相对时间。秒/毫秒需要明确选择；规则时间轴上的缺口拆段，重复、倒序及不规则采样进入审计失败记录，不自动排序或重采样。ABP 逐拍压力要求明确的 mmHg；归一化压力不能直接发布为绝对压力。

可选有效掩码和估计掩码应为布尔值或 0/1，与通道等长。来源无效支持和估计支持随输入保留，不算作 Agent 新增修改。无患者标识时，显示记录数和未知患者数。

保存的映射示例（也可通过界面填写）：

```json
{
  "time_key": "time",
  "time_unit": "s",
  "fs": 125,
  "channels": [
    {
      "task": "PPG",
      "key": "signals",
      "sample_axis": 0,
      "channel": 1,
      "unit": "source_units",
      "valid_key": "valid",
      "estimated_key": "estimated"
    }
  ]
}
```

## 留存与执行

原始运行文件是事实来源，SQLite 只存可重建的运行/数据集索引。历史四轮为只读；工作台不调用当前控制器去重新解释旧版本的实现哈希。

新运行保留：

- `events.jsonl`：追加式模型调用、公开说明、方案提交、工具开始/结果、算子执行、接受/回退、停止及续跑事件。模型内部推理和凭据不通过网页暴露。
- `recipes/`：提交时先保存源码，再编译；编译失败亦留存。
- `steps/` 或 `jobs/<id>/steps/`：逐算子精确替换值和支持掩码，父依赖与校验哈希。完整状态按保存的替换片段恢复，不通过再次执行滤波器伪造中间波形。
- `jobs/<id>/attempts/`：每次 worker 尝试的退出结果、标准输出和错误，包括超时重试。
- `case_checkpoints/`、worker 验证检查点及模型待处理工具：续跑复用已完成工作，不重复应用已完成修改。
- `private/frozen_source/` 与 `implementation_manifest.json`：本次实际实现代码和环境版本；源码草稿、图片、探针与交付继续使用现有版本目录。

新 recipe 可在 `ctx.apply` 中传入 `explanation`（字符串或 JSON 对象），说明观察、规则阈值和操作理由。执行器独立计算局部复测并留存实际状态；未提供的解释显示为缺失。完整 RR/PPI/压力探针只属于实际运行过探针的候选版本，不套用到中间步骤上。

同一时间启动一个 Agent 任务；本地数值处理使用两个 worker。停止在当前模型/工具或记录边界生效，可能需要等待当前计算完成。续跑保持原输入、实现、模型和费用账本。实现变化后需创建新任务；旧运行继续可查看。费用预算耗尽或模型主动完成后，不能通过续跑重置预算。

历史算子中间状态及精确时间若未保存，界面显示“未保存”。r4 的区间解释是已有的离线补充材料，明确标记为事后规则还原，不冒充当时的逐区间模型说明。标坏表示停止用于相应任务，不等于已经修补；没有独立参考时，准确率保持未评估。

## API 与验证

主要接口：`/api/datasets`、`/api/runs`、`/api/runs/{id}/versions`、`events`、`stream`、`cases`、`recipes`、`evidence`、`exports`、`archive`。创建任务使用 `POST /api/runs`，停止/续跑使用 `POST /api/runs/{id}/control/stop|resume`。波形接口支持版本、采样点范围和算子步骤，下载为包含真实数组和掩码的 NPZ。

```bash
.venv/bin/python -m pytest tests/test_workbench.py tests/test_cleaning_v1.py tests/test_cleaning_explanations.py -q
```

真实浏览器检查使用已安装的 Firefox/geckodriver，读取四轮真实历史，并在独立 QA 目录运行合成数据的算子与三基线检查，不调用模型：

```bash
.venv/bin/tsmcp-ui serve --port 8876 --state-dir .signalmind-workbench/qa
# 另一个终端
.venv/bin/python eval/check_workbench_browser.py
```

浏览器检查结果和截图保存在 `.signalmind-workbench/validation/`。QA 合成任务与正式工作台目录分离，不混入 Vital 患者数或 Agent 结果。

新任务可以选择“仅质量探查（本地）”。快速、均衡、细致三档使用相同的输入范围。任务完成后，在“方案与交付档案”下载候选区间、证据、覆盖和未报警抽样记录。该模式在本地执行，模型费用为零。[首轮执行报告](../../eval/results/devkit-inspection-20261118/README.md)
