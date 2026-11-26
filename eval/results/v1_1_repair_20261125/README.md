# SignalMind v1.1 短尖峰开发回归

2026-11-25。一个 Vital 患者的同一段 750 秒 ECG，在相同四个位置分别加入三种扰动。每处两点，125 Hz。检测器只读取扰动后的观测，程序验收后由离线测试复查器统一批准。下表只把两个采样点均有估计来源且相对留存观测的误差下降计为修回。此次新增模型调用和费用均为零。

| 条件 | 原样保留的核心 MAE | 交付核心 MAE | 修回 | 最终处置 | 注入区外改值 |
|---|---:|---:|---:|---|---:|
| 强双相 | 1.32944 | 0.00268 | 4/4 | 3 处线性，1 处 PCHIP | 0 |
| 强单向 | 1.32944 | 1.32944 | 0/4 | 4 处原样保留 | 0 |
| 中等单向 | 0.39883 | 0.00268 | 4/4 | 3 处线性，1 处 PCHIP | 0 |

历史同口径的离线程序回放也是 8/12。历史带真实 DeepSeek 独立复查的交付为 6/12。本次比较支持检查同父版本选择和程序保护是否生效。模型独立复查的修回率还需要在新版本上实跑。[各条件的定位、修回与误差汇总](summary.json)

四处强单向均触发原生心搏保护。独立 XQRS 对其中两处给出一致定位，另两处有分歧。已在原始公开 PPG 和 ABP 记录核对两个通道的完整数值及时间轴，使用邻拍的实测脉搏延迟检查。四处 ABP 的目标延迟均偏离邻拍范围。PPG 有一处偏离、三处无法唯一关联。四处均保留原观测，事件表没有自动增删。另七处自然告警的检测支持落在边界或没有可用事件，均未写值。它们缺少独立人工标签。

在强双相和中等单向两个条件下，每次修补八个样本后，原生 RR 的覆盖从 93,549 降到 92,720 个采样点。每个条件另列四条研究间期，覆盖 416 个样本，均为 `released=false`。其余间期受重新分段后探针边界影响。导出分别保留原生覆盖、研究覆盖及二者并集。

## 原生 Vital 完整记录

用 `dataset_rules` 从原始 MAT 重跑一名既有开发患者。每个通道含 48 个连续块，累计 5,950 秒观测，三个通道全部完成交付。程序记录 1,987 个候选，其中 1,573 个由高心率规则触发。此模式没有模型规划，所有候选的波形动作均为保留，三个通道实际改值和新增停用均为零。原生探针生成 6,134 个 RR、6,119 个 PPI 和 6,142 个逐拍压力结果，合计 16,636 条保留、1,759 条待复查。该患者的天然伪迹没有独立干净参照。

## 公共执行器与独立评分

中等单向条件通过隔离执行器另跑三个确定性基线，它们共享四个检测结果及相同的允许读写范围。原样保留、全部线性和全部 PCHIP 的核心 MAE 分别为 0.39883、0.00281、0.00187。三者在注入区外均修改零点。同父选择策略有时选了线性，开发数据上的同相位邻拍误差并不能始终挑出真实误差最小的算法。[确定性基线汇总](summary.json)

公共执行器接收保留、注册算子、用途停用、影子事件建议或有父版本绑定的数值数组。每次启动一个无网络的 Bubblewrap 进程，只把本题公共文件、提交和数值依赖交给它。私有评分器在独立入口读取分配清单，校验哈希，单列失败与缺失，并在固定分母上计算原样回退的误差。当前公共算子目录覆盖短核心线性与 PCHIP。生产系统中的心搏保护只作用于 SignalMind 发布决策。

```bash
.venv/bin/python eval/verify_planned_repair.py \
  --archive eval/runs/group-repair-v2-20261122 --out /tmp/sm-v11-replay --short-core-comparison

.venv/bin/python -m eval.repair_benchmark.runner prepare \
  --observed eval/runs/group-repair-v2-20261122/moderate/injection-input/waveform.npz \
  --case-id vital-moderate --task ECG --fs 125 --unit source_units --out /tmp/sm-v11-public
.venv/bin/python -m eval.repair_benchmark.runner execute \
  --public /tmp/sm-v11-public --baseline interpolate_pchip --out /tmp/sm-v11-pchip
.venv/bin/python -m eval.repair_benchmark.scoring \
  --manifest eval/runs/v11-executor-final-20261125/organizer/manifest.json --out /tmp/sm-v11-score.json
```

上一版的 12 个历史处置均可回溯，17 份波形、支持掩码及探针工件与旧档案匹配。[兼容性回归汇总](summary.json)

本轮使用同一名开发患者的四个位置。随后完成的同底座对照使用合成 ECG。跨厂商验证、独立患者试点和预注册统计仍待完成。[DeepSeek 三组实测](../v1_1_ds_synthetic_20261125/README.md)
