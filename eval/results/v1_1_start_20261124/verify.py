"""Check the offline PR-01/02 evidence and render the milestone summary."""

from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from tsmcp.cleaning.common import implementation_hash, read, sha, write

here = Path(__file__).resolve().parent
runs = here.parents[1] / "runs"
result = runs / "v1_1_start_20261124/acceptance"
root = runs / "v1_1_start_20261124/certified/domain-release"
old = runs / "planning-v1-20261123/certified-rules"
r = read(result / "verified.json")
wave = read(root / "export/waveform_manifest.json")
old_wave = read(old / "export/waveform_manifest.json")
assert set(wave) == set(old_wave)
for cid, w in wave.items():
    for k in ("source_sha256", "content_hash", "probe_sha256", "trusted_support"):
        assert w[k] == old_wave[cid][k]
for file in ("measurements.csv", "usable_measurements.csv"):
    assert sha(root / "export" / file) == sha(old / "export" / file)
report = read(root / "export/evaluation_report.json")
assert report["status"] == "completed" and report["model_calls"] == 0
snapshot = read(root / "acceptance-spec.json")
assert snapshot["profile"]["calibration"] == "development_only"
assert (
    report["acceptance_spec_id"] == snapshot["id"]
    and read(root / "ledger/run.json")["implementation"] == implementation_hash()
)
domains = Counter()
for f in root.glob("records/*/discovery.json"):
    for h in read(f)["hits"]:
        d = h["repairability"]
        assert d["acceptance_spec_id"] == snapshot["id"]
        domains[d["status"]] += 1
assert sum(domains.values()) == read(root / "ledger/freeze.json")["candidate_count"]
t = ElementTree.parse(result / "regression.xml").getroot().find("testsuite").attrib
assert t["errors"] == t["failures"] == "0"
summary = dict(
    baseline="4e5a09fd38bf94a783359422328e0fcc4b2d0ad4",
    implementation=implementation_hash(),
    tests=int(t["tests"]),
    matched_artifacts=r["identical_numerical_artifacts"],
    historical_conditions=len(r["targets"]),
    statuses=domains,
    verified_blocks=len(wave),
    historical_commits=sum(d["committed"] for d in r["targets"]),
    new_api_calls=0,
)
write(here / "verified.json", summary)
lines = [
    "# v1.1 首批开发验收：修补漏斗与用途域",
    "",
    f"基线 `{summary['baseline']}`。本地实现指纹 `{summary['implementation']}`。全量回归 {summary['tests']:,} 项通过。新增模型调用 0，API 新增支出 0。",
    "",
    "## 修补漏斗的归档回填",
    "",
    "按现有发布工件重建同一名患者四个固定位置的三种注入条件。评分坐标只在离线核验器读入。下表仅统计该目标的短尖峰数值试修，重试仍属于同一目标。",
    "",
    "| 历史条件 | 目标数 | 生成过数值候选的位置 | 实际算出的数值候选（含返工） | 有最终插值提交的位置 |",
    "|---|---:|---:|---:|---:|",
]
for key, label in (("bipolar", "强双相"), ("same-sign", "强单向"), ("moderate", "中等单向")):
    rows = [d for d in r["targets"] if d["shape"] == key]
    lines.append(
        f"| {label} | {len(rows)} | {sum(d['generated'] > 0 for d in rows)} | {sum(d['generated'] for d in rows)} | {sum(d['committed'] for d in rows)} |"
    )
lines += [
    "",
    f"带独立参照的 6/12 属于 v1.0 历史交付。另用固定提议和全同意复查替身重新执行数值链路，与既有基线的 {summary['matched_artifacts']} 份工件逐一比较，波形、时间轴、有效/估计掩码与探针结果一致。旧归档无法还原的字段保持缺失。",
    "",
    "## 显式开发域实跑",
    "",
    f"相同两名 Vital 开发患者，三通道共 {len(wave)} 个连续块，以 v1.1 本地规则模式完成。逐块比对保存的波形内容哈希、来源 SHA 和探针哈希，两张原生测量表逐字节一致。该运行检查用途域和交付链路，与模型组别的对比需另行冻结配置。",
    "",
    "本轮规则模式给冻结的检测候选评估其对应数值操作的适用性。计数为 "
    + "，".join(f"`{k}` {count:,}" for k, count in sorted(domains.items()))
    + "。多数触发类型超出当前登记的短核心算子范围，这些计数衡量算子适用性。",
    "",
    "用途对照测试包含 4 点在 125 Hz 和 1 kHz 下分别对应 32 ms 和 4 ms、未知单位或设备、不同 ECG 用途、时间断点、缺失锚点，以及已知域内通过真实算子和独立测试回执发布的正例。",
    "",
    "[开发映射和使用说明](../../../docs/signalmind/v1.1-development.md) · [本页机器摘要](verified.json) · [归档回填与数值核验](../../verify_repair_outcomes.py)",
]
(here / "README.md").write_text("\n".join(lines) + "\n")
print(summary)
