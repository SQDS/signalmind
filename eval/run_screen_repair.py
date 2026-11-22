"""Hand signed, grouped visual observations to a fresh DS cleaning session with shared billing."""

import argparse
import fcntl
import json
from collections import defaultdict
from pathlib import Path

from tsmcp.cleaning.agent import run
from tsmcp.cleaning.common import digest, read, write
from tsmcp.cleaning.controller import CleaningTools
from tsmcp.followup.screen_rule import rule_id


def execute(out, budget):
    groups = read(out / "selection.json")
    brief = []
    ctl = CleaningTools(out)
    pilots = defaultdict(set)
    for group in groups:
        folder = out / "reviewer" / group
        d = read(folder / "decision.json")
        task = group.split("-")[0]
        rule = d["rescan_rule"]
        rows = d["cases"]
        for r in rows:
            pilots[task].add(r["case"])
        if rule is not None:
            trial = read(folder / "trials.json")[digest(rule)]["findings"]
            target = next(
                r for r in rows if r["verdict"] == "suspected_artifact" and trial[r["id"]]["seed_hits"] and r["samples"]
            )
            rid = rule_id(task, rule)
            ctl.state.setdefault("registered_patterns", {})[rid] = dict(
                rule_id=rid,
                task=task,
                rule=rule,
                detector_family="screen_rule",
                seed_case=target["case"],
                seed_samples=target["seed"],
                suspected_samples=target["samples"],
                seed_version=target["view"]["version"],
                views=[target["view"]],
                suspected_cause=target["possible_cause"],
                request_id=d["request_id"],
                model=d["model"],
                origin="independent_visual_reviewer",
            )
        brief.append(
            dict(
                group=group,
                rule=rule,
                **{k: d[k] for k in ("applies_when", "preserve_when", "split_or_defer", "task_retest") if k in d},
                rows=[
                    {
                        k: r[k]
                        for k in (
                            "id",
                            "case",
                            "seed",
                            "samples",
                            "verdict",
                            "observed",
                            "repair_request",
                            "uncertainty",
                            "source",
                        )
                    }
                    for r in rows
                ],
            )
        )
    ctl.state["reviewer_pilot_cases"] = {k: sorted(v) for k, v in pilots.items()}
    ctl.save()
    path = out / "private/messages.json"
    if not path.exists():
        text = (
            "你是修补 Agent，独立复查 Agent 已按实际图像判断下列候选和未触发对照。请用当前完整患者的三个通道执行 pilot→看图与程序复测→batch→全量复测与视觉复查。"
            '本次复查提交且在候选上实测命中的方法已注册，接受批量后会自动跨所有块回扫。用 ctx.select("review_rule",method=...,threshold=...) 调用。'
            "不要再重复注册同一方法。若本例复查发现适合数值校正的ECG基线偏移，在试跑中与identity比对，观察心搏保护和残余后自主决定。"
            'SignalMind 现有试验扩展代码 ctx.apply("local_detrend",ranges,window_s=...,explanation=...)，只有 category="custom_extension" 才可用。'
            "它限ECG，使用0.18到0.8秒中位数，窗内局部基线与两侧0.7秒锚点插值比较，余弦边缘过渡，绝不插入样本。若损伤心搏可回退。"
            "同一配方可组合标准算子，规则命中不能一律修补或停用，审查不确定性质时保留原始观测，留意正常心搏。"
            "在 baseline_shift 的候选试跑或批量上，inspect_cases 实际看本例有代表性的改变及未改变对照的前后图，核查下游心搏；三种任务各自选择处理。"
            "下列病例编号仅供取证与复查，程序决不能硬编码编号或时刻。说明保留、修补和未解决之处。\n"
            + json.dumps(brief, ensure_ascii=False)
        )
        if any("applies_when" in item for item in brief):
            text += (
                "\n本次为代表样本→条件配方试验。先合并组内意见，若不同动作缺少可观察的区分条件，保持原样。"
                "可用 ctx.select('screen_family', family='ecg_baseline_shift' 等) 定位同类，但必须对每段验证保护条件；"
                "选样角色和患者/时间不可作为判断依据。独立探针分歧也是待查线索，时间差、峰数和波形外观均非人工真值。"
                "保留门槛、对照、试修参数、回退条件及未解决项。每个实际操作将交给独立的批后逐框审查方决定是否发布。"
            )
        write(path, [{"role": "user", "content": text}])
    with (budget / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = run(out, cap=read(budget / "api_budget.json")["cap_usd"], budget_root=budget)
        write(out / "api_budget.json", read(budget / "api_budget.json"))
        print("cleaning result", result["stop"], result["workflow_complete"], result.get("error"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--budget", required=True, type=Path)
    a = p.parse_args()
    execute(a.workspace, a.budget)
