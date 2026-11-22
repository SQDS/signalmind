"""Independent model review of grouped, frozen inspection evidence and deterministic negative controls."""

import argparse
import base64
import fcntl
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from jsonschema import ValidationError, validate

from tsmcp.cleaning.agent import BudgetClient, vision_preflight
from tsmcp.cleaning.common import digest, read, sha, write
from tsmcp.followup.evidence import aligned_pulse_context, view
from tsmcp.followup.reviewer import BOUNDS, bounds, tool
from tsmcp.followup.screen_rule import rescan_rule
from tsmcp.inspection.run import load_channel, version_id

S = {"type": "string", "minLength": 1}
RULE = {
    "type": "object",
    "properties": {
        "method": {"enum": ["baseline_shift", "step_outlier", "high_frequency"]},
        "threshold": {"type": "number", "exclusiveMinimum": 0},
    },
    "required": ["method", "threshold"],
    "additionalProperties": False,
}
ROW = {
    "type": "object",
    "properties": {
        "id": S,
        "verdict": {"enum": ["suspected_artifact", "possible_physiology", "no_visible_issue", "uncertain"]},
        "range_s": {"anyOf": [BOUNDS, {"type": "null"}]},
        "observed": S,
        "possible_cause": S,
        "uncertainty": S,
        "repair_request": S,
    },
    "required": ["id", "verdict", "range_s", "observed", "possible_cause", "uncertainty", "repair_request"],
    "additionalProperties": False,
}
TOOLS = [
    tool("expand_context", "Open a wider waveform around this case.", {"id": S, "flank_s": {"enum": [10, 20, 40]}}),
    tool("try_rule", "Evaluate your proposed threshold on all displayed cases and surrounding beats.", {"rule": RULE}),
    tool(
        "submit_review",
        "Submit each waveform verdict in Chinese, and optionally a previously tested rule for a full-record rescan.",
        {
            "cases": {"type": "array", "items": ROW},
            "rescan_rule": {"anyOf": [RULE, {"type": "null"}]},
            "rule_reason": S,
        },
    ),
]
SYSTEM = """你是独立的质量复查 Agent。每个编号都有真实波形，初筛覆盖全记录，取样中混有未触发对照。
深色仅表示要求查看的位置。按图逐项判断，可回答正常、可能生理、疑似伪差、无法判定。
ECG源单位未标定。区分可见形态、病因猜测和修补必要性；只有证据足够时要求停用或数值修补。
扩大上下文由你决定，每侧10/20/40秒。拟定区间为图的块内秒数。正常不画框。
可选回扫方法 baseline_shift 是 ECG 0.32秒中位数相对前后双侧幅度范围的持续偏移，阈值0.975至2.25；
step_outlier 是相邻样本陡变相对局部10秒 p95-p05 幅度的比值，阈值0.3至12；
high_frequency 是通道高频带的功率占比，ECG 45Hz、PPG 18Hz、ABP 20Hz，阈值0.005至0.5。
若审查 ABP 逐搏变化，请区分真实搏动/压力变化、自动脚点错误和测量伪差。逐搏预警仅提示复核，不构成修补或停用依据。
先试方法，注意正常心搏上的误报，按数量和图像判定是否推荐其作为检索方法，命中只表示待查。
每次只调用一个工具。无独立干净参照，不可自报恢复正确率。"""


def select_cards(inspection, cases):
    frozen = read(inspection / "candidate_manifest.json")
    for name in ("evidence.jsonl", "negative_audit.jsonl"):
        if sha(inspection / name) != frozen["files"][name]:
            raise ValueError("Frozen source mismatch")
    rows = [json.loads(line) for line in (inspection / "evidence.jsonl").open()]
    negatives = [json.loads(line) for line in (inspection / "negative_audit.jsonl").open()]
    grouped = defaultdict(list)
    for task, family in (
        ("ECG", "ecg_baseline_shift"),
        ("ECG", "ecg_excursion"),
        ("ECG", "large_step"),
        ("PPG", "large_step"),
        ("ABP", "abp_pulse_episode"),
        ("ABP", "large_step"),
    ):
        subset = [r for r in rows if r["range"]["channel_id"] == task and r["kind"] == family]
        channel_rows = [r for r in rows if r["range"]["channel_id"] == task]
        if not subset and family != "large_step":
            continue
        if family == "ecg_excursion" and len(subset) > 3:
            ranked = sorted(subset, key=lambda r: (r["value"], r["candidate_id"]))
            subset = [ranked[-1], ranked[len(ranked) // 2], ranked[0]]
        if family == "large_step" and subset:
            strongest = max(subset, key=lambda r: r["value"])
            others = [r for r in subset if r["range"]["block_id"] != strongest["range"]["block_id"]]
            subset = [strongest] + ([min(others, key=lambda r: digest([111101, r["candidate_id"]]))] if others else [])
        if family == "large_step":
            freq = [r for r in channel_rows if r["kind"] == "high_frequency"]
            if freq:
                extra = max(freq, key=lambda r: r["value"])
                if not any(
                    v["range"]["block_id"] == extra["range"]["block_id"]
                    and v["start"] < extra["end"]
                    and extra["start"] < v["end"]
                    for v in subset
                ):
                    subset.append(extra)
        if not subset and any(k.startswith(task + "-") for k in grouped):
            continue
        group = task + "-" + family
        for r in subset:
            cid = r["range"]["block_id"]
            overlaps = [
                q["kind"]
                for q in rows
                if q["range"]["block_id"] == cid and q["start"] < r["end"] and r["start"] < q["end"]
            ]
            grouped[group].append(
                dict(
                    case=cid,
                    seed=[r["start"], r["end"]],
                    source=r["candidate_id"],
                    kind=r["kind"],
                    overlap=sorted(set(overlaps)),
                    version=r["observed_version_id"],
                )
            )
        control_pool = [
            r
            for r in negatives
            if r["range"]["channel_id"] == task
            and r["range"]["end_sample"] - r["range"]["start_sample"] >= round(4 * cases[r["range"]["block_id"]]["fs"])
        ]
        pending = grouped.pop(group, [])
        if not pending:
            group = task + "-quality_audit"
        batches = [pending[i : i + 3] for i in range(0, len(pending), 3)] or [[]]
        for index, batch in enumerate(batches, 1):
            name = group if len(batches) == 1 else f"{group}-{index}"
            anchors = sorted(control_pool, key=lambda r: digest([111101, name, r["range"]]))
            for control in anchors[: 1 if batch else 2]:
                rr = control["range"]
                cid = rr["block_id"]
                fs = cases[cid]["fs"]
                center = (rr["start_sample"] + rr["end_sample"]) // 2
                half = round(fs * (0.6 if family == "ecg_baseline_shift" or not pending else 0.05))
                batch.append(
                    dict(
                        case=cid,
                        seed=[center - half, center + half],
                        source="unflagged_control",
                        kind="control",
                        overlap=[],
                        version=control["observed_version_id"],
                    )
                )
            batch.sort(key=lambda c: digest([111101, name, c["case"], c["seed"]]))
            for n, card in enumerate(batch, 1):
                card["id"] = f"{name}-{n}"
            if batch:
                grouped[name] = batch
    return dict(grouped)


def review_group(client, out, root, cases, group, cards, *, conditional_recipe=False):
    out.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=12)
    def signal(cid):
        path = root / "public" / cid / "input.npz"
        c = cases[cid]
        if sha(path) != c["input_sha256"]:
            raise ValueError("Waveform checksum mismatch")
        x, m = load_channel(path, c)
        return x, ~np.logical_or.reduce(list(m.values()))

    views = {}
    trials = {}
    log = []
    byid = {c["id"]: c for c in cards}

    def attachment(c, flank):
        case = cases[c["case"]]
        x, u = signal(c["case"])
        if version_id(case) != c["version"]:
            raise ValueError("Evidence refers to another raw version")
        ev, pic = view(out / "views", case, x, u, c["version"], c["seed"], flank)
        content = [
            dict(
                type="text",
                text=json.dumps(
                    dict(
                        id=c["id"],
                        modality=case["task"],
                        fs=case["fs"],
                        unit=case["unit"],
                        seed_s=[p / case["fs"] for p in c["seed"]],
                        evidence=ev["id"],
                        visible_s=[p / case["fs"] for p in ev["view_samples"]],
                    )
                ),
            ),
            pic,
        ]
        if c.get("card_image") and flank == 5:
            path = root / c["card_image"]
            if sha(path) != c["card_sha256"]:
                raise ValueError("Frozen representative image changed")
            ev["representative_image_sha256"] = c["card_sha256"]
            content.extend(
                [
                    dict(
                        type="text",
                        text="同一位置的补充图，横轴以图上标注为准。工具提交与扩展统一用第一张图的块内秒。",
                    ),
                    dict(
                        type="image",
                        source=dict(
                            type="base64", media_type="image/png", data=base64.b64encode(path.read_bytes()).decode()
                        ),
                    ),
                ]
            )
            write(out / "views" / (ev["id"] + ".json"), ev)
        elif (conditional_recipe or group.startswith("ABP-abp_pulse_episode")) and case.get("has_source_time"):
            with np.load(root / "public" / case["id"] / "input.npz", allow_pickle=False) as data:
                clock = data["time"].ravel()
            peers = []
            for other in cases.values():
                if (
                    other["record_id"] != case["record_id"]
                    or other.get("rows") != case["rows"]
                    or other.get("fs") != case["fs"]
                    or not other.get("has_source_time")
                ):
                    continue
                with np.load(root / "public" / other["id"] / "input.npz", allow_pickle=False) as data:
                    if not np.array_equal(clock, data["time"].ravel()):
                        continue
                peers.append((other, *signal(other["id"])))
            if len(peers) > 1:
                cross, image = aligned_pulse_context(out / "views", ev, peers, case["fs"])
                ev["cross_channel"] = cross
                content.extend([dict(type="text", text="同一编号的同步原始通道，未作脉搏传导时间校正。"), image])
                write(out / "views" / (ev["id"] + ".json"), ev)
        views[c["id"]] = ev
        return content

    content = [
        dict(type="text", text=f"本组检查 {group}。逐张看所示位置及周围波形，可含未触发对照；无需每组都推荐规则。")
    ]
    tools = list(TOOLS)
    if conditional_recipe:
        submit = tools[-1]
        schema = submit["input_schema"]
        fields = {"applies_when": S, "preserve_when": S, "split_or_defer": S, "task_retest": S}
        tools[-1] = tool(submit["name"], submit["description"], {**schema["properties"], **fields})
        content.append(
            dict(
                type="text",
                text="归纳这一组的条件处方：进入条件、必须保留的反例、需要拆组或暂缓的表现、对应任务的复测。不能用组别代替成员判断；可仅建议保留或复查。",
            )
        )
    if (out / "calls.json").exists():
        log = read(out / "calls.json")
    trials = read(out / "trials.json") if (out / "trials.json").exists() else {}
    for c in cards:
        prior = [read(f) for f in (out / "views").glob("*.json")] if log else []
        widest = max(
            (v["flank_s"] for v in prior if v["case"] == c["case"] and v["seed_samples"] == c["seed"]), default=5
        )
        content += attachment(c, widest)
    if trials:
        content.append(
            dict(type="text", text="续跑，已有实际测试结果：" + json.dumps(list(trials.values()), ensure_ascii=False))
        )
    messages = [dict(role="user", content=content)]
    for turn in range(12):
        conclude = len(trials) >= 3 or turn >= 8
        allowed = [tools[-1]] if conclude else tools
        if conclude:
            messages[-1]["content"].append(
                dict(
                    type="text",
                    text="现在立即调用 submit_review 逐张提交最终意见。即使没有合适检测方法，也可将 rescan_rule 设为 null。",
                )
            )
        response = client.request(messages, allowed, system=SYSTEM)
        parts = [b.model_dump(exclude_none=True) for b in response.content]
        messages.append(dict(role="assistant", content=parts))
        calls = [c for c in parts if c["type"] == "tool_use"]
        replies = []
        log.append(dict(request_id=response.id, views={k: v["sha256"] for k, v in views.items()}, tools=calls))
        write(out / "calls.json", log)
        for call in calls:
            try:
                if len(calls) != 1:
                    raise ValueError("Use one tool per turn")
                name = call["name"]
                data = call["input"]
                validate(data, next(t["input_schema"] for t in allowed if t["name"] == name))
                if name == "expand_context":
                    if data["flank_s"] <= views[data["id"]]["flank_s"]:
                        raise ValueError("Choose a wider context")
                    answer = attachment(byid[data["id"]], data["flank_s"])
                elif name == "try_rule":
                    rule = data["rule"]
                    findings = {}
                    for c in cards:
                        case = cases[c["case"]]
                        fs = case["fs"]
                        x, u = signal(c["case"])
                        hits = rescan_rule(x, u, fs, case["task"], rule)
                        left, right = views[c["id"]]["view_samples"]
                        a, b = c["seed"]
                        local = [[v / fs, w / fs] for v, w in hits if v < right and w > left]
                        findings[c["id"]] = dict(
                            block_hits=len(hits),
                            view_hits=len(local),
                            first_12_s=local[:12],
                            seed_hits=sum(v < b and w > a for v, w in hits),
                        )
                    trials[digest(rule)] = dict(rule=rule, findings=findings)
                    write(out / "trials.json", trials)
                    answer = json.dumps(findings)
                else:
                    if len(data["cases"]) != len(cards) or {r["id"] for r in data["cases"]} != set(byid):
                        raise ValueError("Cover each ID once")
                    rule = data["rescan_rule"]
                    if rule is not None:
                        trial = trials.get(digest(rule))
                        if not trial or not any(
                            trial["findings"][r["id"]]["seed_hits"]
                            and r["range_s"]
                            and r["verdict"] == "suspected_artifact"
                            for r in data["cases"]
                        ):
                            raise ValueError("Select a rule tested on a visually suspected artifact, or use null")
                    for row in data["cases"]:
                        c = byid[row["id"]]
                        result = row["range_s"]
                        if row["verdict"] == "suspected_artifact" and result is None:
                            raise ValueError("Give visible suspected bounds")
                        if row["verdict"] == "no_visible_issue" and result is not None:
                            raise ValueError("No visible issue requires null bounds")
                        row["samples"] = (
                            None
                            if result is None
                            else bounds(result, cases[c["case"]]["fs"], views[c["id"]]["view_samples"], c["seed"])
                        )
                        row.update(case=c["case"], source=c["source"], seed=c["seed"], view=views[c["id"]])
                    data.update(request_id=response.id, model=client.model, group=group)
                    write(out / "decision.json", data)
                    return data
                replies.append(dict(type="tool_result", tool_use_id=call["id"], content=answer))
            except (ValueError, KeyError, StopIteration, ValidationError) as exc:
                replies.append(dict(type="tool_result", tool_use_id=call["id"], content=str(exc)[:600], is_error=True))
        messages.append(
            dict(role="user", content=replies or [dict(type="text", text="请调用提供的工具提交或继续检查。")])
        )
    raise RuntimeError(f"Review not submitted: {group}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("workspace", "inspection", "budget"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    out = args.workspace
    cases = {c["id"]: c for c in read(out / "dataset_manifest.json")["cases"]}
    groups = select_cards(args.inspection, cases)
    write(out / "selection.json", groups)
    with (args.budget / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = BudgetClient(args.budget)
        try:
            vision_preflight(client, workspace=out)
            for g, cards in groups.items():
                dest = out / "reviewer" / g
                if not (dest / "decision.json").exists():
                    review_group(client, dest, out, cases, g, cards)
                print("visual review complete", g, len(cards), flush=True)
        finally:
            client.close()


if __name__ == "__main__":
    main()
