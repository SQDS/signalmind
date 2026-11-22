"""Compare verified raw and accepted artifacts at each model-reviewed location and request a fresh visual review."""

import argparse
import base64
import fcntl
import json
from collections import Counter
from pathlib import Path

import numpy as np
from jsonschema import ValidationError, validate
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.cleaning.agent import BudgetClient
from tsmcp.cleaning.common import array_hash, read, sha, write
from tsmcp.cleaning.engine import load_arrays
from tsmcp.followup.reviewer import tool
from tsmcp.processing import intervals


def seed_counts(probe, lo, hi):
    return [r["sample"] for r in probe["events"] if lo <= r["sample"] < hi]


def make_card(workspace, folder, case, orig, accepted, decision, rescans, page_base):
    raw, t, y, valid, est = load_arrays(workspace, case, accepted)
    if (
        array_hash(y, t, valid, est) != accepted["content_hash"]
        or sha(workspace / "public" / case["id"] / "input.npz") != case["input_sha256"]
    ):
        raise ValueError("Source or accepted waveform hash mismatch")
    _, _, _, raw_valid, raw_est = load_arrays(workspace, case, orig)
    probes = []
    for artifact in (orig, accepted):
        if sha(workspace / artifact["probe_path"]) != artifact["probe_sha256"]:
            raise ValueError("Stale probe")
        probes.append(read(workspace / artifact["probe_path"]))
    fs = case["fs"]
    a, b = decision["samples"] or decision["seed"]
    left, right = max(0, a - round(5 * fs)), min(len(raw), b + round(5 * fs))
    offset = np.arange(left, right)
    # x-axis stays monotone even across minute boundaries.
    page = int((page_base + (a + b) / 2 / fs) // 60) + 1
    clock = page_base - 60 * (page - 1)
    x = clock + offset / fs
    changed = (y[left:right] != raw[left:right]) & ~(np.isnan(raw[left:right]) & np.isnan(y[left:right]))
    pre, post = [seed_counts(p, left, right) for p in probes]
    region_before, region_after = [seed_counts(p, a, b) for p in probes]
    mask = valid & ~est
    current = mask[left:right]
    old = (raw_valid & ~raw_est)[left:right]
    fig = Figure(figsize=(9, 4.4), layout="constrained")
    FigureCanvasAgg(fig)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    lims = np.r_[raw[left:right], y[left:right]]
    finite = lims[np.isfinite(lims)]
    pad = max(0.01, np.ptp(finite) * 0.10)
    for ax, z, events, title, support in ((ax1, raw, pre, "处理前", old), (ax2, y, post, "交付候选", current)):
        ax.axvspan(clock + a / fs, clock + b / fs, color="#dfaa42", alpha=0.18)
        for s, e in intervals(~support):
            ax.axvspan(
                clock + (left + s) / fs,
                clock + (left + e) / fs,
                facecolor="#424b5320",
                edgecolor="#65717a",
                hatch="////",
                lw=0.6,
            )
        ax.plot(x, z[left:right], color="#244f68" if ax is ax1 else "#06705f", lw=0.95)
        if events:
            points = np.array(events)
            ax.plot(clock + points / fs, z[points], ".", ms=3.4, color="#a34a4a")
        ax.set(ylim=(finite.min() - pad, finite.max() + pad), ylabel=case["unit"], title=title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{case['task']} · 第 {page} 分钟    金色为复查位置，斜线为任务停用，红点为程序探针事件", fontsize=11)
    ax2.set_xlabel("本页时间（秒）")
    fig.savefig(folder / (decision["id"] + ".png"), dpi=110)
    fig.clear()
    actual_range = [
        base for op in accepted["operations"] for base in op["ranges_samples"] if base[0] < b and a < base[1]
    ]
    metrics = dict(
        id=decision["id"],
        case=case["id"],
        task=case["task"],
        source=decision["source"],
        verdict=decision["verdict"],
        review_s=[a / fs, b / fs],
        page=page,
        page_range_s=[round(clock + a / fs, 3), round(clock + b / fs, 3)],
        before_events_in_range=region_before,
        after_events_in_range=region_after,
        before_context_event_count=len(pre),
        after_context_event_count=len(post),
        context_events_identical=pre == post,
        max_context_event_shift_ms=(
            round(max(abs(u - v) for u, v in zip(pre, post, strict=True)) / fs * 1000, 3)
            if pre and len(pre) == len(post)
            else None
        ),
        modified_samples_in_range=int(np.count_nonzero(y[a:b] != raw[a:b])),
        modified_samples_in_context=int(changed.sum()),
        old_unsupported_in_range=int((~(raw_valid & ~raw_est)[a:b]).sum()),
        unsupported_in_range=int((~mask[a:b]).sum()),
        estimated_in_range=int(est[a:b].sum()),
        actual_op_ranges=actual_range,
        operations=sorted(
            set(op["op"] for op in accepted["operations"] if any(s < b and a < e for s, e in op["ranges_samples"]))
        ),
        image=decision["id"] + ".png",
        image_sha256=sha(folder / (decision["id"] + ".png")),
        rule_checks=[],
    )
    for pair in rescans.values():
        report, per_case = pair
        if case["id"] in per_case:
            r = per_case[case["id"]]
            metrics["rule_checks"].append(
                dict(
                    rule=report["detector"],
                    **{k: sum(s < b and a < e for s, e in r[k]) for k in ["before", "after_numeric", "after_eligible"]},
                )
            )
    return metrics


def final_review(client, folder, group, cards):
    field = {"type": "string", "minLength": 1}
    row = {
        "type": "object",
        "properties": {
            "id": field,
            "decision": {"enum": ["accept_for_task", "retain_with_caveat", "request_revision"]},
            "observed": field,
            "reason": field,
        },
        "additionalProperties": False,
        "required": ["id", "decision", "observed", "reason"],
    }
    final = tool("submit_result", "提交本组所有局部复测意见", {"cases": {"type": "array", "items": row}})
    content = [
        dict(
            type="text",
            text="每张图上面原始、下面是本次交付候选。依据图和测量复核。ECG用于RR，PPG用于PPI，ABP用于逐拍压力。若数值修补明显损坏可用心搏或不当扩大停用，请明确要求调整。不要把源检测器漏拍当成人工真值。",
        )
    ]
    for c in cards:
        pic = (folder / c["image"]).read_bytes()
        content += [
            dict(type="text", text=json.dumps(c, ensure_ascii=False)),
            dict(type="image", source=dict(type="base64", media_type="image/png", data=base64.b64encode(pic).decode())),
        ]
    messages = [dict(role="user", content=content)]
    for _ in range(3):
        res = client.request(
            messages,
            [final],
            system="你是独立质量复查Agent，当前是批后盲于修补Agent推理的新会话。用中文逐例提交；区分图上可见、程序复测和不确定项。仅使用submit_result工具。",
        )
        parts = [b.model_dump(exclude_none=True) for b in res.content]
        calls = [b for b in parts if b["type"] == "tool_use"]
        try:
            if len(calls) != 1 or calls[0]["name"] != final["name"]:
                raise ValueError("Use submit_result")
            data = calls[0]["input"]
            validate(data, final["input_schema"])
            if sorted(r["id"] for r in data["cases"]) != sorted(c["id"] for c in cards):
                raise ValueError("Cover each ID once")
            write(
                folder / (group + ".json"),
                dict(request_id=res.id, model=client.model, image_hashes=[c["image_sha256"] for c in cards], **data),
            )
            return
        except (ValidationError, ValueError) as e:
            messages.append(dict(role="assistant", content=parts))
            replies = [
                dict(type="tool_result", tool_use_id=c["id"], content=str(e)[:400], is_error=True) for c in calls
            ]
            messages.append(dict(role="user", content=replies or "提交每个编号的意见。"))
    raise RuntimeError("Final review incomplete for " + group)


def retest(workspace, source, budget):
    state = read(workspace / "agent_state.json")
    if not state["complete"]:
        raise ValueError("Waiting for a complete model batch")
    folder = workspace / "independent-retest"
    folder.mkdir(exist_ok=True)
    cases = {c["id"]: c for c in read(workspace / "dataset_manifest.json")["cases"]}
    if (source / "gallery/summary.json").exists():
        meta = read(source / "gallery/summary.json")
        page_base = {cid: b["start"] for b in meta["blocks"] for cid in b["ids"].values()}
    else:
        starts = {}
        for cid, case in cases.items():
            if not case["has_source_time"]:
                raise ValueError("Record-relative minute plots require a source acquisition clock")
            with np.load(source / "public" / cid / "input.npz", allow_pickle=False) as z:
                starts[cid] = float(z["time"][0])
        origin = min(starts.values())
        page_base = {cid: t - origin for cid, t in starts.items()}
    base = read(workspace / "baselines.json")["identity"]
    original = read(workspace / "versions" / base / "version.json")
    cards = []
    groups = {}
    with rc_context({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False}):
        for path in sorted((workspace / "reviewer").glob("*/decision.json")):
            d = read(path)
            group = path.parent.name
            task = group.split("-")[0]
            vid = state["accepted_tasks"][task]
            version = read(workspace / "versions" / vid / "version.json")
            rescans = {}
            for rid, ref in state["pattern_rescans"][vid].items():
                report = read(workspace / ref["report"])
                if (
                    not report["complete"]
                    or report["waveform_version"] != vid
                    or sha(workspace / ref["findings"]) != report["findings_sha256"]
                ):
                    raise ValueError("Unbound sweep")
                rescans[rid] = (report, {r["case_id"]: r for r in read(workspace / ref["findings"])})
            groups[group] = []
            for decision in d["cases"]:
                cid = decision["case"]
                c = make_card(
                    workspace,
                    folder,
                    cases[cid],
                    original["results"][cid],
                    version["results"][cid],
                    decision,
                    rescans,
                    page_base[cid],
                )
                control_check = workspace / "control-morphology.json"
                if decision["source"] == "unflagged_control" and control_check.exists():
                    check = read(control_check)
                    if check["case"] == cid:
                        c["morphology_evidence"] = check
                groups[group].append(c)
                cards.append(c)
    write(folder / "measurements.json", cards)
    with (budget / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = BudgetClient(budget)
        try:
            for group, items in groups.items():
                if not (folder / (group + ".json")).exists():
                    final_review(client, folder, group, items)
                print("independent retest", group, flush=True)
        finally:
            client.close()
        write(workspace / "api_budget.json", read(budget / "api_budget.json"))
    results = Counter(r["decision"] for g in groups for r in read(folder / (g + ".json"))["cases"])
    print("independent decisions", dict(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("workspace", "source", "budget"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    retest(a.workspace, a.source, a.budget)
