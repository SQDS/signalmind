"""Trace every frozen screening candidate to model images, individual decisions and accepted output."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import array_hash, changed, read, sha, write
from tsmcp.cleaning.engine import load_arrays


def audit(workspace):
    output = workspace / "delivery"
    output.mkdir(exist_ok=True)
    inspection = workspace / "inspection"
    frozen = read(inspection / "candidate_manifest.json")
    for name in ("evidence.jsonl", "candidates.jsonl"):
        if sha(inspection / name) != frozen["files"][name]:
            raise ValueError("Inspection evidence changed")
    with (inspection / "evidence.jsonl").open() as stream:
        candidates = list(map(json.loads, stream))
    with (inspection / "candidates.jsonl").open() as stream:
        all_ids = {row["candidate_id"] for row in map(json.loads, stream)}
    if all_ids != {r["candidate_id"] for r in candidates} or len(all_ids) != len(candidates):
        raise ValueError("Candidate ledger is not one-to-one")

    manifest = read(workspace / "dataset_manifest.json")
    cases = {c["id"]: c for c in manifest["cases"]}
    state = read(workspace / "agent_state.json")
    calls = {c["request_id"] for c in read(workspace / "api_budget.json")["calls"] if c["status"] == "returned"}
    final_dir = workspace / "independent-retest"
    measured = {z["id"]: z for z in read(final_dir / "measurements.json")}
    viewports = defaultdict(set)
    direct = {}
    n_controls = 0
    for path in sorted((workspace / "reviewer").glob("*/decision.json")):
        decision = read(path)
        final = read(final_dir / (decision["group"] + ".json"))
        if decision["request_id"] not in calls or final["request_id"] not in calls:
            raise ValueError("Visual verdict has no request receipt")
        final_by_id = {r["id"]: r["decision"] for r in final["cases"]}
        pictures = {v["sha256"]: v for v in (read(f) for f in (path.parent / "views").glob("*.json"))}
        for call in read(path.parent / "calls.json"):
            if call["request_id"] not in calls:
                raise ValueError("Missing reviewer receipt")
            for pic in set(call["views"].values()):
                v = pictures[pic]
                if sha(path.parent / "views" / v["image"]) != pic:
                    raise ValueError("Reviewer image changed")
                viewports[v["case"]].add(("初审", *v["view_samples"]))
        hashes = []
        for r in decision["cases"]:
            if r["source"] != "unflagged_control":
                if r["source"] not in all_ids or r["source"] in direct:
                    raise ValueError("Unexpected or duplicate review source")
                direct[r["source"]] = (r["verdict"], final_by_id[r["id"]])
            else:
                n_controls += 1
            z = measured[r["id"]]
            hashes.append(sha(final_dir / z["image"]))
            c = cases[r["case"]]
            start, end = r["samples"] or r["seed"]
            viewports[c["id"]].add(
                ("批后复查", max(0, start - round(5 * c["fs"])), min(c["n"], end + round(5 * c["fs"])))
            )
        if hashes != final["image_hashes"]:
            raise ValueError("Post-review images changed")

    last_turn = read(workspace / "agent-session.json")["model_turns"]
    with (workspace / "agent_activity.jsonl").open() as stream:
        for call in map(json.loads, stream):
            if call["tool"] != "inspect_cases" or call["is_error"] or call["turn"] >= last_turn:
                continue
            blocks = call["result"]
            ids = json.loads(blocks[0]["text"])["evidence_ids"]
            if sum(b["type"] == "image" for b in blocks) != len(ids):
                raise ValueError("Missing repair image")
            for eid in ids:
                ev = state["evidence"][eid]
                if sha(workspace / ev["image"]) != ev["image_sha256"]:
                    raise ValueError("Repair image changed")
                viewports[ev["case_id"]].add(("修补自检", *ev["range_samples"]))

    original = read(workspace / "versions" / read(workspace / "baselines.json")["identity"] / "version.json")
    versions = {
        task: read(workspace / "versions" / vid / "version.json") for task, vid in state["accepted_tasks"].items()
    }
    groups = defaultdict(list)
    for r in candidates:
        groups[r["range"]["block_id"]].append(r)
    rows, queue = [], []
    counts, per_kind = Counter(), defaultdict(Counter)
    for cid, hits in sorted(groups.items()):
        c = cases[cid]
        task, fs = c["task"], c["fs"]
        ar = versions[task]["results"][cid]
        raw, t, y, valid, estimated = load_arrays(workspace, c, ar)
        _, _, _, prior_valid, prior_est = load_arrays(workspace, c, original["results"][cid])
        if (
            sha(workspace / "public" / cid / "input.npz") != c["input_sha256"]
            or array_hash(y, t, valid, estimated) != ar["content_hash"]
        ):
            raise ValueError("Accepted or source waveform changed")
        new_off = (prior_valid & ~prior_est) & ~(valid & ~estimated)
        modified = changed(raw, y)
        prior = original["results"][cid]
        if sha(workspace / prior["probe_path"]) != prior["probe_sha256"]:
            raise ValueError("Original probe changed")
        peaks = np.array([e["sample"] for e in read(workspace / prior["probe_path"])["events"]])
        episodes = []
        for r in sorted(hits, key=lambda h: (h["start"], h["end"])):
            if not episodes or r["start"] > episodes[-1]["end"]:
                episodes.append(dict(start=r["start"], end=r["end"], hits=[]))
            ep = episodes[-1]
            ep["end"] = max(ep["end"], r["end"])
            ep["hits"].append(r)
        for ep in episodes:
            group_id = f"G{len(queue) + 1:04d}"
            members = []
            for r in ep["hits"]:
                a, b, uid = r["start"], r["end"], r["candidate_id"]
                overlap = {phase for phase, left, right in viewports[cid] if a < right and left < b}
                decision, post = direct.get(uid, ("", ""))
                if decision:
                    status = "逐条复查"
                elif overlap:
                    status = "仅有图像范围交叠"
                else:
                    status = "无图像范围交叠"
                counts[status] += 1
                per_kind[r["kind"]][status] += 1
                midpoint = (a + b) / 2
                near = round(float(np.min(abs(peaks - midpoint))) * 1000 / fs, 2) if len(peaks) else None
                rows.append(
                    [
                        uid,
                        group_id,
                        task,
                        cid,
                        a,
                        b,
                        round(a / fs, 3),
                        round(b / fs, 3),
                        r["kind"],
                        status,
                        "、".join(sorted(overlap)),
                        decision,
                        post,
                        int(modified[a:b].sum()),
                        int(new_off[a:b].sum()),
                        near,
                    ]
                )
                members.append(
                    dict(
                        candidate=uid,
                        rule=r["kind"],
                        duration_ms=round((b - a) * 1000 / fs, 3),
                        relative_offset_ms=round((a - ep["start"]) * 1000 / fs, 3),
                        observed=r["value"],
                        trigger_threshold=r["threshold"],
                        unit=r["unit"],
                        nearest_native_event_ms=near,
                    )
                )
            queue.append(
                dict(
                    group_id=group_id,
                    modality=task,
                    fs_hz=fs,
                    duration_s=round((ep["end"] - ep["start"]) / fs, 4),
                    member_count=len(members),
                    members=members,
                    jev_status="not_run",
                )
            )
    assert len(rows) == len(all_ids) == frozen["candidate_count"] and sum(e["member_count"] for e in queue) == len(rows)
    with (output / "全量初筛候选复查台账.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "候选ID",
                "合并时段ID",
                "通道",
                "采集块",
                "起点样本",
                "终点样本（不含）",
                "块内起点秒",
                "块内终点秒",
                "触发规则",
                "逐条复查状态",
                "相交图像阶段（相交不等于判读）",
                "独立初审判断",
                "批后意见",
                "最终数值改动样本",
                "最终新增停用样本",
                "中点距最近原始探针事件毫秒",
            ]
        )
        writer.writerows(rows)
    with (output / "Jev待评估时段.jsonl").open("w") as stream:
        for ep in queue:
            stream.write(json.dumps(ep, ensure_ascii=False, allow_nan=False) + "\n")
    result = dict(
        candidates=len(rows),
        direct=len(direct),
        controls=n_controls,
        view_overlap="any sample overlap in same channel and continuous block",
        counts=counts,
        by_rule=per_kind,
        proposed_episode_count=len(queue),
        merge="touching or overlapping within the same block/channel only",
        candidates_with_new_exclusion=sum(r[14] > 0 for r in rows),
        jev_calls=0,
    )
    write(output / "候选覆盖审计.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().workspace), ensure_ascii=False))
