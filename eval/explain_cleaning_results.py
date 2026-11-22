"""Retrospective interval receipts for frozen cleaning outputs; never edits decisions or waveforms.

Original operations/parameters and probe artifacts remain original evidence. Trigger values below
are reconstructed from hash-identified recipe rules, and must reproduce the logged ranges exactly.
They are not newly invented model rationales or independently confirmed noise labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import changed, digest, implementation_hash, read, sha, time_support, write
from tsmcp.pipeline import select_ranges
from tsmcp.processing import intervals

PROFILES = {
    "fc9c341a499e41c55d4e26c4823fb12fa33becbe787931e3dcb57c499f548887": "ecg",
    "32c50e28f5a5af0f440f3014e1638c416e579f5f5cb4bfae60aa79774d32555f": "ppg",
    "b1be941db64d4305a7043f72a0bb1cc1d76fdf1909e02cb21d76b30e0a24b5ca": "abp_v2",
    "5c9ee34baec89fa7839a34f2d420ebcf82c183b50c6de77ba58bc2cb4f1f6a82": "abp_v1",
}
CSV_FIELDS = [
    "case_id",
    "task",
    "kind",
    "start_s",
    "stop_s",
    "problem",
    "reason_origin",
    "operation",
    "parameters",
    "trigger_evidence",
    "repair_status",
    "changed_samples",
    "eligible_before",
    "eligible_after",
    "context_measured_before_s",
    "context_measured_after_s",
    "quality_verified",
]


def merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([int(a), int(b)])
    return out


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def reversal_rate(x):
    d = np.diff(x)
    if len(d) < 2:
        return None
    return float(np.mean(((d[:-1] > 0) & (d[1:] < 0)) | ((d[:-1] < 0) & (d[1:] > 0))))


def jitter_windows(x, fs):
    block, hop = int(10 * fs), int(5 * fs)
    if len(x) < 4 * block or block <= 16:
        return [], None, None
    d = np.diff(x)
    flips = ((d[:-1] > 0) & (d[1:] < 0)) | ((d[:-1] < 0) & (d[1:] > 0))
    total = np.r_[0, np.cumsum(flips)]
    starts = np.arange(0, len(x) - block + 1, hop)
    rates = (total[starts + block - 2] - total[starts]) / (block - 2)
    median = float(np.sort(rates)[len(rates) // 2])  # Matches the recipe's upper median.
    threshold = min(0.60, median + 0.12)
    windows = [
        {"range_samples": [int(s), int(s + block)], "value": float(v), "threshold": threshold}
        for s, v in zip(starts, rates, strict=True)
    ]
    return windows, median, threshold


def reconstruct(profile, before, before_valid, after, diagnosis, fs):
    """Reconstruct requests, not signal processing; compare to actual logs at the boundary."""
    if profile not in {"ecg", "ppg", "abp_v1", "abp_v2"}:
        raise ValueError("No verified reconstruction adapter for this exact recipe")
    x, n = np.asarray(before), len(before)
    valid = before_valid.copy()
    calls, skipped = [], []

    def call(op, requested, components, parameters=None):
        if not requested:
            return
        applied = []
        if op == "mark_invalid":
            applied = [[int(a), int(b)] for a, b in requested]
            for a, b in applied:
                valid[a:b] = False
        else:
            for a, b in requested:
                applied.extend([[a + left, a + right] for left, right in intervals(valid[a:b])])
        calls.append({"op": op, "ranges_samples": applied, "components": components, "parameters": parameters or {}})

    def unsupported(values):
        missing = select_ranges(values, valid, fs, {"kind": "missing"})
        flat = select_ranges(values, valid, fs, {"kind": "flat", "min_duration_s": 1.0, "pad_samples": 2})
        components = []
        if missing:
            components.append(
                {
                    "rule_id": "nonfinite_samples",
                    "label": "非有限观测值",
                    "ranges": missing,
                    "parameters": {"selector": "missing"},
                }
            )
        if flat:
            components.append(
                {
                    "rule_id": "exact_flat",
                    "label": "持续精确平坦段（规则判定）",
                    "ranges": flat,
                    "parameters": {"minimum_duration_s": 1.0, "padding_samples": 2},
                }
            )
        return missing + flat, components

    if profile == "ecg":
        bad, components = unsupported(x)
        call("mark_invalid", bad, components)
    for issue in diagnosis.get("issues", []):
        k = str(issue.get("kind", "")).lower()
        threshold = 15 if profile == "ppg" else 20
        if ("narrowband" in k or "tone" in k) and issue.get("frequency_hz") and issue.get("contrast", 0) >= threshold:
            component = {
                "rule_id": "narrowband_contrast",
                "label": "窄带谱峰对比超过当前阈值",
                "ranges": [issue["range_samples"]],
                "parameters": {"contrast_threshold": threshold},
                "observed": issue,
            }
            call(
                "remove_tone",
                [issue["range_samples"]],
                [component],
                {"frequency_hz": issue["frequency_hz"], "window_s": 10},
            )
        elif profile == "ecg" and any(t in k for t in ("flat", "missing", "dropout")):
            call(
                "mark_invalid",
                [issue["range_samples"]],
                [
                    {
                        "rule_id": "diagnostic_unusable",
                        "label": "诊断提示不支持测量",
                        "ranges": [issue["range_samples"]],
                        "observed": issue,
                    }
                ],
            )
    if profile == "ecg":
        return calls, skipped

    if profile.startswith("abp"):
        threshold = max(8.0, 800.0 / fs)
        hits = np.flatnonzero(np.abs(np.diff(x)) > threshold)
        bands = [[max(0, int(i) - 3), min(n, int(i) + 5)] for i in hits]
        merged = merge(bands)
        total = sum(b - a for a, b in merged)
        allowance = 0.02 * n if profile == "abp_v1" else max(0.02 * n, 0.8 * fs)
        passes = total <= allowance and (profile == "abp_v1" or total <= 0.10 * n)
        component = {
            "rule_id": "abp_sample_jump",
            "label": "相邻压力变化超过程序阈值（疑似突跳）",
            "ranges": merged,
            "trigger_indices": hits.tolist(),
            "trigger_ranges": bands,
            "parameters": {
                "jump_threshold": threshold,
                "unit": "mmHg/sample",
                "left_padding_samples": 3,
                "right_stop_offset_samples": 5,
                "candidate_samples": total,
                "allowed_samples": allowance,
                "additional_max_fraction": 0.1 if profile == "abp_v2" else None,
            },
        }
        if merged and passes:
            call("mark_invalid", merged, [component])
        elif merged:
            skipped.append(
                {**component, "applied": False, "skip_reason": "候选范围总量超过该程序的整段预算，整组未执行"}
            )
        bad, components = unsupported(after)
        call("mark_invalid", bad, components)
        return calls, skipped

    disc, components = [], []
    if n > 8:
        ordered = np.sort(x)
        lo, hi = float(ordered[int(0.05 * (n - 1))]), float(ordered[int(0.95 * (n - 1))])
        amplitude = hi - lo
        if amplitude > 0:
            threshold, pad = 0.5 * amplitude, max(2, int(0.25 * fs))
            hits = np.flatnonzero(np.abs(np.diff(x)) > threshold)
            bands = [[max(0, int(i) - pad), min(n, int(i) + 1 + pad)] for i in hits]
            disc = merge(bands)
            total = sum(b - a for a, b in disc)
            component = {
                "rule_id": "ppg_relative_jump",
                "label": "相邻变化超过本段幅度尺度的50%（疑似突跳）",
                "ranges": disc,
                "trigger_indices": hits.tolist(),
                "trigger_ranges": bands,
                "parameters": {
                    "jump_threshold": threshold,
                    "unit": "source_units/sample",
                    "p05_order_value": lo,
                    "p95_order_value": hi,
                    "robust_span": amplitude,
                    "padding_samples": pad,
                    "candidate_samples": total,
                    "allowed_samples": 0.02 * n,
                },
            }
            if total > 0.02 * n:
                skipped.append({**component, "applied": False, "skip_reason": "突跳候选范围超过2%整段预算，整组未执行"})
                disc = []
            elif disc:
                components.append(component)
    windows, median, threshold = jitter_windows(x, fs)
    if windows:
        triggered = [r for r in windows if r["value"] > threshold]
        pad = int(0.5 * fs)
        expanded = [[max(0, r["range_samples"][0] - pad), min(n, r["range_samples"][1] + pad)] for r in triggered]
        guarded = [[max(0, a - int(fs)), min(n, b + int(fs))] for a, b in disc]
        kept = []
        for region in merge(expanded):
            pieces = [region]
            for g in guarded:
                next_pieces = []
                for p in pieces:
                    if not overlaps(p, g):
                        next_pieces.append(p)
                    else:
                        if p[0] < g[0]:
                            next_pieces.append([p[0], g[0]])
                        if g[1] < p[1]:
                            next_pieces.append([g[1], p[1]])
                pieces = next_pieces
            kept.extend([p for p in pieces if p[1] - p[0] > int(0.5 * fs)])
        component = {
            "rule_id": "ppg_reversal_rate",
            "label": "局部差分反向较多，程序视为疑似抖动",
            "ranges": kept,
            "parameters": {
                "window_s": 10,
                "hop_s": 5,
                "median_rate": median,
                "rate_threshold": threshold,
                "padding_samples": pad,
                "discontinuity_guard_samples": int(fs),
            },
            "windows": triggered,
            "guard_ranges": guarded,
        }
        call("lowpass", kept, [component], {"frequency_hz": 12, "filter_order": 2})
    bad, extra = unsupported(after)
    call("mark_invalid", disc + bad, components + extra)
    return calls, skipped


def verify_calls(expected, actual):
    if len(expected) != len(actual):
        raise ValueError(f"Reconstructed call count differs: {len(expected)} versus {len(actual)}")
    for e, a in zip(expected, actual, strict=True):
        if e["op"] != a["op"] or e["ranges_samples"] != a["ranges_samples"]:
            raise ValueError("Reconstructed operation ranges do not match the original execution log")
        for key, value in e["parameters"].items():
            if a["parameters"].get(key) != value:
                raise ValueError("Reconstructed parameter differs from original execution log")


def component_evidence(component, region, before, after):
    result = {
        k: v
        for k, v in component.items()
        if k not in {"ranges", "trigger_indices", "trigger_ranges", "windows", "guard_ranges"}
    }
    if "trigger_indices" in component:
        selected = [
            i
            for i, band in zip(component["trigger_indices"], component["trigger_ranges"], strict=True)
            if overlaps(band, region)
        ]
        result["trigger_pair_count"] = len(selected)
        result["trigger_examples"] = [
            {
                "left_sample": int(i),
                "right_sample": int(i + 1),
                "left_value": float(before[i]),
                "right_value": float(before[i + 1]),
                "absolute_delta": float(abs(before[i + 1] - before[i])),
            }
            for i in selected[:32]
        ]
        result["maximum_trigger_delta"] = max((float(abs(before[i + 1] - before[i])) for i in selected), default=None)
        result["scope_note"] = "标记包括触发采样及其扩展邻域，并非区间内每个采样点都独立触发。"
    if "windows" in component:
        pad = component["parameters"]["padding_samples"]
        selected = [
            r
            for r in component["windows"]
            if overlaps([r["range_samples"][0] - pad, r["range_samples"][1] + pad], region)
        ]
        result["trigger_windows"] = [
            {**r, "after_value_same_window": reversal_rate(after[s:e])}
            for r in selected
            for s, e in [r["range_samples"]]
        ]
        result["trigger_window_count"] = len(selected)
        result["maximum_trigger_rate"] = max((r["value"] for r in selected), default=None)
        result["scope_note"] = "先按10秒窗口判定，再扩展/合并并避开突跳保护带；最终区间不一定等同于判定窗口。"
    return result


def numeric_summary(x):
    finite = np.asarray(x)[np.isfinite(x)]
    return {
        "finite_samples": int(len(finite)),
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
        "difference_reversal_rate": reversal_rate(x),
    }


def covered_samples(ranges, a, b):
    return sum(max(0, min(b, e) - max(a, s)) for s, e in ranges)


def local_probe_summary(measures, a, b, fs):
    event_positions = [e["sample"] for e in measures["events"]]
    count = sum(a <= p < b for p in event_positions)
    periods = [
        p
        for p in measures["intervals"]
        if a <= event_positions[p["left_event_id"]] and event_positions[p["right_event_id"]] < b
    ]
    cycles = [p for p in measures["pressures"] if a <= p["start_sample"] and p["stop_sample"] < b]
    return {
        "detected_events": count,
        "complete_intervals": len(periods),
        "complete_pressure_cycles": len(cycles),
        "measured_seconds": covered_samples(measures["measurement_support"], a, b) / fs,
    }


def retest(region, raw, output, before_valid, valid, estimated, before_probe, after_probe, fs):
    a, b = region
    left, right = max(0, a - round(2 * fs)), min(len(raw), b + round(2 * fs))
    common = np.isfinite(raw[a:b]) & np.isfinite(output[a:b])
    delta = output[a:b][common] - raw[a:b][common]
    before_step = np.diff(raw[max(0, a - 1) : min(len(raw), b + 1)])
    after_step = np.diff(output[max(0, a - 1) : min(len(raw), b + 1)])
    return {
        "evidence_basis": "offline_metrics_on_frozen_arrays_and_original_frozen_probe_outputs",
        "range_samples": region,
        "before": numeric_summary(raw[a:b]),
        "after": numeric_summary(output[a:b]),
        "changed_samples": int(changed(raw[a:b], output[a:b]).sum()),
        "max_absolute_value_change": float(np.max(np.abs(delta))) if len(delta) else None,
        "mean_value_change": float(np.mean(delta)) if len(delta) else None,
        "edge_context_max_step_before": float(np.max(np.abs(before_step))) if len(before_step) else None,
        "edge_context_max_step_after": float(np.max(np.abs(after_step))) if len(after_step) else None,
        "eligible_samples_before": int(before_valid[a:b].sum()),
        "eligible_samples_after": int((valid[a:b] & ~estimated[a:b]).sum()),
        "estimated_samples_after": int(estimated[a:b].sum()),
        "context_range_samples": [left, right],
        "context_padding_s": 2,
        "context_before": local_probe_summary(before_probe, left, right, fs),
        "context_after": local_probe_summary(after_probe, left, right, fs),
        "context_note": "邻域计数从整段固定探针结果截取，可能包含相邻处理影响，不是单次操作的因果分解。",
        "reference_quality_verified": False,
        "quality_verdict": "无独立参考，不能据此声称真实恢复正确",
    }


def flatten(case, region):
    r = region["retest"]
    return {
        "case_id": case["case_id"],
        "task": case["task"],
        "kind": region["kind"],
        "start_s": region["start_s"],
        "stop_s": region["stop_s"],
        "problem": region["problem"],
        "reason_origin": region["reason_origin"],
        "operation": ";".join(sorted({o["op"] for o in region["operations"]})),
        "parameters": json.dumps([o["parameters"] for o in region["operations"]], ensure_ascii=False),
        "trigger_evidence": json.dumps(region["rules"], ensure_ascii=False),
        "repair_status": region["repair_status"],
        "changed_samples": r["changed_samples"],
        "eligible_before": r["eligible_samples_before"],
        "eligible_after": r["eligible_samples_after"],
        "context_measured_before_s": r["context_before"]["measured_seconds"],
        "context_measured_after_s": r["context_after"]["measured_seconds"],
        "quality_verified": False,
    }


def build(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_hash, budget_hash = sha(run / "agent_state.json"), sha(run / "api_budget.json")
    assert read(run / "agent-provenance.json")["implementation_hash"] == implementation_hash()
    state, manifest = read(run / "agent_state.json"), read(run / "dataset_manifest.json")
    versions, profiles = {}, {}

    def version(vid):
        if vid not in versions:
            versions[vid] = read(run / "versions" / vid / "version.json")
        return versions[vid]

    for task, vid in state["accepted_tasks"].items():
        chain = []
        while vid:
            v = version(vid)
            recipe_path = run / "recipes" / v["recipe"]["hash"] / "program.py"
            code = recipe_path.read_text()
            assert digest(code) == v["recipe"]["source_sha256"]
            code_hash = sha(recipe_path)
            profile = PROFILES[code_hash]
            chain.append((v, profile, code_hash))
            vid = v["parent_version"]
        profiles[task] = list(reversed(chain))
    identity = version(read(run / "baselines.json")["identity"])
    summaries, counts, receipt_count = [], Counter(), 0
    csv_file = (output / "intervals.csv").open("w", encoding="utf-8-sig", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    with csv_file, (output / "execution-receipts.jsonl").open("w", encoding="utf-8") as receipts_file:
        for number, case in enumerate(manifest["cases"], 1):
            cid, task, fs = case["id"], case["task"], case["fs"]
            source_path = run / "public" / cid / "input.npz"
            assert sha(source_path) == case["input_sha256"]
            with np.load(source_path, allow_pickle=False) as z:
                raw, times = z["values"].copy(), z["time"].copy()
            source_good = np.isfinite(raw) & time_support(case)

            def arrays(artifact, raw=raw, source_good=source_good):
                if not artifact["arrays"]:
                    return raw, source_good.copy(), np.zeros(len(raw), bool)
                path = run / artifact["arrays"]
                assert sha(path) == artifact["arrays_sha256"]
                with np.load(path, allow_pickle=False) as z:
                    return z["values"].copy(), z["valid"].copy(), z["estimated"].copy()

            base = identity["results"][cid]
            before_values, before_valid, before_estimated = arrays(base)
            previous_artifact = base
            receipts, skipped = [], []
            for step, (v, profile, code_hash) in enumerate(profiles[task]):
                artifact = v["results"][cid]
                values, valid, estimated = arrays(artifact)
                assert not before_estimated.any(), "This reconstruction profile excludes estimated parent inputs"
                expected, suppressed = reconstruct(
                    profile, before_values, before_valid, values, previous_artifact["diagnosis"], fs
                )
                verify_calls(expected, artifact["operations"])
                for call_index, (e, op) in enumerate(zip(expected, artifact["operations"], strict=True)):
                    for range_index, region in enumerate(op["ranges_samples"]):
                        reasons = [
                            component_evidence(c, region, before_values, values)
                            for c in e["components"]
                            if any(overlaps(r, region) for r in c["ranges"])
                        ]
                        assert reasons, (cid, v["version_id"], region)
                        a, b = region
                        receipt = {
                            "id": f"{cid}:{step}:{call_index}:{range_index}",
                            "version_id": v["version_id"],
                            "recipe_id": v["recipe"]["hash"],
                            "code_sha256": code_hash,
                            "range_samples": region,
                            "op": op["op"],
                            "parameters": op["parameters"],
                            "implementation": op["implementation"],
                            "rules": reasons,
                            "reason_origin": "离线追溯：按已执行源码规则重算，区间与原日志逐一一致",
                            "parameters_origin": "原执行日志",
                            "reconstruction_exact": True,
                            "newly_excluded_samples_in_step": int((before_valid[a:b] & ~valid[a:b]).sum()),
                        }
                        receipts.append(receipt)
                        receipts_file.write(
                            json.dumps({"case_id": cid, "task": task, **receipt}, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        receipt_count += 1
                if step == len(profiles[task]) - 1:
                    for item in suppressed:
                        skipped.append(
                            {k: val for k, val in item.items() if k not in {"trigger_indices", "trigger_ranges"}}
                        )
                before_values, before_valid, before_estimated = values, valid, estimated
                previous_artifact = artifact
            final = previous_artifact
            output_values, final_valid, final_estimated = before_values, before_valid, before_estimated
            for artifact in (base, final):
                assert sha(run / artifact["probe_path"]) == artifact["probe_sha256"]
            before_probe, after_probe = read(run / base["probe_path"]), read(run / final["probe_path"])
            masks = {
                "mark_invalid": source_good & ~final_valid,
                "value_change": changed(raw, output_values),
                "source_exclusion": ~source_good,
            }
            regions = []
            final_review = state["reviews"][state["accepted_tasks"][task]]
            images = [
                state["evidence"][eid]
                for eid in final_review["evidence_ids"]
                if state["evidence"][eid]["case_id"] == cid
            ]
            for kind, mask in masks.items():
                for region in intervals(mask):
                    a, b = region
                    related = [
                        r
                        for r in receipts
                        if overlaps(r["range_samples"], region)
                        and (
                            (kind == "mark_invalid" and r["op"] == "mark_invalid")
                            or (kind == "value_change" and r["op"] != "mark_invalid")
                        )
                    ]
                    if kind != "source_exclusion":
                        assert related, (cid, kind, region)
                    rules = list({digest(rule): rule for r in related for rule in r["rules"]}.values())
                    if kind == "source_exclusion":
                        time_conflict = any(overlaps(region, r) for r in case.get("source_time_conflicts", []))
                        rules = [
                            {
                                "rule_id": "source_time_conflict" if time_conflict else "source_nonfinite",
                                "label": "源窗口时间坐标冲突" if time_conflict else "原始非有限观测",
                                "parameters": {"source_ranges": case.get("source_time_conflicts", [])},
                            }
                        ]
                    visible = [e for e in images if overlaps(e["range_samples"], region)]
                    covered = covered_samples(merge([e["range_samples"] for e in visible]), a, b)
                    item = {
                        "id": f"{cid}:{kind}:{a}:{b}",
                        "kind": kind,
                        "range_samples": region,
                        "start_s": a / fs,
                        "stop_s": b / fs,
                        "source_start_s": float(times[a]),
                        "source_stop_s": float(times[b - 1] + 1 / fs),
                        "problem": "；".join(dict.fromkeys(r["label"] for r in rules)),
                        "reason_origin": "原来源审计记录"
                        if kind == "source_exclusion"
                        else "离线规则追溯（不是当时逐区间保存的模型解释）",
                        "rules": rules,
                        "operations": [
                            {
                                k: r[k]
                                for k in (
                                    "id",
                                    "version_id",
                                    "recipe_id",
                                    "code_sha256",
                                    "op",
                                    "range_samples",
                                    "parameters",
                                    "implementation",
                                    "newly_excluded_samples_in_step",
                                )
                            }
                            for r in related
                        ],
                        "repair_status": {
                            "mark_invalid": "标坏停用，未修补",
                            "value_change": "已执行滤波改值，真实恢复质量未验证",
                            "source_exclusion": "来源支持排除，未修补",
                        }[kind],
                        "retest": retest(
                            region,
                            raw,
                            output_values,
                            source_good,
                            final_valid,
                            final_estimated,
                            before_probe,
                            after_probe,
                            fs,
                        ),
                        "final_batch_visual_coverage": {
                            "visible_samples": covered,
                            "total_samples": b - a,
                            "fully_visible": covered == b - a,
                            "evidence_ids": [e["evidence_id"] for e in visible],
                            "meaning": "仅统计该区间是否进入最终批量自检所引用的图片，不代表逐点判断正确。",
                        },
                    }
                    regions.append(item)
                    counts[kind] += 1
            regions.sort(key=lambda r: (r["range_samples"][0], r["kind"]))
            case_result = {
                "schema": "signalmind.interval-explanations.v1",
                "case_id": cid,
                "task": task,
                "version_id": state["accepted_tasks"][task],
                "artifact_id": final["artifact_id"],
                "created_as": "retrospective_enrichment",
                "original_operations_changed": False,
                "model_calls": 0,
                "regions": regions,
                "skipped_rules": skipped,
                "case_retest": {
                    "basis": "原固定探针产物与保存的整段统计",
                    "before": base["stats"],
                    "after": final["stats"],
                    "quality_verified": False,
                },
                "case_note": "本段无Agent改值或新增标坏；不等于人工确认正常。"
                if not any(r["kind"] != "source_exclusion" for r in regions)
                else "标坏与修补分别记录；复测指标不是恢复准确率。",
                "model_batch_review": {
                    "version_id": final_review["version_id"],
                    "observations": final_review["observations"],
                    "remaining_concerns": final_review["remaining_concerns"],
                    "scope": "整体任务及抽查案例，不是逐区间解释",
                },
            }
            write(output / (cid + ".json"), case_result)
            (output / (cid + ".js")).write_text(
                "window.galleryReceiveExplanation("
                + json.dumps(case_result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + ");\n"
            )
            csv_writer.writerows(flatten(case_result, r) for r in regions)
            summaries.append(
                {
                    "id": cid,
                    "region_count": len(regions),
                    "agent_region_count": sum(r["kind"] != "source_exclusion" for r in regions),
                    "skipped_rule_count": len(skipped),
                    "reason_labels": sorted({r["problem"] for r in regions}),
                }
            )
            if number % 100 == 0 or number == len(manifest["cases"]):
                print(
                    json.dumps({"cases": number, "total": len(manifest["cases"]), "regions": dict(counts)}), flush=True
                )
    assert sha(run / "agent_state.json") == state_hash and sha(run / "api_budget.json") == budget_hash
    summary = {
        "source_run": str(run),
        "cases": len(summaries),
        "regions": dict(counts),
        "execution_range_receipts": receipt_count,
        "all_reconstructed_calls_match_original_logs": True,
        "unmapped_agent_regions": 0,
        "new_model_calls": 0,
        "source_state_unchanged": True,
        "api_budget_unchanged": True,
        "explanation_code_sha256": sha(Path(__file__)),
        "origin": "参数与操作来自原日志；触发指标和局部复测为离线追溯/补算，不是当时的逐区间模型解释。",
        "items": summaries,
    }
    write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("eval/runs/vital-cleaning-v1-20261112-r4"))
    parser.add_argument("--out", type=Path, default=Path("eval/runs/vital-all-signal-gallery-20261113/explanations"))
    args = parser.parse_args()
    result = build(args.run, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False, indent=2))
