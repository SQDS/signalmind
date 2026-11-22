"""Reproducible offline locator comparison: prepare -> dev -> freeze -> synthetic/Vital -> report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import selectors
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np  # noqa: E402

from tsmcp.cleaning.common import digest, read, sha, write  # noqa: E402
from tsmcp.locators.api import AUXILIARY, CONFIG, METHODS  # noqa: E402

DEFAULT_OUT = ROOT / "eval/runs/locator-comparison-20261115"


def stamp():
    return datetime.now(UTC).isoformat()


def one(array):
    return array[0] if array.ndim == 2 and array.shape[0] == 1 else array


def prepare(out):
    out = Path(out)
    if (out / "manifest.json").exists():
        return read(out / "manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for version, stage in (("0.1", "dev"), ("0.3", "synthetic")):
        pack = ROOT / f"eval/cases/processing-v{version}/SignalMind_Test_Pack_v{version}"
        manifest = read(pack / "private/manifest.json")
        for item in manifest["cases"]:
            cid = item["case_id"]
            path = pack / "public/cases" / cid / "input.npz"
            task = read(path.parent / "task.json")
            channel = task["signal"]["channels"][0]
            record = {
                "stage": stage,
                "case_id": cid,
                "input_path": str(path),
                "input_sha256": item["input_sha256"],
                "reference_path": str(pack / "private/cases" / cid / "reference.npz"),
                "reference_sha256": item["reference_sha256"],
                "channel": item["kind"],
                "fs": item["fs_hz"],
                "unit": channel.get("unit", channel.get("units", "source_units")),
                "source_cluster": item["source_cluster"],
                "family": item.get("family", item["expected_class"]),
                "generator_family": item.get("generator_family"),
                "duration_s": item["duration_s"],
                "n": round(item["duration_s"] * item["fs_hz"]),
                "source_exclusions": [],
                "reference_available": True,
            }
            records.append(record)
    vital = ROOT / "eval/runs/vital-cleaning-v1-20261112-r4"
    manifest = read(vital / "dataset_manifest.json")
    for case in manifest["cases"]:
        records.append(
            {
                "stage": "vital",
                "case_id": case["id"],
                "input_path": str(vital / "public" / case["id"] / "input.npz"),
                "input_sha256": case["input_sha256"],
                "channel": case["task"],
                "fs": case["fs"],
                "unit": case["unit"],
                "source_cluster": case["record_id"],
                "family": "unlabeled_vital",
                "patient_id": case["record_id"],
                "n": case["n"],
                "duration_s": case["duration_s"],
                "source_exclusions": case.get("source_time_conflicts", []),
                "reference_available": False,
            }
        )
    for row in records:
        source = Path(row["input_path"])
        if sha(source) != row["input_sha256"]:
            raise ValueError(f"Input changed: {row['case_id']}")
        relative = f"{row['stage']}/{row['case_id']}.npz"
        target = out / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        row["input_relative"] = relative
    manifest = {
        "schema": "signalmind.locator-comparison.v1",
        "created_at": stamp(),
        "records": records,
        "stage_counts": {s: sum(r["stage"] == s for r in records) for s in ("dev", "synthetic", "vital")},
        "real_patients": 10,
        "api_calls": 0,
        "dataset_claim": "exposed_development_sets; Vital has no complete pollution truth",
    }
    write(out / "manifest.json", manifest)
    return manifest


def source_files():
    files = sorted(p for p in (ROOT / "src/tsmcp").rglob("*") if p.is_file() and p.suffix in {".py", ".json"})
    files += [ROOT / "eval/locator_worker.py", ROOT / "eval/run_locator_comparison.py", ROOT / "eval/locator_plots.py"]
    return files


def freeze(out, *, formal):
    import platform

    import scipy

    out = Path(out)
    manifest = read(out / "manifest.json")
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files()}
    fingerprint = digest(hashes)
    folder = out / "snapshots" / fingerprint
    for name in hashes:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    protocol = {
        "created_at": stamp(),
        "implementation_hash": fingerprint,
        "files": hashes,
        "config": CONFIG,
        "methods": list(METHODS),
        "auxiliary": AUXILIARY,
        "common_preprocessing": "split genuine time gaps, source exclusions and nonfinite observations identically; no imputation",
        "original_adapter": "original Python bytes unchanged; all methods receive common finite runs; original inclusive event endpoint converted as declared; coverage uncertified",
        "workers": 2,
        "threads_per_worker": 1,
        "timeout_s": 60,
        "memory_gib": 4,
        "event_iou_thresholds": [0.1, 0.3, 0.5],
        "primary_event_iou": 0.3,
        "manifest_sha256": sha(out / "manifest.json"),
        "dataset_status": manifest["dataset_claim"],
        "primary_sensitivity": "balanced",
        "tuning_on_synthetic_80": False,
        "reference_access": "host scorer only; detector subprocesses receive only public NPZ files",
        "metric_claim": "development comparison, no clinical validation or AI-vision claim",
        "model_api_calls": 0,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    if formal:
        path = out / "freeze.json"
        if path.exists() and read(path)["implementation_hash"] != fingerprint:
            raise ValueError("Formal freeze already exists; retain it and create a separate experiment version")
        if not (out / "stage-dev.json").exists():
            raise ValueError("Run development checks before the formal freeze")
        write(path, protocol)
    else:
        write(out / "development-freeze.json", protocol)
    return protocol


def sandbox_command(out, fingerprint):
    python = Path(sys.executable).absolute()
    prefix = Path(sys.base_prefix).resolve()
    cmd = ["bwrap", "--unshare-net", "--unshare-pid", "--new-session", "--die-with-parent", "--clearenv"]
    for name in ("/usr", "/lib", "/lib64", "/bin", "/sbin"):
        if Path(name).exists():
            cmd += ["--ro-bind", name, name]
    for name in ("/etc/ld.so.cache", "/etc/fonts", "/etc/passwd", "/etc/group"):
        if Path(name).exists():
            cmd += ["--ro-bind", name, name]
    cmd += [
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(Path(sys.prefix)),
        str(Path(sys.prefix)),
        "--ro-bind",
        str(prefix),
        str(prefix),
        "--ro-bind",
        str(out / "snapshots" / fingerprint),
        "/impl",
        "--ro-bind",
        str(out / "inputs"),
        "/inputs",
        "--bind",
        str(out / "worker_outputs"),
        "/output",
    ]
    for key, value in {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "MPLCONFIGDIR": "/tmp/mpl",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }.items():
        cmd += ["--setenv", key, value]
    if python.is_symlink():
        alias = python.readlink().parent.parent
        if alias != prefix:
            cmd += ["--ro-bind", str(alias), str(alias)]
    return cmd + ["--chdir", "/output", "--", str(python), "-I", "/impl/eval/locator_worker.py"]


class Worker:
    def __init__(self, out, fingerprint, index):
        self.log = (out / f"worker-{index}.log").open("a")
        self.process = subprocess.Popen(
            sandbox_command(out, fingerprint),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        ready = self.receive(45)
        if not ready.get("ready") or ready["home_credentials_visible"] or ready["reference_visible"]:
            raise RuntimeError(f"Worker isolation preflight failed: {ready}")
        self.ready = ready

    def receive(self, timeout):
        if not self.selector.select(timeout):
            raise TimeoutError("Isolated worker protocol stalled")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Isolated worker exited {self.process.poll()}")
        return json.loads(line)

    def run(self, job):
        self.process.stdin.write(json.dumps(job) + "\n")
        self.process.stdin.flush()
        return self.receive(job["timeout_s"] + 15)

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.selector.close()
        self.log.close()


def run_stage(out, stage, *, limit=None, methods=None):
    out = Path(out).resolve()
    manifest = read(out / "manifest.json")
    protocol = read(out / ("development-freeze.json" if stage == "dev" else "freeze.json"))
    fingerprint = protocol["implementation_hash"]
    for name, checksum in protocol["files"].items():
        if sha(out / "snapshots" / fingerprint / name) != checksum:
            raise ValueError("Frozen implementation snapshot changed")
    if sha(out / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Frozen data scope changed")
    records = [r for r in manifest["records"] if r["stage"] == stage]
    if limit is not None:
        records = records[:limit]
    selected_methods = methods or [*METHODS, AUXILIARY]
    jobs, cached = [], 0
    for record in records:
        for method in selected_methods:
            relative = f"{stage}/{record['case_id']}/{method}"
            path = out / "worker_outputs" / relative / "result.json"
            if path.exists():
                previous = read(path)
                if previous["input_sha256"] != record["input_sha256"] or previous["implementation_hash"] != fingerprint:
                    raise ValueError("Existing receipt belongs to another input/implementation; do not overwrite it")
                cached += 1
                continue
            jobs.append(
                {
                    "stage": stage,
                    "case_id": record["case_id"],
                    "fs": record["fs"],
                    "method": method,
                    "input_relative": record["input_relative"],
                    "input_sha256": record["input_sha256"],
                    "source_exclusions": record["source_exclusions"],
                    "output_relative": relative,
                    "implementation_hash": fingerprint,
                    "timeout_s": 60,
                }
            )
    (out / "worker_outputs").mkdir(exist_ok=True)
    tasks = queue.Queue()
    for job in jobs:
        tasks.put(job)
    started, completed, failures, preflights = time.monotonic(), [], [], []

    def consume(index):
        worker = Worker(out, fingerprint, index)
        preflights.append(worker.ready)
        try:
            while True:
                try:
                    job = tasks.get_nowait()
                except queue.Empty:
                    break
                result = worker.run(job)
                receipt = {"time": stamp(), "case_id": job["case_id"], "method": job["method"], **result}
                completed.append(receipt)
                if result["status"] in {"failed", "timeout"}:
                    failures.append(receipt)
                # Parent-side append is one write per line; separate threads never share a buffered stream.
                with (out / f"progress-{stage}.jsonl").open("a") as stream:
                    stream.write(json.dumps(receipt) + "\n")
                if len(completed) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": stage,
                                "done": len(completed) + cached,
                                "total": len(jobs) + cached,
                                "failures": len(failures),
                                "elapsed_s": round(time.monotonic() - started, 1),
                            }
                        ),
                        flush=True,
                    )
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(consume, i) for i in range(2)]
        for future in futures:
            future.result()
    summary = {
        "stage": stage,
        "cases": len(records),
        "methods": selected_methods,
        "completed_jobs": len(completed) + cached,
        "new_jobs": len(completed),
        "cached_jobs": cached,
        "failures_this_execution": failures,
        "elapsed_s": time.monotonic() - started,
        "implementation_hash": fingerprint,
        "scope_complete": limit is None and selected_methods == [*METHODS, AUXILIARY],
        "preflights": preflights,
        "source_arrays_modified": False,
        "model_api_calls": 0,
    }
    write(out / f"stage-{stage}.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"failures_this_execution", "preflights"}}), flush=True
    )
    return summary


def load_input(out, record):
    with np.load(Path(out) / "inputs" / record["input_relative"], allow_pickle=False) as z:
        values = one(z["values"]).copy()
        times = one(z["time"]).copy() if "time" in z else np.arange(len(values)) / record["fs"]
        flags = {k: one(z[k]).copy() for k in ("device_saturation_flag", "observed_missing_mask") if k in z}
    source = np.ones(len(values), bool)
    for a, b in record["source_exclusions"]:
        source[a:b] = False
    return values, times, flags, source


def score_stage(out, stage):
    from tsmcp.locators.evaluation import aggregate_rows, score_result

    out = Path(out)
    records = [r for r in read(out / "manifest.json")["records"] if r["stage"] == stage]
    scored = []
    for record in records:
        values, times, flags, source = load_input(out, record)
        reference = None
        if record["reference_available"]:
            if sha(record["reference_path"]) != record["reference_sha256"]:
                raise ValueError("Reference changed")
            with np.load(record["reference_path"], allow_pickle=False) as z:
                reference = {
                    k: one(z[k]).copy() for k in z.files if k in {"A", "C", "D", "E", "U"} or k.startswith("component_")
                }
        for method in METHODS:
            path = out / "worker_outputs" / stage / record["case_id"] / method / "result.json"
            result = (
                read(path)
                if path.exists()
                else {
                    "method": method,
                    "status": "failed",
                    "events": [],
                    "coverage_known": True,
                    "evaluated_ranges": [],
                    "runtime": {},
                    "error": "missing_receipt",
                }
            )
            metrics = score_result(
                result,
                input_values=values,
                fs=record["fs"],
                reference=reference,
                device_flags=flags,
                source_support=source,
            )
            scored.append(
                {
                    "case_id": record["case_id"],
                    "channel": record["channel"],
                    "family": record["family"],
                    "source_cluster": record["source_cluster"],
                    "patient_id": record.get("patient_id"),
                    "stage": stage,
                    **metrics,
                }
            )
    write(out / f"scores-{stage}.json", scored)
    summary = aggregate_rows(scored)
    write(out / f"summary-{stage}.json", summary)
    rows = []
    for r in scored:
        row = {k: r.get(k) for k in ("case_id", "method", "channel", "family", "source_cluster", "status")}
        for track, data in r.get("tracks", {}).items():
            if isinstance(data, dict):
                row.update(
                    {
                        track + "_" + key: value
                        for key, value in data.items()
                        if value is None or isinstance(value, (str, int, float, bool))
                    }
                )
        row.update({"runtime_" + k: v for k, v in r.get("runtime", {}).items()})
        rows.append(row)
    keys = sorted({k for r in rows for k in r})
    with (out / f"scores-{stage}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def result_for(out, record, method):
    path = Path(out) / "worker_outputs" / record["stage"] / record["case_id"] / method / "result.json"
    return (
        read(path)
        if path.exists()
        else {"method": method, "status": "failed", "events": [], "coverage_known": True, "evaluated_ranges": []}
    )


def prediction_mask(result, n):
    mask = np.zeros(n, bool)
    for event in result.get("events", []):
        mask[event["start"] : event["end"]] = True
    return mask


def render_figures(out, stage):
    from locator_plots import render_case, render_sphere

    out = Path(out).resolve()
    records = [r for r in read(out / "manifest.json")["records"] if r["stage"] == stage]
    selected = []
    if stage in {"dev", "synthetic"}:
        selected = [{"record": r, "reason": "every_synthetic_case", "window": None} for r in records]
    else:
        groups = {}
        for record in records:
            groups.setdefault((record["patient_id"], record["channel"]), []).append(record)
        for group in groups.values():
            candidates = []
            for r in group:
                masks = [prediction_mask(result_for(out, r, method), r["n"]) for method in METHODS]
                stack = np.stack(masks)
                disagreement = stack.any(axis=0) & ~stack.all(axis=0)
                d_result = result_for(out, r, METHODS[3])
                d_scores = [e["score"] for e in d_result.get("events", [])]
                candidates.append(
                    (r, int(disagreement.sum()), max(d_scores, default=0), bool(stack.any()), disagreement)
                )
            picks = [
                (max(candidates, key=lambda x: x[1]), "largest_method_disagreement"),
                (max(candidates, key=lambda x: x[2]), "highest_fixed_sphere_score"),
                (
                    min([x for x in candidates if not x[3]] or candidates, key=lambda x: digest(x[0]["case_id"])),
                    "no_alarm_control_or_low_available",
                ),
                (min(candidates, key=lambda x: digest([111101, x[0]["case_id"]])), "deterministic_random"),
            ]
            seen = set()
            for (record, _, _, _, disagreement), reason in picks:
                if record["case_id"] in seen:
                    continue
                seen.add(record["case_id"])
                result = result_for(out, record, METHODS[3])
                if reason == "largest_method_disagreement" and disagreement.any():
                    width = max(1, round(10 * record["fs"]))
                    counts = np.convolve(
                        disagreement.astype(int), np.ones(min(width, len(disagreement)), int), mode="same"
                    )
                    center = int(np.argmax(counts))
                elif reason == "highest_fixed_sphere_score" and result.get("events"):
                    event = max(result["events"], key=lambda e: e["score"])
                    center = (event["start"] + event["end"]) // 2
                else:
                    center = record["n"] // 2
                width = min(record["n"], round(20 * record["fs"]))
                left = max(0, min(record["n"] - width, center - width // 2))
                selected.append({"record": record, "reason": reason, "window": [left, left + width]})
    manifest, mesh_picks = [], {}
    for item in selected:
        record, window = item["record"], item["window"]
        values, times, _, _ = load_input(out, record)
        reference = None
        if record["reference_available"]:
            with np.load(record["reference_path"], allow_pickle=False) as z:
                reference = {"A": one(z["A"])}
        results = {m: result_for(out, record, m) for m in METHODS}
        path = out / "figures" / stage / (record["case_id"] + ".png")
        result = render_case(
            values,
            record["fs"],
            results,
            path,
            case_label=f"{record['case_id']} | {record['channel']} | {record['unit']}",
            reference=reference,
            times=times,
            window=window,
        )
        manifest.append(
            {"case_id": record["case_id"], "patient_id": record.get("patient_id"), "reason": item["reason"], **result}
        )
        # One diagnostic sphere per synthetic family/channel, or per Vital patient/channel.
        key = (record.get("patient_id") or record["family"], record["channel"])
        mesh_picks.setdefault(key, record)
    protocol = read(out / ("development-freeze.json" if stage == "dev" else "freeze.json"))
    worker = Worker(out, protocol["implementation_hash"], "plot")
    meshes = []
    try:
        for record in mesh_picks.values():
            relative = f"visualization/{stage}/{record['case_id']}/{METHODS[3]}"
            folder = out / "worker_outputs" / relative
            if not (folder / "result.json").exists():
                worker.run(
                    {
                        "stage": stage,
                        "case_id": record["case_id"],
                        "fs": record["fs"],
                        "method": METHODS[3],
                        "input_relative": record["input_relative"],
                        "input_sha256": record["input_sha256"],
                        "source_exclusions": record["source_exclusions"],
                        "output_relative": relative,
                        "implementation_hash": protocol["implementation_hash"],
                        "timeout_s": 60,
                        "include_mesh": True,
                    }
                )
            replay = read(folder / "result.json")
            original = result_for(out, record, METHODS[3])
            if replay.get("events") != original.get("events"):
                raise ValueError("Mesh generation altered detection results")
            if not replay.get("mesh_files"):
                continue
            payload = max(replay["mesh_files"], key=lambda x: x["range"][1] - x["range"][0])
            with np.load(folder / payload["arrays"], allow_pickle=False) as z:
                mesh = {k: z[k].copy() for k in z.files}
            mesh.update(read(folder / payload["metadata"]))
            result = render_sphere(
                mesh,
                out / "figures" / stage / (record["case_id"] + "-sphere.png"),
                case_label=record["case_id"] + " | D_sphere_fixed",
                fs=record["fs"],
                sample_offset=payload["range"][0],
            )
            meshes.append(
                {
                    "case_id": record["case_id"],
                    "input_sha256": record["input_sha256"],
                    "numerical_result_unchanged": True,
                    **result,
                }
            )
    finally:
        worker.close()
    write(
        out / f"figures-{stage}.json",
        {
            "selection_seed": 111101,
            "waveform_comparisons": manifest,
            "sphere_diagnostics": meshes,
            "model_api_calls": 0,
            "sampling_is_not_complete_visual_review": stage == "vital",
        },
    )
    return manifest


def write_report(out):
    out = Path(out).resolve()
    summaries = {
        s: read(out / f"summary-{s}.json")
        for s in ("dev", "synthetic", "vital")
        if (out / f"summary-{s}.json").exists()
    }

    def fmt(value, percent=False):
        return "null" if value is None else f"{value * 100:.2f}%" if percent else f"{value:.4f}"

    text = [
        "# 时间球定位器 v0.1 离线对照",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-11-15",
        "- Verification Status: EXECUTED (engineering comparison; not clinical validation)",
        "- Version Label: locator_comparison_v1",
        "",
        "只读定位；没有模型调用、波形清洗或 RR 提取。合成题为已曝光开发集，Vital 没有完整污染真值。",
        "",
        "原稿源码字节保持不变；统一适配器对所有方法按有限观测/来源支持/真实断点分段，不插值跨缺口。原稿含尾端的事件按其声明转换为半开；原始输出另存。",
        "",
        "修正版保持四个评分公式与阈值，只修输入、FFT、有效尾圈、回映和边界。覆盖表示至少一项可算，逐项覆盖单独保留；球面渲染不参与评分。",
        "",
    ]
    for stage in ("dev", "synthetic", "vital"):
        if stage not in summaries:
            continue
        summary = summaries[stage]
        text += [
            f"## {stage}",
            "",
            "| 方法 | 主域召回 | 主域精确率 | 时间 F1 | 正常区误报比例 | 告警时间占比 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method, row in summary["by_method"].items():
            track = row["tracks"]["unknown_observed"]["micro"]
            alarm = track["alarm_s"] / track["domain_s"] if track["domain_s"] else None
            text.append(
                f"| {method} | {fmt(track.get('recall'), True)} | {fmt(track.get('precision'), True)} | {fmt(track.get('f1'))} | {fmt(track.get('normal_fp_ratio'), True)} | {fmt(alarm, True)} |"
            )
        text += ["", f"逐条结果：[{stage} CSV](scores-{stage}.csv)；[完整汇总](summary-{stage}.json)。", ""]
        if stage == "vital":
            text += [
                "所有准确率相关字段为 null；告警比例不是污染比例。10名患者，每通道73590秒；三个通道任务时长不能相加解释成独立患者小时。",
                "",
            ]
    text += [
        "## 解释与复现",
        "",
        "- 原稿及修正版的正常尖锐波形误报和窄尖峰漏检属于方法结果，没有调公式掩盖。",
        "- 公共缺失/设备标志的贡献单独计入 all_problems，主域 unknown_observed 排除这些已公开问题。",
        "- change/discord 保存在 X_legacy_auxiliary 中，仅为相对变化候选，不混入污染主评分。",
        "- 逐方法/连续段60秒超时、4 GiB内存、两个隔离worker、单线程数值运算；失败保留，不自动重试。",
        "- 代码及输入哈希见 freeze.json / manifest.json；算法看不到私有参考和凭据。",
        "- 图册仅显示实际数值结果；真实抽查图不构成定位准确率验证。",
        "",
        "```bash",
        f".venv/bin/python eval/run_locator_comparison.py run --out {out} --stage synthetic",
        f".venv/bin/python eval/run_locator_comparison.py run --out {out} --stage vital",
        f".venv/bin/python eval/run_locator_comparison.py score --out {out} --stage synthetic",
        f".venv/bin/python eval/run_locator_comparison.py score --out {out} --stage vital",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(text))
    import html

    cards = []
    for stage in ("synthetic", "vital"):
        index = out / f"figures-{stage}.json"
        if not index.exists():
            continue
        figures = read(index)
        for kind in ("waveform_comparisons", "sphere_diagnostics"):
            for row in figures[kind]:
                path = Path(row["path"]).relative_to(out)
                cards.append(
                    f'<section><h3>{html.escape(stage + " · " + row["case_id"] + " · " + kind)}</h3><a href="{path}"><img loading="lazy" src="{path}" alt="{html.escape(row["case_id"])}"></a></section>'
                )
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>时间球定位对照</title><style>body{font:15px system-ui;background:#f3f5f8;color:#233545;margin:25px auto;max-width:1400px;padding:0 20px}section{background:white;padding:15px;margin:20px 0;border-radius:8px}img{width:100%}h3{font-size:15px}a{color:#167d82}</style><h1>时间球定位器离线对照</h1><p>同窗原始波形与四组候选告警。真实数据没有完整污染真值；球面显示不参与检测。</p><p><a href="README.md">结果报告</a> · <a href="scores-synthetic.csv">80题逐条结果</a> · <a href="scores-vital.csv">Vital逐条结果</a></p>'
        + "\n".join(cards)
        + "</html>"
    )
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "freeze-dev", "freeze", "run", "score", "figures", "report"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stage", choices=["dev", "synthetic", "vital"], default="dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--methods", nargs="+")
    args = p.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.out)["stage_counts"]))
    elif args.command.startswith("freeze"):
        protocol = freeze(args.out, formal=args.command == "freeze")
        print(json.dumps({k: protocol[k] for k in ("created_at", "implementation_hash", "methods", "environment")}))
    elif args.command == "run":
        run_stage(args.out, args.stage, limit=args.limit, methods=args.methods)
    elif args.command == "score":
        score_stage(args.out, args.stage)
    elif args.command == "figures":
        render_figures(args.out, args.stage)
    else:
        write_report(args.out)


if __name__ == "__main__":
    main()
