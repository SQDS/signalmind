"""Prepare an input-selected real PulseDB-Vital pilot with exact time continuity."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from real_processing_helper import PROTOCOL, save_npz  # noqa: E402

SOURCE = Path("/home/yanghaoxiang24/bp2rr_new/dataset/Vital/PulseDB_Vital")
SPLITS = Path("/home/yanghaoxiang24/bp2rr_new/preprocess/data/v1_20260920/patient_splits.json")
PACK = ROOT / "eval/cases/vital-real-20261105"
REPORT = ROOT / "eval/results/vital-ds-20261105"
FIELDS = {"ECG": "ECG_Record", "PPG": "PPG_Record", "ABP": "ABP_Raw"}
FS = 125


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(value):
    return hashlib.sha256(("vital-pilot-20261105:" + value).encode()).hexdigest()


class PulseMat:
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        self.group = self.file["Subj_Wins"]
        self.refs = {}
        field = self.group["SegmentID"]
        self.n = field.size if h5py.check_dtype(ref=field.dtype) is not None else 1

    def array(self, name, index):
        field = self.group[name]
        if h5py.check_dtype(ref=field.dtype) is not None:
            if name not in self.refs:
                self.refs[name] = np.asarray(field).ravel()
            field = self.file[self.refs[name][index]]
        return np.asarray(field).reshape(-1)

    def metadata(self, index):
        time = self.array("T", index)
        case = np.asarray(self.array("CaseID", index), dtype="<u2").tobytes().decode("utf-16-le").rstrip("\0")
        subject = np.asarray(self.array("SubjectID", index), dtype="<u2").tobytes().decode("utf-16-le").rstrip("\0")
        return {
            "index": index,
            "case": case,
            "subject": subject,
            "segment": int(self.array("SegmentID", index)[0]),
            "start": float(time[0]),
            "end": float(time[-1]),
            "regular": len(time) == 1250 and bool(np.all(np.abs(np.diff(time) - 1 / FS) <= 1e-6)),
        }

    def window(self, start):
        meta = [self.metadata(i) for i in range(start, start + 6)]
        if not continuous(meta):
            raise ValueError("Candidate does not form a continuous60-second window")
        time = np.concatenate([self.array("T", i) for i in range(start, start + 6)])
        assert len(time) == 7500 and np.all(np.abs(np.diff(time) - 1 / FS) <= 1e-6)
        waves = {
            kind: np.concatenate([self.array(field, i) for i in range(start, start + 6)]).astype(np.float64)
            for kind, field in FIELDS.items()
        }
        assert all(len(x) == 7500 for x in waves.values())
        return waves, time, meta


def continuous(meta):
    return (
        len(meta) == 6
        and all(r["regular"] for r in meta)
        and all(
            a["subject"] == b["subject"]
            and a["case"] == b["case"]
            and b["segment"] == a["segment"] + 1
            and abs(b["start"] - a["end"] - 1 / FS) <= 1e-6
            for a, b in zip(meta[:-1], meta[1:], strict=True)
        )
    )


def diagnostics(x):
    finite = x[np.isfinite(x)]
    if len(finite) != len(x):
        return {"finite_fraction": float(len(finite) / len(x)), "roughness": 0.0, "line_fraction": 0.0}
    scale = max(float(np.percentile(x, 95) - np.percentile(x, 5)), 1e-12)
    f, p = welch(x, fs=FS, nperseg=1000, detrend="constant")
    line = ((f >= 49) & (f <= 51)) | ((f >= 59) & (f <= 61))
    return {
        "finite_fraction": 1.0,
        "roughness": float(np.percentile(np.abs(np.diff(x)), 99.9) / scale),
        "line_fraction": float(p[line].sum() / max(p[f >= 0.5].sum(), 1e-30)),
    }


def scan():
    assignments = json.loads(SPLITS.read_text())["assignments"]
    files = sorted(
        [p for p in SOURCE.glob("*.mat") if assignments.get("vital::" + p.stem) == "train"], key=lambda p: rank(p.name)
    )[:24]
    candidates, file_stats = [], []
    for number, path in enumerate(files):
        mat = PulseMat(path)
        selected_starts = set()
        for block_start in sorted({0, max(0, mat.n // 2 - 24), max(0, mat.n - 48)}):
            end = min(mat.n, block_start + 48)
            metadata = [mat.metadata(i) for i in range(block_start, end)]
            possible = [
                metadata[j]["index"] for j in range(max(0, len(metadata) - 5)) if continuous(metadata[j : j + 6])
            ]
            if possible:
                selected_starts.update((possible[0], possible[-1]))
        for start in sorted(selected_starts):
            waves, time, metadata = mat.window(start)
            features = {kind: diagnostics(x) for kind, x in waves.items()}
            candidates.append(
                {
                    "source_file": str(path),
                    "start_row": start,
                    "source_start_s": float(time[0]),
                    "case_hash": rank(metadata[0]["case"]),
                    "features": features,
                }
            )
        file_stats.append(
            {"source_file": str(path), "stored_segments": int(mat.n), "candidate_windows": len(selected_starts)}
        )
        mat.file.close()
        print(f"Scanned source {number + 1}/24; candidates {len(candidates)}", flush=True)
    chosen, used = [], set()
    for kind, feature, label in (
        ("ECG", "line_fraction", "ECG窄带成分较突出"),
        ("PPG", "roughness", "PPG局部变化较突出"),
        ("ABP", "roughness", "ABP局部变化较突出"),
    ):
        available = [r for r in candidates if r["source_file"] not in used]
        item = max(available, key=lambda r: r["features"][kind][feature])
        chosen.append({**item, "selection_reason": label})
        used.add(item["source_file"])
    available = [r for r in candidates if r["source_file"] not in used]
    available.sort(key=lambda r: sum(r["features"][k]["roughness"] + r["features"][k]["line_fraction"] for k in FIELDS))
    chosen.append({**available[len(available) // 2], "selection_reason": "输入诊断居中的对照片段"})
    return candidates, chosen, file_stats


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    private = PACK / "private"
    private.mkdir(parents=True, exist_ok=False)
    private.chmod(0o700)
    candidates, chosen, scan_stats = scan()
    (private / "selection.json").write_text(
        json.dumps({"candidates": candidates, "chosen": chosen, "scan": scan_stats}, indent=2, ensure_ascii=False)
        + "\n"
    )
    cases, copies = [], []
    for number, item in enumerate(chosen, 1):
        record = f"V{number:03d}"
        source = Path(item["source_file"])
        target = private / f"{record}.mat"
        source_hash = sha(source)
        shutil.copyfile(source, target)
        assert sha(target) == source_hash == sha(source)
        mat = PulseMat(target)
        waves, time, metadata = mat.window(item["start_row"])
        mat.file.close()
        save_npz(private / f"{record}-original.npz", time=time, **waves)
        copies.append(
            {
                **item,
                "anonymous_record": record,
                "source_sha256": source_hash,
                "copied_source": str(target),
                "bytes": target.stat().st_size,
                "continuity": {
                    "same_full_case_id": True,
                    "segment_id_step": 1,
                    "all_sample_time_steps_checked": True,
                    "fs_hz": FS,
                    "n_samples": 7500,
                },
            }
        )
        for kind, x in waves.items():
            cid = f"{record}_{kind}"
            folder = PACK / "public/cases" / cid
            folder.mkdir(parents=True)
            save_npz(
                folder / "input.npz",
                values=x[None, :],
                observed_missing_mask=(~np.isfinite(x))[None, :],
                device_saturation_flag=np.zeros((1, len(x)), bool),
            )
            unit = "mmHg" if kind == "ABP" else "source_units"
            scale = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
            task = {
                "schema_version": PROTOCOL,
                "case_id": cid,
                "source_type": "real_PulseDB_Vital_v2",
                "signal": {
                    "signal_id": "input_" + cid,
                    "artifact_uri": "input.npz",
                    "sha256": sha(folder / "input.npz"),
                    "channels": [
                        {
                            "name": kind.lower(),
                            "kind": kind,
                            "unit": unit,
                            "fs_hz": FS,
                            "n_samples": len(x),
                            "t0_s": 0.0,
                        }
                    ],
                },
                "objective": {
                    "evaluation_track": "morphology",
                    "purpose": "reference_free_preprocessing_review",
                    "range_samples": [0, len(x)],
                    "preserve": [
                        "sample_count",
                        "time_axis",
                        "source_amplitude_scale",
                        "physiological_events",
                        "slow_pressure_changes",
                    ],
                },
                "public_policy": {
                    "allow_resample": False,
                    "allow_time_shift": False,
                    "max_gap_samples": 2,
                    "max_transient_ms": 80.0,
                    "protected_event_windows": [],
                    "saturation_flag_status": "No device saturation flag was supplied; the zero mask does not certify absence of clipping.",
                    "no_clean_reference": True,
                    "unit_note": "ABP_Raw is mmHg. ECG/PPG retain source amplitude; physical units are not asserted from numeric range.",
                },
                "downstream_contract": {
                    "detector_prominence": max(0.2 * scale, 1e-12),
                    "meaning": "Fixed candidate-event and Welch measurements, not clinical labels or quality scores.",
                },
                "submission_contract": {
                    "metadata": "result.json",
                    "arrays": "output.npz",
                    "shape": [1, len(x)],
                    "intervals": "zero_based_half_open_samples",
                },
            }
            (folder / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")
            prompt = (
                f"对匿名真实记录 {cid} 的60秒{kind}波形做保留信息的预处理。采样率125Hz，幅值保持源数据单位。\n"
                "先判断是否存在有依据的干扰或孤立异常，再决定保留、局部修补、去噪或标记不可用。原样保留是有效选择。\n"
                "这批数据已经过数据集的基础筛选，不代表每个尖峰或不规则搏动都是噪声。ECG的QRS和可能真实的形态变化应保留；"
                "PPG/ABP的收缩上升沿、峰幅、切迹、舒张段和慢变化不能仅为平滑而抹掉。\n"
                "只依据当前观测选择工具、区间与参数。无法区分生理变化与伪迹时，保留观测并说明不确定性。"
                "长缺失、边界缺失及无法支持的重建不应编造。\n"
                "选择一个版本导出result.json和output.npz，解释处理位置、依据和剩余问题。"
                "本任务没有未加噪真值；候选峰、频带功率和前后指标只能用于诊断，不能据此宣称临床准确率提升。"
            )
            (folder / "prompt.txt").write_text(prompt)
            (folder / "SCORING_RULES.txt").write_text(
                "本次为真实来源波形探索性试跑，没有干净参考，没有波形准确率或真实SNR评分。\n"
                "检查交付的float64波形、四种bool掩码、单位、采样率、样本数、时间轴、哈希、操作范围及前后指标。\n"
                "观察修改范围、有效支持、候选事件变化和频带功率。指标改善不自动证明更接近真实生理信号。"
                "不要把其他预滤波字段、预计算峰值或SBP标签当干净真值。无需为通过测试而强制处理。\n"
            )
            (folder / "OUTPUT_FORMAT.txt").write_text(
                "提交/output/result.json和/output/output.npz。values为float64、shape=(1,7500)；"
                "artifact_mask、modified_mask、repaired_mask、valid_mask为同形bool。"
                "action允许keep、repair、denoise、mask_only、partial_repair_and_mask、uncertain。"
                "使用公开submission_helper.write_submission生成记录，SignalMind的ts_export调用同一函数。\n"
            )
            (folder / "submission.schema.json").write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": [
                            "schema_version",
                            "case_id",
                            "action",
                            "signal",
                            "array_layout",
                            "operations",
                            "measurements",
                        ],
                    },
                    indent=2,
                )
            )
            shutil.copyfile(ROOT / "eval/real_processing_helper.py", folder / "submission_helper.py")
            np.savetxt(
                folder / "signal.csv",
                np.column_stack([np.arange(len(x)) / FS, x]),
                delimiter=",",
                header="time_s," + kind.lower(),
                comments="",
                fmt="%.17g",
            )
            cases.append(
                {
                    "case_id": cid,
                    "record": record,
                    "channel": kind,
                    "unit": unit,
                    "samples": len(x),
                    "fs_hz": FS,
                    "source_field": FIELDS[kind],
                    "input_sha256": task["signal"]["sha256"],
                }
            )
    (PACK / "public/index.json").write_text(json.dumps({"case_ids": [r["case_id"] for r in cases]}, indent=2) + "\n")
    (private / "source-copies.json").write_text(json.dumps(copies, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "source_files_available": 2938,
        "source_pool": "Existing BP2RR training partition only",
        "source_records_scanned": len(scan_stats),
        "candidate_windows": len(candidates),
        "records_selected": len(copies),
        "channel_tasks": len(cases),
        "selection": "Input-only coverage sampling: prominent ECG line power, PPG/ABP local changes, and a middle-ranked control; unique source files.",
        "representative_random_sample": False,
        "clean_reference_available": False,
        "added_synthetic_noise": False,
        "source_identifiers_sent_to_model": False,
        "demographics_or_labels_sent_to_model": False,
        "model_processing": "each channel separately",
        "source_mat_bytes_copied": sum(c["bytes"] for c in copies),
        "cases": cases,
        "upstream_url": "https://github.com/pulselabteam/PulseDB",
        "license": "CC BY-NC-SA 4.0",
        "source_mapping": "private local provenance; not mounted in model tool sessions",
    }
    (REPORT / "data-preparation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
