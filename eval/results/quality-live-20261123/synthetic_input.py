"""Synthetic-only inputs for a live end-to-end test. No patient files, features, or waveforms are read."""

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "runs/plan-live-20261123/synthetic-study"


def signals(fs=125, seconds=75):
    rng = np.random.default_rng(230911)
    t = np.arange(fs * seconds) / fs
    beats = [0.6]
    while beats[-1] < seconds + 1:
        b = beats[-1]
        beats.append(b + 0.81 + 0.046 * np.sin(2 * np.pi * b / 17) + rng.normal(0, 0.009))
    ecg = 0.035 * np.sin(2 * np.pi * t / 8)
    ppg = 35.0 + 0.45 * np.sin(2 * np.pi * t / 12)
    abp = 71.0 + 1.5 * np.sin(2 * np.pi * t / 12)
    for i, b in enumerate(beats):
        q = t - b
        amp = 1 + 0.045 * np.sin(i / 4)
        for offset, width, height in [
            (-0.145, 0.028, 0.105),
            (-0.023, 0.008, -0.13),
            (0, 0.012, 1.0),
            (0.027, 0.011, -0.20),
            (0.23, 0.047, 0.20),
        ]:
            ecg += amp * height * np.exp(-0.5 * ((q - offset) / width) ** 2)
        for ch, delay, scale, shape in [(ppg, 0.17, 5.8, 0.072), (abp, 0.10, 43, 0.053)]:
            u = np.maximum(q - delay, 0) / shape
            ch += amp * scale * ((u**2 / 4) * np.exp(2 - u) + 0.095 * np.exp(-0.5 * ((q - delay - 0.34) / 0.045) ** 2))
    for signal, sd in [(ecg, 0.004), (ppg, 0.02), (abp, 0.10)]:
        signal += rng.normal(0, sd, len(t))
    return t, np.asarray(beats), {"ECG": ecg, "PPG": ppg, "ABP": abp}


def main():
    inputs = OUT / "inputs"
    truth = OUT / "evaluator"
    inputs.mkdir(parents=True, exist_ok=True)
    truth.mkdir(exist_ok=True)
    t, beats, clean = signals()
    conditions = [
        ("rec01", "bipolar"),
        ("rec02", "clean"),
        ("rec03", "strong_monophasic"),
        ("rec04", "moderate_monophasic"),
        ("rec05", "pulse_channels"),
    ]
    np.savez(truth / "clean.npz", time=t, **clean)
    answer = {}
    for name, condition in conditions:
        values = {k: v.copy() for k, v in clean.items()}
        targets = []
        if condition in ("bipolar", "strong_monophasic", "moderate_monophasic"):
            for j, index in enumerate((13, 29, 51, 72)):
                start = int(round((beats[index] + 0.148) * 125))
                amp = 0.82 if condition == "moderate_monophasic" else 2.4
                sign = (-1) ** j
                values["ECG"][start : start + 2] += (
                    sign * amp * np.array([1.0, -1.0 if condition == "bipolar" else 1.0])
                )
                targets.append(dict(channel="ECG", a=start, b=start + 2))
        if condition == "pulse_channels":
            for ch, offset, amplitude, indices in [
                ("PPG", 0.61, 12.0, (11, 29, 53, 76)),
                ("ABP", 0.57, 85.0, (17, 38, 60, 83)),
            ]:
                for j, index in enumerate(indices):
                    start = int(round((beats[index] + offset) * 125))
                    values[ch][start : start + 2] += amplitude * ((-1) ** j) * np.array([1.0, -1.0])
                    targets.append(dict(channel=ch, a=start, b=start + 2))
        file = inputs / (name + ".npz")
        np.savez(file, time=t, **values)
        answer[name] = dict(condition=condition, targets=targets, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (truth / "design.json").write_text(json.dumps(answer, indent=2))
    cfg = {
        "dataset": {
            "format": "npz",
            "path": str(inputs.resolve()),
            "mapping": {
                "time_key": "time",
                "fs": 125,
                "channels": [dict(task=k, key=k, unit="mmHg" if k == "ABP" else "source_units") for k in clean],
            },
        },
        "config": {
            "purpose": "保留三路实测波形、原生心搏/RR、脉搏/PPI 与逐拍压力，按观测证据作清洗决策。",
            "method": "dataset_agent",
            "routing": "reason",
            "inspection_preset": "balanced",
            "secondary_probe": True,
            "tasks": list(clean),
            "model": "deepseek-flash",
            "api_cap": 5,
            "budget_root": str((OUT.parents[1] / ".pattern-review-budgets/vital-cleaning-v1-20261112-r4").resolve()),
            "knowledge_root": str((OUT / "fresh-knowledge").resolve()),
            "max_llm_calls_per_record": 96,
            "joint_views": 8,
            "write_views": 50,
            "atlas": False,
        },
    }
    (OUT / "spec.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
