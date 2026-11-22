"""Search methods callable by both the independent reviewer and SignalMind recipes."""

from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import digest, sha
from tsmcp.followup.pattern import detect
from tsmcp.inspection.scan import HIGH_BAND_HZ, ecg_baseline_shifts, windows
from tsmcp.processing import intervals


def rule_id(task, rule):
    root = Path(__file__).resolve().parents[1]
    return digest(
        [task, rule, sha(Path(__file__)), sha(root / "inspection/scan.py"), sha(root / "followup/pattern.py")]
    )


def rescan_rule(x, usable, fs, task, rule, *, evaluated=None):
    method, threshold = rule["method"], rule["threshold"]
    limits = {"baseline_shift": (0.975, 2.25), "step_outlier": (0.3, 12), "high_frequency": (0.005, 0.5)}
    lo, hi = limits[method]
    if not lo <= threshold <= hi or (method == "baseline_shift" and task != "ECG"):
        raise ValueError("Threshold or modality outside the stated method contract")
    if method == "step_outlier":
        return detect(
            x,
            usable,
            fs,
            dict(
                metric="step",
                scale_window_s=10,
                baseline_window_s=0.2,
                seed_ratio=threshold,
                tail_ratio=threshold,
                quiet_s=0.024,
                max_seed_s=0.25,
            ),
            evaluated=evaluated,
        )
    hits = []
    for a, b in intervals(usable & np.isfinite(x)):
        if method == "baseline_shift":
            scored = None if evaluated is None else evaluated[a:b]
            results = ecg_baseline_shifts(x[a:b], fs, threshold / 1.5, evaluated=scored)
            hits.extend([a + left, a + right] for left, right, *_ in results)
        else:
            for left, right in windows(b - a, max(2, round(2 * fs)), max(1, round(fs))):
                y = x[a + left : a + right]
                if len(y) < max(16, round(fs)) or fs / 2 <= HIGH_BAND_HZ[task]:
                    continue
                power = abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y)))) ** 2
                hz = np.fft.rfftfreq(len(y), 1 / fs)
                total = power[hz >= 0.5].sum()
                if total > 0 and evaluated is not None:
                    evaluated[a + left : a + right] = True
                if total > 0 and power[hz >= HIGH_BAND_HZ[task]].sum() >= threshold * total:
                    if hits and hits[-1][1] > a + left:
                        hits[-1][1] = a + right
                    else:
                        hits.append([a + left, a + right])
    return hits
