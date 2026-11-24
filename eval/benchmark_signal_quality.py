"""Known-event development benchmark; no patient inputs, credentials or model calls."""

import argparse
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import write
from tsmcp.cleaning.tasks import probe
from tsmcp.inspection.scan import scan
from tsmcp.processing import intervals

KINDS = (
    "clean",
    "impulse",
    "biphasic",
    "baseline_wander",
    "line",
    "clip",
    "dropout",
    "gain",
    "motion",
    "time_break",
    "missing",
    "distorted_shape",
    "mixed",
)


def example(seed, task, kind, fs, seconds=28):
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * fs)) / fs
    period = rng.uniform(0.72, 0.92)
    phase = (t + 0.3) % period
    pulse = np.exp(-0.5 * ((phase - 0.15) / 0.014) ** 2) + 0.12 * np.exp(-0.5 * ((phase - 0.39) / 0.05) ** 2)
    if task != "ECG":
        pulse = (phase / 0.16) ** 2 * np.exp(-phase / 0.08)
    scale = {"ECG": 1.0, "PPG": 8.0, "ABP": 75.0}[task]
    clean = (75 if task == "ABP" else 30 if task == "PPG" else 0) + scale * pulse
    observed = clean.copy()
    start = int(fs * (12 + rng.uniform(-0.35, 0.35)))
    stop = start + round(2 * fs)
    truth = np.zeros(len(t), bool)
    physiology = np.zeros(len(t), bool)
    if kind in ("impulse", "biphasic"):
        stop = start + 2
        observed[start:stop] += scale * np.array([1, -1 if kind == "biphasic" else 1])
    elif kind == "baseline_wander":
        observed[start:stop] += scale * np.sin(np.linspace(0, np.pi, stop - start)) * 0.6
    elif kind == "line":
        observed[start:stop] += scale * 0.18 * np.sin(2 * np.pi * 50 * t[start:stop])
    elif kind == "clip":
        observed[start:stop] = np.minimum(observed[start:stop], np.percentile(clean, 80))
    elif kind == "dropout":
        observed[start:stop] = observed[start - 1]
    elif kind == "gain":
        observed[start:stop] = (observed[start:stop] - np.median(clean)) * 3 + np.median(clean)
    elif kind == "motion":
        observed[start:stop] += np.interp(
            t[start:stop], t[start:stop:5], rng.normal(0, 0.5 * scale, len(t[start:stop:5]))
        )
    elif kind == "time_break":
        stop = start + 1
        t[start:] += 1
    elif kind == "missing":
        observed[start:stop] = np.nan
    elif kind in ("distorted_shape", "mixed"):
        # Rare broad beat is physiology in the reference; the mixed case additionally contains a technical impulse.
        start = np.flatnonzero((np.arange(len(t)) > start) & (phase < 0.01))[0]
        stop = start + round(0.45 * fs)
        clean[start:stop] += scale * 0.35 * np.exp(-0.5 * ((t[start:stop] - t[start] - 0.14) / 0.06) ** 2)
        observed[:] = clean
        physiology[start:stop] = True
        if kind == "mixed":
            start += round(0.15 * fs)
            stop = start + 2
            observed[start:stop] += scale
    if kind not in ("clean", "distorted_shape"):
        truth[start:stop] = True
    return t, observed, truth, physiology


def scores(truth, alarm, fs):
    refs, proposals = intervals(truth), intervals(alarm)
    used = set()
    errors = []
    for left, right in refs:
        matches = [
            (min(right, b) - max(left, a), i, a, b)
            for i, (a, b) in enumerate(proposals)
            if i not in used and a < right and left < b
        ]
        if matches:
            _, idx, a, b = max(matches)
            used.add(idx)
            errors.append((abs(a - left) + abs(b - right)) / (2 * fs))
    tp = len(used)
    fp = len(proposals) - tp
    fn = len(refs) - tp
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        truth_duration=float(truth.sum() / fs),
        hit_duration=float((truth & alarm).sum() / fs),
        false_positive_seconds=float((~truth & alarm).sum() / fs),
        boundary_errors_s=errors,
    )


def run(out):
    start = time.monotonic()
    cpu = time.process_time()
    rows = []
    for split, seed, fs in (("development", 111101, 125), ("holdout", 514227, 250)):
        for task in ("ECG", "PPG", "ABP"):
            for kind in KINDS:
                t, x, target, phys = example(seed, task, kind, fs)
                v = np.isfinite(x)
                alarm = np.zeros(len(x), bool)
                unknown = ~v.copy()
                cuts = np.flatnonzero(abs(np.diff(t) - 1 / fs) > 1e-6) + 1
                boundaries = [0, *cuts, len(x)]
                alarm[cuts] = True
                for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
                    m = v[left:right]
                    y = x[left:right]
                    ev, _, _ = scan(y, m, {"missing": ~m}, fs, task, "balanced")
                    pr = probe(y, t[left:right], fs, m, np.zeros(len(y), bool), task)["detector_support"]
                    for key in ("unsupported_ranges", "boundary_uncertain_ranges", "empty_event_ranges"):
                        for a, b in pr[key]:
                            unknown[left + a : left + b] = True
                    for z in ev:
                        alarm[left + z["start"] : left + z["end"]] = True
                rows.append(
                    dict(
                        subject=f"synthetic-{seed}",
                        split=split,
                        modality=task,
                        kind=kind,
                        fs=fs,
                        physiological_alarm_s=float((phys & alarm).sum() / fs),
                        scanned_s=len(x) / fs,
                        unknown_s=float(unknown.sum() / fs),
                        **scores(target, alarm, fs),
                    )
                )

    def summarize(sub):
        c = Counter()
        for row in sub:
            c.update(
                {
                    k: row[k]
                    for k in (
                        "tp",
                        "fp",
                        "fn",
                        "truth_duration",
                        "hit_duration",
                        "scanned_s",
                        "false_positive_seconds",
                        "unknown_s",
                    )
                }
            )
        d = 2 * c["tp"] + c["fp"] + c["fn"]
        err = [v for row in sub for v in row["boundary_errors_s"]]
        return dict(
            records=len(sub),
            event_precision=c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None,
            event_recall=c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None,
            event_f1=2 * c["tp"] / d if d else None,
            duration_recall=c["hit_duration"] / c["truth_duration"] if c["truth_duration"] else None,
            false_positive_clean_s=c["false_positive_seconds"],
            boundary_mae_s=float(np.mean(err)) if err else None,
            unknown_fraction=c["unknown_s"] / c["scanned_s"],
            coverage_completeness=1.0,
        )

    duration = sum(r["scanned_s"] for r in rows)
    doc = dict(
        mode="synthetic_discovery_only",
        clinical_accuracy=None,
        rows=rows,
        per_kind={k: summarize([r for r in rows if r["kind"] == k]) for k in KINDS},
        per_split={k: summarize([r for r in rows if r["split"] == k]) for k in ("development", "holdout")},
        signal_hours=duration / 3600,
        wall_s_per_signal_hour=(time.monotonic() - start) / (duration / 3600),
        cpu_s=time.process_time() - cpu,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_calls=0,
        api_usd=0,
    )
    write(out, doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    print(run(p.parse_args().out)["per_split"])
