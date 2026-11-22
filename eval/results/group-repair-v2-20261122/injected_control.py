"""Place four known two-sample impulses on one Vital block; keep original samples evaluator-only."""

import json
from pathlib import Path
import os

import numpy as np

from tsmcp.cleaning.common import read, write
from tsmcp.processing import ProcessingSession

here = Path(__file__).resolve().parent
runs = here.parents[1] / "runs"
experiment = runs / here.name
source = runs / "dataset-entry-20261122/accepted-agent"


def prepare(shape):
    target = experiment if shape == "bipolar" else experiment / shape
    target.mkdir(exist_ok=True)
    cases = read(source / "dataset_manifest.json")["cases"]
    # Longest simultaneous block in the same second patient, tie-broken by case id before looking at trial outcomes.
    ecg = min(
        (c for c in cases if c["record_id"] == os.environ["SIGNALMIND_CONTROL_RECORD"] and c["task"] == "ECG"),
        key=lambda c: (-c["n"], c["id"]),
    )
    peers = [c for c in cases if (c["record_id"], c["rows"]) == (ecg["record_id"], ecg["rows"])]
    assert {c["task"] for c in peers} == {"ECG", "ABP", "PPG"}
    arrays = {}
    for c in peers:
        with np.load(source / "public" / c["id"] / "input.npz") as z:
            if "time" in arrays:
                np.testing.assert_array_equal(z["time"], arrays["time"])
            arrays["time"] = z["time"]
            arrays[c["task"]] = z["values"]
    clean = arrays["ECG"].copy()
    fs = ecg["fs"]
    pr = read(source / read(source / "export/waveform_manifest.json")[ecg["id"]]["probe"])
    peaks = np.array([z["sample"] for z in pr["events"]], int)
    swing = float(np.percentile(clean, 99) - np.percentile(clean, 1))
    magnitude = 0.75 if shape == "moderate" else 2.5
    spans = []
    eligible = peaks[(peaks > 2 * fs) & (peaks < len(clean) - 2 * fs)]
    for j, frac in enumerate((0.15, 0.35, 0.60, 0.80)):
        peak = int(eligible[np.argmin(abs(eligible - frac * len(clean)))])
        a = peak + round(0.144 * fs)
        spans.append([a, a + 2])
        arrays["ECG"][a : a + 2] += (
            np.array([1.0, -1.0 if shape == "bipolar" else 1.0]) * swing * (magnitude if j % 2 == 0 else -magnitude)
        )
    assert len({a for a, _ in spans}) == 4
    inputs = target / "injection-input"
    inputs.mkdir(exist_ok=True)
    np.savez(inputs / "waveform.npz", **arrays)
    control = target / "injection-score"
    control.mkdir(exist_ok=True)
    np.save(control / "observed-before.npy", clean, allow_pickle=False)
    write(control / "key.json", dict(source=ecg["id"], fs=fs, n=len(clean), ranges=spans, scale=swing))
    cfg = read(experiment / "spec.json")["config"]
    for k in ["patient_manifest", "reuse_development", "receipt_seed"]:
        cfg.pop(k, None)
    cfg.update(atlas=False)
    write(
        target / "injected-spec.json",
        dict(
            config=cfg,
            dataset=dict(
                format="npz",
                path="injection-input",
                mapping=dict(
                    time_key="time",
                    fs=fs,
                    channels=[dict(task=c["task"], key=c["task"], unit=c["unit"]) for c in peers],
                ),
            ),
        ),
    )


def score(shape, zoom=False):
    target = experiment if shape == "bipolar" else experiment / shape
    src = target / "injection-score"
    truth = read(src / "key.json")
    clean = np.load(src / "observed-before.npy")
    run = (
        target / "injected-recovery"
        if (target / "injected-recovery/export/evaluation_report.json").exists()
        else target / "injected-agent"
    )
    if zoom:
        run = target / "zoom"
    report = read(run / "export/evaluation_report.json")
    wm = read(run / "export/waveform_manifest.json")
    cid = next(k for k, v in wm.items() if v["channel"] == "ECG")
    with np.load(run / "public" / cid / "input.npz") as z:
        noisy = z["values"]
    with np.load(run / wm[cid]["waveform"]) as z:
        out = z["values"]
        valid = z["valid"] if "valid" in z else np.isfinite(out)
        estimated = z["estimated"] if "estimated" in z else np.zeros(len(out), bool)
    hits = read(next(run.glob("records/*/hit-dispositions.json")))
    trials = read(next(run.glob("records/*/spike-trials.json")))
    mask = np.zeros(len(clean), bool)
    details = []
    for a, b in truth["ranges"]:
        mask[a:b] = True
        hit = [h for h in hits if h["case"] == cid and h["a"] < b and h["b"] > a]
        details.append(
            dict(
                a=a,
                b=b,
                overlapping_alert_reasons=sorted({h["reason"] for h in hit}),
                trial_status=[t["status"] for t in trials if t["case"] == cid and t["a"] < b and t["b"] > a],
                changed=int(np.count_nonzero(out[a:b] != noisy[a:b])),
                disabled=int((~valid[a:b]).sum()),
                estimated=int(estimated[a:b].sum()),
                selected_actions=sorted({h["wave_requested"] for h in hit}),
                before_mae=float(np.mean(abs(noisy[a:b] - clean[a:b]))),
                after_mae=float(np.mean(abs(out[a:b] - clean[a:b]))),
            )
        )
    session = ProcessingSession(noisy, fs=truth["fs"], channel="ECG", unit="source_units")
    oracle = session.process(session.raw.id, "interpolate", truth["ranges"]).values
    result = dict(
        exact_range_linear_mae=float(np.mean(abs(oracle[mask] - clean[mask]))),
        source=truth["source"],
        status=report["status"],
        size=len(clean),
        injected=int(mask.sum()),
        details=details,
        before_mae=float(np.mean(abs(noisy[mask] - clean[mask]))),
        after_mae=float(np.mean(abs(out[mask] - clean[mask]))),
        outside_ecg_changed=int(np.count_nonzero(out[~mask] != clean[~mask])),
        new_calls=report["model_calls"],
        cost_usd=report["estimated_ds_usd"],
    )
    name = ("zoom-" if zoom else "") + shape
    write(here / f"{name}-result.json", result)
    draw(name, truth["fs"], details, clean, noisy, out)
    print(json.dumps(result, ensure_ascii=False))


def draw(shape, fs, details, clean, noisy, out):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 1.6 * len(details) + 0.5), layout="constrained")
    for i, d in enumerate(details, 1):
        a, b = d["a"], d["b"]
        left, right = a - round(0.25 * fs), b + round(0.25 * fs)
        t = (np.arange(left, right) - a) / fs
        ax = fig.add_subplot(len(details), 1, i)
        ax.plot(t, noisy[left:right], color="#757887", label="Injected", lw=1)
        ax.plot(t, clean[left:right], color="#255ca1", label="Withheld original", lw=1.3)
        ax.plot(t, out[left:right], color="#c65422", ls="--", label="Released", lw=1)
        ax.axvspan(0, (b - a) / fs, color="#e3ae53", alpha=0.15)
        ax.set_ylabel(f"Core {i}")
        ax.spines[["top", "right"]].set_visible(False)
        if i == 1:
            ax.legend(fontsize=8, ncol=3)
    ax.set_xlabel("Seconds from first injected sample")
    fig.savefig(here / f"{shape}-comparison.png", dpi=145)


if __name__ == "__main__":
    import sys

    shape = sys.argv[2] if len(sys.argv) > 2 else "bipolar"
    assert shape in ("bipolar", "same-sign", "moderate")
    if sys.argv[1] == "prepare":
        prepare(shape)
    else:
        score(shape, zoom=sys.argv[1] == "score-zoom")
