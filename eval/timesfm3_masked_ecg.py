"""Masked ECG reconstruction with the local Vital-tuned TimesFM. Prepare, infer, and score in separate processes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.timesfm_gap_forecast import bridge, file_hash

TRAIN = Path("eval/runs/timesfm3-vital-full-20261122")
SYNTH = Path("eval/runs/v11-ds-synthetic-20261125")
FS, CONTEXT = 125, 512
LENGTHS = (2, 25, 125, 625)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def flanks(x, peers, a, n):
    b = a + n
    if a < CONTEXT or b + CONTEXT > len(x) or not 0 < n <= 5 * FS:
        raise ValueError("Gap length or continuous context outside this protocol")
    return dict(
        left=x[a - CONTEXT : a].copy(),
        right=x[b : b + CONTEXT][::-1].copy(),
        left_peers=peers[:, a - CONTEXT : b].copy(),
        right_peers=peers[:, a : b + CONTEXT][:, ::-1].copy(),
    )


def prepare(out):
    from tsmcp.cleaning.common import array_hash, digest

    public, private = out / "inputs", out / "scoring"
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir()
    joblist, bank, metadata, gold, cases = [], {}, {}, {}, {}

    def add(x, peers, a, n, label, **info):
        key = f"G{len(joblist):03d}"
        inputs = flanks(x, peers, a, n)
        if not all(np.isfinite(v).all() for v in inputs.values()):
            raise ValueError("Nonfinite model input")
        # Verify intervention invariance before handing the arrays to the inference process.
        hidden = x.copy()
        hidden[a : a + n] = np.nan
        for name, value in flanks(hidden, peers, a, n).items():
            np.testing.assert_array_equal(value, inputs[name])
            bank[f"{key}_{name}"] = value
        joblist.append(dict(key=key, n=n, peers=len(peers)))
        metadata[key] = dict(cohort=label, a=a, n=n, **info)
        return key

    # Reuse the public detector output exactly as it was frozen before the DS experiment.
    manifest = read(SYNTH / "organizer/manifest.json")
    for item in manifest["cases"]:
        cid = item["case_id"]
        directory = SYNTH / cid / "public"
        c = read(directory / "case.json")
        assert c["input_hash"] == item["input_hash"] == digest({k: v for k, v in c.items() if k != "input_hash"})
        with np.load(directory / "observed.npz", allow_pickle=False) as z:
            t, x, valid, est = (z[k] for k in ("t", "x", "valid", "estimated"))
        assert array_hash(t, x, valid, est) == c["observed_hash"] and (valid & ~est).all()
        peers = np.empty((0, len(t)))
        detected = sorted({(d["start_sample"] - c["offset"], d["end_sample"] - c["offset"]) for d in c["candidates"]})
        keys = [add(x, peers, a, b - a, "detector", case=cid) for a, b in detected]
        truth = SYNTH / "organizer" / item["private_npz"]
        assert file_hash(truth) == item["private_sha256"]
        with np.load(truth, allow_pickle=False) as z:
            mask = z["damaged"]
            # A separately labelled known-location probe for a missed injection. Never added to detector scores.
            from tsmcp.processing import intervals

            extra = [
                add(x, peers, a, b - a, "given_location", case=cid)
                for a, b in intervals(mask)
                if (a, b) not in detected
            ]
            gold[cid + "_ref"] = z["reference"]
            gold[cid + "_mask"] = mask
        gold[cid + "_observed"] = x
        cases[cid] = dict(detected=keys, extra=extra)

    # Follow the existing held-out 5 s schedule, selecting solely on split and support for both flanks.
    test = read(TRAIN / "test-data/manifest.json")
    assert list(test["fields"]) == ["ECG", "PPG", "ABP"]
    train_ids = {z["patient"] for z in read(TRAIN / "train-manifest.json")["patients"]}
    test_files = {z["patient"]: z for z in test["patients"] if z["split"] == "test"}
    assert train_ids.isdisjoint(test_files)
    used = set()
    for batch in read(TRAIN / "test-gaps.json"):
        if batch["h"] != max(LENGTHS):
            continue
        for patient, slot, start, reverse in batch["jobs"]:
            if patient in used or reverse:
                continue
            with np.load(TRAIN / "test-data" / test_files[patient]["file"], allow_pickle=False) as z:
                wave = z[f"wave{slot}"]
                peaks = z[f"rpeaks{slot}"]
            a = start + CONTEXT
            if a + max(LENGTHS) + CONTEXT > wave.shape[-1]:
                continue
            for n in LENGTHS:
                key = add(wave[0], wave[1:], a, n, "vital", patient=patient, slot=slot)
                gold[key + "_ref"] = wave[0, a : a + n].copy()
                gold[key + "_peaks"] = peaks[(peaks >= a) & (peaks < a + n)] - a
            used.add(patient)
            if len(used) == 24:
                break
        if len(used) == 24:
            break
    assert len(used) == 24
    np.savez_compressed(public / "flanks.npz", **bank)
    write(public / "jobs.json", joblist)
    np.savez_compressed(private / "targets.npz", **gold)
    write(
        private / "design.json",
        dict(
            cases=cases,
            jobs=metadata,
            synthetic_manifest=file_hash(SYNTH / "organizer/manifest.json"),
            vital_gap_schedule=file_hash(TRAIN / "test-gaps.json"),
            flanks_sha256=file_hash(public / "flanks.npz"),
            targets_sha256=file_hash(private / "targets.npz"),
            context_samples=CONTEXT,
            vital_patients=len(used),
        ),
    )


def predict(inputs, out, checkpoint, device):
    import torch
    from timesfm3 import TimesFM3Forecaster

    expected = read(checkpoint.parent / "export.json")["sha256"]
    actual = file_hash(checkpoint / "model.safetensors")
    if actual != expected:
        raise ValueError("Unexpected fine-tuned checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = TimesFM3Forecaster.from_pretrained(
        str(checkpoint.resolve()), device=device, local_files_only=True, per_core_batch_size=8
    )
    with np.load(inputs / "flanks.npz", allow_pickle=False) as f:
        bank = dict(f)
    batches = defaultdict(list)
    for j in read(inputs / "jobs.json"):
        for count in [0, 2] if j["peers"] == 2 else [0]:
            batches[j["n"], count].extend((j["key"], side) for side in ("left", "right"))
    preds = {}
    with torch.inference_mode():
        for (n, count), rows in batches.items():
            both = {}
            for i in range(0, len(rows), 8):
                part = rows[i : i + 8]
                ans = model.predict_batch(
                    [bank[f"{k}_{s}"] for k, s in part],
                    horizon=n,
                    past_future_covariates=[bank[f"{k}_{s}_peers"] for k, s in part] if count else None,
                    use_znorm=True,
                    return_quantiles=False,
                    use_symmetric_averaging=False,
                    make_positive=False,
                )
                for (k, s), z in zip(part, ans, strict=True):
                    y = np.asarray(z.forecast, dtype=float).reshape(-1)
                    assert len(y) == n and np.isfinite(y).all()
                    both[k, s] = y
            for k, s in rows:
                if s == "left":
                    preds[f"{k}_tfm{count}"] = bridge(both[k, "left"], both[k, "right"][::-1])
            print("decoded", n, "samples;", count, "peers;", len(rows) // 2, "gaps", flush=True)
    np.savez_compressed(out / "predictions.npz", **preds)
    write(
        out / "receipt.json",
        dict(
            checkpoint=str(checkpoint),
            sha256=actual,
            device=device,
            torch=torch.__version__,
            flanks_sha256=file_hash(inputs / "flanks.npz"),
            jobs_sha256=file_hash(inputs / "jobs.json"),
            predictions_sha256=file_hash(out / "predictions.npz"),
            api_calls=0,
        ),
    )


def score(out, results):
    from scipy.interpolate import PchipInterpolator

    from eval.timesfm_gap_trial import basic_bridge, save_csv
    from tsmcp.cleaning.tasks import match_events
    from tsmcp.plugins.ecg.rpeaks import detect_r_peaks

    design = read(out / "scoring/design.json")
    receipt = read(out / "model/receipt.json")
    assert receipt["flanks_sha256"] == design["flanks_sha256"] == file_hash(out / "inputs/flanks.npz")
    assert receipt["jobs_sha256"] == file_hash(out / "inputs/jobs.json")
    assert receipt["predictions_sha256"] == file_hash(out / "model/predictions.npz")
    assert design["targets_sha256"] == file_hash(out / "scoring/targets.npz")
    with (
        np.load(out / "inputs/flanks.npz") as f,
        np.load(out / "scoring/targets.npz") as g,
        np.load(out / "model/predictions.npz") as m,
    ):
        bank, gold, preds = dict(f), dict(g), dict(m)
    rows, curves = [], {}
    for key, job in design["jobs"].items():
        a, n = job["a"], job["n"]
        left, right = bank[key + "_left"], bank[key + "_right"]
        vital = job["cohort"] == "vital"
        ref = gold[key + "_ref"] if vital else gold[job["case"] + "_ref"][a : a + n]
        anchor_x = np.r_[-3:0, n : n + 3]
        methods = dict(
            linear=basic_bridge(left, right, n),
            pchip=PchipInterpolator(anchor_x, np.r_[left[-3:], right[-3:][::-1]], extrapolate=False)(np.arange(n)),
            tfm0=preds[key + "_tfm0"],
        )
        if vital:
            methods["tfm2"] = preds[key + "_tfm2"]
            methods["original_probe"] = ref
        curves[key] = methods
        for method, y in methods.items():
            err = float(np.mean(np.abs(y - ref)))
            row = dict(
                key=key,
                cohort=job["cohort"],
                subject=job.get("patient", job.get("case")),
                samples=n,
                method=method,
                mae=err,
                tp=None,
                fp=None,
                fn=None,
                amplitude_ratio=None,
            )
            if vital:
                peaks = detect_r_peaks(np.r_[left, y, right[::-1]], FS) - len(left)
                found = peaks[(peaks >= 0) & (peaks < n)]
                target = gold[key + "_peaks"]
                tp = len(match_events(found, target, 5))
                row.update(tp=tp, fp=len(found) - tp, fn=len(target) - tp)
                if n >= FS:
                    swing = np.ptp(np.quantile(ref, [0.01, 0.99]))
                    if swing > 0:
                        row["amplitude_ratio"] = float(np.ptp(np.quantile(y, [0.01, 0.99])) / swing)
            rows.append(row)
    results.mkdir(parents=True, exist_ok=True)
    save_csv(results / "gap-scores.csv", rows)
    vit = []
    for n in LENGTHS:
        at = [r for r in rows if r["cohort"] == "vital" and r["samples"] == n]
        methods = {}
        for method in ("linear", "pchip", "tfm0", "tfm2", "original_probe"):
            rr = [r for r in at if r["method"] == method]
            tp, fp, fn = (sum(r[k] for r in rr) for k in ("tp", "fp", "fn"))
            ratios = [r["amplitude_ratio"] for r in rr if r["amplitude_ratio"] is not None]
            methods[method] = dict(
                median_mae=float(np.median([r["mae"] for r in rr])),
                median_amplitude_ratio=float(np.median(ratios)) if ratios else None,
                peak_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        errors = defaultdict(dict)
        for r in at:
            errors[r["key"]][r["method"]] = r["mae"]
        vit.append(
            dict(
                samples=n,
                patients=len(errors),
                methods=methods,
                peer_beats_target_only=sum(d["tfm2"] < d["tfm0"] for d in errors.values()),
                peer_beats_linear=sum(d["tfm2"] < d["linear"] for d in errors.values()),
            )
        )
    trials = {}
    for scope in ("detected", "known_location"):
        trials[scope] = {}
        for method in ("linear", "pchip", "tfm0"):
            error = []
            restored = 0
            outside = 0
            sites = 0
            for cid, c in design["cases"].items():
                original = gold[cid + "_observed"]
                y = original.copy()
                est = np.zeros(len(y), bool)
                mask = gold[cid + "_mask"]
                ref = gold[cid + "_ref"]
                for key in c["detected"] if scope == "detected" else c["detected"] + c["extra"]:
                    j = design["jobs"][key]
                    core = slice(j["a"], j["a"] + j["n"])
                    y[core] = curves[key][method]
                    est[core] = True
                if mask.any():
                    sites += 1
                    error.extend(abs(y[mask] - ref[mask]))
                    restored += int(
                        est[mask].all() and np.mean(abs(y[mask] - ref[mask])) < np.mean(abs(original[mask] - ref[mask]))
                    )
                outside += int(np.sum(y[~mask] != original[~mask]))
                if scope == "detected" and method == "tfm0":
                    np.savez_compressed(out / "model" / (cid + "-candidate.npz"), values=y, estimated=est)
            trials[scope][method] = dict(
                recovered=restored, sites=sites, fixed_core_mae=float(np.mean(error)), outside_changed=outside
            )
    write(
        results / "report.json",
        dict(
            receipt=receipt,
            synthetic=trials,
            vital=vit,
            raw_gap_scores="gap-scores.csv",
            status="offline_candidates_only",
            context_samples_per_side=CONTEXT,
        ),
    )
    plot(results, design, bank, gold, curves)


def plot(folder, design, bank, gold, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = design["jobs"]
    real = next(j["patient"] for j in jobs.values() if j["cohort"] == "vital")
    examples = [next(k for k, j in jobs.items() if j.get("case") == cid) for cid in ("C031", "C052")]
    examples += [next(k for k, j in jobs.items() if j.get("patient") == real and j["n"] == n) for n in (125, 625)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), layout="constrained")
    for i, (ax, key) in enumerate(zip(axes.flat, examples, strict=True)):
        j = jobs[key]
        n = j["n"]
        h = 12 if i < 2 else 35
        ref = gold[j["case"] + "_ref"][j["a"] : j["a"] + n] if i < 2 else gold[key + "_ref"]
        x = np.arange(-h, n + h) / FS
        left = bank[key + "_left"][-h:]
        right = bank[key + "_right"][::-1][:h]
        ax.plot(x, np.r_[left, ref, right], color="#303641", lw=1.6, label="Withheld ECG + observed flanks")
        if i < 2:
            removed = gold[j["case"] + "_observed"][j["a"] : j["a"] + n]
            ax.text(
                0.02,
                0.03,
                "Masked values: " + ", ".join(f"{z:.2f}" for z in removed),
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )
        for method, color, label in [
            ("pchip", "#c17c22", "PCHIP"),
            ("tfm0", "#89669a", "TimesFM: ECG"),
            ("tfm2", "#007f76", "TimesFM: ECG + PPG + ABP"),
        ]:
            if method in curves[key]:
                ax.plot(
                    np.arange(-1, n + 1) / FS,
                    np.r_[left[-1], curves[key][method], right[0]],
                    color=color,
                    lw=1.3,
                    ls="--" if method != "tfm2" else "-",
                    marker="." if i < 2 else None,
                    label=label,
                )
        ax.axvspan(-0.5 / FS, (n - 0.5) / FS, color="#f1bc5c", alpha=0.2)
        ax.set(
            title=[
                "Synthetic: detected interbeat spike",
                "Synthetic: supplied near-QRS location",
                "Vital: 1 s gap",
                "Vital: 5 s gap",
            ][i],
            xlabel="Seconds from gap start",
            ylabel="ECG amplitude",
        )
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(folder / "mask-and-fill.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("run", type=Path)
    pred = sub.add_parser("predict")
    pred.add_argument("inputs", type=Path)
    pred.add_argument("out", type=Path)
    pred.add_argument("--checkpoint", type=Path, default=TRAIN / "checkpoint")
    pred.add_argument("--device", default="cpu")
    scores = sub.add_parser("score")
    scores.add_argument("run", type=Path)
    scores.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.run)
    elif args.action == "predict":
        predict(args.inputs, args.out, args.checkpoint, args.device)
    else:
        score(args.run, args.results)
