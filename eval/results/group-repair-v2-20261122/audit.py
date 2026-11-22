"""Verify submitted group evidence and paired real-data releases, and render every short-spike candidate."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.cleaning.common import changed, digest, read, sha, write
from tsmcp.cleaning.dataset_review import FIELDS
from tsmcp.cleaning.engine import load_arrays
from tsmcp.processing import ProcessingSession

dest = Path(__file__).resolve().parent
runs = dest.parents[1] / "runs"
old = runs / "dataset-entry-20261122/accepted-agent"


def measure(root):
    with (root / "export/measurements.csv").open(encoding="utf-8-sig") as f:
        return {(r["case"], int(r["a"]), int(r["b"])): r for r in csv.DictReader(f)}


def wave(root, c, wm):
    artifact = read(root / "artifacts" / wm[c["id"]]["artifact_id"] / "artifact.json")
    return load_arrays(root, c, artifact)


def main(zoom=False):
    root = runs / dest.name / ("zoom" if zoom else "agent")
    prefix = "zoom-" if zoom else ""
    summary = read(root / "export/evaluation_report.json")
    manifest = read(root / "dataset_manifest.json")
    assert manifest["scope_hash"] == read(old / "dataset_manifest.json")["scope_hash"]
    cases = {c["id"]: c for c in manifest["cases"]}
    wm = read(root / "export/waveform_manifest.json")
    old_wave = read(old / "export/waveform_manifest.json")
    before, after = measure(old), measure(root)
    same = before.keys() & after.keys()

    def counts(data):
        return dict(Counter(r["quality"] for r in data.values()))

    receipts = {p.stem: read(p) for p in (root / "dataset-receipts").glob("*.json")}
    result = dict(
        status=summary["status"],
        groups=0,
        rows=0,
        images=0,
        patients={},
        old=counts(before),
        new=counts(after),
        exact_pairs=len(same),
        old_only=len(before) - len(same),
        new_only=len(after) - len(same),
        unchecked_kept=sum(r["joint_view_unchecked"] == "True" and r["quality"] == "keep" for r in after.values()),
        calls=summary["model_calls"],
        ds_usd=summary["estimated_ds_usd"],
        accuracy=summary["reference_accuracy"],
    )
    spikes = []
    for patient in sorted(root.glob("records/*")):
        hits = {h["id"]: h for h in read(patient / "hit-dispositions.json")}
        groups = read(patient / "groups.json")
        plans = read(patient / "repair-plans.json")
        sent = set()
        for g in groups:
            receipt = receipts[plans[g["id"]]["receipt"]]
            (ev,) = receipt["evidence"]
            name, evidence, images = ev
            assert name == g["id"] and len(images) == len(set(images)) == 2
            assert images == [sha(patient / "views" / f"{name}-{i}.png") for i in range(2)]
            table = evidence["member_rows"]
            assert len(table) == g["count"] and digest(table) == g["member_rows_sha256"]
            for i, (hid, row) in enumerate(zip(g["member_ids"], table, strict=True)):
                h = hits[hid]
                assert hid not in sent
                sent.add(hid)
                assert evidence["cases"][row[0]]["id"] == h["case"]
                assert row[1:4] == [h["a"], h["b"], h["value"]]
                assert row[4:12] == [h["features"][field] for field in FIELDS]
                assert row[12:] == [
                    *[h["peer_bpm"].get(k) for k in ("ECG", "PPG", "ABP")],
                    h["jev_score"],
                    *h.get("anchor_values", []),
                ]
                if zoom and h["reason"] == "isolated_spike_candidate":
                    with np.load(root / "public" / h["case"] / "input.npz") as obs:
                        core = obs["values"][h["a"] : h["b"]].tolist()
                    assert core == evidence["core_samples_by_member"][i]
            result["images"] += len(images)
        assert sent == hits.keys()
        trials = {r["hit"]: r for r in read(patient / "spike-trials.json")}
        for hid, h in hits.items():
            if h["reason"] == "isolated_spike_candidate":
                plan = plans[h["group"]]
                spikes.append(
                    dict(
                        case=h["case"],
                        a=h["a"],
                        b=h["b"],
                        hit=hid,
                        group=h["group"],
                        requested=h["wave_requested"],
                        group_plan=plan,
                        trial=trials.get(hid),
                    )
                )
        result["rows"] += len(hits)
        result["groups"] += len(groups)
        result["patients"][patient.name] = dict(
            groups=len(groups), hits=len(hits), states=read(patient / "report.json")["measurement_states"]
        )
    samples = Counter()
    for cid, c in cases.items():
        raw, _, new, v, e = wave(root, c, wm)
        _, _, prev, pv, pe = wave(old, c, old_wave)
        samples.update(
            total=len(raw),
            numeric=int(changed(raw, new).sum()),
            estimated=int(e.sum()),
            old_numeric=int(changed(raw, prev).sum()),
            disabled=int((~v).sum()),
            old_disabled=int((~pv).sum()),
        )
        assert wm[cid]["source_sha256"] == old_wave[cid]["source_sha256"]
    result["samples"] = samples
    result["candidates"] = len(spikes)
    result["spike_outcomes"] = dict(Counter(s["trial"]["status"] if s["trial"] else "no_trial" for s in spikes))
    write(dest / (prefix + "verified.json"), result)
    write(dest / (prefix + "spike-decisions.json"), spikes)
    if spikes:
        plot(root, cases, wm, spikes, prefix)
    print(result)


def plot(root, cases, wm, spikes, prefix):
    fig = Figure(figsize=(10, 1.52 * len(spikes) + 0.75), layout="constrained")
    FigureCanvasAgg(fig)
    axs = fig.subplots(len(spikes), 2, squeeze=False)
    for i, s in enumerate(spikes):
        c = cases[s["case"]]
        a, b = s["a"], s["b"]
        fs = c["fs"]
        raw, t, released, _, _ = wave(root, c, wm)
        session = ProcessingSession(raw, fs=fs, channel=c["task"], unit=c["unit"])
        alternate = session.process(session.raw.id, "interpolate", [[a, b]]).values
        center = (t[a] + t[b - 1]) / 2
        for ax, width in zip(axs[i], (1.2, 0.12), strict=True):
            lo, hi = np.searchsorted(t, [center - width, center + width])
            rel = t[lo:hi] - center
            ax.plot(rel, raw[lo:hi], color="#424d60", lw=1, label="Observed")
            ax.plot(rel, released[lo:hi], color="#237a71", lw=0.9, ls="--", label="Agent release")
            trial = s["trial"]
            if trial and "candidate" in trial:
                art = read(root / "artifacts" / trial["candidate"] / "artifact.json")
                y = load_arrays(root, c, art)[2]
                ax.plot(t[a - 1 : b + 1] - center, y[a - 1 : b + 1], color="#c65223", lw=1.5, label="Trial (core only)")
            else:
                ax.plot(
                    t[a - 1 : b + 1] - center,
                    alternate[a - 1 : b + 1],
                    color="#c65223",
                    lw=1.5,
                    label="Offline interpolation only",
                )
            ax.axvspan(t[a] - center, t[b - 1] + 1 / fs - center, color="#d6b04c", alpha=0.18, lw=0)
            ax.spines[["right", "top"]].set_visible(False)
        state = s["trial"]["status"] if s["trial"] else "no trial"
        axs[i, 0].set_title(f"{s['case'].split('_')[-1]}  |  source {t[a]:.3f} s  |  {state}", loc="left", fontsize=9)
        axs[i, 0].set_ylabel("ECG")
    axs[0, 1].legend(loc="upper right", fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("Seconds from candidate center")
    fig.suptitle("Vital ECG   |   All eligible cores   |   Context (left) and detail (right)")
    fig.savefig(dest / (prefix + "ecg-candidates.png"), dpi=135)


if __name__ == "__main__":
    import sys

    main("--zoom" in sys.argv)
