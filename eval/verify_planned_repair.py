"""Counterfactual hard-gate evaluation on the archived Vital cases. The fixture reviewer approves every eligible trial.

Only score() reads the pre-injection values and injected coordinates. No model calls or changes to the source workspaces.
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import changed, digest, implementation_hash, read, sha, write
from tsmcp.cleaning.dataset_evidence import acquisition
from tsmcp.cleaning.dataset_pipeline import Record
from tsmcp.cleaning.engine import evaluate_output, load_arrays
from tsmcp.cleaning.event_hypotheses import conflict_evidence
from tsmcp.cleaning.quality_gate import SPEC_ID
from tsmcp.cleaning.tasks import probe
from tsmcp.inspection.scan import scan
from tsmcp.inspection.secondary import match
from tsmcp.plugins.ecg.rpeaks import detect_r_peaks
from tsmcp.processing import intervals


class ApproveFixture:
    enabled = True

    def decide(self, stage, rows):
        if stage != "gate":
            raise ValueError("Fixture supports candidate release decisions only")
        return {
            r["id"]: dict(decision="accept", reason="offline all-approve fixture", receipt=digest(r["data"]))
            for r in rows
        }


def execute(out, case, x, t, valid, cores, config=None):
    cid = case["id"]
    folder = out / "public" / cid
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "input.npz", values=x, time=t, valid=valid)
    case = dict(case, input_sha256=sha(folder / "input.npz"))
    estimated = np.zeros(len(x), bool)
    raw = evaluate_output(
        out,
        case,
        {"hash": "identity"},
        implementation_hash(),
        None,
        x,
        t,
        x,
        x.copy(),
        valid,
        estimated,
        [],
        time.monotonic(),
    )
    record = Record(
        out,
        [case],
        {"identity": {cid: raw}},
        {"write_views": 1000, "purpose": "hard gate benchmark", **(config or {})},
        ApproveFixture(),
        implementation_hash(),
    )
    requests = [
        dict(
            id=digest([cid, a, b])[:18],
            case=cid,
            task="ECG",
            group="discovered_short_excursion",
            a=a,
            b=b,
            reason="isolated_spike_candidate",
            value=float(np.ptp(x[a:b])),
            measurement="keep",
            wave_requested="interpolate_spike",
            decision_source="fixture_all_propose",
            diagnosis=dict(label="artifact", evidence_refs=["observed_candidate"]),
        )
        for a, b in sorted(set(cores))
    ]
    final = record.spike_trials({cid: raw}, requests)
    if config:
        from tsmcp.cleaning.derived_measurements import export_derived

        (out / "export").mkdir(exist_ok=True)
        export_derived(out, [case], final, {cid: raw}, record.repair_domain, final[cid]["artifact_id"], out / "export")
    released = load_arrays(out, case, final[cid])
    return released[2], released[4], read(record.path / "spike-trials.json")


def landmark_scores(reference, output, fs):
    """Use the frozen detector on both finite diagnostic copies, separate from trusted release support."""
    clean_peaks = detect_r_peaks(reference, fs)
    out_peaks = detect_r_peaks(output, fs)
    pairs = match(clean_peaks, out_peaks, round(0.04 * fs))
    amp_error = []
    width_error = []
    timing = []
    radius = round(0.06 * fs)

    def morphology(x, i):
        local = x[i - radius : i + radius + 1]
        height = abs(local - np.median(local))
        peak = int(np.argmax(height))
        width = next((b - a for a, b in intervals(height >= height[peak] / 2) if a <= peak < b), 0)
        return height[peak], width

    for i, j in pairs:
        a, b = clean_peaks[i], out_peaks[j]
        if min(a, b) < radius or max(a, b) + radius >= len(reference):
            continue
        amplitude, width = morphology(reference, a)
        seen, observed = morphology(output, b)
        amp_error.append(abs(amplitude - seen))
        width_error.append(1000 * abs(width - observed) / fs)
        timing.append(1000 * abs(a - b) / fs)
    return dict(
        reference="native probe on original observed ECG",
        missed=len(clean_peaks) - len(pairs),
        added=len(out_peaks) - len(pairs),
        matched=len(pairs),
        timing_mae_ms=float(np.mean(timing)) if timing else None,
        peak_amplitude_mae=float(np.mean(amp_error)) if amp_error else None,
        half_height_width_mae_ms=float(np.mean(width_error)) if width_error else None,
    )


def score(key, ref, x, out, estimated, events, trials, fs):
    affected = np.zeros(len(x), bool)
    for a, b in key:
        affected[a:b] = True
    alarm = np.zeros(len(x), bool)
    for a, b in events:
        alarm[a:b] = True
    before = x[affected] - ref[affected]
    after = out[affected] - ref[affected]
    detected = sum((a, b) in events for a, b in key)
    applied = {(r["a"], r["b"]) for r in trials if r["status"] == "accepted"}
    post, _, _ = scan(out, np.isfinite(out), {}, fs, "ECG", "balanced")
    new = {(z["start"], z["end"]) for z in post if z["kind"] in ("impulsive_core", "biphasic_core")} - set(events)
    outside = [(a, b) for a, b in new if not affected[a:b].any()]
    attempted = sum("candidate" in r for r in trials)
    return dict(
        injected_sites=len(key),
        exact_core_detections=detected,
        accepted_exact_sites=len(set(map(tuple, key)) & applied),
        duration_recall=float(alarm[affected].mean()),
        alarm_duration_outside_injection_s=float(alarm[~affected].sum() / fs),
        before_mae=float(np.abs(before).mean()),
        after_mae=float(np.abs(after).mean()),
        after_rmse=float(np.sqrt((after**2).mean())),
        before_rmse=float(np.sqrt((before**2).mean())),
        snr_gain_db=float(10 * np.log10(np.mean(before**2) / np.mean(after**2))) if np.mean(after**2) > 0 else None,
        released_diagnostic_landmarks=landmark_scores(ref, out, fs),
        introduced_outside_core_alerts_per_channel_hour=len(outside) / (len(ref) / fs / 3600),
        rollback_fraction_of_trials=sum(r["status"] == "rollback" for r in trials) / attempted if attempted else None,
        outside_changed=int(changed(ref, out)[~affected].sum()),
        normal_change_fraction=float(changed(ref, out)[~affected].mean()),
        estimated_samples=int(estimated.sum()),
        all_proposed=len(events),
        states=dict(Counter(r["status"] for r in trials)),
    )


def verified_peers(old, cases, observed, grid):
    """Resolve native acquisition metadata from the two public peer channels, without reading pre-injection ECG."""
    peers = []
    for channel in ("PPG", "ABP"):
        y = observed[channel]
        matches = []
        for c in cases.values():
            if c["task"] != channel or c["n"] != len(y) or not c["has_source_time"]:
                continue
            with np.load(old / "public" / c["id"] / "input.npz", allow_pickle=False) as z:
                if np.array_equal(z["time"], grid) and np.array_equal(z["values"], y, equal_nan=True):
                    valid = z["valid"].copy() if "valid" in z else np.isfinite(y)
                    est = z["estimated"].copy() if "estimated" in z else np.zeros(len(y), bool)
                    matches.append((c, (grid, y, valid, est, probe(y, grid, c["fs"], valid, est, channel))))
        if len(matches) != 1:
            raise ValueError("No unique native source clock for the public peer channel")
        peers.extend(matches)
    keys = {acquisition(c) for c, _ in peers}
    if len(keys) != 1:
        raise ValueError("Public peer channels came from different acquisitions")
    (origin,) = [c for c in cases.values() if c["task"] == "ECG" and acquisition(c) in keys]
    return origin, peers


def main(archive, out, comparison=False):
    old = archive / "zoom"
    cases = {c["id"]: c for c in read(old / "dataset_manifest.json")["cases"]}
    config = None
    if comparison:
        from tsmcp.cleaning.repairability import PROFILE

        config = dict(
            repair_protocol="v1.1",
            repair_domain=dict(
                profile_id=PROFILE["id"],
                profile_version=PROFILE["version"],
                use="development",
                species="human",
                device="Vital",
                goals={"ECG": "ECG_RR", "PPG": "PPG_PPI", "ABP": "ABP_pressure"},
            ),
        )
    groups = {}
    for p in old.glob("records/*/spike-trials.json"):
        for row in read(p):
            groups.setdefault(row["case"], []).append(row)
    result = dict(
        mode="fixed_proposal_and_all_approve_test_fixture",
        acceptance_spec=SPEC_ID,
        live_model_calls=0,
        **({"repair_config": config} if comparison else {}),
        independent_clinical_accuracy=None,
        natural=[],
        injections={},
    )
    for cid, rows in groups.items():
        with np.load(old / "public" / cid / "input.npz") as z:
            x = z["values"]
            t = z["time"]
            valid = z["valid"] if "valid" in z else np.isfinite(x)
        _, _, trials = execute(out / "natural" / cid, cases[cid], x, t, valid, [(r["a"], r["b"]) for r in rows], config)
        for r, s in zip(sorted(rows, key=lambda q: q["a"]), trials, strict=True):
            result["natural"].append(
                dict(
                    case=cid,
                    a=s["a"],
                    b=s["b"],
                    old=r["status"],
                    new=s["status"],
                    reason=s["reason"],
                    **({"reason_codes": s.get("reason_codes", [])} if comparison else {}),
                )
            )
    for name in ("bipolar", "same-sign", "moderate"):
        root = archive if name == "bipolar" else archive / name
        with np.load(root / "injection-input/waveform.npz") as z:
            x = z["ECG"]
            t = z["time"]
            if comparison:
                origin, peers = verified_peers(old, cases, z, t)
        fs = 1.0 / np.median(np.diff(t))
        fs = round(fs, 6)
        events, _, _ = scan(x, np.isfinite(x), {}, fs, "ECG", "balanced")
        cores = {
            (e["start"], e["end"])
            for e in events
            if e["kind"] in ("impulsive_core", "biphasic_core") and e["end"] - e["start"] <= 4
        }
        c = dict(id="target", record_id=name, fs=fs, task="ECG", unit="source_units", n=len(t))
        if comparison:
            c.update(
                {
                    k: origin[k]
                    for k in ("field", "has_source_time", "normalized_windows", "source_sha256", "rows", "record_id")
                }
            )
        y, estimated, trials = execute(out / name, c, x, t, np.isfinite(x), cores, config)
        truth = read(root / "injection-score/key.json")
        ref = np.load(root / "injection-score/observed-before.npy")
        result["injections"][name] = score(truth["ranges"], ref, x, y, estimated, cores, trials, fs)
        if comparison:
            key = set(map(tuple, truth["ranges"]))
            scored = result["injections"][name]
            scored["fixed_denominator_sites"] = [
                dict(
                    a=a,
                    b=b,
                    nominated=r["comparison"]["nominated"],
                    released=r["comparison"]["released_method"],
                    reason=r["reason"],
                    before_mae=float(np.mean(abs(x[a:b] - ref[a:b]))),
                    after_mae=float(np.mean(abs(y[a:b] - ref[a:b]))),
                    estimated_samples=int(estimated[a:b].sum()),
                    alternatives=[
                        dict(method=z["method"], status=z["status"], phase_mae=z.get("phase_mae"))
                        for z in r["comparison"]["alternatives"]
                    ],
                )
                for r in trials
                for a, b in [(r["a"], r["b"])]
                if (a, b) in key
            ]
            scored["missed_sites"] = sorted(key - set(cores))
            index = read(out / name / "export/derived-index.json")
            scored["measurement_coverage"] = dict(
                **index["coverage_samples_by_case"]["target"], derived_rows=index["rows"]
            )
            # Read the actual identity head; the candidate cache may also contain unpublished alternatives.
            identity = next(
                read(p)
                for p in (out / name / "artifacts").glob("*/artifact.json")
                if read(p)["parent_artifact_id"] is None
            )
            scored["measurement_coverage"]["original_native_samples"] = read(out / name / identity["probe_path"])[
                "measured_samples"
            ]
            observed_valid = np.isfinite(x)
            observed_est = np.zeros(len(x), bool)
            before = (t, x, observed_valid, observed_est, probe(x, t, fs, observed_valid, observed_est, "ECG"))
            scored["qrs_conflicts"] = [
                dict(
                    core=[r["a"], r["b"]],
                    action=r["qrs_conflict"]["action"],
                    independent=r["qrs_conflict"]["independent"]["status"],
                    review=r["qrs_conflict"]["review_status"],
                    public_peer_evidence=conflict_evidence(
                        c, before, r["a"], r["b"], r["qrs_conflict"]["sample"], peers
                    )["peers"],
                )
                for r in trials
                if r.get("qrs_conflict")
            ]
    write(out / "verification.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--short-core-comparison", action="store_true")
    args = parser.parse_args()
    main(args.archive, args.out, args.short_core_comparison)
