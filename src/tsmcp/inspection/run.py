"""Run the Development Kit inspection milestone on a prepared SignalMind workspace."""

import argparse
import csv
import fcntl
import json
import time
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import digest, read, sha, time_support, write
from tsmcp.cleaning.provenance import check_stop
from tsmcp.core import dsp, peaks
from tsmcp.inspection.contracts import validate
from tsmcp.inspection.profiles import profile as inspection_profile
from tsmcp.inspection.scan import HIGH_BAND_HZ, HIGH_FRACTION, LINE_FRACTION, PRESETS, scan
from tsmcp.plugins.ecg import rpeaks
from tsmcp.plugins.pressure import pulses
from tsmcp.processing import intervals

HERE = Path(__file__).parent


def sampling_fraction(preset):
    return {"low": 0.02, "medium": 0.05, "high": 0.1}[inspection_profile(preset)["escalation"]["unknown_sampling"]]


HARD = {"source_exclusion", "missing", "device_clipped", "missing_flag", "source_invalid", "source_estimated"}


def coord(case, a, b):
    return dict(record_id=case["record_id"], channel_id=case["task"], block_id=case["id"], start_sample=a, end_sample=b)


def version_id(case):
    return "raw-" + case["input_sha256"]


def specification(cases, preset, run_id, profile_hash, wall_seconds):
    blocks = []
    for c in cases:
        rate = Fraction(str(c["fs"])).limit_denominator(10**6)
        if rate <= 0 or c["n"] < 1:
            raise ValueError(f"Invalid size or rate for {c['id']}")
        blocks.append(
            dict(
                record_id=c["record_id"],
                channel_id=c["task"],
                block_id=c["id"],
                source_version_id=version_id(c),
                n_samples=c["n"],
                clock_id=c["id"],
                sampling_kind="uniform",
                sampling_rate=dict(numerator=rate.numerator, denominator=rate.denominator),
                unit=c["unit"],
                time_map_ref=None,
                content_digest="sha256:" + c["input_sha256"],
            )
        )
    spec = dict(
        schema_version="1.0.0",
        run_id=run_id,
        mode="inspect_only",
        barrier_scope="run",
        source_blocks=blocks,
        acceptance_spec_id="inspection-coverage",
        acceptance_spec_version="1.0.0",
        registry_snapshot_id="impl-" + profile_hash,
        inspection_profile_id="vital-" + preset,
        inspection_profile_version="1.4.0",
        inspection=dict(
            window_seconds=inspection_profile(preset)["multiscale"]["windows_seconds"],
            stride_policy_id=f"hop-{inspection_profile(preset)['multiscale']['hop_ratio']}-end-anchored",
            sensitivity=PRESETS[preset]["sensitivity"],
            audit_fraction=sampling_fraction(preset),
        ),
        repair_policy=dict(
            max_attempts_per_candidate=0,
            allow_imputation=False,
            allowed_tool_ids=[],
            preserve_sampling_rate=True,
            preserve_time_axis=True,
            preserve_length=True,
            preserve_units=True,
        ),
        budget=dict(max_model_tokens=1, max_wall_seconds=wall_seconds, max_parallel_jobs=1),
    )
    validate("run_spec", spec)
    return spec


def load_channel(path, case):
    n = case["n"]
    with np.load(path, allow_pickle=False) as z:

        def vector(key):
            v = z[key]
            if v.shape == (1, n):
                v = v[0]
            if v.shape != (n,):
                raise ValueError(f"{key}: expected one channel of {n} samples")
            return v

        x = vector("values").astype(float)
        if "time" in z:
            t = vector("time")
            if not np.isfinite(t).all() or np.any(
                np.abs(np.diff(t) - 1 / case["fs"]) > max(1e-6, 32 * np.spacing(np.max(np.abs(t))))
            ):
                raise ValueError("Input does not describe one continuous uniform block")
        source = time_support(case)
        masks = {"source_exclusion": ~source, "missing": ~np.isfinite(x)}
        for key, cause in (
            ("valid", "source_invalid"),
            ("estimated", "source_estimated"),
            ("device_saturation_flag", "device_clipped"),
            ("observed_missing_mask", "missing_flag"),
        ):
            if key in z:
                flag = vector(key)
                if not np.isin(flag, (0, 1)).all():
                    raise ValueError(f"{key}: invalid mask")
                masks[cause] = (flag == 0) if key == "valid" else (flag == 1)
    return x, masks


def inspect_block(path, case, preset, run_id):
    x, masks = load_channel(path, case)
    usable = ~np.logical_or.reduce(list(masks.values()))
    events, levels, trends = scan(x, usable, masks, case["fs"], case["task"], preset)
    candidates, evidence = [], []
    alarm = np.zeros(len(x), bool)
    for event in events:
        a, b, kind = event["start"], event["end"], event["kind"]
        region = coord(case, a, b)
        uid = digest([run_id, region, kind])[:24]
        observation = dict(
            result_id="obs-" + uid,
            tool_id=kind,
            tool_version="1.2.0" if kind == "large_step" and case["task"] == "ECG" else "1.0.0",
            status="ok",
            quality="uncertain" if kind not in HARD else "unusable",
            risk_score=None,
            calibration_profile_id=None,
            score_is_probability=False,
            evidence_ids=["ev-" + uid],
        )
        row = dict(
            schema_version="1.0.0",
            candidate_id="c-" + uid,
            run_id=run_id,
            source_version_id=version_id(case),
            range=region,
            state="EVIDENCE_READY",
            final_action=None,
            candidate_kind=(
                "possible_physiology"
                if kind == "abp_pulse_episode"
                else "unknown"
                if kind in HARD or kind == "ecg_baseline_shift"
                else "suspected_artifact"
            ),
            observations=[observation],
            evidence_ids=["ev-" + uid],
            patch_id=None,
        )
        validate("candidate", row)
        evidence.append(
            dict(
                evidence_id="ev-" + uid,
                observed_version_id=version_id(case),
                candidate_id=row["candidate_id"],
                range=region,
                **event,
            )
        )
        candidates.append(row)
        alarm[a:b] = True
    from tsmcp.cleaning.contracts import ToolResult
    from tsmcp.cleaning.ledger import region

    supported = usable.copy()
    tool_results = []
    for name, r in levels["tool_results"].items():
        result = ToolResult(
            tool_id=name,
            status=r["status"],
            evaluated_ranges=[region(case["id"], a, b) for a, b in r["evaluated_ranges"]],
            unsupported_ranges=[region(case["id"], a, b) for a, b in r["unsupported_ranges"]],
            evidence_refs=[k["evidence_id"] for k in evidence if k["kind"] == name],
        )
        tool_results.append(result.model_dump(mode="json"))
        if r["status"] != "not_applicable":
            for a, b in r["unsupported_ranges"]:
                supported[a:b] = False
    tiles = []
    for start, end in intervals(supported & ~alarm):
        size = max(1, round(5 * case["fs"]))
        for a in range(start, end, size):
            region = coord(case, a, min(a + size, end))
            # Seed and scope determine an independent Bernoulli sample; profiles share the raw-data seed.
            if int(digest([111101, version_id(case), region])[:16], 16) / 2**64 < sampling_fraction(preset):
                tiles.append(
                    dict(
                        range=region,
                        observed_version_id=version_id(case),
                        inclusion_probability=sampling_fraction(preset),
                        label_source=None,
                        verdict="unreviewed",
                    )
                )
    unknown_tiles = []
    for start, end in intervals(usable & ~supported & ~alarm):
        size = max(1, round(5 * case["fs"]))
        for a in range(start, end, size):
            coordinate = coord(case, a, min(a + size, end))
            prob = sampling_fraction(preset)
            if int(digest(["unknown", 111101, version_id(case), coordinate])[:16], 16) / 2**64 < prob:
                unknown_tiles.append(
                    dict(
                        range=coordinate,
                        observed_version_id=version_id(case),
                        inclusion_probability=prob,
                        source_status="insufficient_support",
                        verdict="unreviewed",
                    )
                )
    return dict(
        case_id=case["id"],
        status="scanned",
        n=len(x),
        fs=case["fs"],
        eligible=int(usable.sum()),
        candidate_samples=int(alarm.sum()),
        observed_candidate_samples=int((alarm & usable).sum()),
        negative_samples=int((supported & ~alarm).sum()),
        insufficient_samples=int((usable & ~supported & ~alarm).sum()),
        tool_results=tool_results,
        candidates=candidates,
        evidence=evidence,
        context_levels=levels,
        trends=trends,
        audit=tiles,
        unknown_audit=unknown_tiles,
    )


def inspect(workspace, out, preset="balanced", wall_seconds=3600):
    workspace, out = Path(workspace), Path(out)
    if preset not in PRESETS or wall_seconds <= 0:
        raise ValueError("Select a supported preset and a positive wall-time budget")
    manifest = read(workspace / "dataset_manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".inspection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return inspect_locked(workspace, out, preset, wall_seconds, manifest)


def inspect_locked(workspace, out, preset, wall_seconds, manifest):
    start = time.monotonic()
    cases = manifest["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case id")
    profile = dict(
        preset=preset,
        thresholds=PRESETS[preset],
        settings=inspection_profile(preset),
        line_fraction=LINE_FRACTION,
        high_fraction=HIGH_FRACTION,
        high_band=HIGH_BAND_HZ,
        calibrated=False,
        model_api_cap_usd=0,
        evidence_policy="ECG morphology, ABP cycle review, source flags and spectrum",
        files={p.name: sha(p) for p in sorted([*HERE.glob("*.py"), *HERE.glob("schemas/*.json")])},
        qrs_dependencies={p.__name__: sha(Path(p.__file__)) for p in (rpeaks, dsp, peaks)},
        pulse_detector_sha256=sha(Path(pulses.__file__)),
        numpy_version=np.__version__,
        audit_seed=111101,
        audit_fraction=sampling_fraction(preset),
    )
    scope = digest([manifest, profile, wall_seconds])
    run_id = "inspect-" + scope[:24]
    spec = specification(cases, preset, run_id, digest(profile), wall_seconds)
    if (out / "run_spec.json").exists() and read(out / "run_spec.json") != spec:
        raise ValueError("Inspection configuration or input changed; use a new output directory")
    published = out / "candidate_manifest.json"
    if published.exists():
        frozen = read(published)
        if frozen["run_id"] != run_id or any(sha(out / name) != h for name, h in frozen["files"].items()):
            raise ValueError("Frozen inspection artifacts changed")
        for c in cases:
            check_stop(workspace)
            if sha(workspace / "public" / c["id"] / "input.npz") != c["input_sha256"]:
                raise ValueError("Source changed since inspection")
        return read(out / "inspection_summary.json")
    write(out / "profile.json", profile)
    write(out / "run_spec.json", spec)
    # Publication marker is written last; partial, stopped and failed scans cannot claim a frozen manifest.
    (out / "inspection_summary.json").unlink(missing_ok=True)
    totals = Counter()
    errors = []
    attempted = 0
    with ExitStack() as stack:
        streams = {
            k: stack.enter_context((out / (k + ".jsonl")).open("w"))
            for k in ("candidates", "evidence", "coverage", "context", "negative_audit", "unknown_audit")
        }
        csvfile = stack.enter_context((out / "intervals.csv").open("w", encoding="utf-8-sig", newline=""))
        writer = csv.writer(csvfile)
        writer.writerow(
            ["case_id", "observed_version_id", "start_s", "end_s", "trigger", "maximum_value", "threshold", "units"]
        )
        for c in cases:
            check_stop(workspace)
            if time.monotonic() - start > wall_seconds:
                errors.append({"reason": "wall_budget_exhausted", "pending_blocks": len(cases) - attempted})
                for pending in cases[attempted:]:
                    entry = dict(
                        case_id=pending["id"], n=pending["n"], status="pending", reason="wall_budget_exhausted"
                    )
                    streams["coverage"].write(json.dumps(entry) + "\n")
                break
            attempted += 1
            path = workspace / "public" / c["id"] / "input.npz"
            try:
                if sha(path) != c["input_sha256"]:
                    raise ValueError("Input checksum changed")
                cache = out / "blocks" / (digest([run_id, c["id"]]) + ".json")
                if cache.exists():
                    saved = read(cache)
                    block = saved["result"]
                    if saved["digest"] != digest(block):
                        raise ValueError("Inspection checkpoint digest changed")
                else:
                    block = inspect_block(path, c, preset, run_id)
                    write(cache, {"result": block, "digest": digest(block)})
            except (ValueError, OSError) as exc:
                errors.append(dict(case_id=c["id"], reason=str(exc)))
                streams["coverage"].write(
                    json.dumps(dict(case_id=c["id"], n=c["n"], status="failed", reason=str(exc))) + "\n"
                )
                continue
            for name, key in (
                ("candidates", "candidates"),
                ("evidence", "evidence"),
                ("negative_audit", "audit"),
                ("unknown_audit", "unknown_audit"),
            ):
                for row in block[key]:
                    streams[name].write(json.dumps(row, allow_nan=False) + "\n")
            coverage = {
                k: block[k]
                for k in (
                    "case_id",
                    "status",
                    "n",
                    "fs",
                    "eligible",
                    "candidate_samples",
                    "negative_samples",
                    "insufficient_samples",
                    "tool_results",
                    "context_levels",
                )
            }
            streams["coverage"].write(json.dumps(coverage) + "\n")
            for trend in block["trends"]:
                streams["context"].write(
                    json.dumps(dict(case_id=c["id"], observed_version_id=version_id(c), **trend)) + "\n"
                )
            for e in block["evidence"]:
                writer.writerow(
                    [
                        c["id"],
                        version_id(c),
                        e["start"] / c["fs"],
                        e["end"] / c["fs"],
                        e["kind"],
                        e["value"],
                        e["threshold"],
                        e["unit"],
                    ]
                )
            totals.update(
                dict(
                    scanned_blocks=1,
                    scanned_samples=c["n"],
                    candidates=len(block["candidates"]),
                    scanned_channel_s=c["n"] / c["fs"],
                    negative_audit_tiles=len(block["audit"]),
                    candidate_channel_s=block["candidate_samples"] / c["fs"],
                    eligible_channel_s=block["eligible"] / c["fs"],
                    observed_candidate_s=block["observed_candidate_samples"] / c["fs"],
                    insufficient_channel_s=block["insufficient_samples"] / c["fs"],
                )
            )
    complete = attempted == len(cases) and not (
        errors or manifest.get("audit_failures") or manifest.get("unavailable_tasks")
    )
    summary = dict(
        mode="inspect_only",
        run_id=run_id,
        preset=preset,
        patients=manifest.get("patients"),
        planned_blocks=len(cases),
        inspection_complete=complete,
        candidate_manifest_frozen=complete,
        audit_failures=manifest.get("audit_failures", []),
        unavailable_tasks=manifest.get("unavailable_tasks", []),
        errors=errors,
        reference_accuracy=None,
        reference_reason="no independent labels used during scanning",
        elapsed_s=time.monotonic() - start,
        workflow_stage="candidate_discovery",
        diagnosis_complete=False,
        model_calls=0,
        api_usd=0,
        **totals,
    )
    write(out / "inspection_summary.json", summary)
    if complete:
        write(
            out / "candidate_manifest.json",
            dict(
                revision=1,
                run_id=run_id,
                barrier_scope="run",
                coverage_complete=True,
                candidate_count=totals["candidates"],
                files={
                    name: sha(out / name)
                    for name in (
                        "run_spec.json",
                        "profile.json",
                        "candidates.jsonl",
                        "coverage.jsonl",
                        "evidence.jsonl",
                        "context.jsonl",
                        "negative_audit.jsonl",
                        "unknown_audit.jsonl",
                        "intervals.csv",
                        "inspection_summary.json",
                    )
                },
            ),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run contract-based, read-only signal inspection")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="balanced")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    args = parser.parse_args()
    result = inspect(args.workspace, args.out, args.preset, args.wall_seconds)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["inspection_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
