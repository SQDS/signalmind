"""Organizer-side short-core scoring with a fixed assignment list and observed-parent fallback."""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from itertools import product
from pathlib import Path

import numpy as np
from jsonschema import ValidationError, validate

from tsmcp.cleaning.common import array_hash, changed, digest, read, sha, write
from tsmcp.cleaning.tasks import probe, score_events

from .executor import SCHEMAS


def load_result(folder, case, arm):
    result = read(folder / "result.json")
    validate(result, read(SCHEMAS / "method_result.schema.json"))
    if result["arm_id"] != arm or any(case[k] != result[k] for k in ("case_id", "input_hash", "parent_version_id")):
        raise ValueError("Wrong assignment")
    arrays = []
    for key, name in [
        ("released_waveform", "values.npy"),
        ("validity_mask", "task_valid.npy"),
        ("provenance_mask", "estimated.npy"),
    ]:
        ref = result[key]
        if ref is None:
            if result["status"] in ("timeout", "error", "invalid_output", "budget_exhausted", "abstained"):
                return result, None
            raise ValueError("Missing numerical delivery")
        if ref["artifact_id"] != name or sha(folder / name) != ref["sha256"]:
            raise ValueError("Artifact binding mismatch")
        arr = np.load(folder / name, allow_pickle=False)
        if arr.shape != (case["samples"],) or (arr.dtype.kind not in "fiu" if not arrays else arr.dtype != bool):
            raise ValueError("Invalid numerical delivery")
        arrays.append(arr)
    return result, arrays


def read_trace(folder, result):
    ref = result["trace"]
    if ref is None:
        return {}
    path = folder / "trace.json"
    if ref["kind"] != "trace" or ref["artifact_id"] != path.name or sha(path) != ref["sha256"]:
        raise ValueError("Trace binding mismatch")
    return read(path)


def selected_arrays(folder, result, trace, released, n):
    ref = result["selected_candidate"]
    if ref is None:
        return None
    if ref == result["released_waveform"] and released is not None:
        return released
    # A reviewer may return the observed parent after a valid numerical trial. Bind that trial's own masks.
    if ref["kind"] != "waveform" or ref["artifact_id"] != "selected-values.npy":
        raise ValueError("Unknown selected candidate artifact")
    support = trace.get("selected_candidate_support", {})
    arrays = []
    for name, descriptor in [
        ("selected-values.npy", ref),
        ("selected-task_valid.npy", support.get("validity_mask")),
        ("selected-estimated.npy", support.get("provenance_mask")),
    ]:
        if not descriptor or descriptor["artifact_id"] != name or sha(folder / name) != descriptor["sha256"]:
            raise ValueError("Selected candidate support binding mismatch")
        arr = np.load(folder / name, allow_pickle=False)
        if arr.shape != (n,) or (arr.dtype.kind not in "fiu" if not arrays else arr.dtype != bool):
            raise ValueError("Invalid selected candidate array")
        arrays.append(arr)
    return arrays


def wave_quality(x, y, v, e, valid, est, truth, status, delta, scale_floor=None):
    if "damaged" not in truth or "reference" not in truth:
        keys = (
            "core_sites",
            "recovered_sites",
            "reference_core_samples",
            "core_abs_error_sum",
            "identity_abs_error_sum",
            "all_core_mae",
            "recovered_subset_mae",
            "normal_changed_samples",
            "normal_samples",
            "normal_added_abs_error_sum",
            "window_mae",
            "window_nmae",
            "reference_scale",
            "window_reference_samples",
        )
        return dict.fromkeys(keys) | {"waveform_reference_status": "independent_reference_or_mask_unavailable"}
    core = truth["damaged"].astype(bool)
    ref = truth["reference"]
    if core.shape != x.shape or ref.shape != x.shape:
        raise ValueError("Private reference grid mismatch")
    mask = v & ~e & np.isfinite(ref)
    if "reference_support" in truth:
        support = truth["reference_support"]
        if support.shape != mask.shape or support.dtype != bool:
            raise ValueError("Private evaluation support must match the frozen grid")
        mask &= support
    if scale_floor is not None and (not np.isfinite(scale_floor) or scale_floor <= 0):
        raise ValueError("Reference scale floor must be positive and bound to the unit")
    y = np.where(valid, y, x)
    mae = float(np.mean(np.abs(y[mask] - ref[mask]))) if mask.any() else None
    scale = (
        max(float(np.ptp(np.quantile(ref[mask], [0.05, 0.95]))), scale_floor)
        if mask.any() and scale_floor is not None
        else None
    )
    normal = mask & ~core
    scored = mask & core
    from tsmcp.processing import intervals

    sites = intervals(core)
    recovered = 0
    qualified_mae = []
    for a, b in sites:
        m = scored[a:b]
        if not m.all() or not valid[a:b].all() or not est[a:b].all():
            continue
        post = float(np.mean(abs(y[a:b] - ref[a:b])))
        if post < float(np.mean(abs(x[a:b] - ref[a:b]))) and status == "completed":
            recovered += 1
            qualified_mae.append(post)
    return dict(
        waveform_reference_status="supplied",
        window_reference_samples=int(mask.sum()),
        window_mae=mae,
        reference_scale=scale,
        window_nmae=mae / scale if scale is not None else None,
        core_sites=len(sites),
        recovered_sites=recovered,
        reference_core_samples=int(scored.sum()),
        core_abs_error_sum=float(np.abs(y[scored] - ref[scored]).sum()),
        identity_abs_error_sum=float(np.abs(x[scored] - ref[scored]).sum()),
        all_core_mae=float(np.abs(y[scored] - ref[scored]).mean()) if scored.any() else None,
        recovered_subset_mae=float(np.mean(qualified_mae)) if qualified_mae else None,
        normal_changed_samples=int((normal & delta).sum()),
        normal_samples=int(normal.sum()),
        normal_added_abs_error_sum=float((np.abs(y[normal] - ref[normal]) - np.abs(x[normal] - ref[normal])).sum()),
    )


def score_case(case, observed, delivered, truth, status, scale_floor=None, *, has_release=True, trace=None):
    t, x, v, e = observed
    y, valid, est = delivered
    allow = np.zeros_like(v)
    for region in case["allowed_write_ranges"]:
        lo = region["start_sample"] - case["offset"]
        hi = region["end_sample"] - case["offset"]
        allow[lo:hi] = True
    delta = changed(x, y)
    released_valid = valid if has_release else np.zeros_like(valid)
    usable = released_valid & ~est
    native = probe(y, t, case["fs"], released_valid, est, case["task"])
    from tsmcp.cleaning.dataset_evidence import native_rows

    trace = trace or {}
    decisions = {(z["start_sample"], z["end_sample"]): z["quality"] for z in trace.get("measurement_policy", [])}
    unresolved = (trace.get("submission") or {}).get("unresolved_ranges", [])
    kept = np.zeros(len(x), bool)
    dispositions = Counter()
    for a, b, _ in native_rows(native):
        left, right = a + case["offset"], b + case["offset"]
        quality = decisions.get((left, right), "keep")
        if quality == "keep" and any(r["start_sample"] <= right and left < r["end_sample"] for r in unresolved):
            quality = "review"
        dispositions[quality] += 1
        if quality == "keep":
            kept[a:b] = True
    event_metrics = None
    if "event_samples" in truth:
        event_metrics = score_events(
            [z["sample"] + case["offset"] for z in native["events"]], truth["event_samples"], fs=case["fs"]
        )
    pressure_error = None
    if case["task"] == "ABP" and "pressure_cycles" in truth:
        comparisons = []
        # Reference rows are [global foot, next foot, SBP, DBP, time mean]. Compare matched native cycles only.
        outputs = {
            (z["start_sample"] + case["offset"], z["stop_sample"] + case["offset"]): z for z in native["pressures"]
        }
        for left, right, sbp, dbp, mean in truth["pressure_cycles"]:
            z = outputs.get((int(left), int(right)))
            if z:
                comparisons.append([abs(z["sbp"] - sbp), abs(z["dbp"] - dbp), abs(z["mean_pressure"] - mean)])
        pressure_error = dict(
            reference_cycles=len(truth["pressure_cycles"]),
            matched_cycles=len(comparisons),
            mae_mmhg=dict(zip(("SBP", "DBP", "MAP"), np.mean(comparisons, axis=0).tolist(), strict=True))
            if comparisons
            else None,
        )
    return dict(
        pressure_vs_reference=pressure_error,
        **wave_quality(x, y, v, e, valid, est, truth, status, delta, scale_floor),
        outside_write_changes=int(((delta | (v != valid) | (e != est)) & ~allow).sum()),
        disabled_samples=int((v & ~valid).sum()),
        undelivered_samples=0 if has_release else int(v.sum()),
        measured_support_samples=native["measured_samples"],
        approved_native_support_samples=int(kept.sum()),
        measurement_dispositions=dict(dispositions),
        native_sample_support=int(usable.sum()),
        derived_released_samples=0,
        native_events_vs_reference=event_metrics,
        event_reference_status="supplied" if event_metrics is not None else "unavailable",
    )


def macro_errors(rows):
    """Equal weight for windows within type, then types within repeat, repeats within subject, subjects within arm."""
    if any(r["stratum"] is None or r["subject"] is None or r["window_nmae"] is None for r in rows):
        return dict(
            patient_macro_nmae=None, selected_patient_macro_nmae=None, reason="incomplete_stratum_reference_or_scale"
        )
    windows = defaultdict(list)
    for r in rows:
        windows[r["subject"], r["repetition"], r["stratum"]].append(
            [r["window_nmae"], r["selected_candidate"]["window_nmae"]]
        )
    by_repeat = defaultdict(list)
    for (subject, repeat, _), errors in windows.items():
        by_repeat[subject, repeat].append(np.mean(errors, axis=0))
    by_subject = defaultdict(list)
    for (subject, _), errors in by_repeat.items():
        by_subject[subject].append(np.mean(errors, axis=0))
    patients = {p: np.mean(errors, axis=0).tolist() for p, errors in by_subject.items()}
    released, selected = np.mean(list(patients.values()), axis=0).tolist()
    return dict(
        patient_macro_nmae=released,
        selected_patient_macro_nmae=selected,
        patients={p: dict(released=z[0], selected=z[1]) for p, z in patients.items()},
    )


def report(manifest_path):
    manifest = read(manifest_path)
    base = manifest_path.parent
    assignments = manifest["cases"]
    fields = ("subject", "record", "noise_source")
    memberships = {field: {} for field in fields}
    ids = [c["case_id"] for c in assignments]
    arms = manifest["arms"]
    repeats = manifest.get("repetitions", 1)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Expected a positive number of frozen repetitions")
    if len(ids) != len(set(ids)) or len(arms) != len(set(arms)) or not ids or not arms:
        raise ValueError("Empty or duplicate assignment")
    rows = []
    assigned_paths = set()
    for item in assignments:
        for field in fields:
            group = item[field]
            if group is not None and memberships[field].setdefault(group, item["split"]) != item["split"]:
                raise ValueError(f"{field} crosses splits")
        folder = base / item["public_dir"]
        case = read(folder / "case.json")
        if (
            item["input_hash"] != case["input_hash"]
            or item["case_id"] != case["case_id"]
            or digest({k: v for k, v in case.items() if k != "input_hash"}) != case["input_hash"]
        ):
            raise ValueError("Changed public case")
        with np.load(folder / "observed.npz", allow_pickle=False) as z:
            observed = [z[k] for k in ("t", "x", "valid", "estimated")]
        _, x, v, e = observed
        if array_hash(*observed) != case["observed_hash"]:
            raise ValueError("Changed observed arrays")
        private = base / item["private_npz"] if item.get("private_npz") is not None else None
        if private is not None and sha(private) != item["private_sha256"]:
            raise ValueError("Changed private reference")
        with np.load(private, allow_pickle=False) if private is not None else nullcontext({}) as ref:
            floor = manifest.get("reference_scale_floors", {}).get(case["task"], {}).get(case["unit"])
            identity = score_case(case, observed, [x, v, e], ref, "preserve", floor, has_release=False)
            for arm, repeat in product(arms, range(1, repeats + 1)):
                paths = item["outputs"].get(arm, {})
                if isinstance(paths, str) and repeats != 1:
                    raise ValueError("Assign a separate output for each repetition")
                target = paths if isinstance(paths, str) else paths.get(str(repeat))
                path = base / target if target else None
                if path is not None:
                    if path.resolve() in assigned_paths:
                        raise ValueError("Output reused across frozen assignments")
                    assigned_paths.add(path.resolve())
                counters = {}
                failure = None
                arrays = None
                candidate_score = None
                rejected_delivery = None
                mode = None
                status = "missing"
                if path is not None and (path / "result.json").exists():
                    try:
                        result, arrays = load_result(path, case, arm)
                        trace = read_trace(path, result)
                        mode = trace.get("mode", "deterministic")
                        expected = manifest.get("arm_modes", {}).get(arm)
                        if expected is None and arm in ("generic_tool_agent", "one_shot_planner", "signalmind_full"):
                            expected = "live"
                        if expected and expected != mode:
                            raise ValueError("Execution mode differs from frozen assignment")
                        status = result["status"]
                        counters = result["usage"]
                        if arrays is not None and (
                            not np.isfinite(arrays[0][v]).all() or (arrays[1] & ~v).any() or (e & ~arrays[2]).any()
                        ):
                            raise ValueError("Invalid support or nonfinite waveform")
                        candidate = selected_arrays(path, result, trace, arrays, len(x))
                        if candidate is not None:
                            if (
                                not np.isfinite(candidate[0][v]).all()
                                or (candidate[1] & ~v).any()
                                or (e & ~candidate[2]).any()
                            ):
                                raise ValueError("Invalid selected candidate support")
                            candidate_score = score_case(case, observed, candidate, ref, "completed", floor)
                    except (KeyError, TypeError, ValueError, OSError, ValidationError) as exc:
                        status, arrays, failure = "invalid_output", None, str(exc).splitlines()[0][:400]
                fallback = arrays is None or status not in ("completed", "abstained")
                scored = (
                    identity
                    if fallback
                    else score_case(
                        case, observed, arrays, ref, status, floor, has_release=status == "completed", trace=trace
                    )
                )
                if scored["outside_write_changes"]:
                    rejected_delivery = scored
                    status, fallback = "invalid_output", True
                    scored = identity
                    failure = "outside_authorized_write_scope"
                rows.append(
                    dict(
                        case_id=case["case_id"],
                        subject=item["subject"],
                        split=item["split"],
                        arm=arm,
                        repetition=repeat,
                        stratum=item.get("stratum"),
                        status=status,
                        fallback=fallback,
                        detail=failure,
                        usage=counters,
                        mode=mode,
                        candidate_fallback=candidate_score is None or bool(candidate_score["outside_write_changes"]),
                        selected_candidate=identity
                        if candidate_score is None or candidate_score["outside_write_changes"]
                        else candidate_score,
                        rejected_candidate=candidate_score
                        if candidate_score and candidate_score["outside_write_changes"]
                        else None,
                        rejected_delivery=rejected_delivery,
                        **scored,
                    )
                )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["split"], row["arm"]].append(row)
    summaries = []
    for (split, arm), items in grouped.items():
        referenced = [z for z in items if z["core_sites"] is not None]
        n = sum(z["reference_core_samples"] for z in referenced)
        summaries.append(
            dict(
                split=split,
                arm=arm,
                scheduled_cases=len(items),
                unique_cases=len({r["case_id"] for r in items}),
                subjects=len({r["subject"] for r in items}),
                repetitions=repeats,
                **macro_errors(items),
                statuses=dict(Counter(z["status"] for z in items)),
                reference_cases=len(referenced),
                sites=sum(z["core_sites"] for z in referenced) if referenced else None,
                recovered=sum(z["recovered_sites"] for z in referenced) if referenced else None,
                core_mae=sum(z["core_abs_error_sum"] for z in referenced) / n if n else None,
                normal_changed_samples=sum(z["normal_changed_samples"] for z in referenced) if referenced else None,
            )
        )
    paired = []
    for g in summaries:
        if g["arm"] != "generic_tool_agent" or "patients" not in g:
            continue
        s = next((s for s in summaries if s["split"] == g["split"] and s["arm"] == "signalmind_full"), None)
        if s and "patients" in s:
            diffs = {p: item["released"] - s["patients"][p]["released"] for p, item in g["patients"].items()}
            paired.append(
                dict(
                    split=g["split"], patient_nmae_reductions=diffs, mean_reduction=float(np.mean(list(diffs.values())))
                )
            )
    return dict(manifest_sha256=sha(manifest_path), summaries=summaries, paired=paired, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write(args.out, report(args.manifest))
