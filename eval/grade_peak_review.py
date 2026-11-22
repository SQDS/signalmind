"""Post-freeze local execution and annotation-agreement scoring; never imported by model tools."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dataset_workflow import read, sha, write  # noqa: E402
from tsmcp.peak_policy import SCHEMA, apply, digest, observe, validate  # noqa: E402

BASELINES = {
    "keep_original": {"schema_version": SCHEMA, "alignment": "none", "exclude_if": []},
    "fixed_dsp": {
        "schema_version": SCHEMA,
        "alignment": "positive",
        "radius_ms": 80,
        "min_energy_fraction": 0.15,
        "refractory_ms": 250,
        "exclude_if": [
            {"feature": "missing_fraction", "op": "gt", "value": 0.01},
            {"feature": "flat_fraction", "op": "gt", "value": 0.10},
            {"feature": "detector_disagreement_fraction", "op": "gt", "value": 0.30},
            {"feature": "kept_rr_outside_fraction", "op": "gt", "value": 0.20},
        ],
    },
}


def grade_case(truth, original, prediction):
    old = np.asarray(original, int)
    values = prediction.get("new_peak_samples", [])
    delivered = (
        isinstance(prediction.get("exclude"), bool)
        and isinstance(values, list)
        and len(values) == len(old)
        and all(type(v) is int and -1 <= v < 7500 for v in values)
        and len({v for v in values if v >= 0}) == sum(v >= 0 for v in values)
    )
    result = Counter(windows=1, delivered=int(delivered))
    reference_exclude = truth["disposition"] == "EXCLUDED"
    result["reference_excluded" if reference_exclude else "reference_retained"] = 1
    predicted = prediction.get("exclude") if delivered else None
    result["window_agreement"] = int(delivered and predicted == reference_exclude)
    if reference_exclude:
        result["excluded_found"] = int(predicted is True)
        subgroup = "explicit_exclusions" if truth["exclusion_subtype"] == "整窗剔除" else "unresolved_exclusions"
        result[subgroup] = 1
        result[subgroup + "_found"] = int(predicted is True)
        # No per-peak reference exists for excluded windows; do not label their peaks false.
        return dict(result)
    result["retained_found"] = int(predicted is False)
    result["retained_wrongly_excluded"] = int(predicted is True)
    target = np.asarray(truth["target_peak_samples"], int)
    new = np.asarray(values, int) if delivered else np.full(len(old), -2)
    target_delete, predicted_delete = target < 0, new == -1
    keep, present = target >= 0, new >= 0
    moves = keep & (target != old)
    unmodified = keep & (target == old)
    result.update(
        {
            "peak_reference_windows": 1,
            "original_marker_count": len(old),
            "reference_deletions": int(target_delete.sum()),
            "predicted_deletions": int(predicted_delete.sum()),
            "delete_tp": int((target_delete & predicted_delete).sum()),
            "delete_fp": int((~target_delete & predicted_delete).sum()),
            "delete_fn": int((target_delete & ~predicted_delete).sum()),
            "reference_retained_peaks": int(keep.sum()),
            "reference_moves": int(moves.sum()),
            "reference_unchanged_peaks": int(unmodified.sum()),
            "unneeded_edits": int((unmodified & (new != old)).sum()),
            "wrongly_deleted_true_peaks": int((keep & predicted_delete).sum()),
        }
    )
    for tolerance in (1, 5, 10):
        matched = keep & present & (np.abs(new - target) <= tolerance)
        result[f"located_within_{tolerance}_samples"] = int(matched.sum())
        result[f"moves_correct_within_{tolerance}_samples"] = int((moves & matched & (new != old)).sum())
    exact = bool(
        delivered
        and np.array_equal(target_delete, predicted_delete)
        and np.all(present[keep] & (np.abs(new[keep] - target[keep]) <= 1))
    )
    result["peak_windows_all_correct_8ms"] = int(exact)
    result["retained_windows_end_to_end_8ms"] = int(exact and predicted is False)
    common = keep & present
    result["common_kept_peak_count"] = int(common.sum())
    result["common_kept_abs_error_ms_sum"] = float(np.abs(new[common] - target[common]).sum() * 8)
    return dict(result)


def summarize(rows):
    totals = Counter()
    for row in rows:
        totals.update(row["counts"])

    def ratio(a, b):
        return totals[a] / totals[b] if totals[b] else None

    recall, specificity = ratio("excluded_found", "reference_excluded"), ratio("retained_found", "reference_retained")
    f1_den = 2 * totals["delete_tp"] + totals["delete_fp"] + totals["delete_fn"]
    return {
        "counts": dict(totals),
        "subjects": len({r["subject_key"] for r in rows}),
        "window_agreement": ratio("window_agreement", "windows"),
        "exclusion_recall": recall,
        "retention_specificity": specificity,
        "balanced_window_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "deletion_precision": ratio("delete_tp", "predicted_deletions"),
        "deletion_recall": ratio("delete_tp", "reference_deletions"),
        "deletion_f1": 2 * totals["delete_tp"] / f1_den if f1_den else None,
        "retained_peak_location_agreement_8ms": ratio("located_within_1_samples", "reference_retained_peaks"),
        "move_success_8ms": ratio("moves_correct_within_1_samples", "reference_moves"),
        "move_success_40ms": ratio("moves_correct_within_5_samples", "reference_moves"),
        "unneeded_edit_rate": ratio("unneeded_edits", "reference_unchanged_peaks"),
        "conditional_common_peak_mae_ms": ratio("common_kept_abs_error_ms_sum", "common_kept_peak_count"),
        "retained_window_end_to_end_8ms": ratio("retained_windows_end_to_end_8ms", "reference_retained"),
        "note": "Agreement with author-supplied review, not clinical accuracy. Peak metrics include ALL reference-retained windows even if the policy rejects them.",
    }


def execute_and_grade(project):
    policy = validate(read(project / "agent/policy.json"))
    assert digest(policy) == read(project / "agent/freeze.json")["policy_sha256"]
    for name, expected in read(project / "provenance.json")["source_sha256"].items():
        assert sha(ROOT / name) == expected, "Code changed after inference"
    index = read(project / "public/index.json")
    truth = {r["id"]: r for r in read(project / "private/truth.json")}
    arms = {**BASELINES, "DS_shared_peak_policy": policy}
    reports = {arm: [] for arm in arms}
    predictions = {arm: {} for arm in arms}
    started = time.monotonic()
    for case in index["cases"]:
        path = project / "public" / case["id"] / "input.npz"
        assert sha(path) == case["sha256"]
        with np.load(path, allow_pickle=False) as data:
            ecg, old = data["ecg"].copy(), data["old_peaks"].copy()
        observation = observe(ecg, old)
        for arm, spec in arms.items():
            result = apply(ecg, old, spec, observed=observation)
            predictions[arm][case["id"]] = result
            counts = grade_case(truth[case["id"]], old, result)
            reports[arm].append({**truth[case["id"]], "counts": counts, "predicted_exclude": result["exclude"]})
    summary = {}
    for arm, rows in reports.items():
        folder = project / "outputs" / arm
        folder.mkdir(parents=True, exist_ok=True)
        write(folder / "policy.json", arms[arm])
        write(folder / "predictions.json", predictions[arm])
        write(folder / "scored_cases.json", rows)
        summary[arm] = {
            "all_440": summarize(rows),
            "unseen_subjects": summarize([r for r in rows if not r["policy_development_subject"]]),
        }
    all_truth = list(truth.values())
    n_excluded = sum(r["disposition"] == "EXCLUDED" for r in all_truth)
    result = {
        "schema_version": "signalmind.peak-review-scores.v0.1",
        "date": "2026-11-06",
        "arms": summary,
        "always_exclude_baseline": {
            "window_agreement": n_excluded / len(all_truth),
            "balanced_window_accuracy": 0.5,
            "retained_window_end_to_end_8ms": 0.0,
        },
        "batch_elapsed_s": time.monotonic() - started,
        "batch_model_calls": 0,
        "model_api_estimate_usd": read(project / "session.json")["estimated_peak_usd"],
        "quality_retries": 0,
        "input_arrays_changed": False,
        "limitations": [
            "Developer-visible, outcome-selected review set; labels hidden from the evaluated DS session.",
            "Shared policy controls newly added peak primitives; this is not the old waveform-only agent unchanged.",
            "252 reference exclusions encode unresolved judgements, not confirmed false beats.",
            "No per-peak scoring for excluded windows; no full-patient 4Hz RRI or BRS reconstruction scored.",
            "Point estimates on correlated windows; no population-wide or clinical-generalization claim.",
        ],
    }
    write(project / "report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    execute_and_grade(parser.parse_args().project)
