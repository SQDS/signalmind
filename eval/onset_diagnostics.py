"""Task E phase 1: audit frozen development results, without opening any dataset.

The input checkpoints are pinned to the task's initial commit. Suite manifests
are used only through the existing source-qualified record exclusion helper.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.af_episode_metrics import (  # noqa: E402
    LONG_AF_TOL_S,
    clipped,
    dev_records,
    fresh_population,
    intersection,
    longest,
    seconds,
    sha256,
    suite_record_ids,
    summary,
)

CHECKPOINT = "f12e695"
RESULTS = ROOT / "eval/results/af-continuity"
INPUTS = {
    "R": ("baseline-round2.json", "fresh-R.json"),
    "V": ("variant-120.json", "fresh-V.json"),
}


def eligible(row):
    """Apply the annotation-only fresh rule identically to ALL development rows."""
    truth = row["longest_annotated"]
    return bool(truth and truth[1] - truth[0] > 300 and row["true_pct_of_window"] < 90)


def load_checkpoints(cases_dir):
    excluded, manifests = suite_record_ids(cases_dir)
    split = json.loads((ROOT / "eval/tasks/ltafdb_split.json").read_text())
    dev = dev_records(split, excluded)
    fresh, fresh_manifests = fresh_population(cases_dir)
    assert manifests == fresh_manifests
    populations = [[f"{db}/{rec}" for db, rec in group] for group in (dev, fresh)]
    assert len(populations[0]) == 20 and len(populations[1]) == 38
    rows, provenance = {}, {}
    for arm, filenames in INPUTS.items():
        rows[arm] = []
        for filename, population in zip(filenames, populations, strict=True):
            path = RESULTS / filename
            frozen = subprocess.check_output(["git", "show", f"{CHECKPOINT}:{path.relative_to(ROOT)}"], cwd=ROOT)
            if path.read_bytes() != frozen:
                raise ValueError(f"Changed input checkpoint: {filename}")
            data = json.loads(frozen)
            if data["permitted_records"] != population or [r["record"] for r in data["records"]] != population:
                raise ValueError(f"Population mismatch: {filename}")
            if data["suite_manifests"] != manifests or data["long_af_tol_s"] != LONG_AF_TOL_S:
                raise ValueError(f"Protocol mismatch: {filename}")
            expected = "22f74ad" if arm == "R" else "69c9f43"
            if not data["source_commit"].startswith(expected) or (arm == "V" and data["variant_gap_s"] != 120):
                raise ValueError(f"Wrong detector: {filename}")
            if any(r.get("crash") for r in data["records"]):
                raise ValueError(f"Incomplete checkpoint: {filename}")
            rows[arm].extend(data["records"])
            provenance[filename] = {
                "sha256": sha256(path),
                "source_commit": data["source_commit"],
                "runtime_24h": data["runtime_24h"],
                "detector_seconds_sum": sum(r["detector_seconds"] for r in data["records"]),
            }
    for r, v in zip(rows["R"], rows["V"], strict=True):
        for key in (
            "record",
            "window_s",
            "annotated_af_intervals",
            "annotated_episodes",
            "support_intervals",
            "readable_intervals",
        ):
            if r[key] != v[key]:
                raise ValueError(f"R/V disagree on frozen support/truth: {r['record']}/{key}")
    return rows, provenance, manifests


def longest_diagnostic(row):
    truth, detected = row["longest_annotated"], row["longest_detected"]
    assert detected == longest([[e["start"], e["end"]] for e in row["detected_episodes"]])
    overlap = seconds(intersection([truth], [detected])) if truth and detected else 0.0
    union_s = seconds([truth]) + seconds([detected]) - overlap if truth and detected else None
    return {
        "longest_annotated": truth,
        "longest_detected": detected,
        "overlap_s": overlap if truth else None,
        "iou": overlap / union_s if union_s else 0.0 if truth else None,
        "signed_onset_error_s": detected[0] - truth[0] if truth and detected else None,
        "onset_within_60s": bool(truth and detected and abs(detected[0] - truth[0]) <= LONG_AF_TOL_S),
        "fragmentation": row["fragmentation"],
    }


def bridge_gaps(row):
    """Every unsupported gap in V's longest span, with independent truth/quality axes."""
    span = row["longest_detected"]
    if span is None:
        return []
    support = clipped(row["support_intervals"], *span)
    gaps = []
    for left, right in zip(support, support[1:], strict=False):
        a, b = left[1], right[0]
        duration = b - a
        assert 0 < duration < 120
        truth = clipped(row["annotated_af_intervals"], a, b)
        readable = clipped(row["readable_intervals"], a, b)
        af_s, readable_s = seconds(truth), seconds(readable)
        unknown_s = max(0.0, duration - readable_s)
        readable_af_s = seconds(intersection(truth, readable))
        gaps.append(
            {
                "start_s": a,
                "end_s": b,
                "duration_s": duration,
                "new_bridge_vs_R": duration >= 30,
                "annotated_rhythm": "non-AF" if af_s == 0 else "AF" if af_s == duration else "mixed",
                "annotated_af_s": af_s,
                "annotated_non_af_s": duration - af_s,
                "readability": "unknown"
                if readable_s == 0
                else "readable_non_AF_support"
                if unknown_s == 0
                else "mixed",
                "unknown_s": unknown_s,
                "readable_non_af_gap_s": readable_s,
                "readable_annotated_af_s": readable_af_s,
                "readable_annotated_non_af_s": readable_s - readable_af_s,
            }
        )
    return gaps


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        **dict(
            zip(
                ("min", "q25", "median", "q75", "max"),
                map(float, np.quantile(values, [0, 0.25, 0.5, 0.75, 1])),
                strict=True,
            )
        ),
    }


def diagnostic_summary(rows, arm):
    errors = [r[arm]["signed_onset_error_s"] for r in rows]
    present = [v for v in errors if v is not None]
    return {
        "n_records": len(rows),
        "missing_detection": sum(v is None for v in errors),
        "early": sum(v < 0 for v in present),
        "late": sum(v > 0 for v in present),
        "exact": sum(v == 0 for v in present),
        "onset_correct": sum(r[arm]["onset_within_60s"] for r in rows),
        "signed_error_s": distribution(errors),
        "iou": distribution([r[arm]["iou"] for r in rows]),
        "positive_overlap": sum(r[arm]["overlap_s"] > 0 for r in rows),
        "iou_at_least_half": sum(r[arm]["iou"] >= 0.5 for r in rows),
        "overlap_s": distribution([r[arm]["overlap_s"] for r in rows]),
        "near_misses_60_to_300s": [
            {"record": r["record"], "signed_error_s": r[arm]["signed_onset_error_s"]}
            for r in rows
            if r[arm]["signed_onset_error_s"] is not None and 60 < abs(r[arm]["signed_onset_error_s"]) <= 300
        ],
        "median_fragmentation": float(np.median([r[arm]["fragmentation"] for r in rows])),
        "oracle_constant_offset_ceiling": offset_ceiling(present),
    }


def offset_ceiling(errors):
    """Diagnostic upper bound, not a proposed correction: best shared offset using truth.

    Subtracting offset c succeeds iff c lies in [error-60, error+60].
    Count closed endpoints before removals to respect the inclusive tolerance.
    """
    boundaries = sorted({e + d for e in errors for d in (-LONG_AF_TOL_S, LONG_AF_TOL_S)})
    counts = [sum(abs(e - c) <= LONG_AF_TOL_S for e in errors) for c in boundaries]
    best = max(counts, default=0)
    return {
        "max_correct": best,
        "example_offset_subtracted_s": next((c for c, n in zip(boundaries, counts, strict=True) if n == best), None),
        "interpretation": "annotation-informed explanatory upper bound, not a tested candidate",
    }


def gap_summary(gaps):
    return {
        "count": len(gaps),
        "duration_s": distribution([g["duration_s"] for g in gaps]),
        "rhythm_counts": {k: sum(g["annotated_rhythm"] == k for g in gaps) for k in ("AF", "non-AF", "mixed")},
        "readability_counts": {
            k: sum(g["readability"] == k for g in gaps) for k in ("unknown", "readable_non_AF_support", "mixed")
        },
        "totals_s": {
            key: sum(g[key] for g in gaps)
            for key in (
                "duration_s",
                "annotated_af_s",
                "annotated_non_af_s",
                "unknown_s",
                "readable_non_af_gap_s",
                "readable_annotated_af_s",
                "readable_annotated_non_af_s",
            )
        },
    }


def run(cases_dir, output):
    source, provenance, manifests = load_checkpoints(cases_dir)
    rows = [
        {
            "record": r["record"],
            "onset_eligible": eligible(r),
            "R": longest_diagnostic(r),
            "V": longest_diagnostic(v),
            "V_longest_bridged_gaps": bridge_gaps(v),
        }
        for r, v in zip(source["R"], source["V"], strict=True)
        if r["has_af"]
    ]
    gaps = [g for r in rows for g in r["V_longest_bridged_gaps"]]
    result = {
        "phase": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "access": "Frozen development JSON artifacts only; no dataset files or detector execution",
        "checkpoint_commit": CHECKPOINT,
        "script_sha256": sha256(Path(__file__)),
        "inputs": provenance,
        "suite_manifests": manifests,
        "permitted_records": [r["record"] for r in source["R"]],
        "eligibility_rule": "longest <30 s merged annotated AF/AFL episode >300 s AND raw whole-record burden <90%",
        "eligible_records": [r["record"] for r in rows if r["onset_eligible"]],
        "interval_convention": "half-open seconds; longest span ranked by duration then earliest onset; missing detection IoU=0",
        "readability_note": "readable_non_af_gap_s means R-readable without R AF support; readable_annotated_non_af_s uses annotation truth instead",
        "records": rows,
        "summary": {
            arm: {
                "all_af": diagnostic_summary(rows, arm),
                "eligible": diagnostic_summary([r for r in rows if r["onset_eligible"]], arm),
                "burden_overlap_all_records": summary(source[arm]),
            }
            for arm in source
        },
        "bridging": {
            "all": gap_summary(gaps),
            "new_30_to_120s": gap_summary([g for g in gaps if g["new_bridge_vs_R"]]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("eligible_records", "summary", "bridging")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cases_dir, args.output)
