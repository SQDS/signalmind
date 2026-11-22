"""Offline scoring reruns probes from waveform arrays, never trusts submitted event tables."""

from pathlib import Path

import numpy as np

from .common import no_reference, read, sha, time_support
from .engine import load_arrays
from .tasks import fixed_denominator_metrics, match_events, probe, score_events


def score_pressure(predicted, reference, tolerances=None, association_s=None, released=True):
    if tolerances is None or association_s is None:
        return {"content_correct": None, "reason": "pressure_tolerances_not_frozen"}

    def centers(rows):
        return [(r["start_s"] + r["stop_s"]) / 2 for r in rows]

    pairs = match_events(centers(predicted), centers(reference), association_s)
    errors = {k: [abs(predicted[i][k] - reference[j][k]) for i, j in pairs] for k in ("sbp", "dbp", "mean_pressure")}
    correct = bool(reference) and len(predicted) == len(reference) == len(pairs)
    correct = correct and all(all(e <= tolerances[k] for e in values) for k, values in errors.items())
    return {
        "content_correct": bool(correct),
        "correct_delivery": bool(correct and released),
        "released": released,
        "false_positive": len(predicted) - len(pairs),
        "false_negative": len(reference) - len(pairs),
        "mae": {k: float(np.mean(v)) if v else None for k, v in errors.items()},
    }


def score_task(task, measures, reference, fs, released):
    if not reference or not reference.get("complete") or reference.get("reference_type") == "candidate_edits":
        return {"content_correct": None, "reason": "missing_or_incomplete_independent_reference"}
    if reference.get("fiducial_definition") != measures["definition"]:
        return {"content_correct": None, "reason": "reference_probe_fiducial_mismatch"}
    tolerances = reference.get("tolerances")
    if task == "ABP":
        return score_pressure(
            measures["pressures"], reference["cycles"], tolerances, reference.get("association_s"), released
        )
    if not tolerances or not {"association_ms", "peak_ms", "interval_ms"}.issubset(tolerances):
        return {"content_correct": None, "reason": "event_tolerances_not_frozen"}
    return score_events(
        [e["sample"] for e in measures["events"]], reference["peak_samples"], fs=fs, released=released, **tolerances
    )


def event_cores(measures, reference, fs, n, released_support, content_support):
    """Match with full context, then assign reference events and RR right ends to 10 s cores."""
    p = np.asarray([e["sample"] for e in measures["events"]], int)
    r = np.asarray(reference["peak_samples"], int)
    tol = reference["tolerances"]
    pairs = match_events(p, r, fs * tol["association_ms"] / 1000)
    matched = {j: i for i, j in pairs}
    extras = [i for i in range(len(p)) if i not in {a for a, _ in pairs}]
    scorable = np.zeros(n, bool)
    for a, b in reference.get("scorable_support", [[0, n]]):
        scorable[a:b] = True
    rows = []
    for a in range(0, n, round(10 * fs)):
        b = min(n, a + round(10 * fs))
        local = np.flatnonzero((r >= a) & (r < b))
        rr = [j for j in local if j > 0 and scorable[r[j - 1] : r[j] + 1].all()]
        content = None
        if scorable[a:b].all() and rr:
            good_peaks = all(j in matched and abs(p[matched[j]] - r[j]) <= fs * tol["peak_ms"] / 1000 for j in local)
            good_rr = all(
                j in matched
                and j - 1 in matched
                and matched[j] == matched[j - 1] + 1
                and measures["events"][matched[j]]["support_id"] == measures["events"][matched[j - 1]]["support_id"]
                and content_support[r[j - 1] : r[j] + 1].all()
                and abs((p[matched[j]] - p[matched[j - 1]]) - (r[j] - r[j - 1])) <= fs * tol["interval_ms"] / 1000
                for j in rr
            )
            content = bool(good_peaks and good_rr and not any(a <= p[i] < b for i in extras))
        rows.append(
            {
                "range_samples": [a, b],
                "duration_s": (b - a) / fs,
                "released": bool(released_support[a:b].all()),
                "content_correct": content,
            }
        )
    return rows


def evaluate_version(workspace, version_id, reference_path=None):
    workspace = Path(workspace)
    version = read(workspace / "versions" / version_id / "version.json")
    cases = {c["id"]: c for c in read(workspace / "dataset_manifest.json")["cases"]}
    references = read(reference_path).get("cases", {}) if reference_path else {}
    rows = []
    for cid in version["case_ids"]:
        case = cases[cid]
        if cid not in version["results"]:
            rows.append(
                {
                    "task": case["task"],
                    "duration_s": case["duration_s"],
                    "released": False,
                    "content_correct": None,
                    "reason": "execution_failed",
                }
            )
            continue
        artifact = version["results"][cid]
        if sha(workspace / "public" / cid / "input.npz") != case["input_sha256"]:
            raise ValueError("Source changed before evaluation")
        if artifact["arrays"] and sha(workspace / artifact["arrays"]) != artifact["arrays_sha256"]:
            raise ValueError("Candidate changed before evaluation")
        raw, times, values, valid, estimated = load_arrays(workspace, case, artifact)
        content_support = np.isfinite(raw) & np.isfinite(values) & ~estimated & time_support(case)
        actual = probe(values, times, case["fs"], content_support, estimated, case["task"])
        released = bool(np.all(valid & ~estimated))
        score = score_task(case["task"], actual, references.get(cid), case["fs"], released)
        baseline = probe(raw, times, case["fs"], np.isfinite(raw), np.zeros(len(raw), bool), case["task"])
        before = score_task(case["task"], baseline, references.get(cid), case["fs"], True)
        if case["task"] in {"ECG", "PPG"} and score.get("content_correct") is not None:
            core_rows = event_cores(actual, references[cid], case["fs"], case["n"], valid & ~estimated, content_support)
            original_rows = event_cores(
                baseline, references[cid], case["fs"], case["n"], np.isfinite(raw), np.isfinite(raw)
            )
            for row, original in zip(core_rows, original_rows, strict=True):
                rows.append(
                    {"case_id": cid, "task": case["task"], "baseline_correct": original["content_correct"], **row}
                )
            continue
        rows.append(
            {
                "case_id": cid,
                "task": case["task"],
                "duration_s": case["duration_s"],
                "released": released,
                "baseline_correct": before.get("content_correct"),
                **score,
            }
        )
    return {
        "version_id": version_id,
        "probe_source": "rerun_from_actual_waveforms",
        "tasks": {
            task: fixed_denominator_metrics([r for r in rows if r["task"] == task])
            for task in sorted({r["task"] for r in rows})
        },
        "reference_status": no_reference() if not references else "provided_offline",
        "cases": rows,
    }
