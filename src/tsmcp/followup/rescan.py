"""Apply exactly the registered method to source and accepted batches, including previously unflagged blocks."""

from collections import Counter
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import array_hash, digest, read, sha, write
from tsmcp.cleaning.engine import load_arrays
from tsmcp.cleaning.provenance import check_stop
from tsmcp.followup.evidence import comparison
from tsmcp.followup.pattern import detect
from tsmcp.inspection.run import load_channel


def overlaps(interval, others):
    a, b = interval
    return any(a < y and x < b for x, y in others)


def compare(raw, before_support, values, support, fs, rule, *, detector=detect):
    observed = np.zeros(len(raw), bool)
    before = detector(raw, before_support, fs, rule, evaluated=observed)
    # On observed numeric values even if the previous fine pass marked them invalid.
    finite = before_support & np.isfinite(values)
    numeric = detector(values, finite, fs, rule) if not np.array_equal(values, raw, equal_nan=True) else before
    released = support & finite
    scored = np.zeros(len(raw), bool)
    after = detector(values, released, fs, rule, evaluated=scored)
    findings = []
    for a, b in before:
        n = int(released[a:b].sum())
        if not n:
            status = "excluded_from_task"
        elif overlaps((a, b), after):
            status = "still_detected"
        elif n == b - a and not scored[a:b].all():
            status = "insufficient_detection_context"
        elif n < b - a:
            status = "partially_excluded"
        else:
            status = "no_longer_triggered"
        findings.append(
            dict(
                range=[a, b],
                status=status,
                eligible_samples=n,
                still_present_in_numeric_signal=overlaps((a, b), numeric),
            )
        )
    return dict(
        before=before,
        after_numeric=numeric,
        after_eligible=after,
        newly_triggered=[r for r in after if not overlaps(r, before)],
        findings=findings,
        eligible_samples=int(released.sum()),
        evaluated_samples=int(scored.sum()),
        original_evaluated_samples=int(observed.sum()),
    )


def scan_version(workspace, decision, version, out):
    workspace, out = Path(workspace), Path(out)
    task = decision["task"]
    expected_rule = digest([task, decision["rule"], sha(Path(__file__).with_name("pattern.py"))])
    detector = detect
    if decision.get("detector_family") == "screen_rule":
        from tsmcp.followup.screen_rule import rescan_rule, rule_id

        expected_rule = rule_id(task, decision["rule"])

        def detector(x, usable, fs, rule, *, evaluated=None):
            return rescan_rule(x, usable, fs, task, rule, evaluated=evaluated)

    if expected_rule != decision["rule_id"]:
        raise ValueError("Detector or rule differs from the registered method")
    manifest = read(workspace / "dataset_manifest.json")
    cases = [c for c in manifest["cases"] if c["task"] == task]
    if version["stage"] != "batch" or set(version["results"]) != {c["id"] for c in cases}:
        raise ValueError("Follow-up needs a batch snapshot covering the complete modality")
    if not any(
        c["id"] == decision["seed_case"] and "raw-" + c["input_sha256"] == decision["seed_version"] for c in cases
    ):
        raise ValueError("Reviewer seed does not belong to this source scope")
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rows = []
    errors = []
    covered = 0
    scored_s = 0
    for c in cases:
        check_stop(workspace)
        cid = c["id"]
        artifact = version["results"][cid]
        path = workspace / "public" / cid / "input.npz"
        try:
            if sha(path) != c["input_sha256"] or artifact["source_sha256"] != c["input_sha256"]:
                raise ValueError("Source version mismatch")
            if artifact["arrays"] and sha(workspace / artifact["arrays"]) != artifact["arrays_sha256"]:
                raise ValueError("Waveform artifact digest mismatch")
            raw, masks = load_channel(path, c)
            initial = ~np.logical_or.reduce(list(masks.values()))
            _, times, values, valid, estimated = load_arrays(workspace, c, artifact)
            if array_hash(values, times, valid, estimated) != artifact["content_hash"]:
                raise ValueError("Processed data content mismatch")
            result = compare(raw, initial, values, valid & ~estimated, c["fs"], decision["rule"], detector=detector)
            if cid == decision["seed_case"]:
                comparison(out, c, raw, values, valid & ~estimated, decision, version["version_id"], result)
            rows.append(
                dict(
                    case_id=cid,
                    fs=c["fs"],
                    source_version="raw-" + c["input_sha256"],
                    artifact_id=artifact["artifact_id"],
                    content_hash=artifact["content_hash"],
                    **result,
                )
            )
            counts.update({k: len(result[k]) for k in ("before", "after_numeric", "after_eligible", "newly_triggered")})
            counts.update(f["status"] for f in result["findings"])
            covered += result["eligible_samples"] / c["fs"]
            scored_s += result["evaluated_samples"] / c["fs"]
        except (ValueError, OSError) as exc:
            errors.append(dict(case_id=cid, reason=str(exc)))
    write(out / "findings.json", rows)
    summary = dict(
        rule_id=decision["rule_id"],
        detector=decision["rule"],
        cause_hypothesis=decision["suspected_cause"],
        task=task,
        scope_hash=manifest["scope_hash"],
        waveform_version=version["version_id"],
        scanned=len(rows),
        planned=len(cases),
        errors=errors,
        complete=not errors,
        eligible_s=covered,
        evaluated_s=scored_s,
        counts=dict(counts),
        findings_sha256=sha(out / "findings.json"),
        reference_accuracy=None,
        new_model_calls_during_scan=0,
    )
    write(out / "summary.json", summary)
    return summary


def scan_accepted(workspace, decision, out):
    workspace = Path(workspace)
    state = read(workspace / "agent_state.json")
    vid = state["accepted_tasks"][decision["task"]]
    version = read(workspace / "versions" / vid / "version.json")
    for cid, artifact in version["results"].items():
        if state["accepted"].get(cid) != {"version_id": vid, "artifact_id": artifact["artifact_id"]}:
            raise ValueError("First-pass accepted pointers disagree")
    return scan_version(workspace, decision, version, out)
