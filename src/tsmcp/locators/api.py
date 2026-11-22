"""Common locator contract. Detection does not change samples or select cleaning actions."""

from __future__ import annotations

import hashlib

import numpy as np

from tsmcp.processing import intervals

METHODS = ("A_current_30s", "B_legacy_rules", "C_sphere_original", "D_sphere_fixed")
AUXILIARY = "X_legacy_auxiliary"
CONFIG = {
    "sensitivity": "balanced",
    "phase_bins": 128,
    "r_base": 1.0,
    "radial_scale": 0.25,
    "hr_range_bpm": [42.0, 165.0],
    "gap_s": 0.35,
    "min_event_s": 0.08,
}


def _finite_segments(support, times, fs):
    for a, b in intervals(support):
        breaks = (
            np.flatnonzero(
                np.abs(np.diff(times[a:b]) - 1 / fs) > max(1e-6, 32 * np.spacing(np.max(np.abs(times[a:b]))))
            )
            + a
            + 1
        )
        edges = np.r_[a, breaks, b]
        yield from zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True)


def locate_issues(values, fs, *, method, source_support=None, times=None, device_flags=None, include_mesh=False):
    from .legacy import legacy_auxiliary, legacy_diagnosis, legacy_rules
    from .sphere import locate_sphere
    from .vendor.time_sphere_original import TimeSphereArtifactDetector

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Expected nonempty one-dimensional values and a positive finite sampling rate")
    if method not in (*METHODS, AUXILIARY):
        raise ValueError("Unknown locator method")
    n = len(x)
    source = np.ones(n, bool) if source_support is None else np.asarray(source_support, bool)
    time = np.arange(n) / fs if times is None else np.asarray(times, float)
    flags = np.zeros(n, bool) if device_flags is None else np.asarray(device_flags, bool)
    if source.shape != x.shape or time.shape != x.shape or flags.shape != x.shape:
        raise ValueError("Metadata arrays must match the single channel")
    if not np.isfinite(time).all():
        raise ValueError("Input times must be finite")
    original_hash = hashlib.sha256(x.tobytes()).hexdigest()
    x.setflags(write=False)
    eligible = source & np.isfinite(x)
    evaluated = np.zeros(n, bool)
    scores = np.full(n, np.nan)
    events, points, spans, notes, legacy_reports, meshes, diagnostic_blocks = [], [], [], [], [], [], []
    for a, b in _finite_segments(eligible, time, fs):
        piece = x[a:b]
        local, local_points = [], []
        try:
            if method == "A_current_30s":
                local = legacy_diagnosis(piece, fs)
                evaluated[a:b] = True
            elif method == "B_legacy_rules":
                if len(piece) < max(3, int(0.2 * fs) | 1):
                    notes.append({"range": [a, b], "reason": "insufficient_legacy_convolution_context"})
                    continue
                local = legacy_rules(piece, fs)
                evaluated[a:b] = True
            elif method == AUXILIARY:
                local, local_points = legacy_auxiliary(piece, fs)
                evaluated[a:b] = True
            elif method == "C_sphere_original":
                detector = TimeSphereArtifactDetector(fs=fs, sensitivity="balanced")
                report = detector.detect(piece, include_mesh=include_mesh)
                legacy_reports.append(
                    {"range": [a, b], "report": {k: v for k, v in report.items() if k != "visualization_payload"}}
                )
                for ev in report["detected_artifacts"]:
                    start, inclusive_end = ev["sample_indices"]
                    local.append(
                        {
                            "start": int(start),
                            "end": min(len(piece), int(inclusive_end) + 1),
                            "kind": ev["type"],
                            "score": ev["severity_score"],
                            "metric": ev["dominant_metric"],
                            "evidence": {
                                "original_cause": ev["geometric_cause"],
                                "uncalibrated": True,
                                "original_endpoint_convention": "inclusive_as_declared",
                            },
                            "resolution_samples": max(1, int(np.ceil(report["estimated_period_sec"] * fs / 128))),
                        }
                    )
                if include_mesh:
                    meshes.append({"range": [a, b], "original_payload": report["visualization_payload"]})
            else:
                report = locate_sphere(piece, fs, sensitivity="balanced", include_mesh=include_mesh)
                local = report["events"]
                evaluated[a:b] = report["evaluated_mask"]
                scores[a:b] = report["scores"]
                notes.append({"range": [a, b], "notes": report["coverage_notes"]})
                diagnostic_blocks.append(
                    {
                        "range": [a, b],
                        "parameters": report["parameters"],
                        "component_evaluated_samples": report["component_evaluated_samples"],
                        "component_unscored_ranges": report["component_unscored_ranges"],
                    }
                )
                if include_mesh and report.get("mesh") is not None:
                    meshes.append({"range": [a, b], "mesh": report["mesh"]})
            for event in local:
                start, end = int(event["start"]), int(event["end"])
                if not 0 <= start < end <= b - a or not np.isfinite(event["score"]):
                    raise ValueError("Invalid candidate interval or score")
                events.append(
                    {
                        **event,
                        "start": start + a,
                        "end": end + a,
                        "time_range_s": [(start + a) / fs, (end + a) / fs],
                        "source_time_start_s": float(time[start + a]),
                    }
                )
            for point in local_points:
                points.append({**point, "sample": point["sample"] + a})
            spans.append({"start": a, "end": b, "status": "executed"})
        except (ValueError, FloatingPointError, IndexError) as exc:
            spans.append({"start": a, "end": b, "status": "unscored", "reason": str(exc)})
    if hashlib.sha256(x.tobytes()).hexdigest() != original_hash:
        raise RuntimeError("Locator modified its input")
    coverage_known = method != "C_sphere_original"
    complete = all(s["status"] == "executed" for s in spans) and (not coverage_known or np.all(evaluated[eligible]))
    result = {
        "schema": "signalmind.locator.v1",
        "method": method,
        "status": "ok" if complete and eligible.any() else "partial",
        "n": n,
        "fs": float(fs),
        "input_values_sha256": original_hash,
        "events": sorted(events, key=lambda e: (e["start"], e["end"])),
        "point_candidates": points,
        "coverage_known": coverage_known,
        "evaluated_ranges": intervals(evaluated) if coverage_known else None,
        "unscored_ranges": intervals(eligible & ~evaluated) if coverage_known else None,
        "source_unavailable_ranges": intervals(~source),
        "missing_ranges": intervals(~np.isfinite(x)),
        "public_flag_ranges": intervals((~np.isfinite(x) | flags) & source),
        "execution_spans": spans,
        "coverage_notes": notes,
        "parameters": CONFIG if method.startswith(("C_", "D_")) else {"policy": "frozen_existing_thresholds"},
        "classification_claim": "candidate_only_not_noise_truth",
        "confidence_calibrated": False,
        "preprocessing": "common finite/source-supported runs; true time gaps split; no imputation",
        "reference_metrics": None,
        "diagnostic_blocks": diagnostic_blocks,
    }
    if legacy_reports:
        result["original_reports"] = legacy_reports
        result["coverage_notes"].append(
            "Original local-scoring coverage is not certified; its SQI/recommendation is archival only"
        )
    if method == AUXILIARY:
        result["classification_claim"] = "relative_shape_and_change_candidates_not_pollution_labels"
        result["parameters"] = {"discord_k": 5, "max_changes": 10, "limits_preserved": True}
    result["sample_scores"] = scores
    if include_mesh:
        result["meshes"] = meshes
    return result
