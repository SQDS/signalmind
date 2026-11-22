"""Independent, fixed-denominator interval scoring for the locator comparison.

Detection workers must not import this module or receive its reference arrays.
All sample intervals are zero-based and half-open. Scores describe agreement with
the supplied corruption masks; they are not waveform restoration measurements.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLDS = (0.1, 0.3, 0.5)
FAILED_STATUSES = {"failed", "timeout", "invalid_output", "resource_exhausted"}


def _vector(values, n, name, *, dtype=bool):
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} must be a single-channel array with {n} samples")
    return array


def _ranges(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True))


def _range_mask(ranges, n):
    mask = np.zeros(n, bool)
    for start, end in ranges:
        if int(start) != start or int(end) != end or not 0 <= start < end <= n:
            raise ValueError(f"Invalid half-open sample interval [{start}, {end}) for length {n}")
        mask[int(start) : int(end)] = True
    return mask


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values else None


def _classification(tp, fp, fn, tn):
    truth = tp + fn
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, truth),
        # Normal-only cases are assessed through false alarms, never perfect F1.
        "f1": _ratio(2 * tp, 2 * tp + fp + fn) if truth else None,
        "iou": _ratio(tp, tp + fp + fn) if truth else None,
        "normal_fp_ratio": _ratio(fp, fp + tn),
    }


def match_intervals(predicted, reference, *, threshold=0.3, fs=1.0):
    """Maximize one-to-one match count, then total IoU (not greedy IoU matching)."""
    p, r = np.asarray(predicted, dtype=np.int64).reshape(-1, 2), np.asarray(reference, dtype=np.int64).reshape(-1, 2)
    matches = []
    if len(p) and len(r):
        overlap = np.maximum(0, np.minimum(p[:, None, 1], r[None, :, 1]) - np.maximum(p[:, None, 0], r[None, :, 0]))
        union = p[:, None, 1] - p[:, None, 0] + r[None, :, 1] - r[None, :, 0] - overlap
        iou = overlap / union
        qualifies = (overlap > 0) & (iou >= threshold)
        # The entire secondary reward is < 1, so it cannot sacrifice one match.
        reward = np.where(qualifies, 1 + iou / (min(len(p), len(r)) + 1), 0.0)
        pi, ri = linear_sum_assignment(-reward)
        for a, b in zip(pi, ri, strict=True):
            if qualifies[a, b]:
                matches.append(
                    {
                        "prediction_index": int(a),
                        "reference_index": int(b),
                        "prediction": p[a].tolist(),
                        "reference": r[b].tolist(),
                        "iou": float(iou[a, b]),
                        "onset_error_s": abs(int(p[a, 0]) - int(r[b, 0])) / fs,
                        "end_error_s": abs(int(p[a, 1]) - int(r[b, 1])) / fs,
                    }
                )
    tp, fp, fn = len(matches), len(p) - len(matches), len(r) - len(matches)
    return {
        "threshold": threshold,
        "predicted_events": len(p),
        "reference_events": len(r),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn) if len(r) else None,
        "onset_error_median_s": _percentile([m["onset_error_s"] for m in matches], 50),
        "onset_error_p90_s": _percentile([m["onset_error_s"] for m in matches], 90),
        "end_error_median_s": _percentile([m["end_error_s"] for m in matches], 50),
        "end_error_p90_s": _percentile([m["end_error_s"] for m in matches], 90),
        "matches": matches,
    }


def _track(prediction, domain, evaluated, *, fs, reference):
    prediction = prediction & domain
    size = int(domain.sum())
    alarm_count = int(prediction.sum())
    result = {
        "domain_samples": size,
        "domain_s": size / fs,
        "alarm_samples": alarm_count,
        "alarm_s": alarm_count / fs,
        "alarm_fraction": _ratio(alarm_count, size),
        "alarm_regions": len(_ranges(prediction)),
        "evaluated_samples": int((evaluated & domain).sum()) if evaluated is not None else None,
        "evaluated_s": float((evaluated & domain).sum() / fs) if evaluated is not None else None,
        "coverage_fraction": _ratio(int((evaluated & domain).sum()), size) if evaluated is not None else None,
        "unscored_samples": int((domain & ~evaluated).sum()) if evaluated is not None else None,
        "unscored_s": float((domain & ~evaluated).sum() / fs) if evaluated is not None else None,
        "reference_available": reference is not None,
    }
    if reference is None:
        result.update({name: None for name in ("precision", "recall", "f1", "iou", "normal_fp_ratio")})
        result["reference_reason"] = "No independent complete waveform-corruption reference supplied"
        return result

    n = len(domain)
    truth = _vector(reference["A"], n, "A") & domain
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth & domain).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth & domain).sum())
    result.update(_classification(tp, fp, fn, tn))
    for name, value in {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items():
        result[name + "_samples"] = value
        result[name + "_s"] = value / fs
    result.update(
        truth_samples=int(truth.sum()), truth_s=float(truth.sum() / fs), normal_s=float((domain & ~truth).sum() / fs)
    )
    false_regions = len(_ranges(prediction & ~truth & domain))
    result["false_alarm_regions"] = false_regions
    result["false_alarm_regions_per_hour"] = _ratio(false_regions * 3600, result["domain_s"])
    result["normal_only"] = bool(size and not truth.any())
    predicted_intervals, reference_intervals = _ranges(prediction), _ranges(truth)
    result["events"] = {
        str(threshold): match_intervals(predicted_intervals, reference_intervals, threshold=threshold, fs=fs)
        for threshold in IOU_THRESHOLDS
    }
    result["event_definition"] = "Connected components of prediction/reference after clipping to this scoring domain"
    result["event_fp_per_hour"] = _ratio(result["events"]["0.3"]["fp"] * 3600, result["domain_s"])
    clean = _vector(reference.get("C", ~_vector(reference["A"], n, "A")), n, "C")
    protected = _vector(reference.get("E", np.zeros(n, bool)), n, "E") & clean & domain
    result["normal_event_samples"] = int(protected.sum())
    result["normal_event_s"] = float(protected.sum() / fs)
    result["normal_event_fp_samples"] = int((prediction & protected).sum())
    result["normal_event_fp_s"] = float((prediction & protected).sum() / fs)
    result["normal_event_fp_ratio"] = _ratio(result["normal_event_fp_samples"], result["normal_event_samples"])
    result["components"] = {}
    for name in sorted(reference):
        if name.startswith("component_"):
            support = _vector(reference[name], n, name) & domain
            detected = support & prediction
            result["components"][name] = {
                "support_samples": int(support.sum()),
                "support_s": float(support.sum() / fs),
                "alarmed_samples": int(detected.sum()),
                "alarmed_s": float(detected.sum() / fs),
                "support_coverage": _ratio(int(detected.sum()), int(support.sum())),
                "meaning": "Coverage by any alarm, not correct identification of the corruption type",
            }
    return result


def score_result(result, *, input_values, fs, reference, device_flags=None, source_support=None):
    """Score one method/case, retaining source exclusions and failure denominators.

    ``device_flags`` may be a boolean saturation mask or a dictionary containing
    ``device_saturation_flag`` and/or ``observed_missing_mask``. Source exclusions
    are not pollution labels: both tracks use D & source_support. The all-problems
    track adds the same observable missing/device flags to every method's alarms.
    Failed methods have no validated algorithm alarms, but retain that common QC.
    """
    values = np.asarray(input_values, dtype=np.float64)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Expected one channel and a positive finite sampling rate")
    n = len(values)
    source = np.ones(n, bool) if source_support is None else _vector(source_support, n, "source_support")
    finite = np.isfinite(values)
    public = ~finite
    if isinstance(device_flags, Mapping):
        for key in ("device_saturation_flag", "observed_missing_mask"):
            if key in device_flags:
                public = public | _vector(device_flags[key], n, key)
    elif device_flags is not None:
        public = public | _vector(device_flags, n, "device_flags")
    public = public & source
    status = result.get("status", "ok")
    failed = status in FAILED_STATUSES
    events = [] if failed else result.get("events", [])
    prediction = _range_mask([(e["start"], e["end"]) for e in events], n)
    coverage_known = failed or bool(result.get("coverage_known", False))
    evaluated = (
        np.zeros(n, bool)
        if failed
        else _range_mask(result.get("evaluated_ranges", []), n) & source & finite
        if coverage_known
        else None
    )
    domain = np.ones(n, bool) if reference is None else _vector(reference["D"], n, "D")
    all_domain = domain & source
    observed_domain = all_domain & finite & ~public
    tracks = {
        "unknown_observed": _track(prediction, observed_domain, evaluated, fs=fs, reference=reference),
        "all_problems": _track(
            prediction | public,
            all_domain,
            evaluated | public if evaluated is not None else None,
            fs=fs,
            reference=reference,
        ),
    }
    return {
        "schema": "signalmind.locator-evaluation.v1",
        "method": result.get("method", "unknown"),
        "status": status,
        "failed": failed,
        "failure_reason": result.get("error", result.get("reason")) if failed else None,
        "n": n,
        "fs": float(fs),
        "duration_s": n / fs,
        "candidate_event_count": len(events),
        "reference_available": reference is not None,
        "source_unavailable_samples": int((~source).sum()),
        "source_unavailable_s": float((~source).sum() / fs),
        "public_flag_samples": int(public.sum()),
        "public_flag_s": float(public.sum() / fs),
        "public_flag_ranges": [list(pair) for pair in _ranges(public)],
        "coverage_known": coverage_known,
        "coverage": {
            "known": coverage_known,
            "algorithm_evaluated_samples": int(evaluated.sum()) if evaluated is not None else None,
            "algorithm_evaluated_s": float(evaluated.sum() / fs) if evaluated is not None else None,
            "unknown_observed_fraction": tracks["unknown_observed"]["coverage_fraction"],
            "all_problems_fraction": tracks["all_problems"]["coverage_fraction"],
        },
        "tracks": tracks,
        "runtime": dict(result.get("runtime", {})),
        "definitions": {
            "primary_track": "unknown_observed",
            "unknown_observed": "D & source_support & finite(input) & ~public_missing_or_device_flags",
            "all_problems": "D & source_support; prediction union common public missing/device flags",
            "source_exclusions": "Outside both waveform-corruption domains; never added to A",
            "primary_event_iou": 0.3,
            "failed_output": "Algorithm alarms discarded; targets remain in the fixed denominator",
        },
    }


def _mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _summarize_track(tracks):
    labeled = [t for t in tracks if t["reference_available"]]
    micro = {name: sum(t[name] for t in tracks) for name in ("domain_s", "alarm_s", "alarm_regions")}
    known = [t for t in tracks if t["evaluated_s"] is not None]
    micro["evaluated_s"] = sum(t["evaluated_s"] for t in known) if len(known) == len(tracks) else None
    micro["coverage_fraction"] = (
        _ratio(micro["evaluated_s"], micro["domain_s"]) if micro["evaluated_s"] is not None else None
    )
    micro["alarm_fraction"] = _ratio(micro["alarm_s"], micro["domain_s"])
    metrics = ("precision", "recall", "f1", "iou", "normal_fp_ratio", "normal_event_fp_ratio")
    macro = {name: _mean(t.get(name) for t in labeled) for name in metrics}
    denominators = {name: sum(t.get(name) is not None for t in labeled) for name in metrics}
    if labeled:
        for name in ("tp_s", "fp_s", "fn_s", "tn_s", "truth_s", "normal_s", "normal_event_s", "normal_event_fp_s"):
            micro[name] = sum(t[name] for t in labeled)
        micro.update(_classification(micro["tp_s"], micro["fp_s"], micro["fn_s"], micro["tn_s"]))
        micro["normal_event_fp_ratio"] = _ratio(micro["normal_event_fp_s"], micro["normal_event_s"])
        micro["false_alarm_regions"] = sum(t["false_alarm_regions"] for t in labeled)
        labeled_duration = sum(t["domain_s"] for t in labeled)
        micro["false_alarm_regions_per_hour"] = _ratio(micro["false_alarm_regions"] * 3600, labeled_duration)
    else:
        micro.update({name: None for name in metrics})
    event_summary = {}
    for threshold in IOU_THRESHOLDS:
        rows = [t["events"][str(threshold)] for t in labeled]
        tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        matches = [m for row in rows for m in row["matches"]]
        event_summary[str(threshold)] = {
            "tp": tp if labeled else None,
            "fp": fp if labeled else None,
            "fn": fn if labeled else None,
            "precision": _ratio(tp, tp + fp) if labeled else None,
            "recall": _ratio(tp, tp + fn) if labeled else None,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn) if tp + fn else None,
            "onset_error_median_s": _percentile([m["onset_error_s"] for m in matches], 50),
            "onset_error_p90_s": _percentile([m["onset_error_s"] for m in matches], 90),
            "end_error_median_s": _percentile([m["end_error_s"] for m in matches], 50),
            "end_error_p90_s": _percentile([m["end_error_s"] for m in matches], 90),
        }
    return {
        "reference_rows": len(labeled),
        "normal_only_rows": sum(t["normal_only"] for t in labeled),
        "known_coverage_rows": len(known),
        "unknown_coverage_rows": len(tracks) - len(known),
        "micro": micro,
        "macro": macro,
        "macro_denominators": denominators,
        "events": event_summary,
    }


def _summary(rows):
    tracks = {
        name: _summarize_track([r["tracks"][name] for r in rows]) for name in ("unknown_observed", "all_problems")
    }
    cluster_rows = defaultdict(list)
    for row in rows:
        if row.get("source_cluster") is not None:
            cluster_rows[str(row["source_cluster"])].append(row)
    cluster_macro = {}
    for name in tracks:
        cluster_summaries = [
            _summarize_track([r["tracks"][name] for r in group])["micro"] for group in cluster_rows.values()
        ]
        cluster_macro[name] = {
            metric: _mean(s.get(metric) for s in cluster_summaries)
            for metric in ("precision", "recall", "f1", "iou", "normal_fp_ratio", "normal_event_fp_ratio")
        }
    algorithm_times = [r.get("runtime", {}).get("algorithm_s") for r in rows]
    algorithm_times = [v for v in algorithm_times if v is not None]
    load_times = [r.get("runtime", {}).get("load_s") for r in rows]
    peak_rss = [r.get("runtime", {}).get("peak_rss_mb") for r in rows]
    peak_rss = [v for v in peak_rss if v is not None]
    return {
        "rows": len(rows),
        "case_count": len({r.get("case_id", i) for i, r in enumerate(rows)}),
        "source_cluster_count": len(cluster_rows),
        "source_cluster_missing_rows": sum(r.get("source_cluster") is None for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "failure_count": sum(r["failed"] for r in rows),
        "duration_s": sum(r["duration_s"] for r in rows),
        "source_unavailable_s": sum(r["source_unavailable_s"] for r in rows),
        "tracks": tracks,
        "cluster_macro": cluster_macro,
        "runtime": {
            "algorithm_total_s": sum(algorithm_times),
            "algorithm_measured_rows": len(algorithm_times),
            "algorithm_median_s": _percentile(algorithm_times, 50),
            "algorithm_p90_s": _percentile(algorithm_times, 90),
            "load_total_s": sum(v for v in load_times if v is not None),
            "peak_rss_max_mb": max(peak_rss) if peak_rss else None,
        },
    }


def aggregate_rows(rows):
    """Summarize case results; row metadata supplies channel/family/source_cluster.

    Seconds, not sample counts, are pooled across differing sampling rates.
    Cluster macro means weight source clusters equally; no sampling point is
    treated as an independent person and no significance test is performed.
    """
    rows = list(rows)
    grouped = defaultdict(list)
    by_channel, by_family, by_cluster = (
        defaultdict(lambda: defaultdict(list)),
        defaultdict(lambda: defaultdict(list)),
        defaultdict(lambda: defaultdict(list)),
    )
    for row in rows:
        method = row["method"]
        grouped[method].append(row)
        by_channel[str(row.get("channel", row.get("task", "unknown")))][method].append(row)
        by_family[str(row.get("family", "unknown"))][method].append(row)
        by_cluster[str(row.get("source_cluster", "unknown"))][method].append(row)

    def nested(groups):
        return {
            key: {method: _summary(group) for method, group in sorted(methods.items())}
            for key, methods in sorted(groups.items())
        }

    return {
        "schema": "signalmind.locator-comparison-summary.v1",
        "row_count": len(rows),
        "by_method": {method: _summary(group) for method, group in sorted(grouped.items())},
        "by_channel": nested(by_channel),
        "by_family": nested(by_family),
        "by_source_cluster": nested(by_cluster),
        "interpretation": {
            "micro": "Pooled time in seconds across cases, retaining failures and unscored targets",
            "macro": "Case means with explicit per-metric defined-value denominators",
            "cluster_macro": "Equal-weight means of each source cluster's pooled-time metric",
            "control_cases": "Normal-only F1/IoU undefined; false alarms and normal-event alarms are reported",
            "causal_claim": "No claim that an alarm identifies the true cause or that a waveform was repaired",
            "inference": "Descriptive development-set comparison; no significance test",
        },
    }
