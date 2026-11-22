"""Scoring contracts: fixed domains, one-to-one events and missing references."""

import numpy as np
import pytest

from tsmcp.locators.evaluation import aggregate_rows, match_intervals, score_result


def reference(n, intervals=(), protected=()):
    a, e = np.zeros(n, bool), np.zeros(n, bool)
    for start, end in intervals:
        a[start:end] = True
    for start, end in protected:
        e[start:end] = True
    return {"A": a, "D": np.ones(n, bool), "C": ~a, "E": e}


def result(n, intervals=(), **kwargs):
    return {
        "method": "fixture",
        "status": "ok",
        "events": [{"start": a, "end": b} for a, b in intervals],
        "evaluated_ranges": [[0, n]],
        "coverage_known": True,
        **kwargs,
    }


def test_exact_detection_and_half_open_boundaries():
    row = score_result(result(10, [(2, 5)]), input_values=np.ones(10), fs=2, reference=reference(10, [(2, 5)]))
    track = row["tracks"]["unknown_observed"]
    assert track["tp_samples"] == 3 and track["alarm_s"] == 1.5
    assert track["f1"] == 1 and track["iou"] == 1
    assert track["events"]["0.3"]["onset_error_median_s"] == 0


def test_failures_keep_all_positive_time_in_denominator():
    # Even an invalid failed payload cannot claim that every sample was detected.
    row = score_result(
        result(10, [(0, 10)], status="timeout"), input_values=np.ones(10), fs=2, reference=reference(10, [(2, 5)])
    )
    track = row["tracks"]["unknown_observed"]
    assert track["fn_samples"] == 3 and track["f1"] == 0 and track["recall"] == 0
    assert track["coverage_fraction"] == 0 and row["failed"]


def test_unscored_regions_are_not_removed_from_recall():
    row = score_result(
        result(10, [(2, 4)], status="partial", evaluated_ranges=[[0, 5]]),
        input_values=np.ones(10),
        fs=1,
        reference=reference(10, [(2, 4), (7, 9)]),
    )
    track = row["tracks"]["unknown_observed"]
    assert track["recall"] == 0.5 and track["fn_samples"] == 2
    assert track["coverage_fraction"] == 0.5


def test_unknown_coverage_stays_unknown():
    row = score_result(result(8, coverage_known=False), input_values=np.ones(8), fs=1, reference=reference(8))
    assert row["coverage"]["algorithm_evaluated_s"] is None
    assert row["tracks"]["all_problems"]["coverage_fraction"] is None
    assert row["tracks"]["unknown_observed"]["f1"] is None


def test_normal_controls_never_receive_artificial_perfect_f1():
    for predicted in ([], [(2, 4)]):
        row = score_result(result(10, predicted), input_values=np.ones(10), fs=1, reference=reference(10))
        track = row["tracks"]["unknown_observed"]
        assert track["normal_only"] and track["f1"] is None and track["iou"] is None
        assert track["normal_fp_ratio"] == (0.2 if predicted else 0.0)


def test_public_flags_shared_but_not_credited_to_unknown_detection():
    x = np.ones(10)
    x[1] = np.nan
    flags = {"device_saturation_flag": np.arange(10) == 4, "observed_missing_mask": np.arange(10) == 6}
    truth = reference(10, [(1, 2), (4, 5), (6, 7), (8, 9)])
    row = score_result(result(10), input_values=x, fs=1, reference=truth, device_flags=flags)
    assert row["public_flag_ranges"] == [[1, 2], [4, 5], [6, 7]]
    main, all_problems = row["tracks"]["unknown_observed"], row["tracks"]["all_problems"]
    assert main["domain_samples"] == 7 and main["truth_samples"] == 1 and main["recall"] == 0
    assert all_problems["domain_samples"] == 10 and all_problems["tp_samples"] == 3 and all_problems["recall"] == 0.75


def test_source_exclusions_do_not_become_pollution_or_alarms():
    support = np.ones(10, bool)
    support[3:5] = False
    row = score_result(result(10), input_values=np.ones(10), fs=1, reference=reference(10), source_support=support)
    assert row["source_unavailable_s"] == 2 and row["public_flag_s"] == 0
    assert row["tracks"]["all_problems"]["domain_s"] == 8
    assert row["tracks"]["all_problems"]["alarm_s"] == 0


def test_absent_reference_returns_null_accuracy_but_retains_coverage():
    row = score_result(result(10, [(2, 4)]), input_values=np.ones(10), fs=1, reference=None)
    for track in row["tracks"].values():
        assert not track["reference_available"] and track["recall"] is None and track["f1"] is None
        assert track["alarm_s"] == 2 and track["coverage_fraction"] == 1
    summary = aggregate_rows([row])["by_method"]["fixture"]
    assert summary["tracks"]["unknown_observed"]["events"]["0.3"]["fp"] is None


def test_an_empty_scoring_domain_is_not_counted_as_normal_control():
    truth = reference(5, [(0, 5)])
    row = score_result(result(5), input_values=np.full(5, np.nan), fs=1, reference=truth)
    assert row["tracks"]["unknown_observed"]["domain_samples"] == 0
    assert not row["tracks"]["unknown_observed"]["normal_only"]


def test_matching_maximizes_count_before_iou():
    matched = match_intervals([(0, 1), (2, 14)], [(0, 10), (11, 14)], threshold=0.1)
    assert matched["tp"] == 2
    assert {(m["prediction_index"], m["reference_index"]) for m in matched["matches"]} == {(0, 0), (1, 1)}


def test_one_reference_cannot_be_matched_twice():
    matched = match_intervals([(1, 5), (6, 10)], [(0, 10)], threshold=0.3, fs=2)
    assert matched["tp"] == 1 and matched["fp"] == 1 and matched["fn"] == 0
    assert matched["end_error_median_s"] is not None


def test_unmatched_boundaries_do_not_create_zero_error():
    matched = match_intervals([(0, 2)], [(6, 8)], threshold=0.3)
    assert matched["tp"] == 0 and matched["fp"] == matched["fn"] == 1
    assert matched["onset_error_median_s"] is None


def test_normal_morphology_and_components_are_separate_support_metrics():
    truth = reference(12, [(6, 10)], protected=[(1, 3), (6, 8)])
    truth["component_00"] = np.arange(12) >= 6
    truth["component_01"] = np.zeros(12, bool)
    row = score_result(result(12, [(1, 2), (6, 8)]), input_values=np.ones(12), fs=1, reference=truth)
    track = row["tracks"]["unknown_observed"]
    assert track["normal_event_samples"] == 2 and track["normal_event_fp_ratio"] == 0.5
    assert track["components"]["component_00"]["support_coverage"] == pytest.approx(2 / 6)
    assert track["components"]["component_01"]["support_coverage"] is None


def test_domain_masks_clip_events_and_keep_source_index():
    truth = reference(10, [(2, 8)])
    truth["D"][:3] = False
    truth["D"][7:] = False
    row = score_result(result(10, [(0, 10)]), input_values=np.ones(10), fs=1, reference=truth)
    track = row["tracks"]["unknown_observed"]
    assert track["domain_samples"] == 4 and track["tp_samples"] == 4 and track["fp_samples"] == 0
    assert track["events"]["0.3"]["matches"][0]["reference"] == [3, 7]


def test_micro_metrics_pool_seconds_not_sampling_points():
    rows = []
    for fs, detected, case in [(10, True, "a"), (100, False, "b")]:
        n = fs * 10
        scored = score_result(
            result(n, [(0, fs * 2)] if detected else []),
            input_values=np.ones(n),
            fs=fs,
            reference=reference(n, [(0, fs * 2)]),
        )
        rows.append({**scored, "case_id": case, "channel": "ECG", "family": "spike", "source_cluster": case})
    summary = aggregate_rows(rows)
    method = summary["by_method"]["fixture"]
    assert method["tracks"]["unknown_observed"]["micro"]["recall"] == 0.5
    assert method["cluster_macro"]["unknown_observed"]["recall"] == 0.5
    assert method["source_cluster_count"] == 2
    assert summary["by_channel"]["ECG"]["fixture"]["rows"] == 2


def test_aggregation_retains_failed_case_and_reports_defined_denominators():
    rows = [
        score_result(result(10, [(0, 2)]), input_values=np.ones(10), fs=1, reference=reference(10, [(0, 2)])),
        score_result(result(10, status="failed"), input_values=np.ones(10), fs=1, reference=reference(10, [(0, 2)])),
        score_result(result(10), input_values=np.ones(10), fs=1, reference=reference(10)),
    ]
    summary = aggregate_rows(rows)["by_method"]["fixture"]
    assert summary["failure_count"] == 1 and summary["rows"] == 3
    assert summary["tracks"]["unknown_observed"]["macro"]["f1"] == 0.5
    assert summary["tracks"]["unknown_observed"]["macro_denominators"]["f1"] == 2
    assert summary["tracks"]["unknown_observed"]["normal_only_rows"] == 1


@pytest.mark.parametrize("interval", [(-1, 2), (2, 11), (5, 4), (2.5, 4)])
def test_invalid_event_bounds_fail_explicitly(interval):
    with pytest.raises(ValueError, match="half-open"):
        score_result(result(10, [interval]), input_values=np.ones(10), fs=1, reference=reference(10))


def test_single_channel_npz_shapes_are_accepted_without_flattening_channels():
    truth = {k: v[None, :] for k, v in reference(8, [(2, 4)]).items()}
    scored = score_result(result(8, [(2, 4)]), input_values=np.ones((1, 8)), fs=1, reference=truth)
    assert scored["tracks"]["unknown_observed"]["recall"] == 1
    with pytest.raises(ValueError, match="one channel"):
        score_result(result(8), input_values=np.ones((2, 8)), fs=1, reference=truth)
