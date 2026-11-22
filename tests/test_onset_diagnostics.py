import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.onset_diagnostics import bridge_gaps, eligible, longest_diagnostic  # noqa: E402


@pytest.mark.parametrize(
    ("truth", "burden", "expected"),
    [
        ([10, 310], 50, False),
        ([10, 310.001], 89.999, True),
        ([10, 311], 90, False),
        (None, 0, False),
    ],
)
def test_eligibility_uses_strict_annotation_thresholds(truth, burden, expected):
    assert eligible({"longest_annotated": truth, "true_pct_of_window": burden}) is expected


@pytest.mark.parametrize(
    ("detected", "overlap", "iou", "signed_error", "correct"),
    [
        ([200, 300], 0, 0, 200, False),
        ([50, 150], 50, 1 / 3, 50, True),
        ([-60, 40], 40, 0.25, -60, True),
        ([60.001, 160.001], 39.999, 39.999 / 160.001, 60.001, False),
        (None, 0, 0, None, False),
    ],
)
def test_longest_diagnostic_scores_temporal_overlap_and_signed_onset(detected, overlap, iou, signed_error, correct):
    row = {
        "longest_annotated": [0, 100],
        "longest_detected": detected,
        "detected_episodes": [{"start": detected[0], "end": detected[1]}] if detected else [],
        "fragmentation": 1 if overlap else 0,
    }
    result = longest_diagnostic(row)
    # Equal duration alone must not imply the same episode or positive overlap.
    assert result["overlap_s"] == pytest.approx(overlap)
    assert result["iou"] == pytest.approx(iou)
    assert result["signed_onset_error_s"] == signed_error
    assert result["onset_within_60s"] is correct
    assert result["fragmentation"] == row["fragmentation"]


def test_bridge_gaps_separates_annotation_truth_from_detector_readability():
    row = {
        "longest_detected": [0, 100],
        "support_intervals": [[-10, 10], [40, 60], [89, 110]],
        "annotated_af_intervals": [[10, 25], [60, 89]],
        "readable_intervals": [[10, 20], [30, 40], [70, 89]],
    }
    first, second = bridge_gaps(row)
    assert (first["start_s"], first["end_s"], first["duration_s"]) == (10, 40, 30)
    assert first["new_bridge_vs_R"] is True  # R merges strictly less than 30 seconds.
    assert first["annotated_rhythm"] == "mixed"
    assert first["annotated_af_s"] == first["annotated_non_af_s"] == 15
    assert first["readability"] == "mixed"
    assert first["unknown_s"] == 10
    assert first["readable_non_af_gap_s"] == 20
    assert first["readable_annotated_af_s"] == first["readable_annotated_non_af_s"] == 10
    assert second["duration_s"] == 29
    assert second["new_bridge_vs_R"] is False
    assert second["annotated_rhythm"] == "AF"
    assert second["readable_non_af_gap_s"] == second["readable_annotated_af_s"] == 19
    assert second["readable_annotated_non_af_s"] == 0
    for gap in (first, second):
        assert gap["annotated_af_s"] + gap["annotated_non_af_s"] == gap["duration_s"]
        assert gap["unknown_s"] + gap["readable_non_af_gap_s"] == gap["duration_s"]
        assert gap["readable_annotated_af_s"] + gap["readable_annotated_non_af_s"] == gap["readable_non_af_gap_s"]


@pytest.mark.parametrize(
    ("readable", "classification", "unknown"),
    [([], "unknown", 30), ([[10, 40]], "readable_non_AF_support", 0)],
)
def test_non_af_bridge_can_be_unknown_or_readable(readable, classification, unknown):
    row = {
        "longest_detected": [0, 50],
        "support_intervals": [[0, 10], [40, 50]],
        "annotated_af_intervals": [],
        "readable_intervals": readable,
    }
    (gap,) = bridge_gaps(row)
    assert gap["annotated_rhythm"] == "non-AF"
    assert gap["readability"] == classification
    assert gap["unknown_s"] == unknown
    assert gap["readable_annotated_non_af_s"] == 30 - unknown


def test_no_candidate_has_no_bridges_and_exact_120_seconds_is_rejected():
    assert bridge_gaps({"longest_detected": None}) == []
    with pytest.raises(AssertionError):
        bridge_gaps(
            {
                "longest_detected": [0, 140],
                "support_intervals": [[0, 10], [130, 140]],
                "annotated_af_intervals": [],
                "readable_intervals": [],
            }
        )
