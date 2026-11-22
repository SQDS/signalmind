"""Peak review must separate marker edits, waveform values, rejection and missing truth."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.grade_peak_review import grade_case, summarize  # noqa: E402
from eval.run_peak_review import ReviewTools  # noqa: E402
from tsmcp.peak_policy import SCHEMA, apply, validate  # noqa: E402


def test_peak_refinement_changes_only_marker_coordinates():
    t = np.arange(1250)
    centers = np.arange(100, 1200, 125)
    x = sum(np.exp(-0.5 * ((t - p) / 2) ** 2) for p in centers)
    original = x.copy()
    result = apply(x, centers + 4, validate({"schema_version": SCHEMA, "alignment": "positive", "radius_ms": 64}))
    assert result["new_peak_samples"] == centers.tolist()
    np.testing.assert_array_equal(x, original)
    assert not result["waveform_values_modified"] and not result["exclude"]


def test_colliding_refinements_keep_one_marker_not_one_waveform_sample():
    x = np.zeros(1250)
    x[500] = 2
    result = apply(x, np.array([496, 504]), {"schema_version": SCHEMA, "alignment": "positive", "radius_ms": 80})
    assert result["new_peak_samples"].count(500) == 1
    assert result["new_peak_samples"].count(-1) == 1
    assert x[500] == 2


def fixture():
    return {"disposition": "CORRECTED", "target_peak_samples": [100, -1, 300], "subject_key": "fixture"}, [96, 200, 300]


def test_grading_counts_deletion_movement_and_unnecessary_changes_separately():
    truth, old = fixture()
    score = grade_case(truth, old, {"exclude": False, "new_peak_samples": [100, -1, 302]})
    assert score["delete_tp"] == 1 and score["delete_fp"] == 0
    assert score["moves_correct_within_1_samples"] == 1
    assert score["unneeded_edits"] == 1
    assert score["retained_windows_end_to_end_8ms"] == 0


def test_rejecting_a_retained_window_cannot_hide_its_peak_errors():
    truth, old = fixture()
    result = grade_case(truth, old, {"exclude": True, "new_peak_samples": [-1, -1, -1]})
    assert result["reference_deletions"] == 1 and result["delete_fp"] == 2
    assert result["located_within_1_samples"] == 0 and result["retained_windows_end_to_end_8ms"] == 0


def test_unresolved_exclusion_has_no_per_peak_gold_and_is_reported_separately():
    truth = {"disposition": "EXCLUDED", "exclusion_subtype": "需人工复核", "target_peak_samples": None}
    result = grade_case(truth, [10, 50], {"exclude": True, "new_peak_samples": [10, 50]})
    assert result["unresolved_exclusions_found"] == 1
    assert "reference_deletions" not in result


def test_missing_delivery_is_not_treated_as_unchanged_correct_output():
    truth, old = fixture()
    result = grade_case(truth, old, {})
    assert result["delivered"] == result["located_within_1_samples"] == result["window_agreement"] == 0
    assert result["delete_fn"] == 1
    summary = summarize([{"subject_key": "fixture", "counts": result}])
    assert summary["retained_window_end_to_end_8ms"] == 0


def test_duplicate_or_out_of_bounds_marker_output_is_invalid():
    truth, old = fixture()
    for new in ([100, 100, 300], [100, -2, 300], [100, -1, 7500]):
        assert grade_case(truth, old, {"exclude": False, "new_peak_samples": new})["delivered"] == 0
    with pytest.raises(ValueError):
        validate({"schema_version": SCHEMA, "alignment": "positive", "radius_ms": 1000})


def test_unchanged_coordinate_does_not_get_credit_for_requested_small_move():
    truth = {"disposition": "CORRECTED", "target_peak_samples": [101]}
    result = grade_case(truth, [100], {"exclude": False, "new_peak_samples": [100]})
    assert result["located_within_1_samples"] == 1
    assert result["moves_correct_within_1_samples"] == 0


def test_inspection_cannot_access_unlisted_windows_or_labels(tmp_path):
    public, output = tmp_path / "public", tmp_path / "output"
    public.mkdir()
    (public / "index.json").write_text(
        json.dumps({"development_examples": ["visible"], "cases": [], "units": {"ecg": "source_units", "abp": "mmHg"}})
    )
    tools = ReviewTools(public, output)
    for uid in ("hidden", "../private/truth.json"):
        blocks, error = asyncio.run(tools.call("review_inspect", {"id": uid}))
        assert error and "Only the listed" in blocks[0]["text"]
