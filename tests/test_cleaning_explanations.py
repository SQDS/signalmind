"""Reason provenance and fixed-support retest semantics, without model calls."""

import numpy as np
import pytest

from eval.explain_cleaning_results import component_evidence, reconstruct, retest, verify_calls


def test_abp_budget_difference_is_explained_without_changing_data():
    x = np.arange(1250, dtype=float) * 0.01
    for index in (100, 200, 300, 400):
        x[index:] += 20
    valid = np.ones(len(x), bool)
    old, skipped = reconstruct("abp_v1", x, valid, x, {"issues": []}, 125)
    assert old == [] and skipped[0]["parameters"]["candidate_samples"] == 32
    assert skipped[0]["parameters"]["allowed_samples"] == 25
    new, skipped = reconstruct("abp_v2", x, valid, x, {"issues": []}, 125)
    assert not skipped and len(new) == 1
    assert new[0]["ranges_samples"] == [[96, 104], [196, 204], [296, 304], [396, 404]]
    evidence = component_evidence(new[0]["components"][0], [96, 104], x, x)
    assert evidence["trigger_pair_count"] == 1
    assert evidence["maximum_trigger_delta"] == pytest.approx(20.01)


def test_ppg_jump_padding_and_threshold_match_executed_convention():
    t = np.arange(6000) / 125
    x = 40 + np.sin(2 * np.pi * t)
    x[1500] += 30
    calls, skipped = reconstruct("ppg", x, np.ones(len(x), bool), x, {"issues": []}, 125)
    assert not skipped
    marking = next(c for c in calls if c["op"] == "mark_invalid")
    assert marking["ranges_samples"] == [[1468, 1532]]
    parameters = marking["components"][0]["parameters"]
    assert parameters["padding_samples"] == 31
    assert parameters["jump_threshold"] == pytest.approx(parameters["robust_span"] * 0.5)


def test_unverified_source_and_mismatched_ranges_are_not_given_reasons():
    x = np.zeros(100)
    with pytest.raises(ValueError, match="exact recipe"):
        reconstruct("unknown_source_hash", x, np.ones(100, bool), x, {}, 125)
    expected = [{"op": "mark_invalid", "ranges_samples": [[10, 20]], "parameters": {}}]
    with pytest.raises(ValueError, match="ranges"):
        verify_calls(expected, [{"op": "mark_invalid", "ranges_samples": [[11, 20]], "parameters": {}}])


def test_masking_does_not_become_repair_or_numeric_improvement():
    x = np.arange(1000, dtype=float)
    before_valid = np.ones(len(x), bool)
    after_valid = before_valid.copy()
    after_valid[400:410] = False
    before = {"events": [], "intervals": [], "pressures": [], "measurement_support": [[0, 1000]]}
    after = {**before, "measurement_support": [[0, 400], [410, 1000]]}
    result = retest([400, 410], x, x.copy(), before_valid, after_valid, np.zeros(len(x), bool), before, after, 125)
    assert result["changed_samples"] == 0
    assert result["max_absolute_value_change"] == 0
    assert result["eligible_samples_before"] == 10 and result["eligible_samples_after"] == 0
    assert result["edge_context_max_step_before"] == result["edge_context_max_step_after"]
    assert result["estimated_samples_after"] == 0
    assert not result["reference_quality_verified"]
