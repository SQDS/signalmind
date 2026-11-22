"""Second-pass correction must use local evidence while preserving original observations and identity."""

import copy

import numpy as np

from tsmcp.peak_policy import SCHEMA, apply
from tsmcp.peak_refinement import refine


def biphasic():
    t = np.arange(2500)
    centers = np.arange(150, 2400, 125)
    x = sum(np.exp(-0.5 * ((t - p) / 2) ** 2) - 1.3 * np.exp(-0.5 * ((t - p - 8) / 2) ** 2) for p in centers)
    original = centers.copy()
    original[::3] += 8
    return x, original, centers


def test_align_before_morphology_restores_same_beat_wrongly_removed_at_s_trough():
    x, old, centers = biphasic()
    policy = {
        "schema_version": SCHEMA,
        "alignment": "positive",
        "radius_ms": 80,
        "min_template_correlation": 0,
        "refractory_ms": 200,
    }
    batch = apply(x, old, policy)
    assert batch["deleted_count"] > 0
    saved = copy.deepcopy(batch)
    original_x = x.copy()
    result = refine(x, old, batch, policy)
    assert result["new_peak_samples"] == centers.tolist()
    assert any(r["operation"] == "restore" for r in result["refinement_operations"])
    assert batch == saved
    np.testing.assert_array_equal(x, original_x)


def test_distinct_polarity_is_not_itself_a_deletion_reason():
    t = np.arange(2500)
    centers = np.arange(150, 2400, 125)
    x = sum((-1 if i % 4 == 0 else 1) * np.exp(-0.5 * ((t - p) / 2) ** 2) for i, p in enumerate(centers))
    policy = {"schema_version": SCHEMA, "alignment": "none", "min_template_correlation": 0}
    batch = apply(x, centers, policy)
    assert batch["deleted_count"] > 0
    result = refine(x, centers, batch, policy)
    assert result["new_peak_samples"] == centers.tolist()


def test_weak_unmatched_extra_candidate_can_remain_deleted():
    t = np.arange(2500)
    centers = np.arange(150, 2400, 125)
    extra = 705
    x = sum(np.exp(-0.5 * ((t - p) / 2) ** 2) for p in centers) + 0.08 * np.exp(-0.5 * ((t - extra) / 10) ** 2)
    old = np.sort(np.r_[centers, extra])
    policy = {"schema_version": SCHEMA, "alignment": "positive", "min_energy_fraction": 0.5}
    batch = apply(x, old, policy)
    result = refine(x, old, batch, policy)
    assert result["new_peak_samples"][int(np.flatnonzero(old == extra)[0])] == -1
    assert sum(v >= 0 for v in result["new_peak_samples"]) == len(centers)


def test_second_pass_is_coordinate_idempotent_and_does_not_reverse_window_exclusion():
    x, old, _ = biphasic()
    policy = {"schema_version": SCHEMA, "alignment": "positive", "min_template_correlation": 0}
    batch = apply(x, old, policy)
    batch["exclude"] = True
    batch["exclude_reasons"] = [{"feature": "manual_fixture", "op": "gt", "value": 0}]
    first = refine(x, old, batch, policy)
    second = refine(x, old, first, policy)
    assert first["new_peak_samples"] == second["new_peak_samples"]
    assert not second["refinement_operations"]
    assert first["exclude"] and second["exclude"]
    assert not first["window_decision_revised"]


def test_refinement_merges_same_coordinate_without_changing_signal_samples():
    x = np.zeros(1250)
    x[500] = 2
    old = np.array([496, 504])
    policy = {"schema_version": SCHEMA, "alignment": "positive", "radius_ms": 80}
    result = refine(x, old, apply(x, old, policy), policy)
    assert result["new_peak_samples"].count(500) == 1
    assert result["new_peak_samples"].count(-1) == 1
    assert x[500] == 2


def test_distinct_nearby_candidates_are_flagged_instead_of_merged_by_distance_alone():
    t = np.arange(1250)
    old = np.array([150, 275, 400, 525, 544, 775, 900, 1025])
    x = sum(np.exp(-0.5 * ((t - p) / 2) ** 2) for p in old)
    policy = {"schema_version": SCHEMA, "alignment": "positive", "radius_ms": 80, "refractory_ms": 200}
    batch = apply(x, old, policy)
    result = refine(x, old, batch, policy)
    assert 525 in result["new_peak_samples"] and 544 in result["new_peak_samples"]
    assert any(r["reason"] == "nearby_distinct_candidates" for r in result["review_queue"])


def test_incomplete_context_restores_candidate_for_review_not_as_confirmed_beat():
    x, old, _ = biphasic()
    policy = {"schema_version": SCHEMA, "alignment": "none"}
    batch = apply(x, old, policy)
    i = 3
    batch["new_peak_samples"][i] = -1
    x[old[i] - 2 : old[i] + 3] = np.nan
    result = refine(x, old, batch, policy)
    assert result["new_peak_samples"][i] == int(old[i])
    assert any(r["marker_index0"] == i and r["reason"] == "incomplete_context" for r in result["review_queue"])
