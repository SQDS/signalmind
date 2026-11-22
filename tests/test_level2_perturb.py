"""Synthetic transformation invariants; no dataset or subject partition is read."""

import numpy as np
import pytest

from eval import level2_perturb as perturb


@pytest.fixture(autouse=True)
def isolate_transformation_from_native_quality(monkeypatch):
    """The sine fixtures test transformations, not clinical ECG eligibility."""
    monkeypatch.setattr(perturb, "g2_pulse_rate", lambda *args: {"accepted": True, "difference_bpm": 0.0})
    monkeypatch.setattr(perturb, "native_lag_support", lambda *args, **kwargs: {"accepted": True})


def record(duration=180, period_samples=100):
    fs = 125.0
    t = np.arange(round(duration * fs))
    phase = 2 * np.pi * t / period_samples
    x = np.column_stack((np.sin(phase), 90 + 25 * np.sin(phase - 0.6), np.sin(phase - 1.2)))
    return {
        "dataset": "synthetic",
        "subject_key": "synthetic-fixture",
        "record_key": "synthetic-fixture",
        "source_hashes": {},
        "start_s": 0.0,
        "x": x.astype(np.float32),
        "fs": fs,
        "duration_s": duration,
        "channels": ["ECG", "ABP", "PPG"],
        "units": ["mV", "mmHg", "a.u."],
    }


@pytest.mark.parametrize("strength", range(4))
def test_attenuation_has_registered_bias_preserves_prefix_and_other_channels(strength):
    source = record()
    before = source["x"].copy()
    item = perturb.make_candidate(source, "F2", strength, {})
    parameters = item["perturbation"]
    start = parameters["onset_sample"]
    np.testing.assert_array_equal(source["x"], before)
    np.testing.assert_array_equal(item["x"][:, [0, 2]], before[:, [0, 2]])
    np.testing.assert_array_equal(item["x"][: start + 1, 1], before[: start + 1, 1])
    assert parameters["target_peak_bias_mmhg"] == (3, 5, 8, 12)[strength]
    assert abs(parameters["measured_peak_bias_mmhg"] - parameters["target_peak_bias_mmhg"]) < 1e-3
    assert abs(parameters["independent_peak_bias_mmhg"] - parameters["measured_peak_bias_mmhg"]) < 1e-5
    assert 20 <= item["expected_answer"]["onset_s"] < source["duration_s"] - 35


@pytest.mark.parametrize("index", range(6))
def test_signed_shift_matches_exact_source_samples_and_changes_only_target_channel(index):
    source = record()
    item = perturb.make_candidate(source, "F3", index, {})
    j = source["channels"].index(item["expected_answer"]["channel"])
    parameters = item["perturbation"]
    start, shift = parameters["onset_sample"], parameters["shift_samples"]
    expected_indices = np.arange(len(source["x"]))
    expected_indices[start:] -= shift
    expected_indices = expected_indices.clip(0, len(source["x"]) - 1)
    np.testing.assert_array_equal(item["x"][:, j], source["x"][expected_indices, j])
    for other in set(range(3)) - {j}:
        np.testing.assert_array_equal(item["x"][:, other], source["x"][:, other])
    assert item["expected_answer"]["shift_ms"] == 1000 * shift / source["fs"]


@pytest.mark.parametrize("index", [6, 7])
def test_ten_minute_clock_drift_has_exact_endpoint_and_fractional_interior(index):
    source = record(duration=600)
    item = perturb.make_candidate(source, "F3", index, {})
    j = source["channels"].index(item["expected_answer"]["channel"])
    shift = item["perturbation"]["shift_samples"]
    n = len(item["x"])
    probes = np.asarray([0, n // 4, n // 2, n - 1])
    locations = np.clip(probes * (1 - shift / (n - 1)), 0, n - 1)
    expected = np.interp(locations, np.arange(n), source["x"][:, j])
    np.testing.assert_allclose(item["x"][probes, j], expected, atol=1e-5)
    assert item["source_duration_s"] == 600
    assert item["expected_answer"]["drift_ms"] == 1000 * shift / source["fs"]
    with pytest.raises(ValueError, match="insufficient strictly continuous duration"):
        perturb.make_candidate(record(duration=180), "F3", index, {})


def test_missing_pulse_truth_counts_cardiac_cycles_not_refractory_filtered_peaks():
    source = record(period_samples=40)
    n = len(source["x"])
    samples = np.arange(n)
    # A clean 187.5 bpm cardiac train is inside the native ECG RR validity range.
    # The other channels preserve the true count when one pulse train is erased.
    for channel, offset, amplitude, baseline in ((0, 5, 1, 0), (1, 10, 25, 90), (2, 20, 1, 0)):
        distance = (samples - offset + 20) % 40 - 20
        source["x"][:, channel] = baseline + amplitude * np.exp(-0.5 * (distance / 3) ** 2)
    native_pulses = np.arange(20, n, 40)
    item = perturb.make_candidate(source, "F3", 8, {})
    bounds = item["perturbation"]["bounds_samples"]
    affected = native_pulses[(native_pulses >= bounds[0]) & (native_pulses < bounds[1])]
    missing = sum(np.max(item["x"][pulse - 3 : pulse + 4, 2]) < 0.5 for pulse in affected)
    assert missing == item["expected_answer"]["count"]


@pytest.mark.parametrize("index", [10, 11])
def test_polarity_inversion_is_exact_reflection_only_after_onset(index):
    source = record()
    item = perturb.make_candidate(source, "F3", index, {})
    j = source["channels"].index(item["expected_answer"]["channel"])
    parameters = item["perturbation"]
    start = parameters["onset_sample"]
    np.testing.assert_array_equal(item["x"][:start], source["x"][:start])
    np.testing.assert_allclose(item["x"][start:, j], 2 * parameters["center"] - source["x"][start:, j])
    for other in set(range(3)) - {j}:
        np.testing.assert_array_equal(item["x"][:, other], source["x"][:, other])


@pytest.mark.parametrize("index", range(4, 8))
def test_sustained_change_and_transient_have_separate_support(index):
    source = record()
    item = perturb.make_candidate(source, "F6", index, {})
    parameters = item["perturbation"]
    start = parameters["onset_sample"]
    a = round(parameters["distractor_s"] * source["fs"])
    b = a + round(parameters["distractor_duration_s"] * source["fs"])
    expected = np.zeros(len(item["x"]))
    expected[start:] = parameters["displacement_mmhg"]
    observed = item["x"][:, 1] - source["x"][:, 1]
    outside = np.ones(len(observed), dtype=bool)
    outside[a:b] = False
    np.testing.assert_allclose(observed[outside], expected[outside], atol=2e-5)
    np.testing.assert_array_equal(item["x"][:, [0, 2]], source["x"][:, [0, 2]])
    assert abs(parameters["distractor_s"] - item["expected_answer"]["onset_s"]) > 20
    assert item["grounding_rubric"]["verification"]["detects_misleading_output"]


def test_seed_reproduces_intervention_without_mutating_source():
    source = record()
    before = source["x"].copy()
    first = perturb.make_candidate(source, "F3", 2, {})
    second = perturb.make_candidate(source, "F3", 2, {})
    np.testing.assert_array_equal(first["x"], second["x"])
    np.testing.assert_array_equal(source["x"], before)
    assert first["perturbation"] == second["perturbation"]
    assert first["expected_answer"] == second["expected_answer"]


def test_rejected_native_cross_channel_support_cannot_construct_shift(monkeypatch):
    monkeypatch.setattr(perturb, "g2_pulse_rate", lambda *args: {"accepted": False})
    with pytest.raises(ValueError, match="complete-window native cross-channel"):
        perturb.make_candidate(record(), "F3", 0, {})


def test_rejected_native_lag_stability_cannot_construct_shift(monkeypatch):
    monkeypatch.setattr(perturb, "native_lag_support", lambda *args, **kwargs: {"accepted": False})
    with pytest.raises(ValueError, match="native physiological delay"):
        perturb.make_candidate(record(), "F3", 0, {})


def test_long_drift_checks_the_entire_tail_and_every_minute(monkeypatch):
    source = record(duration=600)
    source["x"][-1, 2] = np.nan
    with pytest.raises(ValueError, match="nonfinite samples within complete task window"):
        perturb.make_candidate(source, "F3", 6, {})
    calls = []

    def support(ecg, pulse, fs):
        calls.append((len(ecg), len(pulse), fs))
        return {"accepted": True, "difference_bpm": 0.0}

    monkeypatch.setattr(perturb, "g2_pulse_rate", support)
    perturb.make_candidate(record(duration=600), "F3", 6, {})
    assert calls == [(7500, 7500, 125.0)] * 20


def test_onset_retry_supports_native_ecg_and_abp_without_fabricating_a_pulse_channel():
    source = record()
    source["x"] = source["x"][:, :2]
    source["channels"] = source["channels"][:2]
    source["units"] = source["units"][:2]
    item = perturb.make_candidate(source, "F6", 4, {})
    assert item["channels"] == ["ECG", "ABP"]
    assert item["tolerances"] == {"onset_s": 5.0}
    np.testing.assert_array_equal(item["x"][:, 0], source["x"][:, 0])


def test_retry_schedule_evaluates_every_registered_slot_and_records_failed_attempts(monkeypatch):
    monkeypatch.setattr(perturb, "RETRY_MAX_ATTEMPTS_PER_SLOT", 2)
    monkeypatch.setattr(perturb, "_native_quality", lambda source: {})
    visited = []

    def sources():
        for i in range(5):
            visited.append(i)
            source = record(duration=1)
            source.update(subject_key=f"synthetic-subject-{i}", record_key=f"synthetic-record-{i}")
            yield source

    def candidate(source, family, index, quality):
        if family == "F3" and index == 6:
            raise ValueError("synthetic reference rejection")
        return {"family": family, "subject_key": source["subject_key"]}

    monkeypatch.setattr(perturb, "make_candidate", candidate)
    rows, exclusions = perturb._retry_candidates(sources(), {"F3": set(), "F6": set()})
    assert visited == [0, 1]
    assert len(rows) == 10
    rejected = [row for row in exclusions if row["stage"] == "ground_truth"]
    assert len(rejected) == 2
    assert {row["retry_attempt"] for row in rejected} == {1, 2}
    summaries = [row for row in exclusions if row["stage"] == "retry_schedule_summary"]
    assert len(summaries) == 6
    assert all(row["attempted"] == 2 for row in summaries)
    assert sum(row["reference_eligible"] for row in summaries) == len(rows)
    assert sum(row["reference_rejected"] for row in summaries) == len(rejected)
    assert len({row["retry_rank"] for row in rows + rejected}) == 12
    assert all(row["retry_candidate"] for row in rows + rejected)


def test_retry_native_quality_rejections_are_logged_for_all_slots(monkeypatch):
    monkeypatch.setattr(perturb, "ecg_reference", lambda *args: {"accepted": False})

    def must_not_construct(*args):
        raise AssertionError("Rejected native signals must not reach intervention construction")

    monkeypatch.setattr(perturb, "make_candidate", must_not_construct)
    rows, exclusions = perturb._retry_candidates([record()], {"F3": set(), "F6": set()})
    assert rows == []
    rejected = [row for row in exclusions if row["stage"] == "native_reference"]
    assert {(row["family"], row["design_slot"]) for row in rejected} == {
        (family, index) for family, indices in perturb.RETRY_SLOTS.items() for index in indices
    }


def test_retry_channel_skips_are_bounded_without_consuming_slot_attempts(monkeypatch):
    monkeypatch.setattr(perturb, "RETRY_MAX_SOURCE_RECORDS", 2)
    visited = []

    def sources():
        for i in range(3):
            visited.append(i)
            source = record(duration=1)
            source["channels"] = ["ECG"]
            yield source

    rows, exclusions = perturb._retry_candidates(sources(), {"F3": set(), "F6": set()})
    assert rows == []
    assert visited == [0, 1]
    assert len([row for row in exclusions if row["stage"] == "retry_source_design"]) == 4
    assert all(row["attempted"] == 0 for row in exclusions if row["stage"] == "retry_schedule_summary")


def test_retry_source_allocation_excludes_original_subjects_and_duplicate_records(monkeypatch):
    monkeypatch.setattr(perturb, "_native_quality", lambda source: {})
    monkeypatch.setattr(perturb, "make_candidate", lambda source, family, index, quality: {"family": family})
    source = record(duration=1)
    rows, exclusions = perturb._retry_candidates([source, source], {"F3": {source["subject_key"]}, "F6": set()})
    assert len(rows) == 4
    assert all(row["family"] == "F6" for row in rows)
    assert len([row for row in exclusions if row["stage"] == "retry_source_design"]) == 3
