"""Reference invariants on synthetic waveforms; no source datasets or models."""

import re

import numpy as np
import pytest

from eval import level2_physiology as ref


def test_tfa_independent_implementations_known_transfer(monkeypatch):
    rng = np.random.default_rng(61026)
    pressure = 90 + rng.normal(size=600)
    flow = 60 + 0.7 * (pressure - 90)
    first = ref.tfa_scipy(pressure, flow)

    def forbidden(*args, **kwargs):
        raise AssertionError("FFT reference must not call scipy spectral routines")

    monkeypatch.setattr(ref.signal, "welch", forbidden)
    monkeypatch.setattr(ref.signal, "csd", forbidden)
    second = ref.tfa_fft(pressure, flow)
    band = (first["frequency"] >= 0.02) & (first["frequency"] < 0.20)
    for result in (first, second):
        np.testing.assert_allclose(result["gain"][band], 0.7, atol=1e-12)
        np.testing.assert_allclose(result["phase"][band], 0, atol=1e-12)
        np.testing.assert_allclose(result["coherence"][band], 1, atol=1e-12)
    np.testing.assert_allclose(first["power"], second["power"], atol=1e-12)


def test_tfa_phase_sign_and_insufficient_excitation():
    time = np.arange(600)
    pressure = 90 + np.sin(2 * np.pi * 0.1 * time)
    flow = 60 + np.sin(2 * np.pi * 0.1 * time + 0.4)
    for implementation in (ref.tfa_scipy, ref.tfa_fft):
        result = implementation(pressure, flow)
        assert result["phase"][10] == pytest.approx(0.4)
        with pytest.raises(ValueError, match="300"):
            implementation(pressure[:299], flow[:299])
    result = ref.compare_tfa(np.full(600, 90.0), np.full(600, 60.0), 1)
    assert not result["sufficient1"]
    assert not result["sufficient2"]
    assert not result["agreement"]["accepted"]


def test_period_change_defines_equal_beat_weights_and_half_open_windows():
    # Uneven knot spacing must not accidentally time-weight the beat means.
    stream = {
        "times": np.r_[np.arange(10), 20 + np.arange(10) / 10, 21],
        "values": np.r_[np.full(10, 60), np.full(10, 69), 500],
    }
    result = ref.period_change(stream, stream, (0, 10), (20, 21), "hr_change_bpm")
    expected = {"hr_change_bpm": 9, "response_max_time_s": 20, "response_min_time_s": 20}
    assert result["reference1"] == expected
    assert result["reference2"] == expected
    assert result["agreement"]["accepted"]
    with pytest.raises(ValueError, match="ten"):
        ref.period_change(stream, stream, (0, 9), (20, 21), "hr_change_bpm")


def test_ratio_gate_and_exact_sequence_count():
    streams = {"times": np.arange(30), "rr": np.linspace(0.6, 0.9, 30)}
    result = ref.rr_ratio(streams, streams, (0, 30))
    assert result["reference1"]["rr_ratio"] == pytest.approx(1.5)
    assert result["reference1"]["max_rr_time_s"] == 29
    assert result["reference1"]["min_rr_time_s"] == 0
    assert result["reference2"] == result["reference1"]
    assert result["tolerances"]["rr_ratio"] == {"relative": 0.05}
    assert ref._numeric_gate({"x": 100}, {"x": 102.5}, {"x": {"relative": 0.05}})["accepted"]
    assert not ref._numeric_gate({"x": 100}, {"x": 102.501}, {"x": {"relative": 0.05}})["accepted"]
    assert not ref._numeric_gate({"n": 3}, {"n": 4}, {"n": {"absolute": 0}})["accepted"]


@pytest.mark.parametrize("field,tolerance", [("hr_change_bpm", 2.0), ("sbp_change_mmhg", 3.0)])
def test_period_evidence_is_independently_located_and_gated(field, tolerance):
    first = {"times": np.arange(40, dtype=float), "values": np.r_[np.full(20, 50), np.arange(20) + 60]}
    second = {"times": first["times"] + 0.06, "values": first["values"]}
    result = ref.period_change(first, second, (0, 20), (20, 40), field)
    assert result["reference1"][field] == result["reference2"][field]
    assert result["reference1"]["response_max_time_s"] == 39
    assert result["reference2"]["response_min_time_s"] == pytest.approx(20.06)
    assert result["tolerances"][field] == {"absolute": tolerance}
    assert result["agreement"]["fields"][field]["accepted"]
    assert not result["agreement"]["accepted"]


def test_rr_ratio_evidence_disagreement_rejects_identical_magnitudes():
    first = {"times": np.arange(30, dtype=float), "rr": np.linspace(0.6, 0.9, 30)}
    second = {"times": first["times"] + 0.06, "rr": first["rr"]}
    result = ref.rr_ratio(first, second, (0, 30))
    assert result["agreement"]["fields"]["rr_ratio"]["accepted"]
    assert not result["agreement"]["accepted"]
    first["rr"][[4, 15]] = 1
    result = ref.rr_ratio(first, first, (0, 30))
    for implementation in ("reference1", "reference2"):
        assert result[implementation]["max_rr_time_s"] == 4


@pytest.mark.parametrize("kind", ["hr_change_bpm", "sbp_change_mmhg", "rr_ratio"])
def test_f1_fractional_onset_references_use_exactly_displayed_windows(monkeypatch, kind):
    times = np.arange(0, 180, 0.5)
    stream = {"times": times, "values": 65 + np.sin(times), "rr": 0.8 + 0.1 * np.sin(times)}
    monkeypatch.setattr(ref, "hr_streams", lambda *args: (stream, stream))
    monkeypatch.setattr(ref, "pressure_streams", lambda *args: (stream, stream))
    monkeypatch.setattr(ref.audit, "source_hashes", lambda *args: {})
    record = {
        "dataset": "synthetic",
        "subject_key": "fixture",
        "record_key": "fixture",
        "source_hashes": {},
        "start_s": 0,
        "duration_s": 240,
        "x": np.zeros((240 * 125, 2)),
        "fs": 125,
        "channels": ["ECG", "ABP"],
        "units": ["mV", "mmHg"],
        "events": [{"kind": "valsalva", "t": 100.1234567, "method": "synthetic event"}],
    }
    item = ref._f1(record, kind)
    displayed = [tuple(map(float, period)) for period in re.findall(r"([\d.]+)–([\d.]+) s", item["question"])]
    assert displayed == [tuple(region["window_s"]) for region in item["grounding_rubric"]["regions"]]
    recomputed = (
        ref.rr_ratio(stream, stream, displayed[0])
        if kind == "rr_ratio"
        else ref.period_change(stream, stream, *displayed, kind)
    )
    assert item["expected_answer"] == recomputed["reference1"]


def test_pressure_streams_independent_peak_and_trough_segmentation():
    fs = 125
    time = np.arange(60 * fs) / fs
    pressure = 80 + 40 * np.sin(np.pi * ((time - 0.1) % 1)) ** 2
    first, second = ref.pressure_streams(pressure, fs)
    assert min(len(first["values"]), len(second["values"])) > 50
    np.testing.assert_allclose(first["values"], 120, atol=0.03)
    np.testing.assert_allclose(second["values"], 120, atol=0.03)
    pressure[10 * fs : 11 * fs] = 80
    with pytest.raises(ValueError, match="plateau"):
        ref.pressure_streams(pressure, fs)


@pytest.mark.parametrize(
    "wave,expected",
    [(np.ones(30), True), (np.full(30, np.nan), True), (np.arange(30), False), (np.r_[np.ones(29), np.nan], False)],
)
def test_insufficiency_requires_both_objective_criteria(wave, expected):
    result = ref.insufficiency_checks(wave)
    assert result["reference1"] is expected
    assert result["reference2"] is expected
    assert result["accepted"] is expected


def test_empty_pool_records_unfilled_targets_without_relaxation():
    items, excluded = ref.build_candidates([])
    assert not items
    assert sum(row["count"] for row in excluded if row["family"] == "F1") == 10
    assert sum(row["count"] for row in excluded if row["family"] == "F4") == 10
    assert sum(row["count"] for row in excluded if row["family"] == "F5") == 8


def test_methodological_items_reject_short_records_before_reference_work():
    with pytest.raises(ValueError, match="600"):
        ref._f4({"duration_s": 599.9}, "sequence_brs")
    with pytest.raises(ValueError, match="600"):
        ref._f4({"duration_s": 120}, "tfa_gain")


def test_reference_disagreement_preserves_comparison_for_exclusion():
    comparison = {
        "reference1": {"value": 1},
        "reference2": {"value": 2},
        "agreement": {"accepted": False},
        "tolerances": {"value": {"absolute": 0.1}},
    }
    with pytest.raises(ref.ReferenceDisagreement) as error:
        ref._item({}, "F1", "test", "Question", comparison, [], [])
    assert error.value.details == comparison


def test_heart_rate_insufficiency_pair_contains_no_alternative_pulse_channel(monkeypatch):
    time = np.arange(180 * 125) / 125
    original = np.column_stack((np.sin(2 * np.pi * time), 90 + 20 * np.cos(2 * np.pi * time)))
    record = {
        "dataset": "synthetic",
        "subject_key": "fixture",
        "record_key": "fixture",
        "source_hashes": {},
        "start_s": 0,
        "duration_s": 180,
        "x": original.copy(),
        "fs": 125,
        "channels": ["ECG", "ABP"],
        "units": ["mV", "mmHg"],
        "events": [],
    }
    stream = {"values": np.full(179, 60.0)}
    monkeypatch.setattr(ref, "hr_streams", lambda *args: (stream, stream))
    monkeypatch.setattr(ref.audit, "source_hashes", lambda *args: {})
    control, absent = ref._f5_pair(record, "heart_rate")
    for item in (control, absent):
        assert item["channels"] == ["ECG"]
        assert item["units"] == ["mV"]
        assert item["x"].shape == (len(time), 1)
        assert item["answer_fields"] == ["heart_rate_bpm"]
    assert np.ptp(control["x"]) > 0
    assert np.ptp(absent["x"]) == 0
    assert absent["ground_truth"]["agreement"]["accepted"]
    assert absent["abstention_correct"]
    assert control["pair_id"] == absent["pair_id"]
    assert control["pair_role"] == "control"
    assert absent["pair_role"] == "abstention"
    assert control["question"] == absent["question"]
    assert control["task_type"] == absent["task_type"]
    assert control["source"] == absent["source"]
    assert ref._f5_pair(record, "heart_rate")[0]["pair_id"] == control["pair_id"]
    np.testing.assert_array_equal(record["x"], original)


def test_pair_hold_changes_only_the_requested_channel(monkeypatch):
    time = np.arange(180 * 125) / 125
    original = np.column_stack((np.sin(2 * np.pi * time), 90 + 20 * np.cos(2 * np.pi * time)))
    record = {
        "dataset": "synthetic",
        "subject_key": "fixture",
        "record_key": "fixture",
        "source_hashes": {},
        "start_s": 0,
        "duration_s": 180,
        "x": original.copy(),
        "fs": 125,
        "channels": ["ECG", "ABP"],
        "units": ["mV", "mmHg"],
        "events": [],
    }
    stream = {"values": np.full(179, 110.0)}
    monkeypatch.setattr(ref, "pressure_streams", lambda *args: (stream, stream))
    monkeypatch.setattr(ref.audit, "source_hashes", lambda *args: {})
    control, absent = ref._f5_pair(record, "systolic_pressure")
    np.testing.assert_array_equal(control["x"][:, 0], absent["x"][:, 0])
    np.testing.assert_array_equal(control["x"], original.astype(np.float32))
    assert np.ptp(control["x"][:, 1]) > 0
    assert np.ptp(absent["x"][:, 1]) == 0
    assert control["channels"] == absent["channels"]
    assert control["question"] == absent["question"]
    np.testing.assert_array_equal(record["x"], original)


def test_build_allocates_four_distinct_subject_pairs_atomically(monkeypatch):
    def unavailable(*args):
        raise ValueError("fixture excludes unrelated families")

    def paired(record, kind):
        return [
            {"family": "F5", "pair_type": kind, "subject_key": record["subject_key"], "pair_role": role}
            for role in ("control", "abstention")
        ]

    monkeypatch.setattr(ref, "_f1", unavailable)
    monkeypatch.setattr(ref, "_f4", unavailable)
    monkeypatch.setattr(ref, "_f5_pair", paired)
    records = [{"subject_key": str(index), "record_key": str(index)} for index in range(4)]
    # Duplicate records of one subject cannot consume additional pair slots.
    items, excluded = ref.build_candidates([records[0], records[0], *records[1:]])
    assert len(items) == 8
    for index in range(0, 8, 2):
        first, second = items[index : index + 2]
        assert first["subject_key"] == second["subject_key"]
        assert first["pair_type"] == second["pair_type"]
        assert {first["pair_role"], second["pair_role"]} == {"control", "abstention"}
    assert len({item["subject_key"] for item in items}) == 4
    assert not [row for row in excluded if row["family"] == "F5"]


def test_f5_pair_failure_never_returns_an_unpaired_control(monkeypatch):
    def view(record, kind, abstain):
        if abstain:
            raise ValueError("fixture insufficiency criterion failed")
        return {"pair_role": "control"}

    monkeypatch.setattr(ref, "_f5", view)
    with pytest.raises(ValueError, match="insufficiency"):
        ref._f5_pair({}, "heart_rate")
