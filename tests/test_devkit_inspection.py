"""Contracts and regression cases for the inspect-only milestone."""

from copy import deepcopy

import numpy as np
import pytest

from tsmcp.cleaning.common import read, sha, write
from tsmcp.inspection.contracts import strict_loads
from tsmcp.inspection.run import inspect
from tsmcp.inspection.scan import PRESETS, scan, windows


@pytest.mark.parametrize("text", ['{"x":1e999}', '{"x":-1e999}', '{"x":Infinity}', '{"x":NaN}', '{"x":1,"x":2}'])
def test_finite_unique_json(text):
    with pytest.raises(ValueError):
        strict_loads(text)


def test_scan_has_full_tail_and_short_pulse_path():
    fs = 125
    x = np.zeros(101 * fs + 3)
    x[-2] = 7
    valid = np.ones(len(x), bool)
    valid[31:49] = False
    masks = []
    for preset in PRESETS:
        events, _, _ = scan(x, valid, {}, fs, "PPG", preset)
        assert any(e["kind"] == "large_step" and e["end"] == len(x) for e in events)
        assert all(e["end"] <= 31 or e["start"] >= 49 for e in events)
        alarm = np.zeros(len(x), bool)
        for e in events:
            alarm[e["start"] : e["end"]] = True
        masks.append(alarm)
    assert np.all(~masks[0] | masks[1]) and np.all(~masks[1] | masks[2])
    assert list(windows(101, 50, 25))[-1] == (51, 101)


def test_global_barrier_and_immutable_results(tmp_path):
    src = tmp_path / "source"
    out = tmp_path / "inspection"
    c = dict(id="c1", record_id="patient", task="ABP", fs=125, n=500, unit="mmHg")
    folder = src / "public" / c["id"]
    folder.mkdir(parents=True)
    path = folder / "input.npz"
    np.savez(path, values=np.ones(c["n"]))
    c["input_sha256"] = sha(path)
    bad = {**c, "id": "missing"}
    write(src / "dataset_manifest.json", {"cases": [c, bad]})
    summary = inspect(src, out)
    assert not summary["inspection_complete"] and not (out / "candidate_manifest.json").exists()
    assert summary["errors"][0]["case_id"] == "missing"
    # Restore the missing file and confirm the global barrier can freeze only after every item completes.
    folder = src / "public" / bad["id"]
    folder.mkdir()
    (folder / "input.npz").write_bytes(path.read_bytes())
    assert inspect(src, out)["inspection_complete"]
    marker = sha(out / "candidate_manifest.json")
    assert inspect(src, out)["candidates"] == 2 and sha(out / "candidate_manifest.json") == marker
    (out / "evidence.jsonl").write_text("")
    with pytest.raises(ValueError, match="artifacts changed"):
        inspect(src, out)


def test_fixture_rejects_unrelated_repair_and_incomplete_checks(tmp_path):
    from eval.devkit_stage_example import acceptance, main

    main(tmp_path)
    rev = read(tmp_path / "numeric_review.json")
    patch = read(tmp_path / "patch_plan.json")
    evidence = read(tmp_path / "evidence.json")
    authorized = read(tmp_path / "discovery.json")["authorized_range"]
    assert patch["parent_version_id"] != patch["source_version_id"]
    assert acceptance(rev, patch, authorized, evidence)
    wrong = deepcopy(patch)
    wrong["write_set"][0]["end_sample"] += 1
    with pytest.raises(ValueError, match="authorized range"):
        acceptance(rev, wrong, authorized, evidence)
    del rev["checks"][0]
    with pytest.raises(ValueError, match="mandatory checks"):
        acceptance(rev, patch, authorized, evidence)


def test_ecg_excursion_reports_plateau_and_separates_recovered_events():
    fs = 125
    t = np.arange(12 * fs) / fs
    x = 0.03 * np.sin(2 * np.pi * t)
    x[500:675] -= np.linspace(4, 2, 175)
    x[800:850] += 5
    events, _, _ = scan(x, np.isfinite(x), {}, fs, "ECG", "balanced")
    assert [(v["start"], v["end"]) for v in events if v["kind"] == "ecg_excursion"] == [(500, 675), (800, 850)]
    hole = np.ones(len(x), bool)
    hole[570:630] = False
    broken, _, _ = scan(x, hole, {}, fs, "ECG", "balanced")
    assert not any(v["kind"] == "ecg_excursion" and v["start"] < 570 < v["end"] for v in broken)


def test_ecg_excursion_preserves_narrow_periodic_peaks():
    fs = 125
    t = np.arange(12 * fs) / fs
    x = 0.03 * np.cos(2 * np.pi * t) + np.exp(-(((t % 0.8 - 0.4) / 0.018) ** 2))
    events, _, _ = scan(x, np.isfinite(x), {}, fs, "ECG", "thorough")
    assert not any(v["kind"] == "ecg_excursion" for v in events)


@pytest.mark.parametrize("fs", [125, 250, 500])
def test_ecg_baseline_shift_independent_of_repeating_qrs(fs):
    t = np.arange(24 * fs) / fs
    x = 0.025 * np.sin(2 * np.pi * 0.22 * t)
    for qrs in np.arange(0.6, 23.6, 0.83):
        x += np.exp(-0.5 * ((t - qrs) / 0.015) ** 2) + 0.14 * np.exp(-0.5 * ((t - qrs - 0.24) / 0.055) ** 2)
    valid = np.ones(len(t), bool)

    def warnings(values, mask):
        events, _, _ = scan(values, mask, {}, fs, "ECG", "balanced")
        return [(e["start"] / fs, e["end"] / fs) for e in events if e["kind"] == "ecg_baseline_shift"]

    assert not warnings(x, valid)
    assert not warnings(x + 0.07 * t, valid)
    for sign in (-1, 1):
        displaced = x + sign * 1.4 * np.exp(-0.5 * ((t - 12.22) / 0.38) ** 2)
        [(start, end)] = warnings(displaced, valid)
        assert 11.0 < start < 11.95 and 12.5 < end < 13.5
        assert not warnings(displaced, valid & ((t < 12.0) | (t >= 12.5)))


def test_ecg_steps_distinguish_repeated_qrs_from_superimposed_and_offbeat_spikes():
    fs = 125
    t = np.arange(16 * fs)
    centers = np.arange(50, len(t), 100)
    x = 0.02 * np.cos(2 * np.pi * t / 100)
    for center in centers:
        x += np.exp(-0.5 * ((t - center) / 1.5) ** 2)
    on_beat, off_beat = centers[8], centers[12] + 25
    x[[on_beat, off_beat]] += 2
    events, levels, _ = scan(x, np.ones(len(t), bool), {}, fs, "ECG", "balanced")
    edges = [e for e in events if e["kind"] == "large_step"]
    assert levels["ecg_step_edges_matching_repeated_qrs"] > 0
    assert all(any(e["start"] <= p < e["end"] for e in edges) for p in (on_beat, off_beat))
    assert not any(centers[4] - 10 <= e["start"] <= centers[4] + 10 for e in edges)
    # Without enough neighboring beats the original steep evidence stays visible.
    for task, length in (("ECG", 2 * fs), ("PPG", len(t)), ("ABP", len(t))):
        events, _, _ = scan(x[:length], np.ones(length, bool), {}, fs, task, "balanced")
        assert any(e["kind"] == "large_step" and abs(e["start"] - centers[1]) < 10 for e in events)


@pytest.mark.parametrize("fs", [125, 250, 500])
def test_ecg_sampling_phase_and_gain_with_onbeat_and_offbeat_impulses(fs):
    t = np.arange(24 * fs) / fs
    x = 0.04 * np.cos(2 * np.pi * 0.3 * t)
    centers = np.arange(0.45, 23.7, 0.817)
    for i, c in enumerate(centers):
        gain = (0.90, 1.13, 1.02, 1.07)[i % 4]
        x += gain * (np.exp(-0.5 * ((t - c) / 0.011) ** 2) - 0.21 * np.exp(-0.5 * ((t - c - 0.035) / 0.015) ** 2))
    events, levels, _ = scan(x, np.isfinite(x), {}, fs, "ECG", "balanced")
    assert levels["ecg_step_edges"] > len(centers)
    assert not any(e["kind"] == "large_step" for e in events)
    on_beat, off_beat = round(fs * centers[14]), round(fs * (centers[20] + 0.25))
    x[on_beat] += 1.1
    x[off_beat] -= 1.1
    events, _, _ = scan(x, np.isfinite(x), {}, fs, "ECG", "balanced")
    assert all(
        any(e["kind"] == "large_step" and e["start"] <= p < e["end"] for e in events) for p in (on_beat, off_beat)
    )


@pytest.mark.parametrize("fs", [125, 250])
def test_abp_pressure_cycles_flag_transient_changes_without_labeling_steady_low_pressure(fs):
    from tsmcp.inspection.scan import abp_pulse_episodes

    t = np.arange(50 * fs) / fs
    phase = t % 1
    shape = 40 * np.where(phase < 0.15, phase / 0.15, (1 - phase) / 0.85)
    baseline = 70 + shape
    for values in [baseline, baseline - 35, baseline + 0.7 * t, baseline + 2 * np.sin(2 * np.pi * t / 6)]:
        assert not list(abp_pulse_episodes(values, fs, 1))
    dip = baseline - 18 * np.exp(-0.5 * ((t - 25) / 1.25) ** 2)
    long_cycle = 70 + shape * ((t < 25) | (t >= 26)) - 12 * np.exp(-0.5 * ((t - 25.5) / 0.8) ** 2)
    for values in [dip, long_cycle]:
        result, _, _ = scan(values, np.isfinite(values), {}, fs, "ABP", "balanced")
        found = [r for r in result if r["kind"] == "abp_pulse_episode"]
        assert len(found) == 1 and 22 * fs < found[0]["start"] < 25.5 * fs < found[0]["end"] < 29 * fs
    valid = (t < 24) | (t > 26)
    result, _, _ = scan(dip, valid, {"missing": ~valid}, fs, "ABP", "balanced")
    assert not any(r["kind"] == "abp_pulse_episode" for r in result)
