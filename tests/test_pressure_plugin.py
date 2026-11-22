"""Pressure plugin: per-beat ABP features on synthetic signals, edge cases, port agreement, dev-data smoke."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.errors import SignalError  # noqa: E402
from tsmcp.core.signal import Channel, Signal  # noqa: E402
from tsmcp.plugins.pressure import (  # noqa: E402
    PLUGIN,
    abp_beats,
    abp_dbp,
    abp_pp,
    abp_sbp,
    abp_sbp_ecg,
    register_signal,
)
from tsmcp.plugins.pressure.pulses import detect_pulses_guided, detect_pulses_unguided, savgol_coeffs  # noqa: E402


def make(abp, ecg, fs=125.0, unit="mmHg"):
    s = Signal(
        "t", [Channel("ECG", fs, ecg.astype(np.float32), "mV"), Channel("ABP", fs, abp.astype(np.float32), unit)]
    )
    register_signal(s)
    return s, s.channels[1]


def win(ch, t0, t1):
    return ch.data[int(t0 * ch.fs) : int(t1 * ch.fs)].astype(np.float64)


def truth_mean(beats, t0, t1, key="sbp"):
    return float(np.mean([b[key] for b in beats if b["t_foot_s"] >= t0 and b["t_end_s"] <= t1]))


def test_ordinary_signal_matches_generator():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60)
    s, ch = make(abp, ecg)
    r = abp_sbp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    assert abs(r["value"] - truth_mean(beats, 10, 40)) < 1.0 and r["unit"] == "mmHg"
    assert r["definition_id"] == "abp.sbp.beat_mean.v1" and r["aggregation"] == "beat_weighted"
    assert r["counts"]["rejection_reasons"] == {"incomplete_boundary": 2} and r["coverage"] > 0.9
    assert r["resolution"]["known"] and r["warnings"] == []
    d = abp_dbp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    assert abs(d["value"] - 80.0) < 1.0
    p = abp_pp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    assert abs(p["value"] - (r["value"] - d["value"])) < 0.5
    tbl = abp_beats(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    art = tbl.pop("_artifact")
    assert (
        art["kind"] == "beat_table" and len(art["rows"]) == tbl["n_beats"] and tbl["value"] == r["counts"]["accepted"]
    )
    assert set(art["columns"]) >= {"t_foot_s", "t_peak_s", "t_end_s", "sbp", "dbp", "pp", "accepted", "reject_reason"}
    assert all(row["accepted"] in (True, False) for row in art["rows"])
    json.dumps(tbl)  # reply is JSON-serialisable (no numpy scalars)


def test_boundary_beats_are_excluded_and_counted():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60)
    s, ch = make(abp, ecg)
    tbl = abp_beats(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    rows = tbl["_artifact"]["rows"]
    edge = [r for r in rows if r["reject_reason"] == "incomplete_boundary"]
    assert (
        len(edge) == 2
        and edge[0]["t_foot_s"] < 10 < edge[0]["t_end_s"]
        and edge[-1]["t_foot_s"] < 40 < edge[-1]["t_end_s"]
    )
    inner = [r for r in rows if r["accepted"]]
    assert all(10 <= r["t_foot_s"] and r["t_end_s"] <= 40 for r in inner)


def test_flat_line_and_all_flat_limitation():
    abp, ecg, _ = synth_abp(fs=125, duration_s=60, flat=(20, 30))
    s, ch = make(abp, ecg)
    r = abp_sbp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    assert (
        r["counts"]["rejection_reasons"].get("flat", 0) >= 1
        or r["counts"]["rejection_reasons"].get("implausible_duration", 0) >= 1
    )
    assert any(w["code"] in ("flat_segments", "many_rejected_beats", "low_coverage") for w in r["warnings"])
    assert any(a <= 20.5 and b >= 29.5 for a, b in r["intervals_excluded"])
    flat = np.full(int(60 * 125), 80.0)
    s2, ch2 = make(flat, ecg)
    r2 = abp_sbp(win(ch2, 10, 40), 125, "mmHg", 10.0, ch=ch2)
    assert r2["value"] is None and r2["status"] == "limitation"


def test_clipping_is_rejected_with_reason():
    abp, ecg, _ = synth_abp(fs=125, duration_s=60, clip_at=110.0)
    s, ch = make(abp, ecg)
    r = abp_sbp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    assert r["counts"]["rejection_reasons"].get("clipped", 0) >= 30
    assert any(w["code"] == "clipping_present" for w in r["warnings"])
    assert r["value"] is None or r["value"] < 110.5  # never reports a plausible SBP above the clip level


def test_missing_beats_reduce_coverage_and_are_reported():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, drop_beats=(20, 21, 22))
    s, ch = make(abp, ecg)
    r = abp_sbp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    full = abp_sbp(
        win(make(*synth_abp(fs=125, duration_s=60)[:2])[1], 10, 40),
        125,
        "mmHg",
        10.0,
        ch=make(*synth_abp(fs=125, duration_s=60)[:2])[1],
    )
    assert r["coverage"] < full["coverage"] - 0.05
    assert (
        "implausible_duration" in r["counts"]["rejection_reasons"]
        or "low_pulse_pressure" in r["counts"]["rejection_reasons"]
    )
    assert abs(r["value"] - truth_mean(beats, 10, 40)) < 1.0  # the surviving beats are still right


def test_spikes_and_nan_are_rejected():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, spikes=(25.3,), nan_at=(32.0, 33.0))
    s, ch = make(abp, ecg)
    r = abp_sbp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    reasons = r["counts"]["rejection_reasons"]
    assert reasons.get("nonfinite", 0) >= 1 and reasons.get("spike", 0) >= 1
    assert any(w["code"] == "spikes_present" for w in r["warnings"])
    assert abs(r["value"] - truth_mean(beats, 10, 40)) < 1.5


@pytest.mark.parametrize("fs", [125.0, 250.0, 500.0, 1000.0])
def test_sampling_rates(fs):
    abp, ecg, beats = synth_abp(fs=fs, duration_s=40, hr_bpm=80, sbp=135, dbp=70)
    s, ch = make(abp, ecg, fs=fs)
    r = abp_sbp(win(ch, 5, 35), fs, "mmHg", 5.0, ch=ch)
    assert abs(r["value"] - truth_mean(beats, 5, 35)) < 1.0 and r["counts"]["accepted"] >= 35


def test_unknown_unit_stays_unknown():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60)
    s, ch = make(abp, ecg, unit="?")
    r = abp_sbp(win(ch, 10, 40), 125, "?", 10.0, ch=ch)
    assert r["unit"] == "?" and any(w["code"] == "unit_unknown" for w in r["warnings"])
    assert abs(r["value"] - truth_mean(beats, 10, 40)) < 1.0  # the amplitude is measured, the unit is not asserted


def test_ecg_guidance_is_explicit_and_recorded():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, hrv=0.05, seed=3)
    s, ch = make(abp, ecg)
    pulse = abp_sbp(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    guided = abp_sbp_ecg(win(ch, 10, 40), 125, "mmHg", 10.0, ch=ch)
    assert guided["guidance"] == "ecg" and pulse["guidance"] == "pulse"
    assert guided["timebase"]["sync_status"] == "index_aligned_declared" and "ECG" in guided["channels_effective"]
    assert abs(guided["value"] - pulse["value"]) < 1.0
    only_abp = Signal("u", [Channel("ABP", 125, abp.astype(np.float32), "mmHg")])
    register_signal(only_abp)
    with pytest.raises(SignalError, match="ECG channel"):
        abp_sbp_ecg(win(only_abp.channels[0], 10, 40), 125, "mmHg", 10.0, ch=only_abp.channels[0])


def test_short_window_and_registration():
    abp, ecg, _ = synth_abp(fs=125, duration_s=60)
    s, ch = make(abp, ecg)
    with pytest.raises(SignalError, match="at least 2 s"):
        abp_sbp(win(ch, 10, 11), 125, "mmHg", 10.0, ch=ch)
    assert set(PLUGIN.measures) == {"abp_beats", "abp_sbp", "abp_dbp", "abp_pp", "abp_beats_ecg", "abp_sbp_ecg",
                                    "abp_dbp_ecg", "abp_pp_ecg"}  # fmt: skip


def test_savgol_matches_scipy():
    scipy_signal = pytest.importorskip("scipy.signal")
    for w, p in ((5, 2), (11, 2), (21, 3)):
        assert np.allclose(savgol_coeffs(w, p), scipy_signal.savgol_coeffs(w, p))


def test_guided_port_agrees_with_source_implementation():
    src = Path.home() / "world_model" / "hemoworld" / "beats.py"
    if not src.exists():
        pytest.skip("source implementation not available on this machine")
    pytest.importorskip("scipy")
    pkg = types.ModuleType("hemoworld")
    pkg.__path__ = [str(src.parent)]
    sub = types.ModuleType("hemoworld.pulsedb")
    sub.FS_HZ = 125.0
    sys.modules["hemoworld"], sys.modules["hemoworld.pulsedb"] = pkg, sub
    spec = importlib.util.spec_from_file_location("hemoworld.beats", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # pragma: no cover - environment specific
        pytest.skip(f"cannot import source implementation: {e}")
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, hrv=0.05, seed=5)
    r = np.array([int(round((b["t_foot_s"] - 0.2) * 125)) for b in beats])
    f0, p0 = mod.detect_abp_pulses_guided(abp, r, 125.0)
    f1, p1, _ = detect_pulses_guided(abp, r, 125.0)
    # identical away from the edges (scipy's savgol uses polynomial edge interpolation; ours reflects)
    inner = (f0 > 125) & (f0 < abp.size - 125)
    assert (
        np.array_equal(f0[inner], f1[: inner.sum()][: len(f0[inner])])
        or np.abs(f0[inner] - f1[inner][: inner.sum()]).max() <= 1
    )
    assert abs(len(p0) - len(p1)) <= 2


def test_unguided_detector_counts_beats():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, hrv=0.08, seed=7)
    feet, peaks, _ = detect_pulses_unguided(abp, 125.0)
    assert abs(len(peaks) - len(beats)) <= 1 and (np.diff(feet) > 0).all()


def test_development_case_smoke():
    """Opened development data only (eval/cases, gitignored): consistency checks, no reference answers read."""
    root = Path(__file__).resolve().parents[1] / "eval" / "cases" / "p9level2"
    if not root.exists():
        pytest.skip("development cases not present")
    case = None
    for d in sorted(root.iterdir()):
        v = json.loads((d / "visible.json").read_text())
        names = [c["name"] for c in v["channels"]]
        if "ABP" in names and "ECG" in names and v["channels"][0]["fs"] == 1000.0 and v["dur_s"] <= 200:
            case, visible, names = d, v, names
            break
    if case is None:
        pytest.skip("no suitable development case")
    arr = np.load(case / "signal.npy", allow_pickle=False)
    fs = visible["channels"][0]["fs"]
    s = Signal(
        "dev",
        [Channel(c["name"], fs, arr[:, i].astype(np.float32), c["unit"]) for i, c in enumerate(visible["channels"])],
    )
    register_signal(s)
    ch = s.channel("ABP")
    r = abp_sbp(win(ch, 20, 80), fs, ch.unit, 20.0, ch=ch)
    g = abp_sbp_ecg(win(ch, 20, 80), fs, ch.unit, 20.0, ch=ch)
    if r["value"] is not None:
        assert 50 < r["value"] < 250 and r["counts"]["accepted"] >= 20
    if r["value"] is not None and g["value"] is not None:
        assert abs(r["value"] - g["value"]) < 5.0
