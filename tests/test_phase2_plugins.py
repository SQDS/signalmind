"""Phase-2 specialist plugins: TFA (analytic known answer + scipy oracle), pulse features/quality/change-points,
alignment (delay, delay change, missing beats, polarity). Synthetic signals only; no benchmark reference is used."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.errors import SignalError  # noqa: E402
from tsmcp.core.signal import Channel, Signal  # noqa: E402
from tsmcp.plugins import align, context, pulse, tfa  # noqa: E402
from tsmcp.plugins.optional import OPTIONAL_PLUGINS, plugin_set, register_signal  # noqa: E402
from tsmcp.runtime.units import compatible, convert, family  # noqa: E402

STANDARD = Path.home() / "cbv_true/2027.7.8/code/tfa/standard.py"


def sig_of(*chans):
    s = Signal("t", [Channel(n, fs, np.asarray(x, dtype=np.float32), u) for n, fs, x, u in chans])
    register_signal(s)
    return s


def win(ch, t0, t1):
    return ch.data[int(t0 * ch.fs) : int(t1 * ch.fs)].astype(np.float64)


def strip(r):
    r.pop("_artifact", None)
    json.dumps(r)  # JSON-serialisable
    return r


# ---- TFA ------------------------------------------------------------------------------------------------------------------
def tfa_record(n_s=600, fs=50, g=(0.5, 0.3), noise=0.05, seed=0, flow_unit="cm/s"):
    rng = np.random.default_rng(seed)
    u = np.zeros(n_s)
    for i in range(1, n_s):  # AR(1) input with power across 0.02-0.5 Hz
        u[i] = 0.6 * u[i - 1] + rng.standard_normal()
    y = g[0] * u + g[1] * np.concatenate([[0.0], u[:-1]]) + noise * rng.standard_normal(n_s)
    abp = 90 + 5 * np.repeat(u, fs)  # each 1-s mean recovers the 1 Hz sample exactly
    cbfv = 60 + 5 * np.repeat(y, fs)
    return sig_of(("ABP", fs, abp, "mmHg"), ("CBFV", fs, cbfv, flow_unit)), u, y


def analytic_band(g, lo, hi, quantity):
    f = np.fft.rfftfreq(100, 1.0)
    m = (f >= lo) & (f < hi)
    h = g[0] + g[1] * np.exp(-2j * np.pi * f[m])
    return float(np.mean(np.abs(h) if quantity == "gain" else np.angle(h, deg=True)))


@pytest.mark.parametrize("band", ["vlf", "lf", "hf"])
def test_tfa_known_answer(band):
    g = (0.5, 0.3)
    s, _, _ = tfa_record(g=g)
    ch = s.channel("CBFV")
    gain = strip(tfa.MEASURES[f"tfa_gain_{band}"](win(ch, 0, 600), 50, "cm/s", 0.0, ch=ch))
    phase = strip(tfa.MEASURES[f"tfa_phase_{band}"](win(ch, 0, 600), 50, "cm/s", 0.0, ch=ch))
    coh = strip(tfa.MEASURES[f"tfa_coherence_{band}"](win(ch, 0, 600), 50, "cm/s", 0.0, ch=ch))
    lo, hi = tfa.BANDS[band]
    assert gain["value"] == pytest.approx(analytic_band(g, lo, hi, "gain"), rel=0.05)
    assert phase["value"] == pytest.approx(analytic_band(g, lo, hi, "phase"), abs=3.0)
    assert coh["value"] > 0.9 and gain["n_windows"] == 11 and gain["unit"] == "(cm/s)/mmHg"
    assert gain["conventions"]["band_edges"] == "lo <= f < hi" and gain["channels_effective"] == ["CBFV", "ABP"]
    assert gain["warnings"] == [] and gain["definition_id"] == f"tfa.gain.{band}.v1"


def test_tfa_matches_original_scipy_implementation():
    if not STANDARD.exists():
        pytest.skip("original standard.py not on this machine")
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("tfa_standard", STANDARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, u, y = tfa_record(noise=0.3, seed=3)
    f0, pxx0, pyy0, pxy0 = mod.spectrum_for_window(u, y)
    ref = mod.tfa_from_spectra(f0, pxx0, pyy0, pxy0)
    f1, pxx1, pyy1, pxy1, _ = tfa.welch_cross(u, y)
    got = tfa.transfer(f1, pxx1, pyy1, pxy1)
    assert np.allclose(f0, f1)
    for k_ref, k_got in (("gain", "gain"), ("phase", "phase_deg"), ("coherence", "coherence")):
        assert np.allclose(ref[k_ref], got[k_got], rtol=1e-9, atol=1e-9), k_ref


def test_tfa_validity_and_limitations():
    s, _, _ = tfa_record(noise=3.0, seed=5)
    ch = s.channel("CBFV")
    noisy = strip(tfa.MEASURES["tfa_gain_lf"](win(ch, 0, 600), 50, "cm/s", 0.0, ch=ch))
    assert any(w["code"] == "low_coherence" for w in noisy["warnings"])
    short = strip(tfa.MEASURES["tfa_gain_lf"](win(ch, 0, 250), 50, "cm/s", 0.0, ch=ch))
    assert short["n_windows"] == 4 and any(w["code"] == "few_welch_windows" for w in short["warnings"])
    tiny = strip(tfa.MEASURES["tfa_gain_lf"](win(ch, 0, 60), 50, "cm/s", 0.0, ch=ch))
    assert tiny["value"] is None and tiny["status"] == "limitation"
    flat = sig_of(("ABP", 50, np.full(30000, 80.0), "mmHg"), ("CBFV", 50, np.asarray(s.channel("CBFV").data), "cm/s"))
    c2 = flat.channel("CBFV")
    assert strip(tfa.MEASURES["tfa_gain_lf"](win(c2, 0, 600), 50, "cm/s", 0.0, ch=c2))["status"] == "limitation"
    unk, _, _ = tfa_record(flow_unit="?")
    c3 = unk.channel("CBFV")
    r = strip(tfa.MEASURES["tfa_gain_lf"](win(c3, 0, 600), 50, "?", 0.0, ch=c3))
    assert r["unit"] == "?" and any(w["code"] == "unit_unknown" for w in r["warnings"])
    alone = Signal("a", [Channel("CBFV", 50, np.asarray(s.channel("CBFV").data), "cm/s")])
    register_signal(alone)
    with pytest.raises(SignalError, match="pressure"):
        tfa.MEASURES["tfa_gain_lf"](win(alone.channels[0], 0, 600), 50, "cm/s", 0.0, ch=alone.channels[0])


def test_units_for_new_quantities():
    assert family("(cm/s)/mmHg") == "velocity/pressure" == family("cm/s/mmHg")
    assert compatible("(cm/s)/mmHg", "(m/s)/kPa") and convert(1.0, "(m/s)/mmHg", "(cm/s)/mmHg") == pytest.approx(100)
    assert family("deg") == family("rad") == "angle" and family("count") == "count" and family("?") is None


# ---- pulse ----------------------------------------------------------------------------------------------------------------
def test_pulse_rate_and_quality_on_ppg_like_signal():
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, hr_bpm=84, seed=8, hrv=0.03)
    ppg = (abp - 80) / 40.0
    s = sig_of(("ECG", 125, ecg, "mV"), ("PPG", 125, ppg, "?"))
    ch = s.channel("PPG")
    r = strip(pulse.pulse_rate(win(ch, 10, 50), 125, "?", 10.0, ch=ch))
    rr = np.mean([b["t_end_s"] - b["t_foot_s"] for b in beats if 10 <= b["t_foot_s"] and b["t_end_s"] <= 50])
    assert r["unit"] == "bpm" and abs(r["value"] - 60 / rr) < 1.5 and r["resolution"]["known"]
    q = strip(pulse.pulse_quality(win(ch, 10, 50), 125, "?", 10.0, ch=ch))
    assert q["value"] >= 0.9 and q["sqi_median"] > 0.95
    t = pulse.pulse_beats(win(ch, 10, 50), 125, "?", 10.0, ch=ch)
    assert t["_artifact"]["kind"] == "pulse_beat_table" and all("sqi" in row for row in t["_artifact"]["rows"])


def test_pulse_quality_drops_with_artefact_burst():
    abp, ecg, _ = synth_abp(fs=125, duration_s=60, seed=9)
    rng = np.random.default_rng(1)
    abp[int(25 * 125) : int(35 * 125)] += 25 * rng.standard_normal(int(10 * 125))
    s = sig_of(("ECG", 125, ecg, "mV"), ("ABP", 125, abp, "mmHg"))
    ch = s.channel("ABP")
    q = strip(pulse.pulse_quality(win(ch, 10, 50), 125, "mmHg", 10.0, ch=ch))
    assert q["value"] < 0.85 and any(a <= 35 and b >= 25 for a, b in q["low_quality_intervals"])


def test_amplitude_change_finds_sustained_damping():
    abp, ecg, _ = synth_abp(fs=125, duration_s=90, seed=10)
    k = int(47.0 * 125)
    mean = abp[k : k + 5 * 125].mean()
    abp[k:] = mean + 0.6 * (abp[k:] - mean)  # sustained AC attenuation from 47 s
    s = sig_of(("ECG", 125, ecg, "mV"), ("ABP", 125, abp, "mmHg"))
    ch = s.channel("ABP")
    r = strip(pulse.pulse_amplitude_change(win(ch, 5, 85), 125, "mmHg", 5.0, ch=ch))
    assert abs(r["value"] - 47.0) < 1.5 and r["persistent"] and r["step"] < 0 and r["score"] > 5


def test_level_change_separates_sustained_step_from_transient():
    abp, ecg, _ = synth_abp(fs=125, duration_s=90, seed=11)
    abp[int(60 * 125) :] += 20.0  # sustained baseline displacement at 60 s
    abp[int(25 * 125) : int(28 * 125)] -= 30.0  # brief transient that returns
    s = sig_of(("ECG", 125, ecg, "mV"), ("ABP", 125, abp, "mmHg"))
    ch = s.channel("ABP")
    r = strip(pulse.pulse_level_change(win(ch, 5, 85), 125, "mmHg", 5.0, ch=ch))
    assert abs(r["value"] - 60.0) < 1.5 and r["persistent"]
    assert any(tr["t_start_s"] <= 28 and tr["t_end_s"] >= 25 for tr in r["transients"])


# ---- alignment ------------------------------------------------------------------------------------------------------------
def shifted(x, fs, start_s, shift_s):
    y = x.copy()
    k, d = int(start_s * fs), int(round(shift_s * fs))
    y[k:] = np.roll(x, d)[k:]
    return y


def test_pulse_delay_and_delay_change():
    fs = 500.0
    abp, ecg, _ = synth_abp(fs=fs, duration_s=90, seed=12, hrv=0.03)
    ppg = shifted((abp - 80) / 40.0, fs, 45.0, 0.040)  # PPG acquires an extra +40 ms from 45 s
    s = sig_of(("ECG", fs, ecg, "mV"), ("ABP", fs, abp, "mmHg"), ("PPG", fs, ppg, "?"))
    a, p = s.channel("ABP"), s.channel("PPG")
    da = strip(align.pulse_delay(win(a, 5, 40), fs, "mmHg", 5.0, ch=a))
    assert 150 < da["value"] < 260 and da["timebase"]["sync_status"] == "index_aligned_declared"
    assert da["definition_id"] == "delay.r_to_foot.observed.v1" and da["counts"]["accepted"] > 30
    ca = strip(align.pulse_delay_change(win(a, 5, 85), fs, "mmHg", 5.0, ch=a))
    cp = strip(align.pulse_delay_change(win(p, 5, 85), fs, "?", 5.0, ch=p))
    assert abs(cp["value"] - 40) < 6 and abs(cp["onset_s"] - 45) < 2 and cp["persistent"]
    assert abs(ca["value"]) < 6  # ABP unchanged: the inconsistent channel is PPG


def test_missing_beats_and_polarity():
    fs = 250.0
    abp, ecg, _ = synth_abp(fs=fs, duration_s=60, seed=13, drop_beats=(30, 31))
    s = sig_of(("ECG", fs, ecg, "mV"), ("ABP", fs, abp, "mmHg"))
    a = s.channel("ABP")
    mp = strip(align.beats_missing_in_pulse(win(a, 5, 55), fs, "mmHg", 5.0, ch=a))
    me = strip(align.beats_missing_in_ecg(win(a, 5, 55), fs, "mmHg", 5.0, ch=a))
    assert mp["value"] == 2 and me["value"] == 0 and mp["unit"] == "beats"
    inv = abp.copy()
    m = inv[int(30 * fs) :].mean()
    inv[int(30 * fs) :] = 2 * m - inv[int(30 * fs) :]
    s2 = sig_of(("ABP", fs, inv, "mmHg"))
    c = s2.channel("ABP")
    pc = strip(align.polarity_change(win(c, 0, 60), fs, "mmHg", 0.0, ch=c))
    assert 28 <= pc["value"] <= 32 and pc["skew_before"] > 0 > pc["skew_after"]
    clean = strip(align.polarity_change(win(a, 0, 60), fs, "mmHg", 0.0, ch=a))
    assert clean["value"] is None and clean["status"] == "limitation"


def test_alignment_needs_ecg_and_registry_is_opt_in():
    abp, _, _ = synth_abp(fs=125, duration_s=30, seed=14)
    s = Signal("x", [Channel("ABP", 125, abp.astype(np.float32), "mmHg")])
    context.register(s)
    with pytest.raises(SignalError, match="ECG"):
        align.pulse_delay(win(s.channels[0], 0, 30), 125, "mmHg", 0.0, ch=s.channels[0])
    assert OPTIONAL_PLUGINS == ("pressure", "tfa", "pulse", "align")
    names = {n for p in plugin_set(["tfa", "pulse", "align"]) for n in p.measures}
    assert {"tfa_gain_lf", "pulse_rate", "pulse_delay_change", "polarity_change"} <= names
