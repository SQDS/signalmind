"""Independent validation of the pressure plugin (docs/signalmind/delivery-report.md §3 checks).

  .venv/bin/python eval/validate_pressure.py --out eval/results/signalmind/pressure-validation.json

References that do NOT use the plugin:
  A. scipy `find_peaks` on the raw ABP (distance 0.3 s, prominence 0.25 x (p95 - p5)); beat = trough-to-trough around
     each peak; SBP = beat maximum, DBP = beat minimum (raw samples, no smoothing).
  B. the user's original scipy implementation `~/world_model/hemoworld/beats.py` (R-peaks + guided foot/systolic
     search); SBP = raw maximum between consecutive feet, DBP = raw value at the foot.
  G. generator truth for synthetic signals (tests/synth_abp.py).
The plugin's own outputs are never used as expected values. Development data = opened p9level2 cases (gitignored);
no reference answers are read. Output: per-window agreement tables and summary statistics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.signal import Channel, Signal  # noqa: E402
from tsmcp.plugins.pressure import abp_dbp, abp_dbp_ecg, abp_sbp, abp_sbp_ecg, register_signal  # noqa: E402

PRESSURE = ("mmHg", "kPa", "cmH2O")


# ---- reference A: scipy peaks on raw samples -------------------------------------------------------------------------
def reference_a(abp: np.ndarray, fs: float, t0: float, t1: float) -> dict | None:
    from scipy.signal import find_peaks

    x = np.asarray(abp, dtype=np.float64)
    i0, i1 = int(t0 * fs), int(t1 * fs)
    seg = x[i0:i1]
    if seg.size < 2 * fs or not np.isfinite(seg).any():
        return None
    fin = seg[np.isfinite(seg)]
    lo, hi = np.percentile(fin, [5, 95])
    if hi - lo <= 0:
        return None
    peaks, _ = find_peaks(np.nan_to_num(seg, nan=lo), distance=int(0.3 * fs), prominence=0.25 * (hi - lo))
    sbp, dbp = [], []
    for k in range(1, len(peaks) - 1):
        a, b = peaks[k - 1], peaks[k + 1]
        trough_before = a + int(np.argmin(seg[a : peaks[k]]))
        trough_after = peaks[k] + int(np.argmin(seg[peaks[k] : b]))
        beat = seg[trough_before:trough_after]
        if not np.isfinite(beat).all() or beat.size < 0.3 * fs or beat.size > 2.0 * fs:
            continue
        sbp.append(float(beat.max()))
        dbp.append(float(beat.min()))
    if len(sbp) < 5:
        return None
    return {"sbp": statistics.mean(sbp), "dbp": statistics.mean(dbp), "n": len(sbp)}


# ---- reference B: the original scipy implementation ----------------------------------------------------------------------
def load_original():
    src = Path.home() / "world_model" / "hemoworld" / "beats.py"
    if not src.exists():
        return None
    pkg = types.ModuleType("hemoworld")
    pkg.__path__ = [str(src.parent)]
    sub = types.ModuleType("hemoworld.pulsedb")
    sub.FS_HZ = 125.0
    sys.modules["hemoworld"], sys.modules["hemoworld.pulsedb"] = pkg, sub
    spec = importlib.util.spec_from_file_location("hemoworld.beats", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_b(mod, abp: np.ndarray, ecg: np.ndarray, fs: float, t0: float, t1: float) -> dict | None:
    if mod is None:
        return None
    x = np.asarray(abp, dtype=np.float64)
    r, _ = mod.detect_r_peaks(np.asarray(ecg, dtype=np.float64), fs)
    feet, peaks = mod.detect_abp_pulses_guided(x, r, fs)
    i0, i1 = int(t0 * fs), int(t1 * fs)
    sbp, dbp = [], []
    for k in range(len(feet) - 1):
        a, b = int(feet[k]), int(feet[k + 1])
        if a < i0 or b > i1:
            continue
        beat = x[a:b]
        if not np.isfinite(beat).all() or beat.size < 0.3 * fs or beat.size > 2.0 * fs:
            continue
        sbp.append(float(beat.max()))
        dbp.append(float(x[a]))
    if len(sbp) < 5:
        return None
    return {"sbp": statistics.mean(sbp), "dbp": statistics.mean(dbp), "n": len(sbp)}


# ---- plugin under test -------------------------------------------------------------------------------------------------
def plugin(abp, ecg, fs, unit, t0, t1, guided=False) -> dict:
    chans = [Channel("ABP", fs, np.asarray(abp, dtype=np.float32), unit)]
    if ecg is not None:
        chans.insert(0, Channel("ECG", fs, np.asarray(ecg, dtype=np.float32), "mV"))
    s = Signal("v", chans)
    register_signal(s)
    ch = s.channel("ABP")
    x = ch.data[int(t0 * fs) : int(t1 * fs)].astype(np.float64)
    fs_, fd = (abp_sbp_ecg, abp_dbp_ecg) if guided else (abp_sbp, abp_dbp)
    rs, rd = fs_(x, fs, unit, t0, ch=ch), fd(x, fs, unit, t0, ch=ch)
    rs.pop("_artifact", None)
    rd.pop("_artifact", None)
    return {"sbp": rs["value"], "dbp": rd["value"], "n": rs["counts"]["accepted"], "rejected": rs["counts"]["rejected"],
            "reasons": rs["counts"]["rejection_reasons"], "warnings": [w["code"] for w in rs["warnings"]],
            "status": rs.get("status", "ok")}  # fmt: skip


def synthetic_sweep(seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(40):
        fs = float(rng.choice([100, 125, 250, 500, 1000]))
        hr = float(rng.uniform(45, 150))
        sbp = float(rng.uniform(85, 200))
        dbp = float(rng.uniform(40, min(110, sbp - 25)))
        noise = float(rng.choice([0.0, 0.3, 1.0, 2.0]))
        hrv = float(rng.choice([0.0, 0.03, 0.08]))
        slope = float(rng.choice([0.0, 0.1, -0.2]))
        art = rng.choice(["none", "flat", "clip", "drop", "nan", "spikes"])
        kw = {}
        if art == "flat":
            kw["flat"] = (25.0, 29.0)
        if art == "clip":
            kw["clip_at"] = sbp - 5
        if art == "drop":
            kw["drop_beats"] = (15, 16)
        if art == "nan":
            kw["nan_at"] = (33.0, 34.0)
        if art == "spikes":
            kw["spikes"] = (22.5, 37.1)
        abp, ecg, beats = synth_abp(fs=fs, duration_s=60, hr_bpm=hr, sbp=sbp, dbp=dbp, seed=k, sbp_slope_per_s=slope,
                                    noise=noise, hrv=hrv, **kw)  # fmt: skip
        t0, t1 = 10.0, 50.0
        inner = [b for b in beats if b["t_foot_s"] >= t0 and b["t_end_s"] <= t1]
        if art == "flat":
            inner = [b for b in inner if b["t_end_s"] <= 25 or b["t_foot_s"] >= 29]
        if art == "nan":
            inner = [b for b in inner if b["t_end_s"] <= 33 or b["t_foot_s"] >= 34]
        truth = {"sbp": statistics.mean(b["sbp"] for b in inner), "dbp": statistics.mean(b["dbp"] for b in inner)}
        p = plugin(abp, ecg, fs, "mmHg", t0, t1)
        g = plugin(abp, ecg, fs, "mmHg", t0, t1, guided=True)
        a = reference_a(abp, fs, t0, t1)
        rows.append({
            "case": k, "fs": fs, "hr": round(hr, 1), "sbp": round(sbp, 1), "dbp": round(dbp, 1), "noise": noise,
            "hrv": hrv, "slope": slope, "artefact": art, "truth": truth, "plugin": p, "plugin_ecg": g, "ref_a": a,
            "err_sbp": None if p["sbp"] is None else round(p["sbp"] - truth["sbp"], 3),
            "err_dbp": None if p["dbp"] is None else round(p["dbp"] - truth["dbp"], 3),
            "err_sbp_ecg": None if g["sbp"] is None else round(g["sbp"] - truth["sbp"], 3),
            "err_a_sbp": None if a is None else round(a["sbp"] - truth["sbp"], 3),
        })  # fmt: skip
    return rows


def dev_cases(mod, limit=None) -> list[dict]:
    root = ROOT / "eval" / "cases" / "p9level2"
    if not root.exists():
        return []
    rows = []
    for d in sorted(root.iterdir()):
        v = json.loads((d / "visible.json").read_text())
        names = [c["name"] for c in v["channels"]]
        if "ABP" not in names:
            continue
        arr = np.load(d / "signal.npy", allow_pickle=False)
        fs = v["channels"][0]["fs"]
        abp = arr[:, names.index("ABP")]
        ecg = arr[:, names.index("ECG")] if "ECG" in names else None
        unit = v["channels"][names.index("ABP")]["unit"]
        dur = v["dur_s"]
        windows = [(10.0, min(70.0, dur - 10)), (max(0.0, dur / 2 - 30), dur / 2 + 30)]
        if dur > 200:
            windows.append((dur - 130.0, dur - 10.0))
        for t0, t1 in windows:
            if t1 - t0 < 30:
                continue
            p = plugin(abp, ecg, fs, unit, t0, t1)
            g = plugin(abp, ecg, fs, unit, t0, t1, guided=True) if ecg is not None else None
            a = reference_a(abp, fs, t0, t1)
            b = reference_b(mod, abp, ecg, fs, t0, t1) if (ecg is not None and mod is not None) else None
            rows.append({"case": d.name[:8], "fs": fs, "unit": unit, "window": [t0, t1], "plugin": p, "plugin_ecg": g,
                         "ref_a": a, "ref_b": b})  # fmt: skip
        if limit and len(rows) >= limit:
            break
    return rows


def summarize(synth: list[dict], dev: list[dict]) -> dict:
    def q(xs):
        xs = [abs(x) for x in xs if x is not None]
        return {"n": len(xs), "median": round(statistics.median(xs), 3) if xs else None,
                "p95": round(float(np.percentile(xs, 95)), 3) if xs else None, "max": round(max(xs), 3) if xs else None}  # fmt: skip

    clean = [r for r in synth if r["artefact"] == "none"]
    art = [r for r in synth if r["artefact"] != "none"]
    out = {
        "synthetic": {
            "clean_abs_err_sbp": q([r["err_sbp"] for r in clean]),
            "clean_abs_err_dbp": q([r["err_dbp"] for r in clean]),
            "artefact_abs_err_sbp": q([r["err_sbp"] for r in art]),
            "artefact_abs_err_dbp": q([r["err_dbp"] for r in art]),
            "ecg_guided_abs_err_sbp": q([r["err_sbp_ecg"] for r in synth]),
            "reference_a_abs_err_sbp": q([r["err_a_sbp"] for r in synth]),
            "limitations_returned": sum(1 for r in synth if r["plugin"]["sbp"] is None),
            "by_artefact": {
                a: {
                    "n": sum(1 for r in synth if r["artefact"] == a),
                    "rejected_beats_total": sum(r["plugin"]["rejected"] for r in synth if r["artefact"] == a),
                    "reasons": sorted({k for r in synth if r["artefact"] == a for k in r["plugin"]["reasons"]}),
                }
                for a in sorted({r["artefact"] for r in synth})
            },
        },  # fmt: skip
    }
    if dev:
        pa = [r["plugin"]["sbp"] - r["ref_a"]["sbp"] for r in dev if r["plugin"]["sbp"] is not None and r["ref_a"]]
        pb = [r["plugin"]["sbp"] - r["ref_b"]["sbp"] for r in dev if r["plugin"]["sbp"] is not None and r["ref_b"]]
        ab = [r["ref_a"]["sbp"] - r["ref_b"]["sbp"] for r in dev if r["ref_a"] and r["ref_b"]]
        pad = [r["plugin"]["dbp"] - r["ref_a"]["dbp"] for r in dev if r["plugin"]["dbp"] is not None and r["ref_a"]]
        pe = [r["plugin_ecg"]["sbp"] - r["plugin"]["sbp"] for r in dev
              if r["plugin_ecg"] and r["plugin_ecg"]["sbp"] is not None and r["plugin"]["sbp"] is not None]  # fmt: skip
        agree = [r for r in dev if r["ref_a"] and r["ref_b"] and abs(r["ref_a"]["sbp"] - r["ref_b"]["sbp"]) <= 1.5]
        within = sum(
            1 for r in agree if r["plugin"]["sbp"] is not None and abs(r["plugin"]["sbp"] - r["ref_a"]["sbp"]) <= 3.0
        )
        out["development"] = {
            "windows": len(dev), "cases": len({r["case"] for r in dev}),
            "plugin_vs_A_abs_sbp": q(pa), "plugin_vs_B_abs_sbp": q(pb), "A_vs_B_abs_sbp": q(ab),
            "plugin_vs_A_abs_dbp": q(pad), "ecg_guided_vs_pulse_abs_sbp": q(pe),
            "windows_where_A_and_B_agree_within_1.5": len(agree), "plugin_within_3_of_agreeing_refs": within,
            "plugin_limitations": sum(1 for r in dev if r["plugin"]["sbp"] is None),
            "both_refs_none": sum(1 for r in dev if not r["ref_a"] and not r["ref_b"]),
        }  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/signalmind/pressure-validation.json")
    ap.add_argument("--no-dev", action="store_true")
    args = ap.parse_args()
    mod = load_original()
    synth = synthetic_sweep()
    dev = [] if args.no_dev else dev_cases(mod)
    summary = summarize(synth, dev)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "synthetic": synth, "development": dev}, indent=1, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
