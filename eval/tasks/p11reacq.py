"""Build the p11 evidence-reacquisition suites (pilot: 2 items; main: 18 items). No model is run here.

  .venv/bin/python -m eval.tasks.p11reacq --build
  .venv/bin/python -m eval.tasks.p11reacq --check

Composition (main): 6 natural tasks (3 ABP interval means, 1 ABP change, 2 PPG pulse rates), 6 controlled evidence-gap
tasks (4 ABP, 2 PPG: a dropout is inserted at a position the question does not give; the quantity is asked over the
60 s immediately after the dropout ends, so the first obvious measurement is not the answer and the boundary must be
located first), 6 controls (3 ordinarily answerable: 2 ABP means, 1 PPG rate; 3 genuinely insufficient: ABP NaN over
the whole asked window x2, PPG constant over the whole asked window x1). Every question carries the same
"if the evidence is insufficient, say so" sentence, so no item type is cued by wording.

Sources: opened development records only. ABP items: real diabetes/CVES records (ECG+ABP[+CBFV]), windows that do not
overlap the p10 suites. PPG items: PulseDB records whose PPG and ECG columns are unperturbed (real records, or records
whose Level 2 perturbation touched only the ABP column) exported as ECG+PPG. References never use the plugin:
ABP from implementations A (scipy peaks) and B (original guided detector); PPG rate from the ECG R-peaks of two
independent implementations (scipy band-pass + find_peaks; original detector), retained only when they agree within
1.5 bpm and a scipy PPG peak rate agrees with them (so the pulse rate is well defined). Gap tasks give both arms the
same record; nothing is selected by arm outcome. TaskSpecs use public question fields only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.tasks.p10abp import SRC, agree, refs, sha, sources, write_suite  # noqa: E402
from eval.validate_pressure import load_original  # noqa: E402
from tsmcp.runtime.taskspec import Budgets, QuantitySpec, TaskSpec, abp_interval_spec  # noqa: E402

SEED = 20261029
TOL_P, TOL_R = 3.0, 3.0  # mmHg, bpm
HALF_P, HALF_R = TOL_P / 2, TOL_R / 2
GAP_TOL_S = 2.0  # how close the claimed window start must be to the true dropout end (grading)
AFTER_S = 60.0
GENERATOR_VERSION = "p11reacq.v1"
INSUFF = "If the evidence is insufficient for a quantity, say so instead of giving a number. "


# ---- PPG sources and references ----------------------------------------------------------------------------------------
def ppg_sources() -> list[dict]:
    man = json.loads((ROOT / "eval/tasks/p9level2_manifest.json").read_text())["questions"]
    out = []
    for q in man:
        if q["dataset"] != "pulsedb_mimic" or "PPG" not in q["channels"] or "ECG" not in q["channels"]:
            continue
        pert = q.get("perturbation") or {}
        if q["signal_kind"] != "real_signal" and pert.get("channel") != "ABP":
            continue  # PPG or ECG column altered: not usable as a natural PPG source
        d = SRC / q["case_id"]
        if not (d / "signal.npy").exists():
            continue
        arr = np.load(d / "signal.npy", allow_pickle=False)
        v = json.loads((d / "visible.json").read_text())
        names = [c["name"] for c in v["channels"]]
        keep = [names.index("ECG"), names.index("PPG")]
        vis = {"channels": [v["channels"][i] for i in keep], "dur_s": v["dur_s"], "file": "signal.npy"}
        out.append({"src": q["case_id"], "dataset": q["dataset"], "subject": q["subject_key"], "fs": v["channels"][0]["fs"],
                    "dur": v["dur_s"], "names": ["ECG", "PPG"], "visible": vis, "arr": np.ascontiguousarray(arr[:, keep]),
                    "ppg_intact_reason": "real record" if q["signal_kind"] == "real_signal" else "perturbation on ABP only"})  # fmt: skip
    out.sort(key=lambda r: r["src"])
    return out


def ecg_rate_a(ecg: np.ndarray, fs: float, t0: float, t1: float) -> dict | None:
    """Reference A': scipy band-pass + find_peaks R-peak rate over [t0, t1)."""
    from scipy.signal import butter, find_peaks, sosfiltfilt

    x = np.asarray(ecg, dtype=np.float64)
    sos = butter(3, [5, 20], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, np.nan_to_num(x))
    i0, i1 = int(t0 * fs), int(t1 * fs)
    seg = y[i0:i1]
    if seg.size < 5 * fs:
        return None
    sign = 1.0 if np.percentile(seg, 99.5) >= -np.percentile(seg, 0.5) else -1.0
    seg = sign * seg
    peaks, _ = find_peaks(seg, distance=int(0.3 * fs), prominence=0.4 * np.percentile(seg, 99.5))
    return _rate(peaks, fs)


def ecg_rate_b(mod, ecg: np.ndarray, fs: float, t0: float, t1: float) -> dict | None:
    """Reference B': the original (scipy) R-peak detector."""
    if mod is None:
        return None
    r, _ = mod.detect_r_peaks(np.asarray(ecg, dtype=np.float64), fs)
    i0, i1 = int(t0 * fs), int(t1 * fs)
    return _rate(r[(r >= i0) & (r < i1)] - i0, fs)


def ppg_peak_rate(ppg: np.ndarray, fs: float, t0: float, t1: float) -> dict | None:
    """Consistency check C': scipy peaks on the PPG itself (not the toolkit's period estimator)."""
    from scipy.signal import find_peaks

    x = np.asarray(ppg, dtype=np.float64)
    seg = x[int(t0 * fs) : int(t1 * fs)]
    if seg.size < 5 * fs or not np.isfinite(seg).any() or np.nanstd(seg) == 0:
        return None
    lo, hi = np.nanpercentile(seg, [5, 95])
    peaks, _ = find_peaks(np.nan_to_num(seg, nan=lo), distance=int(0.3 * fs), prominence=0.25 * (hi - lo))
    return _rate(peaks, fs)


def _rate(peaks: np.ndarray, fs: float) -> dict | None:
    if len(peaks) < 8:
        return None
    span = (peaks[-1] - peaks[0]) / fs
    return {"bpm": 60.0 * (len(peaks) - 1) / span, "n": int(len(peaks))} if span > 0 else None


def rate_refs(rec, t0, t1, mod):
    names = rec["names"]
    ecg, ppg = rec["arr"][:, names.index("ECG")], rec["arr"][:, names.index("PPG")]
    return (
        ecg_rate_a(ecg, rec["fs"], t0, t1),
        ecg_rate_b(mod, ecg, rec["fs"], t0, t1),
        ppg_peak_rate(ppg, rec["fs"], t0, t1),
    )


def rate_ok(a, b, c) -> bool:
    return (
        a is not None
        and b is not None
        and c is not None
        and abs(a["bpm"] - b["bpm"]) <= HALF_R
        and abs(c["bpm"] - a["bpm"]) <= HALF_R
    )


# ---- questions and specs ------------------------------------------------------------------------------------------------
def layout(names, dur):
    return (f"Times are seconds from the start of this {dur:g} s case. The array is samples x channels: "
            + ", ".join(f"column {i} = {n}" for i, n in enumerate(names)) + ". When loading, assign these channel names in this order.")  # fmt: skip


def q_abp_mean(t0, t1, names, dur):
    return (f"What are the mean per-beat systolic (SBP) and diastolic (DBP) arterial pressures in mmHg over {t0:g}-{t1:g} s "
            f"of the ABP channel? Report sbp_mean and dbp_mean as means over complete beats. " + INSUFF + layout(names, dur))  # fmt: skip


def q_abp_change(a, b, names, dur):
    return (f"By how much does the mean per-beat systolic arterial pressure (mmHg) change from {a[0]:g}-{a[1]:g} s (baseline) "
            f"to {b[0]:g}-{b[1]:g} s (later minus baseline) in the ABP channel? Report sbp_change. " + INSUFF + layout(names, dur))  # fmt: skip


def q_ppg_rate(t0, t1, names, dur):
    return (f"What is the mean pulse rate in beats per minute represented by the PPG channel over {t0:g}-{t1:g} s? Report "
            f"pulse_rate_bpm from the PPG signal and check it against the ECG rhythm. " + INSUFF + layout(names, dur))  # fmt: skip


def q_abp_gap(t0, t1, names, dur):
    return (f"Within {t0:g}-{t1:g} s the ABP channel contains a stretch where the signal drops out (no pulsatile waveform). "
            f"Report the mean per-beat systolic (sbp_mean) and diastolic (dbp_mean) arterial pressures in mmHg over the "
            f"{AFTER_S:g} s immediately after that dropout ends, as means over complete beats, and state the window you "
            f"used. " + INSUFF + layout(names, dur))  # fmt: skip


def q_ppg_gap(t0, t1, names, dur):
    return (f"Within {t0:g}-{t1:g} s the PPG channel contains a stretch where the signal drops out (flat, no pulses). Report "
            f"the mean pulse rate in beats per minute (pulse_rate_bpm) represented by the PPG over the {AFTER_S:g} s "
            f"immediately after that dropout ends, checked against the ECG rhythm, and state the window you used. "
            + INSUFF + layout(names, dur))  # fmt: skip


def spec_abp(case_id, question, names, interval, quantities, gap=False):
    spec = abp_interval_spec(case_id, question, "ABP", interval, quantities=quantities, ecg_channel="ECG" if "ECG" in names else None,
                             budgets=Budgets())  # fmt: skip
    if gap:
        spec.interval = None
        spec.conventions["window"] = (
            f"the {AFTER_S:g} s immediately after the end of the ABP dropout inside the stated range"
        )
        spec.conventions["window_rule"] = {"type": "after_dropout", "channel": "ABP", "length_s": AFTER_S,
                                           "tolerance_s": GAP_TOL_S}  # fmt: skip
    return spec


def spec_ppg(case_id, question, interval, gap=False):
    return TaskSpec(
        task_id=case_id, request_text=question,
        quantities=[QuantitySpec("pulse_rate_bpm", "rate.from_period.v1", "bpm")],
        channel_roles={"pulse": "PPG", "ecg": "ECG"}, interval=None if gap else (float(interval[0]), float(interval[1])),
        declared_sync={"ECG|PPG": "index_aligned_declared"}, optional_roles=("ecg",),
        conventions={"rate": "rhythm-level pulse rate of the PPG channel, cross-checked against the ECG rhythm",
                     **({"window": f"the {AFTER_S:g} s immediately after the end of the PPG dropout inside the stated range",
                         "window_rule": {"type": "after_dropout", "channel": "PPG", "length_s": AFTER_S,
                                         "tolerance_s": GAP_TOL_S}} if gap else {})},
    )  # fmt: skip


# ---- item builders -------------------------------------------------------------------------------------------------------
def finish(rec, kind, category, question, expected, tolerance, references, intervals, spec, control=None, extra=None):
    arr = rec["arr"]
    body = np.ascontiguousarray(arr.astype(np.float32)).tobytes()
    case_id = sha(body + question.encode())[:24]
    spec.task_id = case_id
    item = {
        "case_id": case_id, "kind": kind, "category": category, "source_case": rec["src"][:8], "dataset": rec["dataset"],
        "subject_key": rec["subject"], "fs": rec["fs"], "duration_s": rec["dur"], "channels": rec["names"],
        "intervals": [list(i) for i in intervals], "question": question, "answer_fields": list(expected),
        "expected": expected, "tolerance": tolerance, "references": references, "control": control,
        "arr": arr, "visible": rec["visible"], "spec": spec.to_dict(), "generator": GENERATOR_VERSION,
    }  # fmt: skip
    if extra:
        item.update(extra)
    return item


def abp_mean_item(rec, t0, t1, mod, category="natural", kind="interval_mean"):
    a, b = refs(rec, t0, t1, mod)
    if not (agree(a, b, "sbp") and agree(a, b, "dbp")) or a["n"] < 30:
        return None
    q = q_abp_mean(t0, t1, rec["names"], rec["dur"])
    return finish(rec, kind, category, q, {"sbp_mean": round(a["sbp"], 2), "dbp_mean": round(a["dbp"], 2)},
                  {"sbp_mean": TOL_P, "dbp_mean": TOL_P}, {"A": a, "B": b}, [(t0, t1)],
                  spec_abp("x", q, rec["names"], (t0, t1), ("sbp_mean", "dbp_mean")))  # fmt: skip


def abp_change_item(rec, iv, mod):
    (a0, a1), (b0, b1) = iv
    aa, ab = refs(rec, a0, a1, mod)
    ba, bb = refs(rec, b0, b1, mod)
    if not (agree(aa, ab, "sbp") and agree(ba, bb, "sbp")):
        return None
    ch_a, ch_b = ba["sbp"] - aa["sbp"], bb["sbp"] - ab["sbp"]
    if abs(ch_a - ch_b) > HALF_P or abs(ch_a) < 4.0:
        return None
    q = q_abp_change((a0, a1), (b0, b1), rec["names"], rec["dur"])
    spec = spec_abp("x", q, rec["names"], (a0, a1), ("sbp_mean",))
    spec.quantities = [QuantitySpec("sbp_change", "abp.sbp.beat_mean_change.v1", "mmHg")]
    spec.interval = None
    spec.conventions["intervals"] = f"baseline {[a0, a1]}, later {[b0, b1]}, change = later - baseline"
    return finish(rec, "change", "natural", q, {"sbp_change": round(ch_a, 2)}, {"sbp_change": TOL_P},
                  {"A": {"baseline": aa, "later": ba, "change": ch_a}, "B": {"baseline": ab, "later": bb, "change": ch_b}},
                  [(a0, a1), (b0, b1)], spec)  # fmt: skip


def ppg_rate_item(rec, t0, t1, mod, category="natural", kind="ppg_rate"):
    a, b, c = rate_refs(rec, t0, t1, mod)
    if not rate_ok(a, b, c):
        return None
    q = q_ppg_rate(t0, t1, rec["names"], rec["dur"])
    return finish(rec, kind, category, q, {"pulse_rate_bpm": round(a["bpm"], 2)}, {"pulse_rate_bpm": TOL_R},
                  {"ecg_A": a, "ecg_B": b, "ppg_peaks_C": c}, [(t0, t1)], spec_ppg("x", q, (t0, t1)))  # fmt: skip


def with_dropout(rec, channel, start, length, mode):
    arr = rec["arr"].copy()
    j = rec["names"].index(channel)
    fs = rec["fs"]
    i0, i1 = int(start * fs), int((start + length) * fs)
    if mode == "constant":
        arr[i0:i1, j] = float(np.nanmedian(arr[i0:i1, j]))
    else:
        arr[i0:i1, j] = np.nan
    return {**rec, "arr": arr}


def abp_gap_item(rec, t0, t1, gap_start, gap_len, mode, mod):
    end = round(gap_start + gap_len, 1)
    rec2 = with_dropout(rec, "ABP", gap_start, gap_len, mode)
    a, b = refs(rec2, end, end + AFTER_S, mod)
    if not (agree(a, b, "sbp") and agree(a, b, "dbp")) or a["n"] < 30:
        return None
    ga, gb = refs(rec2, gap_start, end, mod)  # the dropout itself must carry no beats for either reference
    if ga is not None or gb is not None:
        return None
    whole_a, whole_b = refs(rec2, t0, t1, mod)  # the whole-range shortcut, recorded for transparency
    q = q_abp_gap(t0, t1, rec["names"], rec["dur"])
    return finish(rec2, "abp_gap", "gap", q, {"sbp_mean": round(a["sbp"], 2), "dbp_mean": round(a["dbp"], 2)},
                  {"sbp_mean": TOL_P, "dbp_mean": TOL_P}, {"A": a, "B": b}, [(end, end + AFTER_S)],
                  spec_abp("x", q, rec["names"], (t0, t1), ("sbp_mean", "dbp_mean"), gap=True),
                  control={"type": f"dropout_{mode}", "channel": "ABP", "interval": [gap_start, end], "asked_range": [t0, t1]},
                  extra={"interval_tolerance_s": GAP_TOL_S, "true_dropout_end_s": end,
                         "shortcut_whole_range_A": whole_a, "shortcut_whole_range_B": whole_b})  # fmt: skip


def ppg_gap_item(rec, t0, t1, gap_start, gap_len, mod):
    end = round(gap_start + gap_len, 1)
    rec2 = with_dropout(rec, "PPG", gap_start, gap_len, "constant")
    a, b, c = rate_refs(rec2, end, end + AFTER_S, mod)
    if not rate_ok(a, b, c):
        return None
    if ppg_peak_rate(rec2["arr"][:, 1], rec["fs"], gap_start, end) is not None:
        return None
    whole = rate_refs(rec2, t0, t1, mod)  # the whole-range shortcut, recorded for transparency
    q = q_ppg_gap(t0, t1, rec["names"], rec["dur"])
    return finish(rec2, "ppg_gap", "gap", q, {"pulse_rate_bpm": round(a["bpm"], 2)}, {"pulse_rate_bpm": TOL_R},
                  {"ecg_A": a, "ecg_B": b, "ppg_peaks_C": c}, [(end, end + AFTER_S)], spec_ppg("x", q, (t0, t1), gap=True),
                  control={"type": "dropout_constant", "channel": "PPG", "interval": [gap_start, end], "asked_range": [t0, t1]},
                  extra={"interval_tolerance_s": GAP_TOL_S, "true_dropout_end_s": end,
                         "shortcut_whole_range": {"ecg_A": whole[0], "ecg_B": whole[1], "ppg_peaks_C": whole[2]}})  # fmt: skip


def abp_insufficient_item(rec, t0, t1, mod):
    rec2 = with_dropout(rec, "ABP", t0, t1 - t0, "nan")
    a, b = refs(rec2, t0, t1, mod)
    if a is not None or b is not None:
        return None
    q = q_abp_mean(t0, t1, rec["names"], rec["dur"])
    return finish(rec2, "control_nan_abp", "control_insufficient", q, {"sbp_mean": None, "dbp_mean": None},
                  {"sbp_mean": TOL_P, "dbp_mean": TOL_P}, {"A": None, "B": None}, [(t0, t1)],
                  spec_abp("x", q, rec["names"], (t0, t1), ("sbp_mean", "dbp_mean")),
                  control={"type": "nan_abp", "channel": "ABP", "interval": [t0, t1]})  # fmt: skip


def ppg_insufficient_item(rec, t0, t1, mod):
    rec2 = with_dropout(rec, "PPG", t0, t1 - t0, "constant")
    if ppg_peak_rate(rec2["arr"][:, 1], rec["fs"], t0, t1) is not None:
        return None
    q = q_ppg_rate(t0, t1, rec["names"], rec["dur"])
    return finish(rec2, "control_constant_ppg", "control_insufficient", q, {"pulse_rate_bpm": None}, {"pulse_rate_bpm": TOL_R},
                  {"ecg_A": None, "ecg_B": None, "ppg_peaks_C": None}, [(t0, t1)], spec_ppg("x", q, (t0, t1)),
                  control={"type": "constant_ppg", "channel": "PPG", "interval": [t0, t1]})  # fmt: skip


# ---- planning ------------------------------------------------------------------------------------------------------------
def used_windows() -> dict[str, list]:
    used: dict[str, list] = {}
    for name in ("p10abp", "p10pilot"):
        p = ROOT / "eval/tasks" / f"{name}_manifest.json"
        if p.exists():
            for it in json.loads(p.read_text())["items"]:
                used.setdefault(it["source_case"], []).extend(it["intervals"])
    return used


def free(src, iv, used, taken) -> bool:
    spans = used.get(src[:8], []) + taken.get(src[:8], [])
    return not any(x[0] < s[1] and s[0] < x[1] for x in iv for s in spans)


def plan(abp_recs, ppg_recs, mod, rng):
    used = used_windows()
    taken: dict[str, list] = {}
    subjects_used: set[str] = set()
    items: list[dict] = []

    def take(item, iv):
        taken.setdefault(item["source_case"], []).extend([list(i) for i in iv])
        subjects_used.add(item["subject_key"])
        items.append(item)

    def pick(recs, builder, n, label, long_only=False, need_free=True, new_subject=True):
        got = 0
        order = list(recs)
        rng.shuffle(order)
        for rec in order * 3:
            if got >= n:
                break
            if long_only and rec["dur"] < 500:
                continue
            if new_subject and rec["subject"] in subjects_used:
                continue
            res = builder(rec)
            if res is None:
                continue
            item, iv = res
            if need_free and not free(rec["src"], iv, used, taken):
                continue
            take(item, iv)
            got += 1
        if got < n:
            raise SystemExit(f"could not build {n} {label} items (got {got})")

    def win(rec, length, lo=10.0):
        hi = rec["dur"] - 10.0 - length
        if hi < lo:
            return None
        return round(rng.uniform(lo, hi), 1)

    # natural
    def b_mean(rec):
        s = win(rec, 60.0)
        if s is None:
            return None
        it = abp_mean_item(rec, s, s + 60.0, mod)
        return (it, [(s, s + 60.0)]) if it else None

    def b_change(rec):
        a0 = round(rng.uniform(10, rec["dur"] - 200), 1)
        b0 = round(rng.uniform(a0 + 100, rec["dur"] - 50), 1)
        iv = [(a0, a0 + 40), (b0, b0 + 40)]
        it = abp_change_item(rec, iv, mod)
        return (it, iv) if it else None

    def b_rate(rec):
        s = win(rec, 60.0)
        if s is None:
            return None
        it = ppg_rate_item(rec, s, s + 60.0, mod)
        return (it, [(s, s + 60.0)]) if it else None

    # gaps: asked range covers most of the record; dropout 30 s (180 s records) or 40 s (long records)
    def b_abp_gap(mode):
        def b(rec):
            t0, t1 = (
                (10.0, rec["dur"] - 10.0) if rec["dur"] <= 200 else (round(rng.uniform(10, rec["dur"] - 320), 1), None)
            )
            if t1 is None:
                t1 = t0 + 300.0
            glen = 30.0 if rec["dur"] <= 200 else 40.0
            gs = round(rng.uniform(t0 + 20, t1 - glen - AFTER_S - 5), 1)
            it = abp_gap_item(rec, t0, t1, gs, glen, mode, mod)
            return (it, it["intervals"]) if it else None  # only the answer window must be unused elsewhere

        return b

    def b_ppg_gap(rec):
        t0, t1 = 10.0, rec["dur"] - 10.0
        if rec["dur"] > 200:
            t0 = round(rng.uniform(10, rec["dur"] - 320), 1)
            t1 = t0 + 300.0
        gs = round(rng.uniform(t0 + 20, t1 - 30.0 - AFTER_S - 5), 1)
        it = ppg_gap_item(rec, t0, t1, gs, 30.0, mod)
        return (it, it["intervals"]) if it else None

    def b_abp_insuff(rec):
        s = win(rec, 60.0)
        if s is None:
            return None
        it = abp_insufficient_item(rec, s, s + 60.0, mod)
        return (it, [(s, s + 60.0)]) if it else None

    def b_ppg_insuff(rec):
        s = win(rec, 60.0)
        if s is None:
            return None
        it = ppg_insufficient_item(rec, s, s + 60.0, mod)
        return (it, [(s, s + 60.0)]) if it else None

    def b_mean_ctrl(rec):
        r = b_mean(rec)
        if r:
            r[0]["kind"], r[0]["category"] = "control_interval_mean", "control_answerable"
        return r

    def b_rate_ctrl(rec):
        r = b_rate(rec)
        if r:
            r[0]["kind"], r[0]["category"] = "control_ppg_rate", "control_answerable"
        return r

    pick(abp_recs, b_mean, 3, "ABP mean")
    pick(abp_recs, b_change, 1, "ABP change", long_only=True)
    pick(ppg_recs, b_rate, 2, "PPG rate")
    pick(abp_recs, b_abp_gap("constant"), 3, "ABP gap (constant)", new_subject=False)
    pick(abp_recs, b_abp_gap("nan"), 1, "ABP gap (nan)", new_subject=False)
    pick(ppg_recs, b_ppg_gap, 2, "PPG gap", new_subject=False)
    pick(abp_recs, b_mean_ctrl, 2, "ABP control mean", new_subject=False)
    pick(ppg_recs, b_rate_ctrl, 1, "PPG control rate", new_subject=False)
    pick(abp_recs, b_abp_insuff, 2, "ABP insufficient", new_subject=False)
    pick(ppg_recs, b_ppg_insuff, 1, "PPG insufficient", new_subject=False)
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mod = load_original()
    if mod is None:
        raise SystemExit("reference B (original implementation) is not available")
    abp_recs, ppg_recs = sources(), ppg_sources()
    rng = random.Random(SEED)
    # pilot first: one PPG natural rate and one ABP gap on records held out from main
    pilot_rng = random.Random(SEED + 1)
    pilot = plan_pilot(abp_recs, ppg_recs, mod, pilot_rng)
    held = {it["source_case"] for it in pilot}
    main_items = plan(
        [r for r in abp_recs if r["src"][:8] not in held], [r for r in ppg_recs if r["src"][:8] not in held], mod, rng
    )
    if args.build:
        mp = write_suite("p11pilot", pilot)
        mm = write_suite("p11reacq", main_items)
        print(json.dumps({"pilot": {k: mp[k] for k in ("counts", "subjects", "sources")},
                          "main": {k: mm[k] for k in ("counts", "categories", "subjects", "sources")}}, indent=1))  # fmt: skip
    if args.check:
        for name, items in (("p11pilot", pilot), ("p11reacq", main_items)):
            frozen = json.loads((ROOT / "eval/tasks" / f"{name}_manifest.json").read_text())
            same = [m["case_id"] for m in frozen["items"]] == [g["case_id"] for g in items] and all(
                m["expected"] == g["expected"] for m, g in zip(frozen["items"], items, strict=True)
            )
            files = all(
                sha((ROOT / "eval/cases" / name / m["case_id"] / "signal.npy").read_bytes()) == m["signal_sha256"]
                for m in frozen["items"]
            )
            print(name, "manifest reproduced:", same, "files intact:", files)


def plan_pilot(abp_recs, ppg_recs, mod, rng):
    items = []
    for rec in sorted(ppg_recs, key=lambda r: r["src"], reverse=True):
        s = round(rng.uniform(10, rec["dur"] - 70), 1)
        it = ppg_rate_item(rec, s, s + 60.0, mod)
        if it:
            it["pilot_role"] = "natural_ppg_rate"
            items.append(it)
            break
    for rec in sorted(abp_recs, key=lambda r: r["src"], reverse=True):
        if rec["dur"] > 200:
            continue
        t0, t1 = 10.0, rec["dur"] - 10.0
        gs = round(rng.uniform(t0 + 20, t1 - 30.0 - AFTER_S - 5), 1)
        it = abp_gap_item(rec, t0, t1, gs, 30.0, "constant", mod)
        if it and it["subject_key"] not in {i["subject_key"] for i in items}:
            it["pilot_role"] = "abp_gap"
            items.append(it)
            break
    if len(items) != 2:
        raise SystemExit("could not build the two pilot items")
    return items


if __name__ == "__main__":
    main()
