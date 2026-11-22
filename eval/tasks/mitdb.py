"""MIT-BIH Arrhythmia windows with ground truth from .atr annotations.

de Chazal split: DS1 = dev (tuning), DS2 = test. Paced records excluded. Record IDs, file names and the
dataset name are never shown to the model; windows start at random offsets and time restarts at 0.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MITDB = Path(os.environ.get("MITDB_DIR", "/home/yanghaoxiang24/bp2rr_new/data_process/01_rpeak_detector/mitdb"))
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
BEATS = set("NLRBAaJSVrFejnE/fQ")
WIN_S = 600.0
RHYTHM_NAMES = {
    "(N": "normal sinus rhythm",
    "(AFIB": "atrial fibrillation",
    "(AFL": "atrial flutter",
    "(B": "ventricular bigeminy",
    "(T": "ventricular trigeminy",
    "(VT": "ventricular tachycardia",
    "(SVTA": "supraventricular tachyarrhythmia",
    "(IVR": "idioventricular rhythm",
    "(NOD": "nodal rhythm",
    "(SBR": "sinus bradycardia",
    "(BII": "2nd degree block",
    "(AB": "atrial bigeminy",
    "(PREX": "pre-excitation",
    "(VFL": "ventricular flutter",
    "(P": "paced",
    "(PREX)": "pre-excitation",
}


def _read(rec: int):
    import wfdb

    path = str(MITDB / str(rec))
    h = wfdb.rdheader(path)
    ch = h.sig_name.index("MLII") if "MLII" in h.sig_name else 0
    ann = wfdb.rdann(path, "atr")
    return path, h, ch, ann


def rhythm_segments(ann, fs: float, n: int) -> list[tuple[float, float, str]]:
    marks = [
        (s / fs, a.strip("\x00"))
        for s, sym, a in zip(ann.sample, ann.symbol, ann.aux_note, strict=True)
        if sym == "+" and a
    ]
    segs = []
    for i, (t, lab) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n / fs
        segs.append((t, end, lab))
    return segs


def _signal(path: str, ch: int, s0: int, s1: int) -> np.ndarray:
    import wfdb

    r = wfdb.rdrecord(path, channels=[ch], sampfrom=s0, sampto=s1)
    return r.p_signal[:, 0].astype(np.float32)


def mean_hr_case(rng, rec: int):
    path, h, ch, ann = _read(rec)
    fs, n = h.fs, h.sig_len
    t0 = float(np.round(rng.uniform(0, n / fs - WIN_S - 1), 1))
    s0, s1 = int(t0 * fs), int((t0 + WIN_S) * fs)
    b = np.array([s for s, sym in zip(ann.sample, ann.symbol, strict=True) if sym in BEATS and s0 <= s < s1])
    bpm = 60.0 * (len(b) - 1) / ((b[-1] - b[0]) / fs)
    q = "This is a single-lead ECG. What is the average heart rate over the whole recording, in beats per minute?"
    return _signal(path, ch, s0, s1), fs, {"bpm": float(bpm), "record": rec, "offset_s": t0}, q


def rhythm_change_case(rng, rec: int, want_change: bool):
    """Returns None if the record cannot provide the requested kind of window."""
    path, h, ch, ann = _read(rec)
    fs, n = h.fs, h.sig_len
    segs = rhythm_segments(ann, fs, n)
    changes = [s[0] for s in segs[1:] if s[2] != segs[segs.index(s) - 1][2]]
    dur = n / fs
    if want_change:
        cands = [c for c in changes if 60 < c < dur - 60]
        if not cands:
            return None
        c = float(rng.choice(cands))
        lo, hi = max(0.0, c - WIN_S + 60), min(c - 60, dur - WIN_S)
        if hi < lo:
            return None
        t0 = float(np.round(rng.uniform(lo, hi), 1))
    else:
        quiet = [(a, b) for a, b, _ in segs if b - a >= WIN_S + 2]
        if not quiet:
            return None
        a, b = quiet[int(rng.integers(len(quiet)))]
        t0 = float(np.round(rng.uniform(a + 1, b - WIN_S - 1), 1))
    s0, s1 = int(t0 * fs), int((t0 + WIN_S) * fs)
    inside = [c - t0 for c in changes if t0 < c < t0 + WIN_S]
    first = float(np.round(inside[0], 2)) if inside else None
    start_lab = next((lab for a, b, lab in segs if a <= t0 < b), None)
    q = (
        "This is a single-lead ECG. Does the heart rhythm change type during the recording — for example between "
        "normal sinus rhythm, atrial fibrillation or flutter, ventricular bigeminy or trigeminy, or ventricular "
        "tachycardia? If yes, at what time does the first change occur? Isolated ectopic beats are not a rhythm change."
    )
    ans = {
        "t": first,
        "record": rec,
        "offset_s": t0,
        "start_rhythm": RHYTHM_NAMES.get(start_lab, start_lab),
        "all_changes": inside,
    }
    return _signal(path, ch, s0, s1), fs, ans, q


def build(seed: int, records: list[int], n_hr: int = 5, n_rhythm: int = 5):
    rng = np.random.default_rng(seed)
    recs = list(rng.permutation(records))
    out, i = [], 0
    for rec in recs[:n_hr]:
        x, fs, ans, q = mean_hr_case(rng, rec)
        out.append((f"ecg{i:03d}", "mean_hr", "-", x, fs, ans, q))
        i += 1
    want = [True, False] * n_rhythm
    for rec in recs[n_hr:] + recs[:n_hr]:
        if sum(1 for o in out if o[1] == "rhythm_change") >= n_rhythm:
            break
        w = want[sum(1 for o in out if o[1] == "rhythm_change")]
        r = rhythm_change_case(rng, rec, w)
        if r is None:
            continue
        x, fs, ans, q = r
        out.append((f"ecg{i:03d}", "rhythm_change", "change" if w else "none", x, fs, ans, q))
        i += 1
    return out
