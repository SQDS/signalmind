"""Long-record cases (the hypothesized home turf of TS-MCP).

- LTAFDB (PhysioNet Long-Term AF Database, 24-25 h two-lead ECG at 128 Hz, ODC-By): ground truth from the
  expert .atr beat and rhythm annotations. Questions: 24 h mean heart rate, AF burden, start of the longest AF episode.
  Signal .dat files are downloaded on demand into data/ltafdb (never into the repo).
- Synthetic 24 h pulse train (fs 100 Hz) with exactly one hidden event: a 20 s sensor artifact, or a 3 min episode
  at a clearly higher rate. Exact ground truth; a needle-in-a-haystack test for long records.
Record ids and the dataset name are never shown to the model.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

LTAF = Path(os.environ.get("LTAFDB_DIR", Path(__file__).resolve().parents[2] / "data" / "ltafdb"))
URL = "https://physionet.org/files/ltafdb/1.0.0"
BEATS = set("NLRBAaJSVrFejnE/fQ")
AF_LABELS = ("(AFIB", "(AFL")
EPISODE_MERGE_GAP_S = 30.0  # interruptions shorter than this (e.g. a 0.9 s VT run inside AF) do not end an AF episode


def merge_episodes(segs: list[tuple[float, float]], gap: float = EPISODE_MERGE_GAP_S) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(segs):
        if out and a - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _ensure_dat(rec: str) -> None:
    dat = LTAF / f"{rec}.dat"
    if not dat.exists() or dat.stat().st_size < 1_000_000:
        subprocess.run(["curl", "-s", "-m", "900", "-o", str(dat), f"{URL}/{rec}.dat"], check=True)


def _ann(rec: str):
    import wfdb

    a = wfdb.rdann(str(LTAF / rec), "atr")
    h = wfdb.rdheader(str(LTAF / rec))
    return a, h


def af_segments(a, fs: float, n: int) -> list[tuple[float, float]]:
    marks = [
        (s / fs, x.strip("\x00")) for s, sym, x in zip(a.sample, a.symbol, a.aux_note, strict=True) if sym == "+" and x
    ]
    segs = []
    for i, (t, lab) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n / fs
        if lab in AF_LABELS:
            segs.append((t, end))
    return segs


def record_facts(rec: str) -> dict | None:
    try:
        a, h = _ann(rec)
    except Exception:
        return None
    fs, n = h.fs, h.sig_len
    beats = np.array([s for s, sym in zip(a.sample, a.symbol, strict=True) if sym in BEATS])
    if beats.size < 100:
        return None
    segs = af_segments(a, fs, n)
    af_time = sum(e - s for s, e in segs)
    episodes = merge_episodes(segs)
    longest = max(episodes, key=lambda se: se[1] - se[0]) if episodes else None
    return {
        "rec": rec, "fs": fs, "dur_s": n / fs,
        "bpm": float(60 * (beats.size - 1) / ((beats[-1] - beats[0]) / fs)),
        "af_percent": float(100 * af_time / (n / fs)),
        "longest_af": None if longest is None else {"start": float(longest[0]), "dur_s": float(longest[1] - longest[0])},
        "n_af_segments": len(segs),
    }  # fmt: skip


def _signal(rec: str) -> np.ndarray:
    import wfdb

    _ensure_dat(rec)
    r = wfdb.rdrecord(str(LTAF / rec), channels=[0])
    return r.p_signal[:, 0].astype(np.float32)


Q_HR = "This is a single-lead ECG recorded over about a day. What is the average heart rate over the whole recording, in beats per minute?"
Q_BURDEN = (
    "This is a single-lead ECG recorded over about a day. For what percentage of the total recording time is the heart "
    "in atrial fibrillation or atrial flutter (the AF burden)?"
)
Q_LONGEST = (
    "This is a single-lead ECG recorded over about a day. When does the LONGEST continuous episode of atrial "
    "fibrillation or flutter begin? If there is no atrial fibrillation or flutter at all, answer null."
)


def select_ltafdb(seed: int, n_each: int = 4, records: list[str] | None = None) -> list[tuple[str, dict]]:
    """Deterministic record choice from annotations only (no signal download needed).
    `records` restricts the pool (e.g. the sealed test_ext split); default: every record with annotations."""
    rng = np.random.default_rng(seed)
    recs = sorted(records) if records else sorted({p.stem for p in LTAF.glob("*.atr")})
    facts = [f for f in (record_facts(r) for r in recs) if f]
    rng.shuffle(facts)
    out, used = [], set()

    def take(pred, k):
        got = []
        for f in facts:
            if f["rec"] not in used and pred(f):
                used.add(f["rec"])
                got.append(f)
                if len(got) == k:
                    break
        return got

    for f in take(lambda f: True, n_each):
        out.append(("hr", f))
    for f in take(lambda f: 5 < f["af_percent"] < 95, n_each):
        out.append(("burden", f))
    for f in take(lambda f: f["longest_af"] and f["longest_af"]["dur_s"] > 300 and f["af_percent"] < 90, n_each):
        out.append(("longest", f))
    return out


def build_ltafdb(seed: int, n_each: int = 4, records: list[str] | None = None):
    out = select_ltafdb(seed, n_each, records)
    cases = []
    for i, (kind, f) in enumerate(out):
        x = _signal(f["rec"])
        meta = {"record": f["rec"], "dur_s": f["dur_s"]}
        if kind == "hr":
            cases.append((f"ltaf{i:03d}", "mean_hr", "-", x, f["fs"], {"bpm": f["bpm"], **meta}, Q_HR))
        elif kind == "burden":
            cases.append((f"ltaf{i:03d}", "af_burden", "-", x, f["fs"], {"percent": f["af_percent"], **meta}, Q_BURDEN))
        else:
            ans = {"t": f["longest_af"]["start"], "longest_dur_s": f["longest_af"]["dur_s"], **meta}
            cases.append((f"ltaf{i:03d}", "longest_af", "-", x, f["fs"], ans, Q_LONGEST))
    return cases


# ---------------------------------------------------------------- synthetic 24 h
FS_LONG = 100.0
DUR_LONG = 24 * 3600.0
Q_ART = (
    "This is a day-long recording from a wearable pulse sensor. Somewhere in it there may be a segment where the sensor "
    "output is stuck at a constant value or saturated for tens of seconds. When does that segment start? Answer null if "
    "there is no such segment."
)
Q_EPI = (
    "This is a day-long recording from a wearable pulse sensor. Somewhere in it the pulse rate rises abruptly and stays "
    "clearly elevated for a few minutes before returning to normal. When does that episode start?"
)


def _pulse_train(rng, n: int, rate_hz: np.ndarray) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(rate_hz) / FS_LONG
    ph = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    return (np.exp(-(((ph - 0.3) / 0.05) ** 2)) + 0.35 * np.exp(-(((ph - 0.6) / 0.09) ** 2)) - 0.25).astype(np.float32)


def build_synthetic_long(seed: int, n_cases: int = 5):
    rng = np.random.default_rng(seed)
    n = int(DUR_LONG * FS_LONG)
    t = np.arange(n, dtype=np.float64) / FS_LONG
    cases = []
    kinds = (["artifact", "artifact", "none", "episode", "episode"] * 4)[:n_cases]
    for i, kind in enumerate(kinds):
        base = rng.uniform(1.0, 1.25)  # 60-75 bpm
        rate = base * (1 + 0.12 * np.sin(2 * np.pi * t / 86400 + rng.uniform(0, 6.28)))  # circadian drift
        rate += 0.03 * np.sin(2 * np.pi * t / rng.uniform(600, 1800))  # slower wander
        t0 = float(np.round(rng.uniform(0.05, 0.95) * DUR_LONG, 1))
        if kind == "episode":
            m = (t >= t0) & (t < t0 + 180)
            rate[m] *= rng.uniform(1.6, 1.9)
        x = _pulse_train(rng, n, rate)
        x += (0.05 * rng.standard_normal(n)).astype(np.float32)
        if kind == "artifact":
            i0, L = int(t0 * FS_LONG), int(20 * FS_LONG)
            x[i0 : i0 + L] = x[i0] if rng.random() < 0.5 else 1.3
        if kind == "episode":
            cases.append((f"long{i:03d}", "episode_start", kind, x, FS_LONG, {"t": t0}, Q_EPI))
        else:
            cases.append(
                (f"long{i:03d}", "long_artifact", kind, x, FS_LONG, {"t": t0 if kind == "artifact" else None}, Q_ART)
            )
    return cases
