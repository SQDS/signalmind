"""Synthetic cases with exact ground truth.

Two generator *families* so the test set is not the dev set in disguise:
- dev:  sinusoid carriers, white noise
- test: pulse-train / resonant-AR carriers, 1/f-ish noise (seed sealed: pass --test-seed at build time)
Includes negatives (no change, no artifact) and an adversarial slow amplitude modulation that is NOT a change.
Question types include spike_count, which no P1 tool answers directly.
"""

from __future__ import annotations

import numpy as np

FS = 250.0
DUR = 600.0


def _carrier(rng, family: str, t: np.ndarray, f: np.ndarray) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f) / FS
    if family == "dev":
        return np.sin(phase)
    # test family: narrow periodic pulses (PPG/ECG-like) with a smaller secondary bump
    ph = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    return np.exp(-(((ph - 0.3) / 0.04) ** 2)) + 0.35 * np.exp(-(((ph - 0.6) / 0.08) ** 2)) - 0.25


def _noise(rng, family: str, n: int) -> np.ndarray:
    w = rng.standard_normal(n)
    if family == "dev":
        return w
    # pink-ish: cumulative sum with leak, normalized
    y = np.empty(n)
    acc = 0.0
    for i in range(0, n, 1):  # simple leaky integrator; n=150k is fast enough
        acc = 0.995 * acc + w[i]
        y[i] = acc
    y = y - y.mean()
    return 0.6 * w + 0.4 * y / (y.std() + 1e-12)


def make_case(rng: np.random.Generator, family: str, qtype: str, variant: str) -> tuple[np.ndarray, dict, str]:
    """Returns (signal, answer, question)."""
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f0 = rng.uniform(0.6, 2.5)
    f = np.full(n, f0)
    amp = rng.uniform(0.5, 3.0)
    gain = np.full(n, amp)
    level = np.zeros(n)
    noise_sd = rng.uniform(0.03, 0.15) * amp
    answer: dict = {}
    tc = float(np.round(rng.uniform(0.2, 0.8) * DUR, 1))

    if qtype == "change_time":
        if variant == "freq":
            f[t >= tc] = f0 * rng.choice([rng.uniform(0.6, 0.8), rng.uniform(1.25, 1.6)])
        elif variant == "amp":
            gain[t >= tc] = amp * rng.choice([rng.uniform(0.4, 0.6), rng.uniform(1.7, 2.5)])
        elif variant == "level":
            level[t >= tc] = amp * rng.choice([-1, 1]) * rng.uniform(1.0, 2.0)
        elif variant == "slow_am":  # adversarial: 30% amplitude modulation at 0.02-0.05 Hz is not a change
            gain = amp * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.02, 0.05) * t))
        answer = {"t": tc if variant in ("freq", "amp", "level") else None}
        question = (
            "Does the signal's behavior (its rhythm/frequency, amplitude or baseline level) change abruptly at some "
            "point and then stay changed? If so, when does the change happen? Gradual periodic modulation is not a change."
        )
    x = gain * _carrier(rng, family, t, f) + level + noise_sd * _noise(rng, family, n)

    if qtype == "period":
        a = float(np.round(rng.uniform(0, DUR - 120), 1))
        answer = {"period_s": 1.0 / f0, "window": [a, a + 60]}
        question = (
            f"What is the period (in seconds) of the signal's main oscillation in the window from {a} s to {a + 60} s?"
        )
    elif qtype == "artifact":
        present = variant == "present"
        ts = float(np.round(rng.uniform(0.1, 0.9) * DUR, 1))
        if present:
            L = int(rng.uniform(2, 10) * FS)
            i = int(ts * FS)
            if rng.random() < 0.5:
                x[i : i + L] = x[i]  # sensor stuck (flat line)
            else:
                x[i : i + L] = np.clip(x[i : i + L] * 4, -1.2 * amp, 1.2 * amp)  # saturated / clipped
                x[i : i + L] = np.where(
                    np.abs(x[i : i + L]) >= 1.2 * amp, np.sign(x[i : i + L]) * 1.2 * amp, x[i : i + L]
                )
        answer = {"present": present, "t": ts if present else None}
        question = (
            "Is there a segment (longer than 1 s) where the recording is corrupted — the sensor output is stuck at a "
            "constant value or saturated/clipped? If yes, when does it start?"
        )
    elif qtype == "spike_count":
        k = int(variant)
        pos = (
            np.sort(rng.choice(np.arange(int(5 * FS), n - int(5 * FS), int(3 * FS)), size=k, replace=False))
            if k
            else []
        )
        w = int(0.02 * FS)
        for p in pos:
            x[p : p + w] += rng.choice([-1, 1]) * 6 * amp * np.hanning(w)
        answer = {"count": k, "times": [float(p / FS) for p in pos]}
        question = (
            "How many isolated spikes are there — very short transients (tens of milliseconds) whose size is several "
            "times the normal signal amplitude?"
        )
    return x.astype(np.float32), answer, question


# (qtype, variant) plan per suite
DEV_PLAN = (
    [("change_time", v) for v in ["freq", "amp", "level", "freq", "none", "slow_am", "none", "slow_am"]]
    + [("period", "-")] * 4
    + [("artifact", v) for v in ["present", "present", "absent", "absent"]]
    + [("spike_count", v) for v in ["0", "2", "3", "5"]]
)


def build(seed: int, family: str, plan=DEV_PLAN):
    rng = np.random.default_rng(seed)
    for i, (q, v) in enumerate(plan):
        x, ans, question = make_case(rng, family, q, v)
        yield f"{family}{i:03d}", q, v, x, ans, question
