"""Deterministic synthetic ABP / ECG for pressure-plugin and runtime tests (no patient data)."""

from __future__ import annotations

import numpy as np


def synth_abp(
    fs: float = 125.0,
    duration_s: float = 60.0,
    hr_bpm: float = 72.0,
    sbp: float = 120.0,
    dbp: float = 80.0,
    seed: int = 0,
    sbp_slope_per_s: float = 0.0,
    noise: float = 0.3,
    flat: tuple[float, float] | None = None,
    clip_at: float | None = None,
    drop_beats: tuple[int, ...] = (),
    spikes: tuple[float, ...] = (),
    nan_at: tuple[float, float] | None = None,
    hrv: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (abp, ecg, beats). Each beat: a sharp systolic rise, exponential decay and a small dicrotic bump.

    `beats` lists the generator's own foot/peak times and per-beat SBP/DBP (the generating truth, not a reference
    produced by the toolkit). Beats listed in `drop_beats` are replaced by a flat diastolic run (missing pulse).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    abp = np.full(n, dbp, dtype=np.float64)
    ecg = np.zeros(n)
    beats = []
    t_beat = 0.35
    k = 0
    while t_beat < duration_s - 0.2:
        rr = 60.0 / hr_bpm * (1 + hrv * rng.standard_normal())
        rr = float(np.clip(rr, 0.25, 2.0))
        s_k = sbp + sbp_slope_per_s * t_beat
        foot = t_beat
        rise = min(0.12, 0.4 * rr)  # the upstroke cannot take more than ~40 % of a very short beat
        peak = foot + rise
        i_f, i_n = int(round(foot * fs)), min(n, int(round((foot + rr) * fs)))
        # R peak 0.2 s before the foot (physiological pulse arrival delay)
        i_r = int(round((foot - 0.2) * fs))
        if 0 <= i_r < n:
            ecg[max(0, i_r - 1) : i_r + 2] += 1.0
            ecg[i_r] += 0.5
        if k not in drop_beats:
            seg_t = t[i_f:i_n] - foot
            tau = 0.4 * rr  # diastolic decay scales with the beat so the pulse returns to DBP before the next foot
            notch = 0.3 * rr
            pulse = np.where(
                seg_t < rise,
                dbp + (s_k - dbp) * np.sin(np.pi / 2 * seg_t / rise) ** 2,
                dbp
                + (s_k - dbp)
                * np.exp(-(seg_t - rise) / tau)
                * (1 + 0.12 * np.exp(-((seg_t - rise - notch) ** 2) / (0.004 * (rr / 0.83) ** 2))),
            )
            abp[i_f:i_n] = pulse
            beats.append({"k": k, "t_foot_s": foot, "t_peak_s": peak, "t_end_s": foot + rr, "sbp": s_k, "dbp": dbp})
        t_beat += rr
        k += 1
    abp += noise * rng.standard_normal(n)
    ecg += 0.02 * rng.standard_normal(n)
    if flat is not None:
        a, b = int(flat[0] * fs), int(flat[1] * fs)
        abp[a:b] = abp[a]
    if clip_at is not None:
        abp = np.minimum(abp, clip_at)
    for ts in spikes:
        i = int(ts * fs)
        abp[i : i + 3] += 80.0
    if nan_at is not None:
        a, b = int(nan_at[0] * fs), int(nan_at[1] * fs)
        abp[a:b] = np.nan
    return abp, ecg, beats
