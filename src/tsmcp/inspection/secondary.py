"""Independent detector agreement on one continuous acquisition block, for review ranking only."""

from collections import Counter

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from tsmcp.processing import intervals


def alt_pulse(y, fs):
    sm = sosfiltfilt(butter(3, min(8, fs * 0.35), fs=fs, btype="lowpass", output="sos"), y)
    slope = np.gradient(sm)
    floor = max(float(np.percentile(slope, 95)) * 0.25, 1e-10)
    d = find_peaks(slope, height=floor, prominence=floor, distance=round(0.28 * fs))[0]
    result = []
    for i, p0 in enumerate(d):
        hi = min(len(y), p0 + round(0.35 * fs) + 1, d[i + 1] if i + 1 < len(d) else len(y))
        peak = int(p0 + np.argmax(sm[p0:hi]))
        left = max(0, p0 - round(0.32 * fs))
        base = float(np.min(sm[left : p0 + 1]))
        foot = max(left, int(round(p0 - (sm[p0] - base) / slope[p0])))
        if not result or peak > result[-1][1] and foot > result[-1][0]:
            result.append((foot, peak))
    return result


def match(q, s, tol):
    i = j = 0
    pairs = []
    while i < len(q) and j < len(s):
        if q[i] < s[j] - tol:
            i += 1
        elif s[j] < q[i] - tol:
            j += 1
        else:
            pairs.append((i, j))
            i += 1
            j += 1
    return pairs


def compare_block(x, good, pr, task, fs):
    raw_n = [e["sample"] for e in pr["events"]]
    native_press = {right["start_sample"]: right for right in pr["pressures"]}
    native_feet = sorted(native_press)
    obs = []
    alt = []
    st = Counter(blocks=1)
    for left, right in intervals(good):
        if right - left < 4 * fs:
            st["insufficient_samples"] += right - left
            continue
        y = x[left:right]
        if task == "ECG":
            from wfdb.processing import xqrs_detect

            pp = np.asarray(xqrs_detect(y, fs=fs, verbose=False), int) + left
            feet = []
        else:
            fp = alt_pulse(y, fs)
            pp = np.array([z[1] + left for z in fp], int)
            feet = [z[0] + left for z in fp]
        alt += pp.tolist()
        qq = np.array([i for i in raw_n if left <= i < right], int)
        matched = match(qq, pp, round((0.08 if task == "ECG" else 0.12) * fs))
        used1 = {i for i, j in matched}
        used2 = {j for i, j in matched}
        # Boundary behavior of a detector is scored separately; the outer second is not an onset discrepancy.
        for label, arr, used in [("native_only", qq, used1), ("comparison_only", pp, used2)]:
            obs += [(int(q), label, 1.0) for i, q in enumerate(arr) if i not in used and left + fs <= q < right - fs]
        if task == "ABP":
            pairs = []
            for i, j in matched:
                k = np.searchsorted(native_feet, qq[i], side="right") - 1
                if k >= 0 and j + 1 < len(feet):
                    z = native_press[native_feet[k]]
                    if z["start_sample"] <= qq[i] < z["stop_sample"] and feet[j] < feet[j + 1]:
                        pairs.append((i, j, z))
            shifts = [(feet[j] - z["start_sample"]) / fs for _, j, z in pairs]
            if len(shifts) >= 5:
                typical = np.median(shifts)
                for (_, j, z), shift in zip(pairs, shifts, strict=True):
                    a, b = feet[j : j + 2]
                    samples = x[a : b + 1]
                    mm = float((samples[:-1].sum() + 0.5 * (samples[-1] - samples[0])) / (b - a))
                    err = abs(mm - z["mean_pressure"])
                    if abs(shift - typical) >= 0.15 and err >= 3:
                        obs.append((int(pp[j]), "foot_and_mean_disagreement", round(err, 4)))
        st["evaluated_samples"] += max(0, right - left - 2 * round(fs))
        st["native_matched_pairs"] += len(matched)
    return obs, alt, st
