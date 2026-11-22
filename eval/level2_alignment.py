"""Native timing support for sub-beat interventions; never opens dataset files.

Two event pipelines independently estimate native pre/post cardiac-phase changes.
Their estimates gate whether a controlled 24–80 ms offset can reasonably be scored
within 10 ms without being confounded by native physiological timing changes.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from eval.audit_reference import ecg_reference

MIN_BEATS = 12
NATIVE_LIMIT_MS = 5.0
AGREEMENT_LIMIT_MS = 5.0


def _three_point_vertex(values: np.ndarray, index: int) -> float:
    """Closed-form local extremum, retaining fractional native sample timing."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = values[index - 1 : index + 2]
    curvature = left - 2 * center + right
    offset = 0.5 * (left - right) / curvature if curvature < -1e-15 else 0.0
    return float(index + np.clip(offset, -0.5, 0.5))


def _five_point_vertex(values: np.ndarray, index: int) -> float:
    """Independent least-squares quadratic on five observations."""
    if index < 2 or index >= len(values) - 2:
        return float(index)
    coordinates = np.arange(-2.0, 3.0)
    design = np.column_stack((coordinates**2, coordinates, np.ones(5)))
    quadratic, linear, _ = np.linalg.lstsq(design, values[index - 2 : index + 3], rcond=None)[0]
    if quadratic >= -1e-15:
        return float(index)
    vertex = -linear / (2 * quadratic)
    return float(index + max(-1.0, min(1.0, vertex)))


def _guided_events(x: np.ndarray, fs: float, names: list[str], rpeaks: np.ndarray) -> dict:
    """Primary R events guide local pulse maxima; no unguided pulse detector."""
    ecg = x[:, names.index("ECG")]
    pulse_names = [name for name in ("ABP", "PPG") if name in names]
    smoothed = {
        name: signal.savgol_filter(x[:, names.index(name)], max(5, round(0.056 * fs) | 1), 2) for name in pulse_names
    }
    rows = {"ECG": [], **{name: [] for name in pulse_names}}
    for peak, following in zip(rpeaks[:-1], rpeaks[1:], strict=True):
        peak, following = int(peak), int(following)
        lo, hi = max(1, peak - round(0.04 * fs)), min(len(ecg) - 1, peak + round(0.04 * fs) + 1)
        baseline = float(np.median(ecg[max(0, peak - round(0.12 * fs)) : peak + round(0.12 * fs)]))
        envelope = np.abs(ecg - baseline)
        refined = lo + int(np.argmax(envelope[lo:hi]))
        anchor = _three_point_vertex(envelope, refined)
        left = round(anchor + 0.08 * fs)
        right = min(round(anchor + 0.75 * fs), following - round(0.02 * fs), len(ecg) - 2)
        if right <= left + 2:
            continue
        row = {"ECG": anchor / fs}
        for name in pulse_names:
            values = smoothed[name]
            maximum = left + int(np.argmax(values[left : right + 1]))
            if maximum <= left or maximum >= right:
                break
            row[name] = _three_point_vertex(values, maximum) / fs
        if len(row) == len(rows):
            for name, value in row.items():
                rows[name].append(value)
    return {name: np.asarray(values) for name, values in rows.items()}


def _unguided_events(x: np.ndarray, fs: float, names: list[str], independent_r: np.ndarray) -> dict:
    """Free-running pulse detection with separate smoothing, extrema and pairing."""
    ecg = x[:, names.index("ECG")]
    envelope = np.abs(signal.sosfiltfilt(signal.butter(3, [5, 25], fs=fs, btype="bandpass", output="sos"), ecg))
    rtimes = np.array([_five_point_vertex(envelope, int(index)) / fs for index in independent_r])
    pulse_names = [name for name in ("ABP", "PPG") if name in names]
    pulse_events = {}
    for name in pulse_names:
        values = signal.sosfiltfilt(signal.butter(3, 10, fs=fs, output="sos"), x[:, names.index(name)])
        spread = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        peaks, _ = signal.find_peaks(values, distance=max(1, round(0.30 * fs)), prominence=0.25 * spread)
        pulse_events[name] = np.array([_five_point_vertex(values, int(index)) / fs for index in peaks])
    rows = {"ECG": [], **{name: [] for name in pulse_names}}
    consumed = {name: -1 for name in pulse_names}
    for index in range(len(rtimes) - 1):
        anchor, following = rtimes[index : index + 2]
        choices = {}
        for name in pulse_names:
            times = pulse_events[name]
            left = int(np.searchsorted(times, anchor + 0.08, side="left"))
            right = int(np.searchsorted(times, min(anchor + 0.75, following - 0.02), side="right"))
            # A dicrotic/artefact ambiguity is excluded, rather than selecting a
            # convenient peak or silently counting a pulse twice.
            if right - left != 1 or left <= consumed[name]:
                break
            choices[name] = left
        if len(choices) != len(pulse_names):
            continue
        rows["ECG"].append(float(anchor))
        for name, event_index in choices.items():
            rows[name].append(float(pulse_events[name][event_index]))
            consumed[name] = event_index
    return {name: np.asarray(values) for name, values in rows.items()}


def _window_summary(events: dict, pair: tuple[str, str], window: tuple[float, float]) -> dict:
    mask = (events["ECG"] >= window[0]) & (events["ECG"] < window[1])
    values = (events[pair[1]][mask] - events[pair[0]][mask]) * 1000
    count = len(values)
    return {
        "beats": count,
        "mean_lag_ms": float(np.mean(values)) if count else None,
        "standard_deviation_ms": float(np.std(values, ddof=1)) if count >= 2 else None,
    }


def _comparison(first: dict, second: dict) -> dict:
    supported = min(first["beats"], second["beats"]) >= MIN_BEATS
    if not supported:
        return {"accepted": False, "reason": "fewer than twelve complete matched beats in either window"}
    change = second["mean_lag_ms"] - first["mean_lag_ms"]
    standard_error = float(
        np.sqrt(
            first["standard_deviation_ms"] ** 2 / first["beats"]
            + second["standard_deviation_ms"] ** 2 / second["beats"]
        )
    )
    return {
        "accepted": bool(abs(change) <= NATIVE_LIMIT_MS and standard_error <= NATIVE_LIMIT_MS),
        "native_change_ms": float(change),
        "change_standard_error_ms": standard_error,
        "before": first,
        "after": second,
    }


def native_lag_support(x: np.ndarray, fs: float, channels: list[str], onset_s: float, drift: bool = False) -> dict:
    """Gate native timing changes in the exact comparison regions used by F3.

    Abrupt shifts compare the initial 20 s with [onset+5,onset+25) s; drift
    compares the first and final 30 s. Each method must find >=12 complete beats,
    native mean timing change <=5 ms, and mean-change standard error <=5 ms.
    Independent native-change estimates must agree within 5 ms for every pair.
    """
    values = np.asarray(x, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != len(channels)
        or len(set(channels)) != len(channels)
        or "ECG" not in channels
        or not {"ABP", "PPG"} & set(channels)
        or not np.isfinite(fs)
        or fs < 60
        or not np.isfinite(values).all()
        or not np.isfinite(onset_s)
    ):
        raise ValueError("Native timing audit requires finite aligned ECG and pulse channels")
    duration = len(values) / fs
    before, after = (
        ((0.0, 30.0), (duration - 30.0, duration))
        if drift
        else (
            (0.0, 20.0),
            (onset_s + 5.0, onset_s + 25.0),
        )
    )
    result = {
        "accepted": False,
        "native_change_limit_ms": NATIVE_LIMIT_MS,
        "method_agreement_limit_ms": AGREEMENT_LIMIT_MS,
        "mean_change_standard_error_limit_ms": NATIVE_LIMIT_MS,
        "windows_s": {"before": list(before), "after": list(after)},
        "estimand": "Change in mean same-beat systolic-maximum timing between native channels",
        "methods": [
            "Primary derivative-energy R detector; raw-ECG local extremum and three-point fractional vertex; ECG-guided Savitzky-Golay pulse maxima",
            "Independent bandpass-amplitude R detector; unguided Butterworth pulse prominence maxima; five-point least-squares vertices and one-to-one forward event matching",
        ],
        "pairs": {},
    }
    if after[0] < before[1] or after[1] > duration or before[1] > duration:
        result["reason"] = "Native record lacks nonoverlapping complete timing comparison windows"
        return result
    try:
        ecg = ecg_reference(values[:, channels.index("ECG")], fs)
        if not ecg["accepted"]:
            result["reason"] = "Native ECG fails independent detector quality agreement"
            return result
        result["ecg_quality"] = ecg["quality"]
        implementations = [
            _guided_events(values, fs, channels, ecg["rpeaks"]),
            _unguided_events(values, fs, channels, ecg["independent_rpeaks"]),
        ]
    except ValueError:
        result["reason"] = "Native event reference has insufficient quality or support"
        return result
    pairs = [("ECG", name) for name in ("ABP", "PPG") if name in channels]
    if {"ABP", "PPG"}.issubset(channels):
        pairs.append(("PPG", "ABP"))
    accepted = True
    for pair in pairs:
        estimates = [
            _comparison(_window_summary(events, pair, before), _window_summary(events, pair, after))
            for events in implementations
        ]
        disagreement = (
            abs(estimates[0]["native_change_ms"] - estimates[1]["native_change_ms"])
            if all("native_change_ms" in estimate for estimate in estimates)
            else None
        )
        passed = (
            all(estimate["accepted"] for estimate in estimates)
            and disagreement is not None
            and disagreement <= AGREEMENT_LIMIT_MS
        )
        result["pairs"]["_to_".join(pair)] = {
            "accepted": bool(passed),
            "implementation_1": estimates[0],
            "implementation_2": estimates[1],
            "native_change_disagreement_ms": disagreement,
        }
        accepted &= passed
    result["accepted"] = bool(accepted)
    if not accepted:
        result["reason"] = "Native lag change, uncertainty, independent agreement or matched-beat support exceeds gate"
    return result
