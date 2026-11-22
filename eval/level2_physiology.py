"""Offline physiological candidate construction and independent Level 2 references.

Only caller-supplied development records are inspected. This module writes no
artifacts, loads no datasets and calls no model. Public questions state estimands;
the exact reproducible numerical definitions are retained in GT provenance.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import signal

from eval import audit_reference as audit

TFA_DEFINITION = (
    "Nonoverlapping complete 1-s waveform means; 100-s periodic Hann Welch windows, "
    "50% overlap, constant detrend; H=mean(conj(X)*Y)/mean(abs(X)**2); "
    "equal-bin arithmetic mean gain abs(H), phase angle(H) in radians and "
    "coherence abs(mean(conj(X)*Y))**2/(mean(abs(X)**2)*mean(abs(Y)**2)); "
    "VLF [0.02,0.07), LF [0.07,0.20) Hz. Not a claim of CARNet compliance."
)


class ReferenceDisagreement(ValueError):
    def __init__(self, result):
        super().__init__("dual-reference half-tolerance agreement failed")
        self.details = {key: result[key] for key in ("reference1", "reference2", "agreement", "tolerances")}


def _numeric_gate(first: dict, second: dict, tolerances: dict) -> dict:
    fields = {}
    for name, value in first.items():
        spec = tolerances[name]
        relative = "relative" in spec
        tolerance = spec["relative" if relative else "absolute"]
        if tolerance == 0:
            fields[name] = {
                "accepted": bool(np.isfinite([value, second[name]]).all() and value == second[name]),
                "difference": abs(value - second[name]),
                "half_tolerance": 0,
            }
        else:
            fields[name] = audit.agreement(value, second[name], tolerance, relative=relative)
    return {"accepted": all(v["accepted"] for v in fields.values()), "fields": fields}


def tfa_scipy(abp_1hz: np.ndarray, flow_1hz: np.ndarray) -> dict:
    """Reference A: scipy Welch/CSD, with the complete stated transfer definition."""
    x, y = np.asarray(abp_1hz, float), np.asarray(flow_1hz, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 300 or not np.isfinite([x, y]).all():
        raise ValueError("TFA requires at least 300 finite synchronized seconds")
    kwargs = {"fs": 1.0, "window": "hann", "nperseg": 100, "noverlap": 50, "detrend": "constant"}
    frequency, xx = signal.welch(x, **kwargs)
    _, yy = signal.welch(y, **kwargs)
    _, xy = signal.csd(x, y, **kwargs)
    with np.errstate(divide="ignore", invalid="ignore"):
        transfer = xy / xx
        coherence = abs(xy) ** 2 / (xx * yy)
    return {
        "frequency": frequency,
        "gain": abs(transfer),
        "phase": np.angle(transfer),
        "coherence": coherence,
        "power": xx,
    }


def tfa_fft(abp_1hz: np.ndarray, flow_1hz: np.ndarray) -> dict:
    """Reference B: explicit segment indexing, independent DFT accumulation."""
    pressure = np.asarray(abp_1hz, dtype=float)
    cerebral = np.asarray(flow_1hz, dtype=float)
    if pressure.ndim != 1 or pressure.shape != cerebral.shape or len(pressure) < 300:
        raise ValueError("TFA requires at least 300 finite synchronized seconds")
    if any(not np.isfinite(v).all() for v in (pressure, cerebral)):
        raise ValueError("TFA requires finite synchronized seconds")
    taper = np.sin(np.pi * np.arange(100) / 100) ** 2
    spectra = []
    for left in range(0, len(pressure) - 99, 50):
        p = pressure[left : left + 100]
        c = cerebral[left : left + 100]
        fp = np.fft.rfft(taper * (p - sum(p) / 100))
        fc = np.fft.rfft(taper * (c - sum(c) / 100))
        spectra.append((fp * fp.conjugate(), fc * fc.conjugate(), fp.conjugate() * fc))
    accumulated = sum(np.asarray(row) for row in spectra) / len(spectra)
    pxx, pyy, pxy = accumulated
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.abs(pxy / pxx)
        phase = np.arctan2(pxy.imag, pxy.real)
        coherence = np.abs(pxy) ** 2 / (pxx.real * pyy.real)
    scale = np.full(51, 2 / sum(taper**2))
    scale[[0, -1]] /= 2
    return {
        "frequency": np.arange(51) / 100,
        "gain": gain,
        "phase": phase,
        "coherence": coherence,
        "power": pxx.real * scale,
    }


def compare_tfa(abp: np.ndarray, cbfv: np.ndarray, fs: float, band: str = "lf") -> dict:
    """Both references check excitation/coherence, including negative results."""
    if fs != round(fs) or fs < 1 or len(abp) != len(cbfv):
        raise ValueError("TFA requires integer sampling and equal waveform length")
    samples = int(fs)
    end = len(abp) // samples * samples
    if end / fs < 300:
        raise ValueError("TFA needs at least five minutes")
    if not np.isfinite(abp).all() or not np.isfinite(cbfv).all():
        raise ValueError("TFA requires finite channels throughout the requested interval")
    if np.quantile(abp, 0.01) < 20 or np.quantile(abp, 0.99) > 260:
        raise ValueError("TFA pressure outside declared 20–260 mmHg support range")
    if np.quantile(cbfv, 0.01) <= 0 or np.quantile(cbfv, 0.99) > 200:
        raise ValueError("TFA cerebral velocity outside declared 0–200 cm/s support range")
    # Deliberately independent reduction expressions as well as spectra.
    x = np.asarray(abp[:end], float).reshape(-1, samples).mean(axis=1)
    y = np.asarray(cbfv[:end], float).reshape(-1, samples).mean(axis=1)
    x2 = np.add.reduceat(np.asarray(abp[:end], float), np.arange(0, end, samples)) / samples
    y2 = np.add.reduceat(np.asarray(cbfv[:end], float), np.arange(0, end, samples)) / samples
    first, second = tfa_scipy(x, y), tfa_fft(x2, y2)
    low, high = {"lf": (0.07, 0.20), "vlf": (0.02, 0.07)}[band]
    values, enough = [], []
    for result in (first, second):
        mask = (result["frequency"] >= low - 1e-12) & (result["frequency"] < high - 1e-12)
        values.append({name: float(np.mean(result[name][mask])) for name in ("gain", "phase", "coherence")})
        enough.append(
            bool(
                np.all(result["power"][mask] > 1e-8)
                and np.isfinite(list(values[-1].values())).all()
                and values[-1]["coherence"] >= 0.30
            )
        )
    tolerances = {"gain": {"relative": 0.05}, "phase": {"absolute": 0.05}, "coherence": {"absolute": 0.04}}
    return {
        "reference1": values[0],
        "reference2": values[1],
        "agreement": _numeric_gate(*values, tolerances),
        "sufficient1": enough[0],
        "sufficient2": enough[1],
        "tolerances": tolerances,
        "methods": ["scipy Welch/CSD", "independent explicit FFT accumulation"],
        "definition": TFA_DEFINITION,
        "seconds": len(x),
        "segments": 1 + (len(x) - 100) // 50,
    }


def pressure_streams(abp: np.ndarray, fs: float) -> tuple[dict, dict]:
    """Systolic prominence maxima versus separately detected diastolic cycles."""
    pressure = audit._pressure(abp, fs)
    smooth = signal.savgol_filter(pressure, max(5, round(0.08 * fs) | 1), 2)
    spread = np.quantile(smooth, 0.95) - np.quantile(smooth, 0.05)
    peaks, _ = signal.find_peaks(smooth, distance=round(0.35 * fs), prominence=max(8, 0.25 * spread))
    radius = max(1, round(0.04 * fs))
    peaks = np.asarray(
        [
            p - radius + np.argmax(pressure[p - radius : p + radius + 1])
            for p in peaks
            if radius <= p < len(pressure) - radius
        ],
        dtype=int,
    )
    other = audit._independent_pressure(pressure, fs)
    if min(len(peaks), len(other["peaks"])) < 20 or not 0.90 <= len(peaks) / len(other["peaks"]) <= 1.10:
        raise ValueError("independent systolic event coverage disagrees")
    return ({"times": peaks / fs, "values": pressure[peaks]}, {"times": other["peaks"] / fs, "values": other["sbp"]})


def hr_streams(ecg: np.ndarray, fs: float) -> tuple[dict, dict]:
    result = audit.ecg_reference(ecg, fs)
    if not result["accepted"]:
        raise ValueError("independent ECG/RR/template quality failed")
    streams = []
    for key in ("rpeaks", "independent_rpeaks"):
        peaks = result[key] / fs
        streams.append({"times": (peaks[:-1] + peaks[1:]) / 2, "values": 60 / np.diff(peaks), "rr": np.diff(peaks)})
    return tuple(streams)


def period_change(first: dict, second: dict, baseline: tuple, response: tuple, field: str) -> dict:
    """Beat mean change plus response extrema; all period membership is half-open.

    ECG event times are RR midpoints; pressure event times are systolic maxima.
    Independently select the earliest occurrence when an extremum is tied.
    """
    a = [first["values"][(first["times"] >= left) & (first["times"] < right)] for left, right in (baseline, response)]
    b = [
        [v for t, v in zip(second["times"], second["values"], strict=True) if left <= t < right]
        for left, right in (baseline, response)
    ]
    if min(*(len(v) for v in a), *(len(v) for v in b)) < 10:
        raise ValueError("fewer than ten beats in an estimand period")
    response_times = first["times"][(first["times"] >= response[0]) & (first["times"] < response[1])]
    response_pairs = [
        (t, v) for t, v in zip(second["times"], second["values"], strict=True) if response[0] <= t < response[1]
    ]
    one = {
        field: float(np.mean(a[1]) - np.mean(a[0])),
        "response_max_time_s": float(response_times[np.argmax(a[1])]),
        "response_min_time_s": float(response_times[np.argmin(a[1])]),
    }
    two = {
        field: float(sum(b[1]) / len(b[1]) - sum(b[0]) / len(b[0])),
        "response_max_time_s": float(max(response_pairs, key=lambda pair: (pair[1], -pair[0]))[0]),
        "response_min_time_s": float(min(response_pairs, key=lambda pair: (pair[1], pair[0]))[0]),
    }
    tolerances = {
        field: {"absolute": 2.0 if field == "hr_change_bpm" else 3.0},
        "response_max_time_s": {"absolute": 0.10},
        "response_min_time_s": {"absolute": 0.10},
    }
    return {
        "reference1": one,
        "reference2": two,
        "agreement": _numeric_gate(one, two, tolerances),
        "tolerances": tolerances,
        "definition": "Arithmetic mean of beat estimates in each half-open period, later minus baseline; locate the earliest maximum and minimum beat estimate within the response period. HR uses RR midpoint times; SBP uses systolic peak times.",
        "tolerance_justification": "Mean change tolerance is unchanged: HR 2 bpm or SBP 3 mmHg. Each extremum time has 0.10 s absolute tolerance, resolving the individual beat while allowing native sample quantization and independent localization jitter. Both independently selected extrema must agree within 0.05 s; magnitude agreement must remain within half its original tolerance.",
    }


def rr_ratio(first: dict, second: dict, interval: tuple) -> dict:
    left, right = interval
    mask = (first["times"] >= left) & (first["times"] < right)
    a, times = first["rr"][mask], first["times"][mask]
    pairs = [(t, v) for t, v in zip(second["times"], second["rr"], strict=True) if left <= t < right]
    b = [v for _, v in pairs]
    if min(len(a), len(b)) < 20:
        raise ValueError("fewer than twenty RR intervals around manoeuvre")
    one = {
        "rr_ratio": float(np.max(a) / np.min(a)),
        "max_rr_time_s": float(times[np.argmax(a)]),
        "min_rr_time_s": float(times[np.argmin(a)]),
    }
    two = {
        "rr_ratio": float(max(b) / min(b)),
        "max_rr_time_s": float(max(pairs, key=lambda pair: (pair[1], -pair[0]))[0]),
        "min_rr_time_s": float(min(pairs, key=lambda pair: (pair[1], pair[0]))[0]),
    }
    tolerance = {
        "rr_ratio": {"relative": 0.05},
        "max_rr_time_s": {"absolute": 0.10},
        "min_rr_time_s": {"absolute": 0.10},
    }
    return {
        "reference1": one,
        "reference2": two,
        "agreement": _numeric_gate(one, two, tolerance),
        "tolerances": tolerance,
        "definition": "Longest divided by shortest RR interval whose midpoint lies in the stated half-open manoeuvre window, with the midpoint times of those two intervals; use the earliest midpoint for an exactly tied extremum.",
        "tolerance_justification": "RR-ratio tolerance remains 5% relative. Both RR midpoint times have 0.10 s absolute tolerance to identify the individual intervals while allowing native sample quantization and independent QRS localization jitter. Independent midpoint estimates must agree within 0.05 s and ratio estimates within 2.5%.",
    }


def compare_sequence(ecg: np.ndarray, abp: np.ndarray, fs: float) -> dict:
    result = audit.compare_brs(ecg, abp, fs)
    one = {"brs_ms_per_mmhg": result["reference1"], "valid_sequences": result["quality"]["valid_sequences1"]}
    two = {"brs_ms_per_mmhg": result["reference2"], "valid_sequences": result["quality"]["valid_sequences2"]}
    tolerances = {"brs_ms_per_mmhg": {"relative": 0.05}, "valid_sequences": {"absolute": 0}}
    return {
        "reference1": one,
        "reference2": two,
        "agreement": _numeric_gate(one, two, tolerances),
        "tolerances": tolerances,
        "methods": [audit.METHODS["brs"]],
        "source_hashes": result["source_hashes"],
    }


def insufficiency_checks(values: np.ndarray) -> dict:
    """Independent objective checks for an entirely absent/constant channel."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    one = bool(not finite.any() or (finite.all() and np.ptp(values) <= 1e-9))
    available = [float(v) for v in values if np.isfinite(v)]
    two = not available or (len(available) == len(values) and max(available) - min(available) <= 1e-9)
    return {
        "reference1": one,
        "reference2": bool(two),
        "accepted": one and two,
        "criterion": "No finite samples, or every sample finite with range <=1e-9; no physiological pulsation can be recovered",
    }


def _crop(record: dict, start: float, duration: float) -> dict:
    fs = record["fs"]
    a, b = round(start * fs), min(len(record["x"]), round((start + duration) * fs))
    result = {
        **record,
        "x": np.asarray(record["x"][a:b], dtype=np.float32).copy(),
        "start_s": record["start_s"] + a / fs,
        "duration_s": (b - a) / fs,
    }
    result["events"] = [
        {**event, "t": event["t"] - a / fs, **({"end_t": event["end_t"] - a / fs} if "end_t" in event else {})}
        for event in record.get("events", [])
        if a / fs <= event["t"] < b / fs
    ]
    return result


def _wave(record: dict, channel: str) -> np.ndarray:
    if channel not in record["channels"]:
        raise ValueError("required physiological channel unavailable")
    return record["x"][:, record["channels"].index(channel)]


def _item(
    record,
    family,
    kind,
    question,
    result,
    channels,
    periods,
    *,
    category="G4",
    capability="beat_segmentation",
    confidence="medium",
):
    if not result["agreement"]["accepted"]:
        raise ReferenceDisagreement(result)
    computation = {
        "beat_segmentation": "beat_estimation",
        "BRS": "sequence_brs",
        "TFA": "cross_spectral_estimation",
    }.get(capability, "signal_quality")
    metadata = {
        key: record[key] for key in ("dataset", "subject_key", "record_key", "source_hashes", "start_s", "duration_s")
    }
    reference_hashes = audit.source_hashes(*(["rpeaks"] if "ECG" in channels else []))
    reference_hashes.update(result.get("source_hashes", {}))
    reference_hashes["level2_physiology"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "family": family,
        "task_type": kind,
        "question": question,
        "expected_answer": result["reference1"],
        "answer_fields": list(result["reference1"]),
        "tolerances": result["tolerances"],
        "abstention_correct": False,
        "expected_solvability": "specialist",
        "specialist_capability": capability,
        "signal_kind": "real_signal",
        "x": record["x"],
        "fs": record["fs"],
        "channels": record["channels"],
        "units": record["units"],
        **metadata,
        "source": metadata,
        "ground_truth": {
            "category": category,
            "source": metadata,
            "methods": result.get(
                "methods", ["independently detected beat streams with independently aggregated period estimates"]
            ),
            "agreement": result["agreement"],
            "reference1": result["reference1"],
            "reference2": result["reference2"],
            "confidence": confidence,
            "tolerance_justification": result.get(
                "tolerance_justification",
                "BP 3 mmHg; HR changes 2 bpm; ratios/gain/BRS 5%; phase 0.05 rad; coherence 0.04; counts exact. Every numerical reference difference must be <=half the scoring tolerance.",
            ),
            "definition": result.get("definition"),
            "reference_source_hashes": reference_hashes,
        },
        "grounding_rubric": {
            "regions": [{"channels": channels, "window_s": list(period)} for period in periods],
            "required_computations": [computation],
        },
    }


def _f1(record, kind, protocol=True):
    events = [
        event
        for event in record.get("events", [])
        if event["kind"] in ("tilt", "valsalva") and 50 <= event["t"] <= record["duration_s"] - 70
    ]
    if protocol and not events:
        raise ValueError("no validated protocol onset with complete baseline and response")
    if protocol:
        event = events[0]
        # Uniform over every admissible full-window start, at native resolution.
        seed = int(hashlib.sha256((record["record_key"] + kind).encode()).hexdigest()[:16], 16)
        low = max(0.0, event["t"] - 110)
        high = min(event["t"] - 50, record["duration_s"] - 180)
        if high < low:
            raise ValueError("protocol event lacks a complete randomized 180-s crop")
        begin = np.random.default_rng(seed).uniform(low, high) if high > low else low
        rec = _crop(record, begin, 180)
        onset = event["t"] - (rec["start_s"] - record["start_s"])
        baseline, response = (onset - 45, onset - 10), (onset + 20, onset + 50)
        context = f"The manually aligned {event['kind']} onset is at {onset:g} s. "
    else:
        rec = _crop(record, 0, min(600, record["duration_s"]))
        baseline, response = (10, 40), (rec["duration_s"] - 45, rec["duration_s"] - 15)
        context = ""
    # The half-open membership bounds used by both references and shortcuts must
    # equal the numbers actually shown to the solver, including display rounding.
    baseline, response = [tuple(float(f"{value:g}") for value in period) for period in (baseline, response)]
    if kind == "rr_ratio":
        if not protocol or event["kind"] != "valsalva":
            raise ValueError("no validated Valsalva onset for RR-ratio estimand")
        streams = hr_streams(_wave(rec, "ECG"), rec["fs"])
        period = tuple(float(f"{value:g}") for value in (onset, onset + 50))
        result = rr_ratio(*streams, period)
        question = (
            context
            + f"What is the longest-to-shortest RR interval ratio during {period[0]:g}–{period[1]:g} s, and at what midpoint times do those longest and shortest intervals occur? Return rr_ratio, max_rr_time_s and min_rr_time_s. For tied extrema, report the earliest midpoint."
        )
        periods, channels = [period], ["ECG"]
    else:
        channel = "ECG" if kind == "hr_change_bpm" else "ABP"
        streams = (hr_streams if channel == "ECG" else pressure_streams)(_wave(rec, channel), rec["fs"])
        result = period_change(*streams, baseline, response, kind)
        quantity = "beatwise heart rate (bpm)" if channel == "ECG" else "beatwise systolic pressure (mmHg)"
        question = (
            context
            + f"What is the change in mean {quantity} from {baseline[0]:g}–{baseline[1]:g} s to {response[0]:g}–{response[1]:g} s (later minus baseline), and when are the highest and lowest beatwise values in that later period? Return {kind}, response_max_time_s and response_min_time_s. "
            + ("Report RR midpoint times" if channel == "ECG" else "Report systolic peak times")
            + "; for tied extrema, report the earliest time."
        )
        periods, channels = [baseline, response], [channel]
    result["methods"] = [
        audit.METHODS["ecg"]
        if channels == ["ECG"]
        else "Savitzky-Golay systolic prominence maxima/raw local maxima versus independent Butterworth diastolic-cycle raw systolic extrema; arithmetic beat mean"
    ]
    if protocol:
        result["methods"].append(event["method"])
    item = _item(rec, "F1", kind, question, result, channels, periods, category="G1+G4" if protocol else "G4")
    if protocol:
        item["ground_truth"]["manual_event"] = event
    return item


def _f4(record, kind, band="lf"):
    if record["duration_s"] < 600:
        raise ValueError("methodological BRS/TFA item requires at least 600 seconds")
    rec = _crop(record, 0, min(600, record["duration_s"]))
    if kind == "sequence_brs":
        result = compare_sequence(_wave(rec, "ECG"), _wave(rec, "ABP"), rec["fs"])
        question = "What are the sequence-method BRS (ms/mmHg) and number of valid maximal sequences, pairing each systolic pressure with its contemporaneous RR interval? Return brs_ms_per_mmhg and valid_sequences."
        channels, capability, confidence = ["ECG", "ABP"], "BRS", "medium"
    else:
        result = compare_tfa(_wave(rec, "ABP"), _wave(rec, "CBFV"), rec["fs"], band)
        if not result["sufficient1"] or not result["sufficient2"]:
            raise ValueError("both-reference band excitation/coherence support gate failed")
        fields = {"tfa_gain": ["gain"], "tfa_phase_coherence": ["phase", "coherence"]}[kind]
        result["reference1"] = {field: result["reference1"][field] for field in fields}
        result["reference2"] = {field: result["reference2"][field] for field in fields}
        result["tolerances"] = {field: result["tolerances"][field] for field in fields}
        result["agreement"] = _numeric_gate(result["reference1"], result["reference2"], result["tolerances"])
        band_text = "LF (0.07–0.20 Hz)" if band == "lf" else "VLF (0.02–0.07 Hz)"
        quantity = (
            "mean transfer-function gain from ABP to CBFV, in (cm/s)/mmHg"
            if fields == ["gain"]
            else "mean ABP-to-CBFV transfer phase (radians, positive for CBFV leading ABP) and magnitude-squared coherence"
        )
        question = f"Across {band_text}, what is the {quantity}? The estimand is an equal-frequency-bin mean for the complete recording's nonoverlapping 1-s mean pressure and flow. Report {', '.join(fields)}."
        channels, capability, confidence = ["ABP", "CBFV"], "TFA", "low"
    return _item(
        rec,
        "F4",
        kind + ("_" + band if kind != "sequence_brs" else ""),
        question,
        result,
        channels,
        [(0, rec["duration_s"])],
        capability=capability,
        confidence=confidence,
    )


def _f5(record, kind, abstain):
    rec = _crop(record, 0, min(600 if kind in ("sequence_brs", "tfa_gain") else 180, record["duration_s"]))
    if kind == "heart_rate":
        # Both control and insufficiency views contain ECG alone. Preserving
        # arterial/optical/cerebral pulses would leave cardiac rate recoverable
        # after ECG loss and make an abstention label physiologically invalid.
        index = rec["channels"].index("ECG")
        rec["x"] = rec["x"][:, index : index + 1].copy()
        rec["channels"] = ["ECG"]
        rec["units"] = [rec["units"][index]]
    channel = "ECG" if kind == "heart_rate" else "ABP"
    values = _wave(rec, channel)
    if kind in ("sequence_brs", "tfa_gain"):
        item = _f4(rec, kind)
        item["family"] = "F5"
        item["question"] += " If the evidence is insufficient, report insufficient_evidence with a reason."
    else:
        streams = (hr_streams if kind == "heart_rate" else pressure_streams)(values, rec["fs"])
        field = "heart_rate_bpm" if kind == "heart_rate" else "sbp_mmhg"
        one = {field: float(np.mean(streams[0]["values"]))}
        two = {field: float(sum(streams[1]["values"]) / len(streams[1]["values"]))}
        tolerances = {field: {"absolute": 2 if kind == "heart_rate" else 3}}
        result = {
            "reference1": one,
            "reference2": two,
            "tolerances": tolerances,
            "agreement": _numeric_gate(one, two, tolerances),
        }
        quantity = (
            "mean beatwise heart rate (bpm) from ECG"
            if kind == "heart_rate"
            else "mean beatwise systolic pressure (mmHg)"
        )
        item = _item(
            rec,
            "F5",
            kind,
            f"What is the {quantity} over this recording? Return {field}. If the evidence is insufficient, report insufficient_evidence with a reason.",
            result,
            [channel],
            [(0, rec["duration_s"])],
        )
    # L2-A treats the two counterfactual views as one independent subject pair.
    # No second F5 pair may use this subject, and no source inference is needed
    # to associate the two views: the opaque pair hash is explicit provenance.
    pair_material = "\0".join(
        ("L2-A-F5", rec["subject_key"], rec["record_key"], kind, str(rec["start_s"]), str(rec["duration_s"]))
    )
    item.update(
        pair_type=kind,
        pair_id=hashlib.sha256(pair_material.encode()).hexdigest(),
        pair_role="abstention" if abstain else "control",
    )
    if abstain:
        replacement = float(np.nanmedian(values))
        rec["x"][:, rec["channels"].index(channel)] = replacement
        checks = insufficiency_checks(_wave(rec, channel))
        if not checks["accepted"]:
            raise ValueError("independent insufficiency criteria disagree")
        item.update(
            x=rec["x"],
            expected_answer={
                "insufficient_evidence": True,
                "reason": f"{channel} contains no pulsatile evidence throughout the requested period",
            },
            tolerances={},
            abstention_correct=True,
            expected_solvability="abstention",
            signal_kind="controlled_perturbation",
            specialist_capability=None,
        )
        item["ground_truth"] = {
            "category": "GT-abstain",
            "source": item["source"],
            "methods": ["finite-mask and range criterion", "independent scalar enumeration and extrema criterion"],
            "agreement": checks,
            "confidence": "high",
            "tolerance_justification": "Objective complete loss of requested channel evidence; both independent criteria must agree",
            "reference_source_hashes": {"level2_physiology": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "perturbation": {
                "kind": "entire_channel_hold",
                "channel": channel,
                "window_s": [0, rec["duration_s"]],
                "value": replacement,
            },
        }
        item["grounding_rubric"]["required_computations"] = ["signal_quality"]
    item["ground_truth"]["pairing"] = {
        "pair_id": item["pair_id"],
        "pair_role": item["pair_role"],
        "design": "Exact same-source counterfactual: unchanged public question, channels, interval and nontarget samples; only the requested channel is held constant for the abstention member.",
        "subject_policy": "L2-A exception: one independent pair per development subject in F5, with the same subject used for its two counterfactual members only.",
    }
    return item


def _f5_pair(record, kind):
    """Return both matched views together, or exclude the entire pair on failure."""
    return [_f5(record, kind, False), _f5(record, kind, True)]


def build_candidates(records) -> tuple[list[dict], list[dict]]:
    """Construct the fixed 10/10/8 attempt targets, with all failed gates logged.

    A development subject appears at most once per family, with the L2-A exception
    that the two matched F5 views count as one independent pair. If an exact target
    cannot pass its reference gate, it remains an explicit missing-slot exclusion;
    no tolerance or quality threshold is relaxed.
    """
    targets = {
        "F1": [("hr_change_bpm", True)] * 3
        + [("sbp_change_mmhg", True)] * 3
        + [("rr_ratio", True)] * 2
        + [("hr_change_bpm", False), ("sbp_change_mmhg", False)],
        "F4": [("sequence_brs", "lf")] * 4
        + [("tfa_gain", "lf")] * 2
        + [("tfa_gain", "vlf")] * 2
        + [("tfa_phase_coherence", "lf")] * 2,
        "F5": [(kind,) for kind in ("systolic_pressure", "heart_rate", "sequence_brs", "tfa_gain")],
    }
    candidates, exclusions = [], []
    used = {family: set() for family in targets}
    filled = {family: set() for family in targets}
    for record in records:
        for family, slots in targets.items():
            if record["subject_key"] in used[family]:
                continue
            # One attempt per distinct unresolved estimand, per record.
            tried = set()
            for index, slot in enumerate(slots):
                if index in filled[family] or slot in tried:
                    continue
                tried.add(slot)
                try:
                    result = {"F1": _f1, "F4": _f4, "F5": _f5_pair}[family](record, *slot)
                except (ValueError, IndexError, FloatingPointError) as error:
                    exclusions.append(
                        {
                            "family": family,
                            "task_type": slot[0],
                            "subject_key": record["subject_key"],
                            "record_key": record["record_key"],
                            "stage": "quality_or_reference_agreement",
                            "reason": str(error),
                            **({"count": 2, "pair_excluded": True} if family == "F5" else {}),
                            **({"comparison": error.details} if isinstance(error, ReferenceDisagreement) else {}),
                        }
                    )
                    continue
                candidates.extend(result if family == "F5" else [result])
                used[family].add(record["subject_key"])
                filled[family].add(index)
                break
        if all(len(filled[family]) == len(slots) for family, slots in targets.items()):
            break
    for family, slots in targets.items():
        missing = Counter(slots[index][0] for index in range(len(slots)) if index not in filled[family])
        exclusions.extend(
            {
                "family": family,
                "task_type": kind,
                "stage": "unfilled_target",
                "reason": "No unused development subject passed the unchanged reference/support gate",
                "count": count * (2 if family == "F5" else 1),
            }
            for kind, count in missing.items()
        )
    return candidates, exclusions
