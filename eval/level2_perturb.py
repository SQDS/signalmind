"""Development-only controlled Level 2 candidates, before shortcut rejection.

The registered strengths and task schedule do not depend on baseline outcomes.
No source-derived clinical label is used as reference truth.
"""

from __future__ import annotations

import hashlib
from itertools import islice

import numpy as np
from scipy import signal

from eval.audit_reference import ecg_reference, g2_pulse_rate, independent_rpeaks
from eval.level2_alignment import native_lag_support

SEED = 20261026
TARGETS = {"F2": 12, "F3": 12, "F6": 8}
RETRY_SLOTS = {"F3": (6, 7), "F6": (4, 5, 6, 7)}
RETRY_MAX_ATTEMPTS_PER_SLOT = 24
RETRY_MAX_SOURCE_RECORDS = 64
RETRY_SCHEDULE = {
    "version": "level2-amendment-a-source-retry-v1",
    "order": "Supplied frozen development stream order; every designated slot on each eligible source",
    "max_source_records": RETRY_MAX_SOURCE_RECORDS,
    "max_attempts_per_slot": RETRY_MAX_ATTEMPTS_PER_SLOT,
    "durations_s": {"F3": 600, "F6": 180},
    "selection": "Evaluate every generated alternative with unchanged shortcuts and support gates before slot allocation",
    "interventions": "Original strengths, tolerances and subject-derived seeds unchanged",
}


def _base(record, family, task, x, channels, units, answer, tolerances, question, regions, computation):
    return {
        **{key: record[key] for key in ("dataset", "subject_key", "record_key", "source_hashes", "start_s")},
        "family": family,
        "task_type": task,
        "question": question,
        "x": np.asarray(x, dtype=np.float32),
        "fs": float(record["fs"]),
        "channels": channels,
        "units": units,
        "expected_answer": answer,
        "tolerances": tolerances,
        "abstention_correct": False,
        "expected_solvability": "generic",
        "specialist_capability": None,
        "signal_kind": "controlled_perturbation",
        "ground_truth": {
            "category": "G3",
            "methods": [
                "Recorded intervention sample bounds and parameters",
                "Separate sample-index audit of the applied transformation",
            ],
            "agreement": {"accepted": True},
            "confidence": "medium",
            "tolerance_justification": "Onset tolerance 5 s resolves gradual envelope localization; timing tolerance 10 ms exceeds one native 125 Hz sample.",
        },
        "grounding_rubric": {"regions": regions, "required_computations": [computation]},
    }


def _region(channels, start, end):
    return {"channels": channels, "window_s": [float(start), float(end)]}


def make_candidate(record, family, index, quality):
    fs = float(record["fs"])
    original = np.asarray(record["x"], dtype=np.float32)
    # Long records are necessary for drift; other tasks use 180 s for local morphology.
    duration = min(record["duration_s"], 600 if family == "F3" and index in (6, 7) else 180)
    if duration < (600 if family == "F3" and index in (6, 7) else 120):
        raise ValueError("insufficient strictly continuous duration for registered task")
    x = original[: round(duration * fs)].copy()
    names, units = list(record["channels"]), list(record["units"])
    seed = int(hashlib.sha256(f"{SEED}:{family}:{index}:{record['subject_key']}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    # Uniform over admissible margins; no concentration at the old 80-second shortcut.
    start = int(rng.uniform(20, duration - 35) * fs)
    n = len(x)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite samples within complete task window")
    cross_support = []
    if family == "F3":
        # Every minute of a long alignment case must have usable common cardiac support.
        block_size = round(60 * fs)
        for left in range(0, n, block_size):
            right = min(n, left + block_size)
            if right - left < 20 * fs:
                left = max(0, right - block_size)
            ecg = x[left:right, names.index("ECG")]
            for pulse_name in ("PPG", "ABP"):
                comparison = g2_pulse_rate(ecg, x[left:right, names.index(pulse_name)], fs)
                if not comparison["accepted"]:
                    raise ValueError("complete-window native cross-channel cardiac support failed")
                cross_support.append(
                    {
                        "channel": pulse_name,
                        "window_s": [left / fs, right / fs],
                        "rate_difference_bpm": comparison["difference_bpm"],
                    }
                )
    onset = start / fs
    params = {
        "seed": seed,
        "position_distribution": "Uniform(20,duration-35) s rounded down to native sample",
        "onset_sample": start,
        "onset_s": onset,
    }
    if family == "F2":
        j = names.index("ABP")
        native = x[:, j].copy()
        center = signal.sosfiltfilt(signal.butter(2, 0.3, fs=fs, output="sos"), native)
        ac = native - center
        target_bias = (3.0, 5.0, 8.0, 12.0)[index % 4]
        peaks, _ = signal.find_peaks(native, distance=round(0.4 * fs), prominence=8)
        peaks = peaks[peaks > start + round(3 * fs)]
        if len(peaks) < 15:
            raise ValueError("too few pressure peaks to calibrate graded attenuation")
        ac_at_peaks = float(np.mean(ac[peaks]))
        attenuation = target_bias / ac_at_peaks
        if not 0.04 <= attenuation <= 0.7:
            raise ValueError("graded attenuation outside preregistered plausible range")
        # A one-second onset ramp avoids an unphysiological single-sample seam.
        ramp = np.clip((np.arange(n) - start) / fs, 0, 1)
        x[:, j] = native - attenuation * ramp * ac
        achieved = float(np.mean(native[peaks] - x[peaks, j]))
        independent_bias = sum(float(native[k]) - float(x[k, j]) for k in peaks) / len(peaks)
        if abs(achieved - independent_bias) > 1e-5:
            raise ValueError("independent attenuation audit disagreement")
        params.update(
            kind="AC_attenuation_with_one_second_transition",
            channel="ABP",
            attenuation=float(attenuation),
            target_peak_bias_mmhg=target_bias,
            measured_peak_bias_mmhg=achieved,
            independent_peak_bias_mmhg=independent_bias,
            lowpass_hz=0.3,
            ramp_s=1,
        )
        item = _base(
            record,
            family,
            "damping_onset",
            x,
            names,
            units,
            {"onset_s": onset},
            {"onset_s": 5.0},
            "At what time does the sustained attenuation of arterial pulsations begin in ABP? Report onset_s in seconds, using ECG and pulse-wave consistency to distinguish it from a transient change in pressure.",
            [_region(["ABP", "ECG"], max(0, onset - 10), min(duration, onset + 20))],
            "pressure_pulsatility_change",
        )
        item["expected_solvability"] = "specialist" if target_bias <= 5 else "generic"
        item["specialist_capability"] = "beat_envelope_and_cross_channel_quality" if target_bias <= 5 else None
        item["ground_truth"]["confidence"] = "medium"
        item["ground_truth"]["limitations"] = (
            "Controlled AC attenuation is a surrogate for damping, not a validated catheter dynamic-response model or manual clinical quality label. Recorded bias uses unchanged native peak locations."
        )
    elif family == "F3":
        channel = ("ECG", "ABP", "PPG")[index % 3]
        j = names.index(channel)
        native = x[:, j].copy()
        if index < 8:
            lag_support = native_lag_support(x, fs, names, onset, drift=index >= 6)
            if not lag_support["accepted"]:
                raise ValueError("native physiological delay change exceeds half-tolerance alignment gate")
            shift_ms = (24, -40, 56, -80, 32, -64, 48, -72)[index]
            shift = round(shift_ms * fs / 1000)
            if index < 6:
                offsets = np.where(np.arange(n) >= start, shift, 0)
                task, field = "timing_shift", "shift_ms"
                description = "A recording channel develops an additional timing offset. Identify the inconsistent channel and its added delay relative to the initial synchronized period, preserving the physiological ECG-to-pulse delay. Report channel and signed shift_ms (positive means delayed)."
                region = [
                    _region(names, 0, 20),
                    _region(names, min(duration - 20, onset + 5), min(duration, onset + 25)),
                ]
            else:
                offsets = shift * np.arange(n) / (n - 1)
                task, field = "clock_drift", "drift_ms"
                description = "A recording clock drifts linearly during this ten-minute record. Identify the inconsistent channel and the change in its timing offset from the beginning to the end, preserving physiological pulse delay. Report channel and signed drift_ms (positive means delayed)."
                region = [_region(names, 0, 30), _region(names, duration - 30, duration)]
            source = np.clip(np.arange(n) - offsets, 0, n - 1)
            x[:, j] = np.interp(source, np.arange(n), native)
            audit = np.asarray(
                [
                    native[int(k)] * (1 - (k - int(k))) + native[min(n - 1, int(k) + 1)] * (k - int(k))
                    for k in source[:: max(1, n // 1000)]
                ]
            )
            if not np.allclose(x[:: max(1, n // 1000), j], audit, atol=1e-4):
                raise ValueError("independent timing interpolation audit disagreement")
            params.update(
                kind=task,
                channel=channel,
                shift_samples=shift,
                endpoint_shift_ms=1000 * shift / fs,
                clipped_boundary_samples=int(
                    np.count_nonzero((np.arange(n) - offsets < 0) | (np.arange(n) - offsets > n - 1))
                ),
            )
            item = _base(
                record,
                family,
                task,
                x,
                names,
                units,
                {"channel": channel, field: 1000 * shift / fs},
                {field: 10.0},
                description,
                region,
                "cross_channel_beat_alignment",
            )
            item["ground_truth"]["native_lag_stability"] = lag_support
            item.update(
                expected_solvability="specialist",
                specialist_capability="alignment_and_clock_drift" if index >= 6 else "sub_beat_alignment",
            )
        elif index < 10:
            # Match native cardiac events before defining an entire missing train.
            ecg = x[:, names.index("ECG")]
            reference = ecg_reference(ecg, fs)
            if not reference["accepted"]:
                raise ValueError("missing-beat native ECG support failed")
            if channel == "ECG":
                peaks = reference["rpeaks"]
                lag = 0.0
            else:
                smooth = signal.sosfiltfilt(signal.butter(3, 8, fs=fs, output="sos"), native)
                spread = np.quantile(smooth, 0.95) - np.quantile(smooth, 0.05)
                peaks, _ = signal.find_peaks(smooth, distance=round(0.30 * fs), prominence=0.30 * spread)
                previous = np.searchsorted(reference["rpeaks"], peaks, side="right") - 1
                valid = previous >= 0
                lag = float(np.median(peaks[valid] - reference["rpeaks"][previous[valid]]))
            count = 2 + index % 2
            first = int(np.searchsorted(peaks, start, side="right"))
            if first < 1 or first + count >= len(peaks):
                raise ValueError("missing-beat intervention lacks adjacent native cycles")
            left = int((peaks[first - 1] + peaks[first]) // 2)
            right = int((peaks[first + count - 1] + peaks[first + count]) // 2)
            # Two ECG references must independently support exactly this count.
            second = independent_rpeaks(ecg, fs)
            counts = [
                int(np.count_nonzero((r + lag >= left) & (r + lag <= right))) for r in (reference["rpeaks"], second)
            ]
            if counts != [count, count]:
                raise ValueError("missing-beat cardiac count disagrees across references")
            x[left : right + 1, j] = np.linspace(native[left], native[right], right - left + 1)
            test = x[left : right + 1, j].astype(float)
            if np.ptp(np.diff(test)) > 1e-4:
                raise ValueError("missing train replacement is not independently linear")
            params.update(
                kind="pulse_gap",
                channel=channel,
                missing_peaks=int(count),
                bounds_samples=[left, right + 1],
                independent_cardiac_counts=counts,
                baseline_pulse_lag_samples=lag,
            )
            item = _base(
                record,
                family,
                "missing_beats",
                x,
                names,
                units,
                {"channel": channel, "count": count},
                {"count": 0},
                "One channel contains a short gap in its pulse train while the other channels continue. Identify the channel and count how many cardiac pulses are missing, using the other channels as the timing reference. Report channel and count.",
                [_region(names, max(0, left / fs - 2), min(duration, right / fs + 2))],
                "cross_channel_missing_beats",
            )
            item.update(expected_solvability="specialist", specialist_capability="beat_matching")
        else:
            center = float(np.median(native[:start]))
            x[start:, j] = 2 * center - native[start:]
            params.update(kind="polarity_inversion", channel=channel, center=center)
            item = _base(
                record,
                family,
                "polarity_inversion",
                x,
                names,
                units,
                {"channel": channel, "onset_s": onset},
                {"onset_s": 5.0},
                "A channel changes polarity without a corresponding physiological reversal in the other channels. Identify the channel and the time the inversion starts. Report channel and onset_s in seconds.",
                [_region(names, max(0, onset - 10), min(duration, onset + 15))],
                "cross_channel_polarity_check",
            )
    else:
        if index < 4:
            j = names.index("PPG")
            native = x[:, j].copy()
            rate = g2_pulse_rate(x[:, names.index("ECG")], native, fs)
            if not rate["accepted"]:
                raise ValueError("independent ECG/native pulse rate disagreement")
            bpm = float(rate["value"])
            if not 30 <= 2 * bpm <= 220:
                raise ValueError("second harmonic falls outside preregistered plausible cardiac rate range")
            cardiac = ecg_reference(x[:, names.index("ECG")], fs)["rpeaks"]
            phase = np.interp(np.arange(n), cardiac, np.arange(len(cardiac)))
            amplitude = float(np.std(native)) * (1.8 + index * 0.2)
            phase_offset = float(rng.uniform(0, 2 * np.pi))
            x[:, j] = native + amplitude * np.cos(4 * np.pi * phase + phase_offset)
            item = _base(
                record,
                family,
                "verified_rate",
                x,
                names,
                units,
                {"bpm": bpm},
                {"bpm": 1.5},
                "What is the mean cardiac pulse rate represented by PPG over this record, in bpm? Check the pulse interpretation against ECG before reporting bpm.",
                [_region(["PPG", "ECG"], 0, duration)],
                "cardiac_pulse_rate_with_ecg_check",
            )
            item["ground_truth"].update(
                category="G2+G3",
                methods=[
                    rate["method"],
                    "ECG-phase-locked second harmonic enhancement retains the original cardiac cycle rate",
                ],
                agreement={
                    "accepted": rate["difference_bpm"] <= 0.75,
                    "difference_bpm": rate["difference_bpm"],
                    "half_tolerance": 0.75,
                },
                reference_source_hashes=rate["source_hashes"],
            )
            if rate["difference_bpm"] > 0.75:
                raise ValueError("rate references exceed half tolerance")
            params.update(
                kind="cardiac_second_harmonic",
                channel="PPG",
                amplitude=amplitude,
                phase_offset_rad=phase_offset,
                harmonic=2,
                nominal_frequency_hz=2 * bpm / 60,
            )
            sampled = np.arange(0, n, max(1, n // 1000))
            residual = (
                x[sampled, j].astype(float)
                - native[sampled]
                - amplitude * np.cos(4 * np.pi * phase[sampled] + phase_offset)
            )
            if float(np.max(abs(residual))) > 1e-4:
                raise ValueError("independent harmonic sample audit failed")
            verification = "ECG cardiac intervals and alternating PPG pulse morphology distinguish cardiac cycles from the second harmonic"
        else:
            j = names.index("ABP")
            native = x[:, j].copy()
            displacement = (4, -6, 9, -12)[index - 4]
            x[start:, j] += displacement
            available = [t for t in np.linspace(15, duration - 15, 50) if abs(t - onset) > 20]
            distractor = float(rng.choice(available))
            a = round(distractor * fs)
            b = min(n, a + round(4 * fs))
            x[a:b, j] += 35 * np.sin(np.linspace(0, np.pi, b - a))
            params.update(
                kind="sustained_pressure_displacement_and_transient",
                channel="ABP",
                displacement_mmhg=displacement,
                distractor_s=distractor,
                distractor_duration_s=(b - a) / fs,
                distractor_peak_mmhg=35,
            )
            item = _base(
                record,
                family,
                "verified_onset",
                x,
                names,
                units,
                {"onset_s": onset},
                {"onset_s": 5.0},
                "At what time does the sustained displacement of the ABP pressure baseline begin? A brief transient should not count as a sustained pressure change. Report onset_s in seconds and verify persistence on both sides of the candidate event.",
                [
                    _region(["ABP"], max(0, onset - 10), min(duration, onset + 20)),
                    _region(["ABP"], max(0, distractor - 5), min(duration, distractor + 10)),
                ],
                "sustained_change_vs_transient",
            )
            verification = "Compare ABP baseline before and after both candidates; the large transient returns to its preceding baseline"
        item["grounding_rubric"]["verification"] = {"detects_misleading_output": True, "description": verification}
    # Independently audit altered/unaltered channels and exact intervention support.
    untouched = [k for k in range(x.shape[1]) if k != j]
    untouched_ok = bool(np.array_equal(x[:, untouched], original[:n, untouched]))
    if not untouched_ok or not np.isfinite(x).all():
        raise ValueError("intervention changed another channel or introduced invalid samples")
    extra = {"unaltered_channels_exact": untouched_ok}
    if params["kind"] == "polarity_inversion":
        extra["maximum_reflection_residual"] = float(
            np.max(np.abs((x[start:, j].astype(float) + native[start:]) - 2 * params["center"]))
        )
        if extra["maximum_reflection_residual"] > 1e-4:
            raise ValueError("independent polarity audit failed")
    if params["kind"] == "sustained_pressure_displacement_and_transient":
        keep = np.ones(n, dtype=bool)
        keep[a:b] = False
        delta = x[:, j].astype(float) - native
        expected = np.asarray([displacement if k >= start else 0 for k in range(n)])
        extra["maximum_sustained_displacement_residual"] = float(np.max(abs(delta[keep] - expected[keep])))
        if extra["maximum_sustained_displacement_residual"] > 1e-4:
            raise ValueError("independent sustained displacement audit failed")
    if params["kind"] == "cardiac_second_harmonic":
        extra["ecg_unmodified"] = bool(np.array_equal(x[:, names.index("ECG")], original[:n, names.index("ECG")]))
    item["ground_truth"]["applied_array_audit"] = extra
    item["perturbation"] = params
    item["ground_truth"]["native_ecg_quality"] = quality
    if cross_support:
        item["ground_truth"]["native_cross_channel_support"] = cross_support
    item["source_duration_s"] = duration
    return item


def _native_quality(record):
    fs = record["fs"]
    raw = np.asarray(record["x"][: round(min(180, record["duration_s"]) * fs)])
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite native development signals")
    quality = ecg_reference(raw[:, record["channels"].index("ECG")], fs)
    if not quality["accepted"]:
        raise ValueError("native ECG independent detector quality gate")
    return quality["quality"]


def _retry_candidates(records, used):
    """Build a bounded, outcome-independent alternative pool for L2-A.

    These are alternatives, not additional allocated items. Every successful
    construction must receive all freeze-time baselines; slot and subject
    allocation happens only after the unchanged acceptance gates. A rejected
    reference attempt is recorded even though it cannot receive baselines.
    No baseline feedback changes the source order or intervention parameters.
    """
    candidates, exclusions = [], []
    attempts = {(family, index): 0 for family, slots in RETRY_SLOTS.items() for index in slots}
    seen = {family: set() for family in RETRY_SLOTS}
    for source_rank, record in enumerate(islice(records, RETRY_MAX_SOURCE_RECORDS)):
        quality, quality_error = None, None
        for family, slots in RETRY_SLOTS.items():
            if all(attempts[(family, index)] >= RETRY_MAX_ATTEMPTS_PER_SLOT for index in slots):
                continue
            source = (record["dataset"], record["record_key"], record["start_s"])
            required = {"ECG", "ABP", "PPG"} if family == "F3" else {"ECG", "ABP"}
            reason = None
            if not required.issubset(record["channels"]):
                reason = "Required native channels unavailable for unchanged retry task"
            elif record["subject_key"] in used[family]:
                reason = "Source subject already allocated to another original family slot"
            elif source in seen[family]:
                reason = "Repeated development source in registered retry stream"
            if reason is not None:
                exclusions.append(
                    {
                        "family": family,
                        "stage": "retry_source_design",
                        "reason": reason,
                        "subject_key": record["subject_key"],
                        "source_rank": source_rank,
                        "design_slots": list(slots),
                    }
                )
                continue
            seen[family].add(source)
            if quality is None and quality_error is None:
                try:
                    quality = _native_quality(record)
                except ValueError as exc:
                    quality_error = str(exc)
            for index in slots:
                key = (family, index)
                if attempts[key] >= RETRY_MAX_ATTEMPTS_PER_SLOT:
                    continue
                attempts[key] += 1
                retry = {
                    "retry_candidate": True,
                    "design_slot": index,
                    "retry_attempt": attempts[key],
                    "retry_rank": source_rank * sum(map(len, RETRY_SLOTS.values())) + list(attempts).index(key),
                    "retry_schedule": RETRY_SCHEDULE,
                }
                try:
                    if quality_error is not None:
                        raise ValueError(quality_error)
                    item = make_candidate(record, family, index, quality)
                except ValueError as exc:
                    exclusions.append(
                        {
                            **retry,
                            "family": family,
                            "stage": "native_reference" if quality_error is not None else "ground_truth",
                            "reason": str(exc),
                            "subject_key": record["subject_key"],
                            "record_key": record["record_key"],
                            "source_start_s": record["start_s"],
                        }
                    )
                    continue
                item.update(retry)
                candidates.append(item)
        if all(value >= RETRY_MAX_ATTEMPTS_PER_SLOT for value in attempts.values()):
            break
    for (family, index), count in attempts.items():
        built = sum(row["family"] == family and row["design_slot"] == index for row in candidates)
        exclusions.append(
            {
                "family": family,
                "design_slot": index,
                "stage": "retry_schedule_summary",
                "reason": "Registered source retry schedule exhausted; no gate or tolerance changed",
                "attempted": count,
                "reference_eligible": built,
                "reference_rejected": count - built,
                "retry_schedule": RETRY_SCHEDULE,
            }
        )
    return candidates, exclusions


def build_candidates(records, *, retry_records=None):
    candidates = []
    exclusions = []
    remaining = {family: list(range(count)) for family, count in TARGETS.items()}
    if retry_records is not None:
        for family, slots in RETRY_SLOTS.items():
            remaining[family] = [index for index in remaining[family] if index not in slots]
    used = {family: set() for family in TARGETS}
    for record in records:
        if all(not slots for slots in remaining.values()):
            break
        if not {"ECG", "PPG", "ABP"}.issubset(record["channels"]):
            continue
        try:
            quality = _native_quality(record)
        except ValueError as exc:
            exclusions.append(
                {
                    "family": "F2/F3/F6",
                    "stage": "native_reference",
                    "reason": str(exc),
                    "subject_key": record["subject_key"],
                }
            )
            continue
        for family, slots in remaining.items():
            if record["subject_key"] in used[family]:
                continue
            for index in slots.copy():
                try:
                    item = make_candidate(record, family, index, quality)
                except ValueError as exc:
                    exclusions.append(
                        {
                            "family": family,
                            "design_slot": index,
                            "stage": "ground_truth",
                            "reason": str(exc),
                            "subject_key": record["subject_key"],
                        }
                    )
                    continue
                item["design_slot"] = index
                candidates.append(item)
                slots.remove(index)
                used[family].add(record["subject_key"])
                break
    for family, slots in remaining.items():
        if slots:
            exclusions.append(
                {
                    "family": family,
                    "stage": "unfilled",
                    "reason": "No remaining source met preregistered duration and reference support",
                    "design_slots": slots,
                    "count": len(slots),
                }
            )
    if retry_records is not None:
        retries, retry_exclusions = _retry_candidates(retry_records, used)
        candidates.extend(retries)
        exclusions.extend(retry_exclusions)
    return candidates, exclusions
