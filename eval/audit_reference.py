"""Offline audit references; never imports the evaluated TS-MCP implementation.

Implementation 1 executes only the four explicitly authorized, read-only local
code files. Implementation 2 independently detects events and computes estimates.
These are algorithmic references, not clinical or expert ground truth.
"""

from __future__ import annotations

import hashlib
import sys
import types
from functools import cache
from pathlib import Path

import numpy as np
from scipy import signal, stats
from scipy.integrate import simpson
from scipy.interpolate import interp1d

REFERENCE_PATHS = {
    "rpeaks": Path.home() / "bp2rr_new/data_process/01_rpeak_detector/rpeak_detector.py",
    "beats": Path.home() / "world_model/hemoworld/beats.py",
    "brs": Path.home() / "bp2rr_new/downstream/02_brs_xbrs_seqbrs_v3_1/physiology_core/seqbrs.py",
    "tfa": Path.home() / "cbv_true/2026.9.10_项目分类整理/03_TFA计算/02_TFA代码/01_核心计算/standard.py",
}

METHODS = {
    "ecg": "Local derivative-energy R detector versus independent 5–25 Hz absolute-amplitude maxima; 50 ms event matching, RR/template QC",
    "bp": "Local ECG-guided systolic peaks/diastolic feet, trapezoidal beat MAP versus independent pressure minima, intervening raw maxima/minima and Simpson beat MAP; equal-beat means",
    "pat": "Same minimum-foot definition in both implementations: ECG R to the earliest minimum of centered quadratic-smoothed ABP (max(5,round(0.080*fs)|1) samples) from R+32 ms through 40 ms before its earliest systolic maximum; systolic search R+100 ms to min(R+750 ms,next R+300 ms), inclusive rounded sample bounds; 32–600 ms PAT and >=8 mmHg pulse pressure; arithmetic mean over complete beats with a subsequent detected foot. Local derivative-energy R detection/ECG-guided extrema/greedy pairing versus independent bandpass-amplitude R detection, explicit quadratic-convolution smoothing, globally enumerated extrema with interval indexing and forward event pairing",
    "brs": "Local unique maximal SBP ramps/classify_sequence versus separately enumerated monotonic runs and scipy linear regression; delay 1 (SBP in R_k cycle paired with RR_k), >=3 beats, >=1 mmHg steps, >=5 ms RR steps, r>=0.85, 0<slope<50, >=3 valid ramps; equal-ramp mean",
    "tfa": "Local standard.py scipy Welch/CSD versus independent explicit Hann-window FFT spectra; common nonoverlapping 1 s means, fs=1 Hz, 100 s segments, 50% overlap, constant detrend, mean spectra then abs(Sxy/Sxx); unweighted LF-bin mean [0.07,0.20) Hz",
    "tilt": "Independent ECG/pressure event streams; HR=60/RR interpolated at interval midpoints, SBP at systolic peaks, 1 Hz interpolation; baseline median [-45,-10) s, response median [20,50) s; direction threshold 3 bpm/5 mmHg; latency first 5 consecutive seconds crossing threshold within [0,60] s",
}


@cache
def _reference(name: str):
    """Load approved source without executing packages or creating external pyc.

    beats.py imports only FS_HZ from its package. A private module supplies that
    documented 125 Hz constant; every invoked event function receives fs
    explicitly. No PulseDB loader or code in a data directory is imported.
    """
    path = REFERENCE_PATHS[name]
    module_name = f"_audit_authorized_{name}"
    if name == "beats":
        package = types.ModuleType("_audit_authorized_hemoworld")
        package.__path__ = []
        constants = types.ModuleType("_audit_authorized_hemoworld.pulsedb")
        constants.FS_HZ = 125.0
        sys.modules[package.__name__] = package
        sys.modules[constants.__name__] = constants
        module_name = "_audit_authorized_hemoworld.beats"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module  # dataclasses resolve their defining module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def source_hashes(*names: str) -> dict[str, str]:
    paths = {name: REFERENCE_PATHS[name] for name in names}
    paths["independent_audit_reference"] = Path(__file__)
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _wave(values, fs: float, minimum_s: float = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if fs < 60 or not np.isfinite(fs) or values.ndim != 1 or len(values) < minimum_s * fs:
        raise ValueError("insufficient waveform duration or invalid sample rate")
    if not np.isfinite(values).all() or np.ptp(values) <= 1e-9:
        raise ValueError("nonfinite or constant waveform")
    return values


def independent_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Direct band-limited QRS extrema, without derivative-energy integration."""
    ecg = _wave(ecg, fs)
    filtered = signal.sosfiltfilt(signal.butter(3, [5, 25], fs=fs, btype="bandpass", output="sos"), ecg)
    envelope = abs(filtered)
    scale = float(np.quantile(envelope, 0.995))
    peaks, _ = signal.find_peaks(envelope, distance=round(0.30 * fs), prominence=0.25 * scale, height=0.30 * scale)
    return peaks[(peaks >= round(0.16 * fs)) & (peaks < len(ecg) - round(0.16 * fs))]


def _event_f1(first: np.ndarray, second: np.ndarray, distance: float) -> float:
    i = j = matches = 0
    while i < len(first) and j < len(second):
        if abs(first[i] - second[j]) <= distance:
            i += 1
            j += 1
            matches += 1
        elif first[i] < second[j]:
            i += 1
        else:
            j += 1
    return 2 * matches / max(1, len(first) + len(second))


def _mean_rate(peaks: np.ndarray, fs: float) -> float:
    if len(peaks) < 3:
        raise ValueError("fewer than three detected events")
    return float(60 * fs * (len(peaks) - 1) / (peaks[-1] - peaks[0]))


def ecg_reference(ecg: np.ndarray, fs: float) -> dict:
    ecg = _wave(ecg, fs)
    peaks, filtered = _reference("rpeaks").detect_r_peaks(ecg, fs)
    secondary = independent_rpeaks(ecg, fs)
    if min(len(peaks), len(secondary)) < 20:
        raise ValueError("fewer than twenty R peaks")
    rr = np.diff(peaks) / fs
    width = round(0.12 * fs)
    templates = np.array([filtered[p - width : p + width + 1] for p in peaks if width <= p < len(ecg) - width])
    templates -= templates.mean(axis=1, keepdims=True)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
    template = np.median(templates, axis=0)
    template /= max(np.linalg.norm(template), 1e-12)
    quality = {
        "finite_fraction": 1.0,
        "n_rpeaks": int(len(peaks)),
        "rr_300_2000ms_fraction": float(np.mean((rr >= 0.3) & (rr <= 2))),
        "template_correlation_ge_0_75_fraction": float(np.mean(templates @ template >= 0.75)),
        "independent_peak_f1_50ms": _event_f1(peaks, secondary, 0.05 * fs),
        "mean_hr_difference_bpm": abs(_mean_rate(peaks, fs) - _mean_rate(secondary, fs)),
    }
    accepted = (
        quality["rr_300_2000ms_fraction"] >= 0.98
        and quality["template_correlation_ge_0_75_fraction"] >= 0.90
        and quality["independent_peak_f1_50ms"] >= 0.95
        and quality["mean_hr_difference_bpm"] <= 1.5
    )
    return {
        "accepted": bool(accepted),
        "rpeaks": peaks,
        "independent_rpeaks": secondary,
        "mean_hr_bpm": _mean_rate(peaks, fs),
        "quality": quality,
        "method": METHODS["ecg"],
    }


def _pulse_peaks(pulse: np.ndarray, fs: float) -> np.ndarray:
    pulse = _wave(pulse, fs)
    smooth = signal.sosfiltfilt(signal.butter(3, 8, fs=fs, output="sos"), pulse)
    spread = float(np.quantile(smooth, 0.95) - np.quantile(smooth, 0.05))
    peaks, _ = signal.find_peaks(smooth, distance=round(0.35 * fs), prominence=0.30 * spread)
    return peaks


def g2_pulse_rate(ecg: np.ndarray, pulse: np.ndarray, fs: float) -> dict:
    result = ecg_reference(ecg, fs)
    pulses = _pulse_peaks(pulse, fs)
    rate = _mean_rate(pulses, fs)
    difference = abs(rate - result["mean_hr_bpm"])
    return {
        "accepted": result["accepted"] and difference <= 1.5,
        "value": result["mean_hr_bpm"],
        "pulse_rate_bpm": rate,
        "difference_bpm": difference,
        "quality": result["quality"],
        "method": METHODS["ecg"] + "; pulse channel independently low-passed at 8 Hz and prominence maxima counted",
        "source_hashes": source_hashes("rpeaks"),
    }


def _pressure(abp: np.ndarray, fs: float) -> np.ndarray:
    abp = _wave(abp, fs)
    if np.quantile(abp, 0.01) < 20 or np.quantile(abp, 0.99) > 260:
        raise ValueError("pressure outside 20–260 mmHg QC range")
    # Quantized high-rate recordings naturally repeat adjacent pressure values.
    # Reject sustained plateaus, not the aggregate fraction of equal neighbours.
    flat = np.diff(abp) == 0
    changes = np.diff(np.r_[False, flat, False].astype(int))
    durations = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    if len(durations) and np.max(durations) / fs >= 0.5:
        raise ValueError("pressure flat/clipped plateau lasting >=0.5 s")
    return abp


def _independent_pressure(abp: np.ndarray, fs: float) -> dict:
    """Unguided trough detection with raw extrema inside each trough interval."""
    smooth = signal.sosfiltfilt(signal.butter(3, 8, fs=fs, output="sos"), abp)
    spread = float(np.quantile(smooth, 0.95) - np.quantile(smooth, 0.05))
    feet, _ = signal.find_peaks(-smooth, distance=round(0.35 * fs), prominence=max(8, 0.25 * spread))
    rows = []
    for left, right in zip(feet[:-1], feet[1:], strict=True):
        if not 0.30 <= (right - left) / fs <= 2.0:
            continue
        segment = abp[left : right + 1]
        peak = left + int(np.argmax(segment))
        if np.ptp(segment) < 8:
            continue
        rows.append(
            (
                left,
                right,
                peak,
                float(np.max(segment)),
                float(np.min(segment)),
                simpson(segment, dx=1 / fs) / ((right - left) / fs),
            )
        )
    if len(rows) < 20:
        raise ValueError("fewer than twenty independent arterial beats")
    arr = np.asarray(rows)
    return {key: arr[:, i] for i, key in enumerate(("feet", "ends", "peaks", "sbp", "dbp", "map"))}


def _guided_pressure(abp: np.ndarray, rpeaks: np.ndarray, fs: float) -> dict:
    feet, peaks = _reference("beats").detect_abp_pulses_guided(abp, rpeaks, fs)
    if len(feet) < 20 or len(feet) / len(rpeaks) < 0.90:
        raise ValueError("insufficient ECG-guided arterial beat coverage")
    maps = np.array([np.trapezoid(abp[a : b + 1]) / (b - a) for a, b in zip(feet[:-1], feet[1:], strict=True)])
    return {
        "feet": feet[:-1],
        "ends": feet[1:],
        "peaks": peaks[:-1],
        "sbp": abp[peaks[:-1]],
        "dbp": abp[feet[:-1]],
        "map": maps,
    }


def agreement(first, second, tolerance: float, *, relative: bool = False) -> dict:
    """The agreement gate is HALF the grading tolerance, inclusive."""
    first, second = float(first), float(second)
    limit = tolerance * abs(first) if relative else tolerance
    difference = abs(first - second)
    return {
        "accepted": bool(np.isfinite([first, second, limit]).all() and limit > 0 and difference <= limit / 2),
        "difference": difference if np.isfinite(difference) else None,
        "tolerance": {"value": tolerance, "kind": "relative" if relative else "absolute"},
        "half_tolerance": limit / 2,
    }


def _comparison(kind: str, first, second, gate: dict, quality: dict, *sources: str) -> dict:
    return {
        "reference1": first,
        "reference2": second,
        **gate,
        "quality": quality,
        "methods": METHODS[kind],
        "source_hashes": source_hashes(*sources),
    }


def _ecg_pressure(ecg, abp, fs):
    ecg_result = ecg_reference(ecg, fs)
    if not ecg_result["accepted"]:
        raise ValueError("ECG failed independent-detector/RR/template quality checks")
    abp = _pressure(abp, fs)
    if len(abp) != len(ecg):
        raise ValueError("synchronized channels must have equal length")
    first = _guided_pressure(abp, ecg_result["rpeaks"], fs)
    second = _independent_pressure(abp, fs)
    count_ratio = len(second["sbp"]) / len(first["sbp"])
    if not 0.90 <= count_ratio <= 1.10:
        raise ValueError("independent pressure beat counts disagree by >10%")
    return ecg_result, abp, first, second


def compare_bp(ecg, abp, fs) -> dict:
    e, _, first, second = _ecg_pressure(ecg, abp, fs)
    one = {key: float(np.mean(first[key])) for key in ("sbp", "dbp", "map")}
    two = {key: float(np.mean(second[key])) for key in one}
    gates = {key: agreement(one[key], two[key], 5) for key in one}
    gate = {
        "accepted": all(g["accepted"] for g in gates.values()),
        "difference": {k: g["difference"] for k, g in gates.items()},
        "tolerance": {"value": 5, "kind": "absolute", "unit": "mmHg"},
        "half_tolerance": 2.5,
    }
    return _comparison("bp", one, two, gate, e["quality"], "rpeaks", "beats")


def independent_pat(ecg: np.ndarray, abp: np.ndarray, fs: float) -> np.ndarray:
    """Independently locate the same minimum foot as the authorized reference.

    The estimand includes centered 80-ms quadratic smoothing and the stated
    search intervals. Here smoothing weights come from a polynomial design
    matrix, and global extrema are indexed into intervals; the local reference
    instead uses Savitzky–Golay filtering and scans each ECG-guided interval.
    Neither local events nor its foot-search code are used here.
    """
    rpeaks = independent_rpeaks(ecg, fs)
    abp = _wave(abp, fs)
    if len(ecg) != len(abp):
        raise ValueError("synchronized channels must have equal length")
    window = max(5, round(0.08 * fs) | 1)
    offsets = np.arange(window, dtype=float) - window // 2
    design = np.column_stack((np.ones(window), offsets, offsets**2))
    weights = np.linalg.pinv(design)[0]
    smooth = signal.convolve(abp, weights, mode="same", method="direct")
    radius = window // 2
    for sample_slice, positions in ((slice(None, radius), offsets[:radius]), (slice(-radius, None), offsets[-radius:])):
        edge = abp[:window] if sample_slice.start is None else abp[-window:]
        coefficients = np.linalg.lstsq(design, edge, rcond=None)[0]
        smooth[sample_slice] = np.polynomial.polynomial.polyval(positions, coefficients)
    _, max_properties = signal.find_peaks(smooth, plateau_size=True)
    _, min_properties = signal.find_peaks(-smooth, plateau_size=True)
    # Preserve the earliest sample on a flat extremum, as the definition's
    # first minimum/maximum convention requires.
    maxima = max_properties["left_edges"]
    minima = min_properties["left_edges"]

    def extreme(events, left, right, *, maximum=False):
        start, stop = np.searchsorted(events, [left, right], side="left")
        indexes = np.unique(np.r_[left, events[start:stop], right])
        values = -smooth[indexes] if maximum else smooth[indexes]
        return int(indexes[np.lexsort((indexes, values))[0]])

    feet = []
    previous_foot = -1
    following = np.r_[rpeaks[1:], len(abp) - 1]
    for rpeak, following_r in zip(rpeaks, following, strict=True):
        peak_left = int(rpeak + round(0.10 * fs))
        peak_right = min(len(abp) - 1, int(rpeak + round(0.75 * fs)), int(following_r + round(0.30 * fs)))
        if peak_right <= peak_left:
            continue
        systolic = extreme(maxima, peak_left, peak_right, maximum=True)
        foot_left, foot_right = int(rpeak + round(0.032 * fs)), systolic - round(0.040 * fs)
        if foot_right <= foot_left:
            continue
        foot = extreme(minima, foot_left, foot_right)
        pat = (foot - rpeak) / fs
        if 0.032 <= pat <= 0.60 and smooth[systolic] - smooth[foot] >= 8 and foot > previous_foot:
            feet.append(foot)
            previous_foot = foot
    times = []
    r_index = previous_r = -1
    # As in implementation 1, a complete beat has a subsequent detected foot.
    # Pair with a forward event sweep, independent of its binary-search pairing.
    for foot in feet[:-1]:
        while r_index + 1 < len(rpeaks) and rpeaks[r_index + 1] < foot:
            r_index += 1
        if r_index < 0 or r_index <= previous_r:
            continue
        pat = (foot - rpeaks[r_index]) / fs
        if 0.032 <= pat <= 0.60:
            times.append(1000 * pat)
            previous_r = r_index
    if len(times) < 20 or len(times) / len(rpeaks) < 0.90:
        raise ValueError("insufficient minimum-foot PAT coverage")
    return np.asarray(times)


def compare_pat(ecg, abp, fs) -> dict:
    e, abp, first, _ = _ecg_pressure(ecg, abp, fs)
    ri, fi = _reference("beats").greedy_pair_r_foot(e["rpeaks"], first["feet"], fs=fs)
    if len(ri) < 20 or len(ri) / len(first["feet"]) < 0.90:
        raise ValueError("insufficient local R-foot pairing coverage")
    values1 = 1000 * (first["feet"][fi] - e["rpeaks"][ri]) / fs
    values2 = independent_pat(ecg, abp, fs)
    one, two = float(np.mean(values1)), float(np.mean(values2))
    quality = {
        **e["quality"],
        "paired_beats1": len(values1),
        "paired_beats2": len(values2),
        "pat_sd1_ms": float(np.std(values1)),
        "pat_sd2_ms": float(np.std(values2)),
    }
    return _comparison("pat", one, two, agreement(one, two, 20), quality, "rpeaks", "beats")


def independent_brs(sbp: np.ndarray, rr_ms: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    """Enumerate maximal same-direction runs independently of local ramp code."""
    sbp, rr_ms, cycles = np.asarray(sbp), np.asarray(rr_ms), np.asarray(cycles, dtype=int)
    if len(sbp) != len(cycles):
        raise ValueError("SBP/cycle length mismatch")
    slopes = []
    for direction in (1, -1):
        qualifies = (direction * np.diff(sbp) >= 1) & (np.diff(cycles) == 1)
        changes = np.diff(np.r_[False, qualifies, False].astype(int))
        for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
            if stop - start < 2:
                continue
            indexes = cycles[start : stop + 1]
            # Match the local all-delay support/QC denominator without using it.
            if indexes[0] < 1 or indexes[-1] + 1 >= len(rr_ms):
                continue
            supporting_rr = rr_ms[indexes[0] - 1 : indexes[-1] + 2]
            if not np.isfinite(supporting_rr).all() or np.any((supporting_rr < 300) | (supporting_rr > 2000)):
                continue
            response = rr_ms[indexes]
            if np.any(direction * np.diff(response) < 5):
                continue
            fit = stats.linregress(sbp[start : stop + 1], response)
            if fit.rvalue >= 0.85 and 0 < fit.slope < 50:
                slopes.append(fit.slope)
    return np.asarray(slopes)


def _cycle_pressures(rpeaks, peaks, values):
    chosen = {}
    for peak, value in zip(peaks, values, strict=True):
        cycle = int(np.searchsorted(rpeaks, peak, side="right")) - 1
        if 0 <= cycle < len(rpeaks) - 1 and (cycle not in chosen or value > chosen[cycle][1]):
            chosen[cycle] = (int(peak), float(value))
    cycles = np.array(sorted(chosen), dtype=int)
    return cycles, np.array([chosen[k][0] for k in cycles]), np.array([chosen[k][1] for k in cycles])


def compare_brs(ecg, abp, fs) -> dict:
    e, _, first, second = _ecg_pressure(ecg, abp, fs)
    local = _reference("brs")
    r = e["rpeaks"]
    cycles, peaks, values = _cycle_pressures(r, first["peaks"], first["sbp"])
    n = len(r) - 1
    intervals = local.RRIntervalTable(
        fs,
        np.arange(n),
        np.arange(n),
        np.arange(n) + 1,
        r[:-1],
        r[1:],
        np.diff(r) * 1000 / fs,
        (r[:-1] + r[1:] + 2) / (2 * fs),
        (r[:-1] + r[1:]) / (2 * fs),
    )
    beats = [
        local.CycleSBP(int(c), int(p), float(v), i)
        for i, (c, p, v) in enumerate(zip(cycles, peaks, values, strict=True))
    ]
    slopes1 = []
    for ramp in local.detect_unique_maximal_ramps(beats):
        candidate = local.build_reference_candidate(ramp, intervals)
        validity = local.evaluate_gt_validity(candidate, intervals, delay=1)
        if validity.valid:
            slopes1.append(validity.ols.slope)
    r2 = e["independent_rpeaks"]
    cycles2, _, values2 = _cycle_pressures(r2, second["peaks"], second["sbp"])
    slopes2 = independent_brs(values2, np.diff(r2) * 1000 / fs, cycles2)
    if min(len(slopes1), len(slopes2)) < 3:
        raise ValueError("fewer than three valid sequence-BRS ramps in either implementation")
    one, two = float(np.mean(slopes1)), float(np.mean(slopes2))
    quality = {**e["quality"], "valid_sequences1": len(slopes1), "valid_sequences2": len(slopes2)}
    return _comparison("brs", one, two, agreement(one, two, 0.25, relative=True), quality, "rpeaks", "beats", "brs")


def independent_tfa(abp_1hz: np.ndarray, cbfv_1hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch transfer estimate via explicit FFTs; no scipy spectral routines."""
    if len(abp_1hz) != len(cbfv_1hz) or len(abp_1hz) < 100:
        raise ValueError("TFA needs at least 100 synchronized 1-Hz samples")
    starts = range(0, len(abp_1hz) - 99, 50)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(100) / 100)
    xx, xy = [], []
    for start in starts:
        x, y = abp_1hz[start : start + 100], cbfv_1hz[start : start + 100]
        xfft = np.fft.rfft((x - np.mean(x)) * window)
        yfft = np.fft.rfft((y - np.mean(y)) * window)
        xx.append(abs(xfft) ** 2)
        xy.append(np.conjugate(xfft) * yfft)
    power = np.mean(xx, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = abs(np.mean(xy, axis=0) / power)
    return np.fft.rfftfreq(100), gain


def compare_tfa(abp, cbfv, fs) -> dict:
    abp = _pressure(abp, fs)
    cbfv = _wave(cbfv, fs, minimum_s=180)
    if len(abp) != len(cbfv) or fs != round(fs):
        raise ValueError("TFA requires equal channel lengths and integer waveform fs")
    if np.quantile(cbfv, 0.01) <= 0 or np.quantile(cbfv, 0.99) > 200:
        raise ValueError("CBFV outside 0–200 cm/s QC range")
    samples = int(fs)
    length = len(abp) // samples * samples
    x = abp[:length].reshape(-1, samples).mean(axis=1)
    y = cbfv[:length].reshape(-1, samples).mean(axis=1)
    local = _reference("tfa")
    f, pxx, pyy, pxy = local.spectrum_for_window(x, y)
    curve = local.tfa_from_spectra(f, pxx, pyy, pxy)
    band = local.band_mask(f, "lf")
    coherence = float(np.mean(curve["coherence"][band]))
    if not np.all(pxx[band] > 1e-8) or not np.isfinite(coherence) or coherence < 0.30:
        raise ValueError("insufficient LF excitation/coherence (mean coherence <0.30)")
    one = float(local.band_mean(curve, "lf", "gain"))
    f2, gain2 = independent_tfa(x, y)
    two = float(np.mean(gain2[(f2 >= 0.07 - 1e-12) & (f2 < 0.20 - 1e-12)]))
    return _comparison(
        "tfa",
        one,
        two,
        agreement(one, two, 0.20, relative=True),
        {"mean_lf_coherence": coherence, "seconds": len(x), "welch_segments": 1 + (len(x) - 100) // 50},
        "tfa",
    )


def _response(times, values, onset_s, threshold):
    grid = np.arange(-45, 61, dtype=float)
    if times[0] > onset_s - 45 or times[-1] < onset_s + 60:
        raise ValueError("tilt response lacks required baseline/response support")
    interpolated = np.interp(onset_s + grid, times, values)
    baseline = float(np.median(interpolated[(grid >= -45) & (grid < -10)]))
    difference = float(np.median(interpolated[(grid >= 20) & (grid < 50)]) - baseline)
    direction = "increase" if difference >= threshold else "decrease" if difference <= -threshold else "unchanged"
    latency = None
    if direction != "unchanged":
        post = interpolated[grid >= 0] - baseline
        crossed = post * (1 if direction == "increase" else -1) >= threshold
        for start in range(len(crossed) - 4):
            if crossed[start : start + 5].all():
                latency = float(start)
                break
        if latency is None:
            raise ValueError("tilt direction lacks sustained threshold crossing")
    return direction, latency


def independent_response(times, values, onset_s, threshold):
    """Same frozen tilt estimand via separate interpolation/window arithmetic."""
    times, values = np.asarray(times), np.asarray(values)
    if times[0] > onset_s - 45 or times[-1] < onset_s + 60:
        raise ValueError("tilt response lacks required baseline/response support")
    interpolate = interp1d(times, values, kind="linear", bounds_error=True, assume_sorted=True)
    baseline = np.median(interpolate(onset_s + np.arange(-45, -10)))
    change = np.median(interpolate(onset_s + np.arange(20, 50))) - baseline
    if abs(change) < threshold:
        return "unchanged", None
    sign = 1 if change > 0 else -1
    direction = "increase" if sign == 1 else "decrease"
    differences = interpolate(onset_s + np.arange(61)) - baseline
    crossed = np.asarray(sign * differences >= threshold, dtype=np.int64)
    sustained = np.convolve(crossed, np.ones(5, dtype=np.int64), mode="valid")
    candidates = np.flatnonzero(sustained == 5)
    if not len(candidates):
        raise ValueError("tilt direction lacks sustained threshold crossing")
    return direction, float(candidates[0])


def compare_tilt(ecg, abp, fs, onset_s) -> dict:
    e, _, first, second = _ecg_pressure(ecg, abp, fs)
    outputs = []
    for r, beats, estimator in (
        (e["rpeaks"], first, _response),
        (e["independent_rpeaks"], second, independent_response),
    ):
        hr_direction, hr_latency = estimator((r[:-1] + r[1:]) / (2 * fs), 60 * fs / np.diff(r), onset_s, 3)
        sbp_direction, sbp_latency = estimator(beats["peaks"] / fs, beats["sbp"], onset_s, 5)
        outputs.append(
            {
                "hr_direction": hr_direction,
                "sbp_direction": sbp_direction,
                "hr_latency_s": hr_latency,
                "sbp_latency_s": sbp_latency,
            }
        )
    one, two = outputs
    differences, accepted = {}, True
    for name in ("hr", "sbp"):
        directions_match = one[f"{name}_direction"] == two[f"{name}_direction"]
        if one[f"{name}_latency_s"] is None or two[f"{name}_latency_s"] is None:
            diff = None
            latencies_match = one[f"{name}_latency_s"] is two[f"{name}_latency_s"]
        else:
            diff = abs(one[f"{name}_latency_s"] - two[f"{name}_latency_s"])
            latencies_match = diff <= 5
        differences[name] = {"direction_match": directions_match, "latency_s": diff}
        accepted = accepted and directions_match and latencies_match
    gate = {
        "accepted": accepted,
        "difference": differences,
        "tolerance": {"value": 10, "kind": "absolute", "unit": "s; exact direction"},
        "half_tolerance": 5,
    }
    return _comparison("tilt", one, two, gate, e["quality"], "rpeaks", "beats")
