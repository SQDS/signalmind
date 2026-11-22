"""Session-scoped float64 signal processing with immutable parents and sample masks.

Operations are explicit: the caller chooses the method, intervals and parameters.
No benchmark labels, automatic processing recipe, or reference waveform is used.
SciPy is an optional dependency for spectral inspection and filtering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from tsmcp.core.signal import Channel, Signal

VERSION = "processing.v0.2"


def intervals(mask: np.ndarray) -> list[list[int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return edges.reshape(-1, 2).tolist()


@dataclass
class Processed:
    id: str
    values: np.ndarray
    valid: np.ndarray
    artifact: np.ndarray
    repaired: np.ndarray
    operations: list[dict]


class ProcessingSession:
    """One-channel first version; all ranges are zero-based half-open samples."""

    def __init__(
        self,
        values,
        *,
        fs: float,
        channel: str,
        unit: str,
        source_id: str = "input",
        missing=None,
        saturated=None,
        protected_ranges=(),
        max_gap_samples: int | None = None,
        max_transient_samples: int | None = None,
    ):
        x = np.array(values, dtype=np.float64, copy=True)
        if x.ndim != 1 or not x.size or not np.isfinite(fs) or fs <= 0:
            raise ValueError("values must be a nonempty1-D signal with positive sampling rate")
        self.fs, self.channel, self.unit, self.source_id = float(fs), channel, unit, source_id
        self.n = len(x)
        missing = ~np.isfinite(x) if missing is None else np.asarray(missing, bool)
        self.saturated = np.zeros(x.shape, bool) if saturated is None else np.asarray(saturated, bool).copy()
        if missing.shape != x.shape or self.saturated.shape != x.shape:
            raise ValueError("observation masks must match the signal shape")
        self.protected = np.zeros(x.shape, bool)
        for start, end in self._ranges(protected_ranges):
            self.protected[start:end] = True
        self.max_gap_samples = max_gap_samples
        self.max_transient_samples = max_transient_samples
        self.missing_ranges = intervals(missing | ~np.isfinite(x))
        self.versions: dict[str, Processed] = {}
        self.raw = self._store(
            x,
            np.isfinite(x) & ~missing & ~self.saturated,
            missing | self.saturated | ~np.isfinite(x),
            np.zeros(x.shape, bool),
            [],
        )

    def _ranges(self, ranges) -> list[list[int]]:
        result = []
        for start, end in ranges:
            if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
                raise ValueError("ranges require integer sample indices")
            if not 0 <= start < end <= self.n:
                raise ValueError(f"range [{start},{end}) is outside [0,{self.n})")
            result.append([int(start), int(end)])
        return result

    def _store(self, values, valid, artifact, repaired, operations) -> Processed:
        h = hashlib.sha256(self.source_id.encode())
        for a in (values, valid, artifact, repaired):
            h.update(a.tobytes())
            a.setflags(write=False)
        h.update(json.dumps(operations, sort_keys=True).encode())
        sid = "processed_" + h.hexdigest()[:16]
        state = Processed(sid, values, valid, artifact, repaired, operations)
        self.versions[sid] = state
        return state

    def get(self, signal_id: str | None = None) -> Processed:
        return self.raw if signal_id is None else self.versions[signal_id]

    def signal(self, signal_id: str | None = None) -> Signal:
        state = self.get(signal_id)
        return Signal(state.id, [Channel(self.channel, self.fs, state.values, self.unit)])

    def summary(self, signal_id: str | None = None, top_k: int = 12) -> dict:
        state = self.get(signal_id)
        x, valid = state.values, state.valid
        finite = x[valid & np.isfinite(x)]
        candidates = np.flatnonzero(valid[:-1] & valid[1:]) + 1
        differences = np.diff(x)
        order = candidates[np.argsort(-np.abs(differences[candidates - 1]))[: min(50, max(0, top_k))]]
        return {
            "id": state.id,
            "source_id": self.source_id,
            "channel": self.channel,
            "unit": self.unit,
            "fs_hz": self.fs,
            "n_samples": self.n,
            "duration_s": self.n / self.fs,
            "valid_fraction": float(valid.mean()),
            "invalid_ranges_samples": intervals(~valid),
            "saturation_ranges_samples": intervals(self.saturated),
            "protected_ranges_samples": intervals(self.protected),
            "max_gap_samples": self.max_gap_samples,
            "max_transient_samples": self.max_transient_samples,
            "quantiles": dict(
                zip(
                    ("min", "p1", "median", "p99", "max"),
                    np.percentile(finite, [0, 1, 50, 99, 100]).tolist(),
                    strict=True,
                )
            )
            if finite.size
            else {},
            "largest_steps": [
                {
                    "sample": int(i),
                    "time_s": float(i / self.fs),
                    "before": float(x[i - 1]),
                    "after": float(x[i]),
                    "delta": float(differences[i - 1]),
                }
                for i in order
            ],
            "operations": state.operations,
        }

    def samples(self, signal_id: str, start: int, end: int) -> dict:
        self._ranges([[start, end]])
        if end - start > 1000:
            raise ValueError("inspect at most1000 samples per call")
        state = self.get(signal_id)
        return {
            "id": state.id,
            "columns": ["sample", "time_s", "value", "valid", "artifact", "repaired"],
            "rows": [
                [
                    i,
                    i / self.fs,
                    float(state.values[i]) if np.isfinite(state.values[i]) else None,
                    bool(state.valid[i]),
                    bool(state.artifact[i]),
                    bool(state.repaired[i]),
                ]
                for i in range(start, end)
            ],
        }

    def spectrum(self, signal_id: str, start: int = 0, end: int | None = None, segment_s: float = 4.0) -> dict:
        from scipy.signal import welch

        end = self.n if end is None else end
        self._ranges([[start, end]])
        state = self.get(signal_id)
        runs = intervals(state.valid[start:end] & np.isfinite(state.values[start:end]))
        estimates = []
        for left, right in runs:
            if right - left < 16:
                continue
            nseg = min(right - left, max(16, int(round(segment_s * self.fs))))
            f, p = welch(
                state.values[start + left : start + right],
                fs=self.fs,
                window="hann",
                nperseg=nseg,
                noverlap=nseg // 2,
                detrend="constant",
            )
            peaks = np.argsort(-p[1:])[:12] + 1
            lines = []
            for target in (50.0, 60.0):
                near = np.abs(f - target) <= 1.0
                neighbor = (np.abs(f - target) >= 2.0) & (np.abs(f - target) <= 6.0)
                if near.any() and neighbor.any():
                    i = np.flatnonzero(near)[np.argmax(p[near])]
                    lines.append(
                        {
                            "target_hz": target,
                            "peak_hz": float(f[i]),
                            "peak_to_neighbor_ratio": float(p[i] / max(float(np.median(p[neighbor])), 1e-30)),
                            "band_power_fraction": float(p[near].sum() / max(float(p.sum()), 1e-30)),
                        }
                    )
            estimates.append(
                {
                    "range_samples": [start + left, start + right],
                    "segment_samples": nseg,
                    "peaks": [{"hz": float(f[i]), "power": float(p[i])} for i in peaks],
                    "candidate_lines": lines,
                }
            )
        return {"id": state.id, "method": "Welch Hann,50% overlap,constant detrend", "valid_runs": estimates}

    def process(
        self,
        signal_id: str,
        operation: str,
        ranges_samples: list[list[int]],
        *,
        frequency_hz: float | None = None,
        quality_factor: float = 30.0,
        filter_order: int = 4,
        window_s: float | None = None,
    ) -> Processed:
        ranges = self._ranges(ranges_samples)
        if not ranges:
            raise ValueError("specify at least one processing interval")
        parent = self.get(signal_id)
        x, valid, artifact, repaired = (
            a.copy() for a in (parent.values, parent.valid, parent.artifact, parent.repaired)
        )
        parameters = {}
        applied = []
        if operation == "interpolate":
            for start, end in ranges:
                if start == 0 or end == self.n or not valid[start - 1] or not valid[end]:
                    raise ValueError("interpolation requires finite valid anchors on both sides")
                limit = self.max_transient_samples if self.max_transient_samples is not None else self.max_gap_samples
                if limit is not None and end - start > limit:
                    raise ValueError("interval exceeds the declared maximum repair length")
                if self.max_gap_samples is not None and any(
                    left < end and right > start and right - left > self.max_gap_samples
                    for left, right in self.missing_ranges
                ):
                    raise ValueError("interval overlaps a gap longer than the declared maximum gap length")
                if self.protected[start:end].any() or self.saturated[start:end].any():
                    raise ValueError("interpolation overlaps a protected or confirmed saturated interval")
                x[start:end] = np.interp(np.arange(start, end), [start - 1, end], [x[start - 1], x[end]])
                valid[start:end] = artifact[start:end] = repaired[start:end] = True
                applied.append([start, end])
            parameters = {
                "method": "linear",
                "max_gap_samples": self.max_gap_samples,
                "max_transient_samples": self.max_transient_samples,
            }
        elif operation == "local_detrend":
            if self.channel != "ECG" or window_s is None or not 0.18 <= window_s <= 0.8:
                raise ValueError("Local ECG detrending requires a 0.18–0.8 s median window")
            width = max(3, round(self.fs * window_s) | 1)
            radius = width // 2
            anchor = max(width, round(0.7 * self.fs))
            mask = np.zeros(self.n, bool)
            for start, end in ranges:
                mask[start:end] = True
            for start, end in intervals(mask):
                left, right = start - anchor, end + anchor
                if left < 0 or right > self.n or not np.all(valid[left:right] & ~repaired[left:right]):
                    raise ValueError("Local baseline correction requires observed anchors on both sides")
                if self.protected[start:end].any():
                    raise ValueError("Local baseline correction overlaps a protected interval")
                raw = parent.values
                local = np.lib.stride_tricks.sliding_window_view(raw[start - radius : end + radius], width)
                baseline = np.median(local, axis=1)
                target = np.linspace(np.median(raw[left:start]), np.median(raw[end:right]), end - start)
                taper = min(radius, (end - start) // 2)
                weight = np.ones(end - start)
                if taper:
                    ramp = 0.5 - 0.5 * np.cos(np.arange(taper) * np.pi / taper)
                    weight[:taper], weight[-taper:] = ramp, ramp[::-1]
                x[start:end] = raw[start:end] - weight * (baseline - target)
                artifact[start:end] = True
                applied.append([start, end])
            parameters = dict(window_s=window_s, anchor_samples=anchor, taper="half_cosine", method="median_baseline")
        elif operation == "mark_invalid":
            for start, end in ranges:
                valid[start:end] = False
                artifact[start:end] = True
                applied.append([start, end])
            parameters = {"preserve_observed_values": True}
        elif operation in ("notch", "remove_tone", "lowpass"):
            if frequency_hz is None or not 0 < frequency_hz < self.fs / 2:
                raise ValueError("frequency_hz must be between0 and Nyquist")
            if quality_factor <= 0:
                raise ValueError("quality_factor must be positive")
            parameters = {"frequency_hz": frequency_hz}
            for start, end in ranges:
                for left, right in intervals(valid[start:end]):
                    a, b = start + left, start + right
                    if b - a < 10:
                        raise ValueError("filtering requires at least10 contiguous valid samples")
                    if operation == "notch":
                        from scipy.signal import iirnotch, sosfiltfilt, tf2sos

                        numerator, denominator = iirnotch(frequency_hz, quality_factor, fs=self.fs)
                        x[a:b] = sosfiltfilt(tf2sos(numerator, denominator), x[a:b])
                        parameters.update(
                            quality_factor=quality_factor, phase="forward_backward", padding="scipy_default_odd"
                        )
                    elif operation == "lowpass":
                        from scipy.signal import butter, sosfiltfilt

                        sos = butter(filter_order, frequency_hz, btype="lowpass", fs=self.fs, output="sos")
                        x[a:b] = sosfiltfilt(sos, x[a:b])
                        parameters.update(
                            filter_order=filter_order,
                            family="butterworth",
                            phase="forward_backward",
                            padding="scipy_default_odd",
                        )
                    else:
                        width = b - a if window_s is None else min(b - a, round(window_s * self.fs))
                        if width < 10:
                            raise ValueError("tone fitting requires a window of at least10 samples")
                        starts = list(range(a, b - width + 1, max(1, width // 2)))
                        if starts[-1] != b - width:
                            starts.append(b - width)
                        t = np.arange(width) / self.fs
                        design = np.column_stack(
                            [
                                np.sin(2 * np.pi * frequency_hz * t),
                                np.cos(2 * np.pi * frequency_hz * t),
                                np.ones(width),
                            ]
                        )
                        weights = np.hanning(width + 2)[1:-1]
                        fitted, total_weight = np.zeros(b - a), np.zeros(b - a)
                        for left in starts:
                            coefficients = np.linalg.lstsq(design, x[left : left + width], rcond=None)[0]
                            where = slice(left - a, left - a + width)
                            fitted[where] += (design[:, :2] @ coefficients[:2]) * weights
                            total_weight[where] += weights
                        x[a:b] -= fitted / total_weight
                        parameters.update(
                            method="least_squares_sine_cosine_with_intercept",
                            preserve_fitted_intercept=True,
                            window_s=window_s,
                            overlap_fraction=0.5 if window_s is not None else 0.0,
                            blending="positive_hann_normalized" if window_s is not None else "single_fit",
                        )
                    artifact[a:b] = True
                    applied.append([a, b])
        else:
            raise ValueError(f"unknown processing operation: {operation}")
        if not applied:
            raise ValueError("no valid samples in the selected processing interval")
        op = {
            "op": operation,
            "implementation": "tsmcp.processing",
            "version": VERSION,
            "ranges_samples": applied,
            "parameters": parameters,
            "parent_id": parent.id,
        }
        return self._store(x, valid, artifact, repaired, [*parent.operations, op])
