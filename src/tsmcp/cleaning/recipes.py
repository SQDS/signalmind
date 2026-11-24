"""Ordinary Python recipes compose actual SignalMind operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsmcp.pipeline import apply_policy, features, select_ranges
from tsmcp.processing import ProcessingSession, intervals

from .common import changed, read, time_support
from .vendor.frozen_v01 import fixed

STANDARD_OPERATIONS = {"notch", "remove_tone", "lowpass", "mark_invalid", "interpolate", "local_detrend"}


def diagnose(values, fs, valid=None):
    x = np.asarray(values, float)
    valid = np.isfinite(x) if valid is None else np.asarray(valid, bool) & np.isfinite(x)
    summary = features(np.where(valid, x, np.nan), fs)
    issues = []
    for a in range(0, len(x), max(1, round(30 * fs))):
        b = min(len(x), a + round(30 * fs))
        f = features(np.where(valid[a:b], x[a:b], np.nan), fs)
        for hz in (50, 60):
            if (f[f"line_{hz}_fraction"] or 0) >= 0.005 and (f[f"line_{hz}_ratio"] or 0) >= 5:
                issues.append(
                    {
                        "kind": "narrowband",
                        "range_samples": [a, b],
                        "frequency_hz": hz,
                        "power_fraction": f[f"line_{hz}_fraction"],
                        "contrast": f[f"line_{hz}_ratio"],
                    }
                )
        if f["step_ratio"] > 0.5:
            issues.append({"kind": "large_steps", "range_samples": [a, b], "value": f["step_ratio"]})
        if f["flat_fraction"] > 0.05:
            issues.append({"kind": "flat", "range_samples": [a, b], "value": f["flat_fraction"]})
    return {"features": summary, "issues": issues, "meaning": "Unlabeled diagnostics, not artifact truth"}


class RecipeContext:
    """Recipe returns ctx.current; numerical writes use ctx.apply, not replacement arrays."""

    def __init__(self, values, valid, estimated, metadata, *, extension=False, recorder=None):
        self.fs = metadata["fs"]
        self.task = metadata["task"]
        self.metadata = dict(metadata)
        self.values = np.asarray(values, float).copy()
        self.values.setflags(write=False)
        self.estimated_before = np.asarray(estimated, bool).copy()
        self.session = ProcessingSession(
            values,
            fs=self.fs,
            channel=self.task,
            unit=metadata["unit"],
            source_id=metadata["id"],
            max_gap_samples=2,
            max_transient_samples=10,
        )
        self.current = self.session._store(
            np.asarray(values, float).copy(),
            np.asarray(valid, bool).copy(),
            ~np.asarray(valid, bool),
            np.asarray(estimated, bool).copy(),
            [],
        )
        self.extension = extension
        self.recorder = recorder
        self.diagnosis = diagnose(values, self.fs, valid)
        self.read_context = [0, len(values)]

    def select(self, kind, **parameters):
        if kind == "patient_flag":
            reason = parameters["reason"]
            return [[r["a"], r["b"]] for r in self.metadata["screen_cores"] if r["kind"] == reason]
        if kind == "screen_family":
            return self.screen_family(**parameters)
        if kind == "review_rule":
            from tsmcp.followup.screen_rule import rescan_rule

            return rescan_rule(
                self.current.values, self.current.valid & ~self.current.repaired, self.fs, self.task, parameters
            )
        return select_ranges(self.current.values, self.current.valid, self.fs, {"kind": kind, **parameters})

    def screen_family(self, family, preset="balanced"):
        """Recompute grouped observed intervals on the current waveform. Selection alone never edits it."""
        from tsmcp.inspection.scan import scan
        from tsmcp.plugins.ecg.rpeaks import detect_r_peaks

        selected = []
        current = self.current
        for left, right in intervals(current.valid & ~current.repaired & np.isfinite(current.values)):
            x = current.values[left:right]
            events, _, _ = scan(x, np.ones(len(x), bool), {}, self.fs, self.task, preset)
            cores = []
            for ev in events:
                if cores and ev["start"] <= cores[-1][1]:
                    cores[-1][1] = max(cores[-1][1], ev["end"])
                    cores[-1][2].append(ev)
                else:
                    cores.append([ev["start"], ev["end"], [ev]])
            peaks = detect_r_peaks(x, self.fs) if self.task == "ECG" else []
            for a, b, hits in cores:
                kinds = {h["kind"] for h in hits}
                primary = next(
                    (k for k in ("abp_pulse_episode", "ecg_baseline_shift", "ecg_excursion") if k in kinds), None
                )
                if primary is None and self.task == "ECG" and kinds == {"large_step"}:
                    primary = "step_unknown_qrs"
                    if len(peaks):
                        distance = max(np.min(abs(peaks - (h["start"] + h["end"]) / 2)) for h in hits)
                        primary = "step_near_qrs" if distance <= self.fs * 0.08 else "step_away_from_qrs"
                if family == primary or (primary is None and family in kinds):
                    selected.append([left + a, left + b])
        return selected

    def apply(self, operation, ranges, *, explanation=None, **parameters):
        allowed = STANDARD_OPERATIONS
        if operation not in allowed:
            raise ValueError(f"Operation {operation!r} is not registered in this recipe category")
        before = self.current
        self.current = self.session.process(before.id, operation, ranges, **parameters)
        if self.recorder:
            actual = self.current.operations[-1]
            self.recorder.record(
                before, self.current, operation, actual["ranges_samples"], actual["parameters"], explanation
            )
        self.session.versions = {self.session.raw.id: self.session.raw, self.current.id: self.current}
        return self.current


def execute(values, valid, estimated, metadata, recipe, module=None, recorder=None):
    if recipe["kind"] == "identity":
        return values.copy(), valid.copy(), estimated.copy(), []
    if recipe["kind"] == "fixed_dsp":
        task = {
            "signal": {"channels": [{"fs_hz": metadata["fs"]}]},
            "public_policy": {"max_gap_samples": 2, "candidate_line_hz": [50, 60]},
        }
        result = fixed(
            task,
            {
                "values": values[None, :],
                "observed_missing_mask": (~valid)[None, :],
                "device_saturation_flag": np.zeros((1, len(values)), bool),
            },
        )
        return (
            result["values"][0],
            result["valid_mask"][0],
            estimated | result["repaired_mask"][0],
            result["operations"],
        )
    if recipe["kind"] == "rule_adaptive_dsp":
        policy = read(Path(__file__).parent / "vendor/adaptive_policy.json")
        if metadata.get("source_time_conflicts"):
            y, good, repairs, operations = values.copy(), valid.copy(), estimated.copy(), []
            for a, b in intervals(time_support(metadata)):
                state, _, _ = apply_policy(
                    values[a:b], metadata["fs"], metadata["task"], metadata["unit"], policy, metadata["id"]
                )
                y[a:b] = state.values
                good[a:b] = state.valid & (valid[a:b] | state.repaired)
                repairs[a:b] |= state.repaired
                for op in state.operations:
                    operations.append({**op, "ranges_samples": [[s + a, e + a] for s, e in op["ranges_samples"]]})
            return y, good, repairs, operations
        state, _, _ = apply_policy(values, metadata["fs"], metadata["task"], metadata["unit"], policy, metadata["id"])
        return state.values, state.valid & (valid | state.repaired), estimated | state.repaired, state.operations
    if module is None:
        raise ValueError("Python recipes execute only in the isolated worker")
    ctx = RecipeContext(
        values, valid, estimated, metadata, extension=recipe["category"] == "custom_extension", recorder=recorder
    )
    out = (
        module.run(ctx)
        if not hasattr(module, "applies") or module.applies(ctx.metadata, ctx.diagnosis)
        else ctx.current
    )
    if out is not ctx.current:
        raise ValueError(
            "Return ctx.current; direct replacement waveforms/peak tables are not standard-operator outputs"
        )
    edits = changed(values, out.values)
    written = np.zeros(len(values), bool)
    for op in out.operations:
        for a, b in op["ranges_samples"]:
            written[a:b] = True
        op["read_context_samples"] = ctx.read_context
    if np.any(edits & ~written):
        raise ValueError("Waveform values changed outside actual operator writes")
    return out.values, out.valid, estimated | out.repaired, out.operations
