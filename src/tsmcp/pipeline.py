"""Reusable, data-dependent processing policies; execution never calls an LLM.

A policy is developed on examples, then applied to complete continuous runs.
Features are observable diagnostics, not probabilities or clean-signal truth.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tsmcp.processing import ProcessingSession, intervals

SCHEMA = "signalmind.dataset-policy.v0.1"
OPERATIONS = {"interpolate", "mark_invalid", "notch", "remove_tone", "lowpass"}
SELECTORS = {"all", "missing", "outliers", "flat"}
FEATURES = {
    "finite_fraction",
    "flat_fraction",
    "robust_span",
    "step_ratio",
    "mean",
    "line_50_fraction",
    "line_60_fraction",
    "line_50_ratio",
    "line_60_ratio",
}


def policy_hash(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_policy(policy):
    if policy.get("schema_version") != SCHEMA or not isinstance(policy.get("goal"), str):
        raise ValueError(f"Specify schema_version={SCHEMA!r} and a goal string")
    channels = policy.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("channels must map channel kinds to lists of rules; [] explicitly preserves a channel")
    for kind, rules in channels.items():
        if not isinstance(rules, list):
            raise ValueError(f"{kind}: rules must be a list")
        names = set()
        for rule in rules:
            if not isinstance(rule.get("id"), str) or rule["id"] in names:
                raise ValueError("Each channel's rules require unique string ids")
            names.add(rule["id"])
            if rule.get("operation") not in OPERATIONS or rule.get("selector", {}).get("kind") not in SELECTORS:
                raise ValueError("Unknown operation or selector")
            for name, limits in rule.get("when", {}).items():
                if name not in FEATURES or not isinstance(limits, dict) or set(limits) - {"min", "max"}:
                    raise ValueError("when uses documented feature names with min/max bounds")
                if not limits or not all(np.isfinite(v) for v in limits.values()):
                    raise ValueError("Feature bounds must be finite")
            if rule.get("fallback", "preserve") not in {"preserve", "mark_invalid"}:
                raise ValueError("fallback must be preserve or mark_invalid")
            selector = rule["selector"]
            if selector.get("factor", 1.5) < 0 or selector.get("window_s", 10) <= 0:
                raise ValueError("Selector factor must be nonnegative and window_s positive")
            if selector.get("pad_samples", 0) < 0 or selector.get("min_duration_s", 0) < 0:
                raise ValueError("Selector padding/duration must be nonnegative")
    # Check serializability now, before writing any output.
    policy_hash(policy)
    return policy


def features(values, fs):
    from scipy.signal import welch

    x = np.asarray(values, float)
    finite = np.isfinite(x)
    good = x[finite]
    span = float(np.percentile(good, 95) - np.percentile(good, 5)) if good.size else 0.0
    adjacent = finite[1:] & finite[:-1]
    steps = np.abs(np.diff(x)[adjacent])
    result = {
        "finite_fraction": float(finite.mean()),
        "flat_fraction": float(np.mean(steps == 0)) if steps.size else 0.0,
        "robust_span": span,
        "step_ratio": float(np.percentile(steps, 99.9) / max(span, 1e-12)) if steps.size else 0.0,
        "mean": float(good.mean()) if good.size else None,
    }
    # Spectra use contiguous observations; missing regions are never concatenated.
    spectra, weights = [], []
    for a, b in intervals(finite):
        if b - a < max(32, round(fs)):
            continue
        f, power = welch(x[a:b], fs=fs, nperseg=min(b - a, round(8 * fs)), detrend="constant")
        row = {}
        for hz in (50, 60):
            near = np.abs(f - hz) <= 1
            neighbor = (np.abs(f - hz) >= 2) & (np.abs(f - hz) <= 6)
            row[f"line_{hz}_fraction"] = float(power[near].sum() / max(power[f >= 0.5].sum(), 1e-30))
            row[f"line_{hz}_ratio"] = (
                float(power[near].max() / max(np.median(power[neighbor]), 1e-30))
                if near.any() and neighbor.any()
                else 0.0
            )
        spectra.append(row)
        weights.append(b - a)
    for hz in (50, 60):
        for suffix in ("fraction", "ratio"):
            name = f"line_{hz}_{suffix}"
            result[name] = float(np.average([s[name] for s in spectra], weights=weights)) if spectra else None
    return result


def select_ranges(values, valid, fs, selector):
    from scipy.ndimage import maximum_filter1d

    x, kind = np.asarray(values), selector["kind"]
    if kind == "all":
        return [[0, len(x)]]
    if kind == "missing":
        return intervals(~np.isfinite(x))
    mask = np.zeros(len(x), bool)
    if kind == "outliers":
        width = max(1, round(selector.get("window_s", 10) * fs))
        factor = selector.get("factor", 1.5)
        for a in range(0, len(x), width):
            b = min(len(x), a + width)
            good = x[a:b][valid[a:b] & np.isfinite(x[a:b])]
            if good.size < 16:
                continue
            low, high = np.percentile(good, [5, 95])
            span = max(float(high - low), 1e-12)
            mask[a:b] = (x[a:b] < low - factor * span) | (x[a:b] > high + factor * span)
    elif kind == "flat":
        flat = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (np.diff(x) == 0)
        for a, b in intervals(flat):
            if (b - a + 1) / fs >= selector.get("min_duration_s", 1):
                mask[a : b + 1] = True
    pad = int(selector.get("pad_samples", 0))
    if pad:
        mask = maximum_filter1d(mask, size=2 * pad + 1, mode="constant")
    return intervals(mask)


def apply_policy(values, fs, kind, unit, policy, source_id="run"):
    """Apply one already validated policy to a complete, genuinely continuous run."""
    session = ProcessingSession(
        values,
        fs=fs,
        channel=kind,
        unit=unit,
        source_id=source_id,
        max_gap_samples=max(0, round(policy.get("max_gap_ms", 16) * fs / 1000)),
        max_transient_samples=max(0, round(policy.get("max_repair_ms", 80) * fs / 1000)),
    )
    state, trace = session.raw, []
    if kind not in policy["channels"]:
        raise ValueError(f"Policy must explicitly cover channel kind {kind}")
    for rule in policy["channels"][kind]:
        observed = features(np.where(state.valid, state.values, np.nan), fs)
        matched = all(
            observed[name] is not None
            and ("min" not in bounds or observed[name] >= bounds["min"])
            and ("max" not in bounds or observed[name] <= bounds["max"])
            for name, bounds in rule.get("when", {}).items()
        )
        entry = {"rule": rule["id"], "condition_met": matched, "applied_ranges": [], "skipped": []}
        if matched:
            ranges = select_ranges(state.values, state.valid, fs, rule["selector"])
            for a, b in ranges:
                try:
                    updated = session.process(state.id, rule["operation"], [[a, b]], **rule.get("parameters", {}))
                except ValueError as exc:
                    # Unsupported boundaries/short spans are explicit, never counted as successful repair.
                    entry["skipped"].append({"range": [a, b], "reason": str(exc)})
                    if rule.get("fallback") == "mark_invalid":
                        updated = session.process(state.id, "mark_invalid", [[a, b]])
                    else:
                        continue
                state = updated
                entry["applied_ranges"].append([a, b])
                # The batch executor retains original and current arrays, not a full copy per rule/interval.
                session.versions = {session.raw.id: session.raw, state.id: state}
        trace.append(entry)
    before, after = features(values, fs), features(state.values, fs)
    changed = ~np.isclose(values, state.values, atol=1e-12, rtol=1e-10, equal_nan=True)
    common = np.isfinite(values) & np.isfinite(state.values) & state.valid
    delta = state.values[common] - np.asarray(values)[common]
    diagnostic = {
        "samples": len(values),
        "changed_samples": int(changed.sum()),
        "input_invalid_samples": int((~np.isfinite(values)).sum()),
        "output_invalid_samples": int((~state.valid).sum()),
        "repaired_samples": int(state.repaired.sum()),
        "valid_fraction": float(state.valid.mean()),
        "common_support_mean_shift": float(delta.mean()) if delta.size else None,
        "max_abs_change": float(np.max(np.abs(delta))) if delta.size else None,
        "before": before,
        "after_observed_values": after,
        "rule_trace": trace,
        "operations": state.operations,
        "quality_verdict": None,
    }
    return state, changed, diagnostic
