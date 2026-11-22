from types import SimpleNamespace

import numpy as np

from tsmcp.followup.pattern import context_range
from tsmcp.followup.rescan import compare
from tsmcp.followup.reviewer import review

RULE = dict(
    metric="step",
    scale_window_s=4,
    baseline_window_s=0.25,
    seed_ratio=1.2,
    tail_ratio=0.5,
    quiet_s=0.05,
    max_seed_s=0.1,
)


def signal():
    t = np.arange(5000) / 125
    x = np.sin(t * 2 * np.pi)
    x[2600] -= 9
    return x


def test_context_uses_both_ends_and_handles_boundaries():
    assert context_range(5000, 125, [1400, 1700], 5) == (775, 2325)
    assert context_range(5000, 125, [100, 1400], 10) == (0, 2650)
    assert context_range(5000, 125, [4800, 4990], 5) == (4175, 5000)


def test_mask_is_not_recovery_and_unflagged_signals_are_searchable():
    raw = signal()
    source = np.ones(len(raw), bool)
    post = raw.copy()
    post[3500] += 10
    good = source.copy()
    good[2590:2610] = False
    result = compare(raw, source, post, good, 125, RULE)
    assert result["findings"][0]["status"] == "excluded_from_task"
    assert result["findings"][0]["still_present_in_numeric_signal"]
    assert len(result["after_numeric"]) == 2
    assert len(result["newly_triggered"]) == 1
    assert result["newly_triggered"][0][0] < 3500 < result["newly_triggered"][0][1]
    assert compare(raw, source, raw, np.zeros_like(good), 125, RULE)["evaluated_samples"] == 0


def test_reviewer_receives_expanded_image_then_trials_rule(tmp_path):
    moves = iter(
        [
            ("expand_context", {"flank_s": 10, "reason": "check surrounding pulses"}),
            ("try_detector", {"rule": RULE}),
            (
                "register_detector",
                {
                    "rule": RULE,
                    "observed": "isolated dip",
                    "suspected_cause": "possible dropout",
                    "alternative_or_limitation": "mechanism unknown",
                    "suspected_range_s": [20.79, 20.83],
                    "method_explanation": "look for a disproportionate sample step",
                },
            ),
        ]
    )

    def request(messages, tools, system):
        name, args = next(moves)
        last = messages[-1]["content"]
        if name == "try_detector":
            assert any(b["type"] == "image" for b in last[0]["content"])
        block = {"type": "tool_use", "name": name, "input": args, "id": name}
        return SimpleNamespace(id=name, content=[SimpleNamespace(model_dump=lambda **kwargs: block)])

    x = signal()
    decision = review(
        SimpleNamespace(request=request, model="fixture"),
        tmp_path,
        dict(id="c", task="PPG", unit="au", fs=125),
        x,
        np.ones(len(x), bool),
        "raw-fixture",
        [2600, 2601],
    )
    assert decision["status"] == "registered"
    assert decision["views"][1]["view_samples"] == [1350, 3851]
    assert decision["seed_method_hits"]


def test_reviewer_baseline_selector_and_extension_use_observed_samples():
    import pytest

    from tsmcp.cleaning.recipes import RecipeContext
    from tsmcp.followup.screen_rule import rescan_rule
    from tsmcp.plugins.ecg.rpeaks import detect_r_peaks

    fs = 125
    t = np.arange(20 * fs) / fs
    x = np.zeros_like(t)
    for peak in np.arange(0.5, 19.7, 0.83):
        x += np.exp(-0.5 * ((t - peak) / 0.015) ** 2)
    x += 1.2 * np.exp(-0.5 * ((t - 9.62) / 0.38) ** 2)
    valid = np.ones(len(x), bool)
    rule = {"method": "baseline_shift", "threshold": 1.5}
    scored = np.zeros_like(valid)
    hits = rescan_rule(x, valid, fs, "ECG", rule, evaluated=scored)
    assert len(hits) == 1 and scored[round(9.62 * fs)] and not scored[0]
    metadata = {"id": "trial", "unit": "source_units", "fs": fs, "task": "ECG"}
    ctx = RecipeContext(x, valid, ~valid, metadata, extension=True)
    assert ctx.select("review_rule", **rule) == hits
    out = ctx.apply("local_detrend", hits, window_s=0.32, explanation="local baseline trial")
    a, b = hits[0]
    assert np.array_equal(x[:a], out.values[:a]) and np.array_equal(x[b:], out.values[b:])
    assert np.median(out.values[a:b]) < np.median(x[a:b])
    assert np.array_equal(out.valid, valid) and not out.repaired.any()
    assert max(abs(detect_r_peaks(x, fs) - detect_r_peaks(out.values, fs))) <= 1
    valid[a - 1] = False
    broken = RecipeContext(x, valid, np.zeros_like(valid), metadata, extension=True)
    with pytest.raises(ValueError, match="anchors"):
        broken.apply("local_detrend", hits, window_s=0.32)


def test_screen_family_selector_keeps_supported_cores_and_does_not_write():
    from tsmcp.cleaning.recipes import RecipeContext

    fs = 125
    t = np.arange(20 * fs) / fs
    x = 0.02 * np.sin(2 * np.pi * t)
    x[5 * fs : 6 * fs] += 3
    x[12 * fs : 13 * fs] += 3
    valid = np.ones(len(x), bool)
    valid[11 * fs : 14 * fs] = False
    ctx = RecipeContext(x, valid, np.zeros_like(valid), {"task": "ECG", "fs": fs, "id": "unit", "unit": "mV"})
    cores = ctx.select("screen_family", family="ecg_baseline_shift")
    assert any(a <= round(5.5 * fs) < b for a, b in cores)
    assert all(b <= 11 * fs or a >= 14 * fs for a, b in cores)
    assert np.array_equal(ctx.current.values, x) and np.array_equal(ctx.current.valid, valid)


def test_patient_limits_and_rate_keep_real_support():
    from tsmcp.cleaning.common import recipe_input_key
    from tsmcp.inspection.patient_profile import amplitude_reference, beat_pairs, patient_flags, rate_reference

    amp = amplitude_reference([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], -1.0, 2.0, upper_limit=1.5)
    assert amp["display"][0] < -1 and amp["display"][1] > 2
    rate = rate_reference([60.0, 60.0, 60.0, 60.0, 240.0])
    assert rate["median"] == 60 and rate["low"] == 48 and rate["high"] == 72
    x = np.zeros(40)
    x[10:14] = 2
    observed = np.ones(40, bool)
    observed[11] = False
    p = {"events": [{"support_id": i, "sample": s} for i, s in [(0, 0), (0, 10), (1, 12), (1, 15)]]}
    assert beat_pairs(p) == [(0, 10), (12, 15)]
    found = [(a, b, k) for a, b, k, _, _ in patient_flags(x, observed, 10, beat_pairs(p), amp, rate)]
    assert found == [(10, 11, "y_high"), (12, 14, "y_high"), (12, 15, "rate_high")]
    assert recipe_input_key({"input_sha256": "x"}) == "x"
    c = {"input_sha256": "x", "screen_cores": [], "patient_reference": {"rate": rate}}
    assert recipe_input_key(c) != recipe_input_key({**c, "screen_cores": [{"kind": "y_high", "a": 10, "b": 11}]})
