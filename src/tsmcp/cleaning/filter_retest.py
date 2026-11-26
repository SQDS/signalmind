"""Development-profile checks on both filter seams and the native task quantities."""

import numpy as np

from tsmcp.processing import CONTEXT_FILTER_VERSION


def filter_retest(case, before, after, operation, policy):
    t, x, valid, est, old = before
    _, y, v2, e2, new = after
    fs = case["fs"]
    params = operation["parameters"]
    padding = round(fs * policy["context_s"])
    actual = params.get("anchor_samples", padding) if operation["op"] == "local_detrend" else padding
    checks = dict(
        observed_context=(
            operation.get("version") == CONTEXT_FILTER_VERSION
            and params.get("context_samples") == padding
            and params.get("write_policy") == "crop_to_core"
            and params.get("read_ranges_samples") == [[a - actual, b + actual] for a, b in operation["ranges_samples"]]
        ),
        filter_support_unchanged=np.array_equal(valid, v2) and np.array_equal(est, e2),
    )
    rows = []
    for a, b in operation["ranges_samples"]:
        lo, hi = a - padding, b + padding
        supported = (
            lo >= 0
            and hi <= len(t)
            and (valid[lo:hi] & ~est[lo:hi] & np.isfinite(x[lo:hi])).all()
            and np.all(np.abs(np.diff(t[lo:hi]) - 1 / fs) <= 1e-6)
            and not any(left < hi and lo < right for left, right in case.get("source_time_conflicts", []))
        )
        checks["observed_context"] = bool(checks["observed_context"] and supported)
        if not supported:
            continue
        edge = max(2, round(0.10 * fs))
        seams = []
        for at in (a, b):
            sl = slice(max(lo, at - edge), min(hi, at + edge + 1))
            dx, dy = np.diff(x[sl]), np.diff(y[sl])
            eps = np.spacing(max(1.0, float(np.max(abs(x[sl])))))
            jump0, jump1 = abs(x[at] - x[at - 1]), abs(y[at] - y[at - 1])
            allowance = policy["seam_ratio"] * max(float(np.quantile(abs(dx), 0.95)), jump0, eps)
            seams.append(
                dict(
                    sample=at,
                    post_jump=float(jump1),
                    limit=float(allowance),
                    post_variation=float(np.abs(dy).sum()),
                    pre_variation=float(np.abs(dx).sum()),
                    pass_jump=bool(jump1 <= allowance),
                    pass_ringing=bool(np.abs(dy).sum() <= policy["seam_ratio"] * max(np.abs(dx).sum(), eps)),
                )
            )
        pre, post = ([z["sample"] for z in pr["events"] if lo <= z["sample"] < hi] for pr in (old, new))
        shifts = [abs(i - j) for i, j in zip(pre, post, strict=False)]
        passed = bool(pre and len(pre) == len(post) and max(shifts, default=0) <= policy["event_shift_samples"])
        local = dict(
            filter_seams=all(z["pass_jump"] for z in seams),
            filter_edge_variation=all(z["pass_ringing"] for z in seams),
            filter_native_timing=passed,
        )
        row = dict(core=[a, b], read=[a - actual, b + actual], seams=seams, pre_events=pre, post_events=post)
        if case["task"] == "ABP":
            old_cycles, new_cycles = (
                [m for m in pr["pressures"] if m["start_sample"] < b and m["stop_sample"] >= a] for pr in (old, new)
            )
            linked = bool(
                old_cycles
                and len(old_cycles) == len(new_cycles)
                and all(
                    abs(z[k] - w[k]) <= policy["event_shift_samples"]
                    for z, w in zip(old_cycles, new_cycles, strict=False)
                    for k in ("start_sample", "stop_sample")
                )
            )
            limit = policy["pressure_limit_mmhg"]
            errors = {
                k: max((abs(z[k] - w[k]) for z, w in zip(old_cycles, new_cycles, strict=False)), default=None)
                for k in ("sbp", "dbp", "mean_pressure")
            }
            errors["dc"] = abs(float(np.mean(y[a:b] - x[a:b])))
            row["max_pressure_delta_mmhg"] = errors
            for key, err in errors.items():
                local["filter_abp_" + key] = linked and err is not None and err <= limit
        else:
            swings = []
            radius = round(fs * (0.08 if case["task"] == "ECG" else 0.20))
            if passed:
                for i, j in zip(pre, post, strict=True):
                    if i + radius < a or b <= i - radius or min(i, j) - radius < lo or max(i, j) + radius >= hi:
                        continue
                    amp = np.ptp(x[i - radius : i + radius + 1])
                    if amp > 0:
                        swings.append(float(np.ptp(y[j - radius : j + radius + 1]) / amp))
            row["local_swing_ratios"] = swings
            low, high = policy["swing_ratio"]
            local["filter_native_swing"] = bool(swings and all(low <= z <= high for z in swings))
        for key, value in local.items():
            checks[key] = bool(value and checks.get(key, True))
        rows.append(row)
    return checks, rows
