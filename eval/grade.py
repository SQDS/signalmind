"""Deterministic graders. A missing/unparseable answer is `no_answer` — never silently scored as a negative."""

from __future__ import annotations

import math

CHANGE_TOL_S = 9.0  # max(5 s, 1.5% of a 600 s record)
ARTIFACT_TOL_S = 5.0
RHYTHM_TOL_S = 10.0
PERIOD_REL = 0.05
HR_REL = 0.05
AF_BURDEN_ABS = 5.0  # percentage points
LONG_AF_TOL_S = 60.0
LONG_EVENT_TOL_S = 30.0

# Audit qtypes are separate so no existing suite's scoring changes. Row 12's
# latency and row 15's relative gain tolerances are provisional frozen conventions:
# the approved design did not specify them. See docs/tasks/audit-prep-report.md.
AUDIT_ABS_TOL = {
    "audit_rate": 3.0,
    "audit_motion": 2.0,
    "audit_step": 5.0,
    "audit_unusable": 10.0,
    "audit_bp": 5.0,
    "audit_valsalva": 10.0,
    "audit_damped": 5.0,
    "audit_tilt": 15.0,
    "audit_shift": 0.05,
    "audit_pat": 20.0,
    "audit_response": 10.0,
    "audit_cbfv_tilt": 15.0,
}
AUDIT_REL_TOL = {"audit_brs": 0.25, "audit_tfa": 0.20}
AUDIT_FIELDS = {
    "audit_rate": ("bpm",),
    "audit_motion": ("t",),
    "audit_step": ("t",),
    "audit_unusable": ("percent",),
    "audit_bp": ("sbp", "dbp", "map"),
    "audit_valsalva": ("t",),
    "audit_damped": ("t",),
    "audit_tilt": ("t",),
    "audit_shift": ("channel", "shift_s"),
    "audit_pat": ("pat_ms",),
    "audit_corrupted": ("channel",),
    "audit_response": ("hr_direction", "sbp_direction", "hr_latency_s", "sbp_latency_s"),
    "audit_brs": ("brs_ms_mmhg",),
    "audit_cbfv_tilt": ("t",),
    "audit_tfa": ("gain",),
}


def _audit_num(value) -> float | None:
    """Require a finite JSON number; booleans and numeric strings are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except OverflowError:
        return None


def _audit_within(error: float, tolerance: float) -> bool:
    # Include exact decimal tolerance boundaries despite floating-point subtraction.
    return error <= tolerance or math.isclose(error, tolerance, rel_tol=1e-12, abs_tol=1e-12)


def _grade_audit(qtype: str, answer: dict, pred: dict) -> dict:
    fields = AUDIT_FIELDS[qtype]
    errors = []
    ok = True
    if qtype == "audit_response":
        for channel in ("hr", "sbp"):
            direction, latency = f"{channel}_direction", f"{channel}_latency_s"
            expected = answer[direction]
            actual = pred.get(direction)
            if not isinstance(actual, str) or actual not in ("increase", "decrease", "unchanged"):
                ok = False
            if actual != expected:
                ok = False
            if expected == "unchanged":
                if latency not in pred or pred[latency] is not None:
                    ok = False
            else:
                value = _audit_num(pred.get(latency))
                if value is None or value < 0:
                    ok = False
                else:
                    errors.append(abs(value - answer[latency]))
        tolerance = AUDIT_ABS_TOL[qtype]
    else:
        tolerance = AUDIT_REL_TOL.get(qtype, AUDIT_ABS_TOL.get(qtype, 0.0))
        for field in fields:
            if field == "channel":
                channel = pred.get(field)
                ok = ok and isinstance(channel, str) and bool(channel) and channel == answer[field]
                continue
            value = _audit_num(pred.get(field))
            if value is None or (field != "shift_s" and value < 0):
                ok = False
                continue
            if field == "percent" and value > 100:
                ok = False
                continue
            error = abs(value - answer[field])
            if qtype in AUDIT_REL_TOL:
                # A zero reference has zero allowable absolute error; avoid division by zero.
                if answer[field] == 0:
                    ok = ok and error == 0
                    error = 0.0 if error == 0 else None
                else:
                    error /= abs(answer[field])
            if error is not None:
                errors.append(error)
    ok = ok and all(_audit_within(error, tolerance) for error in errors)
    return {"status": "ok", "correct": bool(ok), "err": max(errors) if errors else None}


def _num(v) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _time_or_null(pred, gt, tol) -> bool:
    p = pred.get("t", "missing")
    if gt is None:
        return p is None
    p = _num(p)
    return p is not None and abs(p - gt) <= tol


def grade(qtype: str, answer: dict, pred: dict | None) -> dict:
    """Returns {"status": "ok"|"no_answer", "correct": bool|None, "err": float|None}."""
    if pred is None:
        return {"status": "no_answer", "correct": None, "err": None}
    if qtype in AUDIT_FIELDS:
        if not isinstance(pred, dict):
            return {"status": "no_answer", "correct": None, "err": None}
        return _grade_audit(qtype, answer, pred)
    a = answer
    err = None
    if qtype == "change_time":
        ok = _time_or_null(pred, a["t"], CHANGE_TOL_S)
    elif qtype == "rhythm_change":
        ok = _time_or_null(pred, a["t"], RHYTHM_TOL_S)
    elif qtype == "period":
        p = _num(pred.get("period_s"))
        err = None if p is None else abs(p - a["period_s"]) / a["period_s"]
        ok = err is not None and err <= PERIOD_REL
    elif qtype == "mean_hr":
        p = _num(pred.get("bpm"))
        err = None if p is None else abs(p - a["bpm"]) / a["bpm"]
        ok = err is not None and err <= HR_REL
    elif qtype == "artifact":
        present = pred.get("present")
        if not isinstance(present, bool):
            return {"status": "no_answer", "correct": None, "err": None}
        if present != a["present"]:
            ok = False
        elif not a["present"]:
            ok = True
        else:
            p = _num(pred.get("t"))
            ok = p is not None and abs(p - a["t"]) <= ARTIFACT_TOL_S
    elif qtype == "spike_count":
        p = _num(pred.get("count"))
        ok = p is not None and int(round(p)) == a["count"]
        err = None if p is None else abs(p - a["count"])
    elif qtype == "af_burden":
        p = _num(pred.get("percent"))
        err = None if p is None else abs(p - a["percent"])
        ok = err is not None and err <= AF_BURDEN_ABS
    elif qtype == "longest_af":
        ok = _time_or_null(pred, a["t"], LONG_AF_TOL_S)
    elif qtype in ("episode_start", "long_artifact"):
        ok = _time_or_null(pred, a["t"], LONG_EVENT_TOL_S)
    elif qtype == "anomaly_time":
        p = _num(pred.get("t"))
        slack = max(100, a["end"] - a["begin"])
        ok = p is not None and a["begin"] - slack <= p <= a["end"] + slack
    else:
        raise ValueError(f"unknown qtype {qtype}")
    return {"status": "ok", "correct": bool(ok), "err": err}


def oracle(qtype: str, a: dict) -> dict:
    """The reference answer in the model's answer format (harness self-test: must score 100%)."""
    if qtype in AUDIT_FIELDS:
        return {field: a[field] for field in AUDIT_FIELDS[qtype]}
    return {
        "change_time": lambda: {"t": a["t"]},
        "rhythm_change": lambda: {"t": a["t"]},
        "period": lambda: {"period_s": a["period_s"]},
        "mean_hr": lambda: {"bpm": a["bpm"]},
        "artifact": lambda: {"present": a["present"], "t": a["t"]},
        "spike_count": lambda: {"count": a["count"]},
        "anomaly_time": lambda: {"t": a["begin"]},
        "af_burden": lambda: {"percent": a["percent"]},
        "longest_af": lambda: {"t": a["t"]},
        "episode_start": lambda: {"t": a["t"]},
        "long_artifact": lambda: {"t": a["t"]},
    }[qtype]()


def null(qtype: str) -> dict:
    """A constant 'nothing there' answer (must NOT score well)."""
    if qtype in AUDIT_FIELDS:
        return {field: None for field in AUDIT_FIELDS[qtype]}
    return {
        "change_time": {"t": None},
        "rhythm_change": {"t": None},
        "period": {"period_s": 1.0},
        "mean_hr": {"bpm": 75},
        "artifact": {"present": False, "t": None},
        "spike_count": {"count": 0},
        "anomaly_time": {"t": 0},
        "af_burden": {"percent": 0},
        "longest_af": {"t": None},
        "episode_start": {"t": 0},
        "long_artifact": {"t": None},
    }[qtype]
