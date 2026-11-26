import numpy as np
import pytest

from tsmcp.cleaning.event_hypotheses import conflict_evidence, protected_event, pulse_evidence
from tsmcp.cleaning.quality_gate import spike_check
from tsmcp.cleaning.tasks import probe


@pytest.mark.parametrize("variant", ["narrow", "wide", "premature", "rate_change", "inverted"])
def test_real_beat_alternatives_remain_protected(variant):
    fs = 125
    t = np.arange(16 * fs) / fs
    at = np.arange(1.0, 15.0)
    if variant == "premature":
        at[7] -= 0.3
    if variant == "rate_change":
        at[7:] = at[7] + np.arange(len(at[7:])) * 0.72
    x = np.zeros(len(t))
    for i, p in enumerate(at):
        width = 0.055 if variant == "wide" and i == 7 else 0.018
        sign = -1 if variant == "inverted" and i == 7 else 1
        x += sign * np.exp(-(((t - p) / width) ** 2))
    c = dict(id="ecg", record_id="r", fs=fs, task="ECG")
    good = np.ones(len(t), bool)
    pr = probe(x, t, fs, good, ~good, "ECG")
    a = round(fs * at[7])
    q = protected_event(c, pr["events"], a, a + 2)
    assert q is not None
    result = conflict_evidence(c, (t, x, good, ~good, pr), a, a + 2, q, [])
    assert set(result["hypotheses"]) == {"H0", "H1", "H2", "H3"}
    assert result["independent"]["status"] in {"agreement", "disagreement"}
    assert len(result["morphology"]["neighbour_samples"]) >= 3
    assert result["action"] == "preserve" and not result["event_amended"]
    assert spike_check(c, x, good, ~good, pr, a, a + 2)["status"] != "evaluated_no_alert"


def test_pulse_evidence_learns_nonzero_delay_and_requires_shared_clock():
    fs = 125
    t = np.arange(fs * 16) / fs
    times = np.arange(1.0, 15.0)
    good = np.ones(len(t), bool)
    pr = {"events": [{"sample": round((z + 0.24) * fs)} for z in times]}
    c = dict(id="c", clock_id="aligned", fs=fs, has_source_time=True, record_id="r")
    p = dict(c, id="p", task="PPG")
    data = (t, np.sin(t), good, ~good, pr)
    out = pulse_evidence(c, p, data, 8.0, times)
    assert out["status"] == "consistent" and out["neighbour_median_lag_s"] == pytest.approx(0.24)
    assert pulse_evidence(c, dict(p, clock_id="other"), data, 8.0, times)["reason"] == "shared_clock_unverified"
    masked = good.copy()
    masked[round(7.8 * fs) : round(8.6 * fs)] = False
    assert pulse_evidence(c, p, (t, np.sin(t), masked, ~good, pr), 8.0, times)["status"] == "unavailable"
