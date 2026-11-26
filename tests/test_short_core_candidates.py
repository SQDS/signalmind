import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tsmcp.cleaning.quality_gate import waveform_gate
from tsmcp.cleaning.repair_candidates import comparison_context
from tsmcp.cleaning.repairability import PROFILE
from tsmcp.cleaning.tasks import probe
from tsmcp.processing import PCHIP_TOOL, ProcessingSession


def test_pchip_really_uses_outer_anchors_and_independent_gate_replays_it():
    t = np.arange(1500) / 125
    x = 45 + np.sin(2 * np.pi * t)
    x[405:407] += 4
    s = ProcessingSession(x, fs=125, channel="PPG", unit="source_units")
    p = s.process(s.raw.id, PCHIP_TOOL, [[405, 407]])
    linear = s.process(s.raw.id, "interpolate", [[405, 407]])
    at = [402, 403, 404, 407, 408, 409]
    np.testing.assert_array_equal(p.values[405:407], PchipInterpolator(at, x[at])([405, 406]))
    assert not np.allclose(linear.values[405:407], p.values[405:407], rtol=0, atol=1e-8)
    assert p.operations[0]["parent_id"] == linear.operations[0]["parent_id"] == s.raw.id
    assert p.id == s.process(s.raw.id, PCHIP_TOOL, [[405, 407]]).id
    case = dict(fs=125, task="PPG")
    pr = probe(x, t, 125, s.raw.valid, s.raw.repaired, "PPG")
    before = t, x, s.raw.valid, s.raw.repaired, pr

    def check(values, est, ops):
        return waveform_gate(case, before, (t, values, p.valid, est, pr), ops, [[405, 407]])

    assert check(p.values, p.repaired, p.operations)["status"] == "passed"
    assert not check(linear.values, p.repaired, p.operations)["checks"]["registered_pchip"]
    assert not check(p.values, s.raw.repaired, p.operations)["checks"]["observed_provenance"]
    assert not check(p.values, p.repaired, [dict(p.operations[0], version="unknown")])["checks"]["registered_pchip"]
    with pytest.raises(ValueError, match="anchors"):
        s.process(s.raw.id, PCHIP_TOOL, [[405, 407], [409, 410]])
    with pytest.raises(ValueError, match="anchors"):
        s.process(p.id, PCHIP_TOOL, [[409, 410]])


@pytest.mark.parametrize("issue", ["boundary", "estimated_anchor", "time_gap", "other_core", "uncorrupted_T"])
def test_ineligible_context_cannot_nominate_an_interpolation(issue):
    t = np.arange(1500) / 125
    phase = np.mod(t, 1)
    x = np.exp(-(((phase - 0.12) / 0.025) ** 2)) + 0.3 * np.exp(-(((phase - 0.4) / 0.055) ** 2))
    a, b = 550, 552  # physiological T maximum
    v = np.ones(len(x), bool)
    e = ~v
    others = []
    if issue != "uncorrupted_T":
        x[a:b] += 3
    if issue == "boundary":
        x[b] += 3
    if issue == "estimated_anchor":
        e[a - 2] = True
    if issue == "time_gap":
        t[b + 2 :] += 1
    if issue == "other_core":
        others.append((b + 1, b + 2))
    result, _ = comparison_context(dict(fs=125), (t, x, v, e, {"events": []}), a, b, others, PROFILE["comparison"])
    assert result["reason"] == {
        "boundary": "suspected_contamination_outside_core",
        "uncorrupted_T": "no_compact_excursion_above_flank_variation",
    }.get(issue, "unsupported_or_overlapping_flanks")
