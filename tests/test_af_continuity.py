import numpy as np
import pytest

from tsmcp.plugins.ecg import af


def analysis():
    times = np.arange(101, dtype=float)
    valid = np.ones(100, bool)
    valid[20:30] = False
    return {
        "supported": [[10, 20], [40, 60]],
        "readable": [[0, 20], [30, 100]],
        "noisy": [[20, 30]],
        "beats": {"p": np.arange(101), "t_s": times, "n_leads": np.ones(101, int), "support": np.ones(101, int)},
        "rr": np.ones(100),
        "valid": valid,
        "statistic": np.full(100, 0.2),
    }


@pytest.mark.parametrize("start,duration,af_s,readable", [(0, 100, 30, 90), (15, 30, 10, 20)])
def test_burden_has_explicit_denominators_and_legacy_aliases(monkeypatch, start, duration, af_s, readable):
    monkeypatch.setattr(af, "af_analysis", lambda *_: analysis())
    out = af.ecg_af_burden(np.zeros(duration), 1, "mV", t0=start)
    assert out["af_s"] == out["irregular_s"] == af_s
    assert out["readable_s"] == readable
    assert out["window_s"] == duration
    assert out["unreadable_s"] == duration - readable and "excluded_s" not in out
    assert out["pct_of_window"] == pytest.approx(100 * af_s / duration, rel=2e-3)  # 3 significant figures
    assert out["pct_of_readable"] == out["value"] == pytest.approx(100 * af_s / readable, rel=2e-3)
    assert out["unit"] == "%" and "pct_of_window" in out["method"] and "readable_s" in out["method"]


def test_burden_zero_af_and_zero_readable(monkeypatch):
    data = analysis()
    data["supported"] = []
    monkeypatch.setattr(af, "af_analysis", lambda *_: data)
    out = af.ecg_af_burden(np.zeros(100), 1, "mV")
    assert out["pct_of_window"] == out["pct_of_readable"] == out["value"] == 0
    data["readable"] = []
    data["noisy"] = [[0, 100]]
    out = af.ecg_af_burden(np.zeros(100), 1, "mV")
    assert out["pct_of_window"] == 0
    assert out["pct_of_readable"] is None and out["value"] is None
    assert out["unreadable_s"] == 100
