import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.af_episode_metrics import (  # noqa: E402
    complement,
    dev_records,
    intersection,
    score_record,
    suite_record_ids,
)


def test_exact_intersection_and_ranking_rules():
    truth = [(0.25, 10.25), (40.25, 60.25), (80.0, 100.0)]
    episodes = [{"t": 40.0, "dur": 60.0, "af_s": 30.0, "intervals_s": [(40, 50), (80, 100)]}]
    row = score_record("synthetic", 120, truth, episodes, [(0, 90)])
    assert row["annotated_episodes"] == [(0.25, 10.25), (40.25, 100.0)]  # exactly 30 s stays split
    assert row["overlap_s"] == 29.75
    assert row["pct_of_window"] == 25
    assert row["fragmentation"] == 1
    assert row["longest_onset_error_s"] == 0.25
    assert row["burden_error_readable_pp"] == pytest.approx(100 * (30 - 40) / 90)
    assert intersection([(0, 2), (1, 3)], [(2, 4)]) == [[2, 3]]
    assert complement([(2, 4), (3, 5)], 6) == [[0, 2], [5, 6]]


def test_suite_exclusions_override_dev_and_manifests_fail_closed(tmp_path):
    allowed = dev_records({"dev": ["00", "11"], "test_ext": ["11"]}, {"mitdb": {"201"}, "ltafdb": set()})
    assert ("mitdb", "201") not in allowed
    assert ("ltafdb", "11") not in allowed
    assert ("ltafdb", "00") in allowed
    with pytest.raises(FileNotFoundError):
        suite_record_ids(tmp_path)


def test_missed_af_is_not_correct_or_missing_from_fragmentation():
    row = score_record("synthetic", 100, [(10, 90)], [], [(0, 100)])
    assert row["longest_within_tolerance"] is False
    assert row["fragmentation"] == 0
    assert row["longest_duration_ratio"] == 0
    assert row["burden_error_window_pp"] == -80


def test_frozen_gates_reject_onset_swap_and_exact_fragmentation_tie():
    from copy import deepcopy

    from eval.af_episode_metrics import gates

    baseline = {
        "summary": {
            "onset_correct": 1,
            "median_fragmentation": 2,
            "burden_mae_window_pp": 1.0,
            "pooled_sensitivity": 0.9,
            "pooled_ppv": 0.9,
        },
        "records": [
            {"record": "a", "longest_within_tolerance": True},
            {"record": "b", "longest_within_tolerance": False},
        ],
        "runtime_24h": {"median_s": 10},
    }
    candidate = deepcopy(baseline)
    candidate["summary"].update(
        median_fragmentation=1, burden_mae_window_pp=1.5, pooled_sensitivity=0.88, pooled_ppv=0.88
    )
    candidate["runtime_24h"]["median_s"] = 15
    result = gates(baseline, candidate)
    assert all(result[k] for k in ("1_onset", "2_fragmentation", "3_burden_overlap", "4_runtime"))
    candidate["records"][0]["longest_within_tolerance"] = False
    candidate["records"][1]["longest_within_tolerance"] = True
    candidate["summary"]["median_fragmentation"] = 2
    result = gates(baseline, candidate)
    assert not result["1_onset"] and not result["2_fragmentation"]
    assert result["onset_regressions"] == ["a"]


def test_inventory_excludes_suite_and_dev_ids():
    from eval.af_fresh_inventory import remaining

    split = {"unused": ["01", "02", "03", "04"], "dev": ["02"], "test_ext": ["03"]}
    assert remaining(split, {"ltafdb": {"04"}}) == ["01"]


def test_inventory_stats_waveform_without_opening_it(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import numpy as np
    import wfdb

    from eval.af_fresh_inventory import inspect_record

    (tmp_path / "01.hea").write_text("test header")
    (tmp_path / "01.atr").write_bytes(b"test annotations")
    (tmp_path / "01.dat").write_bytes(b"\0" * 2000)
    header = SimpleNamespace(sig_len=1000, fs=10, n_sig=1, file_name=["01.dat"], fmt=["16"])
    ann = SimpleNamespace(sample=np.array([100, 600]), symbol=["+", "+"], aux_note=["(AFIB", "(N"])
    monkeypatch.setattr(wfdb, "rdheader", lambda _: header)
    monkeypatch.setattr(wfdb, "rdann", lambda *_: ann)
    monkeypatch.setattr(wfdb, "rdrecord", lambda *_: pytest.fail("inventory opened waveform"))
    monkeypatch.setattr(np, "fromfile", lambda *_: pytest.fail("inventory opened waveform"))
    result = inspect_record(tmp_path, "01")
    assert result["complete"] and result["duration_s"] == 100
    assert result["total_af_s"] == 50
    assert json.loads(json.dumps(result, allow_nan=False))["eligible"]["burden"] is True
    (tmp_path / "01.dat").write_bytes(b"truncated")
    assert not inspect_record(tmp_path, "01")["complete"]
