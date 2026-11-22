"""Dev-only validation must fail closed and exclude held-out signals before opening them."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.tasks import ucr  # noqa: E402
from eval.validate_discord import held_out_ids  # noqa: E402


def test_dev_ids_skip_held_signals_before_loading(monkeypatch):
    paths = [Path(f"{i:03d}_UCR_Anomaly_fake_10_20_30.txt") for i in range(1, 6)]
    loaded = []

    def load(path):
        sid = ucr.file_id(path)
        assert sid not in {1, 3}
        loaded.append(sid)
        return np.zeros(40), {"id": sid}

    monkeypatch.setattr(ucr, "DEV_COUNT", 2)
    monkeypatch.setattr(ucr, "files", lambda: paths)
    monkeypatch.setattr(ucr, "load", load)
    assert ucr.dev_ids({1, 3}) == {2, 4}
    assert loaded == [2, 4]


def test_exclusion_manifests_required_and_other_files_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        held_out_ids(tmp_path)
    for suite, sid in (("p1", 1), ("p3", 3)):
        rows = [{"qtype": "anomaly_time", "answer": {"id": sid}}, {"qtype": "other", "answer": {}}]
        (tmp_path / f"{suite}.answers.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "unrelated.answers.jsonl").write_text("must never be parsed")
    assert held_out_ids(tmp_path) == {1, 3}


def test_append_preserves_entire_original_prefix():
    from eval.discord_baseline import single_scale_discords
    from eval.discord_variants import variant_discords

    x = np.sin(np.arange(4000) * 2 * np.pi / 40) + np.random.default_rng(2).normal(0, 0.1, 4000)
    original = single_scale_discords(x, 10.0, t0=123.4, k=5)
    appended = variant_discords(x, 10.0, t0=123.4, k=8, variant="append")
    assert len(appended) == 8
    assert appended[:5] == original
    assert sorted(appended, key=lambda e: -e["score"]) == appended
    for extra in appended[5:]:
        for other in original:
            assert abs(extra["t"] + extra["dur"] / 2 - other["t"] - other["dur"] / 2) > max(extra["dur"], other["dur"])


def test_rrf_rewards_independent_agreement_without_duplicate_votes():
    from eval.discord_variants import Candidate, rrf

    # A single-source leader loses to a shared second-place candidate; duplicate rows cannot add votes.
    a = [Candidate(0, 10, 100, "a"), Candidate(100, 10, 1, "a"), Candidate(101, 10, 0.9, "a")]
    b = [Candidate(200, 10, 1e6, "b"), Candidate(102, 10, 2, "b")]
    events = rrf([a, b], 3)
    assert events[0].start == 100
    assert events[0].score == pytest.approx(2 / 12)
    assert len(events) == 3
    assert [e.start for e in events[1:]] == [0, 200]


def test_robust_nn_scores_ignore_scale_and_handle_zero_mad():
    from eval.discord_variants import robust_z

    nn = np.array([1.0, 2.0, 3.0, 4.0, 15.0])
    np.testing.assert_allclose(robust_z(nn), robust_z(nn * 7 + 20))
    assert robust_z(nn)[-1] > 5
    assert np.all(robust_z(np.ones(10)) == 0)
    assert np.all(np.isfinite(robust_z(np.r_[np.ones(10), 2, np.inf])))


def test_cheap_scorers_localize_amplitude_change():
    from eval.discord_variants import cheap_rankings

    period = 100
    x = np.sin(np.arange(10_000) * 2 * np.pi / period)
    x[5000:5300] *= 4
    for ranking in cheap_rankings(x, period):
        assert ranking
        assert 4800 <= ranking[0].centre <= 5500


@pytest.mark.parametrize("period", [None, 40, 113])
@pytest.mark.parametrize("k", [1, 3, 5, 8])
@pytest.mark.parametrize("n", [32, 4000])
def test_restored_discord_exact_original_candidates(period, k, n, monkeypatch):
    from eval import discord_baseline
    from tsmcp.core import detect

    x = np.random.default_rng(9).normal(0, 0.1, n)
    if period is not None:
        x += np.sin(np.arange(x.size) * 2 * np.pi / period)
    x[1200:1230] += 2
    x[2000:2004] = np.nan
    # Force period/fallback so this exercises both paths without depending on the estimator's decision.
    monkeypatch.setattr(detect, "_period_samples", lambda *args: period)
    monkeypatch.setattr(discord_baseline, "_period_samples", lambda *args: period)
    expected = discord_baseline.single_scale_discords(x, 13.0, t0=17.2, k=k)
    actual = detect.discords(x, 13.0, t0=17.2, k=k)
    assert all("verify" in e["caveat"] for e in actual)
    assert [{key: value for key, value in e.items() if key != "caveat"} for e in actual] == expected
