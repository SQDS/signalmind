"""v0.1.1 offline fixes: unified precision, rate_from_period, beat duration across heart rates, pinned worktree,
grading of v0.1 runs unchanged."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from eval import grade_signalmind as G  # noqa: E402
from eval.run_signalmind import ROOT as RUN_ROOT  # noqa: E402
from eval.run_signalmind import assert_pinned_worktree  # noqa: E402
from tsmcp.core.signal import Channel, Signal  # noqa: E402
from tsmcp.runtime.ledger import EvidenceLedger, SourceKind  # noqa: E402
from tsmcp.runtime.precision import decimals_of, present, resolution_decimals  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_present_never_adds_precision():
    assert present(148.4437, {"known": True, "value": 0.41}) == 148.0
    assert present(148.4437, {"known": True, "value": 0.04}) == 148.0  # 3 s.f. caps before the resolution does
    assert present(68.7283, {"known": True, "value": 0.25}) == 68.7
    assert present(68.7283, {"known": True, "value": 2.5}) == 69.0
    assert present(0.66173, None) == 0.662 and present(90.6718, {"known": False}) == 90.7
    assert resolution_decimals({"known": True, "value": 0.41}) == 1 and resolution_decimals({"known": False}) is None
    assert decimals_of(148.0) == 0 and decimals_of(68.7) == 1


def test_measure_and_derive_present_the_same_value():
    abp, ecg, _ = synth_abp(fs=125, duration_s=60, seed=31)
    s = Signal(
        "t", [Channel("ECG", 125, ecg.astype(np.float32), "mV"), Channel("ABP", 125, abp.astype(np.float32), "mmHg")]
    )
    from tsmcp.plugins.pressure import abp_beats, abp_sbp, register_signal

    register_signal(s)
    ch = s.channels[1]
    x = ch.data[int(10 * 125) : int(40 * 125)].astype(np.float64)
    meas = abp_sbp(x, 125, "mmHg", 10.0, ch=ch)
    tbl = abp_beats(x, 125, "mmHg", 10.0, ch=ch)
    led = EvidenceLedger("r")
    t = led.record(SourceKind.COMPUTED, "beat_table", tool="ts_measure", tool_call_id="c1", value=tbl["value"], unit="beats",
                   definition_id="abp.beats.v1", parameters={"value_units": {"sbp": "mmHg"}}, artifact_ref="art-x",
                   channels_effective=("ABP",), interval_requested=(10, 40), intervals_effective=[(10, 40)])  # fmt: skip
    d = led.derive(
        "aggregate", [t.evidence_id], column="sbp", rule="beat_weighted", table_rows=tbl["_artifact"]["rows"]
    )
    assert d.value == meas["value"]  # same presentation rule on both paths
    assert abs(d.parameters["value_exact"] - meas["value"]) < 0.5 and d.resolution["known"]


def test_rate_from_period_derivation():
    led = EvidenceLedger("r")
    p = led.record(SourceKind.COMPUTED, "period", tool="ts_measure", tool_call_id="c1", value=0.6617, unit="s",
                   definition_id="generic.period.v1", channels_effective=("PPG",), interval_requested=(0, 180),
                   intervals_effective=[(0, 180)])  # fmt: skip
    r = led.derive("rate_from_period", [p.evidence_id])
    assert (
        r.unit == "bpm"
        and r.value == 90.7
        and r.definition_id == "rate.from_period.v1"
        and r.parents == (p.evidence_id,)
    )
    bad = led.record(SourceKind.COMPUTED, "mean", tool="ts_measure", tool_call_id="c2", value=80.0, unit="mmHg",
                     definition_id="generic.mean.v1", channels_effective=("ABP",))  # fmt: skip
    with pytest.raises(ValueError, match="unit must be s"):
        led.derive("rate_from_period", [bad.evidence_id])


@pytest.mark.parametrize("hr", [40, 60, 90, 130, 170, 205, 240])
@pytest.mark.parametrize("fs", [125.0, 500.0])
def test_beat_duration_rules_across_heart_rates(hr, fs):
    """No special case for any heart rate: accepted fraction and error are bounded from 40 to 240 bpm."""
    from tsmcp.plugins.pressure import abp_sbp, register_signal

    abp, ecg, beats = synth_abp(fs=fs, duration_s=60, hr_bpm=hr, sbp=130, dbp=70, seed=hr, hrv=0.03)
    s = Signal(
        "t", [Channel("ECG", fs, ecg.astype(np.float32), "mV"), Channel("ABP", fs, abp.astype(np.float32), "mmHg")]
    )
    register_signal(s)
    ch = s.channels[1]
    x = ch.data[int(10 * fs) : int(50 * fs)].astype(np.float64)
    r = abp_sbp(x, fs, "mmHg", 10.0, ch=ch)
    truth = np.mean([b["sbp"] for b in beats if b["t_foot_s"] >= 10 and b["t_end_s"] <= 50])
    accepted = r["counts"]["accepted"]
    total = r["counts"]["accepted"] + r["counts"]["rejected"]
    assert accepted / total >= 0.9, (hr, fs, r["counts"])
    assert abs(r["value"] - truth) < 1.5, (hr, fs, r["value"], truth)


def test_pinned_worktree_check():
    with pytest.raises(SystemExit, match="working repository"):
        assert_pinned_worktree(RUN_ROOT)
    for cand in (Path.home() / "tsmcp-server-SM2", Path.home() / "tsmcp-server-SM"):
        if cand.exists():
            sha = assert_pinned_worktree(cand)
            assert len(sha) == 40
            break
    else:
        pytest.skip("no frozen worktree on this machine")


def test_v01_runs_grade_identically(monkeypatch, tmp_path):
    """The v0.1 main-run grading is reproduced on the stable metrics (schema compatibility)."""
    stored = ROOT / "eval/results/signalmind/p10abp-analysis.json"
    if not stored.exists() or not list((ROOT / "eval/runs").glob("p10abp-C-*-main-*")):
        pytest.skip("v0.1 run directories not present")
    monkeypatch.setattr(
        sys, "argv", ["grade", "--suite", "p10abp", "--tag", "main", "main2", "--out", str(tmp_path / "re.json")]
    )
    G.main()
    old = json.loads(stored.read_text())["summary"]["by_arm"]
    new = json.loads((tmp_path / "re.json").read_text())["summary"]["by_arm"]
    for arm in ("C_tools", "C_full"):
        for k in ("grounded_correct", "correct_any_support", "controls_correct", "unsupported_precise_claims",
                  "incorrect_abstentions", "incomplete_execution", "runtime_rejections", "verifier_false_rejections",
                  "cost_usd_total"):  # fmt: skip
            assert old[arm][k] == new[arm][k], (arm, k, old[arm][k], new[arm][k])


def test_git_worktrees_are_detached():
    out = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=ROOT, text=True)
    blocks = [b for b in out.strip().split("\n\n") if "tsmcp-server-SM" in b]
    assert blocks and all("detached" in b for b in blocks)
