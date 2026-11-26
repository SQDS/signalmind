import json

import numpy as np
import pytest

from eval.repair_benchmark.agent_smoke import ScriptedTransport
from eval.repair_benchmark.episode import run_episode
from eval.repair_benchmark.providers import Replay
from eval.repair_benchmark.public import prepare
from eval.repair_benchmark.runner import baseline
from eval.repair_benchmark.scoring import macro_errors, report
from tsmcp.cleaning.common import read, sha, write


def pp_fixture(root):
    t = np.arange(1500) / 125
    x = 45 + np.sin(2 * np.pi * t)
    ref = x.copy()
    x[405:407] += 4
    good = np.ones(len(t), bool)
    public = root / "public"
    c = prepare(
        public,
        "C01",
        "PPG",
        "source_units",
        125,
        t,
        x,
        good,
        ~good,
        offset=100,
        acquisition=dict(
            species="human", device="Vital", field="PPG_Record", has_source_time=True, normalized_windows=False
        ),
    )
    mask = ~good
    mask[405:407] = True
    truth = root / "ref.npz"
    np.savez(truth, damaged=mask, reference=ref)
    return c, public, truth


def test_real_tool_feedback_handles_scoped_replay_and_forced_failure(tmp_path):
    c, public, _ = pp_fixture(tmp_path)
    arm = "generic_tool_agent"
    one = run_episode(public, tmp_path / "one", arm, ScriptedTransport(), episode_id="same")
    assert one["status"] == "completed"
    trace = read(tmp_path / "one/trace.json")
    assert [s["name"] for s in trace["steps"]] == ["run_candidate", "inspect", "submit"]
    assert trace["trials"][0]["handle"] == trace["submission"]["selected_candidate_handle"] == trace["selected_handle"]
    frame = tmp_path / "one/transport.json"
    two = run_episode(public, tmp_path / "two", arm, Replay(frame), episode_id="same")
    assert two["released_waveform"] == one["released_waveform"] and two["usage"]["model_requests"] == 0
    wrong = run_episode(public, tmp_path / "wrong", arm, Replay(frame), episode_id="foreign")
    assert wrong["status"] == "error" and wrong["fallback_used"]
    limit = run_episode(public, tmp_path / "limited", arm, ScriptedTransport(), episode_id="budget", max_turns=1)
    assert limit["status"] == "budget_exhausted" and limit["selected_candidate"] is None
    assert one["input_hash"] == c["input_hash"] and one["usage"]["tool_calls"] == 3


def test_production_review_can_veto_a_real_trial_and_scorer_keeps_both_layers(tmp_path):
    c, public, truth = pp_fixture(tmp_path)
    arm = "signalmind_full"
    runs = {}
    for repeat, decision in [(1, "accept"), (2, "reject")]:
        name = f"run{repeat}"
        runs[str(repeat)] = name
        result = run_episode(public, tmp_path / name, arm, ScriptedTransport(decision), episode_id=name)
        assert result["status"] == "completed", result["reason_codes"]
        assert result["selected_candidate"] is not None
        trace = read(tmp_path / name / "trace.json")
        assert {"diagnosis", "groups", "observation", "gate"} <= {r["stage"] for r in trace["stages"]}
        assert trace["eligible_source"] and len(trace["committed_patches"]) == int(decision == "accept")
    row = dict(
        case_id=c["case_id"],
        input_hash=c["input_hash"],
        public_dir="public",
        private_npz="ref.npz",
        private_sha256=sha(truth),
        subject="fixture",
        record="record",
        noise_source="fixture",
        split="dev",
        stratum="spike",
        outputs={arm: runs},
    )
    manifest = dict(
        cases=[row],
        arms=[arm],
        arm_modes={arm: "fixture"},
        reference_scale_floors={"PPG": {"source_units": 0.01}},
        repetitions=2,
    )
    path = tmp_path / "manifest.json"
    write(path, manifest)
    res = report(path)
    yes, no = res["rows"]
    assert yes["recovered_sites"] == 1
    assert (
        no["recovered_sites"] == 0 and no["selected_candidate"]["recovered_sites"] == 1 and not no["candidate_fallback"]
    )
    assert no["selected_candidate"]["window_nmae"] < no["window_nmae"]
    assert res["summaries"][0]["patient_macro_nmae"] == pytest.approx((yes["window_nmae"] + no["window_nmae"]) / 2)
    # Missing outputs keep identity numerical error but have no released measurement support.
    row["outputs"][arm].pop("2")
    write(path, manifest)
    missing = report(path)["rows"][1]
    assert missing["status"] == "missing" and missing["window_nmae"] == no["window_nmae"]
    assert (
        missing["candidate_fallback"]
        and missing["approved_native_support_samples"] == 0
        and missing["undelivered_samples"] == c["samples"]
    )


def test_macro_weighting_does_not_give_patients_with_extra_windows_extra_weight():
    def item(subject, stratum, repeat, error):
        return dict(
            subject=subject,
            stratum=stratum,
            repetition=repeat,
            window_nmae=error,
            selected_candidate={"window_nmae": error},
        )

    rows = [item("A", "one", 1, 0), item("A", "one", 1, 0), item("A", "two", 1, 2), item("B", "two", 1, 3)]
    assert macro_errors(rows)["patient_macro_nmae"] == 2


def test_shared_experiment_reserve_survives_provider_recreation(tmp_path, monkeypatch):
    from eval.repair_benchmark.providers import DeepSeek, EpisodeLimit
    from tsmcp.cleaning import agent

    class LocalLedger:
        RESERVE = 0.35

        def __init__(self, folder, cap, budget_root, compact_tools):
            self.path = budget_root / "api_budget.json"
            self.ledger = read(self.path)
            self.model = "deepseek-flash"

        def close(self):
            pass

    monkeypatch.setattr(agent, "BudgetClient", LocalLedger)
    shared = tmp_path / "shared"
    marker = tmp_path / "experiment.json"
    ledger = dict(cap_usd=5, calls=[dict(reserve_usd=0.35, estimated_peak_usd=0.22)])
    write(shared / "api_budget.json", ledger)
    p = DeepSeek(shared, 0.50, experiment_budget=marker)
    with p.session(tmp_path, 1e12, 8):
        p.client.before_attempt()
        p.client.ledger["calls"].append(dict(status="transport_failed", reserve_usd=0.35))
        write(p.client.path, p.client.ledger)
    other = DeepSeek(shared, 0.50, experiment_budget=marker)
    with other.session(tmp_path, 1e12, 8), pytest.raises(EpisodeLimit):
        other.client.before_attempt()
    assert read(marker)["first_call"] == 1 and p.accounting()[0]["usd_cost"] is None


def test_one_shot_accepts_exact_json_and_frozen_parameter_restatements(tmp_path):
    case, public, _ = pp_fixture(tmp_path)
    plan = baseline(case, "interpolate_pchip")
    for action in plan["actions"]:
        action["parameters"] = {"anchors_per_side": 3, "max_samples": 4}

    class TextTransport(ScriptedTransport):
        def response(self, *args):
            return [dict(type="text", text=json.dumps(plan))]

    result = run_episode(public, tmp_path / "valid", "one_shot_planner", TextTransport(), episode_id="json")
    assert result["status"] == "completed" and result["decision"] == "waveform_repair"
    plan["actions"][0]["parameters"]["anchors_per_side"] = 8
    wrong = run_episode(public, tmp_path / "invalid", "one_shot_planner", TextTransport(), episode_id="different")
    assert wrong["status"] == "invalid_output" and wrong["selected_candidate"] is None
