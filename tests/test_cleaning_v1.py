"""Risk-focused checks for the waveform cleaning vertical slice."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tsmcp.cleaning.common import TASKS, changed, digest, no_reference, sha, write
from tsmcp.cleaning.controller import CleaningTools
from tsmcp.cleaning.data import VitalReader, continuous, mark_time_conflicts, matlab_indices
from tsmcp.cleaning.engine import execute_version, register_recipe
from tsmcp.cleaning.recipes import execute
from tsmcp.cleaning.tasks import fixed_denominator_metrics, match_events, probe, score_events


def case_workspace(tmp_path, task="ECG"):
    fs = 125
    t = np.arange(2500) / fs
    x = np.sin(2 * np.pi * 1.2 * t)
    case = {
        "id": "test_run",
        "record_id": "test_record",
        "task": task,
        "fs": fs,
        "unit": TASKS[task]["unit"],
        "n": len(x),
        "duration_s": len(t) / fs,
    }
    folder = tmp_path / "public/test_run"
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "input.npz", values=x, time=t)
    case["input_sha256"] = sha(folder / "input.npz")
    write(
        tmp_path / "dataset_manifest.json",
        {"patients": 1, "cases": [case], "audit_failures": [], "unavailable_tasks": [], "scope_hash": digest(case)},
    )
    return case, x, t


def test_source_index_and_time_boundary():
    assert matlab_indices([1, 5], 5).tolist() == [0, 4]
    with pytest.raises(ValueError):
        matlab_indices([0, 4], 5)
    a = {
        "has_time": True,
        "subject": "p",
        "case": "c",
        "fs": 125,
        "segment": 1,
        "start": 0,
        "end": 9.992,
        "time_tolerance": 1e-6,
    }
    b = {**a, "start": 10, "end": 19.992, "segment": 2}
    assert continuous(a, b)
    assert not continuous(a, {**b, "start": 20})
    assert not continuous(a, b, normalized=True)
    assert not continuous(a, {**b, "has_time": False})


def test_shared_endpoint_is_quarantined_on_both_source_windows():
    left = {
        "row": 0,
        "n": 1250,
        "has_time": True,
        "subject": "p",
        "case": "c",
        "fs": 125,
        "start": 16380.016,
        "end": 16390.008,
        "time_tolerance": 1e-6,
    }
    right = {**left, "row": 1, "start": 16390.008, "end": 16400.0}
    conflicts = mark_time_conflicts([left, right])
    assert len(conflicts) == 1
    assert left["time_conflicts"] == [[1249, 1250]]
    assert right["time_conflicts"] == [[0, 1]]


def test_identity_preserves_values_but_not_ambiguous_time_support(tmp_path):
    case, x, _ = case_workspace(tmp_path)
    case["source_time_conflicts"] = [[0, 1], [2499, 2500]]
    result = execute_version(tmp_path, register_recipe(tmp_path, kind="identity"), [case], workers=1)
    a = result["results"][case["id"]]
    assert a["stats"]["eligible_samples"] == len(x) - 2
    with np.load(tmp_path / a["arrays"], allow_pickle=False) as z:
        np.testing.assert_array_equal(z["values"], x)
        assert not z["valid"][0] and not z["valid"][-1]
        assert not z["modified"].any()


def test_missing_channel_and_no_time_window(tmp_path):
    path = tmp_path / "test.mat"
    with h5py.File(path, "w") as f:
        group = f.create_group("Subj_Wins")
        group.create_dataset("ECG_Raw", data=np.arange(1250, dtype=float))
    reader = VitalReader(path)
    try:
        assert reader.channels() == {"ECG": "ECG_Raw", "PPG": None, "ABP": None}
        metadata = reader.metadata(0, "ECG_Raw", "p")
        assert not metadata["has_time"]
    finally:
        reader.close()


def test_probe_never_reconnects_across_estimates():
    fs = 125
    t = np.arange(20 * fs) / fs
    x = np.zeros_like(t)
    for center in np.arange(1, 19):
        x += np.exp(-(((t - center) / 0.025) ** 2))
    estimated = (t >= 7) & (t < 10)
    result = probe(x, t, fs, np.ones(len(t), bool), estimated, "ECG")
    assert result["events"] and result["intervals"]
    assert all(not 7 <= e["time_s"] < 10 for e in result["events"])
    assert all(r["interval_s"] < 2 for r in result["intervals"])
    assert len({e["support_id"] for e in result["events"]}) == 2


def test_native_pressure_cycle_and_integral(monkeypatch):
    monkeypatch.setattr(
        "tsmcp.cleaning.tasks.detect_pulses_unguided",
        lambda x, fs: (np.array([0, 4]), np.array([2, 4]), np.array([2.0, 2.0])),
    )
    x = np.array([80, 100, 120, 100, 80], float)
    result = probe(x, np.arange(5) / 2, 2, np.ones(5, bool), np.zeros(5, bool), "ABP")
    cycle = result["pressures"][0]
    assert (cycle["sbp"], cycle["dbp"], cycle["mean_pressure"]) == (120, 80, 100)
    assert cycle["mean_pressure"] != cycle["dbp"] + (cycle["sbp"] - cycle["dbp"]) / 3


def test_one_to_one_and_missing_beat_interval():
    assert len(match_events([125, 125, 250], [125, 250], 5)) == 2
    score = score_events([0, 250], [0, 125, 250], fs=125)
    assert score["false_negative"] == 1 and score["correct_intervals"] == 0
    shifted = score_events([2, 127, 252], [0, 125, 250], fs=125)
    assert shifted["correct_intervals"] == 2 and not shifted["content_correct"]


def test_fixed_denominator_and_missing_reference():
    rows = [{"duration_s": 10, "released": False, "content_correct": True, "baseline_correct": True}]
    result = fixed_denominator_metrics(rows)
    assert result == {"Y": 0, "C": 0, "Rbad": None, "goodloss": 1}
    rows[0]["content_correct"] = None
    assert fixed_denominator_metrics(rows)["Y"] is None
    assert no_reference()["Y"] is None


def test_recipe_uses_operators_and_preserves_parent(tmp_path):
    case, x, _ = case_workspace(tmp_path)
    initial = x.copy()
    recipe = {"kind": "python", "category": "standard_operators"}

    def run(ctx):
        ctx.apply("mark_invalid", [[10, 20]])
        return ctx.current

    y, valid, estimated, operations = execute(
        x, np.ones(len(x), bool), np.zeros(len(x), bool), case, recipe, SimpleNamespace(run=run)
    )
    np.testing.assert_array_equal(x, initial)
    np.testing.assert_array_equal(y, initial)
    assert not valid[10:20].any() and not estimated.any() and operations
    with pytest.raises(ValueError, match="replacement"):
        execute(
            x,
            np.ones(len(x), bool),
            np.zeros(len(x), bool),
            case,
            recipe,
            SimpleNamespace(run=lambda ctx: {"values": np.zeros(len(x))}),
        )


def test_identity_and_candidate_bound_review(tmp_path):
    case, x, _ = case_workspace(tmp_path)
    recipe = register_recipe(tmp_path, kind="identity")
    baseline = execute_version(tmp_path, recipe, [case], workers=1)
    write(tmp_path / "baselines.json", {"identity": baseline["version_id"]})
    assert baseline["results"][case["id"]]["arrays"] is None
    tools = CleaningTools(tmp_path)
    with pytest.raises(ValueError, match="pilot"):
        tools.run_candidate("ECG", "batch", "No pilot review yet", baseline="identity")
    pilot = tools.run_candidate("ECG", "pilot", "Already usable; inspect identity", baseline="identity")
    vid = pilot["version_id"]
    tools.get_report(vid)
    blocks = tools.inspect_cases(vid)
    ids = json.loads(blocks[0]["text"])["evidence_ids"]
    tools.submit_self_review(vid, "accept", ids, "Observed unchanged source and valid probe output", [])
    batch = tools.run_candidate("ECG", "batch", "Pilot unchanged; execute complete task", recipe_id=pilot["recipe_id"])
    batch_id = batch["version_id"]
    tools.get_report(batch_id)
    with pytest.raises(ValueError, match="Stale"):
        tools.submit_self_review(batch_id, "accept", ids, "Wrong version evidence", [])
    current = json.loads(tools.inspect_cases(batch_id)[0]["text"])["evidence_ids"]
    tools.submit_self_review(batch_id, "rollback", current, "Reject this candidate", [])
    assert not tools.state["accepted"]
    tools.submit_self_review(batch_id, "accept", current, "Checked identical candidate and whole-scope report", [])
    tools.submit_self_review(batch_id, "rollback", current, "Reconsider the accepted version", [])
    assert not tools.state["accepted"]
    tools.submit_self_review(batch_id, "accept", current, "Rechecked identity for this scope", [])
    assert tools.finalize_run("All task inputs checked")["status"] == "FINALIZED_COMPLETE"
    assert not changed(x, x.copy()).any()


def test_cached_artifact_tamper_rejected(tmp_path):
    case, _, _ = case_workspace(tmp_path)
    recipe = register_recipe(tmp_path, kind="identity")
    version = execute_version(tmp_path, recipe, [case], workers=1)
    a = version["results"][case["id"]]
    (tmp_path / a["probe_path"]).write_text("{}")
    with pytest.raises(ValueError, match="probe"):
        execute_version(tmp_path, recipe, [case], workers=1)


def test_sandbox_has_no_reference_or_network_access(tmp_path):
    case, _, _ = case_workspace(tmp_path)
    source = """def run(ctx):
    from pathlib import Path
    import socket
    assert not Path('/home/yanghaoxiang24/.config/tsmcp-eval/deepseek.key').exists(), 'credential exposed'
    assert not Path('/case/development_labels.json').exists()
    try:
        socket.create_connection(('1.1.1.1', 443), timeout=0.2)
    except OSError:
        pass
    else:
        raise AssertionError('network was accessible')
    ctx.apply('mark_invalid', [[10, 20]])
    return ctx.current
"""
    recipe = register_recipe(tmp_path, source=source)
    result = execute_version(tmp_path, recipe, [case], stage="pilot")
    assert result["report"]["program_checks_passed"], result["report"]["errors"]
    assert result["results"][case["id"]]["stats"]["unusable_samples"] == 10


def test_standard_recipe_cannot_hide_unregistered_numerical_edits(tmp_path):
    case, _, _ = case_workspace(tmp_path)
    source = """def run(ctx):
    ctx.apply('lowpass', [[0, len(ctx.values)]], frequency_hz=12)
    ctx.current.values.setflags(write=True)
    ctx.current.values[300] += 1
    return ctx.current
"""
    result = execute_version(tmp_path, register_recipe(tmp_path, source=source), [case], stage="pilot")
    assert not result["report"]["program_checks_passed"]
    assert "reproduce" in result["report"]["errors"][0]["reason"]


def test_support_only_changes_must_be_visible_before_acceptance(tmp_path):
    case, _, _ = case_workspace(tmp_path, task="PPG")
    identity = execute_version(tmp_path, register_recipe(tmp_path, kind="identity"), [case], workers=1)
    write(tmp_path / "baselines.json", {"identity": identity["version_id"]})
    tools = CleaningTools(tmp_path)
    result = tools.run_candidate(
        "PPG",
        "pilot",
        "Test a support-only edit",
        source="def run(ctx):\n    ctx.apply('mark_invalid', [[2200, 2201]])\n    return ctx.current\n",
    )
    vid = result["version_id"]
    report = tools.get_report(vid)
    modified = next(v for v in report["review_cases"] if "modified_cases" in v["reasons"])
    assert modified["focus_sample"] == 2200
    wrong_view = json.loads(tools.inspect_cases(vid, case["id"], 0, 4)[0]["text"])["evidence_ids"]
    with pytest.raises(ValueError, match="support-only"):
        tools.submit_self_review(vid, "accept", wrong_view, "Only reviewed an untouched region", [])
    right_view = json.loads(tools.inspect_cases(vid)[0]["text"])["evidence_ids"]
    assert any(
        tools.state["evidence"][e]["range_samples"][0] <= 2200 < tools.state["evidence"][e]["range_samples"][1]
        for e in right_view
    )
    tools.submit_self_review(
        vid, "accept", right_view, "Inspected the actual invalidated point and preserved waveform", []
    )


def test_budget_persists_across_resume(tmp_path):
    from tsmcp.cleaning.agent import BudgetClient
    from tsmcp.cleaning.common import read

    client = BudgetClient.__new__(BudgetClient)
    client.workspace = tmp_path
    client.cap, client.model, client.path = 0.36, "deepseek-flash", tmp_path / "budget.json"
    client.ledger = {"estimated_peak_usd": 0, "unknown_charge_reserve_usd": 0, "calls": []}
    response = SimpleNamespace(id="fake-request", usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 100000}))
    client.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
    client.request([{"role": "user", "content": "fixture"}])
    assert client.ledger["estimated_peak_usd"] == pytest.approx(0.03)
    client.ledger = read(client.path)
    with pytest.raises(RuntimeError, match="budget"):
        client.request([{"role": "user", "content": "resumed"}])


def test_successful_candidates_need_review_even_if_not_selected(tmp_path):
    case, _, _ = case_workspace(tmp_path)
    identity = execute_version(tmp_path, register_recipe(tmp_path, kind="identity"), [case], workers=1)
    write(tmp_path / "baselines.json", {"identity": identity["version_id"]})
    tools = CleaningTools(tmp_path)
    pilot = tools.run_candidate("ECG", "pilot", "First candidate", baseline="identity")
    with pytest.raises(ValueError, match="pending"):
        tools.run_candidate("ECG", "pilot", "Try another before reviewing", source="def run(ctx): return ctx.current")
    with pytest.raises(ValueError, match="SelfReview"):
        tools.finalize_run("Skipped candidate review")
    tools.get_report(pilot["version_id"])
    ids = json.loads(tools.inspect_cases(pilot["version_id"])[0]["text"])["evidence_ids"]
    tools.submit_self_review(pilot["version_id"], "stop", ids, "Inspected but do not promote this candidate", [])
    assert not tools.pending_reviews()
    result = tools.finalize_run("Stop with explicit unprocessed scope")
    assert result["status"] == "FINALIZED_PARTIAL"
    assert tools.state["finalized"] and not tools.state["complete"]


def test_pressure_reference_and_core_interval_identity():
    from tsmcp.cleaning.evaluation import event_cores, score_pressure

    assert score_pressure([], [])["content_correct"] is None
    reference = [{"start_s": 0, "stop_s": 1, "sbp": 120, "dbp": 80, "mean_pressure": 100}]
    score = score_pressure(reference, reference, {"sbp": 1, "dbp": 1, "mean_pressure": 1}, 0.04, False)
    assert score["content_correct"] and not score["correct_delivery"]
    reference = {
        "peak_samples": [125, 1250, 1375],
        "tolerances": {"association_ms": 40, "peak_ms": 8, "interval_ms": 16},
    }
    events = {"events": [{"sample": p, "support_id": 0} for p in reference["peak_samples"]]}
    support = np.ones(2500, bool)
    rows = event_cores(events, reference, 125, 2500, support, support)
    assert rows[1]["content_correct"] is True
    events["events"].pop(1)
    rows = event_cores(events, reference, 125, 2500, support, support)
    assert rows[1]["content_correct"] is False


def test_registered_rule_rescans_whole_batch_after_acceptance(tmp_path):
    case, x, t = case_workspace(tmp_path, task="PPG")
    x[1000] += 9
    path = tmp_path / "public" / case["id"] / "input.npz"
    np.savez_compressed(path, values=x, time=t)
    case["input_sha256"] = sha(path)
    from tsmcp.cleaning.common import read

    manifest = read(tmp_path / "dataset_manifest.json")
    manifest["cases"] = [case]
    write(tmp_path / "dataset_manifest.json", manifest)
    baseline = execute_version(tmp_path, register_recipe(tmp_path, kind="identity"), [case], workers=1)
    write(tmp_path / "baselines.json", {"identity": baseline["version_id"]})
    tools = CleaningTools(tmp_path)
    pilot = tools.run_candidate("PPG", "pilot", "validate first pass", baseline="identity")
    vid = pilot["version_id"]
    tools.get_report(vid)
    evidence = json.loads(tools.inspect_cases(vid, case["id"], target_samples=[999, 1002], flank_s=5)[0]["text"])[
        "evidence_ids"
    ]
    rule = dict(
        metric="step",
        scale_window_s=3,
        baseline_window_s=0.16,
        seed_ratio=1.5,
        tail_ratio=0.6,
        quiet_s=0.08,
        max_seed_s=0.1,
    )
    receipt = tools.register_pattern(
        evidence[0],
        "raw",
        [999, 1002],
        "sharp spike",
        "possible acquisition transient",
        "no independent cause reference",
        "relative step search",
        rule,
    )
    tools.submit_self_review(vid, "accept", evidence, "pilot OK", [])
    batch = tools.run_candidate("PPG", "batch", "apply pilot to full scope", recipe_id=pilot["recipe_id"])["version_id"]
    tools.get_report(batch)
    evidence = json.loads(tools.inspect_cases(batch)[0]["text"])["evidence_ids"]
    review = tools.submit_self_review(batch, "accept", evidence, "batch OK", [])
    res = review["recurring_patterns"][receipt["rule_id"]]
    assert res["scanned"] == 1 and res["counts"]["still_detected"] == 1
    assert tools.get_pattern_matches(batch, receipt["rule_id"])["total"] == 1
    tools.finalize_run("batch and follow-up complete")
    assert read(tmp_path / "export/pattern_followups.json")["accepted"]["PPG"][receipt["rule_id"]] == res


def test_operation_review_gate_reverts_a_whole_block_when_one_target_is_held(tmp_path, monkeypatch):
    from eval import review_candidate_writes as gate
    from tsmcp.cleaning.common import read

    c, _, _ = case_workspace(tmp_path)
    identity = execute_version(tmp_path, register_recipe(tmp_path, kind="identity"), [c])
    candidate = execute_version(
        tmp_path,
        register_recipe(
            tmp_path, source="def run(ctx):\n    return ctx.apply('mark_invalid', [[220, 230], [300, 310]])\n"
        ),
        [c],
        stage="batch",
    )
    assert candidate["report"]["program_checks_passed"]
    write(tmp_path / "baselines.json", {"identity": identity["version_id"]})
    write(tmp_path / "agent_state.json", {"complete": True, "accepted_tasks": {"ECG": candidate["version_id"]}})
    budget = tmp_path / "budget"
    write(budget / "api_budget.json", {})
    monkeypatch.setattr(gate, "BudgetClient", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))

    def verdicts(client, folder, group, cards):
        items = [z for c in cards for z in c["metrics"]["items"]]
        assert [(z["a"], z["b"]) for z in items] == [(220, 230), (300, 310)]
        return {
            "request_id": "test",
            "items": [
                dict(id=z["id"], verdict=v, observation="test", limitation="test")
                for z, v in zip(items, ["accept", "hold"], strict=True)
            ],
        }

    monkeypatch.setattr(gate, "check_group", verdicts)
    s = gate.run(tmp_path, tmp_path / "run", budget)
    assert s["cards"] == 1 and s["explicitly_reviewed_operations"] == 2 and s["published_operations"] == 0
    assert s["tasks"]["ECG"]["newly_excluded"] == 0
    receipt = read(tmp_path / "run/delivery/waveform_manifest.json")[c["id"]]
    assert receipt["artifact_id"] == identity["results"][c["id"]]["artifact_id"]
    assert receipt["disposition"] == "raw_preserved_pending_review"
