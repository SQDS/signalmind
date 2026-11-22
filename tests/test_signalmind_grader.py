"""The independent grader replays ledgers itself; runtime 'accepted' is never used as correctness evidence."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from mcp import Client

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from eval import grade_signalmind as G  # noqa: E402
from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.server.app import build_server  # noqa: E402

SBP, DBP = "abp.sbp.beat_mean.v1", "abp.dbp.beat_mean.v1"


def make_suite(tmp_path, truth):
    cases = tmp_path / "cases"
    (cases / "p10test" / "case1").mkdir(parents=True)
    (cases / "p10test.answers.jsonl").write_text(json.dumps({
        "case_id": "case1", "expected": {"sbp_mean": truth["sbp"], "dbp_mean": truth["dbp"]},
        "tolerance": {"sbp_mean": 3.0, "dbp_mean": 3.0}, "control": None, "kind": "interval_mean", "category": "ordinary"}) + "\n")  # fmt: skip
    (tmp_path / "eval" / "tasks").mkdir(parents=True)
    (tmp_path / "eval" / "tasks" / "p10test_manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "case_id": "case1",
                        "kind": "interval_mean",
                        "category": "ordinary",
                        "intervals": [[10, 40]],
                        "expected": {},
                    }
                ]
            }
        )
    )
    return cases


async def drive(srv, path, bundles):
    async with Client(srv) as c:
        r = await c.call_tool(
            "ts_load", {"source": str(path), "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]}
        )
        sid = json.loads(r.content[0].text)["id"]
        m = json.loads((await c.call_tool("ts_measure", {"id": sid, "what": ["abp_sbp", "abp_dbp"], "window": [10, 40],
                                                         "channels": ["ABP"]})).content[0].text)["measurements"]  # fmt: skip
        out = []
        for b in bundles(m):
            out.append(json.loads((await c.call_tool("ts_submit", {"bundle": b})).content[0].text))
        return m, out


def test_grader_replays_ledger_independently(tmp_path, monkeypatch):
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, seed=21)
    truth = {
        k: float(np.mean([b[k] for b in beats if 10 <= b["t_foot_s"] and b["t_end_s"] <= 40])) for k in ("sbp", "dbp")
    }
    sig = tmp_path / "work" / "sig.npy"
    sig.parent.mkdir()
    np.save(sig, np.stack([ecg, abp], 1).astype(np.float32))
    cases = make_suite(tmp_path, truth)
    runs = tmp_path / "runs"
    monkeypatch.setattr(G, "CASES", cases)
    monkeypatch.setattr(G, "RUNS", runs)
    monkeypatch.setattr(G, "ROOT", tmp_path)

    def good(m):
        return [{"claims": [
            {"claim_id": "s", "quantity": "sbp_mean", "status": "validated_measurement", "value": m[0]["value"], "unit": "mmHg",
             "interval": [10, 40], "evidence_refs": [m[0]["evidence_id"]], "definition_id": SBP},
            {"claim_id": "d", "quantity": "dbp_mean", "status": "validated_measurement", "value": m[1]["value"], "unit": "mmHg",
             "interval": [10, 40], "evidence_refs": [m[1]["evidence_id"]], "definition_id": DBP}]}]  # fmt: skip

    def fabricated(m):  # right numbers, wrong provenance: log mode records it; the grader must call it unsupported
        return [{"claims": [
            {"claim_id": "s", "quantity": "sbp_mean", "status": "validated_measurement", "value": m[0]["value"], "unit": "mmHg",
             "interval": [10, 40], "evidence_refs": ["ev99"], "definition_id": SBP},
            {"claim_id": "d", "quantity": "dbp_mean", "status": "validated_measurement", "value": round(truth["dbp"]), "unit": "mmHg",
             "interval": [10, 40], "evidence_refs": [m[0]["evidence_id"]], "definition_id": DBP}]}]  # fmt: skip

    for arm, mode, bundles in (("C_full", "verify", good), ("C_tools", "log", fabricated)):
        rd = runs / f"p10test-C-claude-opus-5-5-medium-t-{arm}-w0"
        rt = rd / "runtime" / "case1"
        srv = build_server(
            SignalStore(allow_roots=[sig.parent]), plugins=["pressure"], runtime=mode, runtime_dir=str(rt), run_id=arm
        )
        m, res = asyncio.run(drive(srv, sig, bundles))
        (rd / "results.jsonl").write_text(json.dumps({"case_id": "case1", "cost_usd": 0.1, "cc_cost_usd": 0.11, "latency_s": 30,
                                                      "turns": 3, "tool_calls": 2, "usage": {"output_tokens": 10}}) + "\n")  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["grade", "--suite", "p10test", "--tag", "t", "--out", str(tmp_path / "out.json")])
    G.main()
    result = json.loads((tmp_path / "out.json").read_text())
    full = next(s for s in result["sessions"] if s["arm"] == "C_full")
    tools = next(s for s in result["sessions"] if s["arm"] == "C_tools")
    assert full["grounded_correct"] and full["correct"] and full["finalization"] == "accepted"
    assert tools["correct"] and not tools["grounded_correct"] and tools["finalization"] == "recorded"
    assert tools["final"]["fields"]["sbp_mean"]["why"] == "missing_or_unknown_evidence"
    assert tools["final"]["fields"]["dbp_mean"]["why"] == "no_computed_record_of_the_right_definition"
    assert tools["final"]["unsupported_precise"] is True
    s = result["summary"]["by_arm"]
    assert s["C_full"]["grounded_correct"] == 1 and s["C_tools"]["grounded_correct"] == 0
    assert s["C_tools"]["unsupported_precise_claims"] == 1 and s["C_full"]["verifier_false_rejections"] == 0
    assert result["summary"]["paired"]["grounded_correct_full_only"] == 1


def test_grader_marks_missing_submission_incomplete(tmp_path, monkeypatch):
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, seed=22)
    truth = {
        k: float(np.mean([b[k] for b in beats if 10 <= b["t_foot_s"] and b["t_end_s"] <= 40])) for k in ("sbp", "dbp")
    }
    sig = tmp_path / "work" / "sig.npy"
    sig.parent.mkdir()
    np.save(sig, np.stack([ecg, abp], 1).astype(np.float32))
    cases = make_suite(tmp_path, truth)
    runs = tmp_path / "runs"
    monkeypatch.setattr(G, "CASES", cases)
    monkeypatch.setattr(G, "RUNS", runs)
    monkeypatch.setattr(G, "ROOT", tmp_path)
    for arm, mode in (("C_full", "verify"), ("C_tools", "log")):
        rd = runs / f"p10test-C-claude-opus-5-5-medium-t-{arm}-w0"
        rt = rd / "runtime" / "case1"
        srv = build_server(
            SignalStore(allow_roots=[sig.parent]), plugins=["pressure"], runtime=mode, runtime_dir=str(rt), run_id=arm
        )
        asyncio.run(drive(srv, sig, lambda m: []))  # measured, never submitted
        (rd / "results.jsonl").write_text(json.dumps({"case_id": "case1", "cost_usd": 0.1, "cc_cost_usd": 0.1, "latency_s": 9,
                                                      "turns": 2, "tool_calls": 2, "usage": {}, "text": 'ANSWER: {"sbp_mean": 120}'}) + "\n")  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["grade", "--suite", "p10test", "--tag", "t", "--out", str(tmp_path / "out.json")])
    G.main()
    result = json.loads((tmp_path / "out.json").read_text())
    assert all(s["incomplete_execution"] and not s["correct"] for s in result["sessions"])
    assert result["summary"]["by_arm"]["C_full"]["incomplete_execution"] == 1  # the ANSWER line never counts


def test_gap_tolerance_and_ppg_channel_rules():
    """p11 grading paths: a gap item accepts a claimed window within tolerance of the true dropout end; a PPG rate must
    come from PPG evidence (an ECG-derived rate does not support it)."""
    ledger = {
        "ev1": {"evidence_id": "ev1", "seq": 1, "source_kind": "computed", "definition_id": "abp.sbp.beat_mean.v1", "value": 120.0,
                "unit": "mmHg", "interval_requested": [131.2, 191.2], "channels_effective": ["ABP"], "resolution": {"known": True, "value": 0.4}},
        "ev2": {"evidence_id": "ev2", "seq": 2, "source_kind": "computed", "definition_id": "generic.period.v1", "value": 0.662,
                "unit": "s", "interval_requested": [10.0, 70.0], "channels_effective": ["PPG"], "resolution": {"known": False}},
        "ev3": {"evidence_id": "ev3", "seq": 3, "source_kind": "derived", "definition_id": "rate.from_period.v1", "value": 90.6,
                "unit": "bpm", "parents": ["ev2"], "channels_effective": ["PPG"], "resolution": {"known": False}},
        "ev4": {"evidence_id": "ev4", "seq": 4, "source_kind": "computed", "definition_id": "rate.from_period.v1", "value": 90.6,
                "unit": "bpm", "interval_requested": [10.0, 70.0], "channels_effective": ["ECG"], "resolution": {"known": False}},
    }  # fmt: skip
    gap = {
        "expected": {"sbp_mean": 120.3},
        "tolerance": {"sbp_mean": 3.0},
        "intervals": [[130.0, 190.0]],
        "interval_tolerance_s": 2.0,
    }
    c = {"quantity": "sbp_mean", "status": "validated_measurement", "value": 120.0, "unit": "mmHg", "evidence": ["ev1"]}
    assert G.grade_claims([c], [], gap, ledger)["grounded_correct"]
    far = dict(gap, intervals=[[125.0, 185.0]])
    assert G.grade_claims([c], [], far, ledger)["fields"]["sbp_mean"]["why"] == "interval_differs_from_question"
    ppg = {"expected": {"pulse_rate_bpm": 90.4}, "tolerance": {"pulse_rate_bpm": 3.0}, "intervals": [[10.0, 70.0]]}
    ok = {
        "quantity": "pulse_rate_bpm",
        "status": "validated_measurement",
        "value": 90.6,
        "unit": "bpm",
        "evidence": ["ev3"],
    }
    assert G.grade_claims([ok], [], ppg, ledger)["grounded_correct"]
    ecg_only = dict(ok, evidence=["ev4"])
    assert G.grade_claims([ecg_only], [], ppg, ledger)["fields"]["pulse_rate_bpm"]["why"] == "evidence_not_from_PPG"
