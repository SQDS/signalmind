"""Mock-agent orchestration: scripted drivers against the real pressure plugin on synthetic data.

These tests validate the controller's accept / feedback / repair / stop behaviour. They do not show that a real model
repairs its reasoning.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.runtime.controller import Controller, Stop, Submit, ToolCall  # noqa: E402
from tsmcp.runtime.executor import CoreExecutor  # noqa: E402
from tsmcp.runtime.ledger import EvidenceLedger  # noqa: E402
from tsmcp.runtime.taskspec import Budgets, QuantitySpec, abp_interval_spec  # noqa: E402

SBP = "abp.sbp.beat_mean.v1"


class Scripted:
    """Steps are Actions or callables(obs, ctx) -> Action; the last step repeats. `ctx` remembers the signal id, the
    last measurement row, the last evidence ids and the last deficiencies, like a model would."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.ctx: dict = {}

    def next(self, obs):
        res = obs.get("result") or {}
        if "id" in res:
            self.ctx["sid"] = res["id"]
        if res.get("measurements"):
            self.ctx["m"] = res["measurements"][0]
        if res.get("evidence_id"):
            self.ctx["ev"] = res["evidence_id"]
        if res.get("evidence_ids"):
            self.ctx["evs"] = res["evidence_ids"]
        if res.get("evidence"):
            self.ctx["derived"] = res["evidence"]
        if obs.get("submission"):
            self.ctx["deficiencies"] = obs["submission"]["deficiencies"]
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        return step(obs, self.ctx) if callable(step) else step


@pytest.fixture
def world(tmp_path):
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, seed=1)
    p = tmp_path / "sig.npy"
    np.save(p, np.stack([ecg, abp], 1).astype(np.float32))
    store = SignalStore(allow_roots=[tmp_path])
    ledger = EvidenceLedger("test")
    ex = CoreExecutor(store, ledger, root=tmp_path / "rt")
    truth = float(np.mean([b["sbp"] for b in beats if 10 <= b["t_foot_s"] and b["t_end_s"] <= 40]))
    return {"path": str(p), "store": store, "ledger": ledger, "executor": ex, "truth": truth, "dir": tmp_path}


def load(w):
    return ToolCall("ts_load", {"source": w["path"], "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]})


def measure(window, what="abp_sbp"):
    return lambda obs, ctx: ToolCall(
        "ts_measure", {"id": ctx["sid"], "what": [what], "window": window, "channels": ["ABP"]}
    )


def submit_from_measure(interval=(10, 40), **over):
    def act(obs, ctx):
        m = ctx["m"]
        c = {"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": m["value"],
             "unit": m["unit"], "interval": list(interval), "evidence_refs": [m["evidence_id"]], "definition_id": SBP}  # fmt: skip
        c.update(over)
        return Submit({"claims": [c]})

    return act


def spec(w, **kw):
    return abp_interval_spec("t", "mean SBP over 10-40 s", "ABP", (10, 40), quantities=("sbp_mean",), ecg_channel="ECG",
                             capabilities=CoreExecutor.TOOLS, **kw)  # fmt: skip


def submits(out):
    return [e for e in out.events if "submit" in e]


def test_valid_measurement_is_accepted(world):
    drv = Scripted([load(world), measure([10, 40]), submit_from_measure()])
    out = Controller(spec(world), drv, world["executor"], world["ledger"]).run()
    assert out.status == "success" and out.steps == 3 and out.tool_calls == 2 and out.submissions == 1
    acc = out.answer["accepted_claims"]
    assert len(acc) == 1 and abs(acc[0]["value"] - world["truth"]) < 1.0 and acc[0]["evidence"] == ["ev1"]
    assert "ev1" in out.answer["text"]


def test_wrong_window_rejected_then_repaired(world):
    drv = Scripted([load(world), measure([10, 20]), submit_from_measure(), measure([10, 40]), submit_from_measure()])
    out = Controller(spec(world), drv, world["executor"], world["ledger"]).run()
    assert out.status == "success" and out.submissions == 2
    first = submits(out)[0]
    assert first["submit"] == "retry" and first["deficiencies"][0]["code"] == "scope_mismatch"
    assert out.answer["accepted_claims"][0]["evidence"] == ["ev2"]
    assert world["ledger"].get("ev1").interval_requested == (10.0, 20.0)  # the rejected measurement is kept


def test_fabricated_reference_then_no_progress(world):
    bad = Submit({"claims": [{"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": 118.0,
                              "unit": "mmHg", "interval": [10, 40], "evidence_refs": ["ev42"], "definition_id": SBP}]})  # fmt: skip
    out = Controller(spec(world), Scripted([load(world), bad]), world["executor"], world["ledger"]).run()
    assert out.status == "no_progress" and out.submissions == 2
    assert (
        out.answer["accepted_claims"] == []
        and out.answer["unresolved"][0]["deficiencies"][0]["code"] == "invalid_reference"
    )


def test_repeated_failed_repairs_terminate(world):
    steps = [load(world), measure([10, 20])]
    for k in range(5):  # different (wrong) bundles each time: same issue, never fixed
        steps.append(submit_from_measure(value=118.0 + k))
    out = Controller(spec(world), Scripted(steps), world["executor"], world["ledger"]).run()
    assert out.status == "repeated_unresolved_issue" and out.submissions == 3  # 2 repairs allowed per issue
    assert out.answer["outcome"] == "unresolved"


def test_partial_answer_survives_unresolved_claim(world):
    def bundle(ctx, accept):
        m = ctx["m"]
        return Submit({"claims": [
            {"claim_id": "s", "quantity": "sbp_mean", "status": "validated_measurement", "value": m["value"], "unit": "mmHg",
             "interval": [10, 40], "evidence_refs": [m["evidence_id"]], "definition_id": SBP},
            {"claim_id": "d", "quantity": "dbp_mean", "status": "validated_measurement", "value": 80.0, "unit": "mmHg",
             "interval": [10, 40], "evidence_refs": [m["evidence_id"]], "definition_id": "abp.dbp.beat_mean.v1"}],
            "accept_partial": accept})  # fmt: skip

    steps = [load(world), measure([10, 40]), lambda o, c: bundle(c, True), lambda o, c: bundle(c, True)]
    out = Controller(spec(world), Scripted(steps), world["executor"], world["ledger"]).run()
    assert out.status == "partial" and out.submissions == 2  # first accept_partial ignored (not yet reviewed)
    first = submits(out)[0]
    assert first["submit"] == "retry" and first["deficiencies"][0]["code"] == "method_mismatch"
    assert [c["claim_id"] for c in out.answer["accepted_claims"]] == ["s"]
    assert out.answer["unresolved"][0]["claim_id"] == "d"


def test_step_budget_and_invalid_submissions_consume_budget(world):
    s = spec(world, budgets=Budgets(max_steps=2))
    out = Controller(
        s, Scripted([load(world), measure([10, 40]), submit_from_measure()]), world["executor"], world["ledger"]
    ).run()
    assert out.status == "budget_exhausted" and out.steps == 2
    s = spec(world, budgets=Budgets(max_submissions=1))
    out = Controller(s, Scripted([Submit({"claims": "nonsense"})]), world["executor"], EvidenceLedger("t2")).run()
    assert out.status == "submission_budget_exhausted" and out.submissions == 1


def test_tool_budget_then_explicit_insufficient_evidence(world):
    s = spec(world, budgets=Budgets(max_tool_calls=1))
    abstain = Submit({"claims": [{"claim_id": "c1", "quantity": "sbp_mean", "status": "insufficient_evidence",
                                  "reason": "tool budget exhausted before the interval could be measured"}]})  # fmt: skip
    out = Controller(s, Scripted([load(world), measure([10, 40]), abstain]), world["executor"], world["ledger"]).run()
    assert out.status == "partial" and out.tool_calls == 1
    assert out.events[1]["tool_budget_exhausted"] is True
    assert out.answer["unresolved"][0]["status"] == "insufficient_evidence"


def test_free_text_cannot_bypass_structured_checks(world):
    def sub(obs, ctx):
        m = ctx["m"]
        return Submit({"text": "Mean SBP is 999 mmHg (I am sure).", "claims": [
            {"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": m["value"], "unit": "mmHg",
             "interval": [10, 40], "evidence_refs": [m["evidence_id"]], "definition_id": SBP}]})  # fmt: skip

    out = Controller(
        spec(world), Scripted([load(world), measure([10, 40]), sub]), world["executor"], world["ledger"]
    ).run()
    assert (
        out.status == "success" and "999" not in out.answer["text"] and "999" not in str(out.answer["accepted_claims"])
    )


def test_infrastructure_stop_is_not_a_physiological_statement(world):
    out = Controller(
        spec(world), Scripted([load(world), Stop("infra: gateway unavailable")]), world["executor"], world["ledger"]
    ).run()
    assert out.status == "incomplete_execution" and out.answer["accepted_claims"] == []


def test_visual_estimate_must_be_approximate(world):
    zoom = lambda obs, ctx: ToolCall("ts_zoom", {"id": ctx["sid"], "window": [10, 40], "channels": ["ABP"]})  # noqa: E731

    def exact(obs, ctx):
        return Submit({"claims": [{"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": 120.0,
                                   "unit": "mmHg", "interval": [10, 40], "evidence_refs": [ctx["ev"]], "definition_id": SBP}]})  # fmt: skip

    def approx(obs, ctx):
        return Submit({"claims": [{"claim_id": "c1", "quantity": "sbp_mean", "status": "approximate", "value": 120,
                                   "unit": "mmHg", "interval": [10, 40], "evidence_refs": [ctx["ev"]], "definition_id": SBP,
                                   "precision_decimals": -1}]})  # fmt: skip

    out = Controller(
        spec(world), Scripted([load(world), zoom, exact, approx]), world["executor"], world["ledger"]
    ).run()
    assert out.status == "success"
    assert submits(out)[0]["deficiencies"][0]["code"] == "inadequate_support"
    assert out.answer["accepted_claims"][0]["status"] == "approximate"


def test_derived_time_weighted_mean_is_accepted(world):
    def derive(obs, ctx):
        return ToolCall(
            "ts_derive", {"op": "aggregate", "inputs": [ctx["evs"][0]], "column": "sbp", "rule": "time_weighted"}
        )

    def sub(obs, ctx):
        e = ctx["derived"]
        return Submit({"claims": [{"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement",
                                   "value": round(e["value"], 2), "unit": e["unit"], "interval": [10, 40],
                                   "evidence_refs": [e["evidence_id"]], "definition_id": "abp.sbp.time_weighted_mean.v1",
                                   "derived": {"op": "aggregate"}}]})  # fmt: skip

    s = spec(world)
    s.quantities[0] = QuantitySpec("sbp_mean", "abp.sbp.time_weighted_mean.v1", "mmHg")
    out = Controller(
        s, Scripted([load(world), measure([10, 40], "abp_beats"), derive, sub]), world["executor"], world["ledger"]
    ).run()
    assert (
        out.status == "success" and out.answer["accepted_claims"][0]["definition_id"] == "abp.sbp.time_weighted_mean.v1"
    )
    rec = world["ledger"].get(out.answer["accepted_claims"][0]["evidence"][0])
    assert rec.source_kind == "derived" and rec.parents == ("ev1",) and rec.resolution["known"]


def test_computed_value_is_not_overwritten_by_opinion(world):
    """A second measurement with another method is a new record; the ledger keeps both and the claim cites one."""
    drv = Scripted([load(world), measure([10, 40]), measure([10, 40], "abp_sbp_ecg"), submit_from_measure()])
    out = Controller(spec(world), drv, world["executor"], world["ledger"]).run()
    assert out.status == "success"
    recs = world["ledger"].records()
    assert [r.evidence_id for r in recs] == ["ev1", "ev2"] and recs[0].parameters["guidance"] == "pulse"
    assert recs[1].parameters["guidance"] == "ecg" and recs[1].timebase["sync_status"] == "index_aligned_declared"
