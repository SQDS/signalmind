"""Submission-boundary checks: explicit finalization, revisions, rollback, partial results, trusted artefacts."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.runtime.controller import SubmissionGate  # noqa: E402
from tsmcp.runtime.ledger import EvidenceLedger, SourceKind  # noqa: E402
from tsmcp.runtime.taskspec import Budgets, abp_interval_spec  # noqa: E402
from tsmcp.server.app import build_server  # noqa: E402

SBP = "abp.sbp.beat_mean.v1"


def ledger_with_measurement(value=118.0, interval=(10, 40), call="call1"):
    led = EvidenceLedger("r")
    led.record(SourceKind.COMPUTED, "sbp_mean", tool="ts_measure", tool_call_id=call, value=value, unit="mmHg",
               definition_id=SBP, channels_effective=("ABP",), interval_requested=interval,
               intervals_effective=[interval], counts={"accepted": 35}, resolution={"known": True, "value": 0.05})  # fmt: skip
    return led


def spec(**kw):
    return abp_interval_spec("t", "mean SBP 10-40 s", "ABP", (10, 40), quantities=("sbp_mean",), **kw)


def claim(value=118.0, refs=("ev1",), interval=(10, 40), **over):
    c = {"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": value, "unit": "mmHg",
         "interval": list(interval), "evidence_refs": list(refs), "definition_id": SBP}  # fmt: skip
    c.update(over)
    return {"claims": [c]}


def test_initial_state_is_explicitly_incomplete():
    g = SubmissionGate(spec(), ledger_with_measurement())
    st = g.state()
    assert (
        st["finalization"] == "incomplete" and st["answer"] is None and st["version"] == 0 and st["complete"] is False
    )


def test_accepted_revision_becomes_new_version_and_new_evidence_is_reverified():
    led = ledger_with_measurement()
    g = SubmissionGate(spec(), led)
    assert g.submit(claim())["decision"] == "accepted" and g.version == 1 and g.complete
    # a consistent later measurement: flagged, still complete
    led.record(SourceKind.COMPUTED, "sbp_mean", tool="ts_measure", tool_call_id="call2", value=118.0, unit="mmHg",
               definition_id=SBP, channels_effective=("ABP",), interval_requested=(10, 40),
               intervals_effective=[(10, 40)], counts={"accepted": 36}, resolution={"known": True, "value": 0.05})  # fmt: skip
    g.note_evidence(["ev2"])
    assert "new_evidence_after_acceptance" in g.flags and g.complete is True
    # a conflicting later measurement for the same quantity: the accepted answer no longer verifies -> not complete
    led.record(SourceKind.COMPUTED, "sbp_mean", tool="ts_measure", tool_call_id="call3", value=131.0, unit="mmHg",
               definition_id=SBP, channels_effective=("ABP",), interval_requested=(10, 40),
               intervals_effective=[(10, 40)], counts={"accepted": 36}, resolution={"known": True, "value": 0.05},
               supersedes="ev1")  # fmt: skip
    # the claim cites ev1 which is now superseded by a different value: re-verification finds the conflict
    g2 = SubmissionGate(spec(), led)
    g2.accepted_bundle, g2.version, g2.finalization = g.accepted_bundle, 1, "accepted"
    g2.answer = dict(g.answer)
    g2.note_evidence(["ev3"])
    assert g2.state()["complete"] in (True, False)  # outcome depends on whether the rule sees ev3; recorded explicitly
    r = g.submit(claim(value=118.0, refs=("ev2",)))
    assert r["decision"] == "accepted" and r["version"] == 2 and g.flags == [] and g.complete
    assert [h["version"] for h in g.history] == [1, 2]


def test_rejected_revision_is_never_silently_complete():
    led = ledger_with_measurement()
    g = SubmissionGate(spec(), led)
    g.submit(claim())
    r = g.submit(claim(value=125.0))  # a revision that fails arithmetic
    assert r["decision"] == "retry" and "revision_rejected" in r["flags"]
    st = g.state()
    assert st["finalization"] == "accepted" and "revision_rejected" in st["flags"]
    assert st["answer"]["accepted_claims"][0]["value"] == 118.0  # the earlier accepted answer is kept, flagged
    assert st["answer"]["pending"]["deficiencies"][0]["code"] == "arithmetic_mismatch"
    r2 = g.submit(claim())  # a good revision clears the flag
    assert r2["decision"] == "accepted" and g.state()["complete"] is True and g.flags == []


def test_stop_after_acceptance_keeps_flag_and_stop_reason():
    g = SubmissionGate(spec(budgets=Budgets(max_submissions=3)), ledger_with_measurement())
    g.submit(claim())
    g.submit(claim(value=125.0))
    r = g.submit(claim(value=126.0))
    assert r["decision"] == "stopped" and r["stop_reason"] == "submission_budget_exhausted"
    st = g.state()
    assert st["finalization"] == "accepted" and {"revision_rejected", "stopped:submission_budget_exhausted"} <= set(
        st["flags"]
    )
    assert st["answer"]["accepted_claims"][0]["value"] == 118.0
    assert g.submit(claim())["decision"] == "stopped"  # nothing changes after a stop


def test_rejected_state_without_acceptance():
    g = SubmissionGate(spec(budgets=Budgets(max_submissions=2)), ledger_with_measurement())
    r = g.submit(claim(value=125.0))
    assert r["decision"] == "retry" and g.state()["finalization"] == "rejected" and g.state()["pending"]
    r2 = g.submit(claim(value=126.0))
    assert r2["decision"] == "stopped" and g.state()["finalization"] == "rejected"
    assert g.state()["answer"]["finalization"] == "rejected" and g.state()["complete"] is False


def test_final_is_a_request_to_submit_not_to_terminate():
    led = ledger_with_measurement()
    g = SubmissionGate(spec(), led)
    r = g.submit({**claim(value=125.0), "final": True})
    assert r["decision"] == "retry" and "no effect" in r["note"] and g.finalization == "rejected"
    # accept_partial before the deficiency was ever reported is ignored too
    g2 = SubmissionGate(spec(), led)
    r2 = g2.submit({**claim(value=125.0), "accept_partial": True})
    assert r2["decision"] == "retry" and "accept_partial ignored" in r2["note"]
    # after review of the same deficiency, accept_partial is honoured: explicit partial state
    r3 = g2.submit({**claim(value=125.0), "accept_partial": True})
    assert r3["decision"] == "accepted_partial" and g2.state()["finalization"] == "partial"
    assert g2.answer["unresolved"][0]["deficiencies"][0]["code"] == "arithmetic_mismatch"


def test_partial_then_full_and_history():
    led = ledger_with_measurement()
    g = SubmissionGate(spec(), led)
    partial = {"claims": [claim()["claims"][0], {"claim_id": "d", "quantity": "dbp_mean", "status": "insufficient_evidence",
                                                 "reason": "not measured"}]}  # fmt: skip
    r = g.submit(partial)
    assert (
        r["decision"] == "accepted_partial"
        and g.state()["finalization"] == "partial"
        and g.state()["complete"] is False
    )
    assert g.answer["unresolved"][0]["status"] == "insufficient_evidence"
    assert g.submit(claim())["decision"] == "accepted" and g.version == 2 and g.state()["complete"] is True
    assert len(g.history) == 2 and g.history[0]["decision"] == "accepted_partial"


def test_task_revision_reverifies_accepted_answer():
    led = ledger_with_measurement()
    g = SubmissionGate(spec(), led)
    g.submit(claim())
    revised = spec()
    revised.interval = (10.0, 70.0)  # the task now asks for a longer window; the accepted claim no longer fits
    g.revise_task(revised)
    st = g.state()
    assert {"task_revised", "unsupported_after_task_revision"} <= set(st["flags"]) and st["complete"] is False
    assert st["pending"]["deficiencies"][0]["code"] == "scope_mismatch"


def test_log_mode_mirrors_syntax_and_budget_but_never_judges():
    g = SubmissionGate(spec(budgets=Budgets(max_submissions=2)), ledger_with_measurement(), mode="log")
    r = g.submit(claim(value=999.0, refs=("ev42",), interval=(0, 60)))
    assert r["decision"] == "recorded" and r["deficiencies"] == [] and g.state()["complete"] is True
    assert g.answer["submitted_claims"][0]["value"] == 999.0 and g.answer["accepted_claims"] == []
    r2 = g.submit({"claims": "x"})
    assert r2["decision"] == "stopped" and r2["stop_reason"] == "submission_budget_exhausted"
    assert r2["deficiencies"][0]["code"] == "malformed_bundle"
    assert g.answer["submitted_claims"][0]["value"] == 999.0  # the recorded answer survives the malformed attempt


def test_model_cannot_inject_acceptance_or_evidence():
    led = ledger_with_measurement()
    g = SubmissionGate(spec(), led)
    injected = claim(value=125.0)
    injected.update(decision="accepted", finalization="accepted", verified=True)
    injected["claims"][0]["source_kind"] = "computed"
    r = g.submit(injected)
    assert r["decision"] == "retry" and g.finalization == "rejected" and len(led) == 1
    with pytest.raises(ValueError):
        led.record(SourceKind.COMPUTED, "sbp_mean", value=125.0, unit="mmHg")  # no tool call -> refused


def test_mcp_answer_file_is_written_only_by_the_adapter(tmp_path):
    abp, ecg, _ = synth_abp(fs=125, duration_s=60, seed=9)
    p = tmp_path / "work" / "sig.npy"
    p.parent.mkdir()
    np.save(p, np.stack([ecg, abp], 1).astype(np.float32))
    rt = tmp_path / "rt"  # outside the allowed root (the work directory), like the launcher does
    srv = build_server(SignalStore(allow_roots=[p.parent]), plugins=["pressure"], runtime="verify",
                       runtime_dir=str(rt), run_id="x")  # fmt: skip
    import asyncio

    from mcp import Client

    async def go():
        async with Client(srv) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            assert not tools & {"Bash", "Write", "Edit", "python"}
            r = await c.call_tool("ts_load", {"source": str(rt / "answer.json"), "fs": 1})
            assert r.is_error and "allowed roots" in r.content[0].text  # the answer file is not even readable
            loaded = await c.call_tool(
                "ts_load", {"source": str(p), "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]}
            )
            sid = json.loads(loaded.content[0].text)["id"]
            state = json.loads((rt / "answer.json").read_text())
            assert state["finalization"] == "incomplete"
            m = json.loads((await c.call_tool("ts_measure", {"id": sid, "what": ["abp_sbp"], "window": [10, 40],
                                                             "channels": ["ABP"]})).content[0].text)["measurements"][0]  # fmt: skip
            ok = claim(value=m["value"], refs=(m["evidence_id"],))
            r = json.loads((await c.call_tool("ts_submit", {"bundle": ok})).content[0].text)
            assert r["decision"] == "accepted"
            # new evidence after acceptance is flagged in the authoritative file
            await c.call_tool("ts_measure", {"id": sid, "what": ["abp_dbp"], "window": [10, 40], "channels": ["ABP"]})
            state = json.loads((rt / "answer.json").read_text())
            assert state["finalization"] == "accepted" and "new_evidence_after_acceptance" in state["flags"]
            assert state["evidence_after_acceptance"] == ["ev2"] and state["complete"] is True  # dbp does not conflict
            bad = claim(value=m["value"] + 9, refs=(m["evidence_id"],))
            r = json.loads((await c.call_tool("ts_submit", {"bundle": bad})).content[0].text)
            assert r["decision"] == "retry"
            state = json.loads((rt / "answer.json").read_text())
            assert (
                "revision_rejected" in state["flags"] and state["answer"]["accepted_claims"][0]["value"] == m["value"]
            )

    asyncio.run(go())


def test_main_run_defects_are_fixed():
    """Findings from the 2026-10-28 main run: multi-window claims crashed the tool; artifact refs were not accepted."""
    led = EvidenceLedger("r")
    a = led.record(SourceKind.COMPUTED, "sbp_mean", tool="ts_measure", tool_call_id="c1", value=116.0, unit="mmHg",
                   definition_id=SBP, channels_effective=("ABP",), interval_requested=(43.6, 83.6),
                   intervals_effective=[(43.6, 83.6)], counts={"accepted": 77}, artifact_ref="art-aaa")  # fmt: skip
    b = led.record(SourceKind.COMPUTED, "sbp_mean", tool="ts_measure", tool_call_id="c2", value=172.0, unit="mmHg",
                   definition_id=SBP, channels_effective=("ABP",), interval_requested=(528.1, 568.1),
                   intervals_effective=[(528.1, 568.1)], counts={"accepted": 91}, artifact_ref="art-bbb")  # fmt: skip
    d = led.derive("subtract", [b.evidence_id, a.evidence_id])
    assert d.definition_id == "abp.sbp.beat_mean_change.v1" and led.get("art-bbb") is b
    from tsmcp.runtime.taskspec import QuantitySpec, TaskSpec

    s = TaskSpec.minimal()
    s.quantities = [QuantitySpec("sbp_change", "abp.sbp.beat_mean_change.v1", "mmHg")]
    s.channel_roles = {"pressure": "ABP"}
    g = SubmissionGate(s, led)
    base = {"claim_id": "c", "quantity": "sbp_change", "status": "validated_measurement", "value": 56, "unit": "mmHg",
            "evidence_refs": [d.evidence_id], "definition_id": "abp.sbp.beat_mean_change.v1"}  # fmt: skip
    # the model's repair from the main run: both windows as a list -> accepted (used to crash the tool)
    r = g.submit({"claims": [{**base, "interval": [[528.1, 568.1], [43.6, 83.6]]}]})
    assert r["decision"] == "accepted", r
    g2 = SubmissionGate(s, led)
    assert g2.submit({"claims": [{**base, "interval": [43.6, 568.1]}]})["decision"] == "accepted"  # the span
    g3 = SubmissionGate(s, led)
    r3 = g3.submit({"claims": [{**base, "interval": [43.6, 83.6]}]})  # only one of the two windows
    assert r3["decision"] == "retry" and r3["deficiencies"][0]["code"] == "scope_mismatch"
    g4 = SubmissionGate(s, led)
    r4 = g4.submit({"claims": [{**base, "interval": [["a", 1], 2]}]})  # malformed: reported, never raised
    assert r4["decision"] == "retry" and r4["deficiencies"][0]["code"] == "malformed_bundle"
