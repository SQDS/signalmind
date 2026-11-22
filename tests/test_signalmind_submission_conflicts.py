"""Contradictory per-quantity conclusions are rejected without rewriting a prior answer."""

import copy

import pytest

from tsmcp.runtime.controller import SubmissionGate
from tsmcp.runtime.ledger import EvidenceLedger, SourceKind
from tsmcp.runtime.taskspec import abp_interval_spec


def gate(mode):
    ledger = EvidenceLedger("submission-conflicts")
    ledger.record(
        SourceKind.COMPUTED,
        "sbp_mean",
        tool="ts_measure",
        tool_call_id="m1",
        value=120.0,
        unit="mmHg",
        definition_id="abp.sbp.beat_mean.v1",
        channels_effective=("ABP",),
        interval_requested=(0, 30),
        intervals_effective=[(0, 30)],
        counts={"accepted": 30},
        resolution={"known": True, "value": 1.0},
    )
    spec = abp_interval_spec("t", "Mean beatwise SBP from 0 to 30 s", "ABP", (0, 30), quantities=("sbp_mean",))
    return SubmissionGate(spec, ledger, mode=mode)


def measured():
    return {
        "claim_id": "sbp",
        "quantity": "sbp_mean",
        "status": "validated_measurement",
        "value": 120.0,
        "unit": "mmHg",
        "interval": [0, 30],
        "evidence_refs": ["ev1"],
    }


@pytest.mark.parametrize("mode", ["log", "verify"])
@pytest.mark.parametrize("kind", ["limitation", "insufficient_evidence", "unresolved_conflict", "incomplete_execution"])
def test_conflicting_revision_is_rejected_and_can_be_repaired_with_notes(mode, kind):
    g = gate(mode)
    accepted = "recorded" if mode == "log" else "accepted"
    assert g.submit({"claims": [measured()]})["decision"] == accepted
    bundle = {"claims": [measured()]}
    unresolved = {
        "quantity": "sbp_mean",
        "status": "insufficient_evidence" if kind == "limitation" else kind,
        "reason": "uncertainty remains",
    }
    if kind == "limitation":
        bundle["limitations"] = [unresolved]
    else:
        bundle["claims"].append({"claim_id": "uncertain", **unresolved})
    original = copy.deepcopy(bundle)
    reply = g.submit(bundle)
    assert reply["decision"] == "retry" and g.version == 1
    assert reply["deficiencies"][0]["code"] == "malformed_bundle"
    message = reply["deficiencies"][0]["message"]
    assert "sbp_mean" in message and "notes" in message and "withdraw" in message
    stored = g.answer["submitted_claims" if mode == "log" else "accepted_claims"]
    assert stored[0]["value"] == 120.0 and not g.answer["limitations"]
    assert bundle == original
    assert g.submit({"claims": [measured()], "notes": ["uncertainty remains"]})["decision"] == accepted
    assert g.accepted_bundle.notes == ["uncertainty remains"]


@pytest.mark.parametrize("mode", ["log", "verify"])
def test_other_quantity_can_remain_unresolved_in_partial_answer(mode):
    g = gate(mode)
    bundle = {
        "claims": [measured()],
        "limitations": [
            {"quantity": "dbp_mean", "status": "insufficient_evidence", "reason": "DBP was not measured"},
        ],
    }
    reply = g.submit(bundle)
    assert reply["decision"] == ("recorded" if mode == "log" else "accepted_partial")
    assert g.answer["limitations"][0]["quantity"] == "dbp_mean"


@pytest.mark.parametrize("mode", ["log", "verify"])
def test_withdrawing_measurement_allows_pure_abstention(mode):
    g = gate(mode)
    g.submit({"claims": [measured()]})
    reply = g.submit(
        {
            "claims": [],
            "limitations": [
                {"quantity": "sbp_mean", "status": "insufficient_evidence", "reason": "withdrawing the estimate"},
            ],
        }
    )
    assert reply["decision"] == ("recorded" if mode == "log" else "accepted_partial")
    assert not g.answer.get("submitted_claims") and not g.answer["accepted_claims"]
    assert g.answer["limitations"][0]["quantity"] == "sbp_mean"
