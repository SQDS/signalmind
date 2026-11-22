"""Several evidence errors in one submission must leave the model a chance to repair."""

from tsmcp.runtime.controller import SubmissionGate
from tsmcp.runtime.ledger import EvidenceLedger, SourceKind
from tsmcp.runtime.taskspec import TaskSpec


def gate():
    ledger = EvidenceLedger("repair-budget")
    ledger.record(
        SourceKind.COMPUTED,
        "mean",
        tool="ts_measure",
        tool_call_id="measure1",
        value=80,
        unit="mmHg",
        definition_id="generic.mean.v1",
        channels_effective=("ABP",),
        interval_requested=(0, 30),
        intervals_effective=[(0, 30)],
    )
    return SubmissionGate(TaskSpec.minimal(), ledger)


def bundle(refs):
    return {
        "claims": [
            {
                "claim_id": "mean",
                "quantity": "mean",
                "status": "validated_measurement",
                "value": 80,
                "unit": "mmHg",
                "interval": [0, 30],
                "evidence_refs": refs,
                "definition_id": "generic.mean.v1",
            }
        ]
    }


def test_multiple_bad_references_in_first_submission_can_be_repaired():
    g = gate()
    first = g.submit(bundle(["missing-a", "missing-b", "missing-c"]))
    assert first["decision"] == "retry" and len(first["deficiencies"]) == 3
    assert {d["code"] for d in first["deficiencies"]} == {"invalid_reference"}
    assert g.submit(bundle(["ev1"]))["decision"] == "accepted"
    assert g.submissions == 2 and g.answer["accepted_claims"][0]["value"] == 80


def test_same_issue_across_distinct_submissions_still_exhausts_repair_budget():
    g = gate()
    decisions = []
    for attempt in range(3):
        reply = g.submit(bundle([f"missing-{attempt}-{i}" for i in range(3)]))
        decisions.append(reply["decision"])
    assert decisions == ["retry", "retry", "stopped"]
    assert reply["stop_reason"] == "repeated_unresolved_issue" and g.submissions == 3
    assert not g.answer["accepted_claims"]
