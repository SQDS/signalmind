"""Render the final answer from validated structured claims. Free text never supplies a number."""

from __future__ import annotations

from tsmcp.runtime.claims import SUPPORTED, AnswerBundle, ClaimStatus
from tsmcp.runtime.ledger import EvidenceLedger
from tsmcp.runtime.verifier import VerificationReport


def render_answer(bundle: AnswerBundle, report: VerificationReport, ledger: EvidenceLedger, outcome: str) -> dict:
    accepted, unresolved = [], []
    for c in bundle.claims:
        issues = report.for_claim(c.claim_id)
        st = ClaimStatus(c.status)
        entry = {
            "claim_id": c.claim_id,
            "quantity": c.quantity,
            "value": c.value,
            "unit": c.unit,
            "interval": list(c.interval) if c.interval else None,
            "definition_id": c.definition_id,
            "evidence": list(c.evidence_refs),
        }
        if issues:
            entry["status"] = "unresolved"
            entry["deficiencies"] = [{"code": d.code, "message": d.message} for d in issues]
            unresolved.append(entry)
        elif st in SUPPORTED:
            entry["status"] = st.value
            entry["precision_decimals"] = c.precision_decimals
            entry["acknowledged_warnings"] = list(c.acknowledged_warnings)
            accepted.append(entry)
        else:
            entry["status"] = st.value
            entry["reason"] = c.reason
            unresolved.append(entry)
    limitations = [dict(lim) for lim in bundle.limitations]
    lines = []
    for e in accepted:
        label = "" if e["status"] == "validated_measurement" else f" ({e['status']})"
        if isinstance(e["value"], (int, float)):
            lines.append(
                f"{e['quantity']} = {e['value']} {e['unit'] or ''}{label}; evidence {', '.join(e['evidence'])}".strip()
            )
        else:
            lines.append(f"{e['quantity']}: {e['value']}{label}; evidence {', '.join(e['evidence'])}")
    for e in unresolved:
        why = e.get("reason") or "; ".join(d["message"] for d in e.get("deficiencies", []))
        lines.append(f"{e['quantity']}: {e['status']} — {why}")
    for lim in limitations:
        lines.append(f"{lim['quantity']}: {lim['status']} — {lim['reason']}")
    return {
        "outcome": outcome,
        "accepted_claims": accepted,
        "unresolved": unresolved,
        "limitations": limitations,
        "notes": list(bundle.notes),
        "text": "\n".join(lines) if lines else "no supported claims",
        "ledger_fingerprint": ledger.fingerprint(),
    }


def render_submitted(bundle: AnswerBundle, ledger: EvidenceLedger) -> dict:
    """log mode: the model's claims as submitted, labelled unverified; values are never altered or filtered."""
    claims = []
    lines = []
    for c in bundle.claims:
        claims.append({
            "claim_id": c.claim_id, "quantity": c.quantity, "value": c.value, "unit": c.unit,
            "interval": list(c.interval) if c.interval else None, "definition_id": c.definition_id,
            "evidence": list(c.evidence_refs), "submitted_status": c.status, "status": "submitted_unverified",
            "reason": c.reason, "precision_decimals": c.precision_decimals,
        })  # fmt: skip
        if isinstance(c.value, (int, float)):
            lines.append(f"{c.quantity} = {c.value} {c.unit or ''} (submitted as {c.status}, unverified); "
                         f"evidence {', '.join(c.evidence_refs) or 'none cited'}")  # fmt: skip
        else:
            lines.append(f"{c.quantity}: {c.value if c.value is not None else c.status} (submitted, unverified)"
                         + (f" — {c.reason}" if c.reason else ""))  # fmt: skip
    for lim in bundle.limitations:
        lines.append(f"{lim['quantity']}: {lim['status']} — {lim['reason']}")
    return {
        "outcome": "recorded", "submitted_claims": claims, "accepted_claims": [], "unresolved": [],
        "limitations": [dict(lim) for lim in bundle.limitations],
        "text": "\n".join(lines) if lines else "no claims submitted", "ledger_fingerprint": ledger.fingerprint(),
    }  # fmt: skip
