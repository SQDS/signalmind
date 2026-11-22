"""SignalMind runtime: task specification, evidence ledger, claims, deterministic verification, thin controller.

Pure Python + numpy. Nothing here imports `mcp` or anything under `eval/`; the MCP adapter lives in
`tsmcp.server.runtime_tools`. Everything is opt-in; the default server never loads this package.
"""

from tsmcp.runtime.claims import AnswerBundle, Claim, ClaimStatus
from tsmcp.runtime.controller import Controller, SubmissionGate
from tsmcp.runtime.ledger import EvidenceLedger, EvidenceRecord, SourceKind
from tsmcp.runtime.taskspec import Budgets, QuantitySpec, TaskSpec
from tsmcp.runtime.verifier import Deficiency, VerificationReport, verify

__all__ = [
    "AnswerBundle",
    "Budgets",
    "Claim",
    "ClaimStatus",
    "Controller",
    "Deficiency",
    "EvidenceLedger",
    "EvidenceRecord",
    "QuantitySpec",
    "SourceKind",
    "SubmissionGate",
    "TaskSpec",
    "VerificationReport",
    "verify",
]
