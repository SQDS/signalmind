"""Thin controller: the model chooses how to investigate; tools measure; the gate checks the submitted claims.

`SubmissionGate` is the SDK-independent core: it verifies a bundle, returns actionable deficiencies, counts repair
cycles per issue and stops on success, exhausted budget, repetition or lack of progress. `Controller` drives it with
an abstract `ModelDriver` (a scripted driver in tests; the real model through `ts_submit` in the MCP deployment).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from tsmcp.runtime.claims import SUPPORTED, AnswerBundle, ClaimStatus
from tsmcp.runtime.ledger import EvidenceLedger, EvidenceRecord
from tsmcp.runtime.render import render_answer, render_submitted
from tsmcp.runtime.taskspec import Budgets, TaskSpec
from tsmcp.runtime.verifier import Deficiency, VerificationReport, verify


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


@dataclass(frozen=True)
class Submit:
    bundle: dict


@dataclass(frozen=True)
class Stop:
    reason: str


Action = ToolCall | Submit | Stop


@dataclass
class ToolResult:
    output: dict
    evidence: list[EvidenceRecord] = field(default_factory=list)
    error: str | None = None


class ToolExecutor(Protocol):
    def call(self, name: str, args: dict, call_id: str) -> ToolResult: ...


class ModelDriver(Protocol):
    def next(self, observation: dict) -> Action: ...


class SubmissionGate:
    """Verifies bundles and keeps an explicit answer state.

    States (`finalization`): incomplete (no well-formed submission yet) | rejected (submissions seen, none accepted) |
    partial (accepted with unresolved components) | accepted | recorded (log mode: unverified). Flags qualify the
    state without hiding it: revision_rejected, new_evidence_after_acceptance, unsupported_after_new_evidence,
    task_revised, unsupported_after_task_revision, stopped:<reason>.
    verify mode: accept / retry with deficiencies / stop (budget, repeated issue, no progress). `accept_partial: true`
    is honoured only for deficiencies that were already reported to the model in an earlier retry; `final` has no
    effect (a submission is a request to submit, not authority to terminate). Evidence recorded after an acceptance
    re-verifies the accepted bundle; if it no longer verifies the answer is marked unsupported and `complete` is false.
    log mode: the same syntax handling and budgets, no semantic check: a well-formed submission is `recorded` as the
    answer with the model's values untouched and no feedback. Invalid submissions consume budget in both modes.
    """

    STATES = ("incomplete", "rejected", "partial", "accepted", "recorded")
    ANSWER_STATES = ("partial", "accepted", "recorded")

    def __init__(self, spec: TaskSpec, ledger: EvidenceLedger, budgets: Budgets | None = None, mode: str = "verify"):
        if mode not in ("verify", "log"):
            raise ValueError("mode must be 'verify' or 'log'")
        self.spec = spec
        self.ledger = ledger
        self.budgets = budgets or spec.budgets
        self.mode = mode
        self.submissions = 0
        self.issues: dict[tuple[str | None, str], int] = {}
        self.seen_issues: set[tuple[str | None, str]] = set()
        self.last_fingerprint: str | None = None
        self.decision: str | None = None
        self.stop_reason: str | None = None
        self.answer: dict | None = None  # authoritative application answer (rendered), or None
        self.last_report: VerificationReport | None = None
        self.finalization = "incomplete"
        self.flags: list[str] = []
        self.version = 0
        self.history: list[dict] = []
        self.evidence_after_acceptance: list[str] = []
        self.pending: dict | None = None  # deficiencies of the latest rejected submission or re-verification
        self.accepted_bundle: AnswerBundle | None = None
        self.stopped = False

    # ---- state ------------------------------------------------------------------------------------------------------
    @property
    def complete(self) -> bool:
        return self.finalization in ("accepted", "recorded") and not any(
            f.startswith("unsupported_after") for f in self.flags
        )

    def state(self) -> dict:
        return {
            "mode": self.mode,
            "finalization": self.finalization,
            "flags": list(self.flags),
            "complete": self.complete,
            "version": self.version,
            "submissions": self.submissions,
            "stop_reason": self.stop_reason,
            "evidence_after_acceptance": list(self.evidence_after_acceptance),
            "pending": self.pending,
            "history": list(self.history),
            "answer": self.answer,
        }

    def _flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)
        if self.answer is not None:
            self.answer["flags"] = list(self.flags)
            self.answer["complete"] = self.complete

    def _reverify(self, flag: str) -> None:
        """Re-check the accepted bundle against the current ledger and task; never keep a stale answer silently."""
        if self.mode != "verify" or self.accepted_bundle is None:
            return
        report = verify(self.accepted_bundle, self.ledger, self.spec)
        if not report.ok:
            self.pending = {"source": flag, "deficiencies": [d.to_dict() for d in report.deficiencies]}
            self._flag(flag)
            if self.answer is not None:
                self.answer["pending"] = self.pending

    def note_evidence(self, evidence_ids: list[str]) -> None:
        """Called by the execution layer after every tool call; new evidence after an acceptance is never hidden."""
        if self.version and evidence_ids and self.finalization in self.ANSWER_STATES:
            self.evidence_after_acceptance += list(evidence_ids)
            self._flag("new_evidence_after_acceptance")
            if self.answer is not None:
                self.answer["evidence_after_acceptance"] = list(self.evidence_after_acceptance)
            self._reverify("unsupported_after_new_evidence")

    def revise_task(self, spec: TaskSpec) -> None:
        """A changed task re-verifies the accepted answer; it is never reused silently."""
        self.spec = spec
        if self.version and self.finalization in self.ANSWER_STATES:
            self._flag("task_revised")
            self._reverify("unsupported_after_task_revision")

    def remaining(self) -> dict:
        return {
            "submissions": self.budgets.max_submissions - self.submissions,
            "repairs_per_issue": {
                f"{k[0]}:{k[1]}": self.budgets.max_repair_cycles_per_issue - n for k, n in self.issues.items()
            },
        }

    # ---- submission ----------------------------------------------------------------------------------------
    def submit(self, bundle_dict: dict) -> dict:
        if self.stopped:
            return {"decision": "stopped", "stop_reason": self.stop_reason, "deficiencies": [], "answer": self.answer,
                    "finalization": self.finalization, "flags": list(self.flags)}  # fmt: skip
        self.submissions += 1
        try:
            bundle = AnswerBundle.from_dict(bundle_dict)
        except Exception as e:  # noqa: BLE001 - any malformed input is reported, never raised to the tool layer
            d = Deficiency("malformed_bundle", None, f"{type(e).__name__}: {e}")
            report = VerificationReport(ok=False, deficiencies=[d])
            self.ledger.record_bundle(
                bundle_dict if isinstance(bundle_dict, dict) else {"raw": str(bundle_dict)}, report.to_dict()
            )
            self._count([d])
            return self._retry_or_stop(None, report)
        if self.mode == "log":
            return self._record(bundle)
        report = verify(bundle, self.ledger, self.spec)
        self.last_report = report
        self.ledger.record_bundle(bundle.to_dict(), report.to_dict())
        unresolved = bool(bundle.limitations) or any(ClaimStatus(c.status) not in SUPPORTED for c in bundle.claims)
        if report.ok:
            return self._accept(bundle, report, "partial" if unresolved else "accepted")
        keys = {(d.claim_id, d.code) for d in report.deficiencies}
        if bundle.accept_partial and keys <= self.seen_issues:
            return self._accept(bundle, report, "partial")  # the model reviewed these deficiencies and accepts them
        self._count(report.deficiencies)
        note = None
        if bundle.accept_partial:
            note = "accept_partial ignored: these deficiencies had not been reported before; review them first"
        elif bundle.final:
            note = "`final` has no effect: a submission is checked like any other; use accept_partial after review"
        return self._retry_or_stop(bundle, report, note)

    def _accept(self, bundle: AnswerBundle, report: VerificationReport, finalization: str) -> dict:
        self.version += 1
        self.finalization = finalization
        self.decision = "accepted" if finalization == "accepted" else "accepted_partial"
        self.flags = []
        self.pending = None
        self.accepted_bundle = bundle
        self.evidence_after_acceptance = []
        outcome = "success" if finalization == "accepted" else "partial"
        self.answer = render_answer(bundle, report, self.ledger, outcome)
        self.answer.update(finalization=finalization, version=self.version, flags=[], complete=self.complete)
        self.history.append({"version": self.version, "decision": self.decision, "submission": self.submissions,
                             "fingerprint": report.bundle_fingerprint})  # fmt: skip
        return {"decision": self.decision, "outcome": outcome, "version": self.version,
                "deficiencies": [d.to_dict() for d in report.deficiencies], "answer": self.answer,
                "claims": report.claims, "remaining": self.remaining(), "finalization": finalization}  # fmt: skip

    def _record(self, bundle: AnswerBundle) -> dict:
        """log mode: no semantic verification; the model's claims become the answer as submitted."""
        self.version += 1
        self.decision = self.finalization = "recorded"
        self.flags = []
        self.pending = None
        self.accepted_bundle = bundle
        self.evidence_after_acceptance = []
        self.ledger.record_bundle(bundle.to_dict(), {"ok": None, "mode": "log", "deficiencies": [],
                                                    "bundle_fingerprint": bundle.fingerprint()})  # fmt: skip
        self.answer = render_submitted(bundle, self.ledger)
        self.answer.update(finalization="recorded", version=self.version, flags=[], complete=True)
        self.history.append({"version": self.version, "decision": "recorded", "submission": self.submissions,
                             "fingerprint": bundle.fingerprint()})  # fmt: skip
        if self.submissions >= self.budgets.max_submissions:
            self.stopped, self.stop_reason = True, "submission_budget_exhausted"
            self._flag(f"stopped:{self.stop_reason}")
        return {"decision": "recorded", "version": self.version, "deficiencies": [], "answer": self.answer,
                "remaining": self.remaining(), "finalization": "recorded"}  # fmt: skip

    def _count(self, deficiencies: list[Deficiency]) -> None:
        for key in {(d.claim_id, d.code) for d in deficiencies}:
            self.issues[key] = self.issues.get(key, 0) + 1
            self.seen_issues.add(key)

    def _retry_or_stop(self, bundle: AnswerBundle | None, report: VerificationReport, note: str | None = None) -> dict:
        fp = report.bundle_fingerprint
        exhausted = [k for k, n in self.issues.items() if n > self.budgets.max_repair_cycles_per_issue]
        reason = None
        if self.submissions >= self.budgets.max_submissions:
            reason = "submission_budget_exhausted"
        elif exhausted:
            reason = "repeated_unresolved_issue"
        elif fp and fp == self.last_fingerprint:
            reason = "no_progress"
        self.last_fingerprint = fp or self.last_fingerprint
        deficiencies = [d.to_dict() for d in report.deficiencies]
        self.pending = {"source": "submission", "submission": self.submissions, "deficiencies": deficiencies,
                        "fingerprint": fp}  # fmt: skip
        if self.version:  # an accepted answer exists: keep it, flag the rejected revision
            self._flag("revision_rejected")
            self.answer["pending"] = self.pending
        else:
            self.finalization = "rejected"
        if reason:
            self.stopped, self.decision, self.stop_reason = True, "stopped", reason
            self._flag(f"stopped:{reason}")
            if not self.version:
                if bundle is not None:
                    self.answer = render_answer(bundle, report, self.ledger, "unresolved")
                else:
                    self.answer = {
                        "outcome": "unresolved", "accepted_claims": [], "unresolved": [], "limitations": [],
                        "text": "no well-formed submission", "ledger_fingerprint": self.ledger.fingerprint(),
                    }  # fmt: skip
                self.answer.update(finalization="rejected", version=0, flags=list(self.flags), complete=False,
                                   pending=self.pending)  # fmt: skip
            return {"decision": "stopped", "stop_reason": reason, "deficiencies": deficiencies, "answer": self.answer,
                    "finalization": self.finalization, "flags": list(self.flags)}  # fmt: skip
        out = {
            "decision": "retry",
            "deficiencies": deficiencies,
            "claims": report.claims,
            "remaining": self.remaining(),
            "finalization": self.finalization,
            "flags": list(self.flags),
            "hint": "fix the listed deficiencies (measure the claimed interval/method, cite the right evidence, "
            "qualify or abstain); after reviewing them you may resubmit with accept_partial=true",
        }
        if note:
            out["note"] = note
        return out


@dataclass
class Outcome:
    # success | partial | unresolved | budget_exhausted | no_progress | repeated_unresolved_issue | incomplete_execution
    status: str
    answer: dict | None
    steps: int
    tool_calls: int
    submissions: int
    elapsed_s: float
    events: list[dict]


class Controller:
    def __init__(
        self,
        spec: TaskSpec,
        driver: ModelDriver,
        executor: ToolExecutor,
        ledger: EvidenceLedger,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.spec, self.driver, self.executor, self.ledger, self.clock = spec, driver, executor, ledger, clock
        self.gate = SubmissionGate(spec, ledger)

    def run(self) -> Outcome:
        b = self.spec.budgets
        start = self.clock()
        steps = tool_calls = 0
        events: list[dict] = []
        obs: dict = {"task": self.spec.to_dict(), "capabilities": list(self.spec.capabilities)}
        while True:
            if steps >= b.max_steps:
                return self._finish("budget_exhausted", steps, tool_calls, start, events, "max_steps")
            if self.clock() - start > b.max_seconds:
                return self._finish("budget_exhausted", steps, tool_calls, start, events, "max_seconds")
            action = self.driver.next(obs)
            steps += 1
            if isinstance(action, Stop):
                events.append({"step": steps, "stop": action.reason})
                status = "incomplete_execution" if action.reason.startswith("infra") else "unresolved"
                return self._finish(status, steps, tool_calls, start, events, action.reason)
            if isinstance(action, ToolCall):
                if tool_calls >= b.max_tool_calls:
                    obs = {"error": "tool-call budget exhausted; submit what is supported or abstain"}
                    events.append({"step": steps, "tool_budget_exhausted": True})
                    continue
                tool_calls += 1
                res = self.executor.call(action.name, action.args, call_id=f"call{tool_calls}")
                self.gate.note_evidence([r.evidence_id for r in res.evidence])
                events.append(
                    {
                        "step": steps,
                        "tool": action.name,
                        "args": action.args,
                        "error": res.error,
                        "evidence": [r.evidence_id for r in res.evidence],
                    }
                )
                obs = {
                    "tool": action.name,
                    "error": res.error,
                    "result": res.output,
                    "evidence": [r.summary() for r in res.evidence],
                }
                continue
            verdict = self.gate.submit(action.bundle)
            events.append(
                {
                    "step": steps,
                    "submit": verdict["decision"],
                    "deficiencies": verdict.get("deficiencies", []),
                    "stop_reason": verdict.get("stop_reason"),
                }
            )
            if verdict["decision"] == "accepted":
                return self._finish(verdict["outcome"], steps, tool_calls, start, events)
            if verdict["decision"] in ("accepted_partial", "recorded"):
                return self._finish("partial" if verdict["decision"] == "accepted_partial" else "recorded",
                                    steps, tool_calls, start, events)  # fmt: skip
            if verdict["decision"] == "stopped":
                return self._finish(verdict["stop_reason"], steps, tool_calls, start, events)
            obs = {"submission": verdict}

    def _finish(self, status, steps, tool_calls, start, events, reason=None) -> Outcome:
        answer = self.gate.answer
        if answer is None:
            answer = {
                "outcome": status,
                "accepted_claims": [],
                "unresolved": [],
                "limitations": [],
                "text": f"no accepted answer ({reason or status})",
                "ledger_fingerprint": self.ledger.fingerprint(),
            }
        return Outcome(status, answer, steps, tool_calls, self.gate.submissions, round(self.clock() - start, 3), events)
