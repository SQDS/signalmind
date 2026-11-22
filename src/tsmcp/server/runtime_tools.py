"""Opt-in MCP adapter for the SignalMind runtime (`--runtime log|verify`).

Both modes expose the same tools (ts_submit, ts_derive, ts_artifact, ts_evidence), the same evidence metadata in
replies and the same submission format and budgets. `log`: ts_submit records a well-formed bundle as the answer with
the model's values untouched (no semantic check, no feedback). `verify`: ts_submit verifies claims against the ledger
and returns structured deficiencies; the answer is the last accepted bundle with an explicit finalization state.
The authoritative output is `<runtime-dir>/answer.json`, written only by this adapter; raw model text is never the
application answer. The system controls the accepted application output; it does not stop the model from writing
arbitrary free text in its own reply.
"""

from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path

from mcp_types import TextContent, ToolAnnotations

from tsmcp.core.errors import SignalError
from tsmcp.runtime.artifacts import ArtifactStore
from tsmcp.runtime.controller import SubmissionGate
from tsmcp.runtime.evidence_builders import events_records, measurement_records, view_record
from tsmcp.runtime.ledger import DERIVED_OPS, EvidenceLedger
from tsmcp.runtime.taskspec import TaskSpec

SHARED_INSTRUCTIONS = """
Evidence and submission:
- Every tool reply lists `evidence_ids` for what was actually computed or viewed. A number you report must cite the
  evidence_id of the measurement that produced it, over the same window and channel, with its unit.
- Report conclusions with ts_submit({"claims": [...], "limitations": [...], "notes": [...], "final": false}).
  Each claim: {"claim_id", "quantity", "status", "value", "unit", "interval": [t0, t1], "evidence_refs": [...],
   "definition_id", "derived": {"op": "<ts_derive operation>"}, "acknowledged_warnings": [...],
   "precision_decimals"}. Statuses: validated_measurement
  (exact, computed evidence), approximate (state precision_decimals; visual evidence allowed), qualitative,
  observed_delay, physiological_timing, or insufficient_evidence / unresolved_conflict with a "reason".
  Each limitation: {"quantity": "<name>", "status": "insufficient_evidence", "reason": "<why>"}; free-text
  remarks go in "notes" (list of strings). A value is reported no finer than its evidence resolution.
- Do not report a quantity both as a supported claim and as a limitation or unresolved claim.
  Put caveats for a retained value in notes; to abstain on that quantity, withdraw its supported claim.
- Values read off images are observations, not measurements. Unknown units stay unknown. ts_derive creates
  traceable derived values (subtract, ratio, unit_convert, aggregate, rate_from_period: 60 / a period in s -> bpm);
  ts_artifact shows beat tables with exclusions.
  For a derived claim, include derived.op and cite the resulting ts_derive evidence_id, not just its inputs.
  If that record has no definition_id, use approximate with precision_decimals instead of inventing a definition.
- Your final ANSWER line is recorded but the application answer is what you submit with ts_submit.
"""
VERIFY_INSTRUCTIONS = """\
- ts_submit checks each claim against the evidence and returns deficiencies with the evidence they refer to. You
  choose how to repair: measure the right interval or method, cite other evidence, derive, qualify, or abstain.
  After reviewing reported deficiencies you may resubmit with "accept_partial": true to accept a partial answer;
  a submission is never final by itself. Repairs are limited; invalid submissions count.
"""
LOG_INSTRUCTIONS = """\
- ts_submit records your claims as the answer (the last well-formed submission counts). Submissions are limited;
  invalid submissions count.
"""
RW = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
RO = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


class RuntimeAdapter:
    def __init__(self, mode: str, root: str | Path | None = None, task_spec: str | Path | None = None,
                 run_id: str | None = None):  # fmt: skip
        if mode not in ("log", "verify"):
            raise ValueError("runtime mode must be 'log' or 'verify'")
        self.mode = mode
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run-{secrets.token_hex(4)}"
        self.ledger = EvidenceLedger(self.run_id, path=self.root / "ledger.jsonl" if self.root else None)
        self.artifacts = ArtifactStore(self.root / "artifacts" if self.root else None)
        self.spec = TaskSpec.from_dict(json.loads(Path(task_spec).read_text())) if task_spec else TaskSpec.minimal()
        self.gate = SubmissionGate(self.spec, self.ledger, mode=mode)
        self._calls = 0
        self._lock = threading.Lock()
        self._persist()  # finalization "incomplete" is written up front: no submission is an explicit state

    # ---- execution-layer hooks (the only creators of computed / visual evidence) ------------------------------------
    def _call_id(self) -> str:
        with self._lock:
            self._calls += 1
            return f"call{self._calls}"

    def on_measure(self, sid: str, out: dict, rows: list[dict], window, artifacts) -> dict:
        recs = measurement_records(self.ledger, sid, rows, window, self._call_id(), self.artifacts, artifacts)
        for row, rec in zip(rows, recs, strict=True):
            row["evidence_id"] = rec.evidence_id
            if rec.artifact_ref:
                row["artifact_ref"] = rec.artifact_ref
        self._noted([r.evidence_id for r in recs])
        return {**out, "evidence_ids": [r.evidence_id for r in recs]}

    def on_view(self, sid: str, view: str, channels: list[str], window, head: dict) -> dict:
        rec = view_record(self.ledger, sid, view, channels, window, self._call_id(), head.get("ref"))
        self._noted([rec.evidence_id])
        return {**head, "evidence_id": rec.evidence_id, "evidence_kind": "visual_observation"}

    def on_events(self, sid: str, res: dict, types: list[str], channels: list[str], window, skipped) -> dict:
        recs = events_records(self.ledger, sid, res["events"], types, channels, window, self._call_id(), skipped)
        self._noted([r.evidence_id for r in recs])
        return {**res, "evidence_ids": [r.evidence_id for r in recs]}

    def _noted(self, ids: list[str]) -> None:
        before = (self.gate.finalization, tuple(self.gate.flags), self.gate.pending)
        self.gate.note_evidence(ids)
        if (self.gate.finalization, tuple(self.gate.flags), self.gate.pending) != before:
            self._persist()

    def instructions(self) -> str:
        return SHARED_INSTRUCTIONS + (VERIFY_INSTRUCTIONS if self.mode == "verify" else LOG_INSTRUCTIONS)

    def final_answer(self) -> dict | None:
        return self.gate.answer

    def session_state(self) -> dict:
        return {"run_id": self.run_id, **self.gate.state()}

    def _persist(self) -> None:
        if not self.root:
            return
        (self.root / "bundles.json").write_text(json.dumps(self.ledger.bundles, indent=1, default=str))
        (self.root / "answer.json").write_text(json.dumps(self.session_state(), indent=1, default=str))

    # ---- model-facing tools (verify mode) ---------------------------------------------------------------------------
    def register(self, srv, text, guard) -> None:
        adapter = self

        @srv.tool(annotations=RW, structured_output=False)
        @guard
        def ts_submit(bundle: dict) -> list[TextContent]:
            """Submit structured claims (see the server instructions for the format). Returns `decision`:
            accepted | accepted_partial | retry | stopped (verify) or recorded | stopped (log), any deficiencies
            (code, claim_id, message, evidence_ids), the answer state (finalization, flags) and the remaining
            budget. `accept_partial: true` is honoured only for deficiencies reported in an earlier reply."""
            result = adapter.gate.submit(bundle)
            adapter._persist()
            return [text({"run_id": adapter.run_id, **result})]

        @srv.tool(annotations=RW, structured_output=False)
        @guard
        def ts_derive(
            op: str, inputs: list[str], to: str | None = None, column: str | None = None, rule: str | None = None
        ) -> list[TextContent]:
            """Create a derived evidence record from existing evidence with an allowlisted deterministic operation:
            subtract(a, b), ratio(a, b), unit_convert(a, to=unit), aggregate(beat_table, column=sbp|dbp|pp,
            rule=beat_weighted|time_weighted|median), rate_from_period(period_evidence) -> beats per minute
            (60 / period in s; e.g. a pulse rate from the `period` measure of a PPG channel). The result cites its
            parents and can be cited in claims."""
            if op not in DERIVED_OPS:
                raise SignalError(f"unknown derivation {op!r}; allowed: {', '.join(DERIVED_OPS)}")
            kw: dict = {}
            if op == "unit_convert":
                if not to:
                    raise SignalError("unit_convert needs `to`")
                kw["to"] = to
            if op == "aggregate":
                if len(inputs) != 1 or not column:
                    raise SignalError("aggregate needs one beat-table evidence id and a column")
                parent = adapter.ledger.get(inputs[0])
                if parent is None or not parent.artifact_ref:
                    raise SignalError("aggregate: the input must be evidence that carries a beat-table artifact")
                body = adapter.artifacts.get(parent.artifact_ref)
                kw.update(column=column, rule=rule or "beat_weighted", table_rows=body["rows"] if body else [])
            try:
                rec = adapter.ledger.derive(op, list(inputs), **kw)
            except ValueError as e:
                raise SignalError(str(e)) from None
            return [text({"run_id": adapter.run_id, "evidence": rec.summary()})]

        @srv.tool(annotations=RO, structured_output=False)
        @guard
        def ts_artifact(ref: str, start: int = 0, limit: int = 50, rejected_only: bool = False) -> list[TextContent]:
            """Rows of a stored table (e.g. the beat table behind abp_beats / abp_sbp). Rejected beats are returned
            with their reason; nothing is hidden. Use rejected_only=true to review exclusions."""
            page = adapter.artifacts.rows(ref, start, limit, rejected_only)
            if page is None:
                raise SignalError(f"unknown artifact {ref!r}")
            return [text({"run_id": adapter.run_id, **page, "summary": adapter.artifacts.summary(ref)})]

        @srv.tool(annotations=RO, structured_output=False)
        @guard
        def ts_evidence(id: str | None = None) -> list[TextContent]:
            """Summaries of the evidence recorded in this run (one record when `id` is given)."""
            if id:
                rec = adapter.ledger.get(id)
                if rec is None:
                    raise SignalError(f"unknown evidence {id!r}")
                return [text({"run_id": adapter.run_id, "evidence": rec.summary()})]
            recs = adapter.ledger.records()
            return [text({"run_id": adapter.run_id, "n": len(recs),
                          "evidence": [r.summary() for r in recs[-40:]], "truncated": len(recs) > 40})]  # fmt: skip
