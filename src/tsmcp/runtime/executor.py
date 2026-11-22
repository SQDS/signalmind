"""In-process tool executor over tsmcp.core (no MCP): the execution layer for offline runs, tests and the demo.

It exposes the same tool names and evidence semantics as the MCP adapter, so a scripted driver can be replayed
against either. Images are not rendered here; a ts_zoom call still registers a visual-observation record.
"""

from __future__ import annotations

from pathlib import Path

from tsmcp.core.compact import sig, tround
from tsmcp.core.errors import SignalError
from tsmcp.core.evidence import ref
from tsmcp.core.measure import measure
from tsmcp.core.signal import clip_window, slice_channel, time_decimals
from tsmcp.core.store import SignalStore
from tsmcp.core.views.describe import describe
from tsmcp.plugins import all_detectors, all_measures
from tsmcp.plugins.optional import plugin_set, pop_artifacts, register_signal
from tsmcp.runtime.artifacts import ArtifactStore
from tsmcp.runtime.controller import ToolResult
from tsmcp.runtime.evidence_builders import events_records, measurement_records, view_record
from tsmcp.runtime.ledger import DERIVED_OPS, EvidenceLedger


class CoreExecutor:
    TOOLS = (
        "ts_load",
        "ts_describe",
        "ts_measure",
        "ts_zoom",
        "ts_find_events",
        "ts_derive",
        "ts_artifact",
        "ts_evidence",
    )

    def __init__(self, store: SignalStore, ledger: EvidenceLedger, artifacts: ArtifactStore | None = None,
                 plugins: tuple[str, ...] = ("pressure",), root: Path | None = None):  # fmt: skip
        self.store, self.ledger = store, ledger
        self.artifacts = artifacts or ArtifactStore(root / "artifacts" if root else None)
        pl = plugin_set(plugins)
        self.measures, self.detectors = all_measures(pl), all_detectors(pl)

    def call(self, name: str, args: dict, call_id: str) -> ToolResult:
        fn = getattr(self, f"_{name}", None)
        if fn is None:
            return ToolResult({}, [], error=f"unknown tool {name!r}; available: {', '.join(self.TOOLS)}")
        try:
            return fn(call_id, **args)
        except SignalError as e:
            return ToolResult({}, [], error=str(e))
        except TypeError as e:
            return ToolResult({}, [], error=f"bad arguments for {name}: {e}")

    # ---- tools ----------------------------------------------------------------------------------------------
    def _ts_load(self, call_id, source=None, data=None, fs=None, format=None, channels=None, units=None):
        s = self.store.load(source=source, data=data, fs=fs, format=format, channels=channels, units=units)
        return ToolResult(describe(s), [])

    def _ts_describe(self, call_id, id, detail="normal"):
        return ToolResult(describe(self.store.get(id), detail), [])

    def _ts_measure(self, call_id, id, what, window=None, channels=None):
        s = self.store.get(id)
        t0, t1, warn = clip_window(s, window)
        chans = s.select(channels)
        results, artifacts = [], []
        register_signal(s)
        for c in chans:
            x = slice_channel(c, t0, t1)
            start = max(0, int(t0 * c.fs) // 1) / c.fs
            rows = measure(id, c, x, start, t1, what, time_decimals(c.fs), self.measures)
            pop_artifacts(rows, artifacts)
            results += rows
        recs = measurement_records(self.ledger, id, results, (t0, t1), call_id, self.artifacts, artifacts)
        for row, rec in zip(results, recs, strict=True):
            row["evidence_id"] = rec.evidence_id
            if rec.artifact_ref:
                row["artifact_ref"] = rec.artifact_ref
        tdec = time_decimals(max(c.fs for c in chans))
        out = {"id": id, "window": [tround(t0, tdec), tround(t1, tdec)], "warn": warn, "measurements": results,
               "evidence_ids": [r.evidence_id for r in recs]}  # fmt: skip
        return ToolResult(out, recs)

    def _ts_zoom(self, call_id, id, window, channels=None, view="wave", detail="normal", rows=None):
        s = self.store.get(id)
        t0, t1, warn = clip_window(s, window)
        chans = s.select(channels)
        tdec = time_decimals(max(c.fs for c in chans))
        win = f"{tround(t0, tdec):g}-{tround(t1, tdec):g}"
        head = {"id": id, "view": view, "window": [tround(t0, tdec), tround(t1, tdec)],
                "channels": [c.name for c in chans], "ref": ref(id, f"zoom.{view}", win), "warn": warn,
                "summary": [{"ch": c.name, "min": sig(float(slice_channel(c, t0, t1).min())),
                             "max": sig(float(slice_channel(c, t0, t1).max()))} for c in chans[:8]]}  # fmt: skip
        rec = view_record(self.ledger, id, view, [c.name for c in chans], (t0, t1), call_id, head["ref"])
        return ToolResult({**head, "evidence_id": rec.evidence_id, "evidence_kind": "visual_observation"}, [rec])

    def _ts_find_events(self, call_id, id, types, window=None, channels=None, top_k=15):
        unknown = [t for t in types if t not in self.detectors]
        if unknown:
            raise SignalError(f"unknown event type(s) {unknown}")
        s = self.store.get(id)
        t0, t1, warn = clip_window(s, window)
        chans = s.select(channels) if channels else s.channels[:1]
        found, skipped = [], []
        for c in chans:
            x = slice_channel(c, t0, t1)
            start = max(0, int(t0 * c.fs)) / c.fs
            for typ in types:
                try:
                    for e in self.detectors[typ](x, c.fs, c.unit, start, ch=c):
                        found.append({**e, "ch": c.name})
                except SignalError as err:
                    skipped.append(f"{typ} on {c.name}: {err}")
        keep = sorted(found, key=lambda e: -e.get("score", 0))[: max(1, top_k)]
        keep.sort(key=lambda e: e["t"])
        tdec = time_decimals(max(c.fs for c in chans))
        events = [
            {**e, "t": tround(e["t"], tdec), "ref": ref(id, f"ev.{e['type']}", f"{tround(e['t'], tdec):g}")}
            for e in keep
        ]
        recs = events_records(self.ledger, id, events, types, [c.name for c in chans], (t0, t1), call_id, skipped)
        out = {"id": id, "window": [tround(t0, tdec), tround(t1, tdec)], "events": events, "warn": warn,
               "skipped": skipped, "evidence_ids": [r.evidence_id for r in recs]}  # fmt: skip
        return ToolResult(out, recs)

    def _ts_derive(self, call_id, op, inputs, to=None, column=None, rule=None):
        if op not in DERIVED_OPS:
            raise SignalError(f"unknown derivation {op!r}")
        kw: dict = {}
        if op == "unit_convert":
            kw["to"] = to
        if op == "aggregate":
            parent = self.ledger.get(inputs[0]) if inputs else None
            if parent is None or not parent.artifact_ref:
                raise SignalError("aggregate: the input must be evidence that carries a beat-table artifact")
            body = self.artifacts.get(parent.artifact_ref)
            kw.update(column=column, rule=rule or "beat_weighted", table_rows=body["rows"] if body else [])
        try:
            rec = self.ledger.derive(op, list(inputs), **kw)
        except ValueError as e:
            raise SignalError(str(e)) from None
        return ToolResult({"evidence": rec.summary()}, [rec])

    def _ts_artifact(self, call_id, ref, start=0, limit=50, rejected_only=False):
        page = self.artifacts.rows(ref, start, limit, rejected_only)
        if page is None:
            raise SignalError(f"unknown artifact {ref!r}")
        return ToolResult({**page, "summary": self.artifacts.summary(ref)}, [])

    def _ts_evidence(self, call_id, id=None):
        if id:
            rec = self.ledger.get(id)
            if rec is None:
                raise SignalError(f"unknown evidence {id!r}")
            return ToolResult({"evidence": rec.summary()}, [])
        return ToolResult({"evidence": [r.summary() for r in self.ledger.records()]}, [])
