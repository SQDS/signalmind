"""Append-only evidence ledger: trusted measurement receipts created by the execution layer, never by the model.

A record says what was computed, observed or assumed, on which channels and effective intervals, with which method,
warnings and resolution. `computed` records need a tool call; `derived` records come only from the allowlisted
deterministic operations in `derive()`; a correction is a new record linked by `supersedes`.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from tsmcp.runtime.precision import present
from tsmcp.runtime.units import compatible, convert, family


class SourceKind(StrEnum):
    COMPUTED = "computed"  # produced by a tool / measurement code path (trusted execution)
    DERIVED = "derived"  # allowlisted deterministic operation on existing records
    VISUAL_OBSERVATION = "visual_observation"  # an image was rendered and looked at; no number was computed
    IMPORTED = "imported"  # supplied metadata (e.g. declared sampling rate, declared synchrony)
    HYPOTHESIS = "hypothesis"  # the model's stated assumption; never supports a validated measurement


@dataclass(frozen=True)
class Warning:
    code: str
    message: str
    severity: str = "warning"  # info | warning | error

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    run_id: str
    seq: int
    source_kind: str
    quantity: str
    definition_id: str | None = None
    value: float | int | str | None = None
    unit: str | None = None
    signal_id: str | None = None
    signal_hash: str | None = None
    tool: str | None = None
    tool_call_id: str | None = None
    method: str | None = None
    method_version: str | None = None
    parameters: dict = field(default_factory=dict)
    channels_requested: tuple[str, ...] = ()
    channels_effective: tuple[str, ...] = ()
    interval_requested: tuple[float, float] | None = None
    intervals_effective: tuple[tuple[float, float], ...] = ()
    intervals_excluded: tuple[tuple[float, float], ...] = ()
    gaps: tuple[tuple[float, float], ...] = ()
    counts: dict = field(default_factory=dict)  # accepted, rejected, rejection_reasons{reason: n}
    coverage: float | None = None  # fraction of the requested interval covered by accepted data
    quality: dict = field(default_factory=dict)
    timebase: dict = field(default_factory=dict)  # TimebaseInfo.to_dict()
    warnings: tuple[Warning, ...] = ()
    resolution: dict = field(default_factory=lambda: {"known": False})  # {"known": True, "value": x, "kind": "..."}
    artifact_ref: str | None = None
    parents: tuple[str, ...] = ()
    supersedes: str | None = None
    ref: str | None = None  # tsmcp self-describing ref, for citation compatibility
    status: str = "ok"  # ok | limitation (method could not produce a defensible value)
    note: str | None = None
    fingerprint: str = ""

    @property
    def numeric(self) -> float | None:
        v = self.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return float(v) if math.isfinite(float(v)) else None

    def warning_codes(self, min_severity: str = "warning") -> list[str]:
        order = {"info": 0, "warning": 1, "error": 2}
        return [w.code for w in self.warnings if order.get(w.severity, 1) >= order[min_severity]]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["warnings"] = [w.to_dict() for w in self.warnings]
        return d

    def summary(self) -> dict:
        """What the model sees: compact, with exclusions and warnings visible."""
        return {
            "evidence_id": self.evidence_id,
            "source_kind": self.source_kind,
            "quantity": self.quantity,
            "definition_id": self.definition_id,
            "value": self.value,
            "unit": self.unit,
            "channels": list(self.channels_effective),
            "intervals_s": [list(i) for i in self.intervals_effective],
            "excluded_s": [list(i) for i in self.intervals_excluded],
            "counts": self.counts,
            "coverage": self.coverage,
            "warnings": [w.to_dict() for w in self.warnings],
            "resolution": self.resolution,
            "timebase": self.timebase,
            "artifact_ref": self.artifact_ref,
            "ref": self.ref,
            "status": self.status,
            "note": self.note,
        }


def _fingerprint(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


class EvidenceLedger:
    def __init__(self, run_id: str, path: Path | None = None):
        self.run_id = run_id
        self._records: list[EvidenceRecord] = []
        self._by_id: dict[str, EvidenceRecord] = {}
        self._by_ref: dict[str, EvidenceRecord] = {}
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        self.bundles: list[dict] = []  # submitted bundles and their reports, in order (append-only too)

    # ---- creation -------------------------------------------------------------------------------------------------
    def record(self, source_kind: SourceKind | str, quantity: str, **fields) -> EvidenceRecord:
        kind = SourceKind(source_kind)
        if kind is SourceKind.COMPUTED and not (fields.get("tool") and fields.get("tool_call_id")):
            raise ValueError("computed evidence needs tool and tool_call_id: only the execution layer creates it")
        if kind is SourceKind.DERIVED and "derived_op" not in fields.get("parameters", {}):
            raise ValueError("derived evidence is created through EvidenceLedger.derive()")
        warnings = tuple(w if isinstance(w, Warning) else Warning(**w) for w in fields.pop("warnings", ()))
        with self._lock:
            seq = len(self._records) + 1
            eid = f"ev{seq}"
            body = dict(
                evidence_id=eid,
                run_id=self.run_id,
                seq=seq,
                source_kind=kind.value,
                quantity=quantity,
                warnings=warnings,
                **_normalise(fields),
            )
            rec = EvidenceRecord(
                **body,
                fingerprint=_fingerprint(
                    {k: v for k, v in body.items() if k != "warnings"} | {"warnings": [w.to_dict() for w in warnings]}
                ),
            )
            self._records.append(rec)
            self._by_id[eid] = rec
            if rec.ref:
                self._by_ref.setdefault(rec.ref, rec)
            if self._path:
                with open(self._path, "a") as f:
                    f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        return rec

    def derive(self, op: str, parents: list[str], **kw) -> EvidenceRecord:
        """Allowlisted deterministic derivations. The result cites its parents; nothing is overwritten."""
        recs = [self.get(p) for p in parents]
        if any(r is None for r in recs):
            raise ValueError("derive: unknown parent evidence")
        if op not in DERIVED_OPS:
            raise ValueError(f"derive: operation {op!r} is not allowlisted ({sorted(DERIVED_OPS)})")
        value, unit, detail = DERIVED_OPS[op](recs, **kw)
        chans = tuple(sorted({c for r in recs for c in r.channels_effective}))
        ints = tuple(i for r in recs for i in r.intervals_effective)
        warnings = tuple(w for r in recs for w in r.warnings)
        res = {"known": False}
        known = [r.resolution.get("value") for r in recs if r.resolution.get("known") and r.resolution.get("value")]
        if known and op in ("subtract", "unit_convert"):
            res = {"known": True, "value": max(known) if op == "subtract" else known[0], "kind": "propagated_max"}
        res = detail.pop("resolution", res)
        exact = value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = present(value, res)  # the same presentation rule as ts_measure: no invented precision
        return self.record(
            SourceKind.DERIVED,
            detail.pop("quantity"),
            value=value,
            unit=unit,
            definition_id=detail.pop("definition_id", None),
            parameters={"derived_op": op, **kw, **detail, "value_exact": exact},
            parents=tuple(p for p in parents),
            channels_effective=chans,
            intervals_effective=ints,
            warnings=warnings,
            resolution=res,
            signal_id=recs[0].signal_id,
            timebase=recs[0].timebase,
            method=f"derived:{op}",
        )

    def record_bundle(self, bundle: dict, report: dict) -> None:
        with self._lock:
            self.bundles.append({"seq": len(self.bundles) + 1, "bundle": bundle, "report": report})

    # ---- access -----------------------------------------------------------------------------------------------------
    def get(self, key: str) -> EvidenceRecord | None:
        hit = self._by_id.get(key) or self._by_ref.get(key)
        if hit is None and isinstance(key, str) and key.startswith("art-"):  # an artifact ref names its latest record
            hit = next((r for r in reversed(self._records) if r.artifact_ref == key), None)
        return hit

    def records(self) -> list[EvidenceRecord]:
        return list(self._records)

    def latest_for(self, definition_id: str) -> list[EvidenceRecord]:
        """Records with this definition that have not been superseded."""
        superseded = {r.supersedes for r in self._records if r.supersedes}
        return [r for r in self._records if r.definition_id == definition_id and r.evidence_id not in superseded]

    def __len__(self) -> int:
        return len(self._records)

    def fingerprint(self) -> str:
        return _fingerprint([r.fingerprint for r in self._records])


def _normalise(fields: dict) -> dict:
    out = dict(fields)
    for k in ("channels_requested", "channels_effective", "parents"):
        if k in out and out[k] is not None:
            out[k] = tuple(str(c) for c in out[k])
    for k in ("intervals_effective", "intervals_excluded", "gaps"):
        if k in out and out[k] is not None:
            out[k] = tuple((float(a), float(b)) for a, b in out[k])
    if out.get("interval_requested") is not None:
        a, b = out["interval_requested"]
        out["interval_requested"] = (float(a), float(b))
    return out


# ---- allowlisted derived operations ----------------------------------------------------------------------------------
def _num(r: EvidenceRecord) -> float:
    v = r.numeric
    if v is None:
        raise ValueError(f"{r.evidence_id} has no finite numeric value")
    return v


def _subtract(recs, **kw):
    if len(recs) != 2:
        raise ValueError("subtract needs exactly two parents (a - b)")
    a, b = recs
    if not compatible(a.unit, b.unit):
        raise ValueError(f"subtract: incompatible units {a.unit!r} and {b.unit!r}")
    detail = {"quantity": f"{a.quantity}_minus_{b.quantity}"}
    if {a.definition_id, b.definition_id} <= {"abp.sbp.beat_mean.v1", "abp.sbp.time_weighted_mean.v1"}:
        detail["definition_id"] = "abp.sbp.beat_mean_change.v1"
    return _num(a) - convert(_num(b), b.unit, a.unit), a.unit, detail


def _ratio(recs, **kw):
    if len(recs) != 2:
        raise ValueError("ratio needs exactly two parents (a / b)")
    a, b = recs
    d = _num(b)
    if d == 0:
        raise ValueError("ratio: division by zero")
    unit = "" if a.unit == b.unit else f"{a.unit}/{b.unit}"
    return _num(a) / d, unit, {"quantity": f"{a.quantity}_over_{b.quantity}"}


def _unit_convert(recs, to: str = "", **kw):
    if len(recs) != 1:
        raise ValueError("unit_convert needs one parent")
    (a,) = recs
    if family(a.unit) is None:
        raise ValueError(f"unit_convert: source unit {a.unit!r} is unknown; cannot convert")
    return convert(_num(a), a.unit, to), to, {"quantity": a.quantity, "definition_id": a.definition_id}


def _aggregate(recs, column: str = "", rule: str = "beat_weighted", table_rows: list | None = None, **kw):
    """Aggregate a column of a beat table that the parent evidence references. The caller passes the rows it
    retrieved from the artifact store (the ledger holds no arrays); accepted rows only, a stated rule."""
    if len(recs) != 1 or table_rows is None or not column:
        raise ValueError("aggregate needs one beat-table parent, its rows and a column")
    (a,) = recs
    rows = [r for r in table_rows if r.get("accepted") is True and r.get(column) is not None]
    if not rows:
        raise ValueError("aggregate: no accepted rows")
    vals = [float(r[column]) for r in rows]
    if rule == "beat_weighted":
        v = sum(vals) / len(vals)
    elif rule == "time_weighted":
        w = [float(r.get("duration_s") or 0) for r in rows]
        if sum(w) <= 0:
            raise ValueError("aggregate: time_weighted needs duration_s on every row")
        v = sum(x * y for x, y in zip(vals, w, strict=True)) / sum(w)
    elif rule == "median":
        s = sorted(vals)
        v = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    else:
        raise ValueError(f"aggregate: unknown rule {rule!r}")
    unit = a.parameters.get("value_units", {}).get(column, a.unit)
    resolution = {"known": False}
    if rule in ("beat_weighted", "time_weighted") and len(vals) > 1:
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        resolution = {"known": True, "value": sd / len(vals) ** 0.5, "kind": "standard_error_of_mean"}
    did = {
        ("sbp", "time_weighted"): "abp.sbp.time_weighted_mean.v1",
        ("sbp", "beat_weighted"): "abp.sbp.beat_mean.v1",
        ("dbp", "beat_weighted"): "abp.dbp.beat_mean.v1",
        ("pp", "beat_weighted"): "abp.pp.beat_mean.v1",
    }.get((column, rule))
    return (
        v,
        unit,
        {"quantity": f"{column}_{rule}", "definition_id": did, "n_rows": len(rows), "resolution": resolution},
    )


def _rate_from_period(recs, **kw):
    """Beats per minute from a period in seconds (60 / T); the generic `period` measure is the usual parent."""
    if len(recs) != 1:
        raise ValueError("rate_from_period needs one parent (a period in s)")
    (a,) = recs
    if a.unit != "s":
        raise ValueError(f"rate_from_period: parent unit must be s, not {a.unit!r}")
    t = _num(a)
    if t <= 0:
        raise ValueError("rate_from_period: period must be positive")
    rate = 60.0 / t
    res = {"known": False}
    if a.resolution.get("known") and a.resolution.get("value"):
        res = {"known": True, "value": 60.0 * float(a.resolution["value"]) / t**2, "kind": "propagated"}
    return rate, "bpm", {"quantity": "rate_from_period", "definition_id": "rate.from_period.v1", "resolution": res}


DERIVED_OPS = {
    "subtract": _subtract,
    "ratio": _ratio,
    "unit_convert": _unit_convert,
    "aggregate": _aggregate,
    "rate_from_period": _rate_from_period,
}
