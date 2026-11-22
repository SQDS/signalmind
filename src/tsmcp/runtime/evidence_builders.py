"""Turn actual tool outputs into evidence records. Shared by the offline executor and the MCP adapter.

Only these functions create `computed` / `visual_observation` records, and only from results the toolkit produced.
"""

from __future__ import annotations

from tsmcp.runtime.artifacts import ArtifactStore
from tsmcp.runtime.ledger import EvidenceLedger, EvidenceRecord, SourceKind
from tsmcp.runtime.timebase import SyncStatus, TimebaseInfo

GENERIC_DEFINITIONS = {
    "mean": "generic.mean.v1",
    "rms": "generic.rms.v1",
    "amplitude_pp": "generic.amplitude_pp.v1",
    "period": "generic.period.v1",
    "dominant_freq": "generic.dominant_freq.v1",
}


def definition_for(name: str, row: dict) -> str:
    if row.get("definition_id"):
        return row["definition_id"]
    if name in GENERIC_DEFINITIONS:
        return GENERIC_DEFINITIONS[name]
    return f"{name}.v1"  # plugin measures without an explicit definition keep a stable, name-based id


def single_channel_timebase(sid: str, ch: str) -> dict:
    return TimebaseInfo(
        identity=sid, sync_status=SyncStatus.UNKNOWN.value, basis="single channel", channels=(ch,)
    ).to_dict()


def measurement_records(
    ledger: EvidenceLedger,
    sid: str,
    rows: list[dict],
    window: tuple[float, float],
    tool_call_id: str,
    artifacts: ArtifactStore | None = None,
    artifact_rows: list[tuple[str, dict]] | None = None,
    tool: str = "ts_measure",
) -> list[EvidenceRecord]:
    """One computed record per measurement row (the rows are exactly what the tool returned to the model)."""
    out = []
    by_ref = {ref: art for ref, art in (artifact_rows or [])}
    for row in rows:
        name, ch = row["name"], row.get("ch", "")
        value = row.get("value")
        warnings = list(row.get("warnings", []))
        status = row.get("status", "ok" if value is not None else "limitation")
        note = row.get("note")
        art_ref = None
        art = by_ref.get(row.get("ref"))
        if artifacts is not None and art:
            art_ref = artifacts.put(art["kind"], art["columns"], art["rows"], art.get("meta"))
        counts = dict(row.get("counts", {}))
        if "n_beats" in row and "accepted" not in counts:
            counts["accepted"] = row["n_beats"]
        eff = row.get("intervals_effective") or [list(window)]
        parameters = {k: row[k] for k in ("method", "aggregation", "guidance", "value_units") if k in row}
        rec = ledger.record(
            SourceKind.COMPUTED,
            name,
            definition_id=definition_for(name, row),
            value=value,
            unit=row.get("unit"),
            signal_id=sid,
            signal_hash=sid,
            tool=tool,
            tool_call_id=tool_call_id,
            method=row.get("method"),
            method_version=row.get("method_version"),
            parameters=parameters,
            channels_requested=(ch,),
            channels_effective=tuple(row.get("channels_effective") or (ch,)),
            interval_requested=window,
            intervals_effective=tuple(tuple(i) for i in eff),
            intervals_excluded=tuple(tuple(i) for i in row.get("intervals_excluded", [])),
            gaps=tuple(tuple(i) for i in row.get("gaps", [])),
            counts=counts,
            coverage=row.get("coverage"),
            quality=dict(row.get("quality", {})),
            timebase=row.get("timebase") or single_channel_timebase(sid, ch),
            warnings=warnings,
            resolution=row.get("resolution") or {"known": False},
            artifact_ref=art_ref,
            ref=row.get("ref"),
            status=status,
            note=note,
        )
        out.append(rec)
    return out


def view_record(
    ledger: EvidenceLedger,
    sid: str,
    view: str,
    channels: list[str],
    window: tuple[float, float],
    tool_call_id: str,
    ref: str | None,
    tool: str = "ts_zoom",
) -> EvidenceRecord:
    """An image was rendered for these channels over this window. No number was computed."""
    return ledger.record(
        SourceKind.VISUAL_OBSERVATION,
        "view",
        definition_id="view.v1",
        signal_id=sid,
        signal_hash=sid,
        tool=tool,
        tool_call_id=tool_call_id,
        method=f"view:{view}",
        channels_requested=tuple(channels),
        channels_effective=tuple(channels),
        interval_requested=window,
        intervals_effective=(window,),
        ref=ref,
        timebase=TimebaseInfo(
            identity=sid,
            sync_status=SyncStatus.INDEX_ALIGNED_DECLARED.value if len(channels) > 1 else SyncStatus.UNKNOWN.value,
            basis="channels drawn on one time axis of the same record",
            channels=tuple(channels),
        ).to_dict(),
        resolution={"known": False, "note": "visual; no numeric resolution"},
    )


def events_records(
    ledger: EvidenceLedger,
    sid: str,
    events: list[dict],
    types: list[str],
    channels: list[str],
    window: tuple[float, float],
    tool_call_id: str,
    skipped: list[str] | None = None,
    tool: str = "ts_find_events",
) -> list[EvidenceRecord]:
    """One computed record per (type, channel): value = number of candidates; caveats become warnings."""
    out = []
    for typ in types:
        for ch in channels:
            evs = [e for e in events if e["type"] == typ and e.get("ch") == ch]
            caveats = sorted({e["caveat"] for e in evs if e.get("caveat")})
            warnings = [{"code": "detector_caveat", "message": cv, "severity": "info"} for cv in caveats]
            if skipped and any(s.startswith(f"{typ} on {ch}") for s in skipped):
                warnings.append({"code": "detector_skipped", "message": "; ".join(skipped), "severity": "warning"})
            out.append(
                ledger.record(
                    SourceKind.COMPUTED,
                    f"events.{typ}",
                    definition_id="events.v1",
                    value=len(evs),
                    unit="count",
                    signal_id=sid,
                    signal_hash=sid,
                    tool=tool,
                    tool_call_id=tool_call_id,
                    method=f"detector:{typ}",
                    parameters={
                        "candidates": [
                            {"t": e["t"], "dur": e.get("dur"), "score": e.get("score"), "ref": e.get("ref")}
                            for e in evs
                        ]
                    },
                    channels_requested=(ch,),
                    channels_effective=(ch,),
                    interval_requested=window,
                    intervals_effective=(window,),
                    warnings=warnings,
                    timebase=single_channel_timebase(sid, ch),
                    resolution={
                        "known": True,
                        "value": max((e.get("resolution_s", 0) or 0) for e in evs) if evs else 0.0,
                        "kind": "detector_time_resolution",
                    },
                )
            )
    return out
