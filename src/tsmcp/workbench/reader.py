"""Read historical runs without invoking today's cleaning controller or DSP."""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import changed, digest, read, sha, time_support
from tsmcp.cleaning.provenance import restore_steps
from tsmcp.processing import intervals


@lru_cache(maxsize=32)
def _cached_json(path, stamp):
    return read(path)


def document(path, default=None):
    path = Path(path)
    return _cached_json(str(path), path.stat().st_mtime_ns) if path.exists() else default


@lru_cache(maxsize=256)
def _verify(path, stamp, size, expected):
    if sha(path) != expected:
        raise ValueError("Saved artifact changed; its recorded measurements no longer describe this file")


def verified(path, expected):
    stat = Path(path).stat()
    _verify(str(path), stat.st_mtime_ns, stat.st_size, expected)


def lines(path):
    if not path.exists():
        return []
    out = []
    with path.open() as stream:
        for line in stream:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A writer can be between write and flush; never manufacture an event.
                break
    return out


def public_result(event):
    for block in event.get("result", []):
        if block.get("type") == "text":
            try:
                value = json.loads(block["text"])
                if isinstance(value, dict):
                    return value
            except (ValueError, KeyError):
                pass
    return {}


class Reader:
    def __init__(self, store):
        self.store = store
        self.resources = {}

    def resource(self, path, name=None):
        path = Path(path).resolve()
        token = digest(str(path))[:32]
        self.resources[token] = path
        return {"id": token, "name": name or path.name, "url": f"/api/resources/{token}"}

    def manifest(self, rid):
        return document(
            self.store.run_path(rid) / "dataset_manifest.json", {"cases": [], "patients": None, "records": []}
        )

    def state(self, rid):
        return document(self.store.run_path(rid) / "agent_state.json", {})

    def version(self, rid, vid):
        if not vid.startswith("v_") or "/" in vid or ".." in vid:
            raise ValueError("Invalid version")
        result = document(self.store.run_path(rid) / "versions" / vid / "version.json")
        if result is None:
            raise KeyError("Unknown version")
        return result

    def events(self, rid):
        root = self.store.run_path(rid)
        native = lines(root / "events.jsonl")
        if native:
            rows = []
            for i, event in enumerate(native):
                row = {**event, "sequence": i, "recording": "native"}
                if event.get("job_id") and event.get("attempt_id"):
                    folder = root / "jobs" / event["job_id"] / "attempts" / event["attempt_id"]
                    row["logs"] = [
                        self.resource(folder / name)
                        for name in ("execution.json", "stdout.txt", "stderr.txt")
                        if (folder / name).is_file()
                    ]
                    if (folder / "stderr.txt").is_file():
                        row["stderr_preview"] = (folder / "stderr.txt").read_text(errors="replace")[:12000]
                rows.append(row)
            return rows
        events = []
        for i, event in enumerate(lines(root / "agent_activity.jsonl")):
            events.append(
                {
                    **event,
                    "sequence": i,
                    "event_id": event.get("tool_use_id", f"legacy-{i}"),
                    "type": "tool_failed" if event.get("is_error") else "tool_completed",
                    "timestamp": None,
                    "actor": event.get("origin", "model"),
                    "recording": "legacy",
                }
            )
        # Some invalid drafts only survived as an assistant's public tool arguments.
        recorded = {e.get("tool_use_id") for e in events}
        for message in document(root / "private/messages.json", []):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", [])
            for b in content if isinstance(content, list) else []:
                if b.get("type") == "tool_use" and b.get("id") not in recorded:
                    events.append(
                        {
                            "sequence": len(events),
                            "event_id": b.get("id"),
                            "timestamp": None,
                            "type": "tool_submitted_outcome_unknown",
                            "actor": "model",
                            "tool": b.get("name"),
                            "arguments": b.get("input", {}),
                            "recording": "recovered_public_tool_arguments",
                        }
                    )
        return events

    def versions(self, rid):
        root, state = self.store.run_path(rid), self.state(rid)
        reviews = {}
        for e in self.events(rid):
            if e.get("tool") == "submit_self_review" and e["type"] == "tool_completed" and not e.get("is_error"):
                r = e.get("arguments", {})
                reviews.setdefault(r.get("version_id"), []).append(r)
        result = []
        for p in sorted((root / "versions").glob("*/version.json")):
            v = document(p)
            result.append({k: v[k] for k in ("version_id", "parent_version", "recipe", "stage", "report", "elapsed_s")})
            result[-1].update(
                case_count=len(v["case_ids"]),
                reviews=reviews.get(v["version_id"], []),
                selected=v["version_id"] in state.get("accepted_tasks", {}).values(),
                implementation_hash=v["implementation_hash"],
            )
        for v in result:
            v["pilot_versions"] = [
                p["version_id"]
                for p in result
                if v["stage"] == "batch"
                and p["stage"] == "pilot"
                and p["recipe"]["hash"] == v["recipe"]["hash"]
                and p["parent_version"] == v["parent_version"]
                and set(p["report"]["tasks"]) == set(v["report"]["tasks"])
            ]
        return result

    def summary(self, rid):
        root, manifest, state = self.store.run_path(rid), self.manifest(rid), self.state(rid)
        job = document(root / "run.json", {})
        if job.get("status") == "running" and (root / "stop.requested").exists():
            job = {**job, "status": "stopping"}
        report = document(root / "export/evaluation_report.json", {})
        budget = document(root / "api_budget.json", {})
        selection = document(root / "private/selection.json", {})
        baseline_reports = document(root / "export/baseline_reports.json", {})
        return {
            "id": rid,
            "name": job.get("name", rid),
            "job": job,
            "status": job.get("status", report.get("status", "historical")),
            "historical": not bool(job),
            "patients": manifest.get("patients"),
            "patient_keys": selection.get("patients", manifest.get("patient_keys", [])),
            "records": len(manifest.get("records", [])),
            "cases": len(manifest["cases"]),
            "tasks": report.get("tasks", {}),
            "task_versions": state.get("accepted_tasks", {}),
            "scope_hash": manifest.get("scope_hash"),
            "report": report,
            "baseline_reports": baseline_reports,
            "enabled_tasks": (manifest.get("task_config") or {}).get("tasks", job.get("config", {}).get("tasks", [])),
            "unavailable_tasks": manifest.get("unavailable_tasks", []),
            "audit_failures": manifest.get("audit_failures", []),
            "budget": {
                k: budget.get(k)
                for k in ("estimated_peak_usd", "unknown_charge_reserve_usd", "cap_usd", "actual_billed_usd")
            },
            "superseded": document(root / "post_run_status.json"),
            "reference_metrics": report.get("reference_metrics", {"Y": None, "C": None, "reference_verified": False}),
        }

    def overview(self):
        items, keys, call_ids, cost, unknown = [], set(), set(), 0.0, 0.0
        for row in self.store.runs():
            s = self.summary(row["id"])
            keys.update(s.pop("patient_keys"))
            budget = document(Path(row["root"]) / "api_budget.json", {})
            # Full ledgers can inherit the same requests across run revisions.
            for c in budget.get("calls", []):
                cid = c.get("request_id") or digest([c.get("number"), c.get("request_hash"), c.get("attempt")])
                if cid not in call_ids:
                    call_ids.add(cid)
                    cost += c.get("estimated_peak_usd", 0)
                    if c.get("status") != "returned":
                        unknown += c.get("reserve_usd", 0)
            items.append(s)
        # Old attempt increments are explicit evidence, not the sum of cumulative ledgers.
        attempts = document(self.store.repo / "eval/results/vital-cleaning-v1-20261112/attempts.json", [])
        increments = {a["run"]: a["incremental_cost"] for a in attempts}
        for s in items:
            s["incremental_cost_usd"] = increments.get(s["id"], s["budget"].get("estimated_peak_usd"))
        return {
            "runs": items,
            "unique_patients": len(keys) if keys else None,
            "estimated_cost_usd": cost,
            "unknown_charge_reserve_usd": unknown,
            "actual_billed_usd": None,
        }

    def chosen(self, rid, cid, vid=None):
        state = self.state(rid)
        if not vid or vid == "final":
            vid = state.get("accepted", {}).get(cid, {}).get("version_id")
            if not vid:
                vid = document(self.store.run_path(rid) / "baselines.json", {}).get("identity")
        if vid == "raw" or not vid:
            return None, None
        version = self.version(rid, vid)
        return vid, version["results"].get(cid)

    def cases(self, rid, *, task="", action="all", search="", version=None, offset=0, limit=80):
        rows = []
        state = self.state(rid)
        counts = {"all": 0, "changed": 0, "value": 0, "mask": 0, "keep": 0}
        for c in self.manifest(rid)["cases"]:
            if task and c["task"] != task:
                continue
            if search and search.lower() not in (c["id"] + c["record_id"]).lower():
                continue
            vid, a = self.chosen(rid, c["id"], version)
            if version and version not in ("final", "raw") and not a:
                continue
            stats = a["stats"] if a else {}
            value, mask = stats.get("modified_samples", 0) > 0, stats.get("support_changed_samples", 0) > 0
            kind = "mixed" if value and mask else "value" if value else "mask" if mask else "keep"
            counts["all"] += 1
            counts["changed" if value or mask else "keep"] += 1
            counts["value"] += int(value)
            counts["mask"] += int(mask)
            if (
                action == "changed"
                and not (value or mask)
                or action == "value"
                and not value
                or action == "mask"
                and not mask
                or action == "keep"
                and (value or mask)
            ):
                continue
            rows.append(
                {
                    **c,
                    "action": kind,
                    "version_id": vid,
                    "stats": stats,
                    "accepted": c["id"] in state.get("accepted", {}),
                }
            )
        return {
            "total": len(rows),
            "counts": counts,
            "offset": offset,
            "items": rows[offset : offset + min(limit, 300)],
        }

    def case(self, rid, cid):
        for c in self.manifest(rid)["cases"]:
            if c["id"] == cid:
                return c
        raise KeyError("Unknown case")

    def arrays(self, rid, cid, vid=None, step=None):
        root, case = self.store.run_path(rid), self.case(rid, cid)
        verified(root / "public" / cid / "input.npz", case["input_sha256"])
        with np.load(root / "public" / cid / "input.npz", allow_pickle=False) as z:
            raw, times = z["values"].copy(), z["time"].copy()
            valid = z["valid"].astype(bool) if "valid" in z else np.isfinite(raw)
            estimated = z["estimated"].astype(bool) if "estimated" in z else np.zeros(len(raw), bool)
        valid &= time_support(case)
        source_valid = valid.copy()
        selected, a = self.chosen(rid, cid, vid)
        if selected and a is None:
            raise ValueError("This candidate did not execute the selected case")
        values = raw.copy()
        if a and step is not None:
            if not a.get("operator_trace"):
                raise ValueError("This historical intermediate waveform was not recorded")
            parent = self.version(rid, selected)["parent_version"]
            _, _, values, valid, estimated, _, _ = self.arrays(rid, cid, parent or "raw")
            values, valid, estimated = restore_steps(root, a["operator_trace"], values, valid, estimated, step)
        elif a and a.get("arrays"):
            verified(root / a["arrays"], a["arrays_sha256"])
            with np.load(root / a["arrays"], allow_pickle=False) as z:
                values, valid, estimated = z["values"].copy(), z["valid"].copy(), z["estimated"].copy()
        return raw, times, values, valid, estimated, source_valid, a

    def waveform(self, rid, cid, *, version=None, start=0, stop=None, points=2400, step=None):
        case = self.case(rid, cid)
        raw, times, values, valid, estimated, source_valid, artifact = self.arrays(rid, cid, version, step)
        stop = min(len(raw), stop if stop is not None else len(raw))
        start = max(0, min(start, stop - 1))
        points = max(50, min(points, 100000))
        stride = max(1, int(np.ceil((stop - start) / max(1, points // 8))))
        indices = set()
        for a in range(start, stop, stride):
            b = min(stop, a + stride)
            indices.update((a, b - 1))
            for x in (raw, values):
                finite = np.flatnonzero(np.isfinite(x[a:b]))
                if finite.size:
                    part = x[a:b][finite]
                    indices.update((a + int(finite[part.argmin()]), a + int(finite[part.argmax()])))
        # Keep every discontinuity visible even when overview bins are wide.
        for mask in (valid, source_valid, estimated, np.isfinite(raw), np.isfinite(values)):
            edges = np.flatnonzero(mask[start + 1 : stop] != mask[start : stop - 1]) + start + 1
            for i in edges:
                indices.update((int(i - 1), int(i)))
        ix = np.array(sorted(indices), int)

        def finite_list(x):
            return [float(v) if np.isfinite(v) else None for v in x]

        return {
            "case": case,
            "range": [start, stop],
            "indices": ix.tolist(),
            "time": finite_list(times[ix]),
            "raw": finite_list(raw[ix]),
            "values": finite_list(values[ix]),
            "valid": valid[ix].tolist(),
            "estimated": estimated[ix].tolist(),
            "source_valid": source_valid[ix].tolist(),
            "masks": {
                "modified": intervals(changed(raw, values)),
                "invalid": intervals(~valid),
                "source_invalid": intervals(~source_valid),
                "estimated": intervals(estimated),
                "agent_invalid": intervals(source_valid & ~valid),
            },
            "exact": len(ix) == stop - start,
            "artifact_id": artifact["artifact_id"] if artifact else None,
        }

    def detail(self, rid, cid, version=None):
        root, case = self.store.run_path(rid), self.case(rid, cid)
        vid, a = self.chosen(rid, cid, version)
        trace = document(root / a["operator_trace"]["path"]) if a and a.get("operator_trace") else None
        explanation = None
        if rid == "vital-cleaning-v1-20261112-r4":
            path = self.store.repo / "eval/runs/vital-all-signal-gallery-20261113/explanations" / (cid + ".json")
            candidate = document(path)
            if candidate and candidate["version_id"] == vid:
                explanation = candidate
        evidence = [
            {**e, "resource": self.resource(root / e["image"])}
            for e in self.state(rid).get("evidence", {}).values()
            if e["case_id"] == cid and e["version_id"] == vid
        ]
        reviews = [
            e["arguments"]
            for e in self.events(rid)
            if e.get("tool") == "submit_self_review"
            and e["type"] == "tool_completed"
            and not e.get("is_error")
            and e.get("arguments", {}).get("version_id") == vid
        ]
        if a:
            verified(root / a["probe_path"], a["probe_sha256"])
        probe = document(root / a["probe_path"]) if a else None
        return {
            "case": case,
            "version_id": vid,
            "artifact": a,
            "trace": trace,
            "available_versions": [
                p.parent.name for p in (root / "versions").glob("*/version.json") if cid in document(p)["results"]
            ],
            "intermediate_recording": "recorded" if trace else "not_recorded",
            "explanation": explanation,
            "evidence": evidence,
            "reviews": reviews,
            "probe": probe,
        }

    def recipes(self, rid):
        root, rows = self.store.run_path(rid), []
        for p in sorted((root / "recipes").glob("*/recipe.json")):
            recipe = document(p)
            source = p.parent / "program.py"
            rows.append(
                {
                    **recipe,
                    "source": source.read_text() if source.exists() else None,
                    "resource": self.resource(source) if source.exists() else None,
                }
            )
        known = {r.get("source") for r in rows}
        for e in self.events(rid):
            source = e.get("arguments", {}).get("source")
            if source and source not in known:
                rows.append(
                    {
                        "hash": digest(source),
                        "source": source,
                        "kind": "python",
                        "category": "submitted_draft",
                        "recording": "public_tool_arguments",
                        "resource": None,
                    }
                )
                known.add(source)
        return rows

    def recipe_diff(self, rid, left, right):
        recipes = {r["hash"]: r.get("source") or "" for r in self.recipes(rid)}
        return "\n".join(
            difflib.unified_diff(
                recipes[left].splitlines(), recipes[right].splitlines(), fromfile=left, tofile=right, lineterm=""
            )
        )

    def exports(self, rid):
        root = self.store.run_path(rid)
        files = sorted((root / "export").glob("*"))
        if (root / "inspection/inspection_summary.json").exists():
            files.extend(sorted((root / "inspection").glob("*.json*")))
            files.append(root / "inspection/intervals.csv")
        artifacts = [self.resource(p) for p in files if p.is_file()]
        for record in sorted((root / "records").glob("*")):
            for name in ("atlas-30s.pdf", "report.json", "groups.json", "hit-dispositions.json", "direct-review.json"):
                path = record / name
                if path.is_file():
                    artifacts.append(self.resource(path, f"{record.name}/{name}"))
        return artifacts
