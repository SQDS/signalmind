"""Unreleased measurements on the accepted waveform, with explicit numerical dependencies."""

import json
from collections import defaultdict

import numpy as np

from tsmcp.processing import intervals

from .common import digest, read, sha, write
from .dataset_evidence import native_rows
from .engine import load_arrays
from .tasks import probe


def accepted_chain(head, patches):
    lineage = []
    while head in patches:
        patch = patches[head]
        if patch in lineage:
            raise ValueError("Cycle in accepted patch lineage")
        lineage.append(patch)
        head = patch["plan"]["parent_version_id"]
    return lineage


def derived_rows(case, arrays, source_probe, final_probe, artifact, lineage, domain):
    raw, t, y, valid, estimated = arrays
    task = case["task"]
    native_coverage = np.zeros(len(t), bool)
    for a, b in final_probe["measurement_support"]:
        native_coverage[a:b] = True
    research = np.zeros(len(t), bool)
    rows = []
    if (valid & estimated).any():
        diagnostic = probe(y, t, case["fs"], valid, np.zeros_like(estimated), task)
        if task == "ABP":
            source_marks = {r[k] for r in source_probe["pressures"] for k in ("start_sample", "stop_sample")}
        else:
            source_marks = {z["sample"] for z in source_probe["events"]}
        for a, b, m in native_rows(diagnostic):
            if not estimated[a : b + 1].any():
                continue
            research[a:b] = True
            spans = [[a + left, a + right] for left, right in intervals(estimated[a : b + 1])]
            deps = [
                p
                for p in lineage
                if any(
                    z["block_id"] == case["id"] and z["start_sample"] <= b and a < z["end_sample"]
                    for z in p["plan"]["write_ranges"]
                )
            ]
            endpoints = [
                dict(
                    sample=q,
                    time_s=float(t[q]),
                    origin="estimated"
                    if estimated[q]
                    else "observed_original_location"
                    if q in source_marks and y[q] == raw[q]
                    else "processed_location",
                    matched_source_location=q in source_marks,
                )
                for q in (a, b)
            ]
            entry = dict(
                case=case["id"],
                task=task,
                artifact_id=artifact["artifact_id"],
                content_hash=artifact["content_hash"],
                read_range=[a, b + 1],
                endpoints=endpoints,
                estimates=spans,
                profile=domain.metadata(case),
                released=False,
                status="requires_validation",
                accepted_patch_ids=[p["idempotency_key"] for p in deps],
                waveform_lineage=[p["idempotency_key"] for p in lineage],
                source_input_sha256=artifact["source_sha256"],
                source_probe=source_probe["definition_hash"],
                method="accepted_waveform_including_estimates",
                validation_needed="event_identity_and_hidden_beats" if task != "ABP" else "cycle_and_pressure_fidelity",
            )
            if task == "ABP":
                values = y[a:b]
                quantities = {}
                for name, fn in (("sbp", np.max), ("dbp", np.min)):
                    at = np.flatnonzero(values == fn(values)) + a
                    quantities[name] = dict(
                        value=m[name],
                        unit="mmHg",
                        comparison_range=[a, b],
                        extrema_samples=at.tolist(),
                        estimated_extremum=bool(estimated[at].any()),
                    )
                quantities["mean_pressure"] = dict(
                    value=m["mean_pressure"],
                    unit="mmHg",
                    integration_range=[a, b + 1],
                    method="trapezoid_on_source_time",
                    estimated_samples=int(estimated[a : b + 1].sum()),
                )
            else:
                quantities = {"RR" if task == "ECG" else "PPI": dict(value=m["interval_s"], unit="s")}
            rows.append(dict(entry, quantities=quantities))
    coverage = dict(
        native_samples=int(native_coverage.sum()),
        research_samples=int(research.sum()),
        union_samples=int((research | native_coverage).sum()),
        released_derived_samples=0,
    )
    return rows, coverage


def export_derived(root, cases, final, originals, domain, version, folder):
    path = folder / "derived-measurements.jsonl"
    tmp = path.with_suffix(".tmp")
    coverage = {}
    count = 0
    patches = defaultdict(dict)
    for rid in {c["record_id"] for c in cases}:
        for p in (root / "records" / rid / "patches").glob("*.json"):
            row = read(p)
            patches[rid][row["candidate_version_id"]] = row
    with tmp.open("w", encoding="utf-8") as handle:
        for c in cases:
            cid = c["id"]
            art = final[cid]
            native = read(root / art["probe_path"])
            if not native["estimated_samples_excluded"]:
                n = native["measured_samples"]
                coverage[cid] = dict(native_samples=n, research_samples=0, union_samples=n, released_derived_samples=0)
                continue
            chain = accepted_chain(art["artifact_id"], patches[c["record_id"]])
            source = read(root / originals[cid]["probe_path"])
            rows, coverage[cid] = derived_rows(c, load_arrays(root, c, art), source, native, art, chain, domain)
            for row in rows:
                handle.write(json.dumps(dict(row, release_version=version), allow_nan=False) + "\n")
            count += len(rows)
    tmp.replace(path)
    index = dict(
        release_version=version,
        waveform_heads=digest({cid: a["artifact_id"] for cid, a in final.items()}),
        rows=count,
        released=False,
        file=path.name,
        sha256=sha(path),
        coverage_samples_by_case=coverage,
    )
    write(folder / "derived-index.json", index)
    return index
