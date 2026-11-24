"""Produce an immutable rollback export. Descendants of a reverted patch lose their prior review."""

import csv
from pathlib import Path

from tsmcp.processing import intervals

from .common import array_hash, read, sha, write
from .contracts import Patch, published_run
from .dataset_evidence import acquisition, native_rows
from .dataset_pipeline import csv_table
from .engine import load_arrays


def rollback(workspace, patch_id, out):
    import tempfile

    from .common import locked

    workspace, out = Path(workspace).resolve(), Path(out).resolve()
    if out == workspace or workspace in out.parents:
        raise ValueError("Choose a separate rollback export outside the released run")
    out.parent.mkdir(parents=True, exist_ok=True)
    with locked(out.parent / ("." + out.name + ".rollback-lock")):
        if (out / "rollback.json").exists():
            return _export(workspace, patch_id, out)
        if out.exists() and any(out.iterdir()):
            raise ValueError("Use an empty rollback output")
        with tempfile.TemporaryDirectory(dir=out.parent, prefix=".rollback-") as tmp:
            staging = Path(tmp) / "release"
            result = _export(workspace, patch_id, staging)
            staging.replace(out)
            return result


def _export(workspace, patch_id, out):
    workspace, out = Path(workspace).resolve(), Path(out).resolve()
    if len(patch_id) != 64 or any(s not in "0123456789abcdef" for s in patch_id):
        raise ValueError("Expected a patch key")
    if not published_run(workspace) or out == workspace or workspace in out.parents:
        raise ValueError("Use a released source and a separate export directory")
    (receipt,) = workspace.glob("records/*/patches/" + patch_id + ".json")
    patch = Patch.model_validate(read(receipt))
    if patch.idempotency_key != patch_id:
        raise ValueError("Patch identity changed")
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "rollback.json"
    scope = dict(source=str(workspace), patch=patch_id, receipt_sha256=sha(receipt))
    if marker.exists():
        result = read(marker)
        if result["scope"] != scope or any(sha(out / k) != v for k, v in result["files"].items()):
            raise ValueError("Conflicting or changed rollback export")
        return result
    if any(out.iterdir()):
        raise ValueError("Use an empty rollback output")
    waves = read(workspace / "export/waveform_manifest.json")
    cases = {c["id"]: c for c in read(workspace / "dataset_manifest.json")["cases"]}
    cid = patch.plan.write_ranges[0].block_id
    case = cases[cid]
    aid = waves[cid]["artifact_id"]
    dropped = []
    while aid != patch.candidate_version_id:
        art = read(workspace / "artifacts" / aid / "artifact.json")
        dropped.append(aid)
        aid = art.get("parent_artifact_id")
        if not aid:
            raise ValueError("Patch is not in the released lineage")
    dropped.append(aid)
    parent = read(workspace / "artifacts" / patch.plan.parent_version_id / "artifact.json")
    _, t, y, v, e = load_arrays(workspace, case, parent)
    if (
        array_hash(y, t, v, e) != parent["content_hash"]
        or sha(workspace / parent["probe_path"]) != parent["probe_sha256"]
    ):
        raise ValueError("Rollback parent content changed")
    affected = {key for key, c in cases.items() if acquisition(c) == acquisition(case)}
    waves[cid].update(
        artifact_id=parent["artifact_id"],
        content_hash=parent["content_hash"],
        probe=parent["probe_path"],
        probe_sha256=parent["probe_sha256"],
        waveform=parent["arrays"] or f"public/{cid}/input.npz",
        valid_ranges=intervals(v),
        estimated_ranges=intervals(e),
        trusted_support=intervals(v & ~e),
        modified_samples=parent["stats"]["modified_samples"],
    )
    for w in waves.values():
        w["probe"] = str(workspace / w["probe"])
        w["waveform"] = str(workspace / w["waveform"])
    with (workspace / "export/measurements.csv").open(encoding="utf-8-sig") as f:
        table = csv.DictReader(f)
        fields = table.fieldnames
        rows = [r for r in table if r["case"] != cid]
    for r in rows:
        if r["case"] in affected:
            r.update(quality="review", pending_evidence=True, groups="", direct_ids="", joint_view_unchecked=True)
    for a, b, z in native_rows(read(workspace / parent["probe_path"])):
        rows.append(
            dict(
                z,
                case=cid,
                channel=case["task"],
                a=a,
                b=b,
                start_s=float(t[a]),
                end_s=float(t[b]),
                quality="review",
                pending_evidence=True,
                joint_view_unchecked=True,
                waveform_version=parent["content_hash"],
            )
        )
    from collections import Counter

    counts = {k: Counter() for k in affected}
    for r in rows:
        if r["case"] in counts:
            counts[r["case"]][r["quality"]] += 1
    for k, n in counts.items():
        waves[k]["measurements"] = dict(n)
    csv_table(out / "measurements.csv", rows, fields)
    csv_table(out / "usable_measurements.csv", (r for r in rows if r["quality"] == "keep"), fields)
    write(out / "waveform_manifest.json", waves)
    result = dict(
        scope=scope,
        status="partial",
        affected_blocks=sorted(affected),
        invalidated_descendants=dropped,
        reason="restored_parent_requires_fresh_review",
        files={f.name: sha(f) for f in out.iterdir() if f.is_file()},
    )
    write(marker, result)
    return result
