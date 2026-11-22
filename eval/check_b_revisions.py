"""Replay retained public tool replies and hash-check the saved B evidence bundles.

This checks deterministic selection/assembly without invoking a server or model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import b_revisions as br
from eval import b_workflow as bw


class ReplayClient:
    def __init__(self, bundle):
        if any(d.get("executed") for d in bundle["dropped"]):
            raise ValueError("replay requires every executed response; use a live rebuild for text-trimmed bundles")
        self.items = iter(bundle["items"])
        self.calls = 0

    async def call_tool(self, tool, arguments):
        item = next(self.items)
        if item["tool"] != tool or item["arguments"] != arguments:
            raise ValueError(f"replay request differs at {item['key']}")
        self.calls += 1
        blocks = []
        for b in item["content"]:
            if b["type"] == "text":
                blocks.append(SimpleNamespace(type="text", text=b["text"]))
            else:
                blocks.append(
                    SimpleNamespace(type="image", mime_type=b["source"]["media_type"], data=b["source"]["data"])
                )
        return SimpleNamespace(content=blocks, is_error=item["is_error"])


def replay(meta: dict, cache: Path, workflow: str) -> dict:
    raw = (cache / "objects" / f"{meta['bundle_sha256']}.json").read_bytes()
    if bw.digest(raw) != meta["bundle_sha256"]:
        raise ValueError("cached bundle hash mismatch")
    saved = json.loads(raw)
    identity = meta["identity"]
    if bw.digest(bw.canonical(identity).encode()) != meta["request_sha256"]:
        raise ValueError("saved request identity hash mismatch")
    expected_meta = bw.manifest(saved, meta["bundle_sha256"], identity, meta["request_sha256"])
    if workflow != "v0":
        expected_meta.update(b_workflow=workflow, b_workflow_effective=br.effective_workflow(workflow, meta["qtype"]))
        if (
            identity.get("b_workflow") != workflow
            or identity.get("b_workflow_effective") != expected_meta["b_workflow_effective"]
        ):
            raise ValueError("saved version mismatch")
    if expected_meta != meta or identity["workflow_sha256"] != br.workflow_hash(workflow):
        raise ValueError("manifest or workflow source mismatch")
    case = {key: identity[key] for key in ("case_id", "qtype", "question", "visible")}
    case["dir"] = str(Path(identity["signal_path"]).parent)
    client = ReplayClient(saved)
    rebuilt = asyncio.run(br.collect(case, client, workflow=workflow))
    sha = bw.digest((bw.canonical(rebuilt) + "\n").encode())
    if sha != meta["bundle_sha256"] or client.calls != saved["sizes"]["calls"] or next(client.items, None) is not None:
        raise ValueError(f"bundle replay mismatch: {case['case_id']}")
    return {"case_id": case["case_id"], "bundle_sha256": sha, "byte_identical": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-cache", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=bw.CACHE)
    parser.add_argument("--results", type=Path, default=bw.ROOT / "eval/results/b-workflow")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    for workflow, folder, cache, expected in (
        ("v0", "p7dev", args.v0_cache, 66),
        ("r1", "p7dev-r1", args.cache, 34),
        ("r2", "p7dev-r2", args.cache, 32),
    ):
        rows = []
        for path in sorted((args.results / folder / "manifests").glob("*.json")):
            meta = json.loads(path.read_text())
            if meta["status"] != "ready":
                raise ValueError("non-ready bundle")
            rows.append(replay(meta, cache, workflow))
        if len(rows) != expected:
            raise ValueError(f"expected {expected} {workflow} bundles, found {len(rows)}")
        checks[workflow] = rows
    bw.write_json(args.out, {"replays": checks, "workflow_sha256": br.workflow_hash("final")})
    print(bw.canonical({key: len(rows) for key, rows in checks.items()}))


if __name__ == "__main__":
    main()
