"""Freeze amendment F1 check: the B allowlist fix leaves every B v0 bundle's evidence unchanged.

  .venv/bin/python eval/b_guard_equivalence.py --rebuilt /tmp/b-equiv/out/p7dev/manifests --rebuilt-cache /tmp/b-equiv/cache \
      [--json out.json]

Compares each rebuilt p7dev v0 bundle with the frozen development bundle (manifest in
eval/results/b-workflow/p7dev/manifests, object in eval/cache/b-workflow/objects): the model-facing `content`
blocks must be byte-identical, and every item must match on tool, arguments, response hash, sizes, key, priority and
execution; the dropped list, sizes and status must match. Only identity metadata (e.g. the workflow file hash) and
therefore the bundle hash may differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN_MANIFESTS = ROOT / "eval/results/b-workflow/p7dev/manifests"
FROZEN_OBJECTS = ROOT / "eval/cache/b-workflow/objects"
ITEM_FIELDS = (
    "key",
    "tool",
    "arguments",
    "response_sha256",
    "sizes",
    "priority",
    "required",
    "is_error",
    "call_number",
)


def canon(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compare(frozen: dict, rebuilt: dict) -> list[str]:
    diffs = []
    if canon(frozen.get("content")) != canon(rebuilt.get("content")):
        diffs.append("content")
    fi, ri = frozen.get("items", []), rebuilt.get("items", [])
    if len(fi) != len(ri):
        diffs.append(f"item count {len(fi)} != {len(ri)}")
    for n, (a, b) in enumerate(zip(fi, ri, strict=False)):
        for f in ITEM_FIELDS:
            if canon(a.get(f)) != canon(b.get(f)):
                diffs.append(f"item {n} {f}")
    for f in ("dropped", "sizes", "status", "qtype"):
        if canon(frozen.get(f)) != canon(rebuilt.get(f)):
            diffs.append(f)
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuilt", type=Path, required=True, help="rebuilt manifests directory")
    ap.add_argument("--rebuilt-cache", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows, ok = [], 0
    for fm in sorted(FROZEN_MANIFESTS.glob("*.json")):
        f_meta = json.loads(fm.read_text())
        r_meta = json.loads((args.rebuilt / fm.name).read_text())
        frozen = json.loads((FROZEN_OBJECTS / f"{f_meta['bundle_sha256']}.json").read_text())
        rebuilt = json.loads((args.rebuilt_cache / "objects" / f"{r_meta['bundle_sha256']}.json").read_text())
        diffs = compare(frozen, rebuilt)
        ok += not diffs
        rows.append({"case_id": fm.stem, "frozen_bundle": f_meta["bundle_sha256"], "rebuilt_bundle": r_meta["bundle_sha256"],
                     "evidence_identical": not diffs, "diffs": diffs})  # fmt: skip
    out = {"cases": len(rows), "evidence_identical": ok, "rows": rows}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"cases": len(rows), "evidence_identical": ok,
                      "first_diffs": [r for r in rows if r["diffs"]][:3]}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
