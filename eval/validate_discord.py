"""Paired discord validation on the fixed 120 UCR dev series.

Only anomaly_time ids from the two exclusion manifests are consumed; held-out signals are never opened.
No other case files are accessed. Run with --cases-dir when manifests live outside this worktree.
  uv run python eval/validate_discord.py [--cases-dir /path/to/eval/cases] [--output /tmp/discord.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import CASES  # noqa: E402
from eval.discord_baseline import single_scale_discords  # noqa: E402
from eval.tasks import ucr  # noqa: E402
from tsmcp.core.detect import discords  # noqa: E402


def held_out_ids(cases_dir: Path) -> set[int]:
    """Extract exclusion ids only. Missing manifests must never silently widen the dev set."""
    held = set()
    for suite in ("p1", "p3"):
        with (cases_dir / f"{suite}.answers.jsonl").open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row["qtype"] == "anomaly_time":
                        held.add(int(row["answer"]["id"]))
    return held


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120, help="prefix of the fixed 120-series dev set")
    ap.add_argument("--cases-dir", type=Path, default=CASES)
    ap.add_argument("--output", type=Path, help="write dev-only per-series results and aggregate metrics")
    ap.add_argument(
        "--variant",
        choices=("current", "append", "rrf", "robust_z", "independent"),
        default="current",
        help="pair the original with the current implementation or a dev-only ranking experiment",
    )
    args = ap.parse_args()
    if not 1 <= args.n <= ucr.DEV_COUNT:
        ap.error(f"--n must be between 1 and {ucr.DEV_COUNT}")
    held = held_out_ids(args.cases_dir)
    dev = ucr.dev_ids(held)
    if len(dev) != ucr.DEV_COUNT:
        raise SystemExit(f"expected {ucr.DEV_COUNT} dev series, found {len(dev)}; check UCR_DIR")
    assert not dev & held
    methods = {"old": single_scale_discords, "new": discords}
    if args.variant != "current":
        from eval.discord_variants import variant_discords

        methods["new"] = partial(variant_discords, variant=args.variant)
    rows = []
    for p in ucr.files():
        if ucr.file_id(p) not in dev:
            continue
        x, meta = ucr.load(p)
        row = {"id": meta["id"], "n": int(x.size)}
        candidates = {}
        # Alternate order to avoid consistently favouring the second method's warm caches.
        names = list(methods) if len(rows) % 2 == 0 else list(reversed(methods))
        for name in names:
            start = time.perf_counter()
            events = methods[name](x, 1.0, k=5)
            elapsed = time.perf_counter() - start
            candidates[name] = [{key: value for key, value in e.items() if key != "caveat"} for e in events]
            tolerance = max(100, meta["end"] - meta["begin"])
            hits = [meta["begin"] - tolerance <= e["t"] + e["dur"] / 2 <= meta["end"] + tolerance for e in events]
            row[name] = {"seconds": elapsed, **{f"top{k}": any(hits[:k]) for k in (1, 3, 5)}}
        row["matches_original"] = candidates["old"] == candidates["new"]
        rows.append(row)
        if len(rows) == args.n:
            break
    summary = {}
    for name in methods:
        seconds = [r[name]["seconds"] for r in rows]
        summary[name] = {
            **{f"top{k}": sum(r[name][f"top{k}"] for r in rows) / len(rows) for k in (1, 3, 5)},
            "median_s": float(np.median(seconds)),
            "max_s": max(seconds),
        }
        s = summary[name]
        print(
            f"{name}: n={len(rows)} top-1={s['top1']:.2%} top-3={s['top3']:.2%} "
            f"top-5={s['top5']:.2%} median={s['median_s']:.4f}s max={s['max_s']:.4f}s"
        )
    fingerprint = hashlib.sha256(",".join(str(r["id"]) for r in rows).encode()).hexdigest()
    result = {
        "n": len(rows),
        "dev_ids_sha256": fingerprint,
        "variant": args.variant,
        "summary": summary,
        "matches_original": sum(r["matches_original"] for r in rows),
        "series": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
