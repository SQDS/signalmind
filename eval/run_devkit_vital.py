"""Execute a frozen cohort profile, verify its published files, and draw the discovery review sample."""
import argparse
from pathlib import Path

from devkit_review_queue import summarize

from tsmcp.cleaning.common import read
from tsmcp.inspection.run import inspect


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=root / "configs/cleaning/vital-10-inspection.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    profile = read(args.config)
    if profile["mode"] != "inspect_only" or profile["max_api_usd"] != 0:
        raise ValueError("This entry point runs the offline inspection profile")
    workspace = root / profile["workspace"]
    manifest = read(workspace / "dataset_manifest.json")
    actual = (manifest["patients"], len(manifest["cases"]), manifest["scope_hash"])
    expected = (profile["patients"], profile["expected_blocks"], profile["source_scope_hash"])
    if actual != expected:
        raise ValueError("Input cohort changed")
    run = args.out / ("vital-" + profile["preset"])
    result = inspect(workspace, run, profile["preset"], profile["wall_seconds"])
    if not result["inspection_complete"]:
        raise RuntimeError("Incomplete coverage; see inspection_summary.json")
    summarize(workspace, run, args.out / "visual-review")
    print(f'{result["scanned_blocks"]} blocks, {result["candidates"]} candidates; {args.out / "visual-review/index.html"}')


if __name__ == "__main__":
    main()
