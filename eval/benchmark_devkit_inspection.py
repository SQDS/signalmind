"""Build the public inspection inputs, run all three profiles, then score frozen candidates."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsmcp.cleaning.common import read, sha, write  # noqa: E402
from tsmcp.inspection.run import inspect  # noqa: E402
from tsmcp.inspection.scan import PRESETS  # noqa: E402
from tsmcp.locators.evaluation import aggregate_rows, score_result  # noqa: E402
from tsmcp.processing import intervals  # noqa: E402


def prepare_public(pack, root):
    cases = []
    (root / "public").mkdir(parents=True, exist_ok=True)
    for cid in read(pack / "public/index.json")["case_ids"]:
        source = pack / "public/cases" / cid
        task = read(source / "task.json")
        channel = task["signal"]["channels"][0]
        link = root / "public" / cid
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=True)
        cases.append(
            dict(
                id=cid,
                record_id=cid,
                fs=channel["fs_hz"],
                n=channel["n_samples"],
                unit=channel["unit"],
                task=channel["kind"],
                input_sha256=task["signal"]["sha256"],
            )
        )
    write(root / "dataset_manifest.json", {"patients": None, "cases": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.out / "public-workspace"
    prepare_public(args.pack, workspace)
    summaries = {preset: inspect(workspace, args.out / preset, preset) for preset in PRESETS}
    if any(not s["inspection_complete"] for s in summaries.values()):
        raise ValueError("Incomplete public run; see coverage reports")
    refs = read(args.pack / "private/manifest.json")["cases"]
    rows = []
    for preset in PRESETS:
        folder = args.out / preset
        grouped = defaultdict(list)
        assert sha(folder / "candidates.jsonl") == read(folder / "candidate_manifest.json")["files"]["candidates.jsonl"]
        for line in (folder / "candidates.jsonl").read_text().splitlines():
            c = json.loads(line)
            r = c["range"]
            grouped[r["block_id"]].append(dict(start=r["start_sample"], end=r["end_sample"]))
        for item in refs:
            cid = item["case_id"]
            with np.load(args.pack / "public/cases" / cid / "input.npz", allow_pickle=False) as z:
                x = z["values"][0]
                flags = {k: z[k] for k in ("observed_missing_mask", "device_saturation_flag")}
            path = args.pack / "private/cases" / cid / "reference.npz"
            assert sha(path) == item["reference_sha256"]
            with np.load(path, allow_pickle=False) as z:
                truth = {k: z[k] for k in ("A", "C", "D", "E")}
            scored = score_result(
                dict(
                    method=preset, events=grouped[cid], coverage_known=True, evaluated_ranges=intervals(np.isfinite(x))
                ),
                input_values=x,
                fs=item["fs_hz"],
                device_flags=flags,
                reference=truth,
            )
            rows.append(
                dict(
                    case_id=cid,
                    channel=item["kind"],
                    source_cluster=item["source_cluster"],
                    family=item["family"],
                    **scored,
                )
            )
    write(args.out / "score_by_case.json", rows)
    aggregate = aggregate_rows(rows)["by_method"]
    write(args.out / "scores.json", aggregate)
    for name, s in aggregate.items():
        print(name, s["tracks"]["unknown_observed"]["micro"])


if __name__ == "__main__":
    main()
