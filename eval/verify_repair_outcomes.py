"""Verify the PR-01 v1.0 archival regression, without any model calls or writes to the original runs."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import read, sha, write
from tsmcp.cleaning.repair_outcomes import export_outcomes, json_rows


def content_set(root):
    """Compare actual candidate buffers and probes even though implementation-bound version IDs change."""
    rows = []
    for f in root.glob("**/artifacts/*/artifact.json"):
        doc = read(f)
        from tsmcp.cleaning.common import array_hash
        from tsmcp.cleaning.engine import load_arrays

        ws = f.parents[2]
        cid = doc["case_id"]
        # Fixture workspaces use immutable per-case inputs, no private reference in the probe.
        pub = ws / "public" / cid / "input.npz"
        with np.load(pub, allow_pickle=False) as z:
            n = len(z["values"])
        _, t, y, v, e = load_arrays(ws, dict(id=cid, n=n), doc)
        actual = array_hash(y, t, v, e)
        assert actual == doc["content_hash"]
        rows.append((str(ws.relative_to(root)), cid, actual, sha(ws / doc["probe_path"])))
    return sorted(rows)


def main(archive, out):
    compare = archive.parents[1] / "results" / archive.name
    details = []
    counts = {}
    for name in ("bipolar", "same-sign", "moderate", "synthetic"):
        root = archive / name
        dest = out / name
        summary = export_outcomes(root, dest)
        hits = list(json_rows(dest / "candidates.jsonl"))
        attempts = list(json_rows(dest / "attempts.jsonl"))
        costs = list(json_rows(dest / "api_calls.jsonl"))
        evidence = read(root / "export/evaluation_report.json")
        assert len(hits) == len({r["candidate_id"] for r in hits}) == summary["counts"]["candidates"]
        assert len(attempts) == len({r["attempt_id"] for r in attempts}) == summary["counts"]["attempts"]
        assert Counter(r["outcome"] for r in attempts) == Counter(summary["attempt_outcomes"])
        assert sum(r["in_initial_freeze"] for r in hits) == read(root / "ledger/freeze.json")["candidate_count"]
        assert all(r["waveform_mae"] is None and r["waveform_gain"] is None for r in hits)
        assert abs(summary["api_returned_usd"] - evidence["estimated_ds_usd"]) < 1e-9
        assert sum(r["status"] == "returned" for r in costs) == evidence["model_calls"]
        by_id = {r["attempt_id"]: r for r in attempts}
        counts[name] = summary["counts"]
        if name == "synthetic":
            continue
        # Read evaluator coordinates only after the outcome exporter has closed its products.
        expected = read(compare / (name + "-comparison.json"))
        for item in expected["targets"]:
            matches = [
                h
                for h in hits
                if h["in_initial_freeze"]
                and h["detector"] == "isolated_spike_candidate"
                and (h["region"]["block_id"], h["region"]["start_sample"], h["region"]["end_sample"])
                == (item["case"], item["a"], item["b"])
            ]
            (hit,) = matches
            ops = [by_id[k] for k in hit["attempts"] if by_id[k]["operation"] == "interpolate"]
            committed = any(r["commit"] == "committed" for r in ops)
            assert committed == item["new_recovered"]
            assert (hit["released_changed_samples_in_region"] == item["b"] - item["a"]) == committed
            if not committed:
                assert hit["reason_codes"]
            details.append(
                dict(
                    shape=name,
                    candidate_id=hit["candidate_id"],
                    a=item["a"],
                    b=item["b"],
                    generated=sum(r["candidate_generated"] for r in ops),
                    committed=committed,
                    reason_codes=hit["reason_codes"],
                    attempts=hit["attempts"],
                )
            )
    reference = archive / "local-hard-gate"
    rerun = out / "behavior-regression"
    old = content_set(reference)
    new = content_set(rerun)
    old_report, current_report = (read(p / "verification.json") for p in (reference, rerun))
    # The gate includes its source hash. Adding a registered tool produces a new spec even for a legacy replay.
    spec_ids = [z.pop("acceptance_spec") for z in (old_report, current_report)]
    assert old and old == new and old_report == current_report
    assert len(details) == 12 and sum(d["committed"] for d in details) == 6
    write(
        out / "verified.json",
        dict(
            archive_counts=counts,
            targets=details,
            identical_numerical_artifacts=len(new),
            legacy_gate_spec_ids=spec_ids,
            historical_released=6,
            new_model_calls=0,
        ),
    )
    print(f"{len(details)} target decisions verified, {len(new)} matching waveform/support/probe artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    main(a.archive, a.out)
