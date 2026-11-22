"""Post-freeze inventory: unused LTAFDB headers/annotations only, never detector/signal reads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.af_episode_metrics import suite_record_ids  # noqa: E402
from eval.tasks.longrec import BEATS, af_segments, merge_episodes  # noqa: E402


def remaining(split, excluded):
    return sorted(set(split["unused"]) - set(split["dev"]) - set(split["test_ext"]) - excluded["ltafdb"])


def inspect_record(root, rec):
    import wfdb

    path = root / rec
    files = {ext: path.with_suffix(f".{ext}") for ext in ("dat", "hea", "atr")}
    row = {"record": rec, "present": {ext: p.is_file() for ext, p in files.items()}}
    row.update(duration_s=None, has_af=None, total_af_s=None, complete=False, eligible={})
    if not files["hea"].is_file():
        return row
    try:
        h = wfdb.rdheader(str(path))
        row["duration_s"] = h.sig_len / h.fs
        # LTAFDB uses single-file format 16; byte counts, never read waveform bytes.
        same_file = set(h.file_name) == {f"{rec}.dat"}
        expected = h.sig_len * h.n_sig * 2 if same_file and set(h.fmt) == {"16"} else None
        row["dat_expected_bytes"] = expected
        row["dat_actual_bytes"] = files["dat"].stat().st_size if files["dat"].is_file() else None
        row["dat_complete"] = expected is not None and row["dat_actual_bytes"] == expected
        row["header_complete"] = h.n_sig == len(h.fmt) and h.sig_len > 0 and h.fs > 0
        if not files["atr"].is_file():
            return row
        annotation = wfdb.rdann(str(path), "atr")
        row["annotation_complete"] = bool(
            len(annotation.sample) and (annotation.sample >= 0).all() and (annotation.sample < h.sig_len).all()
        )
        segs = af_segments(annotation, h.fs, h.sig_len)
        af_s = float(sum(b - a for a, b in segs))
        merged = merge_episodes(segs)
        longest_s = float(max((b - a for a, b in merged), default=0.0))
        pct = 100 * af_s / row["duration_s"]
        beats = sum(sym in BEATS for sym in annotation.symbol)
        row.update(has_af=bool(segs), total_af_s=af_s, af_percent=pct, annotated_beat_count=beats)
        row["complete"] = row["dat_complete"] and row["header_complete"] and row["annotation_complete"]
        row["eligible"] = {"burden": 5 < pct < 95, "longest_onset": longest_s > 300 and pct < 90, "hr": beats >= 100}
    except (ValueError, OSError) as exc:
        row["error_type"] = type(exc).__name__  # no header/patient free text
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--ltaf-dir", type=Path, default=ROOT / "data/ltafdb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-commit", required=True, help="Detector freeze commit; inventory must follow it")
    args = parser.parse_args()
    split = json.loads((ROOT / "eval/tasks/ltafdb_split.json").read_text())
    excluded, manifests = suite_record_ids(args.cases_dir)
    rows = [inspect_record(args.ltaf_dir, rec) for rec in remaining(split, excluded)]
    result = {
        "detector_frozen_commit": args.frozen_commit,
        "access": "headers and annotations only; waveform presence/size via stat, no detector or waveform read",
        "suite_manifests": manifests,
        "excluded_unused": sorted(set(split["unused"]) - set(remaining(split, excluded))),
        "eligibility_rule": {
            "burden": "5 < AF percent < 95",
            "longest_onset": "merged longest > 300 s and AF percent < 90",
            "hr": ">=100 annotated beats",
        },
        "records": rows,
        "counts": {"remaining": len(rows), "complete": sum(r["complete"] for r in rows)},
        "annotation_eligible": {
            kind: sum(r["eligible"].get(kind, False) for r in rows) for kind in ("burden", "longest_onset", "hr")
        },
        "complete_eligible": {
            kind: sum(r["complete"] and r["eligible"].get(kind, False) for r in rows)
            for kind in ("burden", "longest_onset", "hr")
        },
        "limitations": "Complete means exact header-derived waveform byte count and parseable in-range annotations; no waveform checksum/content validation. No cases, answers or model runs created.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("counts", "annotation_eligible", "complete_eligible")}, indent=2))


if __name__ == "__main__":
    main()
