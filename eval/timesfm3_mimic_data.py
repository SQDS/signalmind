"""Sample continuous, three-channel clips from the original PulseDB MIMIC records."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval.prepare_vital_processing import FIELDS, FS, PulseMat

RAW = Path("/home/yanghaoxiang24/bp2rr_new/dataset/MIMIC/PulseDB_MIMIC")
SPLITS = Path("/home/yanghaoxiang24/PulseDB/mimic")
INVENTORY = Path("/home/yanghaoxiang24/bp2rr_new/preprocess/data/v1_20260920/inventory")
SEED = 20261121


def seed_for(key):
    return int.from_bytes(hashlib.sha256(f"{SEED}/{key}".encode()).digest()[:8], "little")


def candidate_starts(valid, run, length):
    """Return starts whose entire span is eligible and in the same audited run."""
    if len(run) < length:
        return np.empty(0, dtype=int)
    prefix = np.r_[0, valid.cumsum()]
    return np.flatnonzero(
        (prefix[length:] - prefix[:-length] == length) & (run[length - 1 :] == run[: len(run) - length + 1])
    )


def read_clip(mat, start, length, subject):
    indices = range(start, start + length)
    meta = [mat.metadata(i) for i in indices]
    if any(not m["regular"] or m["subject"] != subject for m in meta):
        raise ValueError(f"Inconsistent subject or sample clock in {subject} at {start}")
    for a, b in zip(meta, meta[1:], strict=False):
        if a["case"] != b["case"] or b["segment"] != a["segment"] + 1 or abs(b["start"] - a["end"] - 1 / FS) > 1e-6:
            raise ValueError(f"Source gap inside audited run in {subject} at {start}")
    rows = np.asarray([[mat.array(field, i) for field in FIELDS.values()] for i in indices], dtype=np.float32)
    if rows.shape != (length, 3, 1250) or not np.isfinite(rows).all():
        return None
    spread = np.diff(np.percentile(rows, [1, 99], axis=2), axis=0)
    if (spread <= 1e-6).any():
        return None
    wave = rows.transpose(1, 0, 2).reshape(3, -1)
    peak_rows = []
    for offset, i in enumerate(indices):
        peaks = mat.array("ECG_RPeaks", i)
        peaks = peaks[np.isfinite(peaks) & (peaks >= 1) & (peaks <= 1250) & (peaks == np.floor(peaks))]
        peak_rows.append(peaks.astype(np.int32) - 1 + offset * 1250)
    return wave, np.unique(np.concatenate(peak_rows))


def prepare_patient(args, *, source="mimic"):
    subject, split, raw, inventory, out = args
    relative = Path(split) / f"{subject}.npz"
    dest = out / relative
    if dest.exists():
        with np.load(dest, allow_pickle=False) as f:
            return json.loads(f["index"].item())
    rng = np.random.default_rng(seed_for(subject))
    with np.load(inventory / f"{source}__{subject}" / "metadata.npz", allow_pickle=False) as m:
        run, valid = m["run_id"], m["include_flag"].copy()
        if not np.array_equal(m["stored_index"], np.arange(len(run))):
            raise ValueError(f"Unexpected raw row mapping for {subject}")
    mat = PulseMat(raw / f"{subject}.mat")
    arrays, clips = {}, []
    try:
        for cap in (7, 2, 1) if split == "train" else (7,):
            for length in range(cap, 0, -1):
                starts = candidate_starts(valid, run, length)
                for start in rng.permutation(starts)[:8]:
                    start = int(start)
                    result = read_clip(mat, start, length, subject)
                    if result is None:
                        continue
                    slot = len(clips)
                    arrays[f"wave{slot}"], arrays[f"rpeaks{slot}"] = result
                    clips.append({"first_row": start, "rows": length, "samples": result[0].shape[1]})
                    valid[start : start + length] = False
                    break
                else:
                    continue
                break
    finally:
        mat.file.close()
    entry = {"patient": subject, "split": split, "file": str(relative), "clips": clips}
    temp = dest.with_suffix(".partial.npz")
    np.savez(temp, index=json.dumps(entry), **arrays)
    temp.replace(dest)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cohort = {}
    jobs = []
    for split in ("train", "val", "test"):
        (args.out / split).mkdir(parents=True, exist_ok=True)
        ids = sorted(json.loads((args.splits / split / "subject_indices.json").read_text()))
        if set(cohort).intersection(ids):
            raise ValueError("Patient overlap across PulseDB partitions")
        cohort.update(dict.fromkeys(ids, split))
        jobs.extend((s, split, args.raw, args.inventory, args.out) for s in ids)
    entries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, entry in enumerate(pool.map(prepare_patient, jobs, chunksize=2), 1):
            entries.append(entry)
            if i % 100 == 0:
                print(f"Prepared {i}/{len(jobs)} patients", flush=True)
    manifest = dict(
        source=str(args.raw.resolve()),
        splits=str(args.splits.resolve()),
        fs=FS,
        fields=FIELDS,
        seed=SEED,
        patients=entries,
    )
    temp = args.out / "manifest.partial.json"
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(args.out / "manifest.json")
    for split in ("train", "val", "test"):
        group = [e for e in entries if e["split"] == split]
        print(
            split,
            "patients",
            sum(bool(e["clips"]) for e in group),
            "clips",
            sum(len(e["clips"]) for e in group),
            flush=True,
        )


if __name__ == "__main__":
    main()
