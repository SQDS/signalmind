"""Read every native PulseDB Vital training row and verify three-channel, 125-Hz support."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from eval.prepare_vital_processing import FS

VITAL = Path("/home/yanghaoxiang24/PulseDB/vital")
RAW = Path("/home/yanghaoxiang24/bp2rr_new/dataset/Vital/PulseDB_Vital")
PREP = Path("/home/yanghaoxiang24/bp2rr_new/preprocess/data/v1_20260920")
ROW = 1250


def json_file(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def patient_source(pid):
    return PREP / "subjects" / f"vital__{pid}"


def flat_rows(waves):
    # Compare near the first and last percentiles, as in the earlier MIMIC experiment.
    bounds = np.partition(waves, (12, ROW - 13), axis=1)
    return bounds[:, ROW - 13] - bounds[:, 12] <= 1e-6


def read_patient(job):
    pid, folder = job
    dest = folder / f"{pid}.npz"
    if dest.is_file():
        with np.load(dest, allow_pickle=False) as pack:
            return json.loads(pack["summary"].item())
    source = patient_source(pid)
    audit = json_file(PREP / "inventory" / f"vital__{pid}" / "COMPLETE.json")
    mat = RAW / f"{pid}.mat"
    stat = mat.stat()
    original = audit["source"]
    extraction = json_file(source / "COMPLETE.json")
    if (stat.st_size, stat.st_mtime_ns) != (original["stat"]["size"], original["stat"]["mtime_ns"]):
        raise ValueError(f"Vital MAT changed since sample-grid audit for {pid}")
    if extraction["source_sha256"] != original["sha256"] or extraction["split"] != "train":
        raise ValueError(f"Cached ECG and ABP source or partition mismatch for {pid}")
    ecg, abp = (np.load(source / name, mmap_mode="r") for name in ("ecg.npy", "abp_raw.npy"))
    n = audit["windows"]
    if ecg.shape != (n, ROW) or abp.shape != ecg.shape:
        raise ValueError(f"Cached waveform row count mismatch for {pid}")
    with np.load(source / "metadata.npz", allow_pickle=False) as meta:
        valid_flag = meta["include_flag"]
        starts, ends = meta["t_start"], meta["t_end"]
        run = meta["run_id"]
        boundaries = (
            (meta["case_id"][:-1] != meta["case_id"][1:])
            | (np.diff(meta["segment_id"]) != 1)
            | (np.abs(starts[1:] - ends[:-1] - 1 / FS) > 1e-6)
        )
        if not np.array_equal(np.diff(run) != 0, boundaries) or not np.array_equal(meta["stored_index"], np.arange(n)):
            raise ValueError(f"Time continuity audit mismatch for {pid}")
    ppg = np.empty((n, ROW), np.float32)
    clock_ok = np.empty(n, bool)
    with h5py.File(mat) as handle:
        group = handle["Subj_Wins"]
        fields = {k: group[k] for k in ("PPG_Record", "T", "ECG_Record", "ABP_Raw")}
        refs = {k: ds[()].ravel() for k, ds in fields.items() if h5py.check_dtype(ref=ds.dtype) is not None}

        def row(name, idx):
            ds = handle[refs[name][idx]] if name in refs else fields[name]
            return ds[()].ravel()

        for idx in {0, n // 2, n - 1}:
            for field, saved in (("ECG_Record", ecg), ("ABP_Raw", abp)):
                if not np.array_equal(row(field, idx).astype(np.float32), saved[idx], equal_nan=True):
                    raise ValueError(f"Raw and cached {field} differ for {pid}/{idx}")
        for idx in range(n):
            ppg[idx] = row("PPG_Record", idx)
            times = row("T", idx)
            clock_ok[idx] = (
                times.size == ROW
                and abs(times[0] - starts[idx]) < 1e-6
                and abs(times[-1] - ends[idx]) < 1e-6
                and np.all(np.abs(np.diff(times) - 1 / FS) <= 1e-6)
            )
    finite = np.isfinite(ecg).all(1) & np.isfinite(ppg).all(1) & np.isfinite(abp).all(1)
    flat = flat_rows(ecg) | flat_rows(ppg) | flat_rows(abp)
    valid = valid_flag & clock_ok & finite & ~flat
    summary = dict(
        patient=pid,
        raw=n,
        eligible=int(valid.sum()),
        excluded=int((~valid).sum()),
        flagged=int((~valid_flag).sum()),
        bad_clock=int((~clock_ok).sum()),
        nonfinite=int((~finite).sum()),
        flat=int(flat.sum()),
        source_sha256=original["sha256"],
    )
    temp = dest.with_suffix(".partial.npz")
    np.savez(temp, ppg=ppg, valid=valid, run=run, summary=json.dumps(summary))
    temp.replace(dest)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    folder = args.run / "train-data"
    folder.mkdir(parents=True, exist_ok=True)
    split_files = {s: VITAL / s / "subject_indices.json" for s in ("train", "val", "test")}
    ids = {s: set(json_file(p)) for s, p in split_files.items()}
    if any(ids[a] & ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Vital patient overlap")
    cases, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case in pool.map(read_patient, [(pid, folder) for pid in sorted(ids["train"])], chunksize=1):
            cases.append(case)
            counts.update(
                {k: case[k] for k in ("raw", "eligible", "excluded", "flagged", "bad_clock", "nonfinite", "flat")}
            )
            if len(cases) % 50 == 0:
                print(
                    f"Verified {len(cases)}/{len(ids['train'])} patients and {counts['eligible']} eligible rows",
                    flush=True,
                )
    manifest = dict(
        split_patients={s: len(v) for s, v in ids.items()},
        split_sha256={s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in split_files.items()},
        raw=str(RAW),
        preprocessing=str(PREP),
        fields=["ECG_Record", "PPG_Record", "ABP_Raw"],
        fs=FS,
        samples_per_row=ROW,
        totals=dict(counts),
        patients=cases,
    )
    atomic_json(args.run / "train-manifest.json", manifest)
    print("Completed", dict(counts), flush=True)


if __name__ == "__main__":
    main()
