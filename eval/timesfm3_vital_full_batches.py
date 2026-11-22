"""Partition every eligible Vital training row into disjoint, time-contiguous reconstruction examples."""

import argparse
import math
import os
from pathlib import Path

import numpy as np

from eval.timesfm3_mimic_data import seed_for
from eval.timesfm3_vital_full_data import ROW, atomic_json, json_file, patient_source
from eval.timesfm_gap_forecast import file_hash

CONTEXT = 512
HORIZONS = np.array([125, 625, 1250, 3750, 7500], dtype=np.int32)
GROUP_ROWS = np.ceil((CONTEXT + HORIZONS) / ROW).astype(int)
PROBS = np.array([0.17, 0.26, 0.26, 0.17, 0.14])
SEED = 20261122


def cache_path(root):
    checksum = file_hash(root / "train-manifest.json")[:20]
    return Path("/tmp") / f"signalmind-vital-{os.getuid()}" / checksum / "waves.npy"


def build(root):
    manifest = json_file(root / "train-manifest.json")
    patients = manifest["patients"]
    total = manifest["totals"]["raw"]
    cache = cache_path(root)
    temp = cache.with_suffix(".partial.npy")
    cache.parent.mkdir(parents=True, exist_ok=True)
    waves = (
        None if cache.is_file() else np.lib.format.open_memmap(temp, mode="w+", dtype="float32", shape=(3, total * ROW))
    )
    jobs = []
    offset = 0
    eligible = np.zeros(total, dtype=bool)
    for person, meta in enumerate(patients):
        pid, count = meta["patient"], meta["raw"]
        with np.load(root / "train-data" / f"{pid}.npz", allow_pickle=False) as pack:
            valid, run = pack["valid"], pack["run"]
            if waves is not None:
                source = patient_source(pid)
                dest = waves[:, offset * ROW : (offset + count) * ROW]
                dest[0] = np.load(source / "ecg.npy", mmap_mode="r").ravel()
                dest[1] = pack["ppg"].ravel()
                dest[2] = np.load(source / "abp_raw.npy", mmap_mode="r").ravel()
        if int(valid.sum()) != meta["eligible"]:
            raise ValueError(f"Validity mismatch for {pid}")
        eligible[offset : offset + count] = valid
        cuts = np.r_[0, np.flatnonzero((run[1:] != run[:-1]) | (valid[1:] != valid[:-1])) + 1, count]
        rng = np.random.default_rng(seed_for("vital-full/" + pid))
        channel = int(rng.integers(3))
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
            if not valid[lo]:
                continue
            while lo < hi:
                choices = np.flatnonzero(GROUP_ROWS <= hi - lo)
                prob = PROBS[choices]
                k = int(choices[np.searchsorted(np.cumsum(prob), rng.random() * prob.sum())])
                size = GROUP_ROWS[k]
                jobs.append((person, offset + lo, size, k, channel))
                channel = (channel + 1) % 3
                lo += size
        offset += count
    if waves is not None:
        waves.flush()
        del waves
        temp.replace(cache)
    if np.load(cache, mmap_mode="r").shape != (3, total * ROW):
        raise ValueError("Incomplete waveform cache")
    plan = np.asarray(jobs, np.int32)
    person, first, counts, kind, channel = plan.T
    diff = np.zeros(total + 1, np.int32)
    np.add.at(diff, first, 1)
    np.add.at(diff, first + counts, -1)
    if not np.array_equal(np.cumsum(diff)[:-1], eligible.astype(int)):
        raise ValueError("Physical rows must occur exactly once in the training partition")
    rows_by_patient = np.array([p["eligible"] for p in patients])
    weights = (counts / np.sqrt(rows_by_patient[person])).astype(np.float32)
    weights /= weights.mean()
    temp = root / "coverage-plan.partial.npz"
    np.savez(temp, person=person, first=first, counts=counts, kind=kind, channel=channel, weights=weights)
    temp.replace(root / "coverage-plan.npz")
    info = dict(
        eligible_rows=int(eligible.sum()),
        groups=len(plan),
        patients=len(np.unique(person)),
        group_counts_by_seconds={str(h // 125): int((kind == n).sum()) for n, h in enumerate(HORIZONS)},
        native_125hz=True,
        every_eligible_row_once_per_traversal=True,
        min_supervised_samples_per_row=64,
        plan_sha256=file_hash(root / "coverage-plan.npz"),
    )
    atomic_json(root / "coverage-plan.json", info)
    print(info, flush=True)


def epoch_layout(plan, epoch):
    rng = np.random.default_rng(SEED + 1009 * epoch)
    kind, counts = plan["kind"], plan["counts"]
    h = HORIZONS[kind]
    # Every constituent raw row has >=64 samples in the supervised target, in either temporal direction.
    lo = np.maximum(0, (counts - 1) * ROW + 64 - CONTEXT - h)
    hi = np.minimum(counts * ROW - CONTEXT - h, ROW - CONTEXT - 64)
    if np.any(lo > hi):
        raise ValueError("Gap cannot cover its listed source rows")
    crop = rng.integers(lo, hi + 1, dtype=np.int32)
    reverse = rng.integers(0, 2, size=len(kind), dtype=np.int8).astype(bool)
    return crop, reverse, (plan["channel"] + epoch - 1) % 3


def batch_schedule(plan, epoch, capacities):
    rng = np.random.default_rng(SEED + 957 * epoch)
    batches = []
    for kind in range(len(HORIZONS)):
        ids = np.flatnonzero(plan["kind"] == kind)
        rng.shuffle(ids)
        n = max(1, math.ceil(len(ids) / sum(c[kind] for c in capacities)))
        batches.extend((kind, batch) for batch in np.array_split(ids, n))
    rng.shuffle(batches)
    return batches


def rank_slice(ids, kind, capacities, rank):
    cuts = np.rint(
        np.r_[0, np.cumsum([c[kind] for c in capacities])] * len(ids) / sum(c[kind] for c in capacities)
    ).astype(int)
    return ids[cuts[rank] : cuts[rank + 1]]


def read_batch(waves, plan, layout, ids, kind):
    crop, reverse, target = layout
    length = CONTEXT + int(HORIZONS[kind])
    rows, offsets = plan["counts"][ids], crop[ids]
    start = plan["first"][ids].astype(np.int64) * ROW
    start += np.where(reverse[ids], rows * ROW - 1 - offsets, offsets)
    t = start[:, None] + np.where(reverse[ids], -1, 1)[:, None] * np.arange(length)
    order = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])[target[ids]]
    return waves[order[:, :, None], t[:, None, :]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    build(parser.parse_args().run)
