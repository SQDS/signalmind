"""Run TimesFM 3.0 on a frozen 2.5 gap benchmark, with and without recorded peer channels."""

import argparse
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.timesfm_gap_forecast import file_hash


def prepare_peers(delivery, frozen, out):
    from eval.timesfm_gap_trial import Release, read_csv
    from tsmcp.cleaning.common import read, write

    source = read(frozen / "scoring/locations.json")
    release = Release(delivery)
    peers = defaultdict(list)
    for cid, c in release.cases.items():
        peers[tuple(c["rows"])].append(cid)
    restricted = defaultdict(list)
    for table in ("RR_PPI全部含状态.csv", "逐拍压力全部含状态.csv"):
        for row in read_csv(delivery / table):
            if row["quality"] != "keep":
                restricted[row["case"]].append((int(row["a"]), int(row["b"])))

    signals, provenance = {}, {}
    with np.load(frozen / "model_input/flanks.npz", allow_pickle=False) as inputs:
        for kind in ("benchmark", "previous_exclusions", "illustrative_joint_event"):
            for gap in source.get(kind, []):
                cid, a, b = gap["case"], gap["a"], gap["b"]
                target_t, _, target_y, _, _ = release.signal(cid)
                block = tuple(release.cases[cid]["rows"])
                for side in ("ql", "qr"):
                    key = gap[side]
                    if key is None:
                        continue
                    size = len(inputs[key])
                    lo, hi = (a - size, b) if side == "ql" else (a, b + size)
                    visible_target = target_y[lo:a] if side == "ql" else target_y[b:hi][::-1]
                    assert np.array_equal(visible_target, inputs[key])
                    covariates, names = [], []
                    for peer in sorted(peers[block]):
                        # The joint-event preview explicitly removes ALL three signals over the same time range.
                        if peer == cid or kind == "illustrative_joint_event" or release.manifest[peer]["block_hold"]:
                            continue
                        pt, _, y, v, e = release.signal(peer)
                        if not np.array_equal(pt[lo:hi], target_t[lo:hi]) or not (v & ~e & np.isfinite(y))[lo:hi].all():
                            continue
                        if any(start < b and a < end for start, end in restricted[peer]):
                            continue
                        values = y[lo:hi] if side == "ql" else y[lo:hi][::-1]
                        covariates.append(values)
                        names.append(release.cases[peer]["task"])
                    signals[key] = np.asarray(covariates, dtype=float).reshape(-1, size + b - a)
                    provenance[key] = dict(target=gap["channel"], peers=names, example=gap["id"], side=side)
                    assert gap["channel"] not in names
        assert set(inputs.files) == set(signals)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "recorded_peer_inputs.npz", **signals)
    write(out / "peer_availability.json", provenance)
    print("available peer counts", Counter(len(x["peers"]) for x in provenance.values()), flush=True)


def forecast(frozen, out, weights, peer_file, threads):
    import json

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import timesfm3
    import torch
    from timesfm3 import TimesFM3Forecaster

    torch.set_num_threads(threads)
    started = time.monotonic()
    lengths = json.loads((frozen / "model_input/requests.json").read_text())
    with np.load(frozen / "model_input/flanks.npz", allow_pickle=False) as f:
        flank = dict(f)
    with np.load(peer_file, allow_pickle=False) as f:
        covariates = dict(f)
    assert set(flank) == set(lengths) == set(covariates)
    model = TimesFM3Forecaster.from_pretrained(
        str(weights.resolve()), device="cpu", local_files_only=True, per_core_batch_size=8
    )
    load_s = time.monotonic() - started
    checksum = file_hash(weights / "model.safetensors")
    if checksum != "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da":
        raise ValueError("Weights differ from the pinned Google 3.0 checkpoint")
    for variant in ("single", "with_peers"):
        start = time.monotonic()
        results = {}
        batches = defaultdict(list)
        for key, n in lengths.items():
            n_peers = len(covariates[key]) if variant == "with_peers" else 0
            batches[n, n_peers].append(key)
        for (h, n_peers), keys in sorted(batches.items()):
            for i in range(0, len(keys), 8):
                batch = keys[i : i + 8]
                with torch.inference_mode():
                    answers = list(
                        model.predict_batch(
                            [flank[k] for k in batch],
                            horizon=h,
                            past_future_covariates=[covariates[k] for k in batch] if n_peers else None,
                            return_quantiles=False,
                            use_znorm=True,
                            use_symmetric_averaging=False,
                            make_positive=False,
                        )
                    )
                for key, answer in zip(batch, answers, strict=True):
                    y = np.asarray(answer.forecast, dtype=float).reshape(-1)
                    assert y.shape == (h,) and np.isfinite(y).all()
                    results[key] = y
        assert set(results) == set(lengths)
        target = out / variant / "predictions"
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / "forecasts.npz", **results)
        info = dict(
            model="TimesFM 3.0",
            variant=variant,
            device="cpu",
            source_tag="v3.0.0",
            implementation=str(timesfm3.__file__),
            source_commit="331c6d33cb1ac2611de3056d0ac7164aab6301eb",
            checkpoint=str(weights.resolve()),
            checkpoint_revision="43046b85ec22d584a13f8098c2ed39c889e129c2",
            checkpoint_sha256=checksum,
            torch_version=str(torch.__version__),
            input_sha256=file_hash(frozen / "model_input/flanks.npz"),
            lengths_sha256=file_hash(frozen / "model_input/requests.json"),
            outputs_sha256=file_hash(target / "forecasts.npz"),
            peer_input_sha256=file_hash(peer_file) if variant == "with_peers" else None,
            requests=len(results),
            calls_with_any_peer=sum(len(v) for (h, n), v in batches.items() if n),
            elapsed_s=round(time.monotonic() - start, 2),
            load_s=round(load_s, 2),
            api_calls=0,
        )
        (target / "model.json").write_text(json.dumps(info, indent=2))
        print(variant, len(results), info["calls_with_any_peer"], info["elapsed_s"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare-peers", "forecast"))
    ap.add_argument("frozen", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delivery", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    if args.action == "prepare-peers":
        prepare_peers(args.delivery, args.frozen, args.out)
    else:
        forecast(args.frozen, args.out, args.weights, args.out / "recorded_peer_inputs.npz", args.threads)


if __name__ == "__main__":
    main()
