"""Offline TimesFM candidate generator. The input contract contains observed flanks and requested lengths only."""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bridge(left_prediction, right_prediction):
    """Right prediction is already in acquisition order. Weight each prediction by proximity to its anchor."""
    if left_prediction is None:
        return right_prediction.copy()
    if right_prediction is None:
        return left_prediction.copy()
    if left_prediction.shape != right_prediction.shape:
        raise ValueError("Flank forecasts have different lengths")
    w = np.arange(1, len(left_prediction) + 1) / (1 + len(left_prediction))
    return (1 - w) * left_prediction + w * right_prediction


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import timesfm
    import torch

    torch.set_num_threads(args.threads)
    started = time.monotonic()
    requests = json.loads((args.input / "requests.json").read_text())
    path = args.weights.resolve()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(path), local_files_only=True, torch_compile=False)
    forecast = {}
    by_horizon = defaultdict(list)
    with np.load(args.input / "flanks.npz", allow_pickle=False) as arrays:
        for key, horizon in requests.items():
            context = arrays[key].copy()
            if not len(context) or not np.all(np.isfinite(context)):
                raise ValueError(f"Missing or nonfinite context for {key}")
            padded_horizon = ((horizon + 127) // 128) * 128
            by_horizon[padded_horizon].append((key, context))
    max_context = max(len(context) for batch in by_horizon.values() for _, context in batch)
    batch_size = 8
    for horizon, batch in sorted(by_horizon.items()):
        model.compile(
            timesfm.ForecastConfig(
                max_context=32 * ((max_context + 31) // 32),
                max_horizon=horizon,
                per_core_batch_size=batch_size,
                normalize_inputs=True,
                use_continuous_quantile_head=False,
                force_flip_invariance=False,
                infer_is_positive=False,
                fix_quantile_crossing=False,
            )
        )
        for start in range(0, len(batch), batch_size):
            chunk = batch[start : start + batch_size]
            with torch.inference_mode():
                points, _ = model.forecast(horizon=horizon, inputs=[context for _, context in chunk])
            for i, (key, _) in enumerate(chunk):
                values = np.asarray(points[i, : requests[key]], dtype=float)
                if len(values) != requests[key] or not np.all(np.isfinite(values)):
                    raise ValueError(f"Incomplete forecast for {key}")
                forecast[key] = values
        print("decoded horizon", horizon, "contexts", len(batch), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "forecasts.npz", **forecast)
    manifest = dict(
        checkpoint=str(path),
        checkpoint_sha256=file_hash(path / "model.safetensors"),
        implementation=str(Path(timesfm.__file__).resolve()),
        torch_version=str(torch.__version__),
        device=str(model.model.device),
        requests=len(requests),
        input_sha256=file_hash(args.input / "flanks.npz"),
        lengths_sha256=file_hash(args.input / "requests.json"),
        outputs_sha256=file_hash(args.out / "forecasts.npz"),
        max_context=max_context,
        max_horizons=sorted(by_horizon),
        batch_size=batch_size,
        elapsed_s=round(time.monotonic() - started, 2),
        api_calls=0,
    )
    (args.out / "model.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
