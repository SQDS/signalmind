"""Fine-tune the local TimesFM 3.0 checkpoint on rotating MIMIC ECG, PPG and ABP gaps."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from timesfm3 import cpm_revin_refine
from timesfm3.model import TimesFM3Torch
from torch import nn

from eval.timesfm3_mimic_data import FS, SEED, seed_for
from eval.timesfm_gap_forecast import file_hash

CHANNELS = ("ECG", "PPG", "ABP")
CONTEXT = 512
HORIZONS = (125, 625, 1250, 3750, 7500)
PROBABILITY = (0.25, 0.40, 0.25, 0.07, 0.03)
VAL_LIMITS = (48, 48, 48, 16, 12)
TEST_LIMITS = (340, 340, 340, 96, 64)
RANK, LAYERS = 8, 4
BASE = Path("/home/yanghaoxiang24/models/timesfm-3.0-pytorch")
BASE_SHA = "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"


def write_json(path, value):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class LowRankLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, RANK, bias=False)
        self.up = nn.Linear(RANK, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


def adapters(model, merge=False):
    """On export, fold every low-rank update back into its original linear layer."""
    for layer in model.transformer_stack.layers[-LAYERS:]:
        for attention in (layer.seq_attn, layer.var_attn):
            for name in ("query_proj", "value_proj"):
                old = getattr(attention, name)
                if merge:
                    with torch.no_grad():
                        old.base.weight.addmm_(old.up.weight, old.down.weight, alpha=old.scale)
                    setattr(attention, name, old.base)
                else:
                    setattr(attention, name, LowRankLinear(old))


def load_bank(folder, split):
    manifest = json.loads((folder / "manifest.json").read_text())
    result = {}
    for entry in manifest["patients"]:
        if entry["split"] == split and entry["clips"]:
            with np.load(folder / entry["file"], allow_pickle=False) as f:
                result[entry["patient"]] = [(f[f"wave{i}"], f[f"rpeaks{i}"]) for i in range(len(entry["clips"]))]
    return result


def choose_crop(bank, patient, horizon, rng, reverse):
    clips = bank[patient]
    matches = [i for i, (x, _) in enumerate(clips) if x.shape[-1] >= CONTEXT + horizon]
    slot = int(rng.choice(matches))
    start = int(rng.integers(clips[slot][0].shape[-1] - CONTEXT - horizon + 1))
    return patient, slot, start, reverse


def train_schedule(bank, epoch):
    rng = np.random.default_rng(SEED + epoch)
    groups = defaultdict(list)
    for patient in sorted(bank):
        max_len = max(x.shape[-1] for x, _ in bank[patient])
        probs = np.asarray(PROBABILITY) * (np.asarray(HORIZONS) + CONTEXT <= max_len)
        horizon = int(rng.choice(HORIZONS, p=probs / probs.sum()))
        groups[horizon].append(choose_crop(bank, patient, horizon, rng, bool(rng.integers(2))))
    batches = []
    for horizon, samples in groups.items():
        rng.shuffle(samples)
        batches.extend(make_batches(horizon, samples))
    rng.shuffle(batches)
    return batches


def eval_schedule(bank, split):
    limits = VAL_LIMITS if split == "val" else TEST_LIMITS
    batches = []
    for horizon, limit in zip(HORIZONS, limits, strict=True):
        candidates = [s for s in bank if max(x.shape[-1] for x, _ in bank[s]) >= CONTEXT + horizon]
        candidates.sort(key=lambda s: seed_for(f"{split}/{horizon}/{s}"))
        jobs = [
            choose_crop(bank, s, horizon, np.random.default_rng(seed_for(f"crop/{split}/{horizon}/{s}")), False)
            for s in candidates[:limit]
        ]
        batches.extend(make_batches(horizon, jobs))
    return batches


def make_batches(horizon, samples):
    patients_per_batch = 12 if horizon <= 1250 else 4
    return [(horizon, samples[i : i + patients_per_batch]) for i in range(0, len(samples), patients_per_batch)]


def crop_values(bank, jobs, horizon):
    views = []
    for patient, slot, start, rev in jobs:
        x = bank[patient][slot][0][:, start : start + CONTEXT + horizon]
        views.append(x[:, ::-1] if rev else x)
    return np.stack(views)


def decoder_inputs(waves):
    """Target normalization sees only the flank. Both peers are observed during the gap."""
    n, _, length = waves.shape
    context = waves[:, :, :CONTEXT]
    center = context.mean(-1, keepdim=True)
    scale = context.std(-1, correction=0, keepdim=True)
    scale = torch.where(scale < 1e-7, 1.0, scale)
    normalized_target = (context - center) / scale
    peer_scale = waves.std(-1, correction=0, keepdim=True)
    normalized_peers = (waves - waves.mean(-1, keepdim=True)) / torch.where(peer_scale < 1e-7, 1.0, peer_scale)
    peers = torch.stack([normalized_peers[:, (1, 2)], normalized_peers[:, (0, 2)], normalized_peers[:, (0, 1)]], dim=1)
    return normalized_target.reshape(n * 3, 1, CONTEXT), peers.reshape(n * 3, 2, length), center, scale


def forecast(model, waves, training=False):
    context, peers, center, scale = decoder_inputs(waves)
    # The public decoder is decorated with no_grad. Its tensor implementation is identical in both modes.
    if training:
        # Treat the inferred RevIN location/scale as normalization statistics. Its inference implementation
        # takes square roots of masked or zero variances, which have undefined derivatives.
        frozen_stats = torch.no_grad()(cpm_revin_refine.cpm_iterative_revin_refine)
        with patch.object(cpm_revin_refine, "cpm_iterative_revin_refine", frozen_stats):
            logits = model.decode.__wrapped__(model, target=context, past_future_covariates=peers)
    else:
        logits = model.decode(target=context, past_future_covariates=peers)
    q = logits[:, 0].reshape(*waves.shape[:2], waves.shape[-1] - CONTEXT, -1).float()
    return q, center, scale


def objective(q, y, quantiles):
    point = q.median(dim=-1).values
    error = y.unsqueeze(-1) - q
    pinball = torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
    point_loss = F.smooth_l1_loss(point, y, beta=0.5)
    slope_loss = F.smooth_l1_loss(point.diff(dim=-1), y.diff(dim=-1), beta=0.2)
    amplitude_loss = F.smooth_l1_loss(point.std(-1, correction=0), y.std(-1, correction=0), beta=0.25)
    return point_loss + 0.25 * pinball + 0.15 * slope_loss + 0.15 * amplitude_loss


@torch.no_grad()
def validate(model, bank, batches, device):
    by_horizon = defaultdict(list)
    for h, jobs in batches:
        waves = torch.from_numpy(crop_values(bank, jobs, h)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            q, center, scale = forecast(model, waves)
        truth = (waves[:, :, CONTEXT:] - center) / scale
        nmae = (q.median(-1).values - truth).abs().mean(-1)
        if not torch.isfinite(nmae).all():
            raise ArithmeticError("Nonfinite validation predictions")
        by_horizon[h].extend(nmae.cpu().numpy())
    groups = {str(h // FS): np.median(rows, axis=0).tolist() for h, rows in by_horizon.items()}
    return {"normalized_mae": float(np.mean(list(groups.values()))), "median_by_seconds_and_channel": groups}


def train(args):
    root, device = args.run, torch.device(args.device)
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    if file_hash(args.base / "model.safetensors") != BASE_SHA:
        raise ValueError("Base weights differ from the SignalMind 3.0 benchmark")
    train_bank, val_bank = (load_bank(root / "data", split) for split in ("train", "val"))
    val_batches = eval_schedule(val_bank, "val")
    model = TimesFM3Torch.from_pretrained(str(args.base.resolve()), local_files_only=True).requires_grad_(False)
    adapters(model)
    model.output_head.requires_grad_(True)
    model.to(device)
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    head = list(model.output_head.parameters())
    lora = [p for k, p in params.items() if not k.startswith("output_head.")]
    optimizer = torch.optim.AdamW([{"params": head, "lr": 1e-4}, {"params": lora, "lr": 8e-4}], weight_decay=0.01)
    quantiles = torch.tensor(model.quantiles, device=device)
    protocol = dict(
        base=str(args.base.resolve()),
        base_sha256=BASE_SHA,
        seed=SEED,
        context=CONTEXT,
        horizons=HORIZONS,
        horizon_probabilities=PROBABILITY,
        channels=CHANNELS,
        fs=FS,
        epochs=args.epochs,
        rank=RANK,
        last_layers=LAYERS,
        trainable_parameters=sum(p.numel() for p in params.values()),
        base_parameters=330710976,
        manifest_sha256=file_hash(root / "data/manifest.json"),
        device=str(device),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        val_limits=VAL_LIMITS,
        test_limits=TEST_LIMITS,
        selection="Lowest equal-weight mean over durations and channels of patient-median validation MAE / observed flank SD",
        input="One target channel hidden. The other two channels are observed over flank and gap. Train direction randomized, evaluation forward.",
        normalization_gradient="Stop gradient only through iterative RevIN statistics, preserve decoder forward values.",
    )
    if (root / "protocol.json").exists() and json.loads((root / "protocol.json").read_text()) != json.loads(
        json.dumps(protocol)
    ):
        raise ValueError("Existing run was started with a different protocol")
    write_json(root / "protocol.json", protocol)
    state_path = root / "training-state.pt"
    history, first_epoch, steps = [], 1, 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["parameters"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        history, first_epoch, steps = state["history"], state["epoch"] + 1, state["steps"]
    else:
        baseline = validate(model.eval(), val_bank, val_batches, device)
        print("Unmodified validation", baseline, flush=True)
        history.append({"epoch": 0, **baseline})
        write_json(root / "history.json", history)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        batches = train_schedule(train_bank, epoch)
        losses = []
        for i, (h, jobs) in enumerate(batches, 1):
            waves = torch.from_numpy(crop_values(train_bank, jobs, h)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                q, center, scale = forecast(model, waves, training=True)
                loss = objective(q, (waves[:, :, CONTEXT:] - center) / scale, quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(loss.detach().item())
            if i == 1 or i % 25 == 0 or i == len(batches):
                print(
                    f"Epoch {epoch}/{args.epochs} batch {i}/{len(batches)} loss {np.mean(losses[-25:]):.4f}", flush=True
                )
        val = validate(model.eval(), val_bank, val_batches, device)
        best_trained = min((v["normalized_mae"] for v in history if v["epoch"]), default=float("inf"))
        row = {"epoch": epoch, "steps": steps, "train_loss": float(np.mean(losses)), **val}
        history.append(row)
        weights = {k: p.detach().cpu().contiguous() for k, p in params.items()}
        if val["normalized_mae"] < best_trained:
            save_file(weights, root / "best-trainable.partial.safetensors")
            (root / "best-trainable.partial.safetensors").replace(root / "best-trainable.safetensors")
        torch.save(
            dict(epoch=epoch, steps=steps, parameters=weights, optimizer=optimizer.state_dict(), history=history),
            root / "training-state.partial.pt",
        )
        (root / "training-state.partial.pt").replace(state_path)
        write_json(root / "history.json", history)
        print("Validation", row, flush=True)
    best = min((v for v in history if v["epoch"]), key=lambda v: v["normalized_mae"])
    model.cpu()
    model.load_state_dict(load_file(root / "best-trainable.safetensors"), strict=False)
    adapters(model, merge=True)
    checkpoint = root / "checkpoint"
    model.eval().save_pretrained(checkpoint)
    shutil.copyfile(args.base / "LICENSE", checkpoint / "LICENSE")
    write_json(
        root / "export.json",
        dict(
            best_epoch=best["epoch"],
            sha256=file_hash(checkpoint / "model.safetensors"),
            base_sha256=BASE_SHA,
            optimizer_steps=steps,
            validation=best,
        ),
    )
    print("Exported best checkpoint", checkpoint, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    train(parser.parse_args())
