"""Distributed patient-split Vital fine-tuning with exhaustive source-row coverage and resumable updates."""

import argparse
import datetime
import math
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from timesfm3 import cpm_revin_refine
from timesfm3.model import TimesFM3Torch

from eval.timesfm3_mimic_finetune import BASE, BASE_SHA, CHANNELS, adapters, load_bank, make_batches, validate
from eval.timesfm3_vital_full_batches import (
    CONTEXT,
    HORIZONS,
    SEED,
    batch_schedule,
    cache_path,
    epoch_layout,
    rank_slice,
    read_batch,
)
from eval.timesfm3_vital_full_data import atomic_json, json_file
from eval.timesfm_gap_forecast import file_hash


def normalized_inputs(x):
    target = x[:, :1, :CONTEXT]
    mu = target.mean(-1, keepdim=True)
    std = target.std(-1, correction=0, keepdim=True)
    std = torch.where(std < 1e-7, 1.0, std)
    peers = x[:, 1:]
    pm = peers.mean(-1, keepdim=True)
    ps = peers.std(-1, correction=0, keepdim=True)
    return (
        (target - mu) / std,
        (peers - pm) / torch.where(ps < 1e-7, 1.0, ps),
        (x[:, 0, CONTEXT:] - mu[:, 0]) / std[:, 0],
    )


def example_losses(q, truth, quantiles):
    point = q.median(dim=-1).values
    error = truth.unsqueeze(-1) - q
    pinball = torch.maximum(quantiles * error, (quantiles - 1) * error).mean((-2, -1))
    point_loss = F.smooth_l1_loss(point, truth, beta=0.5, reduction="none").mean(-1)
    slope = F.smooth_l1_loss(point.diff(dim=-1), truth.diff(dim=-1), beta=0.2, reduction="none").mean(-1)
    size = F.smooth_l1_loss(point.std(-1, correction=0), truth.std(-1, correction=0), beta=0.25, reduction="none")
    return point_loss + 0.25 * pinball + 0.15 * slope + 0.15 * size


def optimize(model, x, weights, optimizer, params, quantiles, dtype, distributed):
    context, peers, truth = normalized_inputs(x)
    for precision, scale in ((dtype, 16.0 if dtype == torch.float16 else 1.0), (torch.bfloat16, 1.0)):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=precision):
            q = model.decode.__wrapped__(model, target=context, past_future_covariates=peers)[:, 0].float()
            loss = (example_losses(q, truth, quantiles) * weights).sum()
        (loss * scale).backward()
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        flat.div_(scale)
        if distributed:
            dist.all_reduce(flat)
        norm = torch.linalg.vector_norm(flat)
        if torch.isfinite(norm).item():
            flat.mul_(torch.clamp(1.0 / (norm + 1e-6), max=1.0))
            for param, grad in zip(params, flat.split([p.numel() for p in params]), strict=True):
                param.grad.copy_(grad.view_as(param))
            optimizer.step()
            reported = loss.detach()
            if distributed:
                dist.all_reduce(reported)
            return reported.item()
    raise FloatingPointError("Nonfinite gradient after retrying the same training examples")


def panel(validation_run):
    bank = load_bank(validation_run / "data", "val")
    frozen = json_file(validation_run / "gaps.json")
    selected = []
    for h, cap in zip(HORIZONS.tolist(), (48, 48, 48, 24, 16), strict=True):
        jobs = [j for batch in frozen if batch["h"] == h for j in batch["jobs"]][:cap]
        selected.extend(make_batches(h, jobs))
    return bank, selected


def atomic_torch(path, thing):
    tmp = path.with_suffix(".partial.pt")
    torch.save(thing, tmp)
    tmp.replace(path)


def save_trainable(path, params):
    tmp = path.with_suffix(".partial.safetensors")
    save_file({k: p.detach().cpu().contiguous() for k, p in params.items()}, tmp)
    tmp.replace(path)


def run(args):
    root = args.run
    rank, world = int(os.getenv("LOCAL_RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30), device_id=device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    # Batches have matching gap lengths across devices; logical batch loss uses the sum of all example weights.
    caps = [[128, 96, 64, 24, 12] for _ in range(world)]
    if world == 3:
        caps[-1] = [144, 120, 80, 32, 16]
    manifest, plan_info = (json_file(root / n) for n in ("train-manifest.json", "coverage-plan.json"))
    if (
        file_hash(BASE / "model.safetensors") != BASE_SHA
        or file_hash(root / "coverage-plan.npz") != plan_info["plan_sha256"]
    ):
        raise ValueError("Unexpected source checkpoint or coverage plan")
    with np.load(root / "coverage-plan.npz", allow_pickle=False) as pack:
        plan = dict(pack)
    waves = np.load(cache_path(root), mmap_mode="r")
    model = TimesFM3Torch.from_pretrained(str(BASE), local_files_only=True).requires_grad_(False)
    adapters(model)
    model.output_head.requires_grad_(True)
    model.to(device)
    for p in model.parameters():
        if not p.requires_grad:
            p.data = p.data.to(dtype)
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    params = list(named.values())
    head = [p for n, p in named.items() if n.startswith("output_head.")]
    lora = [p for n, p in named.items() if not n.startswith("output_head.")]
    optimizer = torch.optim.AdamW([{"params": head}, {"params": lora}], weight_decay=0.01)
    quantiles = torch.tensor(model.quantiles, device=device)
    n_batches = len(batch_schedule(plan, 1, caps))
    protocol = dict(
        source="Vital only",
        base_sha256=BASE_SHA,
        epochs=args.epochs,
        plan_sha256=plan_info["plan_sha256"],
        data_sha256=file_hash(root / "train-manifest.json"),
        channels=CHANNELS,
        context=CONTEXT,
        horizons=HORIZONS.tolist(),
        seed=SEED,
        optimizer="AdamW, 96-step warmup then cosine decay",
        peak_head_lr=6e-5,
        peak_lora_lr=3e-4,
        caps=caps,
        logical_batches_per_epoch=n_batches,
        trainable_parameters=sum(p.numel() for p in params),
        weighting="source rows per gap divided by square root of patient's eligible row count; normalized per logical batch",
        selection="lowest Vital validation macro normalized MAE among completed exhaustive epochs",
        validation_patients_file=str(args.validation / "gaps.json"),
        patient_counts=manifest["split_patients"],
    )
    if rank == 0:
        if (root / "protocol.json").exists() and json_file(root / "protocol.json")["caps"] != caps:
            raise ValueError("Device batch partition differs from the saved run")
        atomic_json(root / "protocol.json", protocol)
        valbank, valjobs = panel(args.validation)
    cursor = root / "training-state.pt"
    epoch0, batch0, steps = 1, 0, 0
    history = []
    seen = np.zeros(len(plan["first"]), np.uint8)
    if cursor.exists():
        old = torch.load(cursor, map_location="cpu", weights_only=True)
        if old["plan_sha256"] != plan_info["plan_sha256"] or old["world_size"] != world or old["epochs"] != args.epochs:
            raise ValueError("Checkpoint schedule mismatch")
        model.load_state_dict(old["params"], strict=False)
        optimizer.load_state_dict(old["optimizer"])
        epoch0, batch0, steps = old["epoch"], old["next_batch"], old["steps"]
        history, seen = old["history"], old["seen"].numpy().copy()
        del old
    elif rank == 0:
        history = [{"epoch": 0, **validate(model.eval(), valbank, valjobs, device)}]
        atomic_json(root / "validation.json", history)
    if world > 1:
        dist.barrier()
    if rank == 0:
        print(
            f"Start at epoch {epoch0}, batch {batch0}. {n_batches} logical batches per complete traversal.", flush=True
        )
    started = time.monotonic()
    starting_steps = steps

    def save_state(epoch, next_batch):
        atomic_torch(
            cursor,
            dict(
                epoch=epoch,
                next_batch=next_batch,
                steps=steps,
                history=history,
                params={k: p.detach().cpu() for k, p in named.items()},
                optimizer=optimizer.state_dict(),
                seen=torch.from_numpy(seen),
                plan_sha256=plan_info["plan_sha256"],
                world_size=world,
                epochs=args.epochs,
            ),
        )

    frozen_stats = torch.no_grad()(cpm_revin_refine.cpm_iterative_revin_refine)
    with patch.object(cpm_revin_refine, "cpm_iterative_revin_refine", frozen_stats):
        for epoch in range(epoch0, args.epochs + 1):
            batches = batch_schedule(plan, epoch, caps)
            layout = epoch_layout(plan, epoch)
            start_at = batch0 if epoch == epoch0 else 0
            committed = int(np.sum(plan["counts"], where=seen == epoch))
            model.train()
            totals = []
            for i in range(start_at, len(batches)):
                kind, ids = batches[i]
                own = rank_slice(ids, kind, caps, rank)
                x = torch.from_numpy(read_batch(waves, plan, layout, own, kind)).to(device)
                weights = torch.from_numpy(plan["weights"][own] / plan["weights"][ids].sum()).to(device)
                progress = steps / max(n_batches * args.epochs - 1, 1)
                lr_factor = min((steps + 1) / 96, 1) * (0.15 + 0.85 * (1 + math.cos(math.pi * progress)) / 2)
                for group, lr in zip(optimizer.param_groups, (6e-5, 3e-4), strict=True):
                    group["lr"] = lr * lr_factor
                totals.append(optimize(model, x, weights, optimizer, params, quantiles, dtype, world > 1))
                steps += 1
                if rank == 0:
                    if not np.all(seen[ids] == epoch - 1):
                        raise ValueError("Duplicate or out-of-order training examples")
                    seen[ids] = epoch
                    committed += int(plan["counts"][ids].sum())
                    if (i + 1) % 25 == 0 or i == start_at or i + 1 == len(batches):
                        average_s = (time.monotonic() - started) / (steps - starting_steps)
                        marker = dict(
                            epoch=epoch,
                            next_batch=i + 1,
                            total_batches=len(batches),
                            total_steps=steps,
                            distinct_rows_committed_this_epoch=committed,
                            mean_recent_loss=float(np.mean(totals[-25:])),
                            elapsed_since_resume_s=round(time.monotonic() - started, 1),
                            seconds_per_step=round(average_s, 3),
                        )
                        atomic_json(root / "progress.json", marker)
                        print(marker, flush=True)
                    if (i + 1) % 100 == 0:
                        save_state(epoch, i + 1)
            if rank == 0:
                if not np.all(seen == epoch) or committed != manifest["totals"]["eligible"]:
                    raise RuntimeError("Optimizer did not cover every eligible row")
                raw_counts = np.bincount(layout[2], weights=plan["counts"], minlength=3).astype(int)
                val = validate(model.eval(), valbank, valjobs, device)
                history.append(
                    dict(epoch=epoch, steps=steps, rows_by_target=raw_counts.tolist(), committed=committed, **val)
                )
                save_trainable(root / f"epoch-{epoch}.safetensors", named)
                atomic_json(root / "validation.json", history)
                save_state(epoch + 1, 0)
                print("Full epoch validation", history[-1], flush=True)
            if world > 1:
                dist.barrier()
    if rank == 0:
        best = min((v for v in history if v["epoch"]), key=lambda v: v["normalized_mae"])
        # Merge the updates into the exact original FP32 checkpoint, independent of the training devices' precision.
        exported = TimesFM3Torch.from_pretrained(str(BASE), local_files_only=True)
        adapters(exported)
        exported.load_state_dict(load_file(root / f"epoch-{best['epoch']}.safetensors"), strict=False)
        adapters(exported, merge=True)
        exported.eval().save_pretrained(root / "checkpoint")
        shutil.copyfile(BASE / "LICENSE", root / "checkpoint/LICENSE")
        atomic_json(
            root / "export.json",
            dict(
                best_epoch=best["epoch"],
                epochs_completed=args.epochs,
                updates=steps,
                base_sha256=BASE_SHA,
                sha256=file_hash(root / "checkpoint/model.safetensors"),
                coverage=plan_info,
                validation=best,
            ),
        )
        print("Full-data checkpoint exported", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--validation", type=Path, default=Path("eval/runs/timesfm3-mimic-on-vital-val-20261122"))
    ap.add_argument("--epochs", type=int, default=3)
    run(ap.parse_args())
