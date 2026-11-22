import base64
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.cleaning.common import digest, sha, write
from tsmcp.followup.pattern import context_range
from tsmcp.processing import intervals


def aligned_pulse_context(out, view_meta, channels, fs):
    """Plot observed ECG/PPG with ABP on a caller-verified common acquisition clock."""
    left, right = view_meta["view_samples"]
    a, b = view_meta["seed_samples"]
    fig = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(channels), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (case, x, usable) in zip(axes, channels, strict=True):
        y = np.where(usable[left:right], x[left:right], np.nan)
        ax.plot(np.arange(left, right) / fs, y, lw=0.9, color="#24576e")
        ax.axvspan(a / fs, b / fs, color="#bc7733", alpha=0.13)
        ax.set_ylabel(case["task"] + "\n" + case["unit"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time in source block (s)")
    axes[-1].ticklabel_format(axis="x", useOffset=False)
    fig.suptitle("Same source times, no pulse transit correction")
    path = Path(out) / (view_meta["id"] + "-channels.png")
    fig.savefig(path, dpi=105)
    data = dict(
        image=path.name, sha256=sha(path), source_versions={c["task"]: c["input_sha256"] for c, _, _ in channels}
    )
    return data, {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(path.read_bytes()).decode()},
    }


def view(out, case, values, usable, version, target, flank_s, *, seed=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fs = case["fs"]
    a, b = target
    seed = target if seed is None else seed
    left, right = context_range(len(values), fs, target, flank_s)
    key = digest([case["id"], version, target, seed, flank_s])[:20]
    fig = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1)
    z0 = max(left, min(seed[0], a) - round(0.3 * fs))
    z1 = min(right, max(seed[1], b) + round(0.3 * fs))
    for ax, (s, e) in zip(axes, [(left, right), (z0, z1)], strict=True):
        ax.plot(np.arange(s, e) / fs, values[s:e], color="#205884", lw=1, marker="." if ax == axes[1] else None, ms=2)
        for u, v in intervals(~usable[s:e]):
            ax.axvspan((u + s) / fs, (v + s) / fs, color="gray", alpha=0.25)
        ax.axvspan(a / fs, b / fs, color="#e99b19", alpha=0.18)
        ax.axvspan(seed[0] / fs, seed[1] / fs, color="#ad532c", alpha=0.32)
        ax.set(xlabel="Time in continuous block (s)", ylabel=case["unit"], xlim=(s / fs, e / fs))
        ax.ticklabel_format(axis="x", useOffset=False)
    axes[0].set_title(f"{case['task']} | context | {flank_s} s on each side (clipped at block ends)")
    axes[1].set_title("Detail | dark: initial trigger; pale: interval under review")
    path = out / (key + ".png")
    fig.savefig(path, dpi=110)
    ev = dict(
        id=key,
        version=version,
        case=case["id"],
        fs=fs,
        unit=case["unit"],
        seed_samples=list(seed),
        target_samples=list(target),
        view_samples=[left, right],
        detail_samples=[z0, z1],
        flank_s=flank_s,
        image=path.name,
        sha256=sha(path),
    )
    write(out / (key + ".json"), ev)
    return ev, {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(path.read_bytes()).decode()},
    }


def comparison(out, case, raw, values, usable, decision, post_version, result):
    """Draw the review footprint, the reviewed episode, and the accepted support separately."""
    fs = case["fs"]
    seed = decision["seed_samples"]
    target = decision.get("suspected_samples", seed)
    flank = decision.get("views", [{"flank_s": 5}])[-1]["flank_s"]
    left, right = context_range(len(raw), fs, target, flank)
    detail = (
        max(0, min(seed[0], target[0]) - round(0.35 * fs)),
        min(len(raw), max(seed[1], target[1]) + round(0.35 * fs)),
    )
    fig = Figure(figsize=(10, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1)
    for ax, arr, (s, e), title in zip(
        axes,
        [raw, raw, values],
        [(left, right), detail, detail],
        ["Original context", "Original detail", "Accepted waveform detail"],
        strict=True,
    ):
        ax.plot(np.arange(s, e) / fs, arr[s:e], lw=1, color="#215b7d")
        ax.axvspan(target[0] / fs, target[1] / fs, color="#edac49", alpha=0.2, label="Reviewer-suspected interval")
        ax.axvspan(seed[0] / fs, seed[1] / fs, color="#b6501c", alpha=0.45, label="Initial trigger")
        if ax is axes[-1]:
            for j, (u, v) in enumerate(intervals(~usable[s:e])):
                ax.axvspan(
                    (u + s) / fs,
                    (v + s) / fs,
                    facecolor="none",
                    hatch="///",
                    edgecolor="#76838b",
                    label="Excluded from task" if j == 0 else None,
                )
        ax.set(xlim=[s / fs, e / fs], ylabel=case["unit"], xlabel="Block time (s)", title=title)
        ax.ticklabel_format(axis="x", useOffset=False)
    axes[1].legend(loc="best", fontsize=8)
    axes[-1].legend(loc="best", fontsize=8)
    axes[-1].set_ylim(axes[1].get_ylim())
    fig.savefig(out / "seed-comparison.png", dpi=120)
    write(
        out / "seed-comparison.json",
        dict(
            case_id=case["id"],
            version=post_version,
            context_samples=[left, right],
            detail_samples=detail,
            seed_samples=seed,
            suspected_samples=target,
            task_excluded_within_detail=[
                [u + detail[0], v + detail[0]] for u, v in intervals(~usable[detail[0] : detail[1]])
            ],
            numeric_changed_within_detail=int(
                np.count_nonzero(
                    ~np.isclose(
                        raw[detail[0] : detail[1]], values[detail[0] : detail[1]], rtol=0, atol=0, equal_nan=True
                    )
                )
            ),
            result=result,
            image_sha256=sha(out / "seed-comparison.png"),
        ),
    )
