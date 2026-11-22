"""Actual same-window pre/post evidence, persisted and bound to candidate content."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.followup.pattern import context_range
from tsmcp.processing import intervals

from .common import digest, read, sha, write
from .engine import load_arrays


def select_cases(version, cases):
    available = [c for c in cases if c["id"] in version["results"]]
    if not available:
        return []
    results = version["results"]
    picks = {}

    def add(case, reason, focus=None):
        focus = case["n"] // 2 if focus is None else focus
        picks.setdefault((case["id"], focus), []).append(reason)

    problem = max(available, key=lambda c: results[c["id"]]["stats"]["remaining_issue_count"])
    issues = results[problem["id"]]["diagnosis"]["issues"]
    add(problem, "remaining_issues", sum(issues[0]["range_samples"]) // 2 if issues else None)
    modified = [c for c in available if results[c["id"]].get("change_focus_sample") is not None]
    if modified:
        edited = max(
            modified,
            key=lambda c: (
                results[c["id"]]["stats"]["modified_samples"] + results[c["id"]]["stats"]["support_changed_samples"]
            ),
        )
        add(edited, "modified_cases", results[edited["id"]]["change_focus_sample"])
    unchanged = [c for c in available if results[c["id"]].get("change_focus_sample") is None]
    add(
        min(unchanged or available, key=lambda c: digest(c["id"])),
        "unchanged_controls" if unchanged else "control_unavailable",
    )
    add(min(available, key=lambda c: digest([111101, c["id"]])), "random_cases")
    add(min(available, key=lambda c: c["n"]), "boundaries", 0)
    return [{"case_id": key[0], "focus_sample": key[1], "reasons": reasons} for key, reasons in picks.items()]


def render(workspace, version, case, reasons=(), start_s=None, end_s=None, focus_sample=None, target=None, flank_s=5):
    workspace = Path(workspace)
    artifact = version["results"][case["id"]]
    if artifact["arrays"] and sha(workspace / artifact["arrays"]) != artifact["arrays_sha256"]:
        raise ValueError("Candidate arrays changed before inspection")
    raw, times, current, valid, estimated = load_arrays(workspace, case, artifact)
    parent = None
    if version["parent_version"]:
        parent = read(workspace / "versions" / version["parent_version"] / "version.json")["results"].get(case["id"])
    previous = load_arrays(workspace, case, parent)[2]
    fs, n = case["fs"], case["n"]
    if target is not None and (start_s is not None or end_s is not None):
        raise ValueError("Select a target and flank, or an exact viewport")
    if target is not None:
        a, b = context_range(n, fs, target, flank_s)
    elif start_s is None:
        if focus_sample is None:
            focus_sample = artifact.get("change_focus_sample")
        difference = np.nan_to_num(np.abs(current - previous))
        issues = artifact["diagnosis"]["issues"]
        center = (
            int(np.argmax(difference))
            if difference.any()
            else (sum(issues[0]["range_samples"]) // 2 if issues else n // 2)
        )
        if focus_sample is not None:
            center = focus_sample
        elif "boundaries" in reasons:
            center = 0
        changed_ranges = artifact.get("support_changed_ranges", []) + [
            span for op in artifact.get("operations", []) for span in op["ranges_samples"]
        ]
        regions = changed_ranges + [issue["range_samples"] for issue in issues]
        target = next((r for r in regions if r[0] <= center < r[1]), [center, center + 1])
        a, b = context_range(n, fs, target, flank_s)
    else:
        a, b = round(start_s * fs), min(n, round((end_s if end_s is not None else start_s + 10) * fs))
    if not 0 <= a < b <= n:
        raise ValueError("Inspection interval is outside this continuous run")
    key = digest([version["version_id"], artifact["content_hash"], a, b, target, flank_s, list(reasons)])[:24]
    folder = workspace / "evidence" / version["version_id"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + ".png")
    fig = Figure(figsize=(11, 5.5), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    t = np.arange(a, b) / fs
    axes[0].plot(t, raw[a:b], color="#888888", lw=0.8, label="Original")
    axes[0].plot(t, previous[a:b], color="#D18C18", lw=0.8, alpha=0.7, label="Parent")
    axes[0].plot(t, current[a:b], color="#0072B2", lw=0.9, label="Candidate")
    p = read(workspace / artifact["probe_path"])
    points = np.asarray([e["sample"] for e in p["events"] if a <= e["sample"] < b], int)
    axes[0].scatter(
        points / fs, current[points], s=20, facecolors="none", edgecolors="#009E73", label="Frozen probe events"
    )
    if target is not None:
        axes[0].axvspan(target[0] / fs, target[1] / fs, color="#df952f", alpha=0.12, label="Inspection target")
    axes[0].set_ylabel(case["unit"])
    if "patient_reference" in case:
        amp = case["patient_reference"]["amplitude"]
        axes[0].set_ylim(*amp["display"])
        for value in (amp["low"], amp["high"]):
            axes[0].axhline(value, ls="--", lw=0.7, color="#a53456")
        for segment in (raw[a:b], current[a:b]):
            for limit, mask, marker in (
                (amp["display"][0], segment < amp["display"][0], "^"),
                (amp["display"][1], segment > amp["display"][1], "v"),
            ):
                if np.any(mask):
                    axes[0].plot(t[mask], np.full(mask.sum(), limit), marker, color="#a53456", clip_on=False, ms=4)
    axes[0].legend(fontsize=8, ncol=4)
    axes[1].plot(t, current[a:b] - previous[a:b], color="#0072B2", lw=0.8, label="Candidate - parent")
    axes[1].fill_between(
        t,
        0,
        1,
        where=~valid[a:b],
        color="#E78A8A",
        alpha=0.3,
        transform=axes[1].get_xaxis_transform(),
        label="Invalid support",
    )
    axes[1].fill_between(
        t,
        0,
        1,
        where=estimated[a:b],
        color="#E6B450",
        alpha=0.3,
        transform=axes[1].get_xaxis_transform(),
        label="Estimated; excluded from task output",
    )
    axes[1].set_xlabel("Relative seconds within this true continuous run")
    axes[1].set_ylabel("Residual")
    axes[1].legend(fontsize=8)
    # A one-sample invalid interval has visible width; fill_between alone hides isolated marks.
    for left, right in intervals(~valid[a:b]):
        axes[0].axvspan((a + left) / fs, (a + right) / fs, color="#E78A8A", alpha=0.25, lw=0)
        axes[1].axvspan((a + left) / fs, (a + right) / fs, color="#E78A8A", alpha=0.25, lw=0)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{case['task']} | {case['id']} | {version['version_id']}\n"
        f"Actual candidate, no clean reference | {', '.join(reasons)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    evidence = {
        "evidence_id": key,
        "version_id": version["version_id"],
        "artifact_id": artifact["artifact_id"],
        "content_hash": artifact["content_hash"],
        "case_id": case["id"],
        "task": case["task"],
        "range_samples": [a, b],
        "focus_sample": focus_sample,
        "target_samples": target,
        "flank_s": flank_s if target is not None else None,
        "reasons": list(reasons),
        "image": str(path.relative_to(workspace)),
        "image_sha256": sha(path),
        "probe_sha256": artifact["probe_sha256"],
        "stats": artifact["stats"],
    }
    write(folder / (key + ".json"), evidence)
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(path.read_bytes()).decode()},
    }
    return evidence, block
