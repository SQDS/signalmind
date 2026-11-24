"""Waveform evidence for a proposed short-core interpolation. Reference beats are observed neighbours."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.processing import intervals


def phase_neighbours(x, good, events, fs, a, b):
    """Align nearby cycles to the target cycle; exclude its samples and never cross untrusted support."""
    radius = max(4, round(0.075 * fs))
    indices = np.arange(max(0, a - radius), min(len(x), b + radius))
    points = np.asarray([z["sample"] for z in events], dtype=int)
    j = int(np.searchsorted(points, a))
    if not j or j == len(points) or points[j] < b:
        return indices, [], [], "core lacks two distinct bracketing landmarks"
    left, right = points[j - 1 : j + 1]
    # Restrict the display and reference to the interior of this cycle.
    indices = indices[(indices > left) & (indices < right)]
    core = (indices >= a) & (indices < b)
    if not core.any() or min((indices < a).sum(), (indices >= b).sum()) < 2 or not good[left : right + 1].all():
        return indices, [], [], "insufficient observed flanks in the target cycle"
    phase = (indices - left) / (right - left)
    candidates = []
    for start, stop in zip(points[:-1], points[1:], strict=True):
        if start == left or max(left - stop, start - right) > 8 * fs or abs((stop - start) / (right - left) - 1) > 0.2:
            continue
        if not good[min(start, left) : max(stop, right) + 1].all():
            continue
        candidates.append((abs(start - left), start, stop))
    ref = []
    cycles = []
    for _, start, stop in sorted(candidates)[:6]:
        at = start + phase * (stop - start)
        values = np.interp(at, np.arange(start, stop + 1), x[start : stop + 1])
        values += np.median(x[indices[~core]] - values[~core])
        ref.append(values)
        cycles.append([int(start), int(stop)])
    if len(ref) < 3:
        return indices, [], cycles, "fewer than three comparable observed cycles"
    return indices, np.stack(ref), cycles, None


def spike_view(path, loc, cases, peers, before, after):
    cid = loc["case"]
    c = cases[cid]
    fs = c["fs"]
    a, b = loc["core_samples"]
    t, x, v, e, pr = before(cid)
    _, y, valid, estimated, _ = after(cid)
    diag = loc["spike_retest"]
    start = t[a]
    indices, ref, cycles, why = phase_neighbours(x, v & ~e & np.isfinite(x), pr["events"], fs, a, b)
    cx = np.arange(a - 1, b + 1)
    full_width = max(2.0, loc.get("context_s", 2.0))
    fig = Figure(figsize=(11.8, 6.5), layout="constrained")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.1, 1))
    left_grid = grid[0, 0].subgridspec(len(peers), 1)
    right_grid = grid[0, 1].subgridspec(2, 1)
    peer_landmarks = {}
    for i, peer in enumerate(peers):
        pt, z, pv, pe, ppr = before(peer["id"])
        ax = fig.add_subplot(left_grid[i, 0])
        l0, r0 = pt.searchsorted([start - full_width, start + full_width])
        ax.plot(pt[l0:r0] - start, z[l0:r0], color="#35556d", lw=0.85)
        ax.axvspan(-0.5 / fs, (b - a - 0.5) / fs, color="#e6ae52", alpha=0.45, lw=0)
        for m, n in intervals((~pv | pe)[l0:r0]):
            ax.axvspan(pt[l0 + m] - start, pt[l0 + n - 1] - start + 1 / peer["fs"], color="#bcbcbc", alpha=0.45, lw=0)
        points = [p["sample"] for p in ppr["events"] if l0 <= p["sample"] < r0]
        if points:
            ax.plot(pt[points] - start, z[points], "v", ms=4, mfc="none", mec="#387a69", label="Detected landmarks")
        peer_landmarks[peer["task"]] = [round((pt[q] - start) * 1000, 2) for q in points]
        ax.set_ylabel(peer["task"] + "\n" + peer["unit"])
        ax.set_xlim(-full_width, full_width)
        if i == 0:
            ax.set_title("Observed context | triangles: detected landmarks", fontsize=10)
        if i == len(peers) - 1:
            ax.set_xlabel("Seconds from first core sample")
    detail = fig.add_subplot(right_grid[0, 0])
    phase_ax = fig.add_subplot(right_grid[1, 0], sharex=detail)
    pad = max(4, round(0.075 * fs))
    ix = np.arange(max(0, a - pad), min(len(x), b + pad))

    def plot_samples(ax):
        ax.plot((ix - a) * 1000 / fs, x[ix], ".-", color="#385b78", lw=1, ms=4, label="Observed samples")
        ax.plot(
            (cx - a) * 1000 / fs,
            y[cx],
            "o--",
            color="#ca5930",
            lw=1.4,
            ms=4,
            mfc="none",
            label="Candidate (anchors + core)",
        )
        ax.axvspan(-500 / fs, (b - a - 0.5) * 1000 / fs, color="#edd5a0", alpha=0.40, lw=0)
        ax.set_ylabel(c["unit"])

    plot_samples(detail)
    detail.set_title(f"{c['task']} | {b - a} core samples, {(b - a) * 1000 / fs:g} ms", fontsize=10)
    detail.legend(fontsize=8, loc="best")
    background = dict(
        method="neighbouring cycles aligned by landmark phase; offset matched on measured flanks",
        cycles=cycles,
        available=why is None,
    )
    if why is None:
        med = np.median(ref, axis=0)
        low, high = np.percentile(ref, [10, 90], axis=0)
        mask = (indices >= a) & (indices < b)
        phase_ax.fill_between((indices - a) * 1000 / fs, low, high, color="#add1c1", alpha=0.7)
        phase_ax.plot((indices - a) * 1000 / fs, med, color="#307558", lw=1.35, label="Neighbour median (10-90% band)")
        phase_ax.plot(
            (cx - a) * 1000 / fs,
            y[cx],
            "o--",
            color="#ca5930",
            lw=1.35,
            ms=4,
            mfc="none",
            label="Candidate and anchors",
        )
        phase_ax.axvspan(-500 / fs, (b - a - 0.5) * 1000 / fs, color="#edd5a0", alpha=0.36, lw=0)
        phase_ax.set_ylabel(c["unit"])
        phase_ax.set_title(f"Same phase in {len(cycles)} observed neighbouring cycles", fontsize=10)
        phase_ax.legend(fontsize=7, loc="best")
        background.update(
            median_core=med[mask].tolist(),
            low_core=low[mask].tolist(),
            high_core=high[mask].tolist(),
            observed_mae_to_median=float(np.mean(abs(x[a:b] - med[mask]))),
            flank_mae_to_median=float(np.mean(abs(x[indices[~mask]] - med[~mask]))),
            candidate_mae_to_median=float(np.mean(abs(y[a:b] - med[mask]))),
        )
    else:
        phase_ax.plot((ix - a) * 1000 / fs, y[ix] - x[ix], ".-", color="#705583", ms=4)
        phase_ax.set(title="Candidate minus observed | phase reference unavailable", ylabel=c["unit"])
        background["reason"] = why
    phase_ax.set_xlabel("Milliseconds from first core sample")
    fig.suptitle("Local interpolation | amber: proposed write | core values remain present as estimates", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    fig.clear()
    parent_marks = peer_landmarks[c["task"]]
    return dict(
        core_samples=[a, b],
        preceding_native_ms=max((v for v in parent_marks if v < 0), default=None),
        following_native_ms=min((v for v in parent_marks if v >= (b - a) * 1000 / fs - 1e-5), default=None),
        parent_landmarks_in_core=[q for q in diag["before_landmarks"] if a <= q < b],
        candidate_landmarks_in_core=[q for q in diag["after_landmarks"] if a <= q < b],
        fs_hz=fs,
        unit=c["unit"],
        parent_core=x[a:b].tolist(),
        candidate_core=y[a:b].tolist(),
        observed_anchors=x[[a - 1, b]].tolist(),
        neighbour_phase=background,
        observed_landmark_offsets_ms=peer_landmarks,
        same_support_landmarks_ms={
            key: [round((q - a) * 1000 / fs, 2) for q in diag[key]] for key in ("before_landmarks", "after_landmarks")
        },
        candidate_estimated_samples=int(estimated[a:b].sum()),
        candidate_invalid_samples=int((~valid[a:b]).sum()),
        release_support_policy=(
            "Candidate core values are present and estimated. Intervals spanning an estimate "
            "are omitted from measured-only output even when diagnostic landmarks stay in place."
        ),
    )
