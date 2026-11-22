"""Read-only figures for locator comparisons; plotting never performs detection.

``window`` in :func:`render_case` is a zero-based, half-open sample interval.
``times`` can supply a source time axis in seconds; otherwise the axis is relative.
Sphere inputs are the saved Python mesh, never the attachment's JS detector.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np
from matplotlib import colormaps, colors, font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_PLOT_LOCK = threading.RLock()
_METHOD_COLORS = ("#377eb8", "#e28e2c", "#8a60a8", "#158b75")
_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def _style():
    family = font_manager.FontProperties(fname=_FONT).get_name() if _FONT.exists() else "DejaVu Sans"
    return {"font.family": family, "font.size": 9, "axes.unicode_minus": False, "pdf.fonttype": 42}


def _ranges(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _mask(ranges, n):
    mask = np.zeros(n, bool)
    for a, b in ranges:
        if not 0 <= a <= b <= n:
            raise ValueError(f"Invalid half-open interval: {(a, b)} for {n} samples")
        mask[a:b] = True
    return mask


def _time_runs(times, fs):
    boundaries = np.flatnonzero(np.abs(np.diff(times) - 1 / fs) > 1e-6) + 1
    edges = np.r_[0, boundaries, len(times)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _extrema_indices(values, start, stop, bins):
    """Keep first/min/max/last observations in each bin, never bin means."""
    width = max(1, int(np.ceil((stop - start) / max(1, bins))))
    indices = []
    for a in range(start, stop, width):
        b = min(stop, a + width)
        block = values[a:b]
        indices.extend(sorted({a, b - 1, a + int(np.argmin(block)), a + int(np.argmax(block))}))
    return np.asarray(indices, dtype=np.int64)


def _draw_trace(ax, values, times, fs, start, stop, bins=1700):
    selected = 0
    for a, b in _time_runs(times, fs):
        a, b = max(start, int(a)), min(stop, int(b))
        if b <= a:
            continue
        for left, right in _ranges(np.isfinite(values[a:b])):
            left, right = a + int(left), a + int(right)
            ids = _extrema_indices(values, left, right, bins)
            ax.plot(
                times[ids], values[ids], color="#374955", lw=0.8, marker="." if len(ids) == 1 else None, markersize=2
            )
            selected += len(ids)
    return selected


def _shade_mask(ax, mask, times, fs, start, stop, color, *, alpha=0.65, hatch=None, band=False):
    for left, right in _time_runs(times, fs):
        left, right = max(start, int(left)), min(stop, int(right))
        if right <= left:
            continue
        for a, b in _ranges(mask[left:right]):
            a, b = left + int(a), left + int(b)
            ax.axvspan(
                times[a],
                times[b - 1] + 1 / fs,
                ymin=0.17 if band else 0,
                ymax=0.83 if band else 1,
                facecolor=color,
                alpha=alpha,
                edgecolor=color if hatch else "none",
                hatch=hatch,
                lw=0,
            )


def _write_figure(fig, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def render_case(values, fs, results, out_path, *, case_label, reference=None, times=None, window=None):
    """Plot one common raw window and per-method candidate ribbons.

    ``results`` maps method names to the unified locator JSON (events contain
    integer ``start``/``end``; coverage uses ``evaluated_ranges``). A reference
    mapping may contain synthetic ``A``; it never enters any detector here.
    """
    started = time.perf_counter()
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Expected a nonempty single channel and positive fs")
    n = len(values)
    time_axis = np.arange(n, dtype=float) / fs if times is None else np.asarray(times, float)
    if time_axis.shape != (n,) or not np.isfinite(time_axis).all() or np.any(np.diff(time_axis) <= 0):
        raise ValueError("Source times must be finite, increasing, and match the samples")
    start, stop = (0, n) if window is None else tuple(map(int, window))
    if not 0 <= start < stop <= n:
        raise ValueError("window must be a zero-based half-open sample interval")
    truth = None
    if reference is not None:
        truth = np.asarray(reference["A"] if isinstance(reference, dict) else reference, bool).reshape(-1)
        if truth.shape != (n,):
            raise ValueError("Reference A does not match the signal")
    with _PLOT_LOCK, rc_context(_style()):
        rows = 1 + int(truth is not None) + len(results)
        fig = Figure(figsize=(13.5, 3.4 + 0.8 * (rows - 1)), facecolor="white")
        FigureCanvasAgg(fig)
        axes = np.atleast_1d(
            fig.subplots(rows, 1, sharex=True, gridspec_kw={"height_ratios": [3.0] + [0.65] * (rows - 1)})
        )
        shown = _draw_trace(axes[0], values, time_axis, fs, start, stop)
        _shade_mask(axes[0], ~np.isfinite(values), time_axis, fs, start, stop, "#9b9b9b", alpha=0.22)
        axes[0].set_ylabel("Original value\n(source units)")
        axes[0].grid(alpha=0.18)
        axes[0].set_title(f"{case_label} | {fs:g} Hz | samples [{start}, {stop})", loc="left", fontsize=11)
        row = 1
        if truth is not None:
            _shade_mask(axes[row], truth, time_axis, fs, start, stop, "#b33445", band=True)
            axes[row].set_ylabel("Injected\nreference A", rotation=0, ha="right", va="center")
            axes[row].set_ylim(0, 1)
            axes[row].set_yticks([])
            row += 1
        summaries = {}
        for i, (method, result) in enumerate(results.items()):
            ax = axes[row + i]
            predictions = _mask([(int(e["start"]), int(e["end"])) for e in result.get("events", [])], n)
            unavailable = _mask(result.get("source_unavailable_ranges", []), n)
            coverage_known = result.get("coverage_known", False)
            evaluated = _mask(result.get("evaluated_ranges", []), n) if coverage_known else None
            status = str(result.get("status", "unknown"))
            if evaluated is not None:
                _shade_mask(
                    ax,
                    ~evaluated & ~unavailable,
                    time_axis,
                    fs,
                    start,
                    stop,
                    "#c5c9ce",
                    alpha=0.4,
                    hatch="////",
                    band=True,
                )
            elif status in {"failed", "timeout"}:
                _shade_mask(
                    ax, np.ones(n, bool), time_axis, fs, start, stop, "#c5c9ce", alpha=0.4, hatch="////", band=True
                )
            _shade_mask(ax, unavailable, time_axis, fs, start, stop, "#8b939b", alpha=0.3, band=True)
            _shade_mask(ax, predictions, time_axis, fs, start, stop, _METHOD_COLORS[i % len(_METHOD_COLORS)], band=True)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            label = result.get("method_label", method)
            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
            alarm_s = float(predictions[start:stop].sum() / fs)
            covered_s = float(evaluated[start:stop].sum() / fs) if evaluated is not None else None
            coverage_text = f"scored {covered_s:.3g}s" if covered_s is not None else "coverage unknown"
            ax.text(
                1.005,
                0.5,
                f"{status}\nalarm {alarm_s:.3g}s\n{coverage_text}",
                transform=ax.transAxes,
                fontsize=8,
                va="center",
            )
            ax.grid(axis="x", alpha=0.16)
            summaries[method] = {
                "status": status,
                "alarm_s_in_window": alarm_s,
                "evaluated_s_in_window": covered_s,
                "events_intersecting_window": sum(
                    e["start"] < stop and e["end"] > start for e in result.get("events", [])
                ),
            }
        for ax in axes:
            ax.set_xlim(time_axis[start], time_axis[stop - 1] + 1 / fs)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1].set_xlabel("Source time [s]" if times is not None else "Relative time [s]")
        note = (
            "Synthetic reference is shown only for evaluation. Candidate alarms are not confirmed artifact types."
            if truth is not None
            else "Real waveform: no independent contamination reference. These figures do not measure localization accuracy."
        )
        fig.text(
            0.135,
            0.03,
            note + "\nGray hatch: unscored; gray fill: source unavailable. Original waveform unchanged.",
            fontsize=8,
            color="#526372",
        )
        fig.subplots_adjust(left=0.14, right=0.86, top=0.91, bottom=0.17, hspace=0.42)
        result = _write_figure(fig, out_path)
        fig.clear()
    result.update(
        window_samples=[start, stop],
        relative_time_s=[start / fs, stop / fs],
        source_time_s=[float(time_axis[start]), float(time_axis[stop - 1] + 1 / fs)],
        input_samples=n,
        drawn_waveform_points=shown,
        waveform_downsampling="first_min_max_last_per_bin",
        reference_available=truth is not None,
        methods=summaries,
        render_elapsed_s=time.perf_counter() - started,
    )
    return result


def _display_pool(scores, valid, max_rows=220, max_columns=128):
    """Visual max pooling preserves isolated high scores; numerical scores stay untouched."""
    k, m = scores.shape
    row_edges = np.unique(np.linspace(0, k, min(k, max_rows) + 1, dtype=int))
    col_edges = np.unique(np.linspace(0, m, min(m, max_columns) + 1, dtype=int))
    pooled = np.full((len(row_edges) - 1, len(col_edges) - 1), np.nan)
    for i, (a, b) in enumerate(zip(row_edges[:-1], row_edges[1:], strict=True)):
        for j, (c, d) in enumerate(zip(col_edges[:-1], col_edges[1:], strict=True)):
            usable = valid[a:b, c:d] & np.isfinite(scores[a:b, c:d])
            if usable.any():
                pooled[i, j] = float(scores[a:b, c:d][usable].max())
    return pooled, row_edges, col_edges


def render_sphere(mesh, out_path, *, case_label, fs=125, sample_offset=0):
    """Render saved positive-radius geometry beside its cycle/phase score map.

    Required K x M fields: display_radius, valid_cell, scores, sample_positions.
    theta and phi are optional radian coordinates with shapes (K, 1)/(1, M).
    The score color scale clips visually at 2 x threshold, never numerically.
    """
    started = time.perf_counter()
    radius = np.asarray(mesh["display_radius"], float)
    valid = np.asarray(mesh["valid_cell"], bool)
    scores = np.asarray(mesh["scores"], float)
    positions = np.asarray(mesh["sample_positions"], float)
    if radius.ndim != 2 or not radius.size or any(a.shape != radius.shape for a in (valid, scores, positions)):
        raise ValueError("Sphere fields must be matching, nonempty K x M arrays")
    if not np.isfinite(fs) or fs <= 0 or np.any(valid & (~np.isfinite(radius) | (radius <= 0))):
        raise ValueError("Valid sphere cells need positive finite display radii and fs")
    k, m = radius.shape
    threshold = float(mesh.get("threshold", 3.2))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Positive score threshold required")
    theta = np.broadcast_to(np.asarray(mesh.get("theta", np.linspace(0.12, np.pi - 0.12, k)[:, None])), radius.shape)
    phi = np.broadcast_to(np.asarray(mesh.get("phi", np.arange(m)[None, :] * (2 * np.pi / m))), radius.shape)
    pooled, row_edges, col_edges = _display_pool(scores / threshold, valid)
    row_ids = np.unique(np.linspace(0, k - 1, min(k, 160), dtype=int))
    col_ids = np.unique(np.linspace(0, m - 1, min(m, 128), dtype=int))
    selected = np.ix_(row_ids, col_ids)
    r = np.where(valid, radius, np.nan)[selected]
    th, ph = theta[selected], phi[selected]
    surface_score = scores[selected] / threshold
    x, y, z = r * np.sin(th) * np.cos(ph), r * np.cos(th), r * np.sin(th) * np.sin(ph)
    # Close longitude only. The first/last cycle are never made neighbours.
    x, y, z, surface_score = [np.c_[a, a[:, :1]] for a in (x, y, z, surface_score)]
    color_map = colormaps["viridis"].copy()
    color_map.set_bad("#b8bfc5")
    norm = colors.Normalize(vmin=0, vmax=2, clip=True)
    scored = valid & np.isfinite(scores) & np.isfinite(positions)
    peak = np.unravel_index(np.argmax(np.where(scored, scores, -np.inf)), scores.shape) if scored.any() else None
    peak_time = float((positions[peak] + sample_offset) / fs) if peak else None
    with _PLOT_LOCK, rc_context(_style()):
        fig = Figure(figsize=(13.5, 6.4), facecolor="white")
        FigureCanvasAgg(fig)
        heat_ax = fig.add_subplot(121)
        sphere_ax = fig.add_subplot(122, projection="3d")
        heat = heat_ax.pcolormesh(
            col_edges / m,
            row_edges,
            np.ma.masked_invalid(pooled),
            cmap=color_map,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        heat_ax.set_ylim(k, 0)
        heat_ax.set_xlabel("Phase within one estimated cycle")
        heat_ax.set_ylabel("Estimated cycle index (not detected beats)")
        heat_ax.set_title("Cycle / phase view of the same scores", loc="left", fontsize=10)
        if peak:
            heat_ax.scatter(
                (peak[1] + 0.5) / m,
                peak[0] + 0.5,
                marker="o",
                s=34,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
            )
        ticks = np.unique(np.linspace(0, k - 1, min(k, 5), dtype=int))
        labels = []
        for row in ticks:
            usable = valid[row] & np.isfinite(positions[row])
            labels.append(f"{(positions[row, usable].min() + sample_offset) / fs:.3f}" if usable.any() else "unscored")
        time_ax = heat_ax.twinx()
        time_ax.set_ylim(k, 0)
        time_ax.set_yticks(ticks + 0.5, labels)
        time_ax.set_ylabel("First sample time of labelled row [s]", fontsize=8)
        if k >= 2 and m >= 2:
            sphere_ax.plot_surface(
                x,
                y,
                z,
                facecolors=color_map(norm(np.ma.masked_invalid(surface_score))),
                rstride=1,
                cstride=1,
                shade=True,
                lightsource=colors.LightSource(azdeg=315, altdeg=35),
                linewidth=0,
                antialiased=False,
            )
            sphere_ax.plot_wireframe(
                x,
                y,
                z,
                rstride=max(1, len(row_ids) // 10),
                cstride=max(1, len(col_ids) // 12),
                linewidth=0.25,
                color="#77848c",
                alpha=0.4,
            )
        else:
            keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            sphere_ax.scatter(
                x[keep], y[keep], z[keep], s=8, c=color_map(norm(np.ma.masked_invalid(surface_score)))[keep]
            )
        sphere_ax.set_box_aspect((1, 1, 1))
        if peak:
            px = float(radius[peak] * np.sin(theta[peak]) * np.cos(phi[peak]))
            py = float(radius[peak] * np.cos(theta[peak]))
            pz = float(radius[peak] * np.sin(theta[peak]) * np.sin(phi[peak]))
            sphere_ax.view_init(
                elev=float(np.degrees(np.arctan2(pz, np.hypot(px, py)))), azim=float(np.degrees(np.arctan2(py, px)))
            )
            sphere_ax.scatter([px], [py], [pz], s=22, color="black", depthshade=False)
        else:
            sphere_ax.view_init(elev=22, azim=38)
        sphere_ax.set_xlabel("Display X")
        sphere_ax.set_ylabel("Display Y")
        sphere_ax.set_zlabel("Display Z")
        sphere_ax.set_title("Positive-radius display; detection does not use this projection", fontsize=9)
        bar_axis = fig.add_axes([0.29, 0.16, 0.43, 0.025])
        bar = fig.colorbar(heat, cax=bar_axis, orientation="horizontal")
        bar.set_label("Heuristic score / threshold (display clipped at 2; not a probability)", fontsize=9)
        fig.suptitle(f"{case_label} | full score grid {k} x {m}", fontsize=12, y=0.98)
        offset = float(mesh.get("display_radius_offset", 0.0))
        peak_label = f"Camera faces peak-score sample at {(peak_time):.3f}s. " if peak else "No scored cell. "
        foot = (
            f"Rendering only: surface {len(row_ids)} x {len(col_ids)} cells; "
            f"heatmap max-pooled to {pooled.shape[0]} x {pooled.shape[1]}. Display radius offset = {offset:.6g}.\n"
            + peak_label
            + "Other side remains occluded. Surface lighting is visual only; quantitative colors are in the 2D map."
        )
        fig.text(0.06, 0.04, foot, fontsize=8, color="#526372")
        fig.subplots_adjust(left=0.07, right=0.94, top=0.86, bottom=0.3, wspace=0.42)
        result = _write_figure(fig, out_path)
        fig.clear()
    finite_positions = positions[valid & np.isfinite(positions)]
    result.update(
        full_grid_shape=[k, m],
        rendered_surface_shape=[len(row_ids), len(col_ids)],
        heatmap_shape=list(pooled.shape),
        heatmap_row_edges=row_edges.tolist(),
        heatmap_phase_bin_edges=col_edges.tolist(),
        rendered_cycle_indices=row_ids.tolist(),
        rendered_phase_indices=col_ids.tolist(),
        sample_offset=int(sample_offset),
        sample_range=[float(finite_positions.min() + sample_offset), float(finite_positions.max() + sample_offset)]
        if finite_positions.size
        else None,
        display_radius_offset=offset,
        peak_sample_time_s=peak_time,
        camera_basis="highest_scored_valid_cell" if peak else "fixed_default",
        score_threshold=threshold,
        display_score_clip=2.0,
        valid_cell_count=int(valid.sum()),
        unscored_cell_count=int((~np.isfinite(scores)).sum()),
        render_elapsed_s=time.perf_counter() - started,
    )
    return result
