"""Thin MCP adapter over tsmcp.core. The single source of tool definitions."""

from __future__ import annotations

import argparse
import base64
import math
import threading

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ImageContent, TextContent, ToolAnnotations

from tsmcp.core.compact import dumps, sig, tround
from tsmcp.core.errors import SignalError
from tsmcp.core.evidence import ref
from tsmcp.core.measure import measure
from tsmcp.core.signal import clip_window, slice_channel, time_decimals
from tsmcp.core.store import SignalStore
from tsmcp.core.views.describe import describe
from tsmcp.core.views.render import render, render_series, render_snippets, render_strip, strip_rows
from tsmcp.plugins import all_detectors, all_measures, all_series, all_views
from tsmcp.plugins.optional import OPTIONAL_PLUGINS, plugin_set, pop_artifacts, register_signal
from tsmcp.server.compact_v2 import AF_CONVENTIONS, AF_DESCRIPTION, EpisodePresenter

INSTRUCTIONS = """\
TS-MCP analyzes time-series signals without putting raw samples in your context.
1. ts_load -> id + overview (quality, period, block profile).
2. Ask the question directly: ts_find_events for "when / where / is there" (e.g. discord for "find the anomaly",
   change, flat, ecg_irregular), ts_measure for "how much". Look at one overview image (view="strip" or a series
   view such as ecg_rr) only when shape matters; zoom into a few windows, not the whole record piece by piece.
3. Event lists are candidates, not confirmation. Compare them with ts_find_events(snippets=True), then
   ts_zoom(view="verify", window=[t0,t1]) to confirm or reject a candidate. Respect detector caveats.
   Never read numbers off an image; ts_measure them. Cite `ref`s.
4. Budget: most questions need fewer than 10 tool calls. Do not sweep a record window by window with
   ts_measure; if a direct tool has not answered after ~15 calls, give your best answer with your confidence.
If quality shows high nan_frac/flat_frac/clip_frac, say so and lower your confidence.
Times are seconds from record start; windows are [t0, t1).
"""

RO = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
CALL_BUDGET = 20  # after this many calls on one signal, every reply carries a nudge to conclude
MAX_SERIES_POINTS_TEXT = 120


def build_server(
    store: SignalStore,
    images: bool = True,
    interface: str = "v1",
    plugins: list[str] | tuple[str, ...] | None = None,
    runtime: str = "off",
    runtime_dir: str | None = None,
    task_spec: str | None = None,
    run_id: str | None = None,
) -> MCPServer:
    """`plugins` adds opt-in plugins (e.g. "pressure"); `runtime` = off (default) | log | verify (SignalMind
    evidence ledger and claim verification). With the defaults the tool list and descriptions are unchanged."""
    if interface not in ("v1", "v2"):
        raise ValueError("interface must be 'v1' or 'v2'")
    if runtime not in ("off", "log", "verify"):
        raise ValueError("runtime must be 'off', 'log' or 'verify'")
    rt = None
    if runtime != "off":
        from tsmcp.server.runtime_tools import RuntimeAdapter

        rt = RuntimeAdapter(runtime, runtime_dir, task_spec, run_id)
    srv = MCPServer("tsmcp", instructions=INSTRUCTIONS + (rt.instructions() if rt else ""))
    episodes = EpisodePresenter() if interface == "v2" else None
    PLUGINS = plugin_set(plugins)  # noqa: N806 - keeps the doc-building code below unchanged
    optional = bool(plugins)
    measures, detectors, series_reg, views = (
        all_measures(PLUGINS),
        all_detectors(PLUGINS),
        all_series(PLUGINS),
        all_views(PLUGINS),
    )
    last_events: dict[str, list[dict]] = {}  # per signal id, for overlaying on later images (convenience only)
    ev_lock = threading.Lock()

    calls: dict[str, int] = {}

    def text(obj) -> TextContent:
        if interface == "v2":
            obj = {**obj, "interface_version": "v2"}
        sid = obj.get("id") if isinstance(obj, dict) else None
        if sid:
            with ev_lock:
                calls[sid] = calls.get(sid, 0) + 1
                n = calls[sid]
            if n >= CALL_BUDGET:
                obj = {**obj, "budget_note": f"{n} tool calls on this signal so far; give your best answer now "
                       "unless one more specific call is clearly needed"}  # fmt: skip
        return TextContent(type="text", text=dumps(obj))

    def image(png: bytes) -> ImageContent:
        return ImageContent(type="image", data=base64.b64encode(png).decode(), mimeType="image/png")

    def guard(fn):
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except SignalError as e:
                raise ToolError(str(e)) from None

        wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
        wrapper.__annotations__ = fn.__annotations__
        wrapper.__wrapped__ = fn
        return wrapper

    def first_time(c, t0):
        return max(0, math.floor(t0 * c.fs)) / c.fs  # time of the first sample slice_channel returns

    @srv.tool(annotations=RO, structured_output=False)
    @guard
    def ts_load(
        source: str | None = None,
        data: list[float] | None = None,
        fs: float | None = None,
        format: str | None = None,
        channels: list[str] | None = None,
        units: list[str] | None = None,
    ) -> list[TextContent]:
        """Load a signal and return its id plus a compact overview (same as ts_describe).

        Pass `source` (local file: .csv, .npy, .edf, or WFDB .hea/.dat) OR `data` (small inline 1-D array).
        `fs` = sampling rate in Hz (required for .npy and csv without a time column; EDF/WFDB carry their own).
        `channels`: names to keep (EDF/WFDB) or to assign (csv/npy). The id is stable across calls and restarts."""
        s = store.load(source=source, data=data, fs=fs, format=format, channels=channels, units=units)
        return [text(describe(s))]

    @srv.tool(annotations=RO, structured_output=False)
    @guard
    def ts_describe(id: str, detail: str = "normal") -> list[TextContent]:
        """Overview of a loaded signal (~300 tokens/channel): duration, per-channel fs/unit, robust stats,
        defined quality fractions (nan/flat/clip), period on a leading probe window, and a block profile
        per equal time block (length dt_s): mean, std, and cross_hz (oscillation frequency from median crossings).
        The block profile shows WHERE level, energy or rhythm changes. detail: "normal" (12 blocks) or "high" (24)."""
        return [text(describe(store.get(id), detail))]

    view_doc = (
        "Look at window [t0, t1) seconds as an image. `view` values:\n"
        + "\n".join(f"- {p.name}: {p.views_summary}" for p in PLUGINS if p.views_summary)
        + "\nEvents found earlier by ts_find_events are marked in red. Use images to judge shape and patterns; "
        "use ts_measure / ts_find_events for exact numbers and times. detail: normal (1200 px) or high (2400 px). "
        "rows: number of strip rows (strip view, default ~100 s per row, max 6)."
    )

    @srv.tool(annotations=RO, structured_output=False, description=view_doc)
    @guard
    def ts_zoom(
        id: str,
        window: list[float],
        channels: list[str] | None = None,
        view: str = "wave",
        detail: str = "normal",
        rows: int | None = None,
    ) -> list[TextContent | ImageContent]:
        """See view_doc."""
        s = store.get(id)
        t0, t1, warn = clip_window(s, window)
        chans = s.select(channels)
        tdec = time_decimals(max(c.fs for c in chans))
        win = f"{tround(t0, tdec):g}-{tround(t1, tdec):g}"
        with ev_lock:
            evs = [dict(e, label=e["type"]) for e in last_events.get(id, [])]
        head = {"id": id, "view": view, "window": [tround(t0, tdec), tround(t1, tdec)],
                "channels": [c.name for c in chans], "ref": ref(id, f"zoom.{view}", win), "warn": warn}  # fmt: skip
        if rt:
            head = rt.on_view(id, view, [c.name for c in chans], (t0, t1), head)
        title = f"{id} [{win}) s"
        c0 = chans[0]
        if view == "wave":
            if len(chans) > 8:
                head["warn"] = ((warn + "; ") if warn else "") + "only the first 8 channels are drawn"
            if not images:
                head["summary"] = [_minmax(c, t0, t1) for c in chans[:8]]
                return [text(head)]
            return [text(head), image(render(chans, t0, t1, detail, title, evs))]
        if len(chans) > 1:
            head["warn"] = ((head["warn"] + "; ") if head["warn"] else "") + f"view {view} uses one channel: {c0.name}"
        if view == "strip":
            n = strip_rows(t0, t1, rows)
            head["rows"], head["row_s"] = n, sig((t1 - t0) / n)
            if not images:
                head["summary"] = [_minmax(c0, t0 + i * (t1 - t0) / n, t0 + (i + 1) * (t1 - t0) / n) for i in range(n)]
                return [text(head)]
            return [text(head), image(render_strip(c0, t0, t1, detail, title, evs, rows))]
        if view in series_reg:
            ser = series_reg[view](slice_channel(c0, t0, t1), c0.fs, c0.unit, first_time(c0, t0), ch=c0)
            head["series"] = {"label": ser["label"], "unit": ser["unit"], "n": int(len(ser["y"]))}
            if not images:
                head["binned"] = _binned_text(ser, t0, t1, tdec)
                return [text(head)]
            return [text(head), image(render_series([ser], t0, t1, detail, title, evs))]
        if view in views:
            if not images:
                raise SignalError(f"view {view!r} is image-only and images are disabled on this server")
            return [text(head), image(views[view](c0, t0, t1, detail, title, evs))]
        allowed = ["wave", "strip", *series_reg, *views]
        raise SignalError(f"unknown view {view!r}; available: {', '.join(allowed)}")

    measure_doc = (
        "Exact measurements on window [t0, t1) (default: whole record). Each result has value, unit, method and a "
        "`ref` to cite. Call this for every number you report — never read numbers off an image. Several measures "
        "and channels can be requested in one call. `what` values:\n"
        + "\n".join(f"- {p.name}: {p.summary}" for p in PLUGINS)
    )

    @guard
    def ts_measure(
        id: str,
        what: list[str],
        window: list[float] | None = None,
        channels: list[str] | None = None,
    ) -> list[TextContent]:
        """See measure_doc."""
        s = store.get(id)
        t0, t1, warn = clip_window(s, window)
        chans = s.select(channels)
        results = []
        artifacts = [] if rt else None
        if optional:
            register_signal(s)
        for c in chans:
            x, start = slice_channel(c, t0, t1), first_time(c, t0)
            rows = measure(id, c, x, start, t1, what, time_decimals(c.fs), measures)
            if optional:
                pop_artifacts(rows, artifacts)
            if episodes is not None:
                rows = [
                    episodes.index(s, c, x, start, row)
                    if row["name"] == "ecg_af_episodes"
                    else {**row, "method": "AF support"}
                    if row["name"] == "ecg_af_burden"
                    else row
                    for row in rows
                ]
            results += rows
        tdec = time_decimals(max(c.fs for c in chans))
        out = {"id": id, "window": [tround(t0, tdec), tround(t1, tdec)], "warn": warn, "measurements": results}
        if episodes is not None and any(w in ("ecg_af_episodes", "ecg_af_burden") for w in what):
            out["af_conventions"] = AF_CONVENTIONS
        if rt:
            out = rt.on_measure(id, out, results, (t0, t1), artifacts)
        return [text(out)]

    if interface == "v1":
        srv.tool(annotations=RO, structured_output=False, description=measure_doc)(ts_measure)
    else:
        v2_measure_doc = (
            measure_doc.replace("Each result has value", "Numeric measurements have value")
            + " ecg_af_episodes returns every candidate ranked by unrounded span then earliest start. "
            "unknown_s is unreadable time inside the span; merged_from counts AF support intervals. "
            "Request what=['ecg_af_episode_detail'], episode='<id>' for exact v1 evidence and readability; "
            "the id preserves channel and analysis window across restarts." + AF_DESCRIPTION
        )

        @srv.tool(name="ts_measure", annotations=RO, structured_output=False, description=v2_measure_doc)
        @guard
        def ts_measure_v2(
            id: str,
            what: list[str],
            window: list[float] | None = None,
            channels: list[str] | None = None,
            episode: str | None = None,
        ) -> list[TextContent]:
            if episode is None:
                if "ecg_af_episode_detail" in what:
                    raise SignalError("ecg_af_episode_detail requires episode='<id>' from ecg_af_episodes")
                return ts_measure(id, what, window, channels)
            if what != ["ecg_af_episode_detail"] or window is not None or channels is not None:
                raise SignalError("episode detail requires only what=['ecg_af_episode_detail'] and episode='<id>'; "
                                  "its id already specifies the window and channel")  # fmt: skip
            detail = episodes.detail(store.get(id), episode)
            return [text({"id": id, "window": detail["window"], "measurements": [detail],
                          "af_conventions": AF_CONVENTIONS})]  # fmt: skip

    events_doc = (
        "Find candidate events in window [t0, t1) (default: whole record). Each event: "
        "type, t (start, s), dur (s), score, optional detail/caveat, and a `ref`. Found events are marked on later "
        "ts_zoom images. For long records search the whole record first, then re-run on a shorter window for finer "
        "time resolution (change events report resolution_s). snippets=True adds one PNG comparing up to four "
        "highest-scored returned events if images are enabled. Scores are detector-specific; verify candidates "
        "with ts_zoom(view='verify') before concluding there is an anomaly. `types` values:\n"
        + "\n".join(f"- {p.name}: {p.detectors_summary}" for p in PLUGINS if p.detectors_summary)
    )
    if interface == "v2":
        events_doc += " Use ecg_af_episodes via ts_measure for a complete span-ranked AF index." + AF_DESCRIPTION

    @srv.tool(annotations=RO, structured_output=False, description=events_doc)
    @guard
    def ts_find_events(
        id: str,
        types: list[str],
        window: list[float] | None = None,
        channels: list[str] | None = None,
        top_k: int = 15,
        snippets: bool = False,
    ) -> list[TextContent | ImageContent]:
        """See events_doc."""
        unknown = [t for t in types if t not in detectors]
        if unknown:
            raise SignalError(f"unknown event type(s) {unknown}; available: {', '.join(detectors)}")
        s = store.get(id)
        t0, t1, warn = clip_window(s, window)
        chans = s.select(channels) if channels else s.channels[:1]
        found, skipped = [], []
        for c in chans:
            x = slice_channel(c, t0, t1)
            for typ in types:
                try:
                    for e in detectors[typ](x, c.fs, c.unit, first_time(c, t0), ch=c):
                        found.append({**e, "ch": c.name})
                except SignalError as err:
                    skipped.append(f"{typ} on {c.name}: {err}")
        total = len(found)
        counts: dict[str, int] = {}
        for e in found:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        keep = sorted(found, key=lambda e: -e.get("score", 0))[: max(1, top_k)]
        keep.sort(key=lambda e: e["t"])
        with ev_lock:
            last_events[id] = keep
        tdec = time_decimals(max(c.fs for c in chans))
        out = []
        for e in keep:
            te = tround(e["t"], tdec)
            row = {
                "type": e["type"],
                "t": te,
                "dur": tround(e.get("dur", 0.0), tdec),
                "score": sig(e.get("score", 0.0)),
                "ch": e["ch"],
                "ref": ref(id, f"ev.{e['type']}", f"{te:g}"),
            }
            for k in ("detail", "resolution_s", "caveat"):
                if k in e:
                    row[k] = e[k] if isinstance(e[k], str) else sig(e[k])
            out.append(row)
        res = {"id": id, "window": [tround(t0, tdec), tround(t1, tdec)], "counts": counts, "events": out, "warn": warn}
        if total > len(out):
            res["truncated"], res["total"] = True, total
        if skipped:
            res["skipped"] = skipped
        if interface == "v2" and any(t in ("ecg_irregular", "ecg_noisy") for t in types):
            res["af_conventions"] = AF_CONVENTIONS
        if rt:
            res = rt.on_events(id, res, types, [c.name for c in chans], (t0, t1), skipped)
        result = [text(res)]
        if snippets and images and keep:
            result.append(image(render_snippets(chans, keep)))
        return result

    @srv.tool(
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False),
        structured_output=False,
    )
    @guard
    def ts_forget(id: str) -> list[TextContent]:
        """Release a loaded signal and its cached recipe. The source file is not touched."""
        with ev_lock:
            last_events.pop(id, None)
        return [text({"id": id, "forgotten": store.forget(id)})]

    if rt:
        rt.register(srv, text, guard)
        srv.runtime = rt  # type: ignore[attr-defined] - lets a launcher read the accepted answer
    return srv


def _minmax(c, a: float, b: float) -> dict:
    x = slice_channel(c, a, b)
    return {"ch": c.name, "t": [sig(a, 6), sig(b, 6)], "min": sig(np.nanmin(x)) if x.size else None,
            "max": sig(np.nanmax(x)) if x.size else None}  # fmt: skip


def _binned_text(ser: dict, t0: float, t1: float, tdec: int) -> dict:
    """Text stand-in for a series image (no-vision clients): per-bin median and p10/p90."""
    t, y = np.asarray(ser["t"], float), np.asarray(ser["y"], float)
    edges = np.linspace(t0, t1, MAX_SERIES_POINTS_TEXT + 1)
    idx = np.searchsorted(t, edges)
    med, p10, p90 = [], [], []
    for i in range(MAX_SERIES_POINTS_TEXT):
        seg = y[idx[i] : idx[i + 1]]
        seg = seg[np.isfinite(seg)]
        med.append(sig(np.median(seg)) if seg.size else None)
        p10.append(sig(np.percentile(seg, 10)) if seg.size else None)
        p90.append(sig(np.percentile(seg, 90)) if seg.size else None)
    return {"bin_s": sig((t1 - t0) / MAX_SERIES_POINTS_TEXT), "median": med, "p10": p10, "p90": p90}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tsmcp", description="TS-MCP stdio server")
    p.add_argument(
        "--allow-root", action="append", default=None, help="directory the server may read (repeatable; default: cwd)"
    )
    p.add_argument("--no-images", action="store_true", help="return text summaries instead of PNG images")
    p.add_argument("--interface", choices=("v1", "v2"), default="v1", help="output contract (default: v1)")
    p.add_argument("--plugins", action="append", choices=OPTIONAL_PLUGINS, default=None,
                   help="opt-in plugin (repeatable), e.g. pressure")  # fmt: skip
    p.add_argument(
        "--runtime",
        choices=("off", "log", "verify"),
        default="off",
        help="SignalMind runtime: off (default), log (passive ledger), verify (ts_submit + checks)",
    )
    p.add_argument("--runtime-dir", default=None, help="directory for ledger.jsonl, artifacts and answer.json")
    p.add_argument("--task-spec", default=None, help="TaskSpec JSON file (verify mode); default: minimal spec")
    args = p.parse_args(argv)
    extra: dict = {}  # new options are passed only when used, so the default call is exactly the historical one
    if args.plugins:
        extra["plugins"] = args.plugins
    if args.runtime != "off":
        extra.update(runtime=args.runtime, runtime_dir=args.runtime_dir, task_spec=args.task_spec)
    build_server(SignalStore(allow_roots=args.allow_root), images=not args.no_images, interface=args.interface,
                 **extra).run("stdio")  # fmt: skip


__all__ = ["build_server", "main"]
