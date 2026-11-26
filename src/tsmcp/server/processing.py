"""Opt-in MCP tools for explicit processing of an already loaded signal session."""

from __future__ import annotations

import base64
from collections.abc import Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ImageContent, TextContent, ToolAnnotations

from tsmcp.core.compact import dumps
from tsmcp.core.views.render import render
from tsmcp.processing import ProcessingSession


def build_processing_server(session: ProcessingSession, export: Callable, measure: Callable) -> MCPServer:
    """Callbacks adapt output formats and public measurement definitions; neither selects a processing action."""
    server = MCPServer(
        "signalmind-processing",
        instructions=(
            "Inspect the signal, then choose explicit operations and sample ranges. Parents are immutable; each "
            "processing call returns a new session-scoped signal id. No processing is required when unsupported. "
            "Use ts_export to deliver the chosen version; final prose alone does not create output artifacts. "
            "All intervals are zero-based half-open SAMPLE indices. Inspection does not identify ground truth."
        ),
    )
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    def text(value):
        return [TextContent(type="text", text=dumps(value))]

    @server.tool(annotations=read_only, structured_output=False)
    def ts_describe(id: str | None = None, top_k: int = 12):
        """Summarize a signal version, masks, public protected ranges and largest adjacent sample steps.
        Omit id to inspect the original input and obtain its handle. Largest steps are candidates, not noise labels."""
        try:
            return text(session.summary(id, top_k))
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=read_only, structured_output=False)
    def ts_samples(id: str, start: int, end: int):
        """Read exact values and masks over [start,end) sample indices, up to1000 samples. Times are in seconds."""
        try:
            return text(session.samples(id, start, end))
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=read_only, structured_output=False)
    def ts_spectrum(id: str, start: int = 0, end: int | None = None, segment_s: float = 4.0):
        """Welch power peaks and50/60Hz neighborhood comparisons for each contiguous valid run in a sample range.
        These are measurements, not an instruction to filter. Narrow the range to examine a local disturbance."""
        try:
            return text(session.spectrum(id, start, end, segment_s))
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=read_only, structured_output=False)
    def ts_zoom(id: str, start: int = 0, end: int | None = None, original_overlay: bool = False):
        """Plot the chosen sample interval. With original_overlay=true show original and selected version in
        stacked panels on the same time axis. No sample values or masks are modified."""
        try:
            end = session.n if end is None else end
            session._ranges([[start, end]])
            channels = session.signal(id).channels
            if original_overlay:
                channels = session.signal().channels + channels
            png = render(channels, start / session.fs, end / session.fs, title="Signal processing view")
            return [ImageContent(type="image", data=base64.b64encode(png).decode(), mimeType="image/png")]
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=write, structured_output=False)
    def ts_process(
        id: str,
        operation: str,
        ranges_samples: list[list[int]],
        frequency_hz: float | None = None,
        quality_factor: float = 30.0,
        filter_order: int = 4,
        window_s: float | None = None,
        context_samples: int | None = None,
    ):
        """Create a new float64 signal version using one explicit operation:
        interpolate: replace selected short intervals by a line between two valid anchors, recording repaired
        points; respects separate missing-gap and finite-transient length limits, protected intervals and saturation.
        mark_invalid: retain observed values and mark the selected intervals unavailable.
        interpolate_pchip: replace a 1–4 sample observed spike using three observed anchors on each side.
        notch: zero-phase IIR notch at frequency_hz with quality_factor, on contiguous valid segments.
        remove_tone: fit sine/cosine plus intercept at frequency_hz separately in each contiguous valid segment,
        subtract only the fitted sine/cosine, preserving the fitted intercept. Optional window_s uses overlapping
        local fits to follow amplitude/phase variation; otherwise one fit over each selected valid segment.
        lowpass: zero-phase Butterworth filter with cutoff frequency_hz and filter_order, on valid segments.
        context_samples: filter using this many observed samples on each side and crop the write to the core.
        Input and all earlier versions remain unchanged. Explicit ranges are half-open sample indices.
        Filtering does not automatically decide whether interference exists or select its parameters."""
        try:
            result = session.process(
                id,
                operation,
                ranges_samples,
                frequency_hz=frequency_hz,
                quality_factor=quality_factor,
                filter_order=filter_order,
                window_s=window_s,
                context_samples=context_samples,
            )
            return text(session.summary(result.id, top_k=6))
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=read_only, structured_output=False)
    def ts_measure(id: str):
        """Compute the task's public display measurements on the chosen signal and its valid mask.
        Before/after support may differ; these are not private quality grades."""
        try:
            return text(measure(session.get(id)))
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=write, structured_output=False)
    def ts_export(id: str, action: str, rationale: str = "", limitations: list[str] | None = None):
        """Write result.json and output.npz for the selected version to the assigned output directory.
        The original can be exported unchanged. action is keep,repair,denoise,mask_only,
        partial_repair_and_mask,or uncertain. Writes all masks, actual operation parameters and public
        before/after measurements through the same packaging helper available to the Python arm."""
        try:
            if action not in {"keep", "repair", "denoise", "mask_only", "partial_repair_and_mask", "uncertain"}:
                raise ValueError("unknown output action")
            return text(export(session.get(id), action, rationale, limitations or []))
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from None

    return server
