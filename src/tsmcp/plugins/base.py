"""Domain plugin protocol (PROVISIONAL during 0.x).

A plugin contributes, by name: measures (numbers), detectors (event lists), series (derived curves to plot) and
views (domain-standard renderings). The MCP tool list stays static: these names appear as allowed values of
ts_measure `what`, ts_find_events `types` and ts_zoom `view`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tsmcp.core.measure import MeasureFn
from tsmcp.core.series import SeriesFn

DetectorFn = Callable[..., list]  # (x, fs, unit, t0, ch=...) -> [{"type", "t", "dur", "score", "detail"?}]
ViewFn = Callable[..., bytes]  # (ch, t0, t1, detail, title, events) -> PNG bytes


@dataclass(frozen=True)
class DomainPlugin:
    name: str
    measures: dict[str, MeasureFn] = field(default_factory=dict)
    summary: str = ""  # one line for the ts_measure description
    detectors: dict[str, DetectorFn] = field(default_factory=dict)
    detectors_summary: str = ""
    series: dict[str, SeriesFn] = field(default_factory=dict)
    views: dict[str, ViewFn] = field(default_factory=dict)
    views_summary: str = ""
