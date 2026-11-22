"""Built-in plugin registry. The core never imports plugins; the server composes them here."""

from __future__ import annotations

from tsmcp.core.detect import DETECTORS as _GENERIC_DETECTORS
from tsmcp.core.measure import MEASURES as _GENERIC
from tsmcp.core.series import SERIES as _GENERIC_SERIES
from tsmcp.core.views.render import render_verify
from tsmcp.plugins.base import DomainPlugin
from tsmcp.plugins.ecg import PLUGIN as ECG

GENERIC = DomainPlugin(
    "generic",
    measures=dict(_GENERIC),
    summary="period (s, autocorrelation; acf_peak = strength), dominant_freq (Hz, Welch), "
    "amplitude_pp (robust p99.5-p0.5), rms, mean",
    detectors=dict(_GENERIC_DETECTORS),
    detectors_summary="spike (short transients), flat (sensor stuck >= 1 s), clip (saturation >= 0.5 s), "
    "change (abrupt persistent change in level / amplitude / oscillation frequency), "
    "discord (unusual stretches across multiple window lengths: use for 'find the anomaly' questions)",
    series=dict(_GENERIC_SERIES),
    views={"verify": render_verify},
    views_summary="wave (default), strip (one channel wrapped into rows; best overview of a long window), "
    "level / rms / freq (derived curves: rolling mean, RMS, oscillation frequency), "
    "verify (focused wave plus freq/RMS over +/-10 windows of context; 1200 x 860 px, image-only)",
)
PLUGINS: list[DomainPlugin] = [GENERIC, ECG]


def _merge(attr: str, plugins: list[DomainPlugin]) -> dict:
    out: dict = {}
    for p in plugins:
        items = getattr(p, attr)
        clash = set(out) & set(items)
        if clash:
            raise ValueError(f"plugin {p.name} redefines {attr} {sorted(clash)}")
        out.update(items)
    return out


def all_measures(plugins: list[DomainPlugin] = PLUGINS) -> dict:
    return _merge("measures", plugins)


def all_detectors(plugins: list[DomainPlugin] = PLUGINS) -> dict:
    return _merge("detectors", plugins)


def all_series(plugins: list[DomainPlugin] = PLUGINS) -> dict:
    return _merge("series", plugins)


def all_views(plugins: list[DomainPlugin] = PLUGINS) -> dict:
    return _merge("views", plugins)
