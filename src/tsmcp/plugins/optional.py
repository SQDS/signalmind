"""Opt-in plugins, imported only when a server or caller asks for them. The default plugin list never changes."""

from __future__ import annotations

from tsmcp.plugins import PLUGINS
from tsmcp.plugins.base import DomainPlugin

OPTIONAL_PLUGINS = ("pressure", "tfa", "pulse", "align")


def optional_plugin(name: str) -> DomainPlugin:
    if name == "pressure":
        from tsmcp.plugins.pressure import PLUGIN as PRESSURE

        return PRESSURE
    if name == "tfa":
        from tsmcp.plugins.tfa import PLUGIN as TFA

        return TFA
    if name == "pulse":
        from tsmcp.plugins.pulse import PLUGIN as PULSE

        return PULSE
    if name == "align":
        from tsmcp.plugins.align import PLUGIN as ALIGN

        return ALIGN
    raise ValueError(f"unknown optional plugin {name!r}; available: {', '.join(OPTIONAL_PLUGINS)}")


def plugin_set(extra: list[str] | tuple[str, ...] | None = None) -> list[DomainPlugin]:
    return PLUGINS + [optional_plugin(n) for n in (extra or ())]


def register_signal(sig) -> None:
    """Let opt-in measures that need another channel of the same record find it (pressure: ECG guidance)."""
    from tsmcp.plugins import context
    from tsmcp.plugins.pressure import register_signal as _register

    _register(sig)
    context.register(sig)


def pop_artifacts(rows: list[dict], sink: list | None) -> None:
    """Remove private `_artifact` tables from measurement rows; keep them for the evidence ledger when asked."""
    for row in rows:
        art = row.pop("_artifact", None)
        if art is not None and sink is not None:
            sink.append((row.get("ref"), art))
