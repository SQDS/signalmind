"""Derived series: a signal re-expressed as a slower quantity over time (e.g. rolling RMS, oscillation frequency,
or a plugin's RR intervals). Plotting a derived series is how a model 'reads' hours of data in one image."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tsmcp.core.detect import feature_series

# SeriesFn(x, fs, unit, t0, ch=...) -> {"t": array (s, absolute), "y": array, "unit": str, "label": str}
SeriesFn = Callable[..., dict]


def _generic(name: str, label: str, unit_fn: Callable[[str], str]) -> SeriesFn:
    def fn(x, fs, unit, t0=0.0, **_):
        dur = x.size / fs
        win = float(np.clip(dur / 400, 2.0, 60.0))
        starts, feats = feature_series(x, fs, win)
        return {
            "t": t0 + starts + win / 2,
            "y": feats[name],
            "unit": unit_fn(unit),
            "label": f"{label} ({win:g} s windows)",
        }

    fn.__doc__ = f"rolling {label}"
    return fn


SERIES: dict[str, SeriesFn] = {
    "level": _generic("level", "mean level", lambda u: u),
    "rms": _generic("rms", "RMS (std)", lambda u: u),
    "freq": _generic("freq", "oscillation frequency", lambda u: "Hz"),
}
