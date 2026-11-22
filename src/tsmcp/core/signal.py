"""Signal container: each channel has its own sampling rate and unit."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from tsmcp.core.errors import SignalError

_NAME_OK = re.compile(r"[^\w\-. ()/]", re.UNICODE)


def sanitize_name(name: str, fallback: str) -> str:
    """Channel names come from untrusted files: keep a short, safe subset."""
    clean = _NAME_OK.sub("", str(name)).strip()[:32]
    return clean or fallback


@dataclass(eq=False)  # identity semantics: hashable, usable as a cache key
class Channel:
    name: str
    fs: float
    data: np.ndarray  # 1-D float32 (may be a read-only memmap)
    unit: str = "?"

    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration(self) -> float:
        return self.n / self.fs


@dataclass
class Signal:
    id: str
    channels: list[Channel]
    start: str | None = None  # absolute start time as recorded, verbatim, no timezone
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(c.duration for c in self.channels)

    def channel(self, name: str | None) -> Channel:
        if name is None:
            return self.channels[0]
        for c in self.channels:
            if c.name == name:
                return c
        names = ", ".join(c.name for c in self.channels[:20])
        raise SignalError(f"unknown channel {name!r}; available: {names}")

    def select(self, names: list[str] | None) -> list[Channel]:
        if not names:
            return list(self.channels)
        return [self.channel(n) for n in names]


def time_decimals(fs: float) -> int:
    """Decimals for times so that one sample is resolvable (clamped to 1..4)."""
    return int(min(4, max(1, math.ceil(math.log10(max(fs, 1.0))))))


def clip_window(sig: Signal, window: list[float] | tuple[float, float] | None) -> tuple[float, float, str | None]:
    """Half-open [t0, t1). Clip to the signal; error if entirely outside."""
    dur = sig.duration
    if window is None:
        return 0.0, dur, None
    if len(window) != 2:
        raise SignalError("window must be [t0, t1] in seconds")
    t0, t1 = float(window[0]), float(window[1])
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        raise SignalError(f"window [{t0}, {t1}) is empty or invalid; need t0 < t1")
    if t1 <= 0 or t0 >= dur:
        raise SignalError(f"window [{t0:g}, {t1:g}) is outside the signal (0–{dur:g} s)")
    warn = None
    if t0 < 0 or t1 > dur:
        warn = f"window clipped to [{max(t0, 0):g}, {min(t1, dur):g}) (signal is 0–{dur:g} s)"
    return max(t0, 0.0), min(t1, dur), warn


def slice_channel(ch: Channel, t0: float, t1: float) -> np.ndarray:
    i0 = int(math.floor(t0 * ch.fs))
    i1 = int(math.ceil(t1 * ch.fs))
    return np.asarray(ch.data[max(i0, 0) : min(i1, ch.n)], dtype=np.float64)
