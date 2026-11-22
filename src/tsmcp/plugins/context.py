"""Which Signal a Channel belongs to, so opt-in cross-channel measures (TFA, alignment, guided beats) can find the
other channels of the same record. The server / executor registers each signal before measuring."""

from __future__ import annotations

import threading
import weakref

_SIGNALS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # Channel -> Signal
_LOCK = threading.Lock()

PRESSURE_UNITS = ("mmHg", "kPa", "cmH2O")


def register(sig) -> None:
    with _LOCK:
        for c in sig.channels:
            _SIGNALS[c] = sig


def signal_of(ch):
    with _LOCK:
        return _SIGNALS.get(ch)


def find(ch, kind: str):
    """Another channel of the same record by role: 'ecg', 'pressure' (ABP), 'ppg', 'flow' (CBFV)."""
    sig = signal_of(ch)
    if sig is None:
        return None
    for c in sig.channels:
        if c is ch:
            continue
        name = c.name.lower()
        if kind == "ecg" and ("ecg" in name or "ekg" in name):
            return c
        if kind == "pressure" and ("abp" in name or "art" in name or c.unit in PRESSURE_UNITS):
            return c
        if kind == "ppg" and ("ppg" in name or "pleth" in name):
            return c
        if kind == "flow" and ("cbfv" in name or "mca" in name or "flow" in name or "tcd" in name):
            return c
    return None
