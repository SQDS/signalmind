"""One presentation rule for every reported number: never more precision than the toolkit convention (3 significant
figures) and never finer than the first significant digit of a known resolution. Used by the pressure measures and
by derived values alike, so `ts_measure` and `ts_derive` cannot disagree about how precise a value is."""

from __future__ import annotations

import math
from decimal import Decimal

from tsmcp.core.compact import sig


def decimals_of(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0
    return max(0, -Decimal(repr(float(value))).normalize().as_tuple().exponent)


def resolution_decimals(resolution: dict | None) -> int | None:
    """Decimals supported by a known resolution: the position of its first significant digit; None if unknown."""
    res = resolution or {}
    if not (res.get("known") and res.get("value") is not None):
        return None
    try:
        r = float(res["value"])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(r) and r > 0):
        return None
    return max(0, -math.floor(math.log10(r)))


def present(value: float, resolution: dict | None = None) -> float:
    """Reported form of a measured or derived number (adds no precision; may remove unsupported precision)."""
    v = sig(float(value))
    d = resolution_decimals(resolution)
    if d is not None and decimals_of(v) > d:
        v = round(v, d)
    return float(v)
