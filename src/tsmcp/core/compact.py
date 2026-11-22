"""Compact, type-aware JSON. Never round globally: times, measurements and codec params differ."""

from __future__ import annotations

import json
import math
from typing import Any


def sig(x: float, n: int = 3) -> float | None:
    """Round to n significant figures (measurements and stats)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    r = round(x, n - 1 - int(math.floor(math.log10(abs(x)))))
    return float(r)


def tround(t: float, decimals: int) -> float:
    """Round a time in seconds to the given decimals."""
    return round(float(t), decimals)


def exact(x: float) -> str:
    """Full-precision string for codec parameters that must survive serialization."""
    return repr(float(x))


def _clean(obj: Any) -> Any:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return int(obj) if obj.is_integer() and abs(obj) < 1e15 else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return _clean(obj.item())
    return obj


def dumps(obj: Any) -> str:
    """Compact JSON: no spaces, None fields dropped, integral floats as ints. Values must be pre-rounded."""
    return json.dumps(_clean(obj), separators=(",", ":"), ensure_ascii=False)
