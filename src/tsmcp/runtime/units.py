"""Small explicit unit table. Unknown units stay unknown; nothing is inferred from plausible-looking amplitudes."""

from __future__ import annotations

UNKNOWN = "?"

# unit -> (family, factor to the family's base unit)
UNITS: dict[str, tuple[str, float]] = {
    "mmHg": ("pressure", 1.0),
    "kPa": ("pressure", 7.50061683),
    "cmH2O": ("pressure", 0.73555912),
    "s": ("time", 1.0),
    "ms": ("time", 1e-3),
    "min": ("time", 60.0),
    "bpm": ("rate", 1.0),
    "Hz": ("frequency", 1.0),
    "beats": ("count", 1.0),
    "count": ("count", 1.0),
    "": ("dimensionless", 1.0),
    "cm/s": ("velocity", 1.0),
    "m/s": ("velocity", 100.0),
    "rad": ("angle", 1.0),
    "deg": ("angle", 0.017453292519943295),
    "mV": ("voltage", 1.0),
}


def _parse(unit: str) -> tuple[str, float] | None:
    """Known unit, or a ratio 'a/b' / '(a)/b' of known units (left-associative: 'cm/s/mmHg' = '(cm/s)/mmHg')."""
    if unit in UNITS:
        return UNITS[unit]
    if "/" not in unit:
        return None
    if unit.startswith("(") and ")/" in unit:
        num, den = unit[1 : unit.index(")/")], unit[unit.index(")/") + 2 :]
    else:
        num, den = unit.rsplit("/", 1)
    a, b = _parse(num), _parse(den)
    if a is None or b is None:
        return None
    return f"{a[0]}/{b[0]}", a[1] / b[1]


def family(unit: str | None) -> str | None:
    if unit is None or unit == UNKNOWN:
        return None
    hit = _parse(unit)
    return hit[0] if hit else None


def compatible(a: str | None, b: str | None) -> bool:
    fa, fb = family(a), family(b)
    return fa is not None and fa == fb


def convert(value: float, src: str, dst: str) -> float:
    if not compatible(src, dst):
        raise ValueError(f"cannot convert {src!r} to {dst!r}")
    return value * _parse(src)[1] / _parse(dst)[1]
