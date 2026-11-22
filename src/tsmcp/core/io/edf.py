"""Self-written EDF / EDF+ reader (numpy only).

Header free text (patient and recording identification) is untrusted and may contain personal data:
it is never returned. Only the start date/time is kept (verbatim, no timezone). "EDF Annotations"
channels are skipped. Each channel keeps its own sampling rate and physical unit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsmcp.core.errors import SignalError
from tsmcp.core.signal import Channel, sanitize_name


def _field(raw: bytes, n: int, width: int) -> list[str]:
    return [raw[i * width : (i + 1) * width].decode("ascii", "replace").strip() for i in range(n)]


def _num(s: str, what: str) -> float:
    try:
        return float(s)
    except ValueError:
        raise SignalError(f"EDF header field '{what}' is not a number") from None


def load_edf(path: Path, fs: float | None, names: list[str] | None, units: list[str] | None):
    with open(path, "rb") as f:
        head = f.read(256)
        if len(head) < 256:
            raise SignalError("file is too short to be EDF")
        header_bytes = int(_num(head[184:192].decode("ascii", "replace"), "header bytes"))
        n_records = int(_num(head[236:244].decode("ascii", "replace"), "number of data records"))
        rec_dur = _num(head[244:252].decode("ascii", "replace"), "record duration")
        ns = int(_num(head[252:256].decode("ascii", "replace"), "number of signals"))
        if ns <= 0 or header_bytes != 256 * (ns + 1):
            raise SignalError("inconsistent EDF header (number of signals / header size)")
        sh = f.read(256 * ns)
    start = f"{head[168:176].decode('ascii', 'replace').strip()} {head[176:184].decode('ascii', 'replace').strip()}"
    off = 0
    cols = {}
    for key, width in [("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8),
                       ("dmax", 8), ("prefilter", 80), ("spr", 8), ("reserved", 32)]:  # fmt: skip
        cols[key] = _field(sh[off : off + width * ns], ns, width)
        off += width * ns
    spr = [int(_num(v, "samples per record")) for v in cols["spr"]]
    rec_len = sum(spr)
    size = path.stat().st_size - header_bytes
    if n_records < 0:  # allowed by the spec while recording; infer from size
        n_records = size // (2 * rec_len)
    n_records = min(n_records, size // (2 * rec_len))
    if n_records <= 0 or rec_dur <= 0:
        raise SignalError("EDF file contains no data records")
    data = np.memmap(path, dtype="<i2", mode="r", offset=header_bytes, shape=(n_records, rec_len))
    chans: list[Channel] = []
    starts = np.concatenate([[0], np.cumsum(spr)])
    for i in range(ns):
        label = cols["label"][i]
        if label.lower().startswith("edf annotations"):
            continue
        pmin, pmax = _num(cols["pmin"][i], "physical min"), _num(cols["pmax"][i], "physical max")
        dmin, dmax = _num(cols["dmin"][i], "digital min"), _num(cols["dmax"][i], "digital max")
        gain = (pmax - pmin) / (dmax - dmin) if dmax != dmin else 1.0
        dig = np.asarray(data[:, starts[i] : starts[i + 1]]).reshape(-1)
        x = ((dig.astype(np.float32) - np.float32(dmin)) * np.float32(gain) + np.float32(pmin)).astype(np.float32)
        chans.append(Channel(sanitize_name(label, f"ch{i}"), spr[i] / rec_dur, x, sanitize_name(cols["unit"][i], "?")))
    if not chans:
        raise SignalError("EDF file has no signal channels")
    if names:
        keep = [c for c in chans if c.name in names]
        missing = set(names) - {c.name for c in keep}
        if missing:
            raise SignalError(f"channels not in file: {sorted(missing)}; available: {[c.name for c in chans]}")
        chans = keep
    return chans, start
