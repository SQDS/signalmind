"""Loader registry. Each loader returns (channels, start). `names`: assigned to columns for csv/npy/inline,
selects channels for edf/wfdb."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from tsmcp.core.errors import SignalError
from tsmcp.core.signal import Channel, sanitize_name

FORMATS = ("csv", "npy", "edf", "wfdb")


def detect_format(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in ("csv", "txt", "tsv"):
        return "csv"
    if ext == "npy":
        return "npy"
    if ext == "edf":
        return "edf"
    if ext in ("hea", "dat"):
        return "wfdb"
    raise SignalError(f"cannot infer format from extension '.{ext}'; supported: {', '.join(FORMATS)}")


def _names(names: list[str] | None, n: int) -> list[str]:
    if names and len(names) != n:
        raise SignalError(f"got {len(names)} channel names for {n} channels")
    return [sanitize_name(names[i] if names else "", f"ch{i}") for i in range(n)]


def from_array(arr: np.ndarray, fs: float, names: list[str] | None, units: list[str] | None) -> list[Channel]:
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim == 2:
        # samples x channels is the common file layout; channels are the short axis
        if a.shape[0] > a.shape[1]:
            a = a.T
    else:
        raise SignalError(f"expected a 1-D or 2-D array, got shape {a.shape}")
    if a.shape[1] < 2:
        raise SignalError("signal must have at least 2 samples")
    if not (fs and math.isfinite(fs) and fs > 0):
        raise SignalError("fs (sampling rate in Hz) is required and must be > 0")
    nm = _names(names, a.shape[0])
    un = units or ["?"] * a.shape[0]
    if len(un) != a.shape[0]:
        raise SignalError(f"got {len(un)} units for {a.shape[0]} channels")
    return [
        Channel(
            nm[i],
            float(fs),
            np.ascontiguousarray(a[i], dtype=np.float32),
            sanitize_name(un[i], "?"),
        )
        for i in range(a.shape[0])
    ]


def load_npy(path: Path, fs: float | None, names, units):
    if fs is None:
        raise SignalError("npy files carry no sampling rate; pass fs (Hz)")
    arr = np.load(path, allow_pickle=False, mmap_mode="r")
    if not np.issubdtype(arr.dtype, np.number):
        raise SignalError(f"npy dtype {arr.dtype} is not numeric")
    return from_array(arr, fs, names, units), None


def load_csv(path: Path, fs: float | None, names, units):
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        except csv.Error:
            dialect = csv.excel
        rows = csv.reader(f, dialect)
        first = next(rows, None)
        if first is None:
            raise SignalError("csv file is empty")
        header = None
        try:
            [float(x) for x in first if x.strip()]
            data_rows = [first]
        except ValueError:
            header = [h.strip() for h in first]
            data_rows = []
        for r in rows:
            if r and any(x.strip() for x in r):
                data_rows.append(r)
    try:
        arr = np.array(
            [[float(x) if x.strip() else np.nan for x in r] for r in data_rows],
            dtype=np.float64,
        )
    except ValueError as e:
        raise SignalError("csv contains non-numeric values outside the header row") from e
    if header and header[0].lower() in ("t", "time", "time_s", "timestamp", "seconds", "sec") and arr.shape[1] >= 2:
        t = arr[:, 0]
        dt = np.diff(t)
        if dt.size == 0 or np.any(dt <= 0):
            raise SignalError("time column must be strictly increasing")
        med = float(np.median(dt))
        if np.max(np.abs(dt - med)) > 0.01 * med:
            raise SignalError("time column is irregularly sampled (>1% jitter); resample to a constant rate first")
        fs = fs or 1.0 / med
        arr, header = arr[:, 1:], header[1:]
    if fs is None:
        raise SignalError("csv has no time column; pass fs (Hz)")
    return from_array(arr.T, fs, names or header, units), None


from tsmcp.core.io.edf import load_edf  # noqa: E402
from tsmcp.core.io.wfdb import load_wfdb  # noqa: E402

LOADERS = {"npy": load_npy, "csv": load_csv, "edf": load_edf, "wfdb": load_wfdb}
