"""Minimal WFDB reader (formats 212 and 16), ported from bp2rr_new/data_process/01_rpeak_detector/rpeak_detector.py.

Reads the .hea header and .dat signal file; header comment lines (which may hold patient data) are ignored.
Supports single-file multi-signal records; for format 212 any even/odd number of signals interleaved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsmcp.core.errors import SignalError
from tsmcp.core.signal import Channel, sanitize_name


def read_header(hea: Path) -> dict:
    lines = [ln for ln in hea.read_text(errors="replace").splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise SignalError("empty WFDB header")
    first = lines[0].split()
    if len(first) < 4:
        raise SignalError("WFDB header needs record name, number of signals, sampling rate and length")
    n_sig, fs, n_samp = int(first[1]), float(first[2].split("/")[0]), int(first[3])
    sigs = []
    for i, line in enumerate(lines[1 : 1 + n_sig]):
        f = line.split()
        gain_field = f[2] if len(f) > 2 else "200"
        gain = float(gain_field.split("/")[0].split("(")[0]) or 200.0
        baseline = int(gain_field.split("(")[1].split(")")[0]) if "(" in gain_field else None
        unit = gain_field.split("/")[1] if "/" in gain_field else "mV"
        adc_zero = int(f[4]) if len(f) > 4 else 0
        sigs.append({
            "file": f[0], "format": f[1].split("x")[0].split(":")[0].split("+")[0], "gain": gain,
            "baseline": adc_zero if baseline is None else baseline, "unit": unit,
            "name": " ".join(f[8:]) if len(f) > 8 else f"ch{i}",
        })  # fmt: skip
    return {"n_sig": n_sig, "fs": fs, "n_samp": n_samp, "signals": sigs}


def _digital(dat: Path, fmt: str, n_sig: int, n_samp: int) -> np.ndarray:
    if fmt == "16":
        raw = np.fromfile(dat, dtype="<i2")
        return raw[: (raw.size // n_sig) * n_sig].reshape(-1, n_sig)[:n_samp]
    if fmt == "212":
        raw = np.fromfile(dat, dtype=np.uint8)
        t = raw[: (raw.size // 3) * 3].reshape(-1, 3).astype(np.int32)
        a = t[:, 0] | ((t[:, 1] & 0x0F) << 8)
        b = t[:, 2] | ((t[:, 1] >> 4) << 8)
        flat = np.empty(2 * t.shape[0], dtype=np.int32)
        flat[0::2], flat[1::2] = a, b
        flat = np.where(flat > 2047, flat - 4096, flat)
        return flat[: (flat.size // n_sig) * n_sig].reshape(-1, n_sig)[:n_samp]
    raise SignalError(f"WFDB format {fmt} is not supported (supported: 212, 16)")


def load_wfdb(path: Path, fs: float | None, names: list[str] | None, units: list[str] | None):
    hea = path.with_suffix(".hea")
    if not hea.exists():
        raise SignalError("WFDB record needs its .hea header next to the .dat file")
    h = read_header(hea)
    files = {s["file"] for s in h["signals"]}
    fmts = {s["format"] for s in h["signals"]}
    if len(files) != 1 or len(fmts) != 1:
        raise SignalError("WFDB records with several .dat files or mixed formats are not supported yet")
    dat = hea.parent / next(iter(files))
    if dat.resolve().parent != hea.resolve().parent:
        raise SignalError("WFDB .dat must be in the same directory as the header")
    dig = _digital(dat, next(iter(fmts)), h["n_sig"], h["n_samp"])
    chans = []
    for i, s in enumerate(h["signals"]):
        x = ((dig[:, i] - s["baseline"]) / s["gain"]).astype(np.float32)
        chans.append(Channel(sanitize_name(s["name"], f"ch{i}"), fs or h["fs"], x, sanitize_name(s["unit"], "?")))
    if names:
        keep = [c for c in chans if c.name in names]
        if len(keep) != len(names):
            raise SignalError(f"channels not in record; available: {[c.name for c in chans]}")
        chans = keep
    return chans, None
