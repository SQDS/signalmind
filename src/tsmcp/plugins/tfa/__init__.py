"""Transfer-function analysis plugin (opt-in): ABP (input) -> flow/CBFV (output) gain, phase and coherence.

Method (port of the project's validated `tfa/standard.py`, numpy only): both channels reduced to non-overlapping 1-s
means over the requested window; Welch auto- and cross-spectra with a periodic Hann window of 100 samples (100 s),
50 % overlap, constant detrend, density scaling; spectra averaged first, then H(f) = Pxy/Pxx, gain = |H|,
phase = angle(H) (degrees, positive = output leads input), coherence = |Pxy|^2 / (Pxx Pyy). Band value = equal-weight
mean over frequency bins with lo <= f < hi. Bands: VLF 0.02-0.07, LF 0.07-0.20, HF 0.20-0.50 Hz. Validity is reported,
not hidden: number of Welch windows, band-mean coherence, NaN handling; a window shorter than one segment is a
limitation. No CARNet-compliance claim is made.

Measure on the output channel (e.g. CBFV); the input is the record's pressure channel.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from tsmcp.core.errors import SignalError
from tsmcp.plugins import context
from tsmcp.plugins.base import DomainPlugin
from tsmcp.runtime.precision import present

METHOD_VERSION = "tfa.welch1s.v1"
NPERSEG = 100
BANDS = {"vlf": (0.02, 0.07), "lf": (0.07, 0.20), "hf": (0.20, 0.50)}
MIN_WINDOWS_RECOMMENDED = 5
LOW_COHERENCE = 0.4
MAX_NAN_FRACTION = 0.05

_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def one_second_means(x: np.ndarray, fs: float, t0: float, t1: float) -> tuple[np.ndarray, float]:
    """Non-overlapping 1-s means of the complete seconds inside [t0, t1); returns (means, nan_fraction)."""
    first, last = int(np.ceil(t0)), int(np.floor(t1))
    n = max(0, last - first)
    out = np.full(n, np.nan)
    for k in range(n):
        a, b = int(round((first + k) * fs)), int(round((first + k + 1) * fs))
        seg = np.asarray(x[a:b], dtype=np.float64)
        if seg.size and np.isfinite(seg).mean() >= 0.5:
            out[k] = float(np.nanmean(seg))
    nan_frac = float(np.mean(~np.isfinite(out))) if n else 1.0
    if 0 < nan_frac <= MAX_NAN_FRACTION:
        idx = np.arange(n)
        good = np.isfinite(out)
        out = np.interp(idx, idx[good], out[good])
    return out, nan_frac


def periodic_hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def welch_cross(x: np.ndarray, y: np.ndarray, fs: float = 1.0, nperseg: int = NPERSEG):
    """Averaged one-sided density spectra (f, Pxx, Pyy, Pxy, n_windows), scipy.signal.welch/csd conventions."""
    n = min(int(nperseg), x.size, y.size)
    step = n // 2 if n > 1 else 1
    w = periodic_hann(n)
    scale = fs * (w**2).sum()
    pxx = pyy = pxy = None
    k = 0
    for s in range(0, min(x.size, y.size) - n + 1, step):
        xs = x[s : s + n] - x[s : s + n].mean()
        ys = y[s : s + n] - y[s : s + n].mean()
        fx, fy = np.fft.rfft(xs * w), np.fft.rfft(ys * w)
        a, b, c = np.abs(fx) ** 2 / scale, np.abs(fy) ** 2 / scale, np.conj(fx) * fy / scale
        pxx, pyy, pxy = (a, b, c) if pxx is None else (pxx + a, pyy + b, pxy + c)
        k += 1
    if k == 0:
        raise SignalError("window shorter than one TFA segment")
    pxx, pyy, pxy = pxx / k, pyy / k, pxy / k
    sl = slice(1, -1) if n % 2 == 0 else slice(1, None)  # one-sided: double all but DC (and Nyquist when n is even)
    pxx[sl] *= 2
    pyy[sl] *= 2
    pxy[sl] *= 2
    return np.fft.rfftfreq(n, 1 / fs), pxx, pyy, pxy, k


def transfer(f, pxx, pyy, pxy) -> dict:
    h = pxy / np.where(np.abs(pxx) < 1e-12, 1e-12, pxx)
    coherence = np.clip(np.abs(pxy) ** 2 / (np.abs(pxx) * np.abs(pyy) + 1e-12), 0.0, 1.0)
    return {"frequency": f, "gain": np.abs(h), "phase_deg": np.angle(h, deg=True), "coherence": coherence}


def band_mean(f: np.ndarray, v: np.ndarray, lo: float, hi: float) -> tuple[float, int]:
    m = (f >= lo) & (f < hi)
    return (float(v[m].mean()) if m.any() else float("nan")), int(m.sum())


def _analysis(x_out, fs, t0, t1, ch) -> dict:
    inp = context.find(ch, "pressure")
    if inp is None:
        raise SignalError("TFA needs a pressure (ABP) input channel in the same record; measure on the flow channel")
    key = (inp, round(t0, 3), round(t1, 3))
    with _LOCK:
        hit = _CACHE.get(ch, {}).get(key)
    if hit is not None:
        return hit
    if inp.fs != ch.fs:
        raise SignalError("TFA in v1 needs input and output at the same sampling rate")
    xi, nan_i = one_second_means(inp.data, inp.fs, t0, t1)
    xo, nan_o = one_second_means(ch.data, ch.fs, t0, t1)
    res = {
        "input": inp.name,
        "input_unit": inp.unit,
        "output_unit": ch.unit,
        "n_seconds": int(xi.size),
        "nan_fraction": max(nan_i, nan_o),
        "window": [float(np.ceil(t0)), float(np.floor(t1))],
    }
    if res["nan_fraction"] > MAX_NAN_FRACTION:
        res["limitation"] = f"{res['nan_fraction']:.0%} of 1-s means missing (more than {MAX_NAN_FRACTION:.0%})"
    elif xi.size < NPERSEG:
        res["limitation"] = f"{xi.size} s of data; TFA needs at least one {NPERSEG}-s segment"
    elif np.std(xi) == 0 or np.std(xo) == 0:
        res["limitation"] = "input or output is constant over the window (no spectral power)"
    else:
        f, pxx, pyy, pxy, k = welch_cross(xi, xo)
        res.update(transfer(f, pxx, pyy, pxy), n_windows=k)
    with _LOCK:
        _CACHE.setdefault(ch, {})[key] = res
    return res


def _gain_unit(out_u: str, in_u: str) -> str:
    if out_u in ("?", "", None) or in_u in ("?", "", None):
        return "?"
    return f"({out_u})/{in_u}"


def _measure(quantity: str, band: str, x, fs, unit, t0, ch) -> dict:
    if ch is None:
        raise SignalError("TFA measures need the loaded channel")
    t1 = t0 + x.size / fs
    res = _analysis(x, fs, t0, t1, ch)
    lo, hi = BANDS[band]
    unit_out = {"gain": _gain_unit(res["output_unit"], res["input_unit"]), "phase": "deg", "coherence": ""}[quantity]
    base = {
        "unit": unit_out,
        "method": f"Welch TFA on 1-s means (Hann {NPERSEG} s, 50 % overlap), band mean over {lo}-{hi} Hz [lo, hi)",
        "method_version": METHOD_VERSION,
        "definition_id": f"tfa.{quantity}.{band}.v1",
        "channels_effective": [ch.name, res["input"]],
        "intervals_effective": [res["window"]],
        "conventions": {
            "input": res["input"],
            "output": ch.name,
            "resampling": "non-overlapping 1-s means",
            "nperseg_s": NPERSEG,
            "overlap": 0.5,
            "window": "periodic Hann",
            "detrend": "constant",
            "band_hz": [lo, hi],
            "band_edges": "lo <= f < hi",
            "phase_sign": "positive = output leads",
        },
        "timebase": {
            "identity": "same record",
            "sync_status": "index_aligned_declared",
            "channels": [res["input"], ch.name],
            "basis": "input and output are columns of one record",
        },
    }
    if "limitation" in res:
        return {**base, "value": None, "status": "limitation", "note": res["limitation"], "warnings": []}
    f = res["frequency"]
    v, nbins = band_mean(
        f, {"gain": res["gain"], "phase": res["phase_deg"], "coherence": res["coherence"]}[quantity], lo, hi
    )
    coh, _ = band_mean(f, res["coherence"], lo, hi)
    warnings = []
    if res["n_windows"] < MIN_WINDOWS_RECOMMENDED:
        warnings.append(
            {
                "code": "few_welch_windows",
                "severity": "warning",
                "message": f"{res['n_windows']} Welch windows (< {MIN_WINDOWS_RECOMMENDED}); estimates are noisy",
            }
        )
    if np.isfinite(coh) and coh < LOW_COHERENCE and quantity != "coherence":
        warnings.append(
            {
                "code": "low_coherence",
                "severity": "warning",
                "message": f"band-mean coherence {coh:.2f} < {LOW_COHERENCE}: gain/phase may not be interpretable",
            }
        )
    if res["input_unit"] in ("?", "") or res["output_unit"] in ("?", ""):
        warnings.append({"code": "unit_unknown", "severity": "warning", "message": "input or output unit unknown"})
    if res["nan_fraction"] > 0:
        warnings.append(
            {
                "code": "interpolated_seconds",
                "severity": "info",
                "message": f"{res['nan_fraction']:.1%} of 1-s means interpolated",
            }
        )
    rows = [
        {
            "f_hz": float(f[i]),
            "gain": float(res["gain"][i]),
            "phase_deg": float(res["phase_deg"][i]),
            "coherence": float(res["coherence"][i]),
            "in_band": bool(lo <= f[i] < hi),
        }
        for i in range(f.size)
        if f[i] <= 0.5
    ]
    return {
        **base,
        "value": present(v) if np.isfinite(v) else None,
        "band_bins": nbins,
        "n_windows": res["n_windows"],
        "coherence_band_mean": round(coh, 3) if np.isfinite(coh) else None,
        "frequency_resolution_hz": 1.0 / NPERSEG,
        "counts": {"accepted": res["n_windows"], "rejected": 0, "rejection_reasons": {}},
        "coverage": round(res["n_seconds"] / max(1.0, t1 - t0), 4),
        "resolution": {"known": False, "note": "no closed-form uncertainty reported; see coherence and window count"},
        "warnings": warnings,
        "_artifact": {
            "kind": "tfa_spectrum",
            "columns": ["f_hz", "gain", "phase_deg", "coherence", "in_band"],
            "rows": rows,
            "meta": {"input": res["input"], "output": ch.name, "method_version": METHOD_VERSION},
        },
    }


def _make(quantity: str, band: str):
    def fn(x, fs, unit, t0=0.0, ch=None, **_):
        return _measure(quantity, band, x, fs, unit, t0, ch)

    fn.__name__ = f"tfa_{quantity}_{band}"
    return fn


MEASURES = {f"tfa_{q}_{b}": _make(q, b) for q in ("gain", "phase", "coherence") for b in BANDS}

PLUGIN = DomainPlugin(
    "tfa",
    measures=MEASURES,
    summary="transfer function pressure -> flow on the flow channel (e.g. CBFV; input = the record's ABP): "
    "tfa_gain_{vlf,lf,hf}, tfa_phase_{vlf,lf,hf} (deg, + = flow leads), tfa_coherence_{vlf,lf,hf}; 1-s means, "
    "Welch Hann 100 s, 50 % overlap, band mean lo <= f < hi (VLF .02-.07, LF .07-.2, HF .2-.5 Hz); reports window "
    "count, band coherence and validity warnings",
)
