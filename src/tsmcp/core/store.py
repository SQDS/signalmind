"""Content-addressed handles with on-disk load recipes: handles survive restarts, eviction and multiple clients."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from tsmcp.core.errors import SignalError
from tsmcp.core.io import FORMATS, LOADERS, detect_format, from_array
from tsmcp.core.signal import Signal

DEFAULT_MAX_BYTES = 2 * 1024**3
DEFAULT_MAX_INLINE = 100_000
DEFAULT_CACHE_BYTES = 1 * 1024**3


def _cache_dir() -> Path:
    d = Path(os.environ.get("TSMCP_CACHE_DIR", Path.home() / ".cache" / "tsmcp")) / "handles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash(*parts: object) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()[:8]


class SignalStore:
    def __init__(
        self,
        allow_roots: list[str | Path] | None = None,
        max_file_bytes: int = DEFAULT_MAX_BYTES,
        max_inline: int = DEFAULT_MAX_INLINE,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ):
        roots = allow_roots or [Path.cwd()]
        self.roots = [Path(r).expanduser().resolve() for r in roots]
        self.max_file_bytes = max_file_bytes
        self.max_inline = max_inline
        self.cache_bytes = cache_bytes
        self._lru: OrderedDict[str, Signal] = OrderedDict()
        self._lock = threading.RLock()

    # ---- path policy -------------------------------------------------
    def _check_path(self, source: str) -> Path:
        if "://" in source:
            raise SignalError("URL sources are not supported; pass a local file path")
        p = Path(source).expanduser()
        try:
            p = p.resolve(strict=True)
        except FileNotFoundError:
            raise SignalError(f"file not found: {source}") from None
        if not any(p == r or r in p.parents for r in self.roots):
            raise SignalError("path is outside the allowed roots; start the server with --allow-root <dir>")
        if not p.is_file():
            raise SignalError(f"not a file: {source}")
        if p.stat().st_size > self.max_file_bytes:
            raise SignalError(f"file is larger than the {self.max_file_bytes // 1024**2} MB limit")
        return p

    # ---- loading -----------------------------------------------------
    def load(
        self,
        source: str | None = None,
        data: list[float] | list[list[float]] | None = None,
        fs: float | None = None,
        format: str | None = None,
        channels: list[str] | None = None,
        units: list[str] | None = None,
    ) -> Signal:
        if (source is None) == (data is None):
            raise SignalError("pass exactly one of source (file path) or data (inline array)")
        if data is not None:
            arr = np.asarray(data, dtype=np.float64)
            if arr.size > self.max_inline:
                raise SignalError(
                    f"inline data has {arr.size} values; limit is {self.max_inline}. Save to a file instead"
                )
            chans = from_array(arr, fs, channels, units)
            sid = _hash(arr.tobytes(), fs, channels, units)
            np.save(_cache_dir() / f"{sid}.npy", arr, allow_pickle=False)
            recipe = {
                "inline": f"{sid}.npy",
                "fs": fs,
                "channels": channels,
                "units": units,
            }
            sig = Signal(sid, chans)
        else:
            p = self._check_path(source)
            fmt = format or detect_format(p)
            if fmt not in LOADERS:
                raise SignalError(f"unsupported format {fmt!r}; supported: {', '.join(FORMATS)}")
            st = p.stat()
            sid = _hash(str(p), st.st_size, st.st_mtime_ns, fmt, fs, channels, units)
            with self._lock:
                if sid in self._lru:
                    self._lru.move_to_end(sid)
                    return self._lru[sid]
            chans, start = LOADERS[fmt](p, fs, channels, units)
            recipe = {
                "source": str(p),
                "format": fmt,
                "fs": fs,
                "channels": channels,
                "units": units,
            }
            sig = Signal(sid, chans, start=start)
        (_cache_dir() / f"{sid}.json").write_text(json.dumps(recipe))
        self._put(sig)
        return sig

    def _put(self, sig: Signal) -> None:
        with self._lock:
            self._lru[sig.id] = sig
            self._lru.move_to_end(sig.id)
            while len(self._lru) > 1 and self._bytes() > self.cache_bytes:
                self._lru.popitem(last=False)

    def _bytes(self) -> int:
        return sum(c.data.nbytes for s in self._lru.values() for c in s.channels if not isinstance(c.data, np.memmap))

    def get(self, sid: str) -> Signal:
        with self._lock:
            if sid in self._lru:
                self._lru.move_to_end(sid)
                return self._lru[sid]
        recipe_path = _cache_dir() / f"{sid}.json"
        if not sid.isalnum() or not recipe_path.exists():
            raise SignalError(f"unknown signal id {sid!r}; call ts_load first")
        r = json.loads(recipe_path.read_text())
        if "inline" in r:
            arr = np.load(_cache_dir() / r["inline"], allow_pickle=False)
            sig = Signal(sid, from_array(arr, r["fs"], r["channels"], r["units"]))
        else:
            p = self._check_path(r["source"])
            chans, start = LOADERS[r["format"]](p, r["fs"], r["channels"], r["units"])
            sig = Signal(sid, chans, start=start)
        self._put(sig)
        return sig

    def forget(self, sid: str) -> bool:
        with self._lock:
            existed = self._lru.pop(sid, None) is not None
        for ext in (".json", ".npy"):
            f = _cache_dir() / f"{sid}{ext}"
            if f.exists():
                f.unlink()
                existed = True
        return existed
