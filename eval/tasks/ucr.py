"""UCR Time Series Anomaly Archive (third-party, one labeled anomaly per series).

File name: <id>_UCR_Anomaly_<name>_<trainEnd>_<anomBegin>_<anomEnd>.txt. The name (which reveals the source dataset)
is never shown to the model. No sampling rate is given, so fs=1 and times are sample indices.
Scoring follows the archive's convention: a location is correct if within the anomaly +- max(100, anomaly length).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

UCR_DIR = Path(os.environ.get("UCR_DIR", Path(__file__).resolve().parents[2] / "data" / "ucr"))
_NAME = re.compile(r"(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")


def files() -> list[Path]:
    return sorted(p for p in UCR_DIR.rglob("*.txt") if _NAME.search(p.name))


def file_id(p: Path) -> int:
    """Read only the id from the filename, so held-out signals can be excluded before opening them."""
    return int(p.name.split("_", 1)[0])


def load(p: Path) -> tuple[np.ndarray, dict]:
    m = _NAME.search(p.name)
    _, name, train_end, a0, a1 = m.groups()
    txt = p.read_text().split()
    x = np.array([float(v) for v in txt], dtype=np.float32)
    return x, {"begin": int(a0), "end": int(a1), "train_end": int(train_end), "source": name, "id": int(m.group(1))}


DEV_COUNT = 120  # eval/validate_discord.py tuned on the first 120 eligible series (archive order); never test on them


def dev_ids(held: set[int], max_len: int = 200_000) -> set[int]:
    """Ids of the series used for discord tuning: the first DEV_COUNT eligible ones in archive order."""
    out: set[int] = set()
    for p in files():
        if len(out) >= DEV_COUNT:
            break
        if file_id(p) in held:
            continue
        x, m = load(p)
        if x.size > max_len:
            continue
        out.add(m["id"])
    return out


def build(seed: int, n: int, max_len: int = 200_000, exclude: set[int] | None = None):
    rng = np.random.default_rng(seed)
    fs_ = [p for p in files()]
    rng.shuffle(fs_)
    exclude = exclude or set()
    out = []
    for p in fs_:
        if len(out) >= n:
            break
        x, meta = load(p)
        if x.size > max_len or meta["id"] in exclude:
            continue
        q = (
            "This recording contains exactly one anomalous segment (an unusual pattern that differs from the rest). "
            "At what time does the anomaly start? (Time unit: samples; the sampling rate is given as 1 Hz.)"
        )
        out.append((f"ucr{len(out):03d}", "anomaly_time", "-", x, 1.0, meta, q))
    return out
