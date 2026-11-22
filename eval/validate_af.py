"""Dev-set validation of ecg_irregular (AF-like episodes) against expert rhythm annotations.

Dev set only (tuning allowed): full MIT-BIH DS1 records + LTAFDB records NOT used by the long-record eval suite.
Reports per record: annotated AF/AFL burden vs detected burden, and time-overlap sensitivity / PPV.

  uv run --group eval python eval/validate_af.py [--ltaf 07 00 01]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.tasks.longrec import LTAF, af_segments, select_ltafdb  # noqa: E402
from eval.tasks.mitdb import DS1, MITDB  # noqa: E402
from tsmcp.core.io.wfdb import load_wfdb  # noqa: E402
from tsmcp.plugins.ecg import ecg_irregular  # noqa: E402


def mask_from(segs, n_sec: int) -> np.ndarray:
    m = np.zeros(n_sec, bool)
    for a, b in segs:
        m[int(a) : int(np.ceil(b))] = True
    return m


def one(path: Path, ch_name: str | None = None) -> dict:
    import wfdb

    chans, _ = load_wfdb(path.with_suffix(".hea"), None, None, None)
    c = next((c for c in chans if c.name == ch_name), chans[0]) if ch_name else chans[0]
    a = wfdb.rdann(str(path), "atr")
    n = int(np.ceil(c.duration))
    truth = mask_from(af_segments(a, c.fs, c.n), n)
    ev = ecg_irregular(np.asarray(c.data, float), c.fs, "mV", 0.0, ch=c)
    det = mask_from([(e["t"], e["t"] + e["dur"]) for e in ev], n)
    tp = (truth & det).sum()
    return {"af_true_%": 100 * truth.mean(), "af_det_%": 100 * det.mean(),
            "se": tp / truth.sum() if truth.sum() else None, "ppv": tp / det.sum() if det.sum() else None}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ltaf", nargs="*", default=None, help="default: 07 + the dev records of eval/tasks/ltafdb_split.json"
    )
    args = ap.parse_args()
    import json

    split = json.loads((Path(__file__).resolve().parent / "tasks" / "ltafdb_split.json").read_text())
    if args.ltaf is None:
        args.ltaf = ["07", *split["dev"]]
    held_out = {f["rec"] for _, f in select_ltafdb(20260930 + 3, 4)} | set(split["test_ext"])
    rows = []
    for r in DS1:
        rows.append((f"mitdb/{r}", one(MITDB / str(r), "MLII")))
    for r in args.ltaf:
        if r in held_out:
            print(f"skip LTAFDB {r}: it is in the eval suite (held out)")
            continue
        if (LTAF / f"{r}.dat").exists():
            rows.append((f"ltafdb/{r}", one(LTAF / r)))
    fp_sec = tot_nonaf = 0.0
    for name, m in rows:
        se = "-" if m["se"] is None else f"{m['se']:.2f}"
        ppv = "-" if m["ppv"] is None else f"{m['ppv']:.2f}"
        print(f"{name:12s} AF true {m['af_true_%']:5.1f}%  detected {m['af_det_%']:5.1f}%  Se {se}  PPV {ppv}")
        if m["af_true_%"] == 0:
            fp_sec += m["af_det_%"]
            tot_nonaf += 1
    errs = [abs(m["af_true_%"] - m["af_det_%"]) for _, m in rows]
    print(f"mean |burden error| {np.mean(errs):.1f} pp; records within 5 pp: {sum(e <= 5 for e in errs)}/{len(errs)}; "
          f"mean false burden on AF-free records {fp_sec / max(tot_nonaf, 1):.1f}%")  # fmt: skip


if __name__ == "__main__":
    main()
