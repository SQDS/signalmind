"""Validate the numpy port of the R-peak detector on all 48 MIT-BIH records against annotations AND the scipy original.

  uv run --group eval python eval/validate_rpeaks.py [--records 100 101 ...]
Acceptance: pooled F1 >= 0.99 (original: 0.992) and per-record F1 within 0.01 of the original.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.tasks.mitdb import MITDB  # noqa: E402
from tsmcp.plugins.ecg.rpeaks import detect_r_peaks  # noqa: E402

ORIG = MITDB.parent / "rpeak_detector.py"
spec = importlib.util.spec_from_file_location("orig_rpeaks", ORIG)
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="*", default=None)
    args = ap.parse_args()
    recs = args.records or sorted({p.stem for p in MITDB.glob("*.hea")})
    tot = {"port": [0, 0, 0], "orig": [0, 0, 0]}  # tp, n_ref, n_det
    rows, worst = [], 0.0
    t_port = t_orig = 0.0
    for r in recs:
        rec = MITDB / r
        sig, hdr = orig.read_signal(rec)
        fs = hdr["fs"]
        s, sym = orig.read_annotations(rec)
        ref = s[np.isin(sym, list(orig.BEAT_SYMBOLS))]
        x = sig[:, 0]
        t = time.perf_counter()
        p_port = detect_r_peaks(x, fs)
        t_port += time.perf_counter() - t
        t = time.perf_counter()
        p_orig, _ = orig.detect_r_peaks(x, fs)
        t_orig += time.perf_counter() - t
        margin = int(round(0.16 * fs))
        ref = ref[(ref >= margin) & (ref < len(x) - margin)]
        m_port = orig.match_beats(ref, p_port, 0.15 * fs)
        m_orig = orig.match_beats(ref, p_orig, 0.15 * fs)
        for k, m in (("port", m_port), ("orig", m_orig)):
            tot[k][0] += m["true_positive"]
            tot[k][1] += m["n_reference"]
            tot[k][2] += m["n_detected"]
        worst = max(worst, m_orig["f1"] - m_port["f1"])
        rows.append((r, round(m_port["f1"], 4), round(m_orig["f1"], 4)))
        print(r, "port F1", round(m_port["f1"], 4), "orig F1", round(m_orig["f1"], 4), flush=True)
    out = {}
    for k, (tp, nr, nd) in tot.items():
        se, ppv = tp / nr, tp / nd
        out[k] = {"se": round(se, 4), "ppv": round(ppv, 4), "f1": round(2 * se * ppv / (se + ppv), 4)}
    out["worst_record_f1_drop_vs_orig"] = round(worst, 4)
    out["seconds"] = {"port": round(t_port, 1), "orig": round(t_orig, 1)}
    print(json.dumps(out, indent=1))
    Path(__file__).with_name("validate_rpeaks_result.json").write_text(
        json.dumps({"summary": out, "per_record": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
