"""Synthetic 200k-sample timings; same signals/seed/repetitions as the initial generic anomaly report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.discord_baseline import single_scale_discords  # noqa: E402
from eval.discord_variants import variant_discords  # noqa: E402
from tsmcp.core.detect import discords  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("old", "current", "append", "rrf", "robust_z", "independent"),
        default=["old", "current", "append", "rrf", "robust_z", "independent"],
    )
    args = parser.parse_args()
    methods = {"old": single_scale_discords, "current": discords}
    methods.update(
        {name: partial(variant_discords, variant=name) for name in ("append", "rrf", "robust_z", "independent")}
    )
    rng = np.random.default_rng(1)
    rows = []
    n = 200_000
    for period in (73, 100, 1000, None):
        noise = rng.standard_normal(n)
        x = noise if period is None else np.sin(2 * np.pi * np.arange(n) / period) + 0.03 * noise
        row = {"period_samples": period, "n": n, "methods": {}}
        for name in args.methods:
            seconds = []
            for _ in range(5):
                start = time.perf_counter()
                methods[name](x, 100.0, k=5)
                seconds.append(time.perf_counter() - start)
            result = {"seconds": seconds, "median_s": float(np.median(seconds)), "max_s": max(seconds)}
            row["methods"][name] = result
            print(f"period={period} {name}: median={result['median_s']:.4f}s max={result['max_s']:.4f}s", flush=True)
        rows.append(row)
    args.output.write_text(
        json.dumps(
            {
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
                "numpy": np.__version__,
                "seed": 1,
                "fs": 100.0,
                "k": 5,
                "repetitions": 5,
                "warmup": 0,
                "discarded": 0,
                "series": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
