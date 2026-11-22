"""Build a case suite.

  uv run --group eval python eval/build_cases.py --suite p1 --synthetic dev --mitdb ds1 --ucr 5

Layout: eval/cases/<suite>/<case_id>/{signal.npy, visible.json}   <- the only things a model arm may see
        eval/cases/<suite>.questions.jsonl                         <- public: case_id, qtype, question
        eval/cases/<suite>.answers.jsonl                           <- private: ground truth (never inside a case dir)
Test-family seeds are passed on the command line and are not committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import CASES, read_jsonl  # noqa: E402
from eval.tasks import afdb, confirm, longrec, mitdb, p7dev, synthetic, ucr  # noqa: E402


def write_case(suite_dir: Path, cid: str, x: np.ndarray, fs: float, unit: str) -> dict:
    d = suite_dir / cid
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "signal.npy", np.asarray(x, dtype=np.float32), allow_pickle=False)
    visible = {
        "file": "signal.npy",
        "dur_s": round(x.size / fs, 3),
        "channels": [{"name": "ch0", "unit": unit, "fs": fs}],
    }
    (d / "visible.json").write_text(json.dumps(visible))
    return visible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--synthetic", choices=["dev", "test", "none"], default=None)
    ap.add_argument("--mitdb", choices=["ds1", "ds2", "none"], default=None)
    ap.add_argument("--mitdb-n", type=int, default=5, help="cases per MIT-BIH question type")
    ap.add_argument("--ucr", type=int, default=0)
    ap.add_argument("--ucr-exclude-dev", action="store_true", help="exclude p1's UCR series and the discord dev series")
    ap.add_argument("--ltafdb", type=int, default=0, help="cases per LTAFDB question type (24 h ECG)")
    ap.add_argument(
        "--afdb",
        action="store_true",
        help="all frozen AFDB endpoint assignments; one case per eligible record and endpoint",
    )
    ap.add_argument(
        "--confirm", action="store_true", help="sealed p6confirm patient-level SHDB-AF / IRIDIA-AF endpoints"
    )
    ap.add_argument("--p7dev", action="store_true", help="opened LTAFDB records with frozen Part 1 + C1 definitions")
    ap.add_argument("--synthetic-long", type=int, default=0, help="number of synthetic 24 h needle cases")
    ap.add_argument(
        "--ltafdb-split", choices=["all", "test_ext"], default="all", help="restrict LTAFDB records to a split"
    )
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--test-seed", type=int, default=None, help="sealed seed for the test family (do not commit)")
    args = ap.parse_args()

    args.synthetic = args.synthetic or ("none" if args.afdb or args.confirm or args.p7dev else "dev")
    args.mitdb = args.mitdb or ("none" if args.afdb or args.confirm or args.p7dev else "ds1")
    frozen_afdb = None
    frozen_confirm = None
    frozen_p7dev = None
    if args.p7dev:
        if args.suite != "p7dev":
            ap.error("--p7dev requires --suite p7dev")
        if (
            args.synthetic != "none"
            or args.mitdb != "none"
            or args.ucr
            or args.ltafdb
            or args.synthetic_long
            or args.afdb
            or args.confirm
        ):
            ap.error("--p7dev must not be combined with other datasets")
        try:
            frozen_p7dev = p7dev.prepare_p7dev()
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    elif args.suite == "p7dev":
        ap.error("p7dev requires --p7dev")
    if args.confirm:
        if args.suite != "p6confirm":
            ap.error("--confirm requires --suite p6confirm")
        if (
            args.synthetic != "none"
            or args.mitdb != "none"
            or args.ucr
            or args.ltafdb
            or args.synthetic_long
            or args.afdb
        ):
            ap.error("--confirm must not be combined with other datasets")
        try:
            frozen_confirm = confirm.prepare_confirm()
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    elif args.suite == "p6confirm":
        ap.error("p6confirm requires --confirm")
    if args.afdb:
        if args.synthetic != "none" or args.mitdb != "none" or args.ucr or args.ltafdb or args.synthetic_long:
            ap.error("--afdb builds independently eligible AFDB endpoints; do not combine it with other datasets")
        try:
            frozen_afdb = afdb.prepare_afdb()
        except (OSError, ValueError) as exc:
            ap.error(str(exc))

    sdir = CASES / args.suite
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    qs, ans = [], []

    def add(cid, qtype, variant, x, fs, unit, answer, question, source):
        write_case(sdir, cid, x, fs, unit)
        qs.append({"case_id": cid, "qtype": qtype, "question": question, "source": source})
        ans.append({"case_id": cid, "qtype": qtype, "variant": variant, "answer": answer})

    if args.synthetic != "none":
        seed = args.seed if args.synthetic == "dev" else args.test_seed
        if seed is None:
            sys.exit("--test-seed is required for the test family")
        for cid, q, v, x, a, question in synthetic.build(seed, args.synthetic):
            add(cid, q, v, x, synthetic.FS, "a.u.", a, question, f"synthetic-{args.synthetic}")
    if args.mitdb != "none":
        recs = mitdb.DS1 if args.mitdb == "ds1" else mitdb.DS2
        for cid, q, v, x, fs, a, question in mitdb.build(args.seed + 1, recs, args.mitdb_n, args.mitdb_n):
            add(cid, q, v, x, fs, "mV", a, question, f"ecg-{args.mitdb}")
    if args.ucr:
        p1_held = {r["answer"]["id"] for r in read_jsonl(CASES / "p1.answers.jsonl") if r["qtype"] == "anomaly_time"}
        excl = set()
        if args.ucr_exclude_dev:
            excl = p1_held | ucr.dev_ids(p1_held)
        for cid, q, v, x, fs, a, question in ucr.build(args.seed + 2, args.ucr, exclude=excl):
            add(cid, q, v, x, fs, "a.u.", a, question, "ucr")

    if args.ltafdb:
        recs = None
        if args.ltafdb_split == "test_ext":
            recs = json.loads((Path(__file__).resolve().parent / "tasks" / "ltafdb_split.json").read_text())["test_ext"]
            recs = [
                r
                for r in recs
                if (longrec.LTAF / f"{r}.dat").exists() and (longrec.LTAF / f"{r}.dat").stat().st_size > 30_000_000
            ]
        for cid, q, v, x, fs, a, question in longrec.build_ltafdb(args.seed + 3, args.ltafdb, recs):
            add(cid, q, v, x, fs, "mV", a, question, "ecg-long")
    if args.synthetic_long:
        for cid, q, v, x, fs, a, question in longrec.build_synthetic_long(args.seed + 4, args.synthetic_long):
            add(cid, q, v, x, fs, "a.u.", a, question, "synthetic-long")

    if frozen_afdb is not None:
        for cid, q, v, x, fs, a, question in afdb.build_afdb(frozen_afdb):
            add(cid, q, v, x, fs, "mV", a, question, "ecg-long")

    if frozen_confirm is not None:
        for cid, q, v, x, fs, a, question in confirm.build_confirm(frozen_confirm):
            add(cid, q, v, x, fs, "mV", a, question, "ecg-long")

    if frozen_p7dev is not None:
        for cid, q, v, x, fs, a, question in p7dev.build_p7dev(frozen_p7dev):
            add(cid, q, v, x, fs, "mV", a, question, "ecg-long")

    (CASES / f"{args.suite}.questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qs))
    (CASES / f"{args.suite}.answers.jsonl").write_text(
        "".join(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n" for r in ans)
    )
    if frozen_p7dev is not None:
        p7dev.write_manifest(sdir, frozen_p7dev, qs, ans)
    by = {}
    for r in qs:
        by[r["qtype"]] = by.get(r["qtype"], 0) + 1
    print(f"suite {args.suite}: {len(qs)} cases {by}")


if __name__ == "__main__":
    main()
