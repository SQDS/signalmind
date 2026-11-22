"""Amendment F3 analyses on p6confirm: (a) protocol attempts, (b) all attempts, (c) complete-case paired, plus
missingness. Uses the frozen eval/confirm_analysis.analyse unchanged; (c) only restricts the case list.

  .venv/bin/python eval/confirm_sensitivity.py --out eval/results/p6confirm
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import eval.confirm_analysis as ca

M = "claude-opus-5-5-medium"
C_PROTOCOL = "+".join(f"p6confirm-C-{M}-conf-w{k}" for k in range(3))
B_PROTOCOL = "+".join(f"p6confirm-B-{M}-conf-w{k}" for k in range(3))
C_ALL = C_PROTOCOL + f"+p6confirm-C-{M}-conf-recovery-w0+p6confirm-C-{M}-conf-final-w0"
B_ALL = B_PROTOCOL + "+" + "+".join(f"p6confirm-B-{M}-conf-final-w{k}" for k in range(3))


def answered(runs: str) -> set[str]:
    rows, _ = ca.rows_of(runs)
    return {c for c, r in rows.items() if r.get("status") == "ok"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a0 = Path("eval/results/p6confirm/a0-R.predictions.json")
    res = {"a_protocol_attempts": ca.analyse("p6confirm", a0, C_PROTOCOL, B_PROTOCOL),
           "b_all_attempts": ca.analyse("p6confirm", a0, C_ALL, B_ALL)}  # fmt: skip
    answers = [json.loads(x) for x in (ca.CASES / "p6confirm.answers.jsonl").read_text().splitlines()]
    both = answered(C_ALL) & answered(B_ALL)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "p6confirm.answers.jsonl").write_text(
            "".join(json.dumps(a) + "\n" for a in answers if a["case_id"] in both)
        )
        real = ca.CASES
        ca.CASES = Path(tmp)
        try:
            res["c_complete_case"] = ca.analyse("p6confirm", a0, C_ALL, B_ALL)
        finally:
            ca.CASES = real
    sched = json.loads(Path("eval/results/p6confirm-interleaved/conf-schedule.json").read_text())["plan"]
    pos = {p["case_id"]: i for i, p in enumerate(sched)}
    b_ok, c_ok = answered(B_ALL), answered(C_ALL)
    miss = {}
    for key, fn in (("endpoint", lambda a: a["qtype"]), ("database", lambda a: a["answer"]["database"])):
        groups: dict = {}
        for a in answers:
            g = groups.setdefault(fn(a), {"n": 0, "B_missing": 0, "C_missing": 0})
            g["n"] += 1
            g["B_missing"] += a["case_id"] not in b_ok
            g["C_missing"] += a["case_id"] not in c_ok
        miss[key] = groups
    quart = {}
    for a in answers:
        q = min(3, pos[a["case_id"]] * 4 // len(sched))
        g = quart.setdefault(f"schedule_quarter_{q + 1}", {"n": 0, "B_missing": 0})
        g["n"] += 1
        g["B_missing"] += a["case_id"] not in b_ok
    miss["schedule_position"] = dict(sorted(quart.items()))
    res["missingness"] = miss
    res["complete_case_n"] = len(both)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis-f3.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"complete_case_n": len(both), "missingness": miss}, indent=1))


if __name__ == "__main__":
    main()
