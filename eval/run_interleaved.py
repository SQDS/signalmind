"""Interleaved B/C execution for the registered Part 2 runs (docs/tasks/verification-experiment.md).

  .venv/bin/python eval/run_interleaved.py --suite p6confirm --tag conf --workers 3 --seed 20261021 --cap 140 \
      --server-root ~/tsmcp-server-R [--only ID ...] [--dry-run]

One fixed seeded case order, split round-robin over workers. Each worker runs both arms for a case back to back, the arm
order randomised per case (seeded). Each worker writes its own run directories (`<suite>-{B,C}-<model>-<effort>-<tag>-w<k>`),
so no two processes append to the same file. Before every session the stage spend (max of harness and Claude Code
cost, over every run dir matching `--spend-glob`) is checked; no new session starts once spend + reserve reaches the
cap. After the main pass, sessions that ended in an infrastructure error (`api_error`, no result row) get exactly one
retry pass. Timeouts and B single-turn contract failures are never retried. The schedule is written to
`eval/results/<suite>-interleaved/<tag>-schedule.json`.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"
CASES = ROOT / "eval" / "cases"
MODEL_TAG = "claude-opus-5-5-medium"
LOCK = threading.Lock()


def spend(patterns: list[str]) -> float:
    total = 0.0
    for pat in patterns:
        for d in glob.glob(str(RUNS / pat)):
            f = Path(d) / "results.jsonl"
            if f.exists():
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    total += max(r.get("cost_usd") or 0, r.get("cc_cost_usd") or 0)
    return total


def run_dir(suite: str, arm: str, tag: str, worker: int) -> Path:
    return RUNS / f"{suite}-{arm}-{MODEL_TAG}-{tag}-w{worker}"


def done(suite: str, arm: str, tag: str, worker: int, cid: str) -> bool:
    f = run_dir(suite, arm, tag, worker) / "results.jsonl"
    return f.exists() and any(json.loads(x)["case_id"] == cid for x in f.read_text().splitlines())


def infra_failed(suite: str, arm: str, tag: str, worker: int, cid: str) -> bool:
    f = run_dir(suite, arm, tag, worker) / "errors.jsonl"
    if not f.exists() or done(suite, arm, tag, worker, cid):
        return False
    return any(json.loads(x).get("case_id") == cid and json.loads(x).get("class") == "api_error"
               for x in f.read_text().splitlines() if x.strip())  # fmt: skip


def command(args, arm: str, worker: int, cid: str) -> list[str]:
    cmd = [sys.executable, str(ROOT / "eval" / "run_claude_code.py"), "--suite", args.suite, "--arm", arm,
           "--server-root", str(args.server_root), "--label", f"{args.tag}-w{worker}", "--effort", "medium",
           "--case-timeout", "900", "--only", cid]  # fmt: skip
    if arm == "B":
        cmd += ["--b-workflow", "v0"]
    return cmd


def worker_loop(args, worker: int, items: list[dict], log: list) -> None:
    for it in items:
        for arm in it["arm_order"]:
            if done(args.suite, arm, args.tag, worker, it["case_id"]):
                continue
            with LOCK:
                s = spend(args.spend_glob)
                if s + args.reserve >= args.cap:
                    log.append(
                        {"event": "budget_stop", "worker": worker, "case_id": it["case_id"], "arm": arm, "spend": s}
                    )
                    return
            subprocess.run(command(args, arm, worker, it["case_id"]), cwd=ROOT, check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--reserve", type=float, default=3.0, help="spend held back for in-flight sessions")
    ap.add_argument("--server-root", type=Path, required=True)
    ap.add_argument("--spend-glob", nargs="+", required=True, help="run-dir globs that count toward the cap")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--arms", nargs="+", choices=("B", "C"), default=["B", "C"], help="arms to run (default both)")
    ap.add_argument(
        "--no-retry", action="store_true", help="skip the retry pass (used for the single recovery attempt)"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ids = [json.loads(x)["case_id"] for x in (CASES / f"{args.suite}.questions.jsonl").read_text().splitlines()]
    if args.only:
        ids = [i for i in ids if i in set(args.only)]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    plan = [
        {"case_id": c, "worker": k % args.workers, "arm_order": rng.sample(["B", "C"], 2)} for k, c in enumerate(ids)
    ]
    # The schedule (order, workers, arm order) is drawn exactly as in the frozen run; --arms only filters what executes.
    plan = [{**p, "arm_order": [a for a in p["arm_order"] if a in args.arms]} for p in plan]
    out = ROOT / "eval" / "results" / f"{args.suite}-interleaved"
    suffix = "" if sorted(args.arms) == ["B", "C"] else "-" + "".join(args.arms)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}-schedule{suffix}.json").write_text(json.dumps({"seed": args.seed, "workers": args.workers,
                                                                "cap": args.cap, "plan": plan}, indent=1))  # fmt: skip
    if args.dry_run:
        for p in plan[:4]:
            print(p["worker"], p["arm_order"], " ".join(command(args, p["arm_order"][0], p["worker"], p["case_id"])))
        return
    log: list = []
    for phase in ("main",) if args.no_retry else ("main", "retry"):
        threads = []
        for w in range(args.workers):
            items = [p for p in plan if p["worker"] == w]
            if phase == "retry":
                items = [{**p, "arm_order": [a for a in p["arm_order"]
                                             if infra_failed(args.suite, a, args.tag, w, p["case_id"])]} for p in items]  # fmt: skip
                items = [p for p in items if p["arm_order"]]
                log += [{"event": "retry", "worker": w, "case_id": p["case_id"], "arms": p["arm_order"]} for p in items]
            t = threading.Thread(target=worker_loop, args=(args, w, items, log))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    (out / f"{args.tag}-execution-log{suffix}.json").write_text(
        json.dumps({"spend": spend(args.spend_glob), "events": log}, indent=1)
    )
    print(json.dumps({"spend": spend(args.spend_glob), "events": len(log)}))


if __name__ == "__main__":
    main()
