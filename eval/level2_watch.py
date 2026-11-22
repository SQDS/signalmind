"""Independent hard-cap watcher for the Level 2 run (docs/tasks/level2-analysis-plan.md §2).

  .venv/bin/python eval/level2_watch.py --tag l2 --cap 40 --runner-pid <pid>

Every 20 s: writes the stage spend (same accounting as the runner, including unfinished transcripts and the smoke run)
to `eval/results/p9level2-run/<tag>-spend.txt`; at or above the cap it sends SIGTERM to the runner and to every session
process the runner registered, then exits.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_level2 import STATE, spend  # noqa: E402

PATTERNS = ["p9level2-C-*", "p9level2-D-*", "p7dev-C-*-l2smoke*"]


def alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--runner-pid", type=int, required=True)
    args = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    while alive(args.runner_pid):
        s = spend(PATTERNS)
        (STATE / f"{args.tag}-spend.txt").write_text(f"{s:.4f}\n")
        if s >= args.cap:
            pids = [args.runner_pid]
            f = STATE / f"{args.tag}-pids.txt"
            if f.exists():
                pids += [int(x) for x in f.read_text().split()]
            for pid in pids:
                if alive(pid):
                    os.kill(pid, signal.SIGTERM)
            print(time.ctime(), "CAP STOP spend", round(s, 4), flush=True)
            return
        time.sleep(20)
    print(time.ctime(), "runner exited; final spend", round(spend(PATTERNS), 4), flush=True)


if __name__ == "__main__":
    main()
