"""Interleaved C/D execution of the frozen Level 2 development suite p9level2 (docs/tasks/level2-analysis-plan.md).

  .venv/bin/python eval/run_level2.py run --tag l2 --workers 2 --seed 20261027 --cap 40 [--only ID ...] [--dry-run]

The frozen harness `eval/run_claude_code.py` is used unchanged (byte-identical; its `main()` runs every session). This
wrapper adds exactly what the frozen L2-A dry run (`eval/level2_checks.py::_dry_runs`) did around `prepare_case`:
the per-question answer schema registered as qtype `level2`, and a copy of the public `visible.json` into the work
directory, identically for both arms. Nothing else about either arm changes.

One seeded item order, split round-robin over workers; each worker runs both arms for an item back to back, arm order
randomised per item (seeded). Each worker writes its own run directories (`p9level2-{C,D}-<model>-<effort>-<tag>-w<k>`).
Before every session the stage spend (results rows, plus usage of any unfinished or timed-out partial transcript,
plus the smoke run) is checked; no session starts once spend + reserve reaches the cap. A run stops when more than
10 % of planned sessions have failed for infrastructure reasons (`api_error`). After the main pass, sessions whose only
outcome is `api_error` get exactly one retry. Timeouts are model outcomes and are never retried.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "eval" / "runs"
CASES = ROOT / "eval" / "cases"
MANIFEST = ROOT / "eval" / "tasks" / "p9level2_manifest.json"
SUITE = "p9level2"
MODEL_TAG = "claude-opus-5-5-medium"
SERVER_ROOT = Path.home() / "tsmcp-server-R"
STATE = ROOT / "eval" / "results" / "p9level2-run"
LOCK = threading.Lock()


def answer_schema(case_id: str) -> str:
    """The answer line exactly as the frozen dry run built it (field names only; no reference value is read)."""
    item = next(q for q in json.loads(MANIFEST.read_text())["questions"] if q["case_id"] == case_id)
    schema = "{" + ", ".join(json.dumps(key) + ": <value>" for key in item["answer_fields"]) + "}"
    return schema + ' OR {"insufficient_evidence": true, "reason": "<reason>"}'


def session(arm: str, tag: str, worker: int, case_id: str) -> None:
    """One harness session through the unmodified `run_claude_code.main()`."""
    from eval import common
    from eval import run_claude_code as rcc

    common.ANSWER_FORMATS["level2"] = answer_schema(case_id)
    original = rcc.prepare_case

    def prepare_with_visible(case, *a, **kw):
        plan = original(case, *a, **kw)
        shutil.copyfile(Path(case["dir"]) / "visible.json", plan["cwd"] / "visible.json")
        return plan

    rcc.prepare_case = prepare_with_visible
    sys.argv = ["run_claude_code.py", "--suite", SUITE, "--arm", arm, "--server-root", str(SERVER_ROOT),
                "--label", f"{tag}-w{worker}", "--effort", "medium", "--case-timeout", "900", "--only", case_id]  # fmt: skip
    rcc.main()


def partial_cost(path: Path) -> float:
    """Usage of an unfinished transcript (timeout or in flight): last usage per message id, frozen prices."""
    from eval.common import MODEL, cost_usd

    last: dict[str, dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "assistant" and isinstance(e.get("message"), dict) and e["message"].get("usage"):
            last[e["message"].get("id", str(len(last)))] = e["message"]["usage"]
    return sum(cost_usd(MODEL, u) for u in last.values())


def spend(patterns: list[str]) -> float:
    total = 0.0
    for pat in patterns:
        for d in glob.glob(str(RUNS / pat)):
            f = Path(d) / "results.jsonl"
            if f.exists():
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    total += max(r.get("cost_usd") or 0, r.get("cc_cost_usd") or 0)
            total += sum(partial_cost(p) for p in Path(d).glob("transcripts/*.partial.jsonl"))
    return total


def run_dir(arm: str, tag: str, worker: int) -> Path:
    return RUNS / f"{SUITE}-{arm}-{MODEL_TAG}-{tag}-w{worker}"


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def done(arm: str, tag: str, worker: int, cid: str) -> bool:
    return any(r["case_id"] == cid for r in rows(run_dir(arm, tag, worker) / "results.jsonl"))


def errors(arm: str, tag: str, worker: int, cid: str) -> list[str]:
    return [r.get("class") for r in rows(run_dir(arm, tag, worker) / "errors.jsonl") if r.get("case_id") == cid]


def infra_failures(tag: str, workers: int) -> int:
    """Sessions (arm, item) with at least one api_error and no result row."""
    n = 0
    for arm in ("C", "D"):
        for w in range(workers):
            ids = {r["case_id"] for r in rows(run_dir(arm, tag, w) / "errors.jsonl") if r.get("class") == "api_error"}
            n += sum(not done(arm, tag, w, cid) for cid in ids)
    return n


def worker_loop(args, worker: int, items: list[dict], log: list, planned: int, stop: threading.Event) -> None:
    from eval.run_claude_code import HOST_SESSION_VARS

    env = {k: v for k, v in os.environ.items() if k not in HOST_SESSION_VARS}
    env["DISABLE_AUTOUPDATER"] = "1"  # incident E1: the client must not replace itself mid-run
    for it in items:
        for arm in it["arm_order"]:
            if stop.is_set() or done(arm, args.tag, worker, it["case_id"]):
                continue
            with LOCK:
                s = spend(args.spend_glob)
                if s + args.reserve >= args.cap:
                    log.append(
                        {"event": "budget_stop", "worker": worker, "case_id": it["case_id"], "arm": arm, "spend": s}
                    )
                    stop.set()
                    return
                failed = infra_failures(args.tag, args.workers) + sum(e["event"] == "session_failed" for e in log)
                if failed > 0.10 * planned:
                    log.append({"event": "infra_stop", "worker": worker, "case_id": it["case_id"], "arm": arm,
                                "infra_failures": failed})  # fmt: skip
                    stop.set()
                    return
            cmd = [sys.executable, str(Path(__file__)), "session", "--arm", arm, "--tag", args.tag,
                   "--worker", str(worker), "--case", it["case_id"]]  # fmt: skip
            p = subprocess.Popen(cmd, cwd=ROOT, env=env)
            with LOCK, open(STATE / f"{args.tag}-pids.txt", "a") as f:
                f.write(f"{p.pid}\n")
            p.wait()
            if not done(arm, args.tag, worker, it["case_id"]) and not errors(arm, args.tag, worker, it["case_id"]):
                with LOCK:
                    log.append({"event": "session_failed", "worker": worker, "case_id": it["case_id"], "arm": arm,
                                "returncode": p.returncode})  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("session")
    s.add_argument("--arm", choices=("C", "D"), required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--worker", type=int, required=True)
    s.add_argument("--case", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--cap", type=float, required=True)
    r.add_argument("--reserve", type=float, default=3.0, help="spend held back for in-flight sessions")
    r.add_argument("--spend-glob", nargs="+", default=["p9level2-C-*", "p9level2-D-*", "p7dev-C-*-l2smoke*"])
    r.add_argument("--only", nargs="*")
    r.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "session":
        session(args.arm, args.tag, args.worker, args.case)
        return
    ids = [json.loads(x)["case_id"] for x in (CASES / f"{SUITE}.questions.jsonl").read_text().splitlines()]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    plan = [
        {"case_id": c, "worker": k % args.workers, "arm_order": rng.sample(["C", "D"], 2)} for k, c in enumerate(ids)
    ]
    if args.only:
        plan = [p for p in plan if p["case_id"] in set(args.only)]
    STATE.mkdir(parents=True, exist_ok=True)
    suffix = "" if not args.only else "-only"
    (STATE / f"{args.tag}-schedule{suffix}.json").write_text(json.dumps({"seed": args.seed, "workers": args.workers,
                                                             "cap": args.cap, "plan": plan}, indent=1))  # fmt: skip
    if args.dry_run:
        for p in plan[:4]:
            print(p["worker"], p["arm_order"], p["case_id"], answer_schema(p["case_id"]))
        return
    planned = 2 * len(ids)
    log: list = []
    stop = threading.Event()
    for phase in ("main", "retry"):
        if stop.is_set():
            break
        threads = []
        for w in range(args.workers):
            items = [p for p in plan if p["worker"] == w]
            if phase == "retry":
                items = [{**p, "arm_order": [a for a in p["arm_order"] if not done(a, args.tag, w, p["case_id"])
                                             and set(errors(a, args.tag, w, p["case_id"])) == {"api_error"}
                                             and len(errors(a, args.tag, w, p["case_id"])) == 1]} for p in items]  # fmt: skip
                items = [p for p in items if p["arm_order"]]
                log += [{"event": "retry", "worker": w, "case_id": p["case_id"], "arms": p["arm_order"]} for p in items]
            t = threading.Thread(target=worker_loop, args=(args, w, items, log, planned, stop))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    (STATE / f"{args.tag}-execution-log{suffix}.json").write_text(
        json.dumps({"spend": spend(args.spend_glob), "events": log}, indent=1)
    )
    print(json.dumps({"spend": spend(args.spend_glob), "events": len(log)}))


if __name__ == "__main__":
    main()
