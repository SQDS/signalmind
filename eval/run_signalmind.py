"""C_tools vs C_full sessions through the frozen harness (docs/signalmind/evaluation-freeze.md).

  .venv/bin/python eval/run_signalmind.py dry-run --arm C_full --suite p10pilot --only <case>
  .venv/bin/python eval/run_signalmind.py run --suite p10pilot --tag pilot --cap 25 --live   (authorised runs only)
  .venv/bin/python eval/run_signalmind.py watch --suite p10abp --tag main --cap 25 --runner-pid <pid>

Arms (one backend, same model/effort/timeout/prompt template; only the tsmcp server flags differ):
  C_tools  --plugins pressure --runtime log      same tools and submission format, no validation feedback
  C_full   --plugins pressure --runtime verify   claim validation, structured deficiencies, bounded repair

The frozen harness `eval/run_claude_code.py` runs every session unchanged (its `main()`); this wrapper registers the
suite's per-question answer schema (as the Level 2 wrapper did), appends the server flags, points the server at a
per-session runtime directory inside the run directory (never inside the model's work directory) and passes the
item's TaskSpec. The application answer is `runtime/<case>/answer.json`; the ANSWER line is kept but not graded.
Budget: spend = result rows + unfinished transcripts + all run dirs matching the suite tags; no session starts once
spend + reserve >= cap; a watcher kills everything at the cap. Infrastructure stop at > 10 % failed sessions; one
retry only for `api_error`. DISABLE_AUTOUPDATER=1; client version recorded before and after the run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "eval" / "runs"
CASES = ROOT / "eval" / "cases"
STATE = ROOT / "eval" / "results" / "signalmind-run"
MODEL_TAG = "claude-opus-5-5-medium"
ARMS = {
    "C_tools": ["--plugins", "pressure", "--runtime", "log"],
    "C_full": ["--plugins", "pressure", "--runtime", "verify"],
}
LOCK = threading.Lock()


def question(suite: str, case_id: str) -> dict:
    for line in (CASES / f"{suite}.questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        if q["case_id"] == case_id:
            return q
    raise SystemExit(f"unknown case {case_id} in {suite}")


def answer_schema(q: dict) -> str:
    schema = "{" + ", ".join(json.dumps(k) + ": <value>" for k in q["answer_fields"]) + "}"
    return schema + ' OR {"insufficient_evidence": true, "reason": "<reason>"}'


def run_dir(suite: str, tag: str, arm: str, worker: int) -> Path:
    return RUNS / f"{suite}-C-{MODEL_TAG}-{tag}-{arm}-w{worker}"


def prepare_patch(suite: str, case_id: str, arm: str, rt_dir: Path):
    """Return a prepare_case wrapper that appends the arm's server flags, runtime dir and TaskSpec."""
    from eval import common
    from eval import run_claude_code as rcc

    q = question(suite, case_id)
    common.ANSWER_FORMATS[q["qtype"]] = answer_schema(q)
    original = rcc.prepare_case
    spec = CASES / suite / case_id / "spec.json"
    server_args = [*ARMS[arm], "--runtime-dir", str(rt_dir)] + (["--task-spec", str(spec)] if spec.exists() else [])

    def prepare(case, *a, **kw):
        plan = original(case, *a, **kw)
        cfg = plan["mcp_config"]["mcpServers"]["tsmcp"]
        cfg["args"] = cfg["args"] + server_args
        Path(plan["base"], "mcp.json").write_text(json.dumps(plan["mcp_config"]))
        return plan

    return prepare, server_args, original


def session(suite: str, arm: str, tag: str, worker: int, case_id: str, server_root: Path) -> None:
    from eval import run_claude_code as rcc

    rd = run_dir(suite, tag, arm, worker)
    rt_dir = rd / "runtime" / case_id
    rt_dir.mkdir(parents=True, exist_ok=True)
    prepare, _, _ = prepare_patch(suite, case_id, arm, rt_dir)
    rcc.prepare_case = prepare
    sys.argv = ["run_claude_code.py", "--suite", suite, "--arm", "C", "--server-root", str(server_root), "--label",
                f"{tag}-{arm}-w{worker}", "--effort", "medium", "--case-timeout", "900", "--only", case_id]  # fmt: skip
    rcc.main()


def dry_run(suite: str, arm: str, case_id: str, server_root: Path) -> None:
    from eval import run_claude_code as rcc
    from eval.common import load_provider, load_suite

    case = next(c for c in load_suite(suite) if c["case_id"] == case_id)
    rt_dir = Path("/tmp") / f"signalmind-dry-{arm}-{case_id}"
    prepare, server_args, _ = prepare_patch(suite, case_id, arm, rt_dir)
    plan = prepare(case, "C", load_provider("anthropic"), "medium", True, server_root)
    work_files = sorted(p.name for p in plan["cwd"].iterdir())
    print(json.dumps({"arm": arm, "server_args": server_args, "work_files": work_files,
                      "mcp_server_args": plan["mcp_config"]["mcpServers"]["tsmcp"]["args"],
                      "allowed_tools": plan["command"][plan["command"].index("--allowedTools") + 1],
                      "prompt_tail": plan["stdin"][-260:], "runtime_dir_inside_work": str(rt_dir).startswith(str(plan["cwd"]))},
                     indent=1))  # fmt: skip
    shutil.rmtree(plan["base"], ignore_errors=True)
    assert rcc.CASE_TIMEOUT_S == 900


# ---- scheduling, budget, watcher (same rules as eval/run_level2.py) -----------------------------------------------------
def partial_cost(path: Path) -> float:
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


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


RESUME_FROM: list[str] = []  # earlier tags whose completed sessions count as done (resume after a stop)


def done(suite, tag, arm, worker, cid) -> bool:
    for t in [tag, *RESUME_FROM]:
        for w in {worker, 0, 1, 2, 3}:  # a resumed schedule keeps the seeded order; worker assignment may differ
            if any(r["case_id"] == cid for r in rows(run_dir(suite, t, arm, w) / "results.jsonl")):
                return True
    return False


def errors(suite, tag, arm, worker, cid) -> list[str]:
    return [r.get("class") for r in rows(run_dir(suite, tag, arm, worker) / "errors.jsonl") if r.get("case_id") == cid]


def infra_failures(suite, tag, workers) -> int:
    n = 0
    for arm in ARMS:
        for w in range(workers):
            ids = {
                r["case_id"]
                for r in rows(run_dir(suite, tag, arm, w) / "errors.jsonl")
                if r.get("class") == "api_error"
            }
            n += sum(not done(suite, tag, arm, w, cid) for cid in ids)
    return n


def claude_version() -> str:
    try:
        return subprocess.check_output(["claude", "--version"], text=True, timeout=30).strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


def worker_loop(args, worker, items, log, planned, stop) -> None:
    from eval.run_claude_code import HOST_SESSION_VARS

    env = {k: v for k, v in os.environ.items() if k not in HOST_SESSION_VARS}
    env["DISABLE_AUTOUPDATER"] = "1"
    for it in items:
        for arm in it["arm_order"]:
            if stop.is_set() or done(args.suite, args.tag, arm, worker, it["case_id"]):
                continue
            with LOCK:
                s = spend(args.spend_glob)
                if s + args.reserve >= args.cap:
                    log.append(
                        {"event": "budget_stop", "worker": worker, "case_id": it["case_id"], "arm": arm, "spend": s}
                    )
                    stop.set()
                    return
                failed = infra_failures(args.suite, args.tag, args.workers) + sum(
                    e["event"] == "session_failed" for e in log
                )
                if failed > 0.10 * planned:
                    log.append(
                        {
                            "event": "infra_stop",
                            "worker": worker,
                            "case_id": it["case_id"],
                            "arm": arm,
                            "failed": failed,
                        }
                    )
                    stop.set()
                    return
            cmd = [sys.executable, str(Path(__file__)), "session", "--suite", args.suite, "--arm", arm, "--tag", args.tag,
                   "--worker", str(worker), "--case", it["case_id"], "--server-root", str(args.server_root)]  # fmt: skip
            p = subprocess.Popen(cmd, cwd=ROOT, env=env)
            with LOCK, open(STATE / f"{args.tag}-pids.txt", "a") as f:
                f.write(f"{p.pid}\n")
            p.wait()
            if not done(args.suite, args.tag, arm, worker, it["case_id"]) and not errors(
                args.suite, args.tag, arm, worker, it["case_id"]
            ):
                with LOCK:
                    log.append({"event": "session_failed", "worker": worker, "case_id": it["case_id"], "arm": arm,
                                "returncode": p.returncode})  # fmt: skip


def assert_pinned_worktree(root: Path) -> str:
    """Controlled runs are served from a detached, clean worktree at a pinned commit, never from the working repo."""
    root = root.expanduser().resolve()
    if root == ROOT.resolve():
        raise SystemExit("server root must be a detached worktree at a pinned commit, not the working repository")
    if subprocess.run(["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 0:
        raise SystemExit(f"server root {root} is on a branch; use a detached worktree (git worktree add --detach)")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise SystemExit(f"server root {root} has uncommitted changes")
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def run(args) -> None:
    if not args.live:
        raise SystemExit("live runs need --live (authorised evaluation only; see docs/signalmind/evaluation-freeze.md)")
    pinned = assert_pinned_worktree(args.server_root)
    RESUME_FROM[:] = list(args.resume_from or [])
    ids = [json.loads(x)["case_id"] for x in (CASES / f"{args.suite}.questions.jsonl").read_text().splitlines()]
    if args.only:
        ids = [i for i in ids if i in set(args.only)]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    plan = [
        {"case_id": c, "worker": k % args.workers, "arm_order": rng.sample(list(ARMS), 2)} for k, c in enumerate(ids)
    ]
    if args.arms:  # the schedule is drawn as for both arms; --arms only filters what executes
        plan = [{**p, "arm_order": [a for a in p["arm_order"] if a in args.arms]} for p in plan]
    STATE.mkdir(parents=True, exist_ok=True)
    meta = {"seed": args.seed, "workers": args.workers, "cap": args.cap, "suite": args.suite, "plan": plan,
            "server_root": str(args.server_root), "server_commit": pinned,
            "claude_version_start": claude_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    (STATE / f"{args.tag}-schedule.json").write_text(json.dumps(meta, indent=1))
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
                items = [{**p, "arm_order": [a for a in p["arm_order"] if not done(args.suite, args.tag, a, w, p["case_id"])
                                             and errors(args.suite, args.tag, a, w, p["case_id"]) == ["api_error"]]} for p in items]  # fmt: skip
                items = [p for p in items if p["arm_order"]]
                log += [{"event": "retry", "worker": w, "case_id": p["case_id"], "arms": p["arm_order"]} for p in items]
            t = threading.Thread(target=worker_loop, args=(args, w, items, log, planned, stop))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    meta.update(claude_version_end=claude_version(), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (STATE / f"{args.tag}-schedule.json").write_text(json.dumps(meta, indent=1))
    (STATE / f"{args.tag}-execution-log.json").write_text(
        json.dumps({"spend": spend(args.spend_glob), "events": log}, indent=1)
    )
    print(json.dumps({"spend": spend(args.spend_glob), "events": len(log)}))


def watch(args) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    while Path(f"/proc/{args.runner_pid}").exists():
        s = spend(args.spend_glob)
        (STATE / f"{args.tag}-spend.txt").write_text(f"{s:.4f}\n")
        if s >= args.cap:
            pids = [args.runner_pid]
            f = STATE / f"{args.tag}-pids.txt"
            if f.exists():
                pids += [int(x) for x in f.read_text().split()]
            for pid in pids:
                if Path(f"/proc/{pid}").exists():
                    os.kill(pid, signal.SIGTERM)
            print(time.ctime(), "CAP STOP spend", round(s, 4), flush=True)
            return
        time.sleep(20)
    print(time.ctime(), "runner exited; final spend", round(spend(args.spend_glob), 4), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--arm", choices=sorted(ARMS), required=True)
    d.add_argument("--suite", required=True)
    d.add_argument("--only", required=True)
    d.add_argument("--server-root", type=Path, default=ROOT)
    s = sub.add_parser("session")
    s.add_argument("--suite", required=True)
    s.add_argument("--arm", choices=sorted(ARMS), required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--worker", type=int, required=True)
    s.add_argument("--case", required=True)
    s.add_argument("--server-root", type=Path, default=ROOT)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--seed", type=int, default=20261028)
    r.add_argument("--cap", type=float, required=True)
    r.add_argument("--reserve", type=float, default=1.5)
    r.add_argument("--spend-glob", nargs="+", default=["p10pilot-C-*", "p10abp-C-*", "p7dev-C-*-smsmoke*"])
    r.add_argument("--only", nargs="*")
    r.add_argument("--server-root", type=Path, default=ROOT)
    r.add_argument("--live", action="store_true")
    r.add_argument("--resume-from", nargs="*", help="earlier tag(s) whose completed sessions are not repeated")
    r.add_argument("--arms", nargs="+", choices=sorted(ARMS), help="run only these arms (labelled follow-ups)")
    w = sub.add_parser("watch")
    w.add_argument("--tag", required=True)
    w.add_argument("--cap", type=float, required=True)
    w.add_argument("--runner-pid", type=int, required=True)
    w.add_argument("--spend-glob", nargs="+", default=["p10pilot-C-*", "p10abp-C-*", "p7dev-C-*-smsmoke*"])
    args = ap.parse_args()
    if args.cmd == "dry-run":
        dry_run(args.suite, args.arm, args.only, args.server_root)
    elif args.cmd == "session":
        session(args.suite, args.arm, args.tag, args.worker, args.case, args.server_root)
    elif args.cmd == "run":
        run(args)
    else:
        watch(args)


if __name__ == "__main__":
    main()
