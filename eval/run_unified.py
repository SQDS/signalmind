"""Matched-arm capability-amplification runs (docs/signalmind/phase2-protocol.md).

  .venv/bin/python eval/run_unified.py --arm DS-Base --suite p9level2 --tag dev --dry-run [--only ID ...]
  .venv/bin/python eval/run_unified.py --arm DS-Agent --suite p9level2 --tag dev --cap 10 --live   (approved runs only)

Arms: <M>-Base (model + sandboxed Python, no network; image viewing for vision models), <M>-Agent (model + full
SignalMind: TS-MCP tools + pressure/tfa/pulse/align plugins + TaskSpec + evidence ledger + verification + ts_submit),
<M>-Tools (ablation, not run without approval: the same tools and plugins, passive ledger, no verification feedback).
M = DS (deepseek-flash), H (claude-haiku), T (gpt-6-terra), A (gpt-6-astra). Within a model, provider settings, limits and public task
constraints are shared; the tool set and submission/verification instructions differ by arm.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.common import ANSWER_FORMATS, case_prompt, load_suite, parse_answer  # noqa: E402
from eval.unified.drivers import MODELS, Step, make_driver  # noqa: E402
from eval.unified.loop import Limits, run_session  # noqa: E402
from eval.unified.tasks import task_spec_for_case  # noqa: E402
from eval.unified.tools import PythonTools, SignalMindTools, copy_case  # noqa: E402
from tsmcp.runtime.taskspec import TaskSpec  # noqa: E402

RUNS = ROOT / "eval" / "runs"
PLUGINS = ["pressure", "tfa", "pulse", "align"]
MODEL_OF = {"DS": "deepseek-flash", "H": "claude-haiku", "T": "gpt-6-terra", "A": "gpt-6-astra"}
KINDS = ("Base", "Agent", "Tools")
REFERENCE_PRICE = {"in": 4.0, "out": 20.0, "cache_read": 0.2, "cache_write": 5.0}  # budgeting estimate, not billing
SYSTEM_BASE = ("You are an expert in physiological signal analysis. You can run Python in a sandbox to analyse the "
               "recorded signal. Measure what the question asks; do not guess numbers. If the evidence is insufficient, "
               "say so.")  # fmt: skip
SYSTEM_AGENT = ("You are an expert in physiological signal analysis. You can use the TS-MCP signal tools (including "
                "specialist physiological measures) to analyse the recorded signal. Measure what the question asks; do not "
                "guess numbers. If the evidence is insufficient, say so.")  # fmt: skip
AGENT_SUBMIT = ("\nSubmit your answer with ts_submit: one claim per answer field, with the field name as the claim's "
                "quantity, citing the evidence_ids of the measurements you used. Then end with the ANSWER line.")  # fmt: skip


def answer_schema(fields: list[str]) -> str:
    return "{" + ", ".join(json.dumps(k) + ": <value>" for k in fields) + "}" + \
        ' OR {"insufficient_evidence": true, "reason": "<reason>"}'  # fmt: skip


def prompt_for(case: dict, work: Path, kind: str, spec: TaskSpec) -> str:
    ANSWER_FORMATS[case["qtype"]] = answer_schema([q.name for q in spec.quantities])
    fs = case["visible"]["channels"][0]["fs"]
    p = (
        case_prompt(case["visible"], case["question"], case["qtype"])
        + f"\nThe signal file is {work / 'signal.npy'} (sampling rate {fs:g} Hz)."
        + "\nPublic task constraints (identical for all arms): "
        + json.dumps(spec.to_dict())
    )
    return p + (AGENT_SUBMIT if kind in ("Agent", "Tools") else "")


def application_answer(
    runtime_dir: Path, *, required_quantities: set[str] | None = None
) -> tuple[dict | None, dict | None, str]:
    """Extract accepted/recorded values; only requested quantities can make the answer an abstention.

    Required fields default to the saved task spec. Older runtime directories without a spec retain the prior
    extraction behaviour because their requested quantities are unknown.
    """
    f = runtime_dir / "answer.json"
    if not f.exists():
        return None, None, "absent"
    spec_path = runtime_dir / "spec.json"
    if required_quantities is None and spec_path.exists():
        required_quantities = {
            q["name"] for q in json.loads(spec_path.read_text())["quantities"] if q.get("required", True)
        }
    st = json.loads(f.read_text())
    ans = st.get("answer") or {}
    claims = ans.get("submitted_claims", []) if st.get("mode") == "log" else ans.get("accepted_claims", [])
    acc = {c["quantity"]: c["value"] for c in claims if c.get("value") is not None}
    allc = dict(acc) | {c["quantity"]: c["value"] for c in ans.get("unresolved", []) if c.get("value") is not None}
    for lim in ans.get("limitations", []) + [
        c
        for c in claims + ans.get("unresolved", [])
        if c.get("submitted_status", c.get("status")) == "insufficient_evidence"
    ]:
        if required_quantities is not None and lim["quantity"] not in required_quantities:
            continue
        acc.setdefault("insufficient_evidence", True)
        acc.setdefault("reason", lim.get("reason") or "insufficient evidence")
    allc.update({k: acc[k] for k in ("insufficient_evidence", "reason") if k in acc})
    if st.get("finalization") == "rejected" or any(f.startswith("unsupported_after") for f in st.get("flags", [])):
        acc = {}  # a rejected or invalidated bundle is not an accepted application answer
    return (acc or None), (allc or None), st.get("finalization", "absent")


class MockDriver:
    """Dry runs: a scripted model that exercises the tools and ends with an ANSWER line (no network)."""

    def __init__(self, kind: str, fields: list[str], visible: dict):
        self.kind, self.fields, self.visible, self.n = kind, fields, visible, 0

    def step(self, system, messages, tools):
        self.n += 1
        names = {t["name"] for t in tools}
        reason = "Offline wiring check; the scripted driver does not solve the task."
        ans = "ANSWER: " + json.dumps({"insufficient_evidence": True, "reason": reason})
        if self.kind == "Base":
            if self.n == 1:
                return Step([{"type": "tool_use", "id": "t1", "name": "python",
                              "input": {"code": "import numpy as np; x=np.load('signal.npy'); print(x.shape)"}}], {}, "tool_use", "mock")  # fmt: skip
            return Step([{"type": "text", "text": "dry run. " + ans}], {"output_tokens": 5}, "end_turn", "mock")
        work = next(m for m in messages[0]["content"].splitlines() if m.startswith("The signal file is")).split()[4]
        if self.n == 1 and "ts_load" in names:
            return Step(
                [
                    {
                        "type": "tool_use",
                        "id": "a1",
                        "name": "ts_load",
                        "input": {
                            "source": work,
                            "fs": self.visible["channels"][0]["fs"],
                            "channels": [c["name"] for c in self.visible["channels"]],
                            "units": [c["unit"] for c in self.visible["channels"]],
                        },
                    }
                ],
                {},
                "tool_use",
                "mock",
            )
        if self.n == 2:
            sid = json.loads(messages[-1]["content"][0]["content"][0]["text"])["id"]
            return Step(
                [{"type": "tool_use", "id": "a2", "name": "ts_measure", "input": {"id": sid, "what": ["mean"]}}],
                {},
                "tool_use",
                "mock",
            )
        if self.n == 3 and "ts_submit" in names:
            return Step(
                [
                    {
                        "type": "tool_use",
                        "id": "a3",
                        "name": "ts_submit",
                        "input": {
                            "bundle": {
                                "claims": [],
                                "limitations": [
                                    {"quantity": f, "status": "insufficient_evidence", "reason": reason}
                                    for f in self.fields
                                ],
                            }
                        },
                    }
                ],
                {},
                "tool_use",
                "mock",
            )
        return Step([{"type": "text", "text": "dry run. " + ans}], {"output_tokens": 5}, "end_turn", "mock")


def price_cost(model_key: str, usage: dict) -> tuple[float | None, float]:
    p = MODELS[model_key].get("prices")
    ref = p or REFERENCE_PRICE  # verified peak rates bound this model; unknown providers use the reference rate

    def c(pr):
        return (usage.get("input_tokens", 0) * pr["in"] + usage.get("output_tokens", 0) * pr["out"]
                + usage.get("cache_read_input_tokens", 0) * pr["cache_read"]
                + usage.get("cache_creation_input_tokens", 0) * pr["cache_write"]) / 1e6  # fmt: skip

    return (c(p) if p else None), c(ref)


async def one_session(args, case: dict, run_dir: Path) -> dict:
    model_key = MODEL_OF[args.arm.split("-")[0]]
    kind = args.arm.split("-")[1]
    cfg = MODELS[model_key]
    base = Path(tempfile.mkdtemp(prefix=f"unified-{args.arm}-"))
    tools = None
    driver = None
    try:
        work = copy_case(Path(case["dir"]), base)
        rt = run_dir / "runtime" / case["case_id"]
        rt.mkdir(parents=True, exist_ok=True)
        source_spec = Path(case["dir"]) / "spec.json"
        spec = (
            TaskSpec.from_dict(json.loads(source_spec.read_text()))
            if source_spec.exists()
            else task_spec_for_case(case)
        )
        spec_path = rt / "spec.json"
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
        vision = bool(cfg.get("vision"))
        if kind == "Base":
            tools, system = PythonTools(work, vision, isolate=not args.no_isolate), SYSTEM_BASE
        else:
            tools = SignalMindTools(
                work,
                rt,
                PLUGINS,
                "verify" if kind == "Agent" else "log",
                spec_path,
                vision,
                f"{args.arm}-{case['case_id'][:8]}",
            )
            await tools.open()
            system = SYSTEM_AGENT + "\n" + tools.instructions
        driver = (
            MockDriver(kind, [q.name for q in spec.quantities], case["visible"])
            if args.dry_run
            else make_driver(model_key, max_tokens=args.max_response_tokens)
        )
        prompt = prompt_for(case, work, kind, spec)
        meta = {
            "arm": args.arm,
            "model": cfg["model"],
            "case_id": case["case_id"],
            "plugins": PLUGINS if kind != "Base" else [],
            "git_commit": args.git_commit,
            "max_response_tokens": args.max_response_tokens,
            "api": cfg["api"],
            "reasoning_effort": cfg.get("reasoning_effort", "provider_default"),
        }
        limits = Limits(
            max_turns=spec.budgets.max_steps,
            max_tool_calls=spec.budgets.max_tool_calls,
            max_seconds=spec.budgets.max_seconds,
        )
        prior_spend = spent(campaign_dirs(args.suite, args.tag)) if not args.dry_run else 0.0

        def budget_exhausted(usage):
            estimate, reference = price_cost(model_key, usage)
            return not args.dry_run and prior_spend + max(estimate or 0.0, reference) + args.reserve >= args.cap

        r = await run_session(
            driver,
            tools,
            system,
            prompt,
            limits,
            run_dir / "transcripts" / f"{case['case_id']}.jsonl",
            meta,
            budget_exhausted=budget_exhausted,
        )
        app, app_all, fin = application_answer(rt) if kind != "Base" else (None, None, "n/a")
        cost, cost_ref = price_cost(model_key, r["usage"])
        return {"case_id": case["case_id"], "arm": args.arm, "model": cfg["model"], "served_models": r["served_models"],
                "pred": parse_answer(r["text"]), "application_answer": app, "application_answer_all": app_all,
                "finalization": fin, "stop": r["stop"], "error": r["error"], "usage": r["usage"], "cost_usd": cost,
                "cost_usd_actual": None, "cost_usd_estimated": cost, "git_commit": args.git_commit,
                "max_response_tokens": args.max_response_tokens,
                "api": cfg["api"], "reasoning_effort": cfg.get("reasoning_effort", "provider_default"),
                "price_basis": cfg.get("price_basis", "configured_token_prices"),
                "cost_usd_conservative": round(cost_ref, 5), "latency_s": r["latency_s"], "turns": r["turns"],
                "tool_calls": r["tool_calls"], "text": r["text"][-3000:], "dry_run": args.dry_run}  # fmt: skip
    finally:
        try:
            if tools is not None:
                await tools.close()
        finally:
            close_driver = getattr(driver, "close", None)
            if close_driver is not None:
                close_driver()
            shutil.rmtree(base, ignore_errors=True)


def campaign_dirs(suite: str, tag: str) -> list[Path]:
    """One sequential campaign shares a budget across all its model/arm directories; dry runs are separate."""
    return [RUNS / f"{suite}-U-{m}-{k}-{tag}" for m in MODEL_OF for k in KINDS]


def result_rows(run_dirs: list[Path]):
    for d in run_dirs:
        f = d / "results.jsonl"
        if f.exists():
            yield from (json.loads(line) for line in f.read_text().splitlines() if line.strip())


def spent(run_dirs: list[Path]) -> float:
    total = 0.0
    for r in result_rows(run_dirs):
        total += max(
            r.get("cost_usd_actual") or 0,
            r.get("cost_usd_estimated", r.get("cost_usd")) or 0,
            r.get("cost_usd_conservative") or 0,
        )
    return total


def assert_pinned_checkout(root: Path, package_root: Path) -> str:
    """The runner and its in-process package must share a clean detached checkout."""
    if root.resolve() != package_root.resolve():
        raise SystemExit("runner and tsmcp must be installed from the same pinned checkout")
    if subprocess.run(["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 0:
        raise SystemExit("live runs require a detached checkout (git worktree add --detach)")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise SystemExit("live runs require a clean checkout")
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=[f"{m}-{k}" for m in MODEL_OF for k in KINDS])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--max-response-tokens",
        type=int,
        default=16384,
        help="per-response output allowance, identical for model-matched arms",
    )
    ap.add_argument(
        "--cap", type=float, default=0.0, help="estimated USD budget shared by this suite/tag; run arms sequentially"
    )
    ap.add_argument("--reserve", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-isolate", action="store_true")
    ap.add_argument("--allow-tools-arm", action="store_true", help="the Tools ablation needs explicit approval")
    args = ap.parse_args()
    if args.dry_run == args.live:
        raise SystemExit("choose exactly one of --dry-run or --live")
    if args.live and args.no_isolate:
        raise SystemExit("live Base runs require the Python sandbox")
    if args.live and args.arm.endswith("-Tools") and not args.allow_tools_arm:
        raise SystemExit(
            "the Tools ablation arm is prepared but not approved; pass --allow-tools-arm only after approval"
        )
    if args.live:
        import tsmcp

        args.git_commit = assert_pinned_checkout(ROOT, Path(tsmcp.__file__).resolve().parents[2])
        if args.cap <= 0:
            raise SystemExit("live runs need --cap")
    else:
        args.git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    cases = [c for c in load_suite(args.suite) if not args.only or c["case_id"] in args.only][: args.limit]
    run_dir = RUNS / f"{args.suite}-U-{args.arm}-{args.tag}{'-dry' if args.dry_run else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = list(result_rows([run_dir]))
    if args.live and any(
        r.get("git_commit") != args.git_commit for r in result_rows(campaign_dirs(args.suite, args.tag))
    ):
        raise SystemExit("this tag contains a different revision; use a new tag")
    if args.live and any(
        r.get("max_response_tokens") != args.max_response_tokens
        for r in result_rows(campaign_dirs(args.suite, args.tag))
    ):
        raise SystemExit("this tag contains a different response budget; use a new tag")
    done = {r["case_id"] for r in previous}
    completed = 0
    for c in cases:
        if c["case_id"] in done:
            continue
        if args.live and spent(campaign_dirs(args.suite, args.tag)) + args.reserve >= args.cap:
            print("budget stop", flush=True)
            break
        row = asyncio.run(one_session(args, c, run_dir))
        with open(run_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        completed += 1
        print(c["case_id"], args.arm, row["stop"], row["pred"], row.get("finalization"), f"turns={row['turns']}",
              f"tools={row['tool_calls']}", f"${row['cost_usd_conservative']:.4f}*", flush=True)  # fmt: skip
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "planned_sessions": len(cases),
                "completed_now": completed,
                "previous_sessions": len(done),
                "budget_estimate_usd": round(spent([run_dir]), 4),
            }
        )
    )


if __name__ == "__main__":
    main()
