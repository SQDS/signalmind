"""Single-case model runner for the public processing-pack contract, with isolated tools."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.processing_tools import ProcessingPythonTools, ProcessingSignalMindTools  # noqa: E402
from eval.unified.drivers import MODELS, make_driver  # noqa: E402
from eval.unified.loop import Limits, run_session  # noqa: E402

PUBLIC_FILES = (
    "task.json",
    "prompt.txt",
    "input.npz",
    "signal.csv",
    "preview.png",
    "OUTPUT_FORMAT.txt",
    "SCORING_RULES.txt",
    "submission.schema.json",
    "submission_helper.py",
)
SYSTEM = (
    "You are a signal-processing agent. Complete the current public task using observed evidence. "
    "Choose whether to preserve, repair, denoise or mark unavailable. Preserve original units and time axes. "
    "Deliver result.json and output.npz under /output using the published contract. After completing artifacts, "
    "finish with a concise explanation. The task has no private reference or grader available to you."
    " You have25 tool calls in total, including export/writing files. Reserve enough calls to deliver artifacts "
    "before this limit. Each tool result reports the remaining calls."
)
MODEL_OF = {"A": "gpt-6-astra", "DS": "deepseek-flash"}


def enable_response_streaming(driver):
    from eval.stream_final_transport import stream_final_post_with_reconnect

    driver.transport = MethodType(stream_final_post_with_reconnect, driver)


class BudgetFeedbackTools:
    """Expose the same remaining-call feedback to both processing tool configurations."""

    def __init__(self, tools, limit=25):
        self.tools, self.schemas, self.remaining = tools, tools.schemas, limit

    async def call(self, name, args):
        blocks, error = await self.tools.call(name, args)
        self.remaining -= 1
        message = f"Runner: {self.remaining} tool calls remaining, including any artifact export."
        if self.remaining <= 3:
            message += " Complete result.json and output.npz now if they are not already delivered."
        return [*blocks, {"type": "text", "text": message}], error

    async def close(self):
        await self.tools.close()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upper_cost(usage, model_key="gpt-6-astra"):
    p = MODELS[model_key]["prices"]
    return (
        sum(
            usage.get(field, 0) * p[key]
            for field, key in (
                ("input_tokens", "in"),
                ("output_tokens", "out"),
                ("cache_read_input_tokens", "cache_read"),
                ("cache_creation_input_tokens", "cache_write"),
            )
        )
        / 1e6
    )


def result_rows(out):
    return [json.loads(line) for f in out.glob("*/results.jsonl") for line in f.read_text().splitlines() if line]


async def run(args, commit):
    model_key = MODEL_OF[args.arm.split("-")[0]]
    config = MODELS[model_key]
    signalmind = args.arm.endswith("-Agent")
    source = args.pack / "public/cases" / args.case_id
    run_dir = args.out / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = run_dir / "transcripts" / f"{args.case_id}.jsonl"
    output = run_dir / "submissions" / args.case_id
    output.mkdir(parents=True, exist_ok=False)
    temporary = Path(tempfile.mkdtemp(prefix="processing-case-"))
    case = temporary / "public"
    case.mkdir()
    tools = driver = None
    hashes = {name: sha(source / name) for name in PUBLIC_FILES}
    try:
        for name in PUBLIC_FILES:
            shutil.copyfile(source / name, case / name)
            (case / name).chmod(0o444)
        task = json.loads((case / "task.json").read_text())
        assert hashes["input.npz"] == task["signal"]["sha256"]
        text = "\n\n".join(
            f"FILE /case/{name}\n{(case / name).read_text()}"
            for name in ("prompt.txt", "task.json", "SCORING_RULES.txt", "OUTPUT_FORMAT.txt")
        )
        text += (
            "\n\nOnly /case and /output are available. To use the shared helper in Python, insert '/case' into "
            "sys.path, load task.json, set task['_case_dir']='/case', and call submission_helper.write_submission. "
            "Its before/after measurements and hashes are provided equally to both tool configurations. "
            "SignalMind ts_export uses this same public helper. All processing choices must be your own."
        )
        prompt = [
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode((case / "preview.png").read_bytes()).decode(),
                },
            },
        ]
        tools = (
            await ProcessingSignalMindTools(case, output).open() if signalmind else ProcessingPythonTools(case, output)
        )
        tools = BudgetFeedbackTools(tools)
        driver = make_driver(model_key, max_tokens=16384)
        if args.stream:
            enable_response_streaming(driver)
        prior = sum(r["cost_upper_usd"] for r in result_rows(args.out))
        meta = {
            "suite": "processing-" + task["schema_version"].rsplit("processing.", 1)[-1],
            "arm": args.arm,
            "case_id": args.case_id,
            "model": config["model"],
            "api": config["api"],
            "reasoning_effort": config.get("reasoning_effort", "provider_default"),
            "source_commit": commit,
            "input_hashes": hashes,
            "prompt_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "tool_budget_feedback": True,
            "transport": "responses_sse_final" if args.stream else config["api"],
            "max_response_tokens": 16384,
            "processing_backend": "tsmcp.processing" if signalmind else None,
        }
        result = await run_session(
            driver,
            tools,
            SYSTEM,
            prompt,
            Limits(max_turns=30, max_tool_calls=25, max_seconds=900),
            transcript,
            meta,
            budget_exhausted=lambda usage: prior + upper_cost(usage, model_key) + args.reserve >= args.cap,
        )
        files = {name: sha(output / name) for name in ("result.json", "output.npz") if (output / name).exists()}
        row = {
            **meta,
            **result,
            "cost_upper_usd": upper_cost(result["usage"], model_key),
            "actual_billed_usd": None,
            "network_disconnects": getattr(driver, "_network_disconnects", 0),
            "submission_files": files,
            "output_dir": str(output),
            "transcript": str(transcript),
        }
        assert hashes == {name: sha(source / name) for name in PUBLIC_FILES}
        with (run_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("arm", "case_id", "stop", "error", "turns", "tool_calls", "latency_s", "cost_upper_usd")
                }
            ),
            flush=True,
        )
    finally:
        if tools is not None:
            await tools.close()
        if driver is not None:
            driver.close()
        shutil.rmtree(temporary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=["A-Base", "A-Agent", "DS-Agent"], required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--reserve", type=float, default=2.0)
    p.add_argument("--live", action="store_true", required=True)
    p.add_argument("--stream", action="store_true", help="Receive Astra Responses through the validated SSE transport")
    args = p.parse_args()
    if args.stream and not args.arm.startswith("A-"):
        p.error("--stream is only supported for Astra Responses")
    args.pack, args.out = args.pack.resolve(), args.out.resolve()
    assert args.case_id in json.loads((args.pack / "public/index.json").read_text())["case_ids"]
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "-C", str(ROOT), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 0:
        raise SystemExit("live runs require a clean detached snapshot")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise SystemExit("live runs require a clean snapshot")
    rows = result_rows(args.out)
    if any(r["arm"] == args.arm and r["case_id"] == args.case_id for r in rows):
        print("already completed")
        return
    if any(r["source_commit"] != commit for r in rows):
        raise SystemExit("output directory contains a different source version")
    if sum(r["cost_upper_usd"] for r in rows) + args.reserve >= args.cap:
        raise SystemExit("allocation reached before starting another case")
    asyncio.run(run(args, commit))


if __name__ == "__main__":
    main()
