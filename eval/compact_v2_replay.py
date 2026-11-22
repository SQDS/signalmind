"""Replay all saved p5afdb M/R tool calls against the same R detector, v1/v2.

No model is called. The real MCP server request handler validates arguments,
serializes tool results, catches errors, and renders images. The transport alone
is bypassed. Transcript order, arguments, failed calls and distinct subsequent
assistant request IDs are retained; only sandbox file paths and handles returned
by an earlier successful ts_load are remapped. Counts are Unicode characters,
not tokens; base64 images are excluded and image counts are reported separately.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mcp_types import CallToolRequestParams

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from eval.compact_v2_equivalence import frozen_source_hashes  # noqa: E402
from eval.token_audit import MODELS, payload_details, text_and_images  # noqa: E402
from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.server.app import build_server  # noqa: E402

INTERFACES = ("v1", "v2")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_digest(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def result_stats(texts: list[str], images: int, error: bool, following: int) -> dict:
    chars = sum(map(len, texts))
    return {
        "chars": chars,
        "utf8_bytes": sum(len(text.encode()) for text in texts),
        "images": images,
        "is_error": error,
        "history_exposure_chars": chars * following,
        "resend_chars": chars * max(0, following - 1),
        "text_sha256": text_digest(texts),
        "error_text": "\n".join(texts) if error else None,
    }


def returned_handle(texts: list[str]) -> str | None:
    for text in texts:
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            return obj["id"]
    return None


def parse_transcript(path: Path) -> dict:
    """Stream large image-bearing JSONL; never retain image bytes or SDK envelopes."""
    requests, calls = {}, {}
    completed = False
    with path.open() as stream:
        for step, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "assistant":
                message = event["message"]
                requests.setdefault(message["id"], step)
                for block in message.get("content", []):
                    if block.get("type") != "tool_use" or block["id"] in calls:
                        continue
                    calls[block["id"]] = {
                        "tool_use_id": block["id"],
                        "request_id": message["id"],
                        "call_step": step,
                        "tool": block["name"].removeprefix("mcp__tsmcp__"),
                        "arguments": block["input"],
                    }
            if event.get("type") == "user":
                content = event.get("message", {}).get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    call = calls[block["tool_use_id"]]
                    if "result_step" in call:
                        raise ValueError(f"duplicate tool result at {path}:{step}")
                    texts, images = text_and_images(block.get("content"))
                    call["result_step"] = step
                    call["original"] = result_stats(texts, images, bool(block.get("is_error")), 0)
                    call["original_returned_handle"] = returned_handle(texts) if call["tool"] == "ts_load" else None
            if event.get("type") == "result":
                completed = True
    if not completed:
        raise ValueError(f"incomplete transcript: {path}")
    for call in calls.values():
        if "result_step" not in call:
            raise ValueError(f"missing tool result: {path}: {call['tool_use_id']}")
        following = sum(step > call["result_step"] for step in requests.values())
        call["following_requests"] = following
        old = call["original"]
        old["history_exposure_chars"] = old["chars"] * following
        old["resend_chars"] = old["chars"] * max(0, following - 1)
    return {
        "transcript": str(path),
        "transcript_sha256": digest(path),
        "requests": len(requests),
        "request_first_steps": requests,
        "calls": list(calls.values()),
    }


def remap_arguments(arguments: dict, signal_path: Path, handles: dict[str, str]) -> dict:
    """Do not fix bad handles, unknown kwargs, windows, channels, or list order."""
    result = dict(arguments)
    if isinstance(result.get("source"), str):
        source = Path(result["source"])
        if (
            source.name == "signal.npy"
            and source.parent.name == "work"
            and source.parent.parent.name.startswith("tsmcp-C-")
        ):
            result["source"] = str(signal_path)
        else:
            raise ValueError(f"unrecognized recorded source; cannot safely remap: {source}")
    if isinstance(result.get("id"), str) and result["id"] in handles:
        result["id"] = handles[result["id"]]
    return result


class ReplayStore(SignalStore):
    """Separate stores/recipes retain forget semantics; only immutable Channels share.

    A pool lives for one case. SignalStore performs normal validation/loading;
    pooling channels with identical source, fs, name and unit only reuses the
    frozen plugin's weak-key beat cache and does not cache any tool response.
    """

    def __init__(self, root: Path, channels: dict):
        super().__init__(allow_roots=[root])
        self.shared_channels = channels

    def load(self, **kwargs):
        signal = super().load(**kwargs)
        if kwargs.get("source") is not None:
            source = str(Path(kwargs["source"]).resolve())
            signal.channels = [
                self.shared_channels.setdefault((source, ch.fs, ch.name, ch.unit), ch) for ch in signal.channels
            ]
        return signal


async def dispatch(server, name: str, arguments: dict):
    # This is the SDK's transport request handler, including its ToolError path.
    return await server._handle_call_tool(None, CallToolRequestParams(name=name, arguments=arguments))


async def replay_transcript(parsed: dict, signal: Path, channel_pool: dict, cache_root: Path) -> dict:
    servers, handles = {}, {}
    for version in INTERFACES:
        servers[version] = build_server(ReplayStore(signal.parent, channel_pool), interface=version)
        handles[version] = {}
    for call in parsed["calls"]:
        for version in INTERFACES:
            os.environ["TSMCP_CACHE_DIR"] = str(cache_root / version)
            arguments = remap_arguments(call["arguments"], signal, handles[version])
            response = await dispatch(servers[version], call["tool"], arguments)
            texts = [block.text for block in response.content if block.type == "text"]
            images = sum(block.type == "image" for block in response.content)
            call[version] = result_stats(texts, images, bool(response.is_error), call["following_requests"])
            call[version]["arguments"] = arguments
            call[version]["measurement_chars"] = payload_details(texts)["measurement_chars"]
            if call["tool"] == "ts_load" and not response.is_error:
                old, new = call["original_returned_handle"], returned_handle(texts)
                if old is not None and new is not None:
                    handles[version][old] = new
            normalized = texts
            for old, new in handles[version].items():
                normalized = [text.replace(new, old) for text in normalized]
            call[version]["matches_original_text_after_handle_remap"] = (
                text_digest(normalized) == call["original"]["text_sha256"]
            )
            del response
        if call["v1"]["is_error"] != call["v2"]["is_error"]:
            raise AssertionError(f"v1/v2 success mismatch: {call['tool_use_id']}")
        if call["v1"]["images"] != call["v2"]["images"]:
            raise AssertionError(f"v1/v2 image count mismatch: {call['tool_use_id']}")
    parsed["handle_remaps"] = handles
    return parsed


def summary(calls: list[dict]) -> dict:
    metrics = ("chars", "utf8_bytes", "images", "history_exposure_chars", "resend_chars")
    result = {"calls": len(calls)}
    for version in ("original", *INTERFACES):
        result[version] = {key: sum(call[version][key] for call in calls) for key in metrics}
        result[version]["errors"] = sum(call[version]["is_error"] for call in calls)
        result[version]["mean_chars"] = result[version]["chars"] / len(calls) if calls else 0
    result["v1_to_v2_reduction_pct"] = {
        key: 100 * (1 - result["v2"][key] / result["v1"][key]) if result["v1"][key] else 0
        for key in ("chars", "history_exposure_chars", "resend_chars")
    }
    result["error_status_mismatches"] = sum(c["v1"]["is_error"] != c["v2"]["is_error"] for c in calls)
    result["image_count_mismatches"] = sum(c["v1"]["images"] != c["v2"]["images"] for c in calls)
    result["original_to_R_v1_error_status_changes"] = sum(
        c["original"]["is_error"] != c["v1"]["is_error"] for c in calls
    )
    result["original_to_R_v1_text_changes_after_handle_remap"] = sum(
        not c["v1"]["matches_original_text_after_handle_remap"] for c in calls
    )
    return result


def summarize_cases(cases: list[dict]) -> dict:
    calls = [call for case in cases for call in case["calls"]]
    return {
        "transcripts": len(cases),
        "cases": len({case["case_id"] for case in cases}),
        "requests": sum(case["requests"] for case in cases),
        "all_tools": summary(calls),
        "per_tool": {
            name: summary([c for c in calls if c["tool"] == name]) for name in sorted({c["tool"] for c in calls})
        },
    }


async def replay_case(paths: list[Path], signal: Path, temporary: Path) -> list[dict]:
    """One case per process; keep its immutable channel cache only across four runs."""
    channels, cases = {}, []
    old_cache = os.environ.get("TSMCP_CACHE_DIR")
    try:
        for path in paths:
            parsed = parse_transcript(path)
            parsed.update(
                case_id=signal.parent.name,
                run=path.parent.parent.name,
                source_transcript_arm=path.parent.parent.name[-1],
            )
            target = temporary / path.parent.parent.name / signal.parent.name
            case = await replay_transcript(parsed, signal, channels, target)
            case["summary"] = summarize_cases([case])
            cases.append(case)
            print(f"{path.parent.parent.name}/{signal.parent.name}: {len(case['calls'])} calls", flush=True)
    finally:
        if old_cache is None:
            os.environ.pop("TSMCP_CACHE_DIR", None)
        else:
            os.environ["TSMCP_CACHE_DIR"] = old_cache
        del channels
        gc.collect()
    return cases


def replay_case_job(job: tuple[list[Path], Path, Path]) -> list[dict]:
    """Spawn entrypoint: each worker has independent global state and plotting lock."""
    return asyncio.run(replay_case(*job))


async def replay(
    runs_root: Path, cases_root: Path, output: Path, selected: list[str] | None = None, workers: int = 1
) -> dict:
    started = time.monotonic()
    frozen_hashes = frozen_source_hashes()
    source_hashes = {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO / "src/tsmcp").rglob("*.py"))}
    script_hash = digest(Path(__file__))
    run_paths = [runs_root / f"p5afdb-C-{model}-{arm}" for model in MODELS for arm in ("M", "R")]
    transcripts = defaultdict(list)
    input_runs = []
    for run in run_paths:
        results_path = run / "results.jsonl"
        rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        expected = {row["case_id"] for row in rows}
        paths = sorted((run / "transcripts").glob("*.jsonl"))
        if len(rows) != 28 or len(expected) != 28 or {p.stem for p in paths} != expected:
            raise ValueError(f"expected all 28 unique saved cases and transcripts: {run}")
        if selected is not None and not set(selected) <= expected:
            raise ValueError(f"unknown --case selection for {run}: {selected}")
        input_runs.append({"path": str(results_path), "sha256": digest(results_path), "cases": sorted(expected)})
        for path in paths:
            if selected is None or path.stem in selected:
                transcripts[path.stem].append(path)
    if not transcripts:
        raise ValueError("no transcripts found")
    if workers < 1:
        raise ValueError("workers must be positive")
    cases, signals = [], []
    with tempfile.TemporaryDirectory(prefix="compact-v2-replay-") as temporary:
        jobs = []
        for case_id, paths in sorted(transcripts.items()):
            signal = (cases_root / case_id / "signal.npy").resolve()
            visible = signal.parent / "visible.json"
            signals.append(
                {
                    "case_id": case_id,
                    "signal": str(signal),
                    "signal_sha256": digest(signal),
                    "visible_sha256": digest(visible),
                }
            )
            jobs.append((paths, signal, Path(temporary)))
        if workers == 1:
            for job in jobs:
                cases.extend(await replay_case(*job))
        else:
            # Spawn (never fork a running asyncio loop); map retains sorted case order.
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for group in pool.map(replay_case_job, jobs):
                    cases.extend(group)
    finished_hashes = {str(p.relative_to(REPO)): digest(p) for p in sorted((REPO / "src/tsmcp").rglob("*.py"))}
    if finished_hashes != source_hashes or digest(Path(__file__)) != script_hash:
        raise ValueError("source or replay script changed during execution; rerun from a stable tree")
    report = {
        "schema_version": 1,
        "workers": workers,
        "kind": "deterministic tool replay, not a model run",
        "detector": "baseline-R-2026-10-12 in both interface arms, including source-M transcripts",
        "method": {
            "dispatch": "MCPServer._handle_call_tool(None, CallToolRequestParams): actual SDK validation, handlers, serialization, errors and PNG rendering; transport bypassed",
            "ordering": "original first-seen tool-use order per transcript, v1 then v2 for each call; no inserted detail calls",
            "arguments": "unchanged except recognized sandbox signal.npy source and earlier successfully returned handle identities; failed calls included",
            "isolation": "fresh server/store per transcript and interface, separate recipe dirs; immutable identical Channels pooled per case to reuse frozen weak-key beat cache",
            "history_exposure": "sum Unicode response characters times distinct assistant request IDs first appearing after recorded result; same audit proxy, not actual tokens",
            "excluded": "base64 images, duplicate SDK envelopes, schemas, hidden wrappers, assistant reasoning and cache billing; image counts separate",
            "accuracy": "fixed recorded policy cannot measure model accuracy or extra on-demand detail use",
        },
        "inputs": {"runs": input_runs, "signals": signals},
        "source_hashes": source_hashes,
        "frozen_detector_sha256": frozen_hashes,
        "source_unchanged_during_run": True,
        "script_sha256": script_hash,
        "elapsed_s": round(time.monotonic() - started, 3),
        "summary": summarize_cases(cases),
        "per_run": {run.name: summarize_cases([c for c in cases if c["run"] == run.name]) for run in run_paths},
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=REPO / "eval/runs")
    parser.add_argument("--cases-root", type=Path, default=REPO / "eval/cases/p5afdb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="optional smoke subset; omit for all 112 transcripts")
    parser.add_argument("--workers", type=int, default=1, help="bounded independent case processes (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = asyncio.run(
        replay(args.runs_root.resolve(), args.cases_root.resolve(), args.output, args.case, args.workers)
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
