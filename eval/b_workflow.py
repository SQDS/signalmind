"""Fixed arm-B evidence collection. This module never calls a model.

Only public case metadata and actual interface-v1 MCP replies drive the plan.
Run the worker with the frozen R checkout's Python, not the harness's Python.
Bundles (including PNGs) stay in the ignored cache; reviewable manifests contain
hashes, sizes and the complete omission ledger, but no signal samples or images.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_R = "1d344ae403616056d3bd7888ec097d2f8144f015"
SERVER_ROOT = Path.home() / "tsmcp-server-R"
CACHE = ROOT / "eval/cache/b-workflow"
# Higher numbers are dropped first; ties drop later entries first.
PRIORITIES = {"common": 0, "top3_onset": 10, "top3_gap": 20, "other_onset": 30, "other_gap": 40}


@dataclass(frozen=True)
class Budget:
    calls: int = 20
    # B1: the original envelope supplies remaining space, with a verification floor.
    text_chars: int = 32_000
    verification_floor_chars: int = 16_000
    images: int = 8

    def verification_limit(self, core_chars: int) -> int:
        return max(self.text_chars - core_chars, self.verification_floor_chars)


BUDGET = Budget()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication: an interrupted worker cannot leave a valid cache pointer.
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as out:
        out.write(canonical(value) + "\n")
        temporary = Path(out.name)
    temporary.replace(path)


def public_cases(cases_root: Path, suite: str) -> list[dict]:
    # Explicit development allowlist, checked BEFORE opening even the questions file.
    if suite not in ("p5afdb", "p7dev", "p6confirm"):
        raise ValueError("B collection is restricted to p5afdb, p7dev and the frozen p6confirm run")
    rows = [json.loads(line) for line in (cases_root / f"{suite}.questions.jsonl").read_text().splitlines() if line]
    cases = []
    for row in rows:
        directory = (cases_root / suite / row["case_id"]).resolve()
        if directory.parent != (cases_root / suite).resolve():
            raise ValueError("case id escapes suite")
        cases.append(
            {
                "case_id": row["case_id"],
                "qtype": row["qtype"],
                "question": row["question"],
                "dir": str(directory),
                "visible": json.loads((directory / "visible.json").read_text()),
            }
        )
    return cases


def signal_path(case: dict) -> Path:
    directory = Path(case["dir"]).resolve()
    source = (directory / case["visible"]["file"]).resolve()
    if not source.is_relative_to(directory) or source.suffix != ".npy":
        raise ValueError("signal must be a .npy inside this case")
    if case["qtype"] not in ("longest_af", "af_burden"):
        raise ValueError("unsupported B endpoint")
    return source


def verify_server(root: Path) -> dict:
    """Verify actual imported source bytes, including untracked Python additions."""
    import tsmcp

    root = root.expanduser().resolve()
    package = Path(tsmcp.__file__).resolve()
    if not package.is_relative_to(root / "src") or Path(sys.prefix).resolve() != root / ".venv":
        raise ValueError("B must run in the R venv and import its src/tsmcp")
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != FROZEN_R:
        raise ValueError(f"expected frozen R {FROZEN_R}, got {sha}")
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", sha, "src"], text=True
    ).splitlines()
    hashes = {}
    for name in names:
        expected = subprocess.check_output(["git", "-C", str(root), "show", f"{sha}:{name}"])
        if (root / name).read_bytes() != expected:
            raise ValueError(f"modified frozen source: {name}")
        hashes[name] = digest(expected)
    extras = {str(p.relative_to(root)) for p in (root / "src").rglob("*.py")} - set(names)
    if extras:
        raise ValueError(f"extra frozen source: {sorted(extras)}")
    return {
        "server_root": str(root),
        "server_git_sha": sha,
        "server_package_path": str(package),
        "server_source_sha256": digest(canonical(hashes).encode()),
        "interface": "v1",
    }


def request_identity(case: dict, provenance: dict) -> dict:
    source = signal_path(case)
    stat = source.stat()
    return {
        "case_id": case["case_id"],
        "qtype": case["qtype"],
        "question": case["question"],
        "visible": case["visible"],
        "signal_sha256": file_hash(source),
        "signal_path": str(source),
        # v1 handles depend on source path, size and mtime: all are part of the key.
        "signal_size": stat.st_size,
        "signal_mtime_ns": stat.st_mtime_ns,
        "workflow_sha256": file_hash(Path(__file__)),
        "prompt_source_sha256": file_hash(ROOT / "eval/common.py"),
        "provenance": provenance,
        "budget": asdict(BUDGET),
    }


def call_item(key: str, tool: str, arguments: dict, priority: int = 0, required: bool = False) -> dict:
    return {"key": key, "tool": tool, "arguments": arguments, "priority": priority, "required": required}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def item_content(item: dict) -> list[dict]:
    # Only provenance framing; no interpretation or new clinical instructions.
    head = canonical({"tool": item["tool"], "arguments": item["arguments"]}) + "\n"
    return [text_block(head), *item["content"]]


def evidence_content(items: list[dict], arithmetic: list[dict]) -> list[dict]:
    blocks = [block for item in items for block in item_content(item)]
    return blocks + [text_block(canonical(row)) for row in arithmetic]


def sizes(content: list[dict]) -> dict:
    return {
        "text_chars": sum(len(b["text"]) for b in content if b["type"] == "text"),
        "images": sum(b["type"] == "image" for b in content),
    }


def payload(item: dict) -> dict:
    if item.get("is_error"):
        raise ValueError(f"required tool failed: {item['key']}")
    return json.loads(next(b["text"] for b in item["content"] if b["type"] == "text"))


def measurement(item: dict, name: str) -> dict:
    rows = [r for r in payload(item)["measurements"] if r["name"] == name]
    if len(rows) != 1:
        raise ValueError("B currently requires one visible ECG channel")
    return rows[0]


def ranked_episodes(index: dict) -> list[dict]:
    return sorted(index["episodes"], key=lambda e: (-e["span_s"], e["t"]))


def candidates(index: dict) -> list[dict]:
    ranked = ranked_episodes(index)
    return [e for i, e in enumerate(ranked) if i < 5 or e["span_s"] >= ranked[0]["span_s"] / 2]


def internal_gaps(episodes: list[dict]) -> list[tuple[float, float]]:
    return sorted(
        {
            (a[1], b[0])
            for e in episodes
            for a, b in zip(e["intervals_s"], e["intervals_s"][1:], strict=False)
            if b[0] > a[1]
        },
        key=lambda w: (-(w[1] - w[0]), w[0], w[1]),
    )


def window(start: float, end: float, duration: float) -> list[float] | None:
    lo, hi = max(0.0, start), min(duration, end)
    return [lo, hi] if hi > lo else None


def longest_plan(sid: str, index: dict, duration: float) -> tuple[list[dict], list[dict]]:
    selected = candidates(index)
    plan, skipped = [], []
    for i, ep in enumerate(selected):
        t = ep["t"]
        priority = PRIORITIES["top3_onset" if i < 3 else "other_onset"]
        for suffix, tool, span, extra in (
            ("strip", "ts_zoom", (t - 30, t + 30), {"view": "strip", "rows": 6}),
            ("before", "ts_measure", (t - 120, t), {"what": ["ecg_rr_stats"]}),
            ("after", "ts_measure", (t, t + 120), {"what": ["ecg_rr_stats"]}),
        ):
            win = window(*span, duration)
            item = call_item(f"onset.{i + 1}.{suffix}", tool, {"id": sid, "window": win, **extra}, priority)
            if win:
                plan.append(item)
            else:
                skipped.append({**item, "reason": "empty_window_at_record_boundary", "executed": False})
    top_gaps = set(internal_gaps(selected[:3]))
    chronological = sorted(selected, key=lambda e: e["t"])
    between = {
        (a["t"] + a["span_s"], b["t"])
        for a, b in zip(chronological, chronological[1:], strict=False)
        if 0 < b["t"] - (a["t"] + a["span_s"]) < 300
    }
    # Between top-three candidates also outranks lower-candidate onset inspection.
    top_starts = {e["t"] for e in selected[:3]}
    top_gaps |= {
        gap
        for gap in between
        if any(e["t"] in top_starts and e["t"] + e["span_s"] == gap[0] for e in selected) and gap[1] in top_starts
    }
    gaps = sorted(top_gaps | between, key=lambda w: (w not in top_gaps, -(w[1] - w[0]), w[0], w[1]))
    for i, gap in enumerate(gaps):
        priority = PRIORITIES["top3_gap" if gap in top_gaps else "other_gap"]
        for suffix, tool, extra in (
            ("strip", "ts_zoom", {"view": "strip", "rows": 6}),
            ("rr", "ts_measure", {"what": ["ecg_rr_stats"]}),
        ):
            plan.append(call_item(f"gap.{i + 1}.{suffix}", tool, {"id": sid, "window": list(gap), **extra}, priority))
    return sorted(
        plan,
        key=lambda item: (item["priority"], item["key"].startswith("onset.") and not item["key"].endswith(".strip")),
    ), skipped


def burden_arithmetic(index: dict, burden: dict, index_ref: str, burden_ref: str) -> list[dict]:
    duration = burden["window_s"]
    rows = []
    # Integrate support, NEVER grouped spans. The original index remains intact.
    for start in range(0, int(duration) + 1, 3600):
        end = min(start + 3600, duration)
        if end <= start:
            continue
        seconds = sum(max(0.0, min(b, end) - max(a, start)) for ep in index["episodes"] for a, b in ep["intervals_s"])
        rows.append({"window": [start, end], "af_s": round(seconds, 6)})
    return [
        {"arithmetic": "hourly_af_s_from_intervals_s", "source_ref": index_ref, "hours": rows},
        {
            "arithmetic": "numeric_consistency",
            "source_ref": burden_ref,
            **{key: burden[key] for key in ("pct_of_window", "af_s", "window_s", "readable_s")},
        },
    ]


def burden_plan(sid: str, index: dict, noisy: dict, duration: float) -> list[dict]:
    plan = []
    # ecg_noisy's score equals duration; top_k=3 selects the largest unknown stretches.
    stretches = sorted(noisy["events"], key=lambda e: (-e["dur"], e["t"]))
    groups = [
        ("unknown", [(e["t"], e["t"] + e["dur"]) for e in stretches[:3]]),
        ("internal_gap", internal_gaps(candidates(index))[:3]),
    ]
    for priority, (kind, spans) in enumerate(groups, 10):
        for i, span in enumerate(spans):
            win = window(*span, duration)
            if win:
                plan.append(
                    call_item(
                        f"{kind}.{i + 1}.strip",
                        "ts_zoom",
                        {"id": sid, "window": win, "view": "strip", "rows": 6},
                        priority,
                    )
                )
    return plan


class Collector:
    def __init__(self, client, budget: Budget):
        self.client, self.budget = client, budget
        self.items: list[dict] = []
        self.dropped: list[dict] = []
        self.calls = 0

    async def call(self, item: dict) -> dict:
        self.calls += 1
        result = await self.client.call_tool(item["tool"], item["arguments"])
        content = []
        for block in result.content:
            if block.type == "text":
                content.append(text_block(block.text))  # no JSON parse/re-encode
            elif block.type == "image":
                content.append(
                    {"type": "image", "source": {"type": "base64", "media_type": block.mime_type, "data": block.data}}
                )
            else:
                raise ValueError(f"unexpected MCP content: {block.type}")
        done = {**item, "content": content, "is_error": bool(result.is_error), "call_number": self.calls}
        self.items.append(done)
        return done

    def drop(self, item: dict, reason: str, executed: bool) -> None:
        entry = {k: v for k, v in item.items() if k != "content"}
        if executed:
            entry["response_sha256"] = digest(canonical(item["content"]).encode())
            entry["response_sizes"] = sizes(item["content"])
        self.dropped.append({**entry, "reason": reason, "executed": executed})

    async def optional(self, plan: list[dict], arithmetic: list[dict]) -> None:
        for item in plan:
            if self.calls >= self.budget.calls:
                self.drop(item, "call_budget", False)
            elif (
                item["tool"] == "ts_zoom"
                and sizes(evidence_content(self.items, arithmetic))["images"] >= self.budget.images
            ):
                self.drop(item, "image_budget", False)
            else:
                await self.call(item)

    def trim_verification(self, arithmetic: list[dict], limit: int) -> None:
        # Core replies are never eligible. Arithmetic is verification too, and keeps
        # its existing highest priority; later arithmetic blocks lose ties first.
        while sizes(evidence_content([i for i in self.items if not i["required"]], arithmetic))["text_chars"] > limit:
            eligible = [item for item in self.items if not item["required"]]
            candidates = [(i["priority"], i["call_number"], i) for i in eligible]
            candidates += [(PRIORITIES["common"], self.calls + n + 1, row) for n, row in enumerate(arithmetic)]
            _, _, victim = max(candidates, key=lambda entry: entry[:2])
            if "arithmetic" in victim:
                arithmetic.remove(victim)
                self.dropped.append(
                    {
                        "key": victim["arithmetic"],
                        "priority": PRIORITIES["common"],
                        "reason": "text_budget",
                        "executed": False,
                        "kind": "arithmetic",
                        "response_sha256": digest(canonical(victim).encode()),
                        "response_sizes": sizes([text_block(canonical(victim))]),
                    }
                )
            else:
                self.items.remove(victim)
                self.drop(victim, "text_budget", True)


async def collect(case: dict, client, budget: Budget = BUDGET) -> dict:
    c = Collector(client, budget)
    source = signal_path(case)
    channels = case["visible"]["channels"]
    if len(channels) != 1:
        raise ValueError("B currently requires one visible ECG channel")
    loaded = await c.call(call_item("load", "ts_load", {"source": str(source), "fs": channels[0]["fs"]}, required=True))
    info = payload(loaded)
    sid, duration = info["id"], info["dur_s"]
    await c.call(call_item("describe", "ts_describe", {"id": sid}, required=True))
    burden_item = await c.call(call_item("burden", "ts_measure", {"id": sid, "what": ["ecg_af_burden"]}, required=True))
    index_item = await c.call(
        call_item("episode_index", "ts_measure", {"id": sid, "what": ["ecg_af_episodes"]}, required=True)
    )
    burden, index = measurement(burden_item, "ecg_af_burden"), measurement(index_item, "ecg_af_episodes")
    arithmetic = burden_arithmetic(index, burden, index["ref"], burden["ref"]) if case["qtype"] == "af_burden" else []
    await c.call(call_item("whole_rr", "ts_zoom", {"id": sid, "window": [0, duration], "view": "ecg_rr"}, priority=5))
    if case["qtype"] == "longest_af":
        plan, skipped = longest_plan(sid, index, duration)
        c.dropped.extend(skipped)
    else:
        noisy_item = await c.call(
            call_item("unknown_index", "ts_find_events", {"id": sid, "types": ["ecg_noisy"], "top_k": 3}, priority=5)
        )
        plan = burden_plan(sid, index, payload(noisy_item), duration)
    core_size = sizes(evidence_content([item for item in c.items if item["required"]], []))["text_chars"]
    verification_limit = budget.verification_limit(core_size)
    await c.optional(plan, arithmetic)
    c.trim_verification(arithmetic, verification_limit)
    content = evidence_content(c.items, arithmetic)
    metrics = {
        **sizes(content),
        "core_text_chars": core_size,
        "verification_text_chars": sizes(evidence_content([i for i in c.items if not i["required"]], arithmetic))[
            "text_chars"
        ],
        "tool_text_chars": sum(sizes(i["content"])["text_chars"] for i in c.items),
        "calls": c.calls,
        "retained_calls": len(c.items),
        "dropped": len(c.dropped),
    }
    valid = (
        metrics["verification_text_chars"] <= verification_limit
        and metrics["images"] <= budget.images
        and c.calls <= budget.calls
    )
    return {
        "schema": "b-workflow-b1",
        "case_id": case["case_id"],
        "qtype": case["qtype"],
        "status": "ready" if valid else "budget_exceeded",
        "budget": asdict(budget),
        "verification_text_budget": verification_limit,
        "sizes": metrics,
        "items": c.items,
        "arithmetic": arithmetic,
        "content": content,
        "dropped": c.dropped,
    }


def manifest(bundle: dict, bundle_hash: str, identity: dict, key: str) -> dict:
    return {
        "case_id": bundle["case_id"],
        "qtype": bundle["qtype"],
        "status": bundle["status"],
        "bundle_sha256": bundle_hash,
        "request_sha256": key,
        "identity": identity,
        "budget": bundle["budget"],
        "sizes": bundle["sizes"],
        "verification_text_budget": bundle["verification_text_budget"],
        "items": [
            {
                **{k: v for k, v in i.items() if k != "content"},
                "sizes": sizes(i["content"]),
                "response_sha256": digest(canonical(i["content"]).encode()),
            }
            for i in bundle["items"]
        ],
        "dropped": bundle["dropped"],
    }


def load_cached(cache: Path, key: str) -> tuple[dict, dict] | None:
    pointer = cache / "requests" / f"{key}.json"
    if not pointer.exists():
        return None
    meta = json.loads(pointer.read_text())
    raw = (cache / "objects" / f"{meta['bundle_sha256']}.json").read_bytes()
    if digest(raw) != meta["bundle_sha256"] or meta["request_sha256"] != key:
        raise ValueError("bundle cache integrity mismatch")
    bundle = json.loads(raw)
    if manifest(bundle, meta["bundle_sha256"], meta["identity"], key) != meta:
        raise ValueError("bundle manifest integrity mismatch")
    return bundle, meta


def build(case: dict, server_root: Path = SERVER_ROOT, cache: Path = CACHE, rebuild: bool = False) -> tuple[dict, dict]:
    provenance = verify_server(server_root)
    identity = request_identity(case, provenance)
    key = digest(canonical(identity).encode())
    if not rebuild and (hit := load_cached(cache, key)):
        return hit

    async def session():
        from mcp import Client

        from tsmcp.core.store import SignalStore
        from tsmcp.server.app import build_server

        # R predates the interface switch: its only/default interface IS v1.
        with tempfile.TemporaryDirectory(prefix="b-handles-") as handles:
            previous = os.environ.get("TSMCP_CACHE_DIR")
            os.environ["TSMCP_CACHE_DIR"] = handles
            try:
                store = SignalStore(allow_roots=[Path(case["dir"])])
                async with Client(build_server(store, images=True)) as client:
                    return await collect(case, client)
            finally:
                if previous is None:
                    os.environ.pop("TSMCP_CACHE_DIR", None)
                else:
                    os.environ["TSMCP_CACHE_DIR"] = previous

    bundle = asyncio.run(session())
    if provenance != verify_server(server_root) or identity != request_identity(case, provenance):
        raise ValueError("source or public case changed during bundle construction")
    raw = (canonical(bundle) + "\n").encode()
    bundle_hash = digest(raw)
    meta = manifest(bundle, bundle_hash, identity, key)
    old = load_cached(cache, key)
    if old and old[1]["bundle_sha256"] != bundle_hash:
        raise ValueError("same request produced a different bundle")
    write_json(cache / "objects" / f"{bundle_hash}.json", bundle)
    write_json(cache / "requests" / f"{key}.json", meta)
    return bundle, meta


def build_in_worker(case: dict, server_root: Path = SERVER_ROOT, cache: Path = CACHE) -> tuple[dict, dict]:
    """Harness entry: never import its local tsmcp as a substitute for frozen R."""
    command = [
        str(server_root / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--worker",
        "--server-root",
        str(server_root),
        "--cache",
        str(cache),
    ]
    # Preserve sandbox sitecustomize when supplied; verify_server rejects wrong package imports.
    result = subprocess.run(command, input=canonical(case), capture_output=True, text=True, check=True)
    reply = json.loads(result.stdout)
    hit = load_cached(cache, reply["request_sha256"])
    if hit is None:
        raise ValueError("worker did not publish a bundle")
    return hit


def user_message(prompt: str, bundle: dict) -> str:
    if bundle["status"] != "ready":
        raise ValueError("verification evidence exceeds B budget; no model message can be sent")
    return (
        canonical({"type": "user", "message": {"role": "user", "content": [text_block(prompt), *bundle["content"]]}})
        + "\n"
    )


def summarize_images(stdin: str) -> str:
    value = json.loads(stdin)
    for block in value["message"]["content"]:
        if block["type"] == "image":
            source = block["source"]
            data = source.pop("data")
            source["data_summary"] = {"base64_chars": len(data), "base64_sha256": digest(data.encode())}
    return canonical(value) + "\n"


def save_table(output: Path, manifests: list[dict]) -> None:
    text = "| Case | Endpoint | Core chars (uncapped) | Verification chars / cap | Total chars | Images / 8 | Calls / 20 | Dropped | Status |\n"
    text += "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    for m in manifests:
        s = m["sizes"]
        text += f"| {m['case_id']} | {m['qtype']} | {s['core_text_chars']} | {s['verification_text_chars']} / {m['verification_text_budget']} | {s['text_chars']} | {s['images']} | {s['calls']} | {s['dropped']} | {m['status']} |\n"
    (output / "size-table.md").write_text(text)
    write_json(
        output / "sizes.json",
        [
            {
                k: m[k]
                for k in ("case_id", "qtype", "status", "bundle_sha256", "sizes", "budget", "verification_text_budget")
            }
            for m in manifests
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--server-root", type=Path, default=SERVER_ROOT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--cases-root", type=Path, default=ROOT / "eval/cases")
    parser.add_argument("--suite", choices=("p5afdb", "p7dev"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/results/b-workflow")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.cache / "matplotlib"))
    if args.worker:
        _, meta = build(json.load(sys.stdin), args.server_root, args.cache, args.rebuild)
        print(canonical({"request_sha256": meta["request_sha256"]}))
        return
    if not args.suite:
        parser.error("--suite is required")
    output = args.output / args.suite
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for case in public_cases(args.cases_root, args.suite):
        if args.only and case["case_id"] not in args.only:
            continue
        _, meta = build(case, args.server_root, args.cache, args.rebuild)
        write_json(output / "manifests" / f"{case['case_id']}.json", meta)
        manifests.append(meta)
        save_table(output, manifests)
        print(canonical({"case_id": case["case_id"], "status": meta["status"], **meta["sizes"]}), flush=True)


if __name__ == "__main__":
    main()
