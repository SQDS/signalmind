"""Selectable deterministic B revisions; the original b_workflow stays frozen.

Only public metadata and tool replies drive collection. No model is invoked.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import b_revision_burden as r2
from eval import b_workflow as bw

VERSIONS = ("v0", "r1", "r2", "final")


def effective_workflow(workflow: str, qtype: str) -> str:
    if workflow not in VERSIONS or qtype not in ("longest_af", "af_burden"):
        raise ValueError("unsupported B workflow or endpoint")
    if workflow == "final":
        return "r1" if qtype == "longest_af" else "r2"
    if (workflow, qtype) in (("r1", "af_burden"), ("r2", "longest_af")):
        return "v0"
    return workflow


def workflow_hash(workflow: str) -> str:
    if workflow not in VERSIONS:
        raise ValueError("unsupported B workflow")
    if workflow == "v0":
        return bw.file_hash(Path(bw.__file__))
    return bw.digest(
        bw.canonical(
            {Path(m.__file__).name: bw.file_hash(Path(m.__file__)) for m in (bw, r2, sys.modules[__name__])}
        ).encode()
    )


def sample_window(start: float | Fraction, end: float | Fraction, fs: float, n_samples: int) -> list[float] | None:
    """Clip outward to sample bounds; inputs are seconds."""
    rate = Fraction(str(fs))
    lo = max(0, min(n_samples, math.floor(Fraction(str(start)) * rate)))
    hi = max(0, min(n_samples, math.ceil(Fraction(str(end)) * rate)))
    return [lo / fs, hi / fs] if hi > lo else None


def profile_bins(item: dict, fs: float) -> list[dict]:
    """Exact fractional-sample edges, never rounded response t0/dt_s."""
    lo, hi = (round(t * fs) for t in item["arguments"]["window"])
    failed = [{"start": Fraction(lo), "end": Fraction(hi), "qualifies": False}]
    if item.get("is_error"):
        return failed
    row = bw.measurement(item, "ecg_rr_profile")
    counts = row.get("n_rr")
    if not isinstance(counts, list) or not counts:
        return failed
    cvs, alts = row.get("rr_cv", []), row.get("alt", [])
    bins = []
    for i, n in enumerate(counts):
        cv = cvs[i] if isinstance(cvs, list) and i < len(cvs) else None
        alt = alts[i] if isinstance(alts, list) and i < len(alts) else None
        valid = all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (n, cv, alt))
        bins.append(
            {
                "start": Fraction(lo) + Fraction((hi - lo) * i, len(counts)),
                "end": Fraction(lo) + Fraction((hi - lo) * (i + 1), len(counts)),
                "qualifies": bool(valid and n >= 8 and cv >= 0.12 and alt <= 0.75),
            }
        )
    return bins


def search_leads(bins: list[dict]) -> list[dict]:
    leads = []
    active = None
    for row in bins:
        if not row["qualifies"]:
            active = None
            continue
        if active is not None and active["end"] == row["start"]:
            active["end"] = row["end"]
        else:
            active = {"start": row["start"], "end": row["end"]}
            leads.append(active)
    return sorted(leads, key=lambda r: (-(r["end"] - r["start"]), r["start"]))


def choose_seeds(index: dict, leads: list[dict], fs: float, n_samples: int) -> list[dict]:
    ranked = bw.ranked_episodes(index)
    seeds = []

    def add(sample, source, rank, separation=0):
        sample = max(0, min(n_samples - 1, math.floor(sample)))
        if all(abs(sample - s["sample"]) > separation for s in seeds):
            seeds.append({"sample": sample, "source": source, "rank": rank})

    if ranked:
        add(ranked[0]["t"] * fs, "r_rank", 1)
    for rank, lead in enumerate(leads, 1):
        if len(seeds) == 3:
            break
        add(lead["start"], "search_lead", rank, separation=60 * fs)
    for rank, ep in enumerate(ranked[1:], 2):
        if len(seeds) == 3:
            break
        add(ep["t"] * fs, "r_rank", rank)
    return seeds


def refined_start(bins: list[dict], seed_sample: int) -> int:
    for a, b in zip(bins, bins[1:], strict=False):
        if a["qualifies"] and b["qualifies"] and a["end"] == b["start"]:
            return math.floor(a["start"])
    return seed_sample


class Collector(bw.Collector):
    async def request(self, item: dict, reserve_calls: int = 0) -> dict | None:
        if item["arguments"].get("window", True) is None:
            self.drop(item, "empty_window_at_record_boundary", False)
        elif self.calls + reserve_calls >= self.budget.calls:
            self.drop(item, "call_budget", False)
        elif (
            item["tool"] == "ts_zoom" and bw.sizes(bw.evidence_content(self.items, []))["images"] >= self.budget.images
        ):
            self.drop(item, "image_budget", False)
        else:
            return await self.call(item)
        return None

    def trim_verification(self, arithmetic: list[dict], limit: int) -> None:
        # R1 before/after evidence loses budget as a pair, never a misleading half.
        while (
            bw.sizes(bw.evidence_content([i for i in self.items if not i["required"]], arithmetic))["text_chars"]
            > limit
        ):
            eligible = [i for i in self.items if not i["required"]]
            candidates = [(i["priority"], i["call_number"], i) for i in eligible]
            candidates += [(0, self.calls + n + 1, row) for n, row in enumerate(arithmetic)]
            _, _, victim = max(candidates, key=lambda entry: entry[:2])
            if "arithmetic" in victim:
                arithmetic.remove(victim)
                self.dropped.append(
                    {
                        "key": victim["arithmetic"],
                        "priority": 0,
                        "reason": "text_budget",
                        "executed": False,
                        "kind": "arithmetic",
                        "response_sha256": bw.digest(bw.canonical(victim).encode()),
                        "response_sizes": bw.sizes([bw.text_block(bw.canonical(victim))]),
                    }
                )
                continue
            victims = [victim]
            if victim["key"].startswith("refine.") and victim["key"].endswith((".before", ".after")):
                prefix = victim["key"].rsplit(".", 1)[0]
                victims = [i for i in eligible if i["key"] in (prefix + ".before", prefix + ".after")]
            for item in victims:
                self.items.remove(item)
                self.drop(item, "text_budget", True)


async def collect_onset(c: Collector, sid: str, index: dict, fs: float, n: int) -> list[dict]:
    bins = []
    for i in range(3):
        lo, hi = n * i // 3, n * (i + 1) // 3
        item = bw.call_item(
            f"search.{i + 1}.profile",
            "ts_measure",
            {"id": sid, "window": [lo / fs, hi / fs] if hi > lo else None, "what": ["ecg_rr_profile"]},
            0,
        )
        reply = await c.request(item)
        bins.extend(
            profile_bins(reply, fs) if reply else [{"start": Fraction(lo), "end": Fraction(hi), "qualifies": False}]
        )
    leads = search_leads(bins)
    seeds = choose_seeds(index, leads, fs, n)
    # Local profiles depend on search, but must leave capacity for higher-priority strips.
    reserved_strips = min(len(seeds), max(0, c.budget.calls - c.calls))
    for i, seed in enumerate(seeds, 1):
        t = Fraction(seed["sample"]) / Fraction(str(fs))
        reply = await c.request(
            bw.call_item(
                f"refine.{i}.profile",
                "ts_measure",
                {"id": sid, "window": sample_window(t - 600, t + 600, fs, n), "what": ["ecg_rr_profile"]},
                10,
            ),
            reserve_calls=reserved_strips,
        )
        seed["refined_sample"] = refined_start(profile_bins(reply, fs), seed["sample"]) if reply else seed["sample"]
    # All strips precede statistics so call-cap omissions prefer the latter.
    for i, seed in enumerate(seeds, 1):
        t = Fraction(seed["refined_sample"]) / Fraction(str(fs))
        await c.request(
            bw.call_item(
                f"refine.{i}.strip",
                "ts_zoom",
                {"id": sid, "window": sample_window(t - 30, t + 30, fs, n), "view": "strip", "rows": 6},
                0,
            )
        )
    for i, seed in enumerate(seeds, 1):
        t = Fraction(seed["refined_sample"]) / Fraction(str(fs))
        for suffix, lo, hi in (("before", t - 120, t), ("after", t, t + 120)):
            await c.request(
                bw.call_item(
                    f"refine.{i}.{suffix}",
                    "ts_measure",
                    {"id": sid, "window": sample_window(lo, hi, fs, n), "what": ["ecg_rr_stats"]},
                    20,
                )
            )
    return [
        {
            "arithmetic": "onset_search_leads",
            "source_keys": [f"search.{i}.profile" for i in (1, 2, 3)],
            "search_leads_s": [[float(r["start"] / fs), float(r["end"] / fs)] for r in leads],
            "seeds": [
                {"source": s["source"], "rank": s["rank"], "t": s["sample"] / fs, "refined_t": s["refined_sample"] / fs}
                for s in seeds
            ],
        }
    ]


async def collect(case: dict, client, budget: bw.Budget = bw.BUDGET, workflow: str = "v0") -> dict:
    effective = effective_workflow(workflow, case["qtype"])
    if effective == "v0":
        return await bw.collect(case, client, budget)
    if budget.calls < 4:
        raise ValueError("B requires all four core calls")
    c = Collector(client, budget)
    channels = case["visible"]["channels"]
    if len(channels) != 1:
        raise ValueError("B currently requires one visible ECG channel")
    info = bw.payload(
        await c.call(
            bw.call_item(
                "load",
                "ts_load",
                {"source": str(bw.signal_path(case)), "fs": channels[0]["fs"]},
                required=True,
            )
        )
    )
    sid = info["id"]
    # Exact record bounds from the observable load reply, not rounded dur_s.
    fs, n = info["channels"][0]["fs"], info["channels"][0]["n"]
    # R2 preserves v0's whole-RR and gap requests, including its duration field.
    duration = n / fs if effective == "r1" else info["dur_s"]
    await c.call(bw.call_item("describe", "ts_describe", {"id": sid}, required=True))
    burden = bw.measurement(
        await c.call(
            bw.call_item(
                "burden",
                "ts_measure",
                {"id": sid, "what": ["ecg_af_burden"]},
                required=True,
            )
        ),
        "ecg_af_burden",
    )
    index = bw.measurement(
        await c.call(
            bw.call_item(
                "episode_index",
                "ts_measure",
                {"id": sid, "what": ["ecg_af_episodes"]},
                required=True,
            )
        ),
        "ecg_af_episodes",
    )
    await c.request(
        bw.call_item(
            "whole_rr",
            "ts_zoom",
            {"id": sid, "window": [0, duration], "view": "ecg_rr"},
            0 if effective == "r1" else 5,
        )
    )
    if effective == "r1":
        arithmetic = await collect_onset(c, sid, index, fs, n)
    else:
        arithmetic = r2.burden_arithmetic(index, burden, index["ref"], burden["ref"])
        noisy = await c.request(
            bw.call_item(
                "unknown_index",
                "ts_find_events",
                {"id": sid, "types": ["ecg_noisy"], "top_k": 3},
                5,
            )
        )
        if noisy and not noisy["is_error"]:
            for item in r2.burden_plan(sid, index, bw.payload(noisy), duration, fs):
                await c.request(item)
        else:
            c.dropped.append({"key": "unknown_views", "reason": "unknown_index_unavailable", "executed": False})
            # Internal gaps remain observable even if unknown-region discovery fails.
            for item in r2.burden_plan(sid, index, {"events": []}, duration, fs):
                await c.request(item)
    core_size = bw.sizes(bw.evidence_content([i for i in c.items if i["required"]], []))["text_chars"]
    limit = budget.verification_limit(core_size)
    c.trim_verification(arithmetic, limit)
    content = bw.evidence_content(c.items, arithmetic)
    metrics = {
        **bw.sizes(content),
        "core_text_chars": core_size,
        "verification_text_chars": bw.sizes(bw.evidence_content([i for i in c.items if not i["required"]], arithmetic))[
            "text_chars"
        ],
        "tool_text_chars": sum(bw.sizes(i["content"])["text_chars"] for i in c.items),
        "calls": c.calls,
        "retained_calls": len(c.items),
        "dropped": len(c.dropped),
    }
    valid = (
        metrics["verification_text_chars"] <= limit and metrics["images"] <= budget.images and c.calls <= budget.calls
    )
    return {
        "schema": "b-workflow-revisions-b1",
        "b_workflow": workflow,
        "b_workflow_effective": effective,
        "case_id": case["case_id"],
        "qtype": case["qtype"],
        "status": "ready" if valid else "budget_exceeded",
        "budget": asdict(budget),
        "verification_text_budget": limit,
        "sizes": metrics,
        "items": c.items,
        "arithmetic": arithmetic,
        "content": content,
        "dropped": c.dropped,
    }


def manifest(bundle: dict, bundle_hash: str, identity: dict, key: str, workflow: str) -> dict:
    return {
        **bw.manifest(bundle, bundle_hash, identity, key),
        "b_workflow": workflow,
        "b_workflow_effective": effective_workflow(workflow, bundle["qtype"]),
    }


def load_cached(cache: Path, key: str) -> tuple[dict, dict] | None:
    import json

    pointer = cache / "requests" / f"{key}.json"
    if not pointer.exists():
        return None
    meta = json.loads(pointer.read_text())
    if "b_workflow" not in meta:
        return bw.load_cached(cache, key)
    raw = (cache / "objects" / f"{meta['bundle_sha256']}.json").read_bytes()
    bundle = json.loads(raw)
    selected = meta["identity"].get("b_workflow")
    effective = effective_workflow(selected, bundle["qtype"])
    if (
        selected != meta["b_workflow"]
        or effective != meta["b_workflow_effective"]
        or effective != meta["identity"].get("b_workflow_effective")
        or bw.digest(bw.canonical(meta["identity"]).encode()) != key
        or (
            effective != "v0"
            and (bundle.get("b_workflow"), bundle.get("b_workflow_effective")) != (selected, effective)
        )
    ):
        raise ValueError("bundle version or request identity mismatch")
    if (
        bw.digest(raw) != meta["bundle_sha256"]
        or manifest(bundle, meta["bundle_sha256"], meta["identity"], key, meta["b_workflow"]) != meta
    ):
        raise ValueError("bundle cache integrity mismatch")
    return bundle, meta


def build(
    case: dict, server_root: Path = bw.SERVER_ROOT, cache: Path = bw.CACHE, rebuild: bool = False, workflow: str = "v0"
) -> tuple[dict, dict]:
    effective_workflow(workflow, case["qtype"])
    if workflow == "v0":
        bundle, meta = bw.build(case, server_root, cache, rebuild)
        return bundle, {**meta, "b_workflow": "v0", "b_workflow_effective": "v0"}
    provenance = bw.verify_server(server_root)

    def identity_now():
        return {
            **bw.request_identity(case, provenance),
            "workflow_sha256": workflow_hash(workflow),
            "b_workflow": workflow,
            "b_workflow_effective": effective_workflow(workflow, case["qtype"]),
        }

    identity = identity_now()
    key = bw.digest(bw.canonical(identity).encode())
    if not rebuild and (hit := load_cached(cache, key)):
        return hit

    async def session():
        from mcp import Client

        from tsmcp.core.store import SignalStore
        from tsmcp.server.app import build_server

        with tempfile.TemporaryDirectory(prefix="b-revision-handles-") as handles:
            previous = os.environ.get("TSMCP_CACHE_DIR")
            os.environ["TSMCP_CACHE_DIR"] = handles
            try:
                store = SignalStore(allow_roots=[Path(case["dir"])])
                async with Client(build_server(store, images=True)) as client:
                    return await collect(case, client, workflow=workflow)
            finally:
                if previous is None:
                    os.environ.pop("TSMCP_CACHE_DIR", None)
                else:
                    os.environ["TSMCP_CACHE_DIR"] = previous

    bundle = asyncio.run(session())
    if provenance != bw.verify_server(server_root) or identity != identity_now():
        raise ValueError("source or public case changed during bundle construction")
    bundle_hash = bw.digest((bw.canonical(bundle) + "\n").encode())
    meta = manifest(bundle, bundle_hash, identity, key, workflow)
    old = load_cached(cache, key)
    if old and old[1]["bundle_sha256"] != bundle_hash:
        raise ValueError("same request produced a different bundle")
    bw.write_json(cache / "objects" / f"{bundle_hash}.json", bundle)
    bw.write_json(cache / "requests" / f"{key}.json", meta)
    return bundle, meta


def build_in_worker(
    case: dict, server_root: Path = bw.SERVER_ROOT, cache: Path = bw.CACHE, workflow: str = "v0"
) -> tuple[dict, dict]:
    import json

    command = [
        str(server_root / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--worker",
        "--server-root",
        str(server_root),
        "--cache",
        str(cache),
        "--b-workflow",
        workflow,
    ]
    reply = subprocess.run(command, input=bw.canonical(case), capture_output=True, text=True, check=True)
    hit = load_cached(cache, json.loads(reply.stdout)["request_sha256"])
    if hit is None:
        raise ValueError("worker did not publish a bundle")
    bundle, meta = hit
    return bundle, {**meta, "b_workflow": workflow, "b_workflow_effective": effective_workflow(workflow, case["qtype"])}


def save_table(output: Path, manifests: list[dict]) -> None:
    bw.save_table(output, manifests)
    # Keep a self-contained, machine-readable omission ledger beside the full manifests.
    bw.write_json(
        output / "sizes.json",
        [
            {
                k: m[k]
                for k in (
                    "case_id",
                    "qtype",
                    "b_workflow",
                    "b_workflow_effective",
                    "status",
                    "bundle_sha256",
                    "sizes",
                    "budget",
                    "verification_text_budget",
                    "dropped",
                )
            }
            for m in manifests
        ],
    )
    text = (output / "size-table.md").read_text()
    text += "\nOmissions (each item is also recorded in its case manifest):\n\n"
    for m in manifests:
        omitted = "; ".join(f"{d['key']}: {d['reason']} (executed={d['executed']})" for d in m["dropped"]) or "none"
        text += f"- {m['case_id']} ({m['b_workflow']}/{m['b_workflow_effective']}): {omitted}\n"
    (output / "size-table.md").write_text(text)


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-workflow", choices=VERSIONS, default="v0")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--server-root", type=Path, default=bw.SERVER_ROOT)
    parser.add_argument("--cache", type=Path, default=bw.CACHE)
    parser.add_argument("--cases-root", type=Path, default=bw.ROOT / "eval/cases")
    parser.add_argument("--suite", choices=("p5afdb", "p7dev"))
    parser.add_argument("--endpoint", choices=("longest_af", "af_burden"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--output", type=Path, default=bw.ROOT / "eval/results/b-workflow")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.cache / "matplotlib"))
    if args.worker:
        _, meta = build(json.load(sys.stdin), args.server_root, args.cache, args.rebuild, args.b_workflow)
        print(bw.canonical({"request_sha256": meta["request_sha256"]}))
        return
    if not args.suite:
        parser.error("--suite is required")
    output = args.output / f"{args.suite}-{args.b_workflow}"
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for case in bw.public_cases(args.cases_root, args.suite):
        if (args.only and case["case_id"] not in args.only) or (args.endpoint and case["qtype"] != args.endpoint):
            continue
        _, meta = build(case, args.server_root, args.cache, args.rebuild, args.b_workflow)
        bw.write_json(output / "manifests" / f"{case['case_id']}.json", meta)
        manifests.append(meta)
        save_table(output, manifests)
        print(
            bw.canonical(
                {"case_id": case["case_id"], "b_workflow": args.b_workflow, "status": meta["status"], **meta["sizes"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
