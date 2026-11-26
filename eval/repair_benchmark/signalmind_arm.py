"""Run the production diagnosis, repair, independent release and bounded rework on frozen T1 candidates."""

import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from tsmcp.cleaning.common import digest, implementation_hash, read, sha, write
from tsmcp.cleaning.dataset_pipeline import Record
from tsmcp.cleaning.dataset_review import Reviewer
from tsmcp.cleaning.engine import evaluate_output, load_arrays
from tsmcp.cleaning.knowledge import Knowledge
from tsmcp.cleaning.quality_gate import SPEC_ID, spike_check
from tsmcp.cleaning.repairability import PROFILE
from tsmcp.inspection.scan import scan

from .providers import EpisodeLimit
from .public import SIMULATOR
from .runner import isolated


class FrozenRecord(Record):
    """Production record with the organizer's detector scope and a common numerical trial budget."""

    def coarse(self):
        return self.raw.copy()

    def scan_version(self, versions):
        get = self.loader(versions)
        ((cid, c),) = self.cases.items()
        t, x, v, e, pr = get(cid)
        regions = self.public_case["candidates"]
        offset = self.public_case["offset"]
        if self.round > 1:
            residual, _, _ = scan(x, v & ~e, {}, c["fs"], c["task"], "balanced")
            current = {
                (q["start"] + offset, q["end"] + offset)
                for q in residual
                if q["kind"] in ("biphasic_core", "impulsive_core")
            }
            regions = [r for r in regions if (r["start_sample"], r["end_sample"]) in current]
        hits = []
        for row in regions:
            a, b = row["start_sample"] - offset, row["end_sample"] - offset
            pre = spike_check(c, x, v, e, pr, a, b)
            observed = x[a:b][v[a:b] & ~e[a:b]]
            anchors = [
                x[a - 1] if a else None,
                min(observed) if len(observed) else None,
                max(observed) if len(observed) else None,
                x[b] if b < len(x) else None,
            ]
            hits.append(
                dict(
                    id=row["candidate_id"],
                    case=cid,
                    task=c["task"],
                    a=a,
                    b=b,
                    reason="isolated_spike_candidate",
                    frozen_trigger=row["reason"],
                    applicability=pre,
                    support_status="suspected_artifact" if pre["status"] == "evaluated_no_alert" else pre["status"],
                    value=float(np.ptp(observed)) if len(observed) else None,
                    anchor_values=[float(z) if z is not None and np.isfinite(z) else None for z in anchors],
                )
            )
        return self.describe_hits(versions, get, [c], hits)

    def charge(self, kind, trial=False):
        if time.monotonic() >= self.toolset.deadline:
            raise TimeoutError("episode deadline")
        if len(self.work) >= self.max_tools or (trial and self.toolset.candidates_used >= self.toolset.max_candidates):
            raise EpisodeLimit("shared_tool_or_candidate_limit")
        self.work.append(kind)
        self.toolset.tool_count = len(self.work)
        self.toolset.candidates_used += int(trial)

    def picture(self, *args, **kwargs):
        self.charge("plot")
        return super().picture(*args, **kwargs)

    def spike_candidate(self, *args, **kwargs):
        self.charge("spike_trial", trial=True)
        return super().spike_candidate(*args, **kwargs)

    def spike_trials(self, parents, hits, all_hits=None):
        log = []
        head = parents
        path = self.path / ("spike-trials.json" if self.round == 1 else "spike-trials-rescan.json")
        for hit in hits:
            if hit["wave_requested"] != "interpolate_spike":
                continue
            if self.toolset.max_candidates - self.toolset.candidates_used < 2:
                hit["measurement"] = "review"
                self.unresolved.append(dict(stage="spike_trial", hit=hit["id"], reason="candidate_budget"))
                log.append(
                    dict(
                        hit=hit["id"],
                        case=hit["case"],
                        a=hit["a"],
                        b=hit["b"],
                        status="pending",
                        reason="candidate_budget",
                    )
                )
                continue
            head = super().spike_trials(head, [hit], all_hits=hits if all_hits is None else all_hits)
            log.extend(read(path))
        write(path, log)
        return head

    def fine(self, parents, hits):
        if any(h["wave_requested"] == "mark_invalid" for h in hits):
            self.charge("mask_trial", trial=True)
        return super().fine(parents, hits)


def run_production(tools, ask, common, budgets):
    from tsmcp.cleaning.derived_measurements import export_derived

    c = tools.case
    goal = {"ECG": "ECG_RR", "ABP": "ABP_pressure", "PPG": "PPG_PPI"}[c["task"]]
    origin = c.get("acquisition", {})
    domain = dict(
        profile_id=PROFILE["id"],
        profile_version=PROFILE["version"],
        use="development",
        species=origin.get("species"),
        device=origin.get("device"),
        goals={c["task"]: goal},
    )
    root = tools.folder / "production"
    config = dict(
        method="dataset_agent",
        model="deepseek-flash",
        inspection_preset="balanced",
        routing="reason",
        purpose=goal,
        write_views=budgets["max_tools"],
        joint_views=0,
        repair_protocol="v1.1",
        repair_domain=domain,
        max_llm_calls_per_record=budgets["max_turns"],
    )
    out = root / "public" / "block" / "input.npz"
    out.parent.mkdir(parents=True)
    np.savez_compressed(out, time=tools.t, values=tools.x, valid=tools.valid, estimated=tools.estimated)
    block = dict(
        id="block",
        record_id="record",
        unit=c["unit"],
        fs=c["fs"],
        task=c["task"],
        n=len(tools.x),
        input_sha256=sha(out),
        **origin,
    )
    implementation = implementation_hash()
    raw = evaluate_output(
        root,
        block,
        {"hash": "identity"},
        implementation,
        None,
        tools.x,
        tools.t,
        tools.x,
        tools.x.copy(),
        tools.valid,
        tools.estimated,
        [],
        time.monotonic(),
    )
    reviewer = Reviewer(root, config)
    reviewer.knowledge = Knowledge(root / "episode-memory")
    reviewer.snapshot = reviewer.knowledge.freeze(root)
    reviewer.deadline = tools.deadline
    reviewer.record = "record"
    record = FrozenRecord(root, [block], {"identity": {"block": raw}}, config, reviewer, implementation)
    if (
        c.get("simulation_profile") == "analytic-ecg-rr.dev1"
        and origin == SIMULATOR
        and c["fs"] == 125
        and c["task"] == "ECG"
        and c["unit"] == "synthetic units"
    ):
        sim = deepcopy(PROFILE)
        sim.update(
            id=c["simulation_profile"],
            version="1.0",
            calibration="engineering_simulation_only",
            species=origin["species"],
            device=origin["device"],
        )
        sim["modalities"] = {"ECG": dict(sim["modalities"]["ECG"], unit=c["unit"], field=origin["field"])}
        domain.update(profile_id=sim["id"], profile_version=sim["version"])
        record.repair_domain.profile = sim
        record.repair_domain.spec_id = digest([SPEC_ID, sim, domain])
        record.acceptance_id = record.repair_domain.spec_id
        write(root / "simulation-profile.json", sim)
    record.public_case = c
    record.toolset = tools
    record.work = []
    record.max_tools = budgets["max_tools"]
    first = True

    def request(messages, schemas, system):
        nonlocal first
        record.charge("model_submission")
        if first:
            messages = [dict(role="user", content=[*common["content"], *messages[0]["content"]]), *messages[1:]]
            first = False
        content, reqid = ask(system, messages, schemas)
        blocks = [SimpleNamespace(model_dump=lambda *, exclude_none, b=b: b) for b in content]
        return SimpleNamespace(id=reqid, content=blocks)

    reviewer.client = SimpleNamespace(request=request)
    record.prepare(c["input_hash"])
    if not record.prepared_ok:
        raise ValueError("Frozen detector observations could not be prepared")
    final, _, measures, report = record.run()
    export = root / "export"
    export.mkdir(exist_ok=True)
    write(export / "task-dispositions.json", measures)
    export_derived(root, [block], final, record.raw, record.repair_domain, final["block"]["artifact_id"], export)
    art = final["block"]
    _, _, y, v, e = load_arrays(root, block, art)
    package = root / "proposed-waveform.npz"
    np.savez(package, values=y, valid=v, estimated=e, time=tools.t, unit=c["unit"])
    envelope = dict(
        kind="submit_waveform",
        input_hash=c["input_hash"],
        parent_version_id=c["parent_version_id"],
        sha256=sha(package),
    )
    release = tools.folder / "structural-release"
    remaining = tools.deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("episode deadline")
    checked = isolated(tools.public, envelope, release, tools.arm, package, timeout=min(remaining, 120))
    if checked["status"] != "completed":
        raise ValueError("Released production waveform failed common structural validation")
    nominations = []
    # Order is the actual production order (mask then numerical proposals for each round).
    for suffix in ("", "-rescan"):
        masks = record.path / ("fine-gate" + suffix + ".json")
        if masks.is_file():
            nominations.extend(z["candidate"] for z in read(masks).values() if z.get("candidate"))
        trials = record.path / ("spike-trials" + suffix + ".json")
        for row in read(trials) if trials.is_file() else []:
            if row.get("comparison", {}).get("nominated") and row.get("candidate"):
                nominations.append(row["candidate"])
    selected = None
    support = {}
    if nominations:
        nominated = read(root / "artifacts" / nominations[-1] / "artifact.json")
        _, _, values, valid, estimated = load_arrays(root, block, nominated)
        for name, kind, arr in [
            ("selected-values.npy", "waveform", values),
            ("selected-task_valid.npy", "validity_mask", valid),
            ("selected-estimated.npy", "provenance_mask", estimated),
        ]:
            np.save(tools.folder / name, arr, allow_pickle=False)
            descriptor = dict(artifact_id=name, kind=kind, sha256=sha(tools.folder / name))
            if kind == "waveform":
                selected = descriptor
            else:
                support[kind] = descriptor
    receipts = [read(path) for path in sorted((root / "dataset-receipts").glob("*.json"))]
    patches = [read(path) for path in sorted((record.path / "patches").glob("*.json"))]
    details = dict(
        implementation=implementation,
        acceptance_spec=record.acceptance_id,
        knowledge_snapshot=reviewer.snapshot["id"],
        outstanding=report["unresolved"],
        measurement_states=report["measurement_states"],
        measurement_policy=[
            dict(start_sample=m["a"] + c["offset"], end_sample=m["b"] + c["offset"], quality=m["quality"])
            for m in measures
            if m["quality"] != "keep"
        ],
        committed_patches=[
            dict(
                key=p["idempotency_key"],
                operation=p["plan"]["tool"]["tool_id"],
                gate=p["receipt"]["hard_gate"]["status"],
                review=p["receipt"]["review_decision"],
            )
            for p in patches
        ],
        eligible_source=not record.repair_domain.metadata(block)["reason_codes"],
        record_scope="single-channel repair-only; frozen candidates; isolated empty initial memory",
        selected_semantics="last actually nominated full waveform on its then-current parent",
        nominations=nominations,
        selected_candidate_support=support,
        release_status=art["artifact_id"],
        stages=[dict(stage=z["stage"], binding=z["binding"], request_id=z["request_id"]) for z in receipts],
        work=record.work,
    )
    return release, selected, details
