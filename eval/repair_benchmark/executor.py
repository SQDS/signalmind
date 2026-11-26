"""Closed ActionDSL worker. Entrypoint receives exactly one public case, one plan and an empty output."""

import sys
import time
from pathlib import Path

import numpy as np
import scipy
from jsonschema import ValidationError, validate

from tsmcp import processing
from tsmcp.cleaning.common import array_hash, changed, digest, read, sha, write
from tsmcp.cleaning.tasks import probe
from tsmcp.processing import ProcessingSession, intervals

from .public import CATALOG

SCHEMAS = Path(__file__).with_name("contracts")


def spans(case, requested, permitted):
    result = []
    for item in requested:
        a, b = item["start_sample"], item["end_sample"]
        if (
            item["channel_id"] != case["task"]
            or type(a) is not int
            or type(b) is not int
            or not any(p["start_sample"] <= a < b <= p["end_sample"] for p in permitted)
        ):
            raise ValueError("Range outside the frozen channel or scope")
        result.append((a - case["offset"], b - case["offset"]))
    return result


def apply_plan(case, plan, t, x, v, e):
    validate(plan, read(SCHEMAS / "action_plan.schema.json"))
    if any(plan[k] != case[k] for k in ("case_id", "input_hash", "parent_version_id", "profile_id")):
        raise ValueError("Stale plan binding")
    if len({z["action_id"] for z in plan["actions"]}) != len(plan["actions"]):
        raise ValueError("Duplicate action ID")
    ids = {z["candidate_id"] for z in case["candidates"]}
    if any(z["candidate_id"] not in ids for z in plan["diagnoses"]):
        raise ValueError("Unknown detected candidate")
    y, estimated = x.copy(), e.copy()
    unavailable = np.zeros(len(x), bool)
    claimed = np.zeros(len(x), bool)
    ops = []
    amendments = []
    session = ProcessingSession(
        x,
        fs=case["fs"],
        channel=case["task"],
        unit=case["unit"],
        missing=~v | e,
        source_id=case["parent_version_id"],
        max_transient_samples=4,
    )
    for action in plan["actions"]:
        kind = action["kind"]
        params = action["parameters"]
        reads = spans(case, action["read_ranges"], case["public_read_ranges"])
        writes = spans(case, action["write_ranges"], case["allowed_write_ranges"])
        if kind == "preserve":
            if params:
                raise ValueError("Preserve has no parameters")
        elif kind == "mark_unusable":
            regions = spans(case, params["ranges"], case["allowed_write_ranges"])
            if not regions or set(params) != {"ranges"}:
                raise ValueError("Expected task ranges for mark_unusable")
            for a, b in regions:
                unavailable[a:b] = True
        elif kind == "propose_event_amendment":
            for change in params["declared_changes"]:
                if set(change) != {"kind", "range"} or change["kind"] not in ("add", "remove", "relocate"):
                    raise ValueError("Invalid shadow event proposal")
                spans(case, [change["range"]], case["allowed_write_ranges"])
                amendments.append(change)
        else:
            name = action["operator_id"]
            spec = CATALOG.get(name)
            if spec is None or spec["version"] != action["operator_version"]:
                raise ValueError("Operator ID and version must match the frozen catalog")
            # These are verifiable declarations about the fixed operation. None are tunable DSP inputs.
            fixed = dict(
                anchors_per_side=spec["anchors_per_side"],
                max_samples=spec["max_samples"],
                channel_id=case["task"],
                target_channel=case["task"],
                estimation_label="estimated",
                method="linear" if name == "interpolate" else "pchip",
            )
            if len(writes) == 1:
                fixed["write_span_samples"] = writes[0][1] - writes[0][0]
            for k, value in params.items():
                if k not in fixed or type(value) is not type(fixed[k]) or value != fixed[k]:
                    raise ValueError(f"Unexpected parameter {k!r}; use empty parameters for this fixed operator")
            k = spec["anchors_per_side"]
            for a, b in writes:
                if b - a > spec["max_samples"] or not any(lo <= a - k and b + k <= hi for lo, hi in reads):
                    raise ValueError("Operator read or write range is insufficient")
                if claimed[a:b].any():
                    raise ValueError("Conflicting writes on the shared parent")
                claimed[a:b] = True
            trial = session.process(session.raw.id, name, writes)
            for a, b in writes:
                y[a:b], estimated[a:b] = trial.values[a:b], True
            ops.append(dict(action_id=action["action_id"], operator=trial.operations[-1]))
    for op in ops:
        k = CATALOG[op["operator"]["op"]]["anchors_per_side"]
        for a, b in op["operator"]["ranges_samples"]:
            if claimed[a - k : a].any() or claimed[b : b + k].any():
                raise ValueError("Another proposed write overlaps an interpolation anchor")
    return y, estimated, unavailable, ops, amendments


def run(public, output, arm):
    started = time.monotonic()
    case = read(public / "case.json")
    if digest({k: v for k, v in case.items() if k != "input_hash"}) != case["input_hash"] or case["catalog"] != CATALOG:
        raise ValueError("Public capability manifest changed")
    if case["public_tool_catalog_hash"] != digest([CATALOG, sha(processing.__file__)]):
        raise ValueError("The numerical operator source changed since the case was frozen")
    with np.load(public / "observed.npz", allow_pickle=False) as z:
        t, x, v, e = (z[k] for k in ("t", "x", "valid", "estimated"))
    if array_hash(t, x, v, e) != case["observed_hash"]:
        raise ValueError("Public parent changed")
    plan = read(public / "plan.json")
    try:
        if plan.get("kind") == "submit_waveform":
            file = public / "submitted.npz"
            if (
                plan["input_hash"] != case["input_hash"]
                or plan["parent_version_id"] != case["parent_version_id"]
                or sha(file) != plan["sha256"]
            ):
                raise ValueError("Unbound waveform submission")
            with np.load(file, allow_pickle=False) as arrays:
                y, valid, est = (arrays[k] for k in ("values", "valid", "estimated"))
                if not np.array_equal(arrays["time"], t) or arrays["unit"].item() != case["unit"]:
                    raise ValueError("Changed source grid or unit")
            if (
                y.shape != x.shape
                or y.dtype.kind not in "fiu"
                or any(z.shape != v.shape or z.dtype != bool for z in (est, valid))
            ):
                raise ValueError("Changed waveform shape or mask type")
            allowed = np.zeros(len(x), bool)
            for a, b in spans(case, case["allowed_write_ranges"], case["public_read_ranges"]):
                allowed[a:b] = True
            delta = changed(x, y)
            if (
                ((delta | (est != e) | (valid != v)) & ~allowed).any()
                or (e & ~est).any()
                or (valid & ~v).any()
                or not np.isfinite(y[v]).all()
            ):
                raise ValueError("Submitted arrays exceed write or observed support")
            disabled, ops, shadow = v & ~valid, [], []
            status = "completed"
        else:
            y, est, disabled, ops, shadow = apply_plan(case, plan, t, x, v, e)
            status = "abstained" if plan["needs_evidence"] and not (ops or disabled.any() or shadow) else "completed"
        reasons = []
    except (ValidationError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        y, est, disabled, ops, shadow = x, e, np.zeros_like(v), [], []
        status, reasons = "invalid_output", [str(exc).splitlines()[0][:400]]
    result = dict(
        result_schema_version="proposal.v1",
        case_id=case["case_id"],
        arm_id=arm,
        run_id=digest([arm, plan, case["input_hash"]]),
        input_hash=case["input_hash"],
        parent_version_id=case["parent_version_id"],
        status=status,
        decision="waveform_repair"
        if changed(x, y).any()
        else "mark_unusable"
        if disabled.any()
        else "event_amendment"
        if shadow
        else "need_evidence"
        if status == "abstained"
        else "preserve",
        fallback_used=status == "invalid_output",
        reason_codes=reasons,
        selected_candidate=None,
        derived_measurements=None,
        provider_receipt=None,
    )
    for key, kind, name, arr in [
        ("released_waveform", "waveform", "values", y),
        ("validity_mask", "validity_mask", "task_valid", v & ~disabled),
        ("provenance_mask", "provenance_mask", "estimated", est),
    ]:
        path = output / (name + ".npy")
        np.save(path, arr, allow_pickle=False)
        result[key] = dict(artifact_id=path.name, kind=kind, sha256=sha(path))
    result["selected_candidate"] = result["released_waveform"] if result["decision"] == "waveform_repair" else None
    for key, doc in [
        (
            "trace",
            dict(
                operations=ops,
                shadow_event_requests=shadow,
                plan_hash=digest(plan),
                numerical_environment=dict(
                    numpy=np.__version__, scipy=scipy.__version__, python=sys.version.split()[0]
                ),
                offset=case["offset"],
                modified_global_ranges=[[a + case["offset"], b + case["offset"]] for a, b in intervals(changed(x, y))],
            ),
        ),
        ("native_measurements", probe(y, t, case["fs"], v & ~disabled, est, case["task"])),
    ]:
        path = output / (key + ".json")
        write(path, doc)
        result[key] = dict(artifact_id=path.name, kind=key, sha256=sha(path))
    result["usage"] = dict(
        model_requests=0,
        tool_calls=len(ops),
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cached_input_tokens=0,
        usd_cost=0,
        wall_seconds=time.monotonic() - started,
        unknown_fields=[],
    )
    validate(result, read(SCHEMAS / "method_result.schema.json"))
    write(output / "result.json", result)


if __name__ == "__main__":
    run(Path("/case"), Path("/output"), sys.argv[1])
