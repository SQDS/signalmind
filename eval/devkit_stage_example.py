"""Executable numeric example for source lineage and acceptance, on a generated waveform."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.cleaning.common import array_hash, digest, sha, write
from tsmcp.inspection.contracts import validate
from tsmcp.inspection.scan import scan
from tsmcp.processing import ProcessingSession

# Frozen, fixture-specific protection criteria, registered before processing.
CHECKS = ("outside_authorized_range", "source_grid", "estimate_visibility", "target_error")
SPEC = {"id": "synthetic-local-spike", "version": "1.0.0", "mandatory": CHECKS, "error_ratio_max": 0.1}


def acceptance(review, patch, authorized, evidence):
    validate("patch_plan", patch)
    validate("review_decision", review)
    for key in (
        "run_id",
        "patch_id",
        "source_version_id",
        "parent_version_id",
        "candidate_version_id",
        "patch_digest",
        "candidate_digest",
        "acceptance_spec_id",
        "acceptance_spec_version",
    ):
        if review[key] != patch[key]:
            raise ValueError(f"Review bound to different {key}")
    if (review["acceptance_spec_id"], review["acceptance_spec_version"]) != (SPEC["id"], SPEC["version"]):
        raise ValueError("Unexpected acceptance spec")
    for region in patch["write_set"]:
        if (
            any(region[k] != authorized[k] for k in ("record_id", "channel_id", "block_id"))
            or not authorized["start_sample"]
            <= region["start_sample"]
            < region["end_sample"]
            <= authorized["end_sample"]
        ):
            raise ValueError("Write outside authorized range")
    checks = review["checks"]
    if len(checks) != len(CHECKS) or {c["check_id"] for c in checks} != set(CHECKS):
        raise ValueError("Incomplete or unexpected mandatory checks")
    for c in checks:
        if (
            not c["mandatory"]
            or not c["evidence_ids"]
            or not all(
                e in evidence
                and evidence[e]["check_id"] == c["check_id"]
                and evidence[e]["result"] == c["result"]
                and evidence[e]["version"] == patch["candidate_version_id"]
                for e in c["evidence_ids"]
            )
        ):
            raise ValueError("Missing or inconsistent check evidence")
    passed = all(c["result"] == "pass" for c in checks)
    if review["mandatory_checks_pass"] != passed:
        raise ValueError("Inconsistent check summary")
    return passed and review["verdict"] == "accepted"


def pressure_control(out):
    # A generated pressure step is a protected signal feature, with a known clean reference.
    x = np.full(2500, 80.0)
    x[1250:] = 110.0
    session = ProcessingSession(x, fs=125, channel="ABP", unit="mmHg")
    candidate = session.process(session.raw.id, "lowpass", [[0, len(x)]], frequency_hz=12, filter_order=4)
    excursion = (candidate.values < 79.9) | (candidate.values > 110.1)
    from tsmcp.processing import intervals

    evidence = dict(
        raw_version_id=session.raw.id,
        observed_version_id=candidate.id,
        unit="mmHg",
        protected_range=[80, 110],
        tolerance_mmhg=0.1,
        new_excursions=intervals(excursion),
        min_after=float(candidate.values.min()),
        max_after=float(candidate.values.max()),
        reason="outside_generated_pressure_envelope",
        decision="reject_coarse" if excursion.any() else "accept_coarse",
        accepted_version_id=session.raw.id if excursion.any() else candidate.id,
    )
    if evidence["decision"] != "reject_coarse":
        raise ValueError("Expected to detect introduced ringing on the protected step")
    write(out / "introduced_issue.json", evidence)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "acceptance_spec.json", SPEC)
    pressure_control(out)
    fs = 250
    time = np.arange(5000) / fs
    clean = np.sin(2 * np.pi * 1.2 * time)
    raw = clean + 0.45 * np.sin(2 * np.pi * 50 * time)
    raw[3010:3013] += 3
    sess = ProcessingSession(raw, fs=fs, channel="PPG", unit="a.u.", max_transient_samples=5)
    coarse = sess.process(sess.raw.id, "remove_tone", [[0, len(raw)]], frequency_hz=50)
    # The generated reference is available only to the numerical acceptance check.
    raw_rmse = float(np.sqrt(np.mean((raw - clean) ** 2)))
    coarse_rmse = float(np.sqrt(np.mean((coarse.values - clean) ** 2)))
    coarse_pass = coarse_rmse < raw_rmse and np.array_equal(coarse.valid, sess.raw.valid)
    write(
        out / "coarse_gate.json",
        dict(
            passed=coarse_pass,
            original_rmse=raw_rmse,
            candidate_rmse=coarse_rmse,
            basis="synthetic reference and unchanged support",
        ),
    )
    if not coarse_pass:
        raise ValueError("Coarse candidate did not pass")
    # Select the two sharp edges using observations on the accepted coarse version.
    events, _, _ = scan(coarse.values, coarse.valid, {}, fs, "PPG", "balanced")
    edges = [x["start"] for x in events if x["kind"] == "large_step"]
    if len(edges) != 2 or not 0 < edges[1] - edges[0] <= 5:
        raise ValueError("No unique short transient supported by two observed edges")
    a, b = edges[0] + 1, edges[1] + 1
    region = {"record_id": "fixture", "channel_id": "PPG", "block_id": "block-0", "start_sample": a, "end_sample": b}
    write(
        out / "discovery.json",
        {
            "source_version_id": sess.raw.id,
            "observed_version_id": coarse.id,
            "events": events,
            "authorized_range": region,
        },
    )
    fine = sess.process(coarse.id, "interpolate", [[a, b]])
    for name, state in (("raw", sess.raw), ("coarse", coarse), ("fine", fine)):
        np.savez(out / (name + ".npz"), time=time, values=state.values, valid=state.valid, estimated=state.repaired)
    stored = np.load(out / "fine.npz", allow_pickle=False)
    with stored:
        same_grid = np.array_equal(stored["time"], time) and stored["values"].shape == clean.shape
    outside = np.ones(len(raw), bool)
    outside[a:b] = False

    def err(arr):
        return float(np.abs(arr[a:b] - clean[a:b]).mean())

    evidence = {}
    facts = [
        (
            "outside_authorized_range",
            np.array_equal(fine.values[outside], coarse.values[outside])
            and np.array_equal(fine.valid[outside], coarse.valid[outside])
            and np.array_equal(fine.repaired[outside], coarse.repaired[outside]),
            {"authorized": region},
        ),
        ("source_grid", same_grid, {"source_samples": len(time), "fs_hz": fs}),
        (
            "estimate_visibility",
            np.array_equal(fine.repaired, ~outside),
            {"estimated_samples": int(fine.repaired.sum())},
        ),
        (
            "target_error",
            err(fine.values) <= SPEC["error_ratio_max"] * err(coarse.values),
            {
                "before_mae": err(coarse.values),
                "after_mae": err(fine.values),
                "reference": "generated analytic waveform",
            },
        ),
    ]
    checks = []
    for key, flag, detail in facts:
        eid = "e-" + key
        evidence[eid] = dict(check_id=key, result="pass" if flag else "fail", version=fine.id, **detail)
        checks.append(dict(check_id=key, mandatory=True, result=evidence[eid]["result"], evidence_ids=[eid]))
    patch = dict(
        schema_version="1.0.0",
        patch_id="local-spike",
        run_id="lineage-fixture",
        candidate_id="short-step",
        record_id="fixture",
        source_version_id=sess.raw.id,
        parent_version_id=coarse.id,
        candidate_version_id=fine.id,
        parent_manifest_digest="sha256:" + sha(out / "coarse.npz"),
        candidate_digest="sha256:" + sha(out / "fine.npz"),
        candidate_artifact_ref="urn:fixture:fine",
        acceptance_spec_id=SPEC["id"],
        acceptance_spec_version=SPEC["version"],
        requirements_id="bounded-transient",
        operations=[dict(tool_id="interpolate", tool_version="0.2.0", parameters=[], seed=None)],
        read_set=[
            dict(
                range={**region, "start_sample": a - 1, "end_sample": b + 1},
                source_version_id=coarse.id,
                content_digest="sha256:" + array_hash(coarse.values[a - 1 : b + 1], coarse.valid[a - 1 : b + 1]),
            )
        ],
        write_set=[region],
        effect_set=[{**region, "start_sample": a - 1, "end_sample": b + 1}],
        global_dependencies=[],
        idempotency_key="fixture-fine",
    )
    patch["patch_digest"] = "sha256:" + digest(patch)
    passed = all(c["result"] == "pass" for c in checks)
    review = {
        k: patch[k]
        for k in (
            "schema_version",
            "run_id",
            "patch_id",
            "source_version_id",
            "parent_version_id",
            "candidate_version_id",
            "patch_digest",
            "candidate_digest",
            "acceptance_spec_id",
            "acceptance_spec_version",
        )
    }
    review.update(
        review_id="numeric-fixture-review",
        reviewer_policy_id="deterministic-fixture-checks",
        reviewer_policy_version="1.0.0",
        verdict="accepted" if passed else "rejected",
        mandatory_checks_pass=passed,
        checks=checks,
        reason_codes=["numerical_fixture_checks"],
        evidence_ids=list(evidence),
    )
    write(out / "patch_plan.json", patch)
    write(out / "numeric_review.json", review)
    write(out / "evidence.json", evidence)
    if not acceptance(review, patch, region, evidence):
        raise ValueError("Synthetic numerical gate failed")
    fig = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for name, x, style in [
        ("input", raw, "0.7"),
        ("coarse", coarse.values, "#c77718"),
        ("fine", fine.values, "#168776"),
    ]:
        ax.plot(time[a - 35 : b + 35], x[a - 35 : b + 35], label=name, color=style)
    ax.plot(time[a - 35 : b + 35], clean[a - 35 : b + 35], label="synthetic reference", color="black", ls=":")
    ax.axvspan(time[a], time[b], alpha=0.12, color="grey")
    ax.set(xlabel="Time (s)", ylabel="PPG (a.u.)", title="Synthetic PPG: tone removal and local interpolation")
    ax.legend()
    fig.savefig(out / "before-after.png", dpi=140)
    write(
        out / "summary.json",
        dict(
            source=sess.raw.id,
            coarse=coarse.id,
            fine=fine.id,
            authorized_samples=[a, b],
            before_mae=err(coarse.values),
            after_mae=err(fine.values),
            estimated_samples=int(fine.repaired.sum()),
            model_api_calls=0,
            reviewer="offline deterministic fixture",
            status="numeric_checks_passed",
            operations=fine.operations,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
