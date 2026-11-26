"""Prepare observed inputs and execute a proposed short-core plan in the existing isolated numerical environment."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from eval.processing_tools import PYTHON, ROOT, sandbox_command
from tsmcp.cleaning.common import read, sha, write

from .public import CATALOG, prepare


def baseline(case, method):
    actions = []
    if method in CATALOG:
        k = CATALOG[method]["anchors_per_side"]
        for i, r in enumerate(case["allowed_write_ranges"]):
            actions.append(
                dict(
                    action_id=f"trial-{i}",
                    kind="invoke_registered_operator",
                    operator_id=method,
                    operator_version=CATALOG[method]["version"],
                    parameters={},
                    read_ranges=[dict(r, start_sample=r["start_sample"] - k, end_sample=r["end_sample"] + k)],
                    write_ranges=[dict(r)],
                    expected_effect="remove observed short excursion",
                    stop_conditions=["invalid numerical support"],
                )
            )
    if not actions:
        actions.append(
            dict(
                action_id="identity",
                kind="preserve",
                operator_id=None,
                operator_version=None,
                parameters={},
                read_ranges=[],
                write_ranges=[],
                expected_effect="retain observations",
                stop_conditions=["no numerical write"],
            )
        )
    return dict(
        plan_schema_version="proposal.v1",
        plan_status="proposed",
        diagnoses=[],
        actions=actions,
        needs_evidence=[],
        summary="Deterministic short-core baseline",
        **{k: case[k] for k in ("case_id", "input_hash", "parent_version_id", "profile_id")},
    )


def isolated(public, plan, destination, arm, waveform=None, *, timeout=120):
    if destination.exists():
        raise ValueError("Choose a fresh method output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".repair-run-") as temp:
        temp = Path(temp)
        stage = temp / "input"
        out = temp / "result"
        stage.mkdir()
        out.mkdir()
        for name in ("case.json", "observed.npz"):
            shutil.copyfile(public / name, stage / name)
        write(stage / "plan.json", plan)
        if waveform is not None:
            shutil.copyfile(waveform, stage / "submitted.npz")
        cmd = sandbox_command(stage, out)[:-1]
        # Mount just the public worker and numerical package. The private scorer and other runs stay outside.
        cmd += ["--ro-bind", str(ROOT / "src/tsmcp"), "/app/tsmcp"]
        base = Path(__file__).parent
        for name in (
            "__init__.py",
            "public.py",
            "executor.py",
            "contracts/action_plan.schema.json",
            "contracts/method_result.schema.json",
        ):
            cmd += ["--ro-bind", str((base / name).resolve()), "/app/repair_benchmark/" + name]
        cmd += ["--setenv", "PYTHONPATH", "/app", "--", str(PYTHON), "-m", "repair_benchmark.executor", arm]
        subprocess.run(cmd, check=True, timeout=timeout)
        result = read(out / "result.json")
        out.rename(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--observed", type=Path, required=True)
    prep.add_argument("--case-id", required=True)
    prep.add_argument("--task", choices=["ECG", "PPG", "ABP"], required=True)
    prep.add_argument("--fs", type=float, required=True)
    prep.add_argument("--unit", required=True)
    prep.add_argument("--offset", type=int, default=0)
    prep.add_argument("--acquisition", type=Path, help="Optional public acquisition metadata JSON")
    prep.add_argument("--out", type=Path, required=True)
    execute = subs.add_parser("execute")
    execute.add_argument("--public", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--arm-id")
    group = execute.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", type=Path)
    group.add_argument("--waveform", type=Path)
    group.add_argument("--baseline", choices=["identity", *CATALOG])
    args = parser.parse_args()
    if args.command == "prepare":
        with np.load(args.observed, allow_pickle=False) as z:
            x = z["values"] if "values" in z else z[args.task]
            case = prepare(
                args.out,
                args.case_id,
                args.task,
                args.unit,
                args.fs,
                z["time"],
                x,
                z["valid"] if "valid" in z else np.isfinite(x),
                z["estimated"] if "estimated" in z else np.zeros(len(x), bool),
                args.offset,
                acquisition=read(args.acquisition) if args.acquisition else None,
            )
        print(case["input_hash"])
    else:
        if args.waveform:
            with np.load(args.waveform, allow_pickle=False) as arrays:
                plan = dict(
                    kind="submit_waveform",
                    input_hash=arrays["input_hash"].item(),
                    parent_version_id=arrays["parent_version_id"].item(),
                    sha256=sha(args.waveform),
                )
        else:
            plan = read(args.plan) if args.plan else baseline(read(args.public / "case.json"), args.baseline)
        arm = args.arm_id or args.baseline or ("submitted_waveform" if args.waveform else "submitted_plan")
        print(isolated(args.public, plan, args.out, arm, args.waveform)["status"])


if __name__ == "__main__":
    main()
