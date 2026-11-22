"""Single-case adapter: generic SignalMind operations + the case's public output helper.

Launched inside bubblewrap with only this case and output mounted. No grader,
generator, other case, baseline source or private answer is accessible.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from tsmcp.processing import ProcessingSession
from tsmcp.server.processing import build_processing_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    task = json.loads((case / "task.json").read_text())
    task["_case_dir"] = str(case)
    with np.load(case / "input.npz", allow_pickle=False) as a:
        values = a["values"].copy()
        missing, saturated = a["observed_missing_mask"].copy(), a["device_saturation_flag"].copy()
    channel = task["signal"]["channels"][0]
    policy = task["public_policy"]
    protected = [
        p["range_samples"] if isinstance(p, dict) else p
        for p in policy.get("protected_event_windows", []) + policy.get("protected_event_windows_optional", [])
    ]
    session = ProcessingSession(
        values[0],
        fs=channel["fs_hz"],
        channel=channel["name"],
        unit=channel["unit"],
        source_id=task["signal"]["signal_id"],
        missing=missing[0],
        saturated=saturated[0],
        protected_ranges=protected,
        max_gap_samples=policy.get("max_gap_samples"),
        max_transient_samples=round(policy["max_transient_ms"] * channel["fs_hz"] / 1000)
        if "max_transient_ms" in policy
        else None,
    )
    spec = importlib.util.spec_from_file_location("public_submission_helper", case / "submission_helper.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    def measure(state):
        return helper.measure_signal(state.values[None, :], state.valid[None, :], task)

    def export(state, action, rationale, limitations):
        result = helper.write_submission(
            args.output,
            task,
            values,
            state.values[None, :],
            state.artifact[None, :],
            state.repaired[None, :],
            state.valid[None, :],
            action,
            state.operations,
            rationale,
            limitations,
        )
        return {
            "new_signal_id": result["signal"]["new_signal_id"],
            "files": ["/output/result.json", "/output/output.npz"],
            "measurements": result["measurements"],
            "operations": result["operations"],
        }

    build_processing_server(session, export, measure).run("stdio")


if __name__ == "__main__":
    main()
