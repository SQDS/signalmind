"""Reproducible offline demonstration of the SignalMind v0.1 vertical slice (no model, no network).

  .venv/bin/python examples/signalmind_demo.py [--out DIR]

A synthetic 90 s ECG+ABP record (125 Hz, SBP drifting +0.1 mmHg/s, one clipped stretch) and the request
"mean SBP and DBP over 10-70 s". A scripted agent first cites a 10-20 s measurement for the 10-70 s claim and reads a
number off a view; the verifier rejects both with actionable deficiencies; the agent then measures the requested
interval, acknowledges the clipping warning and submits a supported bundle. The final
answer is rendered from the validated claims. Everything written to --out is deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.runtime.controller import Controller, Submit, ToolCall  # noqa: E402
from tsmcp.runtime.executor import CoreExecutor  # noqa: E402
from tsmcp.runtime.ledger import EvidenceLedger  # noqa: E402
from tsmcp.runtime.taskspec import abp_interval_spec  # noqa: E402

SBP, DBP = "abp.sbp.beat_mean.v1", "abp.dbp.beat_mean.v1"


class DemoAgent:
    """A scripted stand-in for the model: it decides what to call next from what it has seen."""

    def __init__(self, path: str):
        self.path, self.sid, self.step, self.m, self.view, self.defs = path, None, 0, {}, None, []

    def next(self, obs):
        res = obs.get("result") or {}
        if "id" in res:
            self.sid = res["id"]
        for row in res.get("measurements", []):
            self.m[(row["name"], tuple(res["window"]))] = row
        if res.get("evidence_kind") == "visual_observation":
            self.view = res["evidence_id"]
        if obs.get("submission"):
            self.defs = obs["submission"]["deficiencies"]
        self.step += 1
        s = self.step
        if s == 1:  # inline load: the handle is content-addressed, so the whole run is byte-reproducible
            arr = np.load(self.path, allow_pickle=False)
            return ToolCall(
                "ts_load", {"data": arr.tolist(), "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]}
            )
        if s == 2:  # a quick look and a short measurement: plausible, but not what was asked
            return ToolCall("ts_zoom", {"id": self.sid, "window": [10, 70], "channels": ["ABP"]})
        if s == 3:
            return ToolCall(
                "ts_measure", {"id": self.sid, "what": ["abp_sbp"], "window": [10, 20], "channels": ["ABP"]}
            )
        if s == 4:
            short = self.m[("abp_sbp", (10.0, 20.0))]
            return Submit(
                {
                    "text": "SBP about 122, DBP about 80.",
                    "claims": [
                        {
                            "claim_id": "sbp",
                            "quantity": "sbp_mean",
                            "status": "validated_measurement",
                            "value": short["value"],
                            "unit": "mmHg",
                            "interval": [10, 70],
                            "evidence_refs": [short["evidence_id"]],
                            "definition_id": SBP,
                        },
                        {
                            "claim_id": "dbp",
                            "quantity": "dbp_mean",
                            "status": "validated_measurement",
                            "value": 80.0,
                            "unit": "mmHg",
                            "interval": [10, 70],
                            "evidence_refs": [self.view],
                            "definition_id": DBP,
                        },
                    ],
                }
            )
        if s == 5:  # repair: measure the requested interval with the right method
            return ToolCall(
                "ts_measure", {"id": self.sid, "what": ["abp_sbp", "abp_dbp"], "window": [10, 70], "channels": ["ABP"]}
            )
        if s == 6:
            sbp, dbp = self.m[("abp_sbp", (10.0, 70.0))], self.m[("abp_dbp", (10.0, 70.0))]
            return Submit(
                {
                    "claims": [
                        {
                            "claim_id": "sbp",
                            "quantity": "sbp_mean",
                            "status": "validated_measurement",
                            "value": sbp["value"],
                            "unit": "mmHg",
                            "interval": [10, 70],
                            "evidence_refs": [sbp["evidence_id"]],
                            "definition_id": SBP,
                            "acknowledged_warnings": [w["code"] for w in sbp["warnings"]],
                        },
                        {
                            "claim_id": "dbp",
                            "quantity": "dbp_mean",
                            "status": "validated_measurement",
                            "value": dbp["value"],
                            "unit": "mmHg",
                            "interval": [10, 70],
                            "evidence_refs": [dbp["evidence_id"]],
                            "definition_id": DBP,
                            "acknowledged_warnings": [w["code"] for w in dbp["warnings"]],
                        },
                    ]
                }
            )
        return Submit({"claims": [], "final": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="signalmind-demo-"))
    out.mkdir(parents=True, exist_ok=True)
    abp, ecg, beats = synth_abp(fs=125, duration_s=90, sbp=118, dbp=78, sbp_slope_per_s=0.1, clip_at=124.0, seed=11)
    path = out / "demo_record.npy"
    np.save(path, np.stack([ecg, abp], 1).astype(np.float32))
    spec = abp_interval_spec(
        "demo",
        "What are the mean per-beat SBP and DBP over 10-70 s?",
        "ABP",
        (10, 70),
        quantities=("sbp_mean", "dbp_mean"),
        ecg_channel="ECG",
        capabilities=CoreExecutor.TOOLS,
    )
    ledger = EvidenceLedger("demo", path=out / "ledger.jsonl")
    store = SignalStore(allow_roots=[out])
    executor = CoreExecutor(store, ledger, root=out)
    outcome = Controller(spec, DemoAgent(str(path)), executor, ledger).run()
    truth = {
        k: float(np.mean([b[k] for b in beats if 10 <= b["t_foot_s"] and b["t_end_s"] <= 70])) for k in ("sbp", "dbp")
    }
    report = {
        "status": outcome.status,
        "steps": outcome.steps,
        "tool_calls": outcome.tool_calls,
        "submissions": outcome.submissions,
        "events": outcome.events,
        "answer": outcome.answer,
        "generator_truth_for_reference_only": truth,
        "ledger": [r.summary() for r in ledger.records()],
        "ledger_fingerprint": ledger.fingerprint(),
    }
    (out / "demo_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(
        f"outcome: {outcome.status} after {outcome.steps} steps, {outcome.tool_calls} tool calls, "
        f"{outcome.submissions} submissions"
    )
    for e in outcome.events:
        if "submit" in e:
            print(
                f"  step {e['step']}: submit -> {e['submit']}; deficiencies: "
                + ", ".join(f"{d['claim_id']}:{d['code']}" for d in e["deficiencies"])
            )
        elif "tool" in e:
            print(f"  step {e['step']}: {e['tool']} {json.dumps(e['args'])[:80]} -> evidence {e['evidence']}")
    print("final answer (rendered from validated claims):")
    print("  " + outcome.answer["text"].replace("\n", "\n  "))
    print(f"generator truth (not used by the system): SBP {truth['sbp']:.2f}, DBP {truth['dbp']:.2f}")
    print(f"ledger: {len(ledger)} records, fingerprint {ledger.fingerprint()}; files in {out}")


if __name__ == "__main__":
    main()
