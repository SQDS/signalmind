"""Synthetic known answers through real tools, evidence, submission and frozen numeric grading.

The scripted driver only copies tool measurements into claims. It never sees the generator's reference answer.
These are offline integration checks, not model-capability or independently coded grounding evaluations.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth_abp import synth_abp  # noqa: E402
from test_phase2_plugins import shifted, tfa_record  # noqa: E402
from test_unified_runner import Scripted  # noqa: E402

from eval.level2_grade import grade_level2  # noqa: E402
from eval.run_unified import application_answer  # noqa: E402
from eval.unified.loop import Limits, run_session  # noqa: E402
from eval.unified.tools import SignalMindTools  # noqa: E402
from tsmcp.runtime.taskspec import QuantitySpec, TaskSpec  # noqa: E402
from tsmcp.runtime.timebase import SyncStatus, pair_key  # noqa: E402


def scenario(family):
    if family == "tfa":
        signal, _, _ = tfa_record(g=(0.5, 0.0), noise=0.0)
        channels = [(c.name, c.unit, c.data) for c in signal.channels]
        return (
            channels,
            50,
            [0, 600],
            ["tfa"],
            ["tfa_gain_lf"],
            [QuantitySpec("gain", "tfa.gain.lf.v1", "(cm/s)/mmHg")],
            {"gain": 0.5},
            {"gain": 0.01},
        )
    fs = 500 if family == "alignment" else 125
    abp, ecg, _ = synth_abp(fs=fs, duration_s=90, hr_bpm=72, seed=12, hrv=0.03 if family == "alignment" else 0)
    if family == "alignment":
        ppg = shifted((abp - 80) / 40, fs, 45.0, 0.040)
        return (
            [("ECG", "mV", ecg), ("PPG", "?", ppg)],
            fs,
            [5, 85],
            ["align"],
            ["pulse_delay_change"],
            [QuantitySpec("shift_ms", "delay.r_to_foot.change.v1", "ms")],
            {"shift_ms": 40.0},
            {"shift_ms": 6.0},
        )
    return (
        [("ECG", "mV", ecg), ("ABP", "mmHg", abp)],
        fs,
        [10, 50],
        ["pressure", "pulse"],
        ["abp_sbp", "pulse_rate"],
        [QuantitySpec("sbp_mmhg", "abp.sbp.beat_mean.v1", "mmHg"), QuantitySpec("bpm", "pulse.rate.beats.v1", "bpm")],
        {"sbp_mmhg": 120.0, "bpm": 72.0},
        {"sbp_mmhg": 2.0, "bpm": 1.0},
    )


@pytest.mark.parametrize("mode", ["verify", "log"])
@pytest.mark.parametrize("family", ["pressure_pulse", "tfa", "alignment"])
def test_known_answer_survives_tools_to_frozen_grader(tmp_path, family, mode):
    channels, fs, window, plugins, measures, quantities, expected, tolerances = scenario(family)
    work = tmp_path / "work"
    work.mkdir()
    np.save(work / "signal.npy", np.stack([c[2] for c in channels], axis=1).astype(np.float32))
    names = [c[0] for c in channels]
    roles = {role: name for role, name in (("ecg", "ECG"), ("pressure", "ABP"), ("flow", "CBFV")) if name in names}
    spec = TaskSpec(
        "synthetic",
        "Measure the requested quantities over the stated interval.",
        quantities,
        roles,
        interval=tuple(window),
        declared_sync={pair_key(*names): SyncStatus.INDEX_ALIGNED_DECLARED.value},
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()))
    rt = tmp_path / "runtime"
    submissions, cited_ids = [], []
    state = {}

    def call(name, args):
        return [{"type": "tool_use", "id": f"call-{len(cited_ids)}-{name}", "name": name, "input": args}]

    def reply(messages):
        result = messages[-1]["content"][0]
        assert not result.get("is_error"), result
        return json.loads(result["content"][0]["text"])

    def measure(messages):
        state["signal_id"] = reply(messages)["id"]
        # One incorrect evidence window exercises real rejection and evidence reacquisition in verify mode.
        interval = [10, 20] if family == "pressure_pulse" else window
        return call(
            "ts_measure", {"id": state["signal_id"], "what": measures, "window": interval, "channels": [names[-1]]}
        )

    def submit(messages):
        rows = reply(messages)["measurements"]
        claims = []
        for q, row in zip(quantities, rows, strict=True):
            cited_ids.append(row["evidence_id"])
            claims.append(
                {
                    "claim_id": q.name,
                    "quantity": q.name,
                    "definition_id": q.definition_id,
                    "status": "observed_delay" if family == "alignment" else "validated_measurement",
                    "value": row["value"],
                    "unit": row["unit"],
                    "interval": window,
                    "evidence_refs": [row["evidence_id"]],
                }
            )
        return call("ts_submit", {"bundle": {"claims": claims}})

    def repair(messages):
        submissions.append(reply(messages))
        return call(
            "ts_measure", {"id": state["signal_id"], "what": measures, "window": window, "channels": [names[-1]]}
        )

    def finish(messages):
        submissions.append(reply(messages))
        return [{"type": "text", "text": "The structured submission contains the result.\nANSWER: {}"}]

    steps = [
        call(
            "ts_load",
            {"source": str(work / "signal.npy"), "fs": fs, "channels": names, "units": [c[1] for c in channels]},
        ),
        measure,
        submit,
    ]
    if family == "pressure_pulse":
        steps += [repair, submit]
    driver = Scripted(steps + [finish])

    async def run():
        tools = await SignalMindTools(work, rt, plugins, mode, spec_path, False, "synthetic").open()
        try:
            return await run_session(
                driver, tools, tools.instructions, spec.request_text, Limits(), tmp_path / "trace.jsonl", {}
            )
        finally:
            await tools.close()

    result = asyncio.run(run())
    assert result["error"] is None, result
    answer, _, finalization = application_answer(rt)
    assert finalization == ("accepted" if mode == "verify" else "recorded"), submissions
    score = grade_level2(answer, {"expected_answer": expected, "tolerances": tolerances})
    assert score["correct"], {"answer": answer, "score": score}
    assert not score["unnecessary_abstention"]

    ledger = {r["evidence_id"]: r for r in map(json.loads, (rt / "ledger.jsonl").read_text().splitlines())}
    assert all(ledger[e]["source_kind"] == "computed" and ledger[e]["tool"] == "ts_measure" for e in cited_ids)
    if family == "pressure_pulse" and mode == "verify":
        assert submissions[0]["decision"] == "retry"
        assert {d["code"] for d in submissions[0]["deficiencies"]} == {"scope_mismatch"}
        assert submissions[-1]["decision"] == "accepted"
        assert set(cited_ids[:2]).isdisjoint(cited_ids[2:])
