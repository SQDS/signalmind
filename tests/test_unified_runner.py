"""Unified model adapter: endpoint security, key handling, API-format conversion, loop/transcript compatibility.
No network: every driver is exercised with a fake client or transport."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from eval.level2_traces import render  # noqa: E402
from eval.run_unified import application_answer, assert_pinned_checkout, campaign_dirs, spent  # noqa: E402
from eval.unified.drivers import AnthropicDriver, InsecureEndpoint, OpenAIChatDriver, read_key, secure_url  # noqa: E402
from eval.unified.loop import Limits, run_session  # noqa: E402
from eval.unified.tools import PythonTools, SignalMindTools  # noqa: E402


def test_credentials_only_over_https_or_local_tunnel():
    assert secure_url("https://api.deepseek.com/anthropic/") == "https://api.deepseek.com/anthropic"
    assert secure_url("http://127.0.0.1:8443") == "http://127.0.0.1:8443"
    for bad in ("http://107.173.251.87:8080/", "http://example.com", "", "ftp://x"):
        with pytest.raises(InsecureEndpoint):
            secure_url(bad)


def test_key_handling(tmp_path, monkeypatch):
    f = tmp_path / "k"
    f.write_text("secret-value\n")
    f.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        read_key({"key_file": str(f)})
    f.chmod(0o600)
    assert read_key({"key_file": str(f)}) == "secret-value"
    monkeypatch.delenv("GPT6_TERRA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not set") as e:
        read_key({"key_env": "GPT6_TERRA_API_KEY"})
    assert "secret" not in str(e.value)


def test_anthropic_driver_marks_cache_and_maps_usage():
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}], stop_reason="end_turn", model="m", id="x",
                                   usage={"input_tokens": 3, "output_tokens": 2, "cache_read_input_tokens": 7})  # fmt: skip

    d = AnthropicDriver({"model": "m"}, max_tokens=16384, client=SimpleNamespace(messages=FakeMessages()))
    st = d.step("sys", [{"role": "user", "content": "hello"}], [])
    assert seen["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert seen["max_tokens"] == 16384
    assert st.usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 7,
    }


def test_openai_driver_converts_tools_and_messages():
    sent = {}

    def transport(body):
        sent.update(body)
        return {"model": "gpt-6-terra", "id": "c1", "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                                                             "prompt_tokens_details": {"cached_tokens": 6}},
                "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                    {"id": "call1", "function": {"name": "ts_load", "arguments": '{"source": "x.npy"}'}}]}}]}  # fmt: skip

    d = OpenAIChatDriver({"model": "gpt-6-terra", "vision": True}, transport=transport)
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "python", "input": {"code": "1"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "is_error": False, "content": [
                {"type": "text", "text": "out"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}]}]  # fmt: skip
    st = d.step("sys", msgs, [{"name": "ts_load", "description": "d", "input_schema": {"type": "object"}}])
    roles = [m["role"] for m in sent["messages"]]
    assert (
        roles == ["system", "user", "assistant", "tool", "user"] and sent["tools"][0]["function"]["name"] == "ts_load"
    )
    assert sent["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"code": "1"}'
    assert sent["messages"][4]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert st.stop_reason == "tool_use" and st.content[0] == {
        "type": "tool_use",
        "id": "call1",
        "name": "ts_load",
        "input": {"source": "x.npy"},
    }
    assert st.usage["input_tokens"] == 4 and st.usage["cache_read_input_tokens"] == 6


class Scripted:
    def __init__(self, steps):
        self.steps = steps

    def step(self, system, messages, tools):
        from eval.unified.drivers import Step

        nxt = self.steps.pop(0)
        return Step(nxt(messages) if callable(nxt) else nxt, {"output_tokens": 1}, "end_turn", "fake", "id")


@pytest.mark.parametrize("mode,finalization", [("verify", "accepted"), ("log", "recorded")])
def test_loop_with_signalmind_tools_writes_compatible_transcript_and_answer(tmp_path, mode, finalization):
    abp, ecg, _ = synth_abp(fs=125, duration_s=60, seed=41)
    work = tmp_path / "work"
    work.mkdir()
    np.save(work / "signal.npy", np.stack([ecg, abp], 1).astype(np.float32))
    rt = tmp_path / "rt"

    def measure(msgs):
        sid = json.loads(msgs[-1]["content"][0]["content"][0]["text"])["id"]
        return [
            {
                "type": "tool_use",
                "id": "b",
                "name": "ts_measure",
                "input": {"id": sid, "what": ["pulse_rate"], "window": [10, 40], "channels": ["ABP"]},
            }
        ]

    def submit(msgs):
        m = json.loads(msgs[-1]["content"][0]["content"][0]["text"])["measurements"][0]
        return [{"type": "tool_use", "id": "c", "name": "ts_submit", "input": {"bundle": {"claims": [
            {"claim_id": "r", "quantity": "bpm", "status": "validated_measurement", "value": m["value"], "unit": "bpm",
             "interval": [10, 40], "evidence_refs": [m["evidence_id"]]}]}}}]  # fmt: skip

    driver = Scripted([[{"type": "tool_use", "id": "a", "name": "ts_load", "input": {"source": str(work / "signal.npy"), "fs": 125,
                                                                                    "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]}}],
                       measure, submit, [{"type": "text", "text": 'done\nANSWER: {"bpm": 72}'}]])  # fmt: skip
    tr = tmp_path / "t.jsonl"

    async def go():  # open, run and close the in-process MCP client inside one event loop (as the runner does)
        tools = await SignalMindTools(work, rt, ["pressure", "pulse"], mode, None, False, "t").open()
        names = {t["name"] for t in tools.schemas}
        assert {"ts_load", "ts_measure", "ts_submit"} <= names and "python" not in names
        try:
            return await run_session(driver, tools, "sys", "question", Limits(), tr, {"arm": "test"})
        finally:
            await tools.close()

    r = asyncio.run(go())
    assert r["stop"] == "end_turn" and r["tool_calls"] == 3 and r["error"] is None
    state = json.loads((rt / "answer.json").read_text())
    assert state["finalization"] == finalization
    answer, _, status = application_answer(rt)
    assert status == finalization and answer["bpm"] == pytest.approx(72, abs=1)
    events = [json.loads(x) for x in tr.read_text().splitlines()]
    md = render(
        events,
        "question",
        {
            "dur_s": 60,
            "channels": [{"name": "ECG", "unit": "mV", "fs": 125}, {"name": "ABP", "unit": "mmHg", "fs": 125}],
        },
    )
    assert "tool call: ts_measure" in md and "tool call: ts_submit" in md and events[-1]["type"] == "result"


def test_loop_budget_and_errors_are_contained(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    pt = PythonTools(work, vision=False, isolate=False)
    loop_forever = Scripted([[{"type": "tool_use", "id": f"x{i}", "name": "python", "input": {"code": "print(1)"}}] for i in range(5)]
                            + [[{"type": "text", "text": "ANSWER: {}"}]])  # fmt: skip
    r = asyncio.run(run_session(loop_forever, pt, "s", "p", Limits(max_tool_calls=2), tmp_path / "t.jsonl", {}))
    assert r["tool_calls"] == 2 and r["stop"] == "end_turn"

    class Boom:
        def step(self, *a):
            raise RuntimeError("upstream failure")

    r2 = asyncio.run(run_session(Boom(), pt, "s", "p", Limits(), tmp_path / "t2.jsonl", {}))
    assert r2["stop"] == "error" and "upstream failure" in r2["error"]
    assert json.loads((tmp_path / "t2.jsonl").read_text().splitlines()[-1])["is_error"] is True


def test_signalmind_waveform_image_reaches_model_as_png(tmp_path):
    import base64

    work = tmp_path / "work"
    work.mkdir()
    np.save(work / "signal.npy", np.sin(np.arange(1000) * 0.1).astype(np.float32))

    async def go():
        tools = await SignalMindTools(work, tmp_path / "rt", [], "log", None, True, "image-test").open()
        try:
            loaded, failed = await tools.call("ts_load", {"source": str(work / "signal.npy"), "fs": 100})
            assert not failed
            sid = json.loads(loaded[0]["text"])["id"]
            blocks, failed = await tools.call("ts_zoom", {"id": sid, "window": [0, 5]})
            assert not failed
            return next(b for b in blocks if b["type"] == "image")
        finally:
            await tools.close()

    block = asyncio.run(go())
    assert block["source"]["media_type"] == "image/png"
    assert base64.b64decode(block["source"]["data"]).startswith(b"\x89PNG\r\n\x1a\n")


def test_no_key_material_in_repository_files():
    root = Path(__file__).resolve().parents[1]
    for f in list((root / "eval" / "unified").glob("*.py")) + [root / "eval" / "run_unified.py"]:
        txt = f.read_text()
        assert (
            "sk-" not in txt.replace('"sk-" not in', "")
            and os.environ.get("ANTHROPIC_AUTH_TOKEN", "zz-none") not in txt
        )


def test_python_tools_does_not_inherit_driver_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFIED_TEST_CREDENTIAL", "sentinel-for-test-only")
    work = tmp_path / "work"
    work.mkdir()
    pt = PythonTools(work, vision=False, isolate=False)
    blocks, failed = asyncio.run(
        pt.call("python", {"code": "import os; print('UNIFIED_TEST_CREDENTIAL' in os.environ)"})
    )
    assert not failed and blocks[0]["text"].strip() == "False"


def test_budget_stops_before_next_model_request(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    pt = PythonTools(work, vision=False, isolate=False)
    driver = Scripted(
        [[{"type": "tool_use", "id": f"t{i}", "name": "python", "input": {"code": "print(1)"}}] for i in range(3)]
    )
    result = asyncio.run(
        run_session(
            driver, pt, "s", "p", Limits(), tmp_path / "t.jsonl", {}, budget_exhausted=lambda u: u["output_tokens"] >= 2
        )
    )
    assert result["stop"] == "estimated_cost_budget" and result["turns"] == 2
    assert len(driver.steps) == 1


def test_campaign_budget_includes_other_arms_but_not_dry_runs(tmp_path, monkeypatch):
    import eval.run_unified as runner

    monkeypatch.setattr(runner, "RUNS", tmp_path)
    for arm, suffix, cost in (("DS-Base", "", 1.0), ("H-Agent", "", 2.0), ("DS-Agent", "-dry", 99.0)):
        d = tmp_path / f"p9level2-U-{arm}-pilot{suffix}"
        d.mkdir()
        (d / "results.jsonl").write_text(json.dumps({"cost_usd_actual": None, "cost_usd_conservative": cost}) + "\n")
    assert spent(campaign_dirs("p9level2", "pilot")) == 3.0


def test_pinned_check_accepts_runner_and_package_in_same_detached_tree(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "fixture")
    with pytest.raises(SystemExit, match="detached"):
        assert_pinned_checkout(tmp_path, tmp_path)
    git("checkout", "--detach")
    assert assert_pinned_checkout(tmp_path, tmp_path) == git("rev-parse", "HEAD")
    with pytest.raises(SystemExit, match="same pinned"):
        assert_pinned_checkout(tmp_path, tmp_path / "elsewhere")
    (tmp_path / "dirty").write_text("uncommitted")
    with pytest.raises(SystemExit, match="clean"):
        assert_pinned_checkout(tmp_path, tmp_path)


def test_log_abstention_and_invalidated_answer_extraction(tmp_path):
    from tsmcp.runtime.controller import SubmissionGate
    from tsmcp.runtime.ledger import EvidenceLedger
    from tsmcp.runtime.taskspec import TaskSpec

    gate = SubmissionGate(TaskSpec.minimal(), EvidenceLedger("test"), mode="log")
    gate.submit(
        {
            "claims": [
                {
                    "claim_id": "x",
                    "quantity": "gain",
                    "status": "insufficient_evidence",
                    "reason": "flow channel absent",
                }
            ]
        }
    )
    f = tmp_path / "answer.json"
    f.write_text(json.dumps(gate.state()))
    answer, _, finalization = application_answer(tmp_path)
    assert finalization == "recorded" and answer["insufficient_evidence"] is True
    f.write_text(
        json.dumps(
            {
                "mode": "verify",
                "finalization": "accepted",
                "flags": ["unsupported_after_task_revision"],
                "answer": {"accepted_claims": [{"quantity": "gain", "value": 1.5}]},
            }
        )
    )
    answer, diagnostic, _ = application_answer(tmp_path)
    assert answer is None and diagnostic == {"gain": 1.5}


@pytest.mark.parametrize(
    ("values", "limitation_quantity"),
    [
        ({"channel": "ABP", "onset_s": 20.5}, "onset_s at sub-0.1 s precision"),
        ({"onset_s": 119.7}, "onset_s (sub-beat timing)"),
        ({"bpm": 82.7}, "pulse_rate_from_ppg_beat_table"),
    ],
)
def test_unrelated_limitations_do_not_turn_accepted_values_into_abstention(tmp_path, values, limitation_quantity):
    (tmp_path / "spec.json").write_text(json.dumps({"quantities": [{"name": q} for q in values]}))
    state = {
        "mode": "verify",
        "finalization": "partial",
        "flags": [],
        "answer": {
            "accepted_claims": [{"quantity": q, "value": value} for q, value in values.items()],
            "limitations": [
                {
                    "quantity": limitation_quantity,
                    "status": "insufficient_evidence",
                    "reason": "This additional estimate is unavailable.",
                }
            ],
        },
    }
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    answer, diagnostic, finalization = application_answer(tmp_path)
    assert answer == diagnostic == values
    assert finalization == "partial" and path.read_bytes() == before


@pytest.mark.parametrize("source", ["limitations", "unresolved", "submitted_claims"])
def test_required_quantity_abstention_survives_extraction(tmp_path, source):
    log = source == "submitted_claims"
    refusal = {
        "quantity": "gain",
        "value": None,
        "reason": "Pressure is constant.",
        "status": "submitted_unverified" if log else "insufficient_evidence",
    }
    if log:
        refusal["submitted_status"] = "insufficient_evidence"
    (tmp_path / "answer.json").write_text(
        json.dumps(
            {
                "mode": "log" if log else "verify",
                "finalization": "recorded" if log else "partial",
                "answer": {source: [refusal]},
            }
        )
    )
    answer, diagnostic, status = application_answer(tmp_path, required_quantities={"gain"})
    assert answer == diagnostic == {"insufficient_evidence": True, "reason": "Pressure is constant."}
    assert status == ("recorded" if log else "partial")


def test_optional_limitations_and_missing_required_values_are_not_invented(tmp_path):
    (tmp_path / "spec.json").write_text(
        json.dumps(
            {
                "quantities": [
                    {"name": "channel"},
                    {"name": "onset_s"},
                    {"name": "detail", "required": False},
                ]
            }
        )
    )
    (tmp_path / "answer.json").write_text(
        json.dumps(
            {
                "mode": "verify",
                "finalization": "partial",
                "answer": {
                    "accepted_claims": [
                        {"quantity": "channel", "value": "ABP"},
                        {"quantity": "onset_s", "value": None},
                    ],
                    "limitations": [{"quantity": "detail", "status": "insufficient_evidence", "reason": "unavailable"}],
                    "text": 'ANSWER: {"channel": "ABP", "onset_s": 20.5}',
                },
            }
        )
    )
    answer, diagnostic, status = application_answer(tmp_path)
    assert answer == diagnostic == {"channel": "ABP"} and status == "partial"


def test_rejected_values_stay_diagnostic_and_absent_values_stay_absent(tmp_path):
    state = {
        "mode": "verify",
        "finalization": "rejected",
        "answer": {
            "accepted_claims": [],
            "unresolved": [{"quantity": "onset_s", "value": 20.5, "status": "unresolved"}],
            "limitations": [
                {"quantity": "extra_precision", "status": "insufficient_evidence", "reason": "unavailable"}
            ],
            "text": 'ANSWER: {"onset_s": 20.5}',
        },
    }
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(state))
    assert application_answer(tmp_path, required_quantities={"onset_s"}) == (None, {"onset_s": 20.5}, "rejected")
    state["finalization"] = "incomplete"
    state["answer"]["unresolved"] = []
    path.write_text(json.dumps(state))
    assert application_answer(tmp_path, required_quantities={"onset_s"}) == (None, None, "incomplete")
