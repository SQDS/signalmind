"""Through the real MCP layer: defaults unchanged; log and verify modes; submission round trips; access limits."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

sys.path.insert(0, str(Path(__file__).parent))
from synth_abp import synth_abp  # noqa: E402

from tsmcp.core.store import SignalStore  # noqa: E402
from tsmcp.server.app import INSTRUCTIONS, build_server  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "golden" / "default_tool_surface.json").read_text())
SBP = "abp.sbp.beat_mean.v1"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def record(tmp_path):
    abp, ecg, beats = synth_abp(fs=125, duration_s=60, seed=2)
    p = tmp_path / "sig.npy"
    np.save(p, np.stack([ecg, abp], 1).astype(np.float32))
    return {"path": str(p), "dir": tmp_path, "beats": beats}


async def session(server, calls):
    out = []
    async with Client(server) as c:
        out.append(await c.list_tools())
        for name, args in calls:
            out.append(await c.call_tool(name, args))
    return out


def text(result):
    return json.loads(result.content[0].text)


def test_default_server_surface_is_byte_identical_to_baseline():
    tools = run(session(build_server(SignalStore()), []))[0].tools
    body = [
        {"name": t.name, "description": t.description, "schema": t.input_schema}
        for t in sorted(tools, key=lambda t: t.name)
    ]
    assert hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() == GOLDEN["tools_sha256"]
    assert hashlib.sha256(INSTRUCTIONS.encode()).hexdigest() == GOLDEN["instructions_sha256"]
    assert sorted(t.name for t in tools) == [
        "ts_describe",
        "ts_find_events",
        "ts_forget",
        "ts_load",
        "ts_measure",
        "ts_zoom",
    ]


def test_default_replies_carry_no_runtime_fields(record):
    st = SignalStore(allow_roots=[record["dir"]])
    _, loaded = run(
        session(build_server(st), [("ts_load", {"source": record["path"], "fs": 125, "channels": ["ECG", "ABP"]})])
    )
    sid = text(loaded)["id"]
    _, meas, zoom, bad = run(
        session(
            build_server(st),
            [
                ("ts_measure", {"id": sid, "what": ["mean"], "channels": ["ABP"]}),
                ("ts_zoom", {"id": sid, "window": [0, 10]}),
                ("ts_measure", {"id": sid, "what": ["abp_sbp"], "channels": ["ABP"]}),
            ],
        )
    )
    assert "evidence_ids" not in text(meas) and "evidence_id" not in text(zoom)
    assert bad.is_error and "unknown measure" in bad.content[0].text  # no pressure measures without the plugin


def test_log_mode_records_without_feedback(record):
    """C_tools contrast: same tools and submission format, no semantic check, values kept exactly as submitted."""
    st = SignalStore(allow_roots=[record["dir"]])
    srv = build_server(st, plugins=["pressure"], runtime="log", runtime_dir=str(record["dir"] / "rt"), run_id="log1")
    lst, loaded = run(
        session(
            srv,
            [("ts_load", {"source": record["path"], "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]})],
        )
    )
    names = {t.name for t in lst.tools}
    assert {"ts_submit", "ts_derive", "ts_artifact", "ts_evidence"} <= names
    assert "abp_sbp" in next(t.description for t in lst.tools if t.name == "ts_measure")
    sid = text(loaded)["id"]
    state0 = json.loads((record["dir"] / "rt" / "answer.json").read_text())
    assert state0["finalization"] == "incomplete" and state0["answer"] is None and state0["mode"] == "log"
    _, meas, zoom, ev = run(
        session(
            srv,
            [
                ("ts_measure", {"id": sid, "what": ["abp_sbp", "mean"], "window": [10, 40], "channels": ["ABP"]}),
                ("ts_zoom", {"id": sid, "window": [10, 40], "channels": ["ABP"]}),
                ("ts_find_events", {"id": sid, "types": ["change"], "channels": ["ABP"]}),
            ],
        )
    )
    m = text(meas)
    assert m["evidence_ids"] == ["ev1", "ev2"] and m["measurements"][0]["artifact_ref"].startswith("art-")
    assert "_artifact" not in json.dumps(m) and len(meas.content[0].text) < 6000
    assert text(zoom)["evidence_kind"] == "visual_observation" and text(ev)["evidence_ids"] == ["ev4"]
    lines = (record["dir"] / "rt" / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[2])["source_kind"] == "visual_observation"
    # a wrong, unsupported claim is recorded verbatim: no filtering, no correction, no feedback
    wrong = {"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": 999.0,
             "unit": "mmHg", "interval": [0, 60], "evidence_refs": ["ev77"], "definition_id": SBP}  # fmt: skip
    _, s1, bad, s2 = run(
        session(
            srv,
            [
                ("ts_submit", {"bundle": {"claims": [wrong]}}),
                ("ts_submit", {"bundle": {"claims": "nonsense"}}),
                ("ts_submit", {"bundle": {"claims": [dict(wrong, value=998.0)]}}),
            ],
        )
    )
    r1 = text(s1)
    assert r1["decision"] == "recorded" and r1["deficiencies"] == [] and r1["version"] == 1
    assert r1["answer"]["submitted_claims"][0]["value"] == 999.0 and r1["answer"]["accepted_claims"] == []
    assert r1["answer"]["submitted_claims"][0]["status"] == "submitted_unverified"
    rb = text(bad)  # syntax errors are reported in both modes
    assert rb["decision"] == "retry" and rb["deficiencies"][0]["code"] == "malformed_bundle"
    r2 = text(s2)
    assert r2["decision"] == "recorded" and r2["version"] == 2
    state = json.loads((record["dir"] / "rt" / "answer.json").read_text())
    assert state["finalization"] == "recorded" and state["submissions"] == 3 and len(state["history"]) == 2
    assert state["answer"]["submitted_claims"][0]["value"] == 998.0
    bundles = json.loads((record["dir"] / "rt" / "bundles.json").read_text())
    assert [b["report"].get("ok") for b in bundles] == [None, False, None]  # every submission preserved


def verify_server(record, task_spec=None, run_id="v1"):
    st = SignalStore(allow_roots=[record["dir"]])
    return build_server(st, plugins=["pressure"], runtime="verify", runtime_dir=str(record["dir"] / run_id),
                        task_spec=task_spec, run_id=run_id)  # fmt: skip


def claim(m, **over):
    c = {"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": m["value"],
         "unit": m["unit"], "interval": [10, 40], "evidence_refs": [m["evidence_id"]], "definition_id": SBP}  # fmt: skip
    c.update(over)
    return c


def test_verify_mode_round_trip_and_repair(record):
    srv = verify_server(record)
    lst, loaded = run(
        session(
            srv,
            [("ts_load", {"source": record["path"], "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]})],
        )
    )
    names = {t.name for t in lst.tools}
    assert {"ts_submit", "ts_derive", "ts_artifact", "ts_evidence"} <= names and "Bash" not in names
    sid = text(loaded)["id"]
    short_claim = {"claim_id": "c1", "quantity": "sbp_mean", "status": "validated_measurement", "value": 120.0,
                   "unit": "mmHg", "interval": [10, 40], "evidence_refs": ["ev1"], "definition_id": SBP}  # fmt: skip
    _, short, s1, full = run(
        session(
            srv,
            [
                ("ts_measure", {"id": sid, "what": ["abp_sbp"], "window": [10, 20], "channels": ["ABP"]}),
                ("ts_submit", {"bundle": {"claims": [short_claim]}}),
                ("ts_measure", {"id": sid, "what": ["abp_sbp"], "window": [10, 40], "channels": ["ABP"]}),
            ],
        )
    )
    r1 = text(s1)
    assert r1["decision"] == "retry" and [d["code"] for d in r1["deficiencies"]] == ["scope_mismatch"]
    assert r1["remaining"]["repairs_per_issue"] == {"c1:scope_mismatch": 1}
    m = text(full)["measurements"][0]
    _, s2 = run(session(srv, [("ts_submit", {"bundle": {"claims": [claim(m)]}})]))
    r2 = text(s2)
    assert r2["decision"] == "accepted" and r2["answer"]["accepted_claims"][0]["evidence"] == [m["evidence_id"]]
    ans = json.loads((record["dir"] / "v1" / "answer.json").read_text())
    assert ans["finalization"] == "accepted" and ans["version"] == 1 and str(m["value"]) in ans["answer"]["text"]
    bundles = json.loads((record["dir"] / "v1" / "bundles.json").read_text())
    assert len(bundles) == 2 and bundles[0]["report"]["ok"] is False and bundles[1]["report"]["ok"] is True
    # a later identical revision is accepted as version 2 (explicit), not silently ignored
    _, again = run(session(srv, [("ts_submit", {"bundle": {"claims": [claim(m)]}})]))
    assert text(again)["decision"] == "accepted" and text(again)["version"] == 2


def test_verify_mode_rejections(record):
    srv = verify_server(record, run_id="v2")
    _, loaded = run(
        session(
            srv,
            [("ts_load", {"source": record["path"], "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]})],
        )
    )
    sid = text(loaded)["id"]
    _, meas, zoom, amp = run(
        session(
            srv,
            [
                ("ts_measure", {"id": sid, "what": ["abp_sbp"], "window": [10, 40], "channels": ["ABP"]}),
                ("ts_zoom", {"id": sid, "window": [10, 40], "channels": ["ABP"]}),
                ("ts_measure", {"id": sid, "what": ["amplitude_pp"], "window": [10, 40], "channels": ["ABP"]}),
            ],
        )
    )
    m, z, a = text(meas)["measurements"][0], text(zoom), text(amp)["measurements"][0]
    cases = {
        "invalid_reference": claim(m, evidence_refs=["ev77"]),
        "inadequate_support": claim(m, evidence_refs=[z["evidence_id"]]),  # visual as exact
        "method_mismatch": claim(m, evidence_refs=[a["evidence_id"]]),  # whole-window p-p as SBP
        "arithmetic_mismatch": claim(m, value=m["value"] + 7),
        "unknown_or_incompatible_unit": claim(m, unit="kPa"),
    }
    for code, c in cases.items():
        srv_k = verify_server(record, run_id=f"v2-{code}")
        _, ld = run(
            session(
                srv_k,
                [
                    (
                        "ts_load",
                        {"source": record["path"], "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]},
                    )
                ],
            )
        )
        sid_k = text(ld)["id"]
        _, mk, zk, ak, sub = run(
            session(
                srv_k,
                [
                    ("ts_measure", {"id": sid_k, "what": ["abp_sbp"], "window": [10, 40], "channels": ["ABP"]}),
                    ("ts_zoom", {"id": sid_k, "window": [10, 40], "channels": ["ABP"]}),
                    ("ts_measure", {"id": sid_k, "what": ["amplitude_pp"], "window": [10, 40], "channels": ["ABP"]}),
                    ("ts_submit", {"bundle": {"claims": [c]}}),
                ],
            )
        )
        assert [d["code"] for d in text(sub)["deficiencies"]] == [code], code


def test_derive_and_artifact_tools(record):
    srv = verify_server(record, run_id="v3")
    _, loaded = run(
        session(
            srv,
            [("ts_load", {"source": record["path"], "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]})],
        )
    )
    sid = text(loaded)["id"]
    _, tbl = run(
        session(srv, [("ts_measure", {"id": sid, "what": ["abp_beats"], "window": [10, 40], "channels": ["ABP"]})])
    )
    t = text(tbl)["measurements"][0]
    _, rows, rej, der, bad = run(
        session(
            srv,
            [
                ("ts_artifact", {"ref": t["artifact_ref"], "limit": 5}),
                ("ts_artifact", {"ref": t["artifact_ref"], "rejected_only": True}),
                (
                    "ts_derive",
                    {"op": "aggregate", "inputs": [t["evidence_id"]], "column": "sbp", "rule": "time_weighted"},
                ),
                ("ts_derive", {"op": "exec", "inputs": [t["evidence_id"]]}),
            ],
        )
    )
    assert text(rows)["returned"] == 5 and text(rows)["summary"]["n_rejected"] == 2
    assert all(r["reject_reason"] == "incomplete_boundary" for r in text(rej)["rows"])
    e = text(der)["evidence"]
    assert (
        e["source_kind"] == "derived"
        and e["definition_id"] == "abp.sbp.time_weighted_mean.v1"
        and abs(e["value"] - 120) < 1.5
    )
    assert bad.is_error and "unknown derivation" in bad.content[0].text
    _, sub = run(
        session(
            srv,
            [
                (
                    "ts_submit",
                    {
                        "bundle": {
                            "claims": [
                                {
                                    "claim_id": "c1",
                                    "quantity": "sbp_mean",
                                    "status": "validated_measurement",
                                    "value": round(e["value"], 2),
                                    "unit": "mmHg",
                                    "interval": [10, 40],
                                    "evidence_refs": [e["evidence_id"]],
                                    "definition_id": "abp.sbp.time_weighted_mean.v1",
                                    "derived": {"op": "aggregate"},
                                }
                            ]
                        }
                    },
                )
            ],
        )
    )
    assert text(sub)["decision"] == "accepted"


def test_task_spec_file_drives_required_quantities(record):
    from tsmcp.runtime.taskspec import abp_interval_spec

    spec = abp_interval_spec("t", "SBP and DBP 10-40 s", "ABP", (10, 40), ecg_channel="ECG")
    f = record["dir"] / "spec.json"
    f.write_text(json.dumps(spec.to_dict()))
    srv = verify_server(record, task_spec=str(f), run_id="v4")
    _, loaded = run(
        session(
            srv,
            [("ts_load", {"source": record["path"], "fs": 125, "channels": ["ECG", "ABP"], "units": ["mV", "mmHg"]})],
        )
    )
    sid = text(loaded)["id"]
    _, meas, s1 = run(
        session(
            srv,
            [
                ("ts_measure", {"id": sid, "what": ["abp_sbp"], "window": [10, 40], "channels": ["ABP"]}),
                ("ts_submit", {"bundle": {"claims": []}}),
            ],
        )
    )
    r = text(s1)
    assert r["decision"] == "retry" and {d["message"].split("'")[1] for d in r["deficiencies"]} == {
        "sbp_mean",
        "dbp_mean",
    }
    m = text(meas)["measurements"][0]
    _, s2 = run(
        session(
            srv,
            [
                (
                    "ts_submit",
                    {
                        "bundle": {
                            "claims": [claim(m)],
                            "limitations": [{"quantity": "dbp_mean", "reason": "not measured in this run"}],
                        }
                    },
                )
            ],
        )
    )
    assert text(s2)["decision"] == "accepted_partial" and text(s2)["finalization"] == "partial"


def test_new_configurations_keep_access_limits(record, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.npy"
    np.save(outside, np.zeros(1000, dtype=np.float32))
    try:
        srv = verify_server(record, run_id="v5")
        lst, bad = run(session(srv, [("ts_load", {"source": str(outside), "fs": 10})]))
        assert bad.is_error and "allowed roots" in bad.content[0].text
        assert not any(t.name in ("Bash", "python", "exec") for t in lst.tools)
    finally:
        outside.unlink(missing_ok=True)
