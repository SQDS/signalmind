"""Contract tests for real provenance, import semantics, history, and task control."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tsmcp.cleaning.common import implementation_hash, read, sha, write
from tsmcp.cleaning.engine import execute_version, load_arrays, process_one, register_recipe
from tsmcp.cleaning.provenance import restore_steps
from tsmcp.workbench.app import create_app
from tsmcp.workbench.datasets import import_dataset, preview
from tsmcp.workbench.jobs import Jobs
from tsmcp.workbench.reader import Reader, lines
from tsmcp.workbench.runner import execute_run
from tsmcp.workbench.store import Store


def source(tmp_path, *, fs=125, task="PPG"):
    path = tmp_path / "source.npz"
    t = np.arange(fs * 12) / fs
    values = np.sin(2 * np.pi * 1.2 * t) + 0.08 * np.sin(2 * np.pi * 30 * t)
    valid = np.ones(len(t), bool)
    valid[300:320] = False
    estimated = np.zeros(len(t), bool)
    estimated[500:510] = True
    np.savez(path, time=t, signal=values, valid=valid, estimated=estimated)
    dataset = {"id": "fixture", "path": str(path), "format": "npz"}
    mapping = {
        "time_key": "time",
        "fs": fs,
        "channels": [
            {
                "task": task,
                "key": "signal",
                "unit": "mmHg" if task == "ABP" else "source_units",
                "valid_key": "valid",
                "estimated_key": "estimated",
            }
        ],
    }
    return dataset, mapping


def workspace(tmp_path):
    dataset, mapping = source(tmp_path)
    root = tmp_path / "run"
    manifest = import_dataset(dataset, mapping, root, ["PPG"])
    return root, manifest["cases"][0]


def http(app, method, path, body=None, origin=None):
    async def run():
        sent = []
        body_bytes = json.dumps(body).encode() if body is not None else b""
        headers = [(b"host", b"localhost"), (b"content-type", b"application/json")]
        if origin:
            headers.append((b"origin", origin.encode()))
        target, _, query = path.partition("?")
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": target,
            "raw_path": target.encode(),
            "query_string": query.encode(),
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 1),
            "server": ("localhost", 80),
        }
        consumed = False

        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            await asyncio.sleep(30)
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return status, json.loads(data) if data else None

    return asyncio.run(run())


def test_csv_gaps_units_masks_and_unknown_patient(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("time,ECG,ok\n0,1,1\n8,2,0\n16,3,1\n40,4,1\n48,5,1\n")
    dataset = {"id": "csv", "path": str(path), "format": "csv"}
    mapping = {
        "time_key": "time",
        "time_unit": "ms",
        "fs": 125,
        "channels": [{"task": "ECG", "key": "ECG", "valid_key": "ok", "unit": "mV"}],
    }
    manifest = import_dataset(dataset, mapping, tmp_path / "run", ["ECG", "PPG"])
    assert manifest["patients"] is None
    assert [c["n"] for c in manifest["cases"]] == [3, 2]
    assert manifest["cases"][0]["unit"] == "mV"
    assert manifest["unavailable_tasks"][0]["task"] == "PPG"
    _, t, _, valid, _ = load_arrays(tmp_path / "run", manifest["cases"][0])
    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_allclose(t, [0, 0.008, 0.016])


def test_blank_csv_observation_is_missing_not_invented(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text('signal\n1\n""\n3\n')
    dataset = {"id": "csv", "path": str(path), "format": "csv"}
    root = tmp_path / "run"
    manifest = import_dataset(dataset, {"fs": 125, "channels": [{"task": "ECG", "key": "signal"}]}, root, ["ECG"])
    raw, _, _, valid, _ = load_arrays(root, manifest["cases"][0])
    assert np.isnan(raw[1]) and not valid[1] and len(raw) == 3


def test_npz_axes_and_files_do_not_fake_continuity(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    for i in range(2):
        np.savez(folder / f"{i}.npz", matrix=np.arange(40).reshape(2, 20))
    dataset = {"id": "npz", "path": str(folder), "format": "npz"}
    mapping = {
        "fs": 250,
        "patient_id": "same",
        "channels": [{"task": "ECG", "key": "matrix", "sample_axis": 1, "channel": 1}],
    }
    manifest = import_dataset(dataset, mapping, tmp_path / "run", ["ECG"])
    assert len(manifest["cases"]) == 2 and manifest["patients"] == 1
    assert not manifest["cases"][0]["has_source_time"]
    with np.load(tmp_path / "run/public" / manifest["cases"][0]["id"] / "input.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["values"], np.arange(20, 40))


@pytest.mark.parametrize("times", [[0, 0.008, 0.008], [0, 0.016, 0.008], [0, 0.008, 0.017]])
def test_bad_time_is_accounted_without_reordering(tmp_path, times):
    path = tmp_path / "bad.npz"
    np.savez(path, t=times, x=[1, 2, 3])
    ds = {"id": "bad", "path": str(path), "format": "npz"}
    m = import_dataset(
        ds, {"time_key": "t", "fs": 125, "channels": [{"task": "ECG", "key": "x"}]}, tmp_path / "run", ["ECG"]
    )
    assert not m["cases"] and m["audit_failures"]


def test_abp_units_fail_only_that_task(tmp_path):
    path = tmp_path / "units.npz"
    np.savez(path, x=np.arange(200), pressure=np.arange(200))
    ds = {"id": "units", "path": str(path), "format": "npz"}
    m = import_dataset(
        ds,
        {
            "fs": 125,
            "channels": [{"task": "ECG", "key": "x"}, {"task": "ABP", "key": "pressure", "unit": "normalized"}],
        },
        tmp_path / "run",
        ["ECG", "ABP"],
    )
    assert [c["task"] for c in m["cases"]] == ["ECG"]
    assert m["audit_failures"][0]["task"] == "ABP"


def test_source_support_is_not_counted_as_agent_edit(tmp_path):
    root, case = workspace(tmp_path)
    result = execute_version(root, register_recipe(root, kind="identity"), [case], workers=1)
    a = result["results"][case["id"]]
    assert a["stats"]["support_changed_samples"] == 0
    assert a["stats"]["estimated_samples"] == 10
    assert a["stats"]["unusable_samples"] == 20


def test_exact_operator_history_and_cache_no_double_processing(tmp_path):
    root, case = workspace(tmp_path)
    source_code = "def run(ctx):\n    return ctx.current\n"
    recipe = register_recipe(root, source=source_code)

    def run(ctx):
        ctx.apply(
            "lowpass",
            [[100, 250]],
            frequency_hz=12,
            filter_order=2,
            explanation={"problem": "fixture jitter", "threshold": 0.3},
        )
        ctx.apply("mark_invalid", [[700, 710]], explanation="fixture exclusion")
        return ctx.current

    result = process_one(root, case, recipe, implementation_hash(), module=SimpleNamespace(run=run))
    trace = read(root / result["operator_trace"]["path"])
    assert len(trace["steps"]) == 2
    _, _, raw, valid, estimated = load_arrays(root, case)
    first = restore_steps(root, result["operator_trace"], raw, valid, estimated, 0)
    last = restore_steps(root, result["operator_trace"], raw, valid, estimated, 1)
    assert first[1][700:710].all() and not last[1][700:710].any()
    with np.load(root / result["arrays"], allow_pickle=False) as output:
        for actual, key in zip(last, ("values", "valid", "estimated"), strict=True):
            np.testing.assert_array_equal(actual, output[key])
    cached = process_one(
        root,
        case,
        recipe,
        implementation_hash(),
        module=SimpleNamespace(run=lambda ctx: pytest.fail("must not reapply")),
    )
    assert cached["operator_trace"] == result["operator_trace"]


def test_invalid_source_is_saved_before_compile(tmp_path):
    with pytest.raises(SyntaxError):
        register_recipe(tmp_path, source="def run(:\n")
    assert next((tmp_path / "recipes").glob("*/program.py")).read_text() == "def run(:\n"
    assert [e["type"] for e in lines(tmp_path / "events.jsonl")] == ["recipe_submitted", "recipe_compile_failed"]


def test_legacy_reviews_and_missing_timestamps_remain_truthful(tmp_path):
    root, case = workspace(tmp_path)
    (root / "events.jsonl").unlink()  # A legacy run predates the native journal.
    reviews = [
        {
            "tool": "submit_self_review",
            "tool_use_id": str(i),
            "arguments": {"version_id": "v_a", "action": action},
            "result": [],
        }
        for i, action in enumerate(["inspect", "accept", "rollback"])
    ]
    (root / "agent_activity.jsonl").write_text("\n".join(json.dumps(r) for r in reviews) + "\n")
    store = Store(tmp_path, tmp_path / "state")
    store.add_run(root, historical=True)
    before = sha(root / "agent_activity.jsonl")
    rows = Reader(store).events(root.name)
    assert [r["arguments"]["action"] for r in rows] == ["inspect", "accept", "rollback"]
    assert all(r["timestamp"] is None for r in rows)
    assert sha(root / "agent_activity.jsonl") == before


def test_cost_and_patient_dedup_of_inherited_ledgers(tmp_path):
    store = Store(tmp_path, tmp_path / "state")
    for i in range(2):
        root = tmp_path / f"r{i}"
        write(root / "dataset_manifest.json", {"cases": [], "patients": 1, "patient_keys": ["same"], "records": []})
        write(
            root / "api_budget.json",
            {"calls": [{"request_id": "same-call", "estimated_peak_usd": 0.1, "status": "returned"}]},
        )
        store.add_run(root)
    result = Reader(store).overview()
    assert result["unique_patients"] == 1 and result["estimated_cost_usd"] == 0.1
    assert result["actual_billed_usd"] is None


def test_task_api_and_stop_resume_budget_contract(tmp_path):
    app = create_app(tmp_path, tmp_path / "state", run_jobs=False)
    dataset, mapping = source(tmp_path)
    code, ds = http(app, "POST", "/api/datasets", {**dataset, "mapping": mapping})
    assert code == 201
    code, run = http(
        app, "POST", "/api/runs", {"dataset_id": ds["id"], "tasks": ["PPG"], "method": "agent", "api_cap": 0.5}
    )
    assert code == 201
    rid = run["id"]
    code, stopped = http(app, "POST", f"/api/runs/{rid}/control/stop", {})
    assert code == 200 and stopped["status"] == "stopped"
    root = app.state.store.run_path(rid)
    write(root / "api_budget.json", {"cap_usd": 0.5, "estimated_peak_usd": 0.1, "calls": []})
    previous = sha(root / "api_budget.json")
    code, resumed = http(app, "POST", f"/api/runs/{rid}/control/resume", {})
    assert code == 200 and resumed["config"]["api_cap"] == 0.5
    assert sha(root / "api_budget.json") == previous
    assert http(app, "POST", "/api/runs", {}, origin="http://other-site.test")[0] == 403


def test_offline_job_runs_real_baselines_and_keeps_all_reports(tmp_path):
    store = Store(tmp_path, tmp_path / "state")
    ds, mapping = source(tmp_path)
    ds = store.save_dataset({**ds, "mapping": mapping})
    job = Jobs(store).create({"dataset_id": ds["id"], "tasks": ["PPG"], "method": "baselines"})
    root = store.run_path(job["id"])
    result = execute_run(root, agent_runner=lambda *a, **k: pytest.fail("No paid model calls in baseline mode"))
    assert result["status"] == "completed"
    assert set(read(root / "baselines.json")) == {"identity", "fixed_dsp", "rule_adaptive_dsp"}
    assert (root / "export/baseline_reports.json").exists()
    assert not (root / "api_budget.json").exists()


def test_resume_rejects_changed_implementation(tmp_path):
    store = Store(tmp_path, tmp_path / "state")
    ds, mapping = source(tmp_path)
    ds = store.save_dataset(ds)
    jobs = Jobs(store)
    run = jobs.create({"dataset_id": ds["id"], "method": "baselines", "mapping": mapping})
    root = store.run_path(run["id"])
    jobs.stop(run["id"])
    config = read(root / "run.json")
    config["implementation_hash"] = "changed"
    write(root / "run.json", config)
    with pytest.raises(ValueError, match="Implementation changed"):
        jobs.resume(run["id"])


def test_api_extrema_and_mask_boundaries_survive_overview(tmp_path):
    root, case = workspace(tmp_path)
    app = create_app(tmp_path, tmp_path / "state", run_jobs=False)
    app.state.store.add_run(root)
    status, wave = http(app, "GET", f"/api/runs/run/cases/{case['id']}/waveform?version=raw&points=50")
    assert status == 200
    assert 299 in wave["indices"] and 300 in wave["indices"] and 319 in wave["indices"] and 320 in wave["indices"]
    _, _, raw, _, _ = load_arrays(root, case)
    assert max(wave["raw"]) == raw.max() and min(wave["raw"]) == raw.min()
    assert wave["masks"]["agent_invalid"] == []


def test_changed_source_is_not_displayed_with_stale_result(tmp_path):
    root, case = workspace(tmp_path)
    store = Store(tmp_path, tmp_path / "state")
    store.add_run(root)
    reader = Reader(store)
    reader.waveform(root.name, case["id"])
    np.savez(
        root / "public" / case["id"] / "input.npz", values=np.zeros(case["n"]), time=np.arange(case["n"]) / case["fs"]
    )
    with pytest.raises(ValueError, match="Saved artifact changed"):
        reader.waveform(root.name, case["id"])


def test_stopped_agent_retains_pending_tool_and_resume_uses_same_request(tmp_path, monkeypatch):
    from tsmcp.cleaning import agent
    from tsmcp.cleaning.cli import baselines

    root, case = workspace(tmp_path)
    baselines(root)
    calls = []

    class FakeClient:
        def __init__(self, workspace, cap, model):
            self.ledger = {"estimated_peak_usd": 0}

        def request(self, *args):
            calls.append(True)
            content = [{"type": "tool_use", "id": "pending-one", "name": "inspect_dataset", "input": {}}]
            # Stop after the model returned but before its pending tool is executed.
            (root / "stop.requested").touch()
            return SimpleNamespace(
                id="same-model-response", content=[SimpleNamespace(model_dump=lambda **k: content[0])]
            )

        def close(self):
            pass

    monkeypatch.setattr(agent, "BudgetClient", FakeClient)
    monkeypatch.setattr(agent, "vision_preflight", lambda client: {})
    result = agent.run(root, max_turns=2)
    assert result["stop"] == "stopped"
    assert not read(root / "agent_state.json").get("finalized")
    (root / "stop.requested").unlink()
    original = agent.CleaningTools.call

    def call_and_stop(self, *args):
        output = original(self, *args)
        (root / "stop.requested").touch()
        return output

    monkeypatch.setattr(agent.CleaningTools, "call", call_and_stop)
    result = agent.run(root, max_turns=3)
    assert result["stop"] == "stopped" and len(calls) == 1
    activity = lines(root / "agent_activity.jsonl")
    assert len(activity) == 1 and activity[0]["request_id"] == "same-model-response"


def test_worker_timeout_attempts_are_retained(tmp_path, monkeypatch):
    import subprocess

    from eval import processing_tools
    from tsmcp.cleaning.sandbox import run_job

    root, case = workspace(tmp_path)
    recipe = register_recipe(root, source="def run(ctx):\n    return ctx.current\n")
    monkeypatch.setattr(processing_tools, "sandbox_command", lambda *a, **k: ["placeholder"])
    attempts = []

    def failing(command, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired(command, 600, output=b"first attempt output", stderr=b"timeout detail")
        return subprocess.CompletedProcess(command, 1, "second attempt output", "execution failure")

    monkeypatch.setattr(subprocess, "run", failing)
    results, errors = run_job(root, [case], recipe, implementation_hash(), {})
    assert not results and errors and len(attempts) == 2
    folders = list((root / "jobs").glob("*/attempts/*"))
    assert len(folders) == 2
    assert sorted((p / "stdout.txt").read_text() for p in folders) == ["first attempt output", "second attempt output"]
    assert {read(p / "execution.json").get("status", "returned") for p in folders} == {"timeout", "returned"}


def test_object_array_cannot_be_unpickled(tmp_path):
    path = tmp_path / "object.npz"
    np.savez(path, x=np.array([{"value": 1}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        preview({"path": str(path), "format": "npz"})


def test_vital_audit_stop_preserves_patient_scope_on_resume(tmp_path, monkeypatch):
    import h5py

    from tsmcp.cleaning import data
    from tsmcp.cleaning.provenance import StopRequested

    dataset = tmp_path / "mat"
    dataset.mkdir()
    records = []
    for i in range(2):
        path = dataset / f"p{i}.mat"
        with h5py.File(path, "w") as file:
            file.create_group("Subj_Wins").create_dataset("ECG_Record", data=np.arange(1250))
        records.append(
            {"subject_key": f"vital::p{i}", "split": "train", "source": {"path": str(path), "sha256": sha(path)}}
        )
    inventory = tmp_path / "inventory.json"
    write(inventory, records)
    root = tmp_path / "run"
    checks = []

    def stop_second_record(output):
        checks.append(True)
        if len(checks) == 2:
            raise StopRequested("fixture stop")

    monkeypatch.setattr(data, "check_stop", stop_second_record)
    with pytest.raises(StopRequested):
        data.audit(dataset, inventory, root, count=2, repo=tmp_path, config={"tasks": ["ECG"]})
    selection = read(root / "private/selection.json")
    saved = {p: sha(p) for p in (root / "public").glob("*/input.npz")}
    monkeypatch.setattr(data, "check_stop", lambda output: None)
    result = data.audit(
        dataset,
        inventory,
        root,
        count=2,
        repo=tmp_path,
        config={"tasks": ["ECG"]},
        excluded_patients=selection["patients"],
        resume=True,
    )
    assert len(result["cases"]) == 2
    assert read(root / "private/selection.json") == selection
    assert all(sha(p) == digest for p, digest in saved.items())


def test_inspect_only_publishes_candidates_without_model(tmp_path):
    app = create_app(tmp_path, tmp_path / "state", run_jobs=False)
    dataset, mapping = source(tmp_path)
    code, ds = http(app, "POST", "/api/datasets", {**dataset, "mapping": mapping})
    assert code == 201
    code, job = http(
        app,
        "POST",
        "/api/runs",
        {"dataset_id": ds["id"], "method": "inspect_only", "tasks": ["PPG"], "inspection_preset": "thorough"},
    )
    assert code == 201 and job["config"]["api_cap"] == 0
    root = app.state.store.run_path(job["id"])
    assert (
        execute_run(root, agent_runner=lambda *a, **k: pytest.fail("Model called during scan"))["status"] == "completed"
    )
    result = read(root / "inspection/inspection_summary.json")
    assert result["preset"] == "thorough" and result["planned_blocks"] == 1 and result["model_calls"] == 0
    assert not (root / "versions").exists()
    code, files = http(app, "GET", f"/api/runs/{job['id']}/exports")
    assert code == 200 and {"intervals.csv", "candidate_manifest.json"} <= {f["name"] for f in files}


def test_candidate_ledger_endpoint_pages_and_filters(tmp_path):
    app = create_app(tmp_path, tmp_path / "ui", run_jobs=False)
    root = tmp_path / "r"
    app.state.store.add_run(root)
    assert http(app, "GET", "/api/runs/r/quality")[1]["status"] == "pending"
    rows = [
        dict(candidate_id=f"c{i}", state=state)
        for i, state in enumerate(["preserved", "pending_budget", "pending_budget"])
    ]
    write(root / "ledger/release.json", dict(candidates=len(rows)))
    (root / "ledger/candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    status, page = http(app, "GET", "/api/runs/r/quality?state=pending_budget&offset=1&limit=1")
    assert status == 200 and page["status"] == "published" and page["items"] == [rows[2]]
