import csv
from argparse import Namespace

import numpy as np
import pytest

from tsmcp.cleaning.common import read, sha, write
from tsmcp.cleaning.dataset_entry import dataset_cli
from tsmcp.cleaning.dataset_evidence import acquisition
from tsmcp.cleaning.dataset_pipeline import Record


def test_dataset_entry_resume_materializes_same_outputs(tmp_path, monkeypatch):
    data = tmp_path / "input"
    data.mkdir()
    t = np.arange(1250) / 125
    for i in range(2):
        np.savez(
            data / f"p{i}.npz",
            time=t,
            PPG=np.sin(t * 7 + i),
            ABP=90 + 22 * np.sin(t * 7 + i),
            valid=np.ones(len(t), bool),
        )
    spec = tmp_path / "spec.json"
    write(
        spec,
        dict(
            dataset=dict(
                format="npz",
                path="input",
                mapping=dict(
                    time_key="time",
                    fs=125,
                    channels=[
                        dict(task="PPG", key="PPG", unit="source_units"),
                        dict(task="ABP", key="ABP", unit="mmHg"),
                    ],
                ),
            ),
            config=dict(method="dataset_rules", tasks=["PPG", "ABP"], purpose="native pulse and pressure", atlas=False),
        ),
    )
    root = tmp_path / "interrupted"
    args = Namespace(spec=spec, workspace=root, live=False, resume=False)
    run = Record.run

    def pause(self):
        result = run(self)
        (self.root / "stop.requested").touch()
        return result

    monkeypatch.setattr(Record, "run", pause)
    with pytest.raises(RuntimeError, match="Stopped"):
        dataset_cli(args)
    assert read(root / "run.json")["status"] == "stopped"
    (cp,) = list((root / "records").glob("*/checkpoint.json"))
    stamp = cp.stat().st_mtime_ns
    monkeypatch.setattr(Record, "run", run)
    args.resume = True
    result = dataset_cli(args)
    assert result["status"] == "completed" and len(result["records"]) == 2
    assert cp.stat().st_mtime_ns == stamp
    clean = tmp_path / "uninterrupted"
    result2 = dataset_cli(Namespace(spec=spec, workspace=clean, live=False, resume=False))
    assert result2 == result
    for name in ("waveform_manifest.json", "measurements.csv", "usable_measurements.csv"):
        assert sha(root / "export" / name) == sha(clean / "export" / name)
    with (root / "export/measurements.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["channel"] for r in rows} == {"PPG", "ABP"}
    assert all(float(r["end_s"]) > float(r["start_s"]) for r in rows)
    image = next(cp.parent.glob("views/*.png"))
    image.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint"):
        dataset_cli(args)


def test_acquisition_does_not_join_patient_or_unsupported_clock():
    case = {"id": "a", "record_id": "p", "rows": [0], "has_source_time": True}
    assert acquisition(case) != acquisition(dict(case, record_id="q"))
    assert acquisition(case) == acquisition(dict(case, id="b"))
    assert acquisition(dict(case, has_source_time=False)) != acquisition(dict(case, id="b", has_source_time=False))


def test_group_review_replays_only_bound_responses(tmp_path):
    from types import SimpleNamespace

    from tsmcp.cleaning.dataset_review import Reviewer

    cfg = {"method": "dataset_agent", "model": "deepseek-flash"}
    root = tmp_path / "live"
    root.mkdir()
    image = root / "wave.png"
    image.write_bytes(b"fixture-evidence")
    rows = [{"id": "coarse-block", "images": [image], "data": {"available": {"identity": "hash"}}}]
    reviewer = Reviewer(root, cfg)
    block = {
        "type": "tool_use",
        "id": "unit",
        "name": "submit",
        "input": {"items": [{"id": "coarse-block", "method": "identity", "reason": "fixture kept"}]},
    }
    reviewer.client = SimpleNamespace(
        request=lambda *args, **kwargs: SimpleNamespace(
            id="unit-receipt", content=[SimpleNamespace(model_dump=lambda **kwargs: block)]
        )
    )
    live = reviewer.decide("coarse", rows)
    replay = Reviewer(tmp_path / "new", dict(cfg, replay_from=str(root)))
    assert replay.decide("coarse", rows) == live
    entry = next((replay.root / "dataset-receipts").glob("*.json"))
    rec = read(entry)
    rec["decision"]["items"][0]["reason"] = "modified after review"
    write(entry, rec)
    with pytest.raises(ValueError, match="evidence"):
        replay.decide("coarse", rows)
    image.write_bytes(b"new evidence")
    with pytest.raises(FileNotFoundError):
        replay.decide("coarse", rows)


def test_jev_uses_every_hit_and_cached_calls_cost_no_additional_tokens(tmp_path, monkeypatch):
    import io
    import json
    import urllib.request

    from tsmcp.cleaning.dataset_evidence import jev_routes

    rows = [
        dict(id=f"h{i}", task="PPG", reason="rate_high", features={"peer_count": 2}, peer_bpm={"ABP": 72}, value=110)
        for i in range(25)
    ]
    calls = []

    def respond(req, timeout):
        data = json.loads(req.data)
        calls.append(data)
        return io.BytesIO(
            json.dumps(
                dict(
                    model="jev-1.14.0",
                    answers={k: dict(probability=0.7) for k in data["questions"]},
                    usage=dict(input_tokens=100, output_tokens=10),
                )
            ).encode()
        )

    monkeypatch.setenv("WULUSAI_API_KEY", "fixture")
    from types import SimpleNamespace

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: SimpleNamespace(open=respond))
    config = {"jev_token_cap": 25000}
    routes = jev_routes(tmp_path, config, rows)
    assert len(calls) == 2 and routes == {h["id"]: 0.7 for h in rows}
    assert jev_routes(tmp_path, config, rows) == routes and len(calls) == 2
    assert read(tmp_path / "jev_usage.json")["reported_tokens"] == 220


def test_workbench_dataset_mode_shares_local_contract(tmp_path):
    from pathlib import Path

    from tsmcp.workbench.jobs import Jobs
    from tsmcp.workbench.reader import Reader
    from tsmcp.workbench.runner import execute_run
    from tsmcp.workbench.store import Store

    fs = 125
    data = tmp_path / "record.npz"
    t = np.arange(750) / fs
    np.savez(data, PPG=np.sin(t * 6.1), time=t)
    store = Store(Path.cwd(), tmp_path / "workbench")
    dataset = store.save_dataset(
        dict(
            name="one",
            path=str(data),
            format="npz",
            mapping=dict(time_key="time", fs=fs, channels=[dict(task="PPG", key="PPG", unit="source_units")]),
        )
    )
    job = Jobs(store).create(dict(dataset_id=dataset["id"], method="dataset_rules", tasks=["PPG", "ABP"], atlas=False))
    root = store.run_path(job["id"])
    result = execute_run(root)
    assert result["status"] == "partial"  # Missing ABP is accounted for; PPG is still delivered.
    report = read(root / "export/evaluation_report.json")
    assert report["unavailable_tasks"][0]["task"] == "ABP" and report["scope_cases"] == 1
    reader = Reader(store)
    (case,) = read(root / "dataset_manifest.json")["cases"]
    _, artifact = reader.chosen(job["id"], case["id"])
    assert artifact["content_hash"] == read(root / "export/waveform_manifest.json")[case["id"]]["content_hash"]


def test_coarse_interpolation_next_to_observed_ecg_peak_goes_back_to_review(tmp_path):
    from types import SimpleNamespace

    case = {"id": "ecg", "task": "ECG", "unit": "source_units", "fs": 125, "record_id": "r", "n": 120}
    raw = {"content_hash": "unchanged", "stats": {"estimated_samples": 0}}
    repaired = {"content_hash": "replacement", "stats": {"estimated_samples": 4}}
    bank = {"identity": {"ecg": raw}, "rule_adaptive_dsp": {"ecg": raw}, "fixed_dsp": {"ecg": repaired}}

    def decide(stage, rows):
        assert rows == []
        return {}

    record = Record(
        tmp_path,
        [case],
        bank,
        {"write_views": 10, "purpose": "RR"},
        SimpleNamespace(enabled=True, decide=decide),
        "test",
    )

    def load(arts):
        def data(cid):
            estimated = np.zeros(120, bool)
            if arts[cid]["content_hash"] == "replacement":
                estimated[48:52] = True
            return np.arange(120) / 125, np.zeros(120), np.ones(120, bool), estimated, {"events": [{"sample": 50}]}

        return data

    record.loader = load
    assert record.coarse() == bank["identity"]
    assert read(record.path / "coarse-protected.json") == [{"case": "ecg", "a": 48, "b": 52, "method": "fixed_dsp"}]


def test_numeric_coarse_candidate_requires_independent_gate(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from tsmcp.cleaning.dataset_review import Reviewer

    fs = 125
    time = np.arange(1250) / fs
    pulse = 45 + 10 * np.sin(2 * np.pi * 1.2 * time)
    values = pulse + 2 * np.sin(2 * np.pi * 50 * time)
    path = tmp_path / "noisy.npz"
    np.savez(path, time=time, signal=values)
    spec = tmp_path / "numeric.json"
    write(
        spec,
        dict(
            dataset=dict(
                format="npz",
                path="noisy.npz",
                mapping=dict(time_key="time", fs=fs, channels=[dict(task="PPG", key="signal", unit="source_units")]),
            ),
            config=dict(method="dataset_agent", tasks=["PPG"], api_cap=0.5, atlas=False, write_views=10),
        ),
    )

    @contextmanager
    def offline(self):
        yield self

    monkeypatch.setattr(Reviewer, "session", offline)
    # These fixture decisions exercise both selection and release; the actual DSP and native probe run normally.
    answer = "rollback"

    def decide(self, stage, rows):
        replies = {}
        for z in rows:
            decision = {
                "coarse": {"method": "fixed_dsp"},
                "gate": {"decision": answer},
                "groups": dict(wave="preserve", wave_if=[], measure="keep", measure_if=[], otherwise="keep"),
                "joint": dict(default="keep", exceptions=[]),
            }[stage]
            k = z["id"]
            binding = f"{stage}-{k}"
            row = dict(id=k, receipt=binding, request_id="fixture", reason="fixture", **decision)
            write(self.root / "dataset-receipts" / (binding + ".json"), row)
            replies[k] = row
        return replies

    monkeypatch.setattr(Reviewer, "decide", decide)
    for answer in ("rollback", "accept"):
        root = tmp_path / answer
        report = dataset_cli(Namespace(spec=spec, workspace=root, resume=False, live=True))
        (item,) = read(root / "export/waveform_manifest.json").values()
        assert report["status"] == "completed"
        assert (item["modified_samples"] > 0) == (answer == "accept")
        with np.load(root / item["waveform"], allow_pickle=False) as z:
            if answer == "accept":
                assert np.mean((z["values"] - pulse) ** 2) < np.mean((values - pulse) ** 2) / 2
            else:
                np.testing.assert_array_equal(z["values"], values)


def test_grouped_submission_repairs_missing_ids_without_losing_rows(tmp_path):
    from types import SimpleNamespace

    from tsmcp.cleaning.dataset_review import Reviewer

    cfg = {"method": "dataset_agent", "model": "deepseek-flash"}
    root = tmp_path / "live"
    root.mkdir()
    img = root / "card.png"
    img.write_bytes(b"bound-evidence")
    rows = [dict(id=f"group-{i}", data={"available": ["identity"]}, images=[img]) for i in range(3)]
    reviewer = Reviewer(root, cfg)
    calls = []

    def request(messages, tools, system):
        fields = tools[0]["input_schema"]["properties"]
        calls.append(set(fields))
        if "items" in fields:
            payload = {"items": [{"id": rows[0]["id"], "method": "identity", "reason": "fixture"}]}
        else:
            payload = {k: {"method": "identity", "reason": "all three assessed"} for k in fields}
        value = {"type": "tool_use", "id": "fixture", "name": "submit", "input": payload}
        return SimpleNamespace(id="mock-" + str(len(calls)), content=[SimpleNamespace(model_dump=lambda **kw: value)])

    reviewer.client = SimpleNamespace(request=request)
    result = reviewer.decide("coarse", rows)
    assert calls == [{"items"}, {"row1", "row2", "row3"}]
    assert set(result) == {z["id"] for z in rows}
    assert all(r["request_id"] == "mock-2" for r in result.values())
    assert Reviewer(tmp_path / "replay", dict(cfg, replay_from=str(root))).decide("coarse", rows) == result


def test_group_input_and_unseen_joint_preserve_other_channel_decision(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from tsmcp.cleaning import dataset_pipeline
    from tsmcp.cleaning.dataset_review import Reviewer

    t = np.arange(1875) / 125
    signal = 10 + 2 * np.sin(t * 7)
    ecg = np.exp(-(((t % 0.8) - 0.4) / 0.025) ** 2)
    np.savez(tmp_path / "patient.npz", time=t, ECG=ecg, ABP=signal + 80, PPG=signal + 30)
    channels = [dict(task=k, key=k, unit="mmHg" if k == "ABP" else "source_units") for k in ("ABP", "PPG", "ECG")]
    spec = tmp_path / "spec.json"
    write(
        spec,
        dict(
            dataset=dict(format="npz", path="patient.npz", mapping=dict(time_key="time", fs=125, channels=channels)),
            config=dict(method="dataset_agent", api_cap=0.5, joint_views=0, atlas=False),
        ),
    )
    monkeypatch.setattr(Reviewer, "session", lambda self: nullcontext(self))
    seen = {}

    def discover(workspace, cases, *args):
        return [
            dict(
                id=c["task"] + str(a),
                case=c["id"],
                task=c["task"],
                a=a,
                b=a + 200,
                reason="probe_disagreement",
                value=float(a),
            )
            for c in cases
            if c["task"] != "ECG"
            for a in ([375, 750, 1000] if c["task"] == "PPG" else [375])
        ]

    def decide(self, stage, rows):
        outputs = {}
        for row in rows:
            key, data = row["id"], row["data"]
            if stage == "groups":
                seen[data["task"]] = data
                assert len(row["images"]) == len(set(sha(p) for p in row["images"])) == 2
                fields = dict(
                    wave="preserve",
                    wave_if=[],
                    measure="review" if data["task"] == "ABP" else "keep",
                    measure_if=[],
                    otherwise="keep",
                )
            elif stage == "joint":
                fields = dict(default="keep", exceptions=[])
            elif stage == "coarse":
                fields = dict(method="identity")
            else:
                raise AssertionError(stage)
            outputs[key] = dict(id=key, reason="fixture", request_id=key, receipt=key, **fields)
            write(self.root / "dataset-receipts" / (key + ".json"), outputs[key])
        return outputs

    monkeypatch.setattr(dataset_pipeline, "discover", discover)
    monkeypatch.setattr(Reviewer, "decide", decide)
    root = tmp_path / "run"
    result = dataset_cli(Namespace(spec=spec, workspace=root, resume=False, live=True))
    assert result["status"] == "completed"
    assert seen["PPG"]["scope"] == len(seen["PPG"]["member_rows"]) == 3
    assert [r[3] for r in seen["PPG"]["member_rows"]] == [375, 750, 1000]
    assert len(set(seen["PPG"]["image_rows"])) == 2
    assert seen["ABP"]["image_roles"] == ["typical", "detail"]
    rows = read(next(root.glob("records/*/measurements.json")))
    unchecked = [r for r in rows if r["joint_view_unchecked"] and r["channel"] == "ECG"]
    assert unchecked and all(r["quality"] == "keep" and not r["pending_evidence"] for r in unchecked)


def test_local_spike_trials_accept_and_reject_individually(tmp_path):
    import time
    from types import SimpleNamespace

    from tsmcp.cleaning.engine import evaluate_output, load_arrays
    from tsmcp.cleaning.tasks import probe

    t = np.arange(2000) / 125
    clean = 45 + np.sin(2 * np.pi * t)
    y = clean.copy()
    spans = [(405, 407), (905, 907)]
    for a, b in spans:
        y[a:b] += 4
    v = np.ones(len(t), bool)
    e = ~v
    case = dict(id="x", task="PPG", record_id="r", fs=125, unit="source_units", n=len(t))
    public = tmp_path / "public/x"
    public.mkdir(parents=True)
    np.savez(public / "input.npz", time=t, values=y, valid=v)
    case["input_sha256"] = sha(public / "input.npz")
    art = evaluate_output(tmp_path, case, {"hash": "identity"}, "fixture", None, y, t, y, y, v, e, [], time.monotonic())
    hits = [
        dict(
            id=str(a),
            case="x",
            a=a,
            b=b,
            reason="isolated_spike_candidate",
            measurement="keep",
            group="impulses",
            decision_source="group-plan",
            wave_requested="interpolate_spike",
        )
        for a, b in spans
    ]

    def decide(stage, rows):
        k = rows[0]["id"]
        return {k: dict(decision="accept" if k == "spike-405" else "rollback", reason="fixture decision", receipt=k)}

    record = Record(
        tmp_path, [case], {"identity": {"x": art}}, {"write_views": 4}, SimpleNamespace(decide=decide), "fixture"
    )
    final = record.spike_trials({"x": art}, hits)
    _, _, out, valid, estimated = load_arrays(tmp_path, case, final["x"])
    trials = read(record.path / "spike-trials.json")
    assert [r["status"] for r in trials] == ["accepted", "rollback"]
    assert np.flatnonzero(estimated).tolist() == [405, 406]
    assert np.array_equal(out[~estimated], y[~estimated])
    assert np.max(np.abs(out[estimated] - clean[estimated])) < 0.005
    assert all(i["support_id"] >= 0 for i in probe(out, t, 125, valid, estimated, "PPG")["intervals"])
