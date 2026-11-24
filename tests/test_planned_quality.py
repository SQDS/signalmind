"""Regression cases for the failure modes identified in the project's November 23 architecture review."""

from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tsmcp.cleaning.common import digest, read, sha, write
from tsmcp.cleaning.context import ContextBuilder
from tsmcp.cleaning.contracts import Patch, published_run
from tsmcp.cleaning.dataset_entry import dataset_cli
from tsmcp.cleaning.dataset_review import Reviewer
from tsmcp.cleaning.knowledge import Knowledge
from tsmcp.cleaning.ledger import commit_patch
from tsmcp.cleaning.quality_gate import spike_check, waveform_gate
from tsmcp.cleaning.rollback import rollback
from tsmcp.cleaning.tasks import probe
from tsmcp.followup.cli import load_seed
from tsmcp.inspection.scan import scan
from tsmcp.workbench.datasets import import_dataset


def test_zero_events_short_support_and_edges_are_unknown():
    fs = 125
    y = np.zeros(1000)
    t = np.arange(len(y)) / fs
    v = np.ones(len(y), bool)
    v[400:800] = False
    p = probe(y, t, fs, v, np.zeros(len(y), bool), "ECG")
    s = p["detector_support"]
    assert s["unsupported_ranges"] == [[400, 1000]] and s["boundary_uncertain_ranges"] == [[0, 20], [380, 400]]
    assert s["empty_event_ranges"] == [[20, 380]] and s["applicability"] == "insufficient_support"
    assert spike_check(dict(fs=fs, task="ECG"), y, v, ~v, p, 200, 201)["status"] == "insufficient_support"


def test_missed_recurrent_peak_protected_independently():
    fs = 125
    t = np.arange(1500)
    x = np.zeros(len(t))
    peaks = np.arange(60, 1450, 100)
    for q in peaks:
        x += np.exp(-0.5 * ((t - q) / 2) ** 2)
    a = peaks[5] - 1
    b = a + 3
    ev = [dict(sample=int(q)) for q in peaks if q != peaks[5]]
    good = np.ones(len(t), bool)
    result = spike_check(dict(fs=fs, task="ECG"), x, good, ~good, dict(events=ev), a, b)
    assert result["status"] == "possible_physiology" and result["matching_shapes"] >= 3


@pytest.mark.parametrize("modality", ["ECG", "PPG", "ABP"])
def test_local_detector_ignores_distant_gain_and_finds_biphasic_bounds(modality):
    fs = 125
    t = np.arange(70 * fs) / fs
    x = np.sin(2 * np.pi * t)
    a = round(20.25 * fs)
    x[a : a + 2] += [4, -4]

    def findings(v):
        events, coverage, _ = scan(v, np.isfinite(v), {}, fs, modality, "balanced")
        return [
            (r["start"], r["end"], r["value"], r["kind"])
            for r in events
            if a - 2 <= r["start"] <= a + 2 and r["kind"] in ("biphasic_core", "large_step")
        ]

    bigger = x.copy()
    bigger[42 * fs :] *= 200
    assert findings(x) == findings(bigger)
    assert any(left == a and right == a + 2 and k == "biphasic_core" for left, right, _, k in findings(x))


def test_explicit_storage_shards_match_single_file_and_split_at_gap(tmp_path):
    t = np.r_[np.arange(502) / 125, 9 + np.arange(498) / 125]
    y = 50 + np.sin(t * 6)

    def imported(name, parts):
        src = tmp_path / name
        src.mkdir()
        for i, p in enumerate(parts):
            np.savez(src / f"{i}.npz", t=t[p], x=y[p])
        out = tmp_path / (name + "-run")
        out.mkdir()
        mapping = dict(
            fs=125,
            time_key="t",
            channels=[dict(task="PPG", key="x", unit="u")],
            storage_shards={"patient/A": [f"{i}.npz" for i in range(len(parts))]},
        )
        return out, import_dataset(dict(id="ds", path=str(src), format="npz"), mapping, out, ["PPG"])

    one, m = imported("single", [slice(None)])
    many, n = imported("sharded", [slice(0, 130), slice(130, 881), slice(881, None)])
    assert [(c["id"], c["n"]) for c in m["cases"]] == [(c["id"], c["n"]) for c in n["cases"]]
    assert [c["n"] for c in n["cases"]] == [502, 498]
    for c in m["cases"]:
        assert sha(one / "public" / c["id"] / "input.npz") == sha(many / "public" / c["id"] / "input.npz")


def test_hard_gate_rejects_out_of_scope_writes_and_pressure_dc_loss():
    fs = 125
    t = np.arange(1000) / fs
    x = 90 + 15 * np.sin(2 * np.pi * t) + 2 * np.sin(2 * np.pi * 50 * t)
    v = np.ones(len(x), bool)
    e = ~v
    pr = probe(x, t, fs, v, e, "ABP")
    before = (t, x, v, e, pr)
    c = dict(task="ABP", fs=fs)
    y = x.copy()
    y[10] += 4
    test = waveform_gate(
        c, before, (t, y, v, e, pr), [dict(op="mark_invalid", ranges_samples=[[22, 25]], parameters={})], [[22, 25]]
    )
    assert test["status"] == "rejected" and not test["checks"]["write_ranges"]
    op = [dict(op="remove_tone", ranges_samples=[[0, len(t)]], parameters=dict(frequency_hz=50))]
    smooth = 90 + 15 * np.sin(2 * np.pi * t)
    assert waveform_gate(c, before, (t, smooth, v, e, pr), op)["status"] == "passed"
    assert waveform_gate(c, before, (t, smooth - 30, v, e, pr), op)["status"] == "rejected"


def test_all_members_accounted_for_across_bounded_requests(tmp_path):
    from PIL import Image

    im = tmp_path / "view.png"
    Image.new("RGB", (20, 15)).save(im)
    cfg = ContextBuilder(dict(max_text_bytes=10000, max_request_tokens=20000))
    data = dict(
        cases=[dict(id="c", fs=125, unit="u")],
        columns=["case", "sample", "score"],
        member_rows=[[0, i, i / 7] for i in range(1201)],
        image_rows=[0, 1200],
        scope=1201,
    )
    req = dict(id="g", data=data, images=[im, im])
    pages = cfg.group_pages(req)
    assert len(pages) > 1
    flattened = [row for page, _, _ in pages for row in page["data"]["member_rows"]]
    assert flattened == data["member_rows"] and all(cfg.fits(cfg.usage([row], "short", {})) for row, _, _ in pages)


def test_record_budget_survives_new_reviewer_and_pending_is_explicit(tmp_path):
    cfg = dict(method="dataset_agent", model="mock", max_llm_calls_per_record=1)
    first = Reviewer(tmp_path, cfg)
    first.record = "r"
    first.deadline = float("inf")
    first.debit()
    retry = Reviewer(tmp_path, cfg)
    retry.record = "r"
    retry.deadline = float("inf")
    with pytest.raises(RuntimeError, match="budget"):
        retry.debit()
    none = Reviewer(tmp_path, dict(cfg, max_images_per_call=1))
    none.record = "r"
    im = tmp_path / "x"
    im.write_bytes(b"x")
    assert none.decide("coarse", [dict(id="g", data={"available": ["identity"]}, images=[im, im])]) == {}
    assert "decision_pending" in (tmp_path / "events.jsonl").read_text()


def test_memory_namespaces_and_pinned_snapshot(tmp_path):
    lib = Knowledge(tmp_path / "lib")

    def episode(label):
        return dict(
            modality="ECG", trigger="impulse", note=label, source=digest(label), scope="record", confidence=None
        )

    lib.record("repair", episode("repair-only"))
    lib.record("review", episode("review-only"))
    old = lib.freeze(tmp_path / "run-a")
    lib.record("review", episode("new observation"))
    assert lib.freeze(tmp_path / "run-a") == old
    new = lib.freeze(tmp_path / "run-b")
    assert new["id"] != old["id"]
    assert [z["note"] for z in lib.context(new, "repair", "ECG", "impulse")["memory"]] == ["repair-only"]
    assert {z["note"] for z in lib.context(new, "review", "ECG", "impulse")["memory"]} == {
        "review-only",
        "new observation",
    }


def test_global_freeze_new_alert_reopened_and_rollback_export(tmp_path, monkeypatch):
    import json

    from tsmcp.cleaning import dataset_pipeline
    from tsmcp.followup import cli as followup

    fs = 125
    t = np.arange(15 * fs) / fs
    src = tmp_path / "source"
    src.mkdir()
    for i in (1, 2):
        np.savez(src / f"{i}.npz", time=t, PPG=40 + np.sin(2 * np.pi * t))
    spec = tmp_path / "spec.json"
    root = tmp_path / "run"
    write(
        spec,
        dict(
            dataset=dict(
                format="npz",
                path="source",
                mapping=dict(fs=fs, time_key="time", channels=[dict(key="PPG", task="PPG", unit="u")]),
            ),
            config=dict(method="dataset_agent", api_cap=0.5, tasks=["PPG"], atlas=False, joint_views=0),
        ),
    )
    monkeypatch.setattr(Reviewer, "session", lambda self: nullcontext(self))

    def discover(workspace, cases, data, *args):
        hits = []
        for c in cases:
            valid = data(c["id"])[2]
            changed = not valid[500]
            a, b, reason = (770, 810, "new_issue") if changed else (500, 525, "seed")
            hits.append(dict(id=c["id"] + reason, a=a, b=b, case=c["id"], reason=reason, value=1.0, task="PPG"))
        return hits

    monkeypatch.setattr(dataset_pipeline, "discover", discover)

    def decide(self, stage, rows):
        answers = {}
        for r in rows:
            key = r["id"]
            payload = r["data"]
            if stage == "groups":
                assert read(root / "ledger/freeze.json")["status"] == "candidate_frozen"
                assert len(list(root.glob("records/*/discovery.json"))) == 2
                if payload["trigger"] == "new_issue":
                    continue
                action = dict(wave="mark_invalid", wave_if=[], measure="review", measure_if=[], otherwise="keep")
            elif stage == "gate":
                action = dict(decision="accept" if "prior_review" in payload else "need_evidence")
            elif stage == "coarse":
                action = dict(method="identity")
            else:
                action = dict(default="keep", exceptions=[])
            ref = digest([key, payload, action])
            z = dict(id=key, reason="fixture", request_id=ref, receipt=ref, **action)
            answers[key] = z
            write(root / "dataset-receipts" / (ref + ".json"), z)
        return answers

    monkeypatch.setattr(Reviewer, "decide", decide)
    result = dataset_cli(Namespace(spec=spec, workspace=root, resume=False, live=True))
    assert result["status"] == "partial" and published_run(root)
    rows = [json.loads(line) for line in (root / "ledger/candidates.jsonl").read_text().splitlines()]
    news = [r for r in rows if r["detector"] == "new_issue"]
    assert len(news) == 2 and all(r["change"] == "new" and r["state"] == "pending_budget" for r in news)
    for p in root.glob("records/*/rescan.json"):
        assert read(p)["changes"]["new"] == read(p)["reassessed"]
    seed = news[0]["candidate_id"]
    case, region, _, good, _, revision = load_seed(root, root / "ledger", seed)
    assert good[region["start_sample"] : region["end_sample"]].all()
    # A genuine follow-up on a new release reaches the reviewer via the same terminal contract.
    write(root / "api_budget.json", dict(calls=[], cap_usd=0.5, estimated_peak_usd=0))
    monkeypatch.setattr(followup, "vision_preflight", lambda *_: None)
    monkeypatch.setattr(followup, "review", lambda *args: dict(status="registered", task="PPG"))
    d = followup.investigate(
        root,
        root / "ledger",
        seed,
        tmp_path / "review",
        5,
        client_factory=lambda *a, **k: SimpleNamespace(close=lambda: None),
    )
    assert d["status"] == "registered" and d["binding"]["inspection"] == revision
    f = next(root.glob("records/*/patches/*.json"))
    patch = Patch.model_validate(read(f))
    before = sha(f)
    commit_patch(f.parents[1], patch, patch.plan.parent_version_id)
    assert sha(f) == before
    bad = patch.model_dump()
    bad["receipt"]["candidate_revision"] = "stale"
    with pytest.raises(ValueError, match="different version"):
        Patch.model_validate(bad)
    other = patch.model_copy(deep=True)
    other.candidate_version_id = "other"
    other.receipt.candidate_version_id = "other"
    other.idempotency_key = digest([other.plan.plan_id, "other"])
    with pytest.raises(ValueError, match="Conflicting patches"):
        commit_patch(f.parents[1], other, other.plan.parent_version_id)
    reverted = rollback(root, f.stem, tmp_path / "reverted")
    assert reverted == rollback(root, f.stem, tmp_path / "reverted")
    assert published_run(root) and sha(f) == before
    restored = read(tmp_path / "reverted/waveform_manifest.json")[patch.plan.write_ranges[0].block_id]
    assert restored["artifact_id"] == patch.plan.parent_version_id
    from concurrent.futures import ThreadPoolExecutor

    def compare_and_swap(proposal):
        try:
            commit_patch(tmp_path / "parallel", proposal, proposal.plan.parent_version_id)
            return "committed"
        except ValueError as error:
            assert "Conflicting patches" in str(error)
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(compare_and_swap, [patch, other])) == ["committed", "conflict"]


def test_respiration_ingestion_into_inspection_only(tmp_path):
    fs = 20
    t = np.arange(22 * fs) / fs
    np.savez(tmp_path / "resp.npz", t=t, breath=np.sin(0.3 * 2 * np.pi * t))
    source = dict(
        format="npz",
        path="resp.npz",
        mapping=dict(time_key="t", fs=fs, channels=[dict(task="RESP", key="breath", unit="a.u.")]),
    )
    path = tmp_path / "spec.json"
    write(path, dict(dataset=source, config=dict(method="inspect_only", tasks=["RESP"], inspection_preset="deep")))
    report = dataset_cli(Namespace(workspace=tmp_path / "run", spec=path, live=False, resume=False))
    assert report["inspection_complete"]
    import json

    cov = json.loads((tmp_path / "run/inspection/coverage.jsonl").read_text().strip())
    statuses = {r["tool_id"]: r["status"] for r in cov["tool_results"]}
    assert statuses["high_frequency"] == statuses["line_50"] == "not_applicable"
    assert cov["context_levels"]["multiscale"]["5"]["hop_seconds"] == 1.25


def test_skill_promotion_needs_sandbox_and_independent_paired_references(tmp_path):
    from tsmcp.cleaning.engine import execute_version, register_recipe

    proc = tmp_path / "processing"
    refs = tmp_path / "references"
    refs.mkdir()
    lib = Knowledge(tmp_path / "library")
    fs = 125
    t = np.arange(1400) / fs
    cases = []
    entries = []
    for i, split in enumerate(("regression", "holdout")):
        cid = "pulse" + str(i)
        clean = 40 + np.sin(2 * np.pi * (1 + 0.15 * i) * t)
        bad = clean.copy()
        mask = np.zeros(len(t), bool)
        mask[450 + i] = True
        bad[mask] += 100
        root = proc / "public" / cid
        root.mkdir(parents=True)
        np.savez(root / "input.npz", values=bad, time=t)
        cases.append(
            dict(
                id=cid,
                record_id=cid,
                n=len(t),
                duration_s=len(t) / fs,
                fs=fs,
                task="PPG",
                unit="u",
                input_sha256=sha(root / "input.npz"),
            )
        )
        truth = refs / (cid + ".npz")
        np.savez(truth, clean=clean, affected=mask)
        entries.append(
            dict(
                workspace="../processing",
                case=cid,
                subject=cid,
                split=split,
                reference=truth.name,
                reference_sha256=sha(truth),
            )
        )
    write(proc / "dataset_manifest.json", dict(cases=cases))
    recipe = register_recipe(
        proc,
        source="def run(ctx):\n    for i in range(1,len(ctx.values)-1):\n        if ctx.values[i]>100:\n            ctx.apply('interpolate',[[i,i+1]])\n    return ctx.current\n",
    )
    ver = execute_version(proc, recipe, cases, stage="skill_trial")
    from tsmcp.processing import VERSION

    card = dict(
        skill_id="short-pulse",
        version="1.0.0",
        role="repair",
        problem_type="isolated_spike_candidate",
        modalities=["PPG"],
        applicability={"device": "synthetic"},
        prerequisites=["two observed anchors"],
        tools={"interpolate": VERSION},
        parameters={},
        read_scope="two sided anchors",
        write_scope="core only",
        expected_effects=["recover core"],
        stop_conditions=["missing support"],
        acceptance_criteria={"min_mae_gain": 1, "max_clean_change_fraction": 0},
        regression_suite="two synthetic subjects",
        known_failure_modes=["real pulses"],
        sources=["fixture"],
    )
    lib.propose(card)
    with pytest.raises(ValueError, match="transition"):
        lib.transition("short-pulse", "1.0.0", "active")
    lib.transition("short-pulse", "1.0.0", "sandboxed", workspace=proc, version_id=ver["version_id"])
    for e in entries:
        e["version_id"] = ver["version_id"]
    bench = refs / "suite.json"
    write(bench, dict(suite=card["regression_suite"], cases=[entries[0], dict(entries[0], split="holdout")]))
    with pytest.raises(ValueError, match="independent"):
        lib.transition("short-pulse", "1.0.0", "evaluated", benchmark=bench)
    write(bench, dict(suite=card["regression_suite"], cases=entries))
    lib.transition("short-pulse", "1.0.0", "evaluated", benchmark=bench)
    lib.transition("short-pulse", "1.0.0", "approved")
    lib.transition("short-pulse", "1.0.0", "active")
    snap = lib.freeze(tmp_path / "a")
    assert lib.context(snap, "repair", "PPG", "isolated_spike_candidate")["skills"]
    lib.transition("short-pulse", "1.0.0", "deprecated")
    assert not lib.context(lib.freeze(tmp_path / "b"), "repair", "PPG", "isolated_spike_candidate")["skills"]
    assert lib.freeze(tmp_path / "a") == snap


def test_diagnosis_and_planner_receive_separate_version_bound_contexts(tmp_path):
    from PIL import Image

    cfg = dict(method="dataset_agent", model="fixture", max_llm_calls_per_record=4)
    reviewer = Reviewer(tmp_path, cfg)
    pics = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (14, 14), (i, 0, 0)).save(p)
        pics.append(p)
    observed = []

    def request(messages, tools, system):
        content = messages[0]["content"]
        observed.append((system, [x["type"] for x in content]))
        if "Classify the observed group" in system:
            item = dict(
                id="g",
                reason="fixture observation",
                diagnosis=dict(
                    label="physiology",
                    confidence=0.8,
                    evidence_refs=["view_1", "members"],
                    alternative_hypotheses=["artifact"],
                ),
            )
        else:
            assert "diagnostic_receipt" in content[0]["text"] and "fixture observation" in content[0]["text"]
            item = dict(
                id="g",
                reason="fixture policy",
                wave="preserve",
                wave_if=[],
                measure="keep",
                measure_if=[],
                otherwise="keep",
                parameters={},
            )
        block = dict(type="tool_use", name="submit", input={"items": [item]})
        return SimpleNamespace(
            id=f"fixture-{len(observed)}", content=[SimpleNamespace(model_dump=lambda **kwargs: block)]
        )

    reviewer.client = SimpleNamespace(request=request)
    req = dict(
        id="g",
        images=pics,
        data=dict(
            task="ECG",
            trigger="isolated_spike_candidate",
            revision="frozen",
            evidence_ids=["view_1", "view_2", "members"],
            member_rows=[[0, 3, 5]],
        ),
    )
    d = reviewer.decide("groups", [req])["g"]
    assert [kinds.count("image") for _, kinds in observed] == [2, 0]
    assert d["diagnosis"]["label"] == "physiology" and d["diagnosis_receipt"] != d["receipt"]
    assert Reviewer(tmp_path / "replay", dict(cfg, replay_from=str(tmp_path))).decide("groups", [req])["g"] == d


def test_blind_observation_precedes_independent_release_receipt(tmp_path):
    from PIL import Image

    reviewer = Reviewer(tmp_path, dict(method="dataset_agent", model="fixture"))
    pic = tmp_path / "pair.png"
    Image.new("RGB", (14, 14)).save(pic)
    trace = []

    def request(messages, tools, system):
        prompt = messages[0]["content"][0]["text"]
        if "before seeing the proposed action" in system:
            assert "secret_retest_label" not in prompt
            item = dict(id="k", reason="blind-observation")
        else:
            assert "blind-observation" in prompt and "secret_retest_label" in prompt
            item = dict(id="k", reason="gate", decision="preserve")
        trace.append(system)
        return SimpleNamespace(
            id=f"mock-{len(trace)}",
            content=[
                SimpleNamespace(model_dump=lambda **kw: dict(type="tool_use", name="submit", input={"items": [item]}))
            ],
        )

    reviewer.client = SimpleNamespace(request=request)
    d = reviewer.decide(
        "gate",
        [
            dict(
                id="k",
                images=[pic],
                data=dict(task="ECG", revision="r2", parent="v0", candidate="v1", secret_retest_label="data"),
            )
        ],
    )["k"]
    assert len(trace) == 2 and d["decision"] == "rollback" and d["review_decision"] == "preserve"


def test_visual_pair_is_not_resubmitted_for_every_numeric_page(tmp_path):
    import json

    from PIL import Image

    cfg = dict(method="dataset_agent", model="test", max_text_bytes=10500, max_llm_calls_per_record=80)
    reviewer = Reviewer(tmp_path, cfg)
    pic = tmp_path / "wave.png"
    Image.new("RGB", (20, 15)).save(pic)
    members = [[0, i, i / 7] for i in range(700)]
    base = dict(
        id="group",
        images=[pic, pic],
        data=dict(
            task="PPG",
            trigger="y_low",
            cases=[{"id": "c"}],
            member_rows=members,
            image_rows=[0, 699],
            scope=len(members),
            columns=["case", "sample", "value"],
            evidence_ids=["view_1", "view_2", "members"],
            summary={"min": 0, "max": 100},
        ),
    )
    pages = reviewer.context.group_pages(base)
    assert len(pages) > 1
    received = []
    num_images = 0

    def request(messages, tools, system):
        nonlocal num_images
        content = messages[0]["content"]
        obj = json.loads(content[0]["text"])
        num_images += sum(r["type"] == "image" for r in content)
        if "Classify the observed group" in system:
            assert "member_rows" not in obj["data"]
            d = dict(
                id=obj["id"],
                reason="shape",
                diagnosis=dict(label="unknown", confidence=0.0, evidence_refs=["view_1"], alternative_hypotheses=[]),
            )
        else:
            received.extend(obj["data"]["member_rows"])
            d = dict(
                id=obj["id"],
                reason="plan",
                wave="preserve",
                wave_if=[],
                measure="review",
                measure_if=[],
                otherwise="keep",
                parameters={},
            )
        return SimpleNamespace(
            id=d["id"],
            content=[
                SimpleNamespace(model_dump=lambda **k: dict(type="tool_use", name="submit", input={"items": [d]}))
            ],
        )

    reviewer.client = SimpleNamespace(request=request)
    plans = reviewer.decide("groups", [p for p, _, _ in pages])
    assert len(plans) == len(pages) and received == members and num_images == 2
    assert len({p["diagnosis_receipt"] for p in plans.values()}) == 1
