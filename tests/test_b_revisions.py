"""Revision contracts use only synthetic replies and the saved v0 manifests."""

import ast
import asyncio
import copy
import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import b_revisions as br  # noqa: E402
from eval import b_workflow as bw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def case(tmp_path):
    directory = tmp_path / "synthetic"
    directory.mkdir()
    np.save(directory / "signal.npy", np.zeros(100), allow_pickle=False)
    return {
        "case_id": "synthetic",
        "qtype": "longest_af",
        "question": "When does the longest AF begin?",
        "dir": str(directory),
        "visible": {
            "file": "signal.npy",
            "dur_s": 10000.3,
            "channels": [{"name": "ch0", "unit": "mV", "fs": 10}],
        },
    }


def episode(start, span):
    return {"t": start, "span_s": span, "af_s": span, "intervals_s": [[start, start + span]]}


def profile_item(window, n_rr, rr_cv, alt, *, error=False):
    row = {
        "name": "ecg_rr_profile",
        "ref": "synthetic:profile",
        # Deliberately wrong rounded coordinates must never influence bin geometry.
        "t0": 99999,
        "dt_s": 99999,
        "n_rr": n_rr,
        "rr_cv": rr_cv,
        "alt": alt,
    }
    return {
        "key": "synthetic.profile",
        "arguments": {"window": window},
        "content": [bw.text_block(json.dumps({"measurements": [row]}))],
        "is_error": error,
    }


class RevisionClient:
    def __init__(self, *, episodes=None, empty_profiles=False, core_padding=0):
        self.episodes = episodes if episodes is not None else [episode(1000, 3000), episode(6000, 1000)]
        self.empty_profiles = empty_profiles
        self.core_padding = core_padding
        self.calls = []
        self.replies = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, copy.deepcopy(arguments)))
        data = {"id": "12345678"}
        core = name in ("ts_load", "ts_describe")
        if core:
            data.update(dur_s=10000.3, channels=[{"name": "ch0", "fs": 10, "n": 100003}])
        elif name == "ts_find_events":
            data["events"] = []
        elif name == "ts_measure":
            what = arguments["what"][0]
            core = what in ("ecg_af_burden", "ecg_af_episodes")
            row = {"name": what, "ref": f"synthetic:{what}", "ch": "ch0"}
            if what == "ecg_af_episodes":
                row.update(episodes=self.episodes, merge_gap_s=30)
            elif what == "ecg_af_burden":
                row.update(pct_of_window=40, af_s=4000, window_s=10000.3, readable_s=9000)
            elif what == "ecg_rr_profile":
                row.update(
                    t0=99999,
                    dt_s=99999,
                    hr_bpm=[None, 80, 90, None],
                    n_rr=[0, 8, 9, 0],
                    rr_cv=[None, 0.12, 0.2, None],
                    alt=[None, 0.75, 0.1, None],
                )
                if self.empty_profiles:
                    row.update(n_rr=[0] * 4, rr_cv=[None] * 4, alt=[None] * 4)
            else:
                row.update(cv=0.123, n_rr=80)
            data["measurements"] = [row]
        if core and self.core_padding:
            data["padding"] = "x" * self.core_padding
        raw = json.dumps(data, indent=1)
        blocks = [SimpleNamespace(type="text", text=raw)]
        if name == "ts_zoom":
            blocks.append(SimpleNamespace(type="image", mime_type="image/png", data="cG5n"))
        self.replies.append(raw)
        return SimpleNamespace(content=blocks, is_error=False)


def test_profile_edges_use_requested_samples_not_rounded_metadata():
    item = profile_item([1.1, 3.2], [8, 9, 10, 11], [0.12] * 4, [0.75] * 4)
    bins = br.profile_bins(item, 10)
    assert [(b["start"], b["end"]) for b in bins] == [
        (Fraction(11), Fraction(65, 4)),
        (Fraction(65, 4), Fraction(43, 2)),
        (Fraction(43, 2), Fraction(107, 4)),
        (Fraction(107, 4), Fraction(32)),
    ]
    assert all(b["qualifies"] for b in bins)


def test_profile_search_thresholds_missing_and_negative_values_break_runs():
    item = profile_item(
        [0, 90],
        [8, 8, 7, 8, 8, 8, None, -1, 8],
        [0.12, 0.13, 0.9, 0.119, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.75, 0.5, 0, 0, 0.751, -1, 0, 0, None],
    )
    bins = br.profile_bins(item, 10)
    assert [b["qualifies"] for b in bins] == [True, True, False, False, False, False, False, False, False]
    assert br.search_leads(bins) == [{"start": Fraction(0), "end": Fraction(200)}]


def test_short_or_missing_profile_arrays_leave_explicit_nonqualifying_bins():
    item = profile_item([0, 30], [8, 8, 8], [0.2, 0.2], [0.1])
    bins = br.profile_bins(item, 1)
    assert [b["qualifies"] for b in bins] == [True, False, False]
    assert bins[-1]["end"] == 30
    empty = profile_item([0, 30], [], [], [])
    assert br.profile_bins(empty, 1) == [{"start": Fraction(0), "end": Fraction(30), "qualifies": False}]


@pytest.mark.parametrize(
    "start,end,expected",
    [(-1, 2.03, [0, 2.1]), (8.04, 12, [8, 10]), (10, 11, None), (-2, -1, None), (5, 5, None)],
)
def test_requested_windows_clip_to_observable_sample_bounds(start, end, expected):
    assert br.sample_window(start, end, 10, 100) == expected


def test_search_leads_join_only_adjacent_bins_and_rank_span_then_start():
    bins = [
        {"start": Fraction(a), "end": Fraction(b), "qualifies": yes}
        for a, b, yes in [(0, 10, True), (10, 20, True), (20, 30, False), (30, 50, True), (60, 65, True)]
    ]
    assert br.search_leads(bins) == [
        {"start": Fraction(0), "end": Fraction(20)},
        {"start": Fraction(30), "end": Fraction(50)},
        {"start": Fraction(60), "end": Fraction(65)},
    ]
    failed = br.profile_bins(profile_item([20, 30], [], [], [], error=True), 1)
    assert failed == [{"start": Fraction(20), "end": Fraction(30), "qualifies": False}]
    assert len(br.search_leads(bins[:2] + failed + bins[3:])) == 3


def test_seeds_reserve_top_r_use_strict_separation_and_fill_from_r():
    index = {"episodes": [episode(300, 100), episode(100, 300), episode(900, 200)]}
    leads = [
        {"start": Fraction(1600), "end": Fraction(8000)},
        {"start": Fraction(1601), "end": Fraction(4000)},
        {"start": Fraction(2201), "end": Fraction(5000)},
    ]
    seeds = br.choose_seeds(index, leads, 10, 100000)
    assert [s["sample"] for s in seeds] == [1000, 1601, 9000]
    assert [s["source"] for s in seeds] == ["r_rank", "search_lead", "r_rank"]
    assert seeds[0]["rank"] == 1
    assert [s["sample"] for s in br.choose_seeds(index, [], 10, 100000)] == [1000, 9000, 3000]


def test_empty_index_can_take_three_scan_leads():
    leads = [{"start": Fraction(t), "end": Fraction(t + 100)} for t in (0, 601, 1202, 1803)]
    seeds = br.choose_seeds({"episodes": []}, leads, 10, 10000)
    assert [s["sample"] for s in seeds] == [0, 601, 1202]
    assert {s["source"] for s in seeds} == {"search_lead"}
    assert br.choose_seeds({"episodes": []}, [], 10, 10000) == []


def test_scan_separation_does_not_filter_further_r_ranked_fallbacks():
    index = {"episodes": [episode(100, 30), episode(140, 20), episode(180, 10)]}
    # The proposal's >60 s rule applies to scan starts. Remaining R ranks fill
    # unused slots in rank order, even when distinct starts are nearer than 60 s.
    leads = [{"start": Fraction(1400), "end": Fraction(1600)}]
    seeds = br.choose_seeds(index, leads, 10, 10000)
    assert [(s["sample"], s["source"]) for s in seeds] == [(1000, "r_rank"), (1400, "r_rank"), (1800, "r_rank")]


def test_local_refinement_uses_first_consecutive_pair_or_original_seed():
    bins = [
        {"start": Fraction(a), "end": Fraction(b), "qualifies": yes}
        for a, b, yes in [(0, 10, True), (10, 20, False), (20, 30, True), (30, 40, True), (40, 50, True)]
    ]
    assert br.refined_start(bins, 35) == 20
    assert br.refined_start(bins[:3], 35) == 35
    assert br.refined_start([bins[0], bins[2]], 35) == 35
    assert br.refined_start([], 35) == 35


@pytest.mark.parametrize(
    "workflow,qtype,effective",
    [
        ("v0", "longest_af", "v0"),
        ("v0", "af_burden", "v0"),
        ("r1", "longest_af", "r1"),
        ("r1", "af_burden", "v0"),
        ("r2", "longest_af", "v0"),
        ("r2", "af_burden", "r2"),
        ("final", "longest_af", "r1"),
        ("final", "af_burden", "r2"),
    ],
)
def test_version_dispatch(workflow, qtype, effective):
    assert br.effective_workflow(workflow, qtype) == effective


@pytest.mark.parametrize(
    "workflow,qtype", [("v0", "longest_af"), ("v0", "af_burden"), ("r1", "af_burden"), ("r2", "longest_af")]
)
def test_v0_dispatch_keeps_byte_identical_bundles(case, workflow, qtype):
    case["qtype"] = qtype
    expected = asyncio.run(bw.collect(case, RevisionClient()))
    actual = asyncio.run(br.collect(case, RevisionClient(), workflow=workflow))
    assert bw.canonical(actual) == bw.canonical(expected)


@pytest.mark.parametrize("qtype,effective", [("longest_af", "r1"), ("af_burden", "r2")])
def test_final_contains_exact_endpoint_revision_evidence(case, qtype, effective):
    case["qtype"] = qtype
    selected = asyncio.run(br.collect(case, RevisionClient(), workflow=effective))
    final = asyncio.run(br.collect(case, RevisionClient(), workflow="final"))
    assert final["b_workflow"] == "final" and final["b_workflow_effective"] == effective
    assert final["content"] == selected["content"]
    assert final["items"] == selected["items"]
    assert final["arithmetic"] == selected["arithmetic"]
    assert {k: v for k, v in final.items() if k != "b_workflow"} == {
        k: v for k, v in selected.items() if k != "b_workflow"
    }


@pytest.mark.parametrize("workflow", ["v0", "r1", "r2", "final"])
@pytest.mark.parametrize("qtype", ["longest_af", "af_burden"])
def test_cache_manifest_preserves_selected_and_effective_versions(case, tmp_path, workflow, qtype):
    case["qtype"] = qtype
    bundle = asyncio.run(br.collect(case, RevisionClient(), workflow=workflow))
    effective = br.effective_workflow(workflow, qtype)
    identity = {
        **bw.request_identity(case, {"server_git_sha": bw.FROZEN_R}),
        "b_workflow": workflow,
        "b_workflow_effective": effective,
    }
    request_sha = bw.digest(bw.canonical(identity).encode())
    bundle_sha = bw.digest((bw.canonical(bundle) + "\n").encode())
    meta = br.manifest(bundle, bundle_sha, identity, request_sha, workflow)
    assert meta["b_workflow"] == workflow and meta["b_workflow_effective"] == effective
    bw.write_json(tmp_path / "objects" / f"{bundle_sha}.json", bundle)
    bw.write_json(tmp_path / "requests" / f"{request_sha}.json", meta)
    assert br.load_cached(tmp_path, request_sha) == (bundle, meta)


def test_cache_rejects_changed_selected_version_even_if_effective_version_is_same(case, tmp_path):
    bundle = asyncio.run(br.collect(case, RevisionClient(), workflow="r1"))
    identity = {
        **bw.request_identity(case, {"server_git_sha": bw.FROZEN_R}),
        "b_workflow": "r1",
        "b_workflow_effective": "r1",
    }
    request_sha = bw.digest(bw.canonical(identity).encode())
    bundle_sha = bw.digest((bw.canonical(bundle) + "\n").encode())
    meta = br.manifest(bundle, bundle_sha, identity, request_sha, "r1")
    meta["b_workflow"] = "final"
    bw.write_json(tmp_path / "objects" / f"{bundle_sha}.json", bundle)
    bw.write_json(tmp_path / "requests" / f"{request_sha}.json", meta)
    with pytest.raises(ValueError, match="integrity|identity mismatch"):
        br.load_cached(tmp_path, request_sha)


def test_r1_full_search_refinement_is_deterministic_and_verbatim(case):
    first, second = RevisionClient(), RevisionClient()
    a = asyncio.run(br.collect(case, first, workflow="r1"))
    b = asyncio.run(br.collect(case, second, workflow="r1"))
    assert bw.canonical(a) == bw.canonical(b)
    assert a["status"] == "ready"
    assert a["sizes"]["calls"] == len(first.calls) == 20
    assert a["sizes"]["images"] == 4
    assert a["sizes"]["verification_text_chars"] <= a["verification_text_budget"]
    assert [i["key"] for i in a["items"][:4]] == ["load", "describe", "burden", "episode_index"]
    assert all(i["content"][0]["text"] == first.replies[i["call_number"] - 1] for i in a["items"])
    profiles = [args for name, args in first.calls if name == "ts_measure" and args["what"] == ["ecg_rr_profile"]]
    assert len(profiles) == 6
    thirds = [[round(t * 10) for t in p["window"]] for p in profiles[:3]]
    assert thirds == [[0, 33334], [33334, 66668], [66668, 100003]]
    assert all(0 <= p["window"][0] < p["window"][1] <= 10000.3 for p in profiles)
    assert len([args for name, args in first.calls if name == "ts_zoom" and args["view"] == "strip"]) == 3


def test_r1_empty_index_still_searches_and_no_leads_stops_cleanly(case):
    client = RevisionClient(episodes=[])
    bundle = asyncio.run(br.collect(case, client, workflow="r1"))
    assert bundle["status"] == "ready" and bundle["sizes"]["calls"] == 20
    assert bundle["sizes"]["images"] == 4
    empty = asyncio.run(br.collect(case, RevisionClient(episodes=[], empty_profiles=True), workflow="r1"))
    assert empty["status"] == "ready" and empty["sizes"]["calls"] == 8
    assert empty["sizes"]["images"] == 1
    assert not any(i["arguments"].get("view") == "strip" for i in empty["items"])


def test_r1_keeps_complete_core_and_reports_call_and_text_omissions(case):
    budget = bw.Budget(calls=10, text_chars=0, verification_floor_chars=1500)
    client = RevisionClient(core_padding=9000)
    bundle = asyncio.run(br.collect(case, client, budget, workflow="r1"))
    assert bundle["status"] == "ready"
    assert bundle["sizes"]["calls"] == len(client.calls) == 10
    assert bundle["sizes"]["core_text_chars"] > 36000
    assert bundle["sizes"]["verification_text_chars"] <= 1500
    assert {i["key"] for i in bundle["items"] if i["required"]} == {"load", "describe", "burden", "episode_index"}
    reasons = {d["reason"] for d in bundle["dropped"]}
    assert {"call_budget", "text_budget"} <= reasons
    assert all(not d.get("required") and "content" not in d for d in bundle["dropped"])
    assert all("response_sha256" in d for d in bundle["dropped"] if d["executed"])
    accounted = {i["call_number"] for i in bundle["items"]}
    accounted |= {d["call_number"] for d in bundle["dropped"] if d.get("executed")}
    assert accounted == set(range(1, 11))


FROZEN_V0_WORKFLOW = "f89804cf831408b5451c19ae7fdbb5f662269b4b07b6ce02d245a48bb1874532"


def test_saved_v0_manifests_bind_unchanged_workflow_and_answer_prompt():
    manifests = [json.loads(p.read_text()) for p in (ROOT / "eval/results/b-workflow/p7dev/manifests").glob("*.json")]
    assert len(manifests) == 66
    # Saved v0 manifests bind the frozen pre-F1 workflow; F1 changed only the suite allowlist, and the rebuild check
    # shows every one of the 66 bundles carries byte-identical evidence under the amended file.
    assert {m["identity"]["workflow_sha256"] for m in manifests} == {FROZEN_V0_WORKFLOW}
    equivalence = json.loads((ROOT / "eval/results/p6confirm/b-guard-equivalence.json").read_text())
    assert equivalence["cases"] == equivalence["evidence_identical"] == 66
    assert {m["identity"]["prompt_source_sha256"] for m in manifests} == {bw.file_hash(ROOT / "eval/common.py")}


def test_revision_workflow_has_no_answer_annotation_or_eligibility_file_access():
    modules = [ROOT / "eval/b_workflow.py", ROOT / "eval/b_revisions.py", ROOT / "eval/b_revision_burden.py"]
    modules.extend(sorted((ROOT / "eval").glob("b_revisions_*.py")))
    forbidden = ("answers", "eligibility", "p6confirm", ".atr", "annotations", "reference_onset")
    # F1: the suite name in public_cases' allowlist and its refusal message (no file access)
    allowed_constants = {"p6confirm", "b collection is restricted to p5afdb, p7dev and the frozen p6confirm run"}
    reads = ("open", "read_text", "read_bytes", "load", "loadtxt", "read_json", "read_csv", "rdann")
    for path in modules:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name in reads:
                    text = ast.unparse(node).lower()
                    assert not any(word in text for word in forbidden), f"forbidden data access: {path.name}: {text}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # File construction is checked as well as calls, so separating a
                # forbidden path from its read does not bypass this contract.
                value = node.value.lower()
                if "\n" not in value and len(value) < 300 and value not in allowed_constants:
                    assert not any(word in value for word in forbidden), (
                        f"forbidden path constant: {path.name}: {value}"
                    )
