"""R1 image limits, verification priorities and failed-profile preservation."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_b_revisions import RevisionClient, episode
from test_b_revisions import case as shared_case

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import b_revisions as br  # noqa: E402
from eval import b_workflow as bw  # noqa: E402

case = shared_case


@pytest.mark.parametrize("fs,sample", [(10, 359), (100, 359), (250, 544), (360, 988), (128, 359)])
def test_r1_windows_stay_exactly_centered_on_samples_at_nonbinary_rates(fs, sample):
    client = RevisionClient(empty_profiles=True)
    collector = br.Collector(client, bw.BUDGET)
    collector.calls = 5  # The four core calls and whole-record RR precede this phase.
    n = 1000000
    index = {"episodes": [episode(sample / fs, 1000)]}
    seed = br.choose_seeds(index, [], fs, n)[0]["sample"]
    asyncio.run(br.collect_onset(collector, "synthetic", index, fs, n))
    items = {i["key"]: i for i in collector.items}
    for suffix, before, after in (("profile", 600, 600), ("strip", 30, 30), ("before", 120, 0), ("after", 0, 120)):
        lo, hi = items[f"refine.1.{suffix}"]["arguments"]["window"]
        assert [round(lo * fs), round(hi * fs)] == [max(0, seed - before * fs), min(n, seed + after * fs)]


@pytest.mark.parametrize("calls", [8, 9, 10, 11, 12, 13, 14])
def test_r1_call_cap_reserves_strips_before_lower_priority_local_profiles(case, calls):
    bundle = asyncio.run(br.collect(case, RevisionClient(), bw.Budget(calls=calls), workflow="r1"))
    assert bundle["status"] == "ready" and bundle["sizes"]["calls"] == calls
    keys = {i["key"] for i in bundle["items"]}
    assert len([key for key in keys if key.startswith("refine.") and key.endswith(".strip")]) == min(3, calls - 8)
    assert len([key for key in keys if key.startswith("refine.") and key.endswith(".profile")]) == min(
        3, max(0, calls - 11)
    )
    assert all(not d["executed"] for d in bundle["dropped"] if d["reason"] == "call_budget")


@pytest.mark.parametrize("images", [0, 1, 2, 3, 4])
def test_r1_image_cap_records_each_unexecuted_view(case, images):
    client = RevisionClient()
    bundle = asyncio.run(br.collect(case, client, bw.Budget(images=images), workflow="r1"))
    assert bundle["status"] == "ready"
    assert bundle["sizes"]["images"] == images
    assert bundle["sizes"]["calls"] == len(client.calls) == 16 + images
    views = ["whole_rr", "refine.1.strip", "refine.2.strip", "refine.3.strip"]
    omitted = [d for d in bundle["dropped"] if d["reason"] == "image_budget"]
    assert [d["key"] for d in omitted] == views[images:]
    assert all(not d["executed"] and "content" not in d for d in omitted)
    assert [i["key"] for i in bundle["items"] if i["tool"] == "ts_zoom"] == views[:images]
    assert not any(i["required"] for i in omitted)


@pytest.mark.parametrize("remove_all_stats", [False, True])
def test_r1_text_drops_stats_as_pairs_before_profiles_and_preserves_search_and_strips(case, remove_all_stats):
    original = asyncio.run(br.collect(case, RevisionClient(), workflow="r1"))
    stats = [i for i in original["items"] if i["key"].endswith((".before", ".after"))]
    stats_chars = bw.sizes(bw.evidence_content(stats, []))["text_chars"]
    limit = original["sizes"]["verification_text_chars"] - (stats_chars if remove_all_stats else 0) - 1
    budget = bw.Budget(text_chars=0, verification_floor_chars=limit)
    bundle = asyncio.run(br.collect(case, RevisionClient(), budget, workflow="r1"))
    drops = [d for d in bundle["dropped"] if d["reason"] == "text_budget"]
    expected = (
        {"refine.3.before", "refine.3.after"}
        if not remove_all_stats
        else {i["key"] for i in stats} | {"refine.3.profile"}
    )
    assert {d["key"] for d in drops} == expected
    assert bundle["status"] == "ready"
    assert bundle["sizes"]["calls"] == 20
    assert bundle["sizes"]["verification_text_chars"] <= limit
    assert all(d["executed"] and "response_sha256" in d and "response_sizes" in d for d in drops)
    keys = {i["key"] for i in bundle["items"]}
    assert {"whole_rr", *(f"search.{i}.profile" for i in (1, 2, 3)), *(f"refine.{i}.strip" for i in (1, 2, 3))} <= keys
    for i in (1, 2, 3):
        assert (f"refine.{i}.before" in keys) == (f"refine.{i}.after" in keys)


class FailedProfileClient(RevisionClient):
    def __init__(self, *, local_only):
        super().__init__(episodes=None if local_only else [])
        self.local_only = local_only

    async def call_tool(self, name, arguments):
        reply = await super().call_tool(name, arguments)
        if name == "ts_measure" and arguments["what"] == ["ecg_rr_profile"]:
            start, end = arguments["window"]
            if not self.local_only or end - start <= 1200:
                reply.is_error = True
                reply.content = [SimpleNamespace(type="text", text="profile unavailable")]
        return reply


@pytest.mark.parametrize("local_only", [False, True])
def test_r1_profile_errors_remain_visible_and_never_supply_a_search_lead(case, local_only):
    bundle = asyncio.run(br.collect(case, FailedProfileClient(local_only=local_only), workflow="r1"))
    errors = [i for i in bundle["items"] if i["is_error"]]
    assert bundle["status"] == "ready"
    assert len(errors) == 3
    assert all(i["content"] == [bw.text_block("profile unavailable")] for i in errors)
    ledger = bundle["arithmetic"][0]
    if local_only:
        assert all(seed["refined_t"] == seed["t"] for seed in ledger["seeds"])
        assert {i["key"] for i in errors} == {f"refine.{i}.profile" for i in (1, 2, 3)}
    else:
        assert ledger["search_leads_s"] == ledger["seeds"] == []
        assert bundle["sizes"]["calls"] == 8
        assert {i["key"] for i in errors} == {f"search.{i}.profile" for i in (1, 2, 3)}
