import pytest

from tsmcp.cleaning.repair_outcomes import attempt, export_outcomes


def test_funnel_distinguishes_early_protection_and_stale_input_from_failed_numerics(tmp_path):
    trial = dict(
        case="block",
        a=10,
        b=12,
        method="interpolate",
        status="ineligible",
        parent="raw",
        precheck=dict(status="insufficient_support", reason="stale_parent_context"),
        initial_precheck=dict(status="possible_physiology", reason="near_native_qrs"),
        hard_gate=dict(status="rejected", checks={"support_and_morphology": False}),
    )
    out = attempt(tmp_path, tmp_path, trial, ["core"], "fine", 1, {}, {})
    assert out.reason_codes == ["protected_landmark_conflict", "stale_evidence"]
    assert not out.operator_invoked and not out.candidate_generated
    assert (out.hard_gate, out.review, out.commit, out.outcome) == ("not_run", "not_run", "not_committed", "preserve")


def test_multiple_review_reasons_and_retries_remain_separate(tmp_path):
    trial = dict(
        case="block",
        candidate_hash="hash",
        method="mark_invalid",
        hard_gate={"status": "passed"},
        ranges={"a": {"decision": "rollback", "receipt": "r1"}, "b": {"decision": "need_evidence", "receipt": "r2"}},
    )
    replies = {k: {"request_id": "one-batch"} for k in ("r1", "r2")}
    first = attempt(tmp_path, tmp_path, trial, ["a", "b"], "fine", 1, {}, replies)
    second = attempt(tmp_path, tmp_path, trial, ["a", "b"], "fine", 2, {}, replies)
    assert first.attempt_id != second.attempt_id and first.candidate_ids == second.candidate_ids
    assert set(first.reason_codes) == {"review_rejected", "need_evidence"}
    assert first.operator_invoked and first.candidate_generated and first.outcome == "preserve"
    assert first.request_ids == ["one-batch"]


def test_funnel_output_collision_leaves_frozen_ledger_untouched(tmp_path):
    directory = tmp_path / "ledger"
    directory.mkdir()
    prior = directory / "candidates.jsonl"
    prior.write_text("immutable\n")
    with pytest.raises(ValueError, match="separate directory"):
        export_outcomes(tmp_path, directory)
    assert prior.read_text() == "immutable\n"


def test_normal_dataset_exports_funnel_and_archival_rebuild_is_idempotent(tmp_path):
    from argparse import Namespace

    import numpy as np

    from tsmcp.cleaning.common import read, sha, write
    from tsmcp.cleaning.dataset_entry import dataset_cli

    t = np.arange(1000) / 125
    np.savez(tmp_path / "signals.npz", time=t, p=42 + np.sin(t * 6))
    cfg = dict(
        method="dataset_rules",
        tasks=["PPG"],
        api_cap=0,
        atlas=False,
        repair_protocol="v1.1",
        repair_domain={"species": "rat", "goals": {"PPG": "PPG_PPI"}},
    )
    spec = tmp_path / "spec.json"
    write(
        spec,
        dict(
            dataset=dict(
                format="npz",
                path="signals.npz",
                mapping=dict(time_key="time", fs=125, channels=[{"key": "p", "task": "PPG", "unit": "unknown"}]),
            ),
            config=cfg,
        ),
    )
    root = tmp_path / "run"
    report = dataset_cli(Namespace(spec=spec, workspace=root, resume=False, live=False))
    assert report["status"] == "completed"
    funnel = read(root / report["repair_outcomes"])
    assert funnel["waveform_delivery"]["modified_samples"] == 0
    assert funnel["counts"]["initial_candidates"] == read(root / "ledger/freeze.json")["candidate_count"]
    assert sum(r["measurement_states"].get("keep", 0) for r in report["records"]) == 0
    files = ["export/measurements.csv", "export/usable_measurements.csv", "export/waveform_manifest.json"]
    sums = [sha(root / f) for f in files]
    assert export_outcomes(root, tmp_path / "offline") == funnel
    assert export_outcomes(root, tmp_path / "offline") == funnel
    assert [sha(root / f) for f in files] == sums
