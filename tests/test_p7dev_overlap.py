"""Synthetic overlap checks; never access actual confirmation artifacts."""

import json

import pytest

from eval.tasks.p7dev_overlap import compare_overlap, load_confirmation_index


def _eligibility(tmp_path):
    path = tmp_path / "eligibility.json"
    path.write_text(
        json.dumps(
            {
                "suite": "p6confirm",
                "data_hashes": {"source.dat": "a" * 64},
                "patients": [
                    {
                        "database": "confirmation",
                        "segments": [
                            {"database": "confirmation", "record": "01", "segment": "01:0"},
                            {"database": "confirmation", "record": "02", "segment": "02:0"},
                        ],
                        "case_ids": {"longest_af": "c1", "af_burden": "c2"},
                        "selected": {"longest_af": "01:0", "af_burden": "01:0"},
                    }
                ],
            }
        )
    )
    return path


def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"suite": "p6confirm", "cases": rows}))
    return path


def _development(record="01", digest="a" * 64):
    return [{"case_id": "d1", "database": "ltafdb", "record": record, "signal_sha256": digest}]


def test_source_hashes_cannot_substitute_for_case_signal_hashes(tmp_path):
    index = load_confirmation_index(_eligibility(tmp_path))
    result = compare_overlap(_development(), index)
    assert result["status"] == "incomplete"
    assert result["zero_overlap"] is None
    assert result["identity_check"]["status"] == "passed"
    assert result["signal_hash_check"]["confirmation_hashed_case_count"] == 0
    assert result["signal_hash_check"]["confirmation_missing_hash_case_ids"] == ["c1", "c2"]
    assert "data_hashes" not in index


def test_complete_case_hashes_establish_zero_overlap(tmp_path):
    manifest = _manifest(tmp_path, [{"case_id": cid, "signal_sha256": "b" * 64} for cid in ("c2", "c1")])
    index = load_confirmation_index(_eligibility(tmp_path), manifest)
    result = compare_overlap(_development(), index)
    assert result["status"] == "passed"
    assert result["zero_overlap"] is True
    assert result["signal_hash_check"]["confirmation_hashed_case_count"] == 2


def test_partial_hash_manifest_does_not_pass(tmp_path):
    manifest = _manifest(tmp_path, [{"case_id": "c1", "signal_sha256": "b" * 64}])
    result = compare_overlap(_development(), load_confirmation_index(_eligibility(tmp_path), manifest))
    assert result["status"] == "incomplete"
    assert result["signal_hash_check"]["confirmation_missing_hash_case_ids"] == ["c2"]


def test_matching_hash_is_failure_even_if_coverage_incomplete(tmp_path):
    manifest = _manifest(tmp_path, [{"case_id": "c1", "signal_sha256": "a" * 64}])
    result = compare_overlap(_development(), load_confirmation_index(_eligibility(tmp_path), manifest))
    assert result["status"] == "failed"
    assert result["zero_overlap"] is False
    assert result["signal_hash_check"]["overlaps"][0]["p6confirm_case_ids"] == ["c1"]


def test_identity_checks_all_eligibility_records(tmp_path):
    development = _development(record="02")
    development[0]["database"] = "confirmation"
    result = compare_overlap(development, load_confirmation_index(_eligibility(tmp_path)))
    assert result["status"] == "failed"
    assert result["identity_check"]["overlaps"][0]["record"] == "02"


def test_same_record_separate_endpoints_are_not_duplicate_records(tmp_path):
    development = _development()
    development.append({**development[0], "case_id": "d2"})
    index = load_confirmation_index(_eligibility(tmp_path))
    assert not compare_overlap(development, index)["p7dev_duplicate_signals_across_records"]
    development[1]["record"] = "02"
    duplicates = compare_overlap(development, index)["p7dev_duplicate_signals_across_records"]
    assert duplicates[0]["case_ids"] == ["d1", "d2"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"case_id": "unknown", "signal_sha256": "b" * 64}],
        [{"case_id": "c1", "signal_sha256": "b" * 64}] * 2,
        [{"case_id": "c1", "signal_sha256": "bad"}],
        [{"case_id": "c1", "signal_sha256": "b" * 64, "record": "wrong"}],
    ],
)
def test_invalid_manifest_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        load_confirmation_index(_eligibility(tmp_path), _manifest(tmp_path, rows))
