"""AFDB preflight and public/private layout, using synthetic inputs only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import build_cases, common  # noqa: E402
from eval.grade import grade, oracle  # noqa: E402
from eval.tasks import afdb  # noqa: E402


def test_preflight_rejects_changed_frozen_facts(tmp_path, monkeypatch):
    frozen_path = tmp_path / "eligibility.json"
    frozen_path.write_text(json.dumps({"records": {"04015": {"answer_ready": True}}}))
    monkeypatch.setattr(afdb, "collect_eligibility", lambda data_dir: {"records": {}})
    with pytest.raises(ValueError, match="differ from the frozen"):
        afdb.prepare_afdb(tmp_path, frozen_path)


def test_preflight_rejects_missing_selected_answer_before_waveform_read(tmp_path, monkeypatch):
    import wfdb

    frozen = {"records": {"04043": {"types": ["longest_af"], "answer_ready": False}}}
    frozen_path = tmp_path / "eligibility.json"
    frozen_path.write_text(json.dumps(frozen))
    monkeypatch.setattr(afdb, "collect_eligibility", lambda data_dir: frozen)
    monkeypatch.setattr(wfdb, "rdrecord", lambda *a, **kw: pytest.fail("must not read waveform"))
    with pytest.raises(ValueError, match="trusted answers.*04043"):
        afdb.prepare_afdb(tmp_path, frozen_path)
    with pytest.raises(ValueError, match="trusted answers"):
        list(afdb.build_afdb(frozen, tmp_path))


@pytest.mark.parametrize("types", [["hr"], ["mean_hr"], ["longest_af", "longest_af"]])
def test_build_rejects_unsupported_or_duplicate_endpoints_before_reading_waveforms(monkeypatch, types):
    import wfdb

    monkeypatch.setattr(wfdb, "rdrecord", lambda *a, **kw: pytest.fail("must not read waveform"))
    frozen = {"records": {"04043": {"types": types, "answer_ready": True}}}
    with pytest.raises(ValueError, match="unsupported|Unsupported|duplicate|Duplicate"):
        list(afdb.build_afdb(frozen))


@pytest.mark.parametrize(
    "facts",
    [
        {"types": ["longest_af"], "answer_ready": True, "longest_af": None},
        {"types": ["longest_af"], "answer_ready": True, "longest_af": {"start": None, "dur_s": 301}},
        {"types": ["af_burden"], "answer_ready": True},
        {"types": ["af_burden"], "answer_ready": True, "af_percent": float("nan")},
    ],
)
def test_build_rejects_absent_or_invalid_selected_answer_even_when_marked_ready(monkeypatch, facts):
    import wfdb

    monkeypatch.setattr(wfdb, "rdrecord", lambda *a, **kw: pytest.fail("must not read waveform"))
    with pytest.raises(ValueError, match="trusted answers.*04043"):
        list(afdb.build_afdb({"records": {"04043": facts}}))


def test_failed_preflight_preserves_existing_suite(tmp_path, monkeypatch):
    suite = tmp_path / "p5afdb"
    suite.mkdir()
    sentinel = suite / "keep.txt"
    sentinel.write_text("existing suite")
    answers = tmp_path / "p5afdb.answers.jsonl"
    answers.write_text("existing answers")
    questions = tmp_path / "p5afdb.questions.jsonl"
    questions.write_text("existing questions")
    monkeypatch.setattr(build_cases, "CASES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_cases.py", "--suite", "p5afdb", "--afdb"])

    def blocked():
        raise ValueError("AFDB data or annotation facts differ from the frozen eligibility JSON")

    monkeypatch.setattr(afdb, "prepare_afdb", blocked)
    with pytest.raises(SystemExit) as exc:
        build_cases.main()
    assert exc.value.code == 2
    assert sentinel.read_text() == "existing suite"
    assert answers.read_text() == "existing answers"
    assert questions.read_text() == "existing questions"


def test_afdb_cli_uses_only_requested_dataset_and_keeps_answers_private(tmp_path, monkeypatch):
    monkeypatch.setattr(build_cases, "CASES", tmp_path)
    monkeypatch.setattr(common, "CASES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_cases.py", "--suite", "p5afdb", "--afdb"])
    frozen = {"records": {}}
    monkeypatch.setattr(afdb, "prepare_afdb", lambda: frozen)
    private = [
        {"t": 120.004, "longest_dur_s": 301, "record": "04043", "dur_s": 4},
        {"percent": 77.25, "record": "04043", "dur_s": 4},
    ]
    questions = ["When does the longest AF begin?", "What percentage of the recording is AF?"]

    def fake_cases(selected):
        assert selected is frozen
        yield "afdb000", "longest_af", "-", np.zeros(1000), 250, private[0], questions[0]
        yield "afdb001", "af_burden", "-", np.zeros(1000), 250, private[1], questions[1]

    monkeypatch.setattr(afdb, "build_afdb", fake_cases)
    monkeypatch.setattr(build_cases.synthetic, "build", lambda *a: pytest.fail("unexpected synthetic builder"))
    monkeypatch.setattr(build_cases.mitdb, "build", lambda *a: pytest.fail("unexpected MITDB builder"))
    build_cases.main()
    suite = common.load_suite("p5afdb")
    assert [case["case_id"] for case in suite] == ["afdb000", "afdb001"]
    assert [case["qtype"] for case in suite] == ["longest_af", "af_burden"]
    assert suite[0]["dir"] != suite[1]["dir"]
    for index, case in enumerate(suite):
        case_dir = Path(case["dir"])
        assert {p.name for p in case_dir.iterdir()} == {"signal.npy", "visible.json"}
        signal = np.load(case_dir / "signal.npy", allow_pickle=False)
        assert signal.shape == (1000,) and signal.dtype == np.float32
        assert case["visible"] == {
            "file": "signal.npy",
            "dur_s": 4,
            "channels": [{"name": "ch0", "unit": "mV", "fs": 250}],
        }
        prompt = common.case_prompt(case["visible"], case["question"], case["qtype"])
        assert questions[index] in prompt
        assert questions[1 - index] not in prompt
        assert "04043" not in prompt and "120.004" not in prompt and "77.25" not in prompt
        assert "answer" not in case
    public = (tmp_path / "p5afdb.questions.jsonl").read_text()
    assert "04043" not in public and "120.004" not in public and "77.25" not in public
    rows = common.read_jsonl(tmp_path / "p5afdb.answers.jsonl")
    assert [row["answer"] for row in rows] == private
    for case, row in zip(suite, rows, strict=True):
        assert (row["case_id"], row["qtype"]) == (case["case_id"], case["qtype"])
        assert grade(row["qtype"], row["answer"], oracle(row["qtype"], row["answer"]))["correct"]
