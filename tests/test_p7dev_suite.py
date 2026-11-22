"""P7 development construction contracts, using synthetic inputs only."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import build_cases, common  # noqa: E402
from eval.grade import grade, oracle  # noqa: E402
from eval.tasks import p7dev  # noqa: E402

FS = 128
MIN_SAMPLES = 72000 * FS
QUESTIONS = {
    "longest_af": (
        "When does the longest episode of atrial fibrillation or flutter begin? "
        "Count AF or flutter separated by less than 30 seconds of other rhythm "
        "as one continuous episode. Answer in seconds from the start."
    ),
    "af_burden": "For what percentage of the total recording time is the heart in atrial fibrillation or flutter?",
}


def header(n_samples=MIN_SAMPLES, **overrides):
    values = {
        "fs": FS,
        "sig_len": n_samples,
        "n_sig": 2,
        "fmt": ["16", "16"],
        "file_name": ["private-record.dat", "private-record.dat"],
        "units": ["mV", "mV"],
    }
    return SimpleNamespace(**(values | overrides))


def annotation(intervals):
    samples, labels = [], []
    for start, end in intervals:
        samples.extend((start, end))
        labels.extend(("(AFIB", "(N"))
    return SimpleNamespace(sample=np.array(samples, dtype=int), aux_note=labels, symbol=["N"] * len(samples))


def selected_record(**overrides):
    values = {
        "database": "ltafdb",
        "record": "private-record",
        "patient_id": "private-record",
        "fs": FS,
        "n_samples": MIN_SAMPLES,
        "dur_s": 72000,
        "longest_af": {"start": 120, "end": 7320, "dur_s": 7200},
        "af_percent": 10,
        "eligible": {"longest_af": True, "af_burden": True},
        "case_ids": {"longest_af": "a" * 32, "af_burden": "b" * 32},
        "sources": ["dev", "p2long"],
    }
    return values | overrides


def test_question_wording_is_byte_identical_to_frozen_part_one():
    assert p7dev.QUESTIONS == QUESTIONS
    assert set(p7dev.ENDPOINTS) == set(QUESTIONS)


@pytest.mark.parametrize("gap,episodes", [(30 * FS - 1, 1), (30 * FS, 2), (30 * FS + 1, 2)])
def test_gap_boundary_uses_samples_and_merged_gap_does_not_increase_burden(gap, episodes):
    start, end = 120 * FS, 320 * FS
    next_start, next_end = end + gap, end + gap + 200 * FS
    facts = p7dev.annotation_facts(header(), annotation([(start, end), (next_start, next_end)]))
    assert facts["af_s"] == 400
    assert facts["af_percent"] == pytest.approx(400 / 72000 * 100)
    assert facts["n_af_segments"] == 2
    assert facts["n_merged_af_episodes"] == episodes
    expected_end = next_end / FS if episodes == 1 else end / FS
    assert facts["longest_af"] == {"start": 120, "end": expected_end, "dur_s": expected_end - 120}


def test_longest_merged_episode_tie_uses_earliest_onset():
    intervals = [(120 * FS, 320 * FS), (325 * FS, 530 * FS), (1000 * FS, 1410 * FS)]
    facts = p7dev.annotation_facts(header(), annotation(intervals))
    assert facts["longest_af"] == {"start": 120, "end": 530, "dur_s": 410}
    assert facts["n_merged_af_episodes"] == 2
    assert facts["af_s"] == 815


def test_rhythm_notes_count_afib_and_flutter_regardless_of_beat_symbol():
    labels = ["(AT", "(PAT", "(NOD", "(N", "(AFIB", "", "(AFL\x00", "(AB", "(AFIB", "(AT", "(AFL", "beat"]
    annotations = SimpleNamespace(sample=np.arange(12) * FS, aux_note=labels, symbol=["N"] * len(labels))
    facts = p7dev.annotation_facts(header(12 * FS), annotations)
    assert facts["af_s"] == 6
    assert facts["af_percent"] == 50
    assert facts["n_af_segments"] == 3


def test_repeated_af_markers_and_flutter_transition_do_not_double_count_time():
    annotations = SimpleNamespace(
        sample=np.array([120, 200, 300, 500]) * FS,
        aux_note=["(AFIB", "(AFIB", "(AFL", "(N"],
        symbol=["N", "V", "+", "N"],
    )
    facts = p7dev.annotation_facts(header(), annotations)
    assert facts["af_s"] == 380
    assert facts["n_af_segments"] == facts["n_merged_af_episodes"] == 1
    assert facts["longest_af"] == {"start": 120, "end": 500, "dur_s": 380}


def test_absent_af_has_no_episode_and_initial_unannotated_time_is_not_af():
    facts = p7dev.annotation_facts(header(), annotation([]))
    assert facts["af_s"] == facts["af_percent"] == 0
    assert facts["longest_af"] is None
    assert not any(facts["eligible"].values())
    annotations = SimpleNamespace(sample=np.array([MIN_SAMPLES - FS]), aux_note=["(AFL"])
    facts = p7dev.annotation_facts(header(), annotations)
    assert facts["af_s"] == 1
    assert facts["longest_af"]["start"] == 71999


@pytest.mark.parametrize(
    "start,duration,longest_eligible,burden_eligible",
    [
        (120 * FS, 300 * FS, False, False),
        (120 * FS, 300 * FS + 1, True, False),
        (120 * FS - 1, 301 * FS, False, False),
        (120 * FS, 3600 * FS, True, False),
        (120 * FS, 3600 * FS + 1, True, True),
        (120 * FS, 64800 * FS - 1, True, True),
        (120 * FS, 64800 * FS, False, True),
        (120 * FS, 68400 * FS - 1, False, True),
        (120 * FS, 68400 * FS, False, False),
    ],
)
def test_frozen_endpoint_thresholds_at_one_sample_precision(start, duration, longest_eligible, burden_eligible):
    facts = p7dev.annotation_facts(header(), annotation([(start, start + duration)]))
    expected = {"longest_af": longest_eligible, "af_burden": burden_eligible}
    assert facts["eligible"] == expected
    assert {kind: not reasons for kind, reasons in facts["exclusion_reasons"].items()} == expected


@pytest.mark.parametrize(
    "n_samples,eligible", [(MIN_SAMPLES - 1, False), (MIN_SAMPLES, True), (86399 * FS, True), (90000 * FS, True)]
)
def test_c1_whole_record_duration_and_denominator(n_samples, eligible):
    facts = p7dev.annotation_facts(header(n_samples), annotation([(120 * FS, 7320 * FS)]))
    assert facts["n_samples"] == n_samples
    assert facts["dur_s"] == n_samples / FS
    assert facts["af_percent"] == pytest.approx(7200 * FS / n_samples * 100)
    assert facts["eligible"] == {kind: eligible for kind in QUESTIONS}


def test_whole_file_preserves_annotation_tail_after_24_hours():
    annotations = SimpleNamespace(sample=np.array([86401 * FS]), aux_note=["(AFIB"])
    facts = p7dev.annotation_facts(header(90000 * FS), annotations)
    assert facts["longest_af"] == {"start": 86401, "end": 90000, "dur_s": 3599}
    assert facts["af_percent"] == pytest.approx(3599 / 90000 * 100)
    assert facts["eligible"] == {"longest_af": True, "af_burden": False}


@pytest.mark.parametrize("overrides", [{"fs": 200}, {"n_sig": 1}, {"fmt": ["212", "212"]}, {"sig_len": 0}])
def test_unexpected_ltafdb_header_is_rejected(overrides):
    with pytest.raises(ValueError):
        p7dev.annotation_facts(header(**overrides), annotation([]))


@pytest.mark.parametrize(
    "samples,notes", [([-1], ["(AFIB"]), ([MIN_SAMPLES], ["(AFIB"]), ([128, 0], ["(AFIB", "(N"]), ([128], [])]
)
def test_invalid_annotation_coordinates_are_rejected(samples, notes):
    annotations = SimpleNamespace(sample=np.array(samples), aux_note=notes)
    with pytest.raises(ValueError):
        p7dev.annotation_facts(header(), annotations)


@pytest.mark.parametrize("committed,returncode", [(b'{"changed": true}', 0), (b"{}", 128)])
def test_commit_gate_compares_file_bytes_to_head_not_worktree_index(tmp_path, monkeypatch, committed, returncode):
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_bytes(b"{}")
    monkeypatch.setattr(p7dev, "ROOT", tmp_path)
    calls = []

    def git(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=committed)

    monkeypatch.setattr(p7dev.subprocess, "run", git)
    with pytest.raises(ValueError, match="[Cc]ommit"):
        p7dev.require_committed(eligibility)
    assert calls[0][0] == ["git", "show", "HEAD:eligibility.json"]
    assert calls[0][1]["cwd"] == tmp_path


def test_commit_gate_accepts_exact_head_bytes(tmp_path, monkeypatch):
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_bytes(b"{}\n")
    monkeypatch.setattr(p7dev, "ROOT", tmp_path)
    monkeypatch.setattr(p7dev.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=b"{}\n"))
    p7dev.require_committed(eligibility)


def test_uncommitted_eligibility_stops_before_metadata_or_waveform_access(tmp_path, monkeypatch):
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_bytes(b"{}")
    monkeypatch.setattr(p7dev, "ROOT", tmp_path)
    monkeypatch.setattr(p7dev.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=b"different"))
    monkeypatch.setattr(p7dev, "collect_eligibility", lambda *a, **kw: pytest.fail("metadata read before commit gate"))
    monkeypatch.setattr(p7dev, "read_signal", lambda *a: pytest.fail("waveform read before commit gate"))
    with pytest.raises(ValueError, match="[Cc]ommit"):
        p7dev.prepare_p7dev(tmp_path, eligibility)


@pytest.mark.parametrize("changed", [False, True])
def test_preflight_rechecks_frozen_namespace_source_root_and_facts(tmp_path, monkeypatch, changed):
    frozen = {
        "case_namespace": "private-namespace",
        "sources": {"opened_cases": str(tmp_path / "opened")},
        "records": [],
    }
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text(json.dumps(frozen))
    checked = []
    monkeypatch.setattr(p7dev, "require_committed", lambda path: checked.append(path))

    def collect(data_dir, *, opened_cases, case_namespace):
        assert checked == [eligibility]
        assert data_dir == tmp_path / "ltafdb"
        assert opened_cases == tmp_path / "opened"
        assert case_namespace == "private-namespace"
        return frozen | ({"records": [{"changed": True}]} if changed else {})

    monkeypatch.setattr(p7dev, "collect_eligibility", collect)
    if changed:
        with pytest.raises(ValueError, match="differ.*eligibility"):
            p7dev.prepare_p7dev(tmp_path / "ltafdb", eligibility)
    else:
        assert p7dev.prepare_p7dev(tmp_path / "ltafdb", eligibility) == frozen


def test_opened_source_union_deduplicates_records_and_reads_only_authorized_suites(tmp_path, monkeypatch):
    tasks = tmp_path / "eval/tasks"
    tasks.mkdir(parents=True)
    inventory = tmp_path / "eval/results/af-continuity"
    inventory.mkdir(parents=True)
    opened = tmp_path / "opened"
    opened.mkdir()
    dev = [str(record) for record in range(100, 108)]
    former_fresh = [str(record) for record in range(200, 238)]
    (tasks / "ltafdb_split.json").write_text(json.dumps({"dev": dev, "test_ext": ["999"]}))
    (inventory / "fresh-inventory.json").write_text(
        json.dumps({"records": [{"record": record} for record in former_fresh]})
    )
    p2 = [
        {"case_id": "ltaf001", "answer": {"record": "100"}},
        {"case_id": "ltaf002", "answer": {"record": "100"}},
        {"case_id": "ltaf003", "answer": {"record": "300"}},
        {"case_id": "synthetic001", "answer": {}},
    ]
    p4 = [{"case_id": "ltaf004", "answer": {"record": "200"}}, {"case_id": "ltaf005", "answer": {"record": "300"}}]
    for suite, rows in (("p2long", p2), ("p4long", p4)):
        (opened / f"{suite}.answers.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(p7dev, "ROOT", tmp_path)
    original_read = Path.read_text
    paths = []

    def restricted_read(path, *args, **kwargs):
        assert path.name in {
            "ltafdb_split.json",
            "fresh-inventory.json",
            "p2long.answers.jsonl",
            "p4long.answers.jsonl",
        }
        paths.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", restricted_read)
    sources = p7dev.opened_sources(opened)
    assert len(paths) == 4
    assert set(sources["membership"]) == set(dev + former_fresh + ["300"])
    assert sources["membership"]["100"] == ["dev", "p2long"]
    assert sources["membership"]["200"] == ["former_fresh", "p4long"]
    assert sources["membership"]["300"] == ["p2long", "p4long"]
    assert sources["repeated_record_rows_within_opened_suites"] == {"p2long": ["100"], "p4long": []}
    assert sources["records_by_source"]["p2long"] == ["100", "300"]


def test_eligibility_reads_only_allowed_headers_and_manual_atr_and_assigns_one_case_per_endpoint(tmp_path, monkeypatch):
    import wfdb

    allowed = ["100", "200"]
    sources = {
        "membership": {"100": ["dev", "p2long"], "200": ["former_fresh"]},
        "records_by_source": {"dev": ["100"], "p2long": ["100"], "former_fresh": ["200"]},
    }
    monkeypatch.setattr(p7dev, "opened_sources", lambda _: sources)
    hashes, headers, annotations = [], [], []
    for record in allowed:
        with (tmp_path / f"{record}.dat").open("wb") as stream:
            stream.truncate(MIN_SAMPLES * 4)

    def hash_file(path):
        assert path.stem in allowed and path.suffix in {".hea", ".atr", ".dat"}
        hashes.append(path.name)
        return path.name

    def read_header(path):
        record = Path(path).name
        assert record in allowed
        headers.append(record)
        return header(file_name=[f"{record}.dat"] * 2)

    def read_annotation(path, extension):
        record = Path(path).name
        assert record in allowed and extension == "atr"
        annotations.append(record)
        return annotation([(120 * FS, 7320 * FS)])

    monkeypatch.setattr(p7dev, "sha256", hash_file)
    monkeypatch.setattr(wfdb, "rdheader", read_header)
    monkeypatch.setattr(wfdb, "rdann", read_annotation)
    monkeypatch.setattr(wfdb, "rdrecord", lambda *a, **kw: pytest.fail("eligibility decoded waveform"))
    frozen = p7dev.collect_eligibility(
        tmp_path, opened_cases=tmp_path, case_namespace="00000000-0000-0000-0000-000000000001"
    )
    assert headers == annotations == allowed
    assert set(hashes) == {f"{record}.{extension}" for record in allowed for extension in ("hea", "atr", "dat")}
    assert len(frozen["records"]) == 2
    assert frozen["identity_check"]["unique_records"] == frozen["identity_check"]["unique_patients"] == 2
    ids = [case_id for row in frozen["records"] for case_id in row["case_ids"].values()]
    assert len(ids) == len(set(ids)) == 4
    assert all(len(case_id) == 32 and set(case_id) <= set("0123456789abcdef") for case_id in ids)
    assert all(row["database"] == "ltafdb" and row["record"] == row["patient_id"] for row in frozen["records"])
    assert frozen["counts"]["total"]["longest_af"]["eligible"] == 2
    assert frozen["counts"]["total"]["af_burden"]["eligible"] == 2


def test_waveform_reader_copies_entire_first_channel_including_partial_final_chunk(tmp_path, monkeypatch):
    import wfdb

    size = 1_000_003
    calls = []

    def read_record(path, *, sampfrom, sampto, channels):
        assert path == str(tmp_path / "private-record")
        assert channels == [0]
        calls.append((sampfrom, sampto))
        return SimpleNamespace(fs=FS, p_signal=np.full((sampto - sampfrom, 1), len(calls), dtype=np.float64))

    monkeypatch.setattr(wfdb, "rdrecord", read_record)
    values = p7dev.read_signal(tmp_path, selected_record(n_samples=size))
    assert calls == [(0, 1_000_000), (1_000_000, size)]
    assert values.shape == (size,) and values.dtype == np.float32
    assert np.all(values[:1_000_000] == 1) and np.all(values[1_000_000:] == 2)


@pytest.mark.parametrize(
    "shape,dtype", [((MIN_SAMPLES - 1,), np.float32), ((MIN_SAMPLES, 1), np.float32), ((MIN_SAMPLES,), np.float64)]
)
def test_builder_rejects_partial_multichannel_or_wrong_dtype_signal(monkeypatch, shape, dtype):
    monkeypatch.setattr(p7dev, "read_signal", lambda *a: np.broadcast_to(dtype(0), shape))
    with pytest.raises(ValueError):
        list(p7dev.build_p7dev({"records": [selected_record()]}))


def test_builder_has_separate_endpoint_cases_and_annotation_only_answers(monkeypatch):
    calls = []
    row = selected_record()

    def read_signal(data_dir, selected):
        calls.append(selected)
        return np.broadcast_to(np.float32(0), (selected["n_samples"],))

    monkeypatch.setattr(p7dev, "read_signal", read_signal)
    cases = list(p7dev.build_p7dev({"records": [row]}))
    assert calls == [row]
    assert len(cases) == 2
    assert len({case[0] for case in cases}) == 2
    for case_id, kind, variant, values, fs, answer, question in cases:
        assert case_id == row["case_ids"][kind]
        assert variant == "-" and fs == FS
        assert values.shape == (MIN_SAMPLES,) and values.dtype == np.float32
        assert question == QUESTIONS[kind]
        assert answer["record"] == answer["patient_id"] == "private-record"
        assert answer["database"] == "ltafdb"
        assert grade(kind, answer, oracle(kind, answer))["correct"]
        if kind == "longest_af":
            assert answer["t"] == 120 and answer["longest_dur_s"] == 7200
        else:
            assert answer["percent"] == 10


def test_ineligible_records_do_not_read_waveforms(monkeypatch):
    monkeypatch.setattr(p7dev, "read_signal", lambda *a: pytest.fail("ineligible waveform accessed"))
    row = selected_record(eligible={kind: False for kind in QUESTIONS}, case_ids={})
    assert list(p7dev.build_p7dev({"records": [row]})) == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["--suite", "other", "--p7dev"],
        ["--suite", "p7dev"],
        ["--suite", "p7dev", "--p7dev", "--confirm"],
        ["--suite", "p7dev", "--p7dev", "--afdb"],
        ["--suite", "p7dev", "--p7dev", "--synthetic", "dev"],
        ["--suite", "p7dev", "--p7dev", "--mitdb", "ds1"],
        ["--suite", "p7dev", "--p7dev", "--ltafdb", "1"],
        ["--suite", "p7dev", "--p7dev", "--ucr", "1"],
        ["--suite", "p7dev", "--p7dev", "--synthetic-long", "1"],
    ],
)
def test_cli_rejects_wrong_suite_and_combined_datasets_before_access(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(build_cases, "CASES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_cases.py", *arguments])
    for module, name in (
        (p7dev, "prepare_p7dev"),
        (build_cases.confirm, "prepare_confirm"),
        (build_cases.afdb, "prepare_afdb"),
    ):
        monkeypatch.setattr(module, name, lambda *a: pytest.fail("invalid CLI accessed dataset"))
    with pytest.raises(SystemExit) as error:
        build_cases.main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_failed_preflight_preserves_existing_p7dev_suite(tmp_path, monkeypatch):
    suite = tmp_path / "p7dev"
    suite.mkdir()
    sentinel = suite / "keep.txt"
    sentinel.write_text("existing suite")
    for kind in ("answers", "questions"):
        (tmp_path / f"p7dev.{kind}.jsonl").write_text(f"existing {kind}")
    monkeypatch.setattr(build_cases, "CASES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_cases.py", "--suite", "p7dev", "--p7dev"])

    def blocked():
        raise ValueError("annotation facts differ from frozen eligibility")

    monkeypatch.setattr(p7dev, "prepare_p7dev", blocked)
    with pytest.raises(SystemExit) as error:
        build_cases.main()
    assert error.value.code == 2
    assert sentinel.read_text() == "existing suite"
    for kind in ("answers", "questions"):
        assert (tmp_path / f"p7dev.{kind}.jsonl").read_text() == f"existing {kind}"


def test_cli_isolates_requested_dataset_and_keeps_private_answers_outside_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(build_cases, "CASES", tmp_path)
    monkeypatch.setattr(common, "CASES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_cases.py", "--suite", "p7dev", "--p7dev"])
    frozen = {"records": [selected_record()]}
    monkeypatch.setattr(p7dev, "prepare_p7dev", lambda: frozen)
    monkeypatch.setattr(p7dev, "write_manifest", lambda *a: None)
    monkeypatch.setattr(p7dev, "read_signal", lambda *a: np.broadcast_to(np.float32(0), (MIN_SAMPLES,)))
    build = p7dev.build_p7dev

    def small_cases(selected):
        assert selected is frozen
        for case_id, kind, variant, _, fs, answer, question in build(selected):
            yield case_id, kind, variant, np.arange(256, dtype=np.float32), fs, answer, question

    monkeypatch.setattr(p7dev, "build_p7dev", small_cases)
    for module, name in (
        (build_cases.synthetic, "build"),
        (build_cases.mitdb, "build"),
        (build_cases.longrec, "build_ltafdb"),
        (build_cases.afdb, "build_afdb"),
        (build_cases.confirm, "build_confirm"),
    ):
        monkeypatch.setattr(module, name, lambda *a: pytest.fail("unexpected dataset builder"))
    build_cases.main()
    suite = common.load_suite("p7dev")
    answers = common.read_jsonl(tmp_path / "p7dev.answers.jsonl")
    assert len(suite) == len(answers) == 2
    assert {case["qtype"] for case in suite} == set(QUESTIONS)
    assert len({case["dir"] for case in suite}) == 2
    public = (tmp_path / "p7dev.questions.jsonl").read_text()
    for case, answer in zip(suite, answers, strict=True):
        case_dir = Path(case["dir"])
        assert {path.name for path in case_dir.iterdir()} == {"signal.npy", "visible.json"}
        signal = np.load(case_dir / "signal.npy", allow_pickle=False)
        assert signal.shape == (256,) and signal.dtype == np.float32
        assert case["question"] == QUESTIONS[case["qtype"]]
        assert case["source"] == "ecg-long"
        assert case["visible"] == {
            "file": "signal.npy",
            "dur_s": 2,
            "channels": [{"name": "ch0", "unit": "mV", "fs": FS}],
        }
        prompt = common.case_prompt(case["visible"], case["question"], case["qtype"])
        assert case["question"] in prompt
        assert "answer" not in case
        assert answer["answer"]["record"] == "private-record"
        assert grade(answer["qtype"], answer["answer"], oracle(answer["qtype"], answer["answer"]))["correct"]
        public += json.dumps(case["visible"]) + prompt + case["case_id"]
    assert all(secret not in public for secret in ("ltafdb", "private-record", "p2long"))
