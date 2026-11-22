"""Confirmation preparation uses synthetic annotations/files, never patient signals."""

import json
import stat
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import build_cases, common  # noqa: E402
from eval.grade import grade, oracle  # noqa: E402
from eval.tasks import confirm, confirm_iridia, confirm_shdb  # noqa: E402

FS = 200
DAY_SAMPLES = 24 * 60 * 60 * FS
MIN_SEGMENT_SAMPLES = 20 * 60 * 60 * FS
QUESTIONS = {
    "longest_af": (
        "When does the longest episode of atrial fibrillation or flutter begin? "
        "Count AF or flutter separated by less than 30 seconds of other rhythm "
        "as one continuous episode. Answer in seconds from the start."
    ),
    "af_burden": "For what percentage of the total recording time is the heart in atrial fibrillation or flutter?",
}


def segment(
    *,
    database="shdb-af",
    patient="private-patient",
    record="private-record",
    file_index=1,
    start=120,
    duration=5000,
    n_samples=DAY_SAMPLES,
    annotation_available=True,
):
    return {
        "database": database,
        "patient_id": patient,
        "record": record,
        "segment": f"{record}/{file_index}",
        "file_index": file_index,
        "start_sample": (file_index - 1) * DAY_SAMPLES,
        "n_samples": n_samples,
        "fs": FS,
        "af_intervals": [(round(start * FS), round((start + duration) * FS))],
        "annotation_available": annotation_available,
    }


@pytest.mark.parametrize(("gap", "episodes"), [(5999, 1), (6000, 2), (6001, 2)])
def test_episode_gap_is_strictly_less_than_30_seconds_and_burden_excludes_gap(gap, episodes):
    first_start, first_end = 24000, 64000
    second_start = first_end + gap
    second_end = second_start + 40000
    facts = confirm.annotation_facts([(first_start, first_end), (second_start, second_end)], DAY_SAMPLES, FS)
    assert facts["af_s"] == 400
    assert facts["af_percent"] == pytest.approx(400 / 86400 * 100)
    assert facts["n_af_segments"] == 2
    assert facts["n_merged_af_episodes"] == episodes
    expected_end = second_end / FS if episodes == 1 else first_end / FS
    assert facts["longest_af"] == {"start": 120, "end": expected_end, "dur_s": expected_end - 120}


def test_longest_episode_tie_uses_earliest_onset_after_merging():
    facts = confirm.annotation_facts([(24000, 64000), (65000, 106000), (200000, 282000)], DAY_SAMPLES)
    assert facts["n_merged_af_episodes"] == 2
    assert facts["longest_af"] == {"start": 120, "end": 530, "dur_s": 410}
    assert facts["af_s"] == 815


def test_absent_af_has_zero_burden_and_no_episode():
    facts = confirm.annotation_facts([], DAY_SAMPLES)
    assert facts["af_s"] == facts["af_percent"] == 0
    assert facts["longest_af"] is None
    assert facts["n_af_segments"] == facts["n_merged_af_episodes"] == 0


def test_overlapping_annotations_count_raw_af_only_once():
    facts = confirm.annotation_facts([(30000, 70000), (24000, 64000)], DAY_SAMPLES)
    assert facts["af_s"] == 230
    assert facts["n_af_segments"] == 1
    assert facts["longest_af"] == {"start": 120, "end": 350, "dur_s": 230}


@pytest.mark.parametrize(
    ("start", "duration", "longest_eligible", "burden_eligible"),
    [
        (120, 300, False, False),
        (120, 300.005, True, False),
        (119.995, 301, False, False),
        (120, 4320, True, False),
        (120, 4320.005, True, True),
        (120, 77759.995, True, True),
        (120, 77760, False, True),
        (120, 82079.995, False, True),
        (120, 82080, False, False),
    ],
)
def test_frozen_endpoint_thresholds(start, duration, longest_eligible, burden_eligible):
    source = segment(start=start, duration=duration)
    [patient] = confirm.assign_patients([source])
    [facts] = patient["segments"]
    expected = {"longest_af": longest_eligible, "af_burden": burden_eligible}
    assert facts["eligible"] == expected
    for kind, eligible in expected.items():
        assert patient["selected"][kind] == (source["segment"] if eligible else None)
        assert bool(facts["exclusion_reasons"][kind]) is not eligible


@pytest.mark.parametrize("database", ["shdb-af", "iridia-af"])
@pytest.mark.parametrize(
    ("n_samples", "eligible"),
    [
        (MIN_SEGMENT_SAMPLES - 1, False),
        (MIN_SEGMENT_SAMPLES, True),
        (86399 * FS, True),
        (DAY_SAMPLES, True),
        (DAY_SAMPLES + 3600 * FS, True),
    ],
)
def test_c1_whole_file_duration_boundary_and_burden_denominator(database, n_samples, eligible):
    source = segment(database=database, n_samples=n_samples)
    [patient] = confirm.assign_patients([source])
    [facts] = patient["segments"]
    assert facts["n_samples"] == n_samples
    assert facts["dur_s"] == n_samples / FS
    assert facts["af_s"] == 5000
    assert facts["af_percent"] == pytest.approx(5000 * FS / n_samples * 100)
    assert patient["selected"] == {kind: source["segment"] if eligible else None for kind in QUESTIONS}
    assert facts["eligible"] == dict.fromkeys(QUESTIONS, eligible)
    assert facts["exclusion_reasons"] == {kind: [] if eligible else ["shorter_than_20_hours"] for kind in QUESTIONS}


@pytest.mark.parametrize("n_samples", [MIN_SEGMENT_SAMPLES, DAY_SAMPLES + 1])
def test_missing_manual_annotations_never_contribute(n_samples):
    source = segment(n_samples=n_samples, annotation_available=False)
    [patient] = confirm.assign_patients([source])
    [facts] = patient["segments"]
    assert patient["selected"] == {"longest_af": None, "af_burden": None}
    assert facts["eligible"] == {"longest_af": False, "af_burden": False}
    assert all(facts["exclusion_reasons"][kind] for kind in QUESTIONS)


def test_patient_scan_uses_record_order_then_numeric_file_order_and_independent_endpoints():
    early_burden = segment(record="record02", file_index=2, start=0, duration=5000)
    later_both = segment(record="record02", file_index=10)
    next_record = segment(record="record10", file_index=1)
    ineligible = segment(record="record01", duration=100)
    [patient] = confirm.assign_patients([next_record, later_both, early_burden, ineligible])
    assert [row["segment"] for row in patient["segments"]] == [
        ineligible["segment"],
        early_burden["segment"],
        later_both["segment"],
        next_record["segment"],
    ]
    assert patient["selected"] == {
        "longest_af": later_both["segment"],
        "af_burden": early_burden["segment"],
    }
    assert len(set(patient["case_ids"].values())) == 2


def test_patient_identity_is_database_scoped_and_case_ids_do_not_disclose_identity():
    sources = [segment(database=database) for database in ("shdb-af", "iridia-af")]
    patients = confirm.assign_patients(sources)
    assert len(patients) == 2
    ids = [case_id for patient in patients for case_id in patient["case_ids"].values()]
    assert len(set(ids)) == 4
    assert confirm.assign_patients(list(reversed(sources))) == patients
    for case_id in ids:
        assert all(secret not in case_id for secret in ("shdb", "iridia", "private", "record", "patient"))


def test_shdb_rhythms_include_only_fibrillation_and_flutter():
    intervals = confirm_shdb.rhythm_intervals(
        np.arange(12) * 200,
        ["(AT", "(PAT", "(NOD", "(N", "(AFIB", "", "(AFL\x00", "(AB", "(AFIB", "(AT", "(AFL", "beat"],
        2400,
    )
    assert intervals == [[800, 1400], [1600, 1800], [2000, 2400]]


def test_shdb_inspection_reads_manual_atr_only_and_never_automatic_qrs(tmp_path, monkeypatch):
    import wfdb

    (tmp_path / "AdditionalData.csv").write_text(
        "Data_ID,Subject_ID,Annotated\n001,patient1,True\n002,patient2,False\n"
    )
    (tmp_path / "RECORDS.txt").write_text("001\n002\n")
    (tmp_path / "001.atr").write_bytes(b"synthetic annotation")
    for record in ("001", "002"):
        (tmp_path / f"{record}.qrs").write_bytes(b"automatic annotations must never be read")
    monkeypatch.setattr(confirm_shdb, "_read_header", lambda *_: SimpleNamespace(sig_len=DAY_SAMPLES))
    calls = []

    def read_annotation(path, extension):
        calls.append((Path(path).name, extension))
        return SimpleNamespace(sample=np.array([24000]), aux_note=["(AFIB"])

    monkeypatch.setattr(wfdb, "rdann", read_annotation)
    inspected = confirm_shdb.inspect_shdb(tmp_path)
    assert calls == [("001", "atr")]
    assert inspected["segments"][0]["af_intervals"] == [[24000, DAY_SAMPLES]]
    assert inspected["segments"][1]["annotation_available"] is False
    assert inspected["segments"][1]["af_intervals"] == []


def test_shdb_inspection_keeps_the_whole_distributed_file_and_annotations_after_24h(tmp_path, monkeypatch):
    import wfdb

    n_samples = DAY_SAMPLES + 3600 * FS
    (tmp_path / "AdditionalData.csv").write_text("Data_ID,Subject_ID,Annotated\n001,patient1,True\n")
    (tmp_path / "RECORDS.txt").write_text("001\n")
    (tmp_path / "001.atr").write_bytes(b"synthetic annotation")
    monkeypatch.setattr(confirm_shdb, "_read_header", lambda *_: SimpleNamespace(sig_len=n_samples))
    monkeypatch.setattr(
        wfdb,
        "rdann",
        lambda *_: SimpleNamespace(sample=np.array([DAY_SAMPLES + FS]), aux_note=["(AFIB"]),
    )
    [source] = confirm_shdb.inspect_shdb(tmp_path)["segments"]
    assert source["n_samples"] == source["record_n_samples"] == n_samples
    assert source["start_sample"] == source["file_index"] == 0
    assert source["af_intervals"] == [[DAY_SAMPLES + FS, n_samples]]
    [patient] = confirm.assign_patients([source])
    [facts] = patient["segments"]
    assert facts["longest_af"] == {"start": 86401, "end": 90000, "dur_s": 3599}
    assert facts["af_percent"] == pytest.approx(3599 / 90000 * 100)
    assert patient["selected"] == {"longest_af": source["segment"], "af_burden": None}


def test_shdb_waveform_copy_selects_channel_zero_and_converts_to_millivolts(tmp_path, monkeypatch):
    raw = np.array([[10, 999], [12, 998], [-32768, 997], [4, 996]], dtype="<i2")
    raw.tofile(tmp_path / "001.dat")
    header = SimpleNamespace(sig_len=4, file_name=["001.dat", "001.dat"], baseline=[10, 0], adc_gain=[2, 1])
    monkeypatch.setattr(confirm_shdb, "_read_header", lambda *_: header)
    monkeypatch.setattr(confirm_shdb, "MIN_SEGMENT_SAMPLES", 3)
    row = segment(record="001", n_samples=4)
    row["record_n_samples"] = 4
    values = confirm_shdb.read_signal(tmp_path, row)
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, [0, 1, np.nan, -3], equal_nan=True)
    with pytest.raises(ValueError, match="frozen segment"):
        confirm_shdb.read_signal(tmp_path, {**row, "n_samples": 3})


@pytest.mark.parametrize("length", [400, 43200 * FS])
def test_iridia_cross_file_annotations_are_clipped_without_concatenating_files(tmp_path, monkeypatch, length):
    (tmp_path / confirm_iridia.METADATA).write_text(
        "record_id,patient_id,record_files,record_samples\n"
        f"record_001,patient_001,2,{2 * length}\nrecord_002,patient_001,1,{length}\n"
    )
    for record, file_count in (("record_001", 2), ("record_002", 1)):
        directory = tmp_path / "records" / confirm_iridia.ARCHIVE_ROOT / record
        directory.mkdir(parents=True)
        for index in range(file_count):
            (directory / f"{record}_ecg_{index:02d}.h5").write_bytes(b"header mocked; no patient signal")
        (directory / f"{record}_ecg_labels.csv").write_text(
            "start_file_index,start_qrs_index,end_file_index,end_qrs_index\n"
            + (f"0,{length * 9 // 10},1,{length // 10}\n" if record == "record_001" else "")
        )
    monkeypatch.setattr(confirm_iridia, "_waveform_header", lambda _: {"n_samples": length})
    inspected = confirm_iridia.inspect_iridia(tmp_path)
    rows = inspected["segments"]
    assert [row["start_sample"] for row in rows] == [0, length, 0]
    assert [row["n_samples"] for row in rows] == [length] * 3
    assert [row["af_intervals"] for row in rows] == [[[length * 9 // 10, length]], [[0, length // 10]], []]
    assert len({row["segment"] for row in rows}) == 3
    [patient] = confirm.assign_patients(rows)
    assert patient["patient_id"] == "patient_001"
    assert len(patient["segments"]) == 3
    assert patient["selected"] == dict.fromkeys(QUESTIONS)
    assert all(row["exclusion_reasons"]["af_burden"] == ["shorter_than_20_hours"] for row in patient["segments"])


def test_iridia_waveform_copy_selects_whole_file_lead_ii_without_rescaling(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    records = tmp_path / "records"
    records.mkdir()
    path = records / "synthetic.h5"
    values = np.array([[10, 99], [12, -98], [15, 97], [4, -96]], dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ecg", data=values)
    monkeypatch.setattr(confirm_iridia, "MIN_SEGMENT_SAMPLES", 3)
    row = {"waveform_file": "records/synthetic.h5", "n_samples": 4}
    actual = confirm_iridia.read_lead_ii(tmp_path, row)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, values[:, 1].astype(np.float32))
    with pytest.raises(ValueError, match="waveform file"):
        confirm_iridia.read_lead_ii(tmp_path, {**row, "n_samples": 3})
    with pytest.raises(ValueError, match="waveform file"):
        confirm_iridia.read_lead_ii(tmp_path, {**row, "n_samples": 3, "file_start_sample": 1})


@pytest.mark.parametrize("unsafe", ["/absolute/file", "../escape", "ok/../../escape", "ok/../file", r"..\escape"])
def test_zip_rejects_unsafe_member_before_extracting_any_file(tmp_path, unsafe):
    archive_path = tmp_path / "records.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe/file.txt", "safe member comes first")
        archive.writestr(unsafe, "untrusted")
    destination = tmp_path / "records"
    with pytest.raises(ValueError, match="[Uu]nsafe|path"):
        confirm_iridia.extract_iridia_archive(archive_path, destination)
    assert not destination.exists()


def test_zip_rejects_symbolic_link_members_before_extraction(tmp_path):
    archive_path = tmp_path / "records.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", "safe member comes first")
        archive.writestr(member, "../outside")
    destination = tmp_path / "records"
    with pytest.raises(ValueError, match="[Uu]nsafe|symlink"):
        confirm_iridia.extract_iridia_archive(archive_path, destination)
    assert not destination.exists()


def test_zip_does_not_follow_an_existing_destination_symlink(tmp_path):
    archive_path = tmp_path / "records.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/file.txt", "untrusted")
    destination, outside = tmp_path / "records", tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes|symlink"):
        confirm_iridia.extract_iridia_archive(archive_path, destination)
    assert list(outside.iterdir()) == []


def test_zip_extracts_only_beneath_records_directory(tmp_path):
    archive_path = tmp_path / "records.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/annotations.txt", "120 500")
    destination = tmp_path / "records"
    confirm_iridia.extract_iridia_archive(archive_path, destination)
    assert (destination / "nested" / "annotations.txt").read_text() == "120 500"
    assert {path.name for path in tmp_path.iterdir()} == {"records.zip", "records"}


@pytest.mark.parametrize(
    ("n_samples", "start", "duration"),
    [(MIN_SEGMENT_SAMPLES, 120, 5000), (86399 * FS, 120, 5000), (90000 * FS, 84000, 6000)],
)
def test_c1_builder_copies_whole_file_once_for_separate_endpoint_sessions(monkeypatch, n_samples, start, duration):
    source = segment(n_samples=n_samples, start=start, duration=duration)
    frozen = {"patients": confirm.assign_patients([source])}
    values = np.broadcast_to(np.float32(1), (n_samples,))
    copied = []

    def copy_signal(row):
        copied.append(row["segment"])
        return values

    monkeypatch.setattr(confirm, "read_signal", copy_signal)
    cases = list(confirm.build_confirm(frozen))
    assert copied == [source["segment"]]
    assert len(cases) == len({case[0] for case in cases}) == 2
    assert {case[1] for case in cases} == set(QUESTIONS)
    for _, kind, _, signal, fs, answer, question in cases:
        assert signal is values
        assert signal.shape == (n_samples,)
        assert fs == FS
        assert answer["dur_s"] == n_samples / FS
        assert question == QUESTIONS[kind]
        if kind == "longest_af":
            assert answer["t"] == start
            assert answer["longest_dur_s"] == duration
        else:
            assert answer["percent"] == pytest.approx(duration * FS / n_samples * 100)


def test_confirm_cli_preserves_public_private_boundary_and_exact_frozen_questions(tmp_path, monkeypatch):
    frozen = {"patients": confirm.assign_patients([segment()])}
    monkeypatch.setattr(build_cases, "CASES", tmp_path)
    monkeypatch.setattr(common, "CASES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_cases.py", "--suite", "p6confirm", "--confirm"])
    monkeypatch.setattr(confirm, "prepare_confirm", lambda: frozen)
    monkeypatch.setattr(confirm, "read_signal", lambda row: np.broadcast_to(np.float32(0), (row["n_samples"],)))
    build = confirm.build_confirm

    def small_cases(selected):
        for case_id, kind, variant, _, fs, answer, question in build(selected):
            yield case_id, kind, variant, np.arange(400, dtype=np.float32), fs, answer, question

    monkeypatch.setattr(confirm, "build_confirm", small_cases)
    monkeypatch.setattr(build_cases.synthetic, "build", lambda *a: pytest.fail("unexpected synthetic builder"))
    monkeypatch.setattr(build_cases.mitdb, "build", lambda *a: pytest.fail("unexpected MITDB builder"))
    build_cases.main()
    suite = common.load_suite("p6confirm")
    answers = common.read_jsonl(tmp_path / "p6confirm.answers.jsonl")
    assert len(suite) == len(answers) == 2
    assert {case["qtype"] for case in suite} == set(QUESTIONS)
    assert len({case["dir"] for case in suite}) == 2
    public = (tmp_path / "p6confirm.questions.jsonl").read_text()
    for case, answer in zip(suite, answers, strict=True):
        case_dir = Path(case["dir"])
        assert {path.name for path in case_dir.iterdir()} == {"signal.npy", "visible.json"}
        signal = np.load(case_dir / "signal.npy", allow_pickle=False)
        assert signal.shape == (400,) and signal.dtype == np.float32
        assert case["question"] == QUESTIONS[case["qtype"]]
        assert case["visible"] == {
            "file": "signal.npy",
            "dur_s": 2,
            "channels": [{"name": "ch0", "unit": "mV", "fs": FS}],
        }
        prompt = common.case_prompt(case["visible"], case["question"], case["qtype"])
        assert case["question"] in prompt
        assert "answer" not in case
        assert grade(answer["qtype"], answer["answer"], oracle(answer["qtype"], answer["answer"]))["correct"]
        public += json.dumps(case["visible"]) + prompt + case["case_id"]
    assert all(secret not in public for secret in ("shdb-af", "iridia-af", "private-patient", "private-record"))
    assert all(row["answer"]["patient_id"] == "private-patient" for row in answers)
    assert all(row["answer"]["record"] == "private-record" for row in answers)


def test_failed_confirmation_preflight_preserves_existing_suite(tmp_path, monkeypatch):
    suite = tmp_path / "p6confirm"
    suite.mkdir()
    sentinel = suite / "keep.txt"
    sentinel.write_text("existing suite")
    for kind in ("answers", "questions"):
        (tmp_path / f"p6confirm.{kind}.jsonl").write_text(f"existing {kind}")
    monkeypatch.setattr(build_cases, "CASES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_cases.py", "--suite", "p6confirm", "--confirm"])

    def blocked():
        raise ValueError("confirmation facts differ from frozen eligibility")

    monkeypatch.setattr(confirm, "prepare_confirm", blocked)
    with pytest.raises(SystemExit) as error:
        build_cases.main()
    assert error.value.code == 2
    assert sentinel.read_text() == "existing suite"
    for kind in ("answers", "questions"):
        assert (tmp_path / f"p6confirm.{kind}.jsonl").read_text() == f"existing {kind}"


def test_uncommitted_eligibility_is_rejected_before_data_or_waveform_access(tmp_path, monkeypatch):
    eligibility = tmp_path / "confirm_eligibility.json"
    eligibility.write_text("{}")
    monkeypatch.setattr(confirm, "ROOT", tmp_path)
    monkeypatch.setattr(
        confirm.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=b'{"different": true}'),
    )
    monkeypatch.setattr(confirm, "collect_eligibility", lambda *_: pytest.fail("data accessed before commit check"))
    monkeypatch.setattr(confirm, "read_signal", lambda *_: pytest.fail("waveform accessed before commit check"))
    with pytest.raises(ValueError, match="Commit.*before building"):
        confirm.prepare_confirm(tmp_path, eligibility)


def test_changed_annotation_facts_fail_against_committed_eligibility(tmp_path, monkeypatch):
    eligibility = tmp_path / "confirm_eligibility.json"
    eligibility.write_text('{"patients": [], "case_namespace": "private-namespace"}')
    monkeypatch.setattr(confirm, "require_committed", lambda _: None)
    monkeypatch.setattr(confirm, "collect_eligibility", lambda *a, **kw: {"patients": [{"changed": True}]})
    with pytest.raises(ValueError, match="differ from the frozen"):
        confirm.prepare_confirm(tmp_path, eligibility)
