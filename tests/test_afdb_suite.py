"""AFDB preparation checks using synthetic headers, annotations, and file bytes only."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.tasks import afdb  # noqa: E402


@pytest.mark.parametrize(
    ("start", "duration", "burden", "expected"),
    [
        (120, 300.004, 50, ["longest_af", "af_burden"]),
        (120, 300.004, 1, ["longest_af"]),
        (119.996, 301, 50, ["af_burden"]),
        (0, 301, 1, []),
        (120, 300, 50, ["af_burden"]),
        (120, 301, 5, ["longest_af"]),
        (120, 301, 89.999, ["longest_af", "af_burden"]),
        (120, 301, 90, ["af_burden"]),
        (120, 301, 95, []),
        (None, None, 5, []),
        (None, None, 5.001, ["af_burden"]),
        (None, None, 94.999, ["af_burden"]),
        (None, None, 95, []),
    ],
)
def test_independent_endpoints_and_frozen_boundaries(start, duration, burden, expected):
    longest = None if start is None else {"start": start, "dur_s": duration}
    assert afdb.select_types({"longest_af": longest, "af_percent": burden}) == expected


def test_frozen_inventory_has_28_questions_with_13_overlapping_records():
    frozen = json.loads(afdb.ELIGIBILITY.read_text())
    records = frozen["records"]
    assert frozen["schema_version"] == 2
    assert frozen["counts"] == {"longest_af": 15, "af_burden": 13}
    assert frozen["build_ready"] is True
    assert frozen["blocker"] is None
    assert len(records) == 23
    assert set(frozen["excluded"]) == {"00735", "03665"}
    for facts in records.values():
        assert facts["types"] == afdb.select_types(facts)
        assert facts["answer_ready"] is True
        assert facts["atr_beat_count"] == 0
        assert "type" not in facts and "bpm" not in facts
    assert sum("longest_af" in facts["types"] for facts in records.values()) == 15
    assert sum("af_burden" in facts["types"] for facts in records.values()) == 13
    assert sum(len(facts["types"]) == 2 for facts in records.values()) == 13
    assert sum(not facts["types"] for facts in records.values()) == 8


def header(duration=1000, fs=250):
    return SimpleNamespace(fs=fs, sig_len=int(duration * fs), n_sig=2, fmt=["212", "212"])


def rhythm(samples, labels):
    return SimpleNamespace(sample=np.array(samples), symbol=["+"] * len(samples), aux_note=labels)


def test_annotation_times_are_from_record_start_and_final_rhythm_reaches_end():
    annotation = rhythm([100, 30001, 75000, 100000], ["(N", "(AFIB", "(J", "(AFL\x00"])
    facts = afdb.annotation_facts("04015", header(), annotation)
    assert facts["fs"] == 250
    assert facts["n_samples"] == 250000
    assert facts["dur_s"] == 1000
    assert facts["af_s"] == pytest.approx(779.996)
    assert facts["af_percent"] == pytest.approx(77.9996)
    assert facts["n_af_segments"] == 2
    assert facts["n_merged_af_episodes"] == 2
    assert facts["longest_af"] == {"start": 400, "end": 1000, "dur_s": 600}
    assert facts["types"] == ["longest_af", "af_burden"]


def test_initial_af_marker_does_not_shift_or_extend_to_record_origin():
    facts = afdb.annotation_facts("04015", header(), rhythm([174], ["(AFIB"]))
    assert facts["longest_af"]["start"] == 0.696
    assert facts["af_s"] == pytest.approx(999.304)
    assert facts["af_percent"] == pytest.approx(99.9304)
    assert facts["types"] == []
    assert facts["answer_ready"] is True


@pytest.mark.parametrize(("gap_samples", "episodes", "longest_s"), [(7499, 1, 401), (7500, 2, 201)])
def test_merge_gap_is_strict_and_raw_burden_excludes_gap(gap_samples, episodes, longest_s):
    first_start, first_end = 30000, 80000
    second_start = first_end + gap_samples
    second_end = second_start + 42751 if gap_samples == 7499 else second_start + 50250
    facts = afdb.annotation_facts(
        "04015",
        header(),
        rhythm([0, first_start, first_end, second_start, second_end], ["(N", "(AFIB", "(N", "(AFL", "(N"]),
    )
    raw_s = ((first_end - first_start) + (second_end - second_start)) / 250
    assert facts["n_af_segments"] == 2
    assert facts["n_merged_af_episodes"] == episodes
    assert facts["af_s"] == pytest.approx(raw_s)
    assert facts["af_percent"] == pytest.approx(raw_s / 10)
    assert facts["longest_af"]["dur_s"] == pytest.approx(longest_s)


def test_rhythm_only_record_without_eligible_endpoint_has_no_hr_fallback():
    facts = afdb.annotation_facts("04015", header(), rhythm([125], ["(N"]))
    assert facts["types"] == []
    assert facts["atr_beat_count"] == 0
    assert "bpm" not in facts
    assert facts["answer_ready"] is True


def test_beat_annotations_are_counted_for_audit_but_never_create_hr_answers():
    annotation = SimpleNamespace(
        sample=np.arange(100) * 250 + 125,
        symbol=["N"] * 100,
        aux_note=[""] * 100,
    )
    facts = afdb.annotation_facts("04015", header(), annotation)
    assert facts["atr_beat_count"] == 100
    assert facts["types"] == []
    assert "bpm" not in facts
    assert facts["answer_ready"] is True


def test_wrong_sampling_rate_is_rejected():
    with pytest.raises(ValueError, match="250|sampling|fs"):
        afdb.annotation_facts("04015", header(fs=128), rhythm([0], ["(N"]))


@pytest.mark.parametrize("samples", [[-1], [250000], [200, 100]])
def test_invalid_annotation_coordinates_are_rejected(samples):
    with pytest.raises(ValueError, match="coordinates|unsorted"):
        afdb.annotation_facts("04015", header(), rhythm(samples, ["(N"] * len(samples)))


RECORDS = (
    "00735 03665 04015 04043 04048 04126 04746 04908 04936 05091 05121 05261 06426 "
    "06453 06995 07162 07859 07879 07910 08215 08219 08378 08405 08434 08455"
).split()


@pytest.fixture
def checked_data(tmp_path):
    files = {"RECORDS": "\n".join(RECORDS) + "\n"}
    for rec in RECORDS:
        extensions = ("hea", "atr", "qrs") if rec in {"00735", "03665"} else ("hea", "atr", "qrs", "dat")
        files.update({f"{rec}.{ext}": f"synthetic {rec} {ext}\n" for ext in extensions})
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in sorted(files.items())}
    manifest = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    manifest += f"{hashlib.sha256(b'optional').hexdigest()}  notes.txt\n"
    (tmp_path / "SHA256SUMS.txt").write_text(manifest)
    return tmp_path, hashes


def test_checksums_cover_every_downloaded_file_and_report_absent_optional_files(checked_data):
    data, hashes = checked_data
    verified = afdb.verify_data(data)
    assert verified == {
        "algorithm": "sha256",
        "manifest": "SHA256SUMS.txt",
        "manifest_sha256": hashlib.sha256((data / "SHA256SUMS.txt").read_bytes()).hexdigest(),
        "verified_file_count": 99,
        "files": hashes,
        "not_downloaded_manifest_entries": ["notes.txt"],
    }


@pytest.mark.parametrize("name", ["04015.dat", "04015.atr", "04015.qrs", "RECORDS"])
def test_checksum_mismatch_fails_even_for_files_not_used_as_truth(checked_data, name):
    data, _ = checked_data
    (data / name).write_text("changed bytes")
    with pytest.raises(ValueError, match="SHA256|checksum|hash|mismatch"):
        afdb.verify_data(data)


@pytest.mark.parametrize("extension", ["hea", "atr", "dat"])
def test_required_waveform_record_file_cannot_be_silently_omitted(checked_data, extension):
    data, _ = checked_data
    (data / f"04015.{extension}").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="04015|required|missing"):
        afdb.verify_data(data)


def test_unlisted_download_is_rejected(checked_data):
    data, _ = checked_data
    (data / "unlisted.txt").write_text("unverified")
    with pytest.raises(ValueError, match="unlisted|manifest|checksum|SHA256"):
        afdb.verify_data(data)


@pytest.mark.parametrize(
    ("kind", "qtype", "expected_answer", "question_part"),
    [
        ("af_burden", "af_burden", {"percent": 50}, "percentage of the total recording time"),
        ("longest_af", "longest_af", {"t": 120.004, "longest_dur_s": 301}, "LONGEST continuous episode"),
    ],
)
def test_build_uses_channel_zero_and_existing_answer_schema(
    tmp_path, monkeypatch, kind, qtype, expected_answer, question_part
):
    import wfdb

    facts = {
        "fs": 250,
        "n_samples": 250000,
        "dur_s": 1000,
        "types": [kind],
        "answer_ready": True,
        "af_percent": 50,
        "longest_af": {"start": 120.004, "dur_s": 301},
    }
    signal = np.zeros((250000, 1), dtype=np.float64)
    signal[:3, 0] = [0.125, -0.25, 0.5]
    reads = []

    def read(path, *, channels):
        reads.append((path, channels))
        return SimpleNamespace(fs=250, p_signal=signal)

    monkeypatch.setattr(wfdb, "rdrecord", read)
    rows = list(afdb.build_afdb({"records": {"04015": facts}}, data_dir=tmp_path))
    assert reads == [(str(tmp_path / "04015"), [0])]
    assert len(rows) == 1
    case_id, actual_qtype, variant, values, fs, answer, question = rows[0]
    assert case_id == "afdb000"
    assert actual_qtype == qtype
    assert variant == "-"
    assert values.shape == (250000,)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, signal[:, 0])
    assert fs == 250
    assert answer == {**expected_answer, "record": "04015", "dur_s": 1000}
    assert question_part in question
    assert "about 10 hours" in question


def test_build_decodes_overlapping_record_once_and_skips_ineligible_records(tmp_path, monkeypatch):
    import wfdb

    facts = {
        "fs": 250,
        "n_samples": 1000,
        "dur_s": 4,
        "answer_ready": True,
        "af_percent": 50,
        "longest_af": {"start": 120.004, "dur_s": 301},
    }
    records = {
        "04015": {"types": [], "answer_ready": True},
        "04043": {**facts, "types": ["longest_af", "af_burden"]},
        "04048": {"types": [], "answer_ready": True},
        "05261": {**facts, "types": ["longest_af"]},
    }
    reads = []

    def read(path, *, channels):
        reads.append((Path(path).name, channels))
        return SimpleNamespace(fs=250, p_signal=np.zeros((1000, 1)))

    monkeypatch.setattr(wfdb, "rdrecord", read)
    rows = list(afdb.build_afdb({"records": records}, data_dir=tmp_path))
    assert reads == [("04043", [0]), ("05261", [0])]
    assert [(row[0], row[1], row[5]["record"]) for row in rows] == [
        ("afdb000", "longest_af", "04043"),
        ("afdb001", "af_burden", "04043"),
        ("afdb002", "longest_af", "05261"),
    ]
    np.testing.assert_array_equal(rows[0][3], rows[1][3])
    assert rows[0][6] != rows[1][6]
    assert rows[0][5] == {"t": 120.004, "longest_dur_s": 301, "record": "04043", "dur_s": 4}
    assert rows[1][5] == {"percent": 50, "record": "04043", "dur_s": 4}


def test_build_rejects_missing_truth_before_reading_any_waveforms(monkeypatch):
    import wfdb

    monkeypatch.setattr(wfdb, "rdrecord", lambda *_, **__: pytest.fail("waveform decoded before truth was available"))
    with pytest.raises(ValueError, match="trusted answers"):
        list(
            afdb.build_afdb(
                {
                    "records": {
                        "04043": {"types": ["longest_af"], "answer_ready": True},
                        "04908": {"types": ["af_burden"], "answer_ready": False},
                    }
                }
            )
        )
