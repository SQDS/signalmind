"""Frozen AFDB detector protocol checks; all waveform inputs are synthetic."""

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import af_episode_metrics as metrics  # noqa: E402
from eval.tasks import afdb  # noqa: E402

RECORDS = (
    "04015 04043 04048 04126 04746 04908 04936 05091 05121 05261 06426 06453 "
    "06995 07162 07859 07879 07910 08215 08219 08378 08405 08434 08455"
).split()
ONSET = set("04043 04746 04908 04936 05121 05261 06995 07879 07910 08215 08219 08378 08405 08434 08455".split())
BURDEN = ONSET - {"05261", "08434"}


@pytest.fixture
def evidence():
    return json.loads(afdb.ELIGIBILITY.read_text())


def arguments(tmp_path, eligibility=afdb.ELIGIBILITY):
    return SimpleNamespace(
        afdb=True,
        fresh=False,
        baseline=None,
        prepare_eligible=False,
        afdb_dir=tmp_path,
        afdb_eligibility=eligibility,
        source_root=metrics.ROOT,
        source_ref="baseline-R-2026-10-12",
        variant_gap=None,
        cases_dir=None,
        output=tmp_path / "result.json",
    )


def forbid_detector_and_waveform(monkeypatch):
    import wfdb

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("tsmcp.plugins.ecg", "tsmcp.core.io.wfdb")):
            pytest.fail("detector or waveform loader imported before AFDB validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(wfdb, "rdrecord", lambda *a, **kw: pytest.fail("waveform read before AFDB validation"))
    monkeypatch.setattr(np, "fromfile", lambda *a, **kw: pytest.fail("waveform read before AFDB validation"))


def test_afdb_population_includes_all_23_and_preserves_independent_a1_endpoints(evidence):
    permitted = metrics.afdb_records(evidence)
    assert permitted == [("afdb", rec) for rec in RECORDS]
    assert {rec for rec, row in evidence["records"].items() if "longest_af" in row["types"]} == ONSET
    assert {rec for rec, row in evidence["records"].items() if "af_burden" in row["types"]} == BURDEN
    assert len(ONSET) == 15 and len(BURDEN) == 13
    assert len(ONSET & BURDEN) == 13
    assert sum(not row["types"] for row in evidence["records"].values()) == 8


@pytest.mark.parametrize(
    "name",
    ["afdb/00735", "afdb/03665", "afdb/99999", "mitdb/04043", "ltafdb/04043", "afdb/../04043", "04043"],
)
def test_afdb_runtime_allowlist_rejects_excluded_other_database_and_path_ids(evidence, name):
    permitted = metrics.afdb_records(evidence)
    metrics.require_afdb_record("afdb/04043", permitted)
    with pytest.raises(ValueError):
        metrics.require_afdb_record(name, permitted)


@pytest.mark.parametrize(
    "replacement", [None, "00735", "03665", "99999", "../04015", "afdb/04015", "mitdb/04015", "０４０１５"]
)
def test_afdb_population_cannot_be_shrunk_or_substituted(evidence, replacement):
    row = evidence["records"].pop("04015")
    if replacement is not None:
        evidence["records"][replacement] = row
    with pytest.raises(ValueError):
        metrics.afdb_records(evidence)


def test_afdb_population_rejects_an_extra_record(evidence):
    evidence["records"]["99999"] = evidence["records"]["04015"]
    with pytest.raises(ValueError):
        metrics.afdb_records(evidence)


@pytest.mark.parametrize("change", ["unsupported", "duplicate", "swap", "counts"])
def test_afdb_a1_membership_is_frozen_even_when_endpoint_counts_are_preserved(evidence, change):
    if change == "unsupported":
        evidence["records"]["04015"]["types"] = ["hr"]
    elif change == "duplicate":
        evidence["records"]["04043"]["types"].append("longest_af")
    elif change == "swap":
        first, second = evidence["records"]["04015"], evidence["records"]["04043"]
        first["types"], second["types"] = second["types"], first["types"]
    else:
        evidence["counts"]["longest_af"] = 14
    with pytest.raises(ValueError):
        metrics.afdb_records(evidence)


@pytest.mark.parametrize("change", ["missing", "population", "annotation", "hashes"])
def test_checkpoint_failures_precede_detector_imports_and_waveform_access(tmp_path, monkeypatch, evidence, change):
    path = tmp_path / "eligibility.json"
    if change == "population":
        evidence["records"].pop("04015")
    elif change == "annotation":
        evidence["records"]["04043"]["longest_af"]["start"] += 1
    elif change == "hashes":
        evidence["data_hashes"]["files"]["04043.atr"] = "0" * 64
    if change != "missing":
        path.write_text(json.dumps(evidence))
    forbid_detector_and_waveform(monkeypatch)
    monkeypatch.setattr(afdb, "prepare_afdb", lambda *a, **kw: pytest.fail("changed checkpoint reached data access"))
    with pytest.raises((ValueError, FileNotFoundError)):
        metrics.run(arguments(tmp_path, path))
    assert not (tmp_path / "result.json").exists()


def test_afdb_runs_checksum_annotation_preflight_before_detector_import(tmp_path, monkeypatch):
    calls = []

    def reject(data_dir, eligibility_path):
        calls.append((data_dir, eligibility_path))
        raise ValueError("synthetic SHA256 mismatch")

    monkeypatch.setattr(afdb, "prepare_afdb", reject)
    forbid_detector_and_waveform(monkeypatch)
    args = arguments(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        metrics.run(args)
    assert calls == [(args.afdb_dir, args.afdb_eligibility)]


@pytest.mark.parametrize("options", [["--fresh"], ["--baseline", "baseline.json"], ["--prepare-eligible"]])
def test_afdb_cli_rejects_other_protocols_and_gates(tmp_path, monkeypatch, options):
    monkeypatch.setattr(
        sys, "argv", ["af_episode_metrics.py", "--afdb", "--output", str(tmp_path / "out.json"), *options]
    )
    monkeypatch.setattr(metrics, "run", lambda _: pytest.fail("invalid AFDB CLI reached detector runner"))
    monkeypatch.setattr(metrics, "prepare_fresh_eligibility", lambda _: pytest.fail("AFDB prepared fresh eligibility"))
    with pytest.raises(SystemExit) as exc:
        metrics.main()
    assert exc.value.code == 2


def test_afdb_cli_does_not_require_unrelated_suite_manifests(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "argv", ["af_episode_metrics.py", "--afdb", "--output", str(tmp_path / "out.json")])
    monkeypatch.setattr(metrics, "run", seen.append)
    metrics.main()
    assert len(seen) == 1
    assert seen[0].afdb and not seen[0].fresh
    assert seen[0].cases_dir is None


@pytest.mark.parametrize(
    ("source_ref", "gap", "expected", "commit"),
    [
        ("baseline-M-2026-10-12", None, "M", "114b99372b20e80feb6c2bc69f3da0696921b5b7"),
        ("baseline-R-2026-10-12", None, "R", "1d344ae403616056d3bd7888ec097d2f8144f015"),
        ("69c9f43", 120, "V", "69c9f43f14ff3584a5d1f0be243ece3cb4e98160"),
    ],
)
def test_afdb_accepts_only_the_frozen_candidate_commits(tmp_path, monkeypatch, source_ref, gap, expected, commit):
    verified = []

    def verify(source, resolved):
        verified.append((source, resolved))
        return {"src/frozen.py": "synthetic-hash"}

    monkeypatch.setattr(metrics, "verify_snapshot_source", verify)
    args = arguments(tmp_path)
    args.source_ref, args.variant_gap = source_ref, gap
    assert metrics.verify_afdb_snapshot(args) == (expected, {"src/frozen.py": "synthetic-hash"})
    assert verified == [(args.source_root, commit)]


@pytest.mark.parametrize(
    ("source_ref", "gap"),
    [
        ("baseline-M-2026-10-12", 120),
        ("baseline-R-2026-10-12", 120),
        ("69c9f43", None),
        ("69c9f43", 60),
        ("69c9f43", 300),
        ("7a97b80", None),  # Earlier M source ref lacks the frozen baseline burden contract.
        ("22f74ad", None),
    ],
)
def test_afdb_refuses_other_snapshots_and_parameter_overrides(tmp_path, monkeypatch, source_ref, gap):
    monkeypatch.setattr(metrics, "verify_snapshot_source", lambda *a: pytest.fail("unfrozen snapshot accepted"))
    args = arguments(tmp_path)
    args.source_ref, args.variant_gap = source_ref, gap
    with pytest.raises(ValueError):
        metrics.verify_afdb_snapshot(args)


@pytest.mark.parametrize("change", ["unchanged", "modified", "extra"])
def test_snapshot_source_must_be_unmodified_without_extra_python_modules(tmp_path, monkeypatch, change):
    source = tmp_path / "src"
    source.mkdir()
    original = b"frozen = True\n"
    path = source / "frozen.py"
    path.write_bytes(b"frozen = False\n" if change == "modified" else original)
    if change == "extra":
        (source / "extra.py").write_text("injected = True\n")

    def git_output(command, **kwargs):
        if command[1] == "ls-tree":
            return "src/frozen.py\n"
        assert command == ["git", "show", "synthetic-commit:src/frozen.py"]
        return original

    monkeypatch.setattr(metrics.subprocess, "check_output", git_output)
    if change == "unchanged":
        assert metrics.verify_snapshot_source(tmp_path, "synthetic-commit") == {"src/frozen.py": metrics.sha256(path)}
    else:
        with pytest.raises(ValueError):
            metrics.verify_snapshot_source(tmp_path, "synthetic-commit")


def test_afdb_runner_reads_atr_only_and_scores_channel_zero_for_all_23(tmp_path, monkeypatch, evidence):
    import wfdb

    import tsmcp.core.io.wfdb as waveform
    import tsmcp.plugins.ecg as ecg

    permitted = metrics.afdb_records(evidence)
    for row in evidence["records"].values():
        row.update(fs=250, n_samples=250000, dur_s=1000)
    monkeypatch.setattr(metrics, "load_afdb_eligibility", lambda _: (permitted, evidence))
    monkeypatch.setattr(metrics, "verify_afdb_snapshot", lambda _: ("R", {"synthetic": "verified"}))
    annotation_calls, waveform_calls, detector_calls = [], [], []
    annotation = SimpleNamespace(sample=np.array([30001, 200001]), symbol=["+", "+"], aux_note=["(AFL", "(N"])
    first = SimpleNamespace(data=np.zeros(250000), fs=250, n=250000, duration=1000, unit="mV", name="ECG1")
    second = SimpleNamespace(data=np.ones(250000), fs=250, n=250000, duration=1000, unit="mV", name="MLII")

    def read_annotation(path, extension):
        annotation_calls.append((Path(path).name, extension))
        assert extension == "atr"
        return annotation

    def read_waveform(path, *args):
        waveform_calls.append(path.name)
        return [first, second], {}

    def irregular(data, fs, unit, *, ch):
        assert ch is first and data is first.data and fs == 250 and unit == "mV"
        detector_calls.append(ch)
        return [{"t": 120.004, "dur": 680}]

    def burden(data, fs, unit, *, ch):
        assert ch is first and data is first.data and fs == 250 and unit == "mV"
        return {"pct_of_window": 68, "pct_of_readable": 85, "value": 85}

    def noisy(data, fs, unit, *, ch):
        assert ch is first and data is first.data and fs == 250 and unit == "mV"
        return [{"t": 800, "dur": 200}]

    monkeypatch.setattr(wfdb, "rdann", read_annotation)
    monkeypatch.setattr(waveform, "load_wfdb", read_waveform)
    monkeypatch.setattr(ecg, "ecg_irregular", irregular)
    monkeypatch.setattr(ecg, "ecg_af_burden", burden)
    monkeypatch.setattr(ecg, "ecg_noisy", noisy)
    args = arguments(tmp_path)
    metrics.run(args)
    result = json.loads(args.output.read_text())
    assert annotation_calls == [(rec, "atr") for rec in RECORDS]
    assert waveform_calls == [f"{rec}.hea" for rec in RECORDS]
    assert len(detector_calls) == 23
    assert result["counts"] == {"longest_af": 15, "af_burden": 13}
    assert result["summary"]["n_records"] == 23
    assert result["summary_onset_eligible"]["n_records"] == 15
    assert result["summary_burden_eligible"]["n_records"] == 13
    assert result["summary_onset_eligible"]["onset_correct"] == 15
    assert result["summary"]["pooled_sensitivity"] == 1
    assert result["summary"]["pooled_ppv"] == 1
    assert result["crashes"] == []
    assert "gates" not in result and "runtime_24h" not in result
    for row in result["records"]:
        assert row["channel_index"] == 0 and row["fs"] == 250
        assert row["longest_annotated"] == [120.004, 800.004]
        assert row["longest_detected"] == [120.004, 800.004]
        assert row["longest_iou"] == 1
        assert row["burden_error_window_pp"] == 0
        assert row["pct_of_window"] == 68
        assert row["pct_of_readable"] == 85
    assert not args.output.with_suffix(".partial.json").exists()
    with pytest.raises(FileExistsError):
        metrics.run(args)
    assert len(detector_calls) == 23


def test_250_hz_truth_merges_only_below_30_seconds_and_iou_uses_episode_spans():
    # AFIB -> AFL is continuous AF; J is not AF. Initial unlabeled samples are non-AF.
    annotation = SimpleNamespace(
        sample=np.array([30001, 55001, 80001, 87500, 125001, 132501, 157501]),
        symbol=["+"] * 7,
        aux_note=["(AFIB", "(AFL", "(J", "(AFIB", "(N", "(AFL", "(N"],
    )
    truth = metrics.af_segments(annotation, 250, 250000)
    assert truth == [(120.004, 220.004), (220.004, 320.004), (350.0, 500.004), (530.004, 630.004)]
    episodes = [{"t": 130.004, "dur": 350, "intervals_s": [(130.004, 300.004), (360.004, 480.004)]}]
    row = metrics.score_record("afdb/synthetic", 1000, truth, episodes, [(0, 800)])
    assert row["annotated_episodes"] == [(120.004, 500.004), (530.004, 630.004)]
    assert row["longest_iou"] == pytest.approx(350 / 380)
    assert row["longest_onset_error_s"] == pytest.approx(10)
    assert row["longest_within_tolerance"] is True
    assert row["fragmentation"] == 1
    assert row["af_s"] == pytest.approx(290)
    assert row["true_af_s"] == pytest.approx(450.004)
    assert row["burden_error_window_pp"] == pytest.approx(-16.0004)
    assert row["sensitivity"] == pytest.approx(290 / 450.004)
    assert row["ppv"] == pytest.approx(1)


@pytest.mark.parametrize(
    ("truth", "episodes", "iou", "correct"),
    [
        ([(10, 90)], [], 0, False),
        ([(10, 90)], [{"t": 100, "dur": 10}], 0, False),
        ([(10, 90)], [{"t": 0, "dur": 100}], 0.8, True),
        ([], [{"t": 0, "dur": 100}], None, None),
        ([], [], None, None),
    ],
)
def test_longest_iou_handles_misses_disjoint_spans_and_absent_truth(truth, episodes, iou, correct):
    row = metrics.score_record("afdb/synthetic", 120, truth, episodes, [(0, 120)])
    assert row["longest_iou"] == iou
    assert row["longest_within_tolerance"] is correct


@pytest.mark.parametrize(("onset", "expected"), [(180.004, True), (180.008, False)])
def test_onset_tolerance_keeps_the_unchanged_grader_boundary(onset, expected):
    row = metrics.score_record("afdb/synthetic", 1000, [(120.004, 800)], [{"t": onset, "dur": 400}], [(0, 1000)])
    assert metrics.LONG_AF_TOL_S == 60
    assert row["longest_within_tolerance"] is expected


def test_numpy_episode_coordinates_produce_json_serializable_fragmentation():
    row = metrics.score_record(
        "afdb/synthetic", 1000, [(120, 800)], [{"t": np.float64(130), "dur": np.float64(400)}], [(0, 1000)]
    )
    assert type(row["fragmentation"]) is int
    assert json.loads(json.dumps(row, allow_nan=False))["fragmentation"] == 1
