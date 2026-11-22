import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import af_episode_metrics as metrics  # noqa: E402


def population():
    inventory = json.loads(metrics.FRESH_INVENTORY.read_text())
    split = json.loads((metrics.ROOT / "eval/tasks/ltafdb_split.json").read_text())
    excluded = {"mitdb": set(), "ltafdb": set()}
    for manifest in inventory["suite_manifests"].values():
        for db, records in manifest["records"].items():
            excluded[db].update(records)
    return inventory, split, excluded


@pytest.mark.parametrize("name", ["mitdb/203", "ltafdb/00", "ltafdb/07", "ltafdb/999", "ltafdb/../00"])
def test_fresh_refuses_records_outside_inventory(name):
    permitted = metrics.fresh_records(*population())
    assert len(permitted) == 38
    metrics.require_fresh_record("ltafdb/01", permitted)
    with pytest.raises(ValueError, match="outside frozen fresh allowlist"):
        metrics.require_fresh_record(name, permitted)


@pytest.mark.parametrize("change", ["dev", "test_ext", "suite", "unknown", "duplicate", "missing"])
def test_fresh_inventory_fails_closed_on_exclusions_and_population_changes(change):
    inventory, split, excluded = population()
    if change == "missing":
        inventory["records"].pop()
    elif change == "suite":
        excluded["ltafdb"].add(inventory["records"][0]["record"])
    else:
        replacement = {
            "dev": split["dev"][0],
            "test_ext": split["test_ext"][0],
            "unknown": "999",
            "duplicate": inventory["records"][1]["record"],
        }[change]
        inventory["records"][0]["record"] = replacement
    with pytest.raises(ValueError, match="exactly the 38"):
        metrics.fresh_records(inventory, split, excluded)


def test_eligibility_is_annotation_only_and_uses_strict_boundaries(tmp_path, monkeypatch):
    import wfdb

    names = ["01", "03", "06", "08"]
    monkeypatch.setattr(metrics, "fresh_population", lambda _: ([("ltafdb", r) for r in names], {}))
    for rec in names:
        (tmp_path / f"{rec}.hea").write_text("synthetic")
        (tmp_path / f"{rec}.atr").write_text("synthetic")
    # 01: exactly 300 s; 03: exactly 90%; 06: eligible but <24 h; 08: runtime record.
    durations = {"01": 1000, "03": 1000, "06": 1000, "08": 86400}
    ends = {"01": 300, "03": 900, "06": 301, "08": 500}
    monkeypatch.setattr(wfdb, "rdheader", lambda path: SimpleNamespace(fs=1, sig_len=durations[Path(path).name]))
    monkeypatch.setattr(
        wfdb,
        "rdann",
        lambda path, _: SimpleNamespace(
            sample=np.array([0, ends[Path(path).name]]), symbol=["+", "+"], aux_note=["(AFIB", "(N"]
        ),
    )
    monkeypatch.setattr(wfdb, "rdrecord", lambda *_: pytest.fail("waveform read during eligibility"))
    monkeypatch.setattr(np, "fromfile", lambda *_: pytest.fail("waveform read during eligibility"))
    args = SimpleNamespace(cases_dir=tmp_path, ltaf_dir=tmp_path, output=tmp_path / "eligible.json")
    metrics.prepare_fresh_eligibility(args)
    evidence = json.loads(args.output.read_text())
    assert evidence["eligible_records"] == ["ltafdb/06", "ltafdb/08"]
    assert evidence["runtime_record"] == "ltafdb/08"
    with pytest.raises(FileExistsError):
        metrics.prepare_fresh_eligibility(args)


def paired_results():
    def result(correct):
        return {
            "records": [
                metrics.score_record(str(i), 1000, [(0, 400)], [{"t": 0 if hit else 100, "dur": 300}], [(0, 1000)])
                for i, hit in enumerate(correct)
            ],
            "runtime_24h": {"record": "synthetic", "window_s": 86400, "median_s": 10},
            "crashes": [],
        }

    return result([True, False, False, False]), result([False, True, True, True])


def test_fresh_gates_allow_regression_with_net_two_and_fragmentation_tie():
    from eval.af_fresh_gates import compare, exact_mcnemar

    baseline, candidate = paired_results()
    candidate["runtime_24h"]["median_s"] = 15
    gates = compare(baseline, candidate, ["0", "1", "2", "3"])
    assert gates["all_gates_pass"]
    assert gates["G1"]["net_gain"] == 2
    assert gates["G1"]["regressions"] == ["0"]
    assert gates["G1"]["exact_two_sided_mcnemar_p"] == 0.625
    assert gates["G4"]["M_median_fragmentation"] == gates["G4"]["C_median_fragmentation"]
    assert exact_mcnemar(0, 0) == 1
    assert exact_mcnemar(2, 0) == 0.5
    # An AF record outside the eligible list cannot supply a primary-gate gain.
    assert not compare(baseline, candidate, ["0", "1", "2"])["G1"]["pass"]


def test_fresh_secondary_gate_boundaries_and_pairing():
    from copy import deepcopy

    from eval.af_fresh_gates import compare

    baseline, candidate = paired_results()
    eligible = ["0", "1", "2", "3"]
    for row in candidate["records"]:
        row["burden_error_window_pp"] -= 0.5
    assert compare(baseline, candidate, eligible)["G2"]["pass"]
    candidate["records"][0]["burden_error_window_pp"] -= 0.0001
    assert not compare(baseline, candidate, eligible)["G2"]["pass"]
    low_sensitivity = deepcopy(candidate)
    for row in low_sensitivity["records"]:
        row["overlap_s"] = 291.9  # 72.975% versus 75% baseline
    assert not compare(baseline, low_sensitivity, eligible)["G3"]["pass"]
    low_ppv = deepcopy(candidate)
    for row in low_ppv["records"]:
        row["af_s"] = 310  # 96.77% PPV versus 100% baseline, sensitivity unchanged
    assert not compare(baseline, low_ppv, eligible)["G3"]["pass"]
    candidate["runtime_24h"]["median_s"] = 15.001
    assert not compare(baseline, candidate, eligible)["G5"]["pass"]
    candidate["records"].pop()
    with pytest.raises(ValueError, match="populations differ"):
        compare(baseline, candidate, eligible)


def test_fresh_selection_follows_all_three_tiebreakers():
    from copy import deepcopy

    from eval.af_fresh_gates import select

    entry = {"all_gates_pass": True, "G1": {"net_gain": 2}, "G2": {"C_mae_pp": 3.0}}
    comparisons = {"R": deepcopy(entry), "V": deepcopy(entry)}
    assert select(comparisons) == "R"
    comparisons["V"]["G2"]["C_mae_pp"] = 2.9
    assert select(comparisons) == "V"
    comparisons["R"]["G1"]["net_gain"] = 3
    assert select(comparisons) == "R"
    for entry in comparisons.values():
        entry["all_gates_pass"] = False
    assert select(comparisons) == "M"
