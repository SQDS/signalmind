from types import SimpleNamespace

import numpy as np
import pytest

from tsmcp.cleaning.common import read, sha
from tsmcp.cleaning.dataset_pipeline import Record
from tsmcp.cleaning.engine import evaluate_output, load_arrays
from tsmcp.cleaning.repairability import PROFILE, domain_for
from tsmcp.cleaning.tasks import probe


def declaration(goal="PPG_PPI"):
    return dict(
        repair_protocol="v1.1",
        repair_domain=dict(
            profile_id=PROFILE["id"],
            profile_version=PROFILE["version"],
            species="human",
            device="Vital",
            use="development",
            goals={"PPG": goal},
        ),
        write_views=5,
    )


def inputs():
    t = np.arange(1500) / 125
    y = 45 + np.sin(2 * np.pi * t)
    y[405:407] += 4
    v = np.ones(len(y), bool)
    e = ~v
    c = dict(
        id="c",
        record_id="r",
        task="PPG",
        fs=125,
        unit="source_units",
        field="PPG_Record",
        n=len(t),
        has_source_time=True,
        normalized_windows=False,
    )
    return c, (t, y, v, e, probe(y, t, 125, v, e, "PPG"))


def test_profile_has_no_species_rate_unit_or_goal_fallback():
    c, before = inputs()
    domain = domain_for(declaration())
    allowed = domain.assess(c, 405, 409, before, "interpolate")
    assert allowed["status"] == "eligible" and allowed["duration_s"] == 0.032 and allowed["period_fraction"] > 0
    bad = declaration()
    bad["repair_domain"]["species"] = "rat"
    other = domain_for(bad).assess(dict(c, fs=1000, unit="unknown"), 405, 409, before, "interpolate")
    assert other["duration_s"] == 0.004 and other["status"] == "unsupported"
    assert set(other["reason_codes"]) >= {
        "species_outside_profile",
        "fs_outside_profile",
        "native_unit_or_field_unverified",
    }
    assert domain_for({"repair_protocol": "v1.1"}).assess(c, 405, 409, before, "interpolate")["status"] == "unsupported"
    ecg = dict(c, task="ECG", field="ECG_Record")
    cfg = declaration()
    cfg["repair_domain"]["goals"] = {"ECG": "ECG_morphology"}
    assert "goal_uncalibrated" in domain_for(cfg).metadata(ecg)["reason_codes"]
    cfg["repair_domain"]["goals"] = {"ECG": "ECG_RR"}
    assert not domain_for(cfg).metadata(ecg)["reason_codes"]
    assert domain_for({}) is None


@pytest.mark.parametrize("known,decision", [(True, "accept"), (True, "rollback"), (False, None)])
def test_profile_controls_real_trial_and_binds_published_patch(tmp_path, known, decision):
    import time

    c, before = inputs()
    t, x, v, e, pr = before
    folder = tmp_path / "public/c"
    folder.mkdir(parents=True)
    np.savez(folder / "input.npz", time=t, values=x, valid=v)
    c["input_sha256"] = sha(folder / "input.npz")
    art = evaluate_output(tmp_path, c, {"hash": "identity"}, "fixture", None, x, t, x, x, v, e, [], time.monotonic())
    config = declaration()
    if not known:
        config["repair_domain"]["device"] = "unspecified"

    def decide(stage, rows):
        assert known
        return {
            r["id"]: dict(decision=decision, receipt="fixture", reason="fixture", request_id="fixture") for r in rows
        }

    record = Record(tmp_path, [c], {"identity": {"c": art}}, config, SimpleNamespace(decide=decide), "test")
    h = dict(
        id="short",
        case="c",
        a=405,
        b=407,
        reason="isolated_spike_candidate",
        wave_requested="interpolate_spike",
        decision_source="fixture",
        group="group",
        measurement="keep",
    )
    final = record.spike_trials(record.raw, [h])
    trow = read(record.path / "spike-trials.json")[0]
    _, _, y, _, est = load_arrays(tmp_path, c, final["c"])
    if known:
        comp = trow["comparison"]
        assert comp["identity"] == art["artifact_id"]
        assert [r["method"] for r in comp["alternatives"]] == ["interpolate", "interpolate_pchip"]
        for branch in comp["alternatives"]:
            artifact = read(tmp_path / "artifacts" / branch["candidate"] / "artifact.json")
            assert artifact["parent_artifact_id"] == comp["identity"] == branch["parent"]
            assert branch["hard_gate"]["checks"]["interpolation_support"]
        assert len({b["plan_id"] for b in comp["alternatives"]}) == 2
        if decision == "rollback":
            assert final["c"] == art and comp["released_method"] == "identity" and not est.any()
            assert not list((record.path / "patches").glob("*.json"))
            return
        assert trow["status"] == "accepted" and est.sum() == 2 and (x[405:407] != y[405:407]).all()
        np.testing.assert_array_equal(x[~est], y[~est])
        again = record.spike_trials(record.raw, [h])
        assert again["c"]["artifact_id"] == final["c"]["artifact_id"] == comp["released"]
        assert len(list((record.path / "patches").glob("*.json"))) == 1
        patch = read(next((record.path / "patches").glob("*.json")))
        assert patch["plan"]["tool"]["parameters"]["repair_domain"]["profile_version"] == PROFILE["version"]
        assert patch["receipt"]["acceptance_spec_id"] == record.repair_domain.spec_id
        from tsmcp.cleaning.common import write
        from tsmcp.cleaning.derived_measurements import export_derived
        from tsmcp.cleaning.rollback import rollback

        out = tmp_path / "export"
        out.mkdir()
        idx = export_derived(tmp_path, [c], final, record.raw, record.repair_domain, "selected", out)
        assert idx["rows"] >= 1
        head = final["c"]
        write(
            out / "waveform_manifest.json",
            {"c": dict(artifact_id=head["artifact_id"], waveform=head["arrays"], probe=head["probe_path"])},
        )
        (out / "measurements.csv").write_text("case,quality,waveform_version\n")
        write(tmp_path / "dataset_manifest.json", {"cases": [c]})
        write(tmp_path / "agent_state.json", {"finalized": True})
        write(tmp_path / "baselines.json", {"identity": "original"})
        write(tmp_path / "versions/original/version.json", {"results": record.raw})
        write(tmp_path / "acceptance-spec.json", {"deployment": config["repair_domain"]})
        undone = rollback(tmp_path, patch["idempotency_key"], tmp_path.parent / (tmp_path.name + "-undo"))
        assert undone["derived_measurements"]["rows"] == 0 and head["artifact_id"] in undone["invalidated_descendants"]
        assert read(out / "derived-index.json") == idx
    else:
        assert not est.any() and np.array_equal(x, y) and "candidate" not in trow
        assert "device_outside_profile" in trow["reason_codes"]


def test_profile_records_physical_gap_and_missing_anchors():
    c, (t, x, v, e, pr) = inputs()
    domain = domain_for(declaration())
    grid = t.copy()
    grid[406:] += 1
    assert "time_discontinuity" in domain.assess(c, 405, 407, (grid, x, v, e, pr), "interpolate")["reason_codes"]
    valid = v.copy()
    valid[404] = False
    d = domain.assess(c, 405, 407, (t, x, valid, e, pr), "interpolate")
    assert not d["observed_anchors"] and d["status"] == "need_evidence" and "anchor_unsupported" in d["reason_codes"]
    valid[405:407] = False
    assert domain.assess(c, 405, 407, (t, x, valid, e, pr), "interpolate")["status"] == "unrecoverable_for_goal"
