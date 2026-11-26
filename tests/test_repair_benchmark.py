import json

import numpy as np
import pytest

from eval.repair_benchmark.executor import apply_plan
from eval.repair_benchmark.public import prepare
from eval.repair_benchmark.runner import baseline, isolated
from eval.repair_benchmark.scoring import report, score_case
from tsmcp.cleaning.common import sha, write


def test_public_plan_binding_actual_artifacts_and_fixed_denominator(tmp_path):
    t = np.arange(2500) / 125
    ref = np.sin(2 * np.pi * t)
    x = ref.copy()
    x[955:957] += [10, -10]
    valid = np.ones(len(t), bool)
    public = tmp_path / "public"
    case = prepare(public, "s", "PPG", "u", 125, t, x, valid, ~valid, offset=2000)
    assert any(z["start_sample"] == 2955 for z in case["candidates"])
    plan = baseline(case, "interpolate_pchip")
    result = isolated(public, plan, tmp_path / "pchip", "pchip")
    assert result["status"] == "completed" and result["usage"]["model_requests"] == 0
    stale = dict(plan, parent_version_id="older")
    assert isolated(public, stale, tmp_path / "invalid", "invalid")["status"] == "invalid_output"
    submitted = tmp_path / "wave.npz"
    damaged_core = np.zeros_like(valid)
    damaged_core[955:957] = True
    np.savez(submitted, values=ref, valid=valid, estimated=damaged_core, time=t, unit="u")
    binding = dict(
        kind="submit_waveform",
        input_hash=case["input_hash"],
        parent_version_id=case["parent_version_id"],
        sha256=sha(submitted),
    )
    direct = isolated(public, binding, tmp_path / "external", "external", submitted)
    assert direct["status"] == "completed" and direct["decision"] == "waveform_repair"
    tampered = ref.copy()
    tampered[970] += 1
    np.savez(submitted, values=tampered, valid=valid, estimated=damaged_core, time=t, unit="u")
    rejected = isolated(public, dict(binding, sha256=sha(submitted)), tmp_path / "bad-array", "external", submitted)
    assert rejected["status"] == "invalid_output" and rejected["decision"] == "preserve"
    plan["actions"][0]["write_ranges"][0]["start_sample"] -= 1
    with pytest.raises(ValueError, match="outside"):
        apply_plan(case, plan, t, x, valid, ~valid)
    damaged = valid.copy()
    damaged[:955] = damaged[957:] = False
    private = tmp_path / "reference.npz"
    np.savez(private, reference=ref, damaged=damaged)
    arms = ["pchip", "invalid", "absent"]
    row = dict(
        case_id="s",
        subject="a",
        record="r",
        noise_source="wave",
        split="dev",
        public_dir="public",
        input_hash=case["input_hash"],
        private_npz=private.name,
        private_sha256=sha(private),
        outputs={k: k for k in arms},
    )
    path = tmp_path / "manifest.json"
    write(path, dict(cases=[row], arms=arms))
    out = report(path)["rows"]
    assert [r["status"] for r in out] == ["completed", "invalid_output", "missing"]
    assert [r["core_sites"] for r in out] == [1, 1, 1] and [r["recovered_sites"] for r in out] == [1, 0, 0]
    assert out[0]["all_core_mae"] < out[1]["all_core_mae"] == out[2]["all_core_mae"]
    assert not (tmp_path / "pchip" / "reference.npz").exists()
    assert "private" not in json.dumps(case)


def test_private_score_direction_for_perfect_waveform_and_landmark_damage():
    t = np.arange(2500) / 125
    phase = np.mod(t, 1)
    ref = np.exp(-(((phase - 0.15) / 0.028) ** 2))
    x = ref.copy()
    x[701:703] += 4
    core = np.zeros(len(t), bool)
    core[701:703] = True
    good = np.ones(len(t), bool)
    case = dict(task="ECG", fs=125, offset=0, allowed_write_ranges=[dict(start_sample=701, end_sample=703)])
    truth = dict(damaged=core, reference=ref, event_samples=np.flatnonzero(ref > 0.99))
    before = (t, x, good, ~good)
    exact = score_case(case, before, (ref, good, core), truth, "completed")
    damaged = ref.copy()
    damaged[890:920] = 0
    harmed = score_case(case, before, (damaged, good, core), truth, "completed")
    assert exact["all_core_mae"] == 0 and exact["recovered_sites"] == 1
    assert harmed["normal_changed_samples"] > 0 and harmed["outside_write_changes"] > 0
    assert (
        harmed["native_events_vs_reference"]["false_negative"] > exact["native_events_vs_reference"]["false_negative"]
    )
    unlabelled = score_case(case, before, (ref, good, core), {}, "completed")
    assert unlabelled["core_sites"] is None and unlabelled["all_core_mae"] is None
    assert unlabelled["measured_support_samples"] == exact["measured_support_samples"]
