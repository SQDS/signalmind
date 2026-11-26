import numpy as np
import pytest

from tsmcp.cleaning.derived_measurements import accepted_chain, derived_rows
from tsmcp.cleaning.repairability import PROFILE, RepairDomain
from tsmcp.cleaning.tasks import probe


@pytest.mark.parametrize("task", ["ECG", "PPG", "ABP"])
def test_research_intervals_use_released_lineage_and_never_add_native_coverage(task):
    t = np.arange(2400) / 125
    phase = np.mod(t, 1)
    x = np.exp(-(((phase - 0.1) / 0.025) ** 2)) if task == "ECG" else 85 + 12 * np.sin(2 * np.pi * t)
    valid = np.ones(len(x), bool)
    est = ~valid
    src = probe(x, t, 125, valid, est, task)
    a, b = 801, 803
    est[a:b] = True
    y = x.copy()
    y[a:b] += 0.001
    released = probe(y, t, 125, valid, est, task)
    c = dict(
        id="c",
        task=task,
        fs=125,
        unit=PROFILE["modalities"][task]["unit"],
        field=PROFILE["modalities"][task]["field"],
        has_source_time=True,
    )
    domain = RepairDomain(
        {
            "repair_domain": dict(
                profile_id=PROFILE["id"],
                profile_version=PROFILE["version"],
                species="human",
                device="Vital",
                use="development",
                goals={task: PROFILE["modalities"][task]["goals"][0]},
            )
        }
    )
    patch = dict(
        idempotency_key="p1",
        plan=dict(parent_version_id="original", write_ranges=[dict(block_id="c", start_sample=a, end_sample=b)]),
    )
    patches = {"new": patch}
    art = dict(artifact_id="new", content_hash="h", source_sha256="input")
    rows, counts = derived_rows(c, (x, t, y, valid, est), src, released, art, accepted_chain("new", patches), domain)
    assert rows and all(r["accepted_patch_ids"] == ["p1"] and not r["released"] for r in rows)
    assert counts["native_samples"] == released["measured_samples"] and counts["released_derived_samples"] == 0
    assert counts["native_samples"] <= counts["union_samples"] <= counts["native_samples"] + counts["research_samples"]
    if task == "ABP":
        m = rows[0]["quantities"]
        assert set(m) == {"sbp", "dbp", "mean_pressure"}
        assert m["sbp"]["comparison_range"][1] + 1 == m["mean_pressure"]["integration_range"][1]
        assert m["mean_pressure"]["estimated_samples"] == 2
    else:
        assert rows[0]["validation_needed"] == "event_identity_and_hidden_beats"
    restored, _ = derived_rows(
        c,
        (x, t, x, valid, ~valid),
        src,
        src,
        dict(art, artifact_id="original"),
        accepted_chain("original", patches),
        domain,
    )
    assert restored == []
