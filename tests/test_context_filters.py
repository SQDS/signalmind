import numpy as np
import pytest

from tsmcp.cleaning.filter_retest import filter_retest
from tsmcp.cleaning.quality_gate import waveform_gate
from tsmcp.cleaning.repairability import PROFILE
from tsmcp.cleaning.tasks import probe
from tsmcp.processing import ProcessingSession


@pytest.mark.parametrize(
    "method,params",
    [
        ("notch", dict(frequency_hz=50)),
        ("remove_tone", dict(frequency_hz=50)),
        ("lowpass", dict(frequency_hz=15)),
        ("local_detrend", dict(window_s=0.4)),
    ],
)
def test_contextual_filters_write_only_authorized_samples(method, params):
    t = np.arange(2000) / 125
    x = 90 + 12 * np.sin(2 * np.pi * t) + 0.3 * np.sin(2 * np.pi * 50 * t)
    task = "ECG" if method == "local_detrend" else "ABP"
    s = ProcessingSession(x, fs=125, channel=task, unit="u")
    a, b = 501, 1399
    state = s.process(s.raw.id, method, [[a, b]], context_samples=125, **params)
    op = state.operations[-1]
    n = op["parameters"].get("anchor_samples", 125)
    assert op["parameters"]["read_ranges_samples"] == [[a - n, b + n]]
    assert not state.artifact[:a].any() and not state.artifact[b:].any()
    np.testing.assert_array_equal(state.values[~state.artifact], x[~state.artifact])
    assert not state.repaired.any()
    with pytest.raises(ValueError, match="observations"):
        s.process(s.raw.id, method, [[0, b]], context_samples=125, **params)
    if task == "ABP":
        before = (t, x, s.raw.valid, s.raw.repaired, probe(x, t, 125, s.raw.valid, s.raw.repaired, task))
        after = (
            t,
            state.values,
            state.valid,
            state.repaired,
            probe(state.values, t, 125, state.valid, state.repaired, task),
        )
        case = dict(task=task, fs=125)
        hard = waveform_gate(case, before, after, state.operations, [[a, b]], filter_policy=PROFILE["filter_policy"])
        assert hard["status"] == "passed", hard
        shifted = state.values.copy()
        shifted[a:b] += 10
        bad = (
            *after[:1],
            shifted,
            state.valid,
            state.repaired,
            probe(shifted, t, 125, state.valid, state.repaired, task),
        )
        checks, _ = filter_retest(case, before, bad, op, PROFILE["filter_policy"])
        assert not checks["filter_abp_dc"] and not checks["filter_seams"]
        case["source_time_conflicts"] = [[b + 24, b + 25]]
        assert not filter_retest(case, before, after, op, PROFILE["filter_policy"])[0]["observed_context"]


def test_record_uses_bound_context_filter_and_retests_actual_native_pressure(tmp_path):
    import time
    from types import SimpleNamespace

    from tsmcp.cleaning.common import array_hash, read, sha
    from tsmcp.cleaning.dataset_pipeline import Record
    from tsmcp.cleaning.engine import evaluate_output

    fs = 125
    t = np.arange(2000) / fs
    x = 90 + 12 * np.sin(2 * np.pi * t) + 0.3 * np.sin(2 * np.pi * 50 * t)
    v = np.ones(len(t), bool)
    p = tmp_path / "public/c/input.npz"
    p.parent.mkdir(parents=True)
    np.savez(p, values=x, time=t, valid=v)
    c = dict(
        id="c",
        record_id="r",
        task="ABP",
        fs=fs,
        n=len(t),
        field="ABP_Raw",
        unit="mmHg",
        has_source_time=True,
        input_sha256=sha(p),
    )
    raw = evaluate_output(tmp_path, c, dict(hash="raw"), "fixture", None, x, t, x, x, v, ~v, [], time.monotonic())
    cfg = dict(
        write_views=2,
        repair_protocol="v1.1",
        repair_domain=dict(
            profile_id=PROFILE["id"],
            profile_version=PROFILE["version"],
            device="Vital",
            species="human",
            use="development",
            goals={"ABP": "ABP_pressure"},
        ),
    )
    reviewer = SimpleNamespace(
        decide=lambda stage, rows: {r["id"]: dict(decision="accept", reason="fixture", receipt="checked") for r in rows}
    )
    record = Record(tmp_path, [c], {"identity": {"c": raw}}, cfg, reviewer, "fixture")
    a, b = 501, 1399
    h = dict(
        id="line",
        case="c",
        reason="line_50",
        wave_requested="remove_tone",
        decision_source="fixture",
        a=a,
        b=b,
        measurement="keep",
        context_digest=array_hash(x, v, ~v),
    )
    final = record.filter_trials(record.raw, [h])
    assert final["c"]["parent_artifact_id"] == raw["artifact_id"]
    row = read(record.path / "filter-trials.json")[0]
    assert row["status"] == "accepted" and all(
        row["hard_gate"]["checks"]["filter_abp_" + m] for m in ["sbp", "dbp", "mean_pressure", "dc"]
    )
