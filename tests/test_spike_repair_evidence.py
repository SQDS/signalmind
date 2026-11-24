from types import SimpleNamespace

import numpy as np
from PIL import Image

from tsmcp.cleaning.common import read
from tsmcp.cleaning.dataset_evidence import view
from tsmcp.cleaning.dataset_pipeline import Record
from tsmcp.cleaning.dataset_review import FIELDS, Reviewer
from tsmcp.cleaning.spike_evidence import phase_neighbours


def test_same_phase_uses_other_observed_cycles_and_respects_breaks():
    fs = 125
    x = np.sin(2 * np.pi * np.arange(1200) / 100)
    events = [{"sample": i} for i in range(50, len(x), 100)]
    a, b = 568, 570
    good = np.ones(len(x), bool)
    idx, ref, cycles, why = phase_neighbours(x, good, events, fs, a, b)
    assert why is None and len(cycles) == 6
    assert all(pair != [550, 650] for pair in cycles)
    np.testing.assert_allclose(np.median(ref, axis=0), x[idx], atol=1e-12)
    disturbed = x.copy()
    disturbed[a:b] += [2, -2]
    _, ref2, _, _ = phase_neighbours(disturbed, good, events, fs, a, b)
    np.testing.assert_array_equal(ref, ref2)
    good[[549, 651]] = False
    assert phase_neighbours(x, good, events, fs, a, b)[3] is not None


def test_spike_card_shows_estimated_values_even_if_no_interval_is_published(tmp_path):
    fs = 125
    t = np.arange(1250) / fs
    truth = np.sin(2 * np.pi * t * 1.25)
    x = truth.copy()
    a, b = 568, 570
    x[a:b] += [2, -2]
    y = x.copy()
    y[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
    good = np.ones(len(t), bool)
    e = np.zeros(len(t), bool)
    e[a:b] = True
    pr = dict(events=[{"sample": i} for i in range(50, len(t), 100)])
    c = dict(id="c", record_id="r", fs=fs, task="ECG", unit="source_units")
    markers = [450, 550, 650]
    out = tmp_path / "spike.png"
    report = view(
        out,
        dict(
            case="c",
            start=t[a],
            end=t[b],
            core_samples=[a, b],
            spike_retest=dict(before_landmarks=markers, after_landmarks=markers),
        ),
        {"c": c},
        lambda k: (t, x, good, ~good, pr),
        lambda k: (t, y, good, e, {"events": []}),
    )
    assert (
        out.stat().st_size > 0
        and report["candidate_estimated_samples"] == 2
        and report["candidate_invalid_samples"] == 0
    )
    assert report["parent_landmarks_in_core"] == report["candidate_landmarks_in_core"] == []
    assert report["preceding_native_ms"] == -144 and report["following_native_ms"] == 656
    np.testing.assert_array_equal(report["candidate_core"], y[a:b])
    assert report["neighbour_phase"]["candidate_mae_to_median"] < report["neighbour_phase"]["observed_mae_to_median"]


def test_same_group_keeps_both_protected_and_eligible_members_but_only_trials_eligible(tmp_path):
    c = dict(id="c", record_id="r", fs=125, task="ECG", unit="u")
    hits = []
    for i, status in enumerate(
        ("possible_physiology", "evaluated_no_alert", "possible_physiology", "insufficient_support")
    ):
        hits.append(
            dict(
                id=f"h{i}",
                case="c",
                task="ECG",
                reason="isolated_spike_candidate",
                value=1.0,
                a=100 * i + 40,
                b=100 * i + 42,
                features={k: float(i) for k in FIELDS},
                peer_bpm={},
                anchor_values=[0.0, 1.0, 1.0, 0.0],
                support_status="suspected_artifact" if status == "evaluated_no_alert" else status,
                applicability={"status": status},
            )
        )

    def decide(stage, rows):
        assert stage == "groups" and len(rows) == 1
        r = rows[0]
        data = r["data"]
        col = data["columns"].index("trial_eligible")
        assert [z[col] for z in data["member_rows"]] == [0, 1, 0, 0]
        assert data["image_rows"][0] == 1 and data["image_rows"][1] != 1 and len(r["images"]) == 2
        assert data["trial_scope"]["eligible"] == 1
        return {
            r["id"]: dict(
                wave="interpolate_spike",
                wave_if=[],
                measure_if=[],
                measure="keep",
                otherwise="keep",
                diagnosis={"label": "mixed"},
                trial_diagnosis={"label": "artifact"},
                receipt="test-request",
            )
        }

    r = Record(
        tmp_path,
        [c],
        {"identity": {"c": {"content_hash": "raw"}}},
        {"purpose": "RR", "routing": "reason"},
        SimpleNamespace(enabled=True, decide=decide),
        "fixture",
    )
    image = tmp_path / "image.png"
    Image.new("RGB", (30, 20)).save(image)
    r.picture = lambda *args: (image, {})
    t = np.arange(500) / 125
    groups = r.groups(hits, lambda k: (t, np.sin(t), None, None, None))
    assert len(groups) == 1 and [h["wave_requested"] for h in hits] == [
        "preserve",
        "interpolate_spike",
        "preserve",
        "preserve",
    ]
    assert len({h["group"] for h in hits}) == 1 and hits[1]["diagnosis"]["label"] == "mixed"


def test_zero_match_spike_policy_is_corrected_on_real_page_rows(tmp_path):
    image = tmp_path / "view.png"
    Image.new("RGB", (25, 10)).save(image)

    def diagnosis(label):
        return dict(label=label, confidence=0.8, evidence_refs=["view_1"], alternative_hypotheses=[])

    reviewer = Reviewer(tmp_path, {"method": "dataset_agent", "model": "fixture"})
    prompts = []

    def request(messages, tools, system):
        prompts.append(messages)
        if len(prompts) == 1:
            result = {
                "items": [
                    dict(id="g", diagnosis=diagnosis("mixed"), trial_diagnosis=diagnosis("artifact"), reason="fixture")
                ]
            }
        else:
            item = dict(
                wave="interpolate_spike",
                parameters={},
                wave_if=[dict(field="seconds", op="ge", value=0.03)],
                measure="review",
                measure_if=[],
                otherwise="keep",
                reason="fixture",
            )
            if len(prompts) == 2:
                result = {"items": [dict(item, id="g")]}
            else:
                assert "No eligible member" in messages[0]["content"][-1]["text"]
                result = {"row1": dict(item, wave_if=[dict(field="seconds", op="le", value=0.02)])}
        call = dict(type="tool_use", name="submit", input=result)
        return SimpleNamespace(id=f"test-{len(prompts)}", content=[SimpleNamespace(model_dump=lambda **k: call)])

    reviewer.client = SimpleNamespace(request=request)
    data = dict(
        task="ECG",
        trigger="isolated_spike_candidate",
        evidence_ids=["view_1", "view_2", "members"],
        trial_scope=dict(eligible=1, protected=1),
        columns=["seconds", "trial_eligible"],
        member_rows=[[0.032, 0], [0.016, 1]],
    )
    rows = [dict(id="g", images=[image, image], data=data)]
    decision = reviewer.decide("groups", rows)["g"]
    assert (
        len(prompts) == 3
        and decision["diagnosis"]["label"] == "mixed"
        and decision["trial_diagnosis"]["label"] == "artifact"
    )
    assert decision["wave_if"][0]["op"] == "le"
    assert read(tmp_path / "dataset-receipts" / (decision["receipt"] + ".json"))["submission_format"] == "ordered_keys"
    replay = Reviewer(tmp_path / "replay", dict(reviewer.config, replay_from=str(tmp_path)))
    assert replay.decide("groups", rows)["g"] == decision
