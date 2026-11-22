import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.onset_boundary import refine_events  # noqa: E402


def analysis_with(evidence, supported, *, duration=600, readable=None):
    times = np.arange(duration + 1, dtype=float)
    score = np.zeros(duration)
    for start, end in evidence:
        score[start:end] = 1
    return {
        "beats": {"t_s": times},
        "score": score,
        "valid": np.ones(duration, dtype=bool),
        "supported": supported,
        "readable": [[0, duration]] if readable is None else readable,
    }


def event_with(intervals):
    start, end = intervals[0][0], intervals[-1][1]
    return {
        "t": start,
        "start": start,
        "dur": end - start,
        "span_s": end - start,
        "af_s": sum(b - a for a, b in intervals),
        "intervals_s": intervals,
    }


def test_earliest_qualifying_boundary_moves_backward_without_changing_event():
    analysis = analysis_with([[100, 160], [200, 400]], [[130, 160], [200, 400]])
    events = [event_with(analysis["supported"])]
    saved_analysis, saved_events = copy.deepcopy(analysis), copy.deepcopy(events)
    (refined,) = refine_events(analysis, events, 600)
    assert refined["onset_s"] == refined["onset_lo_s"] == refined["onset_hi_s"] == 100
    assert refined["onset_confident"] is True
    for key, value in events[0].items():
        assert refined[key] == value
    assert events == saved_events
    for key in ("score", "valid"):
        np.testing.assert_array_equal(analysis[key], saved_analysis[key])
    assert analysis["supported"] == saved_analysis["supported"]
    refined["intervals_s"][0][0] = -1
    assert events == saved_events


def test_weak_precursor_is_skipped_for_sustained_boundary_inside_event():
    analysis = analysis_with([[100, 110], [200, 400]], [[100, 110], [200, 400]])
    (refined,) = refine_events(analysis, [event_with(analysis["supported"])], 600)
    assert refined["onset_s"] == 200
    assert refined["merged_from"] == 2
    assert refined["unknown_s"] == 0
    assert refined["readable_non_af_gap_s"] == 90
    assert refined["af_s"] == 210
    assert "candidate" in refined["description"] and "gaps" in refined["description"]


@pytest.mark.parametrize(("event_start", "expected"), [(279, 100), (280, None)])
def test_backward_connection_gap_is_strictly_below_120_seconds(event_start, expected):
    # Final support itself connects the preceding evidence; no annotation or
    # alteration of the detector's support is necessary to make that decision.
    analysis = analysis_with([[100, 160]], [[event_start, 400]])
    (refined,) = refine_events(analysis, [event_with(analysis["supported"])], 600)
    assert refined["onset_s"] == expected


def test_forward_boundary_outside_event_is_rejected_and_bracket_is_not_event_clipped():
    analysis = analysis_with([[200, 400]], [[100, 120]])
    (refined,) = refine_events(analysis, [event_with(analysis["supported"])], 600)
    assert refined["onset_s"] is None
    assert refined["onset_confident"] is False
    assert (refined["onset_lo_s"], refined["onset_hi_s"]) == (0, 400)


@pytest.mark.parametrize(("run", "bracket"), [([20, 100], (0, 320)), ([560, 600], (260, 600))])
def test_boundary_requires_complete_context_at_both_record_edges(run, bracket):
    analysis = analysis_with([run], [run])
    (refined,) = refine_events(analysis, [event_with([run])], 600)
    assert refined["onset_s"] is None
    assert (refined["onset_lo_s"], refined["onset_hi_s"]) == bracket


@pytest.mark.parametrize(
    ("prior_evidence_end", "prior_readable_end", "after_readable_end", "after_evidence_end", "expected"),
    [
        (174, 197, 254, 245, 200),  # inclusive 3/27/54/45-second thresholds
        (175, 197, 254, 245, None),
        (174, 196, 254, 245, None),
        (174, 197, 253, 245, None),
        (174, 197, 254, 244, None),
    ],
)
def test_each_declared_coverage_requirement(
    prior_evidence_end, prior_readable_end, after_readable_end, after_evidence_end, expected
):
    support = [[200, after_readable_end], [260, 400]]
    analysis = analysis_with(
        [[171, prior_evidence_end], [200, after_evidence_end]],
        support,
        readable=[[0, prior_readable_end], [200, after_readable_end], [260, 600]],
    )
    (refined,) = refine_events(analysis, [event_with(support)], 600)
    assert refined["onset_s"] == expected
    assert refined["unknown_s"] == 260 - after_readable_end
    assert refined["readable_non_af_gap_s"] == 0


def test_short_runs_cannot_establish_onset_even_with_enough_total_evidence():
    runs = [[100 + i * 10, 109 + i * 10] for i in range(5)]
    analysis = analysis_with(runs, [[100, 150]])
    (refined,) = refine_events(analysis, [event_with([[100, 150]])], 600)
    assert refined["onset_s"] is None


def test_invalid_rr_breaks_boundary_evidence_and_empty_inputs_are_supported():
    analysis = analysis_with([[100, 160]], [[100, 160]])
    analysis["valid"][100:160] = False
    (refined,) = refine_events(analysis, [event_with([[100, 160]])], 600)
    assert refined["onset_s"] is None
    assert refine_events(analysis, [], 600) == []
    empty = {"beats": {"t_s": []}, "score": [], "valid": [], "supported": [], "readable": []}
    assert refine_events(empty, [], 600) == []


def test_supplied_v_provenance_is_preserved_despite_floating_sum_order():
    intervals = [[100.1, 100.2], [100.3, 100.4]]
    analysis = analysis_with([], intervals, readable=intervals)
    event = event_with(intervals)
    # Equivalent interval arithmetic has different final bits: V's retained
    # unknown gap sum is authoritative, not a recomputed cumulative prefix.
    event["unknown_s"] = np.nextafter(100.3 - 100.2, np.inf)
    event["merged_from"] = 2
    event["af_s"] = np.nextafter(event["af_s"], np.inf)
    (refined,) = refine_events(analysis, [event], 600)
    assert refined["unknown_s"] == event["unknown_s"]
    assert refined["merged_from"] == event["merged_from"]
    assert refined["af_s"] == event["af_s"]
    assert refined["readable_non_af_gap_s"] == max(0, event["dur"] - event["af_s"] - event["unknown_s"])
