"""One frozen Task E boundary candidate; consumes R analysis without changing it.

The declaration is committed at c80f339 in docs/tasks/onset-refine-report.md.
No annotation, waveform loading, detector call, or classification occurs here.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy

import numpy as np

PARAMETERS = {
    "search_radius_s": 300.0,
    "minimum_run_s": 10.0,
    "before_s": 30.0,
    "before_readable_min_s": 27.0,
    "before_evidence_max_s": 3.0,
    "after_s": 60.0,
    "after_readable_min_s": 54.0,
    "after_evidence_min_s": 45.0,
    "connection_gap_s": 120.0,
}


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([float(start), float(end)])
    return result


class _Coverage:
    """Exact occupied duration up to each query time, using interval prefixes."""

    def __init__(self, intervals):
        self.intervals = _union(intervals)
        self.starts = [start for start, _ in self.intervals]
        self.prefix = [0.0]
        for start, end in self.intervals:
            self.prefix.append(self.prefix[-1] + end - start)

    def until(self, time):
        i = bisect_right(self.starts, time) - 1
        if i < 0:
            return 0.0
        start, end = self.intervals[i]
        return self.prefix[i] + min(time, end) - start

    def between(self, start, end):
        return self.until(end) - self.until(start)


def _evidence_runs(analysis):
    times = np.asarray(analysis["beats"]["t_s"], dtype=float)
    mask = (np.asarray(analysis["score"]) >= 1.0) & np.asarray(analysis["valid"], dtype=bool)
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [
        [float(times[a]), float(times[b])]
        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)
    ]


def _connected_groups(intervals):
    groups = []
    for start, end in _union(intervals):
        if groups and start - groups[-1][1] < PARAMETERS["connection_gap_s"]:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    return groups


def refine_events(analysis, episodes, duration):
    """Return V event copies with boundary estimates and explicit gap provenance.

    Event order, spans, support membership and burden fields remain untouched.
    An unresolved bracket records the search bounds, not calibrated coverage.
    Readable non-AF gaps mean readable time without final R AF support.
    """
    evidence_runs = _evidence_runs(analysis)
    evidence = _Coverage(evidence_runs)
    readable = _Coverage(analysis["readable"])
    support = _Coverage(analysis["supported"])
    groups = _connected_groups(evidence_runs + support.intervals)
    group_starts = [start for start, _ in groups]

    # These confidence checks depend only on the proposed beat boundary, so
    # evaluate once even when neighboring candidate events share a search area.
    qualifying = []
    for start, end in evidence_runs:
        before = start - PARAMETERS["before_s"]
        after = start + PARAMETERS["after_s"]
        if end - start < PARAMETERS["minimum_run_s"] or before < 0 or after > duration:
            continue
        if (
            readable.between(before, start) >= PARAMETERS["before_readable_min_s"]
            and evidence.between(before, start) <= PARAMETERS["before_evidence_max_s"]
            and readable.between(start, after) >= PARAMETERS["after_readable_min_s"]
            and evidence.between(start, after) >= PARAMETERS["after_evidence_min_s"]
        ):
            qualifying.append(start)

    result = []
    for original in episodes:
        event = deepcopy(original)
        start, end = float(event["t"]), float(event["t"] + event["dur"])
        lo = max(0.0, start - PARAMETERS["search_radius_s"])
        hi = min(float(duration), start + PARAMETERS["search_radius_s"])
        onset = None
        for proposed in qualifying:
            if proposed < lo:
                continue
            if proposed > hi:
                break
            if proposed < start:
                group_index = bisect_right(group_starts, proposed) - 1
                if group_index < 0 or groups[group_index][1] <= start:
                    continue
            elif proposed >= end:
                continue
            onset = proposed
            break

        clipped_support = [[max(start, a), min(end, b)] for a, b in support.intervals if b > start and a < end]
        readable_s = readable.between(start, end)
        support_s = sum(b - a for a, b in clipped_support)
        # Retain V's existing provenance exactly: equivalent sums performed in
        # a different order can round differently on nonbinary sample grids.
        unknown_s = event.get("unknown_s", max(0.0, end - start - readable_s))
        af_s = event.get("af_s", support_s)
        event.update(
            onset_s=onset,
            onset_lo_s=lo if onset is None else onset,
            onset_hi_s=hi if onset is None else onset,
            onset_confident=onset is not None,
            merged_from=event.get("merged_from", len(clipped_support)),
            unknown_s=unknown_s,
            readable_non_af_gap_s=max(0.0, event["dur"] - af_s - unknown_s),
            description="AF candidate event; bridged gaps do not establish uninterrupted AF",
        )
        result.append(event)
    return result
