"""Where native event evidence can be used. All coordinates are block-local half-open samples."""

import numpy as np

from tsmcp.processing import intervals


def native_support(mask, fs, task, events):
    good = np.asarray(mask, bool)
    assessed = np.zeros(len(good), bool)
    edge = np.zeros(len(good), bool)
    unknown = np.zeros(len(good), bool)
    points = np.array([e["sample"] for e in events], int)
    ecg = task == "ECG"
    # ECG explicitly discards the first/last 160 ms. For pulse landmarks the
    # smoothed extrema and first foot need neighbouring observed samples.
    margin = max(1, round((0.16 if ecg else 0.60) * fs))
    for a, b in intervals(good):
        if b - a < int((3 if ecg else 2) * fs) or (ecg and fs <= 40) or 2 * margin >= b - a:
            continue
        edge[a : a + margin] = edge[b - margin : b] = True
        assessed[a + margin : b - margin] = True
        if not np.any((points >= a) & (points < b)):
            unknown[a + margin : b - margin] = True
    return dict(
        detector="qrs_fir_absolute" if ecg else "native_pressure_pulse",
        evaluated_ranges=intervals(assessed),
        boundary_uncertain_ranges=intervals(edge),
        unsupported_ranges=intervals(~(assessed | edge)),
        empty_event_ranges=intervals(unknown),
        applicability="evaluated" if (assessed & ~unknown).any() else "insufficient_support",
    )
