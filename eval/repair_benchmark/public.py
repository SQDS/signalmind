"""Build one public repair-only input from detector output on observed samples."""

from pathlib import Path

import numpy as np

from tsmcp import processing
from tsmcp.cleaning.common import array_hash, digest, sha, write
from tsmcp.processing import PCHIP_TOOL, PCHIP_VERSION, VERSION

CATALOG = {
    "interpolate": dict(version=VERSION, anchors_per_side=1, max_samples=4),
    PCHIP_TOOL: dict(version=PCHIP_VERSION, anchors_per_side=3, max_samples=4),
}

SIMULATOR = dict(
    species="simulated-human",
    device="SignalMind-analytic-ECG-v1",
    field="formula_ecg",
    has_source_time=True,
    normalized_windows=False,
)


def prepare(out, case_id, task, unit, fs, t, x, valid, estimated, offset=0, *, acquisition=None):
    from tsmcp.inspection.scan import scan

    out = Path(out)
    t, x, valid, estimated = (
        np.asarray(z, dtype=k) for z, k in [(t, float), (x, float), (valid, bool), (estimated, bool)]
    )
    if (
        task not in ("ECG", "PPG", "ABP")
        or fs <= 0
        or offset < 0
        or t.ndim != 1
        or len(t) < 2
        or any(z.shape != t.shape for z in (x, valid, estimated))
        or not np.isfinite(t).all()
        or not np.all(np.abs(np.diff(t) - 1 / fs) <= 1e-6)
        or not np.isfinite(x[valid]).all()
    ):
        raise ValueError("Expected one continuous observed channel on its declared source grid")
    hits, _, _ = scan(x, valid & ~estimated, {}, fs, task, "balanced")
    selected = sorted(
        {
            (z["start"], z["end"], z["kind"])
            for z in hits
            if z["kind"] in ("biphasic_core", "impulsive_core") and z["end"] - z["start"] <= 4
        }
    )
    candidates = [
        dict(
            candidate_id=digest([case_id, a + offset, b + offset, reason])[:24],
            start_sample=a + offset,
            end_sample=b + offset,
            reason=reason,
        )
        for a, b, reason in selected
    ]

    def region(a, b):
        return dict(channel_id=task, start_sample=a + offset, end_sample=b + offset)

    source = array_hash(t, x, valid, estimated)
    from tsmcp.inspection import scan as detector

    case = dict(
        case_id=case_id,
        task=task,
        unit=unit,
        fs=fs,
        offset=offset,
        samples=len(x),
        track="repair_only_short_core",
        profile_id="public.short-core.1",
        observed_hash=source,
        parent_version_id=digest([source, case_id]),
        candidates=candidates,
        public_read_ranges=[region(0, len(x))],
        allowed_write_ranges=[region(a, b) for a, b in sorted({(a, b) for a, b, _ in selected})],
        catalog=CATALOG,
        public_tool_catalog_hash=digest([CATALOG, sha(processing.__file__)]),
        detector=dict(preset="balanced", source_sha256=sha(detector.__file__)),
        guard_ownership="Shared numerical prerequisites, observed anchors and declared write scope. Physiology decisions belong to each method.",
    )
    if acquisition is not None:
        keys = {"species", "device", "field", "has_source_time", "normalized_windows"}
        if (
            set(acquisition) != keys
            or any(type(acquisition[k]) is not bool for k in ("has_source_time", "normalized_windows"))
            or any(not isinstance(acquisition[k], str) or not acquisition[k] for k in ("species", "device", "field"))
        ):
            raise ValueError("Expected public species, device, native field, time and normalization declarations")
        case["acquisition"] = dict(acquisition)
        if acquisition == SIMULATOR and task == "ECG" and fs == 125 and unit == "synthetic units":
            case["simulation_profile"] = "analytic-ecg-rr.dev1"
    case["input_hash"] = digest(case)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "observed.npz", t=t, x=x, valid=valid, estimated=estimated)
    write(out / "case.json", case)
    return case
