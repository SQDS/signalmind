"""Real ABP quality candidates from strictly mapped development manual labels.

Only exact frozen development Day2 records can select a cleaned NPZ. The cleaned
signal is never treated as physiological truth: it supplies a manually reviewed
artefact interval and an auditable processing mask. Cases contain native raw ABP.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from eval.level2_data import ROOT, _development_candidates, _wfdb_record, frozen_splits
from eval.tasks.audit_sources import file_hash

MANUAL_ROOT = Path.home() / "cbv_true/data/disbate_data/manual_physiocal_cleaned_day2"
SEED = 20261026
ELIGIBLE_TYPES = frozenset(
    {"physiocal_dropout_step_recovery", "physiocal_small_step_like", "physiocal_step_train_long"}
)
ACTION_MASK = {
    "interpolate_short_main": "manual_physio_interp_mask",
    "review_exclude_no_interpolation": "manual_physio_review_exclude_mask",
}


def _eligible(event: dict) -> bool:
    return (
        event["confidence"] == "high"
        and event["mark_layer"] == "main"
        and event["problem_type"] in ELIGIBLE_TYPES
        and event["applied_action"] in ACTION_MASK
        and float(event["end_sec"]) - float(event["start_sec"]) >= 2.5
    )


def _mapped_cleaned(path: Path) -> Path | None:
    """Resolve exact record-stem matches; no cleaned contents are read here."""
    pattern = re.compile(re.escape(path.stem) + r"(?:[_\-.].*)?\.npz", re.IGNORECASE)
    matches = [p for p in (MANUAL_ROOT / "strict/processed_records").rglob("*.npz") if pattern.fullmatch(p.name)]
    if len(matches) > 1:
        raise ValueError("Ambiguous development raw-to-cleaned filename mapping")
    return matches[0] if matches else None


def _native_agreement(native: np.ndarray, cleaned_original: np.ndarray, scalar: float) -> float:
    if native.shape != cleaned_original.shape or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError("Manual cleaned array has different native sample support or calibration")
    if not np.allclose(native * scalar, cleaned_original, rtol=1e-6, atol=1e-4, equal_nan=True):
        raise ValueError("Manual cleaned original does not match frozen native ABP samples")
    difference = np.abs(native * scalar - cleaned_original)
    return float(np.nanmax(difference))


def _choose_crop(event: dict, events: list[dict], total_s: float, fs: float, rng) -> tuple[float, float]:
    duration = 180.0
    onset, end = float(event["start_sec"]), float(event["end_sec"])
    lower, upper = max(0, end - duration + 20), min(total_s - duration, onset - 20)
    if lower >= upper:
        raise ValueError("Manual interval lacks native context on both sides")
    begin = round(float(rng.uniform(lower, upper)) * fs) / fs
    overlaps = [
        other
        for other in events
        if other["mark_layer"] == "main"
        and float(other["start_sec"]) < begin + duration
        and float(other["end_sec"]) > begin
    ]
    competing = [
        min(float(other["end_sec"]), begin + duration) - max(float(other["start_sec"]), begin)
        for other in overlaps
        if other is not event
    ]
    if competing and end - onset <= max(competing) + 2:
        raise ValueError("No uniquely longest reviewed ABP defect in randomized crop")
    return begin, duration


def build_candidates(limit: int = 4) -> tuple[list[dict], list[dict]]:
    parent, level2 = frozen_splits()
    # This gate resolves only frozen development paths, before any cleaned NPZ is opened.
    pool = _development_candidates("diabetes", parent, level2, prefer_unused=True)
    candidates, exclusions, used = [], [], set()
    for subject, path, key, frozen, previously_used in pool:
        if len(candidates) >= limit:
            break
        if subject in used or not path.parent.name.endswith("Day2"):
            continue
        try:
            cleaned = _mapped_cleaned(path)
            if cleaned is None:
                raise ValueError("No exact development cleaned-file mapping")
            cleaned_hash = file_hash(cleaned)
            with np.load(cleaned, allow_pickle=False) as archive:
                native_subject = re.sub(r"D[AB]$", "", path.stem, flags=re.IGNORECASE).lower()
                if str(archive["subject_key"].item()).lower() != native_subject:
                    raise ValueError("Manual cleaned subject differs from frozen development mapping")
                if str(archive["manual_physio_mode"].item()) != "strict":
                    raise ValueError("Manual quality truth requires strict main-label processing")
                events = json.loads(str(archive["manual_physio_events_json"].item()))
                if any(event["record_stem"].lower() != path.stem.lower() for event in events):
                    raise ValueError("Manual event record differs from frozen development source")
                eligible = sorted(
                    [event for event in events if _eligible(event)],
                    key=lambda event: -(float(event["end_sec"]) - float(event["start_sec"])),
                )
                if not eligible:
                    raise ValueError("No high-confidence main ABP interval of at least 2.5 seconds")
                if file_hash(path.with_suffix(".dat")) != frozen["source_sha256"]:
                    raise ValueError("Frozen development raw bytes changed before manual mapping")
                record = _wfdb_record("diabetes", path, frozen)
                fs = record["fs"]
                if float(archive["fs"]) != fs:
                    raise ValueError("Manual sample frequency differs from frozen native record")
                abp = record["x"][:, record["channels"].index("ABP")]
                scalar = float(archive["abp_scalar"])
                difference = _native_agreement(abp, archive["abp_original_calibrated"], scalar)
                seed = int(hashlib.sha256(f"{SEED}:F2:manual:{subject}".encode()).hexdigest()[:16], 16)
                rng = np.random.default_rng(seed)
                chosen = None
                for event in eligible:
                    try:
                        begin, duration = _choose_crop(event, events, record["duration_s"], fs, rng)
                        mask = archive[ACTION_MASK[event["applied_action"]]]
                        a, b = (round(float(event[k]) * fs) for k in ("mask_start_sec", "mask_end_sec"))
                        if len(mask) != len(abp) or b <= a or float(np.mean(mask[a:b])) < 0.99:
                            raise ValueError("Manual event interval and processing mask disagree")
                        chosen = event, begin, duration, float(np.mean(mask[a:b]))
                        break
                    except ValueError as exc:
                        exclusions.append(
                            {"family": "F2", "stage": "manual_interval", "reason": str(exc), "subject_key": subject}
                        )
                if chosen is None:
                    continue
                event, begin, duration, mask_fraction = chosen
                a, b = round(begin * fs), round((begin + duration) * fs)
                raw = np.asarray(record["x"][a:b], dtype=np.float32)
                if not np.isfinite(raw).all():
                    raise ValueError("Manual candidate crop contains nonfinite native samples")
                onset = float(event["start_sec"]) - begin
                length = float(event["end_sec"]) - float(event["start_sec"])
                candidates.append(
                    {
                        "dataset": "diabetes",
                        "subject_key": subject,
                        "record_key": key,
                        "source_hashes": {k: v for k, v in frozen.items() if k.endswith("sha256")},
                        "start_s": begin,
                        "source_duration_s": duration,
                        "used_in_level1": previously_used,
                        "family": "F2",
                        "task_type": "manual_abp_artifact",
                        "question": (
                            "Identify the longest acquisition-related ABP artefact in this recording. "
                            "Report its onset_s and duration_s in seconds. Distinguish an unreliable pressure "
                            "segment from a physiological pressure response by checking its morphology and "
                            "the simultaneous ECG."
                        ),
                        "x": raw,
                        "fs": fs,
                        "channels": record["channels"],
                        "units": record["units"],
                        "expected_answer": {"onset_s": onset, "duration_s": length},
                        "tolerances": {"onset_s": 2.0, "duration_s": 1.0},
                        "abstention_correct": False,
                        "expected_solvability": "generic",
                        "specialist_capability": None,
                        "signal_kind": "real_signal",
                        "ground_truth": {
                            "category": "G1",
                            "methods": [
                                "High-confidence main visual-review artefact interval in strict manual cleaning",
                                "Native raw ABP/calibrated-original sample identity and processing-mask support audit",
                            ],
                            "agreement": {
                                "accepted": True,
                                "max_calibrated_original_sample_difference_mmhg": difference,
                                "processing_mask_interval_coverage": mask_fraction,
                                "same_native_sample_count_and_fs": True,
                            },
                            "confidence": "medium",
                            "manual_label_confidence": "high",
                            "manual_problem_type": event["problem_type"],
                            "manual_action": event["applied_action"],
                            "manual_interval_native_s": [float(event["start_sec"]), float(event["end_sec"])],
                            "reference_source_hashes": {
                                "development_cleaned_npz_sha256": cleaned_hash,
                                "manual_cleaning_readme_sha256": file_hash(MANUAL_ROOT / "README.md"),
                            },
                            "tolerance_justification": (
                                "Manual review labels have subsecond sample-aligned boundaries; 2 s onset and "
                                "1 s duration tolerances allow visual boundary uncertainty and exceed the "
                                "0.1–0.2 s processing padding. Native label interval, not padding, is scored."
                            ),
                            "limitations": (
                                "Single local visual review, not adjudicated clinical annotation. Only high-confidence "
                                "main marks are truth; cleaned values are not physiological SBP truth. Low-confidence, "
                                "micro-possible, multi-signal-uncertain and terminal intervals are excluded."
                            ),
                        },
                        "grounding_rubric": {
                            "regions": [{"channels": ["ABP", "ECG"], "window_s": [onset - 5, onset + length + 5]}],
                            "required_computations": ["ABP artefact interval and cross-channel quality verification"],
                        },
                        "crop_randomization": {
                            "seed": seed,
                            "distribution": "Uniform crop start over native record support with 20 s event margins",
                        },
                    }
                )
                used.add(subject)
        except (ValueError, KeyError) as exc:
            # Exceptions use fixed private messages; arbitrary data values are never echoed.
            reason = str(exc) if isinstance(exc, ValueError) else "Missing required development manual source field"
            exclusions.append({"family": "F2", "stage": "manual_mapping", "reason": reason, "subject_key": subject})
    return candidates, exclusions


def persist_candidates(candidates: list[dict], exclusions: list[dict]) -> Path:
    cache = ROOT / "eval/cache/level2/manual"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, candidate in enumerate(candidates):
        name = f"candidate-{index:02d}.npy"
        np.save(cache / name, candidate["x"], allow_pickle=False)
        metadata.append(
            {
                **{k: v for k, v in candidate.items() if k != "x"},
                "signal_file": name,
                "signal_cache": str((cache / name).relative_to(ROOT)),
            }
        )
    destination = cache / "index.json"
    destination.write_text(json.dumps({"candidates": metadata, "exclusions": exclusions}, indent=2) + "\n")
    (cache / "candidates.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (cache / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    candidates, exclusions = build_candidates()
    persist_candidates(candidates, exclusions)
    print(json.dumps({"manual_candidates": len(candidates), "manual_exclusions": len(exclusions)}))
