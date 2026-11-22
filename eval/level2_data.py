"""Development-only Level 2 data access and raw-byte separation evidence.

Native identities and paths never leave the reader. The frozen partition is
validated before source decoding; only the selected development marker row is
parsed. PulseDB derived labels are never accessed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import h5py
import numpy as np

from eval.tasks.audit_sources import ROOTS, _pulse_field, _pulse_runs, file_hash, inventory, record_key
from tsmcp.core.io.wfdb import read_header

ROOT = Path(__file__).resolve().parents[1]
PARENT_SPLIT = ROOT / "eval/tasks/audit_splits.json"
LEVEL2_SPLIT = ROOT / "eval/tasks/level2_splits.json"
SEALED_HASHES = ROOT / "eval/tasks/audit_sealed_hashes.json"
ALLOWED = frozenset({"diabetes", "cves", "pulsedb_mimic"})


def frozen_splits(*, verify_commit: bool = True) -> tuple[dict, dict]:
    """Check the authoritative parent/reservation binding before any data access."""
    parent = json.loads(PARENT_SPLIT.read_text())
    level2 = json.loads(LEVEL2_SPLIT.read_text())
    if level2["parent_split_sha256"] != file_hash(PARENT_SPLIT):
        raise ValueError("Level 2 reservation no longer binds the frozen parent split")
    if level2["source_hash_inventory_sha256"] != file_hash(SEALED_HASHES):
        raise ValueError("Level 2 reservation no longer binds the sealed raw-byte inventory")
    if set(parent["datasets"]) != ALLOWED or set(level2["datasets"]) != ALLOWED:
        raise ValueError("Unexpected source in frozen partitions")
    for name in ALLOWED:
        first, second = parent["datasets"][name], level2["datasets"][name]
        development = set(first["development"])
        sealed = set(first["sealed_confirmation"])
        confirmation = set(second["level2_confirmation"])
        reserve = set(second["future_reserve"])
        if (
            development != set(second["development"])
            or development & sealed
            or confirmation & reserve
            or confirmation | reserve != sealed
            or len(confirmation) != len(sealed) // 2
        ):
            raise ValueError("Frozen development/confirmation/reserve partitions disagree")
    if verify_commit:
        result = subprocess.run(
            ["git", "show", "HEAD:eval/tasks/level2_splits.json"], cwd=ROOT, capture_output=True, check=False
        )
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != file_hash(LEVEL2_SPLIT):
            raise ValueError("Level 2 split must be committed unchanged before reading signals")
    return parent, level2


def _development_candidates(dataset: str, parent: dict, level2: dict, prefer_unused: bool) -> list[tuple]:
    if dataset not in ALLOWED:
        raise ValueError("Source excluded by Level 2 policy")
    entry = parent["datasets"][dataset]
    development = set(entry["development"])
    sealed = set(entry["sealed_confirmation"])
    second = level2["datasets"][dataset]
    if development != set(second["development"]) or development & (
        sealed | set(second["level2_confirmation"]) | set(second["future_reserve"])
    ):
        raise ValueError("Attempt to open a sealed or unassigned subject")
    order = list(entry["candidate_subjects"])
    if any(subject not in development or subject in sealed for subject in order):
        raise ValueError("Attempt to open a sealed or unassigned subject")
    used = set()
    if prefer_unused:
        previous = json.loads((ROOT / "eval/tasks/p8audit_manifest.json").read_text())
        used = {q["source"]["subject_key"] for q in previous["questions"]}
        order.sort(key=lambda subject: subject in used)  # stable frozen order within groups
    local = inventory()[dataset]
    metadata = {row["record_key"]: row for row in entry["records"]}
    result = []
    for subject in order:
        if subject not in local:
            raise ValueError("Frozen development subject missing from local inventory")
        for path in local[subject]:
            key = record_key(dataset, path)
            row = metadata.get(key)
            if row is None or row["subject_key"] != subject or "source_sha256" not in row:
                raise ValueError("Development record missing or inconsistent in frozen source hashes")
            result.append((subject, path, key, row, subject in used))
    return result


def _selected_marker_row(table: Path, native: str) -> list[str] | None:
    """Match the identity column as bytes; never decode another subject's annotations.

    Source marker files use one unquoted subject field followed by comma-separated
    annotations on one physical line. Unrelated lines remain opaque bytes.
    """
    target = native.lower().encode("ascii")
    with table.open("rb") as stream:
        for line in stream:
            identity = line.split(b",", 1)[0].strip().strip(b'"').lower()
            if identity == target:
                return next(csv.reader([line.decode("latin1")]))
    return None


def _marker_events(path: Path, marker: np.ndarray, fs: float) -> tuple[list[dict], dict]:
    day = 1 if path.parent.name.endswith("Day1") else 2
    table = ROOTS["diabetes"] / "Data_Description" / f"GE-71_Head-up-tilt-Day{day}_Markers_per_subject.csv"
    native = re.sub(r"D[AB]$", "", path.stem, flags=re.IGNORECASE).lower()
    row = _selected_marker_row(table, native)
    diagnostics = {"status": "excluded", "reason": "No aligned development manual marker row"}
    if row is None or not np.any(np.isfinite(marker)):
        return [], diagnostics
    low, high = np.nanquantile(marker, [0.05, 0.999])
    if high - low < 0.05:
        high = float(np.nanmax(marker))
    if high - low < 0.05:
        return [], diagnostics
    active = marker > low + 0.5 * (high - low)
    starts = np.flatnonzero(np.diff(active.astype(np.int8), prepend=0) == 1)
    starts = starts[np.r_[True, np.diff(starts) > fs]]
    expected = max([int(v.strip()) for v in row[2:-1] if v.strip().isdigit()], default=0)
    matched = expected == len(starts) and expected > 0
    diagnostics.update(
        expected_pulses=expected,
        detected_pulses=len(starts),
        status="accepted" if matched else "excluded",
        reason="Exact manual ordinal maximum / native marker pulse count agreement required",
    )
    if not matched:
        return [], diagnostics
    labels = [("valsalva", i) for i in ([5, 7, 9] if day == 1 else [3, 5, 7, 9])]
    labels.append(("tilt", 11 if day == 1 else 15))
    events = []
    for kind, column in labels:
        try:
            ordinal = int(row[column].strip())
        except (ValueError, IndexError):
            continue
        if 0 < ordinal <= len(starts):
            event = {
                "kind": kind,
                "t": float(starts[ordinal - 1] / fs),
                "marker_ordinal": ordinal,
                "marker_pulse_count": len(starts),
                "method": "Development manual ordinal aligned to native marker rising edge",
                "marker_file_sha256": file_hash(table),
            }
            # Paired stop ordinal, when documented and present, preserves the
            # manoeuvre interval without inferring a duration from physiology.
            try:
                stop = int(row[column + 1].strip())
            except (ValueError, IndexError):
                stop = 0
            if ordinal < stop <= len(starts):
                event["end_t"] = float(starts[stop - 1] / fs)
            events.append(event)
    return events, diagnostics


def _wfdb_record(dataset: str, path: Path, frozen: dict) -> dict:
    if file_hash(path) != frozen["header_sha256"]:
        raise ValueError("Development WFDB header changed after freeze")
    header = read_header(path)
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if any(s["format"] != "16" for s in header["signals"]):
        raise ValueError("Level 2 requires native WFDB format 16")
    if not np.isfinite(header["fs"]) or header["fs"] <= 0:
        raise ValueError("Invalid native sampling frequency")
    if path.with_suffix(".dat").stat().st_size != header["n_samp"] * header["n_sig"] * 2:
        raise ValueError("Development WFDB byte length disagrees with frozen header")
    raw = np.memmap(path.with_suffix(".dat"), dtype="<i2", mode="r").reshape(-1, header["n_sig"])
    native_names = [s["name"].lower() for s in header["signals"]]
    cerebral = "mcar" if "mcar" in native_names else "mcal"
    names, units, columns, marker = [], [], [], None
    mapping = {"ecg": "ECG", "abp": "ABP", cerebral: "CBFV"}
    for i, (description, line) in enumerate(zip(header["signals"], lines[1:], strict=False)):
        name = description["name"].lower()
        if name not in mapping and not (name == "marker" and dataset == "diabetes"):
            continue
        values = (raw[:, i].astype(np.float64) - description["baseline"]) / description["gain"]
        values[raw[:, i] == -32768] = np.nan
        if name == "marker":
            marker = values
            continue
        gain_field = line.split()[2]
        names.append(mapping[name])
        units.append(gain_field.split("/", 1)[1] if "/" in gain_field else "mV")
        columns.append(values)
    events, alignment = _marker_events(path, marker, header["fs"]) if marker is not None else ([], {})
    return {
        "x": np.column_stack(columns),
        "fs": float(header["fs"]),
        "channels": names,
        "units": units,
        "start_s": 0.0,
        "duration_s": header["n_samp"] / header["fs"],
        "events": events,
        "marker_alignment": alignment,
        "protocol": frozen["protocol"],
    }


def records(
    dataset: str,
    *,
    duration_s: float = 600,
    min_duration_s: float = 60,
    prefer_unused: bool = True,
    max_records: int | None = None,
) -> Iterator[dict]:
    """Yield full native WFDB records or the longest qualifying strict PulseDB run.

    WFDB records remain full length to support 10-minute and protocol analyses.
    PulseDB requests are capped at 600 s; every included 10-s window and boundary
    must satisfy native timestamp, case, segment and subject continuity. A 600-s
    request with min_duration_s=600 never silently falls back to a short run.
    """
    if dataset not in ALLOWED:
        raise ValueError("Source excluded by Level 2 policy")
    if not (math.isfinite(duration_s) and math.isfinite(min_duration_s) and 0 < min_duration_s <= duration_s):
        raise ValueError("Require finite positive durations and min_duration_s <= duration_s")
    if dataset == "pulsedb_mimic" and not 60 <= duration_s <= 600:
        raise ValueError("PulseDB Level 2 runs require 60 to 600 seconds")
    if dataset == "pulsedb_mimic" and math.ceil(min_duration_s / 10) > math.floor(duration_s / 10):
        raise ValueError("Requested PulseDB duration bounds contain no complete 10-second run")
    parent, level2 = frozen_splits()
    candidates = _development_candidates(dataset, parent, level2, prefer_unused)
    if dataset == "diabetes":
        for filename, expected in parent["datasets"][dataset]["marker_files"].items():
            if file_hash(ROOTS[dataset] / "Data_Description" / filename) != expected:
                raise ValueError("Manual marker source changed after split freeze")
    yielded = 0
    for subject, path, key, frozen, previously_used in candidates:
        if max_records is not None and yielded >= max_records:
            return
        source = path if dataset == "pulsedb_mimic" else path.with_suffix(".dat")
        if file_hash(source) != frozen["source_sha256"]:
            raise ValueError("Development source changed after split freeze")
        base = {
            "dataset": dataset,
            "subject_key": subject,
            "record_key": key,
            "source_hashes": {k: v for k, v in frozen.items() if k.endswith("sha256")},
            "events": [],
            "used_in_level1": previously_used,
        }
        if dataset == "pulsedb_mimic":
            minimum = max(6, math.ceil(min_duration_s / 10))
            cap = min(60, math.floor(duration_s / 10))
            with h5py.File(path, "r") as handle:
                # Search for the full requested support before falling back. A
                # short early run must not hide a later ten-minute drift record.
                runs = _pulse_runs(handle, subject, first_min_windows=cap, cap_windows=cap)
                eligible = [run for run in runs if len(run) >= minimum]
                selected = max(eligible, key=len)[:cap] if eligible else None
                if selected is None:
                    continue
                group = handle["Subj_Wins"]
                columns = [
                    np.concatenate([_pulse_field(handle, group[field], i) for i in selected])
                    for field in ("ECG_Record", "PPG_Record", "ABP_Raw")
                ]
                if any(len(column) != len(selected) * 1250 for column in columns):
                    raise ValueError("Unexpected development PulseDB waveform shape")
                base.update(
                    x=np.column_stack(columns).astype(np.float64),
                    fs=125.0,
                    channels=["ECG", "PPG", "ABP"],
                    units=["source amplitude", "source amplitude", "mmHg"],
                    start_s=float(_pulse_field(handle, group["T"], selected[0])[0]),
                    duration_s=len(selected) * 10.0,
                    continuity="SubjectID + CaseID + SegmentID+1 + every native timestamp increment at 125 Hz",
                    window_indices=[selected[0], selected[-1]],
                )
        else:
            base.update(_wfdb_record(dataset, path, frozen))
        if base["duration_s"] < min_duration_s:
            continue
        yielded += 1
        yield base


def separation(
    case_sources: Sequence[dict],
    signal_paths: Sequence[Path],
    *,
    sealed_path: Path = SEALED_HASHES,
    ground_truth_sources: Sequence[dict] | None = None,
) -> dict:
    """Compare all frozen development and emitted source bytes to sealed bytes.

    Supplemental G1 cleaned-file/document hashes are taken from case ground truth
    or explicit reference_source_hashes mappings. Reference Python code hashes are
    excluded because they are executable methods, not subject source material.
    """
    parent, level2 = frozen_splits()
    sealed = json.loads(sealed_path.read_text())
    bound = json.loads(SEALED_HASHES.read_text())
    if sealed["split_sha256"] != file_hash(PARENT_SPLIT) or sealed.get("waveform_decoding") is not False:
        raise ValueError("Sealed hash evidence has incorrect split binding or decoding policy")
    if sealed["records"] != bound["records"]:
        raise ValueError("Fresh sealed source-byte inventory differs from frozen reservation")
    sealed_hashes = {
        value for row in sealed["records"] for key, value in row.items() if key in {"source_sha256", "header_sha256"}
    }
    development_hashes = []
    metadata = {}
    subject_overlap = 0
    development_records = 0
    for name, entry in parent["datasets"].items():
        development = set(entry["development"])
        reserved = level2["datasets"][name]
        subject_overlap += len(development & (set(reserved["level2_confirmation"]) | set(reserved["future_reserve"])))
        for row in entry["records"]:
            if row["subject_key"] not in development:
                continue
            development_records += 1
            metadata[(name, row["record_key"])] = row
            if "source_sha256" not in row:
                raise ValueError("Frozen development record lacks raw source hash")
            development_hashes.extend(row[k] for k in ("source_sha256", "header_sha256") if k in row)
    for source in case_sources:
        name = source["dataset"]
        if name not in ALLOWED or source["subject_key"] not in parent["datasets"][name]["development"]:
            raise ValueError("Case uses a sealed or unassigned subject")
        row = metadata.get((name, source["record_key"]))
        if row is None or row["subject_key"] != source["subject_key"]:
            raise ValueError("Case source is absent from frozen development inventory")
        hashes = source["source_hashes"]
        if hashes.get("source_sha256") != row["source_sha256"] or any(row.get(k) != v for k, v in hashes.items()):
            raise ValueError("Case source hashes differ from frozen development inventory")
    if ground_truth_sources is None:
        ground_truth_sources = [
            source["ground_truth"].get("reference_source_hashes", {})
            for source in case_sources
            if "G1" in source.get("ground_truth", {}).get("category", "")
        ]
    manual_fields = {"development_cleaned_npz_sha256", "manual_cleaning_readme_sha256"}
    manual_hashes = [
        value for reference in ground_truth_sources for key, value in reference.items() if key in manual_fields
    ]
    signals = [file_hash(Path(path)) for path in signal_paths]
    raw_overlap = len(set(development_hashes) & sealed_hashes)
    case_overlap = len(set(signals) & sealed_hashes)
    manual_overlap = len(set(manual_hashes) & sealed_hashes)
    if subject_overlap or raw_overlap or case_overlap or manual_overlap:
        raise ValueError("Development or emitted-case overlap with sealed inventory")
    return {
        "subject_overlap": subject_overlap,
        "level2_confirmation_overlap": 0,
        "development_records_compared": development_records,
        "development_source_header_hashes_compared": len(development_hashes),
        "case_source_records_checked": len(case_sources),
        "case_signal_hashes_compared": len(signals),
        "manual_ground_truth_source_hashes_compared": len(manual_hashes),
        "unique_manual_ground_truth_source_hashes_compared": len(set(manual_hashes)),
        "sealed_records": sealed["record_count"],
        "sealed_files": sealed["raw_file_count"],
        "sealed_raw_bytes": sealed["source_bytes"],
        "development_raw_hash_overlap": raw_overlap,
        "case_signal_raw_hash_overlap": case_overlap,
        "manual_ground_truth_raw_hash_overlap": manual_overlap,
        "sealed_inventory_sha256": file_hash(sealed_path),
        "reservation_sha256": file_hash(LEVEL2_SPLIT),
        "confirmation_waveforms_decoded": False,
        "scope": "Raw source/header bytes and emitted NPY bytes; does not prove decoded-window disjointness",
    }
