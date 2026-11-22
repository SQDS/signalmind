"""Privacy-preserving audit source inventory, frozen split, and development readers.

Only opaque hashes leave this module. Native subject/path mappings stay in memory.
The split operation reads directory/header metadata and hashes selected development
files; it does not decode waveform samples. Confirmation waveforms are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from tsmcp.core.io.wfdb import read_header

SEED = 20261026
SPLIT_PATH = Path(__file__).with_name("audit_splits.json")
ROOTS = {
    "diabetes": Path.home() / "cbv_true/data/disbate_data/raw/cerebral-vasoregulation-in-diabetes-1.0.0",
    "cves": Path.home() / "cbv_true/data/physionet/cves/1.0.0/head-up-tilt",
    "pulsedb_mimic": Path.home() / "bp2rr_new/dataset/MIMIC",
}


def opaque(dataset: str, identifier: str) -> str:
    return hashlib.sha256(f"p8audit-subject-v1:{dataset}:{identifier.lower()}".encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_key(dataset: str, path: Path) -> str:
    return opaque(dataset, "record:" + str(path.relative_to(ROOTS[dataset])))


def inventory() -> dict[str, dict[str, list[Path]]]:
    """Resolve identifiers only in memory, grouping diabetes visits by subject."""
    result = {}
    for dataset, root in ROOTS.items():
        subjects: dict[str, list[Path]] = {}
        paths = root.rglob("*.mat" if dataset == "pulsedb_mimic" else "*.hea")
        for path in sorted(paths):
            if dataset == "diabetes":
                if path.parent.name not in {"Head-up-tilt_Day1", "Head-up-tilt_Day2"}:
                    continue
                match = re.fullmatch(r"(s\d+)(DA|DB)", path.stem, flags=re.IGNORECASE)
                if match is None:
                    raise ValueError("Unrecognized diabetes subject metadata; no identifier emitted")
                native_subject = match.group(1)
            elif dataset == "cves":
                match = re.fullmatch(r"(s\d+)-head-up-tilt", path.stem, flags=re.IGNORECASE)
                if match is None:
                    raise ValueError("Unrecognized CVES subject metadata; no identifier emitted")
                native_subject = match.group(1)
            else:
                if re.fullmatch(r"p\d+", path.stem, flags=re.IGNORECASE) is None:
                    raise ValueError("Unrecognized PulseDB subject filename; no identifier emitted")
                native_subject = path.stem
            subjects.setdefault(opaque(dataset, native_subject), []).append(path)
        result[dataset] = subjects
    return result


def split_rank(dataset: str, subject_key: str) -> str:
    return hashlib.sha256(f"{SEED}:{dataset}:{subject_key}".encode()).hexdigest()


def freeze_splits(destination: Path = SPLIT_PATH, pulse_candidates: int | None = None) -> dict:
    """Freeze subject assignment before selecting or examining any waveform."""
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "development_fraction": 0.30,
        "assignment": "sort SHA256(seed:dataset:opaque_subject); first ceil(0.30*n) development",
        "subject_key": "SHA256(p8audit-subject-v1:dataset:lowercase_subject); native identifiers never persisted",
        "source_policy": "Metadata-only inventory; development candidate bytes hashed without waveform decoding",
        "confirmation_policy": "Sealed: filename/header metadata only; no waveform bytes decoded or hashed",
        "datasets": {},
        "excluded_sources": {
            "uci": "Quarantined: no verifiable subject linkage; possible overlap with MIMIC cannot be excluded",
            "jlu": "Excluded entirely by instruction; never accessed",
            "pulsedb_vitaldb": "Excluded entirely by instruction; never accessed",
        },
        "separation": {
            "subject_overlap": 0,
            "confirmation_waveforms_opened": False,
            "confirmation_signal_hash_comparison": "incomplete: no sealed signal hash inventory available",
        },
    }
    for dataset, subjects in inventory().items():
        ranked = sorted(subjects, key=lambda s: split_rank(dataset, s))
        n_dev = math.ceil(len(ranked) * 0.30)
        dev, sealed = ranked[:n_dev], ranked[n_dev:]
        candidates = dev[:pulse_candidates] if dataset == "pulsedb_mimic" and pulse_candidates is not None else dev
        entry = {
            "subject_count": len(ranked),
            "development": dev,
            "sealed_confirmation": sealed,
            "candidate_subjects": candidates,
            "subject_grouping": {
                "diabetes": "Day1/Day2 suffix removed from subject filename; visits stay together",
                "cves": "One subject per head-up-tilt record",
                "pulsedb_mimic": "One subject MAT per audited PulseDB MIMIC filename; loader verifies SubjectID",
            }[dataset],
            "records": [],
        }
        for subject in ranked:
            for path in subjects[subject]:
                rec = {"subject_key": subject, "record_key": record_key(dataset, path)}
                if dataset != "pulsedb_mimic":
                    header = read_header(path)
                    rec.update(
                        {
                            "header_sha256": file_hash(path),
                            "fs": header["fs"],
                            "n_samples": header["n_samp"],
                            "channels": [s["name"] for s in header["signals"]],
                            "protocol": path.parent.name if dataset == "diabetes" else "head-up-tilt",
                        }
                    )
                if subject in candidates:
                    source = path if dataset == "pulsedb_mimic" else path.with_suffix(".dat")
                    rec["source_sha256"] = file_hash(source)
                    rec["source_bytes"] = source.stat().st_size
                entry["records"].append(rec)
        if dataset == "diabetes":
            entry["marker_files"] = {
                path.name: file_hash(path)
                for path in sorted((ROOTS[dataset] / "Data_Description").glob("*Markers*.csv"))
            }
            entry["marker_semantics"] = (
                "CSV values identify ordinal marker-channel pulses, not seconds; zero/missing = absent. "
                "Day1 Val 1/2/3 onset columns 5/7/9 and Tilt column 11; Day2 Val columns 3/5/7/9 and Tilt column 15. "
                "Alignment is first rising edge of each marker pulse, sample index divided by native fs."
            )
        manifest["datasets"][dataset] = entry
    add_published_checksums(manifest)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def add_published_checksums(manifest: dict) -> None:
    """Use distributed checksum metadata, without opening sealed waveform bytes."""
    dataset = "diabetes"
    checksum_file = ROOTS[dataset] / "SHA256SUMS.txt"
    known = {}
    for line in checksum_file.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        known[name.lstrip("*")] = digest
    paths = inventory()[dataset]
    records = {row["record_key"]: row for row in manifest["datasets"][dataset]["records"]}
    matched = 0
    for sources in paths.values():
        for path in sources:
            record = records[record_key(dataset, path)]
            relative = str(path.with_suffix(".dat").relative_to(ROOTS[dataset]))
            if relative not in known:
                continue
            record["published_source_sha256"] = known[relative]
            if "source_sha256" in record and record["source_sha256"] != known[relative]:
                raise ValueError("Development source differs from distributed checksum inventory")
            matched += 1
    manifest["datasets"][dataset]["published_checksum_inventory"] = {
        "sha256": file_hash(checksum_file),
        "records_covered": matched,
        "sealed_hashes_status": "published source-file hashes only; sealed bytes not reread",
    }
    manifest["separation"]["confirmation_signal_hash_comparison"] = (
        "partial: diabetes published source-byte hashes available; CVES and PulseDB sealed hashes unavailable; "
        "no claim of canonical-window signal-hash separation"
    )


def load_manifest(path: Path = SPLIT_PATH) -> dict:
    return json.loads(path.read_text())


def local_subject_paths(dataset: str, subject_key: str) -> list[Path]:
    return inventory()[dataset][subject_key]


def _pulse_field(handle: h5py.File, field: h5py.Dataset, index: int) -> np.ndarray:
    if h5py.check_dtype(ref=field.dtype) is not None:
        return np.asarray(handle[field[0, index]]).reshape(-1)
    if index != 0:
        raise ValueError("Scalar-struct PulseDB has only one window")
    return np.asarray(field).reshape(-1)


def _text(value: np.ndarray) -> str:
    return "".join(chr(int(c)) for c in value).lower()


def _pulse_runs(
    handle: h5py.File, expected_subject: str, first_min_windows: int | None = None, cap_windows: int = 18
) -> list[list[int]]:
    """Strict continuity: SubjectID, CaseID, SegmentID +1 and every T increment."""
    group = handle["Subj_Wins"]
    field = group["SubjectID"]
    count = field.shape[1] if h5py.check_dtype(ref=field.dtype) is not None else 1
    runs, current = [], []
    previous = None
    for index in range(count):
        subject = _text(_pulse_field(handle, group["SubjectID"], index))
        if opaque("pulsedb_mimic", subject) != expected_subject:
            raise ValueError("PulseDB subject mismatch; no identifier emitted")
        case = _text(_pulse_field(handle, group["CaseID"], index))
        segment = int(_pulse_field(handle, group["SegmentID"], index)[0])
        time = _pulse_field(handle, group["T"], index).astype(float)
        valid = len(time) == 1250 and np.allclose(np.diff(time), 1 / 125, rtol=0, atol=1e-6)
        continuous = (
            valid
            and previous is not None
            and previous[0] == case
            and previous[1] + 1 == segment
            and abs(time[0] - previous[2] - 1 / 125) < 1e-6
        )
        if not continuous and current:
            runs.append(current)
            if first_min_windows is not None and len(current) >= first_min_windows:
                return [current[:cap_windows]]
            current = []
        if valid:
            current.append(index)
            previous = (case, segment, time[-1])
            if first_min_windows is not None and len(current) >= cap_windows:
                return [current]
        else:
            previous = None
    if current:
        runs.append(current)
    return runs


def iter_dev_windows(dataset: str, manifest: dict | None = None, duration_s: float = 180):
    """Yield a strict >=60s PulseDB run or full native development WFDB record.

    Prefer the first >=120s run; if none exists, use the first >=60s run.

    Result keys: subject_key, record_key, x, fs, channels, units, start_s,
    duration_s, source_hashes, events. Only pre-frozen candidate subjects are read.
    WFDB event times are absolute record seconds; x begins at start_s=0.
    """
    manifest = load_manifest() if manifest is None else manifest
    entry = manifest["datasets"][dataset]
    if dataset == "diabetes":
        for filename, expected in entry["marker_files"].items():
            if file_hash(ROOTS[dataset] / "Data_Description" / filename) != expected:
                raise ValueError("Manual marker metadata changed after split freeze")
    local = inventory()[dataset]
    metadata = {row["record_key"]: row for row in entry["records"]}
    candidates = []
    for subject in entry["candidate_subjects"]:
        if subject not in entry["development"] or subject in entry["sealed_confirmation"]:
            raise ValueError("Attempt to open a sealed subject")
        candidates.extend((subject, path) for path in local[subject])

    def verify(candidate):
        subject, path = candidate
        key = record_key(dataset, path)
        record = metadata[key]
        source = path if dataset == "pulsedb_mimic" else path.with_suffix(".dat")
        if file_hash(source) != record["source_sha256"]:
            raise ValueError("Development source changed after split freeze")
        return subject, path, key, record

    # Independent source hashing is parallel; waveform decoding and yields retain
    # frozen candidate order. Every partition is checked before hashing starts.
    with ThreadPoolExecutor(max_workers=4) as workers:
        for subject, path, key, record in workers.map(verify, candidates):
            base = {
                "dataset": dataset,
                "subject_key": subject,
                "record_key": key,
                "source_hashes": {k: v for k, v in record.items() if k.endswith("sha256")},
                "events": [],
            }
            if dataset == "pulsedb_mimic":
                with h5py.File(path, "r") as handle:
                    all_runs = _pulse_runs(
                        handle, subject, first_min_windows=12, cap_windows=max(12, int(duration_s // 10))
                    )
                    runs = [run for run in all_runs if len(run) >= 12] or [run for run in all_runs if len(run) >= 6]
                    if not runs:
                        continue
                    selected = runs[0][: max(12, int(duration_s // 10))]
                    group = handle["Subj_Wins"]
                    columns = [
                        np.concatenate([_pulse_field(handle, group[field], i) for i in selected])
                        for field in ("ECG_Record", "PPG_Record", "ABP_Raw")
                    ]
                    if any(len(x) != len(selected) * 1250 for x in columns):
                        raise ValueError("Unexpected PulseDB window shape")
                    base.update(
                        {
                            "x": np.column_stack(columns).astype(np.float64),
                            "fs": 125.0,
                            "channels": ["ECG", "PPG", "ABP"],
                            "units": ["source amplitude", "source amplitude", "mmHg"],
                            "start_s": float(_pulse_field(handle, group["T"], selected[0])[0]),
                            "duration_s": len(selected) * 10.0,
                            "continuity": "SubjectID + CaseID + SegmentID+1 + complete native T grid at 125 Hz",
                            "window_indices": [selected[0], selected[-1]],
                        }
                    )
            else:
                base.update(_read_wfdb_dev(dataset, path, record))
            yield base


def _read_wfdb_dev(dataset: str, path: Path, record: dict) -> dict:
    header = read_header(path)
    header_lines = [line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    for signal, line in zip(header["signals"], header_lines[1:], strict=False):
        # Preserve compound units (cm/s); the core reader exposes only the first component.
        gain_field = line.split()[2]
        signal["unit"] = gain_field.split("/", 1)[1] if "/" in gain_field else "mV"
    if file_hash(path) != record["header_sha256"]:
        raise ValueError("Development WFDB header changed")
    if any(s["format"] != "16" for s in header["signals"]):
        raise ValueError("Audit WFDB reader requires native format 16")
    if not np.isfinite(header["fs"]) or header["fs"] <= 0:
        raise ValueError("Invalid WFDB sampling rate")
    if path.with_suffix(".dat").stat().st_size != header["n_samp"] * header["n_sig"] * 2:
        raise ValueError("WFDB source size differs from header sample count")
    raw = np.memmap(path.with_suffix(".dat"), dtype="<i2", mode="r").reshape(-1, header["n_sig"])
    channels, units, columns = [], [], []
    marker = None
    for i, signal in enumerate(header["signals"]):
        values = (raw[: header["n_samp"], i].astype(float) - signal["baseline"]) / signal["gain"]
        values[raw[: header["n_samp"], i] == -32768] = np.nan
        name = signal["name"].lower()
        if name == "marker":
            marker = values
        elif name in {"ecg", "abp", "mcar", "mcal"}:
            channels.append({"ecg": "ECG", "abp": "ABP", "mcar": "CBFV_R", "mcal": "CBFV_L"}[name])
            units.append(signal["unit"])
            columns.append(values)
    marker_alignment = {}
    events = (
        diabetes_events(path, marker, header["fs"], marker_alignment)
        if dataset == "diabetes" and marker is not None
        else []
    )
    return {
        "x": np.column_stack(columns),
        "fs": header["fs"],
        "channels": channels,
        "units": units,
        "start_s": 0.0,
        "duration_s": header["n_samp"] / header["fs"],
        "events": events,
        "marker_alignment": marker_alignment,
        "marker": marker,
        "protocol": record["protocol"],
    }


def diabetes_events(path: Path, marker: np.ndarray, fs: float, diagnostics: dict | None = None) -> list[dict]:
    """Map manually annotated ordinal pulse labels to recording time."""
    day = 1 if path.parent.name.endswith("Day1") else 2
    table = ROOTS["diabetes"] / "Data_Description" / f"GE-71_Head-up-tilt-Day{day}_Markers_per_subject.csv"
    rows = list(csv.reader(table.open()))
    native = re.sub(r"D[AB]$", "", path.stem, flags=re.IGNORECASE).lower()
    row = next((r for r in rows[1:] if r[0].strip().lower() == native), None)
    if row is None:
        return []
    low, high = np.nanquantile(marker, [0.05, 0.999])
    if high - low < 0.05:
        high = float(np.nanmax(marker))
    if high - low < 0.05:
        return []
    active = marker > low + 0.5 * (high - low)
    starts = np.flatnonzero(np.diff(active.astype(np.int8), prepend=0) == 1)
    starts = starts[np.r_[True, np.diff(starts) > fs]]
    expected_count = max([int(value.strip()) for value in row[2:-1] if value.strip().isdigit()], default=0)
    matched = expected_count == len(starts) and expected_count > 0
    if diagnostics is not None:
        diagnostics.update(
            {
                "expected_pulses": expected_count,
                "detected_pulses": len(starts),
                "status": "accepted" if matched else "excluded",
                "reason": "Exact manual ordinal maximum / detected pulse count agreement required",
            }
        )
    if not matched:
        return []
    labels = [("valsalva", i) for i in ([5, 7, 9] if day == 1 else [3, 5, 7, 9])]
    labels.append(("tilt", 11 if day == 1 else 15))
    events = []
    for kind, column in labels:
        try:
            ordinal = int(row[column].strip())
        except (ValueError, IndexError):
            continue
        if 0 < ordinal <= len(starts):
            events.append(
                {
                    "kind": kind,
                    "t": float(starts[ordinal - 1] / fs),
                    "marker_ordinal": ordinal,
                    "marker_pulse_count": len(starts),
                    "method": "Manual protocol CSV ordinal matched to marker-channel rising edge",
                    "semantics": "CSV values are 1-based pulse ordinals; zero means absent",
                    "alignment": f"marker native sample / {fs:g} Hz; first rising edge at half pulse height",
                    "marker_file_sha256": file_hash(table),
                }
            )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    manifest = freeze_splits() if args.freeze else load_manifest()
    print(
        json.dumps(
            {
                k: {
                    "subjects": v["subject_count"],
                    "dev": len(v["development"]),
                    "sealed": len(v["sealed_confirmation"]),
                    "candidates": len(v["candidate_subjects"]),
                }
                for k, v in manifest["datasets"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
