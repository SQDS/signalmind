"""IRIDIA inputs: checksums, safe extraction, annotation facts, and lead-II copying.

No detector or model is imported or run. HDF5 waveform values are read only by
``read_lead_ii`` when copying an already-selected case.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import stat
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import numpy as np

ARCHIVE = "iridia-af-records-v1.0.1.zip"
ZIP_NAME = ARCHIVE
METADATA = "iridia-af-metadata-v1.0.1.csv"
ARCHIVE_ROOT = "iridia-af-records-v1.0.1"
FS = 200
MIN_SEGMENT_SAMPLES = 20 * 60 * 60 * FS


def verify_iridia(data_dir: Path, checksum_file: Path = Path("/tmp/trackB/iridia-files.txt")) -> dict:
    """Fail on a missing or mismatched Zenodo checksum for either required file."""
    expected = {}
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        url, name, checksum = line.split()
        if name not in {ARCHIVE, METADATA} or name in expected:
            raise ValueError(f"Unexpected or duplicate IRIDIA checksum entry: {name}")
        algorithm, digest = checksum.split(":", maxsplit=1)
        if algorithm != "md5" or len(digest) != 32:
            raise ValueError(f"Unexpected IRIDIA checksum: {checksum}")
        expected[name] = (algorithm, digest, url)
    if set(expected) != {ARCHIVE, METADATA}:
        raise ValueError("IRIDIA checksum manifest must include both ZIP and metadata")
    verified = {}
    for name, (algorithm, digest, url) in sorted(expected.items()):
        path = data_dir / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, algorithm).hexdigest()
        if actual != digest:
            raise ValueError(f"IRIDIA {algorithm} mismatch: {name}")
        verified[name] = {
            "algorithm": algorithm,
            "checksum": actual,
            "size_bytes": path.stat().st_size,
            "source_url": url,
        }
    return {"manifest": str(checksum_file), "verified_file_count": len(verified), "files": verified}


def validate_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Validate the whole archive before writing; reject unsafe/ambiguous paths."""
    members = archive.infolist()
    seen = set()
    files = set()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        mode = member.external_attr >> 16
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or "\x00" in name
            or stat.S_ISLNK(mode)
            or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR})
        ):
            raise ValueError(f"Unsafe ZIP member: {name!r}")
        normalized = path.as_posix()
        if normalized in {"", "."} or normalized in seen:
            raise ValueError(f"Empty or duplicate ZIP member: {name!r}")
        seen.add(normalized)
        if not member.is_dir():
            files.add(path)
    for member in members:
        if any(parent in files for parent in PurePosixPath(member.filename).parents):
            raise ValueError(f"ZIP file shadows a parent directory: {member.filename!r}")
    return members


def extract_iridia_archive(zip_path: Path, records_dir: Path) -> Path:
    """Extract checked names beneath records_dir; never execute downloaded files.

    Existing members are retained only after checking their ZIP CRC and size.
    New files pass zipfile's decompression/CRC check before atomic replacement.
    """
    if records_dir.is_symlink():
        raise ValueError("IRIDIA extraction directory must not be a symlink")
    with zipfile.ZipFile(zip_path) as archive:
        members = validate_zip_members(archive)
        records_dir.mkdir(parents=True, exist_ok=True)
        root = records_dir.resolve()
        for member in members:
            destination = records_dir.joinpath(*PurePosixPath(member.filename).parts)
            if destination.is_symlink() or not destination.resolve().is_relative_to(root):
                raise ValueError(f"ZIP destination escapes extraction root: {member.filename!r}")
            current = records_dir
            for part in destination.relative_to(records_dir).parts:
                current /= part
                if current.is_symlink():
                    raise ValueError(f"ZIP destination contains a symlink: {member.filename!r}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == member.file_size:
                checksum = 0
                with destination.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        checksum = zlib.crc32(block, checksum)
                if checksum == member.CRC:
                    continue
            temporary = destination.with_name(destination.name + ".extracting")
            if temporary.is_symlink():
                raise ValueError(f"ZIP temporary destination is a symlink: {temporary.name!r}")
            try:
                with archive.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if temporary.stat().st_size != member.file_size:
                    raise ValueError(f"Incomplete ZIP extraction: {member.filename!r}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return records_dir / ARCHIVE_ROOT


def _metadata_rows(data_dir: Path) -> list[dict]:
    with (data_dir / METADATA).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = set()
    for row in rows:
        record, patient = row["record_id"], row["patient_id"]
        if (
            not record.startswith("record_")
            or not record.removeprefix("record_").isdigit()
            or not patient.startswith("patient_")
            or not patient.removeprefix("patient_").isdigit()
            or record in records
        ):
            raise ValueError(f"Invalid or duplicate IRIDIA identity: {record!r}, {patient!r}")
        records.add(record)
        if int(row["record_files"]) <= 0 or int(row["record_samples"]) <= 0:
            raise ValueError(f"Invalid IRIDIA recording size: {record}")
    return sorted(rows, key=lambda row: row["record_id"])


def _waveform_header(path: Path) -> dict:
    """Read HDF5 structure only, never values, while collecting eligibility."""
    import h5py

    with h5py.File(path, "r") as handle:
        if list(handle) != ["ecg"]:
            raise ValueError(f"Unexpected IRIDIA HDF5 dataset names: {path.name}")
        ecg = handle["ecg"]
        if not isinstance(ecg, h5py.Dataset) or ecg.ndim != 2 or ecg.shape[1] != 2 or ecg.shape[0] <= 0:
            raise ValueError(f"Unexpected IRIDIA waveform shape: {path.name}")
        if ecg.dtype != np.dtype("float64"):
            raise ValueError(f"Unexpected IRIDIA waveform dtype: {path.name}")
        return {"n_samples": int(ecg.shape[0]), "dtype": str(ecg.dtype), "n_channels": 2}


def _record_intervals(path: Path, lengths: list[int]) -> list[tuple[int, int]]:
    """Read manual AF intervals using sample coordinates, not rounded durations."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    intervals = []
    for row in rows:
        start_file = int(row["start_file_index"])
        end_file = int(row["end_file_index"])
        start_index = int(row["start_qrs_index"])
        end_index = int(row["end_qrs_index"])
        if not (0 <= start_file <= end_file < len(lengths)):
            raise ValueError(f"IRIDIA annotation file indices out of range: {path.name}")
        if not (0 <= start_index <= lengths[start_file] and 0 <= end_index <= lengths[end_file]):
            raise ValueError(f"IRIDIA annotation sample indices out of range: {path.name}")
        start, end = offsets[start_file] + start_index, offsets[end_file] + end_index
        if start >= end or (intervals and start < intervals[-1][1]):
            raise ValueError(f"IRIDIA annotations overlap, are unsorted, or empty: {path.name}")
        intervals.append((start, end))
    return intervals


def inspect_iridia(data_dir: Path) -> dict:
    """Return every source-file segment with metadata and manual AF annotations.

    Files are *nominal* daily splits: their exact duration remains visible in
    n_samples and is never padded or silently rounded to 86,400 seconds. The
    caller applies the frozen duration/eligibility policy. RR data is not read.
    """
    rows = _metadata_rows(data_dir)
    records_root = data_dir / "records" / ARCHIVE_ROOT
    segments = []
    durations = []
    for row in rows:
        record = row["record_id"]
        record_dir = records_root / record
        file_count = int(row["record_files"])
        paths = [record_dir / f"{record}_ecg_{index:02d}.h5" for index in range(file_count)]
        if sorted(record_dir.glob(f"{record}_ecg_*.h5")) != paths:
            raise ValueError(f"IRIDIA ECG files differ from metadata: {record}")
        headers = [_waveform_header(path) for path in paths]
        lengths = [header["n_samples"] for header in headers]
        if sum(lengths) != int(row["record_samples"]):
            raise ValueError(f"IRIDIA sample count differs from metadata: {record}")
        intervals = _record_intervals(record_dir / f"{record}_ecg_labels.csv", lengths)
        offset = 0
        for index, (path, length) in enumerate(zip(paths, lengths, strict=True)):
            local = [
                [max(start, offset) - offset, min(end, offset + length) - offset]
                for start, end in intervals
                if start < offset + length and end > offset
            ]
            segments.append(
                {
                    "database": "iridia-af",
                    "patient_id": row["patient_id"],
                    "record": record,
                    "segment": f"{record}:{index:02d}",
                    "file_index": index,
                    "start_sample": offset,
                    "n_samples": length,
                    "fs": FS,
                    "af_intervals": local,
                    "annotation_available": True,
                    "waveform_file": path.relative_to(data_dir).as_posix(),
                    "source_n_samples": length,
                    "channel": 1,
                    "lead": "II",
                    "units": "mV",
                }
            )
            offset += length
            durations.append(length / FS)
    return {
        "formats": {
            "layout": f"records/{ARCHIVE_ROOT}/record_NNN/record_NNN_ecg_XX.h5",
            "waveform_format": "HDF5 ecg dataset; float64 [samples, 2]; no HDF5 attributes",
            "annotation_format": "CSV manual AF onset/offset file indices and waveform sample indices",
            "annotation_coordinates": "start_qrs_index/end_qrs_index at 200 Hz; [start,end) intervals",
            "unused_files": "RR HDF5 and RR-label CSV; never parsed as truth",
            "metadata_file": METADATA,
            "patient_identity_column": "patient_id",
            "lead_order": ["I", "II"],
            "selected_lead": "II",
            "selected_channel": 1,
            "units": "mV",
            "units_and_annotation_source": "https://github.com/cedricgilon/iridia-af/blob/main/iridia_af/record.py",
            "fs": FS,
            "format_source": "https://www.nature.com/articles/s41597-023-02621-1",
            "record_count": len(rows),
            "patient_count": len({row["patient_id"] for row in rows}),
            "file_count": len(segments),
            "file_duration_s_min": min(durations),
            "file_duration_s_max": max(durations),
            "files_at_least_20h": sum(duration >= MIN_SEGMENT_SAMPLES / FS for duration in durations),
            "exact_24h_file_count": sum(duration == 86400 for duration in durations),
            "nominal_daily_split": "Source files are retained with their actual sample counts; no padding or duration rounding",
            "calibration": "First 30 seconds of each recording retained, as part of the whole-segment denominator",
        },
        "segments": segments,
    }


def read_lead_ii(data_dir: Path, segment: dict) -> np.ndarray:
    """Copy exactly the chosen source segment's lead II as float32, unprocessed."""
    import h5py

    relative = PurePosixPath(segment["waveform_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe IRIDIA waveform path")
    path = data_dir.joinpath(*relative.parts)
    allowed = (data_dir / "records").resolve()
    if not path.resolve().is_relative_to(allowed):
        raise ValueError("IRIDIA waveform path escapes records directory")
    length = int(segment["n_samples"])
    local_start = int(segment.get("file_start_sample", 0))
    with h5py.File(path, "r") as handle:
        ecg = handle["ecg"]
        if length < MIN_SEGMENT_SAMPLES or local_start != 0 or length != ecg.shape[0] or ecg.shape[1] != 2:
            raise ValueError("IRIDIA frozen segment must be one whole waveform file of at least 20 hours")
        return np.asarray(ecg[:, 1], dtype=np.float32)
