"""SHDB-AF inputs for the sealed confirmation suite; annotations only as truth.

Waveforms are decoded only by ``read_signal``, after the caller freezes eligibility.
No automatic QRS annotations, detector, or model are used here.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

FS = 200
MIN_SEGMENT_SAMPLES = 20 * 60 * 60 * FS
AF_LABELS = {"(AFIB", "(AFL"}


def verify_shdb(data_dir: Path) -> dict:
    """Require every manifest file to exist and match its SHA256 digest."""
    data_dir = Path(data_dir)
    manifest = data_dir / "SHA256SUMS.txt"
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or name in expected
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"Invalid or duplicate SHDB SHA256 entry: {name}")
        expected[name] = digest
    if not expected or "AdditionalData.csv" not in expected or "RECORDS.txt" not in expected:
        raise ValueError("SHDB manifest lacks required metadata")
    actual_names = {path.relative_to(data_dir).as_posix() for path in data_dir.rglob("*") if path.is_file()}
    if actual_names != set(expected) | {manifest.name}:
        raise ValueError("SHDB files differ from SHA256 manifest")
    for name, digest in expected.items():
        path = data_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlink SHDB input: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"SHDB SHA256 mismatch: {name}")
    return {
        "algorithm": "sha256",
        "manifest": manifest.name,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "verified_file_count": len(expected),
        "files": expected,
    }


def rhythm_intervals(samples, aux_notes, n_samples: int) -> list[list[int]]:
    """Read rhythm changes regardless of beat symbol; initial unknown time is non-AF."""
    samples = np.asarray(samples)
    if (
        samples.ndim != 1
        or len(samples) != len(aux_notes)
        or np.any(samples < 0)
        or np.any(samples >= n_samples)
        or np.any(np.diff(samples) < 0)
    ):
        raise ValueError("SHDB ATR samples must be sorted and within the recording")
    intervals = []
    start = None
    for sample, note in zip(samples, aux_notes, strict=True):
        note = note.rstrip("\x00").strip()
        if not note.startswith("("):
            continue
        sample = int(sample)
        is_af = note in AF_LABELS
        if start is not None and not is_af:
            if sample > start:
                intervals.append([start, sample])
            start = None
        elif start is None and is_af:
            start = sample
    if start is not None and start < n_samples:
        intervals.append([start, n_samples])
    return intervals


def _read_header(data_dir: Path, record: str):
    import wfdb

    header = wfdb.rdheader(str(data_dir / record))
    if (
        header.fs != FS
        or header.n_sig != 2
        or header.fmt != ["16", "16"]
        or header.sig_len <= 0
        or header.file_name != [f"{record}.dat", f"{record}.dat"]
        or header.samps_per_frame != [1, 1]
        or any(value not in (None, 0) for value in header.byte_offset + header.skew)
        or header.sig_name != ["ECG1", "ECG2"]
        or header.units != ["mV", "mV"]
        or any(not np.isfinite(value) or value <= 0 for value in header.adc_gain)
    ):
        raise ValueError(f"Unexpected SHDB header: {record}")
    if (data_dir / f"{record}.dat").stat().st_size != header.sig_len * 2 * 2:
        raise ValueError(f"SHDB waveform byte size differs from header: {record}")
    return header


def inspect_shdb(data_dir: Path) -> dict:
    """Inspect metadata, headers and manual ATR only; retain short/missing-truth rows.

    Under C1 each distributed recording is one whole segment. The caller excludes
    recordings shorter than 20 hours; longer recordings are never truncated.
    """
    import wfdb

    data_dir = Path(data_dir)
    with (data_dir / "AdditionalData.csv").open(newline="") as stream:
        metadata = list(csv.DictReader(stream))
    records = (data_dir / "RECORDS.txt").read_text().split()
    by_record = {row["Data_ID"]: row for row in metadata}
    if (
        len(by_record) != len(metadata)
        or len(set(records)) != len(records)
        or set(by_record) != set(records)
        or any(len(record) != 3 or not record.isdigit() for record in records)
    ):
        raise ValueError("SHDB records and patient metadata disagree")
    segments = []
    labels = Counter()
    durations = []
    annotated_records = []
    for record in sorted(records):
        row = by_record[record]
        patient_id = row["Subject_ID"]
        if not patient_id or row["Annotated"] not in ("True", "False"):
            raise ValueError(f"Invalid SHDB patient identity or annotation availability: {record}")
        header = _read_header(data_dir, record)
        available = (data_dir / f"{record}.atr").is_file()
        if available != (row["Annotated"] == "True"):
            raise ValueError(f"SHDB ATR presence differs from patient metadata: {record}")
        n_samples = header.sig_len
        intervals = []
        if available:
            annotated_records.append(record)
            annotation = wfdb.rdann(str(data_dir / record), "atr")
            labels.update(note.rstrip("\x00").strip() for note in annotation.aux_note if note.startswith("("))
            intervals = rhythm_intervals(annotation.sample, annotation.aux_note, n_samples)
        durations.append(header.sig_len / FS)
        segments.append(
            {
                "database": "shdb-af",
                "patient_id": patient_id,
                "record": record,
                "segment": f"{record}:0",
                "file_index": 0,
                "start_sample": 0,
                "n_samples": n_samples,
                "record_n_samples": header.sig_len,
                "fs": FS,
                "af_intervals": intervals,
                "annotation_available": available,
            }
        )
    return {
        "formats": {
            "waveform": "WFDB .dat format 16, interleaved signed little-endian int16",
            "header": "WFDB text .hea",
            "annotations": "WFDB .atr aux_note rhythm transitions (manual); .qrs never parsed",
            "metadata": "AdditionalData.csv: Data_ID maps recordings to Subject_ID",
            "channels": ["ECG1", "ECG2"],
            "selected_channel": 0,
            "units": "mV",
            "fs": FS,
            "record_count": len(records),
            "patient_count": len({row["Subject_ID"] for row in metadata}),
            "annotated_record_count": len(annotated_records),
            "annotated_records": annotated_records,
            "rhythm_labels": dict(sorted(labels.items())),
            "duration_s_min": min(durations),
            "duration_s_max": max(durations),
            "records_at_least_20h": sum(duration >= MIN_SEGMENT_SAMPLES / FS for duration in durations),
            "records_shorter_than_24h": sum(duration < 86400 for duration in durations),
            "records_exactly_24h": sum(duration == 86400 for duration in durations),
            "records_longer_than_24h": sum(duration > 86400 for duration in durations),
            "initial_unannotated_time": "Non-AF until the first rhythm marker",
            "non_af_labels": "All rhythm labels other than (AFIB and (AFL, including observed (AB",
        },
        "segments": segments,
    }


def read_signal(data_dir: Path, row: dict) -> np.ndarray:
    """Decode only frozen segment channel 0 to physical millivolts, float32."""
    data_dir = Path(data_dir)
    header = _read_header(data_dir, row["record"])
    start, size = row["start_sample"], row["n_samples"]
    if (
        row["database"] != "shdb-af"
        or row["fs"] != FS
        or header.sig_len != row["record_n_samples"]
        or start != 0
        or size < MIN_SEGMENT_SAMPLES
        or size != header.sig_len
    ):
        raise ValueError("SHDB frozen segment differs from the waveform header")
    raw = np.memmap(data_dir / header.file_name[0], dtype="<i2", mode="r", shape=(header.sig_len, 2))
    digital = raw[start : start + size, 0]
    signal = digital.astype(np.float32)
    signal -= header.baseline[0]
    signal /= header.adc_gain[0]
    signal[digital == -32768] = np.nan
    return signal
