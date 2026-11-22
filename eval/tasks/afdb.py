"""Sealed AFDB cases: annotation-only eligibility, checked local inputs, no detectors.

Freeze ``collect_eligibility()`` as afdb_eligibility.json and commit it before building.
AFDB ATR files contain rhythms only. Each independently eligible rhythm endpoint
becomes a separate case; no HR question or automatic QRS truth is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from eval.tasks import longrec

AFDB = Path(__file__).resolve().parents[2] / "data" / "afdb"
ELIGIBILITY = Path(__file__).with_name("afdb_eligibility.json")
NO_WAVEFORM = {"00735", "03665"}


def verify_data(data_dir: Path) -> dict:
    """Hash every downloaded file; require the inputs used for every waveform record.

    The upstream manifest also lists undownloaded backups and ancillary files. Keep
    their names in the audit, but never confuse missing backups with hash mismatches.
    QRS bytes are hashed for integrity only; they are never parsed as truth.
    """
    manifest = data_dir / "SHA256SUMS.txt"
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        if name in expected or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid or duplicate SHA256 manifest entry: {name}")
        expected[name] = digest
    hashes = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        name = path.relative_to(data_dir).as_posix()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected.get(name) != digest:
            raise ValueError(f"SHA256 mismatch or unlisted file: {name}")
        hashes[name] = digest
    if "RECORDS" not in hashes:
        raise ValueError("Missing required input: RECORDS")
    records = (data_dir / "RECORDS").read_text().split()
    if len(records) != len(set(records)) or any(not rec.isdigit() or len(rec) != 5 for rec in records):
        raise ValueError("Invalid or duplicate AFDB record IDs")
    for rec in records:
        for ext in ("hea", "atr") if rec in NO_WAVEFORM else ("hea", "atr", "dat"):
            if f"{rec}.{ext}" not in hashes:
                raise ValueError(f"Missing required input: {rec}.{ext}")
    return {
        "algorithm": "sha256",
        "manifest": "SHA256SUMS.txt",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "verified_file_count": len(hashes),
        "files": hashes,
        "not_downloaded_manifest_entries": sorted(expected.keys() - hashes.keys()),
    }


def select_types(facts: dict) -> list[str]:
    """Independent endpoints; the longest-AF rule has no lower burden bound."""
    longest, burden = facts["longest_af"], facts["af_percent"]
    types = []
    if longest and longest["dur_s"] > 300 and longest["start"] >= 120 and burden < 90:
        types.append("longest_af")
    if 5 < burden < 95:
        types.append("af_burden")
    return types


def annotation_facts(rec: str, header, annotation) -> dict:
    if header.fs != 250 or header.n_sig != 2 or header.fmt != ["212", "212"] or header.sig_len <= 0:
        raise ValueError(f"Unexpected AFDB sampling rate/channel layout/format: {rec}")
    samples = np.asarray(annotation.sample)
    if np.any(np.diff(samples) < 0) or np.any(samples < 0) or np.any(samples >= header.sig_len):
        raise ValueError(f"ATR sample coordinates outside the record or unsorted: {rec}")
    segments = longrec.af_segments(annotation, header.fs, header.sig_len)
    episodes = longrec.merge_episodes(segments)
    longest = max(episodes, key=lambda pair: pair[1] - pair[0]) if episodes else None
    duration = header.sig_len / header.fs
    af_s = float(sum(end - start for start, end in segments))
    beat_count = sum(sym in longrec.BEATS for _, sym in zip(samples, annotation.symbol, strict=True))
    facts = {
        "fs": float(header.fs),
        "n_samples": header.sig_len,
        "dur_s": duration,
        "af_s": af_s,
        "af_percent": 100 * af_s / duration,
        "n_af_segments": len(segments),
        "n_merged_af_episodes": len(episodes),
        "longest_af": None
        if longest is None
        else {"start": float(longest[0]), "end": float(longest[1]), "dur_s": float(longest[1] - longest[0])},
        "atr_beat_count": beat_count,
    }
    return {"types": select_types(facts), **facts, "answer_ready": True}


def collect_eligibility(data_dir: Path = AFDB) -> dict:
    """Inspect headers and ATR only, after verifying all downloaded data hashes."""
    import wfdb

    hashes = verify_data(data_dir)
    records, excluded = {}, {}
    for rec in (data_dir / "RECORDS").read_text().split():
        if rec in NO_WAVEFORM:
            excluded[rec] = "No waveform distributed"
            continue
        header = wfdb.rdheader(str(data_dir / rec))
        annotation = wfdb.rdann(str(data_dir / rec), "atr")
        records[rec] = annotation_facts(rec, header, annotation)
    counts = {kind: sum(kind in row["types"] for row in records.values()) for kind in ("longest_af", "af_burden")}
    return {
        "schema_version": 2,
        "dataset": "MIT-BIH Atrial Fibrillation Database 1.0.0",
        "data_dir": "data/afdb",
        "truth": ".atr only; .qrs never used as truth",
        "time_coordinates": "Seconds from record start = sample index / 250; before the first rhythm marker is non-AF",
        "rules": {
            "af_labels": ["(AFIB", "(AFL"],
            "merge_gap_strictly_below_s": 30,
            "burden": "100 * raw AF seconds / whole-record seconds; merged gaps excluded",
            "endpoints": {"primary": "longest_af", "secondary": "af_burden"},
            "selection": "Independent eligibility per endpoint; one question per qualifying record per endpoint; each case is a separate session",
            "longest_af": "longest merged duration > 300 s and onset >= 120 s and burden < 90%",
            "af_burden": "5% < burden < 95%",
        },
        "counts": counts,
        "build_ready": True,
        "blocker": None,
        "records": records,
        "excluded": excluded,
        "data_hashes": hashes,
    }


def _validate_assignments(frozen: dict) -> None:
    """Reject invalid assignments or missing truth before decoding any waveform."""
    for rec, facts in frozen["records"].items():
        types = facts.get("types")
        if not isinstance(types, list) or any(kind not in ("longest_af", "af_burden") for kind in types):
            raise ValueError(f"Unsupported AFDB endpoint types: {rec}: {types!r}")
        if len(types) != len(set(types)):
            raise ValueError(f"Duplicate AFDB endpoint types: {rec}: {types!r}")
        truth = []
        if "longest_af" in types:
            longest = facts.get("longest_af")
            if not isinstance(longest, dict):
                raise ValueError(f"Cannot build AFDB without trusted answers for {rec}: missing longest AF episode")
            truth.extend((longest.get("start"), longest.get("dur_s")))
        if "af_burden" in types:
            truth.append(facts.get("af_percent"))
        if not facts.get("answer_ready") or any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in truth):
            raise ValueError(f"Cannot build AFDB without trusted answers for {rec}")


def prepare_afdb(data_dir: Path = AFDB, eligibility_path: Path = ELIGIBILITY) -> dict:
    """Preflight before creating/removing case directories or decoding waveforms."""
    frozen = json.loads(eligibility_path.read_text())
    if collect_eligibility(data_dir) != frozen:
        raise ValueError("AFDB data or annotation facts differ from the frozen eligibility JSON")
    _validate_assignments(frozen)
    return frozen


def build_afdb(frozen: dict, data_dir: Path = AFDB):
    """Stream channel 0 in the LTAFDB layout, with private answers outside cases."""
    import wfdb

    _validate_assignments(frozen)
    questions = {"af_burden": longrec.Q_BURDEN, "longest_af": longrec.Q_LONGEST}
    i = 0
    for rec, facts in frozen["records"].items():
        if not facts["types"]:
            continue
        record = wfdb.rdrecord(str(data_dir / rec), channels=[0])
        if record.fs != facts["fs"] or record.p_signal.shape != (facts["n_samples"], 1):
            raise ValueError(f"Waveform layout differs from frozen header: {rec}")
        x = record.p_signal[:, 0].astype(np.float32)
        meta = {"record": rec, "dur_s": facts["dur_s"]}
        for kind in facts["types"]:
            if kind == "af_burden":
                answer = {"percent": facts["af_percent"], **meta}
            else:
                longest = facts["longest_af"]
                answer = {"t": longest["start"], "longest_dur_s": longest["dur_s"], **meta}
            # Preserve the question form while correctly describing this ~10 h database.
            question = questions[kind].replace("about a day", "about 10 hours")
            yield f"afdb{i:03d}", kind, "-", x, facts["fs"], answer, question
            i += 1
