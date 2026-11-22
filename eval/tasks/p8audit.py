"""Free capability audit builder. Never imports a model client or opens sealed waveforms.

Run ``.venv/bin/python -m eval.tasks.p8audit`` after committing audit_splits.json.
The public cases contain only signal.npy and visible.json; all truth stays outside.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal

from eval.common import CASES

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/tasks/p8audit_manifest.json"
SEALED_HASHES = ROOT / "eval/tasks/audit_sealed_hashes.json"
SUPERSEDED = ROOT / "eval/results/audit-prep/superseded-27/p8audit_manifest.json"
SEED = 20261026
TILT_EXTRA_CROPS = [(120, 50), (120, 55), (125, 60), (135, 65), (150, 70), (150, 80)]
# Row, family, qtype, GT, requested n, expected capability, scoring group.
ROWS = [
    (1, "PPG", "audit_rate", "G2", 3, "solvable", "core"),
    (2, "PPG", "audit_motion", "G3", 2, "solvable", "core"),
    (3, "PPG", "audit_step", "G3", 2, "solvable", "core"),
    (4, "PPG", "audit_unusable", "G3", 1, "partial", "core"),
    (5, "ABP", "audit_bp", "G4", 2, "partial", "core"),
    (6, "ABP", "audit_valsalva", "G1", 2, "solvable", "core"),
    (7, "ABP", "audit_damped", "G3", 2, "solvable", "core"),
    (8, "ABP", "audit_tilt", "G1", 2, "solvable", "core"),
    (9, "Multi", "audit_shift", "G3", 2, "partial", "core"),
    (10, "Multi", "audit_pat", "G4", 2, "partial", "core"),
    (11, "Multi", "audit_corrupted", "G3", 2, "solvable", "core"),
    (12, "Multi", "audit_response", "G1+G4", 2, "partial", "core"),
    (13, "Multi", "audit_brs", "G4", 2, "gap-probe", "gap-probe"),
    (14, "CBFV", "audit_cbfv_tilt", "G1", 2, "solvable", "core"),
    (15, "CBFV", "audit_tfa", "G4", 2, "gap-probe", "gap-probe"),
]
TOLERANCES = {
    "audit_rate": {"absolute": 3, "unit": "bpm"},
    "audit_motion": {"absolute": 2, "unit": "s"},
    "audit_step": {"absolute": 5, "unit": "s"},
    "audit_unusable": {"absolute": 10, "unit": "percentage points"},
    "audit_bp": {"absolute": 5, "unit": "mmHg", "all_components": True},
    "audit_valsalva": {"absolute": 10, "unit": "s"},
    "audit_damped": {"absolute": 5, "unit": "s"},
    "audit_tilt": {"absolute": 15, "unit": "s"},
    "audit_shift": {"absolute": 0.05, "unit": "s", "channel": "exact"},
    "audit_pat": {"absolute": 20, "unit": "ms"},
    "audit_corrupted": {"channel": "exact"},
    "audit_response": {"absolute": 10, "unit": "s", "directions": "exact"},
    "audit_brs": {"relative": 0.25, "unit": "ms/mmHg"},
    "audit_cbfv_tilt": {"absolute": 15, "unit": "s"},
    "audit_tfa": {"relative": 0.20, "unit": "(cm/s)/mmHg"},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def perturb(x: np.ndarray, fs: float, names: list[str], qtype: str, seed: int) -> tuple:
    """Perturb a private copy; parameters, not detector output, define G3 truth."""
    x = np.array(x, dtype=np.float32, copy=True)
    rng = np.random.default_rng(seed)
    n = len(x)
    start = int(rng.uniform(0.38, 0.57) * n)
    end = min(n, start + int(rng.uniform(18, 28) * fs))
    name = "PPG" if qtype in ("audit_motion", "audit_step", "audit_unusable") else "ABP"
    if qtype in ("audit_shift", "audit_corrupted"):
        name = ("PPG", "ABP")[seed % 2]
    j = names.index(name)
    y = x[:, j].copy()
    scale = float(np.std(y))
    params = {"seed": seed, "channel": name, "start_s": start / fs, "end_s": end / fs}
    answer = {"t": start / fs}
    if qtype == "audit_motion":
        noise = rng.normal(size=end - start)
        noise = signal.sosfiltfilt(signal.butter(2, [0.4, 8], fs=fs, btype="bandpass", output="sos"), noise)
        noise *= 2 * scale / np.std(noise)
        x[start:end, j] += noise
        params.update(kind="bandlimited_motion_burst", noise_sd=2 * scale, band_hz=[0.4, 8])
    elif qtype == "audit_step":
        baseline = signal.sosfiltfilt(signal.butter(2, 0.3, fs=fs, output="sos"), y)
        factor = float(rng.choice([0.45, 1.65]))
        x[start:, j] = baseline[start:] + factor * (y[start:] - baseline[start:])
        params.update(kind="AC_amplitude_step", factor=factor, baseline_lowpass_hz=0.3, end_s=n / fs)
    elif qtype == "audit_unusable":
        # Two defect types prevent a single flat-fraction summary from answering.
        spans = [(int(0.19 * n), int(0.31 * n)), (start, min(n, start + int(0.17 * n)))]
        a, b = spans[0]
        x[a:b, j] = float(np.median(y[a:b]))
        a, b = spans[1]
        center = float(np.median(y[a:b]))
        limits = [center - 0.02 * scale, center + 0.02 * scale]
        x[a:b, j] = np.clip(y[a:b], *limits)
        params.update(
            kind="flat_and_severe_clipping",
            intervals_s=[[a / fs, b / fs] for a, b in spans],
            interval_kinds=["flat", "clipped"],
            clipping_limits=limits,
        )
        answer = {"percent": 100 * sum(b - a for a, b in spans) / n}
    elif qtype == "audit_damped":
        if seed % 2:
            damped = signal.sosfiltfilt(signal.butter(3, 0.35, fs=fs, output="sos"), y)
            x[start:end, j] = damped[start:end]
            params.update(kind="lowpass_damping", cutoff_hz=0.35, order=3)
        else:
            x[start:end, j] = float(np.median(y[start:end]))
            params.update(kind="flat", value=float(x[start, j]))
    elif qtype == "audit_shift":
        shift = int(round(rng.choice([-0.32, 0.24, 0.40]) * fs))
        indices = np.clip(np.arange(start, n) - shift, 0, n - 1)
        x[start:, j] = y[indices]
        params.update(kind="late_channel_shift", shift_samples=shift, shift_s=shift / fs, end_s=n / fs)
        answer = {"channel": name, "shift_s": shift / fs}
    elif qtype == "audit_corrupted":
        # Replace only the late part by an implausibly fast oscillation, retaining baseline and scale.
        t = np.arange(n - start) / fs
        hz = 3.7
        x[start:, j] = np.median(y) + 1.8 * scale * np.sin(2 * np.pi * hz * t)
        params.update(kind="rate_corruption", frequency_hz=hz, amplitude=1.8 * scale, end_s=n / fs)
        answer = {"channel": name}
    else:
        raise ValueError("unknown perturbation")
    return x, answer, params


def public_question(qtype: str, names: list[str], duration: float, event_s: float | None = None) -> str:
    questions = {
        "audit_rate": "What is the mean PPG pulse rate over the full window, in beats/minute (bpm)?",
        "audit_motion": "At what time does the inserted motion artefact burst begin in PPG (seconds)?",
        "audit_step": "At what time does the persistent step in PPG pulsatile (AC) amplitude begin (seconds)?",
        "audit_unusable": "What percentage of the full PPG window is unusable for pulse morphology because of inserted flat or severely clipped stretches? Count the complete duration of both inserted defects, without attributing native imperfections to the perturbation.",
        "audit_bp": "Estimate mean beatwise SBP, mean beatwise DBP, and mean beatwise MAP over the full ABP window, all in mmHg. SBP and DBP are the maximum and minimum within complete cardiac cycles; Each beat MAP is its pressure integral divided by beat duration; average complete beats with equal weights.",
        "audit_valsalva": "At what time does the first Valsalva manoeuvre in this case begin, inferred from the ABP response (seconds)?",
        "audit_damped": "At what time does the inserted damped or flat ABP stretch begin (seconds)?",
        "audit_tilt": "At what time does head-up tilt begin, inferred from the ABP response (seconds)?",
        "audit_shift": "One pulse channel has an additional time shift in the second part of the window. Using the initial third as its synchronized baseline, identify the shifted channel and the added shift in seconds. Positive means delayed; negative means advanced. Preserve the physiological baseline ECG-to-pulse delay; report only the added shift.",
        "audit_pat": "Estimate mean pulse arrival time from each ECG R peak to the following ABP minimum foot, in milliseconds, averaging complete accepted beats equally. Define the foot on ABP smoothed by a centered quadratic fit with max(5, round(0.080*fs)|1) samples (the bitwise OR makes the window odd). Search the earliest systolic maximum inclusively from R+100 ms through min(R+750 ms, next R+300 ms); the foot is the earliest minimum from R+32 ms through 40 ms before that maximum. Round time offsets to native samples. Retain arrivals 32–600 ms after R with systolic-minus-foot pressure >=8 mmHg. A complete beat has a subsequent detected foot; pair each retained foot to its immediately preceding R once.",
        "audit_corrupted": "The channel rates disagree in the latter part of the window. Which one channel was corrupted? Compare with the initially synchronized ECG and pulse channels; return its channel name.",
        "audit_response": f"Tilt onset is at {event_s} s. Estimate HR and SBP response directions (increase, decrease, unchanged) and each latency in seconds after tilt. Use the response convention below.",
        "audit_brs": "Estimate sequence baroreflex sensitivity, in ms/mmHg, using the sequence convention below.",
        "audit_cbfv_tilt": "At what time does head-up tilt begin, inferred jointly from MCA cerebral blood-flow velocity CBFV and ABP (seconds)?",
        "audit_tfa": "Estimate absolute ABP-to-CBFV transfer-function gain in the LF band 0.07–0.20 Hz, in (cm/s)/mmHg, using the spectral convention below.",
    }
    mapping = ", ".join(f"column {i} = {n}" for i, n in enumerate(names))
    return (
        questions[qtype]
        + f" All times refer to the start of this {duration:g} s case, with half-open intervals. The array is samples × channels: {mapping}. When loading, assign these channel names in this order."
    )


def write_case(case_id: str, x: np.ndarray, fs: float, names: list[str], units: list[str]) -> dict:
    d = CASES / "p8audit" / case_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "signal.npy", np.asarray(x, dtype=np.float32), allow_pickle=False)
    visible = {
        "file": "signal.npy",
        "dur_s": len(x) / fs,
        "channels": [{"name": name, "unit": unit, "fs": fs} for name, unit in zip(names, units, strict=True)],
    }
    (d / "visible.json").write_text(json.dumps(visible, indent=2) + "\n")
    return visible


def _cache_pool(splits: dict) -> list[dict]:
    """Materialize small, predeclared development windows, never entire records."""
    from eval.tasks.audit_sources import iter_dev_windows

    cache = ROOT / "eval/cache/p8audit"
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "index.json"
    split_sha = digest(ROOT / "eval/tasks/audit_splits.json")
    reader_sha = digest(Path(__file__).with_name("audit_sources.py"))
    window_builder_sha = hashlib.sha256(inspect.getsource(_cache_pool).encode()).hexdigest()
    if index.exists():
        old = json.loads(index.read_text())
        if (
            old["split_sha256"] == split_sha
            and old["reader_sha256"] == reader_sha
            and old.get("window_builder_sha256") == window_builder_sha
        ):
            for window in old["windows"]:
                if digest(cache / f"{window['cache_key']}.npy") != window["cache_sha256"]:
                    raise ValueError("Cached development waveform changed; rebuild cache")
            return old["windows"]
    windows = []
    for dataset in ("pulsedb_mimic", "diabetes", "cves"):
        for record in iter_dev_windows(dataset, splits, duration_s=180):
            raw = record["x"]
            fs = record["fs"]
            names = list(record["channels"])
            units = list(record["units"])
            # Choose the right MCA where present; left is a deterministic fallback.
            cerebral = next((name for name in ("CBFV_R", "CBFV_L") if name in names), None)
            keep = [i for i, name in enumerate(names) if not name.startswith("CBFV_") or name == cerebral]
            names = ["CBFV" if names[i] == cerebral else names[i] for i in keep]
            units = [units[i] for i in keep]
            specs = [(0.0, min(300.0, record["duration_s"]), None)]
            if dataset == "pulsedb_mimic":
                specs = [(0.0, record["duration_s"], None)]
            else:
                for event in record["events"]:
                    offset_seed = int(
                        hashlib.sha256((record["record_key"] + str(event["t"])).encode()).hexdigest()[:8], 16
                    )
                    before = 60 + offset_seed % 31
                    begin = max(0.0, event["t"] - before)
                    end = min(record["duration_s"], begin + 180)
                    if begin + 45 <= event["t"] <= end - 60:
                        specs.append((begin, end, event))
            for begin, end, event in specs:
                if end - begin < (60 if dataset == "pulsedb_mimic" else 120):
                    continue
                a, b = round(begin * fs), round(end * fs)
                data = np.asarray(raw[a:b, keep], dtype=np.float32)
                key = hashlib.sha256(f"{record['record_key']}:{a}:{b}".encode()).hexdigest()[:24]
                np.save(cache / f"{key}.npy", data, allow_pickle=False)
                meta = {
                    "cache_key": key,
                    "cache_sha256": digest(cache / f"{key}.npy"),
                    "dataset": dataset,
                    "subject_key": record["subject_key"],
                    "record_key": record["record_key"],
                    "source_hashes": record["source_hashes"],
                    "window_start_s": record["start_s"] + a / fs,
                    "window_duration_s": (b - a) / fs,
                    "fs": fs,
                    "channels": names,
                    "units": units,
                    "event": event,
                    "event_case_s": None if event is None else event["t"] - a / fs,
                    "earlier_same_kind_in_window": event is not None
                    and any(
                        a / fs <= other["t"] < event["t"] and other["kind"] == event["kind"]
                        for other in record["events"]
                    ),
                    "original_channel_names": [record["channels"][i] for i in keep],
                    "marker_alignment": record.get("marker_alignment"),
                }
                if "continuity" in record:
                    meta.update(continuity=record["continuity"], window_indices=record["window_indices"])
                windows.append(meta)
            print(json.dumps({"cached_dataset": dataset, "windows": len(windows)}), flush=True)
    index.write_text(
        json.dumps(
            {
                "split_sha256": split_sha,
                "reader_sha256": reader_sha,
                "window_builder_sha256": window_builder_sha,
                "windows": windows,
            },
            indent=2,
        )
        + "\n"
    )
    return windows


def _extra_tilt_windows(pool: list[dict], splits: dict) -> list[dict]:
    """P1: search fixed shorter crops while preserving the tilt estimand support.

    Original candidates retain priority. Extra crops use only previously verified
    development windows, with >=50 s before and >=65 s after the same marker;
    neither the [-45,-10) baseline nor [0,60] response interval is changed.
    """
    cache = ROOT / "eval/cache/p8audit"
    extra = []
    for meta in pool:
        if not _eligible(meta, "audit_response"):
            continue
        split = splits["datasets"][meta["dataset"]]
        if meta["subject_key"] not in split["development"] or meta["subject_key"] in split["sealed_confirmation"]:
            raise ValueError("Attempt to crop a sealed subject")
        x = np.load(cache / f"{meta['cache_key']}.npy", allow_pickle=False, mmap_mode="r")
        fs = meta["fs"]
        for duration, before in TILT_EXTRA_CROPS:
            a = round((meta["event_case_s"] - before) * fs)
            b = a + round(duration * fs)
            if a < 0 or b > len(x):
                continue
            start = meta["window_start_s"] + a / fs
            key = hashlib.sha256(f"p1-tilt:{meta['record_key']}:{start}:{duration}".encode()).hexdigest()[:24]
            path = cache / f"{key}.npy"
            np.save(path, np.asarray(x[a:b], dtype=np.float32), allow_pickle=False)
            extra.append(
                {
                    **meta,
                    "cache_key": key,
                    "cache_sha256": digest(path),
                    "window_start_s": start,
                    "window_duration_s": (b - a) / fs,
                    "event_case_s": meta["event_case_s"] - a / fs,
                    "candidate_search": {
                        "amendment": "P1",
                        "parent_cache_sha256": meta["cache_sha256"],
                        "crop_duration_s": duration,
                        "marker_before_s": before,
                        "rule": "Fixed alternate crop order after all original tilt windows; unchanged reference and quality gates",
                    },
                }
            )
    return extra


def _sealed_separation(items: list[dict], splits: dict) -> dict:
    """Compare frozen source/case bytes without decoding any sealed waveform."""
    from eval.audit_checks import _sealed_inventory

    errors = []
    inventory = _sealed_inventory(
        {"amendment": "P1", "sealed_hash_inventory": {"path": SEALED_HASHES.name, "sha256": digest(SEALED_HASHES)}},
        MANIFEST,
        splits,
        ROOT / "eval/tasks/audit_splits.json",
        errors,
    )
    if errors or not inventory["complete"]:
        raise ValueError("Invalid sealed raw-byte inventory: " + "; ".join(errors))
    sealed = inventory["hashes"]
    source_hashes = {value for item in items for value in item["source"]["source_hashes"].values()}
    signal_hashes = {item["signal_sha256"] for item in items}
    if sealed & (source_hashes | signal_hashes):
        raise ValueError("Development case overlaps sealed raw-file hash inventory")
    return {
        **{key: value for key, value in splits["separation"].items() if key != "confirmation_waveforms_opened"},
        "confirmation_waveforms_decoded": False,
        "confirmation_raw_source_bytes_hashed": True,
        "confirmation_signal_hash_comparison": "complete: all sealed source/header raw SHA-256 values compared with every development case source and signal; no waveform decoding or canonical-waveform duplicate claim",
        "sealed_source_records_hashed": inventory["records"],
        "sealed_raw_files_hashed": inventory["raw_files"],
        "development_case_source_hashes_compared": len(source_hashes),
        "development_case_signal_hashes_compared": len(signal_hashes),
        "source_hash_overlap": 0,
        "signal_hash_overlap": 0,
    }


def _eligible(meta: dict, qtype: str) -> bool:
    names = set(meta["channels"])
    event = meta["event"]
    if qtype == "audit_valsalva":
        return (
            event is not None
            and event["kind"] == "valsalva"
            and "ABP" in names
            and not meta.get("earlier_same_kind_in_window", False)
        )
    if qtype in ("audit_tilt", "audit_response", "audit_cbfv_tilt"):
        needed = {"ABP", "CBFV"} if qtype == "audit_cbfv_tilt" else {"ECG", "ABP"}
        return event is not None and event["kind"] == "tilt" and needed <= names
    if event is not None:
        return False
    if qtype == "audit_tfa":
        return {"ABP", "CBFV"} <= names and meta["window_duration_s"] >= 180
    return {"ECG", "PPG", "ABP"} <= names


def _truth(meta: dict, qtype: str, seed: int) -> tuple:
    from eval import audit_reference as ref

    x = np.load(ROOT / "eval/cache/p8audit" / f"{meta['cache_key']}.npy", allow_pickle=False)
    names, fs = meta["channels"], meta["fs"]
    col = {name: x[:, i] for i, name in enumerate(names)}
    if not np.isfinite(x).all():
        raise ValueError("nonfinite source samples")
    if qtype in {"audit_valsalva", "audit_tilt", "audit_cbfv_tilt"}:
        # G1 truth comes from protocol labels; signal must at least carry pulsatile evidence.
        if np.ptp(col["ABP"]) < 10 or (qtype == "audit_cbfv_tilt" and np.ptp(col["CBFV"]) < 2):
            raise ValueError("insufficient response signal amplitude")
        comparison = {"accepted": True, "marker": meta["event"], "method": meta["event"]["method"]}
        return x, {"t": meta["event_case_s"]}, comparison
    if qtype == "audit_rate":
        result = ref.g2_pulse_rate(col["ECG"], col["PPG"], fs)
        return x, {"bpm": result["value"]}, result
    functions = {"audit_bp": ref.compare_bp, "audit_pat": ref.compare_pat, "audit_brs": ref.compare_brs}
    if qtype in functions:
        result = functions[qtype](col["ECG"], col["ABP"], fs)
        value = result["reference1"]
        answer = value if qtype == "audit_bp" else {"pat_ms" if qtype == "audit_pat" else "brs_ms_mmhg": value}
        return x, answer, result
    if qtype == "audit_tfa":
        result = ref.compare_tfa(col["ABP"], col["CBFV"], fs)
        return x, {"gain": result["reference1"]}, result
    if qtype == "audit_response":
        result = ref.compare_tilt(col["ECG"], col["ABP"], fs, meta["event_case_s"])
        result["marker"] = meta["event"]
        return x, result["reference1"], result
    # Establish usable native pulse channels before inserting defects.
    quality_ppg = ref.g2_pulse_rate(col["ECG"], col["PPG"], fs)
    quality_abp = ref.g2_pulse_rate(col["ECG"], col["ABP"], fs)
    if not quality_ppg["accepted"] or not quality_abp["accepted"]:
        raise ValueError("native ECG/PPG/ABP failed clean cross-channel rate gate")
    x, answer, params = perturb(x, fs, names, qtype, seed)
    return (
        x,
        answer,
        {
            "accepted": True,
            "perturbation": params,
            "baseline_quality": {"PPG": quality_ppg, "ABP": quality_abp},
            "method": "Seeded operation on quality-gated real synchronized channels; exact sample boundaries define truth",
        },
    )


def build() -> dict:
    from eval.tasks.audit_sources import SPLIT_PATH, load_manifest

    # Require the precise split bytes in git before any source waveform can be opened.
    committed = subprocess.check_output(["git", "show", "HEAD:eval/tasks/audit_splits.json"], cwd=ROOT)
    if committed != SPLIT_PATH.read_bytes():
        raise ValueError("Commit the frozen splits before constructing audit questions")
    split_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "eval/tasks/audit_splits.json"], cwd=ROOT, text=True
    ).strip()
    splits = load_manifest()
    pool = _cache_pool(splits)
    extra_tilt = _extra_tilt_windows(pool, splits)
    suite_dir = CASES / "p8audit"
    if suite_dir.exists():
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True)
    questions, answers, items, exclusions = [], [], [], []
    used = {family: set() for family in ("PPG", "ABP", "Multi", "CBFV")}
    # Allocate scarce G4 items first; no subject may be reused within its family.
    order = [10, 13, 12, 15, 5, 1, 2, 3, 4, 6, 7, 8, 9, 11, 14]
    for number in order:
        row, family, qtype, gt, requested, expected, group = ROWS[number - 1]
        count = 0
        for meta in pool + extra_tilt if qtype == "audit_response" else pool:
            if count == requested:
                break
            if meta["subject_key"] in used[family] or not _eligible(meta, qtype):
                continue
            seed = int(hashlib.sha256(f"{SEED}:{row}:{meta['cache_key']}".encode()).hexdigest()[:8], 16)
            details = {
                "row": row,
                "qtype": qtype,
                "gt_category": gt,
                "dataset": meta["dataset"],
                "subject_key": meta["subject_key"],
                "record_key": meta["record_key"],
                "window_start_s": meta["window_start_s"],
                "window_duration_s": meta["window_duration_s"],
            }
            if "candidate_search" in meta:
                details["candidate_search"] = meta["candidate_search"]
            try:
                x, answer, provenance = _truth(meta, qtype, seed)
            except (ValueError, FileNotFoundError) as error:
                exclusions.append({**details, "reason": str(error), "stage": "quality_or_reference_support"})
                continue
            if not provenance["accepted"]:
                exclusions.append(
                    {
                        **details,
                        "reason": "independent agreement/quality gate failed",
                        "stage": "agreement",
                        "comparison": provenance,
                    }
                )
                continue
            selected = meta["channels"]
            if family == "PPG":
                selected = ["PPG"]
            elif family == "ABP":
                selected = ["ABP"]
            elif family == "CBFV":
                selected = ["ABP", "CBFV"]
            indices = [meta["channels"].index(name) for name in selected]
            x = x[:, indices]
            units = [meta["units"][i] for i in indices]
            cid = hashlib.sha256(f"p8audit:{SEED}:{row}:{meta['cache_key']}".encode()).hexdigest()[:24]
            write_case(cid, x, meta["fs"], selected, units)
            question = public_question(qtype, selected, len(x) / meta["fs"], meta["event_case_s"])
            conventions = {
                "audit_brs": "Use maximal same-direction SBP ramps of at least 3 beats, successive SBP changes >=1 mmHg and concordant RR changes >=5 ms, correlation r>=0.85, slope strictly between 0 and 50 ms/mmHg. Pair SBP in R_k cycle with RR_k (delay 1). Discard ramps lacking support RR k-1 through k+1, or any support RR outside 300–2000 ms. Average slopes of accepted ramps equally; require at least 3 ramps.",
                "audit_tfa": "Use source-declared cm/s for CBFV without rescaling. Average each nonoverlapping second; fs=1 Hz, 100-s periodic Hann windows with 50% overlap and mean removal. Average auto/cross spectra, compute abs(Sxy/Sxx), and average LF bins 0.07<=f<0.20 Hz equally. This is a numerical source-unit gain, with physical calibration inherited from the recording header.",
                "audit_response": "Interpolate HR=60/RR at RR midpoints and SBP at systolic peaks onto a 1-s grid. Baseline is median [-45,-10) s relative to tilt; response is median [20,50) s. Direction thresholds are 3 bpm HR and 5 mmHg SBP; latency is the first of 5 consecutive seconds crossing the corresponding threshold within [0,60] s. For unchanged direction, return null latency.",
            }
            if qtype in conventions:
                question += " Convention: " + conventions[qtype]
            source = {
                k: meta[k]
                for k in (
                    "dataset",
                    "subject_key",
                    "record_key",
                    "source_hashes",
                    "window_start_s",
                    "window_duration_s",
                    "original_channel_names",
                )
            }
            for key in ("continuity", "window_indices", "candidate_search"):
                if key in meta:
                    source[key] = meta[key]
            confidence = "low" if qtype == "audit_tfa" else "medium" if "G4" in gt else "high"
            reason = (
                "Two separately implemented estimates passed quality checks and agreed within half the scoring tolerance."
                if "G4" in gt
                else "Manual protocol ordinal aligned to native marker samples."
                if gt == "G1"
                else "Clean ECG agreed with independently detected pulse rate."
                if gt == "G2"
                else "Seeded perturbation boundaries and parameters are known exactly; native channels pass cross-channel quality checks."
            )
            items.append(
                {
                    "id": cid,
                    "case_id": cid,
                    "row": row,
                    "family": family,
                    "task_type": qtype,
                    "gt_category": gt,
                    "source": source,
                    "method": provenance.get("methods", provenance.get("method")),
                    "tolerance": TOLERANCES[qtype],
                    "confidence": confidence,
                    "reason": reason,
                    "expected_solvability": expected,
                    "score_group": group,
                    "signal_sha256": digest(CASES / "p8audit" / cid / "signal.npy"),
                    "provenance": provenance,
                }
            )
            questions.append({"case_id": cid, "qtype": qtype, "question": question, "source": "audit"})
            answers.append({"case_id": cid, "qtype": qtype, "answer": answer})
            used[family].add(meta["subject_key"])
            count += 1
        print(json.dumps({"row": row, "accepted": count, "requested": requested}), flush=True)
    items.sort(key=lambda item: (item["row"], item["id"]))
    positions = {item["id"]: i for i, item in enumerate(items)}
    for collection, suffix in ((questions, "questions"), (answers, "answers")):
        collection.sort(key=lambda item: positions[item["case_id"]])
        (CASES / f"p8audit.{suffix}.jsonl").write_text(
            "".join(json.dumps(item, allow_nan=False) + "\n" for item in collection)
        )
    row_counts = [
        {"row": r, "family": f, "qtype": q, "requested": n, "actual": sum(i["row"] == r for i in items)}
        for r, f, q, _, n, _, _ in ROWS
    ]
    manifest = {
        "schema_version": 1,
        "suite": "p8audit",
        "seed": SEED,
        "code_sha256": {
            name: digest(ROOT / name)
            for name in (
                "eval/tasks/p8audit.py",
                "eval/tasks/audit_sources.py",
                "eval/tasks/audit_sealed.py",
                "eval/audit_reference.py",
                "eval/audit_checks.py",
                "eval/audit_formats.py",
                "eval/__init__.py",
                "eval/common.py",
                "eval/grade.py",
            )
        },
        "questions_sha256": digest(CASES / "p8audit.questions.jsonl"),
        "answers_sha256": digest(CASES / "p8audit.answers.jsonl"),
        "split_sha256": digest(SPLIT_PATH),
        "split_commit": split_commit,
        "denominators": {
            "core": sum(i["score_group"] == "core" for i in items),
            "gap_probe": sum(i["score_group"] == "gap-probe" for i in items),
            "total": len(items),
            "planned_core": 26,
            "planned_gap_probe": 4,
        },
        "rows": row_counts,
        "questions": items,
        "exclusions": exclusions,
        "separation": _sealed_separation(items, splits),
        "sealed_hash_inventory": {"path": SEALED_HASHES.name, "sha256": digest(SEALED_HASHES)},
        "amendment": "P1",
        "amendment_details": {
            "date": "2026-10-26",
            "supersedes": {
                "path": str(SUPERSEDED.relative_to(ROOT)),
                "sha256": digest(SUPERSEDED),
                "denominators": json.loads(SUPERSEDED.read_text())["denominators"],
            },
            "pat_agreement_gate_ms": 10,
            "tilt_latency_agreement_gate_s": 5,
            "tilt_directions": "exact",
        },
        "allocation": "Scarce G4 first in fixed row order; first acceptable frozen candidate; one subject per family",
        "conventions_added": {
            "row9": "partial maps design hard to required solvability labels",
            "row12_latency_tolerance_s": 10,
            "row15_gain_relative_tolerance": 0.20,
        },
        "pool_windows": len(pool),
        "candidate_eligibility": {
            "tilt_p1": {
                "original_event_windows": sum(_eligible(meta, "audit_response") for meta in pool),
                "additional_crop_windows": len(extra_tilt),
                "crop_order_duration_and_pre_marker_s": TILT_EXTRA_CROPS,
                "rule": "Try all original candidates first, then fixed shorter crops of already verified development event windows in original record order; preserve [-45,-10) baseline and [0,60] response support; all gates unchanged",
            },
            "pulsedb_mimic": {
                "development_subjects_checked": len(splits["datasets"]["pulsedb_mimic"]["candidate_subjects"]),
                "eligible_subjects": len({w["subject_key"] for w in pool if w["dataset"] == "pulsedb_mimic"}),
                "without_contiguous_60s_run": sorted(
                    set(splits["datasets"]["pulsedb_mimic"]["candidate_subjects"])
                    - {w["subject_key"] for w in pool if w["dataset"] == "pulsedb_mimic"}
                ),
                "window_rule": "First >=120 s run, otherwise first >=60 s run; cap at 180 s",
            },
        },
        "source_marker_alignment": {m["record_key"]: m["marker_alignment"] for m in pool if m.get("marker_alignment")},
        "tfa_limitations": "Numerical agreement only; physical CBFV scale inherits header calibration and is not independently validated. One-second boxcar averaging has limited antialias rejection.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    build()
