"""Preregistered free Level 2 shortcuts; no agent, API or dataset discovery.

Constants and detector policies below are fixed before candidate evaluation.
Only leave-one-out predictions read peer truth; other predictions receive an
explicit public-input projection. Unsupported predictions are missing answers,
never an assertion of insufficient evidence. F5 uses a paired shortcut gate:
the same baseline must pass both the answerable control and abstention member.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from eval.b_workflow import FROZEN_R, SERVER_ROOT
from eval.level2_grade import grade_level2, grade_level2_pair, validate_level2_pair

BASELINE_VERSION = "level2-preregistered-v2-l2a"
BASELINE_NAMES = ("constant", "development_median", "detector_only", "constant_abstention")
FIELD_CONSTANTS = {
    "bpm": 72.0,
    "hr_bpm": 72.0,
    "heart_rate_bpm": 72.0,
    "sbp": 120.0,
    "sbp_mmhg": 120.0,
    "dbp": 80.0,
    "map": 93.0,
    "onset_s": 80.0,
    "duration_s": 10.0,
    "t": 80.0,
    "shift_ms": 0.0,
    "drift_ms": 0.0,
    "channel": "PPG",
    "count": 0,
    "n_sequences": 3,
    "valid_sequences": 3,
    "ratio": 1.2,
    "rr_ratio": 1.2,
    "valsalva_ratio": 1.2,
    "ei_ratio": 1.2,
    "ratio_30_15": 1.2,
    "brs_ms_mmhg": 20.0,
    "brs_ms_per_mmhg": 20.0,
    "brs": 20.0,
    "gain": 1.0,
    "phase_rad": 0.0,
    "phase": 0.0,
    "coherence": 0.5,
    "fraction": 1.0,
    "reliable_fraction": 1.0,
    "percent": 100.0,
    "trustworthy": True,
    "interpretable": True,
    "hr_change_bpm": 0.0,
    "sbp_change_mmhg": 0.0,
    "overshoot_mmhg": 0.0,
    "max_rr_time_s": 80.0,
    "min_rr_time_s": 80.0,
    "response_max_time_s": 80.0,
    "response_min_time_s": 80.0,
}
CONSTANT_ABSTENTION = {"insufficient_evidence": True, "reason": "Insufficient evidence for the requested quantity."}
PUBLIC_KEYS = {
    "task_type",
    "x",
    "fs",
    "channels",
    "units",
    "baseline_channel",
    "baseline_window_s",
    "baseline_windows_s",
    "answer_fields",
}


def _finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _answer_fields(candidate: dict) -> list[str]:
    # The schema, never values, is public. A caller can provide answer_fields
    # explicitly so answerable and nonanswerable controls use identical schemas.
    fields = candidate.get("answer_fields")
    if fields is None:
        fields = list(candidate.get("expected_answer", candidate.get("answer", {})))
    return [field for field in fields if isinstance(field, str) and field not in {"reason", "insufficient_evidence"}]


def constant_answer(public: dict) -> dict | None:
    fields = public.get("answer_fields", [])
    if not fields or any(field not in FIELD_CONSTANTS for field in fields):
        return None
    return {field: FIELD_CONSTANTS[field] for field in fields}


def development_median(candidates: list[dict], index: int) -> tuple[dict | None, dict]:
    """Numeric medians and exact category modes, excluding the target row."""
    target = candidates[index]
    peers = [
        candidate.get("expected_answer", candidate.get("answer", {}))
        for i, candidate in enumerate(candidates)
        if i != index and candidate.get("task_type") == target.get("task_type")
    ]
    answer, support = {}, {}
    for field in _answer_fields(target):
        values = [peer[field] for peer in peers if field in peer and peer[field] is not None]
        support[field] = len(values)
        if not values:
            return None, support
        if all(_finite_number(value) for value in values):
            answer[field] = float(statistics.median(values))
        elif all(isinstance(value, (str, bool)) for value in values):
            counts = Counter(json.dumps(value) for value in values)
            encoded = min(counts, key=lambda value: (-counts[value], value))
            answer[field] = json.loads(encoded)
        else:
            return None, support
    return answer or None, support


def verify_frozen_sources(root: Path = SERVER_ROOT) -> dict:
    """The source checks from b_workflow.verify_server, with a truthful local interpreter.

    The B worker additionally requires the R virtualenv. Baselines use the
    user's required local interpreter and verify the same imported source bytes
    instead of changing or pretending to change the interpreter environment.
    """
    root = root.expanduser().resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != FROZEN_R:
        raise ValueError("baseline source is not the frozen R revision")
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", revision, "src"], text=True
    ).splitlines()
    hashes = {}
    for name in names:
        expected = subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:{name}"])
        if (root / name).read_bytes() != expected:
            raise ValueError("frozen R source bytes differ from committed bytes")
        hashes[name] = hashlib.sha256(expected).hexdigest()
    if {str(path.relative_to(root)) for path in (root / "src").rglob("*.py")} - set(names):
        raise ValueError("frozen R contains additional Python source")
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return {
        "server_git_sha": revision,
        "server_source_sha256": hashlib.sha256(canonical).hexdigest(),
        "interpreter": sys.executable,
        "source_verification": "committed bytes, Python inventory and actual imported module origin",
    }


@contextmanager
def frozen_generic_tools(root: Path = SERVER_ROOT):
    """Import only verified R source for this single-threaded evaluation context."""
    provenance = verify_frozen_sources(root)
    source = root.expanduser().resolve() / "src"
    saved_modules = {key: value for key, value in sys.modules.items() if key == "tsmcp" or key.startswith("tsmcp.")}
    saved_path = sys.path[:]
    for key in saved_modules:
        del sys.modules[key]
    try:
        sys.path.insert(0, str(source))
        package = importlib.import_module("tsmcp")
        measures = importlib.import_module("tsmcp.core.measure")
        detectors = importlib.import_module("tsmcp.core.detect")
        ecg = importlib.import_module("tsmcp.plugins.ecg")
        if any(
            not Path(module.__file__).resolve().is_relative_to(source) for module in (package, measures, detectors, ecg)
        ):
            raise ValueError("baseline imported outside the verified frozen R source")
        provenance["server_package_path"] = str(Path(package.__file__).resolve())
        yield {"measures": measures, "detectors": detectors, "ecg": ecg, "provenance": provenance}
    finally:
        for key in list(sys.modules):
            if key == "tsmcp" or key.startswith("tsmcp."):
                del sys.modules[key]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(part) for key, part in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(part) for part in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _evidence_locations(public: dict, tools: dict, field: str, interval: list) -> tuple[dict, dict]:
    """Mechanical extrema locations from frozen RR output or raw pressure samples."""
    fs = float(public["fs"])
    x = np.asarray(public["x"])
    if x.ndim == 1:
        x = x[:, None]
    channels = public["channels"]
    channel = public.get("baseline_channel", "ABP" if field == "sbp_change_mmhg" else "ECG")
    requested = set(public.get("answer_fields", []))
    maximum, minimum = (
        ("max_rr_time_s", "min_rr_time_s") if field == "rr_ratio" else ("response_max_time_s", "response_min_time_s")
    )
    if not requested.intersection({maximum, minimum}):
        return {}, {}
    left, right = int(interval[0] * fs), int(interval[1] * fs)
    segment = x[left:right, channels.index(channel)]
    unit = public.get("units", ["?"] * len(channels))[channels.index(channel)]
    record = {"channel": channel, "window_s": interval}
    try:
        if field == "sbp_change_mmhg":
            # Deliberately generic sample extrema; no beatwise SBP oracle.
            values = np.asarray(segment, dtype=float)
            times = np.arange(left, right, dtype=float) / fs
            record["operation"] = "raw_sample_extrema"
        else:
            record["operation"] = "series_rr_extrema_midpoints"
            series = getattr(tools["ecg"], "series_rr", None)
            if series is None:
                raise ValueError("frozen ECG primitive unavailable")
            output = series(segment, fs, unit, t0=left / fs)
            rr = np.asarray(output["y"], dtype=float)
            times = np.asarray(output["t"], dtype=float) - rr / 2
            values = rr if field == "rr_ratio" else 60 / rr
        usable = np.isfinite(values) & np.isfinite(times)
        values, times = values[usable], times[usable]
        if not len(values):
            raise ValueError("no finite generic localization output")
        locations = {maximum: float(times[np.argmax(values)]), minimum: float(times[np.argmin(values)])}
        record["output"] = locations
        return {name: value for name, value in locations.items() if name in requested}, record
    except (ValueError, RuntimeError, IndexError, KeyError, FloatingPointError):
        record["error"] = "generic operation could not localize the requested extrema"
        return {}, record


def detector_only(public: dict, tools: dict) -> tuple[dict | None, dict]:
    """Mechanical dominant-frequency, top change/flat and ECG-detector outputs.

    No truth values enter this function. All channels are inspected. An explicit
    public baseline_channel restricts the selected initial output; otherwise
    rate uses PPG, ECG, ABP priority and onset uses the highest scored change.
    """
    x = np.asarray(public["x"])
    channels = public["channels"]
    fs = float(public["fs"])
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] != len(channels) or not math.isfinite(fs) or fs <= 0:
        raise ValueError("baseline input requires samples by channels and a finite positive sample rate")
    units = public.get("units", ["?"] * len(channels))
    if len(units) != len(channels):
        raise ValueError("one unit is required per channel")
    window = public.get("baseline_window_s", [0.0, len(x) / fs])
    start, stop = max(0, int(window[0] * fs)), min(len(x), int(window[1] * fs))
    if stop <= start:
        raise ValueError("baseline public window is empty")
    outputs = []
    for i, channel in enumerate(channels):
        segment = x[start:stop, i]
        row = {"channel": channel, "window_s": [start / fs, stop / fs]}
        operations = {
            "dominant_freq": tools["measures"].dominant_freq,
            "mean": tools["measures"].mean,
            "change": tools["detectors"].changes,
            "flat": tools["detectors"].flat,
        }
        if public.get("task_type") in {"verified_rate", "missing_beats"}:
            operations["ecg_hr"] = tools["ecg"].ecg_hr
        for name, function in operations.items():
            try:
                row[name] = _json_value(function(segment, fs, units[i], t0=start / fs))
            except (ValueError, RuntimeError, IndexError, FloatingPointError):
                row[name] = {"error": "generic operation had insufficient usable input"}
        outputs.append(row)
    requested = public.get("baseline_channel")
    selectable = [row for row in outputs if requested is None or row["channel"] == requested]
    task_type = public.get("task_type", "")
    answer, first_output = None, None
    fields = set(public.get("answer_fields", []))
    location_fields = {"max_rr_time_s", "min_rr_time_s", "response_max_time_s", "response_min_time_s"}
    magnitude_fields = fields - location_fields
    period_outputs = []
    localization_output = {}
    if magnitude_fields in ({"hr_change_bpm"}, {"sbp_change_mmhg"}, {"rr_ratio"}):
        windows = public.get("baseline_windows_s", [])
        field = next(iter(magnitude_fields))
        channel = public.get("baseline_channel", "ABP" if field == "sbp_change_mmhg" else "ECG")
        operation, function = {
            "hr_change_bpm": ("ecg_hr", tools["ecg"].ecg_hr),
            "sbp_change_mmhg": ("mean", tools["measures"].mean),
            "rr_ratio": ("ecg_rr_stats", tools["ecg"].ecg_rr_stats),
        }[field]
        required = 1 if field == "rr_ratio" else 2
        if len(windows) == required and channel in channels:
            for interval in windows:
                left, right = int(interval[0] * fs), int(interval[1] * fs)
                if left < 0 or right > len(x) or right <= left:
                    raise ValueError("question-stated baseline window is outside the supplied signal")
                j = channels.index(channel)
                try:
                    output = _json_value(function(x[left:right, j], fs, units[j], t0=left / fs))
                except (ValueError, RuntimeError, IndexError, FloatingPointError):
                    output = {"error": "generic operation had insufficient usable input"}
                period_outputs.append(
                    {"operation": operation, "channel": channel, "window_s": interval, "output": output}
                )
            if field == "rr_ratio":
                bounds = period_outputs[0]["output"].get("min_max_s", [])
                if len(bounds) == 2 and all(_finite_number(value) for value in bounds) and bounds[0] > 0:
                    answer = {field: bounds[1] / bounds[0]}
            elif all(_finite_number(row["output"].get("value")) for row in period_outputs):
                answer = {field: period_outputs[1]["output"]["value"] - period_outputs[0]["output"]["value"]}
            first_output = period_outputs[0]
            locations, localization_output = _evidence_locations(public, tools, field, windows[-1])
            if locations:
                answer = {**(answer or {}), **locations}
    elif fields in ({"sbp"}, {"sbp_mmhg"}, {"map"}):
        pressure = next((row for row in selectable if row["channel"] == "ABP"), None)
        if pressure is not None:
            output = pressure["mean"]
            first_output = {"operation": "mean", "channel": "ABP", "output": output}
            if _finite_number(output.get("value")):
                answer = {next(iter(fields)): output["value"]}
    elif fields in ({"bpm"}, {"hr_bpm"}, {"heart_rate_bpm"}):
        order = {channel: i for i, channel in enumerate(("PPG", "ECG", "ABP"))}
        selectable.sort(key=lambda row: (order.get(row["channel"], 3), channels.index(row["channel"])))
        if selectable:
            row = selectable[0]
            result = row["dominant_freq"]
            first_output = {"operation": "dominant_freq", "channel": row["channel"], "output": result}
            if _finite_number(result.get("value")):
                answer = {next(iter(fields)): 60 * result["value"]}
    elif "onset_s" in fields or fields == {"t"}:
        events = [
            (event, row["channel"], operation)
            for operation in ("change", "flat")
            for row in selectable
            if isinstance(row.get(operation), list)
            for event in row[operation]
        ]
        # Generic change first, then flat; score ranking matches the interface.
        events.sort(key=lambda entry: (entry[2] != "change", -entry[0]["score"], entry[0]["t"], entry[1]))
        if events:
            event, channel, operation = events[0]
            first_output = {"operation": operation, "channel": channel, "output": event}
            answer = {"onset_s" if "onset_s" in fields else "t": event["t"]}
            if "duration_s" in fields and _finite_number(event.get("dur")):
                answer["duration_s"] = event["dur"]
            if "channel" in fields:
                answer["channel"] = channel
    elif task_type == "missing_beats":
        counts = [
            (row["ecg_hr"].get("n_beats"), row["channel"])
            for row in selectable
            if _finite_number(row.get("ecg_hr", {}).get("n_beats"))
        ]
        if len(counts) >= 2:
            count, channel = min(counts)
            answer = {"channel": channel, "count": max(value for value, _ in counts) - count}
            first_output = {"operation": "ecg_hr.n_beats", "channel": channel, "output": {"counts": counts}}
    return answer, {
        "channel_outputs": outputs,
        "period_outputs": period_outputs,
        "localization_output": localization_output,
        "first_output": first_output,
        "unsupported_fields": sorted(fields - set(answer or {})),
        "unsupported": answer is None or bool(fields - set(answer)),
    }


def apply_pair_gate(candidates: list[dict], rows: list[dict]) -> list[dict]:
    """Apply item gates and F5 conjunctions after every prediction is evaluated.

    ``successful_baselines`` always reports raw item correctness. ``rejected_by``
    reports effective shortcuts, which for F5 must pass both members. Malformed
    pairs fail closed via ``pair_gate.valid`` and ``retained=False`` without
    falsely labelling a baseline as a successful shortcut.
    """
    if len(candidates) != len(rows):
        raise ValueError("baseline rows must align with candidates")
    groups = {}
    for index, (candidate, row) in enumerate(zip(candidates, rows, strict=True)):
        row["successful_baselines"] = [name for name in BASELINE_NAMES if row[name]["grade"]["correct"]]
        row["rejected_by"] = list(row["successful_baselines"])
        row["retained"] = not row["rejected_by"]
        row.pop("pair_gate", None)
        if candidate.get("family") == "F5" or "pair_id" in candidate or "pair_role" in candidate:
            pair_id = candidate.get("pair_id")
            key = ("pair", pair_id) if isinstance(pair_id, str) else ("invalid", index)
            groups.setdefault(key, []).append(index)
    for indices in groups.values():
        items = [candidates[index] for index in indices]
        gate = {"valid": False, "member_indices": indices, "rejected_by": [], "baseline_scores": {}}
        try:
            members = validate_level2_pair(items)
        except ValueError as error:
            gate["error"] = str(error)
        else:
            gate.update(valid=True, pair_id=members["control"]["pair_id"])
            gate["baseline_scores"] = {
                name: grade_level2_pair([rows[index][name]["answer"] for index in indices], items)
                for name in BASELINE_NAMES
            }
            gate["rejected_by"] = [name for name in BASELINE_NAMES if gate["baseline_scores"][name]["pair_correct"]]
        for index in indices:
            rows[index]["pair_gate"] = gate
            rows[index]["rejected_by"] = list(gate["rejected_by"])
            rows[index]["retained"] = gate["valid"] and not gate["rejected_by"]
    return rows


def run_baselines(candidates: list[dict], *, root: Path = SERVER_ROOT) -> list[dict]:
    """Evaluate every item, then apply the family-appropriate shortcut gate."""
    rows = []
    with frozen_generic_tools(root) as tools:
        for index, candidate in enumerate(candidates):
            public = {key: candidate[key] for key in PUBLIC_KEYS if key in candidate}
            public["answer_fields"] = _answer_fields(candidate)
            constant = constant_answer(public)
            median, support = development_median(candidates, index)
            detector, metadata = detector_only(public, tools)
            predictions = {
                "constant": (constant, {}),
                "development_median": (median, {"peer_support": support}),
                "detector_only": (detector, metadata),
                "constant_abstention": (dict(CONSTANT_ABSTENTION), {}),
            }
            result = {
                name: {"answer": answer, "grade": grade_level2(answer, candidate), **extra}
                for name, (answer, extra) in predictions.items()
            }
            result["provenance"] = {"baseline_version": BASELINE_VERSION, **tools["provenance"]}
            rows.append(result)
    return apply_pair_gate(candidates, rows)
