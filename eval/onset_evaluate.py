"""Run the single pre-declared Task E boundary candidate on development data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from eval.af_episode_metrics import clipped, sha256, summary  # noqa: E402
from eval.onset_diagnostics import (  # noqa: E402
    LONG_AF_TOL_S,
    diagnostic_summary,
    eligible,
    load_checkpoints,
    longest_diagnostic,
)
from eval.tasks.longrec import af_segments  # noqa: E402
from eval.tasks.mitdb import MITDB  # noqa: E402

DECLARATION = "c80f339"
BASELINE = "1d344ae"
V_SOURCE = "69c9f43"


def frozen_sources():
    """Check all production files against R, and load V's original grouping code."""
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE, "src"], cwd=ROOT, text=True
    ).splitlines()
    hashes = {}
    for name in paths:
        original = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT)
        if (ROOT / name).read_bytes() != original:
            raise ValueError(f"Baseline source changed: {name}")
        hashes[name] = sha256(ROOT / name)
    extra = {str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")} - set(paths)
    if extra:
        raise ValueError("Unexpected production modules")
    code = subprocess.check_output(["git", "show", f"{V_SOURCE}:src/tsmcp/plugins/ecg/af.py"], cwd=ROOT)
    module = types.ModuleType("task_e_frozen_v")
    exec(compile(code, "69c9f43:src/tsmcp/plugins/ecg/af.py", "exec"), module.__dict__)
    module.CONTINUITY_GAP_S = 120.0
    return module, hashes, hashlib.sha256(code).hexdigest()


def load_allowed(name, permitted):
    """Guard the record AND header-referenced signal before any waveform read."""
    from tsmcp.core.io.wfdb import load_wfdb, read_header

    if name not in permitted:
        raise ValueError("Record outside the fixed development population")
    db, rec = name.split("/")
    root = ROOT / "data/ltafdb" if db == "ltafdb" else MITDB if db == "mitdb" else None
    if root is None or not rec.isascii() or not rec.isdigit():
        raise ValueError("Invalid development record")
    path = root / rec
    for ext in ("hea", "atr", "dat"):
        file = path.with_suffix(f".{ext}")
        if not file.is_file() or file.resolve().parent != root.resolve() or file.resolve().name != f"{rec}.{ext}":
            raise ValueError("Development file absent or redirected")
    header = read_header(path.with_suffix(".hea"))
    if {s["file"] for s in header["signals"]} != {f"{rec}.dat"}:
        raise ValueError("Header references a different record")
    channels, _ = load_wfdb(path.with_suffix(".hea"), None, None, None)
    ch = next((c for c in channels if c.name == "MLII"), channels[0]) if db == "mitdb" else channels[0]
    return ch, path


def candidate_diagnostic(base, events):
    diagnostic = longest_diagnostic(base)
    event = events[0] if events else None  # frozen V ranking: longest span, earliest tie
    truth = base["longest_annotated"]
    point = event["onset_s"] if event and event["onset_confident"] else None
    error = point - truth[0] if point is not None and truth else None
    diagnostic.update(
        onset_s=point,
        onset_confident=event["onset_confident"] if event else False,
        onset_lo_s=event["onset_lo_s"] if event else None,
        onset_hi_s=event["onset_hi_s"] if event else None,
        missing_episode=event is None,
        signed_onset_error_s=error,
        onset_within_60s=bool(error is not None and abs(error) <= LONG_AF_TOL_S),
        interval_contains_truth=bool(event and truth and event["onset_lo_s"] <= truth[0] <= event["onset_hi_s"]),
    )
    return diagnostic


def summarize_candidate(rows, originals):
    af_rows = [r for r in rows if r["has_af"]]
    selected = [r for r in af_rows if r["onset_eligible"]]
    result = {}
    for label, population in (("all_af", af_rows), ("eligible", selected)):
        stats = diagnostic_summary(population, "V_ref")
        stats["missing_point"] = stats.pop("missing_detection")
        # The oracle offset is a Phase 1 diagnostic only, not a second-candidate suggestion.
        stats.pop("oracle_constant_offset_ceiling")
        confident = [r for r in population if r["V_ref"]["onset_confident"]]
        stats.update(
            missing_episode=sum(r["V_ref"]["missing_episode"] for r in population),
            onset_confident_false=sum(not r["V_ref"]["onset_confident"] for r in population),
            confident_count=len(confident),
            confident_correct=sum(r["V_ref"]["onset_within_60s"] for r in confident),
            confident_accuracy=sum(r["V_ref"]["onset_within_60s"] for r in confident) / len(confident)
            if confident
            else None,
            unresolved_interval_contains_truth=sum(
                r["V_ref"]["interval_contains_truth"]
                for r in population
                if not r["V_ref"]["onset_confident"] and not r["V_ref"]["missing_episode"]
            ),
        )
        result[label] = stats
    events = [e for r in rows for e in r["events"]]
    burden = summary(originals)
    burden["onset_correct"] = result["all_af"]["onset_correct"]
    result.update(
        burden_overlap_all_records=burden,
        all_events={"count": len(events), "onset_confident_false": sum(not e["onset_confident"] for e in events)},
    )
    return result


def acceptance(baseline_summaries, candidate_summary, support_equal):
    c = candidate_summary
    gains = {a: c["eligible"]["onset_correct"] - s["eligible"]["onset_correct"] for a, s in baseline_summaries.items()}
    metrics = c["burden_overlap_all_records"]
    no_loss = all(
        metrics["burden_mae_window_pp"] <= s["burden_overlap_all_records"]["burden_mae_window_pp"]
        and metrics["burden_mae_readable_pp"] <= s["burden_overlap_all_records"]["burden_mae_readable_pp"]
        and metrics["pooled_sensitivity"] >= s["burden_overlap_all_records"]["pooled_sensitivity"]
        and metrics["pooled_ppv"] >= s["burden_overlap_all_records"]["pooled_ppv"]
        for s in baseline_summaries.values()
    )
    return {
        "required_gain_over_V": 3,
        "onset_gain": gains,
        "onset_pass": gains["V"] >= 3 and gains["R"] > 0,
        "burden_se_ppv_no_loss": no_loss,
        "support_readability_identical": support_equal,
        "pass": gains["V"] >= 3 and gains["R"] > 0 and no_loss and support_equal,
    }


def cold_runtime(permitted, v, refine_events):
    from tsmcp.core.signal import Channel
    from tsmcp.plugins.ecg import af

    original, _ = load_allowed("ltafdb/102", permitted)
    if original.duration < 86400:
        raise ValueError("Runtime record is shorter than 24 hours")
    result = {}
    for arm in ("R", "V", "V_ref"):
        trials, overhead = [], []
        for _ in range(3):
            ch = Channel(original.name, original.fs, original.data[: round(86400 * original.fs)], original.unit)
            module = af if arm == "R" else v
            begin = time.perf_counter()
            events = module.ecg_af_episodes(ch.data, ch.fs, ch.unit, ch=ch)["episodes"]
            module.ecg_af_burden(ch.data, ch.fs, ch.unit, ch=ch)
            boundary_begin = time.perf_counter()
            if arm == "V_ref":
                refine_events(module.af_analysis(ch.data, ch.fs, ch=ch), events, ch.duration)
            end = time.perf_counter()
            trials.append(end - begin)
            overhead.append(end - boundary_begin if arm == "V_ref" else 0.0)
            print(f"24h {arm} trial {len(trials)}: {trials[-1]:.3f} s", flush=True)
        result[arm] = {"cold_seconds": trials, "median_s": float(np.median(trials)), "boundary_seconds": overhead}
    return {
        "record": "ltafdb/102",
        "window_s": 86400,
        "includes": "cold episode generation + burden; V_ref also boundary; excludes loading",
        "arms": result,
    }


def run(args):
    import wfdb

    from eval.onset_boundary import PARAMETERS, refine_events
    from tsmcp.plugins.ecg import af

    partial = args.output.with_suffix(".partial.json")
    if args.output.exists() or partial.exists():
        raise FileExistsError("Candidate output exists; refusing another development run")
    declaration = subprocess.check_output(["git", "show", f"{DECLARATION}:docs/tasks/onset-refine-report.md"], cwd=ROOT)
    if not (ROOT / "docs/tasks/onset-refine-report.md").read_bytes().startswith(declaration):
        raise ValueError("Pre-declaration changed")
    source, inputs, manifests = load_checkpoints(args.cases_dir)
    permitted = [r["record"] for r in source["R"]]
    v, hashes, v_hash = frozen_sources()
    script_hashes = {
        name: sha256(ROOT / name)
        for name in ("eval/onset_boundary.py", "eval/onset_evaluate.py", "eval/onset_diagnostics.py")
    }
    result = {
        "started_utc": datetime.now(UTC).isoformat(),
        "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "phase1_commit": "2e374f482661b965edd4ff76d6e8586ba1b7c885",
        "declaration_commit": DECLARATION,
        "declaration_sha256": hashlib.sha256(declaration).hexdigest(),
        "candidate_parameters": PARAMETERS,
        "source_hashes": hashes,
        "V_af_source_sha256": v_hash,
        "experiment_source_hashes": script_hashes,
        "baseline_inputs": inputs,
        "suite_manifests": manifests,
        "permitted_records": permitted,
        "eligible_records": [r["record"] for r in source["R"] if eligible(r)],
        "long_af_tol_s": LONG_AF_TOL_S,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for r, baseline_v in zip(source["R"], source["V"], strict=True):
        name = r["record"]
        ch, path = load_allowed(name, permitted)
        truth = clipped(af_segments(wfdb.rdann(str(path), "atr"), ch.fs, ch.n), 0, ch.duration)
        if ch.duration != r["window_s"] or truth != r["annotated_af_intervals"]:
            raise ValueError(f"Development data disagree with checkpoint: {name}")
        begin = time.perf_counter()
        analysis = af.af_analysis(ch.data, ch.fs, ch=ch)
        episodes_r = af._episode_rows(analysis, 0, ch.duration, 30)
        events = v._episode_rows(analysis, 0, ch.duration, 120)
        detector_s = time.perf_counter() - begin
        if analysis["supported"] != r["support_intervals"] or analysis["readable"] != r["readable_intervals"]:
            raise ValueError(f"Frozen R support/readability changed: {name}")
        for actual, expected in ((episodes_r, r), (events, baseline_v)):
            spans = sorted([[e["t"], e["t"] + e["dur"]] for e in actual])
            if spans != [[e["start"], e["end"]] for e in expected["detected_episodes"]]:
                raise ValueError(f"Frozen grouping changed: {name}")
        begin = time.perf_counter()
        refined = refine_events(analysis, events, ch.duration)
        boundary_s = time.perf_counter() - begin
        for before, after in zip(events, refined, strict=True):
            for key in ("t", "dur", "span_s", "af_s", "intervals_s", "merged_from", "unknown_s"):
                if before[key] != after[key]:
                    raise ValueError(f"Boundary changed frozen event/support: {name}/{key}")
        row = {
            "record": name,
            "has_af": r["has_af"],
            "onset_eligible": eligible(r),
            "R": longest_diagnostic(r),
            "V": longest_diagnostic(baseline_v),
            "V_ref": candidate_diagnostic(baseline_v, refined),
            "events": [
                {
                    k: e[k]
                    for k in (
                        "t",
                        "dur",
                        "span_s",
                        "af_s",
                        "intervals_s",
                        "merged_from",
                        "unknown_s",
                        "readable_non_af_gap_s",
                        "onset_s",
                        "onset_lo_s",
                        "onset_hi_s",
                        "onset_confident",
                    )
                }
                for e in refined
            ],
            "event_interpretation": "Grouped candidate events; gaps are not uninterrupted AF",
            "input_sha256": {ext: sha256(path.with_suffix(f".{ext}")) for ext in ("hea", "atr")},
            "waveform_bytes": path.with_suffix(".dat").stat().st_size,
            "R_analysis_and_R_V_grouping_seconds": detector_s,
            "boundary_seconds": boundary_s,
            "support_readability_and_grouping_equal": True,
        }
        result["records"].append(row)
        partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        d = row["V_ref"]
        print(
            f"{name}: eligible={row['onset_eligible']}, confident={d['onset_confident']}, signed_error={d['signed_onset_error_s']}, boundary={boundary_s:.4f}s",
            flush=True,
        )
    rows = result["records"]
    af_rows = [r for r in rows if r["has_af"]]
    result["summary"] = {
        a: {
            "all_af": diagnostic_summary(af_rows, a),
            "eligible": diagnostic_summary([r for r in af_rows if r["onset_eligible"]], a),
            "burden_overlap_all_records": summary(source[a]),
        }
        for a in ("R", "V")
    }
    result["summary"]["V_ref"] = summarize_candidate(rows, source["V"])
    # Independently recompute burden/overlap from retained support, not onset estimates.
    from eval.af_episode_metrics import score_record

    rescored = [
        score_record(r["record"], b["window_s"], b["annotated_af_intervals"], r["events"], b["readable_intervals"])
        for r, b in zip(rows, source["V"], strict=True)
    ]
    recomputed = summary(rescored)
    for key in ("burden_mae_window_pp", "burden_mae_readable_pp", "pooled_sensitivity", "pooled_ppv"):
        assert recomputed[key] == result["summary"]["V_ref"]["burden_overlap_all_records"][key]
    result["gate"] = acceptance({a: result["summary"][a] for a in ("R", "V")}, result["summary"]["V_ref"], True)
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["runtime_24h"] = cold_runtime(permitted, v, refine_events)
    result["decision"] = (
        "pass: eligible for explicit opt-in freeze"
        if result["gate"]["pass"]
        else "reject: negative result; stop after one candidate; production R unchanged"
    )
    result["finished_utc"] = datetime.now(UTC).isoformat()
    assert script_hashes == {name: sha256(ROOT / name) for name in script_hashes}
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.unlink()
    print(json.dumps({k: result[k] for k in ("summary", "gate", "runtime_24h", "decision")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
