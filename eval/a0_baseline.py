"""Frozen, annotation-blind A0 predictions, followed by a separate grading command.

Run ``predict`` once per frozen server tree, then ``grade`` on the saved predictions.
Prediction reads only the public questions, signal.npy and visible.json. It never
imports an annotation reader or opens answer, eligibility, or diagnostic files.
M's equivalent episode API is ecg_irregular; its <=15 s native grouping is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "M": "114b99372b20e80feb6c2bc69f3da0696921b5b7",
    "R": "1d344ae403616056d3bd7888ec097d2f8144f015",
}
MODELS = ("claude-opus-5-5-medium", "deepseek-flash_deepseek-flash-high")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def select_longest(episodes):
    """Rank complete native spans, never supported seconds, score, or list order.

    No intervals are merged here. R already supplies its frozen <30 s grouping;
    M supplies its frozen <=15 s grouping. Equal spans choose the earliest start.
    """
    if not episodes:
        return None
    return min(episodes, key=lambda e: (-float(e.get("span_s", e["dur"])), float(e["t"])))


def answer_from_payload(qtype, payload):
    if qtype == "af_burden":
        return {"percent": payload["pct_of_window"]}
    if qtype == "longest_af":
        selected = select_longest(payload["episodes"])
        return {"t": None if selected is None else float(selected["t"])}
    raise ValueError(f"Unsupported A0 question: {qtype}")


def verify_source(source_root, arm):
    source_root = Path(source_root).resolve()
    commit = FROZEN[arm]
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "src"], cwd=ROOT, text=True
    ).splitlines()
    hashes = {}
    for name in files:
        expected = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        actual = (source_root / name).read_bytes()
        if actual != expected:
            raise ValueError(f"Frozen source differs: {source_root / name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    extras = {str(p.relative_to(source_root)) for p in (source_root / "src").rglob("*.py")} - set(files)
    if extras:
        raise ValueError(f"Extra source modules in frozen tree: {sorted(extras)}")
    return hashes


def predict(args):
    """Prediction phase: this function has no answer-file argument or reader."""
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source_root.resolve()
    hashes = verify_source(source, args.arm)
    if any(name == "tsmcp" or name.startswith("tsmcp.") for name in sys.modules):
        raise RuntimeError("Run each frozen arm in a fresh process before importing tsmcp")
    sys.path.insert(0, str(source / "src"))
    ecg = importlib.import_module("tsmcp.plugins.ecg")
    package = importlib.import_module("tsmcp")
    channel_class = importlib.import_module("tsmcp.core.signal").Channel
    if not Path(package.__file__).resolve().is_relative_to(source / "src"):
        raise RuntimeError("Imported detector is outside the requested frozen tree")
    questions = read_jsonl(args.questions)
    ids = [q["case_id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate public case IDs")
    counts = {qtype: sum(q["qtype"] == qtype for q in questions) for qtype in ("longest_af", "af_burden")}
    expected = {"longest_af": args.expect_longest, "af_burden": args.expect_burden}
    if counts != expected or len(questions) != sum(expected.values()):
        raise ValueError(f"Expected the complete public question manifest {expected}, got {counts}")
    rows = []
    for question in questions:
        case_id, qtype = question["case_id"], question["qtype"]
        case_dir = args.cases_dir / case_id
        visible_path = case_dir / "visible.json"
        visible = json.loads(visible_path.read_text())
        if visible["file"] != "signal.npy" or len(visible["channels"]) != 1:
            raise ValueError(f"Expected one visible channel in signal.npy: {case_id}")
        channel = visible["channels"][0]
        signal_path = case_dir / "signal.npy"
        x = np.load(signal_path, allow_pickle=False, mmap_mode="r")
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        fs, unit = float(channel["fs"]), channel["unit"]
        if x.ndim != 1 or not math.isfinite(fs) or fs <= 0:
            raise ValueError(f"Invalid single-channel array or sampling frequency: {case_id}")
        duration = len(x) / fs
        if not math.isclose(duration, visible["dur_s"], abs_tol=1 / fs):
            raise ValueError(f"Visible duration differs from sample count: {case_id}")
        # M's burden function reads beats three times; mirror the server's per-case
        # Channel cache. Never reuse a channel or result across separate cases.
        context = {"ch": channel_class(channel["name"], fs, x, unit)} if args.arm == "M" else {}
        start = time.perf_counter()
        if qtype == "af_burden":
            detector = "ecg_af_burden"
            payload = ecg.ecg_af_burden(x, fs, unit, **context)
        elif args.arm == "R":
            detector = "ecg_af_episodes"
            payload = ecg.ecg_af_episodes(x, fs, unit)
            if payload["merge_gap_s"] != 30:
                raise ValueError("R's native 30 s grouping changed")
        else:
            detector = "ecg_irregular"
            payload = {"episodes": ecg.ecg_irregular(x, fs, unit, **context)}
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "case_id": case_id,
                "qtype": qtype,
                "pred": answer_from_payload(qtype, payload),
                "detector": detector,
                "detector_payload": payload,
                "detector_seconds": elapsed,
                "window_s": [0.0, duration],
                "fs": fs,
                "unit": unit,
                "n_samples": len(x),
                "input_sha256": {"signal.npy": sha256(signal_path), "visible.json": sha256(visible_path)},
            }
        )
        print(json.dumps({"arm": args.arm, "case_id": case_id, "seconds": round(elapsed, 3)}), flush=True)
    if verify_source(source, args.arm) != hashes:
        raise ValueError("Source tree changed during prediction")
    result = {
        "schema_version": 1,
        "phase": "predictions; references not read",
        "baseline": "A0" if args.arm == "R" else "A0-M",
        "arm": args.arm,
        "source_root": str(source),
        "source_commit": FROZEN[args.arm],
        "imported_package": package.__file__,
        "source_sha256": hashes,
        "questions_sha256": sha256(args.questions),
        "rule": "whole-window pct_of_window; longest native merged span, earliest start on ties; null if empty",
        "native_grouping": "strictly <30 s (frozen R)" if args.arm == "R" else "<=15 s (frozen M; no extra regrouping)",
        "rows": rows,
    }
    write_new(args.output, result)


def grade_rows(predictions, answers):
    """Use the unchanged public grader; retain a separate explicit onset error."""
    sys.path.insert(0, str(ROOT))
    from eval.grade import grade

    rows = []
    for prediction in predictions:
        row = dict(prediction)
        truth = answers[row["case_id"]]
        row["grade"] = grade(row["qtype"], truth, row["pred"])
        row["answer"] = truth
        key = "t" if row["qtype"] == "longest_af" else "percent"
        value = row["pred"].get(key) if row["pred"] else None
        row["absolute_error"] = None if value is None or truth[key] is None else abs(float(value) - truth[key])
        row["error_unit"] = "s" if key == "t" else "pp"
        rows.append(row)
    return rows


def summarize(rows):
    return {
        qtype: {
            "correct": sum(row["grade"]["correct"] is True for row in rows if row["qtype"] == qtype),
            "total": sum(row["qtype"] == qtype for row in rows),
        }
        for qtype in ("longest_af", "af_burden")
    }


def grade_saved(args):
    predictions = json.loads(args.predictions.read_text())
    answers = {row["case_id"]: row["answer"] for row in read_jsonl(args.answers)}
    if set(answers) != {row["case_id"] for row in predictions["rows"]}:
        raise ValueError("Prediction and reference case sets differ")
    rows = grade_rows(predictions["rows"], answers)
    comparisons = {}
    for model in MODELS:
        for arm in ("M", "R"):
            path = args.runs_dir / f"p5afdb-C-{model}-{arm}" / "results.jsonl"
            agent_rows = read_jsonl(path)
            if len(agent_rows) != len(rows) or {r["case_id"] for r in agent_rows} != set(answers):
                raise ValueError(f"Agent case set differs: {path}")
            graded = grade_rows(agent_rows, answers)
            name = f"{model}-{arm}"
            comparisons[name] = {"summary": summarize(graded), "results_sha256": sha256(path)}
            indexed = {row["case_id"]: row for row in graded}
            for row in rows:
                agent = indexed[row["case_id"]]
                if agent["qtype"] != row["qtype"]:
                    raise ValueError(f"Question type differs: {row['case_id']}")
                baseline_ok = row["grade"]["correct"] is True
                agent_ok = agent["grade"]["correct"] is True
                outcome = (
                    "both_correct"
                    if baseline_ok and agent_ok
                    else "a0_only"
                    if baseline_ok
                    else "agent_only"
                    if agent_ok
                    else "both_wrong"
                )
                row.setdefault("agents", {})[name] = {
                    "pred": agent["pred"],
                    "grade": agent["grade"],
                    "absolute_error": agent["absolute_error"],
                    "outcome": outcome,
                }
            comparisons[name]["paired"] = {
                outcome: sum(row["agents"][name]["outcome"] == outcome for row in rows)
                for outcome in ("both_correct", "a0_only", "agent_only", "both_wrong")
            }
    result = {
        **{key: value for key, value in predictions.items() if key != "rows"},
        "phase": "offline grading after saving annotation-blind predictions",
        "predictions_file": args.predictions.name,
        "predictions_sha256": sha256(args.predictions),
        "answers_sha256": sha256(args.answers),
        "grader_sha256": sha256(ROOT / "eval/grade.py"),
        "summary": summarize(rows),
        "agent_comparisons": comparisons,
        "rows": rows,
    }
    write_new(args.output, result)
    print(json.dumps({"baseline": predictions["baseline"], "summary": result["summary"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prediction = subparsers.add_parser("predict", help="visible inputs only, no truth access")
    prediction.add_argument("--arm", choices=FROZEN, required=True)
    prediction.add_argument("--source-root", type=Path, required=True)
    prediction.add_argument("--cases-dir", type=Path, required=True)
    prediction.add_argument("--questions", type=Path, required=True)
    prediction.add_argument("--expect-longest", type=int, default=15, help="guard: number of longest_af questions")
    prediction.add_argument("--expect-burden", type=int, default=13, help="guard: number of af_burden questions")
    prediction.add_argument("--output", type=Path, required=True)
    grading = subparsers.add_parser("grade", help="offline references only after prediction is saved")
    grading.add_argument("--predictions", type=Path, required=True)
    grading.add_argument("--answers", type=Path, required=True)
    grading.add_argument("--runs-dir", type=Path, required=True)
    grading.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (predict if args.command == "predict" else grade_saved)(args)


if __name__ == "__main__":
    main()
