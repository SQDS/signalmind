"""Two-stage offline execution: consume saved batch predictions, refine, then score separately."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dataset_workflow import read, sha, write  # noqa: E402
from tsmcp.peak_policy import digest  # noqa: E402
from tsmcp.peak_refinement import VERSION, RefinementConfig, refine  # noqa: E402


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(project, out, config):
    out.mkdir(parents=True, exist_ok=False)
    index = read(project / "public/index.json")
    policy = read(project / "agent/policy.json")
    batch_path = project / "outputs/DS_shared_peak_policy/predictions.json"
    batch = read(batch_path)
    assert digest(policy) == read(project / "agent/freeze.json")["policy_sha256"]
    sources = [
        "src/tsmcp/peak_refinement.py",
        "src/tsmcp/peak_policy.py",
        "src/tsmcp/plugins/ecg/rpeaks.py",
        "src/tsmcp/core/dsp.py",
        "src/tsmcp/core/peaks.py",
        "src/tsmcp/processing.py",
        "eval/run_peak_refinement.py",
        "eval/grade_peak_review.py",
    ]
    provenance = {
        "date": "2026-11-06",
        "version": VERSION,
        "config": asdict(config),
        "project": str(project.resolve()),
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "batch_prediction_sha256": sha(batch_path),
        "batch_policy_sha256": digest(policy),
        "public_index_sha256": sha(project / "public/index.json"),
        "protocol_sha256": sha(ROOT / "docs/signalmind/peak-refinement-protocol.md"),
        "algorithm_origin": "Developer-authored local program after reviewing previous failures; not a new DS session.",
        "labels_read_by_refinement": False,
        "evaluation_role": "Post-hoc development regression on already observed cases.",
    }
    for name in sources:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    write(out / "provenance.json", provenance)
    results, operations, queue, counts = {}, [], [], Counter()
    started = time.monotonic()
    for case in index["cases"]:
        uid = case["id"]
        path = project / "public" / uid / "input.npz"
        assert sha(path) == case["sha256"]
        with np.load(path, allow_pickle=False) as data:
            x, old = data["ecg"].copy(), data["old_peaks"].copy()
        original = x.copy()
        result = refine(x, old, batch[uid], policy, config=config)
        np.testing.assert_array_equal(x, original)
        assert result["exclude"] == batch[uid]["exclude"]
        results[uid] = result
        counts["windows"] += 1
        counts["windows_with_edits"] += bool(result["refinement_operations"])
        counts["windows_for_review"] += result["review_required"]
        counts["review_queue_entries"] += len(result["review_queue"])
        for item in result["refinement_operations"]:
            operations.append({"id": uid, **item})
            counts[item["operation"]] += 1
        queue.extend({"id": uid, **item} for item in result["review_queue"])
    elapsed = time.monotonic() - started
    assert sha(batch_path) == provenance["batch_prediction_sha256"]
    write(out / "predictions.json", results)
    save_csv(out / "operations.csv", operations)
    save_csv(out / "review_queue.csv", queue)
    summary = {
        "version": VERSION,
        "config": asdict(config),
        "counts": dict(counts),
        "elapsed_s": elapsed,
        "new_model_calls": 0,
        "added_api_cost_usd": 0,
        "source_waveforms_modified": False,
        "stage1_predictions_modified": False,
        "window_decisions_revised": 0,
        "review_queue_is_exclusion_decision": False,
        "predictions_sha256": sha(out / "predictions.json"),
        "quality_verdict": None,
    }
    write(out / "run-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def score(project, out):
    # References only enter this explicit post-execution stage.
    from eval.grade_peak_review import grade_case, summarize

    provenance = read(out / "provenance.json")
    for name, value in provenance["source_sha256"].items():
        assert sha(ROOT / name) == sha(out / "source" / name) == value
    assert sha(out / "predictions.json") == read(out / "run-summary.json")["predictions_sha256"]
    truth = {r["id"]: r for r in read(project / "private/truth.json")}
    first = read(project / "outputs/DS_shared_peak_policy/predictions.json")
    second = read(out / "predictions.json")
    policy = read(project / "agent/policy.json")
    config = RefinementConfig(**provenance["config"])
    summary, details, queue_counts = {}, [], Counter()
    oracle_sources = {"batch_only": first, "batch_then_program": second}
    rescored = {name: [] for name in oracle_sources}
    for case in read(project / "public/index.json")["cases"]:
        uid = case["id"]
        with np.load(project / "public" / uid / "input.npz", allow_pickle=False) as data:
            old = data["old_peaks"].copy()
            rerun = refine(data["ecg"], old, first[uid], policy, config=config)
        assert rerun == second[uid]
        target = truth[uid]
        if second[uid]["review_required"]:
            queue_counts["reference_excluded" if target["disposition"] == "EXCLUDED" else "reference_retained"] += 1
        for arm, predictions in oracle_sources.items():
            result = grade_case(target, old, predictions[uid])
            rescored[arm].append({**target, "counts": result})
            details.append(
                {
                    "arm": arm,
                    "group_id": target["group_id"],
                    "subject_key": target["subject_key"],
                    "reference_action": target["disposition"],
                    "review_required": second[uid]["review_required"],
                    "delete_tp": result.get("delete_tp", ""),
                    "delete_fp": result.get("delete_fp", ""),
                    "delete_fn": result.get("delete_fn", ""),
                    "correct_moves_8ms": result.get("moves_correct_within_1_samples", ""),
                    "reference_moves": result.get("reference_moves", ""),
                    "retained_window_pass_8ms": result.get("retained_windows_end_to_end_8ms", ""),
                }
            )
    for arm, rows in rescored.items():
        summary[arm] = {
            "all_440": summarize(rows),
            "original_unseen_subject_subset": summarize([r for r in rows if not r["policy_development_subject"]]),
        }
    assert summary["batch_only"]["all_440"] == read(project / "report.json")["arms"]["DS_shared_peak_policy"]["all_440"]
    report = {
        "date": "2026-11-06",
        "arms": summary,
        "fixed_dsp_reference": read(project / "report.json")["arms"]["fixed_dsp"],
        "review_queue_reference_composition": dict(queue_counts),
        "run": read(out / "run-summary.json"),
        "labels_used_in_processing": False,
        "prior_scores_seen_by_developer": True,
        "new_blind_evaluation": False,
        "deterministic_prediction_replay_exact": True,
        "note": "New development regression after previous failure analysis. Review flags are not counted as correct exclusions. No fresh DS call or full RRI reconstruction.",
    }
    write(out / "scores.json", report)
    save_csv(out / "case_results.csv", details)
    for name, scopes in summary.items():
        print(name, json.dumps(scopes["all_440"], ensure_ascii=False, indent=2))
    print("review_queue", dict(queue_counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "score"])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        config = RefinementConfig(**read(args.config)) if args.config else RefinementConfig()
        run(args.project, args.out, config)
    else:
        score(args.project, args.out)


if __name__ == "__main__":
    main()
