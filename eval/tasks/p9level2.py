"""Free Level 2 builder: candidate caches, frozen-R shortcuts and private freeze.

Run with .venv/bin/python -m eval.tasks.p9level2. Never imports a model client.
The split must already be committed. Sources are development-only and all emitted
answers remain outside case directories. Cached generation is hash-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.level2_baselines import BASELINE_VERSION, FIELD_CONSTANTS, development_median, run_baselines
from eval.level2_checks import ROOT, dry_runs, sha, validate
from eval.level2_grade import grade_level2, grade_level2_pair, validate_level2_pair

CACHE = ROOT / "eval/cache/level2"
MANIFEST = ROOT / "eval/tasks/p9level2_manifest.json"
RESULTS = ROOT / "eval/results/level2-build"
TARGETS = {"F1": 10, "F2": 12, "F3": 12, "F4": 10, "F5": 8, "F6": 8}


def abstention_pairs(items):
    pairs = defaultdict(list)
    for item in items:
        if item["family"] == "F5":
            pairs[item.get("pair_id")].append(item)
    for pair in pairs.values():
        validate_level2_pair(pair)
    return dict(pairs)


def select_retry_slots(items):
    """Deterministic maximum slot matching, without repeating a subject."""
    selected = [row for row in items if not row.get("retry_candidate")]
    used = {(row["family"], row["subject_key"]) for row in selected}
    options = defaultdict(list)
    for row in sorted(
        (row for row in items if row.get("retry_candidate")),
        key=lambda row: (row["family"], row["design_slot"], row["retry_rank"]),
    ):
        subject = (row["family"], row["subject_key"])
        slot = (row["family"], row["design_slot"])
        if subject not in used:
            options[slot].append(row)
    assigned = {}
    owners = {}

    def assign(slot, visited):
        if slot in visited:
            return False
        visited.add(slot)
        # Preserve early ranks when an unused subject can fill the slot.
        ordered = sorted(options[slot], key=lambda row: (row["family"], row["subject_key"]) in owners)
        for row in ordered:
            subject = (row["family"], row["subject_key"])
            owner = owners.get(subject)
            if owner is None or assign(owner, visited):
                owners[subject] = slot
                assigned[slot] = row
                return True
        return False

    for slot in sorted(options):
        assign(slot, set())
    selected.extend(assigned.values())
    return sorted(
        selected, key=lambda row: (row["family"], row["task_type"], row["subject_key"], row.get("pair_role", ""))
    )


def retained_median_gate(items):
    """Per-item median gate except for complete F5 counterfactual pairs."""
    reject = set()
    for i, row in enumerate(items):
        prediction, support = development_median(items, i)
        result = grade_level2(prediction, row)
        row["retained_pool_median"] = {"answer": prediction, "peer_support": support, "grade": result}
        if row["family"] != "F5" and result["correct"]:
            reject.add(row["case_id"])
    for pair in abstention_pairs(items).values():
        result = grade_level2_pair([row["retained_pool_median"]["answer"] for row in pair], pair)
        for row in pair:
            row["retained_pool_median"]["pair_grade"] = result
            if result["pair_correct"]:
                reject.add(row["case_id"])
    for row in items:
        row.setdefault("retained_pool_median_history", []).append(row["retained_pool_median"])
    return reject


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cache_binding(part):
    from eval.audit_reference import source_hashes

    modules = [
        "eval/level2_data.py",
        "eval/tasks/audit_sources.py",
        f"eval/level2_{part if part != 'perturb' else 'perturb'}.py",
    ]
    if part == "perturb":
        modules.append("eval/level2_alignment.py")
    return {
        "split_sha256": sha(ROOT / "eval/tasks/level2_splits.json"),
        "source_hashes": {name: sha(ROOT / name) for name in modules},
        "reference_hashes": source_hashes("rpeaks", "beats", "brs", "tfa"),
        "generation_function_sha256": hashlib.sha256(inspect.getsource(generate).encode()).hexdigest(),
    }


def generate():
    from itertools import chain

    from eval import level2_manual, level2_perturb, level2_physiology
    from eval.level2_data import records

    streams = chain(
        records("diabetes"),
        records("cves"),
        records("pulsedb_mimic", duration_s=600, min_duration_s=60, max_records=40),
    )
    builders = [
        ("physiology", lambda: level2_physiology.build_candidates(streams)),
        (
            "perturb",
            lambda: level2_perturb.build_candidates(
                records("pulsedb_mimic", duration_s=600, min_duration_s=120, max_records=100),
                retry_records=chain(
                    records("diabetes", min_duration_s=120, max_records=8),
                    records("cves", min_duration_s=120, max_records=8),
                    records("pulsedb_mimic", duration_s=600, min_duration_s=600, max_records=48),
                ),
            ),
        ),
        ("manual", level2_manual.build_candidates),
    ]
    for part, builder in builders:
        binding = cache_binding(part)
        candidates, exclusions = builder()
        directory = CACHE / part
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, row in enumerate(candidates):
            path = directory / f"{index:03d}.npy"
            np.save(path, row["x"], allow_pickle=False)
            metadata = {key: value for key, value in row.items() if key != "x"}
            metadata["signal_cache"] = str(path.relative_to(ROOT))
            metadata["cache_signal_sha256"] = sha(path)
            rows.append(metadata)
        dump(directory / "candidates.json", rows)
        dump(directory / "exclusions.json", exclusions)
        if binding != cache_binding(part):
            raise ValueError("Generator sources changed while candidates were built")
        dump(
            directory / "generation.json",
            {
                "binding": binding,
                "metadata_sha256": sha(directory / "candidates.json"),
                "exclusions_sha256": sha(directory / "exclusions.json"),
            },
        )
        print(
            json.dumps(
                {
                    "generation": part,
                    "counts": dict(Counter(r["family"] for r in rows)),
                    "failed_attempts": sum(row["stage"] != "retry_schedule_summary" for row in exclusions),
                }
            ),
            flush=True,
        )


def load_candidates():
    candidates = []
    exclusions = []
    for part in ("physiology", "perturb", "manual"):
        path = CACHE / part / "candidates.json"
        if not path.exists():
            raise ValueError("Required candidate generation cache missing")
        stamp_path = path.parent / "generation.json"
        if not stamp_path.is_file():
            raise ValueError("Candidate cache lacks generation binding; rerun --generate")
        stamp = json.loads(stamp_path.read_text())
        if (
            stamp["binding"] != cache_binding(part)
            or stamp["metadata_sha256"] != sha(path)
            or stamp["exclusions_sha256"] != sha(path.parent / "exclusions.json")
        ):
            raise ValueError("Candidate cache sources or metadata changed; rerun --generate")
        rows = json.loads(path.read_text())
        for row in rows:
            signal_path = Path(row.pop("signal_cache"))
            if not signal_path.is_absolute():
                signal_path = ROOT / signal_path
            row["candidate_signal_sha256"] = sha(signal_path)
            if row.get("cache_signal_sha256") != row["candidate_signal_sha256"]:
                raise ValueError("Cached candidate signal hash changed")
            row["x"] = np.load(signal_path, allow_pickle=False)
            row["candidate_cache_part"] = part
        candidates.extend(rows)
        exclusions.extend(json.loads((CACHE / part / "exclusions.json").read_text()))
    # Manual real artefacts replace the last four controlled F2 design slots,
    # before any shortcut is run. Superseded controls still receive every baseline.
    manual = [r for r in candidates if r["candidate_cache_part"] == "manual"]
    surplus = max(0, sum(r["family"] == "F2" for r in candidates) - TARGETS["F2"])
    for row in reversed(candidates.copy()):
        if surplus and manual and row["family"] == "F2" and row["candidate_cache_part"] == "perturb":
            row["design_superseded"] = True
            surplus -= 1
            exclusions.append(
                {
                    "family": "F2",
                    "stage": "design_allocation",
                    "reason": "Replaced controlled slot with validated real manual artefact before baselines",
                    "subject_key": row["subject_key"],
                }
            )
    candidates.sort(key=lambda r: (r["family"], r.get("task_type", ""), r["subject_key"]))
    pairs = abstention_pairs(candidates)
    seen = {}
    for row in candidates:
        key = (row["family"], row["subject_key"])
        previous = seen.get(key)
        same_pair = previous and row["family"] == "F5" and previous.get("pair_id") == row.get("pair_id")
        retry = row.get("retry_candidate") and (not previous or previous.get("retry_candidate"))
        if previous and not (same_pair or retry):
            raise ValueError("Candidate subject reused within a family")
        seen[key] = row
        duration = len(row["x"]) / row["fs"]
        row["duration_s"] = duration
        row.setdefault("source_duration_s", duration)
        row.setdefault(
            "source",
            {
                key: row[key]
                for key in ("dataset", "subject_key", "record_key", "source_hashes", "start_s", "duration_s")
            },
        )
        row["ground_truth"].setdefault("source", row["source"])
        row.setdefault(
            "answer_fields", [key for key in row["expected_answer"] if key not in ("insufficient_evidence", "reason")]
        )
        if not row["answer_fields"]:
            row["answer_fields"] = row.get("control_answer_fields", ["sbp_mmhg"])
        mapping = ", ".join(f"column {i} = {name}" for i, name in enumerate(row["channels"]))
        row["question"] += (
            f" Times are seconds from the start of this {duration:g} s case. The array is samples × channels: {mapping}. When loading, assign these channel names in this order."
        )
        identity = "level2-v1:" + row["family"] + ":" + row["task_type"] + ":" + row["subject_key"]
        if row["family"] == "F5":
            identity += ":" + row["pair_role"]
        if row.get("retry_candidate"):
            identity += f":retry:{row['design_slot']}:{row['retry_rank']}"
        row["case_id"] = hashlib.sha256(identity.encode()).hexdigest()[:24]
        if row["task_type"] in ("damping_onset", "verified_onset", "manual_abp_artifact"):
            row["baseline_channel"] = "ABP"
        if row["task_type"] == "verified_rate":
            row["baseline_channel"] = "PPG"
        if row["family"] == "F5" and row["task_type"] == "heart_rate" and not row["abstention_correct"]:
            row["expected_solvability"] = "generic"
            row["specialist_capability"] = None
            row["solvability_justification"] = (
                "Existing ECG HR/RR summaries and arithmetic on question-stated windows support this estimand; numerical baseline is still rejected if it passes"
            )
        if row["family"] == "F1":
            regions = row["grounding_rubric"]["regions"]
            # These are the exact periods stated publicly in the question.
            for region in regions:
                a, b = region["window_s"]
                if f"{a:g}–{b:g} s" not in row["question"]:
                    raise ValueError("F1 shortcut window is not public in question")
            row["baseline_windows_s"] = [region["window_s"] for region in regions]
            row["baseline_channel"] = regions[0]["channels"][0]
    if sum(len(pair) for pair in pairs.values()) != sum(row["family"] == "F5" for row in candidates):
        raise ValueError("F5 candidate pairing is incomplete")
    return candidates, exclusions


def freeze(*, run_dry=True):
    candidates, exclusions = load_candidates()
    baseline_hash = sha(ROOT / "eval/level2_baselines.py")
    rows = run_baselines(candidates)
    eligible = []
    evaluated = []
    for row, baseline in zip(candidates, rows, strict=True):
        row["baselines"] = baseline
        support_eligible = True
        if row["family"] == "F6":
            row["misleading_output"] = baseline["detector_only"].get("first_output")
            initial = baseline["detector_only"]["answer"]
            field = "bpm" if row["task_type"] == "verified_rate" else "onset_s"
            value = initial.get(field) if isinstance(initial, dict) else None
            plausible = isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
            plausible = plausible and (30 <= value <= 220 if field == "bpm" else 0 <= value < row["duration_s"])
            if row["task_type"] == "verified_onset" and plausible:
                perturbation = row["perturbation"]
                left = perturbation["distractor_s"] - 5
                right = perturbation["distractor_s"] + perturbation["distractor_duration_s"] + 5
                plausible = left <= value <= right
            row["initial_output_gate"] = {
                "accepted": bool(plausible and not baseline["detector_only"]["grade"]["correct"]),
                "criterion": "finite cardiac rate30–220bpm or onset within registered transient±5s; outside answer tolerance",
                "prediction": initial,
            }
            if row["misleading_output"] is None or not row["initial_output_gate"]["accepted"]:
                exclusions.append(
                    {
                        "family": "F6",
                        "stage": "initial_output",
                        "reason": "No successful plausible but incorrect first generic answer",
                        "case_id": row["case_id"],
                    }
                )
                row["eligibility_exclusions"] = ["initial_output_gate"]
                support_eligible = False
        if baseline["retained"] and support_eligible and not row.get("design_superseded"):
            eligible.append(row)
        elif not baseline["retained"]:
            exclusions.append(
                {
                    "family": row["family"],
                    "stage": "shortcut_baseline",
                    "reason": "At least one preregistered shortcut passed",
                    "case_id": row["case_id"],
                    "rejected_by": baseline["rejected_by"],
                }
            )
        evaluated.append(row)
    # Report a fixed-point retained-pool median check as an additional gate.
    iterations = []
    retained = select_retry_slots(eligible)
    while retained:
        reject = retained_median_gate(retained)
        iterations.append({"pool_size": len(retained), "rejections": len(reject)})
        if not reject:
            break
        for row in retained:
            if row["case_id"] in reject:
                exclusions.append(
                    {
                        "family": row["family"],
                        "stage": "retained_pool_median",
                        "reason": "Retained-pool leave-one-out shortcut passed",
                        "case_id": row["case_id"],
                    }
                )
        eligible = [row for row in eligible if row["case_id"] not in reject]
        retained = select_retry_slots(eligible)
    if baseline_hash != sha(ROOT / "eval/level2_baselines.py"):
        raise ValueError("Baseline code changed during execution")
    retained_ids = {row["case_id"] for row in retained}
    for row in eligible:
        if row["case_id"] not in retained_ids:
            exclusions.append(
                {
                    "family": row["family"],
                    "stage": "retry_allocation",
                    "reason": "Earlier gate-passing retry fills slot or subject is already used in family",
                    "case_id": row["case_id"],
                }
            )
    for row in evaluated:
        row["decision"] = "retained" if row["case_id"] in retained_ids else "excluded"
    output = []
    answers = []
    for row in retained:
        directory = ROOT / "eval/cases/p9level2" / row["case_id"]
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "signal.npy", row["x"], allow_pickle=False)
        visible = {
            "file": "signal.npy",
            "dur_s": row["duration_s"],
            "channels": [
                {"name": n, "unit": u, "fs": row["fs"]} for n, u in zip(row["channels"], row["units"], strict=True)
            ],
        }
        dump(directory / "visible.json", visible)
        row["signal_sha256"] = sha(directory / "signal.npy")
        row["visible_sha256"] = sha(directory / "visible.json")
        row["question_sha256"] = hashlib.sha256(row["question"].encode()).hexdigest()
        output.append({k: v for k, v in row.items() if k != "x"})
        answers.append(
            {
                "case_id": row["case_id"],
                "qtype": "level2",
                "question": row["question"],
                "answer": row["expected_answer"],
                "tolerances": row["tolerances"],
            }
        )
    answer_path = ROOT / "eval/cases/p9level2.answers.jsonl"
    answer_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in answers))
    question_path = ROOT / "eval/cases/p9level2.questions.jsonl"
    question_path.write_text(
        "".join(
            json.dumps({k: row[k] for k in ("case_id", "qtype", "question")}, sort_keys=True) + "\n" for row in answers
        )
    )
    sources = [
        "eval/tasks/p9level2.py",
        "eval/tasks/level2_split.py",
        "eval/level2_data.py",
        "eval/level2_perturb.py",
        "eval/level2_alignment.py",
        "eval/level2_physiology.py",
        "eval/level2_manual.py",
        "eval/level2_baselines.py",
        "eval/level2_grade.py",
        "eval/level2_checks.py",
    ]
    primary_counts = Counter(
        row["family"] for row in candidates if not row.get("design_superseded") and not row.get("retry_candidate")
    )
    primary_counts.update(
        family
        for family, _ in {(row["family"], row["design_slot"]) for row in candidates if row.get("retry_candidate")}
    )
    pairs = abstention_pairs(output)
    superseded_path = RESULTS / "superseded-39.json"
    manifest = {
        "schema_version": 2,
        "amendment": "L2-A",
        "suite": "p9level2",
        "seed": 20261026,
        "model_calls": 0,
        "target_counts": TARGETS,
        "candidate_counts": dict(Counter(r["family"] for r in candidates)),
        "primary_candidate_counts": dict(primary_counts),
        "final_counts": dict(Counter(r["family"] for r in output)),
        "solvability_counts": dict(Counter(r["expected_solvability"] for r in output)),
        "signal_kind_counts": dict(Counter(r["signal_kind"] for r in output)),
        "ground_truth_counts": dict(Counter(r["ground_truth"]["category"] for r in output)),
        "abstention_pair_count": len(pairs),
        "abstention_pairs": [
            {
                "pair_id": pair_id,
                "members": {row["pair_role"]: row["case_id"] for row in pair},
                "oracle_grade": grade_level2_pair([row["expected_answer"] for row in pair], pair),
            }
            for pair_id, pair in sorted(pairs.items())
        ],
        "supersedes": json.loads(superseded_path.read_text()) if superseded_path.exists() else None,
        "split_sha256": sha(ROOT / "eval/tasks/level2_splits.json"),
        "baseline_registration": {
            "version": BASELINE_VERSION,
            "sha256": baseline_hash,
            "constants": FIELD_CONSTANTS,
            "constant_abstention": True,
            "gate": "F5: same baseline must pass both pair members; all other families: per-item",
            "median_reference": "leave-one-out initial candidate task_type; additional retained-pool fixed-point check",
        },
        "generator_hashes": {p: sha(ROOT / p) for p in sources if (ROOT / p).exists()},
        "generation_bindings": {
            part: json.loads((CACHE / part / "generation.json").read_text())
            for part in ("physiology", "perturb", "manual")
        },
        "questions": output,
        "candidate_evaluations": [{k: v for k, v in r.items() if k != "x"} for r in evaluated],
        "exclusions": exclusions,
        "retained_pool_median_iterations": iterations,
        "answers_sha256": sha(answer_path),
        "questions_sha256": sha(question_path),
        "evaluation_protocol": {
            "arms": ["C", "D"],
            "model": "claude-opus-5-5",
            "effort": "medium",
            "timeout_s": 900,
            "paid_run_authorized": False,
            "report_groups": ["generic", "specialist", "abstention"],
            "abstention_scoring": "per pair: correct_abstention, fabrication, unnecessary_abstention, pair_correct",
            "subject_unit": "one subject per family except the two counterfactual members of one F5 pair",
            "grounding": "audited trace records checked against frozen regions and computations; numeric correctness alone is ungrounded",
            "verification": "independent linked trace check with changed method/channel/window; F6 requires detected_misleading_output",
            "failure_taxonomy": "T0-T6 and Tg as Level1; manual earliest-cause audit",
        },
    }
    fresh = RESULTS / "sealed-hashes.json"
    manifest["separation"] = validate(manifest, fresh_sealed=fresh if fresh.exists() else None)
    if run_dry:
        manifest["dry_runs"] = dry_runs(output)
    dump(MANIFEST, manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "candidate_counts",
                    "final_counts",
                    "solvability_counts",
                    "signal_kind_counts",
                    "ground_truth_counts",
                )
            }
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dry-runs", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate()
    freeze(run_dry=not args.skip_dry_runs)
