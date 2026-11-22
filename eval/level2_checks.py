"""Free Level 2 integrity checks and exact existing C/D preparation, no model launch."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.level2_data import separation
from eval.level2_grade import grade_level2, grade_level2_pair, validate_level2_pair

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dry_runs(items):
    """Prepare both frozen arms while restoring the answer registry on all paths."""
    from eval import common

    absent = object()
    previous = common.ANSWER_FORMATS.get("level2", absent)
    try:
        return _dry_runs(items)
    finally:
        if previous is absent:
            common.ANSWER_FORMATS.pop("level2", None)
        else:
            common.ANSWER_FORMATS["level2"] = previous


def _dry_runs(items):
    """Use unmodified preparation; only non-model outer isolation probes execute."""
    from eval import common
    from eval.audit_checks import _probe_home
    from eval.run_claude_code import BASH_SANDBOX, CASE_TIMEOUT_S, prepare_case

    chosen = [items[0], next(item for item in items if len(item["channels"]) > 1)]
    # Prefer a different item when the first is already multichannel.
    if chosen[0] is chosen[1]:
        chosen[1] = next(item for item in items[1:] if len(item["channels"]) > 1)
    results = []
    for ordinal, item in enumerate(chosen, 1):
        case_dir = ROOT / "eval/cases/p9level2" / item["case_id"]
        visible = json.loads((case_dir / "visible.json").read_text())
        fields = item["answer_fields"]
        schema = "{" + ", ".join(json.dumps(key) + ": <value>" for key in fields) + "}"
        common.ANSWER_FORMATS["level2"] = schema + ' OR {"insufficient_evidence": true, "reason": "<reason>"}'
        case = {"dir": str(case_dir), "visible": visible, "question": item["question"], "qtype": "level2"}
        for arm in ("C", "D"):
            plan = prepare_case(
                case,
                arm,
                {"model": "claude-opus-5-5", "vision": True, "base_url": None},
                "medium",
                True,
                server_root=Path.home() / "tsmcp-server-R",
            )
            try:
                work = plan["cwd"]
                # The frozen harness supplies metadata in the prompt. Level 2's
                # public metadata file is additionally delivered equally to both
                # arms, without modifying either arm or its sandbox settings.
                shutil.copyfile(case_dir / "visible.json", work / "visible.json")
                checks = {
                    "same_signal_bytes": sha(work / "signal.npy") == item["signal_sha256"],
                    "same_visible_bytes": sha(work / "visible.json") == item["visible_sha256"],
                    "same_visible_metadata_in_prompt": all(
                        f"{c['name']} ({c['unit']}, {c['fs']} Hz)" in plan["stdin"] for c in visible["channels"]
                    ),
                    "public_work_files_only": sorted(p.name for p in work.iterdir()) == ["signal.npy", "visible.json"],
                    "question_preserved": item["question"] in plan["stdin"],
                    "timeout_s": CASE_TIMEOUT_S,
                    "medium_effort": "--effort" in plan["command"]
                    and plan["command"][plan["command"].index("--effort") + 1] == "medium",
                }
                row = {"item_ordinal": ordinal, "arm": arm, "model_invoked": False, "checks": checks}
                if arm == "C":
                    row["mcp_configured"] = "tsmcp" in plan["mcp_config"]["mcpServers"]
                else:
                    row["network_configuration"] = BASH_SANDBOX["sandbox"]
                    row["network_runtime"] = (
                        "Not exercised: existing Claude Bash network sandbox needs a model session; no model called"
                    )
                    home = _probe_home(plan["command"], {}, work, ROOT)
                    row["home_runtime"] = {k: v for k, v in home.items() if k != "home"}
                row["status"] = (
                    "pass"
                    if all(
                        checks[k]
                        for k in (
                            "same_signal_bytes",
                            "same_visible_bytes",
                            "same_visible_metadata_in_prompt",
                            "public_work_files_only",
                            "question_preserved",
                            "medium_effort",
                        )
                    )
                    and checks["timeout_s"] == 900
                    else "fail"
                )
                if arm == "C" and not row["mcp_configured"]:
                    row["status"] = "fail"
                if arm == "D" and row["home_runtime"]["status"] != "pass" and row["status"] == "pass":
                    row["status"] = "configuration_only"
                results.append(row)
            finally:
                shutil.rmtree(plan["base"])
    return {
        "model_calls": 0,
        "runs": results,
        "configuration_pass": all(r["status"] in ("pass", "configuration_only") for r in results),
        "runtime_network_verified": False,
    }


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_reference_agreement(item):
    truth = item["ground_truth"]
    agreement = truth["agreement"]
    if agreement.get("accepted") is not True:
        raise ValueError("Retained reference lacks agreement")
    first, second = truth.get("reference1"), truth.get("reference2")
    if "G4" in truth["category"] and not (isinstance(first, dict) and isinstance(second, dict)):
        raise ValueError("G4 reference values are missing")
    if isinstance(first, dict) and isinstance(second, dict):
        tolerances = item["tolerances"]
        numeric_fields = [key for key, value in item["expected_answer"].items() if _finite(value)]
        if any(key not in first or key not in second for key in numeric_fields):
            raise ValueError("Independent numerical reference coverage is incomplete")
        half = {}
        for key in numeric_fields:
            value = tolerances[key]
            half[key] = (
                {name: bound / 2 for name, bound in value.items() if name in {"absolute", "relative"}}
                if isinstance(value, dict)
                else value / 2
            )
        if numeric_fields:
            reference_item = {"expected_answer": {key: first[key] for key in numeric_fields}, "tolerances": half}
            if not grade_level2({key: second[key] for key in numeric_fields}, reference_item)["correct"]:
                raise ValueError("Independent numerical references exceed half tolerance")
    records = [agreement, *agreement.get("fields", {}).values()]
    for record in records:
        difference = record.get("difference", record.get("difference_bpm"))
        bound = record.get("half_tolerance")
        if difference is not None and bound is not None:
            if not _finite(difference) or not _finite(bound) or difference < 0 or bound < 0:
                raise ValueError("Reference agreement contains invalid numerical bounds")
            if difference > bound and not math.isclose(difference, bound, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Recorded reference disagreement exceeds half tolerance")


def _initial_prediction(item):
    first = item.get("misleading_output")
    if not isinstance(first, dict) or not isinstance(first.get("output"), dict):
        raise ValueError("F6 lacks an observed numerical initial output")
    operation, output = first.get("operation"), first["output"]
    fields = set(item["answer_fields"])
    if operation == "dominant_freq" and fields in ({"bpm"}, {"hr_bpm"}, {"heart_rate_bpm"}):
        if not _finite(output.get("value")):
            raise ValueError("F6 initial frequency is not a usable number")
        rate = 60 * output["value"]
        if not 30 <= rate <= 220:
            raise ValueError("F6 initial cardiac rate is outside the preregistered plausible range")
        return {next(iter(fields)): rate}
    if operation in {"change", "flat"} and ("onset_s" in fields or fields == {"t"}):
        if not _finite(output.get("t")):
            raise ValueError("F6 initial onset is not a usable number")
        if not 0 <= output["t"] < item["duration_s"]:
            raise ValueError("F6 initial onset is outside the supplied recording")
        if item.get("task_type") == "verified_onset":
            perturbation = item.get("perturbation", {})
            distractor = perturbation.get("distractor_s")
            duration = perturbation.get("distractor_duration_s")
            if not _finite(distractor) or not _finite(duration) or duration <= 0:
                raise ValueError("F6 misleading transient has no valid recorded support")
            if not distractor - 5 <= output["t"] <= distractor + duration + 5:
                raise ValueError("F6 initial onset does not identify the recorded misleading transient")
        prediction = {"onset_s" if "onset_s" in fields else "t": output["t"]}
        if "channel" in fields:
            prediction["channel"] = first.get("channel")
        if "duration_s" in fields:
            prediction["duration_s"] = output.get("dur")
        return prediction
    raise ValueError("F6 initial output has no registered numerical interpretation")


def _validate_baselines(item, pair=None):
    baselines = item["baselines"]
    expected_names = {"constant", "development_median", "detector_only", "constant_abstention"}
    if not expected_names <= baselines.keys():
        raise ValueError("Retained item is missing a preregistered baseline")
    passing = []
    for name in sorted(expected_names):
        record = baselines[name]
        current = grade_level2(record.get("answer"), item)
        if current["correct"] != record["grade"]["correct"] or current["outcome"] != record["grade"]["outcome"]:
            raise ValueError("Stored baseline grade disagrees with recomputed grade")
        if current["correct"]:
            passing.append(name)
    if item["family"] == "F5":
        if pair is None or not baselines.get("pair_gate", {}).get("valid"):
            raise ValueError("Retained F5 item lacks a valid pair gate")
        if passing != sorted(baselines["successful_baselines"]):
            raise ValueError("F5 per-item baseline successes differ from stored record")
        for name in sorted(expected_names):
            current = grade_level2_pair([row["baselines"][name]["answer"] for row in pair], pair)
            if current != baselines["pair_gate"]["baseline_scores"][name] or current["pair_correct"]:
                raise ValueError("Retained pair passes a shortcut or its stored pair grade differs")
        if baselines["rejected_by"] or baselines["pair_gate"]["rejected_by"]:
            raise ValueError("Retained F5 pair has a recorded shortcut rejection")
    elif passing or baselines["rejected_by"]:
        raise ValueError("Retained item passes a preregistered baseline")
    if "retained_pool_median" in item:
        median = item["retained_pool_median"]
        current = grade_level2(median.get("answer"), item)
        if current != median["grade"]:
            raise ValueError("Retained-pool median shortcut does not satisfy the rejection gate")
        if item["family"] == "F5":
            pair_grade = grade_level2_pair([row["retained_pool_median"]["answer"] for row in pair], pair)
            if pair_grade != median.get("pair_grade") or pair_grade["pair_correct"]:
                raise ValueError("Retained-pool pair median shortcut does not satisfy the rejection gate")
        elif current["correct"]:
            raise ValueError("Retained-pool median shortcut does not satisfy the rejection gate")
    if item["family"] == "F6":
        initial = _initial_prediction(item)
        detector = baselines["detector_only"]
        if detector.get("first_output") != item["misleading_output"] or detector.get("answer") != initial:
            raise ValueError("F6 initial output does not match the actual shortcut prediction")
        if grade_level2(initial, item)["correct"]:
            raise ValueError("F6 initial output is correct rather than misleading")


def _validate_counterfactual(pair):
    roles = validate_level2_pair(pair)
    control, abstention = roles["control"], roles["abstention"]
    for field in (
        "subject_key",
        "record_key",
        "source_hashes",
        "start_s",
        "duration_s",
        "question",
        "channels",
        "units",
        "fs",
        "answer_fields",
    ):
        if control[field] != abstention[field]:
            raise ValueError("F5 pair does not share its source and public task")
    one = np.load(ROOT / "eval/cases/p9level2" / control["case_id"] / "signal.npy", allow_pickle=False)
    two = np.load(ROOT / "eval/cases/p9level2" / abstention["case_id"] / "signal.npy", allow_pickle=False)
    perturbation = abstention["ground_truth"]["perturbation"]
    column = control["channels"].index(perturbation["channel"])
    if one.shape != two.shape or perturbation["kind"] != "entire_channel_hold":
        raise ValueError("F5 counterfactual shape or intervention differs")
    other = [index for index in range(one.shape[1]) if index != column]
    if not np.array_equal(one[:, other], two[:, other], equal_nan=True):
        raise ValueError("F5 intervention changes an unrelated channel")
    if not np.all(two[:, column] == perturbation["value"]) or np.array_equal(one[:, column], two[:, column]):
        raise ValueError("F5 held channel does not implement the frozen insufficiency")
    if not grade_level2_pair([row["expected_answer"] for row in pair], pair)["pair_correct"]:
        raise ValueError("F5 oracle pair fails grading")


def validate(manifest, *, fresh_sealed=None):
    items = manifest["questions"]
    pairs = defaultdict(list)
    for item in items:
        if item["family"] == "F5":
            pairs[item.get("pair_id")].append(item)
    for pair in pairs.values():
        _validate_counterfactual(pair)
    seen = {}
    paths = []
    for item in items:
        marker = (item["family"], item["subject_key"])
        pair_id = item.get("pair_id") if item["family"] == "F5" else None
        if marker in seen and (pair_id is None or seen[marker] != pair_id):
            raise ValueError("More than one subject question in a family")
        seen[marker] = pair_id
        directory = ROOT / "eval/cases/p9level2" / item["case_id"]
        if sorted(p.name for p in directory.iterdir()) != ["signal.npy", "visible.json"]:
            raise ValueError("Case directory includes nonpublic files")
        for file, key in [("signal.npy", "signal_sha256"), ("visible.json", "visible_sha256")]:
            if sha(directory / file) != item[key]:
                raise ValueError("Frozen case bytes changed")
        data = np.load(directory / "signal.npy", allow_pickle=False, mmap_mode="r")
        if data.shape[1] != len(item["channels"]):
            raise ValueError("Case channel mapping disagrees")
        if not grade_level2(item["expected_answer"], item)["correct"]:
            raise ValueError("Oracle does not pass grader")
        _validate_baselines(item, pairs.get(pair_id))
        _validate_reference_agreement(item)
        paths.append(directory / "signal.npy")
    kwargs = {"sealed_path": fresh_sealed} if fresh_sealed else {}
    result = separation(items, paths, **kwargs)
    result.update(
        oracle_passes=len(items),
        one_subject_per_family=not bool(pairs),
        one_subject_per_family_or_counterfactual_pair=True,
        counterfactual_pairs_checked=len(pairs),
        answers_outside_case_dirs=True,
    )
    return result
