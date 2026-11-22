"""Offline analysis of the fixed 30-question development capability audit.

No collection/model path, source-dataset discovery, or sealed-partition access.
Q01..Q30 are positional aliases in the frozen manifest, not participant IDs.
Reviewed judgments are explicit inputs; geometry alone cannot establish grounding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from eval.grade import AUDIT_FIELDS, grade
from eval.part2_dev_audit import parse_transcript as parse_raw_transcript

TOKEN_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
RUNS = {arm: f"p8audit-{arm}-claude-opus-5-5-medium-audit" for arm in ("C", "D")}
ARGUMENT_KEYS = {"channels", "window", "what", "view", "series", "detail", "detector", "top_k", "snippets"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("blank JSONL line would invalidate physical line citations")
    return [json.loads(line) for line in lines]


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row["case_id"]
        if key in result:
            raise ValueError("duplicate case key")
        result[key] = row
    return result


def parse_transcript(events: list[dict]) -> dict:
    """Join by tool IDs privately; export only ordinal/line-addressed metadata.

    Uses the earlier audit's tested SDK-envelope/stream deduplication. Never
    exports thinking, waveform arrays, images, paths, handles or raw commands.
    Python-bearing Bash calls differ from Python invocations (reviewed separately).
    """
    try:
        parsed = parse_raw_transcript(events)
    except ValueError as error:
        raise ValueError(str(error).split(":")[0]) from None
    calls = []
    for ordinal, call in enumerate(parsed["calls"], 1):
        if "result_line" not in call:
            raise ValueError("tool call missing result")
        args = call["arguments"]
        delivered_channels = None
        delivered_window = None
        single_channel_warning = None
        for payload in call["evidence"]["payloads"]:
            if not isinstance(payload, dict):
                continue
            if isinstance(payload.get("window"), list):
                delivered_window = payload["window"]
            channels = payload.get("channels")
            if channels and all(isinstance(c, str) for c in channels):
                delivered_channels = channels
            warning = re.search(r"uses one channel: ([A-Za-z0-9_ -]+)", payload.get("warn", ""))
            if warning:
                single_channel_warning = warning.group(1).strip()
                delivered_channels = [single_channel_warning]
        calls.append(
            {
                "ordinal": ordinal,
                "tool": call["tool"],
                "call_line": call["call_line"],
                "result_line": call["result_line"],
                "arguments": {k: v for k, v in args.items() if k in ARGUMENT_KEYS},
                "usable": call["evidence_usable"],
                "is_error": call["is_error"],
                "presentation_error": call["presentation_error"],
                "text_chars": call["evidence"]["text_chars"],
                "images": call["evidence"]["images"],
                "delivered_channels": delivered_channels,
                "delivered_window": delivered_window,
                "single_channel_warning": single_channel_warning,
                "python_bash_call": call["tool"] == "Bash"
                and bool(re.search(r"\bpython(?:3(?:\.\d+)?)?\b", args.get("command", ""))),
            }
        )
    finals = [e for e in events if e.get("type") == "result"]
    if len(finals) != 1:
        raise ValueError("expected one terminal result")
    final = finals[0]
    return {
        "calls": calls,
        "tool_counts": dict(sorted(Counter(c["tool"] for c in calls).items())),
        "python_bash_calls": sum(c["python_bash_call"] for c in calls),
        "unusable_tool_results": sum(not c["usable"] for c in calls),
        "api_retry_events": sum(e.get("subtype") == "api_retry" for e in events),
        "terminal": final,  # private reconciliation only; never serialized
    }


def numeric_errors(answer: dict, pred: dict | None) -> dict:
    errors = {}
    for key, truth in answer.items():
        value = (pred or {}).get(key)
        if isinstance(truth, (int, float)) and not isinstance(truth, bool):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            errors[key] = {
                "signed": value - truth if valid else None,
                "absolute": abs(value - truth) if valid else None,
                "relative": abs(value - truth) / abs(truth) if valid and truth != 0 else None,
            }
    return errors


def c_geometry(calls: list[dict], visible: dict, truth: dict, task: str) -> dict:
    """Conservative successful-call geometry; broad delivery is not attention.

    Full-record load/describe summaries do not establish waveform inspection.
    Locate feature overlap only for event perturbations and manual onset tasks.
    Aggregate, pairing and D-code coverage need the reviewed judgment input.
    """
    duration = visible["dur_s"]
    names = [c["name"] for c in visible["channels"]]
    windows = []
    for call in calls:
        if not call["usable"] or call["tool"] not in ("ts_zoom", "ts_measure", "ts_find_events"):
            continue
        args = call["arguments"]
        window = call.get("delivered_window") or args.get("window") or [0, duration]
        window = [max(0, window[0]), min(duration, window[1])]
        channels = call.get("delivered_channels") or args.get("channels") or names
        windows.append({"call_line": call["call_line"], "window": window, "channels": channels, "tool": call["tool"]})
    perturbation = truth.get("perturbation")
    target = perturbation.get("start_s") if perturbation else None
    channel = perturbation.get("channel") if perturbation else None
    if task in ("audit_valsalva", "audit_tilt", "audit_cbfv_tilt"):
        target = truth["answer"]["t"]
        channel = "ABP"
    hits = []
    if target is not None:
        hits = [
            w["call_line"] for w in windows if w["window"][0] <= target < w["window"][1] and channel in w["channels"]
        ]
    return {
        "successful_windows": windows,
        "target_onset_s": target,
        "target_channel": channel,
        "onset_covered_by_lines": hits,
        "scope": "delivery geometry only; consult reviewed judgment for feature visibility, baseline and aggregate support",
    }


def reconcile(row: dict, trace: dict) -> None:
    final = trace["terminal"]
    if row["status"] != "ok" or final.get("is_error") or final.get("subtype") != "success":
        raise ValueError("unexpected unsuccessful frozen session")
    if any(row["usage"].get(k, 0) != final["usage"].get(k, 0) for k in TOKEN_KEYS):
        raise ValueError("terminal usage mismatch")
    if row["turns"] != final["num_turns"] or row["text"] != final["result"]:
        raise ValueError("terminal answer/turn mismatch")
    if not math.isclose(row["cc_cost_usd"], final["total_cost_usd"], abs_tol=1e-9):
        raise ValueError("terminal cost mismatch")
    expected = Counter(t.removeprefix("mcp__tsmcp__") for t in row["tool_calls"])
    if expected != Counter(trace["tool_counts"]):
        raise ValueError("tool-count mismatch")


def summarize(rows: list[dict]) -> dict:
    result = {}
    for group in ("core", "gap-probe"):
        subset = [r for r in rows if r["score_group"] == group]
        result[group] = {}
        for arm in RUNS:
            arms = [r["arms"][arm] for r in subset]
            metrics = {
                "n": len(subset),
                "correct": sum(a["correct"] for a in arms),
                "correct_but_weakly_grounded": [
                    r["question"] for r in subset if r["arms"][arm]["judgment"]["correct_but_weakly_grounded"]
                ],
                "failure_attribution": dict(
                    Counter(a["judgment"]["failure_attribution"] for a in arms if not a["correct"])
                ),
                "tokens": {k: sum(a["tokens"][k] for a in arms) for k in TOKEN_KEYS},
                "cost_usd": sum(a["cost_usd"] for a in arms),
                "cc_cost_usd": sum(a["cc_cost_usd"] for a in arms),
                "latency_s_median": statistics.median(a["latency_s"] for a in arms),
                "latency_s_total": sum(a["latency_s"] for a in arms),
                "turns_total": sum(a["turns"] for a in arms),
                "tool_counts": dict(sum((Counter(a["tool_counts"]) for a in arms), Counter())),
                "python_steps": sum(a["judgment"].get("python_steps", 0) for a in arms) if arm == "D" else None,
                "unusable_tool_results": sum(a["unusable_tool_results"] for a in arms),
                "api_retry_events": sum(a["api_retry_events"] for a in arms),
            }
            for field in ("family", "gt_category", "confidence", "expected_solvability"):
                metrics[f"by_{field}"] = {
                    value: {
                        "n": sum(r[field] == value for r in subset),
                        "correct": sum(r[field] == value and r["arms"][arm]["correct"] for r in subset),
                    }
                    for value in sorted({r[field] for r in subset})
                }
            result[group][arm] = metrics
    return result


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "# Frozen per-question results",
        "",
        "Q aliases follow manifest order. JSON has full-precision references/errors, reviewed evidence and physical trace lines. Tokens are uncached/cache-write/cache-read/output; cost is saved evaluator estimate. `Y*` is correct but weakly grounded under the documented review rule. BRS/TFA grade error is relative; other errors use answer units. See the report for accounting and denominator caveats.",
        "",
        "| Q | Family / task | GT / confidence | Expected / group | Reference | C answer / correct / grade error | D answer / correct / grade error |",
        "|---|---|---|---|---|---|---|",
    ]

    def short(value: object) -> str:
        if isinstance(value, dict):
            return ", ".join(f"{k}={short(v)}" for k, v in value.items())
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    for row in rows:
        scores = []
        for arm in RUNS:
            a = row["arms"][arm]
            correct = ("Y*" if a["judgment"]["correct_but_weakly_grounded"] else "Y") if a["correct"] else "N"
            scores.append(f"{short(a['answer'])} / {correct} / {short(a['grade_error'])}")
        lines.append(
            f"| {row['question']} | {row['family']} / {row['task']} | {row['gt_category']} / {row['confidence']} | {row['expected_solvability']} / {row['score_group']} | {short(row['truth']['answer'])} | {' | '.join(scores)} |"
        )
    lines.extend(
        [
            "",
            "| Q | C tokens U/W/R/O | C cost / latency s / turns | C calls by tool | D tokens U/W/R/O | D cost / latency s / turns | D Python steps / Bash calls |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        c, d = row["arms"]["C"], row["arms"]["D"]
        tok = lambda a: "/".join(str(a["tokens"][k]) for k in TOKEN_KEYS)  # noqa: E731
        resource = lambda a: f"${a['cost_usd']:.5f} / {a['latency_s']:.2f} / {a['turns']}"  # noqa: E731
        lines.append(
            f"| {row['question']} | {tok(c)} | {resource(c)} | {short(c['tool_counts'])} | {tok(d)} | {resource(d)} | {d['judgment']['python_steps']} / {d['tool_counts'].get('Bash', 0)} |"
        )
    return "\n".join(lines) + "\n"


def analyze(root: Path, judgments: Path, out: Path, bind_review: bool = False) -> None:
    manifest_path = root / "eval/tasks/p8audit_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    answers_path = root / "eval/cases/p8audit.answers.jsonl"
    questions_path = root / "eval/cases/p8audit.questions.jsonl"
    if digest(answers_path) != manifest["answers_sha256"] or digest(questions_path) != manifest["questions_sha256"]:
        raise ValueError("frozen questions/answers hash mismatch")
    answers = keyed(read_jsonl(answers_path))
    questions = keyed(read_jsonl(questions_path))
    expected = {q["case_id"] for q in manifest["questions"]}
    if (
        len(manifest["questions"]) != 30
        or len(expected) != 30
        or set(answers) != expected
        or set(questions) != expected
    ):
        raise ValueError("frozen question set mismatch")
    reviewed = {arm: json.loads((judgments / f"{arm.lower()}-judgments.json").read_text()) for arm in RUNS}
    difficulty = {r["question"]: r for r in json.loads((judgments / "difficulty.json").read_text())["questions"]}
    aliases = {f"Q{i:02d}" for i in range(1, 31)}
    if any(set(j) != aliases for j in reviewed.values()):
        raise ValueError("reviewed judgments must cover Q01..Q30 exactly")
    if set(difficulty) != aliases:
        raise ValueError("difficulty review must cover Q01..Q30 exactly")
    sources = {"manifest": digest(manifest_path), "answers": digest(answers_path), "questions": digest(questions_path)}
    results = {}
    for arm, run in RUNS.items():
        path = root / "eval/runs" / run / "results.jsonl"
        results[arm] = keyed(read_jsonl(path))
        if set(results[arm]) != expected:
            raise ValueError("run question set mismatch")
        sources[f"results_{arm}"] = digest(path)
    rows = []
    for index, question in enumerate(manifest["questions"], 1):
        alias = f"Q{index:02d}"
        key = question["case_id"]
        task = question["task_type"]
        answer = {k: answers[key]["answer"][k] for k in AUDIT_FIELDS[task]}
        visible_path = root / "eval/cases/p8audit" / key / "visible.json"
        visible = json.loads(visible_path.read_text())
        signal_path = visible_path.with_name("signal.npy")
        signal_hash = digest(signal_path)
        if signal_hash != question["signal_sha256"]:
            raise ValueError(f"{alias} signal hash mismatch")
        sources[alias] = {"visible": digest(visible_path), "signal": signal_hash}
        truth = {"answer": answer, "perturbation": question["provenance"].get("perturbation")}
        if task == "audit_bp":
            signal = np.load(signal_path, allow_pickle=False)
            channel = [c["name"] for c in visible["channels"]].index("ABP")
            time_mean = float(np.mean(signal[:, channel], dtype=np.float64))
            reviewed_mean = difficulty[alias]["observed_facts"]["sample_weighted_abp_mean"]
            if not math.isclose(time_mean, reviewed_mean, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"{alias} pressure-mean difficulty check changed")
            truth["sample_time_mean_map_check"] = {
                "mean": time_mean,
                "difference_from_equal_beat_map": time_mean - answer["map"],
                "purpose": "post-hoc estimator-difficulty check, not a replacement reference",
            }
        row = {
            "question": alias,
            "manifest_position": index,
            **{
                k: question[k]
                for k in (
                    "row",
                    "family",
                    "gt_category",
                    "confidence",
                    "expected_solvability",
                    "score_group",
                    "tolerance",
                )
            },
            "task": task,
            "duration_s": visible["dur_s"],
            "channels": visible["channels"],
            "truth": truth,
            "difficulty": difficulty.get(alias),
            "arms": {},
        }
        for arm, run in RUNS.items():
            saved = results[arm][key]
            if saved["qtype"] != task or answers[key]["qtype"] != task or questions[key]["qtype"] != task:
                raise ValueError(f"{alias} task mismatch")
            trace_path = root / "eval/runs" / run / "transcripts" / f"{key}.jsonl"
            sources[alias][arm] = digest(trace_path)
            trace = parse_transcript(read_jsonl(trace_path))
            reconcile(saved, trace)
            scored = grade(task, answer, saved["pred"])
            judgment = reviewed[arm][alias]
            failure = judgment["failure_attribution"]
            if (scored["correct"] and failure is not None) or (
                not scored["correct"] and failure not in {f"T{i}" for i in range(1, 7)}
            ):
                raise ValueError(f"{alias} reviewed attribution conflicts with score")
            if judgment["correct_but_weakly_grounded"] and not scored["correct"]:
                raise ValueError(f"{alias} weak-correct flag on wrong answer")
            row["arms"][arm] = {
                "answer": {k: saved["pred"][k] for k in AUDIT_FIELDS[task] if k in saved["pred"]}
                if isinstance(saved["pred"], dict)
                else None,
                "correct": scored["correct"],
                "grade_error": scored["err"],
                "numerical_errors": numeric_errors(answer, saved["pred"]),
                "tokens": {k: saved["usage"].get(k, 0) for k in TOKEN_KEYS},
                "sdk_model_usage_tokens": {
                    k: sum(v.get(k, 0) for v in trace["terminal"].get("modelUsage", {}).values())
                    for k in ("inputTokens", "cacheCreationInputTokens", "cacheReadInputTokens", "outputTokens")
                },
                **{k: saved[k] for k in ("cost_usd", "cc_cost_usd", "latency_s", "turns")},
                **{
                    k: trace[k]
                    for k in ("calls", "tool_counts", "python_bash_calls", "unusable_tool_results", "api_retry_events")
                },
                "geometry": c_geometry(trace["calls"], visible, truth, task) if arm == "C" else None,
                "judgment": judgment,
            }
        rows.append(row)
    if Counter(r["score_group"] for r in rows) != {"core": 26, "gap-probe": 4}:
        raise ValueError("frozen score-group mismatch")
    sources["judgments"] = {
        p.name: digest(p)
        for p in [judgments / "c-judgments.json", judgments / "d-judgments.json", judgments / "difficulty.json"]
    }
    sources["analysis_code"] = digest(Path(__file__))
    sources["grader"] = digest(Path(__file__).with_name("grade.py"))
    sources["formats"] = digest(Path(__file__).with_name("audit_formats.py"))
    sources["parser_dependency"] = digest(Path(__file__).with_name("part2_dev_audit.py"))
    if sources["grader"] != digest(root / "eval/grade.py") or sources["formats"] != digest(
        root / "eval/audit_formats.py"
    ):
        raise ValueError("grader or formats changed from frozen input root")
    binding = {k: v for k, v in sources.items() if k != "analysis_code"}
    binding_path = judgments / "review-binding.json"
    if bind_review:
        if binding_path.exists():
            raise ValueError("review binding already exists; do not silently replace it")
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    elif not binding_path.exists() or json.loads(binding_path.read_text()) != binding:
        raise ValueError("inputs or judgments changed; explicit fresh review binding required")
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("per-question.json", rows),
        ("summary.json", summarize(rows)),
        ("input-manifest.json", sources),
    ):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "per-question.md").write_text(markdown_table(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/home/yanghaoxiang24/signal_agent"))
    parser.add_argument("--judgments", type=Path, default=Path("eval/results/p8audit"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/p8audit"))
    parser.add_argument(
        "--bind-review", action="store_true", help="bind the completed review once; refuses replacement"
    )
    args = parser.parse_args()
    analyze(args.input_root, args.judgments, args.out, args.bind_review)
    print("Analyzed 30 development questions: core 26, gap probes 4. No model calls.")


if __name__ == "__main__":
    main()
