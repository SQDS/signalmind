"""Offline p7dev audit of retained attempts, evidence, and reviewed judgments.

This script reads only the named development artifacts. It never loads a model,
starts a tool server, recomputes a detector, or collects a proposed B revision.
Reference annotations are used only for retrospective geometry, never selection.
Run with the existing project's .venv/bin/python; see the audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

if __package__:
    from .grade import grade
else:
    from grade import grade

HERE = Path(__file__).resolve().parent
RUNS = {
    "B": "p7dev-B-claude-opus-5-5-medium-v0",
    "C": "p7dev-C-claude-opus-5-5-medium-dev",
}
ENDPOINTS = ("longest_af", "af_burden")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row["case_id"]
        if key in result:
            raise ValueError(f"duplicate case_id: {key}")
        result[key] = row
    return result


def overlap(a: list | tuple, b: list | tuple) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def union(intervals: list, gap: float = 0) -> list[list[float]]:
    """Union intervals; positive gap merges only strictly smaller interruptions."""
    out = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("interval must have positive duration")
        if out and (start <= out[-1][1] or start - out[-1][1] < gap):
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([float(start), float(end)])
    return out


def interval_seconds(intervals: list) -> float:
    return sum(end - start for start, end in union(intervals))


def annotation_intervals(samples: list, notes: list, fs: float, duration: float) -> list:
    """Rhythm labels persist until the next rhythm label, not the next beat."""
    if len(samples) != len(notes):
        raise ValueError("sample/note lengths differ")
    marks = []
    for sample, note in zip(samples, notes, strict=True):
        label = note.strip("\x00").strip()
        if label.startswith("("):
            marks.append((float(sample) / fs, label))
    if marks != sorted(marks, key=lambda x: x[0]):
        raise ValueError("rhythm changes must be chronological")
    return [
        [start, marks[i + 1][0] if i + 1 < len(marks) else duration]
        for i, (start, label) in enumerate(marks)
        if label in ("(AFIB", "(AFL") and start < (marks[i + 1][0] if i + 1 < len(marks) else duration)
    ]


def project_payload(value: object) -> object:
    """Keep all episode ranks/key values but summarize long support arrays."""
    if isinstance(value, list):
        return [project_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key == "intervals_s":
            out["support_interval_count"] = len(item)
            out["support_intervals_sha256"] = hashlib.sha256(compact(item).encode()).hexdigest()
        else:
            out[key] = project_payload(item)
    return out


def content_evidence(content: object) -> dict:
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("tool content is neither text nor blocks")
    text, payloads, errors = [], [], []
    images = 0
    for block in content:
        if block.get("type") == "image":
            images += 1
        elif block.get("type") == "text":
            raw = block["text"]
            text.append(raw)
            try:
                payloads.append(project_payload(json.loads(raw)))
            except ValueError:
                errors.append(raw)
    return {
        "text_chars": sum(map(len, text)),
        "images": images,
        "text_sha256": hashlib.sha256(compact(text).encode()).hexdigest(),
        "payloads": payloads,
        "non_json_text": errors,
    }


def parse_transcript(events: list[dict]) -> dict:
    """Join streamed tool uses/results by ID, preserving physical JSONL lines.

    Ignore hidden thinking and duplicate SDK envelopes. Deduplicate repeated
    streamed blocks, but reject inconsistent duplicates and orphan results.
    A result's presence is not evidence that a failed payload was usable.
    """
    calls, statements, seen_statements = {}, [], set()
    final = None
    for line, event in enumerate(events, 1):
        message = event.get("message") or {}
        blocks = message.get("content", [])
        if not isinstance(blocks, list):
            continue
        if event.get("type") == "assistant":
            for block in blocks:
                if block.get("type") == "text":
                    identity = (message.get("id"), block["text"])
                    if identity not in seen_statements:
                        statements.append({"line": line, "text": block["text"]})
                        seen_statements.add(identity)
                if block.get("type") != "tool_use":
                    continue
                key = block["id"]
                tool = block["name"].removeprefix("mcp__tsmcp__")
                if key in calls:
                    if (calls[key]["tool"], calls[key]["arguments"]) != (tool, block["input"]):
                        raise ValueError(f"inconsistent tool-use ID: {key}")
                    continue
                calls[key] = {
                    "tool_use_id": key,
                    "call_line": line,
                    "tool": tool,
                    "arguments": block["input"],
                }
        elif event.get("type") == "user":
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                key = block["tool_use_id"]
                if key not in calls:
                    raise ValueError(f"orphan tool result: {key}")
                evidence = content_evidence(block.get("content", []))
                is_error = bool(block.get("is_error"))
                # Claude's client sometimes delivers its size-limit notice as an
                # ordinary text tool_result without setting is_error.
                presentation_error = any("exceeds maximum allowed tokens" in text for text in evidence["non_json_text"])
                text_error = any(
                    text.startswith(("Error:", "Error executing tool")) for text in evidence["non_json_text"]
                )
                if "result_line" in calls[key]:
                    if calls[key]["evidence"] != evidence or calls[key]["is_error"] != is_error:
                        raise ValueError(f"inconsistent repeated tool result: {key}")
                    continue
                calls[key].update(
                    result_line=line,
                    is_error=is_error,
                    presentation_error=presentation_error,
                    evidence_usable=not (is_error or text_error or presentation_error),
                    evidence=evidence,
                )
        if event.get("type") == "result":
            final = {
                "line": line,
                **{k: event.get(k) for k in ("subtype", "num_turns", "is_error", "result", "total_cost_usd")},
            }
    return {"calls": list(calls.values()), "public_statements": statements, "session_result": final}


def score(answer: dict, row: dict | None, error: dict | None = None) -> dict:
    """Execution errors stay wrong even if the rejected transcript has an answer."""
    if row is None or row.get("status", "ok") != "ok":
        category = (error or {}).get("class")
        if category == "B_single_turn_contract":
            failure = "execution_contract"
        elif category == "api_error":
            failure = "infrastructure"
        else:
            raise ValueError(f"unclassified missing/failed case: {answer['case_id']}")
        return {"pred": None, "correct": False, "abs_error": None, "failure_class": failure, "error": error}
    pred = row.get("pred")
    graded = grade(answer["qtype"], answer["answer"], pred)
    key = "t" if answer["qtype"] == "longest_af" else "percent"
    value = (pred or {}).get(key)
    truth = answer["answer"][key]
    err = abs(float(value) - truth) if isinstance(value, (int, float)) and math.isfinite(value) else None
    return {
        "pred": pred,
        "correct": bool(graded["correct"]),
        "abs_error": err,
        "failure_class": "none" if graded["correct"] else "model_workflow",
    }


def episode_geometry(episodes: list[dict], reference: list, longest: list) -> dict:
    """Rank/coverage facts are geometry, not proof of rhythm or recoverability."""
    ranked = sorted(episodes, key=lambda ep: (-ep.get("span_s", ep["dur"]), ep["start"]))
    selected = [
        i
        for i, ep in enumerate(ranked)
        if i < 5 or ep.get("span_s", ep["dur"]) >= 0.5 * ranked[0].get("span_s", ranked[0]["dur"])
    ]
    rows = []
    for i, ep in enumerate(ranked):
        span = [ep["start"], ep["start"] + ep.get("span_s", ep["dur"])]
        touched = [j + 1 for j, ref in enumerate(reference) if overlap(span, ref) > 0]
        rows.append(
            {
                "rank": i + 1,
                "span": span,
                "span_s": span[1] - span[0],
                "af_s": ep["af_s"],
                "support_interval_count": len(ep["intervals_s"]),
                "b_candidate": i in selected,
                "overlap_longest_s": overlap(span, longest),
                "reference_episodes_overlapped": touched,
                "start_error_s": abs(span[0] - longest[0]),
            }
        )
    matching = [row for row in rows if row["overlap_longest_s"] > 0]
    return {
        "episodes": rows,
        "count": len(rows),
        "reference_longest": longest,
        "overlap_count": len(matching),
        "missing": not matching,
        "fragmented": len(matching) > 1,
        "overmerged_across_reference_episodes": any(len(x["reference_episodes_overlapped"]) > 1 for x in matching),
        "candidate_overlaps_longest": any(x["b_candidate"] for x in matching),
        "candidate_start_within_60s": any(x["b_candidate"] and x["start_error_s"] <= 60 for x in rows),
    }


def summarize(cases: list[dict]) -> dict:
    summary = {"development_only": True, "endpoints": {}}
    for endpoint in ENDPOINTS:
        rows = [row for row in cases if row["qtype"] == endpoint]
        out = {"n": len(rows), "attempts": {}}
        for phase in ("original", "after_retry"):
            out["attempts"][phase] = {}
            for arm in ("A0", "B", "C"):
                scores = [r["scores"][phase][arm] for r in rows]
                counts = Counter(s["failure_class"] for s in scores)
                out["attempts"][phase][arm] = {
                    "correct": sum(s["correct"] for s in scores),
                    **{
                        key: counts[key]
                        for key in ("infrastructure", "execution_contract", "model_workflow", "detector_rule")
                    },
                }
        out["c_vs_a0"] = {
            "corrections": [r["case_id"] for r in rows if r["c_a0_change"] == "correction"],
            "introduced_errors": [r["case_id"] for r in rows if r["c_a0_change"] == "introduced_error"],
        }
        out["c_vs_b"] = {
            "c_only_correct": [
                r["case_id"]
                for r in rows
                if r["scores"]["after_retry"]["C"]["correct"] and not r["scores"]["after_retry"]["B"]["correct"]
            ],
            "b_only_correct": [
                r["case_id"]
                for r in rows
                if r["scores"]["after_retry"]["B"]["correct"] and not r["scores"]["after_retry"]["C"]["correct"]
            ],
        }
        for arm in ("B", "C"):
            wrong = [r for r in rows if not r["scores"]["after_retry"][arm]["correct"]]
            out[f"{arm}_earliest"] = dict(
                Counter(
                    r.get("judgment", {}).get("arms", {}).get(arm, {}).get("earliest_failure", "pending") for r in wrong
                )
            )
            out[f"{arm}_workflow"] = dict(
                Counter(
                    r.get("judgment", {}).get("arms", {}).get(arm, {}).get("actionable_workflow_failure", "pending")
                    for r in wrong
                )
            )
        summary["endpoints"][endpoint] = out
    onset = [row for row in cases if row["qtype"] == "longest_af"]
    summary["coverage"] = {
        "onset_n": len(onset),
        "strip_contains_reference_onset": sum(r["b"]["strip_contains_reference_onset"] for r in onset),
        "strip_within_60s_of_reference_onset": sum(r["b"]["strip_within_60s_of_reference_onset"] for r in onset),
        "candidate_overlaps_longest": sum(r["detector"]["candidate_overlaps_longest"] for r in onset),
        "candidate_start_within_60s": sum(r["detector"]["candidate_start_within_60s"] for r in onset),
        "fragmented_reference_longest": sum(r["detector"]["fragmented"] for r in onset),
        "cases_with_drops": sum(bool(r["b"]["dropped"]) for r in cases),
        "drop_reasons": dict(Counter(x["reason"] for r in cases for x in r["b"]["dropped"])),
    }
    burden = [row for row in cases if row["qtype"] == "af_burden"]
    for arm in ("B", "C"):
        summary[f"{arm}_burden_adjustments"] = dict(
            Counter(
                "improved_absolute_error"
                if r["scores"]["after_retry"][arm]["abs_error"] < r["scores"]["after_retry"]["A0"]["abs_error"]
                else "worsened_absolute_error"
                if r["scores"]["after_retry"][arm]["abs_error"] > r["scores"]["after_retry"]["A0"]["abs_error"]
                else "unchanged_absolute_error"
                for r in burden
            )
        )
    summary["bundle_sizes"] = {
        key: {
            "min": min(r["b"]["sizes"][key] for r in cases),
            "median": statistics.median(r["b"]["sizes"][key] for r in cases),
            "max": max(r["b"]["sizes"][key] for r in cases),
        }
        for key in ("text_chars", "core_text_chars", "verification_text_chars", "images", "calls")
    }
    summary["c_presentation_rejections"] = [
        {
            "case_id": row["case_id"],
            "record": row["record"],
            "qtype": row["qtype"],
            "calls": [
                {
                    "call_line": call["call_line"],
                    "result_line": call["result_line"],
                    "arguments": call["arguments"],
                    "notice": call["evidence"]["non_json_text"],
                }
                for call in row["transcripts"]["C"]["calls"]
                if call.get("presentation_error")
            ],
        }
        for row in cases
        if any(call.get("presentation_error") for call in row["transcripts"]["C"]["calls"])
    ]
    return summary


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def render_case_notes(cases: list[dict]) -> str:
    lines = [
        "# p7dev case traces (development diagnosis only)",
        "",
        "All 66 questions are included, including numeric B/C disagreements that do not change correctness. "
        "Onset units are seconds; burden units are percent. ✓/✗ use the unchanged 60 s / 5 pp grader. "
        "Tool line numbers refer to the hashed per-case C transcript. Public claims of morphology are attributed "
        "to the answering model, not independent image re-annotation. Full arguments, payload values, support "
        "geometry, all B omissions and public decisions are in `cases.json`.",
        "",
        "| Record | Endpoint | Reference | A0 | B | C | C vs A0 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    def formatted(row, arm):
        result = row["scores"]["after_retry"][arm]
        key = "t" if row["qtype"] == "longest_af" else "percent"
        value = (result["pred"] or {}).get(key)
        number = "null" if value is None else f"{value:.6g}"
        return number + (" ✓" if result["correct"] else " ✗")

    for row in cases:
        key = "t" if row["qtype"] == "longest_af" else "percent"
        lines.append(
            f"| {row['record']} | {row['qtype']} | {row['answer'][key]:.6g} | "
            f"{formatted(row, 'A0')} | {formatted(row, 'B')} | {formatted(row, 'C')} | {row['c_a0_change']} |"
        )
    for row in cases:
        judgment = row.get("judgment", {})
        geometry = row["detector"]
        support = geometry["support_diagnosis"]
        lines.extend(
            [
                "",
                f"## {row['record']} / {row['qtype']} / {row['case_id']}",
                "",
                f"Reference longest interval: {row['reference']['longest']}; total AF {row['reference']['af_s']:.3f} s "
                f"of {row['reference']['duration_s']:.3f} s. R has {geometry['count']} episodes; "
                f"{geometry['overlap_count']} overlap the reference longest. R support overlaps "
                f"{support['overlap_reference_af_s']:.3f} reference AF seconds, misses {support['missed_reference_af_s']:.3f} s "
                f"and adds {support['false_positive_s']:.3f} false-support seconds.",
                "",
                judgment.get("detector_diagnosis", judgment.get("detector_attribution", "")),
                "",
                f"B retained {row['b']['sizes']['retained_calls']} calls / {row['b']['sizes']['images']} images; "
                f"{len(row['b']['dropped'])} omitted requests. Exact omitted keys/windows/reasons are in `cases.json`. "
                "Retained item keys: " + ", ".join(f"`{i['key']}`" for i in row["b"]["items"]) + ".",
            ]
        )
        if row["qtype"] == "longest_af":
            lines.extend(
                [
                    "",
                    f"Candidate overlaps reference longest: {geometry['candidate_overlaps_longest']}; "
                    f"candidate starts within 60 s: {geometry['candidate_start_within_60s']}; "
                    f"retained strip contains onset: {row['b']['strip_contains_reference_onset']}. "
                    "Overlap alone does not establish episode identity, continuity or an accurate start.",
                ]
            )
        for arm in ("B", "C"):
            note = judgment.get("arms", {}).get(arm, {})
            lines.extend(
                [
                    "",
                    f"**{arm}: {formatted(row, arm)}.** Earliest: {note.get('earliest_failure') or 'none'}; "
                    f"workflow: {note.get('actionable_workflow_failure') or 'none'}; "
                    f"onset mode: {note.get('onset_error_mode', 'not applicable')}. "
                    + note.get("reason", note.get("decision_and_attribution", "")),
                ]
            )
            if arm == "B":
                used = [i["key"] for i in note.get("reported_used_items", [])]
                used += judgment.get("B_evidence", {}).get("items_cited_by_public_answer", [])
                if used:
                    lines.append("\nPublicly cited bundle items: " + ", ".join(f"`{key}`" for key in used) + ".")
            if note.get("public_decision_quote"):
                lines.append("\nPublic decision excerpt: " + note["public_decision_quote"])
        lines.extend(["", "C retrieval trace (complete key values in `cases.json`):", ""])
        for call in row["transcripts"]["C"]["calls"]:
            arguments = {key: val for key, val in call["arguments"].items() if key not in ("id", "source")}
            state = "usable" if call.get("evidence_usable") else "unavailable/error"
            lines.append(
                f"- L{call['call_line']}→{call.get('result_line', '?')}: `{call['tool']}` "
                f"`{compact(arguments)}`; {state}."
            )
    return "\n".join(lines) + "\n"


def build(root: Path, output: Path, judgments_path: Path | None) -> dict:
    import wfdb  # dev-only annotation reader; no signal loading or network request

    files = {}

    def track(path: Path) -> Path:
        key = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        files[key] = sha256(path)
        return path

    def jsonl(relative: str) -> dict:
        return keyed(read_jsonl(track(root / relative)))

    answers = jsonl("eval/cases/p7dev.answers.jsonl")
    questions = jsonl("eval/cases/p7dev.questions.jsonl")
    a0 = keyed(json.loads(track(root / "eval/results/part2-dev/a0-R.predictions.json").read_text())["rows"])
    if answers.keys() != questions.keys() or answers.keys() != a0.keys():
        raise ValueError("question/answer/A0 identities differ")
    if Counter(a["qtype"] for a in answers.values()) != Counter(longest_af=34, af_burden=32):
        raise ValueError("expected exactly 34 onset and 32 burden p7dev cases")
    runs, errors, original, original_errors = {}, {}, {}, {}
    for arm, run in RUNS.items():
        runs[arm] = jsonl(f"eval/runs/{run}/results.jsonl")
        errors[arm] = jsonl(f"eval/runs/{run}/errors.jsonl")
        label = "B-v0" if arm == "B" else "C-dev"
        original[arm] = jsonl(f"eval/results/part2-dev/original/{label}.results.jsonl")
        original_errors[arm] = jsonl(f"eval/results/part2-dev/original/{label}.errors.jsonl")
        for mapping in (runs[arm], errors[arm], original[arm], original_errors[arm]):
            if not mapping.keys() <= answers.keys():
                raise ValueError("run contains a non-p7dev case")
        if original[arm].keys() & original_errors[arm].keys():
            raise ValueError("original attempt has both success and failure")
        if original[arm].keys() | original_errors[arm].keys() != answers.keys():
            raise ValueError("original attempt inventory incomplete")
        for cid, row in original[arm].items():
            if row != runs[arm].get(cid):
                raise ValueError("a successful original attempt was replaced")
    track(root / "eval/results/part2-dev/analysis-v0-after-retry.json")
    annotations = {}
    references = {}
    cases = []
    for cid, answer in answers.items():
        record = answer["answer"]["record"]
        if not record.isascii() or not record.isdigit():
            raise ValueError("invalid development record name")
        if record not in references:
            base = root / "data/ltafdb" / record
            track(base.with_suffix(".atr"))
            track(base.with_suffix(".hea"))
            header = wfdb.rdheader(str(base))
            ann = wfdb.rdann(str(base), "atr")
            duration = header.sig_len / header.fs
            raw = annotation_intervals(ann.sample, ann.aux_note, header.fs, duration)
            merged = union(raw, gap=30)
            longest = min(merged, key=lambda iv: (-(iv[1] - iv[0]), iv[0]))
            references[record] = {
                "duration_s": duration,
                "af_s": interval_seconds(raw),
                "percent": 100 * interval_seconds(raw) / duration,
                "raw_af_intervals": raw,
                "merged_episodes": merged,
                "longest": longest,
            }
        ref = references[record]
        truth_key = "t" if answer["qtype"] == "longest_af" else "percent"
        truth = ref["longest"][0] if truth_key == "t" else ref["percent"]
        if abs(truth - answer["answer"][truth_key]) > 1e-8:
            raise ValueError(f"reference annotation does not reproduce answer: {cid}")
        manifest_path = root / "eval/results/b-workflow/p7dev/manifests" / f"{cid}.json"
        manifest = json.loads(track(manifest_path).read_text())
        bundle_path = root / "eval/cache/b-workflow/objects" / f"{manifest['bundle_sha256']}.json"
        bundle = json.loads(track(bundle_path).read_text())
        if sha256(bundle_path) != manifest["bundle_sha256"]:
            raise ValueError("B bundle digest differs from manifest")
        if cid in runs["B"] and runs["B"][cid]["bundle_sha256"] != manifest["bundle_sha256"]:
            raise ValueError("B run used a different bundle")
        items = [
            {**{k: v for k, v in item.items() if k != "content"}, "evidence": content_evidence(item["content"])}
            for item in bundle["items"]
        ]
        if [item["key"] for item in items] != [item["key"] for item in manifest["items"]]:
            raise ValueError("manifest retained items differ from bundle")
        measurements = [
            m
            for item in items
            for payload in item["evidence"]["payloads"]
            if isinstance(payload, dict)
            for m in payload.get("measurements", [])
        ]
        # Geometry uses original support arrays, never the projected public table.
        episode_item = next(item for item in bundle["items"] if item["key"] == "episode_index")
        episode_payload = json.loads(next(b["text"] for b in episode_item["content"] if b["type"] == "text"))
        episodes = next(m["episodes"] for m in episode_payload["measurements"] if m["name"] == "ecg_af_episodes")
        if answer["qtype"] == "longest_af" and episodes != a0[cid]["detector_payload"]["episodes"]:
            raise ValueError("B index differs from the saved A0 R index")
        geometry = episode_geometry(episodes, ref["merged_episodes"], ref["longest"])
        burden = next(m for m in measurements if m["name"] == "ecg_af_burden")
        geometry["burden"] = burden
        support = union([iv for ep in episodes for iv in ep["intervals_s"]])
        tp_s = sum(overlap(det, gt) for det in support for gt in ref["raw_af_intervals"])
        geometry["support_diagnosis"] = {
            "detector_supported_s": interval_seconds(support),
            "overlap_reference_af_s": tp_s,
            "false_positive_s": interval_seconds(support) - tp_s,
            "missed_reference_af_s": ref["af_s"] - tp_s,
        }
        scores = {}
        for phase, result_rows, error_rows in (("original", original, original_errors), ("after_retry", runs, errors)):
            scores[phase] = {
                "A0": {
                    **score(answer, a0[cid]),
                    "failure_class": "none"
                    if grade(answer["qtype"], answer["answer"], a0[cid]["pred"])["correct"]
                    else "detector_rule",
                },
                **{arm: score(answer, result_rows[arm].get(cid), error_rows[arm].get(cid)) for arm in ("B", "C")},
            }
        transcripts = {}
        for arm, run in RUNS.items():
            path = track(root / "eval/runs" / run / "transcripts" / f"{cid}.jsonl")
            transcripts[arm] = {"path": str(path.relative_to(root)), **parse_transcript(read_jsonl(path))}
        strips = [
            item["arguments"]["window"]
            for item in items
            if item["tool"] == "ts_zoom" and item["arguments"].get("view") == "strip"
        ]
        a_ok, c_ok = scores["after_retry"]["A0"]["correct"], scores["after_retry"]["C"]["correct"]
        cases.append(
            {
                "case_id": cid,
                "record": record,
                "qtype": answer["qtype"],
                "question": questions[cid]["question"],
                "answer": answer["answer"],
                "reference": ref,
                "scores": scores,
                "detector": geometry,
                "c_a0_change": "correction"
                if c_ok and not a_ok
                else "introduced_error"
                if a_ok and not c_ok
                else "unchanged",
                "b_c_correctness_disagreement": scores["after_retry"]["B"]["correct"] != c_ok,
                "b_c_answer_disagreement": scores["after_retry"]["B"]["pred"] != scores["after_retry"]["C"]["pred"],
                "b": {
                    "bundle_sha256": manifest["bundle_sha256"],
                    "sizes": manifest["sizes"],
                    "budget": manifest["budget"],
                    "verification_text_budget": manifest["verification_text_budget"],
                    "items": items,
                    "arithmetic": bundle["arithmetic"],
                    "dropped": manifest["dropped"],
                    "strip_contains_reference_onset": any(w[0] <= ref["longest"][0] < w[1] for w in strips),
                    "strip_within_60s_of_reference_onset": any(
                        w[0] - 60 <= ref["longest"][0] <= w[1] + 60 for w in strips
                    ),
                    "decision": runs["B"].get(cid, {}).get("text"),
                },
                "c": {"decision": runs["C"][cid]["text"]},
                "transcripts": transcripts,
            }
        )
    input_manifest = {"suite": "p7dev", "files": dict(sorted(files.items()))}
    fingerprint = hashlib.sha256(compact(input_manifest).encode()).hexdigest()
    if judgments_path:
        reviewed = json.loads(judgments_path.read_text())
        if reviewed["input_manifest_sha256"] != fingerprint:
            raise ValueError("reviewed judgments bind different inputs; re-review before regeneration")
        annotations = reviewed["cases"]
        if annotations.keys() != answers.keys():
            raise ValueError("reviewed table must cover all 66 cases, including numeric disagreements")
        for row in cases:
            row["judgment"] = annotations.get(row["case_id"], {})
            if (
                not all(row["scores"]["after_retry"][arm]["correct"] for arm in ("B", "C"))
                or row["c_a0_change"] != "unchanged"
            ) and not row["judgment"]:
                raise ValueError(f"required judgment missing: {row['case_id']}")
            if row["judgment"]["record"] != row["record"] or row["judgment"]["qtype"] != row["qtype"]:
                raise ValueError("judgment case identity mismatch")
            for arm in ("B", "C"):
                note = row["judgment"]["arms"][arm]
                if note["correct"] != row["scores"]["after_retry"][arm]["correct"]:
                    raise ValueError("reviewed correctness differs from unchanged grader")
    summary = summarize(cases)
    summary["input_manifest_sha256"] = fingerprint
    summary["reviewed"] = judgments_path is not None
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input-manifest.json", input_manifest)
    # One case per physical line avoids multi-megabyte pretty-print whitespace.
    envelope = '{"development_only":true,"input_manifest_sha256":' + compact(fingerprint) + ',"cases":[\n'
    (output / "cases.json").write_text(envelope + ",\n".join(compact(row) for row in cases) + "\n]}\n")
    write_json(output / "summary.json", summary)
    (output / "case-traces.md").write_text(render_case_notes(cases))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/home/yanghaoxiang24/signal_agent"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    summary = build(args.input_root.resolve(), args.out, args.judgments)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
