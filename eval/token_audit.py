"""Offline, matched M/R transcript size and usage audit; never calls a model.

Counts the tool_result content presented to the model once, excluding the duplicate
SDK tool_use_result envelope and image base64. Request IDs deduplicate streamed
assistant fragments. Character exposures describe an un-compacted history; they
are not provider tokens or a reconstruction of undocumented wire requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ("claude-opus-5-5-medium", "deepseek-flash_deepseek-flash-high")
TOKEN_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
EPISODE_FIELDS = ("intervals_s", "span_s", "af_s", "start")


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_arguments(arguments: dict, duration: float) -> dict:
    """Only normalize opaque handles, sandbox paths, and exact whole windows."""
    result = dict(arguments)
    if "id" in result:
        result["id"] = "<signal>"
    if "source" in result:
        result["source"] = "<signal.npy>"
    if result.get("window") == [0, duration]:
        result.pop("window")
    if isinstance(result.get("what"), list):
        result["what"] = sorted(result["what"])
    return result


def text_and_images(content: object) -> tuple[list[str], int]:
    if isinstance(content, str):
        return [content], 0
    if not isinstance(content, list):
        return [], 0
    return [b["text"] for b in content if b.get("type") == "text"], sum(b.get("type") == "image" for b in content)


def field_size(value: object, key: str) -> int:
    """Exact compact-JSON reduction when this key is omitted recursively."""

    def without(item: object) -> object:
        if isinstance(item, list):
            return [without(x) for x in item]
        if isinstance(item, dict):
            return {k: without(v) for k, v in item.items() if k != key}
        return item

    return len(compact(value)) - len(compact(without(value)))


def payload_details(texts: list[str]) -> dict:
    measurements: Counter = Counter()
    field_chars: Counter = Counter()
    episodes = intervals = 0
    for text in texts:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for measurement in payload.get("measurements", []):
            measurements[measurement["name"]] += len(compact(measurement))
            episodes += len(measurement.get("episodes", []))
        episodes += len(payload.get("events", []))
        for field in EPISODE_FIELDS:
            field_chars[field] += field_size(payload, field)

        def count_intervals(value: object) -> int:
            if isinstance(value, list):
                return sum(count_intervals(x) for x in value)
            if isinstance(value, dict):
                return len(value.get("intervals_s", [])) + sum(
                    count_intervals(v) for k, v in value.items() if k != "intervals_s"
                )
            return 0

        intervals += count_intervals(payload)
    return {
        "measurement_chars": dict(measurements),
        "field_chars": dict(field_chars),
        "episodes_or_events": episodes,
        "support_intervals": intervals,
    }


def audit_case(directory: Path, row: dict, duration: float) -> dict:
    path = directory / "transcripts" / f"{row['case_id']}.jsonl"
    events = read_jsonl(path)
    requests: dict[str, dict] = {}
    calls: dict[str, dict] = {}
    retries = []
    result = None
    for step, event in enumerate(events, 1):
        if event.get("type") == "assistant":
            message = event["message"]
            request = requests.setdefault(
                message["id"], {"id": message["id"], "first_step": step, "usage": dict.fromkeys(TOKEN_KEYS, 0)}
            )
            # Stream fragments share input usage; max avoids counting partial output repeatedly.
            for key in TOKEN_KEYS:
                request["usage"][key] = max(request["usage"][key], message.get("usage", {}).get(key) or 0)
            for block in message.get("content", []):
                if block.get("type") != "tool_use":
                    continue
                tool_id = block["id"]
                if tool_id in calls:
                    continue
                arguments = canonical_arguments(block["input"], duration)
                name = block["name"].removeprefix("mcp__tsmcp__")
                calls[tool_id] = {
                    "tool_use_id": tool_id,
                    "request_id": message["id"],
                    "call_step": step,
                    "tool": name,
                    "arguments": arguments,
                    "signature": json.dumps([name, arguments], sort_keys=True, separators=(",", ":")),
                }
        if event.get("type") == "user":
            for block in event.get("message", {}).get("content", []):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call = calls[block["tool_use_id"]]
                if "result_step" in call:
                    raise ValueError(f"duplicate result: {path}:{step}")
                texts, images = text_and_images(block.get("content"))
                call.update(
                    result_step=step,
                    chars=sum(map(len, texts)),
                    utf8_bytes=sum(len(t.encode()) for t in texts),
                    images=images,
                    is_error=bool(block.get("is_error")),
                    error_text="\n".join(texts)[:500] if block.get("is_error") else None,
                    **payload_details(texts),
                )
        if event.get("subtype") == "api_retry":
            retries.append({k: event.get(k) for k in ("attempt", "error_status", "error", "retry_delay_ms")})
        if event.get("type") == "result":
            result = event
    if result is None:
        raise ValueError(f"missing final result: {path}")
    seen: dict[str, list[dict]] = defaultdict(list)
    for call in calls.values():
        if "result_step" not in call:
            raise ValueError(f"missing tool result: {path}: {call['tool_use_id']}")
        later = sum(r["first_step"] > call["result_step"] for r in requests.values())
        call["following_requests"] = later
        call["history_exposure_chars"] = call["chars"] * later
        call["resend_chars"] = call["chars"] * max(0, later - 1)
        earlier = seen[call["signature"]]
        call["repeated_query"] = bool(earlier)
        call["repeat_after_error"] = bool(earlier and earlier[-1]["is_error"])
        seen[call["signature"]].append(call)
    usage = {k: row.get("usage", {}).get(k) or 0 for k in TOKEN_KEYS}
    if any(usage[k] != (result.get("usage", {}).get(k) or 0) for k in TOKEN_KEYS):
        raise ValueError(f"result usage mismatch: {path}")
    request_usage = {k: sum(r["usage"][k] for r in requests.values()) for k in TOKEN_KEYS}
    return {
        "case_id": row["case_id"],
        "qtype": row["qtype"],
        "model": row["model"],
        "server_root": row.get("server_root"),
        "server_git_sha": row.get("server_git_sha"),
        "arm": directory.name[-1],
        "transcript": str(path),
        "transcript_sha256": digest(path),
        "turns": row["turns"],
        "requests": len(requests),
        "usage": usage,
        "total_input_tokens": sum(usage[k] for k in TOKEN_KEYS[:3]),
        "request_usage": request_usage,
        "usage_minus_request_usage": {k: usage[k] - request_usage[k] for k in TOKEN_KEYS},
        "recorded_cost_usd": row.get("cost_usd"),
        "cc_cost_usd": row.get("cc_cost_usd"),
        "model_usage": result.get("modelUsage"),
        "calls": list(calls.values()),
        "api_retries": retries,
        "compaction_events": sum("compact" in str(e.get("subtype", "")) for e in events if e["type"] == "system"),
        "request_usage_trace": list(requests.values()),
    }


def size_summary(calls: list[dict]) -> dict:
    sizes = [c["chars"] for c in calls]
    return {
        "calls": len(calls),
        "chars": sum(sizes),
        "mean_chars": statistics.mean(sizes) if sizes else 0,
        "median_chars": statistics.median(sizes) if sizes else 0,
        "max_chars": max(sizes, default=0),
        "history_exposure_chars": sum(c["history_exposure_chars"] for c in calls),
        "resend_chars": sum(c["resend_chars"] for c in calls),
        "images": sum(c["images"] for c in calls),
        "errors": sum(c["is_error"] for c in calls),
        "repeated_queries": sum(c["repeated_query"] for c in calls),
        "repeat_after_error": sum(c["repeat_after_error"] for c in calls),
        "field_chars": {f: sum(c["field_chars"].get(f, 0) for c in calls) for f in EPISODE_FIELDS},
    }


def summarize(cases: list[dict]) -> dict:
    calls = [c for case in cases for c in case["calls"]]
    names = sorted({c["tool"] for c in calls})
    measurements = sorted({m for c in calls for m in c["measurement_chars"]})
    total_input = sum(case["total_input_tokens"] for case in cases)
    return {
        "cases": len(cases),
        "turns": sum(case["turns"] for case in cases),
        "requests": sum(case["requests"] for case in cases),
        "usage": {k: sum(case["usage"][k] for case in cases) for k in TOKEN_KEYS},
        "total_input_tokens": total_input,
        "cache_read_fraction": sum(case["usage"]["cache_read_input_tokens"] for case in cases) / total_input,
        "recorded_cost_usd": sum(case["recorded_cost_usd"] or 0 for case in cases),
        "cc_cost_usd": sum(case["cc_cost_usd"] or 0 for case in cases),
        "api_retries": sum(len(case["api_retries"]) for case in cases),
        "cases_with_retries": [case["case_id"] for case in cases if case["api_retries"]],
        "compaction_events": sum(case["compaction_events"] for case in cases),
        "tools": {name: size_summary([c for c in calls if c["tool"] == name]) for name in names},
        "measurements": {
            name: {
                "calls": sum(name in c["measurement_chars"] for c in calls),
                "cases": sum(any(name in c["measurement_chars"] for c in case["calls"]) for case in cases),
                "chars": sum(c["measurement_chars"].get(name, 0) for c in calls),
                "history_exposure_chars": sum(
                    c["measurement_chars"].get(name, 0) * c["following_requests"] for c in calls
                ),
            }
            for name in measurements
        },
        "all_tools": size_summary(calls),
    }


def match_cases(cases: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(dict)
    for case in cases:
        groups[case["model"], case["case_id"]][case["arm"]] = case
    matched = []
    for (model, case_id), pair in sorted(groups.items()):
        if set(pair) != {"M", "R"}:
            raise ValueError(f"unpaired case: {model} {case_id}")
        queries: dict[str, dict[str, list]] = defaultdict(lambda: {"M": [], "R": []})
        for arm, case in pair.items():
            for call in case["calls"]:
                if not call["is_error"]:
                    queries[call["signature"]][arm].append(call)
        identical = []
        for signature, by_arm in sorted(queries.items()):
            for m_call, r_call in zip(by_arm["M"], by_arm["R"], strict=False):
                identical.append(
                    {
                        "signature": signature,
                        "tool": m_call["tool"],
                        "M_chars": m_call["chars"],
                        "R_chars": r_call["chars"],
                        "delta_chars": r_call["chars"] - m_call["chars"],
                        "M_result_step": m_call["result_step"],
                        "R_result_step": r_call["result_step"],
                        "M_field_chars": m_call["field_chars"],
                        "R_field_chars": r_call["field_chars"],
                        "M_episodes_or_events": m_call["episodes_or_events"],
                        "R_episodes_or_events": r_call["episodes_or_events"],
                        "M_support_intervals": m_call["support_intervals"],
                        "R_support_intervals": r_call["support_intervals"],
                    }
                )
        matched.append(
            {
                "model": model,
                "case_id": case_id,
                "qtype": pair["M"]["qtype"],
                **{
                    f"{arm}_{metric}": case[metric]
                    for arm, case in pair.items()
                    for metric in ("total_input_tokens", "turns", "requests")
                },
                "delta_input_tokens": pair["R"]["total_input_tokens"] - pair["M"]["total_input_tokens"],
                "identical_queries": identical,
            }
        )
    return matched


def render_report(result: dict) -> str:
    lines = [
        "# Offline M/R token-growth audit",
        "",
        "All 56 model/case pairs are matched; 112 complete transcripts. Input is the sum of uncached input, cache creation and cache read fields. Counts below are totals over 28 cases per arm unless marked mean. SDK `num_turns` and actual distinct assistant request IDs are different: parallel tool actions share a request, so both counts are shown.",
        "",
        "| Model | Arm | Input tokens/case | Uncached input | Cache creation | Cache read | Output | Turns/case | Requests/case | Tool calls | API retries | Recorded $ | SDK $ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, pair in result["summary"].items():
        for arm, s in pair.items():
            u = s["usage"]
            lines.append(
                f"| {model} | {arm} | {s['total_input_tokens'] / s['cases']:,.1f} | {u['input_tokens']:,} | {u['cache_creation_input_tokens']:,} | {u['cache_read_input_tokens']:,} | {u['output_tokens']:,} | {s['turns'] / s['cases']:.2f} | {s['requests'] / s['cases']:.2f} | {s['all_tools']['calls']} | {s['api_retries']} | {s['recorded_cost_usd']:.5f} | {s['cc_cost_usd']:.5f} |"
            )
    lines += [
        "",
        "`cost_usd` is the evaluator's saved price-based estimate, not a bill. Claude uses the run's frozen rates ($4/$20/$0.20/$5 per million uncached input/output/cache read/cache creation); the SDK reports list-based accounting with some additional usage. DeepSeek's zero evaluator cost reflects unspecified prices; its SDK `costBasis=unknown` dollar estimate is not a valid measured spend. No current prices or API calls were used. No provider balance/invoice was retained here, so realized spend cannot be reconstructed. Retry costs are unknown when no usage was emitted.",
        "",
        "## Tool text and history exposure",
        "",
        "| Model | Tool | Calls M/R | Mean response chars M/R | Total chars M/R | History exposure chars M/R | Images M/R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, pair in result["summary"].items():
        for tool in sorted(set(pair["M"]["tools"]) | set(pair["R"]["tools"])):
            m = pair["M"]["tools"].get(tool, size_summary([]))
            r = pair["R"]["tools"].get(tool, size_summary([]))
            lines.append(
                f"| {model} | {tool} | {m['calls']}/{r['calls']} | {m['mean_chars']:,.1f}/{r['mean_chars']:,.1f} | {m['chars']:,}/{r['chars']:,} | {m['history_exposure_chars']:,}/{r['history_exposure_chars']:,} | {m['images']}/{r['images']} |"
            )
    lines += [
        "",
        "History exposure sums each tool text's characters multiplied by the number of subsequent distinct assistant request IDs. Resend characters subtract its first exposure. This is a transparent full-history proxy, not an exact provider prompt reconstruction: system/tool schemas, assistant output/thinking, image encoding, hidden wrapper messages, cache policy, and failed requests are not reconstructible from these events. No compaction events are observed. Base64 image bytes and duplicate SDK envelopes are excluded. Streamed assistant blocks sharing one message ID are counted once. Per-request input usage reconciles to session totals as recorded in the JSON; output fragments need not expose final billed output.",
        "",
        "No local tokenizer for either served model is available. Per-tool exact token attribution is therefore unavailable; character counts are exact Unicode text lengths, never labelled tokens. Provider usage is the source of actual token totals. Image counts are reported separately and no characters-to-tokens conversion is imposed.",
        "",
        "## Within-case identical-query comparisons",
        "",
        "Only successful queries with the same tool and arguments are paired, in occurrence order. Handles/sandbox paths and an explicit exact whole-record window are normalized; `what` list order is ignored. Different windows or candidate limits are not equated. These comparisons isolate response differences more closely than aggregate averages but are a selected subset, not randomized estimates.",
        "",
        "| Tool | Matched calls | M chars | R chars | Change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for tool, s in result["matched_query_summary"].items():
        lines.append(f"| {tool} | {s['calls']} | {s['M_chars']:,} | {s['R_chars']:,} | {s['delta_chars']:+,} |")
    lines += [
        "",
        "No identical matched query grew. The response route changed: frozen M does not register the `ecg_af_episodes` measurement, whereas R does. M typically retrieves compact `ts_find_events` candidates; R often retrieves the full nested episode list with `ts_measure`. Thus this is not evidence that identical prompts to the same tool became larger. The route, completeness, detector output and agent choices changed together.",
        "",
        "Actual growth examples, paired by case/model but explicitly using different candidate queries:",
        "",
        "| Model/case | M first candidate tool, chars | R first episode result, chars | R episode component chars | R episodes / support intervals | Subsequent requests M/R | M/R result lines |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for example in result["candidate_route_examples"]:
        lines.append(
            f"| {example['model']}/{example['case_id']} | {example['M_tool']}, {example['M_chars']:,} | {example['R_chars']:,} | {example['R_episode_chars']:,} | {example['R_episodes']}/{example['R_intervals']} | {example['M_following_requests']}/{example['R_following_requests']} | {example['M_result_step']}/{example['R_result_step']} |"
        )
    lines += [
        "",
        "The JSON retains the exact two argument sets, including M's `top_k` and either window, so these examples do not imply identical evidence or response completeness. For example, a result of 48,472 characters retained for nine later requests contributes 436,248 character-exposures, including its first presentation; the extra eight replays contribute 387,776.",
        "",
        "| Model | Episode calls / cases in R | Episode chars M→R | Episode history exposure M→R | Net all-tool text growth | Net all-tool history growth | R support-array chars | R start/span/AF-seconds chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, pair in result["summary"].items():
        ep = pair["R"]["measurements"]["ecg_af_episodes"]
        m, r = pair["M"]["all_tools"], pair["R"]["all_tools"]
        fields = r["field_chars"]
        extra = sum(fields[k] for k in ("start", "span_s", "af_s"))
        lines.append(
            f"| {model} | {ep['calls']} / {ep['cases']} | 0→{ep['chars']:,} | 0→{ep['history_exposure_chars']:,} | {r['chars'] - m['chars']:+,} | {r['history_exposure_chars'] - m['history_exposure_chars']:+,} | {fields['intervals_s']:,} | {extra:,} |"
        )
    lines += [
        "",
        "Field counts are exact compact-JSON size reductions from recursively omitting each named field, including its key/delimiter. The last column includes `af_s` from burden payloads too, and is not all removable redundancy: supported AF seconds must remain distinct from span. Episode text exceeds the net growth because other tool text shrank. Support arrays account for about half of episode text; raw nested episode metadata, repeated aliases and floating-point tails account for much of the rest. No output was actually removed or changed.",
    ]
    lines += ["", "## Repetition, retries and mechanism", ""]
    for model, pair in result["summary"].items():
        m, r = pair["M"]["all_tools"], pair["R"]["all_tools"]
        delta = pair["R"]["total_input_tokens"] / pair["M"]["total_input_tokens"] - 1
        lines.append(
            f"- {model}: input {delta:+.1%}; raw tool text {m['chars']:,} → {r['chars']:,}; full-history tool-text exposure {m['history_exposure_chars']:,} → {r['history_exposure_chars']:,}; resends beyond first exposure {m['resend_chars']:,} → {r['resend_chars']:,}. Repeated normalized queries {m['repeated_queries']} → {r['repeated_queries']} (after errors {m['repeat_after_error']} → {r['repeat_after_error']}); tool errors {m['errors']} → {r['errors']}. Repetition is a diagnostic count, not proof every repeat was unnecessary."
        )
    lines += [
        "",
        "The principal structural growth is detailed `ecg_af_episodes` payloads under R: support interval arrays plus repeated episode metadata and long floating-point representations. M/R candidate counts and requested measurements differ, so aggregate tool-size changes combine representation, detector output and agent policy. Identical-query comparisons show no growth; the new episode route and field-removal accounting explain the observed text growth. No factorial counterfactual separates detector changes from interface and agent policy. The large result remains in later context, multiplying its input exposure even when R takes fewer turns. Existing input usage also includes images and model reasoning/context; no exact causal token percentage is assigned to a tool.",
        "",
        "## Proposed compaction only",
        "",
        "- Return a compact, complete episode index with start/end/span, AF-supported seconds, grouping gap, channel, units, requested/clipped window, candidate count, ranking rule and stable evidence references. Preserve every candidate needed to establish the longest event, deterministic earliest-start ties and unambiguous completeness; paging must retain a global complete index or a guaranteed global longest selection, never silently truncate alternatives.",
        "- Put full `intervals_s` support behind explicit per-episode detail retrieval with stable refs; keep the full evidence accessible and preserve the distinction between supported AF seconds and merged span. Return burden numerator, whole-window/readable denominators, unreadable seconds and both percentages together.",
        "- Apply existing type-aware precision consistently within nested episode objects: timestamps at the sampling precision and measurements at three significant figures; avoid redundant `t/start` and `dur/span_s` aliases in the proposed view while preserving explicit labels. Do not round before ranking or interval integration.",
        "- Summarize old results into a factual evidence ledger keyed by ref with bounds, units, provenance, denominators, selected and alternative candidates, confidence caveats and pending checks; retain an explicit route back to original full results. Cache or reuse identical successful queries, distinguish failed calls and corrected arguments, and avoid running dependent calls before a load handle exists.",
        "- Separate retry diagnostics from reasoning turns, persist provider usage and retry billing when supplied, and report unknown prices as unknown. Evaluate any compact interface with fixed detector, prompts, grader and a new held-out protocol; this opened AFDB audit is diagnostic only.",
        "",
        "Full per-case calls, request IDs/usage, field sizes, repetitions, hashes and matched comparisons: `token-audit.json`. No detector, tool or model prompt was changed. Existing project tests were unchanged; new synthetic accounting tests protect the audit parser.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/home/yanghaoxiang24/signal_agent/eval/runs"))
    parser.add_argument("--output", type=Path, default=Path("eval/results/audit-a0/token-audit.json"))
    args = parser.parse_args()
    cases = []
    provenance = {}
    for model in MODELS:
        for arm in ("M", "R"):
            directory = args.runs_root / f"p5afdb-C-{model}-{arm}"
            rows = read_jsonl(directory / "results.jsonl")
            if len(rows) != 28 or len({r["case_id"] for r in rows}) != 28:
                raise ValueError(f"expected 28 unique completed cases: {directory}")
            provenance[str(directory / "results.jsonl")] = digest(directory / "results.jsonl")
            for row in rows:
                visible = args.runs_root.parent / "cases" / "p5afdb" / row["case_id"] / "visible.json"
                duration = json.loads(visible.read_text())["dur_s"]
                cases.append(audit_case(directory, row, duration))
    matched = match_cases(cases)
    summary = {
        model: {arm: summarize([c for c in cases if c["model"] == model and c["arm"] == arm]) for arm in ("M", "R")}
        for model in sorted({c["model"] for c in cases})
    }
    identical = [
        {"case_id": p["case_id"], "model": p["model"], **query} for p in matched for query in p["identical_queries"]
    ]
    query_summary = {}
    for tool in sorted({q["tool"] for q in identical}):
        subset = [q for q in identical if q["tool"] == tool]
        query_summary[tool] = {
            "calls": len(subset),
            **{key: sum(q[key] for q in subset) for key in ("M_chars", "R_chars", "delta_chars")},
        }
    candidate_examples = []
    for pair in matched:
        if pair["case_id"] not in {"afdb000", "afdb006", "afdb026"}:
            continue
        m_case = next(
            c for c in cases if c["model"] == pair["model"] and c["case_id"] == pair["case_id"] and c["arm"] == "M"
        )
        r_case = next(
            c for c in cases if c["model"] == pair["model"] and c["case_id"] == pair["case_id"] and c["arm"] == "R"
        )
        m_call = next((c for c in m_case["calls"] if c["tool"] == "ts_find_events" and not c["is_error"]), None)
        r_call = next((c for c in r_case["calls"] if "ecg_af_episodes" in c["measurement_chars"]), None)
        if m_call and r_call:
            candidate_examples.append(
                {
                    "model": pair["model"],
                    "case_id": pair["case_id"],
                    "M_tool": m_call["tool"],
                    "M_arguments": m_call["arguments"],
                    "R_arguments": r_call["arguments"],
                    "M_chars": m_call["chars"],
                    "R_chars": r_call["chars"],
                    "R_episode_chars": r_call["measurement_chars"]["ecg_af_episodes"],
                    "R_episodes": r_call["episodes_or_events"],
                    "R_intervals": r_call["support_intervals"],
                    "M_following_requests": m_call["following_requests"],
                    "R_following_requests": r_call["following_requests"],
                    "M_result_step": m_call["result_step"],
                    "R_result_step": r_call["result_step"],
                }
            )
    result = {
        "schema_version": 1,
        "method": "offline paired transcript audit; exact text characters and recorded provider token usage",
        "per_tool_tokens": None,
        "per_tool_tokens_reason": "No model-matched local tokenizer; no API calls permitted. Characters are not token estimates.",
        "history_proxy": "sum(tool_text_chars * number_of_distinct_assistant_requests_after_result)",
        "actual_spend_usd": None,
        "actual_spend_reason": "No invoice/balance evidence; evaluator and SDK dollar accounting retained separately.",
        "input_provenance": provenance,
        "summary": summary,
        "matched_query_summary": query_summary,
        "candidate_route_examples": candidate_examples,
        "largest_matched_increases": sorted(
            [q for q in identical if q["delta_chars"] > 0], key=lambda q: q["delta_chars"], reverse=True
        )[:20],
        "matched_cases": matched,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    args.output.with_name("token-report.md").write_text(render_report(result))
    print(f"Audited {len(cases)} transcripts, {len(matched)} matched cases; wrote {args.output}")


if __name__ == "__main__":
    main()
