"""Run the T1 one-shot planner or single tool agent on a frozen public case."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from jsonschema import ValidationError, validate

from tsmcp.cleaning.common import digest, read, sha, write

from .agent_tools import SCHEMAS, PublicTools, tool_specs
from .providers import DeepSeek, EpisodeLimit, Replay
from .runner import baseline

ARMS = ("generic_tool_agent", "one_shot_planner", "signalmind_full")


def text_block(obj):
    return dict(type="text", text=json.dumps(obj, sort_keys=True, allow_nan=False, separators=(",", ":")))


def final_json(content):
    """Accept one complete JSON proposal, including a single JSON code fence."""
    text = "".join(b.get("text", "") for b in content if b["type"] == "text").strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def initial_message(tools, budgets):
    c = tools.case
    manifest = {
        k: c[k]
        for k in (
            "case_id",
            "input_hash",
            "parent_version_id",
            "task",
            "unit",
            "fs",
            "offset",
            "samples",
            "profile_id",
            "catalog",
            "candidates",
            "public_read_ranges",
            "allowed_write_ranges",
            "track",
        )
    }
    manifest["purpose"] = {
        "ECG": "native heartbeats and RR",
        "PPG": "native pulses and PPI",
        "ABP": "native beat pressure",
    }[c["task"]]
    manifest["budget"] = budgets
    manifest["action_contract"] = dict(
        parameters="Use {} for linear and PCHIP. Their anchors and maximum core length are fixed.",
        interpolation_ranges="For write [a,b), include [a-k,b+k) in a declared read range, with k=anchors_per_side. All coordinates are global, end exclusive.",
        diagnosis_ids=[row["candidate_id"] for row in c["candidates"]],
        empty_candidate_rule="With zero detected candidates use diagnoses=[] and no waveform writes.",
        preserve_example=baseline(c, "identity"),
    )
    manifest["acquisition"] = c.get("acquisition")
    if "simulation_profile" in c:
        manifest["simulation_profile"] = c["simulation_profile"]
    ref = tools.evidence_ref("frozen_case", dict(input_hash=c["input_hash"]))
    blocks = [text_block(dict(case=manifest, evidence_ref=ref))]
    # Both arms see the same initial numbers and two scales of the same observed region.
    for row in c["candidates"]:
        a, b = row["start_sample"], row["end_sample"]
        left, right = c["offset"], c["offset"] + c["samples"]
        item, _ = tools.inspect(
            dict(handle="observed", start_sample=max(left, a - 3), end_sample=min(right, b + 3), format="samples")
        )
        blocks.append(text_block(dict(candidate_id=row["candidate_id"], measurements=item)))
    left, right = c["offset"], c["offset"] + c["samples"]
    a, b = (
        (c["candidates"][0][k] for k in ("start_sample", "end_sample"))
        if c["candidates"]
        else ((left + right) // 2, (left + right) // 2 + 1)
    )
    for seconds in (5, 0.12):
        margin = max(1, round(seconds * c["fs"]))
        item, image = tools.inspect(
            dict(
                handle="observed", start_sample=max(left, a - margin), end_sample=min(right, b + margin), format="plot"
            )
        )
        blocks.extend([text_block(item), image])
    return dict(role="user", content=blocks)


def generic_dialog(toolset, system, specs, messages, budgets, ask, steps):
    arm = toolset.arm
    turns, max_tools, deadline = budgets["max_turns"], budgets["max_tools"], toolset.deadline
    schemas = {s["name"]: s["input_schema"] for s in specs}
    calls = 0
    selected = None
    status, reason = "budget_exhausted", "turn_limit"
    for turn in range(turns):
        if time.monotonic() >= deadline:
            raise TimeoutError("episode deadline")
        last = turn == turns - 1 and arm == "generic_tool_agent"
        if last:
            messages[-1]["content"].append(
                text_block(
                    dict(instruction="Final turn. Submit one previously tried candidate or a complete final plan now.")
                )
            )
        content, _ = ask(system, messages, [s for s in specs if s["name"] == "submit"] if last else specs)
        messages.append(dict(role="assistant", content=content))
        answers = []
        uses = [block for block in content if block["type"] == "tool_use"]
        text_submission = False
        if not uses:
            obj = final_json(content)
            if obj is not None:
                uses = [
                    dict(
                        id=f"json-final-{turn}",
                        name="submit_plan" if arm == "one_shot_planner" else "submit",
                        input=obj,
                    )
                ]
                text_submission = True
                steps.append(dict(turn=turn + 1, submission_format="json_text"))
        if not uses:
            status, reason = "invalid_output", "no_tool_submission"
            if arm == "one_shot_planner":
                break
            messages.append(dict(role="user", content=[text_block({"required": "Use the submit tool to finish."})]))
            continue
        if len({u["id"] for u in uses}) != len(uses):
            status, reason = "invalid_output", "duplicate_tool_call_id"
            break
        if any(u["name"] in ("submit", "submit_plan") for u in uses) and len(uses) != 1:
            status, reason = "invalid_output", "final_submission_must_be_a_single_call"
            if arm == "one_shot_planner":
                break
            # Return an error for every tool call ID to keep the provider's conversation valid.
            if calls + len(uses) > max_tools:
                raise EpisodeLimit("tool_limit")
            calls += len(uses)
            toolset.tool_count = calls
            steps.append(dict(turn=turn + 1, error=reason))
            messages.append(
                dict(
                    role="user",
                    content=[
                        dict(type="tool_result", tool_use_id=u["id"], is_error=True, content=reason) for u in uses
                    ],
                )
            )
            continue
        for call in uses:
            if calls >= max_tools:
                raise EpisodeLimit("tool_limit")
            if time.monotonic() >= deadline:
                raise TimeoutError("episode deadline")
            calls += 1
            toolset.tool_count = calls
            name, arg = call["name"], call["input"]
            entry = dict(turn=turn + 1, call_id=call["id"], name=name, arguments_hash=digest(arg))
            image = None
            failed = False
            try:
                if name not in schemas:
                    raise ValueError("Tool unavailable in this arm")
                if last and name != "submit":
                    raise ValueError("The final tool turn is reserved for submission")
                validate(arg, schemas[name])
                if name == "inspect":
                    reply, image = toolset.inspect(arg)
                elif name == "run_candidate":
                    reply = toolset.trial(arg)
                elif name == "submit_plan":
                    reply = toolset.trial(arg)
                    if not toolset.trials[reply["handle"]]["eligible"]:
                        raise ValueError("Invalid one-shot plan: " + ", ".join(reply["reasons"]))
                    selected = toolset.trials[reply["handle"]]["folder"]
                    toolset.submission = arg
                else:
                    selected = toolset.submit(arg)
                    reply = dict(selected_handle=selected.name)
            except (ValueError, KeyError, TypeError, ValidationError) as exc:
                failed = True
                reply = dict(error=(exc.message if isinstance(exc, ValidationError) else str(exc))[:400])
                reason = reply["error"]
                status = "invalid_output"
            steps.append(dict(entry, result=reply, success=not failed))
            if text_submission:
                answers.append(text_block(reply))
                continue
            answers.append(
                dict(
                    type="tool_result",
                    tool_use_id=call["id"],
                    is_error=failed,
                    content=[text_block(reply)] + ([image] if image else []),
                )
            )
        if selected is not None:
            if time.monotonic() >= deadline:
                raise TimeoutError("episode deadline")
            break
        if arm == "one_shot_planner":
            break
        answers.append(
            text_block(
                dict(
                    remaining_turns=turns - turn - 1,
                    remaining_tools=max_tools - calls,
                    remaining_new_candidates=toolset.max_candidates - toolset.candidates_used,
                )
            )
        )
        messages.append(dict(role="user", content=answers))
        status, reason = "budget_exhausted", "turn_limit"
    return selected, status, reason


def run_episode(public, out, arm, provider, *, episode_id, max_turns=8, max_tools=16, max_candidates=2, seconds=180):
    if arm not in ARMS or min(max_turns, max_tools, seconds) < 1 or max_candidates < 0 or not episode_id:
        raise ValueError("Invalid repair episode settings")
    started = time.monotonic()
    deadline = started + seconds
    turns = 1 if arm == "one_shot_planner" else max_turns
    budgets = dict(max_turns=turns, max_tools=max_tools, max_candidates=max_candidates, wall_seconds=seconds)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "public-input"
    snapshot.mkdir()
    for name in ("case.json", "observed.npz"):
        shutil.copyfile(public / name, snapshot / name)
    toolset = PublicTools(snapshot, out, arm, episode_id, max_candidates, deadline)
    c = toolset.case
    if arm == "signalmind_full":
        from tsmcp.cleaning.dataset_review import PROMPTS
        from tsmcp.cleaning.dataset_review import SCHEMAS as PRODUCTION_SCHEMAS

        system, specs = PROMPTS, PRODUCTION_SCHEMAS
    else:
        system = (Path(__file__).with_name("prompts") / (arm + ".txt")).read_text()
        specs = tool_specs(arm)
    messages = [initial_message(toolset, budgets)]
    initial_hash = digest(messages)
    initial_images = [z["image_sha256"] for z in toolset.evidence.values() if "image_sha256" in z]
    details = {}
    nominated = None
    frames = []
    steps = []
    selected = None
    status = "budget_exhausted"
    reason = "turn_limit"

    def ask(prompt, history, catalog):
        if len(frames) >= turns:
            raise EpisodeLimit("provider_request_limit")
        if time.monotonic() >= deadline:
            raise TimeoutError("episode deadline")
        req = digest([prompt, catalog, history])
        blocks = provider.response(prompt, history, catalog, req)
        reqid = getattr(provider, "last_id", None) or "local-" + req[:24]
        frames.append(dict(request_hash=req, request_id=reqid, content=blocks))
        write(out / "transport.json", dict(frames=frames))
        return blocks, reqid

    try:
        with provider.session(out, deadline, turns):
            if arm == "signalmind_full":
                from .signalmind_arm import run_production

                selected, nominated, details = run_production(toolset, ask, messages[0], budgets)
            else:
                selected, status, reason = generic_dialog(toolset, system, specs, messages, budgets, ask, steps)
            if selected is not None:
                checked = read(selected / "result.json")
                for key, name in [
                    ("released_waveform", "values.npy"),
                    ("validity_mask", "task_valid.npy"),
                    ("provenance_mask", "estimated.npy"),
                    ("native_measurements", "native_measurements.json"),
                    ("trace", "trace.json"),
                ]:
                    if checked[key]["artifact_id"] != name or sha(selected / name) != checked[key]["sha256"]:
                        raise ValueError("Selected artifact changed since execution")
                if time.monotonic() >= deadline:
                    raise TimeoutError("episode deadline")
    except (TimeoutError, subprocess.TimeoutExpired):
        selected, status, reason = None, "timeout", "episode_deadline"
    except EpisodeLimit as exc:
        selected, status, reason = None, "budget_exhausted", str(exc)
    except Exception as exc:
        # Keep the fixed assignment and partial receipts even when the remote service or worker fails.
        selected, status, reason = None, "error", type(exc).__name__
    calls = toolset.tool_count
    usage, provider_receipt, account = provider.accounting()
    run_id = digest([episode_id, arm, provider.mode, c["input_hash"], digest(system), digest(specs), budgets, frames])
    result = dict(
        result_schema_version="proposal.v1",
        case_id=c["case_id"],
        arm_id=arm,
        run_id=run_id,
        input_hash=c["input_hash"],
        parent_version_id=c["parent_version_id"],
        status=status,
        decision="preserve",
        fallback_used=True,
        selected_candidate=None,
        released_waveform=None,
        validity_mask=None,
        provenance_mask=None,
        native_measurements=None,
        derived_measurements=None,
        reason_codes=[reason],
    )
    execution = {}
    if selected is not None:
        result = read(selected / "result.json")
        result["run_id"] = run_id
        if arm == "signalmind_full":
            result["selected_candidate"] = nominated
            derived = out / "production" / "export" / "derived-measurements.jsonl"
            shutil.copyfile(derived, out / derived.name)
            result["derived_measurements"] = dict(
                kind="derived_measurements", artifact_id=derived.name, sha256=sha(derived)
            )
        for file in ("values.npy", "task_valid.npy", "estimated.npy", "native_measurements.json"):
            shutil.copyfile(selected / file, out / file)
        execution = read(selected / "trace.json")
    trace = dict(
        **execution,
        **details,
        mode=provider.mode,
        arm=arm,
        episode_id=episode_id,
        prompt_sha256=digest(system),
        tool_specs_hash=digest(specs),
        initial_message_hash=initial_hash,
        initial_image_sha256=initial_images,
        budgets=budgets,
        turns=len(frames),
        steps=steps,
        trials=[v["reply"] for v in toolset.trials.values()],
        evidence=toolset.evidence,
        submission=toolset.submission,
        provider_accounting=account,
        selected_handle=(details.get("nominations") or [None])[-1]
        if arm == "signalmind_full"
        else selected.name
        if selected
        else None,
    )
    write(out / "trace.json", trace)
    result.update(
        trace=dict(kind="trace", artifact_id="trace.json", sha256=sha(out / "trace.json")),
        usage=dict(**usage, tool_calls=calls, wall_seconds=time.monotonic() - started),
        provider_receipt=provider_receipt,
    )
    validate(result, read(SCHEMAS / "method_result.schema.json"))
    write(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--episode-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--live-deepseek", action="store_true")
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--episode-cap-usd", type=float)
    parser.add_argument("--experiment-budget", type=Path, help="Share the additional cap across episodes and restarts")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tools", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    if args.live_deepseek and (not args.budget_root or args.episode_cap_usd is None):
        parser.error("Live mode requires the shared --budget-root and an explicit --episode-cap-usd")
    provider = (
        DeepSeek(args.budget_root, args.episode_cap_usd, experiment_budget=args.experiment_budget)
        if args.live_deepseek
        else Replay(args.replay)
    )
    print(
        run_episode(
            args.public,
            args.out,
            args.arm,
            provider,
            episode_id=args.episode_id,
            max_turns=args.max_turns,
            max_tools=args.max_tools,
            max_candidates=args.max_candidates,
            seconds=args.seconds,
        )["status"]
    )


if __name__ == "__main__":
    main()
