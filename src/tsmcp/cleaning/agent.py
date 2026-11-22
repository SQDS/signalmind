"""One persistent visual DS session; shared budget, actual tool calls and no host repair."""

from __future__ import annotations

import base64
import json
import secrets
import time
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .common import digest, implementation_hash, read, sha, write
from .controller import CleaningTools
from .provenance import StopRequested, check_stop, emit, freeze_implementation

SYSTEM = """You are the operational SignalMind waveform cleaning Agent, responsible for the complete supplied
dataset. Three independent tasks: ECG waveform for native QRS/interval extraction, PPG waveform for native
pulse/PPI extraction, ABP waveform for native beat SBP/DBP/time-integrated mean pressure. This is NOT the
old R-marker editing benchmark. Only actual candidate waveforms are processed by frozen probes. No gold
references are available; never invent true accuracy. Raw values/time/units must be preserved outside edits.
Use observable conditions, not hard-coded case or patient identifiers. Recipes must be reusable.
For each ctx.apply supply explanation with the observed issue, rule/threshold, and reason for this operation.
The explanation is your stated basis, not independently verified artifact truth. The executor saves actual
operator snapshots and computes retest statistics independently.

Use inspect_dataset and list_capabilities. Diagnose groups, then for EACH task create or select a reusable
recipe. New waveform numerical processing must use ctx.apply SignalMind operators. Python may choose ranges,
conditions and parameters. Do not import another DSP implementation to bypass the registered operators.
Identity is valid when justified; do not filter or mask merely to show action. Large steps/low SQI are not
artifact truth. Negative ECG morphology may be real. ABP absolute pressure and slow trends matter. Missing
samples, estimated repairs and genuine discontinuities cannot support trusted derived intervals/pressure.

The sequence for each task is run_candidate(stage=pilot), get_report, inspect_cases, submit_self_review.
You must view candidate AFTER processing. To accept, cite real evidence_ids and describe observed preservation
and problems. Then run the SAME recipe_id with stage=batch to cover every continuous run for this task.
Every successful candidate needs explicit SelfReview before another candidate of that task; use revise/rollback
for rejected candidates. Failed executions without candidate output can be corrected from their error report.
Read its full report, call inspect_cases WITHOUT case_id for the required batch sample categories, and submit
batch self-review with ALL returned evidence_ids. You may request additional zooms, revise, reject or rollback.
A good pilot alone is not full-data delivery. If a candidate fails, change the method/parameters using actual
feedback. No developer silently repairs your code or picks a version. Plain text does not finish the task.
To refine an accepted batch further, set parent_version to that task's accepted batch in BOTH pilot and batch.

When inspecting an interval use its boundaries with 5 or 10 seconds of context on EACH side; if baseline,
recurrence, or recovery is unclear, expand using inspect_cases(target_samples, flank_s=10/20/40, case_id=...).
For a recurring defect, document signal evidence, tentative cause, alternatives and a reusable detection method.
Use register_pattern on the inspected raw or candidate signal BEFORE accepting its batch. Compare the rule to the
index trigger, inspect residual and previously unflagged matches after the automatic batch rescan. If none of the
supported methods describes it, record the unresolved method need in the review.
Complete every enabled task through batch review, then call finalize_run. Compare available unlabelled diagnostics
and preserve normal controls. Budget is shared across tasks and resume; do not repeatedly try identical
candidates. Maximum four new recipes per task (two initially, then at most two evidence-based revisions).
This is a grouped conversation, never one LLM session per window. Model self-review is not independent truth.
"""


class BudgetClient:
    compact_tools = False
    # Official peak rates verified 2026-11-12. Billing remains separately unknown.
    RESERVE = 0.35  # 1M input tokens + 16,384 output tokens, conservative peak-price ceiling.

    def __init__(self, workspace, cap=5.0, model="deepseek-flash", *, budget_root=None, compact_tools=False):
        import anthropic

        self.workspace, self.cap, self.model = Path(workspace), cap, model
        self.compact_tools = compact_tools
        if model != "deepseek-flash":
            raise ValueError("This validated visual runner is configured for deepseek-flash")
        key_path = Path.home() / ".config/tsmcp-eval/deepseek.key"
        if key_path.stat().st_mode & 0o077:
            raise ValueError("Credential file must have mode 0600")
        self.client = anthropic.Anthropic(
            api_key=key_path.read_text().strip(),
            base_url="https://api.deepseek.com/anthropic",
            max_retries=0,
            timeout=300,
        )
        self.path = (self.workspace if budget_root is None else Path(budget_root)) / "api_budget.json"
        self.ledger = (
            read(self.path)
            if self.path.exists()
            else {
                "cap_usd": cap,
                "model": model,
                "estimated_peak_usd": 0.0,
                "unknown_charge_reserve_usd": 0.0,
                "calls": [],
                "actual_billed_usd": None,
                "price_source": "https://api-docs.deepseek.com/quick_start/pricing",
                "price_checked": "2026-11-12",
            }
        )
        if self.ledger["cap_usd"] != cap or self.ledger["model"] != model:
            raise ValueError("Resume must retain the original model and total budget")

    def request(self, messages, tools=None, system=SYSTEM):
        for attempt in range(2):
            check_stop(self.workspace)
            committed = self.ledger["estimated_peak_usd"] + self.ledger["unknown_charge_reserve_usd"]
            if committed + self.RESERVE > self.cap:
                raise RuntimeError("api_budget_exhausted")
            call = {
                "number": len(self.ledger["calls"]) + 1,
                "status": "reserved",
                "reserve_usd": self.RESERVE,
                "request_hash": digest(messages),
                "attempt": attempt,
            }
            self.ledger["unknown_charge_reserve_usd"] += self.RESERVE
            self.ledger["calls"].append(call)
            write(self.path, self.ledger)
            emit(
                self.workspace,
                "model_call_started",
                actor="model",
                call_number=call["number"],
                attempt=attempt,
                model=self.model,
            )
            started = time.monotonic()
            try:
                kwargs = {"model": self.model, "system": system, "messages": messages, "max_tokens": 16384}
                if self.compact_tools:
                    kwargs.update(max_tokens=4096, thinking={"type": "disabled"})
                if tools:
                    kwargs["tools"] = tools
                response = self.client.messages.create(**kwargs)
            except Exception as exc:
                call.update(
                    status="transport_failed", error_type=type(exc).__name__, elapsed_s=time.monotonic() - started
                )
                write(self.path, self.ledger)
                emit(
                    self.workspace,
                    "model_call_failed",
                    actor="model",
                    call_number=call["number"],
                    error_type=type(exc).__name__,
                    elapsed_s=call["elapsed_s"],
                )
                if attempt:
                    raise
                continue
            usage = response.usage.model_dump()
            cost = (
                (usage.get("input_tokens", 0) or 0) * 0.3
                + (usage.get("output_tokens", 0) or 0) * 1.2
                + (usage.get("cache_creation_input_tokens", 0) or 0) * 0.3
                + (usage.get("cache_read_input_tokens", 0) or 0) * 0.006
            ) / 1e6
            self.ledger["unknown_charge_reserve_usd"] -= self.RESERVE
            self.ledger["estimated_peak_usd"] += cost
            call.update(
                status="returned",
                request_id=response.id,
                usage=usage,
                estimated_peak_usd=cost,
                elapsed_s=time.monotonic() - started,
            )
            write(self.path, self.ledger)
            emit(
                self.workspace,
                "model_call_completed",
                actor="model",
                request_id=response.id,
                call_number=call["number"],
                usage=usage,
                cost_usd=cost,
                elapsed_s=call["elapsed_s"],
            )
            return response

    def close(self):
        self.client.close()


def vision_preflight(client, *, workspace=None):
    path = (client.workspace if workspace is None else Path(workspace)) / "private/vision-preflight.json"
    if path.exists() and read(path).get("passed"):
        return read(path)
    code = secrets.token_hex(3).upper()
    fig = Figure(figsize=(4, 2), facecolor="white")
    FigureCanvasAgg(fig)
    fig.text(0.5, 0.5, code, ha="center", va="center", fontsize=36, color="black")
    picture = path.with_suffix(".png")
    fig.savefig(picture, dpi=100)
    content = [
        {"type": "text", "text": "Return only the six-character hexadecimal code visible in this image."},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(picture.read_bytes()).decode(),
            },
        },
    ]
    response = client.request([{"role": "user", "content": content}], system="Read the supplied image accurately.")
    answer = " ".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    result = {
        "passed": code in answer.upper(),
        "image_sha256": sha(picture),
        "request_id": response.id,
        "expected": code,
        "answer": answer,
    }
    write(path, result)
    if not result["passed"]:
        raise RuntimeError("Visual input preflight did not pass")
    return result


def run(workspace, *, cap=5.0, max_turns=100, model="deepseek-flash", budget_root=None):
    workspace = Path(workspace).resolve()
    private = workspace / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    toolset = CleaningTools(workspace)
    fingerprint = implementation_hash()
    snapshot = workspace / "agent-provenance.json"
    if snapshot.exists() and read(snapshot)["implementation_hash"] != fingerprint:
        raise ValueError("Agent implementation changed; preserve this run and use a new workspace")
    freeze_implementation(workspace)
    write(
        snapshot,
        {
            "implementation_hash": fingerprint,
            "system_hash": digest(SYSTEM),
            "model": model,
            "cap_usd": cap,
            "scope_hash": toolset.manifest["scope_hash"],
            "reference_access": False,
        },
    )
    client = BudgetClient(workspace, cap, model, **({"budget_root": budget_root} if budget_root is not None else {}))
    messages_path = private / "messages.json"
    messages = (
        read(messages_path)
        if messages_path.exists()
        else [
            {
                "role": "user",
                "content": (
                    f"自主处理当前范围：{toolset.manifest['patients']}名已知患者，"
                    f"{len(toolset.cases)}个连续任务片段，任务为"
                    f"{sorted({c['task'] for c in toolset.cases.values()})}；调用SignalMind算子，"
                    "真实看图自检、调整、全量运行并交付。没有人工标注，准确率不得编造。"
                ),
            }
        ]
    )
    log_path = workspace / "agent_activity.jsonl"
    turn = sum(1 for m in messages if m["role"] == "assistant")
    stop, error = "turn_limit", None
    started = time.monotonic()
    try:
        vision_preflight(client)
        while turn < max_turns and not toolset.state.get("finalized", toolset.state["complete"]):
            check_stop(workspace)
            if implementation_hash() != fingerprint:
                raise RuntimeError("Implementation changed during live run")
            if messages[-1]["role"] == "assistant":
                content = messages[-1]["content"]
                request_id = read(private / "pending.json")["request_id"]
            else:
                response = client.request(messages, toolset.schemas)
                request_id = response.id
                content = [b.model_dump(exclude_none=True) for b in response.content]
                messages.append({"role": "assistant", "content": content})
                turn += 1
                write(private / "pending.json", {"request_id": request_id, "turn": turn})
                write(messages_path, messages)
                for block in content:
                    if block["type"] == "tool_use":
                        emit(
                            workspace,
                            "tool_proposed",
                            actor="model",
                            request_id=request_id,
                            turn=turn,
                            tool_use_id=block["id"],
                            tool=block["name"],
                            arguments=block["input"],
                        )
                    elif block["type"] == "text":
                        emit(
                            workspace,
                            "model_summary",
                            actor="model",
                            request_id=request_id,
                            turn=turn,
                            text=block["text"],
                        )
            calls = [b for b in content if b["type"] == "tool_use"]
            replies = []
            for call in calls:
                check_stop(workspace)
                cached = private / "tool_results" / (digest(call["id"]) + ".json")
                if cached.exists():
                    emit(workspace, "tool_cache_hit", tool_use_id=call["id"], tool=call["name"])
                    replies.append(read(cached))
                    continue
                try:
                    emit(
                        workspace,
                        "tool_started",
                        actor="model",
                        request_id=request_id,
                        tool_use_id=call["id"],
                        tool=call["name"],
                        arguments=call["input"],
                        turn=turn,
                    )
                    blocks = toolset.call(call["name"], call["input"])
                    is_error = False
                except StopRequested:
                    raise
                except (ValueError, KeyError, TypeError, OSError, RuntimeError, SyntaxError) as exc:
                    blocks = [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}]
                    is_error = True
                public_blocks = [
                    {"type": "image", "sha256": digest(b["source"]["data"])} if b["type"] == "image" else b
                    for b in blocks
                ]
                event = {
                    "turn": turn,
                    "origin": "model_tool_call",
                    "request_id": request_id,
                    "tool_use_id": call["id"],
                    "tool": call["name"],
                    "arguments": call["input"],
                    "is_error": is_error,
                    "result": public_blocks,
                }
                with log_path.open("a") as log:
                    log.write(json.dumps(event, ensure_ascii=False) + "\n")
                emit(workspace, "tool_failed" if is_error else "tool_completed", actor="model", **event)
                reply = {"type": "tool_result", "tool_use_id": call["id"], "content": blocks, "is_error": is_error}
                write(cached, reply)
                replies.append(reply)
                print(
                    json.dumps(
                        {
                            "turn": turn,
                            "tool": call["name"],
                            "error": is_error,
                            "cost_usd": client.ledger["estimated_peak_usd"],
                        }
                    ),
                    flush=True,
                )
            messages.append(
                {
                    "role": "user",
                    "content": replies
                    or "Continue using tools; deliver all enabled tasks through batch review and finalize_run.",
                }
            )
            write(messages_path, messages)
        if toolset.state["complete"]:
            stop = "completed"
        elif toolset.state.get("finalized"):
            stop = "finalized_partial"
    except StopRequested as exc:
        error, stop = str(exc), "stopped"
        emit(workspace, "agent_stopped", reason=error, resumable=True)
    except Exception as exc:
        error, stop = f"{type(exc).__name__}: {exc}", "interrupted"
        if "api_budget_exhausted" in str(exc):
            stop = "budget_exhausted"
    finally:
        if (
            stop not in {"stopped", "interrupted"}
            and not toolset.state["complete"]
            and not toolset.state.get("finalized")
        ):
            toolset.finalize_run(
                f"Controller partial export after {stop}: {error}", decision_owner="controller_partial_export"
            )
        elif stop in {"stopped", "interrupted"} and not toolset.state.get("finalized"):
            # A checkpoint export must not finalize or destroy the pending tool call.
            from .common import read as read_json

            saved_state = (
                read_json(workspace / "agent_state.json")
                if (workspace / "agent_state.json").exists()
                else dict(toolset.state)
            )
            toolset.finalize_run("User stop: partial checkpoint", decision_owner="controller_partial_export")
            write(workspace / "agent_state.json", saved_state)
        client.close()
        result = {
            "stop": stop,
            "error": error,
            "model_turns": turn,
            "elapsed_s": time.monotonic() - started,
            "estimated_peak_usd": client.ledger["estimated_peak_usd"],
            "budget": client.ledger,
            "workflow_complete": toolset.state["complete"],
            "delivery": toolset.state.get("delivery"),
        }
        write(workspace / "agent-session.json", result)
    return result
