"""A bounded visual tool loop. The reviewer must reproduce the trigger with its proposed rule."""

import json
from pathlib import Path

import numpy as np
from jsonschema import ValidationError, validate

from tsmcp.cleaning.common import digest, sha, write
from tsmcp.followup.evidence import view
from tsmcp.followup.pattern import detect, pattern_schema

SYSTEM = """Review the waveform for the stated task. Report observations, hypotheses, and alternatives in Chinese.
Read both the overview and sample-level detail. Decide whether to widen the inspection area based on recurrence
and recovery. Use expand_context for 10/20/40 seconds per side. Estimate episode boundaries from the waveform.
For a recurring artifact, use try_detector to check a proposed method against the actual trigger and neighbours.
Compare the core hit, noisy tails, and ordinary beats. Revise if needed. Register the tested rule or use defer.
step=max adjacent absolute difference; median_residual=absolute difference from running median;
second_difference=absolute second difference. Seed and tail scores divide by local p95-p05 amplitude span.
scale_window_s sets the normalization window. quiet_s joins nearby tail hits. max_seed_s limits seed duration.
baseline_window_s applies only to median_residual. Author transferable rules without case identifiers or times.
Record uncertainty when the mechanism or suitable detector remains unclear."""

S = {"type": "string", "minLength": 1}
BOUNDS = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}


def tool(name, description, fields, required=None):
    return dict(
        name=name,
        description=description,
        input_schema=dict(
            type="object",
            properties=fields,
            required=list(fields) if required is None else required,
            additionalProperties=False,
        ),
    )


TOOLS = [
    tool(
        "expand_context",
        "Request a wider context and detail. Optional target_s selects an interval within the current image.",
        {"flank_s": {"type": "integer", "enum": [10, 20, 40]}, "reason": S, "target_s": BOUNDS},
        ["flank_s", "reason"],
    ),
    tool(
        "try_detector",
        "Compute the proposed rule on this block. Return anchor and visible matched ranges in seconds.",
        {"rule": pattern_schema()},
    ),
    tool(
        "register_detector",
        "Submit the diagnosis, estimated boundaries, and an executable reusable detection rule.",
        {
            "observed": S,
            "suspected_cause": S,
            "alternative_or_limitation": S,
            "suspected_range_s": BOUNDS,
            "method_explanation": S,
            "rule": pattern_schema(),
        },
    ),
    tool("defer", "Record insufficient evidence or an unsupported method.", {"reason": S}),
]


def bounds(seconds, fs, visible, seed):
    if not all(np.isfinite(seconds)):
        raise ValueError("Nonfinite target")
    a, b = [round(s * fs) for s in seconds]
    if not visible[0] <= a < b <= visible[1] or not (a < seed[1] and seed[0] < b):
        raise ValueError("Proposed bounds must overlap the seed and lie on the current image")
    return [a, b]


def review(client, out, case, x, usable, version, seed, initial_flank=5):
    out = Path(out)
    fs = case["fs"]
    target = list(seed)
    flank = initial_flank
    ev, picture = view(out / "views", case, x, usable, version, target, flank, seed=seed)

    def caption(ev):
        left, right = ev["view_samples"]
        lo, hi = np.percentile(x[left:right][usable[left:right]], [5, 95])
        z0, z1 = ev["detail_samples"]
        detail = x[z0:z1][usable[z0:z1]]
        return json.dumps(
            {
                "modality": case["task"],
                "fs": fs,
                "unit": case["unit"],
                "seed_s": [s / fs for s in seed],
                "visible_s": [left / fs, right / fs],
                "detail_min_max": [float(np.min(detail)), float(np.max(detail))],
                "detail_largest_adjacent_step": float(
                    np.max(np.abs(np.diff(x[z0:z1]))[usable[z0:z1][:-1] & usable[z0:z1][1:]], initial=0)
                ),
                "evidence_id": ev["id"],
                "local_p05": lo,
                "local_p95": hi,
            }
        )

    messages = [dict(role="user", content=[{"type": "text", "text": caption(ev)}, picture])]
    views = [ev]
    log = []
    trials = {}
    for turn in range(8):
        tools = TOOLS
        if (len(trials) >= 3 and any(t["seed_hits"] and t["view"] == ev["id"] for t in trials.values())) or turn == 7:
            tools = [t for t in TOOLS if t["name"] in {"register_detector", "defer"}]
            messages[-1]["content"].append(
                {
                    "type": "text",
                    "text": (
                        "Conclude now. Register your best previously tested exact rule and describe any missed tails, "
                        "or defer. The detector may target the distinctive core; "
                        "give the wider suspected episode separately."
                    ),
                }
            )
        response = client.request(messages, tools, system=SYSTEM)
        content = [b.model_dump(exclude_none=True) for b in response.content]
        messages.append(dict(role="assistant", content=content))
        calls = [c for c in content if c["type"] == "tool_use"]
        receipt = dict(request_id=response.id, view_ids=[v["id"] for v in views], calls=calls)
        log.append(receipt)
        write(out / "calls.json", log)
        replies = []
        for call in calls:
            data = call["input"]
            name = call["name"]
            try:
                if len(calls) != 1:
                    raise ValueError("Call a single tool, then read its result before the next decision")
                validate(data, next(t["input_schema"] for t in tools if t["name"] == name))
                if name == "expand_context":
                    if data["flank_s"] <= flank:
                        raise ValueError("Select a wider flank or conclude the review")
                    if "target_s" in data:
                        target = bounds(data["target_s"], fs, ev["view_samples"], seed)
                    flank = data["flank_s"]
                    ev, picture = view(out / "views", case, x, usable, version, target, flank, seed=seed)
                    views.append(ev)
                    answer = [{"type": "text", "text": caption(ev)}, picture]
                elif name == "try_detector":
                    matches = detect(x, usable, fs, data["rule"])
                    hits = [r for r in matches if r[0] < seed[1] and seed[0] < r[1]]
                    visible = [r for r in matches if r[0] < ev["view_samples"][1] and ev["view_samples"][0] < r[1]]
                    key = digest(data["rule"])
                    trials[key] = dict(
                        rule=data["rule"],
                        seed_hits=hits,
                        view=ev["id"],
                        block_hits=len(matches),
                        visible_count=len(visible),
                        visible_first_20_s=[[a / fs, b / fs] for a, b in visible[:20]],
                    )
                    write(out / "trials.json", trials)
                    answer = json.dumps(trials[key])
                elif name == "defer":
                    return dict(status="unresolved", reason=data["reason"], views=views, request_id=response.id)
                else:
                    suspected = bounds(data["suspected_range_s"], fs, ev["view_samples"], seed)
                    trial = trials.get(digest(data["rule"]))
                    if not trial or trial["view"] != ev["id"]:
                        raise ValueError("Try this exact rule on the current view before registering")
                    hits = trial["seed_hits"]
                    if not hits:
                        raise ValueError("Rule did not detect the index trigger; revise parameters or defer")
                    return dict(
                        status="registered",
                        **data,
                        task=case["task"],
                        seed_case=case["id"],
                        seed_samples=list(seed),
                        suspected_samples=suspected,
                        seed_method_hits=hits,
                        seed_version=version,
                        views=views,
                        request_id=response.id,
                        model=client.model,
                        rule_id=digest([case["task"], data["rule"], sha(Path(__file__).with_name("pattern.py"))]),
                    )
                replies.append(dict(type="tool_result", tool_use_id=call["id"], content=answer))
            except (ValueError, StopIteration, ValidationError) as exc:
                replies.append(dict(type="tool_result", tool_use_id=call["id"], content=str(exc)[:1000], is_error=True))
        messages.append(
            dict(role="user", content=replies or [{"type": "text", "text": "Choose a provided tool to continue."}])
        )
    return dict(status="unresolved", reason="review_turn_limit", views=views)
