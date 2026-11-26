"""Offline protocol fixture for the T1 arms. Its scripted choices measure no language model capability."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import sha, write

from .episode import ARMS, run_episode
from .public import prepare
from .runner import baseline
from .scoring import report


class ScriptedTransport:
    mode = "fixture"

    def __init__(self, release_decision="accept"):
        self.release_decision = release_decision

    @contextmanager
    def session(self, folder, deadline, max_requests):
        yield self

    def response(self, system, messages, tools, request_hash):
        if (
            len(tools) == 1
            and tools[0]["name"] == "submit"
            and "items" in tools[0]["input_schema"].get("properties", {})
        ):
            from tsmcp.cleaning.dataset_review import PROMPTS

            stage = next(k for k, p in PROMPTS.items() if system.startswith(p.split(".")[0]))
            items = []
            for block in messages[0]["content"]:
                if block["type"] != "text":
                    continue
                row = json.loads(block["text"])
                if "id" not in row:
                    continue
                decision = dict(id=row["id"], reason="Scripted protocol fixture")
                if stage == "diagnosis":
                    diagnosis = dict(
                        label="artifact",
                        confidence=0.5,
                        evidence_refs=["view_1"],
                        alternative_hypotheses=["physiology"],
                    )
                    decision["diagnosis"] = diagnosis
                    if row["data"].get("trial_scope", {}).get("protected") and row["data"]["trial_scope"].get(
                        "eligible"
                    ):
                        decision["trial_diagnosis"] = diagnosis
                elif stage == "groups":
                    choices = tools[0]["input_schema"]["properties"]["items"]["items"]["properties"]["wave"]["enum"]
                    decision.update(
                        wave="interpolate_spike" if "interpolate_spike" in choices else "preserve",
                        wave_if=[],
                        measure_if=[],
                        measure="keep",
                        otherwise="review",
                        parameters={},
                    )
                elif stage == "gate":
                    decision["decision"] = self.release_decision
                elif stage == "joint":
                    decision.update(default="keep", exceptions=[])
                items.append(decision)
            return [dict(type="tool_use", id="fixture-" + request_hash[:12], name="submit", input={"items": items})]
        first = json.loads(messages[0]["content"][0]["text"])
        c = first["case"]
        names = {t["name"] for t in tools}
        turns = sum(m["role"] == "assistant" for m in messages)
        name = "submit_plan" if "submit_plan" in names else "run_candidate"
        arg = baseline(c, "interpolate")
        if "submit" in names and turns:
            links = [
                json.loads(b["content"][0]["text"])
                for m in messages
                if m["role"] == "user"
                for b in m["content"]
                if b["type"] == "tool_result"
            ]
            trial = links[0]
            if turns == 1:
                a = c["offset"] if not c["candidates"] else c["candidates"][0]["start_sample"]
                name = "inspect"
                arg = dict(
                    handle=trial["handle"],
                    start_sample=max(c["offset"], a - 12),
                    end_sample=min(c["offset"] + c["samples"], a + 18),
                    format="plot",
                )
            else:
                name = "submit"
                arg = dict(
                    submission_schema_version="tool-agent.proposed.v1",
                    submission_status="proposed",
                    **{k: c[k] for k in ("case_id", "profile_id", "input_hash")},
                    selected_candidate_handle=trial["handle"],
                    action_plan=None,
                    observed_tool_result_refs=[link["evidence_ref"] for link in links],
                    unresolved_ranges=[],
                    summary="Scripted protocol fixture selected its numerical trial.",
                )
        return [dict(type="tool_use", id=f"fixture-{turns + 1}", name=name, input=arg)]

    def accounting(self):
        return (
            dict(
                model_requests=0,
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                cached_input_tokens=0,
                usd_cost=0,
                unknown_fields=[],
            ),
            None,
            dict(transport="scripted_offline_fixture"),
        )


def smoke(root):
    root.mkdir(parents=True, exist_ok=False)
    organizer = root / "organizer"
    organizer.mkdir()
    cases = []
    for i in range(3):
        cid = f"C{i + 1:03d}"
        t = np.arange(2500) / 125
        phase = np.mod(t, 1)
        ref = np.exp(-(((phase - 0.14) / 0.025) ** 2)) + 0.18 * np.exp(-(((phase - 0.41) / 0.065) ** 2))
        x = ref.copy()
        mask = np.zeros(len(t), bool)
        if i < 2:
            a = 955 if i == 0 else 1017
            x[a : a + 2] += [6.0, -6.0]
            mask[a : a + 2] = True
        v = np.ones(len(t), bool)
        case = prepare(root / cid / "public", cid, "ECG", "synthetic units", 125, t, x, v, ~v)
        for arm in ARMS:
            run_episode(
                root / cid / "public", root / cid / arm, arm, ScriptedTransport(), episode_id="offline-protocol-1"
            )
        truth = organizer / (cid + ".npz")
        np.savez(truth, damaged=mask, reference=ref)
        cases.append(
            dict(
                case_id=cid,
                subject="synthetic-01",
                record="synthetic-base-01",
                noise_source=None if i == 2 else "synthetic-spike",
                split="development",
                input_hash=case["input_hash"],
                public_dir=f"../{cid}/public",
                private_npz=truth.name,
                private_sha256=sha(truth),
                outputs={a: f"../{cid}/{a}" for a in ARMS},
            )
        )
    path = organizer / "manifest.json"
    write(path, dict(cases=cases, arms=list(ARMS), arm_modes=dict.fromkeys(ARMS, "fixture")))
    result = report(path)
    write(organizer / "report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(smoke(p.parse_args().out)["summaries"], indent=2))
