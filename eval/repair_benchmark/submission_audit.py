"""Re-execute recorded proposals offline. A changed tool reply cannot establish the model's next decision."""

import argparse
from pathlib import Path

from jsonschema import ValidationError, validate

from tsmcp.cleaning.common import read, sha, write

from .agent_tools import PLAN
from .episode import final_json
from .runner import isolated


def audit(run, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    manifest = run / "organizer/manifest.json"
    for c in read(manifest)["cases"]:
        public = run / c["case_id"] / "public"
        for arm in ("generic_tool_agent", "one_shot_planner"):
            episode = public.parent / arm
            trace = read(episode / "trace.json")
            emitted = {s.get("result", {}).get("evidence_ref") for s in trace["steps"]}
            visible = set(trace["evidence"]) - emitted
            for turn, frame in enumerate(read(episode / "transport.json")["frames"], 1):
                uses = [b for b in frame["content"] if b["type"] == "tool_use"]
                if not uses and arm == "one_shot_planner":
                    uses = [dict(id="final_text", name="submit_plan", input=final_json(frame["content"]))]
                prior = {s["call_id"]: s for s in trace["steps"] if s["turn"] == turn and "call_id" in s}
                for call in uses:
                    name = call["name"]
                    if name not in ("run_candidate", "submit_plan"):
                        continue
                    original = prior.get(call["id"], {}).get("result", {})
                    row = dict(
                        case_id=c["case_id"],
                        arm=arm,
                        turn=turn,
                        call_id=call["id"],
                        name=name,
                        original_status=original.get("status") or "invalid_output",
                    )
                    arg = call["input"]
                    try:
                        if arg is None:
                            raise ValueError("No complete parseable JSON submission")
                        validate(arg, PLAN)
                        if any(ref not in visible for d in arg["diagnoses"] for ref in d["evidence_refs"]):
                            raise ValueError("Unknown diagnosis evidence reference")
                        dest = out / f"trial-{len(rows):03}"
                        result = isolated(public, arg, dest, arm)
                        row.update(
                            offline_status=result["status"],
                            decision=result["decision"],
                            reason_codes=result["reason_codes"],
                            artifact_sha256=result["released_waveform"]["sha256"],
                        )
                    except (ValueError, ValidationError) as exc:
                        row.update(
                            offline_status="invalid_output",
                            reason_codes=[(exc.message if isinstance(exc, ValidationError) else str(exc))[:240]],
                        )
                    rows.append(row)
                visible.update(
                    s["result"]["evidence_ref"] for s in prior.values() if s.get("result", {}).get("evidence_ref")
                )
    result = dict(mode="offline_recorded_proposals", model_requests=0, live_manifest_sha256=sha(manifest), rows=rows)
    write(out / "audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(len(audit(args.run, args.out)["rows"]))
