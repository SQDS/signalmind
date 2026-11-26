"""Public evidence, same-parent trials and case-bound final submissions for repair-only methods."""

import base64
import copy
import time

import numpy as np
from jsonschema import validate
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp import processing
from tsmcp.cleaning.common import array_hash, digest, read, sha, write

from .executor import SCHEMAS, spans
from .public import CATALOG
from .runner import isolated


def inline_schema(filename):
    """Resolve the two local kit schemas before handing the tool to a provider."""
    doc = read(SCHEMAS / filename)

    def expand(obj):
        if isinstance(obj, list):
            return [expand(v) for v in obj]
        if not isinstance(obj, dict):
            return obj
        ref = obj.get("$ref")
        if ref == "action_plan.schema.json":
            return inline_schema(ref)
        if ref:
            return expand(doc["$defs"][ref.removeprefix("#/$defs/")])
        return {k: expand(v) for k, v in obj.items() if k not in {"$defs", "$schema", "$id", "$comment"}}

    return expand(doc)


PLAN = inline_schema("action_plan.schema.json")
SUBMISSION = inline_schema("tool_agent_submission.schema.json")
INSPECT = dict(
    type="object",
    additionalProperties=False,
    required=["handle", "start_sample", "end_sample", "format"],
    properties=dict(
        handle={"type": "string"},
        start_sample={"type": "integer"},
        end_sample={"type": "integer"},
        format={"enum": ["samples", "plot"]},
    ),
)


def tool_specs(arm):
    if arm == "one_shot_planner":
        return [dict(name="submit_plan", description="Submit the one permitted ActionPlan.", input_schema=PLAN)]
    return [
        dict(
            name="inspect",
            description="Read observed or candidate samples, masks, or a waveform plot. Use global half-open coordinates. The parent handle is observed.",
            input_schema=INSPECT,
        ),
        dict(
            name="run_candidate",
            description="Execute a plan against the frozen parent and return a candidate handle and numerical checks.",
            input_schema=PLAN,
        ),
        dict(
            name="submit",
            description="Choose a previously observed candidate handle or submit a final executable plan.",
            input_schema=SUBMISSION,
        ),
    ]


class PublicTools:
    def __init__(self, public, folder, arm, episode, max_candidates, deadline):
        self.public, self.folder, self.arm = public, folder, arm
        self.deadline = deadline
        self.case = read(public / "case.json")
        c = self.case
        if (
            digest({k: v for k, v in c.items() if k != "input_hash"}) != c["input_hash"]
            or c["catalog"] != CATALOG
            or c["public_tool_catalog_hash"] != digest([CATALOG, sha(processing.__file__)])
        ):
            raise ValueError("Frozen public case or operator catalog changed")
        with np.load(public / "observed.npz", allow_pickle=False) as z:
            self.t, self.x, self.valid, self.estimated = (z[k] for k in ("t", "x", "valid", "estimated"))
        if array_hash(self.t, self.x, self.valid, self.estimated) != c["observed_hash"]:
            raise ValueError("Frozen observed arrays changed")
        self.owner = digest([episode, arm, c["input_hash"], c["parent_version_id"]])
        self.max_candidates = max_candidates
        self.candidates_used = 0
        self.tool_count = 0
        self.trials = {}
        self.evidence = {}
        self.submission = None

    def evidence_ref(self, kind, payload):
        ref = digest([self.owner, kind, payload])
        self.evidence[ref] = dict(kind=kind, **payload)
        return ref

    def check_refs(self, plan):
        refs = {ref for d in plan["diagnoses"] for ref in d["evidence_refs"]}
        if not refs <= self.evidence.keys():
            raise ValueError("Diagnosis cites evidence outside this episode")

    def inspect(self, args):
        handle = args["handle"]
        x, v, e = self.x, self.valid, self.estimated
        if handle != "observed":
            if handle not in self.trials or not self.trials[handle]["eligible"]:
                raise ValueError("Unknown candidate handle")
            path = self.trials[handle]["folder"]
            x, v, e = (
                np.load(path / name, allow_pickle=False, mmap_mode="r")
                for name in ("values.npy", "task_valid.npy", "estimated.npy")
            )
        reg = dict(channel_id=self.case["task"], start_sample=args["start_sample"], end_sample=args["end_sample"])
        ((a, b),) = spans(self.case, [reg], self.case["public_read_ranges"])
        payload = dict(handle=handle, region=reg, format=args["format"], unit=self.case["unit"], fs_hz=self.case["fs"])
        if args["format"] == "samples":
            if b - a > 256:
                raise ValueError("Request at most 256 exact samples or use a plot")
            payload.update(
                values=[float(value) if np.isfinite(value) else None for value in x[a:b]],
                time_s=self.t[a:b].tolist(),
                valid=v[a:b].tolist(),
                estimated=e[a:b].tolist(),
            )
            return dict(payload, evidence_ref=self.evidence_ref("inspection", payload)), None
        fig = Figure(figsize=(8, 2.6), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.plot(self.t[a:b], self.x[a:b], color="#606d7c", lw=0.9, label="Observed")
        if handle != "observed":
            ax.plot(self.t[a:b], x[a:b], color="#c6561e", lw=1, label="Candidate")
            ax.plot(self.t[a:b][e[a:b]], x[a:b][e[a:b]], ".", color="#c6561e", label="Estimated")
            ax.legend()
        visible = []
        for r in self.case["allowed_write_ranges"]:
            lo = max(a, r["start_sample"] - self.case["offset"])
            hi = min(b, r["end_sample"] - self.case["offset"])
            if lo < hi:
                visible.append([lo + self.case["offset"], hi + self.case["offset"]])
                ax.axvspan(
                    self.t[lo] - 0.5 / self.case["fs"],
                    self.t[hi - 1] + 0.5 / self.case["fs"],
                    color="#e8a01e",
                    alpha=0.24,
                )
        ax.set(xlabel="Recorded time (s)", ylabel=self.case["unit"], title=self.case["task"])
        path = self.folder / "views" / (digest([self.owner, args]) + ".png")
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=100)
        payload.update(
            image_sha256=sha(path),
            visible_write_ranges=visible,
            legend="Amber marks detector-authorized write ranges"
            if visible
            else "No authorized write range in this view",
        )
        ref = self.evidence_ref("inspection", payload)
        return dict(payload, evidence_ref=ref), dict(
            type="image",
            source=dict(
                type="base64", media_type="image/png", data=base64.b64encode(path.read_bytes()).decode("ascii")
            ),
        )

    def trial(self, plan):
        validate(plan, PLAN)
        self.check_refs(plan)
        handle = digest([self.owner, "trial", plan])
        if handle not in self.trials:
            numerical_or_mask = any(z["kind"] != "preserve" for z in plan["actions"])
            if numerical_or_mask and self.candidates_used >= self.max_candidates:
                raise ValueError("Candidate budget reached; select an existing handle or preserve")
            folder = self.folder / "trials" / handle
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("episode deadline")
            result = isolated(self.public, plan, folder, self.arm, timeout=min(120, remaining))
            write(folder / "plan.json", plan)
            eligible = result["status"] in ("completed", "abstained")
            self.candidates_used += int(numerical_or_mask and eligible)
            self.trials[handle] = dict(folder=folder, eligible=eligible, plan=copy.deepcopy(plan))
            trace = read(folder / "trace.json")
            native = read(folder / "native_measurements.json")
            evidence = dict(
                handle=handle,
                inspectable=eligible,
                plan_hash=digest(plan),
                status=result["status"],
                decision=result["decision"],
                reasons=result["reason_codes"],
                modified_global_ranges=trace["modified_global_ranges"],
                native_events=len(native["events"]),
                native_support_samples=native["measured_samples"],
                waveform_sha256=result["released_waveform"]["sha256"],
            )
            self.trials[handle]["reply"] = dict(evidence, evidence_ref=self.evidence_ref("trial", evidence))
        return self.trials[handle]["reply"]

    def submit(self, obj):
        validate(obj, SUBMISSION)
        if any(obj[k] != self.case[k] for k in ("case_id", "input_hash", "profile_id")):
            raise ValueError("Stale submission binding")
        if not set(obj["observed_tool_result_refs"]) <= self.evidence.keys():
            raise ValueError("Submission cites unknown tool evidence")
        spans(self.case, obj["unresolved_ranges"], self.case["public_read_ranges"])
        handle = obj["selected_candidate_handle"]
        if handle is None:
            handle = self.trial(obj["action_plan"])["handle"]
        elif (
            handle in self.trials
            and self.trials[handle]["reply"]["evidence_ref"] not in obj["observed_tool_result_refs"]
        ):
            raise ValueError("Cite the selected candidate trial receipt")
        if handle not in self.trials or not self.trials[handle]["eligible"]:
            raise ValueError("Selected handle is unavailable or failed execution")
        self.submission = copy.deepcopy(obj)
        return self.trials[handle]["folder"]
