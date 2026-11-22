"""Dataset-level tools: learn a reusable policy on development examples, then freeze it."""

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from eval.dataset_workflow import FS, read, sha, trial, write
from tsmcp.pipeline import SCHEMA

PLOT_LOCK = threading.Lock()
POLICY_GUIDE = {
    "schema_version": SCHEMA,
    "structure": {
        "goal": "string",
        "max_gap_ms": 16,
        "max_repair_ms": 80,
        "channels": {"ECG": [], "PPG": [], "ABP": []},
    },
    "rule": {
        "id": "unique rule name",
        "operation": "one operation",
        "selector": {"kind": "one selector"},
        "when": {"feature_name": {"min": "optional number", "max": "optional number"}},
        "parameters": {},
        "fallback": "preserve or mark_invalid",
    },
    "selectors": {
        "all": "Entire continuous run; filters operate separately on finite valid spans.",
        "missing": "Original nonfinite runs.",
        "outliers": "Per window_s (default10s), flag values outside p5-factor*(p95-p5), p95+factor*(p95-p5); factor default1.5. Optional pad_samples default0. This is a candidate detector, not truth.",
        "flat": "Exactly equal finite consecutive values for min_duration_s (default1s). Optional pad_samples default0.",
    },
    "operations": {
        "interpolate": "Short linear interpolation between valid anchors; no parameters. Respects policy max_gap_ms and max_repair_ms. Long/edge intervals use explicit fallback.",
        "mark_invalid": "Keep observation, mark unusable; no parameters.",
        "notch": "parameters: frequency_hz, quality_factor(default30).",
        "remove_tone": "parameters: frequency_hz, optional window_s for overlapping local sine fits. Preserve fitted DC.",
        "lowpass": "parameters: frequency_hz=cutoff, filter_order(default4).",
    },
    "features": "finite_fraction, flat_fraction, robust_span=p95-p5, step_ratio=p99.9 absolute adjacent step/robust_span, mean, line_50_fraction, line_60_fraction, line_50_ratio, line_60_ratio. Mean/span retain source units. ECG/PPG are source_units with no confirmed mV or cross-record calibration. Line fractions use ±1Hz relative to AC power; ratios compare peak to median power 2–6Hz away. when bounds are ANDed and use current valid observations before each rule.",
    "execution": "Same rules apply to every run; no sample ids, source-specific parameters or LLM calls in batch execution. Empty rules explicitly preserve a channel. Missing channel definitions are errors.",
    "limits": "Prototype operators and selectors only. No arbitrary code generation, clinical ground truth, device saturation flags, cross-channel reconstruction, or long-gap reconstruction. Large amplitude/span alone does not establish saturation or a bad full run. Whole-run filter selection does not yet provide local time-varying noise classification.",
}


def tool(name, description, properties, required=()):
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": list(required)},
    }


class DatasetTools:
    def __init__(self, project):
        self.project = Path(project)
        self.samples = {s["id"]: s for s in read(self.project / "private/samples.json") if s["role"] == "development"}
        self.trials = {}
        self.schemas = [
            tool(
                "dataset_overview",
                "Read whole-dataset inventory, development-example diagnostics and executable policy contract. Validation examples are not exposed.",
                {},
            ),
            tool(
                "dataset_inspect",
                "View one development channel; optionally compare an actual trial output. Local time starts at zero; output invalid regions remain visible and shaded.",
                {
                    "sample_id": {"type": "string"},
                    "channel": {"type": "string", "enum": ["ECG", "PPG", "ABP"]},
                    "trial_id": {"type": "string"},
                },
                ("sample_id", "channel"),
            ),
            tool(
                "pipeline_trial",
                "Execute a reusable policy on every development example locally. Returns diagnostics and trial id; no clinical quality score. Revise general rules, not per-example exceptions.",
                {"policy": {"type": "object"}},
                ("policy",),
            ),
            tool(
                "pipeline_finalize",
                "Freeze a previously tried policy for dataset-wide reuse. This ends tuning; independent validation happens after the session. Does not launch full-dataset processing.",
                {"trial_id": {"type": "string"}, "rationale": {"type": "string"}},
                ("trial_id", "rationale"),
            ),
        ]

    async def call(self, name, args):
        try:
            if name == "dataset_overview":
                result = {
                    "inventory": read(self.project / "inventory-summary.json"),
                    "policy_guide": POLICY_GUIDE,
                    "development_examples": [
                        {k: v for k, v in s.items() if k != "sha256"} for s in self.samples.values()
                    ],
                }
            elif name == "dataset_inspect":
                return self.inspect(**args), False
            elif name == "pipeline_trial":
                if (self.project / "policy.json").exists():
                    raise ValueError("Policy already frozen; this development session is complete")
                report = trial(self.project, args["policy"])
                tid = report["policy_sha256"]
                self.trials[tid] = args["policy"]
                result = {
                    "trial_id": tid,
                    **report["summary"],
                    "per_example": [
                        {
                            "id": row["id"],
                            "channels": {
                                kind: {k: v for k, v in diag.items() if k not in {"rule_trace", "operations"}}
                                for kind, diag in row["channels"].items()
                            },
                        }
                        for row in report["cases"]
                    ],
                }
            elif name == "pipeline_finalize":
                tid = args["trial_id"]
                if (self.project / "policy.json").exists():
                    raise ValueError("Policy already frozen")
                policy = self.trials[tid]
                write(self.project / "policy.json", policy)
                write(
                    self.project / "policy-freeze.json",
                    {
                        "policy_sha256": tid,
                        "implementation_sha256": read(self.project / "trials" / tid / "development/report.json")[
                            "implementation_sha256"
                        ],
                        "rationale": args["rationale"],
                        "development_sample_hashes": {sid: s["sha256"] for sid, s in self.samples.items()},
                        "validation_seen_during_tuning": False,
                    },
                )
                result = {"status": "frozen", "policy_sha256": tid, "batch_execution_model_calls": 0}
            else:
                raise ValueError("Unknown dataset tool")
            return [{"type": "text", "text": json.dumps(result, ensure_ascii=False, allow_nan=False)}], False
        except (ValueError, KeyError, TypeError) as exc:
            return [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], True

    def inspect(self, sample_id, channel, trial_id=None):
        sample = self.samples[sample_id]
        index = ["ECG", "PPG", "ABP"].index(channel)
        path = self.project / "samples" / (sample_id + ".npz")
        if sha(path) != sample["sha256"]:
            raise ValueError("Development example changed")
        with np.load(path, allow_pickle=False) as data:
            before = data["values"][index].copy()
        after = valid = None
        if trial_id is not None:
            self.trials[trial_id]
            path = self.project / "trials" / trial_id / "development" / (sample_id + ".npz")
            with np.load(path, allow_pickle=False) as data:
                after, valid = data["values"][index].copy(), data["valid_mask"][index].copy()
        with PLOT_LOCK:
            fig = Figure(figsize=(12, 5), facecolor="white")
            FigureCanvasAgg(fig)
            axes = fig.subplots(2, 1)
            t = np.arange(len(before)) / FS
            focus = (
                int(np.nanargmax(np.abs(np.diff(before)))) + 1
                if np.isfinite(before).any() and len(before) > 1 and np.isfinite(np.diff(before)).any()
                else 0
            )
            for ax in axes:
                ax.plot(t, before, label="input", color="#cf7b26", linewidth=0.8)
                if after is not None:
                    ax.plot(t, after, label="actual export", color="#1779ae", linewidth=0.8)
                    ax.fill_between(t, 0, 1, where=~valid, transform=ax.get_xaxis_transform(), color="red", alpha=0.15)
                ax.set_ylabel("mmHg" if channel == "ABP" else "source amplitude")
                ax.grid(alpha=0.2)
            axes[0].legend(loc="upper right")
            axes[1].set_xlim(max(0, focus / FS - 2), min(len(before) / FS, focus / FS + 2))
            axes[1].set_xlabel("Relative time (s); lower panel: largest adjacent step")
            fig.suptitle(f"{sample_id} / {channel} — observations; no clean reference")
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100)
        return [
            {
                "type": "text",
                "text": json.dumps(
                    {"sample_id": sample_id, "channel": channel, "features": sample["features"][channel]}
                ),
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(buffer.getvalue()).decode(),
                },
            },
        ]
