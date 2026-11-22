"""Answer-blind dataset-level peak-policy experiment on the supplied Vital review archive."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import io
import json
import platform
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dataset_tools import PLOT_LOCK, tool  # noqa: E402
from eval.dataset_workflow import read, sha, write  # noqa: E402
from eval.real_processing_helper import save_npz  # noqa: E402
from eval.unified.drivers import cost_of, make_driver  # noqa: E402
from eval.unified.loop import Limits, run_session  # noqa: E402
from tsmcp.peak_policy import FEATURE_NAMES, SCHEMA, apply, digest, observe, validate  # noqa: E402

SYSTEM = """Develop ONE reusable ECG peak-annotation review policy for a whole dataset. Inspect the supplied
unlabelled development examples, test general policies, and freeze one. Use only the public observations.
Original R markers are fallible candidates. ECG source units are unconfirmed, ABP is mmHg; no channel alignment
has been applied. A negative deflection is not automatically a false beat. Preserve waveform samples.
Choose whether a complete 60s window is reliable enough for RR analysis, remove supported extra markers,
and refine the location of retained beats within identifiable QRS complexes. Do not invent missing beats.
Do not remove a real T wave from the ECG when removing its mistaken R annotation. Uncertainty may justify
window exclusion but is not proof of pathological or corrupted signals. There is no clinical ground truth in
your tools. Trials return descriptive changes only, never private scores. Avoid source/example-specific rules.
The same policy will run locally on all windows, with no per-window LLM calls. You have 28 tool calls including
finalize; inspect ordinary observations too, compare trial outputs, then finalize a tested policy and stop.
"""

GUIDE = {
    "schema_version": SCHEMA,
    "alignment": "none(default), positive, negative, absolute. Search original ECG within radius_ms(default80, range0..200) of each retained old marker; original samples do not change.",
    "min_energy_fraction": "Optional0..2. Drop markers whose 5–20Hz QRS-band RMS over ±120ms is below this fraction of the window median marker energy.",
    "min_template_correlation": "Optional-1..1. Drop markers below correlation to the window median normalized QRS-band snippet. This can discard real morphological variants; inspect carefully.",
    "max_detector_distance_ms": "Optional0..500. Drop old markers farther than this from a fixed independent SignalMind QRS detector; detector agreement is not truth.",
    "refractory_ms": "Optional0..500, default0. Among collisions/nearby final markers keep higher QRS-band energy. Exact coordinate collisions are deduplicated even at zero.",
    "exclude_if": "Optional list of {feature, op:lt|gt, value:number}. OR-combined; no rules means retain all windows. All window outputs still retain peak-edit decisions even when excluded.",
    "features": sorted(FEATURE_NAMES),
    "feature_notes": "flat_fraction counts exact flat runs>=0.5s; RR outside fraction uses0.30..2.00s; detector_disagreement_fraction counts old markers>120ms from detector. kept/removed features are after policy processing. All are heuristic diagnostics, not medical labels.",
    "minimal_policy": {"schema_version": SCHEMA, "alignment": "none", "exclude_if": []},
}


def prepare(archive_path, project, examples=12):
    project.mkdir(parents=True, exist_ok=False)
    (project / "private").mkdir(mode=0o700)
    (project / "public").mkdir()
    with ZipFile(archive_path) as archive:
        prefix = archive.namelist()[0].split("/")[0] + "/"

        def table(name):
            return list(csv.DictReader(io.StringIO(archive.read(prefix + name).decode("utf-8-sig"))))

        lookup = table("WINDOW_LOOKUP.csv")
        final = table("00_review_decisions/retained_window_R_peaks.csv")
        exclusion = {r["group_id"]: r for r in table("00_review_decisions/excluded_windows.csv")}
        final_by_g = {}
        for row in final:
            final_by_g.setdefault(row["group_id"], {})[int(row["original_R_id"][1:]) - 1] = int(
                row["corrected_sample0"]
            )
        cases, truth = [], []
        for row in lookup:
            uid = "w_" + hashlib.sha256(("peak-review-v1:" + row["sequence_key"]).encode()).hexdigest()[:16]
            with np.load(io.BytesIO(archive.read(prefix + row["original_npz"])), allow_pickle=False) as data:
                x, bp, old = data["ecg"].copy(), data["abp"].copy(), data["r_current_sample0"].copy()
                if data["r_current_ids"].tolist() != [f"R{i + 1:03d}" for i in range(len(old))]:
                    raise ValueError("Original marker ids must retain their ordinal mapping")
            folder = project / "public" / uid
            folder.mkdir()
            save_npz(folder / "input.npz", ecg=x, abp=bp, old_peaks=old)
            cases.append({"id": uid, "sha256": sha(folder / "input.npz")})
            target = (
                None
                if row["final_disposition"] == "EXCLUDED"
                else [final_by_g[row["group_id"]].get(i, -1) for i in range(len(old))]
            )
            truth.append(
                {
                    "id": uid,
                    "group_id": row["group_id"],
                    "subject_key": row["subject_key"],
                    "disposition": row["final_disposition"],
                    "target_peak_samples": target,
                    "exclusion_subtype": exclusion.get(row["group_id"], {}).get("original_review_status", ""),
                }
            )
    subjects = sorted(
        {r["subject_key"] for r in truth}, key=lambda v: hashlib.sha256(("inspect-v1:" + v).encode()).hexdigest()
    )
    chosen_subjects = subjects[:examples]
    visible = [min(r["id"] for r in truth if r["subject_key"] == subject) for subject in chosen_subjects]
    for row in truth:
        row["policy_development_subject"] = row["subject_key"] in chosen_subjects
        row["visible_example"] = row["id"] in visible
    index = {
        "schema_version": "signalmind.peak-review-input.v0.1",
        "fs_hz": 125,
        "units": {"ecg": "source_units", "abp": "mmHg"},
        "cases": sorted(cases, key=lambda r: r["id"]),
        "development_examples": visible,
        "duration_s": 60,
    }
    write(project / "public/index.json", index)
    write(project / "private/truth.json", truth)
    write(
        project / "preparation.json",
        {
            "archive_sha256": sha(archive_path),
            "windows": len(cases),
            "subjects": len(subjects),
            "development_subjects": len(chosen_subjects),
            "visible_examples": len(visible),
            "labels_exposed_to_model": False,
            "selection": "Deterministic subject hash, one unlabelled window per selected subject; no label stratification.",
        },
    )
    return read(project / "preparation.json")


class ReviewTools:
    def __init__(self, public, output):
        self.public, self.output = Path(public), Path(output)
        self.index = read(self.public / "index.json")
        self.visible = set(self.index["development_examples"])
        self.cache, self.trials = {}, {}
        self.output.mkdir(parents=True, exist_ok=True)
        self.schemas = [
            tool(
                "review_overview",
                "Read unlabelled development-example diagnostics and shared peak-policy contract; no private labels/scores are accessible.",
                {},
            ),
            tool(
                "review_inspect",
                "View ECG and ABP with original R markers in a relative-time interval; optionally overlay actual trial peak edits. Only listed development examples are accessible.",
                {
                    "id": {"type": "string"},
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "trial_id": {"type": "string"},
                },
                ("id",),
            ),
            tool(
                "review_trial",
                "Apply a reusable policy to all visible examples locally. Reports decisions, counts and diagnostics, not annotation agreement.",
                {"policy": {"type": "object"}},
                ("policy",),
            ),
            tool(
                "review_finalize",
                "Freeze a previously tried policy; no changes or private-score feedback after this call.",
                {"trial_id": {"type": "string"}, "rationale": {"type": "string"}},
                ("trial_id", "rationale"),
            ),
        ]

    def load(self, uid):
        if uid not in self.visible:
            raise ValueError("Only the listed unlabelled development examples are available")
        if uid not in self.cache:
            path = self.public / uid / "input.npz"
            expected = next(r["sha256"] for r in self.index["cases"] if r["id"] == uid)
            if sha(path) != expected:
                raise ValueError("Public input changed")
            with np.load(path, allow_pickle=False) as data:
                values = {k: data[k].copy() for k in data.files}
            values["observed"] = observe(values["ecg"], values["old_peaks"])
            self.cache[uid] = values
        return self.cache[uid]

    async def call(self, name, args):
        try:
            if name == "review_overview":
                result = {
                    "windows_in_dataset": len(self.index["cases"]),
                    "fs_hz": 125,
                    "units": self.index["units"],
                    "guide": GUIDE,
                    "examples": [{"id": uid, **self.load(uid)["observed"]["features"]} for uid in sorted(self.visible)],
                }
            elif name == "review_inspect":
                return self.inspect(**args), False
            elif name == "review_trial":
                if (self.output / "policy.json").exists():
                    raise ValueError("Policy is already frozen")
                policy = validate(args["policy"])
                tid = digest(policy)
                results = {}
                for uid in sorted(self.visible):
                    data = self.load(uid)
                    results[uid] = apply(data["ecg"], data["old_peaks"], policy, observed=data["observed"])
                self.trials[tid] = {"policy": policy, "results": results}
                write(self.output / (tid + ".json"), self.trials[tid])
                result = {"trial_id": tid, "examples": [{"id": uid, **row} for uid, row in results.items()]}
            elif name == "review_finalize":
                if (self.output / "policy.json").exists():
                    raise ValueError("Policy already frozen")
                tid = args["trial_id"]
                write(self.output / "policy.json", self.trials[tid]["policy"])
                write(
                    self.output / "freeze.json",
                    {"policy_sha256": tid, "rationale": args["rationale"], "private_labels_accessed": False},
                )
                result = {"status": "frozen", "policy_sha256": tid}
            else:
                raise ValueError("Unknown tool")
            return [{"type": "text", "text": json.dumps(result, ensure_ascii=False, allow_nan=False)}], False
        except (ValueError, TypeError, KeyError) as exc:
            return [{"type": "text", "text": str(exc)}], True

    def inspect(self, id, start_s=0, end_s=10, trial_id=None):
        if not 0 <= start_s < end_s <= 60:
            raise ValueError("Choose 0<=start_s<end_s<=60")
        data = self.load(id)
        old = data["old_peaks"]
        chosen = np.flatnonzero((old >= start_s * 125) & (old < end_s * 125))
        new = np.asarray(self.trials[trial_id]["results"][id]["new_peak_samples"]) if trial_id else None
        with PLOT_LOCK:
            fig = Figure(figsize=(12, 6), facecolor="white")
            FigureCanvasAgg(fig)
            axes = fig.subplots(2, 1)
            a, b = round(start_s * 125), round(end_s * 125)
            t = np.arange(a, b) / 125
            axes[0].plot(t, data["ecg"][a:b], color="#333", linewidth=0.8)
            axes[0].scatter(
                old[chosen] / 125, data["ecg"][old[chosen]], color="#d77a19", marker="v", label="original R"
            )
            if end_s - start_s <= 15:
                for i in chosen:
                    axes[0].annotate(
                        f"R{i + 1:03d}",
                        (old[i] / 125, data["ecg"][old[i]]),
                        xytext=(0, 8),
                        textcoords="offset points",
                        fontsize=7,
                    )
            if new is not None:
                good = (new >= a) & (new < b)
                axes[0].scatter(
                    new[good] / 125,
                    data["ecg"][new[good]],
                    facecolors="none",
                    edgecolors="#247bb5",
                    s=50,
                    label="trial retained R",
                )
            axes[1].plot(t, data["abp"][a:b], color="#9b4242", linewidth=0.9)
            axes[0].set_ylabel("ECG / source units")
            axes[1].set_ylabel("ABP / mmHg")
            axes[1].set_xlabel("Relative seconds; waveforms unchanged")
            for ax in axes:
                ax.set_xlim(start_s, end_s)
                ax.grid(alpha=0.2)
            axes[0].legend()
            fig.suptitle(id)
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=120)
        details = [
            {
                "R_id": f"R{i + 1:03d}",
                "sample0": int(old[i]),
                "energy_fraction": float(data["observed"]["energy_fraction"][i]),
                "template_correlation": float(data["observed"]["template_correlation"][i]),
            }
            for i in chosen
        ]
        return [
            {"type": "text", "text": json.dumps({"markers": details})},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(buffer.getvalue()).decode(),
                },
            },
        ]


async def develop(project, cap):
    target = project / "agent"
    if (project / "session.json").exists() or (target / "policy.json").exists():
        raise ValueError("Preserve prior attempts; use a new project for another model run")
    sources = [
        "src/tsmcp/peak_policy.py",
        "src/tsmcp/plugins/ecg/rpeaks.py",
        "src/tsmcp/core/dsp.py",
        "src/tsmcp/core/peaks.py",
        "eval/run_peak_review.py",
        "eval/grade_peak_review.py",
        "eval/unified/loop.py",
        "eval/unified/drivers.py",
        "src/tsmcp/processing.py",
        "eval/dataset_tools.py",
        "eval/dataset_workflow.py",
        "eval/real_processing_helper.py",
    ]
    provenance = {
        "date": "2026-11-06",
        "model": "deepseek-flash",
        "cap_usd": cap,
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "public_index_sha256": sha(project / "public/index.json"),
        "protocol_sha256": sha(ROOT / "docs/signalmind/vital-peak-evaluation-protocol.md"),
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    for name in sources:
        path = project / "private/source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    write(project / "provenance.json", provenance)
    tools = ReviewTools(project / "public", target)
    driver = make_driver("deepseek-flash", max_tokens=8192)
    try:
        result = await run_session(
            driver,
            tools,
            SYSTEM,
            "Develop and finalize a general peak-review policy from the unlabelled examples. Start with review_overview.",
            Limits(max_turns=30, max_tool_calls=28, max_seconds=900, max_output_tokens_total=40000),
            project / "private/transcript.jsonl",
            provenance,
            budget_exhausted=lambda usage: cost_of("deepseek-flash", usage) + 0.05 >= cap,
        )
    finally:
        driver.close()
    result["estimated_peak_usd"] = cost_of("deepseek-flash", result["usage"])
    result["actual_billed_usd"] = None
    result["policy_finalized"] = (target / "policy.json").exists()
    write(project / "session.json", result)
    return {k: v for k, v in result.items() if k not in {"text", "usage"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "develop"])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--cap", type=float, default=0.50)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.archive, args.project, args.examples)
    else:
        if not args.live or args.cap <= 0.05:
            raise ValueError("Supply --live and cap above the $0.05 reserve")
        result = asyncio.run(develop(args.project, args.cap))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
