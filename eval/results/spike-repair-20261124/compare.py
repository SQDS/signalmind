"""Post-run comparison on identical input files. Withheld pre-injection samples are read by the scorer only."""

import importlib.util
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from tsmcp.cleaning.common import implementation_hash, read, write
from tsmcp.cleaning.quality_gate import SPEC_ID

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs"
NEW = RUNS / "spike-repair-20261124"
OLD = RUNS / "plan-live-20261123"
PREVIOUS = HERE.parent / "quality-live-20261123"
spec = importlib.util.spec_from_file_location("prior_grade", PREVIOUS / "grade.py")
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)
KEYS = ("bipolar", "same-sign", "moderate", "synthetic")
TITLES = {"bipolar": "强双相", "same-sign": "强单向", "moderate": "中等单向"}


def initial_events(root):
    return sorted(
        (z["range"]["block_id"], z["range"]["start_sample"], z["range"]["end_sample"], z["observations"][0]["tool_id"])
        for z in grade.jsonl(root / "inspection/candidates.jsonl")
    )


def run_one(label):
    new = NEW / label
    old = OLD / ("synthetic-study/live" if label == "synthetic" else label)
    design = read(NEW / "design.json")
    assert design["acceptance_spec"] == SPEC_ID and read(old / "ledger/run.json")["acceptance_spec_id"] == SPEC_ID
    assert read(new / "ledger/run.json")["implementation"] == design["implementation_hash"]
    assert read(old / "dataset_manifest.json")["scope_hash"] == read(new / "dataset_manifest.json")["scope_hash"]
    assert initial_events(old) == initial_events(new)
    previous = read(PREVIOUS / (label + "-verified.json"))
    current = grade.score(new, label, output_directory=HERE)

    def key(z):
        return (z["case"], z["a"], z["b"])

    prev_targets = {key(z): z for z in previous["targets"]}
    assert prev_targets.keys() == {key(z) for z in current["targets"]}
    cases = {c["id"]: c for c in read(new / "dataset_manifest.json")["cases"]}
    references = grade.fixtures(label, read(new / "dataset_manifest.json"))
    compared = []
    for z in current["targets"]:
        prior = prev_targets[key(z)]
        assert (
            z["source_hash"] == prior["source_hash"]
            and z["input_mae"] == prior["input_mae"]
            and z["exact_core"] == prior["exact_core"]
        )
        c = cases[z["case"]]
        a, b = z["a"], z["b"]
        record = new / "records" / z["record"]
        trials = [
            r
            for f in sorted(record.glob("spike-trials*.json"))
            for r in read(f)
            if r["case"] == c["id"] and (r["a"], r["b"]) == (a, b)
        ]
        trial = next((r for r in trials if r.get("candidate")), None)
        mae = None
        if trial:
            art = read(new / "artifacts" / trial["candidate"] / "artifact.json")
            _, _, candidate, _, _ = grade.load_arrays(new, c, art)
            truth = references[z["record"]][2][c["task"]]
            mae = float(np.abs(candidate[a:b] - truth[a:b]).mean())
        compared.append(
            dict(
                record=z["record"],
                case=c["id"],
                condition=z["condition"],
                channel=z["channel"],
                a=a,
                b=b,
                source_s=z["source_s"],
                before_input=z["input_mae"],
                old_error=prior["output_mae"],
                new_error=z["output_mae"],
                old_recovered=prior["complete_repair"],
                new_recovered=z["complete_repair"],
                new_trial_mae=mae,
                trial_verdicts=[(r["status"], r.get("reason", "")) for r in trials],
                new_disabled=z["final_disabled"],
                actual_requests=z["group_actions"],
                exact=z["exact_core"],
            )
        )
    aligned = {}
    for k, s in current["scores"].items():
        base = previous["scores"][k]
        aligned[k] = dict(
            n=s["targets"],
            detected=s.get("exact_cores", 0),
            old=base.get("recovered", 0),
            new=s.get("recovered", 0),
            input_mae=s.get("input_mae"),
            old_mae=base.get("output_mae"),
            new_mae=s.get("output_mae"),
            old_outside_changed=base["outside_changed"],
            outside_changed=s["outside_changed"],
            outside_disabled=s["outside_disabled"],
            old_outside_disabled=base["outside_disabled"],
        )
    jr = read(new / "export/evaluation_report.json")
    return dict(
        input_files_identical=True,
        initial_alerts_identical=True,
        initial_events=len(initial_events(old)),
        scored=aligned,
        targets=compared,
        status=current["status"],
        unresolved=dict(Counter(item["reason"] for r in jr["records"] for item in r["unresolved"])),
        group_pages=current["group_pages"],
        group_pages_covered=current["group_pages_with_policy"],
        api_calls=current["calls"],
        usd=current["provider_returned_usd"],
        retries_rejected=current["messages_rejected"],
    )


def comparison_plot(results):
    mpl.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "font.size": 8,
        }
    )
    fig = Figure(figsize=(11.8, 7), layout="constrained")
    axes = fig.subplots(3, 4, sharey="row", squeeze=False)
    colors = {"input": "#83919b", "old": "#bd6e3d", "now": "#304c7b", "reference": "#14846c"}
    for i, label in enumerate(KEYS[:3]):
        new = NEW / label
        old = OLD / label
        manifest = read(new / "dataset_manifest.json")
        references = grade.fixtures(label, manifest)
        for j, z in enumerate(results[label]["targets"]):
            c = next(c for c in manifest["cases"] if c["id"] == z["case"])
            x, t, y, _, _ = grade.wave(new, c, read(new / "export/waveform_manifest.json"))
            _, _, yp, _, _ = grade.wave(old, c, read(old / "export/waveform_manifest.json"))
            a, b = z["a"], z["b"]
            idx = np.arange(a - 7, b + 8)
            u = (t[idx] - t[a]) * 1000
            ax = axes[i, j]
            ax.axvspan(-500 / c["fs"], 1000 * (b - a - 0.5) / c["fs"], color="#ebcd87", alpha=0.38, lw=0)
            for trace, color, style, marker, width, name in [
                (x, "input", "-", None, 2.9, "含干扰输入"),
                (yp, "old", "--", None, 1.5, "旧版 DS"),
                (y, "now", "-", "o", 0.92, "本版 DS"),
                (references[z["record"]][2]["ECG"], "reference", ":", "+", 0.96, "评分留存原值"),
            ]:
                ax.plot(
                    u, trace[idx], c=colors[color], ls=style, marker=marker, ms=3.1, mfc="none", lw=width, label=name
                )
            verdict = "修回" if z["new_recovered"] else "停用" if z["new_disabled"] == b - a else "保留原值"
            ax.set_title(f"{z['source_s']:.3f} s · {verdict}")
            if j == 0:
                ax.set_ylabel(TITLES[label] + "\nECG 原始单位")
            if i == 2:
                ax.set_xlabel("相对尖峰起点（毫秒）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("Vital 同一真实背景 · 两版 DS 在全部 12 处已知尖峰上的实际交付", fontsize=14)
    fig.savefig(HERE / "vital-before-after.png", dpi=145)


if __name__ == "__main__":
    import sys

    labels = sys.argv[1:] or list(KEYS)
    results = {label: run_one(label) for label in labels}
    for label, r in results.items():
        write(HERE / (label + "-comparison.json"), r)
    if set(labels) == set(KEYS):
        assert read(NEW / "design.json")["implementation_hash"] == implementation_hash()
        comparison_plot(results)
        stats = ElementTree.parse(NEW / "regression.xml").getroot().find("testsuite").attrib
        assert stats["failures"] == stats["errors"] == "0"
        notes = read(NEW / "local-hard-gate/verification.json")
        assert len(notes["natural"]) == 7 and all(z["new"] == "ineligible" for z in notes["natural"])
        cfg = read(NEW / "spec/synthetic.json")["config"]
        budget = read(Path(cfg["budget_root"]) / "api_budget.json")
        paid = {}
        for label in KEYS:
            for e in grade.jsonl(NEW / label / "events.jsonl"):
                if e["type"] == "model_call_completed":
                    assert e["request_id"] not in paid
                    paid[e["request_id"]] = e["cost_usd"]
        assert paid.keys() <= {c.get("request_id") for c in budget["calls"][read(NEW / "design.json")["prior_calls"] :]}
        write(
            HERE / "verification.json",
            dict(
                tests=int(stats["tests"]),
                local_protected=len(notes["natural"]),
                actual_new_calls=len(paid),
                api_usd=sum(paid.values()),
                cumulative_usd=budget["estimated_peak_usd"],
                reserve=budget["unknown_charge_reserve_usd"],
                acceptance_spec=SPEC_ID,
                discovery_identical=True,
            ),
        )
    for label, r in results.items():
        print(
            label,
            r["status"],
            r["unresolved"],
            [(k, x["old"], x["new"], x["outside_changed"]) for k, x in r["scored"].items() if x["n"]],
        )
