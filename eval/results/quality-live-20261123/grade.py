"""Score final, content-bound artifacts after live runs. No evaluator inputs are available to the model."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import array_hash, changed, digest, read, sha, write
from tsmcp.cleaning.engine import load_arrays
from tsmcp.cleaning.quality_gate import SPEC_ID
from tsmcp.processing import intervals

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs"
EXPERIMENT = RUNS / "plan-live-20261123"
PREVIOUS = RUNS / "group-repair-v2-20261122"
TARGET_NAMES = {"biphasic_core", "impulsive_core", "isolated_spike_candidate"}
NATURAL_AUDIT = HERE.parent / "group-repair-v2-20261122/accepted-peak-audit.json"


def jsonl(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def wave(root, case, manifest):
    c = case["id"]
    entry = manifest[c]
    artifact = read(root / "artifacts" / entry["artifact_id"] / "artifact.json")
    raw, t, delivered, valid, estimated = load_arrays(root, case, artifact)
    assert sha(root / "public" / c / "input.npz") == entry["source_sha256"] == case["input_sha256"]
    assert array_hash(delivered, t, valid, estimated) == entry["content_hash"]
    assert sha(root / entry["probe"]) == entry["probe_sha256"]
    return raw, t, delivered, valid, estimated


def fixtures(label, manifest):
    if label == "synthetic":
        score = EXPERIMENT / "synthetic-study/evaluator"
        key = read(score / "design.json")
        with np.load(score / "clean.npz") as z:
            clean = {k: z[k] for k in ("ECG", "PPG", "ABP")}
        result = {}
        for record in manifest["records"]:
            file = record["source_name"]
            item = key[Path(file).stem]
            assert record["source_sha256"] == item["sha256"]
            result[record["record_id"]] = (item["condition"], item["targets"], clean)
        return result
    if label == "natural":
        return {}
    old = PREVIOUS if label == "bipolar" else PREVIOUS / label
    truth = read(old / "injection-score/key.json")
    original = np.load(old / "injection-score/observed-before.npy", allow_pickle=False)
    with np.load(old / "injection-input/waveform.npz") as z:
        clean = dict(ECG=original, PPG=z["PPG"], ABP=z["ABP"])
    (rec,) = manifest["records"]
    assert rec["source_sha256"] == sha(old / "injection-input/waveform.npz")
    return {rec["record_id"]: (label, [dict(channel="ECG", a=a, b=b) for a, b in truth["ranges"]], clean)}


def score(root, label, *, output_directory=HERE):
    run = read(root / "export/evaluation_report.json")
    manifest = read(root / "dataset_manifest.json")
    wm = read(root / "export/waveform_manifest.json")
    assert run["acceptance_spec_id"] == SPEC_ID and not run["replayed"]
    assert not read(root / "run.json")["config"].get("receipt_seed")
    assert (root / "private/vision-preflight.json").is_file() and read(root / "private/vision-preflight.json")["passed"]
    events = list(jsonl(root / "events.jsonl"))
    calls = {x["request_id"]: x for x in events if x["type"] == "model_call_completed"}
    docs = [read(f) for f in (root / "dataset-receipts").glob("*.json")]
    assert len(calls) == run["model_calls"] and all(d["request_id"] in calls for d in docs)
    docs_by_hash = {d["binding"]: d for d in docs}
    plans_total = plans_seen = images = 0
    for d in docs:
        assert d["decision_sha256"] == digest([d["binding"], d["request_id"], d["decision"]])
        if d["stage"] == "diagnosis":
            images += sum(len(v[2]) for v in d["evidence"])
    options = fixtures(label, manifest)
    data = defaultdict(list)
    initial = defaultdict(list)
    for h in jsonl(root / "inspection/candidates.jsonl"):
        z = h["range"]
        initial[z["block_id"]].append((z["start_sample"], z["end_sample"], h["observations"][0]["tool_id"]))
    group_actions = {}
    all_trials = defaultdict(list)
    record_status = {}
    for rd in (root / "records").iterdir():
        hits = read(rd / "hit-dispositions.json")
        plans = read(rd / "repair-plans.json")
        rosters = read(rd / "groups.json")
        pages = [g["id"] for roster in rosters for g in roster["pages"]]
        assert sorted(h["id"] for h in hits) == sorted(hid for g in rosters for hid in g["member_ids"])
        for group in rosters:
            bounds = group["pages"]
            assert bounds[0]["start"] == 0 and bounds[-1]["end"] == group["count"]
            assert all(a["end"] == b["start"] for a, b in zip(bounds[:-1], bounds[1:], strict=True))
        for p in plans.values():
            assert docs_by_hash[p["receipt"]]["stage"] == "groups"
            assert docs_by_hash[p["diagnosis_receipt"]]["stage"] == "diagnosis"
        plans_seen += sum(k in plans for k in pages)
        plans_total += len(pages)
        record_status[rd.name] = dict(
            groups=len(rosters),
            pages=len(pages),
            pages_with_policy=sum(k in plans for k in pages),
            alerts=len(hits),
            members_with_policy=sum(h["policy_id"] in plans for h in hits),
            unresolved=read(rd / "report.json")["unresolved"],
        )
        for h in hits:
            data[h["case"]].append(h)
            p = plans.get(h["policy_id"])
            if p:
                group_actions[h["id"]] = dict(
                    wave=p["wave"], wave_if=p["wave_if"], reason=p["reason"], request=p["request_id"]
                )
        for f in rd.glob("*-trials*.json"):
            for trial in read(f):
                all_trials[trial["case"]].append((f.name, trial))
    per_channel = defaultdict(Counter)
    scores = {}
    ranges = []
    conditions = {}
    patches = Counter()
    for f in root.glob("records/*/patches/*.json"):
        patch = read(f)
        gate = patch["receipt"]
        assert gate["hard_gate"]["status"] == "passed" and all(gate["hard_gate"]["checks"].values())
        for ref in gate["evidence_refs"]:
            receipt = docs_by_hash[ref]
            assert receipt["stage"] == "gate"
            assert all(
                "blind_observation" in e[1] and e[1]["blind_observation"]["receipt"] in docs_by_hash
                for e in receipt["evidence"]
            )
        patches[(patch["plan"]["tool"]["tool_id"])] += 1
    for c in manifest["cases"]:
        cid, rid, task = c["id"], c["record_id"], c["task"]
        input_signal, t, release, v, e = wave(root, c, wm)
        _, _, _, v0, e0 = load_arrays(root, c)
        delta = changed(input_signal, release)
        disabled = v0 & ~e0 & ~v
        new_est = e & ~e0
        assert int(delta.sum()) == wm[cid]["modified_samples"]
        per_channel[task].update(
            changed_fragments=len(intervals(delta)),
            disabled_fragments=len(intervals(disabled)),
            changed_seconds=float(delta.sum() / c["fs"]),
            disabled_seconds=float(disabled.sum() / c["fs"]),
            samples=c["n"],
            modified=int(delta.sum()),
            disabled=int(disabled.sum()),
            estimated=int(new_est.sum()),
        )
        if rid not in options:
            continue
        condition, targets, clean = options[rid]
        truth = clean[task]
        assert len(truth) == c["n"] and np.isfinite(truth).all()
        conditions[condition] = rid
        own = [q for q in targets if q["channel"] == task]
        affected = np.zeros(c["n"], bool)
        for q in own:
            affected[q["a"] : q["b"]] = True
        assert np.array_equal(input_signal[~affected], truth[~affected])
        assert delta.shape == affected.shape
        key = condition + "." + task
        scores[key] = dict(
            targets=len(own),
            samples=c["n"],
            outside_changed=int(delta[~affected].sum()),
            outside_disabled=int(disabled[~affected].sum()),
            outside_estimated=int(new_est[~affected].sum()),
            changed=int(delta.sum()),
            disabled=int(disabled.sum()),
        )
        for q in own:
            a, b = q["a"], q["b"]
            assert (input_signal[a:b] != truth[a:b]).all()
            overlaps = [h for h in data[cid] if a < h["b"] and h["a"] < b]
            spikes = [h for h in overlaps if h["reason"] == "isolated_spike_candidate" and (h["a"], h["b"]) == (a, b)]
            before = float(np.mean(abs(input_signal[a:b] - truth[a:b])))
            after = float(np.mean(abs(release[a:b] - truth[a:b])))
            ranges.append(
                dict(
                    record=rid,
                    case=cid,
                    channel=task,
                    condition=condition,
                    a=a,
                    b=b,
                    source_s=float(t[a]),
                    alarm=any(left < b and a < right for left, right, k in initial[cid]),
                    exact_core=any((left, right) == (a, b) and k in TARGET_NAMES for left, right, k in initial[cid]),
                    final_modified=int(delta[a:b].sum()),
                    final_disabled=int(disabled[a:b].sum()),
                    final_estimated=int(new_est[a:b].sum()),
                    complete_repair=bool(delta[a:b].all() and new_est[a:b].all() and v[a:b].all() and after < before),
                    input_mae=before,
                    output_mae=after,
                    group_actions=[
                        dict(
                            selected=group_actions.get(h["id"], {}),
                            applied=h["wave_requested"],
                            diagnosis=h.get("diagnosis"),
                            support=h.get("applicability"),
                            source=h.get("decision_source"),
                        )
                        for h in spikes
                    ],
                    trials=[
                        dict(stage=stage, status=z["status"], reason=z.get("reason"), receipt=z.get("receipt"))
                        for stage, z in all_trials[cid]
                        if a < z["b"] and z["a"] < b
                    ],
                    other_alerts=sorted({h["reason"] for h in overlaps}),
                    source_hash=wm[cid]["source_sha256"],
                )
            )
        if own:
            subset = [row for row in ranges if row["case"] == cid]
            scores[key].update(
                initial_alarms=sum(q["alarm"] for q in subset),
                exact_cores=sum(q["exact_core"] for q in subset),
                recovered=sum(q["complete_repair"] for q in subset),
                full_disabled=sum(q["final_disabled"] == q["b"] - q["a"] for q in subset),
                input_mae=float(np.abs(input_signal[affected] - truth[affected]).mean()),
                output_mae=float(np.abs(release[affected] - truth[affected]).mean()),
            )
    with (root / "export/measurements.csv").open(encoding="utf-8-sig") as stream:
        measures = Counter(r["quality"] for r in csv.DictReader(stream))
    tiers = Counter(z["status"] for items in all_trials.values() for _, z in items)
    assert sum(v["samples"] for v in per_channel.values()) == sum(c["n"] for c in manifest["cases"])
    result = dict(
        label=label,
        status=run["status"],
        records=record_status,
        patches=patches,
        trials=tiers,
        calls=len(calls),
        provider_returned_usd=sum(z["cost_usd"] for z in calls.values()),
        receipt_counts=Counter(d["stage"] for d in docs),
        messages_rejected=sum(z["type"] == "dataset_submission_rejected" for z in events),
        group_pages=plans_total,
        group_pages_with_policy=plans_seen,
        diagnostic_image_transmissions=images,
        measured_states=measures,
        channel_counts=per_channel,
        conditions=conditions,
        scores=scores,
        targets=ranges,
    )
    write(output_directory / (label + "-verified.json"), result)
    return result


def natural_site(root):
    (s,) = read(NATURAL_AUDIT)
    manifest = read(root / "dataset_manifest.json")
    cid = s["case"]
    c = next(c for c in manifest["cases"] if c["id"] == cid)
    w = read(root / "export/waveform_manifest.json")
    x, t, y, v, e = wave(root, c, w)
    a, b = s["a"], s["b"]
    return dict(
        case=cid,
        start_s=float(t[a]),
        stop_s=float(t[b - 1] + 1 / c["fs"]),
        modified=int(changed(x[a:b], y[a:b]).sum()),
        valid=int(v[a:b].sum()),
        estimated=int(e[a:b].sum()),
        n=b - a,
    )


if __name__ == "__main__":
    import sys

    label = sys.argv[1]
    root = EXPERIMENT / ("synthetic-study/live" if label == "synthetic" else label)
    result = score(root, label)
    print(label, result["status"], result["calls"], result["provider_returned_usd"])
    if label == "natural":
        print(natural_site(root))
