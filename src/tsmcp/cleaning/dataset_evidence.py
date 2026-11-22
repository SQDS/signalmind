"""Patient-relative screening, source-aligned visual evidence and cross-channel review selection."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.inspection.patient_profile import amplitude_reference, beat_pairs, patient_flags, rate_reference
from tsmcp.inspection.scan import scan
from tsmcp.processing import intervals

from .common import changed, digest
from .provenance import check_stop


def native_rows(probe):
    if probe["task"] == "ABP":
        return [(v["start_sample"], v["stop_sample"], v) for v in probe["pressures"]]
    events = probe["events"]
    return [
        (events[v["left_event_id"]]["sample"], events[v["right_event_id"]]["sample"], v) for v in probe["intervals"]
    ]


def acquisition(case):
    # Row numbers are local to a Vital patient. Imported CSV/NPZ files use a source sample range.
    if case.get("normalized_windows") or (not case.get("has_source_time", True) and "source_range_samples" not in case):
        return case["record_id"], (case["id"],)
    return case["record_id"], tuple(case.get("rows", case.get("source_range_samples", [case["id"]])))


def patient_profiles(cases, data):
    samples = defaultdict(list)
    rates = defaultdict(list)
    for c in cases:
        _, x, v, e, pr = data(c["id"])
        ok = v & ~e & np.isfinite(x)
        task = c["task"]
        for a, b, _ in native_rows(pr):
            if ok[a:b].all():
                samples[task].append((float(x[a:b].min()), float(x[a:b].max())))
        rates[task].extend(60 * c["fs"] / (b - a) for a, b in beat_pairs(pr) if ok[a : b + 1].all())
    profiles = {}
    for task in sorted({c["task"] for c in cases}):
        amp = rate = None
        if samples[task]:
            lo, hi = np.array(samples[task]).T
            amp = amplitude_reference(lo, hi, min(lo), max(hi))
        pool = rates["ECG"] or rates[task]
        if pool:
            rate = rate_reference(pool)
        profiles[task] = dict(amplitude=amp, rate=rate, rate_source="ECG" if rates["ECG"] else task)
    return profiles


def discover(workspace, cases, data, profiles, preset, secondary=False):
    hits = []
    for c in cases:
        check_stop(workspace)
        cid = c["id"]
        _, x, v, e, pr = data(cid)
        ok = v & ~e & np.isfinite(x)
        events, _, _ = scan(x, ok, {"unavailable": ~ok}, c["fs"], c["task"], preset)
        rows = [(i["start"], i["end"], i["kind"], i["value"]) for i in events]
        if secondary:
            from tsmcp.inspection.secondary import compare_block

            observations, _, _ = compare_block(x, ok, pr, c["task"], c["fs"])
            width = max(1, round(5 * c["fs"]))
            tiles = defaultdict(list)
            for sample, _, value in observations:
                tiles[sample // width].append(value)
            rows.extend(
                (i * width, min((i + 1) * width, len(x)), "probe_disagreement", max(values))
                for i, values in tiles.items()
            )
        pf = profiles[c["task"]]
        # A record can have observed amplitude or beats even when the other reference is unavailable.
        amp = pf["amplitude"] or {"low": -np.inf, "high": np.inf}
        rate = pf["rate"] or {"low": -np.inf, "high": np.inf}
        rows.extend((a, b, k, val) for a, b, k, val, _ in patient_flags(x, ok, c["fs"], beat_pairs(pr), amp, rate))
        for a, b, k, val in rows:
            hits.append(
                dict(
                    id=digest([cid, int(a), int(b), k])[:24],
                    case=cid,
                    task=c["task"],
                    a=int(a),
                    b=int(b),
                    reason=k,
                    value=float(val) if val is not None else None,
                )
            )
    return hits


def locations(cases, data, hits, *, uncertain_only=False):
    """Rank concurrent alerts, optionally requiring a remaining measurement uncertainty."""
    by_case = defaultdict(list)
    for h in hits:
        by_case[h["case"]].append(h)
    blocks = defaultdict(list)
    for c in cases:
        blocks[acquisition(c)].append(c)
    joint = []
    controls = {}
    for block in blocks.values():
        changes = defaultdict(list)
        for c in block:
            cid = c["id"]
            t, x, v, e, _ = data(cid)
            good = v & ~e & np.isfinite(x)
            for h in by_case[cid]:
                a, b = h["a"], h["b"]
                good[a:b] = False
                changes[round(float(t[a]), 8)].append((1, h))
                changes[round(float(t[b - 1] + 1 / c["fs"]), 8)].append((-1, h))
            for a, b in intervals(good):
                if b - a < c["fs"]:
                    continue
                mid = (a + b) // 2
                item = dict(case=cid, start=float(t[mid]) - 0.5, end=float(t[mid]) + 0.5)
                priority = digest([cid, a, b])
                if c["task"] not in controls or priority < controls[c["task"]][0]:
                    controls[c["task"]] = priority, item
        active = {}
        previous = None
        for left, right in zip(sorted(changes), sorted(changes)[1:], strict=False):
            for sign, h in sorted(changes[left], key=lambda e: e[0]):
                if sign < 0:
                    active.pop(h["id"], None)
                else:
                    active[h["id"]] = h
            tasks = {h["task"] for h in active.values()}
            if len(tasks) < 2 or (uncertain_only and not any(h["measurement"] == "review" for h in active.values())):
                previous = None
                continue
            score = (len(tasks), len({h["reason"] for h in active.values()}), right - left)
            # Pick the strongest focal interval within each contiguous episode. Keep full episode bounds.
            loc = dict(
                case=min(h["case"] for h in active.values()),
                start=left,
                end=right,
                episode_start=left,
                episode_end=right,
                score=score,
            )
            if previous is None:
                joint.append(loc)
                previous = loc
            else:
                onset = previous["episode_start"]
                if score > tuple(previous["score"]):
                    previous.update(loc)
                previous.update(episode_start=onset, episode_end=right)
    joint.sort(key=lambda z: (tuple(-v for v in z["score"]), z["case"], z["start"]))
    return joint, [v[1] for _, v in sorted(controls.items())]


def hit_location(h, cases, data):
    c = cases[h["case"]]
    t = data(c["id"])[0]
    return dict(case=c["id"], start=float(t[h["a"]]), end=float(t[h["b"] - 1] + 1 / c["fs"]))


def view(path, loc, cases, before, after=None):
    """Show only simultaneous target intervals and nearby observed context; label native measurement keys."""
    c = cases[loc["case"]]
    block = acquisition(c)
    peers = sorted(
        [p for p in cases.values() if acquisition(p) == block], key=lambda p: ("ECG", "PPG", "ABP").index(p["task"])
    )
    start, end = loc["start"], loc["end"]
    fig = Figure(figsize=(9.2, 1.85 * len(peers) + 0.65), layout="constrained")
    FigureCanvasAgg(fig)
    detail = not after and loc.get("show_detail", False)
    axes = fig.subplots(
        len(peers), 2 if after or detail else 1, squeeze=False, sharex="col" if detail else True, sharey="row"
    )
    metrics = {}
    for col, get in enumerate([before, after] if after else [before, before] if detail else [before]):
        for row, peer in enumerate(peers):
            cid = peer["id"]
            t, x, v, e, pr = get(cid)
            ax = axes[row, col]
            fs = peer["fs"]
            pad = 0.15 if detail and col else loc.get("context_s", 5)
            lo, hi = t.searchsorted([start - pad - 1e-7, end + pad - 1e-7])
            a, b = t.searchsorted([start - 1e-7, end - 1e-7])
            ax.axvspan(0, end - start, color="#ffe7be", lw=0)
            ax.plot(
                t[lo:hi] - start,
                x[lo:hi],
                lw=0.8,
                color="#087469" if after and col else "#23536d",
                marker="." if detail and col else None,
                markersize=3,
            )
            if col and after and peer["task"] == c["task"]:
                _, orig, _, _, _ = before(cid)
                for begin, stop in intervals(changed(orig[lo:hi], x[lo:hi])):
                    ax.axvspan(
                        t[lo + begin] - start,
                        t[lo + stop - 1] + 1 / fs - start,
                        ymin=0.88,
                        ymax=1,
                        color="#875893",
                        lw=0.3,
                    )
            for left, h in intervals((~v | e)[lo:hi]):
                ax.axvspan(
                    t[lo + left] - start,
                    t[lo + h - 1] + 1 / fs - start,
                    facecolor="#85858522",
                    edgecolor="#78848b",
                    hatch="///",
                    lw=0.3,
                )
            ax.set_ylabel(peer["task"] + "\n" + peer["unit"])
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(
                    ("Parent" if not col else "Candidate")
                    if after
                    else "Observed samples (zoom)"
                    if detail and col
                    else "Observed waveforms"
                )
            native = []
            for i, j, z in native_rows(pr):
                if i < b and a < j:
                    native.append(
                        dict(
                            key=f"{cid}:{i}:{j}",
                            **{k: round(z[k], 4) for k in ("interval_s", "sbp", "dbp", "mean_pressure") if k in z},
                        )
                    )
            metrics[("after" if col else "before") + "_" + peer["task"]] = native
    axes[-1, 0].set_xlabel("Seconds from target onset")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    fig.clear()
    return metrics


def change_ranges(before, after):
    _, x, v, e, _ = before
    _, y, v2, e2, _ = after
    return intervals(changed(x, y) | (v != v2) | (e != e2))


def enrich(cases, data, profiles, hits):
    by_block = defaultdict(dict)
    for c in cases.values():
        by_block[acquisition(c)][c["task"]] = c
    grouped = defaultdict(list)
    for h in hits:
        grouped[h["case"]].append(h)
    for block in by_block.values():
        signals = {}
        for task, c in block.items():
            t, x, v, e, pr = data(c["id"])
            pairs = np.asarray(beat_pairs(pr), int).reshape(-1, 2)
            starts, ends = pairs.T
            alarm = np.zeros(len(x), bool)
            for h in grouped[c["id"]]:
                alarm[h["a"] : h["b"]] = True
            signals[task] = (
                t,
                x,
                v & ~e,
                starts,
                ends,
                60 * c["fs"] / (ends - starts),
                np.asarray([z["sample"] for z in pr["events"]]),
                alarm,
            )
        for task, c in block.items():
            t, x, good, starts, ends, bpm, landmarks, _ = signals[task]
            amp = profiles[task]["amplitude"]
            scale = amp["typical_swing"] if amp else 0
            for h in grouped[c["id"]]:
                a, b = h["a"], h["b"]
                mid = (a + b - 1) / 2
                raw = x[a:b][good[a:b]]
                left = max(0, a - 1)
                right = min(len(x), b + 1)
                step = np.abs(np.diff(x[left:right]))[good[left : right - 1] & good[left + 1 : right]]
                own = bpm[ends.searchsorted(a, "right") : starts.searchsorted(b)]
                ix = landmarks.searchsorted(mid)
                near = landmarks[max(0, ix - 1) : ix + 1]
                rates = {}
                alarms = 0
                for peer, (pt, _, _, ps, pe, pb, _, flags) in signals.items():
                    if peer == task:
                        continue
                    lo, hi = pt.searchsorted([t[a] - 1e-7, t[b - 1] + 1 / c["fs"] - 1e-7])
                    vals = pb[pe.searchsorted(lo, "right") : ps.searchsorted(hi)] if lo < hi else []
                    if len(vals):
                        rates[peer] = float(np.median(vals))
                    alarms += int(flags[lo:hi].any())
                h["features"] = dict(
                    seconds=(b - a) / c["fs"],
                    step_norm=float(step.max() / scale) if len(step) and scale else None,
                    span_norm=float(np.ptp(raw) / scale) if len(raw) and scale else None,
                    beat_distance_ms=float(np.min(abs(near - mid)) * 1000 / c["fs"]) if len(near) else None,
                    bpm_min=float(own.min()) if len(own) else None,
                    bpm_max=float(own.max()) if len(own) else None,
                    peer_count=len(rates),
                    peer_alerts=alarms,
                )
                h["peer_bpm"] = rates


def jev_routes(workspace, config, hits):
    """One typed routing judgment per original hit. Only derived numbers are transmitted."""
    import json
    import os
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    from .common import read, write

    outcomes = {}
    batches = defaultdict(list)
    for h in hits:
        batches[h["task"], h["reason"]].append(h)
    ledger_file = workspace / "jev_usage.json"
    ledger = read(ledger_file) if ledger_file.exists() else {"reported_tokens": 0, "reserved_tokens": 0, "requests": 0}
    for (task, reason), members in sorted(batches.items()):
        for i in range(0, len(members), 24):
            check_stop(workspace)
            batch = members[i : i + 24]
            state = (
                f"Channel {task}, alert {reason}. Decide the review route from each separate row. True if "
                "a local waveform or local detector issue is supported with coherent peers. False if "
                "synchronous physiological change or insufficient evidence warrants joint review. Values "
                "step_norm and span_norm are relative to the patient's typical native beat swing. Do not "
                "diagnose."
            )
            questions = {
                f"r{j}": {
                    "type": "boolean",
                    "instructions": json.dumps(dict(**h["features"], peer_bpm=h["peer_bpm"], trigger=h["value"])),
                }
                for j, h in enumerate(batch)
            }
            body = dict(model="jev-1.14.0", state=state, questions=questions)
            binding = digest([body, [h["id"] for h in batch]])
            path = workspace / "jev-receipts" / (binding + ".json")
            replay = Path(config["replay_from"]) / "jev-receipts" / path.name if config.get("replay_from") else None
            cached = Path(config["receipt_seed"]) / "jev-receipts" / path.name if config.get("receipt_seed") else path
            if path.exists() or replay or cached.is_file():
                rec = read(path if path.exists() else replay if replay else cached)
                if rec["binding"] != binding:
                    raise ValueError("JEV evidence changed")
                if not path.exists():
                    write(path, rec)
                    ledger["reused_tokens"] = ledger.get("reused_tokens", 0) + sum(
                        rec["usage"].get(k, 0) for k in ("input_tokens", "output_tokens")
                    )
                    write(ledger_file, ledger)
            else:
                reserve = 24000
                if ledger["reported_tokens"] + ledger["reserved_tokens"] + reserve > config["jev_token_cap"]:
                    outcomes.update({h["id"]: None for h in batch})
                    continue
                key = os.environ.get("WULUSAI_API_KEY")
                if not key:
                    raise ValueError("JEV routing requires WULUSAI_API_KEY or matching saved receipts")
                ledger["reserved_tokens"] += reserve
                ledger["requests"] += 1
                write(ledger_file, ledger)
                request = urllib.request.Request(
                    "https://api.wulusai.com/v1/evaluate",
                    data=json.dumps(body).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "SignalMind-dataset/1.0",
                        "Authorization": "Bearer " + key,
                    },
                )
                with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
                    payload = json.load(response)
                if payload.get("model") != body["model"] or set(payload["answers"]) != set(questions):
                    raise ValueError("Incomplete JEV response")
                rec = dict(
                    binding=binding,
                    scores={h["id"]: payload["answers"][f"r{j}"]["probability"] for j, h in enumerate(batch)},
                    usage=payload["usage"],
                )
                if set(rec["scores"]) != {h["id"] for h in batch} or not all(
                    isinstance(v, (int, float)) and 0 <= v <= 1 for v in rec["scores"].values()
                ):
                    raise ValueError("Invalid JEV scores")
                write(path, rec)
                ledger["reserved_tokens"] -= reserve
                ledger["reported_tokens"] += sum(rec["usage"].get(k, 0) for k in ("input_tokens", "output_tokens"))
                write(ledger_file, ledger)
            if set(rec["scores"]) != {h["id"] for h in batch} or not all(
                isinstance(v, (float, int)) and 0 <= v <= 1 for v in rec["scores"].values()
            ):
                raise ValueError("Missing JEV routing rows")
            outcomes.update(rec["scores"])
    return outcomes
