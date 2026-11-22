"""Independent, per-operation visual gate on a staged native-waveform batch; publish binding manifests."""

import argparse
import base64
import csv
import fcntl
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from jsonschema import ValidationError, validate
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tsmcp.cleaning.agent import BudgetClient
from tsmcp.cleaning.common import array_hash, changed, digest, read, sha, write
from tsmcp.cleaning.engine import load_arrays
from tsmcp.followup.reviewer import tool
from tsmcp.processing import intervals

ROW = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "verdict": {"enum": ["accept", "accept_with_limits", "hold"]},
        "observation": {"type": "string", "minLength": 1},
        "limitation": {"type": "string", "minLength": 1},
    },
    "required": ["id", "verdict", "observation", "limitation"],
    "additionalProperties": False,
}
SUBMIT = tool("submit", "Assess every labeled range separately.", {"items": {"type": "array", "items": ROW}})


def pack_targets(targets, fs):
    cards = []
    for target in sorted(targets, key=lambda z: (z["a"], z["b"], z["id"])):
        if (
            cards
            and target["a"] - cards[-1]["b"] <= round(5 * fs)
            and max(target["b"], cards[-1]["b"]) - cards[-1]["a"] <= round(20 * fs)
        ):
            cards[-1]["b"] = max(cards[-1]["b"], target["b"])
            cards[-1]["targets"].append(target)
        else:
            cards.append(dict(a=target["a"], b=target["b"], targets=[target]))
    return cards


def check_group(client, folder, group_id, cards):
    path = folder / (group_id + ".json")
    binding = digest([{k: v for k, v in c.items() if k != "file"} for c in cards])
    if path.exists():
        result = read(path)
        if result["binding"] != binding:
            raise ValueError("Candidate evidence changed")
        return result
    content = [
        dict(
            type="text",
            text="独立批后复核。以前后图和探针变化为依据。逐框作决定：accept 接受该操作/保留；accept_with_limits 用途可接受但记限制；hold 该候选应回退或延期（例如误删清楚的搏动，过宽停用，制造新波形，仍有无法解决的风险）。对残余/对照如仍不宜使用，请记明证据和局限。原始数值相同但任务掩码可不同。图上方原始，下方候选，虚线框为逐项复核范围，灰斜线为停用。原始探针也可能误检。无干净参照，不可将探针变化直接当作错误标签。每个独立范围一条 verdict，中文。",
        )
    ]
    ids = set()
    for c in cards:
        pic = (folder / c["file"]).read_bytes()
        if sha(folder / c["file"]) != c["image_sha256"]:
            raise ValueError("Postreview figure changed")
        content += [
            dict(type="text", text=json.dumps(c["metrics"], ensure_ascii=False)),
            dict(type="image", source=dict(type="base64", media_type="image/png", data=base64.b64encode(pic).decode())),
        ]
        ids.update(i["id"] for i in c["metrics"]["items"])
    messages = [dict(role="user", content=content)]
    for _ in range(3):
        response = client.request(
            messages,
            [SUBMIT],
            system="You are an independent waveform reviewer. Examine the actual images. Use submit for all listed IDs. Neither detector matches nor model proposals are independent truth.",
        )
        parts = [z.model_dump(exclude_none=True) for z in response.content]
        calls = [z for z in parts if z["type"] == "tool_use"]
        try:
            if len(calls) != 1 or calls[0]["name"] != "submit":
                raise ValueError("Use submit once")
            val = calls[0]["input"]
            validate(val, SUBMIT["input_schema"])
            if sorted(r["id"] for r in val["items"]) != sorted(ids):
                raise ValueError("Include all IDs exactly once")
            result = dict(
                binding=binding,
                request_id=response.id,
                model=client.model,
                images=[c["image_sha256"] for c in cards],
                **val,
            )
            write(path, result)
            return result
        except (ValidationError, ValueError) as exc:
            messages.extend(
                [
                    dict(role="assistant", content=parts),
                    dict(
                        role="user",
                        content=[
                            dict(type="tool_result", tool_use_id=c["id"], content=str(exc)[:450], is_error=True)
                            for c in calls
                        ]
                        or "Call submit",
                    ),
                ]
            )
    raise RuntimeError("No valid independent verdict")


def run(workspace, out, budget, *, extras=()):
    w, out, budget = map(Path, (workspace, out, budget))
    folder = out / "operation-review"
    folder.mkdir(parents=True, exist_ok=True)
    state = read(w / "agent_state.json")
    if not state.get("complete"):
        raise ValueError("No completed candidate batch")
    manifest = read(w / "dataset_manifest.json")
    cases = {c["id"]: c for c in manifest["cases"]}
    base = read(w / "versions" / read(w / "baselines.json")["identity"] / "version.json")
    versions = {k: read(w / "versions" / vid / "version.json") for k, vid in state["accepted_tasks"].items()}
    targets = defaultdict(list)
    ops = []
    cards = []
    locator = {}

    def add(cid, start, end, kind, parameters=None):
        if not 0 <= start < end <= cases[cid]["n"]:
            raise ValueError("Invalid target")
        oid = f"O{len(ops) + 1:04}" if kind == "operation" else f"V{len(locator) + 1:04}"
        item = dict(id=oid, case=cid, a=int(start), b=int(end), kind=kind, parameters=parameters or {})
        if kind == "operation":
            ops.append(item)
        locator[oid] = item
        chunk = max(1, round(20 * cases[cid]["fs"]))
        for a in range(start, end, chunk):
            sub = dict(
                item,
                id=oid if end - start <= chunk else f"{oid}.{1 + (a - start) // chunk}",
                parent_id=oid,
                a=int(a),
                b=int(min(end, a + chunk)),
            )
            targets[cid].append(sub)

    @lru_cache(maxsize=8)
    def data(cid):
        c = cases[cid]
        orig = base["results"][cid]
        art = versions[c["task"]]["results"][cid]
        raw, t, y, valid, est = load_arrays(w, c, art)
        _, _, _, v0, e0 = load_arrays(w, c, orig)
        for z, vals, v, e in ((orig, raw, v0, e0), (art, y, valid, est)):
            if array_hash(vals, t, v, e) != z["content_hash"] or sha(w / z["probe_path"]) != z["probe_sha256"]:
                raise ValueError("Version/probe hash mismatch")
        return raw, t, y, v0, e0, valid, est, read(w / orig["probe_path"]), read(w / art["probe_path"])

    for task, version in versions.items():
        if not version["report"]["program_checks_passed"] or set(version["results"]) != {
            k for k, c in cases.items() if c["task"] == task
        }:
            raise ValueError("Incomplete candidate scope or checks")
        for cid, ar in sorted(version["results"].items()):
            c = cases[cid]
            if sha(w / "public" / cid / "input.npz") != c["input_sha256"]:
                raise ValueError("Raw source mismatch")
            allowed = np.zeros(c["n"], bool)
            link = ar
            chain = []
            while link:
                chain.append(link)
                parent = link.get("parent_artifact_id")
                link = read(w / "artifacts" / parent / "artifact.json") if parent else None
            for stage in reversed(chain):
                for op in stage["operations"]:
                    for a, b in op["ranges_samples"]:
                        add(
                            cid,
                            a,
                            b,
                            "operation",
                            dict(op=op["op"], params=op["parameters"], recipe=stage["recipe_hash"]),
                        )
                        allowed[a:b] = True
            raw, t, y, v0, e0, v, e, _, _ = data(cid)
            if np.any((changed(raw, y) | (v0 != v) | (e0 != e)) & ~allowed):
                raise ValueError("Undeclared change")
    for item in extras:
        add(item["case"], item["a"], item["b"], item["kind"], item.get("parameters"))
    # Include at least one unchanged control in each modality, even if the candidate is identity.
    for task in versions:
        eligible = [
            c
            for c in cases.values()
            if c["task"] == task and not any(z["kind"] == "operation" for z in targets[c["id"]])
        ]
        if eligible:
            c = min(eligible, key=lambda c: digest([111101, "post", c["id"]]))
            mid = c["n"] // 2
            rad = min(mid, round(c["fs"]))
            if rad:
                add(c["id"], mid - rad, mid + rad, "unchanged_control")

    def peaks(probe, a, b):
        return [r["sample"] for r in probe["events"] if a <= r["sample"] < b]

    def beats(probe, a, b):
        return [
            [z["start_sample"], z["stop_sample"], round(z["sbp"], 2), round(z["dbp"], 2), round(z["mean_pressure"], 2)]
            for z in probe["pressures"]
            if z["start_sample"] < b and a < z["stop_sample"]
        ]

    def periods(probe, a, b):
        events = probe["events"]
        return [
            [events[z["left_event_id"]]["sample"], events[z["right_event_id"]]["sample"], round(z["interval_s"], 4)]
            for z in probe["intervals"]
            if events[z["left_event_id"]]["sample"] < b and a < events[z["right_event_id"]]["sample"]
        ]

    origin = min(float(data(k)[1][0]) for k in cases)
    with rc_context({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False}):
        for cid, spans in sorted(targets.items()):
            if not spans:
                continue
            c = cases[cid]
            fs = c["fs"]
            raw, t, y, v0, e0, valid, est, before, after = data(cid)
            off0 = ~(v0 & ~e0)
            off = ~(valid & ~est)
            art = versions[c["task"]]["results"][cid]
            for card in pack_targets(spans, fs):
                a, b = card["a"], card["b"]
                left = max(0, a - round(5 * fs))
                right = min(len(y), b + round(5 * fs))
                page = int(np.floor((t[a] - origin) / 60)) + 1
                offset = origin + (page - 1) * 60
                card_id = f"C{len(cards) + 1:03}"
                fig = Figure(figsize=(10, 4.9), layout="constrained")
                FigureCanvasAgg(fig)
                axes = fig.subplots(2, 1, sharex=True, sharey=True)
                visible = np.concatenate((raw[left:right], y[left:right]))
                finite = visible[np.isfinite(visible)]
                lo, hi = (finite.min(), finite.max()) if len(finite) else (0, 1)
                margin = max(0.01, (hi - lo) * 0.13)
                display = manifest.get("patient_display_limits", {}).get(c["task"], [lo - margin, hi + margin])
                for ax, arr, bad, probe, title, col in zip(
                    axes,
                    (raw, y),
                    (off0, off),
                    (before, after),
                    ("原始", "批量候选"),
                    ("#275570", "#067663"),
                    strict=True,
                ):
                    ax.plot(t[left:right] - offset, arr[left:right], lw=0.9, color=col)
                    for q, z in intervals(bad[left:right]):
                        ax.axvspan(
                            t[left + q] - offset,
                            t[left + z - 1] - offset + 1 / fs,
                            facecolor="#66717a22",
                            edgecolor="#69747c",
                            hatch="///",
                            lw=0.5,
                        )
                    for i, slot in enumerate(card["targets"]):
                        lo_sample, h = slot["a"], slot["b"]
                        ax.axvspan(
                            t[lo_sample] - offset,
                            t[h - 1] - offset + 1 / fs,
                            facecolor="#d8a74d17",
                            edgecolor="#ad772a",
                            ls="--",
                            lw=0.75,
                        )
                        if ax is axes[0]:
                            ax.text(
                                (t[lo_sample] + t[h - 1]) / 2 - offset,
                                0.94 - 0.11 * (i % 3),
                                slot["id"],
                                fontsize=7,
                                ha="center",
                                transform=ax.get_xaxis_transform(),
                            )
                    pp = peaks(probe, left, right)
                    if pp:
                        ax.plot(t[pp] - offset, arr[pp], ".", ms=3, color="#8b4156")
                    ax.set(
                        ylabel=c["unit"],
                        ylim=display,
                        xlim=(t[left] - offset, t[right - 1] - offset),
                    )
                    ax.set_title(title, loc="left", fontsize=9)
                    if "patient_reference" in c:
                        amp = c["patient_reference"]["amplitude"]
                        for bound in (amp["low"], amp["high"]):
                            ax.axhline(bound, ls="--", color="#a53456", lw=0.7)
                        for limit, flags, marker in (
                            (display[0], arr[left:right] < display[0], "v"),
                            (display[1], arr[left:right] > display[1], "^"),
                        ):
                            if flags.any():
                                ax.plot(
                                    t[left:right][flags] - offset,
                                    np.full(flags.sum(), limit),
                                    marker,
                                    ms=4,
                                    color="#a53456",
                                    clip_on=False,
                                )
                axes[-1].set_xlabel(f"第 {page} 分钟内时间（秒）")
                fig.suptitle(f"{card_id}  ·  {c['task']}   红点为探针事件，斜线为任务停用", fontsize=12)
                path = folder / (card_id + ".png")
                fig.savefig(path, dpi=115)
                fig.clear()
                items = []
                for slot in card["targets"]:
                    lo_sample, h = slot["a"], slot["b"]
                    desc = dict(
                        slot,
                        page=page,
                        page_range_s=[
                            round(float(t[lo_sample] - offset), 3),
                            round(float(t[h - 1] - offset + 1 / fs), 3),
                        ],
                        changed_samples=int(changed(raw[lo_sample:h], y[lo_sample:h]).sum()),
                        newly_excluded=int((off[lo_sample:h] & ~off0[lo_sample:h]).sum()),
                        estimated=int(est[lo_sample:h].sum()),
                        native_events_before=peaks(before, lo_sample, h),
                        native_events_after=peaks(after, lo_sample, h),
                    )
                    if c["task"] == "ABP":
                        desc.update(
                            foot_cycles_SBP_DBP_MAP_before=beats(before, lo_sample, h),
                            foot_cycles_SBP_DBP_MAP_after=beats(after, lo_sample, h),
                        )
                    else:
                        desc.update(
                            intervals_before=periods(before, lo_sample, h), intervals_after=periods(after, lo_sample, h)
                        )
                    items.append(desc)
                meta = dict(
                    card_id=card_id,
                    task=c["task"],
                    fs=fs,
                    content_hash=art["content_hash"],
                    case=cid,
                    view_samples=[left, right],
                    original_probe_sha=base["results"][cid]["probe_sha256"],
                    candidate_probe_sha=art["probe_sha256"],
                    context_events_before=peaks(before, left, right),
                    context_events_after=peaks(after, left, right),
                    items=items,
                )
                cards.append(dict(file=path.name, image_sha256=sha(path), metrics=meta))
    write(folder / "cards.json", cards)
    write(folder / "operations.json", ops)
    verdicts = {}
    budget_out = False
    with (budget / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = BudgetClient(w, budget_root=budget)
        try:
            for i in range(0, len(cards), 2):
                try:
                    result = check_group(client, folder, f"group-{i // 2:03}", cards[i : i + 2])
                except RuntimeError as exc:
                    if "api_budget_exhausted" not in str(exc):
                        raise
                    budget_out = True
                    break
                for z in result["items"]:
                    verdicts[z["id"]] = dict(z, request_id=result["request_id"])
                print("postreview cards", i + 1, min(i + 2, len(cards)), "/", len(cards), flush=True)
        finally:
            client.close()
        write(out / "close-budget.json", read(budget / "api_budget.json"))
    holds = defaultdict(list)
    statuses = {}
    for cid, items in targets.items():
        for item in items:
            v = verdicts.get(item["id"])
            verdict = v["verdict"] if v else "not_reviewed"
            statuses[item["id"]] = verdict
            if verdict in ("hold", "not_reviewed"):
                holds[cid].append(item["id"])
    records = {}
    totals = defaultdict(Counter)
    decision_rows = []
    for cid, c in cases.items():
        ar = versions[c["task"]]["results"][cid]
        parent = base["results"][cid]
        held = bool(holds[cid])
        chosen = parent if held else ar
        raw, t, y, v0, e0, v, e, _, _ = data(cid)
        if held:
            y, v, e = raw, v0, e0
        off0 = ~(v0 & ~e0)
        off = ~(v & ~e)
        totals[c["task"]].update(
            blocks=1,
            samples=c["n"],
            modified_samples=int(changed(raw, y).sum()),
            newly_excluded=int((off & ~off0).sum()),
            held_blocks=int(held),
        )
        records[cid] = dict(
            case=cid,
            task=c["task"],
            artifact_id=chosen["artifact_id"],
            source_sha256=c["input_sha256"],
            artifact_ref=str((w / "artifacts" / chosen["artifact_id"] / "artifact.json").resolve()),
            probe_ref=str((w / chosen["probe_path"]).resolve()),
            probe_sha256=chosen["probe_sha256"],
            content_hash=chosen["content_hash"],
            disposition="raw_preserved_pending_review" if held else "accepted_by_policy",
            holds=holds[cid],
        )
    for card in cards:
        for z in card["metrics"]["items"]:
            d = verdicts.get(z["id"], {})
            decision_rows.append(
                dict(
                    id=z["id"],
                    parent=z["parent_id"],
                    kind=z["kind"],
                    case=z["case"],
                    task=card["metrics"]["task"],
                    page=z["page"],
                    start_s=z["page_range_s"][0],
                    end_s=z["page_range_s"][1],
                    start_sample=z["a"],
                    end_sample=z["b"],
                    method=z["parameters"],
                    modified_trial=z["changed_samples"],
                    unusable_trial=z["newly_excluded"],
                    verdict=statuses[z["id"]],
                    released=records[z["case"]]["disposition"],
                    reason=d.get("observation", ""),
                    limitation=d.get("limitation", ""),
                    photo=card["file"],
                    request_id=d.get("request_id"),
                )
            )
    release = out / "delivery"
    release.mkdir(parents=True, exist_ok=True)
    with (release / "逐框复核.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)
    reviewed_parents = [
        op for op in ops if all(z["id"] in verdicts for z in targets[op["case"]] if z["parent_id"] == op["id"])
    ]
    summary = dict(
        source_scope=manifest["scope_hash"],
        tasks=totals,
        candidate_operations=len(ops),
        explicitly_reviewed_operations=len(reviewed_parents),
        published_operations=sum(not holds[op["case"]] for op in ops),
        cards=len(cards),
        all_item_verdicts=Counter(statuses.values()),
        block_holds={k: v for k, v in holds.items() if v},
        budget_exhausted=budget_out,
        reference_accuracy=None,
    )
    write(release / "waveform_manifest.json", records)
    write(release / "gate-summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["workspace", "out", "budget"]:
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--extras", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.workspace, args.out, args.budget, extras=read(args.extras) if args.extras else []),
            ensure_ascii=False,
        ),
        flush=True,
    )
