"""Build the p10abp development suites (pilot: 4 items; main: 25 items) for the C_tools vs C_full comparison.

  .venv/bin/python -m eval.tasks.p10abp --build      # writes eval/cases/p10pilot*, eval/cases/p10abp*, manifests
  .venv/bin/python -m eval.tasks.p10abp --check      # recomputes references and verifies the frozen manifests

Sources: opened development records (eval/cases/p9level2/*/signal.npy, real-signal items only, ABP + ECG present,
ABP not flat). References come from two implementations that are NOT the plugin under test: A (scipy peaks on raw
samples) and B (the user's original scipy guided detector); an item is retained only when A and B agree within half
the tolerance. Controls remove the evidence (constant or NaN ABP over the asked interval) and are labelled as such.
TaskSpec files are written from the question's public fields only (channels, interval(s), unit, quantities).
Nothing here reads p9level2 reference answers, rubrics or tolerances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.validate_pressure import load_original, reference_a, reference_b  # noqa: E402
from tsmcp.runtime.taskspec import Budgets, QuantitySpec, TaskSpec, abp_interval_spec  # noqa: E402

CASES = ROOT / "eval" / "cases"
SRC = CASES / "p9level2"
SEED = 20261028
TOL = 3.0  # mmHg on every reported pressure mean / change
HALF = TOL / 2
GENERATOR_VERSION = "p10abp.v1"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sources() -> list[dict]:
    """Real-signal development records with ABP and ECG; structure only (dataset, subject, channels)."""
    man = json.loads((ROOT / "eval/tasks/p9level2_manifest.json").read_text())["questions"]
    out = []
    for q in man:
        if q["signal_kind"] != "real_signal" or "ABP" not in q["channels"] or "ECG" not in q["channels"]:
            continue
        d = SRC / q["case_id"]
        if not (d / "signal.npy").exists():
            continue
        arr = np.load(d / "signal.npy", allow_pickle=False)
        v = json.loads((d / "visible.json").read_text())
        names = [c["name"] for c in v["channels"]]
        abp = arr[:, names.index("ABP")]
        if not np.isfinite(abp).any() or np.nanstd(abp) < 1.0:
            continue
        out.append({"src": q["case_id"], "dataset": q["dataset"], "subject": q["subject_key"], "fs": v["channels"][0]["fs"],
                    "dur": v["dur_s"], "names": names, "visible": v, "arr": arr})  # fmt: skip
    out.sort(key=lambda r: r["src"])
    return out


def refs(rec, t0, t1, mod):
    names = rec["names"]
    abp, ecg = rec["arr"][:, names.index("ABP")], rec["arr"][:, names.index("ECG")]
    a = reference_a(abp, rec["fs"], t0, t1)
    b = reference_b(mod, abp, ecg, rec["fs"], t0, t1)
    return a, b


def agree(a, b, key) -> bool:
    return a is not None and b is not None and abs(a[key] - b[key]) <= HALF


def question_text(kind, iv, dur, names, extra=""):
    layout = (
        "The array is samples x channels: "
        + ", ".join(f"column {i} = {n}" for i, n in enumerate(names))
        + ". When loading, assign these channel names in this order."
    )
    if kind in ("interval_mean", "control_constant", "control_nan", "precision"):
        t0, t1 = iv[0]
        q = (
            f"What are the mean per-beat systolic (SBP) and diastolic (DBP) arterial pressures in mmHg over "
            f"{t0:g}-{t1:g} s of the ABP channel? Report sbp_mean and dbp_mean as means over complete beats. "
        )
    elif kind == "change":
        (a0, a1), (b0, b1) = iv
        q = (
            f"By how much does the mean per-beat systolic arterial pressure (mmHg) change from {a0:g}-{a1:g} s "
            f"(baseline) to {b0:g}-{b1:g} s (later minus baseline) in the ABP channel? Report sbp_change. "
        )
    else:
        raise ValueError(kind)
    if kind in ("control_constant", "control_nan"):
        q += "If the evidence is insufficient, say so instead of a number. "
    q += extra
    return q + f"Times are seconds from the start of this {dur:g} s case. " + layout


def spec_for(kind, iv, names, case_id, question) -> TaskSpec:
    ecg = "ECG" if "ECG" in names else None
    if kind == "change":
        spec = abp_interval_spec(case_id, question, "ABP", iv[0], quantities=("sbp_mean",), ecg_channel=ecg)
        spec.quantities = [QuantitySpec("sbp_change", "abp.sbp.beat_mean_change.v1", "mmHg")]
        spec.interval = None
        spec.conventions["intervals"] = f"baseline {list(iv[0])}, later {list(iv[1])}, change = later - baseline"
        return spec
    spec = abp_interval_spec(case_id, question, "ABP", iv[0], quantities=("sbp_mean", "dbp_mean"), ecg_channel=ecg,
                             budgets=Budgets())  # fmt: skip
    if kind == "precision":
        spec.quantities = [
            QuantitySpec(q.name, q.definition_id, q.unit, precision_decimals=None) for q in spec.quantities
        ]
    return spec


def build_item(rec, kind, iv, mod, rng, extra="") -> dict | None:
    names, fs, dur = rec["names"], rec["fs"], rec["dur"]
    arr = rec["arr"].copy()
    control = None
    if kind in ("control_constant", "control_nan"):
        t0, t1 = iv[0]
        i0, i1 = int(t0 * fs), int(t1 * fs)
        j = names.index("ABP")
        if kind == "control_constant":
            arr[i0:i1, j] = float(np.nanmedian(arr[i0:i1, j]))
            control = {"type": "constant_abp", "interval": [t0, t1], "value": float(arr[i0, j])}
        else:
            arr[i0:i1, j] = np.nan
            control = {"type": "nan_abp", "interval": [t0, t1]}
        rec2 = {**rec, "arr": arr}
        a, b = refs(rec2, t0, t1, mod)
        if a is not None or b is not None:
            return None  # the evidence removal must leave both references without beats
        expected = {"sbp_mean": None, "dbp_mean": None}
        references = {"A": None, "B": None}
    elif kind == "change":
        (a0, a1), (b0, b1) = iv
        aa, ab = refs(rec, a0, a1, mod)
        ba, bb = refs(rec, b0, b1, mod)
        if not (agree(aa, ab, "sbp") and agree(ba, bb, "sbp")):
            return None
        ch_a, ch_b = ba["sbp"] - aa["sbp"], bb["sbp"] - ab["sbp"]
        if abs(ch_a - ch_b) > HALF or abs(ch_a) < 4.0:  # a real, agreed change; "no change" is not a shortcut
            return None
        expected = {"sbp_change": round(ch_a, 2)}
        references = {
            "A": {"baseline": aa, "later": ba, "change": ch_a},
            "B": {"baseline": ab, "later": bb, "change": ch_b},
        }
    else:
        t0, t1 = iv[0]
        a, b = refs(rec, t0, t1, mod)
        if not (agree(a, b, "sbp") and agree(a, b, "dbp")) or a["n"] < 30:
            return None
        expected = {"sbp_mean": round(a["sbp"], 2), "dbp_mean": round(a["dbp"], 2)}
        references = {"A": a, "B": b}
    question = question_text(kind, iv, dur, names, extra)
    body = np.ascontiguousarray(arr.astype(np.float32)).tobytes()
    case_id = sha(body + question.encode())[:24]
    fields = list(expected)
    return {
        "case_id": case_id, "kind": kind, "category": "control" if control else ("verification" if kind in ("change", "scope")
                                                                                else "ordinary"),
        "source_case": rec["src"][:8], "dataset": rec["dataset"], "subject_key": rec["subject"], "fs": fs, "duration_s": dur,
        "channels": names, "intervals": [list(i) for i in iv], "question": question, "answer_fields": fields,
        "expected": expected, "tolerance": {f: TOL for f in fields}, "references": references,
        "agreement_rule": f"A and B within {HALF} mmHg", "control": control, "arr": arr, "visible": rec["visible"],
        "spec": spec_for(kind, iv, names, case_id, question).to_dict(), "generator": GENERATOR_VERSION,
    }  # fmt: skip


def plan_items(recs, mod, rng) -> tuple[list[dict], list[dict]]:
    """Deterministic assignment: pilot takes 4 records first (distinct subjects); main draws from the rest."""
    by_subject: dict[str, list[dict]] = {}
    for r in recs:
        by_subject.setdefault(r["subject"], []).append(r)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    pilot, main = [], []
    used_subjects_pilot = set()

    def window(rec, length, lo=10.0):
        hi = rec["dur"] - 10.0 - length
        return (round(rng.uniform(lo, max(lo, hi)), 1),) if hi >= lo else None

    # ---- pilot: valid measurement, scope correction (long record), unsupported precision, explicit limitation
    wanted = [("interval_mean", False), ("scope", True), ("precision", False), ("control_constant", False)]
    for kind, need_long in wanted:
        for subj in subjects:
            if subj in used_subjects_pilot:
                continue
            rec = by_subject[subj][0]
            if need_long and rec["dur"] < 500:
                continue
            k = "interval_mean" if kind == "scope" else kind
            length = 60.0
            w = window(rec, length)
            if w is None:
                continue
            iv = [(w[0], w[0] + length)]
            extra = "Report both values to 0.01 mmHg. " if kind == "precision" else ""
            item = build_item(rec, k, iv, mod, rng, extra)
            if kind == "scope" and item is not None:  # the whole-record mean must not pass
                whole_a, whole_b = refs(rec, 10.0, rec["dur"] - 10.0, mod)
                if whole_a is None or abs(whole_a["sbp"] - item["expected"]["sbp_mean"]) < TOL + 1:
                    item = None
            if item is not None:
                item["pilot_role"] = kind
                item["kind"] = kind
                pilot.append(item)
                used_subjects_pilot.add(subj)
                break
    # ---- main: 13 ordinary, 4 change, 3 scope (long records), 5 controls (3 constant, 2 NaN)
    quota = {"interval_mean": 13, "change": 4, "scope": 3, "control_constant": 3, "control_nan": 2}
    pool = [r for s_ in subjects if s_ not in used_subjects_pilot for r in by_subject[s_]]
    attempts = 0
    while sum(quota.values()) and attempts < 400:
        attempts += 1
        rec = pool[attempts % len(pool)]
        kinds = [k for k, n in quota.items() if n > 0]
        kind = kinds[attempts % len(kinds)]
        if kind in ("change", "scope") and rec["dur"] < 500:
            continue
        if sum(1 for m in main if m["source_case"] == rec["src"][:8]) >= 2:
            continue
        if kind == "change":
            a0 = round(rng.uniform(10, rec["dur"] - 200), 1)
            b0 = round(rng.uniform(a0 + 100, rec["dur"] - 50), 1)
            iv = [(a0, a0 + 40), (b0, b0 + 40)]
            item = build_item(rec, "change", iv, mod, rng)
        else:
            w = window(rec, 60.0)
            if w is None:
                continue
            iv = [(w[0], w[0] + 60.0)]
            k = "interval_mean" if kind == "scope" else kind
            item = build_item(rec, k, iv, mod, rng)
            if kind == "scope" and item is not None:
                whole_a, _ = refs(rec, 10.0, rec["dur"] - 10.0, mod)
                if whole_a is None or abs(whole_a["sbp"] - item["expected"]["sbp_mean"]) < TOL + 1:
                    item = None
        if item is None:
            continue
        if any(m["source_case"] == item["source_case"] and overlaps(m["intervals"], item["intervals"]) for m in main):
            continue
        item["kind"] = kind
        item["category"] = (
            "control" if kind.startswith("control") else ("verification" if kind in ("change", "scope") else "ordinary")
        )
        main.append(item)
        quota[kind] -= 1
    return pilot, main


def overlaps(a, b) -> bool:
    return any(x[0] < y[1] and y[0] < x[1] for x in a for y in b)


def write_suite(name: str, items: list[dict]) -> dict:
    root = CASES / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    qs, ans, man = [], [], []
    for it in items:
        d = root / it["case_id"]
        d.mkdir()
        np.save(d / "signal.npy", it["arr"].astype(np.float32), allow_pickle=False)
        vis = {"channels": it["visible"]["channels"], "dur_s": it["duration_s"], "file": "signal.npy"}
        (d / "visible.json").write_text(json.dumps(vis, indent=2))
        (d / "spec.json").write_text(json.dumps(it["spec"], indent=1))
        qs.append(
            {
                "case_id": it["case_id"],
                "qtype": "signalmind",
                "question": it["question"],
                "answer_fields": it["answer_fields"],
            }
        )
        ans.append({"case_id": it["case_id"], "expected": it["expected"], "tolerance": it["tolerance"], "control": it["control"],
                    "kind": it["kind"], "category": it["category"]})  # fmt: skip
        man.append({k: v for k, v in it.items() if k not in ("arr", "visible")}
                   | {"signal_sha256": sha((d / "signal.npy").read_bytes()), "visible_sha256": sha((d / "visible.json").read_bytes()),
                      "spec_sha256": sha((d / "spec.json").read_bytes())})  # fmt: skip
    (CASES / f"{name}.questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    (CASES / f"{name}.answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in ans))
    manifest = {"suite": name, "generator": GENERATOR_VERSION, "seed": SEED, "tolerance_mmHg": TOL, "items": man,
                "counts": {k: sum(1 for m in man if m["kind"] == k) for k in sorted({m["kind"] for m in man})},
                "categories": {k: sum(1 for m in man if m["category"] == k) for k in sorted({m["category"] for m in man})},
                "subjects": len({m["subject_key"] for m in man}), "sources": len({m["source_case"] for m in man})}  # fmt: skip
    (ROOT / "eval/tasks" / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mod = load_original()
    if mod is None:
        raise SystemExit("reference B (original implementation) is not available; cannot build references")
    recs = sources()
    rng = random.Random(SEED)
    pilot, main_items = plan_items(recs, mod, rng)
    if args.build:
        mp = write_suite("p10pilot", pilot)
        mm = write_suite("p10abp", main_items)
        print(json.dumps({"pilot": {k: mp[k] for k in ("counts", "subjects", "sources")},
                          "main": {k: mm[k] for k in ("counts", "categories", "subjects", "sources")}}, indent=1))  # fmt: skip
    if args.check:
        for name, items in (("p10pilot", pilot), ("p10abp", main_items)):
            frozen = json.loads((ROOT / "eval/tasks" / f"{name}_manifest.json").read_text())
            got = [{k: v for k, v in it.items() if k not in ("arr", "visible")} for it in items]
            same = [m["case_id"] for m in frozen["items"]] == [g["case_id"] for g in got] and all(
                m["expected"] == g["expected"]
                and m["references"] == json.loads(json.dumps(g["references"], default=str))
                for m, g in zip(frozen["items"], got, strict=True)
            )
            files = all(
                sha((CASES / name / m["case_id"] / "signal.npy").read_bytes()) == m["signal_sha256"]
                for m in frozen["items"]
            )
            print(name, "manifest reproduced:", same, "files intact:", files)


if __name__ == "__main__":
    main()
