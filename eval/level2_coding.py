"""Blinded trace coding logistics (docs/tasks/level2-analysis-plan.md §4).

  .venv/bin/python eval/level2_coding.py prepare --traces eval/results/p9level2/traces --work /tmp/l2code --batches 4
  .venv/bin/python eval/level2_coding.py collect --work /tmp/l2code --out eval/results/p9level2/coding-codex
  .venv/bin/python eval/level2_coding.py sample --traces eval/results/p9level2/traces

`prepare` copies only `<session>.md` files and the codebook into each batch directory — never `index.json`, the
manifest or any answer — so the coder cannot see arm labels beyond what the trace shows, references or grades.
`collect` validates every coded file against the record schema and reports missing or malformed sessions.
`sample` draws the seeded audit sample (5 C, 5 D sessions; seed 20261027).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODEBOOK = ROOT / "docs" / "tasks" / "level2-coding-codebook.md"
FIELDS = {"trace_ref", "channels", "window_s", "computations", "method", "success", "verified", "independent_of",
          "detected_misleading_output", "quote"}  # fmt: skip
LABELS = {"beat_estimation", "pressure_pulsatility_change", "cross_channel_beat_alignment", "cross_channel_missing_beats",
          "cross_channel_polarity_check", "cardiac_pulse_rate_with_ecg_check", "cross_spectral_estimation",
          "sequence_brs", "ABP artefact interval and cross-channel quality verification", "signal_quality",
          "sustained_change_vs_transient"}  # fmt: skip


def validate(doc: dict, key: str) -> list[str]:
    problems = []
    if doc.get("session") != key or not isinstance(doc.get("records"), list):
        return ["session/records missing"]
    for k, r in enumerate(doc["records"]):
        if not isinstance(r, dict) or not FIELDS <= r.keys():
            problems.append(f"record {k}: missing fields")
            continue
        w = r["window_s"]
        if not (isinstance(w, list) and len(w) == 2 and all(isinstance(x, (int, float)) for x in w) and w[1] > w[0]):
            problems.append(f"record {k}: bad window")
        if not isinstance(r["channels"], list) or not r["channels"]:
            problems.append(f"record {k}: bad channels")
        if not isinstance(r["computations"], list) or set(r["computations"]) - LABELS:
            problems.append(f"record {k}: unknown computation label")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--batches", type=int, default=4)
    c = sub.add_parser("collect")
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sample")
    s.add_argument("--traces", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "prepare":
        keys = sorted(t.stem for t in args.traces.glob("*.md"))
        random.Random(20261027).shuffle(keys)
        for b in range(args.batches):
            d = args.work / f"batch{b}"
            d.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(CODEBOOK, d / "codebook.md")
            for key in keys[b :: args.batches]:
                shutil.copyfile(args.traces / f"{key}.md", d / f"{key}.md")
            print(d, len(keys[b :: args.batches]), "sessions")
    elif args.cmd == "collect":
        args.out.mkdir(parents=True, exist_ok=True)
        missing, bad, ok = [], {}, 0
        for d in sorted(args.work.glob("batch*")):
            for t in sorted(d.glob("*.md")):
                if t.name == "codebook.md":
                    continue
                f = d / f"{t.stem}.json"
                if not f.exists():
                    missing.append(t.stem)
                    continue
                try:
                    doc = json.loads(f.read_text())
                except ValueError:
                    bad[t.stem] = ["invalid JSON"]
                    continue
                problems = validate(doc, t.stem)
                if problems:
                    bad[t.stem] = problems
                shutil.copyfile(f, args.out / f.name)
                ok += 1
        print(json.dumps({"collected": ok, "missing": missing, "schema_problems": bad}, indent=1))
    else:
        index = json.loads((args.traces / "index.json").read_text())
        rng = random.Random(20261027)
        pick = []
        for arm in ("C", "D"):
            keys = sorted(
                k for k, v in index.items() if v["arm"] == arm and not v["transcript"].endswith(".partial.jsonl")
            )
            pick += rng.sample(keys, 5)
        print(json.dumps(pick))


if __name__ == "__main__":
    main()
