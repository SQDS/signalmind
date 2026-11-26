"""Offline development pilot on two archived Vital subjects. Freeze all assignments before running any method."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tsmcp.cleaning.common import digest, read, sha, write
from tsmcp.cleaning.tasks import probe

from .agent_smoke import ScriptedTransport
from .episode import ARMS, run_episode
from .public import prepare
from .scoring import report


def build(archive, root, seed):
    rng = np.random.default_rng(seed)
    pool = defaultdict(list)
    for c in read(archive / "dataset_manifest.json")["cases"]:
        if (
            c["task"] == "ECG"
            and c["fs"] == 125
            and c.get("field") == "ECG_Record"
            and c.get("has_source_time")
            and not c.get("normalized_windows")
            and c["n"] >= 3750
        ):
            pool[c["record_id"]].append(c)
    root.mkdir(parents=True, exist_ok=False)
    private = root / "organizer"
    private.mkdir()
    windows = []
    for subject in sorted(pool)[:2]:
        for i in rng.permutation(len(pool[subject])):
            c = pool[subject][i]
            source = archive / "public" / c["id"] / "input.npz"
            if c["input_sha256"] != sha(source):
                raise ValueError("Archived observed source changed")
            with np.load(source, allow_pickle=False) as z:
                start = int(rng.integers(0, c["n"] - 3750 + 1))
                where = slice(start, start + 3750)
                ref, t = z["values"][where], z["time"][where]
                v = z["valid"][where] if "valid" in z else np.isfinite(ref)
                e = z["estimated"][where] if "estimated" in z else np.zeros(len(t), bool)
            if not (v & ~e & np.isfinite(ref)).all() or not np.allclose(np.diff(t), 1 / c["fs"], rtol=0, atol=1e-6):
                continue
            native = probe(ref, t, c["fs"], v, e, "ECG")
            beats = [q["sample"] for q in native["events"] if 625 <= q["sample"] < 3125]
            swing = float(np.ptp(np.quantile(ref, [0.02, 0.98])))
            if len(beats) < 3 or swing <= 0:
                continue
            k = len(beats) // 2
            for stratum, site in (
                ("between_beats", (beats[k - 1] + beats[k]) // 2),
                ("at_detected_beat", beats[k]),
                ("unchanged", None),
            ):
                windows.append((subject, c, start, t, ref, v, e, stratum, site, swing))
            break
    if len(windows) != 6:
        raise ValueError("The pilot needs one supported 30-second block from each of two development subjects")
    cases = []
    for number, i in enumerate(rng.permutation(len(windows)), 1):
        subject, c, start, t, ref, v, e, stratum, site, swing = windows[i]
        cid = f"C{number:03}"
        x = ref.copy()
        damaged = np.zeros(len(x), bool)
        if site is not None:
            x[site : site + 2] += np.array([3, -3]) * swing
            damaged[site : site + 2] = True
        public = prepare(
            root / cid / "public",
            cid,
            "ECG",
            c["unit"],
            c["fs"],
            t,
            x,
            v,
            e,
            start,
            acquisition=dict(
                species="human",
                device="Vital",
                field=c["field"],
                has_source_time=c["has_source_time"],
                normalized_windows=c["normalized_windows"],
            ),
        )
        truth = private / (cid + ".npz")
        np.savez(truth, reference=ref, damaged=damaged, reference_support=v & ~e)
        cases.append(
            dict(
                case_id=cid,
                subject=subject,
                record=c["id"],
                stratum=stratum,
                noise_source=None if site is None else "paired_spike",
                split="development",
                source_sha256=c["input_sha256"],
                source_range=[start, start + len(t)],
                input_hash=public["input_hash"],
                public_dir=f"../{cid}/public",
                private_npz=truth.name,
                private_sha256=sha(truth),
                outputs={arm: f"../{cid}/{arm}" for arm in ARMS},
            )
        )
    frozen = dict(
        seed=seed,
        arms=list(ARMS),
        repetitions=1,
        arm_modes=dict.fromkeys(ARMS, "fixture"),
        cases=cases,
        purpose="offline engineering regression",
        clean_reference="pre-injection observation",
        independent_expert_events=False,
        reference_scale_floors={},
    )
    write(private / "manifest.json", frozen)
    return private / "manifest.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=111101)
    args = p.parse_args()
    path = build(args.archive, args.out, args.seed)
    bound = sha(path)
    for case in read(path)["cases"]:
        for arm in ARMS:
            run_episode(
                path.parent / case["public_dir"],
                path.parent / case["outputs"][arm],
                arm,
                ScriptedTransport(),
                episode_id=digest(["vital-pilot", case["case_id"], 1])[:24],
            )
    if sha(path) != bound:
        raise ValueError("Assignment manifest changed")
    scored = report(path)
    write(path.parent / "report.json", scored)
    print([(s["arm"], s["statuses"], s["sites"], s["recovered"]) for s in scored["summaries"]])


if __name__ == "__main__":
    main()
