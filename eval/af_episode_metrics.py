"""Detector-only, fail-closed AF diagnostics on dev, fresh LTAFDB, or frozen AFDB.

Example (no downloads, models or suite selection helpers):
  PYTHONPATH=src uv run python eval/af_episode_metrics.py --source-root /tmp/baseline \
    --source-ref main --cases-dir /path/to/eval/cases --output /tmp/baseline.json

Raw annotations score overlap/burden; <30 s merged annotations score ranking.
Suite manifests are read for record IDs only. Never call select_ltafdb here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.grade import LONG_AF_TOL_S  # noqa: E402
from eval.tasks import afdb  # noqa: E402
from eval.tasks.longrec import af_segments, merge_episodes  # noqa: E402
from eval.tasks.mitdb import DS1, MITDB  # noqa: E402

SUITES = ("p1", "p2long", "p3", "p4long")
FRESH_INVENTORY = ROOT / "eval/results/af-continuity/fresh-inventory.json"
FRESH_ELIGIBLE = ROOT / "eval/results/af-continuity/fresh-eligible.json"
FRESH_PROTOCOL = "5f612bc"
FRESH_CANDIDATES = {"M": ("7a97b80", None), "R": ("22f74ad", None), "V": ("69c9f43", 120)}
AFDB_ELIGIBLE = afdb.ELIGIBILITY
AFDB_PROTOCOL = "30f44b4fb7a34c6ab61548c40cda7aa0630018f1"
AFDB_RECORD_IDS = tuple(
    "04015 04043 04048 04126 04746 04908 04936 05091 05121 05261 06426 06453 "
    "06995 07162 07859 07879 07910 08215 08219 08378 08405 08434 08455".split()
)
AFDB_ONSET_IDS = frozenset(
    "04043 04746 04908 04936 05121 05261 06995 07879 07910 08215 08219 08378 08405 08434 08455".split()
)
AFDB_BURDEN_IDS = AFDB_ONSET_IDS - {"05261", "08434"}
AFDB_CANDIDATES = {
    "M": ("114b99372b20e80feb6c2bc69f3da0696921b5b7", None),  # baseline-M-2026-10-12
    "R": ("1d344ae403616056d3bd7888ec097d2f8144f015", None),  # baseline-R-2026-10-12
    "V": ("69c9f43f14ff3584a5d1f0be243ece3cb4e98160", 120),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def afdb_records(evidence):
    """Exactly A1's waveform population and independent endpoint memberships."""
    records = evidence["records"]
    if set(records) != set(AFDB_RECORD_IDS) or len(records) != 23:
        raise ValueError("AFDB inventory must contain exactly the 23 frozen waveform records")
    if evidence.get("schema_version") != 2 or evidence.get("counts") != {"longest_af": 15, "af_burden": 13}:
        raise ValueError("AFDB eligibility must use the frozen Amendment A1 endpoints")
    for rec, facts in records.items():
        expected = (["longest_af"] if rec in AFDB_ONSET_IDS else []) + (["af_burden"] if rec in AFDB_BURDEN_IDS else [])
        if facts.get("types") != expected or afdb.select_types(facts) != expected:
            raise ValueError(f"AFDB endpoint membership differs from frozen eligibility: {rec}")
    return [("afdb", rec) for rec in AFDB_RECORD_IDS]


def require_afdb_record(name, permitted):
    if name not in {f"{db}/{rec}" for db, rec in permitted} or name not in {f"afdb/{rec}" for rec in AFDB_RECORD_IDS}:
        raise ValueError(f"Record outside frozen AFDB allowlist: {name}")


def load_afdb_eligibility(args):
    frozen = subprocess.check_output(["git", "show", f"{AFDB_PROTOCOL}:eval/tasks/afdb_eligibility.json"], cwd=ROOT)
    if args.afdb_eligibility.read_bytes() != frozen:
        raise ValueError("AFDB eligibility differs from the frozen Amendment A1 checkpoint")
    evidence = json.loads(frozen)
    permitted = afdb_records(evidence)
    # Rehash every downloaded input, then recompute all header/ATR facts. No waveform decoding.
    afdb.prepare_afdb(args.afdb_dir, args.afdb_eligibility)
    return permitted, evidence


def verify_afdb_snapshot(args):
    commit = subprocess.check_output(["git", "rev-parse", args.source_ref], cwd=ROOT, text=True).strip()
    matches = [k for k, (ref, gap) in AFDB_CANDIDATES.items() if commit == ref and args.variant_gap == gap]
    if len(matches) != 1:
        raise ValueError("AFDB permits only frozen baseline-M/R tags without overrides, or V@69c9f43 with gap 120")
    return matches[0], verify_snapshot_source(args.source_root, commit)


def fresh_records(inventory, split, excluded):
    """Refuse an invalid inventory rather than silently shrinking the frozen population."""
    records = [r["record"] for r in inventory["records"]]
    allowed = set(split["unused"]) - set(split["dev"]) - set(split["test_ext"]) - excluded["ltafdb"]
    if (
        len(records) != 38
        or len(set(records)) != 38
        or any(not isinstance(r, str) or not r.isascii() or not r.isdigit() for r in records)
        or set(records) != allowed
    ):
        raise ValueError("Fresh inventory must contain exactly the 38 permitted unused LTAFDB records")
    return [("ltafdb", r) for r in sorted(records)]


def require_fresh_record(name, permitted):
    if name not in {f"{db}/{rec}" for db, rec in permitted}:
        raise ValueError(f"Record outside frozen fresh allowlist: {name}")


def fresh_population(cases_dir):
    frozen = subprocess.check_output(
        ["git", "show", f"{FRESH_PROTOCOL}:eval/results/af-continuity/fresh-inventory.json"], cwd=ROOT
    )
    if FRESH_INVENTORY.read_bytes() != frozen:
        raise ValueError("Fresh inventory differs from the pre-waveform protocol commit")
    inventory = json.loads(frozen)
    split = json.loads((ROOT / "eval/tasks/ltafdb_split.json").read_text())
    excluded, manifests = suite_record_ids(cases_dir)
    if manifests != inventory["suite_manifests"]:
        raise ValueError("Suite exclusion manifests differ from the frozen inventory")
    return fresh_records(inventory, split, excluded), manifests


def prepare_fresh_eligibility(args):
    """Write annotation-only eligibility before importing or running any detector."""
    import wfdb

    permitted, manifests = fresh_population(args.cases_dir)
    rows = []
    for db, rec in permitted:
        require_fresh_record(f"{db}/{rec}", permitted)
        path = args.ltaf_dir / rec
        h = wfdb.rdheader(str(path))
        ann = wfdb.rdann(str(path), "atr")
        if not len(ann.sample) or not ((ann.sample >= 0) & (ann.sample < h.sig_len)).all():
            raise ValueError(f"Invalid annotations: {rec}")
        truth = clipped(af_segments(ann, h.fs, h.sig_len), 0, h.sig_len / h.fs)
        lt = longest(merge_episodes(truth))
        duration = h.sig_len / h.fs
        pct = 100 * seconds(truth) / duration
        rows.append(
            {
                "record": f"{db}/{rec}",
                "window_s": duration,
                "fs": h.fs,
                "n_samples": h.sig_len,
                "true_af_s": seconds(truth),
                "true_pct_of_window": pct,
                "annotated_af_intervals": truth,
                "longest_annotated": lt,
                "eligible": bool(lt and lt[1] - lt[0] > 300 and pct < 90),
                "input_sha256": {ext: sha256(path.with_suffix(f".{ext}")) for ext in ("hea", "atr")},
            }
        )
    eligible = [r["record"] for r in rows if r["eligible"]]
    runtime = next(r["record"] for r in rows if r["eligible"] and r["window_s"] >= 86400)
    result = {
        "protocol_commit": FRESH_PROTOCOL,
        "created_utc": datetime.now(UTC).isoformat(),
        "access": "headers and annotations only; no waveform reads or detector imports/runs",
        "inventory_sha256": sha256(FRESH_INVENTORY),
        "split_sha256": sha256(ROOT / "eval/tasks/ltafdb_split.json"),
        "suite_manifests": manifests,
        "eligibility_rule": "longest <30 s merged annotated AF/AFL episode >300 s AND raw whole-record burden <90%",
        "long_af_tol_s": LONG_AF_TOL_S,
        "records": rows,
        "eligible_records": eligible,
        "runtime_record": runtime,
        "runtime_selection": "first eligible record in sorted inventory order with duration >=86400 s",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"eligible_records": eligible, "runtime_record": runtime}, indent=2))


def load_fresh_eligibility(args, permitted, manifests):
    evidence = json.loads(args.eligibility.read_text())  # missing checkpoint fails before waveform access
    names = [f"{db}/{rec}" for db, rec in permitted]
    if (
        evidence["inventory_sha256"] != sha256(FRESH_INVENTORY)
        or evidence["split_sha256"] != sha256(ROOT / "eval/tasks/ltafdb_split.json")
        or evidence["suite_manifests"] != manifests
        or [r["record"] for r in evidence["records"]] != names
        or evidence["long_af_tol_s"] != LONG_AF_TOL_S
    ):
        raise ValueError("Eligibility checkpoint does not match the frozen population")
    for row in evidence["records"]:
        require_fresh_record(row["record"], permitted)
        path = args.ltaf_dir / row["record"].split("/")[1]
        if any(sha256(path.with_suffix(f".{ext}")) != digest for ext, digest in row["input_sha256"].items()):
            raise ValueError("Header/annotation changed after eligibility checkpoint")
        lt = row["longest_annotated"]
        if row["eligible"] != bool(lt and lt[1] - lt[0] > 300 and row["true_pct_of_window"] < 90):
            raise ValueError("Eligibility checkpoint violates frozen definition")
    if evidence["eligible_records"] != [r["record"] for r in evidence["records"] if r["eligible"]]:
        raise ValueError("Eligibility list differs from checkpoint rows")
    runtime = next(r["record"] for r in evidence["records"] if r["eligible"] and r["window_s"] >= 86400)
    if runtime != evidence["runtime_record"]:
        raise ValueError("Runtime record differs from frozen selection rule")
    return evidence


def verify_fresh_snapshot(args):
    commit = subprocess.check_output(["git", "rev-parse", args.source_ref], cwd=ROOT, text=True).strip()
    matches = [k for k, (ref, gap) in FRESH_CANDIDATES.items() if commit.startswith(ref) and args.variant_gap == gap]
    if len(matches) != 1:
        raise ValueError("Fresh mode permits only M@7a97b80, R@22f74ad, V@69c9f43 with gap 120")
    return matches[0], verify_snapshot_source(args.source_root, commit)


def verify_snapshot_source(source_root, commit):
    source = source_root.resolve()
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "src"], cwd=ROOT, text=True)
    hashes = {}
    for name in paths.splitlines():
        original = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if (source / name).read_bytes() != original:
            raise ValueError(f"Source snapshot modified: {name}")
        hashes[name] = sha256(source / name)
    if {str(p.relative_to(source)) for p in (source / "src").rglob("*.py")} - set(hashes):
        raise ValueError("Source snapshot contains extra Python modules")
    return hashes


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([float(a), float(b)])
    return out


def clipped(intervals, a, b):
    return union((max(a, x), min(b, y)) for x, y in intervals)


def seconds(intervals):
    return sum(b - a for a, b in intervals)


def intersection(left, right):
    left, right = union(left), union(right)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            out.append([a, b])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return out


def complement(intervals, duration):
    out, previous = [], 0.0
    for a, b in clipped(intervals, 0, duration):
        if a > previous:
            out.append([previous, a])
        previous = b
    if previous < duration:
        out.append([previous, duration])
    return out


def suite_record_ids(cases_dir):
    """Only return IDs and manifest hashes, never answers or questions."""
    excluded = {"mitdb": set(), "ltafdb": set()}
    manifest = {}
    for suite in SUITES:
        ids = {"mitdb": set(), "ltafdb": set()}
        hashes = {}
        for kind in ("questions", "answers"):
            path = cases_dir / f"{suite}.{kind}.jsonl"
            content = path.read_bytes()  # missing manifests fail closed
            hashes[kind] = hashlib.sha256(content).hexdigest()
            for line in content.splitlines():
                row = json.loads(line)
                case_id = row["case_id"]
                db = "mitdb" if case_id.startswith("ecg") else "ltafdb" if case_id.startswith("ltaf") else None
                answer = row.get("answer", {})
                rec = answer.get("record", row.get("record"))
                if kind == "answers" and db and rec is None:
                    raise ValueError(f"Missing record ID: {suite}/{case_id}")
                if db and rec is not None:
                    ids[db].add(str(rec))
        for db in ids:
            excluded[db].update(ids[db])
        manifest[suite] = {"sha256": hashes, "records": {db: sorted(rs) for db, rs in ids.items()}}
    return excluded, manifest


def dev_records(split, excluded):
    return [("mitdb", str(r)) for r in DS1 if str(r) not in excluded["mitdb"]] + [
        ("ltafdb", r) for r in split["dev"] if r not in excluded["ltafdb"] and r not in split["test_ext"]
    ]


def longest(intervals):
    return min(intervals, key=lambda ab: (-(ab[1] - ab[0]), ab[0])) if intervals else None


def score_record(name, duration, truth, episodes, readable):
    truth = clipped(truth, 0, duration)
    spans = clipped([(e["t"], e["t"] + e["dur"]) for e in episodes], 0, duration)
    # AF support, not a presentation span crossing unknown/non-AF gaps.
    support = clipped([ab for e in episodes for ab in e.get("intervals_s", [(e["t"], e["t"] + e["dur"])])], 0, duration)
    annotated = merge_episodes(truth)
    lt, ld = longest(annotated), longest(spans)
    true_s, af_s, readable_s = seconds(truth), seconds(support), seconds(readable)
    tp = seconds(intersection(truth, support))
    true_readable_s = seconds(intersection(truth, readable))
    err = abs(ld[0] - lt[0]) if ld and lt else None
    longest_overlap = seconds(intersection([lt], [ld])) if lt and ld else 0.0
    longest_union = seconds([lt, ld]) - longest_overlap if lt and ld else 0.0
    return {
        "record": name,
        "window_s": duration,
        "has_af": bool(truth),
        "annotated_episodes": annotated,
        "annotated_af_intervals": truth,
        "support_intervals": support,
        "readable_intervals": readable,
        "detected_episodes": [
            {
                "start": e["t"],
                "end": e["t"] + e["dur"],
                "af_s": e.get("af_s", e["dur"]),
                "merged_from": e.get("merged_from"),
                "unknown_s": e.get("unknown_s"),
            }
            for e in sorted(episodes, key=lambda e: e["t"])
        ],
        "n_annotated": len(annotated),
        "n_detected": len(episodes),
        "true_af_s": true_s,
        "af_s": af_s,
        "overlap_s": tp,
        "readable_s": readable_s,
        "unreadable_s": duration - readable_s,
        "sensitivity": tp / true_s if true_s else None,
        "ppv": tp / af_s if af_s else None,
        "true_pct_of_window": 100 * true_s / duration,
        "pct_of_window": 100 * af_s / duration,
        "burden_error_window_pp": 100 * (af_s - true_s) / duration,
        "true_pct_of_readable": 100 * true_readable_s / readable_s if readable_s else None,
        "pct_of_readable": 100 * af_s / readable_s if readable_s else None,
        "burden_error_readable_pp": 100 * (af_s - true_readable_s) / readable_s if readable_s else None,
        "burden_error_common_readable_denominator_pp": 100 * (af_s - true_s) / readable_s if readable_s else None,
        "longest_annotated": lt,
        "longest_detected": ld,
        "longest_iou": longest_overlap / longest_union if lt and ld else 0.0 if lt else None,
        "longest_onset_error_s": err,
        "longest_within_tolerance": err is not None and err <= LONG_AF_TOL_S if lt else None,
        "longest_duration_ratio": (ld[1] - ld[0]) / (lt[1] - lt[0]) if ld and lt else 0.0 if lt else None,
        "fragmentation": int(sum(e["t"] < lt[1] and e["t"] + e["dur"] > lt[0] for e in episodes)) if lt else None,
    }


def summary(rows):
    af = [r for r in rows if r["has_af"]]
    true_s = sum(r["true_af_s"] for r in rows)
    af_s = sum(r["af_s"] for r in rows)
    tp = sum(r["overlap_s"] for r in rows)
    readable_errors = [abs(r["burden_error_readable_pp"]) for r in rows if r["burden_error_readable_pp"] is not None]
    return {
        "n_records": len(rows),
        "n_af_records": len(af),
        "onset_correct": sum(r["longest_within_tolerance"] for r in af),
        "median_fragmentation": float(np.median([r["fragmentation"] for r in af])) if af else None,
        "burden_mae_window_pp": float(np.mean([abs(r["burden_error_window_pp"]) for r in rows])),
        "burden_mae_readable_pp": float(np.mean(readable_errors)) if readable_errors else None,
        "pooled_sensitivity": tp / true_s if true_s else None,
        "pooled_ppv": tp / af_s if af_s else None,
    }


def gates(baseline, candidate):
    before, after = baseline["summary"], candidate["summary"]
    br = {r["record"]: r for r in baseline["records"]}
    cr = {r["record"]: r for r in candidate["records"]}
    if br.keys() != cr.keys():
        raise ValueError("Gate populations differ")
    regressions = [k for k in br if br[k]["longest_within_tolerance"] and not cr[k]["longest_within_tolerance"]]
    ratio = candidate["runtime_24h"]["median_s"] / baseline["runtime_24h"]["median_s"]
    return {
        "1_onset": after["onset_correct"] >= before["onset_correct"] and not regressions,
        "onset_regressions": regressions,
        "2_fragmentation": after["median_fragmentation"] < before["median_fragmentation"],
        "3_burden_overlap": after["burden_mae_window_pp"] <= before["burden_mae_window_pp"] + 0.5
        and after["pooled_sensitivity"] >= before["pooled_sensitivity"] - 0.02
        and after["pooled_ppv"] >= before["pooled_ppv"] - 0.02,
        "4_runtime": ratio <= 1.5,
        "runtime_ratio": ratio,
        "5_checks": "pending tests/ruff/dependency verification",
    }


def run(args):
    if getattr(args, "afdb", False):
        return run_afdb(args)
    evidence = None
    if args.fresh:
        if args.baseline:
            raise ValueError("Use eval/af_fresh_gates.py for the fresh gates, not the dev --baseline gates")
        if args.output.exists() or args.output.with_suffix(".partial.json").exists():
            raise FileExistsError("Fresh result already exists; refusing an accidental detector rerun")
        permitted, manifests = fresh_population(args.cases_dir)
        evidence = load_fresh_eligibility(args, permitted, manifests)
        candidate, snapshot_hashes = verify_fresh_snapshot(args)
    else:
        split = json.loads((ROOT / "eval/tasks/ltafdb_split.json").read_text())
        excluded, manifests = suite_record_ids(args.cases_dir)
        permitted = dev_records(split, excluded)
    started_utc = datetime.now(UTC).isoformat()
    # Pin imports to the specified snapshot, not the shared environment's editable install.
    sys.path.insert(0, str(args.source_root.resolve() / "src"))
    import wfdb

    import tsmcp.plugins.ecg as ecg
    from tsmcp.core.io.wfdb import load_wfdb
    from tsmcp.core.signal import Channel

    if not Path(ecg.__file__).resolve().is_relative_to(args.source_root.resolve() / "src"):
        raise ValueError("Detector import escaped the requested snapshot")
    if args.variant_gap is not None:
        from tsmcp.plugins.ecg import af

        if not hasattr(af, "CONTINUITY_GAP_S"):
            raise ValueError(
                "Variants require the frozen candidate source snapshot 69c9f43; final detection is restored"
            )
        af.CONTINUITY_GAP_S = args.variant_gap

    rows = []
    for db, rec in permitted:
        name = f"{db}/{rec}"
        if evidence:
            require_fresh_record(name, permitted)
        path = (args.ltaf_dir if db == "ltafdb" else args.mitdb_dir) / rec
        for ext in (".hea", ".atr", ".dat"):
            if not path.with_suffix(ext).is_file():
                raise FileNotFoundError(path.with_suffix(ext))
        fact = next(r for r in evidence["records"] if r["record"] == name) if evidence else None
        stage = "load_waveform"
        try:
            channels, _ = load_wfdb(path.with_suffix(".hea"), None, None, None)
            ch = next((c for c in channels if c.name == "MLII"), channels[0]) if db == "mitdb" else channels[0]
            if fact:
                if ch.fs != fact["fs"] or ch.n != fact["n_samples"]:
                    raise ValueError("Signal dimensions differ from eligibility checkpoint")
                truth = fact["annotated_af_intervals"]
            else:
                annotation = wfdb.rdann(str(path), "atr")
                truth = af_segments(annotation, ch.fs, ch.n)
            stage = "detector"
            begin = time.perf_counter()
            episodes = ecg.ecg_irregular(ch.data, ch.fs, ch.unit, ch=ch)
            burden = ecg.ecg_af_burden(ch.data, ch.fs, ch.unit, ch=ch)
            elapsed = time.perf_counter() - begin
            stage = "readability"
            noisy = ecg.ecg_noisy(ch.data, ch.fs, ch.unit, ch=ch)
            readable = complement([(e["t"], e["t"] + e["dur"]) for e in noisy], ch.duration)
            row = score_record(name, ch.duration, truth, episodes, readable)
            row.update(public_burden=burden, detector_seconds=elapsed, crash=None)
        except Exception as exc:
            if not fact:
                raise
            row = score_record(name, fact["window_s"], fact["annotated_af_intervals"], [], [])
            row.update(public_burden=None, detector_seconds=None, crash={"type": type(exc).__name__, "stage": stage})
            print(f"{name}: FAILED at {stage}: {type(exc).__name__}", flush=True)
        if fact:
            row["onset_eligible"] = fact["eligible"]
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".partial.json").write_text(
            json.dumps({"records": rows}, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"{db}/{rec}: AF={row['has_af']}, onset={row['longest_onset_error_s']}, fragments={row['fragmentation']}",
            flush=True,
        )
    runtime_record = evidence["runtime_record"] if evidence else "ltafdb/102"
    if tuple(runtime_record.split("/")) not in permitted:
        raise ValueError("Runtime record is outside the permitted set")
    channels, _ = load_wfdb(args.ltaf_dir / f"{runtime_record.split('/')[1]}.hea", None, None, None)
    original = channels[0]
    if original.duration < 86400:
        raise ValueError("Runtime record shorter than 24 hours")
    times, runtime_errors = [], []
    for _ in range(3):
        ch = Channel(original.name, original.fs, original.data[: round(86400 * original.fs)], original.unit)
        begin = time.perf_counter()
        try:
            ecg.ecg_irregular(ch.data, ch.fs, ch.unit, ch=ch)
            ecg.ecg_af_burden(ch.data, ch.fs, ch.unit, ch=ch)
            times.append(time.perf_counter() - begin)
            print(f"24h cold runtime: {times[-1]:.3f} s", flush=True)
        except Exception as exc:
            if not evidence:
                raise
            times.append(None)
            runtime_errors.append({"trial": len(times), "type": type(exc).__name__})
    source = args.source_root.resolve() / "src"
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((source / "tsmcp/plugins/ecg").glob("*.py"))
    }
    result = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "variant_gap_s": args.variant_gap,
        "source_ref": args.source_ref,
        "source_commit": subprocess.check_output(["git", "rev-parse", args.source_ref], cwd=ROOT, text=True).strip(),
        "source_hashes": hashes,
        "long_af_tol_s": LONG_AF_TOL_S,
        "annotation_merge": "gap strictly less than 30 s; raw AF/AFL for burden/overlap",
        "single_lead": True,
        "suite_manifests": manifests,
        "permitted_records": [f"{db}/{rec}" for db, rec in permitted],
        "records": rows,
        "summary": summary(rows),
        "summary_af_only": summary([r for r in rows if r["has_af"]]),
        "summary_by_database": {
            db: summary([r for r in rows if r["record"].startswith(db)]) for db in sorted({db for db, _ in permitted})
        },
        "runtime_24h": {
            "record": runtime_record,
            "window_s": 86400,
            "cold_seconds": times,
            "median_s": float(np.median(times)) if all(t is not None for t in times) else None,
            "errors": runtime_errors,
            "includes": "ecg_irregular + ecg_af_burden; excludes file loading",
        },
    }
    if evidence:
        result.update(
            mode="fresh",
            candidate=candidate,
            protocol_commit=FRESH_PROTOCOL,
            eligibility_sha256=sha256(args.eligibility),
            snapshot_hashes=snapshot_hashes,
            eligible_records=evidence["eligible_records"],
            summary_onset_eligible=summary([r for r in rows if r["onset_eligible"]]),
            crashes=[{"record": r["record"], **r["crash"]} for r in rows if r["crash"]],
        )
    if args.baseline:
        result["gates"] = gates(json.loads(args.baseline.read_text()), result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".partial.json").unlink(missing_ok=True)
    print(json.dumps({"summary": result["summary"], "gates": result.get("gates")}, indent=2), flush=True)


def run_afdb(args):
    """Descriptive frozen-candidate evaluation; no selection, gates, or 24 h benchmark."""
    if args.baseline or args.fresh or args.prepare_eligible:
        raise ValueError("--afdb cannot use --baseline, --fresh, or --prepare-eligible")
    if args.output.exists() or args.output.with_suffix(".partial.json").exists():
        raise FileExistsError("AFDB result already exists; refusing an accidental detector rerun")
    permitted, evidence = load_afdb_eligibility(args)
    candidate, snapshot_hashes = verify_afdb_snapshot(args)
    started_utc = datetime.now(UTC).isoformat()
    source = args.source_root.resolve() / "src"
    # A previous in-process import must not supply a different snapshot's detector or helpers.
    for name, module in list(sys.modules.items()):
        if name == "tsmcp" or name.startswith("tsmcp."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(source):
                raise ValueError("Detector import escaped the requested snapshot; use a fresh process")
    sys.path.insert(0, str(source))
    import wfdb

    import tsmcp.plugins.ecg as ecg
    from tsmcp.core.io.wfdb import load_wfdb

    if not Path(ecg.__file__).resolve().is_relative_to(source):
        raise ValueError("Detector import escaped the requested snapshot")
    if candidate == "V":
        from tsmcp.plugins.ecg import af

        af.CONTINUITY_GAP_S = args.variant_gap

    rows = []
    for db, rec in permitted:
        name = f"{db}/{rec}"
        require_afdb_record(name, permitted)
        fact = evidence["records"][rec]
        path = args.afdb_dir / rec
        # Only audited rhythm annotations supply truth. QRS files are integrity-hashed only.
        annotation = wfdb.rdann(str(path), "atr")
        truth = af_segments(annotation, fact["fs"], fact["n_samples"])
        channels, _ = load_wfdb(path.with_suffix(".hea"), None, None, None)
        ch = channels[0]
        if ch.fs != 250 or ch.fs != fact["fs"] or ch.n != fact["n_samples"]:
            raise ValueError(f"Signal dimensions differ from AFDB eligibility checkpoint: {rec}")
        stage = "detector"
        try:
            begin = time.perf_counter()
            episodes = ecg.ecg_irregular(ch.data, ch.fs, ch.unit, ch=ch)
            burden = ecg.ecg_af_burden(ch.data, ch.fs, ch.unit, ch=ch)
            elapsed = time.perf_counter() - begin
            stage = "readability"
            noisy = ecg.ecg_noisy(ch.data, ch.fs, ch.unit, ch=ch)
            readable = complement([(e["t"], e["t"] + e["dur"]) for e in noisy], ch.duration)
            row = score_record(name, ch.duration, truth, episodes, readable)
            row.update(public_burden=burden, detector_seconds=elapsed, crash=None)
        except Exception as exc:
            # Same failure accounting as fresh LTAFDB: retain the record with zero detected support.
            row = score_record(name, fact["dur_s"], truth, [], [])
            row.update(public_burden=None, detector_seconds=None, crash={"type": type(exc).__name__, "stage": stage})
            print(f"{name}: FAILED at {stage}: {type(exc).__name__}", flush=True)
        row.update(
            fs=ch.fs,
            n_samples=ch.n,
            channel_index=0,
            types=fact["types"],
            onset_eligible="longest_af" in fact["types"],
            burden_eligible="af_burden" in fact["types"],
        )
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".partial.json").write_text(
            json.dumps({"candidate": candidate, "records": rows}, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"{name}: onset={row['longest_onset_error_s']}, IoU={row['longest_iou']}, fragments={row['fragmentation']}",
            flush=True,
        )
    result = {
        "mode": "afdb",
        "purpose": "descriptive detector-level failure attribution; all candidates frozen; no selection or gates",
        "candidate": candidate,
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": AFDB_PROTOCOL,
        "eligibility_sha256": sha256(args.afdb_eligibility),
        "data_hashes": evidence["data_hashes"],
        "source_ref": args.source_ref,
        "source_commit": AFDB_CANDIDATES[candidate][0],
        "source_root": str(args.source_root.resolve()),
        "imported_package_path": str(Path(ecg.__file__).resolve()),
        "snapshot_hashes": snapshot_hashes,
        "variant_gap_s": args.variant_gap,
        "long_af_tol_s": LONG_AF_TOL_S,
        "annotation_merge": "gap strictly less than 30 s; raw AF/AFL for burden/overlap; .atr only",
        "detected_ranking": "native detector episode spans; longest duration, earliest start for ties",
        "longest_iou_definition": "intersection / union of longest detected span and longest merged annotated span",
        "single_lead": True,
        "channel_index": 0,
        "time_coordinates": evidence["time_coordinates"],
        "counts": evidence["counts"],
        "permitted_records": [f"{db}/{rec}" for db, rec in permitted],
        "onset_eligible_records": [r["record"] for r in rows if r["onset_eligible"]],
        "burden_eligible_records": [r["record"] for r in rows if r["burden_eligible"]],
        "records": rows,
        "summary": summary(rows),
        "summary_onset_eligible": summary([r for r in rows if r["onset_eligible"]]),
        "summary_burden_eligible": summary([r for r in rows if r["burden_eligible"]]),
        "crashes": [{"record": r["record"], **r["crash"]} for r in rows if r["crash"]],
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".partial.json").unlink()
    print(
        json.dumps(
            {key: result[key] for key in ("summary", "summary_onset_eligible", "summary_burden_eligible")}, indent=2
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--cases-dir", type=Path)
    parser.add_argument("--ltaf-dir", type=Path, default=ROOT / "data/ltafdb")
    parser.add_argument("--mitdb-dir", type=Path, default=MITDB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--variant-gap", type=float, choices=(60, 120, 300))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true", help="Only the 38 records in the frozen fresh inventory")
    mode.add_argument("--afdb", action="store_true", help="Exactly the 23 frozen AFDB waveform records; descriptive")
    parser.add_argument("--afdb-dir", type=Path, default=afdb.AFDB)
    parser.add_argument("--afdb-eligibility", type=Path, default=AFDB_ELIGIBLE)
    parser.add_argument(
        "--prepare-eligible", action="store_true", help="Write annotations-only eligibility; no detector"
    )
    parser.add_argument("--eligibility", type=Path, default=FRESH_ELIGIBLE)
    args = parser.parse_args()
    if not args.afdb and args.cases_dir is None:
        parser.error("--cases-dir is required outside --afdb mode")
    if args.afdb and (args.baseline or args.prepare_eligible):
        parser.error("--afdb cannot use --baseline or --prepare-eligible")
    if args.prepare_eligible:
        if not args.fresh:
            parser.error("--prepare-eligible requires --fresh")
        prepare_fresh_eligibility(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
