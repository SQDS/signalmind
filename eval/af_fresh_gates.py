"""Evaluate the frozen fresh-confirmation gates; reads saved results only, never signals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.af_episode_metrics import (  # noqa: E402
    FRESH_CANDIDATES,
    FRESH_ELIGIBLE,
    FRESH_PROTOCOL,
    LONG_AF_TOL_S,
    sha256,
    summary,
)


def exact_mcnemar(gains, losses):
    n = gains + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n)


def compare(baseline, candidate, eligible):
    before = {r["record"]: r for r in baseline["records"]}
    after = {r["record"]: r for r in candidate["records"]}
    if (
        len(before) != len(baseline["records"])
        or len(after) != len(candidate["records"])
        or before.keys() != after.keys()
    ):
        raise ValueError("Gate populations differ or contain duplicate records")
    if not set(eligible) <= before.keys() or len(set(eligible)) != len(eligible):
        raise ValueError("Invalid onset-eligible population")
    for name in before:
        for key in ("window_s", "annotated_af_intervals", "longest_annotated"):
            if before[name][key] != after[name][key]:
                raise ValueError("Paired truth differs")
    bs, cs = summary(list(before.values())), summary(list(after.values()))
    be, ce = summary([before[k] for k in eligible]), summary([after[k] for k in eligible])
    gains = [k for k in eligible if after[k]["longest_within_tolerance"] and not before[k]["longest_within_tolerance"]]
    losses = [k for k in eligible if before[k]["longest_within_tolerance"] and not after[k]["longest_within_tolerance"]]
    bt, ct = baseline["runtime_24h"], candidate["runtime_24h"]
    if bt["record"] != ct["record"] or bt["window_s"] != 86400 or ct["window_s"] != 86400:
        raise ValueError("Runtime protocols differ")
    ratio = ct["median_s"] / bt["median_s"] if ct["median_s"] is not None and bt["median_s"] else None
    result = {
        "G1": {
            "pass": len(gains) - len(losses) >= 2,
            "eligible_n": len(eligible),
            "M_correct": be["onset_correct"],
            "C_correct": ce["onset_correct"],
            "gains": gains,
            "regressions": losses,
            "net_gain": len(gains) - len(losses),
            "exact_two_sided_mcnemar_p": exact_mcnemar(len(gains), len(losses)),
            "p_is_gate": False,
        },
        "G2": {
            "pass": cs["burden_mae_window_pp"] <= bs["burden_mae_window_pp"] + 0.5,
            "M_mae_pp": bs["burden_mae_window_pp"],
            "C_mae_pp": cs["burden_mae_window_pp"],
            "delta_pp": cs["burden_mae_window_pp"] - bs["burden_mae_window_pp"],
        },
        "G3": {
            "pass": cs["pooled_sensitivity"] is not None
            and cs["pooled_ppv"] is not None
            and cs["pooled_sensitivity"] >= bs["pooled_sensitivity"] - 0.02
            and cs["pooled_ppv"] >= bs["pooled_ppv"] - 0.02,
            "M_sensitivity": bs["pooled_sensitivity"],
            "C_sensitivity": cs["pooled_sensitivity"],
            "M_ppv": bs["pooled_ppv"],
            "C_ppv": cs["pooled_ppv"],
        },
        "G4": {
            "pass": ce["median_fragmentation"] <= be["median_fragmentation"],
            "M_median_fragmentation": be["median_fragmentation"],
            "C_median_fragmentation": ce["median_fragmentation"],
        },
        "G5": {"pass": ratio is not None and ratio <= 1.5, "runtime_ratio": ratio, "M": bt, "C": ct},
        "summary_all": cs,
        "summary_onset_eligible": ce,
        "crashes": candidate["crashes"],
    }
    result["all_gates_pass"] = all(result[f"G{i}"]["pass"] for i in range(1, 6))
    return result


def select(comparisons):
    passing = [name for name in ("R", "V") if comparisons[name]["all_gates_pass"]]
    return min(
        passing,
        key=lambda name: (
            -comparisons[name]["G1"]["net_gain"],
            comparisons[name]["G2"]["C_mae_pp"],
            name != "R",
        ),
        default="M",
    )


def evaluate(results, eligibility, eligibility_hash):
    names = [r["record"] for r in eligibility["records"]]
    if len(names) != 38 or len(set(names)) != 38:
        raise ValueError("Fresh gates require all 38 inventory records")
    for name, result in results.items():
        ref, gap = FRESH_CANDIDATES[name]
        if (
            result["candidate"] != name
            or not result["source_commit"].startswith(ref)
            or result["variant_gap_s"] != gap
            or result["eligibility_sha256"] != eligibility_hash
            or result["eligible_records"] != eligibility["eligible_records"]
            or result["permitted_records"] != names
            or [r["record"] for r in result["records"]] != names
            or result["long_af_tol_s"] != LONG_AF_TOL_S
            or result["runtime_24h"]["record"] != eligibility["runtime_record"]
            or len(result["runtime_24h"]["cold_seconds"]) != 3
        ):
            raise ValueError(f"Result provenance/population violates frozen protocol: {name}")
    comparisons = {name: compare(results["M"], results[name], eligibility["eligible_records"]) for name in ("R", "V")}
    return {
        "protocol_commit": FRESH_PROTOCOL,
        "eligibility_sha256": eligibility_hash,
        "n_records": len(names),
        "eligible_records": eligibility["eligible_records"],
        "baseline": {
            "candidate": "M",
            "summary_all": summary(results["M"]["records"]),
            "summary_onset_eligible": summary(
                [r for r in results["M"]["records"] if r["record"] in eligibility["eligible_records"]]
            ),
            "runtime_24h": results["M"]["runtime_24h"],
            "crashes": results["M"]["crashes"],
        },
        "comparisons": comparisons,
        "selection_rule": "all gates pass; highest net onset gain; lower whole-window MAE; R; otherwise M",
        "selected": select(comparisons),
        "freshness_status": "All 38 records opened for detector selection; never reuse as fresh confirmation for the selected detector",
        "known_dev_risk": "MIT-BIH 203: onset errors M 0.236 s, R 930.178 s, V 819.186 s; R/V sensitivity 0.5083 versus M 0.9880",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=FRESH_ELIGIBLE.parent)
    parser.add_argument("--eligibility", type=Path, default=FRESH_ELIGIBLE)
    args = parser.parse_args()
    paths = {name: args.results_dir / f"fresh-{name}.json" for name in FRESH_CANDIDATES}
    results = {name: json.loads(path.read_text()) for name, path in paths.items()}
    result = evaluate(results, json.loads(args.eligibility.read_text()), sha256(args.eligibility))
    result["result_sha256"] = {name: sha256(path) for name, path in paths.items()}
    (args.results_dir / "fresh-gates.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected": result["selected"], "comparisons": result["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
