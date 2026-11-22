"""Read-only alignment audit of a Vital review ZIP, source MATs and SignalMind record handles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.dataset_workflow import read, sha, write  # noqa: E402
from eval.prepare_vital_processing import PulseMat  # noqa: E402


def workbook_tables(content):
    """Read cell values and G-number tables using the XLSX XML; do not execute formulas/macros."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    tables = {}
    with ZipFile(io.BytesIO(content)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(t.text or "" for t in si.iterfind(".//s:t", ns))
                for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", ns)
        for sheet in sheets:
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rels[rid]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            header, rows = None, []
            for row in ET.fromstring(archive.read(target)).find("s:sheetData", ns):
                values = {}
                for cell in row:
                    column = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                    value = cell.find("s:v", ns)
                    if cell.attrib.get("t") == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iterfind(".//s:t", ns))
                    elif value is None or value.text is None:
                        text = ""
                    elif cell.attrib.get("t") == "s":
                        text = strings[int(value.text)]
                    else:
                        text = value.text
                    values[column] = text
                if "G编号" in values.values():
                    header = {k: v for k, v in values.items() if v}
                elif header:
                    item = {name: values.get(col, "") for col, name in header.items()}
                    if re.fullmatch(r"G\d{4}", item.get("G编号", "")):
                        rows.append({"excel_row": int(row.attrib["r"]), **item})
            tables[sheet.attrib["name"]] = rows
    return tables


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*_mapping.csv\nproblem_details.csv\n")
    problems, counts = [], Counter()

    def check(condition, kind, context):
        counts[kind + "_checked"] += 1
        if not bool(condition):
            problems.append({"check": kind, "context": str(context)})
        return bool(condition)

    manifest = read(args.project / "private/manifest.json")
    records = {"vital::" + r["subject"]: r for r in manifest["records"]}
    cached_metadata = {}

    def metadata(subject):
        if subject not in cached_metadata:
            record = records[subject]
            stat = Path(record["source"]).stat()
            check(stat.st_size == record["size"] and stat.st_mtime_ns == record["mtime_ns"], "source_stat", subject)
            check(sha(record["metadata_path"]) == record["metadata_sha256"], "source_metadata_sha256", subject)
            with np.load(record["metadata_path"], allow_pickle=False) as data:
                cached_metadata[subject] = {k: data[k] for k in data.files}
        return cached_metadata[subject]

    digest = sha(args.archive)
    sidecar = Path(str(args.archive) + ".sha256")
    if sidecar.exists():
        check(digest == sidecar.read_text().split()[0], "zip_sha256", args.archive.name)
    with ZipFile(args.archive) as archive:
        prefix = archive.namelist()[0].split("/")[0] + "/"

        def raw(name):
            return archive.read(prefix + name)

        def obj(name):
            return json.loads(raw(name))

        def rows(name):
            return list(csv.DictReader(io.StringIO(raw(name).decode("utf-8-sig"))))

        package_manifest = obj("MANIFEST.json")
        expected = package_manifest["files"]
        check(
            set(archive.namelist()) == {prefix + n for n in expected} | {prefix + "MANIFEST.json"},
            "zip_file_set",
            "MANIFEST",
        )
        for name, info in expected.items():
            data = raw(name)
            check(
                len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"],
                "package_file_sha256",
                name,
            )
        print(json.dumps({"stage": "package_integrity", "files": len(expected), "issues": len(problems)}), flush=True)

        lookup = rows("WINDOW_LOOKUP.csv")
        patients = rows("PATIENT_INDEX.csv")
        decisions = rows("00_review_decisions/annotations/最终执行方案.csv")
        excluded = rows("00_review_decisions/excluded_windows.csv")
        actions = rows("00_review_decisions/annotations/executed_actions.csv")
        final_peaks = rows("00_review_decisions/retained_window_R_peaks.csv")
        tables = workbook_tables(raw("00_review_decisions/R_peak_final_no_manual_review (1).xlsx"))
        original_problems = {r["G编号"]: r for r in tables["440窗总表"]}
        final_sheet = {r["G编号"]: r for r in tables["最终执行方案"]}
        decision_by_g = {r["G编号"]: r for r in decisions}
        lookup_by_key = {r["sequence_key"]: r for r in lookup}
        check(
            len(lookup) == len(lookup_by_key) == len(decision_by_g) == len(original_problems) == 440,
            "review_key_uniqueness",
            "440 windows",
        )
        for row in decisions:
            match = final_sheet[row["G编号"]]
            for field in ("record", "case", "最终动作", "原因"):
                check(row[field] == match[field], "workbook_final_decision", row["G编号"] + ":" + field)
        for sheet_name, csv_name, time_field in (
            ("最终删除清单", "最终删除清单", "窗内时间（秒）"),
            ("最终峰位修正", "最终峰位修正", "建议时间（秒）"),
        ):
            sheets = {(r["G编号"], r["旧R编号"]): r for r in tables[sheet_name]}
            exported = rows("00_review_decisions/annotations/" + csv_name + ".csv")
            check(len(sheets) == len(exported), "workbook_action_count", sheet_name)
            for row in exported:
                match = sheets[(row["G编号"], row["旧R编号"])]
                check(
                    row["record"] == match["record"]
                    and row["case"] == match["case"]
                    and abs(float(row[time_field]) - float(match[time_field])) < 1e-9,
                    "workbook_peak_action",
                    row["G编号"] + ":" + row["旧R编号"],
                )

        refs, reference_mapping = {}, []
        reference_counts, split_counts = Counter(), Counter()
        global_indices = []
        for patient in patients:
            subject = patient["subject_key"]
            check(subject in records, "reference_patient_source", subject)
            record, meta = records[subject], metadata(subject)
            split_counts[record["split"]] += 1
            with np.load(io.BytesIO(raw(patient["corrected_rri_npz"])), allow_pickle=False) as data:
                fields = (
                    "sequence_key",
                    "global_group_index",
                    "identity_json",
                    "rr_true_s",
                    "rr_valid",
                    "rr_support_id",
                    "excluded_group",
                    "changed_group",
                    "raw_label_changed_group",
                )
                ref = {k: data[k].copy() for k in fields}
            n = len(ref["sequence_key"])
            check(
                ref["rr_true_s"].shape == ref["rr_valid"].shape == ref["rr_support_id"].shape == (n, 240),
                "rri_array_shapes",
                subject,
            )
            valid, drop = ref["rr_valid"], ref["excluded_group"]
            check(np.all((ref["rr_support_id"] >= 0) == valid), "rri_support_mask", subject)
            check(not valid[drop].any(), "excluded_rri_invalid", subject)
            check(np.isfinite(ref["rr_true_s"][valid]).all(), "valid_rri_finite", subject)
            check(
                n == int(patient["original_groups"])
                and int(drop.sum()) == int(patient["excluded_groups"])
                and int(valid[~drop].sum()) == int(patient["valid_RRI_points"]),
                "patient_index_counts",
                subject,
            )
            reference_counts.update(
                {
                    "patients": 1,
                    "rows": n,
                    "excluded": int(drop.sum()),
                    "retained": int((~drop).sum()),
                    "valid_points": int(valid[~drop].sum()),
                    "changed": int(ref["changed_group"].sum()),
                    "below216_retained": int(np.sum((valid.sum(axis=1) < 216) & ~drop)),
                }
            )
            global_indices.extend(ref["global_group_index"].tolist())
            for index, key in enumerate(ref["sequence_key"]):
                key = str(key)
                identity = json.loads(str(ref["identity_json"][index]))
                who, case, run, start, length = json.loads(key)
                end = start + length
                matched = (
                    who == subject
                    and identity["sequence_key"] == key
                    and length == 6
                    and identity["member_rows"] == list(range(start, end))
                    and np.all(meta["case_id"][start:end] == case)
                    and np.all(meta["run_id"][start:end] == run)
                    and meta["segment_id"][start:end].tolist() == identity["member_segment_id"]
                    and abs(float(meta["t_start"][start]) - identity["t_start"]) < 1e-9
                )
                check(matched, "reference_sequence_source_metadata", key)
                check(
                    bool(drop[index])
                    == (key in lookup_by_key and lookup_by_key[key]["final_disposition"] == "EXCLUDED"),
                    "reference_exclusion_key",
                    key,
                )
                gid = lookup_by_key.get(key, {}).get("group_id", "")
                reference_mapping.append(
                    {
                        "sequence_key": key,
                        "group_id": gid,
                        "subject_key": subject,
                        "signalmind_record_id": record["id"],
                        "source_split": record["split"],
                        "case_id": case,
                        "run_id": run,
                        "source_row0_start": start,
                        "source_row0_end_exclusive": end,
                        "t_start_s": identity["t_start"],
                        "corrected_rri_npz": patient["corrected_rri_npz"],
                        "corrected_local_row": index,
                        "global_group_index": int(ref["global_group_index"][index]),
                        "excluded": bool(drop[index]),
                        "changed": bool(ref["changed_group"][index]),
                        "valid_rri_points": int(valid[index].sum()),
                        "source_metadata_match": matched,
                    }
                )
            refs[subject] = ref
        check(sorted(global_indices) == list(range(25048)), "global_reference_indices", "all original rows")
        print(json.dumps({"stage": "all_reference_keys", **reference_counts, "issues": len(problems)}), flush=True)

        by_subject, by_action, by_peak = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in lookup:
            by_subject[row["subject_key"]].append(row)
        for action in actions:
            by_action[action["group_id"]].append(action)
        for peak in final_peaks:
            by_peak[peak["group_id"]].append(peak)
        review_mapping, problem_details = [], []
        action_mapping, raw_matches, refinement_differences = [], 0, []
        for subject, windows in by_subject.items():
            record, meta = records[subject], metadata(subject)
            mat = PulseMat(record["source"])
            try:
                for row in windows:
                    gid, key = row["group_id"], row["sequence_key"]
                    identity = obj(f"01_original_review_440/data/{gid}/identity.json")
                    with np.load(io.BytesIO(raw(row["original_npz"])), allow_pickle=False) as data:
                        wave = {k: data[k].copy() for k in data.files if k != "identity_json"}
                    members = identity["member_rows"]
                    source_rows = [r["mat_row0"] for r in identity["member_details"]]
                    check(
                        identity["subject_key"] == subject
                        and identity["sequence_key"] == key
                        and identity["source_mat"] == record["source"]
                        and source_rows == members,
                        "review_identity",
                        gid,
                    )
                    exact = True
                    for name, field in (("ecg", "ECG_Record"), ("abp", "ABP_Raw"), ("absolute_time_s", "T")):
                        original = np.concatenate([mat.array(field, i) for i in source_rows])
                        exact &= check(
                            np.array_equal(original, wave[name], equal_nan=True), "raw_array_exact", gid + ":" + name
                        )
                    raw_matches += exact
                    observed_meta = [mat.metadata(i) for i in source_rows]
                    check(
                        all(
                            m["regular"] and m["subject"] == record["subject"] and m["case"] == identity["case_id"]
                            for m in observed_meta
                        )
                        and [m["segment"] for m in observed_meta] == identity["member_segment_id"]
                        and np.allclose(np.diff(wave["absolute_time_s"]), 0.008, rtol=0, atol=1e-6),
                        "raw_time_identity",
                        gid,
                    )
                    check(
                        np.allclose(
                            wave["relative_time_s"], wave["absolute_time_s"] - identity["t_start"], rtol=0, atol=1e-9
                        ),
                        "relative_time",
                        gid,
                    )
                    ref, index = refs[subject], int(row["corrected_local_row"])
                    check(
                        str(ref["sequence_key"][index]) == key
                        and int(ref["global_group_index"][index]) == int(row["original_global_group_index"])
                        and bool(ref["excluded_group"][index]) == (row["final_disposition"] == "EXCLUDED")
                        and int(ref["rr_valid"][index].sum()) == int(row["RRI_valid_points"]),
                        "review_reference_row",
                        gid,
                    )
                    old_peaks = dict(
                        zip(wave["r_current_ids"].tolist(), wave["r_current_sample0"].tolist(), strict=True)
                    )
                    remaining = dict(old_peaks)
                    for action in by_action[gid]:
                        old_id, point = action["old_marker_id"], int(action["old_sample0_in_group"])
                        check(
                            old_peaks.get(old_id) == point
                            and action["sequence_key"] == key
                            and int(action["old_stored_sample0"]) == members[0] * 1250 + point,
                            "executed_old_peak_identity",
                            gid + ":" + old_id,
                        )
                        if action["action"] == "DELETE":
                            remaining.pop(old_id)
                            new_point = ""
                        else:
                            new_point = int(action["executed_sample0_in_group"])
                            remaining[old_id] = new_point
                            check(
                                0 <= new_point < 7500
                                and int(action["executed_stored_sample0"]) == members[0] * 1250 + new_point,
                                "executed_move_coordinate",
                                gid + ":" + old_id,
                            )
                            if new_point != int(action["new_sample0_in_group"]):
                                refinement_differences.append(
                                    {
                                        "group_id": gid,
                                        "old_R_id": old_id,
                                        "suggested_sample0": int(action["new_sample0_in_group"]),
                                        "executed_sample0": new_point,
                                    }
                                )
                        action_mapping.append(
                            {
                                "group_id": gid,
                                "sequence_key": key,
                                "signalmind_record_id": record["id"],
                                "old_R_id": old_id,
                                "old_sample0": point,
                                "action": action["action"],
                                "executed_sample0": new_point,
                                "reason": action["reason"],
                            }
                        )
                    final = by_peak[gid]
                    if row["final_disposition"] != "EXCLUDED":
                        final_map = {p["original_R_id"]: int(p["corrected_sample0"]) for p in final}
                        check(
                            remaining == final_map and len(final) == int(row["final_R_peaks"]), "final_peak_replay", gid
                        )
                        for peak in final:
                            point = int(peak["corrected_sample0"])
                            check(
                                peak["sequence_key"] == key
                                and abs(point / 125 - float(peak["corrected_time_in_window_s"])) < 1e-9
                                and abs(identity["t_start"] + point / 125 - float(peak["corrected_record_time_s"]))
                                < 1e-9,
                                "final_peak_time",
                                gid + ":" + peak["corrected_R_id"],
                            )
                    else:
                        check(not final and not by_action[gid], "excluded_no_peak_edits", gid)
                    issue = original_problems[gid]
                    review_mapping.append(
                        {
                            "group_id": gid,
                            "sequence_key": key,
                            "subject_key": subject,
                            "source_split": record["split"],
                            "signalmind_record_id": record["id"],
                            "source_mat": record["source"],
                            "case_id": identity["case_id"],
                            "run_id": identity["run_id"],
                            "source_row0_start": members[0],
                            "source_row0_end_exclusive": members[-1] + 1,
                            "source_t_start_s": identity["t_start"],
                            "source_t_end_exclusive_s": identity["t_start"] + 60,
                            "final_disposition": row["final_disposition"],
                            "corrected_rri_npz": row["corrected_rri_npz"],
                            "corrected_local_row": index,
                            "raw_arrays_exact": bool(exact),
                            "problem_statement": issue["主要问题"],
                            "problem_times_original_text": issue["问题时段（窗内秒）"],
                            "final_reason": row["final_action_reason"],
                        }
                    )
                    problem_details.append(
                        {
                            "group_id": gid,
                            "final_disposition": row["final_disposition"],
                            "original_window_status": issue["窗口结论"],
                            "problem_statement": issue["主要问题"],
                            "problem_times_original_text": issue["问题时段（窗内秒）"],
                            "review_notes": issue["修改与保留说明"],
                            "final_reason": row["final_action_reason"],
                        }
                    )
            finally:
                mat.file.close()
        check(
            {r["sequence_key"] for r in excluded}
            == {r["sequence_key"] for r in lookup if r["final_disposition"] == "EXCLUDED"},
            "exclusion_csv_keys",
            "277 windows",
        )
        save_csv(out / "window_mapping.csv", review_mapping)
        save_csv(out / "rri_mapping.csv", reference_mapping)
        save_csv(out / "peak_action_mapping.csv", action_mapping)
        save_csv(out / "problem_details.csv", problem_details)
        review_ids = {records[s]["id"] for s in by_subject}
        reference_ids = {records[s]["id"] for s in refs}
        used_ids = set(manifest["development_records"] + manifest["validation_records"])
        old_pilot = read(ROOT / "eval/cases/vital-real-20261105/private/source-copies.json")
        old_pilot_subjects = {"vital::" + Path(r["source_file"]).stem for r in old_pilot}
        changed_outside_review = sum(r["changed"] and not r["group_id"] for r in reference_mapping)
        summary = {
            "date": "2026-11-06",
            "archive": str(args.archive),
            "archive_sha256": digest,
            "status": "PASS" if not problems else "MISMATCHES_FOUND",
            "issues": problems,
            "checks": dict(counts),
            "package_files_verified": len(expected),
            "review_windows": len(lookup),
            "review_subjects": len(by_subject),
            "review_dispositions": dict(Counter(r["final_disposition"] for r in lookup)),
            "excluded_original_statuses": dict(Counter(r["original_review_status"] for r in excluded)),
            "review_waveforms_exact_to_source": raw_matches,
            "review_raw_fields": ["ECG_Record", "ABP_Raw", "T"],
            "reference_counts": dict(reference_counts),
            "reference_source_splits": dict(split_counts),
            "changed_reference_rows_outside_440_review": changed_outside_review,
            "executed_actions": dict(Counter(r["action"] for r in actions)),
            "retained_review_peak_rows": len(final_peaks),
            "move_refinement_differences": refinement_differences,
            "xlsx_table_counts": {k: len(v) for k, v in tables.items()},
            "previous_dataset_development_or_validation_record_overlap": len(reference_ids & used_ids),
            "previous_4_record_pilot_subject_overlap": len(set(refs) & old_pilot_subjects),
            "review_records_mapped_to_signalmind": len(review_ids),
            "reference_records_mapped_to_signalmind": len(reference_ids),
            "current_pipeline_has_peak_edit_operations": False,
            "raw_sources_modified": False,
            "model_calls": 0,
            "full_source_mat_files_rehashed": False,
            "interpretation": "Alignment verified, not clinical adjudication. Exclusions apply to named analysis windows, not entire source files. Peak deletions/moves do not alter ECG samples. Original problem text is retained without inventing a mutually exclusive taxonomy.",
        }
        write(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
