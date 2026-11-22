"""Prevent missing annotations, missing sessions and price estimates becoming positive study conclusions."""

import json

import pytest

from eval import grade_unified as grading
from eval.level2_traces import render, session_key


@pytest.fixture
def item():
    return {
        "case_id": "case1",
        "family": "F3",
        "specialist_capability": "sub_beat_alignment",
        "abstention_correct": False,
        "expected_answer": {"shift_ms": 40.0},
        "tolerances": {"shift_ms": 2},
        "grounding_rubric": {
            "regions": [{"channels": ["ECG", "PPG"], "window_s": [20, 60]}],
            "required_computations": ["cross_channel_beat_alignment"],
        },
    }


def write_run(tmp_path, arm, rows):
    run = tmp_path / f"p9level2-U-{arm}-dev"
    run.mkdir()
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def result_row(cid, **extra):
    return {
        "case_id": cid,
        "pred": {"shift_ms": 40},
        "application_answer": {"shift_ms": 40},
        "cost_usd": 0.1,  # Legacy field is a token-price estimate, not a bill.
        **extra,
    }


def test_same_blind_coding_for_all_arms_and_missing_coding_is_pending(tmp_path, monkeypatch, item):
    monkeypatch.setattr(grading, "RUNS", tmp_path)
    coding = tmp_path / "coding"
    coding.mkdir()
    evidence = {
        "trace_ref": "s1",
        "channels": ["ECG", "PPG"],
        "window_s": [20, 60],
        "computations": ["cross_channel_beat_alignment"],
        "method": "matched beat times",
        "success": True,
        "verified": False,
        "independent_of": None,
        "detected_misleading_output": False,
        "quote": "lag = 40 ms",
    }
    for arm in ("DS-Base", "DS-Tools", "DS-Agent"):
        row = result_row("case1", pred={"shift_ms": 40 if arm.endswith("Base") else 400})
        run = write_run(tmp_path, arm, [row])
        ledger = run / "runtime" / "case1" / "ledger.jsonl"
        ledger.parent.mkdir(parents=True)
        ledger.write_text(json.dumps({"source_kind": "computed", "definition_id": "delay.pat.v1"}) + "\n")
        items = {"case1": item}
        pending = grading.grade_arm("dev", arm, items, coding)
        assert pending["case1"]["primary"]["correct"] is True
        assert pending["case1"]["specialist_tool_success"] is True
        assert pending["case1"]["primary"]["grounded_correct"] is None
        assert pending["case1"]["primary"]["correct_but_ungrounded"] is None
        assert grading.summary(pending, items, list(items))["grounding_status"] == "pending"

        key = session_key(arm, "case1")
        path = coding / f"{key}.json"
        path.write_text(json.dumps({"session": key, "records": [evidence]}))
        coded = grading.grade_arm("dev", arm, items, coding)
        assert coded["case1"]["primary"]["grounded_correct"] is True
        # A reviewed trace with no qualifying computation differs from a missing review.
        path.write_text(json.dumps({"session": key, "records": []}))
        coded_empty = grading.grade_arm("dev", arm, items, coding)
        assert coded_empty["case1"]["primary"]["grounded_correct"] is False
        assert coded_empty["case1"]["primary"]["grounding_status"] == "coded"


def test_missing_billed_cost_and_partial_annotations_do_not_become_zero(tmp_path, monkeypatch, item):
    monkeypatch.setattr(grading, "RUNS", tmp_path)
    write_run(
        tmp_path,
        "DS-Base",
        [
            result_row("a", cost_usd_actual=0.03, cost_usd_estimated=0.04),
            result_row("b"),
        ],
    )
    items = {"a": item, "b": item}
    rows = grading.grade_arm("dev", "DS-Base", items, None)
    rows["a"]["primary"] = grading.grade_answer({"shift_ms": 40}, item, [])
    s = grading.summary(rows, items, list(items))
    assert s["correct"] == 2
    assert s["cost_usd_actual_total"] is None
    assert s["cost_usd_actual_known_subtotal"] == 0.03
    assert s["cost_usd_actual_missing_n"] == 1
    assert s["cost_usd_estimated_total"] == 0.14
    assert s["grounding_status"] == "partial" and s["grounding_coded_n"] == 1
    assert s["grounded_correct"] is None and s["grounded_correct_coded_subset"] == 0
    assert s["latency_s_median"] is None


def test_paired_direction_denominators_and_empty_grounding_comparison(item):
    def row(value, status="ok"):
        return {"status": status, "primary": grading.grade_answer({"shift_ms": value}, item, None)}

    first = {"a": row(40), "b": row(40), "c": row(0), "d": {"status": "not_run"}}
    second = {"a": row(40), "b": row(0), "c": row(0, "execution_failure"), "d": row(40)}
    p = grading.paired(first, second, list(first))
    assert p["n"] == 3 and p["excluded_n"] == 1
    assert (p["both"], p["only_first"], p["only_second"], p["neither"]) == (1, 1, 0, 1)
    assert p["diff_first_minus_second"] == pytest.approx(1 / 3)
    reverse = grading.paired(second, first, list(first))
    assert reverse["diff_first_minus_second"] == pytest.approx(-1 / 3)
    assert reverse["newcombe95"] == pytest.approx([-p["newcombe95"][1], -p["newcombe95"][0]])
    unknown = grading.paired(first, second, list(first), metric="grounded_correct")
    assert unknown["n"] == 0
    assert unknown["diff_first_minus_second"] is None
    assert unknown["newcombe95"] is None and unknown["exact_mcnemar_p"] is None


def test_f5_and_dry_runs_do_not_count_unrun_members_as_failures(tmp_path, monkeypatch, item):
    monkeypatch.setattr(grading, "RUNS", tmp_path)
    shared = {"family": "F5", "pair_id": "a" * 64, "task_type": "shift", "answer_fields": ["shift_ms"]}
    control = {**item, **shared, "case_id": "control", "pair_role": "control"}
    abstention = {
        **item,
        **shared,
        "case_id": "abstention",
        "pair_role": "abstention",
        "abstention_correct": True,
        "expected_answer": {"insufficient_evidence": True, "reason": "missing"},
    }
    items = {"control": control, "abstention": abstention}
    run = write_run(tmp_path, "DS-Base", [result_row("control", dry_run=True)])
    dry = grading.grade_arm("dev", "DS-Base", items, None)
    assert grading.summary(dry, items, list(items))["n"] == 0
    assert grading.f5_summary(dry, items)["not_run"] == 1
    (run / "results.jsonl").write_text(json.dumps(result_row("control")) + "\n")
    partial = grading.grade_arm("dev", "DS-Base", items, None)
    assert grading.f5_summary(partial, items) == {
        "n": 0,
        "n_requested": 1,
        "not_run": 0,
        "incomplete": 1,
        "correct_abstention": 0,
        "fabrication": 0,
        "unnecessary_abstention": 0,
        "pair_correct": 0,
    }
    with (run / "results.jsonl").open("a") as stream:
        stream.write(
            json.dumps(result_row("abstention", pred={"insufficient_evidence": True, "reason": "missing"})) + "\n"
        )
    complete = grading.grade_arm("dev", "DS-Base", items, None)
    assert grading.f5_summary(complete, items)["pair_correct"] == 1


def test_trace_export_uses_the_existing_coding_keys_and_excludes_references(tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "ROOT", tmp_path)
    monkeypatch.setattr(grading, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    cases = tmp_path / "eval" / "cases"
    case_dir = cases / "p9level2" / "case1"
    case_dir.mkdir(parents=True)
    (cases / "p9level2.questions.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case1",
                "question": "Find the lag.",
                "expected_answer": {"DO_NOT_EXPORT": 17},
            }
        )
        + "\n"
    )
    (case_dir / "visible.json").write_text(
        json.dumps(
            {
                "dur_s": 60,
                "channels": [{"name": "PPG", "unit": "a.u.", "fs": 125}],
            }
        )
    )
    arm = "DS-Tools"
    run = write_run(tmp_path / "runs", arm, [result_row("case1")])
    (run / "transcripts").mkdir()
    (run / "transcripts" / "case1.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "I computed the lag. /tmp/DS-Tools/data"}]},
            }
        )
        + "\n"
    )
    out = tmp_path / "traces"
    assert grading.export_traces("dev", [arm], out)["sessions"] == 1
    key = session_key(arm, "case1")
    trace = (out / f"{key}.md").read_text()
    assert "Find the lag." in trace and "I computed the lag." in trace
    assert "DO_NOT_EXPORT" not in trace and arm not in trace
    assert json.loads((out / "index.json").read_text())[key]["arm"] == arm


def test_answer_line_primary_keeps_application_failure_and_never_backfills(tmp_path, monkeypatch, item):
    monkeypatch.setattr(grading, "RUNS", tmp_path)
    write_run(
        tmp_path,
        "DS-Agent",
        [
            result_row("text_only", application_answer=None, finalization="rejected"),
            result_row("application_only", pred=None, finalization="accepted"),
            result_row("different", pred={"shift_ms": 400}, finalization="accepted"),
            result_row("both", finalization="accepted"),
        ],
    )
    items = {cid: item for cid in ("text_only", "application_only", "different", "both")}
    legacy = grading.grade_arm("dev", "DS-Agent", items, None)
    shared = grading.grade_arm("dev", "DS-Agent", items, None, primary_output="answer_line")
    assert grading.summary(legacy, items, list(items))["correct"] == 3
    assert grading.summary(shared, items, list(items))["correct"] == 2
    assert grading.summary(shared, items, list(items), "application")["correct"] == 3
    assert shared["text_only"]["primary"]["correct"] and not shared["text_only"]["application"]["correct"]
    assert shared["text_only"]["finalization"] == "rejected"
    assert shared["application_only"]["primary"]["outcome"] == "no_answer"
    assert legacy["text_only"]["primary"]["outcome"] == "no_answer"
    assert shared["different"]["primary"]["correct"] is False
    assert shared["different"]["application"]["correct"] is True
    assert all(r["primary_source"] == "answer_line" for r in shared.values())
    assert all(r["primary"]["grounded_correct"] is None for r in shared.values())


def test_base_application_is_not_applicable_not_a_failed_or_backfilled_answer(tmp_path, monkeypatch, item):
    monkeypatch.setattr(grading, "RUNS", tmp_path)
    write_run(tmp_path, "Astra-Base", [result_row("case1")])
    items = {"case1": item}
    rows = grading.grade_arm("dev", "Astra-Base", items, None)
    assert rows["case1"]["primary_source"] == "answer_line"  # historical Base scoring is preserved
    assert rows["case1"]["application"] is None and rows["case1"]["application_pred"] is None
    app = grading.summary(rows, items, list(items), "application")
    assert app["n"] == 0 and app["not_applicable"] == 1 and app["correct_rate"] is None
    assert app["correct"] is None
    assert grading.paired(rows, rows, list(items), which="application")["n"] == 0


def test_cli_declares_selected_primary_and_separate_output_dimensions(tmp_path, monkeypatch, item):
    monkeypatch.setattr(grading, "RUNS", tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"questions": [{**item, "expected_solvability": "specialist", "signal_kind": "real_signal"}]})
    )
    monkeypatch.setattr(grading, "MANIFEST", manifest)
    write_run(tmp_path, "DS-Base", [result_row("case1", pred={"shift_ms": 400})])
    write_run(tmp_path, "DS-Agent", [result_row("case1", application_answer=None, finalization="rejected")])
    write_run(tmp_path, "Astra-Base", [result_row("case1", pred={"shift_ms": 400})])
    destination = tmp_path / "new-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "grade_unified.py",
            "--tag",
            "dev",
            "--arms",
            "DS-Base",
            "DS-Agent",
            "Astra-Base",
            "--primary-output",
            "answer_line",
            "--compare",
            "DS-Agent",
            "Astra-Base",
            "--out",
            str(destination),
        ],
    )
    grading.main()
    result = json.loads(destination.read_text())
    assert result["primary_output"] == "answer_line"
    assert result["primary_sources"] == {
        "DS-Base": "answer_line",
        "DS-Agent": "answer_line",
        "Astra-Base": "answer_line",
    }
    assert result["summaries"]["DS-Agent"]["answerable"]["correct"] == 1
    assert "answer_line_secondary" not in result["summaries"]["DS-Agent"]
    assert result["output_summaries"]["DS-Agent"]["application"]["answerable"]["correct"] == 0
    assert result["output_summaries"]["DS-Base"]["application"]["answerable"]["not_applicable"] == 1
    assert result["submission_states"]["DS-Agent"]["finalization"] == {"rejected": 1}
    pair = result["paired_within_model"]["DS"]["DS-Agent_vs_DS-Base"]["numeric_correctness"]["answerable"]
    assert pair["only_first"] == 1 and pair["only_second"] == 0
    cross = result["paired_requested"]["DS-Agent_vs_Astra-Base"]
    assert cross["first"] == "DS-Agent" and cross["second"] == "Astra-Base"
    assert cross["numeric_correctness"]["answerable"]["only_first"] == 1
    assert "Astra" not in result["paired_within_model"]


def test_complete_trace_retains_late_measurements_without_thinking_or_image_data():
    visible = {"dur_s": 180, "channels": [{"name": "ECG", "fs": 125, "unit": "mV"}]}
    late_measurement = '"ecg_hr":79.2'
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "PRIVATE_THINKING"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "ts_measure",
                        "input": {"what": ["pulse_quality", "ecg_hr"]},
                    },
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "quality intervals: " + "x" * 5000 + late_measurement},
                            {"type": "image", "source": {"data": "DO_NOT_INCLUDE_IMAGE_DATA"}},
                        ],
                    }
                ]
            },
        },
    ]
    legacy = render(events, "Find the rate", visible)
    assert legacy == render(events, "Find the rate", visible, text_limit=4000)
    assert late_measurement not in legacy and "truncated" in legacy
    full = render(events, "Find the rate", visible, text_limit=0)
    assert late_measurement in full and "truncated" not in full
    assert "[image returned by s1]" in full
    assert "PRIVATE_THINKING" not in full and "DO_NOT_INCLUDE_IMAGE_DATA" not in full
