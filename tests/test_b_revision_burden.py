"""Data-only burden revisions use synthetic tool outputs and no case files."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import b_revision_burden as r2  # noqa: E402
from eval import b_revisions as br  # noqa: E402
from eval import b_workflow as bw  # noqa: E402


def test_unknown_midpoint_views_preserve_ranking_gap_arguments_and_inputs():
    index = {
        "episodes": [
            {"t": 20, "span_s": 80, "intervals_s": [[20, 25], [35, 60], [80, 100]]},
            {"t": 200, "span_s": 50, "intervals_s": [[200, 210], [225, 250]]},
        ]
    }
    noisy = {
        "events": [
            {"t": 600, "dur": 400},
            {"t": 200, "dur": 400},
            {"t": 125, "dur": 50},
            {"t": 50, "dur": 20},
        ]
    }
    before = copy.deepcopy((index, noisy))
    original = bw.burden_plan("signal", index, noisy, 1000)
    revised = r2.burden_plan("signal", index, noisy, 1000, 128)
    assert revised == r2.burden_plan("signal", index, noisy, 1000, 128)
    assert (index, noisy) == before
    assert [i["key"] for i in revised] == [i["key"] for i in original]
    assert revised[3:] == original[3:]
    for i, midpoint in enumerate((400, 800, 150)):
        assert revised[i]["arguments"] == {
            "id": "signal",
            "window": [midpoint - 15, midpoint + 15],
            "view": "ecg_paper",
        }
        assert {k: v for k, v in revised[i].items() if k != "arguments"} == {
            k: v for k, v in original[i].items() if k != "arguments"
        }


@pytest.mark.parametrize(
    "start,end,duration,fs,expected",
    [
        (0, 10, 100, 128, [0, 20]),
        (90, 100, 100, 128, [80, 100]),
        (0, 8, 8, 128, [0, 8]),
        (30.125, 60.125, 100, 4, [30, 60]),
        (30.25, 60.25, 100, 4, [30.25, 60.25]),
        (-100, -80, 100, 128, None),
        (200, 220, 100, 128, None),
    ],
)
def test_midpoint_paper_sample_alignment_and_clipping(start, end, duration, fs, expected):
    win = r2.midpoint_paper_window(start, end, duration, fs)
    assert win == expected
    if win:
        assert all((t * fs).is_integer() for t in win)
        assert 0 <= win[0] < win[1] <= duration


def test_arithmetic_preserves_original_and_classifies_only_support():
    index = {
        "episodes": [
            {"t": 3590, "span_s": 50, "intervals_s": [[3590, 3605], [3625, 3640]]},
            {"t": 7100, "span_s": 200, "intervals_s": [[7100, 7150], [7200, 7300]]},
        ]
    }
    burden = {"pct_of_window": 1.11, "af_s": 80, "window_s": 7200, "readable_s": 7000}
    before = copy.deepcopy((index, burden))
    revised = r2.burden_arithmetic(index, burden, "index-ref", "burden-ref")
    assert revised[:2] == bw.burden_arithmetic(index, burden, "index-ref", "burden-ref")
    assert (index, burden) == before
    extra = revised[2]
    assert extra["hours"] == [
        [0, 3600, 10, 10, 0, 3590],
        [3600, 7200, 70, 140, 70, 3460],
    ]
    assert extra["whole_record"] == [0, 7200, 80, 150, 70, 7050]
    assert extra["denominator_s"] == 7200
    assert extra["pct_of_window"] == burden["pct_of_window"]
    assert extra["source_refs"] == ["index-ref", "burden-ref"]
    assert extra["columns"][-2:] == [
        "within_span_non_support_unclassified_s",
        "outside_span_unclassified_s",
    ]
    assert "readable_s" not in extra


def test_arithmetic_unions_overlaps_clips_and_handles_empty_index():
    index = {
        "episodes": [
            {"t": -10, "span_s": 20, "intervals_s": [[-10, 5]]},
            {"t": 0, "span_s": 20, "intervals_s": [[0, 15]]},
            {"t": 20, "span_s": 50, "intervals_s": [[20, 70]]},
        ]
    }
    burden = {"pct_of_window": 85, "af_s": 25, "window_s": 30, "readable_s": 29}
    extra = r2.burden_arithmetic(index, burden, "i", "b")[2]
    assert extra["whole_record"] == [0, 30, 25, 30, 5, 0]
    empty = r2.burden_arithmetic({"episodes": []}, burden, "i", "b")[2]
    assert empty["whole_record"] == [0, 30, 0, 0, 0, 30]
    assert r2.burden_plan("id", {"episodes": []}, {"events": []}, 30, 128) == []


def test_fractional_arithmetic_final_hour_rounding_and_compact_envelope():
    index = {
        "episodes": [
            {
                "t": 3599.123456789,
                "span_s": 2,
                "intervals_s": [[3599.123456789, 3599.987654321], [3600.1, 3601.123456789]],
            }
        ]
    }
    burden = {"pct_of_window": 0.00085, "af_s": 1.887654321, "window_s": 88500.125, "readable_s": 80000}
    first = r2.burden_arithmetic(index, burden, "i" * 60, "b" * 60)
    second = r2.burden_arithmetic(index, burden, "i" * 60, "b" * 60)
    assert bw.canonical(first) == bw.canonical(second)
    extra = first[2]
    assert len(extra["hours"]) == 25
    assert extra["hours"][-1] == [86400, 88500.125, 0, 0, 0, 2100.125]
    assert extra["whole_record"][2:5] == [1.887654, 2, 0.112346]
    assert len(bw.canonical(extra)) <= 2500


@pytest.mark.parametrize("n,duration", [(128000, 1000), (128001, 1000.008)])
def test_failed_paper_stays_an_error_and_other_r2_evidence_preserves_v0(tmp_path, n, duration):
    class PaperErrorClient:
        async def call_tool(self, name, arguments):
            data = {"id": "synthetic"}
            if name in ("ts_load", "ts_describe"):
                data.update(dur_s=duration, channels=[{"name": "ch0", "fs": 128, "n": n}])
            elif name == "ts_find_events":
                data["events"] = [{"t": 300, "dur": 300}, {"t": 700, "dur": 100}]
            elif name == "ts_measure":
                what = arguments["what"][0]
                row = {"name": what, "ref": f"synthetic:{what}", "ch": "ch0"}
                if what == "ecg_af_burden":
                    row.update(af_s=90, pct_of_window=9, window_s=n / 128, readable_s=600)
                else:
                    row["episodes"] = [{"t": 100, "span_s": 100, "af_s": 90, "intervals_s": [[100, 150], [160, 200]]}]
                data["measurements"] = [row]
            if arguments.get("view") == "ecg_paper":
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="too few readable RR intervals in this window")],
                    is_error=True,
                )
            blocks = [SimpleNamespace(type="text", text=json.dumps(data, indent=1))]
            if name == "ts_zoom":
                blocks.append(SimpleNamespace(type="image", mime_type="image/png", data="cG5n"))
            return SimpleNamespace(content=blocks, is_error=False)

    case = {
        "case_id": "synthetic",
        "qtype": "af_burden",
        "question": "What is the AF burden?",
        "dir": str(tmp_path),
        "visible": {"file": "signal.npy", "channels": [{"name": "ch0", "unit": "mV", "fs": 128}]},
    }
    original = asyncio.run(bw.collect(case, PaperErrorClient()))
    revised = asyncio.run(br.collect(case, PaperErrorClient(), workflow="r2"))
    repeated = asyncio.run(br.collect(case, PaperErrorClient(), workflow="r2"))
    assert bw.canonical(revised) == bw.canonical(repeated)
    assert revised["status"] == "ready"
    assert revised["items"][:6] == original["items"][:6]
    assert revised["items"][-1] == original["items"][-1]
    assert revised["arithmetic"][:2] == original["arithmetic"]
    assert revised["sizes"]["calls"] == original["sizes"]["calls"] == 9
    assert revised["sizes"]["images"] == 2
    assert not revised["dropped"]
    errors = [item for item in revised["items"] if item["is_error"]]
    assert [item["key"] for item in errors] == ["unknown.1.strip", "unknown.2.strip"]
    assert all(item["content"] == [bw.text_block("too few readable RR intervals in this window")] for item in errors)
    assert all(item["arguments"]["view"] == "ecg_paper" for item in errors)
    assert all(bw.item_content(item)[-1] in revised["content"] for item in errors)
    assert revised["items"][4]["arguments"]["window"] == [0, duration]
    assert revised["arithmetic"][2]["whole_record"] == [0, round(n / 128, 6), 90, 100, 10, round(n / 128 - 100, 6)]
    assert revised["arithmetic"][2]["pct_of_window"] == 9
