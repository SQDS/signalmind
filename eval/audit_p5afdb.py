"""Reproduce the offline, evidence-linked p5afdb failure audit (no model calls).

The case judgments below are explicit human-readable audit annotations, not a
classifier trained on reference answers. Input artifacts are read only. The
unchanged grader defines scope; transcript line numbers and projected tool
payloads make each judgment reviewable without copying waveform/image data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .grade import grade
else:
    from grade import grade

MODELS = {
    "C": "claude-opus-5-5-medium",
    "D": "deepseek-flash_deepseek-flash-high",
}
AUDIT_MANIFEST_SHA256 = "7e20dbc4d869b3c21649dac44331f285c4d01d4b93c691369c7a91a4f18f602c"

CATEGORIES = (
    "task_reference_issue",
    "detector_limitation",
    "tool_retrieval_or_presentation",
    "agent_decision_error",
    "calculation_or_reporting_error",
)


def judgment(category, reason, lines, span=None, recovery=False, how="", downstream=""):
    return {
        "earliest_failure": category,
        "reason": reason,
        "decisive_call_lines": lines,
        "agent_selected_span_s": span,
        "recoverable_with_existing_tools": recovery,
        "recovery_how": how,
        "downstream_issue": downstream,
    }


# Each key is model / detector arm / case. The task's required scope has 28 rows.
JUDGMENTS = {
    "C-R-afdb000": judgment(
        "task_reference_issue",
        "The question says continuous but does not disclose reference merging across non-AF gaps <30 s. "
        "Raw truth [10539.644,10959.248) and [10980.648,11118.324) has a real 21.400 s break. "
        "The agent rejects joining after seeing a regular/P-wave stretch and chooses the genuinely "
        "longer raw uninterrupted [1065.992,1505.312) episode. R's top onset is grade-correct, "
        "but its 415 s span omits the second truth segment; this is not simply ignoring the right event.",
        [6, 46, 52],
        [1064, 1505],
        True,
        "Yes, with the <30 s task convention made explicit and verification of the 10959–10981 s break. "
        "Tools already supplied the relevant region; an onset-close candidate alone would not establish continuity.",
        "Detector fragmentation enlarges the observed support gap to >30 s, so blind grouping of R spans "
        "would still split it. The reference convention and rhythm-boundary check both matter.",
    ),
    "C-M-afdb001": judgment(
        "detector_limitation",
        "M's whole-window support underestimates annotation burden: 15.3% vs 21.5425%. "
        "The model reports the received numerator and full-record denominator correctly.",
        [6, 9],
        how="No demonstrated correction from these calls. More waveform/continuity checks could identify missed AF, "
        "but the transcript contains no quantified missing support sufficient to cross the grading threshold.",
    ),
    "C-R-afdb001": judgment(
        "no_failure",
        "R exposes 18.5% of window and the agent uses it; the residual 3.043 pp deficit passes.",
        [6],
        recovery=True,
        how="Already correct; direct whole-window burden is sufficient.",
    ),
    "C-R-afdb008": judgment(
        "detector_limitation",
        "R splits the annotated [19633.164,28472.480) longest into 16 overlapping fragments "
        "and ranks a different [9697.240,13749.396) episode first. The agent then overmerges 7403–31600 s, "
        "crossing multiple annotated non-AF gaps >=30 s, including 19597.784–19633.164 (35.380 s).",
        [6, 21, 56, 81],
        [7403, 31600],
        False,
        "No demonstrated sufficient check: only selected gaps were inspected. A systematic scan of all competing "
        "runs and suspected terminations could help, but early fragments and a broad RR cloud do not prove continuity.",
        "Agent overaggregation extends the final span across the true longest; its error is an inaccurate early "
        "boundary/overmerge, not a disjoint final episode.",
    ),
    "C-R-afdb017": judgment(
        "detector_limitation",
        "Truth is continuous AF [7098.832,36823.040), but R returns two episodes separated "
        "by 54.236 s. The agent interprets a regular RR stretch near 12820–12867 as termination; numerical "
        "regularity does not establish non-AF/non-flutter.",
        [6, 11],
        [12874.8, 36823.04],
        how="No demonstrated recovery from the received RR image alone. ECG paper/beat checks of the disputed "
        "gap are available, but reliable rhythm discrimination remains uncertain; merely selecting 7089.724 is hindsight.",
        downstream="Agent accepts RR regularity as a real termination; it does not overlook a hidden onset.",
    ),
    "C-M-afdb019": judgment(
        "detector_limitation",
        "M's 160 fragmented/false-positive episodes obscure the reference longest. "
        "Its longest [23439.952,24106.612) overlaps truth but starts 110.704 s late. The model instead "
        "merges irregular RR across 17730–24100 s despite multiple annotated breaks >=30 s.",
        [7, 38, 44, 49],
        [17730, 24100],
        False,
        "No demonstrated complete recovery in this M transcript. Local gap verification and competing-episode "
        "checks could help (Claude-R succeeds), but broad irregularity cannot justify this full interval.",
        "Agent overmerges across true terminations; top_k=30 truncation exists but the near-target event was received.",
    ),
    "C-R-afdb019": judgment(
        "no_failure",
        "The agent rejects R's false-positive longest at 3558.052 and selects the third-ranked "
        "23311.608–24106.612 span, refining its start to 23311 (18.248 s early).",
        [6, 59, 60, 69],
        [23311, 24105],
        True,
        "Already correct through candidate comparison and boundary inspection. This is a demonstrated use of "
        "existing tools, not evidence that every gap was correctly interpreted.",
        "Final prose cites 18150 s as a check inside 18872–19710 s, which is outside that interval; "
        "the successful onset does not validate all of its continuity claims.",
    ),
    "C-R-afdb020": judgment(
        "calculation_or_reporting_error",
        "R's 26.4% whole-window result is grade-correct (4.813 pp error). "
        "The agent correctly identifies that denominator, then constructs 27.0% by an unsupported upward "
        "uncertainty adjustment for unreadable time. That 0.6 pp uplift crosses tolerance (5.413 pp error).",
        [6, 53],
        recovery=True,
        how="Yes: retain 26.4 as the evidence-backed point estimate; report unknown-time uncertainty separately. "
        "No new measurement or denominator change is needed.",
        downstream="R still overcalls annotated burden by 4.8 pp, but that upstream residual does not cause the grade miss.",
    ),
    "C-M-afdb025": judgment(
        "task_reference_issue",
        "The agent splits the real approximately 20–24 s regular break at 9262–9283. "
        "Truth raw intervals [8948.752,9261.652) and [9285.956,9956.668) are separated by 24.304 s, "
        "which the undisclosed reference <30 s convention merges. Its selected raw continuous episode "
        "therefore has the right later raw onset but fails the merged-reference task.",
        [6, 20, 25],
        [9283, 9958],
        True,
        "Yes, once the reference gap convention is stated: apply it to verified rhythm intervals, including "
        "the observed <30 s break. M's native <=15 s grouping is insufficient and must not define truth.",
        "M fragments the reference episode into seven detector spans; that is an additional limitation.",
    ),
    "C-R-afdb025": judgment(
        "no_failure",
        "R ranks the reference episode first at 8978.504; ECG onset inspection refines it "
        "to 8949, only 0.248 s from annotation.",
        [6, 21, 27],
        [8949, 9957],
        True,
        "Already correct. Distinguishes the subsequent roughly 30 s true termination and examines the onset.",
    ),
    "D-M-afdb001": judgment(
        "detector_limitation",
        "M exposes only 5626.272 s AF support (15.3%), short of 7932.604 s annotation "
        "(21.5425%). Window checks reproduce underdetection; the agent reports the full-window fraction correctly.",
        [110, 6095, 6097, 6099],
        how="No demonstrated quantified missing support. Verification of a few positive and negative examples "
        "does not validate the burden of every interval.",
    ),
    "D-R-afdb001": judgment(
        "no_failure",
        "The agent takes R's 18.5% of window, separates 18.8% readable-time normalization, "
        "and rejects an inflated RR-CV estimate after spot checks.",
        [136, 17060, 27884],
        recovery=True,
        how="Already correct; direct burden and spot checks support the chosen answer within tolerance.",
    ),
    "D-R-afdb006": judgment(
        "detector_limitation",
        "R severely underdetects/segments the annotated 12662.664–19012.868 episode "
        "(31 overlapping fragments; longest detected span only 494.756 s). The agent verifies a terminal "
        "AF/flutter episode near 33245 and ranks it against detector fragments, missing the earlier 6350 s run.",
        [146, 7429, 15092, 43239],
        [33245, 36186],
        False,
        "No demonstrated full recovery: verifying terminal flutter does not establish the duration of earlier "
        "fragmented AF. Existing wave/RR views permit more comparison, but recognizing all missing support is unresolved.",
        "Wrong episode selected: final 33245–36186 span has zero overlap with the annotated longest.",
    ),
    "D-R-afdb007": judgment(
        "detector_limitation",
        "R's 37.8% whole-window support misses more than half of true AF/AFL time "
        "(truth 81.3393%, sensitivity 45.97%). The agent adds about 3300 s of inferred regular flutter, "
        "reaching 46.5%, but never reconstructs the large remaining missing support.",
        [15, 3607, 13950, 18266],
        how="No demonstrated complete correction. Existing images identify some regular tachycardia, but more "
        "than 12800 s of true AF/AFL remains unaccounted for in the answer. Better AF/AFL classification/continuity "
        "evidence is needed, not a denominator fix.",
        downstream="Approximate flutter durations and subtraction of overlap are secondary aggregation uncertainty; "
        "their small arithmetic discrepancy cannot explain a 34.839 pp miss.",
    ),
    "D-R-afdb008": judgment(
        "detector_limitation",
        "The reference longest 19633.164–28472.480 is fragmented into 16 R episodes. "
        "R instead ranks 9697.240–13749.396 first; the agent validates this AF segment and nearby early gaps, "
        "then answers its onset without establishing that later fragmented runs are shorter.",
        [161, 4059, 6148, 13136],
        [9697.24, 13749.396],
        False,
        "No demonstrated recovery. Inspect all competing long runs and verify their gaps; positive AF validation "
        "of the first-ranked run does not establish longest-event selection.",
        "Wrong episode selected (zero temporal overlap), with incomplete comparison after the detector ranking error.",
    ),
    "D-M-afdb011": judgment(
        "detector_limitation",
        "M fragments the annotated 25097.396–36823.040 continuous episode into 58 pieces. "
        "The chosen 26208.696–27540.352 span is inside it, but begins 1111.3 s late. The agent even sees "
        "an onset near 25090 and irregular gaps but keeps the longest fragment.",
        [],
        [26208.696, 27540.352],
        False,
        "No demonstrated complete M recovery: some gaps were checked, others looked organized, and regular AF/flutter "
        "cannot be settled from an onset-close fragment. DeepSeek-R demonstrates a more extensive verification strategy.",
        "Agent treats the detector's native fragmentation/strict no-gap rule as episode truth despite contrary "
        "local evidence; report this contributor separately from earliest detector failure.",
    ),
    "D-R-afdb011": judgment(
        "no_failure",
        "The agent rejects R's top fragment at 28261.544, inspects earlier gaps, and follows "
        "the run back to an observed beat transition near 25090, within 7.396 s of annotation.",
        [],
        [25090, 36045],
        True,
        "Already correct through gap and onset verification; final claimed termination near 36045 disagrees "
        "with annotation through record end, which this onset-only grader does not test.",
    ),
    "D-M-afdb017": judgment(
        "no_failure",
        "Despite M's 109 fragments of the annotated longest, local beat times bracket onset "
        "at 7091.660–7092.896 and the agent answers 7092 (6.832 s early).",
        [],
        [7092, 36823.04],
        True,
        "Already correct through backward onset search and continuity checks. It does not specifically validate "
        "the regular 12820–12874 s gap that misleads both R agents, so this is not proof that gap is easy to classify.",
    ),
    "D-R-afdb017": judgment(
        "detector_limitation",
        "R splits truth's uninterrupted 7098.832–36823.040 AF into two spans. The "
        "agent verifies 12825–12870 s is regular (CV 0.0881, pNN50 0.0806), then equates organized RR with "
        "true episode termination. This is a classification/continuity limitation, not a timestamp conversion.",
        [],
        [12874.792, 36823.04],
        False,
        "No demonstrated recovery from these RR/beat measurements; ECG morphology in the regular gap may help, "
        "but a reliable AF/flutter-vs-organized rhythm judgment is not established by receipt of the earlier onset.",
        "Agent interpretation reinforces detector fragmentation; both candidate onsets were visible.",
    ),
    "D-R-afdb019": judgment(
        "detector_limitation",
        "R's top 3558.052–4627.768 candidate is entirely outside annotated AF, which "
        "first begins at 12311.696; it is a false-positive episode, not merely a late boundary. The true "
        "longest is present as third-ranked 23311.608–24106.612, but the agent accepts the wrong candidate.",
        [],
        [3558.052, 4627.768],
        False,
        "No demonstrated correction in these calls. Comparing within-candidate gaps and alternate episodes "
        "can help (Claude-R succeeds), but a false-positive rhythm label is not fixed by simply retrieving more timestamps.",
        "The final cites a strip at 5000–5020 as inside 3558–4628; it is outside. Verification of positive "
        "snippets and late-run gaps did not establish this candidate's full continuity or identity.",
    ),
    "D-M-afdb020": judgment(
        "detector_limitation",
        "M's whole-window 30.4% exceeds true 21.5866%; the diagnostic support contains "
        "substantial false positives. The agent faithfully divides 11207.272 by 36823.04 and reports 30.4.",
        [41, 3587, 4738, 9991],
        how="No demonstrated quantified removal of false-positive support. Needs reliable classification across "
        "flagged intervals; positive spot checks do not justify the entire burden.",
    ),
    "D-R-afdb020": judgment(
        "no_failure",
        "R reduces the overcall to 26.4% of window (4.813 pp error), and the agent uses the "
        "correct denominator instead of 28.9% readable time.",
        [67, 5391, 5393],
        recovery=True,
        how="Already correct. Half-window numerators add to the whole-record numerator; success does not "
        "prove unreadable time is non-AF or every visually labeled positive is true AF.",
    ),
    "D-M-afdb022": judgment(
        "detector_limitation",
        "M exposes 15.5% vs true 25.1237%: AF is fragmented/underdetected and the "
        "RR detector misses regular AFL. The agent recognizes ambiguous regular fast bands but declines "
        "to add them. Adding only their approximately 4.2 pp would still narrowly miss tolerance.",
        [57, 12989, 12991, 16979],
        how="No demonstrated full correction using M's received support. Regular rhythm classification plus "
        "missing support inside AF clouds is needed; R's success combines a larger AF numerator with inferred flutter.",
        downstream="Conservative uncertainty about flutter is not itself proof of an agent decision error.",
    ),
    "D-R-afdb022": judgment(
        "no_failure",
        "R exposes 18.4%, itself below tolerance. The agent identifies two regular "
        "tachycardia runs (484.2+1076.7 s), adds them to 6776.5 s AF, and answers 22.6% (2.524 pp low).",
        [72, 1317, 2032, 2034, 4717],
        recovery=True,
        how="Already correct through available waveform/tachycardia evidence and interval construction. "
        "Flutter inference from abrupt rate changes is plausible here but not a validated general diagnostic rule.",
    ),
    "D-M-afdb023": judgment(
        "no_failure",
        "M's longest fragment starts at 23782.832, but the agent checks gaps and follows the "
        "sustained run backward to 10263.6, within 12.476 s of the annotated onset.",
        [53, 1062, 4773, 4775, 5766],
        [10263.6, 36823.04],
        True,
        "Already correct with local gap and onset checks; demonstrates an existing-tool verification path.",
    ),
    "D-R-afdb023": judgment(
        "detector_limitation",
        "R first fragments the truly continuous 10251.124–36823.040 AF into nine "
        "spans, ranking 11616.248 first. Crucially, the agent subsequently receives direct gap evidence "
        "(CV 0.194/0.330/0.264/0.214; pNN50 0.707/0.627/0.915/0.566), explicitly proposes 10258, "
        "then defers to detector segmentation in its final answer.",
        [191, 2211, 2213, 11862, 11864],
        [11616.2, 18535.032],
        True,
        "Yes: the transcript already supports the alternative onset ~10258 and reports it explicitly. "
        "Use verified rhythm continuity, not a detector support gap as proof of a >=30 s sinus interval. "
        "This is stronger evidence than mere presence of an onset-close fragment.",
        "Clear downstream agent decision error: sufficient conflicting evidence was received and acknowledged. "
        "The earliest category remains detector under the chronological causal convention.",
    ),
    "D-M-afdb025": judgment(
        "detector_limitation",
        "M fragments the true longest into seven spans and ranks a late fragment. The received list includes an onset-close 8977.488 candidate but also a false precursor at 8892.480. "
        "The agent compensates by merging the cluster and extrapolating onset to 8886, but truth starts "
        "8948.752 (62.752 s later). It also merges across the true 30.476 s break after 9956.668.",
        [224, 1647, 2869, 3478],
        [8886, 10438],
        False,
        "No demonstrated exact onset recovery: requested onset strips stop at 8920 and do not inspect the "
        "8948.752 transition. A focused boundary check near 8920–8990 is available and would test this "
        "hypothesis; a correct answer cannot be guaranteed from the present evidence.",
        "Downstream early-boundary interpretation and overmerge; the final span overlaps the correct episode, "
        "so this is not selection of a disjoint episode.",
    ),
    "D-R-afdb025": judgment(
        "no_failure",
        "R's longest start 8978.504 is already within tolerance; the agent returns 8978.5. "
        "Its speculation about merging both candidates crosses the real 30.476 s non-AF gap, but does "
        "not change the graded onset.",
        [90, 305, 768],
        [8978.5, 10430.116],
        True,
        "Already correct numerically; onset grading does not validate its overmerge/episode-end interpretation.",
    ),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(value, key=""):
    """Preserve candidates/scalars, summarize large numeric arrays; no waveform/image copies."""
    if isinstance(value, dict):
        return {k: project(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if key == "intervals_s":
            return {"n_intervals": len(value), "first": value[:1], "last": value[-1:]}
        if len(value) > 12 and all(not isinstance(v, dict) for v in value):
            return {"n_values": len(value), "first_3": value[:3], "last_3": value[-3:]}
        return [project(v) for v in value]
    return value


def transcript_trace(path):
    calls, final_text = {}, []
    for line_number, row in enumerate(read_jsonl(path), 1):
        blocks = row.get("message", {}).get("content", [])
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if row.get("type") == "assistant" and block.get("type") == "text":
                final_text.append({"line": line_number, "text": block["text"]})
            if block.get("type") == "tool_use":
                calls[block["id"]] = {
                    "call_line": line_number,
                    "tool": block["name"].removeprefix("mcp__tsmcp__"),
                    "arguments": block["input"],
                }
            if block.get("type") == "tool_result":
                call = calls[block["tool_use_id"]]
                call["result_line"] = line_number
                call["is_error"] = block.get("is_error", False)
                parts = block.get("content", [])
                if isinstance(parts, str):
                    parts = [{"type": "text", "text": parts}]
                call["image_received"] = any(p.get("type") == "image" for p in parts)
                texts = [p["text"] for p in parts if p.get("type") == "text"]
                outputs = []
                for text in texts:
                    try:
                        outputs.append(json.loads(text))
                    except ValueError:
                        outputs.append({"text": text})
                call["returned_evidence"] = [project(out) for out in outputs]
                call["text_chars"] = sum(map(len, texts))
    return sorted(calls.values(), key=lambda c: c["call_line"]), final_text


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def endpoint_geometry(span, target, correct):
    if span is None:
        return None
    intersection = overlap(span, target)
    union = span[1] - span[0] + target[1] - target[0] - intersection
    if correct:
        subtype = "correct_onset_not_a_test_of_full_episode"
    elif intersection == 0:
        subtype = "wrong_episode_selected"
    elif span[0] < target[0] - 60 and span[1] - span[0] > target[1] - target[0]:
        subtype = "right_episode_in_overmerged_span_inaccurate_boundary"
    else:
        subtype = "right_episode_overlap_inaccurate_boundary"
    return {"subtype": subtype, "overlap_s": intersection, "iou": intersection / union if union else 0}


def burden_axes(key, correct):
    """Separate recognition errors from denominator, aggregation and answer construction."""
    result = {
        "classification": "detector support has residual error; see reference/support comparison",
        "denominator": "whole-record window denominator used correctly",
        "interval_aggregation": "uses tool support numerator; no manual span summation",
        "answer_construction": "reports received whole-window percentage",
    }
    if key == "C-R-afdb020":
        result["classification"] = "detector overcall is within tolerance before agent uplift"
        result["answer_construction"] = "unsupported 26.4 to 27.0 upward adjustment causes miss"
    if key == "D-R-afdb007":
        result["classification"] = "severe missing AF/AFL support dominates"
        result["interval_aggregation"] = (
            "adds approximate flutter durations, subtracts estimated already-counted support"
        )
        result["answer_construction"] = (
            "approximate 46.5%; small arithmetic ambiguity does not explain 34.839 pp deficit"
        )
    if key == "D-R-afdb022":
        result["classification"] = (
            "regular tachycardia inferred as AFL; improves score but remains a morphology hypothesis"
        )
        result["interval_aggregation"] = "6776.5 + 484.2 + 1076.7 seconds, divided by 36823.04"
        result["answer_construction"] = "22.6% passes; no demonstrated aggregation error"
    result["grade_correct"] = correct
    return result


def build_audit(input_root):
    manifest_path = Path(__file__).parent / "results/audit-a0/input-manifest.json"
    if digest(manifest_path) != AUDIT_MANIFEST_SHA256:
        raise ValueError("Curated audit is bound to the original input-manifest SHA256; re-review changed inputs")
    manifest = json.loads(manifest_path.read_text())["inputs"]
    answer_path = input_root / "eval/cases/p5afdb.answers.jsonl"
    answers = {r["case_id"]: r for r in read_jsonl(answer_path)}
    inputs = {str(answer_path): digest(answer_path)}
    for relative in ("eval/tasks/afdb_eligibility.json", "eval/cases/p5afdb.questions.jsonl"):
        path = input_root / relative
        if path.exists():
            inputs[str(path)] = digest(path)
    detectors = {}
    for arm in "MR":
        path = input_root / f"eval/results/afdb-detector/{arm}.json"
        artifact = json.loads(path.read_text())
        detectors[arm] = {r["record"].split("/")[-1]: r for r in artifact["records"]}
        inputs[str(path)] = digest(path)
    rows = []
    for short, model in MODELS.items():
        runs = {}
        for arm in "MR":
            path = input_root / f"eval/runs/p5afdb-C-{model}-{arm}/results.jsonl"
            runs[arm] = {r["case_id"]: r for r in read_jsonl(path)}
            inputs[str(path)] = digest(path)
        for case, reference in sorted(answers.items()):
            grades = {a: grade(reference["qtype"], reference["answer"], runs[a][case]["pred"]) for a in "MR"}
            flip = bool(grades["M"]["correct"]) != bool(grades["R"]["correct"])
            for arm in "MR":
                if not (flip or (arm == "R" and not grades["R"]["correct"])):
                    continue
                key = f"{short}-{arm}-{case}"
                annotation = JUDGMENTS[key].copy()
                run = runs[arm][case]
                record = reference["answer"]["record"]
                det = detectors[arm][record]
                transcript = input_root / f"eval/runs/p5afdb-C-{model}-{arm}/transcripts/{case}.jsonl"
                inputs[str(transcript)] = digest(transcript)
                calls, texts = transcript_trace(transcript)
                # Empty lists request the principal evidence, including every targeted measure.
                if not annotation["decisive_call_lines"]:
                    annotation["decisive_call_lines"] = [
                        c["call_line"]
                        for c in calls
                        if c["tool"] in {"ts_measure", "ts_find_events"} and not c.get("is_error")
                    ]
                decisive = [c for c in calls if c["call_line"] in annotation["decisive_call_lines"]]
                assert len(decisive) == len(annotation["decisive_call_lines"]), key
                correct = bool(grades[arm]["correct"])
                assert (annotation["earliest_failure"] == "no_failure") == correct, key
                pred_field = "t" if reference["qtype"] == "longest_af" else "percent"
                error = abs(run["pred"][pred_field] - reference["answer"][pred_field])
                selected = annotation.pop("agent_selected_span_s")
                row = {
                    "key": key,
                    "case_id": case,
                    "model": model,
                    "model_short": short,
                    "arm": arm,
                    "record": record,
                    "qtype": reference["qtype"],
                    "correct": correct,
                    "error_size": error,
                    "error_unit": "s" if pred_field == "t" else "percentage_points",
                    "scope": {"incorrect_R": arm == "R" and not correct, "correctness_flip": flip},
                    "reference": reference["answer"],
                    "reference_trace": {
                        "coordinates": "seconds from record start; half-open; first lead; fs=250 Hz",
                        "definition": "AFIB+AFL; raw support for burden; gaps strictly <30 s merged for longest",
                        "annotated_af_intervals": det["annotated_af_intervals"],
                        "annotated_episodes": det["annotated_episodes"],
                        "longest_annotated": det["longest_annotated"],
                        "true_af_s": det["true_af_s"],
                        "window_s": det["window_s"],
                    },
                    "detector_trace": {
                        "longest_detected": det["longest_detected"],
                        "longest_iou": det["longest_iou"],
                        "longest_onset_error_s": det["longest_onset_error_s"],
                        "longest_within_tolerance": det["longest_within_tolerance"],
                        "fragmentation_of_reference_longest": det["fragmentation"],
                        "public_burden": det["public_burden"],
                        "detected_episodes": project(det["detected_episodes"]),
                        "sensitivity": det["sensitivity"],
                        "ppv": det["ppv"],
                        "support_representation": "native M episode spans"
                        if arm == "M"
                        else "raw R AF support intervals",
                        "source": f"eval/results/afdb-detector/{arm}.json#records/{record}",
                    },
                    "available_tools": "ts_find_events ecg_irregular candidates (both); ecg_af_episodes (R only); ecg_af_burden; "
                    "windowed ecg_rr_profile, ecg_rr_stats, ecg_beats; ts_zoom ecg_rr/ecg_paper/wave. "
                    "Candidate availability is not proof of rhythm identity or continuity.",
                    "transcript_path": str(transcript),
                    "calls_and_received_evidence": calls,
                    "decisive_evidence": decisive,
                    "agent_public_text": texts,
                    "agent_decision_and_final_answer": run["text"],
                    "pred": run["pred"],
                    "agent_selected_span_s": selected,
                    "longest_geometry": endpoint_geometry(selected, det["longest_annotated"], correct),
                    "burden_axes": burden_axes(key, correct) if pred_field == "percent" else None,
                    **annotation,
                }
                rows.append(row)
    for path, sha in inputs.items():
        relative = str(Path(path).relative_to(input_root))
        if relative not in manifest or manifest[relative]["sha256"] != sha:
            raise ValueError(f"Curated diagnosis input changed: {relative}; re-review required")
    assert {r["key"] for r in rows} == set(JUDGMENTS)
    counts = Counter(r["earliest_failure"] for r in rows if not r["correct"])
    r_counts = Counter(r["earliest_failure"] for r in rows if r["arm"] == "R" and not r["correct"])
    return {
        "purpose": "opened AFDB diagnosis only; no evaluated model or A0 receives references",
        "scope": {
            "rows": len(rows),
            "wrong_rows": sum(not r["correct"] for r in rows),
            "incorrect_R_rows": sum(r["scope"]["incorrect_R"] for r in rows),
            "correct_flip_rows": sum(r["correct"] for r in rows),
            "correctness_flip_pairs": sum(r["scope"]["correctness_flip"] for r in rows) // 2,
        },
        "category_counts_wrong_rows": {c: counts[c] for c in CATEGORIES},
        "category_counts_incorrect_R": {c: r_counts[c] for c in CATEGORIES},
        "classification_convention": "Earliest causally relevant failure in reference -> detector -> retrieval -> "
        "decision -> reporting order. A fragmented/false-positive detector can remain the earliest category even "
        "when later verification supplies enough evidence for recovery (D-R-afdb023); later decision errors are "
        "recorded separately. Grade-correct flip rows are no_failure, not additional errors.",
        "recoverability_convention": "Yes requires a concrete existing-evidence correction or stated reference "
        "convention plus already inspected gaps. No means not demonstrated by this audit, not impossible. "
        "An onset-close fragment alone does not establish longest-event identity or continuity. No new model "
        "was tested, and image interpretations are attributed to the agent's public account, not independently validated.",
        "projection_convention": "All calls and arguments retained. All returned candidates and scalar fields "
        "retained; large numeric arrays projected to length/endpoints, interval lists to count/endpoints. "
        "Exact originals are in hashed source transcripts at stated result lines; image bytes excluded.",
        "curated_input_manifest_sha256": AUDIT_MANIFEST_SHA256,
        "source_sha256": inputs,
        "rows": rows,
    }


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def compact_evidence(row):
    """One quoted output anchor per table row; full arguments appear in its trace."""
    parts = []
    for call in row["decisive_evidence"]:
        for output in call.get("returned_evidence", []):
            for measurement in output.get("measurements", []):
                if measurement.get("name") == "ecg_af_burden":
                    quoted = {k: measurement[k] for k in ("pct_of_window", "af_s", "window_s") if k in measurement}
                elif measurement.get("name") == "ecg_af_episodes":
                    first = measurement.get("episodes", [])[:1]
                    quoted = {k: first[0][k] for k in ("t", "span_s") if k in first[0]} if first else {}
                elif measurement.get("name") == "ecg_rr_stats":
                    quoted = {k: measurement[k] for k in ("cv", "pnn50") if k in measurement}
                else:
                    continue
                parts.append(
                    f"L{call['call_line']}→{call.get('result_line')} {measurement['name']}: "
                    f"`{json.dumps(quoted, separators=(',', ':'))}`"
                )
            if "events" in output and not parts:
                first = sorted(output["events"], key=lambda e: -e.get("dur", 0))[:1]
                quoted = {k: first[0][k] for k in ("t", "dur") if k in first[0]} if first else {}
                parts.append(
                    f"L{call['call_line']}→{call.get('result_line')} events: "
                    f"`{json.dumps(quoted, separators=(',', ':'))}`"
                )
        if parts:
            break
    return "; ".join(parts[:2])


def decision_excerpt(row):
    anchors = {
        "C-R-afdb000": "But the ECG trace",
        "C-M-afdb025": "**About 9,262",
        "C-R-afdb020": "That's why I round up",
        "D-R-afdb023": "Using the tool's explicit episode segmentation",
        "D-M-afdb025": "~8886 s onward",
    }
    text = row["agent_decision_and_final_answer"]
    anchor = anchors.get(row["key"])
    start = text.find(anchor) if anchor else 0
    assert start >= 0, row["key"]
    excerpt = text[start:].split("\n\n", 1)[0]
    return excerpt[:450] + ("…" if len(excerpt) > 450 else "")


def render_report(audit):
    lines = [
        "# Case-level p5afdb failure audit",
        "",
        "Diagnostic only, on opened AFDB. C = Claude Opus 5.5 medium; D = DeepSeek flash high. "
        "Arm is the frozen detector tree, not the model's harness arm (which is C throughout).",
        "",
        audit["classification_convention"],
        "",
        audit["recoverability_convention"],
        "",
        f"Scope: **{audit['scope']['rows']} rows, {audit['scope']['wrong_rows']} wrong, "
        f"{audit['scope']['correct_flip_rows']} correct flip counterparts**. "
        "All 10 wrong R answers and both rows of all 10 correctness flips are included once.",
        "",
        "| Earliest failure | All scoped wrong rows | Wrong R only |",
        "| --- | ---: | ---: |",
    ]
    for category in CATEGORIES:
        lines.append(
            f"| {category} | {audit['category_counts_wrong_rows'][category]} | "
            f"{audit['category_counts_incorrect_R'][category]} |"
        )
    lines += [
        "",
        "Counts are reviewer judgments with explicit causal ordering, not independent error probabilities. "
        "Zero primary agent/tool categories does not imply no agent mistakes: detector-first attribution "
        "coexists with the explicit downstream decision failure in D-R-afdb023 and overmerging elsewhere.",
        "",
        "| Case | Model/arm | Type | Correct | Error | Earliest failure | Final episode geometry / burden axis | Supporting tool evidence | Recoverable / how |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in audit["rows"]:
        geometry = row["longest_geometry"]
        subtype = geometry["subtype"] if geometry else row["burden_axes"]["answer_construction"]
        lines.append(
            f"| {row['case_id']} ({row['record']}) | {row['model_short']}/{row['arm']} | "
            f"{row['qtype']} | {'yes' if row['correct'] else 'no'} | "
            f"{row['error_size']:.3f} {row['error_unit']} | {row['earliest_failure']} | {cell(subtype)} | "
            f"{cell(compact_evidence(row))} | "
            f"{'yes' if row['recoverable_with_existing_tools'] else 'no demonstrated recovery'}: "
            f"{cell(row['recovery_how'].split('. ', 1)[0])} |"
        )
    lines += [
        "",
        "The geometry uses the final *claimed interval*, not only the returned onset. Positive overlap "
        "means the selected span intersects the annotated longest; it does not prove correct identity or duration. "
        "C-R-afdb008 and C-M-afdb019 overmerge across several true episodes, so their huge early onset errors "
        "are not mislabeled as disjoint episode choices. The unchanged grader tests only onset ±60 s or burden ±5 pp.",
        "",
        "The only directly demonstrated task-definition mismatches are C-R-afdb000 (true non-AF gap 21.400 s) "
        "and C-M-afdb025 (24.304 s). The visible question omits the reference's <30 s convention. "
        "These are still wrong under unchanged grading, but calling every true regular break a termination "
        "is not necessarily a reasoning failure under the question's ordinary meaning of continuous.",
        "",
        "## Actionable implications",
        "",
        "- Interface: state the annotation-independent episode convention and distinguish supported AF time, "
        "grouped span, readability and clinically inferred continuity. Expose all competing candidates without "
        "raw support-array repetition. No audited miss is proven to originate solely from hidden/truncated evidence.",
        "- Deterministic checks: preserve pct_of_window and full-record denominator; keep uncertainty separate "
        "from a point estimate; validate sum/union of added intervals and that cited verification windows lie "
        "inside the candidate. This directly prevents C-R-afdb020's 26.4 → 27.0 reporting miss.",
        "- Autonomous verification: compare multiple candidates, check *all material* disputed gaps, and "
        "refine onsets only after establishing continuity. D-R-afdb023 already received a supported ~10258 s "
        "alternative but chose the detector fragment; D-R-afdb011 and D-M-afdb023 show successful verification paths.",
        "- Perception/classification: regular AFL versus sinus/ectopy, false-positive irregularity (08219), "
        "and missing/fragmented AF (04936, 05121, 08215) remain substantive evidence gaps. Existing ECG views "
        "may help; this audit does not prove a new perception model is required or would solve them. "
        "No onset-close candidate or successful sibling run licenses automatic correction.",
        "",
        "## Case traces",
        "",
        "Exact calls, arguments, line numbers, projected returned fields, reference intervals, every detector "
        "candidate and artifact SHA256s are in [failures.json](failures.json). Tool excerpts below preserve "
        "received values; image morphology statements are the agent's reported interpretations.",
        "",
    ]
    for row in audit["rows"]:
        lines += [
            f"### {row['key']} — {row['record']} / {row['qtype']}",
            "",
            f"**{'Correct' if row['correct'] else 'Wrong'}**, error {row['error_size']:.3f} "
            f"{row['error_unit']}; earliest: **{row['earliest_failure']}**. {row['reason']}",
            "",
        ]
        ref, det = row["reference_trace"], row["detector_trace"]
        if row["qtype"] == "longest_af":
            lines.append(
                f"Reference longest {ref['longest_annotated']}; native detector longest "
                f"{det['longest_detected']}; {det['fragmentation_of_reference_longest']} "
                f"overlapping fragments. Final claimed span {row['agent_selected_span_s']}; "
                f"overlap {row['longest_geometry']['overlap_s']:.3f} s, "
                f"IoU {row['longest_geometry']['iou']:.4f}."
            )
        else:
            lines.append(
                f"Reference AF time {ref['true_af_s']:.3f} / {ref['window_s']:.3f} s "
                f"= {row['reference']['percent']:.6f}%; detector public payload "
                f"`{json.dumps(det['public_burden'], separators=(',', ':'))}`. "
                f"Axes: {cell(json.dumps(row['burden_axes']))}"
            )
        lines.append("")
        for call in row["decisive_evidence"]:
            args = {k: v for k, v in call["arguments"].items() if k != "id"}
            summary = []
            for output in call.get("returned_evidence", []):
                for measurement in output.get("measurements", []):
                    if measurement.get("name") == "ecg_af_episodes":
                        episodes = measurement.get("episodes", [])
                        summary.append(
                            {
                                "name": "ecg_af_episodes",
                                "value": measurement.get("value"),
                                "merge_gap_s": measurement.get("merge_gap_s"),
                                "top_3": [
                                    {k: e[k] for k in ("start", "t", "dur", "span_s", "af_s") if k in e}
                                    for e in episodes[:3]
                                ],
                            }
                        )
                    else:
                        summary.append({k: v for k, v in measurement.items() if k not in {"ref", "method", "ch"}})
                if "events" in output:
                    events = output["events"]
                    summary.append(
                        {
                            "events_received": len(events),
                            "total": output.get("total"),
                            "largest_3_received": sorted(events, key=lambda e: -e.get("dur", 0))[:3],
                        }
                    )
                if "text" in output:
                    summary.append({"error" if call.get("is_error") else "text": output["text"]})
                if call["tool"] == "ts_zoom" and not call.get("is_error"):
                    summary.append({k: output[k] for k in ("window", "view", "series") if k in output})
            lines.append(
                f"- Transcript L{call['call_line']} → L{call.get('result_line')}: "
                f"`{call['tool']}({json.dumps(args, separators=(',', ':'))})`; "
                f"returned `{json.dumps(summary, separators=(',', ':'))}`"
                + (" plus image." if call.get("image_received") else ".")
            )
        lines += [
            "",
            f"Agent public decision excerpt: “{cell(decision_excerpt(row))}” "
            f"Final parsed answer: `{json.dumps(row['pred'])}`. "
            "Complete public answer is retained in failures.json.",
            "",
            f"Recovery: **{'yes' if row['recoverable_with_existing_tools'] else 'not demonstrated'}**. "
            f"{row['recovery_how']}",
            "",
        ]
        if row["downstream_issue"]:
            lines += [f"Additional issue/qualification: {row['downstream_issue']}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/home/yanghaoxiang24/signal_agent"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/audit-a0"))
    args = parser.parse_args()
    audit = build_audit(args.input_root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "failures.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.out / "failure-report.md").write_text(render_report(audit))
    print(json.dumps({"scope": audit["scope"], "counts": audit["category_counts_wrong_rows"]}, indent=2))


if __name__ == "__main__":
    main()
