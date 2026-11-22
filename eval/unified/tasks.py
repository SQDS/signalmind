"""Task constraints from the public Level 2 question and visible signal metadata.

No files or grading data are read here. A missing definition means that the requested field has no matching
registered method contract yet; its presence, unit and stated interval can still be checked. It does not advertise
a new computation. Event-search questions leave the evidence window to the agent instead of requiring a whole-record
measurement. The full public search interval is recorded separately in conventions.
"""

from __future__ import annotations

import json
import re
from itertools import combinations

from tsmcp.runtime.taskspec import QuantitySpec, TaskSpec
from tsmcp.runtime.timebase import SyncStatus, pair_key

FIELD_UNITS = {
    "hr_change_bpm": "bpm",
    "sbp_change_mmhg": "mmHg",
    "response_max_time_s": "s",
    "response_min_time_s": "s",
    "onset_s": "s",
    "duration_s": "s",
    "channel": "",
    "count": "beats",
    "shift_ms": "ms",
    "brs_ms_per_mmhg": "ms/mmHg",
    "valid_sequences": "count",
    "gain": "(cm/s)/mmHg",
    "phase": "rad",
    "coherence": "",
    "heart_rate_bpm": "bpm",
    "sbp_mmhg": "mmHg",
    "bpm": "bpm",
}


def public_answer_fields(case: dict) -> list[str]:
    """Use supplied public field names, or the question's explicit Return/Report clause."""
    if case.get("answer_fields"):
        return list(case["answer_fields"])
    question = case["question"]
    if "pulse rate represented by PPG" in question:
        return ["bpm"]
    clause = re.search(r"\b(?:Return|Report)\s+([^.]+)", question)
    if clause is None:
        raise ValueError("question does not state its answer fields")
    return [name for name in re.findall(r"\b[a-z][a-z_]*\b", clause[1]) if name in FIELD_UNITS]


def _definition(field: str, question: str) -> str | None:
    if field in ("gain", "phase", "coherence"):
        band = re.search(r"Across (VLF|LF|HF)\b", question)[1].lower()
        return f"tfa.{field}.{band}.v1"
    if field == "sbp_change_mmhg":
        return "abp.sbp.beat_mean_change.v1"
    if field == "sbp_mmhg":
        return "abp.sbp.beat_mean.v1"
    if field == "onset_s":
        if "sustained attenuation" in question:
            return "pulse.amplitude_change.v1"
        if "sustained displacement" in question:
            return "pulse.level_change.v1"
        if "changes polarity" in question:
            return "polarity.change.v1"
    return None


def task_spec_for_case(case: dict) -> TaskSpec:
    """Build a task spec using only case_id, question, visible and optional answer_fields.

    Unbound fields (including BRS, channel identification and extrema) remain required. Their method semantics need
    independent grading; they are not silently mapped to an unrelated existing measurement definition.
    """
    question, visible = case["question"], case["visible"]
    fields = public_answer_fields(case)
    names = [ch["name"] for ch in visible["channels"]]
    by_name = {name.upper(): name for name in names}
    roles = {
        role: by_name[name] for role, name in (("ecg", "ECG"), ("pressure", "ABP"), ("flow", "CBFV")) if name in by_name
    }
    if "PPG" in by_name or "ABP" in by_name:
        roles["pulse"] = by_name.get("PPG", by_name.get("ABP"))
    duration = float(visible["dur_s"])
    conventions = {"spec_source": "public_question_and_visible_metadata"}
    # Only intervals stated in the request are bound. No hidden event bounds enter the task spec.
    number = r"(\d+(?:\.\d+)?)"
    change = re.search(rf"from {number}[–-]{number} s to {number}[–-]{number} s", question)
    interval = None
    response = None
    if change:
        a, b, c, d = map(float, change.groups())
        interval, response = (a, d), (c, d)
        conventions.update(
            baseline_window_s=json.dumps([a, b]),
            response_window_s=json.dumps([c, d]),
            change_direction="later_minus_baseline",
            extrema_ties="earliest_time",
        )
        conventions["extrema_times"] = "RR_midpoints" if "RR midpoint" in question else "systolic_peaks"
    elif "sequence-method BRS" in question or "over this record" in question or "complete recording" in question:
        interval = (0.0, duration)
    else:
        conventions["search_interval_s"] = json.dumps([0.0, duration])
    if "sequence-method BRS" in question:
        conventions.update(
            sequence_selection="maximal",
            rr_pairing="contemporaneous",
            method_support="no_registered_sequence_BRS_definition",
        )
    if "equal-frequency-bin" in question:
        conventions.update(band_statistic="equal_frequency_bin_mean", input_series="nonoverlapping_1s_means")
    if "positive for CBFV leading ABP" in question:
        conventions["phase_sign"] = "positive_for_CBFV_leading_ABP"
    quantities = [
        QuantitySpec(
            name,
            _definition(name, question),
            FIELD_UNITS[name],
            interval=response if name.startswith("response_") else interval,
        )
        for name in fields
    ]
    unbound = [q.name for q in quantities if q.definition_id is None]
    if unbound:
        conventions["unbound_method_fields"] = ",".join(unbound)
    return TaskSpec(
        task_id=case["case_id"],
        request_text=question,
        quantities=quantities,
        channel_roles=roles,
        interval=interval,
        declared_sync={pair_key(a, b): SyncStatus.INDEX_ALIGNED_DECLARED.value for a, b in combinations(names, 2)},
        conventions=conventions,
    )
