"""Explicit v1.1 development domain. Existing v1.0 runs remain available for version comparisons."""

from pathlib import Path

import numpy as np

from tsmcp.processing import PCHIP_ANCHORS, PCHIP_TOOL

from .common import digest, read
from .contracts import Repairability
from .quality_gate import SPEC_ID, spike_check

PROFILES = Path(__file__).with_name("profiles")
LEGACY_PROFILE = read(PROFILES / "human-vital-short-core.v1.json")
PROFILE = read(PROFILES / "human-vital-short-core.v2.json")


class RepairDomain:
    def __init__(self, config):
        self.declaration = config.get("repair_domain", {})
        self.profile = (
            LEGACY_PROFILE if self.declaration.get("profile_version") == LEGACY_PROFILE["version"] else PROFILE
        )
        self.spec_id = digest([SPEC_ID, self.profile, self.declaration])

    def metadata(self, c):
        decl = self.declaration
        goal = decl.get("goals", {}).get(c["task"])
        model = self.profile["modalities"].get(c["task"], {})
        reasons = []
        checks = (
            (decl.get("use") == "development", "development_profile_only"),
            (all(k not in c or c[k] == decl.get(k) for k in ("species", "device")), "source_declaration_conflict"),
            (
                (decl.get("profile_id"), decl.get("profile_version")) == (self.profile["id"], self.profile["version"]),
                "unregistered_profile",
            ),
            (decl.get("species") == self.profile["species"], "species_outside_profile"),
            (decl.get("device") == self.profile["device"], "device_outside_profile"),
            (c["fs"] == self.profile["fs_hz"], "fs_outside_profile"),
            (goal in model.get("goals", []), "goal_uncalibrated"),
            (
                c.get("unit") == model.get("unit") and c.get("field") == model.get("field"),
                "native_unit_or_field_unverified",
            ),
            (bool(c.get("has_source_time")) and not c.get("normalized_windows"), "native_time_or_scale_unknown"),
        )
        reasons.extend(reason for ok, reason in checks if not ok)
        return dict(
            profile_id=decl.get("profile_id"),
            profile_version=decl.get("profile_version"),
            calibration=self.profile["calibration"]
            if (decl.get("profile_id"), decl.get("profile_version")) == (self.profile["id"], self.profile["version"])
            else "unregistered",
            species=decl.get("species"),
            device=decl.get("device"),
            modality=c["task"],
            fs_hz=c["fs"],
            unit=c.get("unit"),
            goal=goal,
            acceptance_spec_id=self.spec_id,
            reason_codes=reasons,
        )

    def assess(self, c, a, b, before, operation, precheck=None):
        item = self.metadata(c)
        t, x, v, e, pr = before
        good = v & ~e & np.isfinite(x)
        reasons = item["reason_codes"]
        duration = (b - a) / c["fs"]
        period = None
        points = [z for z in pr["events"] if abs(z["sample"] - (a + b) / 2) <= 8 * c["fs"]]
        periods = [
            (q["sample"] - p["sample"]) / c["fs"]
            for p, q in zip(points[:-1], points[1:], strict=True)
            if p["support_id"] == q["support_id"]
            and good[p["sample"] : q["sample"] + 1].all()
            and good[min(p["sample"], a) : max(q["sample"] + 1, b)].all()
        ]
        if periods:
            period = float(np.median(periods))
        n_anchor = PCHIP_ANCHORS if operation == PCHIP_TOOL else 1
        filter_policy = self.profile.get("filter_policy", {})
        contextual_filter = operation in filter_policy.get("tools", [])
        if contextual_filter:
            n_anchor = round(c["fs"] * filter_policy["context_s"])
        lo, hi = a - n_anchor, b + n_anchor
        anchors = bool(lo >= 0 and hi <= len(x) and good[lo:a].all() and good[b:hi].all())
        grid = t[max(0, lo) : min(len(t), hi)]
        time_supported = bool(
            len(grid) >= 2
            and np.isfinite(grid).all()
            and np.all(np.abs(np.diff(grid) - 1 / c["fs"]) <= max(1e-6, 32 * np.spacing(abs(grid).max())))
            and not any(left < hi and lo < right for left, right in c.get("source_time_conflicts", []))
        )
        item.update(
            duration_s=duration,
            local_period_s=period,
            period_fraction=duration / period if period else None,
            observed_anchors=anchors,
            time_supported=time_supported,
            operation=operation,
            status="unsupported",
        )
        if reasons:
            return Repairability.model_validate(item).model_dump()
        if not time_supported:
            reasons.append("time_discontinuity")
            item["status"] = "need_evidence"
            return Repairability.model_validate(item).model_dump()
        if contextual_filter:
            if operation == "local_detrend" and c["task"] != "ECG":
                reasons.append("method_not_applicable")
            if not anchors or not good[a:b].all() or b - a < 10:
                reasons.append("filter_read_context_unsupported")
            if period is None:
                reasons.append("local_period_unknown")
            item["status"] = "need_evidence" if reasons else "eligible"
            return Repairability.model_validate(item).model_dump()
        if operation in self.profile["mask_tools"]:
            item["status"] = "eligible"
            return Repairability.model_validate(item).model_dump()
        if (
            operation not in self.profile["spike_tools"]
            or b - a > self.profile["max_samples"]
            or duration > self.profile["max_seconds"]
        ):
            reasons.append("method_not_applicable")
            return Repairability.model_validate(item).model_dump()
        if not good[a:b].any():
            item["status"] = "unrecoverable_for_goal"
            reasons.append("no_native_core_support")
            return Repairability.model_validate(item).model_dump()
        check = precheck if precheck is not None else spike_check(c, x, v, e, pr, a, b)
        item["event_evidence"] = check
        if not anchors:
            reasons.append("anchor_unsupported")
        if not period:
            reasons.append("local_period_unknown")
        elif duration >= period:
            reasons.append("core_spans_local_cycle")
        if check["status"] != "evaluated_no_alert":
            reasons.append(check.get("reason", check["status"]))
        item["status"] = (
            "preserve_physiology"
            if check.get("matching_shapes", 0) >= 3
            and check["status"] == "possible_physiology"
            and check.get("reason") != "near_native_qrs"
            else "need_evidence"
            if reasons
            else "eligible"
        )
        return Repairability.model_validate(item).model_dump()


def domain_for(config):
    mode = config.get("repair_protocol", "v1.1" if "repair_domain" in config else "v1.0")
    if mode not in ("v1.0", "v1.1") or ("repair_domain" in config and mode != "v1.1"):
        raise ValueError("Choose v1.1 to use an explicit repair domain")
    if mode == "v1.1":
        domain = config.get("repair_domain", {})
        if not isinstance(domain, dict) or not isinstance(domain.get("goals", {}), dict):
            raise ValueError("repair_domain and its per-channel goals must be objects")
        return RepairDomain(config)
    return None
