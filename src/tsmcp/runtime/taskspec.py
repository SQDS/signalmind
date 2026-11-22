"""TaskSpec: what is requested, from the request, public method definitions and observable metadata only.

Nothing in this module may read reference answers, grading tolerances, benchmark item ids or category labels. A
TaskSpec constrains the *evidence a conclusion needs*, not the sequence of tool calls that produces it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tsmcp.runtime.timebase import SyncStatus, pair_key

# ---- public measurement definitions --------------------------------------------------------------------------------
# definition_id -> how the quantity is defined. These are the method conventions a claim must name; the pressure plugin
# stamps the same ids on its evidence. `kind` decides the timebase rule; `required_roles` are channel roles the
# measurement genuinely depends on (a per-beat ABP mean depends on the pressure channel only).
MEASUREMENT_DEFINITIONS: dict[str, dict] = {
    "abp.beats.v1": {
        "quantity": "beat_table",
        "unit_family": "count",
        "kind": "measurement",
        "required_roles": ("pressure",),
        "computation": "beat_estimation",
        "min_accepted_beats": 1,
        "description": "Per-beat ABP table: foot, systolic peak, next foot, SBP, DBP, PP, accepted flag and reason.",
    },
    "abp.sbp.beat_mean.v1": {
        "quantity": "sbp_mean",
        "unit_family": "pressure",
        "kind": "measurement",
        "required_roles": ("pressure",),
        "computation": "beat_estimation",
        "aggregation": "beat_weighted",
        "min_accepted_beats": 5,
        "description": "Mean over accepted complete beats of each beat's systolic maximum (beat-weighted).",
    },
    "abp.dbp.beat_mean.v1": {
        "quantity": "dbp_mean",
        "unit_family": "pressure",
        "kind": "measurement",
        "required_roles": ("pressure",),
        "computation": "beat_estimation",
        "aggregation": "beat_weighted",
        "min_accepted_beats": 5,
        "description": "Mean over accepted complete beats of the diastolic minimum at each foot (beat-weighted).",
    },
    "abp.pp.beat_mean.v1": {
        "quantity": "pp_mean",
        "unit_family": "pressure",
        "kind": "measurement",
        "required_roles": ("pressure",),
        "computation": "beat_estimation",
        "aggregation": "beat_weighted",
        "min_accepted_beats": 5,
        "description": "Mean over accepted complete beats of SBP minus DBP (beat-weighted).",
    },
    "abp.sbp.time_weighted_mean.v1": {
        "quantity": "sbp_mean",
        "unit_family": "pressure",
        "kind": "measurement",
        "required_roles": ("pressure",),
        "computation": "beat_estimation",
        "aggregation": "time_weighted",
        "min_accepted_beats": 5,
        "description": "Per-beat SBP weighted by beat duration (derived from a beat table via the aggregate op).",
    },
    "abp.sbp.beat_mean_change.v1": {
        "quantity": "sbp_change",
        "unit_family": "pressure",
        "kind": "measurement",
        "required_roles": ("pressure",),
        "computation": "beat_estimation",
        "min_accepted_beats": None,
        "description": "Later-interval beat-mean SBP minus baseline-interval beat-mean SBP (derived: subtract of two "
        "abp.sbp.beat_mean.v1 records).",
    },
    "delay.r_to_foot.observed.v1": {
        "quantity": "r_to_foot_delay",
        "unit_family": "time",
        "kind": "observed_delay",
        "required_roles": ("ecg", "pressure"),
        "computation": "cross_channel_beat_alignment",
        "min_accepted_beats": 5,
        "description": "Observed sample-index delay from ECG R-peak to the paired ABP foot; not a physiological claim.",
    },
    "pat.r_to_foot.v1": {
        "quantity": "pulse_arrival_time",
        "unit_family": "time",
        "kind": "physiological_timing",
        "required_roles": ("ecg", "pressure"),
        "computation": "cross_channel_beat_alignment",
        "min_accepted_beats": 5,
        "description": "Pulse arrival time: physiological reading of the R-to-foot delay; needs declared synchrony.",
    },
    # generic single-channel measures already in the toolkit (evidence from ts_measure carries these ids)
    "generic.mean.v1": {"quantity": "mean", "unit_family": None, "kind": "measurement", "required_roles": ()},
    "generic.rms.v1": {"quantity": "rms", "unit_family": None, "kind": "measurement", "required_roles": ()},
    "generic.amplitude_pp.v1": {
        "quantity": "amplitude_pp",
        "unit_family": None,
        "kind": "measurement",
        "required_roles": (),
        "description": "Robust whole-window peak-to-peak (p99.5 - p0.5). Not a per-beat quantity.",
    },
    "generic.period.v1": {"quantity": "period", "unit_family": "time", "kind": "measurement", "required_roles": ()},
    "rate.from_period.v1": {
        "quantity": "pulse_rate_bpm",
        "unit_family": "rate",
        "kind": "measurement",
        "required_roles": ("pulse",),
        "computation": "period_estimation",
        "derive_with": "rate_from_period",
        "derived_from": "generic.period.v1",
        "description": "60 / autocorrelation period of the pulse channel over the window (derived from "
        "generic.period.v1); a rhythm-level rate, not a beat count.",
    },
    "generic.dominant_freq.v1": {
        "quantity": "dominant_freq",
        "unit_family": "frequency",
        "kind": "measurement",
        "required_roles": (),
    },
    **{
        f"tfa.{q}.{b}.v1": {
            "quantity": f"tfa_{q}_{b}",
            "unit_family": {"gain": "velocity/pressure", "phase": "angle", "coherence": "dimensionless"}[q],
            "kind": "measurement",
            "required_roles": ("pressure", "flow"),
            "computation": "cross_spectral_estimation",
            "description": f"Transfer function pressure->flow, {q}, {b.upper()} band mean (1-s means, Welch 100 s).",
        }
        for q in ("gain", "phase", "coherence")
        for b in ("vlf", "lf", "hf")
    },
    "pulse.beats.v1": {"quantity": "pulse_beats", "unit_family": "count", "kind": "measurement", "required_roles": ()},
    "pulse.rate.beats.v1": {
        "quantity": "pulse_rate",
        "unit_family": "rate",
        "kind": "measurement",
        "required_roles": (),
        "computation": "beat_estimation",
        "min_accepted_beats": 5,
    },
    "pulse.quality.v1": {
        "quantity": "pulse_quality",
        "unit_family": "dimensionless",
        "kind": "measurement",
        "required_roles": (),
        "computation": "signal_quality",
    },
    "pulse.amplitude_change.v1": {
        "quantity": "amplitude_change_onset",
        "unit_family": "time",
        "kind": "measurement",
        "required_roles": (),
        "computation": "pressure_pulsatility_change",
    },
    "pulse.level_change.v1": {
        "quantity": "level_change_onset",
        "unit_family": "time",
        "kind": "measurement",
        "required_roles": (),
        "computation": "sustained_change_vs_transient",
    },
    "delay.r_to_foot.change.v1": {
        "quantity": "r_to_foot_delay_change",
        "unit_family": "time",
        "kind": "observed_delay",
        "required_roles": ("ecg",),
        "computation": "cross_channel_beat_alignment",
    },
    "beats.missing_in_pulse.v1": {
        "quantity": "beats_missing_in_pulse",
        "unit_family": "count",
        "kind": "measurement",
        "required_roles": ("ecg",),
        "computation": "cross_channel_missing_beats",
    },
    "beats.missing_in_ecg.v1": {
        "quantity": "beats_missing_in_ecg",
        "unit_family": "count",
        "kind": "measurement",
        "required_roles": ("ecg",),
        "computation": "cross_channel_missing_beats",
    },
    "polarity.change.v1": {
        "quantity": "polarity_change_onset",
        "unit_family": "time",
        "kind": "measurement",
        "required_roles": (),
        "computation": "cross_channel_polarity_check",
    },
    "view.v1": {"quantity": "view", "unit_family": None, "kind": "observation", "required_roles": ()},
    "events.v1": {"quantity": "events", "unit_family": "count", "kind": "measurement", "required_roles": ()},
}

# Reusable dependency rules by quantity *kind*, not by case: what else a claim of that kind needs.
DEPENDENCY_RULES: dict[str, dict] = {
    "measurement": {"source_kinds": ("computed", "derived")},
    "observed_delay": {"source_kinds": ("computed", "derived"), "sync": SyncStatus.INDEX_ALIGNED_DECLARED.value},
    "physiological_timing": {"source_kinds": ("computed", "derived"), "sync": SyncStatus.DECLARED_SYNCHRONOUS.value},
    "observation": {"source_kinds": ("computed", "derived", "visual_observation")},
}


@dataclass(frozen=True)
class QuantitySpec:
    name: str  # the user's name for the quantity, e.g. "mean_sbp"
    definition_id: str | None  # a registered definition; None leaves method semantics unbound
    unit: str  # target unit for the answer
    interval: tuple[float, float] | None = None  # overrides the task interval
    precision_decimals: int | None = None  # from the request or the method resolution; None = method decides
    required: bool = True

    @property
    def definition(self) -> dict | None:
        if self.definition_id is None:
            return None
        if self.definition_id not in MEASUREMENT_DEFINITIONS:
            raise ValueError(f"unknown measurement definition {self.definition_id!r}")
        return MEASUREMENT_DEFINITIONS[self.definition_id]


@dataclass(frozen=True)
class Budgets:
    """Engineering safeguards, not physiological thresholds."""

    max_steps: int = 30
    max_tool_calls: int = 25
    max_submissions: int = 4
    max_repair_cycles_per_issue: int = 2
    max_seconds: float = 900.0
    max_cost_usd: float | None = None


@dataclass
class TaskSpec:
    task_id: str
    request_text: str
    quantities: list[QuantitySpec]
    channel_roles: dict[str, str]  # role -> channel name, from the request / observable metadata
    interval: tuple[float, float] | None = None  # seconds from record start, half-open
    coordinate_system: str = "seconds_from_record_start"
    declared_sync: dict[str, str] = field(default_factory=dict)  # pair_key(a, b) -> SyncStatus value
    capabilities: tuple[str, ...] = ()  # measure / tool names available to the agent
    budgets: Budgets = field(default_factory=Budgets)
    conventions: dict[str, str] = field(default_factory=dict)  # explicitly selected conventions, documented
    optional_roles: tuple[str, ...] = ()

    def quantity(self, name: str) -> QuantitySpec | None:
        return next((q for q in self.quantities if q.name == name), None)

    def sync_between(self, a: str, b: str) -> str:
        return self.declared_sync.get(pair_key(a, b), SyncStatus.UNKNOWN.value)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["budgets"] = asdict(self.budgets)
        return d

    @classmethod
    def minimal(cls, task_id: str = "adhoc", request_text: str = "") -> TaskSpec:
        """A spec that constrains nothing beyond the definitions' own rules (used when no spec file is given)."""
        return cls(task_id=task_id, request_text=request_text, quantities=[], channel_roles={})

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        qs = [
            QuantitySpec(**{**q, "interval": tuple(q["interval"]) if q.get("interval") else None})
            for q in d.get("quantities", [])
        ]
        b = d.get("budgets") or {}
        return cls(
            task_id=d["task_id"],
            request_text=d.get("request_text", ""),
            quantities=qs,
            channel_roles=dict(d.get("channel_roles", {})),
            interval=tuple(d["interval"]) if d.get("interval") else None,
            coordinate_system=d.get("coordinate_system", "seconds_from_record_start"),
            declared_sync=dict(d.get("declared_sync", {})),
            capabilities=tuple(d.get("capabilities", ())),
            budgets=Budgets(**b),
            conventions=dict(d.get("conventions", {})),
            optional_roles=tuple(d.get("optional_roles", ())),
        )


def abp_interval_spec(
    task_id: str,
    request_text: str,
    pressure_channel: str,
    interval: tuple[float, float],
    quantities: tuple[str, ...] = ("sbp_mean", "dbp_mean"),
    unit: str = "mmHg",
    ecg_channel: str | None = None,
    same_array: bool = True,
    declared_synchronous: bool = False,
    capabilities: tuple[str, ...] = (),
    budgets: Budgets | None = None,
) -> TaskSpec:
    """The v0.1 vertical slice: per-beat pressure summary over a stated interval.

    Only request-level facts enter: which channel is pressure, the interval, target unit, whether an ECG channel is
    present in the same array (index-aligned, declared by the file) and whether the user stated synchrony.
    """
    defs = {"sbp_mean": "abp.sbp.beat_mean.v1", "dbp_mean": "abp.dbp.beat_mean.v1", "pp_mean": "abp.pp.beat_mean.v1"}
    unknown = [q for q in quantities if q not in defs]
    if unknown:
        raise ValueError(f"unsupported quantities {unknown}; v0.1 supports {sorted(defs)}")
    roles = {"pressure": pressure_channel}
    sync = {}
    if ecg_channel:
        roles["ecg"] = ecg_channel
        status = (
            SyncStatus.DECLARED_SYNCHRONOUS
            if declared_synchronous
            else SyncStatus.INDEX_ALIGNED_DECLARED
            if same_array
            else SyncStatus.UNKNOWN
        )
        sync[pair_key(ecg_channel, pressure_channel)] = status.value
    return TaskSpec(
        task_id=task_id,
        request_text=request_text,
        quantities=[QuantitySpec(q, defs[q], unit) for q in quantities],
        channel_roles=roles,
        interval=(float(interval[0]), float(interval[1])),
        declared_sync=sync,
        capabilities=capabilities,
        budgets=budgets or Budgets(),
        optional_roles=("ecg",) if ecg_channel else (),
        conventions={"aggregation": "beat_weighted", "boundary_beats": "excluded_if_incomplete"},
    )
