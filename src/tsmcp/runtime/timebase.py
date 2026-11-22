"""Time semantics across channels. An unknown offset is not zero; equal sample indices are not proof of simultaneity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class SyncStatus(StrEnum):
    UNKNOWN = "unknown"
    # Channels share one sample index (same array / file); the acquisition system's alignment is declared, not verified.
    INDEX_ALIGNED_DECLARED = "index_aligned_declared"
    # The user or recorded metadata states the channels are synchronous (known offset, possibly non-zero).
    DECLARED_SYNCHRONOUS = "declared_synchronous"
    # Alignment measured by a method that records its own evidence (deferred: no such method exists in v0.1).
    MEASURED = "measured"


_RANK = {
    SyncStatus.UNKNOWN: 0,
    SyncStatus.INDEX_ALIGNED_DECLARED: 1,
    SyncStatus.DECLARED_SYNCHRONOUS: 2,
    SyncStatus.MEASURED: 3,
}

# What a claim kind needs. An *observed* inter-channel delay only needs a common index; a *physiological*
# interpretation (pulse arrival time) needs a declared or measured relationship.
REQUIRED_SYNC = {
    "observed_delay": SyncStatus.INDEX_ALIGNED_DECLARED,
    "physiological_timing": SyncStatus.DECLARED_SYNCHRONOUS,
}


def supports(status: SyncStatus | str | None, needed: SyncStatus | str) -> bool:
    try:
        s = SyncStatus(status) if status is not None else SyncStatus.UNKNOWN
        n = SyncStatus(needed)
    except ValueError:
        return False
    return _RANK[s] >= _RANK[n]


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


@dataclass(frozen=True)
class TimebaseInfo:
    """Attached to evidence that involves more than one channel, or whose time axis matters."""

    identity: str  # the clock the record's times refer to (signal id in v0.1)
    sync_status: str = SyncStatus.UNKNOWN.value
    basis: str = ""  # why this status is claimed, e.g. "columns of one .npy array"
    offset_s: float | None = None  # known offset between the channels, if declared
    channels: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)
