"""Versioned AF presentation only; all detection and integration stay in the frozen plugin."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict

import numpy as np

from tsmcp.core.compact import dumps, tround
from tsmcp.core.errors import SignalError
from tsmcp.core.signal import time_decimals
from tsmcp.plugins.ecg import af

AF_CONVENTIONS = {
    "grouping": "merge AF support separated by gaps strictly < 30 s",
    "af_s": "supported AF seconds, excludes gaps",
    "span_s": "episode end minus start, includes gaps",
    "pct_of_window": "100*af_s/window_s; whole analysed window; use for whole-record burden",
    "pct_of_readable": "value = 100*af_s/readable_s",
    "unknown": "unknown time is not evidence of absence of AF",
}
AF_DESCRIPTION = (
    " AF conventions: merge support across gaps strictly <30 s; af_s excludes gaps; span_s includes gaps. "
    "pct_of_window=100*af_s/window_s uses the whole analysed window (whole-record burden); "
    "pct_of_readable=value=100*af_s/readable_s. Unknown time is not evidence of absence of AF."
)
RANKING = "span_s descending, start ascending (unrounded)"
_EPISODE_ID = re.compile(r"([0-9a-f]{8}):af2:([0-9a-f]+):([0-9a-f]+):([0-9a-f]+):([0-9a-f]{16})\Z")


def _digest(episode: dict) -> str:
    return hashlib.sha256(dumps(episode).encode()).hexdigest()[:16]


class EpisodePresenter:
    """Bounded evidence cache; IDs encode the recipe needed after eviction or restart.

    Cache entries contain episode and quality intervals, never signal samples or beat arrays.
    The channel index and sample bounds are hexadecimal integers, with no rounded times in IDs.
    """

    def __init__(self):
        self._windows: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.RLock()

    def _remember(self, key, raw, analysis):
        entry = {
            "episodes": raw["episodes"],
            "by_digest": {_digest(e): e for e in raw["episodes"]},
            "readable": analysis["readable"],
            "unknown": analysis["noisy"],
        }
        with self._lock:
            self._windows[key] = entry
            self._windows.move_to_end(key)
            while len(self._windows) > 8:
                self._windows.popitem(last=False)
        return entry

    def index(self, signal, channel, x, t0, raw):
        ci = signal.channels.index(channel)
        i0 = round(t0 * channel.fs)
        i1 = i0 + len(x)
        key = (signal.id, ci, i0, i1)
        entry = self._remember(key, raw, af.af_analysis(x, channel.fs, t0, ch=channel))
        tdec = time_decimals(channel.fs)
        prefix = f"{signal.id}:af2:{ci:x}:{i0:x}:{i1:x}:"
        episodes = []
        # Ranking and all interval arithmetic precede presentation rounding.
        for e in sorted(entry["episodes"], key=lambda e: (-e["span_s"], e["start"])):
            start, end = e["start"], e["intervals_s"][-1][1]
            unknown = sum(b - a for a, b in af.clip_intervals(entry["unknown"], start, end))
            episodes.append(
                {
                    "id": prefix + _digest(e),
                    "start": tround(start, tdec),
                    "end": tround(end, tdec),
                    "span_s": tround(e["span_s"], tdec),
                    "af_s": tround(e["af_s"], tdec),
                    "unknown_s": tround(unknown, tdec),
                    "merged_from": len(e["intervals_s"]),
                }
            )
        return {
            "name": "ecg_af_episodes",
            "channel": channel.name,
            "count": len(episodes),
            "units": "s",
            "window": [tround(i0 / channel.fs, tdec), tround(i1 / channel.fs, tdec)],
            "ranking": RANKING,
            "ref": raw["ref"],
            "episodes": episodes,
        }

    def detail(self, signal, episode):
        match = _EPISODE_ID.fullmatch(episode)
        if match is None or match[1] != signal.id:
            raise SignalError("episode must be an id from ecg_af_episodes for this signal")
        ci, i0, i1 = (int(match[i], 16) for i in (2, 3, 4))
        if ci >= len(signal.channels):
            raise SignalError("episode channel is not present; request ecg_af_episodes again")
        channel = signal.channels[ci]
        if not 0 <= i0 < i1 <= channel.n:
            raise SignalError("episode sample window is invalid; request ecg_af_episodes again")
        key = (signal.id, ci, i0, i1)
        with self._lock:
            entry = self._windows.get(key)
            if entry is None:
                x = np.asarray(channel.data[i0:i1], dtype=np.float64)
                t0 = i0 / channel.fs
                raw = af.ecg_af_episodes(x, channel.fs, channel.unit, t0, ch=channel)
                entry = self._remember(key, raw, af.af_analysis(x, channel.fs, t0, ch=channel))
            else:
                self._windows.move_to_end(key)
            raw = entry["by_digest"].get(match[5])
        if raw is None:
            raise SignalError("episode evidence no longer matches; request ecg_af_episodes again")
        start, end = raw["start"], raw["intervals_s"][-1][1]
        tdec = time_decimals(channel.fs)
        return {
            "name": "ecg_af_episode_detail",
            "channel": channel.name,
            "episode": episode,
            "window": [tround(i0 / channel.fs, tdec), tround(i1 / channel.fs, tdec)],
            "units": "s",
            "ref": episode,
            "legacy_episode": raw,
            "readable_intervals_s": af.clip_intervals(entry["readable"], start, end),
            "unknown_intervals_s": af.clip_intervals(entry["unknown"], start, end),
        }
