"""Pure-Python core (no mcp import). Provisional API during 0.x."""

from tsmcp.core.errors import SignalError
from tsmcp.core.signal import Channel, Signal
from tsmcp.core.store import SignalStore

__all__ = ["Channel", "Signal", "SignalError", "SignalStore"]
