"""Unified, model-independent runner for SignalMind capability-amplification experiments (eval-side only).

One tool loop drives any model through a provider driver (Anthropic Messages API for Claude and DeepSeek's
Anthropic-compatible endpoint; OpenAI chat-completions for an OpenAI-compatible gateway). Arms differ only in the
tool set: Base = sandboxed Python (no network) + image viewing; Agent = the SignalMind MCP server in-process (specialist
plugins, TaskSpec, evidence ledger, verification, ts_submit). Historical Agent C and the frozen harness are untouched.
"""
