"""Freeze amendment F1: the bundle comparison flags evidence changes and ignores identity metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.b_guard_equivalence import compare  # noqa: E402

ITEM = {"key": "load", "tool": "ts_load", "arguments": {"fs": 128}, "response_sha256": "a", "sizes": {"text_chars": 5},
        "priority": 0, "required": True, "is_error": False, "call_number": 1}  # fmt: skip
BUNDLE = {"content": [{"type": "text", "text": "x"}], "items": [ITEM], "dropped": [], "sizes": {"calls": 1},
          "status": "ready", "qtype": "af_burden", "identity": {"workflow_sha256": "old"}}  # fmt: skip


def test_identity_only_difference_is_equivalent():
    assert compare(BUNDLE, {**BUNDLE, "identity": {"workflow_sha256": "new"}}) == []


def test_content_or_item_change_is_detected():
    assert "content" in compare(BUNDLE, {**BUNDLE, "content": [{"type": "text", "text": "y"}]})
    changed = {**BUNDLE, "items": [{**ITEM, "response_sha256": "b"}]}
    assert "item 0 response_sha256" in compare(BUNDLE, changed)
