"""Registered confirmation analysis helpers (no data, no model)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.confirm_analysis import holm  # noqa: E402


def test_holm_two_tests():
    adj = holm({"a": 0.01, "b": 0.04})
    assert adj["a"] == pytest.approx(0.02) and adj["b"] == pytest.approx(0.04)


def test_holm_is_monotone_and_capped():
    adj = holm({"a": 0.03, "b": 0.02, "c": 0.9})
    assert adj["b"] == pytest.approx(0.06) and adj["a"] == pytest.approx(0.06) and adj["c"] == pytest.approx(0.9)
