"""Statistics helpers of the p5afdb arm comparison (no model calls, no data)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.compare_p5afdb_arms import newcombe_paired, wilson  # noqa: E402


def test_wilson_known_value():
    lo, hi = wilson(12, 15)
    assert lo == pytest.approx(0.5481, abs=1e-3) and hi == pytest.approx(0.9295, abs=1e-3)


@pytest.mark.parametrize("a,b,c,d", [(10, 2, 0, 3), (12, 0, 0, 3), (5, 3, 3, 4), (0, 0, 0, 15), (9, 3, 1, 2)])
def test_newcombe_contains_difference_and_stays_in_range(a, b, c, d):
    delta, lo, hi = newcombe_paired(a, b, c, d)
    assert delta == pytest.approx((b - c) / (a + b + c + d))
    assert -1 <= lo <= delta <= hi <= 1


def test_newcombe_is_antisymmetric():
    d1, lo1, hi1 = newcombe_paired(9, 3, 1, 2)
    d2, lo2, hi2 = newcombe_paired(9, 1, 3, 2)
    assert d1 == pytest.approx(-d2) and lo1 == pytest.approx(-hi2) and hi1 == pytest.approx(-lo2)
