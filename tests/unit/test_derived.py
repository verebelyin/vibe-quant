"""Tests for vibe_quant.dsl.derived — computed-output helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibe_quant.dsl.derived import (
    compute_bandwidth,
    compute_percent_b,
    compute_position,
)


def _band(upper: float, middle: float, lower: float) -> SimpleNamespace:
    """Build a duck-typed band object matching the NT indicator surface."""
    return SimpleNamespace(upper=upper, middle=middle, lower=lower)


class TestComputePercentB:
    """Golden-value coverage for Bollinger %B."""

    def test_close_equals_upper_returns_one(self) -> None:
        band = _band(upper=110.0, middle=100.0, lower=90.0)
        assert compute_percent_b(band, 110.0) == pytest.approx(1.0)

    def test_close_equals_lower_returns_zero(self) -> None:
        band = _band(upper=110.0, middle=100.0, lower=90.0)
        assert compute_percent_b(band, 90.0) == pytest.approx(0.0)

    def test_close_equals_middle_returns_half(self) -> None:
        band = _band(upper=110.0, middle=100.0, lower=90.0)
        assert compute_percent_b(band, 100.0) == pytest.approx(0.5)

    def test_close_above_upper_overshoots(self) -> None:
        band = _band(upper=110.0, middle=100.0, lower=90.0)
        # 120 is 150% of the way from lower (90) to upper (110)
        assert compute_percent_b(band, 120.0) == pytest.approx(1.5)

    def test_close_below_lower_undershoots(self) -> None:
        band = _band(upper=110.0, middle=100.0, lower=90.0)
        # 80 is -50% of the way from lower (90) to upper (110)
        assert compute_percent_b(band, 80.0) == pytest.approx(-0.5)

    def test_zero_range_returns_neutral_half(self) -> None:
        band = _band(upper=100.0, middle=100.0, lower=100.0)
        assert compute_percent_b(band, 100.0) == 0.5

    def test_negative_range_returns_neutral_half(self) -> None:
        """Pathological inverted band: falls back to 0.5."""
        band = _band(upper=90.0, middle=100.0, lower=110.0)
        assert compute_percent_b(band, 100.0) == 0.5

    def test_asymmetric_band(self) -> None:
        """Non-centered bands still compute correctly."""
        band = _band(upper=120.0, middle=100.0, lower=95.0)
        # close=100 → (100-95)/(120-95) = 5/25 = 0.2
        assert compute_percent_b(band, 100.0) == pytest.approx(0.2)


class TestComputeBandwidth:
    """Golden-value coverage for Bollinger bandwidth."""

    def test_standard_bandwidth(self) -> None:
        band = _band(upper=110.0, middle=100.0, lower=90.0)
        # (110 - 90) / 100 = 0.2
        assert compute_bandwidth(band, 100.0) == pytest.approx(0.2)

    def test_close_is_ignored(self) -> None:
        """Bandwidth does not depend on close price."""
        band = _band(upper=110.0, middle=100.0, lower=90.0)
        assert compute_bandwidth(band, 0.0) == compute_bandwidth(band, 1000.0)

    def test_zero_middle_returns_zero(self) -> None:
        band = _band(upper=10.0, middle=0.0, lower=-10.0)
        assert compute_bandwidth(band, 0.0) == 0.0

    def test_negative_middle_returns_zero(self) -> None:
        """Safety fallback for degenerate input."""
        band = _band(upper=10.0, middle=-5.0, lower=-20.0)
        assert compute_bandwidth(band, 0.0) == 0.0

    def test_narrow_band(self) -> None:
        band = _band(upper=100.1, middle=100.0, lower=99.9)
        # (100.1 - 99.9) / 100 = 0.002
        assert compute_bandwidth(band, 100.0) == pytest.approx(0.002)

    def test_wide_band(self) -> None:
        band = _band(upper=150.0, middle=100.0, lower=50.0)
        # (150 - 50) / 100 = 1.0
        assert compute_bandwidth(band, 100.0) == pytest.approx(1.0)


class TestComputePosition:
    """Golden-value coverage for Donchian position."""

    def test_close_at_upper_returns_one(self) -> None:
        band = _band(upper=120.0, middle=100.0, lower=80.0)
        assert compute_position(band, 120.0) == pytest.approx(1.0)

    def test_close_at_lower_returns_zero(self) -> None:
        band = _band(upper=120.0, middle=100.0, lower=80.0)
        assert compute_position(band, 80.0) == pytest.approx(0.0)

    def test_close_at_middle_returns_half(self) -> None:
        band = _band(upper=120.0, middle=100.0, lower=80.0)
        assert compute_position(band, 100.0) == pytest.approx(0.5)

    def test_close_above_upper_overshoots(self) -> None:
        band = _band(upper=120.0, middle=100.0, lower=80.0)
        assert compute_position(band, 140.0) == pytest.approx(1.5)

    def test_zero_range_returns_neutral_half(self) -> None:
        band = _band(upper=100.0, middle=100.0, lower=100.0)
        assert compute_position(band, 100.0) == 0.5


class TestRegressionEquivalence:
    """Verify derived helpers match the pre-P2 compiler-inlined formulas exactly."""

    @pytest.mark.parametrize(
        ("upper", "lower", "close"),
        [
            (110.0, 90.0, 100.0),
            (110.0, 90.0, 95.0),
            (110.0, 90.0, 105.0),
            (110.0, 90.0, 110.0),
            (110.0, 90.0, 90.0),
            (200.0, 100.0, 150.0),
            (1.0001, 0.9999, 1.0),
        ],
    )
    def test_percent_b_matches_legacy_inline_formula(
        self, upper: float, lower: float, close: float
    ) -> None:
        """Legacy inline formula from compiler._generate_computed_output_code."""
        band = _band(upper=upper, middle=(upper + lower) / 2, lower=lower)
        band_range = upper - lower
        expected = (close - lower) / band_range if band_range > 0 else 0.5
        assert compute_percent_b(band, close) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("upper", "middle", "lower"),
        [
            (110.0, 100.0, 90.0),
            (200.0, 100.0, 50.0),
            (101.0, 100.0, 99.0),
            (1.0, 0.5, 0.0),
        ],
    )
    def test_bandwidth_matches_legacy_inline_formula(
        self, upper: float, middle: float, lower: float
    ) -> None:
        """Legacy inline formula from compiler._generate_computed_output_code."""
        band = _band(upper=upper, middle=middle, lower=lower)
        expected = (upper - lower) / middle if middle > 0 else 0.0
        assert compute_bandwidth(band, 0.0) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("upper", "lower", "close"),
        [
            (120.0, 80.0, 100.0),
            (120.0, 80.0, 80.0),
            (120.0, 80.0, 120.0),
            (120.0, 80.0, 85.0),
            (120.0, 80.0, 115.0),
        ],
    )
    def test_position_matches_legacy_inline_formula(
        self, upper: float, lower: float, close: float
    ) -> None:
        """Legacy inline formula from compiler._generate_computed_output_code."""
        band = _band(upper=upper, middle=(upper + lower) / 2, lower=lower)
        channel_range = upper - lower
        expected = (close - lower) / channel_range if channel_range > 0 else 0.5
        assert compute_position(band, close) == pytest.approx(expected)
