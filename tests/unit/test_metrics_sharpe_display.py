"""One Sharpe annualisation on screen: stored √252 in, display √365 out.

User ruling Q4=A: every Sharpe/Sortino a human reads is annualised with 365 days
(crypto trades every day) while every stored value keeps NT's 252-day meaning.
The conversion is one constant and one helper pair in ``vibe_quant.metrics``.
"""

from __future__ import annotations

import math

import pytest

from vibe_quant.metrics import (
    DISPLAY_SHARPE_DAYS,
    SHARPE_DISPLAY_FACTOR,
    STORED_SHARPE_DAYS,
    fmt_sharpe,
    sharpe_from_display,
    sharpe_to_display,
)

#: The screening value of strategy 239 on main@efc19e47, and its display twin.
STRATEGY_239_SCREENING_SHARPE = 0.8004025908124951
STRATEGY_239_DISPLAY_SHARPE = 0.9632846671540376


def test_display_of_strategy_239_screening() -> None:
    assert sharpe_to_display(STRATEGY_239_SCREENING_SHARPE) == STRATEGY_239_DISPLAY_SHARPE


def test_display_factor_matches_stored_days() -> None:
    from vibe_quant.overfitting import dsr, purged_kfold

    assert math.sqrt(365 / 252) == SHARPE_DISPLAY_FACTOR == 1.2035001862952488
    assert DISPLAY_SHARPE_DAYS == 365
    # Every stored Sharpe is NT's 252-day one; the display basis is the only new one.
    assert dsr.TRADING_DAYS_PER_YEAR == STORED_SHARPE_DAYS == 252
    assert purged_kfold._SHARPE_PERIODS_PER_YEAR == STORED_SHARPE_DAYS  # noqa: SLF001


@pytest.mark.parametrize(
    ("stored", "display"),
    [
        (0.0, 0.0),
        (-0.12, -0.14442002235542986),
        (2.0, 2.4070003725904976),
    ],
)
def test_display_scales_either_sign(stored: float, display: float) -> None:
    assert sharpe_to_display(stored) == display


@pytest.mark.parametrize("stored", [None, math.nan, math.inf, -math.inf])
def test_display_of_missing_or_non_finite_is_none(stored: float | None) -> None:
    assert sharpe_to_display(stored) is None


def test_display_round_trips_a_stored_value() -> None:
    # 0.5 stored -> 0.6017500931476244 shown -> back to 0.5, bit for bit.
    assert sharpe_to_display(0.5) == 0.6017500931476244
    assert sharpe_from_display(sharpe_to_display(0.5)) == 0.5


def test_typed_display_limit_converts_to_stored_units() -> None:
    assert sharpe_from_display(0.6) == 0.4985458305968263


@pytest.mark.parametrize("display", [math.nan, math.inf, -math.inf])
def test_typed_non_finite_limit_is_refused(display: float) -> None:
    with pytest.raises(ValueError):
        sharpe_from_display(display)


def test_fmt_sharpe_prints_display_value_or_na() -> None:
    assert fmt_sharpe(STRATEGY_239_SCREENING_SHARPE) == "0.96"
    assert fmt_sharpe(STRATEGY_239_SCREENING_SHARPE, 3) == "0.963"
    assert fmt_sharpe(-0.12) == "-0.14"
    assert [fmt_sharpe(v) for v in (None, math.nan, math.inf, -math.inf)] == ["n/a"] * 4
