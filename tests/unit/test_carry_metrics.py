"""Funding-carry metrics: equity path, weekly Sharpe and P&L decomposition (task T4).

Contract rows R22-R25, R27, R27b, R27c, R33 and the chief's R48 ruling of the
funding-carry semantics contract (``data/swarm/20261015-funding-carry/design.md``
§4, §7, §8-T4). Every fixture here is a hand-built ``CarryRun`` -- no engine, no
data archive -- so each expected number is a hand-computed contract value. The
weekly Sharpe fixture pins the exact float ``4.163331998932265`` / Lo SE
``12.944325225965066``; the turnover fixture pins the per-decision grouping
(``(4998 + 5046) / 10096 == 0.9948494453248812``, not the per-fill sum).
"""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from vibe_quant.data.catalog import create_instrument
from vibe_quant.portfolio.ledger import CarryFill, CarryRun, FundingPayment
from vibe_quant.portfolio.metrics import (
    _annualised_sharpe,
    _lo_se,
    _require_cost_identity,
    _weekly_sharpe,
    carry_metrics,
)
from vibe_quant.portfolio.sizing import Decision, InstrumentSpec
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryEquityTooSmall,
    CarryExecutionError,
    CarrySpec,
)
from vibe_quant.portfolio.ticks import CarryInputs
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from vibe_quant.portfolio.metrics import CarryReport

_DAY = 86_400 * 10**9
_HOUR = 3_600 * 10**9
_MIN = 60 * 10**9
_START = 1_736_121_600_000_000_000  # 2025-01-06 00:00 UTC, a Monday
_ETH = "ETHUSDT"
_BTC = "BTCUSDT"

# 2023-01-01 / 2023-12-25 / 2024-01-01 / 2024-01-08 / 2025-01-01 (UTC 00:00)
_Y2023 = 1_672_531_200_000_000_000
_BUY_R27B = 1_703_462_400_000_000_000
_Y2024 = 1_704_067_200_000_000_000
_SELL_R27B = 1_704_672_000_000_000_000
_Y2025 = 1_735_689_600_000_000_000
_Y2026 = 1_767_225_600_000_000_000
_EVE2024 = _Y2024 - _DAY  # 2023-12-31 00:00 UTC
_LATE_2023 = _Y2024 - _MIN  # 2023-12-31 23:59 UTC, the last decision of 2023
_ED = 10_096.0  # the decision equity every boundary-turnover fixture normalises by


def _specs() -> dict[str, InstrumentSpec]:
    return {s: InstrumentSpec.from_instrument(create_instrument(s)) for s in (_BTC, _ETH)}


def _fill(
    symbol: str,
    ts: int,
    steps: int,
    price: float = 0.0,
    fee: float = 0.0,
    reduce_only: bool = False,
) -> CarryFill:
    return CarryFill(
        symbol=symbol,
        ts=ts,
        steps=steps,
        qty=float(Decimal(steps) * _specs()[symbol].step),
        price=price,
        fee=fee,
        reduce_only=reduce_only,
    )


def _decision(
    ts: int,
    equity: float,
    signals: dict[str, float | None] | None = None,
    weights: dict[str, float] | None = None,
    rounding_drift: float = 0.0,
) -> Decision:
    return Decision(
        ts=ts,
        equity=equity,
        signals=signals if signals is not None else {},
        excluded={},
        weights=weights if weights is not None else {},
        prices={},
        target_steps={},
        orders=(),
        band_skips=0,
        min_notional_skips=0,
        rounding_drift=rounding_drift,
    )


def _inputs(spec: CarrySpec, boundaries: dict[str, dict[int, float]]) -> CarryInputs:
    return CarryInputs(
        spec=spec,
        decisions=(),
        instruments={s: create_instrument(s) for s in spec.symbols},
        specs={s: InstrumentSpec.from_instrument(create_instrument(s)) for s in spec.symbols},
        ticks={},
        settlements={},
        marks={},
        calc=FundingCalculator(),
        boundary_prices=boundaries,
    )


def _run(
    spec: CarrySpec,
    fills: list[CarryFill],
    payments: list[FundingPayment],
    decisions: tuple[Decision, ...] = (),
) -> CarryRun:
    return CarryRun(
        spec=spec,
        decisions=decisions,
        fills=tuple(fills),
        payments=tuple(payments),
        final_steps={},
        engine="reference",
    )


def _day(index: int) -> int:
    """``_START`` plus ``index`` whole days (Mondays at 0, 7, 14, ...)."""
    return _START + index * _DAY


def test_daily_equity_and_returns() -> None:
    """R22: ``daily[i] == (B_{i+1}, E(B_{i+1}), r_i)``; events at ``ts < B`` only.

    A funding settlement stamped ``00:00:00.003`` (3 ms after boundary B1) does
    not move ``E(B1)`` -- it lands in ``E(B2)``.
    """
    b0 = _START
    b1, b2, b3 = _START + _DAY, _START + 2 * _DAY, _START + 3 * _DAY
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=b0, end_ns=b3)
    fills = [
        _fill(_BTC, b0 + _HOUR, 100, price=50000.0, fee=2.5),
        _fill(_BTC, b2 + _HOUR, -100, price=51000.0, fee=2.55),
    ]
    payments = [FundingPayment(_BTC, b1 + 3_000_000, 1.5, False)]
    inputs = _inputs(spec, {_BTC: {b1: 51000.0, b2: 52000.0}, _ETH: {}})
    decisions = (_decision(b0, 10_000.0, weights={_BTC: 0.5}),)
    report = carry_metrics(_run(spec, fills, payments, decisions), inputs)

    assert [row[0] for row in report.daily] == [b1, b2, b3]
    assert len(report.daily) == 3
    assert report.daily[0][1] == pytest.approx(10097.5)
    assert report.daily[1][1] == pytest.approx(10196.0)  # still held: 0.1 x 52000
    assert report.daily[2][1] == pytest.approx(10093.45)  # closed at b2 + 1h
    assert report.daily[0][2] == pytest.approx(0.009749999999999925, rel=1e-12)
    assert report.daily[1][2] == pytest.approx(0.009754889824213908, rel=1e-12)
    assert report.daily[2][2] == pytest.approx(-0.010057865829737112, rel=1e-12)
    assert report.summary["n_fills"] == 2
    assert report.summary["liquidation"] == "not_modelled"


def test_daily_held_symbol_without_boundary_price_fails() -> None:
    """R22: a held symbol with no close in ``(B-60 min, B]`` fails closed."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + 2 * _DAY)
    fills = [_fill(_BTC, _START + _HOUR, 100, price=50000.0)]
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})  # B1 holds BTC but has no price
    with pytest.raises(CarryDataError, match="no price for held BTCUSDT"):
        carry_metrics(_run(spec, fills, []), inputs)


def test_daily_non_positive_equity_fails() -> None:
    """R22: ``E(B) <= 0`` raises ``CarryEquityTooSmall`` before any ratio (edge at 0)."""
    b0 = _START
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=b0, end_ns=b0 + _DAY)
    fills = [
        _fill(_BTC, b0 + _HOUR, 100, price=100.0, fee=20_000.0),
        _fill(_BTC, b0 + 2 * _HOUR, -100, price=100.0),
    ]
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    with pytest.raises(CarryEquityTooSmall) as error:
        carry_metrics(_run(spec, fills, []), inputs)
    assert error.value.symbol == "<book>"
    assert error.value.decision_ns == b0 + _DAY
    assert error.value.leg_notional == pytest.approx(-10_000.0)
    assert error.value.required == 0.0

    zero = [  # E(B_1) == 0.0 exactly: the check is ``not (equity > 0.0)``, not ``< 0``
        _fill(_BTC, b0 + _HOUR, 100, price=100.0, fee=5_000.0),
        _fill(_BTC, b0 + 2 * _HOUR, -100, price=100.0, fee=5_000.0),
    ]
    with pytest.raises(CarryEquityTooSmall) as zero_error:
        carry_metrics(_run(spec, zero, []), inputs)
    assert zero_error.value.leg_notional == 0.0


def test_weekly_sharpe_value_and_null() -> None:
    """R23: mean/stdev(ddof=1)*sqrt(52); null with < 3 weeks or zero stdev.

    Lo SE ``sqrt((1 + SR^2/2) / years)`` with years = weeks/52.
    """
    returns = [0.01, -0.005, 0.01]
    assert _weekly_sharpe(returns) == 4.163331998932265
    assert _lo_se(_weekly_sharpe(returns), len(returns)) == 12.944325225965066
    assert _weekly_sharpe([0.01, -0.005]) is None  # 2 weeks
    assert _lo_se(None, 2) is None
    assert _weekly_sharpe([0.01, -0.005, 0.01, 0.02, 0.03]) is not None  # >= 3 weeks
    assert _weekly_sharpe([0.01, 0.01, 0.01]) is None  # zero stdev


def test_annualised_sharpe_floor_edge_and_relative_scale() -> None:
    """R50 floor ``stdev <= 1e-12 * max(1, |mean|)``: the edge and the ``|mean| < 1`` branch.

    ``[m-d, m, m+d]`` has mean ``m`` and stdev exactly ``d`` (``fsum`` of ``2 d^2`` over
    ``ddof=1``). Three probes pin the two surviving mutants of that guard line:

    * ``m = 0``, ``d = 1e-12`` sits exactly ON the floor -> null. A strict ``<`` compare
      would return the 0.0 ratio instead (floor mutant 1);
    * ``m = 0.01`` selects the ``max(1.0, |mean|) = 1.0`` branch, so ``d = 5e-13`` is below
      the 1e-12 floor -> null. Dropping the ``max`` would scale the floor to ``1e-14`` and
      return a value (floor mutant 2);
    * ``d = 1.5e-12`` is above the floor -> a value (the floor is not over-tight).

    ``_weekly_sharpe`` re-pins the period minimum: ``n = 2`` -> null, ``n = 3`` -> value.
    """
    assert _annualised_sharpe([-1e-12, 0.0, 1e-12], 52.0) is None
    assert _annualised_sharpe([0.01 - 5e-13, 0.01, 0.01 + 5e-13], 52.0) is None
    assert _annualised_sharpe([-1.5e-12, 0.0, 1.5e-12], 52.0) == 0.0
    assert _annualised_sharpe([-1.5e-12, 0.0, 1.5e-12], 52.0) is not None
    assert _weekly_sharpe([0.01, -0.005]) is None  # n = 2 < _MIN_PERIODS
    assert _weekly_sharpe([0.01, -0.005, 0.01]) == 4.163331998932265  # n = 3


def test_weekly_returns_are_monday_to_monday() -> None:
    """R23: ``weekly[j] == (M_{j+1}, E(M_{j+1})/E(M_j) - 1)`` over Monday boundaries."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_day(0), end_ns=_day(21))
    fills = [_fill(_BTC, _day(0) + _HOUR, 100, price=50000.0)]
    prices = {_day(i): (51000.0 if i <= 7 or i > 14 else 50000.0) for i in range(1, 22)}
    inputs = _inputs(spec, {_BTC: prices, _ETH: {}})
    decisions = (_decision(_day(0), 10_000.0, weights={_BTC: 0.5}),)
    report = carry_metrics(_run(spec, fills, [], decisions), inputs)

    assert [row[0] for row in report.weekly] == [_day(7), _day(14), _day(21)]
    assert report.summary["weeks"] == 3
    assert report.weekly[0][1] == pytest.approx(0.01)
    assert report.weekly[1][1] == pytest.approx(10_000.0 / 10_100.0 - 1.0)
    assert report.weekly[2][1] == pytest.approx(0.01)
    assert report.summary["weekly_sharpe"] == pytest.approx(
        _weekly_sharpe([row[1] for row in report.weekly])
    )
    assert report.summary["weekly_sharpe"] is not None


def test_cost_identity_single_leg() -> None:
    """R24: (BTC,long) price 100.0, fees 5.05, funding 1.5; ``E(end) - E0 == 93.45``."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    fills = [
        _fill(_BTC, _START + _HOUR, 100, price=50000.0, fee=2.5),
        _fill(_BTC, _START + 2 * _HOUR, -100, price=51000.0, fee=2.55),
    ]
    payments = [FundingPayment(_BTC, _START + 90 * _MIN, 1.5, False)]
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    decisions = (_decision(_START, 10_000.0, weights={_BTC: 0.5}),)
    report = carry_metrics(_run(spec, fills, payments, decisions), inputs)

    btc_long = next(leg for leg in report.legs if leg["symbol"] == _BTC and leg["side"] == "long")
    assert btc_long["price_pnl"] == 100.0
    assert btc_long["fees"] == 5.05
    assert btc_long["funding_paid"] == 1.5
    assert report.summary["price_pnl_total"] == 100.0
    assert report.summary["fees_total"] == 5.05
    assert report.summary["funding_paid_total"] == 1.5
    assert report.summary["total_return"] == pytest.approx(93.45 / 10_000.0)
    equity_end = report.daily[-1][1]
    identity = report.summary["price_pnl_total"] - report.summary["funding_paid_total"] - report.summary["fees_total"]
    assert identity == pytest.approx(equity_end - 10_000.0, abs=1e-9 * 10_000.0)
    assert equity_end - 10_000.0 == pytest.approx(93.45, abs=1e-9 * 10_000.0)


def test_cost_identity_violation_fails_closed() -> None:
    """R3a: a run whose legs and equity path disagree raises ``CarryExecutionError``.

    ``_legs`` marks a position held at the end with the catalog's ``spec.step``
    while ``equity_at`` infers its step from ``fill.qty / fill.steps``: a doctored
    fill (``qty`` 2x the exact step) breaks ``price - funding - fees == E(end) - E0``
    beyond ``1e-9 * E0``, which the runtime check must refuse.
    """
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    fill = _fill(_BTC, _START + _HOUR, 100, price=50000.0, fee=2.5)
    doctored = replace(fill, qty=fill.qty * 2.0)
    inputs = _inputs(spec, {_BTC: {_START + _DAY: 51000.0}, _ETH: {}})
    decisions = (_decision(_START, 10_000.0, weights={_BTC: 0.5}),)

    # The same run with the exact quantity satisfies the identity (the control).
    control = carry_metrics(_run(spec, [fill], [], decisions), inputs)
    assert control.summary["price_pnl_total"] == 100.0
    assert control.summary["price_pnl_total"] - control.summary["fees_total"] == pytest.approx(
        control.daily[-1][1] - 10_000.0
    )

    with pytest.raises(CarryExecutionError, match="cost identity"):
        carry_metrics(_run(spec, [doctored], [], decisions), inputs)


def test_cost_identity_tolerance_is_relative_to_e0() -> None:
    """R3a: the tolerance is exactly ``1e-9 * E0`` -- pinned at both edges.

    ``test_cost_identity_violation_fails_closed`` above breaks the identity by about
    ``E0``; this pair pins the tolerance WIDTH instead. With ``E0 == 10_000`` the
    tolerance is ``1e-5``, so a gap of ``1e-7 * E0 == 1e-3`` (100x too big) must
    raise while ``1e-11 * E0 == 1e-7`` (100x within it) must pass. Loosening
    ``_COST_IDENTITY_TOL`` to ``1e-1`` (tolerance ``1e-1 * E0 == 1000``) lets the
    violation through; dropping the ``abs(initial_balance)`` scale to ``1.0``
    (tolerance ``1e-9``) makes the passing gap raise.
    """
    e0 = 10_000.0
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    run = _run(spec, [], [])
    summary = {"price_pnl_total": 0.0, "funding_paid_total": 0.0, "fees_total": 0.0}

    with pytest.raises(CarryExecutionError, match="cost identity"):
        _require_cost_identity(run, e0, summary, e0 + 1e-7 * e0)
    _require_cost_identity(run, e0, summary, e0 + 1e-11 * e0)


def test_flip_attribution() -> None:
    """R24b: the close goes to the old side, the open to the new one (10 / 5 units)."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    fills = [
        _fill(_BTC, _START + _HOUR, 10_000, price=100.0),
        _fill(_BTC, _START + 2 * _HOUR, -10_000, price=110.0, fee=0.55, reduce_only=True),
        _fill(_BTC, _START + 2 * _HOUR, -5_000, price=110.0, fee=0.275),
    ]
    inputs = _inputs(spec, {_BTC: {_START + _DAY: 100.0}, _ETH: {}})
    decisions = (_decision(_START, 10_000.0, weights={_BTC: 0.5}),)
    report = carry_metrics(_run(spec, fills, [], decisions), inputs)

    long_leg = next(leg for leg in report.legs if leg["side"] == "long")
    short_leg = next(leg for leg in report.legs if leg["side"] == "short")
    assert long_leg["price_pnl"] == 100.0
    assert long_leg["fees"] == 0.55
    assert short_leg["price_pnl"] == 50.0
    assert short_leg["fees"] == 0.275
    equity_end = report.daily[-1][1]
    identity = report.summary["price_pnl_total"] - report.summary["funding_paid_total"] - report.summary["fees_total"]
    assert identity == pytest.approx(equity_end - 10_000.0, abs=1e-9 * 10_000.0)


def test_turnover() -> None:
    """R25: ``abs(q*px)`` 4998 + 5046 at ``E_D`` 10096 -> 0.9948494453248812."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    fills = [
        _fill(_BTC, _START + _HOUR, 98, price=51000.0),
        _fill(_ETH, _START + _HOUR, -1682, price=3000.0),
    ]
    inputs = _inputs(
        spec,
        {_BTC: {_START + _DAY: 51000.0}, _ETH: {_START + _DAY: 3000.0}},
    )
    decisions = (_decision(_START, 10_096.0, weights={_BTC: 0.5, _ETH: -0.5}),)
    report = carry_metrics(_run(spec, fills, [], decisions), inputs)

    assert report.summary["turnover_total"] == 0.9948494453248812
    assert report.summary["turnover_annual"] == pytest.approx(0.9948494453248812 * 365.0)
    leg_sum = sum(leg["turnover"] for leg in report.legs)
    assert leg_sum == pytest.approx(report.summary["turnover_total"], abs=1e-9 * 10_000.0)


def test_by_year_sums_to_totals() -> None:
    """R27: per-leg, per-side and per-year sums equal the pooled totals."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + 3 * _DAY)
    fills = [
        _fill(_BTC, _START + _HOUR, 100, price=50000.0, fee=2.5),
        _fill(_ETH, _START + _HOUR, -1000, price=3000.0, fee=1.25),
        _fill(_BTC, _START + _DAY + _HOUR, -50, price=51000.0, fee=1.25),
    ]
    payments = [
        FundingPayment(_BTC, _START + 8 * _HOUR, 0.5, False),
        FundingPayment(_ETH, _START + 8 * _HOUR, -0.25, False),
    ]
    inputs = _inputs(
        spec,
        {
            _BTC: {_START + i * _DAY: 51000.0 for i in range(1, 4)},
            _ETH: {_START + i * _DAY: 3000.0 for i in range(1, 4)},
        },
    )
    decisions = (
        _decision(_START, 10_000.0, weights={_BTC: 0.5, _ETH: -0.5}),
        _decision(_START + _DAY, 10_000.0, weights={_BTC: 0.25, _ETH: -0.25}),
    )
    report = carry_metrics(_run(spec, fills, payments, decisions), inputs)
    summary = report.summary
    tolerance = 1e-9 * 10_000.0

    assert sum(leg["price_pnl"] for leg in report.legs) == pytest.approx(summary["price_pnl_total"], abs=tolerance)
    assert sum(leg["fees"] for leg in report.legs) == pytest.approx(summary["fees_total"], abs=tolerance)
    assert sum(leg["funding_paid"] for leg in report.legs) == pytest.approx(
        summary["funding_paid_total"], abs=tolerance
    )
    long_side = sum(leg["price_pnl"] for leg in report.legs if leg["side"] == "long")
    short_side = sum(leg["price_pnl"] for leg in report.legs if leg["side"] == "short")
    assert long_side + short_side == pytest.approx(summary["price_pnl_total"], abs=tolerance)

    assert sum(row["price_pnl"] for row in report.by_year) == pytest.approx(
        summary["price_pnl_total"], abs=tolerance
    )
    assert sum(row["fees"] for row in report.by_year) == pytest.approx(summary["fees_total"], abs=tolerance)
    assert sum(row["funding_paid"] for row in report.by_year) == pytest.approx(
        summary["funding_paid_total"], abs=tolerance
    )
    assert sum(row["turnover"] for row in report.by_year) == pytest.approx(
        summary["turnover_total"], abs=tolerance
    )
    assert summary["price_pnl_total"] - summary["funding_paid_total"] - summary["fees_total"] == pytest.approx(
        report.daily[-1][1] - 10_000.0, abs=tolerance
    )
    assert isinstance(summary["funding_estimated"], bool)


def _boundary_run(fills: list[CarryFill], decisions: tuple[Decision, ...]) -> CarryReport:
    """Two-day run across the 2024 boundary; BTC marked at 51000 on every close.

    Boundaries are 2023-12-31 00:00, 2024-01-01 00:00 (== ``_Y2024``) and 2024-01-02
    00:00, so 2023 is the short first bucket ``[_EVE2024, _Y2024)``.
    """
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_EVE2024, end_ns=_Y2024 + _DAY)
    prices = {_EVE2024 + index * _DAY: 51000.0 for index in range(3)}
    return carry_metrics(
        _run(spec, fills, [], decisions), _inputs(spec, {_BTC: prices, _ETH: {}})
    )


def test_by_year_turnover_follows_the_fill_not_the_decision() -> None:
    """R27c: a 2023-12-31 23:59 decision whose fill lands 2024-01-01 books 2024.

    The fill is one tick (1 min) after the last decision of 2023 and sits exactly ON
    the 2024-01-01 boundary, so the 2023 bucket ``[.., _Y2024)`` must NOT carry it.
    Bucketing by the decision timestamp books all of ``abs(q*px) == 4998.0 /
    10096.0`` in 2023 and leaves 2024 at 0.0; bucketing with ``<=`` at ``y1`` books
    it in BOTH years.
    """
    fill = _fill(_BTC, _Y2024, 98, price=51000.0)
    decisions = (_decision(_LATE_2023, _ED, weights={_BTC: 0.5}),)
    report = _boundary_run([fill], decisions)

    assert [row["year"] for row in report.by_year] == [2023, 2024]
    assert report.by_year[0]["y0_ns"] == _EVE2024
    assert report.by_year[0]["y1_ns"] == _Y2024
    assert report.by_year[0]["turnover"] == 0.0
    assert report.by_year[1]["turnover"] == 4998.0 / _ED
    assert report.by_year[1]["turnover"] == report.summary["turnover_total"]
    assert report.summary["turnover_total"] == 4998.0 / _ED


def test_by_year_turnover_splits_one_decisions_fills_at_the_boundary() -> None:
    """The last ns of 2023 counts in 2023, and both fills use the decision's ``E_D``.

    One decision at 2023-12-31 23:59 releases two fills -- 98 BTC at
    ``_Y2024 - 1`` ns and 100 BTC at ``_Y2024`` -- so a per-decision grouping (which
    cannot split one decision's fills) misplaces half the year's turnover.
    """
    decisions = (_decision(_LATE_2023, _ED, weights={_BTC: 0.5}),)
    fills = [
        _fill(_BTC, _Y2024 - 1, 98, price=51000.0),
        _fill(_BTC, _Y2024, 100, price=51000.0),
    ]
    report = _boundary_run(fills, decisions)

    assert report.by_year[0]["turnover"] == 4998.0 / _ED
    assert report.by_year[1]["turnover"] == 5100.0 / _ED
    tolerance = 1e-9 * 10_000.0
    assert report.by_year[0]["turnover"] + report.by_year[1]["turnover"] == pytest.approx(
        report.summary["turnover_total"], abs=tolerance
    )
    assert report.summary["turnover_total"] == (4998.0 + 5100.0) / _ED


def test_per_year_price_pnl_is_mark_to_market() -> None:
    """R27b: 2023 = 200.0, 2024 = -100.0 at the 2024-01-01 boundary price 52000."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_Y2023, end_ns=_Y2025)
    fills = [
        _fill(_BTC, _BUY_R27B + _HOUR, 100, price=50000.0),
        _fill(_BTC, _SELL_R27B + _HOUR, -100, price=51000.0),
    ]
    prices = {_Y2023 + i * _DAY: 52000.0 for i in range((_Y2025 - _Y2023) // _DAY + 1)}
    inputs = _inputs(spec, {_BTC: prices, _ETH: {}})
    decisions = (
        _decision(_BUY_R27B - _HOUR, 10_000.0, weights={_BTC: 0.5}),
        _decision(_SELL_R27B - _HOUR, 10_000.0, weights={_BTC: 0.0}),
    )
    report = carry_metrics(_run(spec, fills, [], decisions), inputs)

    assert [row["year"] for row in report.by_year] == [2023, 2024]
    y2023, y2024 = report.by_year
    assert y2023["price_pnl"] == 200.0
    assert y2024["price_pnl"] == -100.0
    assert y2024["y0_ns"] == _Y2024
    assert y2023["y1_ns"] == _Y2024
    assert y2023["equity_start"] == pytest.approx(10_000.0)
    assert y2023["equity_end"] == pytest.approx(10_200.0)
    assert y2024["equity_end"] == pytest.approx(10_100.0)
    tolerance = 1e-9 * 10_000.0
    for row in report.by_year:
        assert row["price_pnl"] - row["funding_paid"] - row["fees"] == pytest.approx(
            row["equity_end"] - row["equity_start"], abs=tolerance
        )


def test_dispersion_and_fee_ratio() -> None:
    """R27c: spreads 0.002 and 0.001 -> dispersion 0.078; fees 15.0 / E 10000 = 0.0015."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_Y2025, end_ns=_Y2026)
    fills = [
        _fill(_BTC, _Y2025 + _HOUR, 100, price=100.0, fee=7.5),
        _fill(_BTC, _Y2025 + 2 * _HOUR, -100, price=100.0, fee=7.5),
    ]
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    decisions = (
        _decision(_Y2025, 10_000.0, signals={_BTC: 0.002, _ETH: 0.0}, weights={_BTC: 0.5}),
        _decision(_Y2025 + 7 * _DAY, 10_000.0, signals={_BTC: 0.001, _ETH: 0.0}, weights={_BTC: 0.5}),
    )
    report = carry_metrics(_run(spec, fills, [], decisions), inputs)

    row = report.by_year[0]
    assert row["year"] == 2025
    assert row["equity_start"] == pytest.approx(10_000.0)
    assert row["fees"] == 15.0
    assert row["fees_pct_equity"] == pytest.approx(0.0015)
    assert row["funding_dispersion_ann"] == pytest.approx(0.078, rel=1e-12)
    assert report.summary["price_pnl_total"] == 0.0


def test_summary_keys_and_liquidation() -> None:
    """R33 (and R48): the summary carries exactly the contract keys."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    decisions = (_decision(_START, 10_000.0, weights={_BTC: 0.0, _ETH: 0.0}),)
    report = carry_metrics(_run(spec, [], [], decisions), inputs)

    assert set(report.summary) == {
        "weekly_sharpe",
        "weekly_sharpe_lo_se",
        "weeks",
        "daily_sharpe_365",
        "total_return",
        "max_drawdown_daily",
        "n_fills",
        "n_decisions",
        "flat_decisions",
        "fees_total",
        "funding_paid_total",
        "funding_fallback_settlements",
        "funding_estimated",
        "funding_fallback_amount",
        "price_pnl_total",
        "turnover_total",
        "turnover_annual",
        "rounding_drift_mean",
        "rounding_drift_max",
        "band_skips",
        "min_notional_skips",
        "liquidation",
    }
    assert report.summary["liquidation"] == "not_modelled"
    assert report.summary["n_fills"] == 0
    assert report.summary["n_decisions"] == 1
    assert report.summary["flat_decisions"] == 1
    assert report.summary["weekly_sharpe"] is None
    assert report.summary["weekly_sharpe_lo_se"] is None
    assert report.summary["weeks"] == 0
    assert report.summary["daily_sharpe_365"] is None
    assert report.summary["total_return"] == 0.0
    assert report.summary["max_drawdown_daily"] == 0.0
    assert report.daily == [(_START + _DAY, 10_000.0, 0.0)]
    assert report.legs == []
    assert report.by_year[0]["year"] == 2025


def test_funding_fallback_accounting() -> None:
    """R48: count / bool / signed amount of the estimated (fallback) settlements."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    clean = carry_metrics(_run(spec, [], []), inputs)
    assert clean.summary["funding_fallback_settlements"] == 0
    assert clean.summary["funding_estimated"] is False
    assert clean.summary["funding_fallback_amount"] == 0.0

    fills = [
        _fill(_BTC, _START + _HOUR, 100, price=50000.0, fee=2.5),
        _fill(_BTC, _START + 20 * _HOUR, -100, price=50000.0),
    ]
    payments = [
        FundingPayment(_BTC, _START + 8 * _HOUR, 0.51, True),
        FundingPayment(_BTC, _START + 16 * _HOUR, 1.02, False),
    ]
    decisions = (_decision(_START, 10_000.0, weights={_BTC: 0.5}),)
    dirty = carry_metrics(_run(spec, fills, payments, decisions), inputs)
    assert dirty.summary["funding_fallback_settlements"] == 1
    assert dirty.summary["funding_estimated"] is True
    assert dirty.summary["funding_fallback_amount"] == 0.51
    assert dirty.summary["funding_paid_total"] == pytest.approx(1.53)


def test_funding_while_flat_fails_closed() -> None:
    """A settlement with no position is a corrupt run: the decomposition must not drop it."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    payments = [FundingPayment(_BTC, _START + 8 * _HOUR, 0.51, False)]
    with pytest.raises(CarryExecutionError, match="no position"):
        carry_metrics(_run(spec, [], payments), inputs)


def _path_run(prices: dict[int, float], days: int) -> CarryReport:
    """A run holding 0.1 BTC (bought at 100000 after ``start``), so ``E(B_i) == 0.1 * p(B_i)``.

    Every boundary in ``prices`` (keyed by day index) carries the close of that day;
    the intermediate days of a week reuse the Monday close.
    """
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + days * _DAY)
    fills = [_fill(_BTC, _START + _HOUR, 100, price=100000.0)]
    inputs = _inputs(spec, {_BTC: {_START + i * _DAY: price for i, price in prices.items()}, _ETH: {}})
    decisions = (_decision(_START, 10_000.0, weights={_BTC: 0.5}),)
    return carry_metrics(_run(spec, fills, [], decisions), inputs)


def test_max_drawdown_daily_uses_running_peak() -> None:
    """max_drawdown_daily is the trough against the RUNNING peak, not the last or final value.

    Path: peak 11000 (B1), trough 9000 (B2), partial recovery to 10800 (B3) and 10400
    (B4). The answer is ``1 - 9000/11000 == 0.1818...``; a dropped running-peak update
    (peak = equity) collapses it to 0.0 and a returns-based drawdown to another number.
    """
    report = _path_run({1: 110000.0, 2: 90000.0, 3: 108000.0, 4: 104000.0}, 4)

    assert [row[1] for row in report.daily] == [11000.0, 9000.0, 10800.0, 10400.0]
    assert report.summary["max_drawdown_daily"] == 1.0 - 9000.0 / 11000.0
    assert report.summary["max_drawdown_daily"] == 0.18181818181818177
    assert report.summary["max_drawdown_daily"] > 1.0 - 10800.0 / 11000.0


def test_short_cover_attribution() -> None:
    """A fill that REDUCES a short (partial cover or full close) belongs to the short leg.

    Short 10 units at 110, cover 6 at 105, close 4 at 100: all three cash terms sum to
    ``+70.0`` and all three fees to ``1.065`` on the short leg; there is no long leg.
    """
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + _DAY)
    fills = [
        _fill(_BTC, _START + _HOUR, -10_000, price=110.0, fee=0.55),
        _fill(_BTC, _START + 2 * _HOUR, 6_000, price=105.0, fee=0.315),
        _fill(_BTC, _START + 3 * _HOUR, 4_000, price=100.0, fee=0.2),
    ]
    decisions = (_decision(_START, 10_000.0, weights={_BTC: -0.5}),)
    report = carry_metrics(_run(spec, fills, [], decisions), _inputs(spec, {_BTC: {}, _ETH: {}}))

    assert len(report.legs) == 1
    short = report.legs[0]
    assert (short["symbol"], short["side"]) == (_BTC, "short")
    assert short["price_pnl"] == 70.0
    assert short["fees"] == 1.065
    identity = report.summary["price_pnl_total"] - report.summary["funding_paid_total"] - report.summary["fees_total"]
    assert identity == pytest.approx(report.daily[-1][1] - 10_000.0, abs=1e-9 * 10_000.0)


def test_by_year_weekly_sharpe_uses_weeks_in_the_year() -> None:
    """Per-year weekly Sharpe uses only weeks whose Monday boundary falls in that year.

    ``_START + 343 d`` is Monday 2025-12-15 and the run ends Monday 2026-01-12, so the
    four week-starts split 3 (2025) + 1 (2026): 2025 gets its own Sharpe, 2026 is below
    the 3-week minimum (null), and neither equals the pooled 4-week value.
    """
    start = _START + 343 * _DAY
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=start, end_ns=start + 28 * _DAY)
    prices = [100000.0, 110000.0, 105000.0, 115000.0, 112000.0]
    inputs = _inputs(spec, {_BTC: {start + i * _DAY: prices[i // 7] for i in range(1, 29)}, _ETH: {}})
    fills = [_fill(_BTC, start + _HOUR, 100, price=100000.0)]
    decisions = (_decision(start, 10_000.0, weights={_BTC: 0.5}),)
    report = carry_metrics(_run(spec, fills, [], decisions), inputs)

    assert report.summary["weeks"] == 4
    assert [row["year"] for row in report.by_year] == [2025, 2026]
    assert report.by_year[0]["weekly_sharpe"] == _weekly_sharpe([row[1] for row in report.weekly][:3])
    assert report.by_year[0]["weekly_sharpe"] == 4.356777141448447
    assert report.by_year[1]["weekly_sharpe"] is None
    assert report.summary["weekly_sharpe"] == 2.8795432666091827


def test_summary_sharpe_and_lo_se_wiring() -> None:
    """R23 through equity: the summary carries the design Sharpe and Lo SE exactly."""
    prices = {i: [100000.0, 101000.0, 100495.0, 101499.95][i // 7] for i in range(1, 22)}
    report = _path_run(prices, 21)

    assert [row[1] for row in report.weekly] == [
        0.010000000000000009,
        -0.0050000000000000044,
        0.010000000000000009,
    ]
    assert report.summary["weeks"] == 3
    assert report.summary["weekly_sharpe"] == 4.163331998932265
    assert report.summary["weekly_sharpe_lo_se"] == 12.944325225965066
    assert report.summary["weekly_sharpe_lo_se"] == _lo_se(report.summary["weekly_sharpe"], report.summary["weeks"])
    assert report.by_year[0]["weekly_sharpe"] == 4.163331998932265


def test_daily_sharpe_365_value_and_minimum() -> None:
    """The informational daily Sharpe is sqrt(365)-scaled and null below 3 returns (R50b)."""
    report = _path_run({1: 110000.0, 2: 90000.0, 3: 108000.0, 4: 104000.0}, 4)
    returns = [row[2] for row in report.daily]
    mean = math.fsum(returns) / len(returns)
    stdev = math.sqrt(math.fsum((value - mean) ** 2 for value in returns) / (len(returns) - 1))

    assert report.summary["daily_sharpe_365"] == mean / stdev * math.sqrt(365.0)
    assert report.summary["daily_sharpe_365"] == 2.333113338097617
    assert _path_run({1: 110000.0, 2: 90000.0, 3: 108000.0}, 3).summary["daily_sharpe_365"] == 3.8012523980739235
    assert _path_run({1: 110000.0, 2: 90000.0}, 2).summary["daily_sharpe_365"] is None


def test_dispersion_skips_decisions_with_fewer_than_two_rankable() -> None:
    """R27c: only decisions with >= 2 rankable signals contribute a spread to the mean."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_Y2025, end_ns=_Y2026)
    fills = [
        _fill(_BTC, _Y2025 + _HOUR, 100, price=100.0, fee=7.5),
        _fill(_BTC, _Y2025 + 2 * _HOUR, -100, price=100.0, fee=7.5),
    ]
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    one_rankable = (
        _decision(_Y2025, 10_000.0, signals={_BTC: 0.002, _ETH: None}, weights={_BTC: 0.5}),
        _decision(_Y2025 + 7 * _DAY, 10_000.0, signals={_BTC: 0.001, _ETH: 0.003}, weights={_BTC: 0.5}),
    )
    row = carry_metrics(_run(spec, fills, [], one_rankable), inputs).by_year[0]

    assert row["fees"] == 15.0
    assert row["fees_pct_equity"] == pytest.approx(0.0015)
    assert row["funding_dispersion_ann"] == pytest.approx(0.002 * 52.0, rel=1e-12)

    none_rankable = (
        _decision(_Y2025, 10_000.0, signals={_BTC: 0.002, _ETH: None}, weights={_BTC: 0.5}),
        _decision(_Y2025 + 7 * _DAY, 10_000.0, signals={_BTC: 0.001, _ETH: None}, weights={_BTC: 0.5}),
    )
    assert carry_metrics(_run(spec, fills, [], none_rankable), inputs).by_year[0]["funding_dispersion_ann"] is None


def test_rounding_drift_mean_and_max() -> None:
    """``rounding_drift_max`` is the maximum over decisions (not the mean, not the last)."""
    spec = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_START + 3 * _DAY)
    inputs = _inputs(spec, {_BTC: {}, _ETH: {}})
    drifts = (0.002, 0.005, 0.001)
    decisions = tuple(
        _decision(_START + index * _DAY, 10_000.0, weights={_BTC: 0.0, _ETH: 0.0}, rounding_drift=drift)
        for index, drift in enumerate(drifts)
    )
    summary = carry_metrics(_run(spec, [], [], decisions), inputs).summary

    assert summary["rounding_drift_max"] == 0.005
    assert summary["rounding_drift_max"] == max(drifts)
    assert summary["rounding_drift_mean"] == pytest.approx(math.fsum(drifts) / 3)
    assert summary["rounding_drift_mean"] < summary["rounding_drift_max"]
    assert summary["flat_decisions"] == 3


def test_sharpe_collapses_on_float_noise_floor() -> None:
    """R50: near-equal returns through equity give a non-zero but ~1e-16 stdev -> null Sharpe.

    The four weekly returns differ in the last bits (so ``stdev != 0.0``) but their
    stdev is 1.81e-16, far below ``1e-12 * max(1, |mean|)``: without the floor the
    summary would carry a bogus 3.98e14.
    """
    prices = {i: 100000.0 * (1.01 ** (i // 7)) for i in range(1, 29)}
    report = _path_run(prices, 28)
    returns = [row[1] for row in report.weekly]

    assert len(returns) == 4
    assert len(set(returns)) > 1
    assert report.summary["weekly_sharpe"] is None
    assert report.summary["weekly_sharpe_lo_se"] is None
    assert report.by_year[0]["weekly_sharpe"] is None
