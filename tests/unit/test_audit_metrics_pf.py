"""Trade-based profit factor shared by screening + validation (bd vibe-quant-e70tl.7).

Previously both tiers reported NT's *daily-return* PF (NT's realized-PnL PF is
unimplemented and returns None), e.g. a trade PF of 1.345 was reported as 3.335.
"""

from __future__ import annotations

import math
import time
from decimal import Decimal
from types import SimpleNamespace

import pytest

from vibe_quant.metrics import PROFIT_FACTOR_CAP, profit_factor
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.extraction import extract_results, extract_trades
from vibe_quant.validation.results import ValidationResult

_DAY_NS = 86_400 * 1_000_000_000
_BASE_NS = 1_735_689_600_000 * 1_000_000  # 2025-01-01T00:00Z


class TestProfitFactor:
    def test_same_day_trades_pf_is_trade_based(self) -> None:
        """[+10, -5, +10, -5] on one day -> PF 20/10 = 2.0 (daily PF would be undefined)."""
        assert profit_factor([10.0, -5.0, 10.0, -5.0]) == 2.0

    def test_no_losing_trades_is_capped_not_inf_or_nan(self) -> None:
        pf = profit_factor([3.0, 4.0])
        assert pf == PROFIT_FACTOR_CAP
        assert math.isfinite(pf)

    def test_no_trades_is_zero(self) -> None:
        assert profit_factor([]) == 0.0

    def test_only_losers_is_zero(self) -> None:
        assert profit_factor([-1.0, -2.0]) == 0.0

    def test_breakeven_trades_ignored(self) -> None:
        assert profit_factor([0.0, 6.0, -3.0, 0.0]) == 2.0

    def test_huge_pf_clamped_to_cap(self) -> None:
        assert profit_factor([1_000_000.0, -1.0]) == PROFIT_FACTOR_CAP

    def test_non_finite_pnls_ignored(self) -> None:
        assert profit_factor([float("nan"), 4.0, -2.0]) == 2.0


# ---------------------------------------------------------------------------
# Screening and validation agree for the same closed-trade list
# ---------------------------------------------------------------------------


class _Position:
    """Closed-position stub exposing what both extractors read."""

    is_closed = True
    is_open = False
    instrument_id = "BTCUSDT-PERP.BINANCE"
    entry = "BUY"
    side = "LONG"
    avg_px_open = 40_000.0
    avg_px_close = 40_100.0
    peak_qty = 0.1
    events: list[object] = []

    def __init__(self, realized_pnl: float, idx: int) -> None:
        self.realized_pnl = realized_pnl
        self.ts_opened = _BASE_NS + idx * 3_600_000_000_000
        self.ts_closed = self.ts_opened + 600_000_000_000  # 10 minutes later, same day

    def commissions(self) -> list[float]:
        return [1.0]


def _fake_engine(pnls: list[float]) -> SimpleNamespace:
    positions = [_Position(p, i) for i, p in enumerate(pnls)]
    cache = SimpleNamespace(
        positions=lambda: positions,
        position_snapshots=list,
        bars=list,
        accounts=list,
    )
    return SimpleNamespace(kernel=SimpleNamespace(cache=cache))


def _bt_result(n_positions: int, daily_pf: float = 3.335) -> SimpleNamespace:
    """BacktestResult stub whose NT stats carry a (wrong) daily-return PF."""
    return SimpleNamespace(
        total_positions=n_positions,
        elapsed_time=0.0,
        stats_pnls={"USDT": {"PnL% (total)": 1.0, "Win Rate": 0.5}},
        stats_returns={"Profit Factor": daily_pf, "Sharpe Ratio (252 days)": 1.0},
    )


def _venue_config() -> SimpleNamespace:
    # prob_slippage > 0 disables post-fill SPEC slippage so net == realized
    return SimpleNamespace(
        default_leverage=Decimal("10"),
        starting_balance_usdt=100_000,
        fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
    )


PNLS = [10.0, -5.0, 10.0, -5.0]


def test_screening_pf_from_trades_not_daily_returns() -> None:
    runner = NTScreeningRunner(dsl_dict={}, symbols=["BTCUSDT"], start_date="", end_date="")
    metrics = runner._extract_metrics(
        {}, _bt_result(len(PNLS)), _fake_engine(PNLS), time.time(), starting_balance=1000.0
    )
    assert metrics.profit_factor == 2.0


def test_validation_pf_from_trades_not_daily_returns() -> None:
    result = extract_results(
        1, "s", _bt_result(len(PNLS)), _fake_engine(PNLS), _venue_config()
    )
    assert result.profit_factor == 2.0


def test_screening_and_validation_pf_agree() -> None:
    runner = NTScreeningRunner(dsl_dict={}, symbols=["BTCUSDT"], start_date="", end_date="")
    pnls = [12.5, -3.0, 7.0, -9.25, 4.0]
    screening = runner._extract_metrics(
        {}, _bt_result(len(pnls)), _fake_engine(pnls), time.time(), starting_balance=1000.0
    )
    validation = extract_results(
        1, "s", _bt_result(len(pnls)), _fake_engine(pnls), _venue_config()
    )
    assert screening.profit_factor == pytest.approx(23.5 / 12.25)
    assert validation.profit_factor == screening.profit_factor


def test_validation_no_trades_pf_zero() -> None:
    result = extract_results(1, "s", _bt_result(0), _fake_engine([]), _venue_config())
    assert result.profit_factor == 0.0


def test_screening_no_trades_pf_zero() -> None:
    runner = NTScreeningRunner(dsl_dict={}, symbols=["BTCUSDT"], start_date="", end_date="")
    metrics = runner._extract_metrics(
        {}, _bt_result(0), _fake_engine([]), time.time(), starting_balance=1000.0
    )
    assert metrics.profit_factor == 0.0


def test_extract_trades_pf_uses_net_pnl_after_costs() -> None:
    """A winner pushed negative by modeled costs counts as a loss in PF."""
    result = ValidationResult(starting_balance=100_000.0)
    engine = _fake_engine([10.0, -5.0])
    venue = _venue_config()
    venue.fill_config = SimpleNamespace(impact_coefficient=0.1, prob_slippage=0.0)
    extract_trades(result, engine, venue)
    expected = profit_factor([t.net_pnl for t in result.trades])
    assert result.profit_factor == expected
    assert result.profit_factor < 2.0  # slippage shrinks the winner, grows the loser
