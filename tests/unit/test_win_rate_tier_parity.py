"""One win-rate definition for screening and validation (bd vibe-quant-neuvu.4).

Win rate = share of a run's trades whose net PnL (after fees, funding and modeled
slippage) is above zero; trades still open at the end count at their mark. Both
tiers report it through :func:`vibe_quant.metrics.trade_win_rate` (user ruling
Q1=A, 2026-10-21). Previously screening reported NT's ``WinRate`` stat -- realized
PnL before funding, an open position counted as a loser at -entry fee -- while
validation already counted ``net_pnl > 0`` over its trades.
"""

from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.fake_engine import FakePosition, fake_engine
from tests.unit.test_audit_metrics_funding import _archive
from vibe_quant.metrics import trade_win_rate
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.extraction import extract_results
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from pathlib import Path

_HOUR_NS = 3_600 * 1_000_000_000
_BASE_NS = 1_735_689_600_000 * 1_000_000  # 2025-01-01T00:00:00Z
_BTC = "BTCUSDT-PERP.BINANCE"


@pytest.fixture(autouse=True)
def _fresh_funding_cache() -> None:
    clear_rate_cache()


def _closed(
    realized: float,
    *,
    opened_hour: float = 7.0,
    closed_hour: float = 9.0,
    qty: float = 0.5,
    px: float = 10_000.0,
    entry_comm: float = 0.0,
) -> FakePosition:
    """Closed-position stub: the fields both extractors read."""
    return FakePosition(
        id="P-CLOSED",
        instrument_id=_BTC,
        ts_opened=_BASE_NS + int(opened_hour * _HOUR_NS),
        ts_closed=_BASE_NS + int(closed_hour * _HOUR_NS),
        events=[],
        realized_pnl=realized,
        avg_px_open=px,
        avg_px_close=px,
        peak_qty=qty,
        commission_list=[entry_comm],
    )


def _open() -> FakePosition:
    """Open-at-end stub: entry commission 0.20, marked at 5000 (+5.00 unrealized)."""
    return FakePosition.open(
        "P-OPEN",
        _BTC,
        entry_ns=_BASE_NS + 1 * _HOUR_NS,
        entry_price=4_950.0,
        qty=0.1,
        entry_fee=0.20,
    )


class _Bar:
    __slots__ = ("close", "high", "low", "open", "ts_init")

    def __init__(self, ts_init: int, price: float) -> None:
        self.ts_init = ts_init
        self.open = self.high = self.low = self.close = price


def _engine(
    positions: list[FakePosition], marks: list[tuple[int, float]] | None = None
) -> SimpleNamespace:
    bars = [_Bar(ts, price) for ts, price in (marks or [])]
    return fake_engine(positions, bars_by_type={f"{_BTC}-1-MINUTE-LAST-EXTERNAL": bars})


def _bt_result(n_positions: int, nt_win_rate: float) -> SimpleNamespace:
    """NT's own stats: the stub's win rate is the value the shared definition replaces."""
    return SimpleNamespace(
        total_positions=n_positions,
        elapsed_time=0.0,
        stats_pnls={"USDT": {"PnL% (total)": 1.0, "Win Rate": nt_win_rate}},
        stats_returns={"Profit Factor": 3.335, "Sharpe Ratio (252 days)": 1.0},
    )


def _venue_config() -> SimpleNamespace:
    # prob_slippage > 0 disables post-fill SPEC slippage so net == realized - funding
    return SimpleNamespace(
        default_leverage=Decimal("10"),
        starting_balance_usdt=100_000,
        fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
    )


def _screening_runner(
    tmp_path: Path, *, archive: object | None = None
) -> NTScreeningRunner:
    return NTScreeningRunner(
        dsl_dict={},
        symbols=["BTCUSDT"],
        start_date="2025-01-01",
        end_date="2025-01-02",
        funding_archive_path=str(archive) if archive else str(tmp_path / "none.db"),  # type: ignore[arg-type]
    )


def _screening(
    runner: NTScreeningRunner,
    engine: SimpleNamespace,
    n_trades: int,
    nt_win_rate: float,
) -> Any:
    return runner._extract_metrics(
        {},
        _bt_result(n_trades, nt_win_rate),
        engine,
        time.time(),
        starting_balance=1000.0,
    )


# --------------------------------------------------------------------- the helper


def test_trade_win_rate_counts_strictly_positive_net() -> None:
    """R4/R7: strict ``> 0.0``; a trade netting exactly 0.0 is not a win."""
    assert trade_win_rate([1.0, -0.5, 0.0]) == 1 / 3
    assert trade_win_rate([]) == 0.0
    assert trade_win_rate([0.0]) == 0.0


def test_trade_win_rate_fails_closed_on_non_finite_pnl() -> None:
    """A non-finite net counts in the denominator and is never a win (unlike PF)."""
    assert trade_win_rate([float("nan"), 2.0]) == 0.5
    assert trade_win_rate([float("inf")]) == 0.0  # +inf is never a win either
    assert trade_win_rate([float("-inf"), 1.0]) == 0.5


# -------------------------------------------------------------------- both tiers


def test_funding_turns_realized_winner_into_loss_both_tiers(
    tmp_path: Path,
) -> None:
    """R5: realized +1.00 (after commissions) pays a 1.50 funding settlement -> net -0.50."""
    archive = _archive(tmp_path / "funding.db", list(range(12)))  # type: ignore[attr-defined]
    pos = _closed(1.0, entry_comm=0.1)
    engine = _engine([pos])
    screening = _screening(_screening_runner(tmp_path, archive=archive), engine, 1, 1.0)
    validation = extract_results(
        1,
        "s",
        _bt_result(1, 1.0),
        engine,
        _venue_config(),
        funding_calculator=FundingCalculator(archive),
        run_start_date="2025-01-01",
        run_end_date="2025-01-02",
    )

    # one 08:00 settlement on the 5000 entry notional: 0.0003 * 5000 = 1.50
    assert screening.total_funding == pytest.approx(1.5)
    assert validation.total_funding == pytest.approx(1.5)
    assert validation.trades[0].net_pnl == pytest.approx(-0.5)
    assert screening.win_rate == trade_win_rate([-0.5]) == 0.0
    assert validation.win_rate == 0.0
    assert screening.win_rate != 1.0  # NT's stub reported the pre-funding winner as a win


def test_open_mark_winner_counts_in_both_tiers(tmp_path: Path) -> None:
    """R6: entry comm 0.20, mark unrealized +5.00, exit fee 0.25 -> net +4.55 -> a win."""
    mark_ns = _BASE_NS + 23 * _HOUR_NS
    engine = _engine([_open()], [(mark_ns, 5_000.0)])
    runner = _screening_runner(tmp_path)
    runner._all_timeframes = {"1m"}  # type: ignore[attr-defined]
    screening = _screening(runner, engine, 1, 0.0)
    validation = extract_results(
        1,
        "s",
        _bt_result(1, 0.0),
        engine,
        _venue_config(),
        run_start_date="2025-01-01",
        run_end_date="2025-01-02",
        execution_timeframe="1m",
    )

    assert validation.trades[0].net_pnl == pytest.approx(4.55)
    assert screening.win_rate == trade_win_rate([4.55]) == 1.0
    assert validation.win_rate == 1.0
    assert screening.win_rate != 0.0  # NT's stub said 0.0 (an open position was a loser)


def test_zero_trades_win_rate_zero_both_tiers(tmp_path: Path) -> None:
    """R8: no trades -> 0.0 in both tiers, not NT's stat."""
    engine = _engine([])
    screening = _screening(_screening_runner(tmp_path), engine, 0, 0.9)
    validation = extract_results(1, "s", _bt_result(0, 0.9), engine, _venue_config())

    assert (screening.total_trades, validation.total_trades) == (0, 0)
    assert screening.win_rate == 0.0
    assert validation.win_rate == 0.0
    assert validation.win_rate != 0.9


def test_two_tiers_agree_on_mixed_closed_trades(tmp_path: Path) -> None:
    """Both tiers forward the same per-trade nets: [10, -5, 0, 7] -> 2/4 = 0.5."""
    pnls = [10.0, -5.0, 0.0, 7.0]
    # 09:00 -> 10:00 never crosses an 08:00/16:00 settlement: net == realized
    engine = _engine([_closed(p, opened_hour=9, closed_hour=10) for p in pnls])
    screening = _screening(_screening_runner(tmp_path), engine, len(pnls), 0.9)
    validation = extract_results(
        1,
        "s",
        _bt_result(len(pnls), 0.9),
        engine,
        _venue_config(),
        run_start_date="2025-01-01",
        run_end_date="2025-01-02",
    )

    assert trade_win_rate(pnls) == 0.5
    assert screening.win_rate == validation.win_rate == 0.5
    assert screening.win_rate == trade_win_rate(t.net_pnl for t in validation.trades)
