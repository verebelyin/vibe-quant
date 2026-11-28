"""CostEdge on real screening runs: what ``metrics.cost_edge`` reports and that it means it.

A maker-authored 4h EMA-cross strategy (not 239) on the real BTCUSDT catalog. Run A is VIP0;
run B charges A's break-even fee on both sides; run C a test tier at half the VIP0 rates (a
stand-in for a cheaper VIP tier). Screening sizes on the costed balance, so a fee change
moves every later notional: "net ~ 0 at the break-even fee" holds only approximately
(design arch-fees-1 § Real engine). Skipped when the catalog is absent.
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.fees import BINANCE_USDTM_TIERS, FeeSchedule
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.screening import nt_runner
from vibe_quant.screening.nt_runner import NTScreeningRunner

if TYPE_CHECKING:
    from vibe_quant.metrics import CostEdge
    from vibe_quant.screening.types import BacktestMetrics

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
needs_catalog = pytest.mark.skipif(
    not (CATALOG / "data" / "bar" / "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL").is_dir(),
    reason="needs the BTCUSDT 4h catalog",
)
pytestmark = [
    needs_catalog,
    pytest.mark.slow(reason="real NT screening runs on the BTCUSDT 4h catalog"),
]

START, END = "2023-01-01", "2024-01-01"
DSL: dict[str, Any] = {
    "name": "cost_edge_ema_cross",
    "timeframe": "4h",
    "indicators": {
        "fast": {"type": "EMA", "period": 12},
        "slow": {"type": "EMA", "period": 48},
    },
    "entry_conditions": {"long": ["fast crosses_above slow"]},
    "exit_conditions": {"long": ["fast crosses_below slow"]},
    "stop_loss": {"type": "fixed_pct", "percent": 3.0},
    "take_profit": {"type": "fixed_pct", "percent": 6.0},
}


class _Run:
    """One screening run plus the trade inputs its ledger priced (spied, not recomputed)."""

    def __init__(self, metrics: BacktestMetrics, balance: float, inputs: list[Any]) -> None:
        self.metrics = metrics
        self.balance = balance
        self.inputs = inputs

    @property
    def ce(self) -> CostEdge:
        assert self.metrics.cost_edge is not None
        return self.metrics.cost_edge


def _run(tier: str, monkeypatch: pytest.MonkeyPatch) -> _Run:
    captured: list[Any] = []
    original = nt_runner.cost_positions

    def spy(trade_inputs: Any, *args: Any, **kwargs: Any) -> Any:
        items = list(trade_inputs)
        captured[:] = items
        return original(items, *args, **kwargs)

    monkeypatch.setattr(nt_runner, "cost_positions", spy)
    runner = NTScreeningRunner(
        DSL,
        ["BTCUSDT"],
        START,
        END,
        catalog_path=str(CATALOG),
        funding_archive_path=str(ARCHIVE),
        launch=LaunchSettings(fee_schedule=FeeSchedule(tier=tier)),
    )
    metrics = runner({})
    return _Run(metrics, runner.last_starting_balance, captured)


def _independent_turnover(inputs: list[Any]) -> float:
    """Per position over ``pos.events`` (+ the run-end mark leg), then over positions."""
    total = 0.0
    for pos, mark in inputs:
        t = 0.0
        for fill in pos.events:
            t += abs(float(fill.last_px) * float(fill.last_qty))
        if mark is not None:
            t += abs(float(pos.signed_qty)) * mark.price
        total += t
    return total


@pytest.fixture(scope="module")
def run_a() -> _Run:
    with pytest.MonkeyPatch.context() as mp:
        return _run("VIP0", mp)


def test_screening_cost_edge_commissions_equal_total_fees(run_a: _Run) -> None:
    """R10: mode off, the cost edge is the run's own fees, net and fill notional.

    The turnover is recomputed independently from the trade inputs the ledger priced (no
    literal pin: a catalog refresh moves it). MEASURED (VIP0, 2023): 23 trades, turnover
    22993.3541, gross edge == break-even 14.813563715144104 bps, cost paid 4.589265225989801.
    """
    m, ce = run_a.metrics, run_a.ce
    assert m.total_trades > 0
    assert ce.commissions == m.total_fees
    assert ce.slippage == 0.0
    assert ce.net_pnl == m.total_return * run_a.balance
    assert ce.turnover == _independent_turnover(run_a.inputs)
    assert ce.breakeven_fee_bps == ce.gross_edge_bps


def test_breakeven_fee_run_nets_within_tenth_of_edge(
    run_a: _Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R14: charging A's break-even fee on both sides nets close to zero per turnover.

    MEASURED: B (fee 14.81 bps per side) 23 trades, net/turnover 0.6453736470826411 bps
    <= 1.4813563715144104 (a tenth of A's edge): the costed balance moves later notionals.
    """
    a = run_a.ce
    assert a.gross_edge_bps is not None and a.gross_edge_bps > 0  # precondition
    assert a.breakeven_fee_bps is not None
    be = Decimal(repr(a.breakeven_fee_bps / 1e4))
    monkeypatch.setitem(BINANCE_USDTM_TIERS, "VIPBE", (be, be))
    run_b = _run("VIPBE", monkeypatch)
    b = run_b.ce

    assert run_b.metrics.total_trades == run_a.metrics.total_trades
    assert abs(b.net_pnl / b.turnover * 1e4) <= 0.10 * a.gross_edge_bps


def test_cheaper_tier_keeps_breakeven_within_tenth(
    run_a: _Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R22: half the VIP0 rates halve the cost paid; the break-even fee barely moves.

    MEASURED: C 23 trades, break-even rel diff 0.028742276184324487, cost-paid ratio
    0.5001140156761438, cost paid 2.295155861172645 < 4.589265225989801 bps, net up.
    """
    monkeypatch.setitem(BINANCE_USDTM_TIERS, "VIPT", (Decimal("0.0001"), Decimal("0.00025")))
    a, c = run_a.ce, _run("VIPT", monkeypatch).ce
    assert a.breakeven_fee_bps is not None and c.breakeven_fee_bps is not None
    assert a.cost_paid_bps is not None and c.cost_paid_bps is not None

    assert abs(c.breakeven_fee_bps - a.breakeven_fee_bps) <= 0.10 * a.breakeven_fee_bps
    assert c.cost_paid_bps < a.cost_paid_bps
    assert c.net_pnl > a.net_pnl
    assert abs(c.cost_paid_bps / a.cost_paid_bps - 0.5) <= 0.05
    assert math.isfinite(c.turnover) and c.turnover > 0
