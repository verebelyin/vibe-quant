"""Cost edge per unit of traded notional: ``CostEdge`` and the ledger's turnover.

Turnover is the notional of every fill that paid a tier fee plus the run-end mark leg;
a liquidation hit leg pays clearance, not a tier fee, so it never counts. The literal
expectations are the design contract's (arch-fees-1 R1-R8).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from nautilus_trader.model.enums import OrderSide

from tests.unit.fake_engine import FakeFill, FakePosition
from vibe_quant.metrics import CostEdge, cost_edge_columns
from vibe_quant.trade_ledger import (
    VALIDATION_RULES,
    accumulate,
    cost_positions,
    position_key,
    position_turnover,
    screening_liquidation_trades,
    validation_liquidated_run,
)
from vibe_quant.validation.extraction import OpenPositionMark
from vibe_quant.validation.funding import FundingAccrual

BTC = "BTCUSDT-PERP.BINANCE"
DAY = 86_400_000_000_000


def _three_trades() -> tuple[list[tuple[FakePosition, OpenPositionMark | None]], dict[str, float]]:
    """A long 1@100->110, B short 2@50->55, C long .5@200 open and marked at 220 (taker 5 bps)."""
    a = FakePosition.closed(
        "A", BTC, entry_ns=0, exit_ns=DAY, entry_price=100.0, exit_price=110.0,
        entry_fee=0.05, exit_fee=0.055,
    )  # fmt: skip
    b = FakePosition.closed(
        "B", BTC, entry_ns=DAY, exit_ns=2 * DAY, entry_price=50.0, exit_price=55.0, qty=2.0,
        entry_side=OrderSide.SELL, entry_fee=0.05, exit_fee=0.055,
    )  # fmt: skip
    c = FakePosition.open("C", BTC, entry_ns=2 * DAY, entry_price=200.0, qty=0.5, entry_fee=0.05)
    mark = OpenPositionMark(position=c, ts=3 * DAY, price=220.0, unrealized=10.0, exit_fee=0.055)
    return [(a, None), (b, None), (c, mark)], {"A": 0.01, "B": 0.02, "C": 0.01}


def test_three_trade_ledger_numbers() -> None:
    """R1: turnover 630 = 210 per trade (C's mark leg 110 included); funding stays inside net."""
    inputs, slips = _three_trades()
    (a, _), _, (c, _) = inputs
    funding = {
        position_key(a): FundingAccrual(total=0.2, payments=[(DAY // 2, 0.2)]),
        position_key(c): FundingAccrual(total=-0.1, payments=[(2 * DAY + 1, -0.1)]),
    }
    costed = cost_positions(
        inputs,
        VALIDATION_RULES,
        slippage=SimpleNamespace(leg=lambda pos, _price, _ts: slips[pos.id]),
        funding_by_key=funding,
    )
    assert [item.turnover for item in costed] == [210.0, 210.0, 210.0]
    ledger = accumulate(costed, VALIDATION_RULES)

    edge = CostEdge(
        turnover=ledger.turnover,
        net_pnl=sum(ledger.net_pnls),
        commissions=ledger.total_fees,
        slippage=ledger.total_slippage,
    )

    assert edge.turnover == 630.0
    assert edge.net_pnl == 9.505
    assert edge.commissions == 0.31500000000000006
    assert edge.slippage == 0.08
    assert edge.gross_edge_bps == 157.14285714285717
    assert edge.cost_paid_bps == 6.2698412698412715
    assert edge.breakeven_fee_bps == 155.87301587301587


def test_position_turnover_marked() -> None:
    """R4: the marked position C adds ``abs(signed_qty) * mark.price`` = 110 to its 100 fill."""
    (_, _, (c, mark)), _ = _three_trades()
    assert position_turnover(c) == 100.0
    assert position_turnover(c, mark) == 210.0  # + 0.5 * 220
    short = FakePosition.open(
        "S", BTC, entry_ns=0, entry_price=40.0, qty=3.0, entry_side=OrderSide.SELL
    )
    assert position_turnover(short, SimpleNamespace(price=30.0)) == 210.0  # 120 + |-3| * 30


def _liquidated() -> tuple[FakePosition, SimpleNamespace]:
    """R5: buy 1@100 ts1, sell .5@95 ts3, sell .5@105 ts5; the hit at ts3 sells .5@90."""
    pos = FakePosition(
        id="L",
        instrument_id=BTC,
        ts_opened=1,
        ts_closed=5,
        events=[
            FakeFill(1, OrderSide.BUY, 1.0, 100.0, 0.05),
            FakeFill(3, OrderSide.SELL, 0.5, 95.0, 0.02375),
            FakeFill(5, OrderSide.SELL, 0.5, 105.0, 0.02625),
        ],
        realized_pnl=0.0,
        avg_px_open=100.0,
        avg_px_close=100.0,
        peak_qty=1.0,
        commission_list=[0.05, 0.02375, 0.02625],
    )
    hit = SimpleNamespace(ts=3, qty=0.5, price=90.0, loss=5.0, clearance_fee=0.225)
    return pos, hit


def test_position_turnover_liquidated_counts_fee_paying_fills_only() -> None:
    """R5: 100 + 47.5 = 147.5; the fill at ts == hit.ts is kept, ts5 is not, the 45.0 hit
    leg never counts (and a hit beats a mark). Both tiers' run sums carry exactly that."""
    pos, hit = _liquidated()
    assert position_turnover(pos, hit=hit) == 147.5
    assert position_turnover(pos, SimpleNamespace(price=1e6), hit=hit) == 147.5
    assert position_turnover(pos, hit=SimpleNamespace(ts=2)) == 100.0
    assert position_turnover(pos) == 200.0  # every fill: 100 + 47.5 + 52.5

    (_, _, (c, mark)), _ = _three_trades()
    dropped = FakePosition.open("D", BTC, entry_ns=1, entry_price=70.0)
    key = position_key(pos)
    overlay = SimpleNamespace(
        dropped_keys=frozenset({position_key(dropped)}),
        hits_by_key={key: hit},
        exit_override={key: (3, 90.0)},
        ledgers={},
    )
    screening = screening_liquidation_trades(
        overlay,  # type: ignore[arg-type]
        [(pos, None), (dropped, None), (c, mark)],
        {key: FundingAccrual(), position_key(c): FundingAccrual()},
    )
    assert screening.turnover == 357.5  # 147.5 + 0 (dropped) + 210 (marked)

    survivor = cost_positions([(c, mark)], VALIDATION_RULES)
    adjusted, _ = validation_liquidated_run(
        survivor,
        [c, pos, dropped],
        overlay,
        slip=SimpleNamespace(leg=lambda _pos, _price, _ts: 0.0),  # type: ignore[arg-type]
        funding_by_key={},
    )
    assert [item.turnover for item in adjusted] == [210.0, 147.5]
    assert accumulate(adjusted, VALIDATION_RULES).turnover == 357.5


def test_cost_edge_zero_turnover() -> None:
    """R6: no fills -> every bps is None (never a ZeroDivisionError); columns keep the order."""
    edge = CostEdge(0.0, 0.0, 0.0, 0.0)
    assert edge.gross_edge_bps is None
    assert edge.cost_paid_bps is None
    assert edge.breakeven_fee_bps is None
    expected = {
        "turnover": 0.0,
        "gross_edge_bps": None,
        "cost_paid_bps": None,
        "breakeven_fee_bps": None,
    }
    assert edge.to_notes() == expected
    assert list(edge.to_notes()) == list(expected)
    assert cost_edge_columns(edge) == expected
    assert cost_edge_columns(None) == dict.fromkeys(expected)
    assert list(cost_edge_columns(None)) == list(expected)
    assert CostEdge(-1.0, 1.0, 1.0, 0.0).gross_edge_bps is None
    # the edge just above zero turnover divides
    assert CostEdge(1.0, 2.0, 1.0, 0.5).to_notes() == {
        "turnover": 1.0,
        "gross_edge_bps": 35000.0,
        "cost_paid_bps": 15000.0,
        "breakeven_fee_bps": 30000.0,
    }


@pytest.mark.parametrize("field", ["turnover", "net_pnl", "commissions", "slippage"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_cost_edge_non_finite_input_is_none(field: str, bad: float) -> None:
    """R7: a non-finite input makes every bps None."""
    values = {"turnover": 100.0, "net_pnl": 1.0, "commissions": 0.5, "slippage": 0.25}
    edge = CostEdge(**{**values, field: bad})
    assert (edge.gross_edge_bps, edge.cost_paid_bps, edge.breakeven_fee_bps) == (None, None, None)


def test_cost_edge_non_finite_result_is_none() -> None:
    """R7: finite inputs whose ratio overflows report None, not inf."""
    edge = CostEdge(1e-300, 1e300, 0.0, 0.0)
    assert edge.gross_edge_bps is None
    assert edge.breakeven_fee_bps is None
    assert edge.cost_paid_bps == 0.0


def test_cost_edge_total() -> None:
    """R8: field sums in item order; any None (or nothing) -> None."""
    a = CostEdge(0.1, 0.2, 0.3, 0.0)
    b = CostEdge(0.2, -0.1, 0.4, 0.05)
    assert CostEdge.total([a, b]) == CostEdge(0.30000000000000004, 0.1, 0.7, 0.05)
    assert CostEdge.total([a, None]) is None
    assert CostEdge.total([None, a]) is None
    assert CostEdge.total([]) is None
    assert CostEdge.total([CostEdge(0.0, 0.0, 0.0, 0.0), b]) == b
