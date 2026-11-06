"""The modeled-cost ledger a strategy sizes against (design-costs.md § 2/§ 6, T1).

Contract rows R1-R9 and R12-R15. ``CostLedger`` finalizes a position's modeled costs (post-hoc
funding to the SAME refined exit the extraction uses, plus the two slippage legs) once, when NT
closes it, and answers ``paid_through(now)`` for the open ones. Nothing installs the ledger in a
run yet (T2), so no published number moves here.

The toy is the design's: 1000 USDT, fees 0, archived 8h funding at +0.0001, price 100. The
archive and the 1m LAST catalog are REAL (temp files), so ``FundingCalculator`` and
``load_bar_windows`` run for real; positions and cache orders are duck-typed stubs.
"""

from __future__ import annotations

import dataclasses
import math
import pickle
import subprocess
import sys
from pathlib import Path  # noqa: TCH003
from types import SimpleNamespace
from typing import Any

import pytest
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import ClientOrderId

from tests.unit.fake_engine import FakePosition
from tests.unit.test_screening_funding_exit_timing import every, interval_archive
from tests.unit.test_screening_liquidation import BTC, Bars, at
from tests.unit.test_screening_liquidation_exit_timing import (
    Order,
    OrderAccepted,
    write_last_bars,
)
from vibe_quant.errors import DataUnavailableError
from vibe_quant.liquidation.exit_timing import MissingExitBars

# The module under test is imported lazily in ``_module()`` so the RED run of this file reports
# one missing-name error per contract row instead of a collection error (see the handoff).
from vibe_quant.trade_ledger import (
    SCREENING_RULES,
    PositionCosts,
    accrue_position_funding,
    cost_positions,
    position_key,
)
from vibe_quant.validation.fill_model import SlippageEstimator
from vibe_quant.validation.funding import FundingAccrual, FundingCalculator

E1D = 86_400_000_000_000
E4H = 4 * 3_600 * 1_000_000_000
MIN = 60_000_000_000
ONE_MS = 1_000_000
ETH = "ETHUSDT-PERP.BINANCE"
SL_LEVEL = 95.0
DAY_MIN = 1440


def _module() -> Any:
    """The module under test (missing on the RED run)."""
    from vibe_quant import modeled_costs

    return modeled_costs


def model(
    archive: Path,
    *,
    refine: frozenset[str] = frozenset(),
    execution_tf: str | None = "1d",
    catalog: Path | None = None,
    slippage: Any | None = None,
) -> Any:
    """A ``CostModel`` over ``archive`` (the design's screening/validation shapes)."""
    bar_ns = {None: 0, "1d": E1D, "4h": E4H}[execution_tf]
    return _module().CostModel(
        funding=FundingCalculator(str(archive)),
        refine=refine,
        execution_tf=execution_tf,
        execution_bar_ns=bar_ns,
        catalog_path=str(catalog) if catalog is not None else None,
        slippage=slippage,
    )


def ledger_for(archive: Path, orders: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """A ``CostLedger`` over a cache that serves exactly ``orders`` (all refinement reads)."""
    cache = SimpleNamespace(order=lambda cid: (orders or {}).get(str(cid)))
    return _module().CostLedger(model(archive, **kwargs), cache)


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    """8h settlements at +0.0001 for BTC and ETH (the toy rate), 24 h around the window."""
    return interval_archive(
        tmp_path / "funding.db",
        {"BTCUSDT": every(8), "ETHUSDT": every(8)},
        rate=0.0001,
    )


def closed(
    *,
    pid: str = "A",
    instrument: str = BTC,
    entry_ns: int,
    exit_ns: int,
    entry_px: float = 100.0,
    exit_px: float = 110.0,
    qty: float = 10.0,
    entry_side: Any = OrderSide.BUY,
) -> FakePosition:
    """A closed round trip: NT's realized PnL is the price PnL (fees 0 in the toy)."""
    return FakePosition.closed(
        pid,
        instrument,
        entry_ns=entry_ns,
        exit_ns=exit_ns,
        entry_price=entry_px,
        exit_price=exit_px,
        qty=qty,
        entry_side=entry_side,
    )


def open_long(
    *,
    pid: str = "O",
    instrument: str = ETH,
    entry_ns: int,
    px: float = 100.0,
    qty: float = 5.0,
) -> FakePosition:
    """An NT position still open at ``now`` (notional ``px * qty``)."""
    from tests.unit.test_screening_liquidation_exit_timing import Fill

    return FakePosition(
        id=pid,
        instrument_id=instrument,
        ts_opened=entry_ns,
        ts_closed=0,
        events=[Fill(entry_ns, OrderSide.BUY, qty, px)],
        realized_pnl=0.0,
        is_closed=False,
        entry=OrderSide.BUY,
        avg_px_open=px,
        peak_qty=qty,
        signed_qty=qty,
    )


def toy_bars(overrides: dict[int, tuple[float, float, float, float]] | None = None) -> Bars:
    """The design's price-100 toy as 1m bars: low 99.9 / high 100.1 unless overridden."""
    n = 3 * DAY_MIN
    o, h, lo, c = ([v] * n for v in (100.0, 100.1, 99.9, 100.0))
    for i, (bo, bh, bl, bc) in (overrides or {}).items():
        o[i], h[i], lo[i], c[i] = bo, bh, bl, bc
    return Bars(o, h, lo, c)


def d1_sl_position(*, pid: str = "A", entry_ns: int | None = None) -> FakePosition:
    """10 @ 100 opened day 0, NT-stopped at the day-1 1d boundary (23:59:59.999) at 95.

    The SL order is live from the entry, so the refinement's ``active_from`` gate admits any 1m
    bar of the exit's strategy bar.
    """
    from tests.unit.test_screening_liquidation_exit_timing import Fill

    start = at(0) + 59_999 * ONE_MS if entry_ns is None else entry_ns
    fill = at(DAY_MIN) + DAY_MIN * MIN - ONE_MS  # day 1 23:59:59.999
    events = [
        Fill(start, OrderSide.BUY, 10.0, 100.0, 0.0, OrderType.MARKET, None),
        Fill(
            fill,
            OrderSide.SELL,
            10.0,
            SL_LEVEL,
            0.0,
            OrderType.STOP_MARKET,
            ClientOrderId("SL-1"),
        ),
    ]
    return FakePosition(
        id=pid,
        instrument_id=BTC,
        ts_opened=start,
        ts_closed=fill,
        events=events,
        realized_pnl=-50.0,
        is_closed=True,
        entry=OrderSide.BUY,
        avg_px_open=100.0,
        avg_px_close=SL_LEVEL,
        peak_qty=10.0,
        signed_qty=0.0,
    )


def sl_order(accepted_at: int) -> Order:
    return Order(
        ClientOrderId("SL-1"),
        OrderType.STOP_MARKET,
        OrderSide.SELL,
        [OrderAccepted(accepted_at)],
        trigger_price=SL_LEVEL,
    )


REFINE_MIN = DAY_MIN + 313  # day 1, 05:13: the 1m bar whose low 94.9 reaches the SL
REFINED_TS = at(REFINE_MIN) + 59_999 * ONE_MS


def refined_catalog(tmp_path: Path) -> Path:
    """A catalog whose 1m LAST series reaches 95 only at day 1 05:13 (R5)."""
    catalog = tmp_path / "catalog"
    write_last_bars(catalog, toy_bars({REFINE_MIN: (95.0, 95.0, 94.9, 95.0)}))
    return catalog


# ---------------------------------------------------------- paid_through


def test_paid_through_counts_closed_and_open_funding(archive: Path) -> None:
    """A closed long 10 @ 100 from 00:00:59.999 to day 2 00:00:59.999 crossed 6 settlements
    -> 0.6000000000000001 paid (NT 1100.0, costed 1099.4 = 9.994 at 110 in 0.001 steps). A
    position still open is paid only through ``now`` (R3: 2 settlements of 0.05 -> 0.1)."""
    entry = at(0) + 59_999 * ONE_MS
    exit_ns = at(2 * DAY_MIN) + 59_999 * ONE_MS
    ledger = ledger_for(archive)
    pos = closed(entry_ns=entry, exit_ns=exit_ns)

    ledger.on_position_closed(pos)

    paid = ledger.paid_through(exit_ns, [])
    assert paid == 0.6000000000000001
    assert ledger.finalized == {position_key(pos): 0.6000000000000001}
    nt_balance = 1000.0 + float(pos.realized_pnl)  # 1100.0
    assert nt_balance == 1100.0
    costed = nt_balance - paid
    assert costed == pytest.approx(1099.4)
    assert math.floor(costed / 110.0 * 1000.0) / 1000.0 == 9.994  # 9.994 vs the old 10.000

    other = open_long(entry_ns=entry)
    other_ledger = ledger_for(archive)
    assert other_ledger.paid_through(at(DAY_MIN) - ONE_MS, [other]) == 0.1  # 2 x 0.05
    assert other_ledger.paid_through(at(DAY_MIN), [other]) == 0.15000000000000002  # + 00:00 (== now)


def test_open_position_on_other_symbol(archive: Path) -> None:
    """An ETH long of notional 500 open since 00:00:59.999 costs 0.1 at day 0's close, and a
    BTC ledger's own finalized costs do not change it (the two symbols are summed, not merged)."""
    entry = at(0) + 59_999 * ONE_MS
    ledger = ledger_for(archive)
    btc = closed(entry_ns=entry, exit_ns=at(DAY_MIN) + 59_999 * ONE_MS)
    eth = open_long(entry_ns=entry, qty=5.0)
    ledger.on_position_closed(btc)

    now = at(DAY_MIN) - ONE_MS  # day 0 23:59:59.999: 08:00 + 16:00, never Jan-2 00:00
    only_eth = ledger_for(archive).paid_through(now, [eth])

    assert only_eth == 0.1
    # the closed BTC position funds to ITS close (day 1 00:00:59.999): 3 settlements
    assert ledger.paid_through(now, []) == 0.30000000000000004
    assert ledger.paid_through(now, [eth]) == 0.4  # + the open ETH position's 2 settlements


def test_settlement_edges(archive: Path) -> None:
    """A settlement EXACTLY at the entry instant is excluded, one exactly at the exit is
    included, and an open position is funded through ``now`` inclusive (the equality edges)."""
    eight = at(8 * 60)  # day 0 08:00, an archived settlement
    sixteen = at(16 * 60)  # day 0 16:00
    assert FundingCalculator(str(archive)).settlements("BTCUSDT")[0].count(eight) == 1

    at_entry = closed(entry_ns=eight, exit_ns=sixteen)
    accrual = accrue_position_funding(FundingCalculator(str(archive)), at_entry)
    assert [(ts, amt) for ts, amt in accrual.payments] == [(sixteen, 0.1)]

    open_pos = open_long(entry_ns=eight, instrument=BTC)
    ledger = ledger_for(archive)
    assert ledger.paid_through(sixteen, [open_pos]) == 0.05  # exactly at ``now``: included
    assert ledger.paid_through(sixteen - 1, [open_pos]) == 0.0
    assert ledger.paid_through(sixteen + 1, [open_pos]) == 0.05


# ------------------------------------------------------------------ Shorts


def test_received_funding_raises_costed(archive: Path) -> None:
    """A short of 1000 at +0.0001 RECEIVES 0.1 per settlement -> paid is -0.1 (one
    settlement), so the costed balance is ABOVE the NT balance."""
    entry = at(60)  # day 0 01:00
    exit_ns = at(9 * 60)  # 09:00: the 08:00 settlement is paid to the short
    ledger = ledger_for(archive)
    pos = closed(entry_ns=entry, exit_ns=exit_ns, entry_side=OrderSide.SELL, exit_px=99.0)

    ledger.on_position_closed(pos)

    paid = ledger.paid_through(exit_ns, [])
    assert paid == -0.1
    nt_balance = 1000.0 + float(pos.realized_pnl)
    assert nt_balance - paid > nt_balance


# ------------------------------------------------- The refined exit


def test_refined_boundary_exit(archive: Path, tmp_path: Path) -> None:
    """A 1d SL refined to the 1m bar that reached it (day 1 05:13:59.999) funds 3 settlements
    -> 0.30000000000000004, NOT the 5 the NT bar close would charge (0.5). NT 950.0, costed
    949.7 at that bar close."""
    from vibe_quant.trade_ledger import funding_refine_instruments

    catalog = refined_catalog(tmp_path)
    calc = FundingCalculator(str(archive))
    refine = funding_refine_instruments(calc, [BTC], E1D, at(0), at(3 * DAY_MIN))
    assert refine == frozenset({BTC})  # a 1d bar outlasts an 8h interval

    pos = d1_sl_position()
    ledger = ledger_for(archive, {"SL-1": sl_order(int(pos.ts_opened))}, refine=refine, catalog=catalog)

    ledger.on_position_closed(pos)

    assert ledger.finalized[position_key(pos)] == 0.30000000000000004
    assert ledger.finalized[position_key(pos)] != 0.5  # the unrefined NT bar close
    assert ledger.paid_through(REFINED_TS, []) == 0.30000000000000004
    costed = 1000.0 + float(pos.realized_pnl) - ledger.paid_through(REFINED_TS, [])
    assert costed == pytest.approx(949.7)


def test_missing_1m_last_raises_from_on_position_closed(archive: Path, tmp_path: Path) -> None:
    """A refine instrument with no 1m LAST series raises ``MissingExitBars`` (a
    DataUnavailableError, so NT's actor re-raises it) out of ``on_position_closed``, and the
    ledger keeps no half-finalized key."""
    ledger = ledger_for(
        archive,
        {"SL-1": sl_order(at(0) + 59_999 * ONE_MS)},
        refine=frozenset({BTC}),
        catalog=tmp_path / "no_catalog",  # never written: no bar directory at all
    )
    pos = d1_sl_position()

    with pytest.raises(MissingExitBars, match=r"^BTCUSDT: no 1m LAST series"):
        ledger.on_position_closed(pos)
    assert ledger.finalized == {}


def test_other_refinement_error_is_wrapped_as_the_expected_text(archive: Path, tmp_path: Path) -> None:
    """Any other refinement failure becomes the R10b ``DataUnavailableError`` naming the
    timeframe, with the original exception as ``__cause__``."""
    catalog = refined_catalog(tmp_path)
    ledger = ledger_for(archive, {}, refine=frozenset({BTC}), catalog=catalog)
    pos = d1_sl_position()

    with pytest.raises(DataUnavailableError) as ei:
        ledger.on_position_closed(pos)

    assert str(ei.value) == (
        f"funding exit refinement (1d): ValueError: {position_key(pos)} fill 1: "
        "order SL-1 is not in the engine cache"
    )
    assert isinstance(ei.value.__cause__, ValueError)


# ------------------------------------------------------------- Netting


def test_netting_cycles(archive: Path) -> None:
    """Two NETTING cycles share the position id; ``id@ts_opened`` keeps them apart and both
    are finalized (a bare-id key would drop the first cycle's costs)."""
    ledger = ledger_for(archive)
    first = closed(pid="N", entry_ns=at(60), exit_ns=at(9 * 60))
    second = closed(pid="N", entry_ns=at(DAY_MIN + 60), exit_ns=at(DAY_MIN + 9 * 60))

    ledger.on_position_closed(first)
    ledger.on_position_closed(second)

    assert list(ledger.finalized) == [position_key(first), position_key(second)]
    assert position_key(first) != position_key(second)
    assert ledger.finalized[position_key(first)] == 0.1
    assert ledger.finalized[position_key(second)] == 0.1
    assert ledger.paid_through(at(2 * DAY_MIN), []) == 0.2


def test_partial_exit_then_close_funds_the_peak_quantity(archive: Path) -> None:
    """Funding is charged on ``peak_qty x avg_px_open`` (2 x 100 here) to the CLOSING fill,
    both in the ledger and in the post-hoc accrual it reconciles against."""
    from tests.unit.test_screening_liquidation_exit_timing import Fill

    entry = at(60)
    half = at(6 * 60)
    exit_ns = at(9 * 60)
    pos = FakePosition(
        id="P",
        instrument_id=BTC,
        ts_opened=entry,
        ts_closed=exit_ns,
        events=[
            Fill(entry, OrderSide.BUY, 2.0, 100.0),
            Fill(half, OrderSide.SELL, 1.0, 110.0),
            Fill(exit_ns, OrderSide.SELL, 1.0, 120.0),
        ],
        realized_pnl=30.0,
        is_closed=True,
        entry=OrderSide.BUY,
        avg_px_open=100.0,
        avg_px_close=120.0,
        peak_qty=2.0,
        signed_qty=0.0,
    )
    ledger = ledger_for(archive)

    ledger.on_position_closed(pos)

    # notional 200, one settlement (08:00) at 0.0001 -> 0.02, not the half-open 0.01
    assert ledger.finalized[position_key(pos)] == 0.02
    post_hoc = accrue_position_funding(FundingCalculator(str(archive)), pos)
    assert ledger.finalized[position_key(pos)] == post_hoc.total


# ------------------------------------------------------- Reconcile


def test_reconcile_mismatch(archive: Path) -> None:
    """A finalized value the post-hoc ledger does not reproduce raises
    ModeledCostIntegrityError with the exact message (sizing first, ledger second)."""
    ledger = ledger_for(archive)
    pos = closed(entry_ns=at(60), exit_ns=at(9 * 60))
    key = position_key(pos)
    ledger.finalized[key] = 0.30000000001
    costed = _position_costs(pos, total=0.30000000000000004)

    with pytest.raises(_module().ModeledCostIntegrityError) as ei:
        _module().reconcile(ledger, [costed])

    assert str(ei.value) == (
        f"modeled costs: sizing ledger and trade ledger disagree for {key}: "
        "sizing=0.30000000001 ledger=0.30000000000000004"
    )
    assert isinstance(ei.value, DataUnavailableError)


def _position_costs(pos: FakePosition, *, total: float, slippage: float = 0.0) -> PositionCosts:
    return PositionCosts(
        pos=pos,
        mark=None,
        realized_pnl=float(pos.realized_pnl),
        exit_price=None,
        exit_ns=int(pos.ts_closed),
        exit_reason=None,
        pos_fees=0.0,
        slippage_cost=slippage,
        accrual=FundingAccrual(total=total),
    )


def test_reconcile_requires_every_finalized_key_to_have_a_closed_trade(archive: Path) -> None:
    """A finalized cost with no post-hoc trade (the hostile no-position / cache-read-error case:
    `reconcile(ledger, [])`) raises; so does a nonempty list that misses one finalized key, a
    key whose item is an OPEN mark, and a missing key after a stop. All keys present passes."""
    a = closed(pid="A", entry_ns=at(60), exit_ns=at(9 * 60))
    b = closed(pid="B", entry_ns=at(60), exit_ns=at(17 * 60))
    ledger = ledger_for(archive)
    ledger.on_position_closed(a)
    ledger.on_position_closed(b)
    both = [*_costed_from_ledger(ledger, a), *_costed_from_ledger(ledger, b)]

    _module().reconcile(ledger, both)  # every key represented: fine

    with pytest.raises(_module().ModeledCostIntegrityError) as ei:
        _module().reconcile(ledger, [])
    assert position_key(a) in str(ei.value)
    with pytest.raises(_module().ModeledCostIntegrityError) as ei:
        _module().reconcile(ledger, both[:1])  # B finalized, no trade
    assert position_key(b) in str(ei.value)
    assert position_key(a) not in str(ei.value)

    mark = SimpleNamespace(ts=at(DAY_MIN), price=110.0, unrealized=10.0, exit_fee=0.0)
    as_open = PositionCosts(**{**both[1].__dict__, "mark": mark})
    with pytest.raises(_module().ModeledCostIntegrityError):
        _module().reconcile(ledger, [both[0], as_open])

    ledger.stopped(at(DAY_MIN))  # the stop forgives unfinalized closes, never finalized ones
    with pytest.raises(_module().ModeledCostIntegrityError):
        _module().reconcile(ledger, [])


SNAP = "5f0c1b2e-3d4a-4b6c-8d9e-0a1b2c3d4e5f"  # a uuid4, as NT's NETTING snapshot suffix


def test_reconcile_matches_a_netting_snapshot_id_to_the_live_close(archive: Path) -> None:
    """NT snapshots a re-entered NETTING position as ``<id>-<uuid4>``: the live close ``X-000``
    and the snapshot trade land on one key; a suffix that is not a uuid4 is NOT stripped; two
    cycles of one base id with different ts_opened stay distinct."""
    live = closed(pid="X-000", entry_ns=at(60), exit_ns=at(9 * 60))
    snap = dataclasses.replace(live, id=f"X-000-{SNAP}")
    ledger = ledger_for(archive)
    ledger.on_position_closed(live)

    assert list(ledger.finalized) == [f"X-000@{at(60)}"]
    _module().reconcile(ledger, _costed_from_ledger(ledger, snap))

    other = dataclasses.replace(live, id="X-000-notauuid")  # v1 uuid, wrong shape, no strip
    v1 = dataclasses.replace(live, id="X-000-5f0c1b2e-3d4a-1b6c-8d9e-0a1b2c3d4e5f")
    for pos in (other, v1):
        assert _module().cost_key(pos) == f"{pos.id}@{at(60)}"
        with pytest.raises(_module().ModeledCostIntegrityError):
            _module().reconcile(ledger, _costed_from_ledger(ledger, pos))

    second = closed(pid="X-000", entry_ns=at(2 * 60), exit_ns=at(10 * 60))
    ledger.on_position_closed(second)
    assert list(ledger.finalized) == [f"X-000@{at(60)}", f"X-000@{at(2 * 60)}"]
    both = [
        *_costed_from_ledger(ledger, snap),
        *_costed_from_ledger(ledger, dataclasses.replace(second, id=f"X-000-{SNAP}")),
    ]
    _module().reconcile(ledger, both)


def test_funded_slippage_close_reconciles_exactly(archive: Path) -> None:
    """Funding 0.1 plus slippage legs 0.15000000000000002 + 0.14250000000000002 (10 @ 100 in,
    95 out, avg volume 1000, volatility 0.01): the post-hoc sum is funding + (entry + exit) =
    0.39250000000000007; the sizing ledger must hold the SAME float, and reconcile stays exact."""
    pos = closed(entry_ns=at(60), exit_ns=at(9 * 60), exit_px=95.0)
    slip = _module().SlippageLegModel(
        SlippageEstimator(impact_coefficient=0.1), {BTC: FixedStats(1000.0, 0.01)}
    )
    ledger = ledger_for(archive, slippage=slip)
    entry = slip.leg(pos, 100.0, pos.ts_opened)
    exit_ = slip.leg(pos, 95.0, pos.ts_closed)
    assert (entry, exit_) == (0.15000000000000002, 0.14250000000000002)

    ledger.on_position_closed(pos)

    assert ledger.finalized[position_key(pos)] == 0.39250000000000007
    item = _position_costs(pos, total=0.1, slippage=entry + exit_)
    _module().reconcile(ledger, [item])
    assert 0.1 + entry + exit_ == 0.3925  # the order that was wrong


def test_reconcile_on_stop_closes(archive: Path) -> None:
    """A closed position the ledger never finalized is allowed only at or after
    ``stopped_ns`` (NT's on_stop closes); before it, or with no stop at all, it raises."""
    stop = at(DAY_MIN)
    later = closed(pid="L", entry_ns=at(60), exit_ns=stop + MIN)  # closed AFTER the stop
    exactly = closed(pid="E", entry_ns=at(60), exit_ns=stop)  # AT the stop: allowed
    earlier = closed(pid="B", entry_ns=at(60), exit_ns=stop - MIN)  # before it: not ours

    ledger = ledger_for(archive)
    ledger.stopped(stop)
    assert ledger.stopped_ns == stop
    _module().reconcile(ledger, [_position_costs(later, total=0.0)])
    _module().reconcile(ledger, [_position_costs(exactly, total=0.0)])

    with pytest.raises(_module().ModeledCostIntegrityError):
        _module().reconcile(ledger, [_position_costs(earlier, total=0.0)])

    never_stopped = ledger_for(archive)
    with pytest.raises(_module().ModeledCostIntegrityError):
        _module().reconcile(never_stopped, [_position_costs(later, total=0.0)])


def test_reconcile_ignores_open_positions(archive: Path) -> None:
    """Amendment (5): reconcile covers closed positions only; an open position's end-of-data
    valuation slippage is a documented addition, never a disagreement."""
    ledger = ledger_for(archive)
    pos = open_long(entry_ns=at(60))
    mark = SimpleNamespace(ts=at(DAY_MIN), price=110.0, unrealized=10.0, exit_fee=0.0)
    item = PositionCosts(
        pos=pos,
        mark=mark,
        realized_pnl=10.0,
        exit_price=110.0,
        exit_ns=at(DAY_MIN),
        exit_reason="end_of_data",
        pos_fees=0.0,
        slippage_cost=0.5,
        accrual=FundingAccrual(total=0.1),
    )

    _module().reconcile(ledger, [item])  # no raise: not a closed position


# -------------------------------------------------------------- Fallback


def test_funding_hole_charges_the_flagged_fallback(tmp_path: Path) -> None:
    """A settlement the archive does not cover is charged the fallback rate and flagged;
    the ledger's number is the accrual's own (same function, same floats)."""
    archive = interval_archive(tmp_path / "holes.db", {"SOLUSDT": every(8)}, rate=0.0001)
    ledger = ledger_for(archive)
    pos = closed(entry_ns=at(60), exit_ns=at(9 * 60))

    ledger.on_position_closed(pos)

    accrual = accrue_position_funding(FundingCalculator(str(archive)), pos)
    assert accrual.fallback_settlements == 1
    assert accrual.total == 0.1  # the 08:00 settlement at the documented fallback rate
    assert ledger.finalized[position_key(pos)] == accrual.total
    _module().reconcile(ledger, _costed_from_ledger(ledger, pos))


def _costed_from_ledger(ledger: Any, pos: FakePosition) -> list[PositionCosts]:
    calc = ledger.model.funding
    return cost_positions([(pos, None)], SCREENING_RULES, funding=calc)


# -------------------------------------------------------------- Slippage


class FixedStats:
    """A ``MarketStatsSeries`` stand-in with hand-set numbers (the R15 toy)."""

    def __init__(self, avg_volume: float, volatility: float) -> None:
        self._stats = (avg_volume, volatility)

    def before(self, ts_ns: int) -> tuple[float, float]:
        return self._stats


def test_slippage_stats_use_bars_before_fill(archive: Path) -> None:
    """The leg is ``estimate_cost(price, peak_qty, avg, vol, 0.0001)``: avg 1000, vol 0.01,
    10 @ 100 -> 0.15000000000000002. An instrument with no series uses the documented defaults
    (1000, 0.02); an estimator of ``None`` (engine slippage on) charges nothing."""
    slip = _module().SlippageLegModel(
        SlippageEstimator(impact_coefficient=0.1),
        {BTC: FixedStats(1000.0, 0.01)},
    )
    btc = closed(entry_ns=at(60), exit_ns=at(9 * 60))
    eth = closed(pid="E", instrument=ETH, entry_ns=at(60), exit_ns=at(9 * 60))

    assert slip.leg(btc, 100.0, at(60)) == 0.15000000000000002
    assert slip.leg(eth, 100.0, at(60)) == 0.25  # defaults (1000, 0.02)
    assert _module().SlippageLegModel(None, {}).leg(btc, 100.0, at(60)) == 0.0


def test_market_stats_series_before_is_causal() -> None:
    """``before(ts)`` is strictly before: the bar AT the fill is excluded (the equality
    edge), so a fat future bar cannot move the slippage the fill is charged."""
    ts = [at(0), at(1), at(2)]
    series = _module().MarketStatsSeries.from_bars(ts, [100.0, 100.0, 100.0], [1000.0, 1000.0, 5000.0])

    assert series.before(at(2)) == (1000.0, 0.02)  # the 5000 bar AT the fill is not read
    assert series.before(at(2) + 1) == (7000.0 / 3.0, 0.02)
    assert series.before(at(0)) == (1000.0, 0.02)  # empty prefix: the documented defaults
    assert series.before(at(0) - 1) == (1000.0, 0.02)


def _reference_stats(closes: list[float], volumes: list[float]) -> tuple[float, float]:
    """The two-pass SPEC formula over one prefix, written out independently of the series."""
    positive_volumes = [v for v in volumes if v > 0]
    avg_volume = (
        sum(positive_volumes) / len(positive_volumes) if positive_volumes else 1000.0
    )
    positive = [c for c in closes if c > 0]
    volatility = 0.02
    if len(positive) >= 3:
        returns = [math.log(positive[i] / positive[i - 1]) for i in range(1, len(positive))]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        if variance > 0:
            volatility = math.sqrt(variance)
    return avg_volume, volatility


def test_market_stats_series_is_the_two_pass_prefix_formula() -> None:
    """Over the same bars ``before(everything)`` equals the two-pass formula (1e-12 relative)."""
    closes = [100.0 + 0.3 * math.sin(i / 3.0) for i in range(60)]
    closes[17] = 0.0  # a non-positive close is dropped (and its ratio skipped)
    volumes = [0.0 if i % 7 == 0 else 500.0 + i for i in range(60)]
    ts = [at(i * 60) for i in range(60)]

    series = _module().MarketStatsSeries.from_bars(ts, closes, volumes)
    avg_volume, volatility = series.before(ts[-1] + 1)

    expected_volume, expected_volatility = _reference_stats(closes, volumes)
    assert avg_volume == pytest.approx(expected_volume, rel=1e-12)
    assert volatility == pytest.approx(expected_volatility, rel=1e-12)
    assert avg_volume == (sum(v for v in volumes if v > 0) / len([v for v in volumes if v > 0]))


def test_market_stats_series_survives_cancelling_variance() -> None:
    """Near-constant returns cancel in raw prefix second moments: the series must still equal the
    two-pass formula (1e-12 relative) in BOTH directions -- a real tiny volatility the prefix
    sums would lose (reference 3.96e-12, not the 0.02 fallback) and a constant-return ramp where
    the reference falls back to 0.02 (prefix sums gave 1.3e-8)."""
    tiny = [100.0, 100.1, 100.2001, 100.3003001, 100.4006004, 100.501001, 100.601502]
    expected = _reference_stats(tiny, [1000.0] * len(tiny))[1]
    assert expected == 3.963769082608259e-12
    assert _module().MarketStatsSeries.from_bars(
        range(7), tiny, [1000.0] * 7
    ).before(7)[1] == pytest.approx(expected, rel=1e-12)

    doubling = [100.0 * 2**i for i in range(7)]
    assert _reference_stats(doubling, [1000.0] * len(doubling))[1] == 0.02
    assert _module().MarketStatsSeries.from_bars(
        range(7), doubling, [1000.0] * 7
    ).before(7)[1] == 0.02

    # the recompute is per prefix: a prefix of the tiny series also matches its own reference
    series = _module().MarketStatsSeries.from_bars(range(7), tiny, [1000.0] * 7)
    assert series.before(4)[1] == pytest.approx(
        _reference_stats(tiny[:4], [1000.0] * 4)[1], rel=1e-12
    )


# ------------------------------------------- lifecycle: idempotence, pickling


def test_closing_the_same_key_twice_changes_nothing(archive: Path) -> None:
    """A second ``on_position_closed`` of the same key keeps the first total (NT can report an
    account-wide close twice): the dict and the running total stay put."""
    ledger = ledger_for(archive)
    pos = closed(entry_ns=at(60), exit_ns=at(9 * 60))

    ledger.on_position_closed(pos)
    first = ledger.paid_through(at(DAY_MIN), [])
    ledger.on_position_closed(pos)

    assert ledger.paid_through(at(DAY_MIN), []) == first == 0.1
    assert len(ledger.finalized) == 1


def test_integrity_error_pickles_across_the_process_pool() -> None:
    """The screening process pool pickles results: the error keeps its type and message."""
    err = _module().ModeledCostIntegrityError("modeled costs: nope")

    back = pickle.loads(pickle.dumps(err))

    assert type(back) is _module().ModeledCostIntegrityError
    assert str(back) == "modeled costs: nope"
    assert isinstance(back, DataUnavailableError)


def test_module_imports_in_a_fresh_process() -> None:
    """Risk 3: no import cycle (the validation/liquidation imports stay lazy)."""
    done = subprocess.run(
        [sys.executable, "-c", "import vibe_quant.modeled_costs"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
