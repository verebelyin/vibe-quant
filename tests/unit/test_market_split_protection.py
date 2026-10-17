"""Market-mode protective re-arm (bead vibe-quant-5byet, Contract C rows R41-R46).

The F3 fix (drop queued SL/TP, ignore orders whose cancel is queued) sat behind
the ``_post_only`` gate of ``_sync_protective_orders``. Contract C widens ONLY
sub-fix (a) - ``_drop_queued_protection()`` now runs in every mode - and keeps the
queued-cancel filter (b) post_only-gated: in market mode a queued cancel must not
hide an order that still has to be replaced (an asymmetric partial exit would
leave the position without its SL).

The generated entry is a market IOC order. On the screening FillModel (an L1 book
sized bar_volume/4) an entry larger than the L1 size fills once and cancels the
rest, so no PositionChanged fires and the entry path is unreachable without a GTC
entry (R41 drives one; R42 documents the IOC under-fill). A split stop exit is
driven end to end in R43, the market-mode queued-cancel rule directly in R44, and
the paper-shaped immediate path (empty outbox) in R46.

Runs use the ``tests/unit/test_maker_entries.py`` harness (``_engine``
FillModel(1.0, 0.0), 5m strategy bars + 1m release bars, outbox ``bar``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderStatus, OrderType, PositionSide, TimeInForce
from nautilus_trader.model.objects import Quantity

from tests.unit import test_maker_entries as h

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nautilus_trader.backtest.engine import BacktestEngine

_M = h._M
_SIGNAL_CLOSES = [100.0, 100.0, 101.0] + [100.5] * 8


@pytest.fixture
def engines() -> Iterator[list[BacktestEngine]]:
    created: list[BacktestEngine] = []
    yield created
    for engine in created:
        engine.dispose()


def _thin(bar: Bar, vol: str) -> Bar:
    """The same bar with a thin volume: the synthetic L1 size is bar_volume/4."""
    return Bar(
        bar_type=bar.bar_type,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=Quantity.from_str(vol),
        ts_event=bar.ts_event,
        ts_init=bar.ts_init,
    )


def _flat_bars(vol: str) -> list[Bar]:
    """5m strategy bars (bar 2 signals) + 60 flat 1m release bars, all volume ``vol``."""
    return [_thin(b, vol) for b in h._strategy_bars(_SIGNAL_CLOSES)] + [
        _thin(b, vol) for b in h._one_minute(60, base=100.5)
    ]


def _sized(qty: str, *, gtc_entry: bool = False) -> Any:
    """Wrap the compiled strategy: fixed entry size, optional GTC-forced market entry,
    and a ``_syncs`` counter on ``_sync_protective_orders``."""

    def wrap(cls: type) -> type:
        class Sized(cls):  # type: ignore[misc, valid-type]
            def _calculate_position_size(self, bar: Any, is_long: bool = True) -> Any:
                return self.instrument.make_qty(qty)

            def _sync_protective_orders(self, pos: Any) -> None:
                self._syncs = getattr(self, "_syncs", 0) + 1
                super()._sync_protective_orders(pos)

            if gtc_entry:

                def submit_order(self, order: Any, *args: Any, **kwargs: Any) -> Any:
                    # Rebuild a non-reduce IOC MARKET entry as GTC so the remainder
                    # fills one tick worse in the SAME matching call (a split entry).
                    if (
                        order.order_type == OrderType.MARKET
                        and not order.is_reduce_only
                        and order.time_in_force == TimeInForce.IOC
                    ):
                        order = self.order_factory.market(
                            instrument_id=order.instrument_id,
                            order_side=order.side,
                            quantity=order.quantity,
                            time_in_force=TimeInForce.GTC,
                        )
                    return super().submit_order(order, *args, **kwargs)

        return Sized

    return wrap


def _fills(order: Any) -> list[tuple[str, str]]:
    return [
        (str(e.last_qty), str(e.last_px)) for e in order.events if type(e).__name__ == "OrderFilled"
    ]


# ---------------------------------------------------------------------------
# R41: a GTC-forced split entry re-arms ONE protective pair (the widened drop)
# ---------------------------------------------------------------------------


def test_gtc_split_entry_keeps_one_protective_pair(engines: list[BacktestEngine]) -> None:
    engine, strat = h._run(
        engines,
        data=_flat_bars("0.010"),
        dsl=h._dsl("mkt_split_gtc"),
        entry_order="market",
        strategy_wrap=_sized("0.004", gtc_entry=True),
    )
    (entry,) = h._entries(engine)
    # L1 size = 0.010/4 = 0.0025 -> 0.002 at the bar price, remainder one tick worse
    assert _fills(entry) == [("0.002", "100.5"), ("0.002", "100.6")]
    (position,) = engine.cache.positions()
    assert position.quantity == Quantity.from_str("0.004")

    protective = h._protective(engine)
    live = [o for o in protective if o.is_open]
    assert sorted((o.order_type.name, str(o.quantity)) for o in live) == [
        ("LIMIT", "0.004"),
        ("STOP_MARKET", "0.004"),
    ], [(o.order_type.name, str(o.quantity), o.status.name) for o in protective]
    assert len(protective) == 2, [(o.order_type.name, str(o.quantity), o.status.name) for o in protective]
    assert strat._outbox == []


# ---------------------------------------------------------------------------
# R42: a generated IOC entry under-fills, no PositionChanged -> 0 syncs
# ---------------------------------------------------------------------------


def test_ioc_entry_underfills_without_position_changed(engines: list[BacktestEngine]) -> None:
    engine, strat = h._run(
        engines,
        data=_flat_bars("0.010"),
        dsl=h._dsl("mkt_split_ioc"),
        entry_order="market",
        strategy_wrap=_sized("0.004"),
    )
    (entry,) = h._entries(engine)
    assert _fills(entry) == [("0.002", "100.5")]
    assert entry.status == OrderStatus.CANCELED  # IOC remainder cancelled
    (position,) = engine.cache.positions()
    assert position.quantity == Quantity.from_str("0.002")

    protective = h._protective(engine)
    assert sorted((o.order_type.name, str(o.quantity)) for o in protective) == [
        ("LIMIT", "0.002"),
        ("STOP_MARKET", "0.002"),
    ], [(o.order_type.name, str(o.quantity), o.status.name) for o in protective]
    assert getattr(strat, "_syncs", 0) == 0  # the entry path never reaches the sync


# ---------------------------------------------------------------------------
# R43: a split stop exit closes the position cleanly
# ---------------------------------------------------------------------------


def test_split_stop_exit_closes_cleanly(engines: list[BacktestEngine]) -> None:
    m5 = h._strategy_bars(_SIGNAL_CLOSES)
    m5[6] = _thin(m5[6], "0.004")
    one_m = h._one_minute(60, base=100.5)
    one_m[30] = _thin(h._bar("1-MINUTE", _M, 30, 79.0, 79.0, 79.0, 79.0), "0.004")
    for i in range(31, 60):
        one_m[i] = h._bar("1-MINUTE", _M, i, 79.0, 79.0, 79.0, 79.0)

    engine, _ = h._run(
        engines,
        data=m5 + one_m,
        dsl=h._dsl("mkt_split_sl"),
        entry_order="market",
        strategy_wrap=_sized("0.006"),
    )
    (position,) = engine.cache.positions()
    assert not position.is_open

    protective = h._protective(engine)
    assert len(protective) == 3, [
        (o.order_type.name, str(o.quantity), o.status.name) for o in protective
    ]
    assert [o for o in protective if o.is_open] == []
    (stop,) = [o for o in protective if o.order_type == OrderType.STOP_MARKET]
    assert _fills(stop) == [("0.001", "79.0"), ("0.005", "78.9")]


# ---------------------------------------------------------------------------
# R44: market mode keeps an order whose cancel is queued (filter stays post_only)
# ---------------------------------------------------------------------------


def test_market_mode_keeps_queued_cancel_orders_in_sync(engines: list[BacktestEngine]) -> None:
    engine, strat = h._run(
        engines,
        data=_flat_bars("0.010"),
        dsl=h._dsl("mkt_split_queued_cancel"),
        entry_order="market",
        strategy_wrap=_sized("0.004", gtc_entry=True),
    )
    (stop,) = [o for o in h._protective(engine) if o.is_open and o.order_type == OrderType.STOP_MARKET]

    # A queued cancel for the live SL must not hide it: the sync replaces it.
    strat._protective_orders = lambda: [stop]  # type: ignore[method-assign]
    strat._outbox = [(0, strat.cancel_order, (stop,))]
    pos = SimpleNamespace(
        side=PositionSide.LONG, quantity=Quantity.from_str("0.006"), avg_px_open=100.5
    )
    strat._sync_protective_orders(pos)

    cancels = [cmd for cmd in strat._outbox if cmd[1] == strat.cancel_order]
    submits = [cmd for cmd in strat._outbox if cmd[1] == strat.submit_order]
    assert [c[2][0] for c in cancels] == [stop, stop]
    (replacement,) = submits
    assert replacement[2][0].order_type == OrderType.STOP_MARKET
    assert replacement[2][0].quantity == Quantity.from_str("0.006")
    assert replacement[2][0].trigger_price == stop.trigger_price


# ---------------------------------------------------------------------------
# R46: the paper-shaped immediate path never queues, the drop is a no-op
# ---------------------------------------------------------------------------


def test_immediate_mode_drop_is_noop(engines: list[BacktestEngine]) -> None:
    engine, strat = h._run(
        engines,
        data=_flat_bars("100000.000"),
        dsl=h._dsl("mkt_immediate"),
        release="",
        entry_order="market",
        strategy_wrap=_sized("0.004"),
    )
    assert strat._outbox == []  # command_release "": everything is sent immediately
    strat._drop_queued_protection()
    assert strat._outbox == []

    (entry,) = h._entries(engine)
    assert entry.order_type == OrderType.MARKET
    assert entry.status == OrderStatus.FILLED
