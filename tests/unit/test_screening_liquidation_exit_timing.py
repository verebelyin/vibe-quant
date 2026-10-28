"""Mode-on screening refines boundary resting exits to their 1m bar (V2, bead vibe-quant-vjno2).

Contract rows R17-R30. A STOP_MARKET / LIMIT fill of a coarser-than-1m screening run lands on
the execution bar's ``ts_init`` (the bar close less 1 ms), so the liquidation scan used to read
the exit a whole bar too late and liquidate a position validation's 1m stop had already closed.
``nt_runner._scan_with_survivor_marks`` now loads the 1m LAST windows of the symbols that HAVE
such a fill (catalog only, never MARK, never the engine cache), refines them
(``liquidation.exit_timing``) and hands the refined timestamps to ``apply_liquidation``. The
ledgers, funding and MTM keep the NT timestamps.

Positions, fills and orders are duck-typed stubs (no NT engine); MARK bars are real parquet and
the 1m LAST bars the refinement reads are real catalog rows, so ``load_bar_windows`` runs for
real.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.enums import OrderSide, OrderType, PriceType
from nautilus_trader.model.identifiers import ClientOrderId

from tests.unit.test_audit_metrics_funding import empty_archive
from tests.unit.test_screening_liquidation import (
    BALANCE,
    BTC,
    START_MS,
    at,
    bt_result,
    engine_for,
    equal_metrics,
    flat_bars,
    spy_scans,
    write_marks,
)
from tests.unit.test_screening_liquidation import Pos as StubPos
from vibe_quant.data.catalog import CatalogManager, create_instrument, get_bar_type, klines_to_bars
from vibe_quant.errors import LiquidationIntegrityError
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.screening.nt_runner import NTScreeningRunner

if TYPE_CHECKING:
    from types import SimpleNamespace

    from tests.unit.test_screening_liquidation import Bars

# 12:00–16:00 is one 4h execution bar of the run window (a 4h grid point: START_NS % 4h == 0).
BAR_OPEN_MIN = 720  # 12:00
BAR_CLOSE_MIN = 960  # 16:00: the boundary fill's NT ts is 15:59:59.999
SL_LEVEL = 39500.0
SL_FILL_PX = 39500.0
E4H = 14_400_000_000_000  # the 4h execution bar in ns
ONE_MS = 1_000_000
# MARK lows: 39400 at 14:19 (above the LP), 39300 at 14:35 (below it)
CRASH_BARS = flat_bars(
    {
        859: (40000.0, 40000.0, 39400.0, 40000.0),
        875: (40000.0, 40000.0, 39300.0, 40000.0),
    }
)
# A 14:19 MARK low below the LP: the liquidation would hit in the exit minute itself
TIE_BARS = flat_bars(
    {
        859: (40000.0, 40000.0, 39350.0, 40000.0),
        875: (40000.0, 40000.0, 39300.0, 40000.0),
    }
)
CALM_BARS = flat_bars()
# ETH's own 1m series: its SL level 39500 is reached at 15:05, not at BTC's 14:19 (R30)
ETH_BARS = flat_bars({905: (40000.0, 40000.0, 39400.0, 40000.0)})
ETH = "ETHUSDT-PERP.BINANCE"
ZERO_TIMING = {"refined": 0, "kept_bar_close": 0, "by_symbol": {}}


def boundary_ts() -> int:
    """The NT fill ts of a boundary resting fill: the 4h bar's close less 1 ms (R18)."""
    return at(BAR_CLOSE_MIN) - ONE_MS


def key_of(pos: StubPos) -> str:
    return f"{pos.id}@{int(pos.ts_opened)}"


@pytest.fixture(autouse=True)
def _fresh_rate_cache() -> None:
    from vibe_quant.validation.funding import clear_rate_cache

    clear_rate_cache()


@dataclass
class Fill:
    """NT fill stub: what ``overlay._collect`` and ``resting_exit_inputs`` read."""

    ts_event: int
    order_side: Any
    last_qty: float
    last_px: float
    commission: float = 0.0
    order_type: Any = OrderType.MARKET
    client_order_id: Any = None


@dataclass
class OrderAccepted:
    ts_event: int


@dataclass
class OrderUpdated:
    ts_event: int


@dataclass
class Order:
    """NT order stub: level, type and the accepted/updated history ``active_from`` reads."""

    client_order_id: Any
    order_type: Any
    side: Any
    events: list[Any]
    trigger_price: float | None = None
    price: float | None = None


def write_last_bars(catalog: Path, bars: Bars, interval: str = "1m", symbol: str = "BTCUSDT") -> None:
    """Materialise ``bars`` as catalog LAST bars (ts_init = the bar close) for ``load_bar_windows``."""
    instrument = create_instrument(symbol)
    bar_type = get_bar_type(symbol, interval, price_type=PriceType.LAST)
    rows = [
        {
            "open_time": START_MS + i * 60_000,
            "close_time": START_MS + i * 60_000 + 59_999,
            "open": bars.open[i],
            "high": bars.high[i],
            "low": bars.low[i],
            "close": bars.close[i],
            "volume": 1.0,
        }
        for i in range(len(bars.open))
    ]
    catalog_manager = CatalogManager(catalog)
    catalog_manager.write_instrument(instrument)
    catalog_manager.write_bars(
        klines_to_bars(
            rows,
            instrument.id,
            bar_type,
            instrument.size_precision,
            instrument.price_precision,
        )
    )


def sl_long(
    pid: str = "A",
    *,
    open_min: int = BAR_OPEN_MIN,
    order_id: str = "SL-1",
    qty: float = 1.0,
    exit_qty: float | None = None,
    order_type: Any = OrderType.STOP_MARKET,
    exit_px: float = SL_FILL_PX,
    instrument: str = BTC,
) -> StubPos:
    """1-unit long: MARKET entry at ``open_min``, exit fill at the 4h boundary of 16:00.

    ``exit_qty`` (a partial exit) defaults to the full ``qty``.
    """
    ts = at(BAR_CLOSE_MIN) - 1_000_000
    fills = [
        Fill(at(open_min), OrderSide.BUY, qty, 40000.0, 0.0, OrderType.MARKET, None),
        Fill(
            ts,
            OrderSide.SELL,
            qty if exit_qty is None else exit_qty,
            exit_px,
            0.0,
            order_type,
            ClientOrderId(order_id),
        ),
    ]
    return StubPos(
        id=pid,
        instrument_id=instrument,
        ts_opened=at(open_min),
        ts_closed=ts,
        events=fills,
        realized_pnl=0.0,
        is_closed=True,
        entry=OrderSide.BUY,
        avg_px_open=40000.0,
        peak_qty=qty,
        signed_qty=0.0,
        commission_list=[],
    )


def market_only_long(
    pid: str = "A", *, instrument: str = BTC, open_min: int = BAR_OPEN_MIN
) -> StubPos:
    """1-unit long with MARKET fills only (no resting exit): the entry and the NT exit."""
    pos = sl_long(pid, order_type=OrderType.MARKET, instrument=instrument, open_min=open_min)
    return pos


def stop_order(
    order_id: str = "SL-1",
    level: float = SL_LEVEL,
    *,
    accepted_at: int,
    updated_at: int | None = None,
    order_type: Any = OrderType.STOP_MARKET,
    side: Any = OrderSide.SELL,
) -> Order:
    events: list[Any] = [OrderAccepted(accepted_at)]
    if updated_at is not None:
        events.append(OrderUpdated(updated_at))
    return Order(ClientOrderId(order_id), order_type, side, events, trigger_price=level)


def limit_order(
    order_id: str = "TP-1",
    level: float = 40500.0,
    *,
    accepted_at: int,
    side: Any = OrderSide.BUY,
) -> Order:
    return Order(ClientOrderId(order_id), OrderType.LIMIT, side, [OrderAccepted(accepted_at)], price=level)


def engine_with_orders(
    positions: list[StubPos], bars: Bars, orders: dict[str, Order]
) -> SimpleNamespace:
    engine = engine_for(positions, bars)
    engine.kernel.cache.order = lambda cid: orders.get(str(cid))
    return engine


def no_cache_engine(positions: list[StubPos]) -> SimpleNamespace:
    """An engine whose cache raises on any read: proves a path never touches it (R17b)."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the exit refinement touched the engine cache")

    engine = engine_for(positions, CALM_BARS)
    engine.kernel.cache.bars = boom
    engine.kernel.cache.order = boom
    return engine


def spy_load_bar_windows(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[tuple[int, int]]]]:
    """Record every ``load_bar_windows`` call of the exit refinement (R17b/R19)."""
    import vibe_quant.validation.extraction as extraction

    calls: list[tuple[str, list[tuple[int, int]]]] = []
    real = extraction.load_bar_windows

    def spy(catalog_path: Any, bar_type: Any, windows: Any) -> Any:
        calls.append((str(bar_type), list(windows)))
        return real(catalog_path, bar_type, windows)

    monkeypatch.setattr(extraction, "load_bar_windows", spy)
    return calls


def resting_inputs(
    positions: list[StubPos], orders: dict[str, Order], bar_ns: int = E4H
) -> list[Any]:
    from vibe_quant.screening.liquidation_metrics import resting_exit_inputs

    return resting_exit_inputs(engine_with_orders(positions, CALM_BARS, orders), positions, bar_ns)


def position_with(
    fills: list[Fill], *, pid: str = "A", instrument: str = BTC, qty: float = 1.0
) -> StubPos:
    """An NT position stub over exactly ``fills`` (in order: the fill index is their index)."""
    return StubPos(
        id=pid,
        instrument_id=instrument,
        ts_opened=fills[0].ts_event,
        ts_closed=fills[-1].ts_event,
        events=list(fills),
        realized_pnl=0.0,
        is_closed=True,
        entry=fills[0].order_side,
        avg_px_open=40000.0,
        peak_qty=qty,
        signed_qty=0.0,
        commission_list=[],
    )


def marks_and_last(tmp_path: Path, bars: Bars = CRASH_BARS) -> Path:
    """A catalog with ``bars`` as MARK 1m (the scan) and as 1m LAST (the refinement, R19)."""
    catalog = write_marks(tmp_path, bars)
    write_last_bars(catalog, bars)
    return catalog


def screening_runner(
    tmp_path: Path, mode: str, catalog: Path | None, *, leverage: float = 10.0
) -> NTScreeningRunner:
    runner = NTScreeningRunner(
        dsl_dict={},
        symbols=["BTCUSDT"],
        start_date="2025-01-01",
        end_date="2025-01-04",
        catalog_path=str(catalog) if catalog else None,
        funding_archive_path=str(empty_archive(tmp_path / "none.db")),
        launch=LaunchSettings(
            liquidation=LiquidationSettings(mode=mode, leverage=leverage),  # type: ignore[arg-type]
        ),
    )
    runner._resolved_catalog_path = catalog  # type: ignore[assignment]
    return runner


# ---------------------------------------------- R18/R19/R21/R27/R28b: the inputs


def test_r18_only_boundary_stop_market_and_limit_fills_are_refined() -> None:
    """R18: a fill with ``(ts + 1 ms) % E == 0`` and order type STOP_MARKET/LIMIT is refined;
    the MARKET exit and the fills 1 ns off the boundary are left alone (and never looked up)."""
    ts = boundary_ts()
    market = Fill(ts, OrderSide.SELL, 1.0, SL_FILL_PX, order_type=OrderType.MARKET)
    stop = Fill(
        ts, OrderSide.SELL, 1.0, SL_FILL_PX, order_type=OrderType.STOP_MARKET,
        client_order_id=ClientOrderId("SL-1"),
    )  # fmt: skip
    limit = Fill(
        ts, OrderSide.BUY, 1.0, 40000.0, order_type=OrderType.LIMIT,
        client_order_id=ClientOrderId("TP-1"),
    )  # fmt: skip
    early = Fill(
        ts - 1, OrderSide.SELL, 1.0, SL_FILL_PX, order_type=OrderType.STOP_MARKET,
        client_order_id=ClientOrderId("SL-early"),
    )  # fmt: skip
    late = Fill(
        ts + 1, OrderSide.SELL, 1.0, SL_FILL_PX, order_type=OrderType.STOP_MARKET,
        client_order_id=ClientOrderId("SL-late"),
    )  # fmt: skip
    pos = position_with([market, stop, limit, early, late])
    orders = {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN)), "TP-1": limit_order(accepted_at=at(BAR_OPEN_MIN))}

    inputs = resting_inputs([pos], orders)

    assert [(i.order_id, i.kind) for i in inputs] == [("SL-1", "stop"), ("TP-1", "limit")]


@pytest.mark.parametrize(
    ("order_type", "side", "falls"),
    [
        (OrderType.STOP_MARKET, OrderSide.SELL, True),
        (OrderType.STOP_MARKET, OrderSide.BUY, False),
        (OrderType.LIMIT, OrderSide.BUY, True),
    ],
)
def test_r19_falls_is_the_direction_the_trigger_watches(
    order_type: Any, side: Any, falls: bool
) -> None:
    """R19: a SELL stop and a BUY limit watch the bar LOW, a BUY stop and a SELL limit the HIGH."""
    fill = Fill(
        boundary_ts(), side, 1.0, 40000.0, order_type=order_type, client_order_id=ClientOrderId("O-1")
    )  # fmt: skip
    order = Order(ClientOrderId("O-1"), order_type, side, [OrderAccepted(at(BAR_OPEN_MIN))], trigger_price=SL_LEVEL, price=SL_LEVEL)

    inputs = resting_inputs([position_with([fill])], {"O-1": order})

    assert [i.falls for i in inputs] == [falls]


def test_r19_level_is_the_order_price_never_the_fill_price() -> None:
    """R19: STOP_MARKET -> ``trigger_price``, LIMIT -> ``price``; ``last_px`` is never read,
    so a fill priced 500 away from its trigger still refines against the trigger."""
    ts = boundary_ts()
    stop = Fill(
        ts, OrderSide.SELL, 1.0, 39000.0, order_type=OrderType.STOP_MARKET,
        client_order_id=ClientOrderId("SL-1"),
    )  # fmt: skip
    limit = Fill(
        ts, OrderSide.BUY, 1.0, 41000.0, order_type=OrderType.LIMIT,
        client_order_id=ClientOrderId("TP-1"),
    )  # fmt: skip
    orders = {
        "SL-1": stop_order(level=39500.0, accepted_at=at(BAR_OPEN_MIN)),
        "TP-1": limit_order(level=40500.0, accepted_at=at(BAR_OPEN_MIN)),
    }

    inputs = resting_inputs([position_with([stop, limit])], orders)

    assert [(i.kind, i.level) for i in inputs] == [("stop", 39500.0), ("limit", 40500.0)]


def test_r21_active_from_is_the_last_accepted_or_updated_event_at_or_before_the_fill() -> None:
    """R21: an ``OrderUpdated`` after the ``OrderAccepted`` wins; an event exactly AT the fill
    counts, one 1 ns later does not (the equality edge)."""
    ts = boundary_ts()
    fill = Fill(
        ts, OrderSide.SELL, 1.0, SL_FILL_PX, order_type=OrderType.STOP_MARKET,
        client_order_id=ClientOrderId("SL-1"),
    )  # fmt: skip
    pos = position_with([fill])
    later = stop_order(accepted_at=at(BAR_OPEN_MIN), updated_at=at(BAR_OPEN_MIN + 1))
    assert [i.active_from for i in resting_inputs([pos], {"SL-1": later})] == [at(BAR_OPEN_MIN + 1)]

    def at_fill(event_ts: int) -> Order:
        return Order(
            ClientOrderId("SL-1"),
            OrderType.STOP_MARKET,
            OrderSide.SELL,
            [OrderAccepted(event_ts)],
            trigger_price=SL_LEVEL,
        )

    assert [i.active_from for i in resting_inputs([pos], {"SL-1": at_fill(ts)})] == [ts]
    with pytest.raises(ValueError, match=r"order SL-1 has no OrderAccepted/OrderUpdated"):
        resting_inputs([pos], {"SL-1": at_fill(ts + 1)})


def test_r27_an_unusable_order_on_the_boundary_names_the_key_and_client_order_id() -> None:
    """R27: an order missing from the cache, with no accepted/updated event, or of another
    resting type (STOP_LIMIT) is a ValueError naming the position key and the client order id."""
    ts = boundary_ts()
    missing = Fill(
        ts, OrderSide.SELL, 1.0, SL_FILL_PX, order_type=OrderType.STOP_MARKET,
        client_order_id=ClientOrderId("SL-1"),
    )  # fmt: skip
    pos = position_with([missing])
    with pytest.raises(ValueError, match=r"A@\d+ fill 0: order SL-1 is not in the engine cache"):
        resting_inputs([pos], {})
    with pytest.raises(ValueError, match=r"A@\d+ fill 0: order SL-1 has no OrderAccepted"):
        resting_inputs(
            [pos],
            {"SL-1": Order(ClientOrderId("SL-1"), OrderType.STOP_MARKET, OrderSide.SELL, [], 39500.0)},
        )

    for bad_type in (OrderType.STOP_LIMIT, OrderType.MARKET_IF_TOUCHED):
        fill = Fill(
            ts, OrderSide.SELL, 1.0, SL_FILL_PX, order_type=bad_type,
            client_order_id=ClientOrderId("SL-bad"),
        )  # fmt: skip
        with pytest.raises(ValueError, match=rf"order SL-bad is a {bad_type.name} fill"):
            resting_inputs([position_with([fill])], {})


def test_r28b_order_id_is_the_client_order_id_string() -> None:
    """R28b: the refined fill carries the client order id, not the position key or an index."""
    pos = sl_long()
    inputs = resting_inputs([pos], {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))})

    assert [i.order_id for i in inputs] == ["SL-1"]
    assert inputs[0].fill_index == 1  # the index of the fill inside position.events


# ---------------------------------------------- R17/R17b/R19/R25b: what is loaded


def test_r17_1m_execution_puts_zero_exit_timing_on_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2/R17/R25b: a mode-on 1m-execution run has nothing to re-time (no 1m LAST file is read,
    even with a boundary STOP_MARKET fill), but the zero counts still travel on the METRICS
    (they are pickled from workers; runner attributes are not), so the pipeline stores 0/0 --
    never NULL -- for a mode-on run."""
    catalog = marks_and_last(tmp_path)
    calls = spy_load_bar_windows(monkeypatch)
    engine = engine_with_orders(
        [sl_long()], CRASH_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)
    runner._all_timeframes = {"1m"}

    metrics = runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)

    assert calls == []
    assert metrics.exit_timing == ZERO_TIMING


def test_r17_mode_off_writes_no_exit_timing_and_only_the_liquidation_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17: mode off runs no new code at all -- no refinement, no notes, no file read."""
    catalog = marks_and_last(tmp_path)
    calls = spy_load_bar_windows(monkeypatch)
    engine = no_cache_engine([sl_long()])
    runner = screening_runner(tmp_path, "off", catalog, leverage=50.0)
    runner._all_timeframes = {"4h"}

    metrics = runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)

    assert metrics.liquidation_count == 0
    assert calls == []
    assert runner.last_liquidation_notes is None
    assert metrics.exit_timing is None


def test_r17b_r19_r25b_only_symbols_with_a_boundary_fill_load_their_1m_last_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R17b/R19 R25b: with BTC holding the only boundary STOP_MARKET fill, exactly BTC's 1m
    LAST bar type is read, over exactly that exit's window ``[ts + 1 ms - E, ts]``. The level
    39500 is never reached in a calm series, so the fill keeps its NT ts (R25) and the run
    logs one WARNING (R25b)."""
    catalog = tmp_path / "catalog"
    write_last_bars(catalog, CALM_BARS)
    calls = spy_load_bar_windows(monkeypatch)
    btc = sl_long()
    eth = market_only_long("E", instrument=ETH)
    engine = engine_with_orders(
        [btc, eth], CALM_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)
    ts = boundary_ts()

    with caplog.at_level(logging.WARNING, logger="vibe_quant.screening.nt_runner"):
        scan_fill_ts, notes = runner._boundary_exit_scan_ts(engine, [btc, eth], E4H, catalog)

    assert calls == [("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL", [(ts + ONE_MS - E4H, ts)])]
    assert scan_fill_ts == {key_of(btc): {1: ts}}
    assert notes == {
        "refined": 0,
        "kept_bar_close": 1,
        "by_symbol": {"BTCUSDT": {"refined": 0, "kept_bar_close": 1}},
    }
    assert any(
        "kept their NT strategy-bar timestamp" in record.getMessage() for record in caplog.records
    )


def test_r17b_no_boundary_resting_fill_reads_no_file_and_leaves_the_engine_cache_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17b: a run whose only exit is a MARKET fill (the screening entry/exit shape) reads no
    file and never calls into the engine cache -- the refinement is entirely catalog-driven."""
    catalog = tmp_path / "catalog"
    write_last_bars(catalog, CALM_BARS)
    calls = spy_load_bar_windows(monkeypatch)
    pos = market_only_long()
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)

    scan_fill_ts, notes = runner._boundary_exit_scan_ts(
        no_cache_engine([pos]), [pos], E4H, catalog
    )

    assert scan_fill_ts is None
    assert notes == ZERO_TIMING
    assert calls == []


# ------------------------------------------------------------------- R23: the probe


def test_r23_stop_inside_liquidation_distance_no_longer_liquidates_after_the_1m_exit(
    tmp_path: Path,
) -> None:
    """R23 probe: long 1 @ 40000, 50x isolated (LP 39357.4297188755), SL 39500 whose 1m exit is
    at 14:19. MARK lows are 39400 at 14:19 and 39300 at 14:35, so before V2 the position
    liquidated at 14:35; refined, the scan's exposure ends at 14:19:59.999 and it does not.
    R24: a zero-hit run equals the mode-off run field for field."""
    catalog = marks_and_last(tmp_path)
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)
    runner._all_timeframes = {"4h"}
    engine = engine_with_orders(
        [sl_long()], CRASH_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )

    metrics = runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)

    assert metrics.liquidation_count == 0
    assert metrics.total_trades == 1
    assert runner.last_liquidation_notes is not None
    assert runner.last_liquidation_notes["exit_timing"] == {
        "refined": 1,
        "kept_bar_close": 0,
        "by_symbol": {"BTCUSDT": {"refined": 1, "kept_bar_close": 0}},
    }
    # D1b/R25b: the counts MUST travel on the METRICS (they are pickled from workers;
    # runner attributes are not): a ``metrics.exit_timing = _no_exit_timing()`` fallback
    # would store zeros for a run that refined one exit.
    assert metrics.exit_timing == {
        "refined": 1,
        "kept_bar_close": 0,
        "by_symbol": {"BTCUSDT": {"refined": 1, "kept_bar_close": 0}},
    }
    off = screening_runner(tmp_path, "off", catalog, leverage=50.0)
    off._all_timeframes = {"4h"}
    off_engine = engine_with_orders(
        [sl_long()], CRASH_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )
    off_metrics = off._extract_metrics({}, bt_result(1), off_engine, time.time(), BALANCE)
    # R24/D1b: every MEASURED field is still identical (equal_metrics ignores the
    # mode-on-only exit_timing diagnostic); off it is None.
    assert off_metrics.exit_timing is None
    assert equal_metrics(metrics, off_metrics)


def test_r23_a_liquidation_inside_the_exit_minute_itself_still_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R23: a 14:19 MARK low of 39350 is below the LP, so the liquidation hits at 14:19:00 --
    the exit minute is scanned (the refined fill is at 14:19:59.999, after the hit) and filled
    at min(LP, open) = the LP."""
    catalog = marks_and_last(tmp_path, TIE_BARS)
    scans = spy_scans(monkeypatch)
    engine = engine_with_orders(
        [sl_long()], TIE_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)
    runner._all_timeframes = {"4h"}

    metrics = runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)

    hit = scans[0].hits_by_key[key_of(sl_long())]
    assert metrics.liquidation_count == 1
    assert hit.ts == at(859)  # 14:19:00, the triggering MARK bar's open
    assert hit.price == pytest.approx(39357.4297188755)  # min(LP, the 40000 bar open)
    assert runner.last_liquidation_notes is not None
    assert runner.last_liquidation_notes["exit_timing"]["refined"] == 1


# ------------------------------------------------------------- R24b/R26/R27: fail closed


def test_r24b_a_partial_refined_exit_then_a_liquidation_fails_the_run_closed(
    tmp_path: Path,
) -> None:
    """R24b: a half exit refined to 14:19:59.999 (NT 15:59:59.999) leaves the position open to
    the 14:35 liquidation; the scan's refined fill straddles the ledger's truncation instant,
    so the run fails closed instead of reconciling a qty the ledgers disagree with."""
    catalog = marks_and_last(tmp_path)
    engine = engine_with_orders(
        [sl_long(exit_qty=0.5)], CRASH_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)
    runner._all_timeframes = {"4h"}

    with pytest.raises(LiquidationIntegrityError, match=r"RefinedExitConflict: A@\d+: fill 1"):
        runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)


def test_r26_a_boundary_fill_without_1m_last_bars_fails_the_run_closed(tmp_path: Path) -> None:
    """R26: a symbol with a boundary resting fill but no 1m LAST bars raises MissingExitBars,
    which the mode-on guard turns into a LiquidationIntegrityError."""
    catalog = write_marks(tmp_path, CALM_BARS)  # MARK bars only: no 1m LAST directory
    engine = engine_with_orders(
        [sl_long()], CALM_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)
    runner._all_timeframes = {"4h"}

    with pytest.raises(
        LiquidationIntegrityError, match=r"liquidation_mode='isolated': MissingExitBars: BTCUSDT"
    ):
        runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)


# ------------------------------------------------------------------- R30: two symbols


def test_r30_each_fill_refines_against_its_own_series_and_the_symbol_order_is_invariant(
    tmp_path: Path,
) -> None:
    """R30: BTC's exit refines to the BTC 1m bar that reached it (14:19) and ETH's to the ETH
    one (15:05); reversing the position order changes neither the timestamps nor the counts."""
    catalog = tmp_path / "catalog"
    write_last_bars(catalog, CRASH_BARS)
    write_last_bars(catalog, ETH_BARS, symbol="ETHUSDT")
    btc = sl_long("B")
    eth = sl_long("E", instrument=ETH, order_id="ETH-SL")
    orders = {
        "SL-1": stop_order("SL-1", accepted_at=at(BAR_OPEN_MIN)),
        "ETH-SL": stop_order("ETH-SL", accepted_at=at(BAR_OPEN_MIN)),
    }
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)

    forward, forward_notes = runner._boundary_exit_scan_ts(
        engine_with_orders([btc, eth], CRASH_BARS, orders), [btc, eth], E4H, catalog
    )
    backward, backward_notes = runner._boundary_exit_scan_ts(
        engine_with_orders([eth, btc], CRASH_BARS, orders), [eth, btc], E4H, catalog
    )

    expected = {key_of(btc): {1: at(860) - ONE_MS}, key_of(eth): {1: at(906) - ONE_MS}}
    assert forward == expected
    assert forward == backward
    assert forward_notes == backward_notes
    assert forward_notes["by_symbol"] == {
        "BTCUSDT": {"refined": 1, "kept_bar_close": 0},
        "ETHUSDT": {"refined": 1, "kept_bar_close": 0},
    }


# -------------------------- R18/R23/R25b: the boundary fill's own index, behind another fill


def test_r18_a_boundary_fill_behind_a_non_boundary_fill_keeps_its_own_index(
    tmp_path: Path,
) -> None:
    """R18/R25b: a position with an entry (fill 0), a scale-in (fill 1) and the boundary stop
    (fill 2) refines the stop at ITS index 2. A mapping that pinned every refined fill to the
    first resting index (``min(index, 1)``) would mis-stamp the scale-in and leave the real
    exit at its NT bar close, so the scan would run a whole bar past the exit: in the R23
    shape the 14:35 MARK low would liquidate the position. Refined, the exposure ends at the
    14:19:59.999 exit and the run does not liquidate."""
    catalog = marks_and_last(tmp_path)  # MARK = LAST = CRASH_BARS
    runner = screening_runner(tmp_path, "isolated", catalog, leverage=50.0)
    runner._all_timeframes = {"4h"}
    entry = Fill(at(BAR_OPEN_MIN), OrderSide.BUY, 1.0, 40000.0)
    scale_in = Fill(at(BAR_OPEN_MIN + 60), OrderSide.BUY, 1.0, 40000.0)
    stop = Fill(
        boundary_ts(), OrderSide.SELL, 2.0, SL_FILL_PX, order_type=OrderType.STOP_MARKET,
        client_order_id=ClientOrderId("SL-1"),
    )  # fmt: skip
    pos = position_with([entry, scale_in, stop], qty=2.0)
    engine = engine_with_orders(
        [pos], CRASH_BARS, {"SL-1": stop_order(accepted_at=at(BAR_OPEN_MIN))}
    )

    scan_fill_ts, notes = runner._boundary_exit_scan_ts(engine, [pos], E4H, catalog)

    refined = at(860) - ONE_MS  # the 14:19 1m bar's ts_init (its low 39400 <= 39500)
    assert scan_fill_ts == {key_of(pos): {2: refined}}
    assert notes["refined"] == 1

    metrics = runner._extract_metrics({}, bt_result(1), engine, time.time(), BALANCE)

    assert metrics.liquidation_count == 0
    assert runner.last_liquidation_notes is not None
    assert runner.last_liquidation_notes["exit_timing"]["refined"] == 1

