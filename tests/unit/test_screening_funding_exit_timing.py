"""A screening run funds a boundary SL/TP exit to its refined 1m instant (beads 6osqd, nxmxd.3).

Contract rows R3-R16b and R18. A coarse execution bar fills a boundary STOP_MARKET / LIMIT exit at
the bar close (``ts_init`` = close - 1 ms), a whole bar after validation's 1m simulation filled
it. ``trade_ledger._funding_close_ns`` moves the FUNDING exit of such a CLOSED position to that
refined instant, mode off AND on, for an instrument whose execution bar is strictly longer than
its shortest funding interval in the run window (``funding_refine_instruments``; 8h when the
archive holds none). The realized PnL, the ledger and the cash flows keep the NT fill
timestamp; a run whose closing fill is a boundary resting exit and has no 1m LAST fails closed
(MissingExitBars is a DataUnavailableError).

Positions, fills and orders are the duck-typed stubs of ``test_screening_liquidation_exit_timing``;
the 1m LAST series the refinement reads is a REAL catalog written by ``write_last_bars``.

The funding archive is absent (``screening_runner`` passes ``tmp/none.db``), so every uncovered
8h settlement is charged at the fallback 0.0001: 40000 notional -> 4.0 USD per settlement.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import ClientOrderId

from tests.unit.test_audit_metrics_funding import empty_archive
from tests.unit.test_screening_liquidation import (
    BALANCE,
    BTC,
    START_MS,
    at,
    bt_result,
    equal_metrics,
    flat_bars,
    ledger_net,
    write_marks,
)
from tests.unit.test_screening_liquidation import Pos as StubPos
from tests.unit.test_screening_liquidation_exit_timing import (
    Fill,
    engine_with_orders,
    key_of,
    limit_order,
    marks_and_last,
    position_with,
    screening_runner,
    stop_order,
    write_last_bars,
)
from vibe_quant import trade_ledger
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.errors import DataUnavailableError
from vibe_quant.liquidation.exit_timing import MissingExitBars
from vibe_quant.screening.nt_runner import NTScreeningRunner, _no_exit_timing
from vibe_quant.trade_ledger import _funding_close_ns
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from tests.unit.test_screening_liquidation import Bars

ONE_MS = 1_000_000
E1D = 86_400_000_000_000
E12H = 12 * 3_600 * 1_000_000_000
E8H = 8 * 3_600 * 1_000_000_000
E4H = 4 * 3_600 * 1_000_000_000
E2H = 2 * 3_600 * 1_000_000_000
E1H = 3_600 * 1_000_000_000
H_MS = 3_600_000
ETH = "ETHUSDT-PERP.BINANCE"
SOL = "SOLUSDT-PERP.BINANCE"
D1_CLOSE_MIN = 1440  # day 0 -> day 1 boundary: the 1d bar's close
SL_LEVEL = 39500.0
# A refined SL at 07:50:59.999 (R3): the 1m bar opening minute 470 reaches 39500.
REFINE_MIN = 470
BEFORE_0800 = flat_bars({REFINE_MIN: (40000.0, 40000.0, 39400.0, 40000.0)})


def d1_position(
    *,
    pid: str = "A",
    open_min: int = 60,
    order_id: str = "SL-1",
    order_type: Any = OrderType.STOP_MARKET,
    exit_px: float = SL_LEVEL,
    exit_qty: float = 1.0,
    n_min: int = D1_CLOSE_MIN,
    instrument: str = BTC,
) -> StubPos:
    """1 BTC long: MARKET entry at ``open_min``, exit fill at the 1d boundary ``n`` (R3 D1)."""
    n = at(n_min) - ONE_MS
    fills = [
        Fill(at(open_min), OrderSide.BUY, 1.0, 40000.0, 0.0, OrderType.MARKET, None),
        Fill(
            n,
            OrderSide.SELL,
            exit_qty,
            exit_px,
            0.0,
            order_type,
            ClientOrderId(order_id),
        ),
    ]
    return position_with(fills, pid=pid, instrument=instrument, qty=1.0)


def open_position(*, pid: str = "B", open_min: int = 600, qty: float = 1.0) -> StubPos:
    """A 1 BTC long NT never closed (R8): screening funds/marks it at the run end."""
    entry = Fill(at(open_min), OrderSide.BUY, qty, 40000.0, 0.0, OrderType.MARKET, None)
    return StubPos(
        id=pid,
        instrument_id=BTC,
        ts_opened=at(open_min),
        ts_closed=0,
        events=[entry],
        realized_pnl=0.0,
        is_closed=False,
        entry=OrderSide.BUY,
        avg_px_open=40000.0,
        peak_qty=qty,
        signed_qty=qty,
        commission_list=[],
    )


def last_bars_catalog(tmp_path: Path, bars: Bars) -> Path:
    """A catalog holding ``bars`` as 1m LAST only (the refinement's series, R19)."""
    catalog = tmp_path / "catalog"
    write_last_bars(catalog, bars)
    return catalog


def d1_runner(
    tmp_path: Path, mode: str, catalog: Path | None, *, leverage: float = 10.0
) -> NTScreeningRunner:
    runner = screening_runner(tmp_path, mode, catalog, leverage=leverage)
    runner._all_timeframes = {"1d"}
    return runner


def extract(
    runner: NTScreeningRunner,
    positions: list[StubPos],
    bars: Bars,
    orders: dict[str, Any],
    balance: float = BALANCE,
) -> Any:
    engine = engine_with_orders(positions, bars, orders)
    return runner._extract_metrics({}, bt_result(len(positions)), engine, time.time(), balance)


def sl_orders(accepted_at: int | None = None) -> dict[str, Any]:
    return {"SL-1": stop_order(accepted_at=at(60) if accepted_at is None else accepted_at)}


# --------------------------------------------------- R3/R3b/R3c: the refined window edge


def test_1d_refined_close_before_0800_pays_nothing(tmp_path: Path) -> None:
    """R3: the refined exit at 07:50:59.999 is before the first 8h settlement, so the position
    pays nothing (pre-fix: the NT bar close at 23:59:59.999 charged 08:00 and 16:00 = 8.0)."""
    catalog = last_bars_catalog(tmp_path, BEFORE_0800)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [d1_position()], BEFORE_0800, sl_orders())

    assert m.total_funding == 0.0
    assert m.funding_fallback_settlements == 0


def test_1d_refined_close_after_0800_pays_one_settlement(tmp_path: Path) -> None:
    """R3b: a low at the 08:00 bar refines to 08:00:59.999, so the 08:00 settlement is charged
    (4.0, one fallback) and the 16:00 one is not."""
    bars = flat_bars({480: (40000.0, 40000.0, 39400.0, 40000.0)})
    catalog = last_bars_catalog(tmp_path, bars)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [d1_position()], bars, sl_orders())

    assert m.total_funding == pytest.approx(4.0)
    assert m.funding_fallback_settlements == 1


def test_1d_refined_close_one_ms_before_settlement(tmp_path: Path) -> None:
    """R3c: the 1m ``ts_init`` ends in :59.999, so a fill refined to 07:59:59.999 is ONE ms
    before 08:00: the settlement window ``(entry, exit]`` charges nothing."""
    bars = flat_bars({479: (40000.0, 40000.0, 39400.0, 40000.0)})
    catalog = last_bars_catalog(tmp_path, bars)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [d1_position()], bars, sl_orders())

    assert m.total_funding == 0.0
    assert m.funding_fallback_settlements == 0


# ----------------------------------------------------------- R4/R5: kept bar close, limits


def test_1d_kept_bar_close_keeps_nt_funding(tmp_path: Path) -> None:
    """R4: a 1m series that never reaches 39500 leaves the fill at its NT bar close, so the
    08:00 and 16:00 settlements are charged as before (8.0, two fallbacks)."""
    bars = flat_bars()
    catalog = last_bars_catalog(tmp_path, bars)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [d1_position()], bars, sl_orders())

    assert m.total_funding == pytest.approx(8.0)
    assert m.funding_fallback_settlements == 2


def test_1d_limit_exit_strict_penetration(tmp_path: Path) -> None:
    """R5: a SELL LIMIT 40500 refines when the 1m HIGH strictly exceeds it (40600 -> 0.0); a
    high of exactly 40500 is only a touch, so the fill keeps its NT ts (8.0)."""
    penetrate = flat_bars({470: (40000.0, 40600.0, 40000.0, 40000.0)})
    catalog = last_bars_catalog(tmp_path, penetrate)
    runner = d1_runner(tmp_path, "off", catalog)
    orders = {
        "TP-1": limit_order("TP-1", 40500.0, accepted_at=at(60), side=OrderSide.SELL)
    }
    pos = d1_position(order_id="TP-1", order_type=OrderType.LIMIT, exit_px=40500.0)
    m = extract(runner, [pos], penetrate, orders)
    assert m.total_funding == 0.0

    touch = flat_bars({470: (40000.0, 40500.0, 40000.0, 40000.0)})
    catalog_touch = last_bars_catalog(tmp_path / "touch", touch)
    runner_touch = d1_runner(tmp_path / "touch", "off", catalog_touch)
    pos_touch = d1_position(order_id="TP-1", order_type=OrderType.LIMIT, exit_px=40500.0)
    m_touch = extract(runner_touch, [pos_touch], touch, orders)
    assert m_touch.total_funding == pytest.approx(8.0)


# ------------------------------------------------- R6/R7: the CLOSING fill decides the exit


def test_refined_partial_exit_does_not_move_funding(tmp_path: Path) -> None:
    """R6: a HALF exit refined to 07:50:59.999 with a MARKET close of the other half at 01:00
    day 1 funds to ``ts_closed`` (the closing fill is not refined): 08:00, 16:00, 00:00 = 12.0
    on the peak qty 1."""
    n = at(D1_CLOSE_MIN) - ONE_MS
    fills = [
        Fill(at(60), OrderSide.BUY, 1.0, 40000.0, 0.0, OrderType.MARKET, None),
        Fill(n, OrderSide.SELL, 0.5, SL_LEVEL, 0.0, OrderType.STOP_MARKET, ClientOrderId("SL-1")),
        Fill(at(1500), OrderSide.SELL, 0.5, 40000.0, 0.0, OrderType.MARKET, None),
    ]
    pos = position_with(fills, qty=1.0)
    catalog = last_bars_catalog(tmp_path, BEFORE_0800)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [pos], BEFORE_0800, sl_orders())

    assert m.total_funding == pytest.approx(12.0)


def test_closing_fill_refined_ts_is_the_funding_exit(tmp_path: Path) -> None:
    """R7: SL-1 0.5 refines to 07:50:59.999 (fill 1) and TP-1 0.5 to 12:10:59.999 (fill 2):
    the CLOSING fill (index 2) is the funding exit, so only 08:00 is charged (4.0)."""
    n = at(D1_CLOSE_MIN) - ONE_MS
    bars = flat_bars(
        {
            470: (40000.0, 40000.0, 39400.0, 40000.0),
            730: (40000.0, 40600.0, 40000.0, 40000.0),
        }
    )
    fills = [
        Fill(at(60), OrderSide.BUY, 1.0, 40000.0, 0.0, OrderType.MARKET, None),
        Fill(n, OrderSide.SELL, 0.5, SL_LEVEL, 0.0, OrderType.STOP_MARKET, ClientOrderId("SL-1")),
        Fill(n, OrderSide.SELL, 0.5, 40500.0, 0.0, OrderType.LIMIT, ClientOrderId("TP-1")),
    ]
    pos = position_with(fills, qty=1.0)
    orders = {
        "SL-1": stop_order(accepted_at=at(60)),
        "TP-1": limit_order("TP-1", 40500.0, accepted_at=at(60), side=OrderSide.SELL),
    }
    catalog = last_bars_catalog(tmp_path, bars)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [pos], bars, orders)

    assert m.total_funding == pytest.approx(4.0)


# ------------------------------------------------ R8: an OPEN position is never refined


def test_open_position_funding_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R8: ``_funding_close_ns`` is never called for an open position, so an empty refine set
    (no instrument refined) changes nothing."""
    import vibe_quant.screening.nt_runner as nt_runner

    bars = BEFORE_0800
    catalog = last_bars_catalog(tmp_path, bars)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [open_position()], bars, {}, balance=BALANCE)

    monkeypatch.setattr(nt_runner, "funding_refine_instruments", lambda *_a, **_k: frozenset())
    runner2 = d1_runner(tmp_path, "off", catalog)
    m2 = extract(runner2, [open_position()], bars, {}, balance=BALANCE)

    assert m.total_funding > 0.0
    assert m.total_funding == m2.total_funding
    assert m.funding_fallback_settlements == m2.funding_fallback_settlements


# ---------------------------------------------------------- R9/R9b/R9c/R9d: the 8h scope


def test_4h_mode_off_unchanged_and_import_isolated() -> None:
    """R9: a 4h execution bar at or below 8h is never refined: mode off imports NO
    ``vibe_quant.liquidation*`` module even with a boundary STOP_MARKET fill."""
    code = (
        "import sys, time\n"
        "from types import SimpleNamespace as NS\n"
        "from nautilus_trader.model.enums import OrderSide, OrderType\n"
        "from nautilus_trader.model.identifiers import ClientOrderId\n"
        "from vibe_quant.screening.nt_runner import NTScreeningRunner\n"
        "class F:\n"
        "    def __init__(self, ts, side, qty, px, ot, coid=None):\n"
        "        self.ts_event=ts; self.order_side=side; self.last_qty=qty; self.last_px=px\n"
        "        self.commission=0.0; self.order_type=ot; self.client_order_id=coid\n"
        "class P:\n"
        "    def __init__(self):\n"
        "        self.id='A'; self.instrument_id='BTCUSDT-PERP.BINANCE'\n"
        "        self.ts_opened=0; self.ts_closed=0; self.events=[]; self.realized_pnl=0.0\n"
        "        self.is_closed=True; self.entry=OrderSide.BUY; self.avg_px_open=40000.0\n"
        "        self.peak_qty=1.0; self.signed_qty=0.0\n"
        "    @property\n"
        "    def is_open(self):\n"
        "        return not self.is_closed\n"
        "    def commissions(self):\n"
        "        return []\n"
        "START=1735689600000000000; H=3600*10**9\n"
        "p=P(); p.ts_opened=START+H; n=START+16*H-1_000_000; p.ts_closed=n\n"
        "p.events=[F(p.ts_opened, OrderSide.BUY, 1.0, 40000.0, OrderType.MARKET),\n"
        "          F(n, OrderSide.SELL, 1.0, 39500.0, OrderType.STOP_MARKET, ClientOrderId('SL-1'))]\n"
        "cache=NS(positions=lambda: [p], position_snapshots=list, bars=list)\n"
        "engine=NS(kernel=NS(cache=cache))\n"
        "r=NTScreeningRunner({}, ['BTCUSDT'], '2025-01-01', '2025-01-02')\n"
        "r._all_timeframes={'4h'}\n"
        "bt=NS(total_positions=1, elapsed_time=0.0, stats_pnls={}, stats_returns={})\n"
        "r._extract_metrics({}, bt, engine, time.time(), 1000.0)\n"
        "bad=sorted(m for m in sys.modules if m=='vibe_quant.liquidation' or m.startswith('vibe_quant.liquidation.'))\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_funding_close_follows_the_refine_set() -> None:
    """R9b/R13: the closing fill's refined instant is the funding exit only for an instrument
    in the refine set; an empty set, another instrument's set and no scan keep ``ts_closed``."""
    pos = d1_position()
    refined = at(471) - ONE_MS
    scan = {key_of(pos): {1: refined}}
    assert _funding_close_ns(pos, scan, frozenset({BTC})) == refined
    assert _funding_close_ns(pos, scan, frozenset()) == int(pos.ts_closed)
    assert _funding_close_ns(pos, scan, frozenset({ETH})) == int(pos.ts_closed)
    assert _funding_close_ns(pos, None, frozenset({BTC})) == int(pos.ts_closed)
    assert _funding_close_ns(pos, {}, frozenset({BTC})) == int(pos.ts_closed)
    # fill 0 is the entry: a refined entry index is not the closing fill (R6)
    assert _funding_close_ns(pos, {key_of(pos): {0: refined}}, frozenset({BTC})) == int(
        pos.ts_closed
    )


def test_12h_bar_refines_settlement_inside_bar(tmp_path: Path) -> None:
    """R9c: a 12h bar (aligned 00:00/12:00) with entry 01:00 and a boundary SL at 11:59:59.999
    refined to 07:50:59.999 pays nothing; at the NT ts it pays the 08:00 settlement (4.0)."""
    start = at(0)
    n = start + 12 * 3_600 * 1_000_000_000 - ONE_MS
    refined = start + (7 * 3_600 + 50 * 60 + 59) * 1_000_000_000 + 999 * ONE_MS
    fills = [
        Fill(start + 3_600 * 1_000_000_000, OrderSide.BUY, 1.0, 40000.0, 0.0, OrderType.MARKET),
        Fill(n, OrderSide.SELL, 1.0, SL_LEVEL, 0.0, OrderType.STOP_MARKET, ClientOrderId("SL-1")),
    ]
    pos = position_with(fills, qty=1.0)

    assert _funding_close_ns(pos, {key_of(pos): {1: refined}}, frozenset({BTC})) == refined
    calc = FundingCalculator(str(empty_archive(tmp_path / "none.db")))
    entry_ns = int(pos.ts_opened)
    assert calc.accrue(str(pos.instrument_id), "LONG", 40000.0, entry_ns, refined).total == 0.0
    assert calc.accrue(str(pos.instrument_id), "LONG", 40000.0, entry_ns, n).total == pytest.approx(
        4.0
    )


def test_unsupported_timeframe_is_not_refined() -> None:
    """R9d: ``"12h"`` is not in ``_TIMEFRAME_SECONDS``, so ``_funding_scan_ts`` returns None
    without importing ``vibe_quant.liquidation*``."""
    code = (
        "import sys\n"
        "from vibe_quant.screening.nt_runner import NTScreeningRunner\n"
        "r=NTScreeningRunner({}, ['BTCUSDT'], '2025-01-01', '2025-01-02')\n"
        "assert r._funding_scan_ts(None, [], '12h', None) is None\n"
        "bad=sorted(m for m in sys.modules if m=='vibe_quant.liquidation' or m.startswith('vibe_quant.liquidation.'))\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr

    pos = d1_position()
    runner = NTScreeningRunner({}, ["BTCUSDT"], "2025-01-01", "2025-01-04")
    assert runner._funding_scan_ts(None, [pos], "12h", None) is None


# ------------------------------------------------------- R10/R10b/R10c/R10d: fail closed


def test_1d_mode_off_missing_1m_fails_closed(tmp_path: Path) -> None:
    """R10: a 1d run with a boundary resting exit and no 1m LAST raises MissingExitBars (a
    DataUnavailableError, so ``__call__`` re-raises it instead of scoring -inf)."""
    catalog = write_marks(tmp_path, flat_bars())  # MARK bars only: no 1m LAST directory
    runner = d1_runner(tmp_path, "off", catalog)
    with pytest.raises(
        MissingExitBars,
        match=r"^BTCUSDT: no 1m LAST series to refine the boundary fill A@\d+ fill 1$",
    ):
        extract(runner, [d1_position()], flat_bars(), sl_orders())


def test_1d_mode_off_missing_1m_call_reraises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R10 (``__call__`` variant): the DataUnavailableError a failed refinement raises is
    re-raised, never masked as a -inf row."""
    runner = d1_runner(tmp_path, "off", None)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise MissingExitBars("BTCUSDT: no 1m LAST series to refine the boundary fill A@0 fill 1")

    monkeypatch.setattr(runner, "_run_backtest", boom)
    with pytest.raises(MissingExitBars):
        runner({})


def test_1d_mode_off_refinement_error_fails_closed(tmp_path: Path) -> None:
    """R10b: an unusable boundary order (SL-1 absent from the cache) becomes a
    DataUnavailableError naming the timeframe, with the ValueError as ``__cause__``."""
    catalog = last_bars_catalog(tmp_path, BEFORE_0800)
    pos = d1_position()
    runner = d1_runner(tmp_path, "off", catalog)
    with pytest.raises(DataUnavailableError) as ei:
        extract(runner, [pos], BEFORE_0800, {})
    assert str(ei.value) == (
        f"funding exit refinement (1d): ValueError: {key_of(pos)} fill 1: "
        "order SL-1 is not in the engine cache"
    )
    assert isinstance(ei.value.__cause__, ValueError)


def test_1d_mode_off_without_catalog_fails_closed(tmp_path: Path) -> None:
    """R10c: no catalog means no 1m LAST, so the refinement raises MissingExitBars."""
    runner = d1_runner(tmp_path, "off", None)
    with pytest.raises(MissingExitBars):
        extract(runner, [d1_position()], BEFORE_0800, sl_orders())


def test_1d_without_resting_exits_needs_no_1m(tmp_path: Path) -> None:
    """R10d: MARKET-only exits are excluded before any rejection, so no 1m series is needed and
    funding stays at the NT bar close (8.0)."""
    n = at(D1_CLOSE_MIN) - ONE_MS
    fills = [
        Fill(at(60), OrderSide.BUY, 1.0, 40000.0, 0.0, OrderType.MARKET, None),
        Fill(n, OrderSide.SELL, 1.0, 40000.0, 0.0, OrderType.MARKET, None),
    ]
    pos = position_with(fills, qty=1.0)
    catalog = write_marks(tmp_path, flat_bars())  # no 1m LAST anywhere
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [pos], flat_bars(), {})

    assert m.total_funding == pytest.approx(8.0)


# ------------------------------------------------------------ R11/R12: mode on


def test_1d_mode_on_identity_equals_mode_off(tmp_path: Path) -> None:
    """R11: with a calm MARK series the overlay is the identity, so the mode-on run (whose scan
    refines the same exit) equals the mode-off run field for field; both pay 0.0."""
    bars = BEFORE_0800
    catalog = marks_and_last(tmp_path, bars)
    off = extract(
        d1_runner(tmp_path, "off", catalog), [d1_position()], bars, sl_orders(), BALANCE
    )
    on = extract(
        d1_runner(tmp_path, "isolated", catalog), [d1_position()], bars, sl_orders(), BALANCE
    )

    assert equal_metrics(on, off)
    assert on.total_funding == 0.0
    assert off.total_funding == 0.0


def test_1d_mode_on_scan_funding_stops_at_refined_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R12: mode on, the overlay's funding for the CLOSED position stops at its refined exit,
    and the truncated ledgers reconcile with the reported return and the gross identity.

    A's closing fill sits in day 0's 1d bar; B's liquidation is in day 2's, so the overlay can
    replay each 1d bar alone (a position with an event in the hit's bar fails closed)."""
    import vibe_quant.liquidation.apply as apply_module
    import vibe_quant.validation.extraction as extraction

    bars = flat_bars(
        {
            REFINE_MIN: (40000.0, 40000.0, 39400.0, 40000.0),
            2 * D1_CLOSE_MIN + 900: (40000.0, 40000.0, 30000.0, 40000.0),
        }
    )
    catalog = marks_and_last(tmp_path, bars)
    funding_calls: list[dict[str, list[tuple[int, float]]]] = []
    ledgers_calls: list[Any] = []
    real_apply = apply_module.apply_liquidation
    real_paths = extraction.mark_to_market_paths

    def spy_apply(*args: Any, **kwargs: Any) -> Any:
        funding_calls.append(kwargs["funding_by_position"])
        return real_apply(*args, **kwargs)

    def spy_paths(*args: Any, **kwargs: Any) -> Any:
        ledgers_calls.append(kwargs["ledgers"])
        return real_paths(*args, **kwargs)

    monkeypatch.setattr(apply_module, "apply_liquidation", spy_apply)
    monkeypatch.setattr(extraction, "mark_to_market_paths", spy_paths)
    closed = d1_position()
    closed.realized_pnl = -500.0  # the stub's NT realized PnL must match its fills
    runner = d1_runner(tmp_path, "isolated", catalog)
    m = extract(
        runner, [closed, open_position(open_min=D1_CLOSE_MIN + 600)], bars, sl_orders(), BALANCE
    )

    refined = at(471) - ONE_MS
    # the closed position's funding has NO payment after its refined exit (here: none at all)
    assert all(ts <= refined for ts, _amt in funding_calls[0].get(key_of(closed), []))
    assert m.liquidation_count == 1
    assert ledgers_calls and ledgers_calls[0] is not None
    assert ledger_net(ledgers_calls[0]) == pytest.approx(m.total_return * BALANCE, abs=1e-9)
    assert m.gross_pnl == pytest.approx(
        m.total_return * BALANCE + m.total_fees + m.total_funding
    )


# ------------------------------------------------- R14/R15/R16/R16b: scope and records


def test_realized_cash_keeps_nt_ts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R15: the realized cash flow keeps the NT fill ts; only the funding cash moves to the
    refined instant, and the gross identity holds."""
    import vibe_quant.validation.extraction as extraction

    catalog = last_bars_catalog(tmp_path, BEFORE_0800)
    captured: list[list[tuple[int, float]]] = []
    real = extraction.daily_sharpe_sortino

    def spy(sb: float, events: Any, start: int, end: int) -> Any:
        captured.append(list(events))
        return real(sb, events, start, end)

    monkeypatch.setattr(extraction, "daily_sharpe_sortino", spy)
    pos = d1_position()
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [pos], BEFORE_0800, sl_orders())

    n = int(pos.ts_closed)
    assert (n, 0.0) in captured[0]  # realized cash at the NT ts
    assert all(ts <= at(471) - ONE_MS for ts, _amt in captured[0] if ts != n)
    assert m.gross_pnl == pytest.approx(
        m.total_return * BALANCE + m.total_fees + m.total_funding
    )


def test_mode_off_refinement_records_no_exit_timing(tmp_path: Path) -> None:
    """R16: mode off records no exit timing: ``_last_exit_timing_notes`` stays the zero notes
    and ``metrics.exit_timing`` is None."""
    catalog = last_bars_catalog(tmp_path, BEFORE_0800)
    runner = d1_runner(tmp_path, "off", catalog)
    m = extract(runner, [d1_position()], BEFORE_0800, sl_orders())

    assert runner._last_exit_timing_notes == _no_exit_timing()
    assert m.exit_timing is None


def test_kept_bar_close_warning_names_its_purpose(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R16b: the kept-bar-close WARNING names the purpose (funding vs liquidation scan) and
    carries the ``liquidation_mode`` suffix ONLY for the liquidation scan."""
    bars = flat_bars()
    catalog = last_bars_catalog(tmp_path, bars)
    runner = d1_runner(tmp_path, "off", catalog)
    with caplog.at_level(logging.WARNING, logger="vibe_quant.screening.nt_runner"):
        extract(runner, [d1_position()], bars, sl_orders())
    funding = [r.getMessage() for r in caplog.records if "Exit refinement" in r.getMessage()]
    assert len(funding) == 1
    assert "Exit refinement for the funding: 1 of 1" in funding[0]
    assert "kept their NT strategy-bar timestamp" in funding[0]
    assert "liquidation_mode" not in funding[0]

    caplog.clear()
    catalog_on = marks_and_last(tmp_path / "on", bars)
    runner_on = d1_runner(tmp_path / "on", "isolated", catalog_on)
    with caplog.at_level(logging.WARNING, logger="vibe_quant.screening.nt_runner"):
        extract(runner_on, [d1_position()], bars, sl_orders())
    scan = [r.getMessage() for r in caplog.records if "Exit refinement" in r.getMessage()]
    assert len(scan) == 1
    assert "for the liquidation scan" in scan[0]
    assert "(liquidation_mode=isolated)" in scan[0]


# --------------------------------------- nxmxd.3: refine when the bar outlasts the coin's interval


def interval_archive(path: Path, rows: dict[str, list[tuple[int, int]]], rate: float = 0.0003) -> Path:
    """An archive holding ``(funding_time_ms, interval_hours)`` rows per symbol at ``rate``."""
    archive = RawDataArchive(path)
    try:
        for symbol, entries in rows.items():
            archive.insert_funding_rates(symbol, [(ms, rate, None) for ms, _h in entries], "test")
            archive.insert_funding_intervals(symbol, entries, "test")
    finally:
        archive.close()
    return path


def every(hours: int, *, offset_h: int = 0, first_h: int = -24, last_h: int = 120) -> list[tuple[int, int]]:
    """Settlement rows every ``hours`` h from START+first_h to START+last_h, ``hours`` interval."""
    return [(START_MS + t * H_MS, hours) for t in range(first_h + offset_h, last_h, hours)]


def refine_set(
    archive: Path,
    ids: list[str],
    bar_ns: int,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> frozenset[str]:
    return trade_ledger.funding_refine_instruments(  # type: ignore[attr-defined]
        FundingCalculator(archive), ids, bar_ns, start_ns, end_ns
    )


def test_two_hour_funding_refines_a_4h_exit(tmp_path: Path) -> None:
    """R: SOL paid every 2h, so a 4h bar outlasts its interval and is refined; BTC on 8h rows is
    not (over two unrelated coins)."""
    archive = interval_archive(
        tmp_path / "a.db", {"SOLUSDT": every(2), "BTCUSDT": every(8), "ETHUSDT": every(2)}
    )
    assert refine_set(archive, [SOL, BTC], E4H) == frozenset({SOL})
    assert refine_set(archive, [BTC, ETH], E4H) == frozenset({ETH})
    assert refine_set(archive, [BTC], E4H) == frozenset()


def test_equal_interval_is_not_refined(tmp_path: Path) -> None:
    """R5/R6: the guard is a strict ``>``: bar == interval and bar < interval keep ts_closed."""
    archive = interval_archive(tmp_path / "a.db", {"HYPEUSDT": every(4), "SOLUSDT": every(2)})
    hype = "HYPEUSDT-PERP.BINANCE"
    assert refine_set(archive, [hype], E4H) == frozenset()
    assert refine_set(archive, [hype], E4H + 1) == frozenset({hype})
    assert refine_set(archive, [hype], E1H) == frozenset()  # 1h bar on 4h funding
    assert refine_set(archive, [SOL], E2H) == frozenset()
    assert refine_set(archive, [SOL], E1H) == frozenset()  # 1h bar on 2h funding (R6)
    assert refine_set(archive, [SOL], E2H + 1) == frozenset({SOL})


def test_unarchived_symbol_uses_eight_hours(tmp_path: Path) -> None:
    """R7: no archived rows -> the 8h default: 4h and exactly 8h are not refined, >8h is."""
    archive = interval_archive(tmp_path / "a.db", {"SOLUSDT": every(2)})
    assert refine_set(archive, [BTC], E4H) == frozenset()
    assert refine_set(archive, [BTC], E8H) == frozenset()
    assert refine_set(archive, [BTC], E8H + 1) == frozenset({BTC})
    assert refine_set(archive, [BTC, ETH], E1D) == frozenset({BTC, ETH})
    assert refine_set(archive, [BTC], E12H) == frozenset({BTC})


def test_one_minute_bar_reads_no_archive(tmp_path: Path) -> None:
    """A 1m (or finer) bar has nothing to refine: empty set, and the (missing) archive is never
    read; a 2m bar with a missing archive fails closed."""
    gone = tmp_path / "missing.db"
    assert refine_set(gone, [BTC, SOL], 60_000_000_000) == frozenset()
    assert refine_set(gone, [BTC], 1) == frozenset()
    with pytest.raises(DataUnavailableError):
        refine_set(gone, [BTC], 120_000_000_000)


def _edge_rows(special: dict[int, int]) -> list[tuple[int, int]]:
    """Eight rows 8h apart from START (index 0..7), 8h each except ``special`` index -> hours."""
    return [(START_MS + 8 * i * H_MS, special.get(i, 8)) for i in range(8)]


def _t(i: int) -> int:
    return (START_MS + 8 * i * H_MS) * 1_000_000


@pytest.mark.parametrize(
    ("start_ns", "end_ns", "expected"),
    [
        (_t(1) - 1, None, True),  # 2h row 0 is the row before the window start (lo - 1)
        (_t(1), None, True),  # start == row 1: bisect_left keeps row 0 as lo - 1
        (_t(1) + 1, None, False),  # start 1 ns past row 1: lo - 1 is row 1 (8h), row 0 is out
        (_t(3), None, False),  # two rows before the start never count
        (None, None, True),  # whole archive
        (_t(1) + 1, _t(7), False),  # 2h row 0 outside: window rows are all 8h
    ],
)
def test_interval_row_before_window_counts(
    tmp_path: Path, start_ns: int | None, end_ns: int | None, expected: bool
) -> None:
    """R8: the row just before the window start counts (over-inclusion costs a 1m load, never a
    number); rows two or more before it do not."""
    archive = interval_archive(tmp_path / "a.db", {"SOLUSDT": _edge_rows({0: 2})})
    got = refine_set(archive, [SOL], E4H, start_ns, end_ns)
    assert got == (frozenset({SOL}) if expected else frozenset())


@pytest.mark.parametrize(
    ("end_ns", "expected"),
    [
        (_t(2), False),  # hi = 3: rows <= 3 only, the 2h row 4 is two past the end
        (_t(3) - 1, False),  # hi = 3 still
        (_t(3), True),  # end == row 3: hi = 4, the one-past row 4 (2h) counts
        (_t(4) - 1, True),  # hi = 4
        (_t(4), True),  # end == row 4 (inside the window): hi = 5
    ],
)
def test_interval_row_after_window_counts(tmp_path: Path, end_ns: int, expected: bool) -> None:
    """The row just after the window end counts too; two rows after does not."""
    archive = interval_archive(tmp_path / "a.db", {"SOLUSDT": _edge_rows({4: 2})})
    got = refine_set(archive, [SOL], E4H, _t(0), end_ns)
    assert got == (frozenset({SOL}) if expected else frozenset())


# ------------------------------- nxmxd.3: a 4h run on a 2h-funding coin, through the extractor

H4_CLOSE_MIN = 240  # the first 4h bar closes at 04:00; a boundary exit is stamped 03:59:59.999
H4_REFINE_MIN = 90  # the 1m bar opening 01:30 reaches the stop: refined exit 01:30:59.999
H4_BARS = flat_bars({H4_REFINE_MIN: (40000.0, 40000.0, 39400.0, 40000.0)})
SETTLEMENT = 12.0  # rate 0.0003 x 40000 notional, a LONG pays
H4_ENTRY_MIN = 10


def h4_position(*, pid: str = "A", order_id: str = "SL-1", instrument: str = BTC) -> StubPos:
    """1 BTC long entered 00:10, stopped on the 4h boundary (03:59:59.999)."""
    return d1_position(
        pid=pid,
        open_min=H4_ENTRY_MIN,
        order_id=order_id,
        n_min=H4_CLOSE_MIN,
        instrument=instrument,
    )


def h4_runner(
    tmp_path: Path, mode: str, catalog: Path | None, rows: dict[str, list[tuple[int, int]]]
) -> NTScreeningRunner:
    runner = screening_runner(tmp_path, mode, catalog)
    runner._all_timeframes = {"4h"}
    runner._funding_archive_path = str(interval_archive(tmp_path / "funding.db", rows))
    return runner


def h4_orders(*ids: str) -> dict[str, Any]:
    return {i: stop_order(i, accepted_at=at(H4_ENTRY_MIN)) for i in ids}


def test_4h_exit_on_two_hour_funding_stops_at_the_1m_fill(tmp_path: Path) -> None:
    """R3: stopped at 01:30 (1m), BTC paid every 2h: the 02:00 settlement is NOT charged (0.0),
    where the NT bar close 03:59:59.999 charged it (12.0)."""
    catalog = last_bars_catalog(tmp_path, H4_BARS)
    runner = h4_runner(tmp_path, "off", catalog, {"BTCUSDT": every(2)})
    m = extract(runner, [h4_position()], H4_BARS, h4_orders("SL-1"))

    assert m.total_funding == 0.0
    assert m.funding_fallback_settlements == 0


def test_4h_exit_on_eight_hour_funding_is_not_scanned(tmp_path: Path) -> None:
    """R1: 8h funding and a 4h bar -> empty refine set: no 1m LAST is read (the catalog holds
    none and a boundary SL does not fail), funding is the unrefined ts_closed funding."""
    catalog = write_marks(tmp_path, flat_bars())  # no 1m LAST directory
    runner = h4_runner(tmp_path, "off", catalog, {"BTCUSDT": every(8, offset_h=2)})
    m = extract(runner, [h4_position()], H4_BARS, h4_orders("SL-1"))

    assert m.total_funding == pytest.approx(SETTLEMENT)  # 02:00 charged at the NT exit, as before
    assert runner._last_exit_timing_notes == _no_exit_timing()


def test_4h_two_hour_funding_without_1m_fails_closed(tmp_path: Path) -> None:
    """R11: a refined instrument with no 1m LAST raises MissingExitBars, never a -inf row."""
    catalog = write_marks(tmp_path, flat_bars())
    runner = h4_runner(tmp_path, "off", catalog, {"BTCUSDT": every(2)})
    with pytest.raises(MissingExitBars):
        extract(runner, [h4_position()], H4_BARS, h4_orders("SL-1"))


def test_only_refine_symbols_load_1m(tmp_path: Path) -> None:
    """R9: BTC (2h funding) and ETH (8h) both stop on the boundary; only BTC is scanned. The
    catalog has BTC 1m LAST only, so scanning ETH would raise MissingExitBars; ETH's funding is
    the unrefined one (02:00 charged, 12.0) and BTC's is refined (0.0)."""
    catalog = last_bars_catalog(tmp_path, H4_BARS)
    rows = {"BTCUSDT": every(2), "ETHUSDT": every(8, offset_h=2)}
    runner = h4_runner(tmp_path, "off", catalog, rows)
    positions = [h4_position(), h4_position(pid="E", order_id="SL-2", instrument=ETH)]
    m = extract(runner, positions, H4_BARS, h4_orders("SL-1", "SL-2"))

    assert m.total_funding == pytest.approx(SETTLEMENT)  # BTC 0.0 + ETH 12.0


def test_mode_on_funding_uses_refine_set(tmp_path: Path) -> None:
    """R10: mode on with an identity overlay funds the 2h coin to the same refined instant as
    mode off (0.0)."""
    catalog = marks_and_last(tmp_path, H4_BARS)
    rows = {"BTCUSDT": every(2)}
    off = extract(h4_runner(tmp_path / "off", "off", catalog, rows), [h4_position()], H4_BARS,
                  h4_orders("SL-1"))  # fmt: skip
    on = extract(h4_runner(tmp_path / "on", "isolated", catalog, rows), [h4_position()], H4_BARS,
                 h4_orders("SL-1"))  # fmt: skip

    assert on.total_funding == off.total_funding == 0.0


def test_scan_ts_ignored_outside_refine_set(tmp_path: Path) -> None:
    """R13: mode on, the liquidation scan refines the 4h stop to 01:30 for its own purpose, but a
    coin on 4h funding (bar == interval) is not in the refine set: its funding exit stays the
    NT ts_closed, so the 02:00 settlement is charged (12.0)."""
    catalog = marks_and_last(tmp_path, H4_BARS)
    runner = h4_runner(tmp_path, "isolated", catalog, {"BTCUSDT": every(4, offset_h=2)})
    m = extract(runner, [h4_position()], H4_BARS, h4_orders("SL-1"))

    assert m.total_funding == pytest.approx(SETTLEMENT)
