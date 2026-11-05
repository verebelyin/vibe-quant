"""Screening/validation parity for the ``max_bars_in_trade`` time exit.

Design § 4 T3, § 6 R16 and § 7 R16b/R9b/R10b. Both tiers compile the same generated class,
so the time exit must fill at the SAME instants in both. Parity is read from the EXECUTED
fills in each engine's cache (every position's open/close fill ``ts_event``, prices and
closing order type), never from serialized result lists: validation's serialization marks a
position still open at the run end closed, and that mark is not a fill (R16b). Screening's
cache shows such a position OPEN.

Price parity: timestamps are exact; each engine may draw up to ``max_adverse_ticks=1``
slippage per fill independently, so the prices of one fill differ by at most 2 ticks
(chief ruling R16c; 1m N=3: screening closes 10031.5 where validation closes 10031.3).

The toy catalog/strategy helpers come from the T2 screening tests; a validation run uses the
named ``catalog_path``, the same value ``DEFAULT_CATALOG_PATH`` would resolve to.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from tests.unit.test_max_bars_exit import (
    _STARVED_BOUNDARY,
    _Close,
    _dsl,
    _ns,
    _Order,
    _Position,
    _utc,
    _write_gapped_catalog,
)
from tests.unit.test_screening_fill_mode import _END, _START, _write_catalog
from vibe_quant.db.state_manager import StateManager
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.screening.types import BacktestMetrics
    from vibe_quant.validation.results import ValidationResult

_TICK = 0.1  # BTCUSDT-PERP price increment in the toy catalog
_PRICE_TOL = 2 * _TICK + 1e-6  # R16c: two independent 1-tick draws, float slack
_MIN_NS = 60_000_000_000


class _Fills(NamedTuple):
    """What one engine actually executed: positions, filled reduce-only closes, orders."""

    positions: list[_Position]
    closes: list[_Close]
    orders: list[_Order]


def _fills_from_engine(engine: Any) -> _Fills:
    """Read positions and orders from the engine cache (no serialized result involved)."""
    cache = engine.kernel.cache
    order_types = {str(o.client_order_id): o.order_type.name for o in cache.orders()}
    rows: dict[tuple[str, int, int | None], _Position] = {}
    for pos in list(cache.positions()) + list(cache.position_snapshots()):
        symbol = str(pos.instrument_id).split("-")[0]
        closed = bool(pos.is_closed)
        key = (symbol, int(pos.ts_opened), int(pos.ts_closed) if closed else None)
        rows[key] = _Position(
            symbol=symbol,
            ts_open=int(pos.ts_opened),
            ts_close=int(pos.ts_closed) if closed else None,
            px_open=float(pos.avg_px_open),
            px_close=float(pos.avg_px_close) if closed else None,
            closing=order_types.get(str(pos.closing_order_id)) if closed else None,
        )
    orders = sorted(
        (
            _Order(
                symbol=str(o.instrument_id).split("-")[0],
                reduce_only=bool(o.is_reduce_only),
                type=o.order_type.name,
                side=o.side.name,
                filled=float(o.filled_qty),
                px=float(o.avg_px),
                ts_last=int(o.ts_last),
                status=o.status.name,
            )
            for o in cache.orders()
        ),
        key=lambda o: (o.ts_last, o.symbol, o.type),
    )
    return _Fills(
        positions=sorted(rows.values(), key=lambda r: (r.ts_open, r.symbol)),
        closes=[
            _Close(symbol=o.symbol, ts=o.ts_last, px=o.px)
            for o in orders
            if o.reduce_only and o.filled > 0
        ],
        orders=orders,
    )


def _screening(
    catalog: Path,
    dsl: dict[str, Any],
    *,
    end: str = _END,
    symbols: tuple[str, ...] = ("BTCUSDT",),
) -> tuple[_Fills, BacktestMetrics]:
    """Run the public screening entry point; the cache is read from its extraction hook."""
    seen: list[_Fills] = []
    orig = NTScreeningRunner._extract_metrics

    def spy(
        self: NTScreeningRunner,
        params: dict[str, float | int],
        bt_result: Any,
        engine: Any,
        start_time: float,
        starting_balance: float = 1000.0,
        cost_ledger: Any = None,
    ) -> BacktestMetrics:
        seen.append(_fills_from_engine(engine))
        return orig(
            self, params, bt_result, engine, start_time,
            starting_balance=starting_balance, cost_ledger=cost_ledger,
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(NTScreeningRunner, "_extract_metrics", spy)
        metrics = NTScreeningRunner(
            dsl, list(symbols), _START, end, catalog_path=str(catalog)
        )({})
    return seen[0], metrics


def _validation(
    tmp_path: Path,
    catalog: Path,
    dsl: dict[str, Any],
    *,
    end: str = _END,
    symbols: tuple[str, ...] = ("BTCUSDT",),
) -> tuple[_Fills, ValidationResult]:
    """Run the public validation entry point; the cache is read from its extraction hook."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "state.db"
    state = StateManager(db)
    strategy_id = state.create_strategy(name=str(dsl["name"]), dsl_config=dsl)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=list(symbols),
        timeframe=str(dsl["timeframe"]),
        start_date=_START,
        end_date=end,
        parameters={},
    )
    state.close()

    seen: list[_Fills] = []
    orig = ValidationRunner._extract_results

    def spy(self: ValidationRunner, **kwargs: Any) -> Any:
        seen.append(_fills_from_engine(kwargs["engine"]))
        return orig(self, **kwargs)

    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", catalog_path=catalog)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ValidationRunner, "_extract_results", spy)
        try:
            result = runner.run(run_id=run_id)
        finally:
            runner.close()
    return seen[0], result


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_catalog(tmp_path_factory.mktemp("catalog"), ("BTCUSDT", "ETHUSDT"))


# ---------------------------------------------------------------------------
# R16 / R16b: identical (entry ts, exit ts) per position in both tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tf", "n", "expected", "end"),
    [("4h", 12, 2, _END), ("4h", 1, 11, _END), ("1m", 3, 288, "2024-01-02")],
    ids=["4h-12", "4h-1", "1m-3"],
)
def test_screening_and_validation_exit_at_the_same_instants(
    catalog: Path, tmp_path: Path, tf: str, n: int, expected: int, end: str
) -> None:
    dsl = _dsl(tf, name=f"parity_{tf}_{n}", n=n)
    scr, _ = _screening(catalog, dsl, end=end)
    val, result = _validation(tmp_path, catalog, dsl, end=end)

    assert len(scr.positions) == len(val.positions) == expected
    # validation serializes one trade per position, the run-end mark included (R16b)
    assert result.total_trades == expected

    assert [p.ts_open for p in scr.positions] == [p.ts_open for p in val.positions]
    assert [p.ts_close for p in scr.positions] == [p.ts_close for p in val.positions]
    assert [p.closing for p in scr.positions] == [p.closing for p in val.positions]

    for screening_fill, validation_fill in zip(scr.positions, val.positions, strict=True):
        assert screening_fill.px_open == pytest.approx(validation_fill.px_open, abs=_PRICE_TOL)
        assert (screening_fill.px_close is None) == (validation_fill.px_close is None)
        if screening_fill.px_close is not None:
            assert screening_fill.px_close == pytest.approx(
                validation_fill.px_close, abs=_PRICE_TOL
            )

    # exactly one EXECUTED close per closed position in each engine
    for fills in (scr, val):
        assert len(fills.closes) == sum(1 for p in fills.positions if p.ts_close is not None)


def test_run_end_mark_is_extraction_not_a_time_exit(catalog: Path, tmp_path: Path) -> None:
    dsl = _dsl("4h", name="parity_run_end", n=12)
    scr, _ = _screening(catalog, dsl)
    val, result = _validation(tmp_path, catalog, dsl)

    # both caches show the 2nd position OPEN: validation never fills an exit for it
    assert (scr.positions[-1].ts_open, scr.positions[-1].ts_close) == (
        _ns("01-03 16:00:59.999"),
        None,
    )
    assert (val.positions[-1].ts_open, val.positions[-1].ts_close) == (
        _ns("01-03 16:00:59.999"),
        None,
    )
    assert len(scr.closes) == len(val.closes) == 1  # only the 12-bar exit executed
    assert (scr.closes[0].ts, scr.closes[0].px) == (_ns("01-03 12:00:59.999"), 13600.4)
    assert (val.closes[0].ts, val.closes[0].px) == (_ns("01-03 12:00:59.999"), 13600.5)

    # the serialized list carries the terminal mark for the open position instead
    assert [t.exit_reason for t in result.trades] == ["signal", "end_of_data"]
    first, mark = result.trades
    assert first.exit_time == "2024-01-03T12:00:59.999000+00:00"
    assert first.exit_price == val.positions[0].px_close
    assert mark.entry_time == "2024-01-03T16:00:59.999000+00:00"
    assert mark.exit_time == "2024-01-04T23:59:59.999000+00:00"
    assert mark.exit_price == 15759.5
    # ...and the mark is NOT an N-bar exit: it is 8 bars after entry, short of N=12
    assert datetime.fromisoformat(mark.exit_time) - datetime.fromisoformat(
        mark.entry_time
    ) < timedelta(hours=12 * 4)


def test_time_exit_queued_on_the_last_bar_never_fills(catalog: Path, tmp_path: Path) -> None:
    """N=3 ends on the bar whose exit would release after the window closes (R16b)."""
    dsl = _dsl("4h", name="parity_queued", n=3)
    end = "2024-01-02"
    scr, _ = _screening(catalog, dsl, end=end)
    val, result = _validation(tmp_path, catalog, dsl, end=end)

    for fills in (scr, val):
        (position,) = fills.positions
        assert (position.ts_open, position.ts_close, position.px_close) == (
            _ns("01-01 12:00:59.999"),
            None,
            None,
        )
        # the queued exit never reached the venue: no reduce-only MARKET order exists
        assert [o for o in fills.orders if o.reduce_only and o.type == "MARKET"] == []
        assert sorted(o.type for o in fills.orders if o.reduce_only) == ["LIMIT", "STOP_MARKET"]

    (mark,) = result.trades
    assert mark.exit_reason == "end_of_data"
    assert mark.exit_time == "2024-01-01T23:59:59.999000+00:00"
    assert mark.exit_price == 11439.5


# ---------------------------------------------------------------------------
# § 7 R9b (validation part): SL/TP precedence, feature on
# ---------------------------------------------------------------------------


def test_validation_sl_before_n_keeps_a_fresh_count(catalog: Path, tmp_path: Path) -> None:
    """1m immediate mode: the SL takes the position 10 bars in, before N=12."""
    dsl = _dsl("1m", name="parity_r9b_sl", n=12, side="short", sl=0.1)
    val, result = _validation(tmp_path, catalog, dsl, end="2024-01-02")

    assert len(val.positions) == 125
    assert result.total_trades == 125
    # every entry is a SELL: the reduce-only exit never opens a reverse position
    assert {o.side for o in val.orders if not o.reduce_only and o.filled > 0} == {"SELL"}
    first, second = val.positions[0], val.positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close, first.closing) == (
        "01-01 00:03:59.999",
        "01-01 00:13:59.999",
        10013.6,
        "STOP_MARKET",
    )
    # fresh count: a carried 10-bar count would time-exit the 2nd at 00:16:59.999; it
    # lives its own 10 minutes to its SL instead
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_close) == (
        "01-01 00:14:59.999",
        "01-01 00:24:59.999",
        10024.6,
    )
    assert second.closing == "STOP_MARKET"
    assert second.ts_close is not None
    assert second.ts_close - second.ts_open == 10 * _MIN_NS
    # exactly one filled close per position: no second (time) exit landed anywhere
    assert [(c.symbol, c.ts) for c in val.closes] == [
        (p.symbol, p.ts_close) for p in val.positions if p.ts_close is not None
    ]


def test_validation_tp_before_n_keeps_a_fresh_count(catalog: Path, tmp_path: Path) -> None:
    """1m immediate mode: the take-profit takes the position 10 bars in, before N=12."""
    dsl = _dsl("1m", name="parity_r9b_tp", n=12, side="long", tp=0.1)
    val, result = _validation(tmp_path, catalog, dsl, end="2024-01-02")

    assert len(val.positions) == 124
    assert result.total_trades == 124
    first, second = val.positions[0], val.positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close, first.closing) == (
        "01-01 00:03:59.999",
        "01-01 00:13:59.999",
        10013.5,
        "LIMIT",
    )
    # fresh count: the 2nd survives past 00:16:59.999 (a carried count) to its own TP
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_close) == (
        "01-01 00:14:59.999",
        "01-01 00:24:59.999",
        10024.5,
    )
    assert second.closing == "LIMIT"
    assert second.ts_close is not None
    assert second.ts_close - second.ts_open == 10 * _MIN_NS
    assert [(c.symbol, c.ts) for c in val.closes] == [
        (p.symbol, p.ts_close) for p in val.positions if p.ts_close is not None
    ]


def test_validation_protective_fill_on_the_time_exit_release_wins(
    catalog: Path, tmp_path: Path
) -> None:
    """TP 26.863 % sits between the 12th bar's high and the release tick's price."""
    dsl = _dsl("4h", name="parity_r9b_release", n=12, tp=26.863)
    val, result = _validation(tmp_path, catalog, dsl, end="2024-01-05")

    first, second = val.positions
    assert (_utc(first.ts_close), first.px_close, first.closing) == (
        "01-03 12:00:59.999",
        13600.3,
        "LIMIT",
    )
    # exactly one filled close: the reduce-only market exit never filled
    assert val.closes == [_Close("BTCUSDT", _ns("01-03 12:00:59.999"), 13600.3)]
    # ...but it WAS submitted on the release tick and rejected as reduce-only
    assert [
        (o.type, o.filled, o.status, _utc(o.ts_last))
        for o in val.orders
        if o.reduce_only and o.type == "MARKET"
    ] == [("MARKET", 0.0, "REJECTED", "01-03 12:00:59.999")]
    # fresh count: a stale 12 would exit the 2nd position at 01-03 20:00:59.999
    assert (second.ts_open, second.ts_close) == (_ns("01-03 16:00:59.999"), None)
    assert [o.side for o in val.orders if not o.reduce_only and o.filled > 0] == ["BUY", "BUY"]
    assert [t.exit_reason for t in result.trades] == ["signal", "end_of_data"]


# ---------------------------------------------------------------------------
# § 7 R10b (validation part): the own-instrument release feed is starved, then resumes
# ---------------------------------------------------------------------------


def test_validation_missing_detail_release_waits_for_the_next_own_bar(
    tmp_path: Path,
) -> None:
    """01-03 12:00/12:01 1m detail is missing; the 4h bars stay complete.

    BTC's queued time exit waits for its own next 1m bar at 12:02:59.999; ETH's own
    datum at 12:00:59.999 does not release it, and no stale exit survives.
    """
    catalog = _write_gapped_catalog(
        tmp_path / "catalog",
        {"BTCUSDT": {_STARVED_BOUNDARY, _STARVED_BOUNDARY + 1}},
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    dsl = _dsl("4h", name="parity_r10b", n=12)
    val, result = _validation(
        tmp_path / "run", catalog, dsl, symbols=("BTCUSDT", "ETHUSDT")
    )

    btc = [p for p in val.positions if p.symbol == "BTCUSDT"]
    eth = [p for p in val.positions if p.symbol == "ETHUSDT"]
    assert (_utc(btc[0].ts_close), btc[0].px_close, btc[0].closing) == (
        "01-03 12:02:59.999",
        13602.5,
        "MARKET",
    )
    # the gap is BTC-only: ETH's exit still fills on its own 12:00 datum
    assert (_utc(eth[0].ts_close), eth[0].px_close) == ("01-03 12:00:59.999", 27201.0)
    # exactly one filled close per instrument: no extra queued exit, no reverse position
    assert {(c.symbol, _utc(c.ts)) for c in val.closes} == {
        ("BTCUSDT", "01-03 12:02:59.999"),
        ("ETHUSDT", "01-03 12:00:59.999"),
    }
    assert {o.side for o in val.orders if not o.reduce_only and o.filled > 0} == {"BUY"}
    # the starved exit never closes the next positions (both re-open and stay open)
    assert [(p.symbol, _utc(p.ts_open), p.ts_close) for p in val.positions] == [
        ("BTCUSDT", "01-01 12:00:59.999", _ns("01-03 12:02:59.999")),
        ("ETHUSDT", "01-01 12:00:59.999", _ns("01-03 12:00:59.999")),
        ("BTCUSDT", "01-03 16:00:59.999", None),
        ("ETHUSDT", "01-03 16:00:59.999", None),
    ]
    assert [t.exit_reason for t in result.trades] == [
        "signal",
        "signal",
        "end_of_data",
        "end_of_data",
    ]


def test_validation_long_detail_gap_exits_once_when_its_own_data_resumes(
    tmp_path: Path,
) -> None:
    """BTC's 4h primary bars close across a 242-minute detail hole; the exit waits it out.

    01-03 12:00..16:01 1m detail is missing for BTC, so the primary 4h bars at 16:00 and
    20:00 keep delivering while the 12:00 exit is queued. The primary bars are not the
    release: the exit fills once, on BTC's own next 1m datum at 16:02:59.999, the
    repeated submissions in the outbox cannot reverse the position, and the position
    reopened at 20:00 is still open at the run end.
    """
    catalog = _write_gapped_catalog(
        tmp_path / "catalog",
        {"BTCUSDT": set(range(_STARVED_BOUNDARY, _STARVED_BOUNDARY + 242))},
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    dsl = _dsl("4h", name="parity_r10b_long", n=12)
    val, result = _validation(tmp_path / "run", catalog, dsl, symbols=("BTCUSDT", "ETHUSDT"))

    btc = [p for p in val.positions if p.symbol == "BTCUSDT"]
    eth = [p for p in val.positions if p.symbol == "ETHUSDT"]
    assert (_utc(btc[0].ts_close), btc[0].px_close, btc[0].closing) == (
        "01-03 16:02:59.999",
        13842.5,
        "MARKET",
    )
    # exactly one filled close per closed position: nothing stale, nothing doubled
    assert [(c.symbol, _utc(c.ts), c.px) for c in val.closes] == [
        ("ETHUSDT", "01-03 12:00:59.999", 27201.0),
        ("BTCUSDT", "01-03 16:02:59.999", 13842.5),
    ]
    # BTC re-earns a fresh 12 bars from 20:00 and the run ends before its own exit
    assert (_utc(btc[1].ts_open), btc[1].ts_close) == ("01-03 20:00:59.999", None)
    assert (_utc(eth[0].ts_close), eth[0].px_close) == ("01-03 12:00:59.999", 27201.0)
    # one reduce-only SELL between two BUY entries: the duplicate exit never reversed
    assert [o.side for o in val.orders if o.symbol == "BTCUSDT" and o.filled > 0] == [
        "BUY",
        "SELL",
        "BUY",
    ]
    assert [t.exit_reason for t in result.trades] == [
        "signal",
        "signal",
        "end_of_data",
        "end_of_data",
    ]


def test_validation_detail_that_never_resumes_leaves_the_position_open(
    tmp_path: Path,
) -> None:
    """BTC's 1m detail stops at 12:00 for the rest of the run: the queued exit never fills.

    The primary 4h bars keep delivering to the run end, but no own datum exists to
    release the exit. BTC stays OPEN in the cache, no BTC reduce-only order fills, and
    validation's terminal mark for it is extraction, not a time exit.
    """
    catalog = _write_gapped_catalog(
        tmp_path / "catalog",
        {"BTCUSDT": set(range(_STARVED_BOUNDARY, 5760))},
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    dsl = _dsl("4h", name="parity_r10b_never", n=12)
    val, result = _validation(tmp_path / "run", catalog, dsl, symbols=("BTCUSDT", "ETHUSDT"))

    (btc,) = [p for p in val.positions if p.symbol == "BTCUSDT"]
    assert (btc.ts_open, btc.ts_close, btc.closing) == (_ns("01-01 12:00:59.999"), None, None)
    # the only filled close of the run is ETH's
    assert [(c.symbol, _utc(c.ts), c.px) for c in val.closes] == [
        ("ETHUSDT", "01-03 12:00:59.999", 27201.0)
    ]
    # the starved exit never reaches the venue: BTC's stop/target never fire either
    assert [o.side for o in val.orders if o.symbol == "BTCUSDT" and o.filled > 0] == ["BUY"]

    # the serialized list marks the still-open BTC position at its last own datum
    btc_trade = next(t for t in result.trades if t.exit_time == "2024-01-03T11:59:59.999000+00:00")
    assert btc_trade.entry_time == "2024-01-01T12:00:59.999000+00:00"
    assert btc_trade.exit_reason == "end_of_data"
    assert [t.exit_reason for t in result.trades] == ["signal", "end_of_data", "end_of_data"]
