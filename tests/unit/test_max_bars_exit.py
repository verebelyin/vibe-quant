"""Time exit ``max_bars_in_trade``: screening outcomes (design § 6 R1-R10, R14, R15, R17
and § 7 R9b, R10b).

Toy catalog: ``_write_catalog`` (rising +0.01 %/min, BTC 10 000 / ETH 20 000), window
``2024-01-01..2024-01-05``. Strategy: always-long ``ema(3) > 0``, SL 10 % / TP 90 % unless
a row says otherwise. Positions come from the ENGINE CACHE (``ts_event`` of the open and
close fills, their prices and the closing order type), never from serialized results; the
``total_trades`` count is the serialized cross-check. Every number below was measured with
the block the compiler now emits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from tests.unit.test_screening_fill_mode import (
    _DAY0_MS,
    _END,
    _MIN_MS,
    _START,
    _write_catalog,
)
from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.discovery.operators import (
    ConditionType as C,
)
from vibe_quant.discovery.operators import (
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.fast_tier import FastTierUnsupported
from vibe_quant.screening.nt_runner import (
    NTScreeningRunner,
    UnknownStrategyParamError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.screening.types import BacktestMetrics

_H4 = 14_400_000_000_000  # 4h in ns
_H1 = 3_600_000_000_000
_N_MIN = 5 * 1440
# 2024-01-03 12:00 = the 12th 4h boundary after the 2024-01-01 12:00 entry
_STARVED_BOUNDARY = 2 * 1440 + 12 * 60  # 1m index (3600) and its +1m neighbour (3601)


class _Position(NamedTuple):
    """One position as the engine cache reports it (fill ts_event, prices, closer)."""

    symbol: str
    ts_open: int
    ts_close: int | None
    px_open: float
    px_close: float | None
    closing: str | None


class _Close(NamedTuple):
    """One FILLED reduce-only order (a protective order or a strategy exit)."""

    symbol: str
    ts: int
    px: float


class _Order(NamedTuple):
    """One order of the run, filled or not (a rejected exit is evidence)."""

    symbol: str
    reduce_only: bool
    type: str
    side: str
    filled: float
    px: float
    ts_last: int
    status: str


def _ns(text: str) -> int:
    """UTC ``MM-DD HH:MM:SS.mmm`` in 2024 -> ns (the toy window's year)."""
    stamp = datetime.strptime("2024-" + text, "%Y-%m-%d %H:%M:%S.%f")
    return int(stamp.replace(tzinfo=UTC).timestamp() * 1_000) * 1_000_000


def _utc(ns: int | None) -> str:
    if ns is None:
        return "OPEN"
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%m-%d %H:%M:%S.%f")[:-3]


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Spy the runner's extraction hook; the run itself stays the public ``runner(...)``."""
    seen: dict[str, list[Any]] = {"positions": [], "closes": [], "entries": [], "orders": []}
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
        cache = engine.kernel.cache
        order_types = {str(o.client_order_id): o.order_type.name for o in cache.orders()}
        rows: dict[tuple[str, int, int | None], _Position] = {}
        for pos in list(cache.position_snapshots()) + list(cache.positions()):
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
        seen["positions"] = sorted(rows.values(), key=lambda r: (r.ts_open, r.symbol))
        seen["orders"] = sorted(
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
        seen["closes"] = [
            _Close(symbol=o.symbol, ts=o.ts_last, px=o.px)
            for o in seen["orders"]
            if o.reduce_only and o.filled > 0
        ]
        seen["entries"] = [
            _Close(symbol=o.symbol, ts=o.ts_last, px=o.px)
            for o in seen["orders"]
            if not o.reduce_only and o.filled > 0
        ]
        return orig(
            self, params, bt_result, engine, start_time,
            starting_balance=starting_balance, cost_ledger=cost_ledger,
        )

    monkeypatch.setattr(NTScreeningRunner, "_extract_metrics", spy)
    return seen


def _run(
    catalog: Path,
    dsl: dict[str, Any],
    *,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    end: str = _END,
) -> tuple[BacktestMetrics, list[_Position], list[_Close], list[_Order]]:
    with pytest.MonkeyPatch.context() as mp:
        seen = _capture(mp)
        metrics = NTScreeningRunner(dsl, list(symbols), _START, end, catalog_path=str(catalog))({})
    return metrics, seen["positions"], seen["closes"], seen["orders"]


def _dsl(
    tf: str = "4h",
    *,
    name: str,
    n: int | None = 12,
    side: str = "long",
    sl: float = 10.0,
    tp: float = 90.0,
    exit_condition: str | None = None,
) -> dict[str, Any]:
    """Always-in toy strategy; ``n=None`` leaves the time exit off."""
    dsl: dict[str, Any] = {
        "name": name,
        "timeframe": tf,
        "indicators": {"ema": {"type": "EMA", "period": 3}},
        "entry_conditions": {side: ["ema > 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": sl},
        "take_profit": {"type": "fixed_pct", "percent": tp},
    }
    if n is not None:
        dsl["max_bars_in_trade"] = n
    if exit_condition:
        dsl["exit_conditions"] = {side: [exit_condition]}
    return dsl


def _toy_kline(base: float, i: int) -> dict[str, Any]:
    """Rising 1m kline ``i`` of the toy: open = base*(1+1e-4 i), close +0.5 minute."""
    return {
        "open_time": _DAY0_MS + i * _MIN_MS,
        "open": base * (1 + 1e-4 * i),
        "high": base * (1 + 1e-4 * (i + 1)),
        "low": base * (1 + 1e-4 * (i - 1)),
        "close": base * (1 + 1e-4 * (i + 0.5)),
        "volume": 10.0,
        "close_time": _DAY0_MS + i * _MIN_MS + 59_999,
    }


def _write_gapped_catalog(
    root: Path,
    drops: dict[str, set[int]],
    *,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    n_min: int = _N_MIN,
    timeframes: tuple[str, ...] = ("1m", "4h"),
) -> Path:
    """Rising toy catalog with whole 1m minutes missing per symbol.

    Coarser timeframes aggregate the COMPLETE 1m series, so primary bars continue
    across a 1m hole while the symbol's fill-tick feed loses that boundary (R10b).
    """
    cm = CatalogManager(root)
    for k, symbol in enumerate(symbols):
        inst = create_instrument(symbol)
        cm.write_instrument(inst)
        base = 10_000.0 * (k + 1)
        klines: list[Any] = [_toy_kline(base, i) for i in range(n_min)]
        complete = klines_to_bars(
            klines,
            inst.id,
            get_bar_type(symbol, "1m"),
            inst.size_precision,
            inst.price_precision,
        )
        if "1m" in timeframes:
            drop = drops.get(symbol, set())
            gapped: list[Any] = [
                _toy_kline(base, i) for i in range(n_min) if i not in drop
            ]
            cm.write_bars(
                klines_to_bars(
                    gapped,
                    inst.id,
                    get_bar_type(symbol, "1m"),
                    inst.size_precision,
                    inst.price_precision,
                )
            )
        for tf, minutes in (("15m", 15), ("1h", 60), ("4h", 240)):
            if tf in timeframes:
                cm.write_bars(
                    aggregate_bars(
                        complete,
                        get_bar_type(symbol, tf),
                        minutes,
                        inst.size_precision,
                        inst.price_precision,
                    )
                )
    return root


def _entries(orders: list[_Order]) -> list[_Order]:
    """Filled entry orders (not reduce-only): their side is the position direction."""
    return [o for o in orders if not o.reduce_only and o.filled > 0]


def _assert_one_filled_close_per_position(
    positions: list[_Position], closes: list[_Close]
) -> None:
    """Exactly one filled close, at each closed position's own instant, price and symbol."""
    assert [(c.symbol, c.ts, c.px) for c in closes] == [
        (p.symbol, p.ts_close, p.px_close) for p in positions if p.ts_close is not None
    ]


def _by_symbol(positions: list[_Position]) -> dict[str, list[_Position]]:
    out: dict[str, list[_Position]] = {}
    for pos in positions:
        out.setdefault(pos.symbol, []).append(pos)
    return out


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_catalog(tmp_path_factory.mktemp("catalog"), ("BTCUSDT", "ETHUSDT"))


# ---------------------------------------------------------------------------
# R1: 4h screening holds exactly N bars, then the position stays open at end
# ---------------------------------------------------------------------------


def test_4h_screening_holds_exactly_n_bars(catalog: Path) -> None:
    metrics, positions, closes, _ = _run(catalog, _dsl("4h", name="nbar_r1", n=12))

    assert metrics.total_trades == 2
    assert [_utc(p.ts_open) for p in positions] == ["01-01 12:00:59.999", "01-03 16:00:59.999"]
    assert (_utc(positions[0].ts_close), positions[0].px_open, positions[0].px_close) == (
        "01-03 12:00:59.999",
        10720.5,
        13600.4,
    )
    # exactly 12 4h bars held: entry at the close of the bar opening 12:00, exit at the
    # release tick N bars later
    assert positions[0].ts_close is not None
    assert positions[0].ts_close - positions[0].ts_open == 12 * _H4
    assert positions[0].closing == "MARKET"
    # second entry on the next bar's tick; the run ends with it OPEN in the cache (R16b)
    assert (positions[1].px_open, positions[1].ts_close, positions[1].px_close) == (
        13840.6,
        None,
        None,
    )
    assert closes == [_Close("BTCUSDT", _ns("01-03 12:00:59.999"), 13600.4)]


# ---------------------------------------------------------------------------
# R2/R4: N=1 exits on the next bar's tick, re-enters one bar after the exit fill
# ---------------------------------------------------------------------------


def test_4h_n1_exits_next_bar_and_reenters_bar_after(catalog: Path) -> None:
    metrics, positions, _, _ = _run(catalog, _dsl("4h", name="nbar_r2", n=1))

    assert metrics.total_trades == 11
    assert len(positions) == 11
    assert (_utc(positions[0].ts_open), _utc(positions[0].ts_close), positions[0].px_close) == (
        "01-01 12:00:59.999",
        "01-01 16:00:59.999",
        10960.4,
    )
    base = _ns("01-01 12:00:59.999")
    for i, pos in enumerate(positions[:-1]):
        assert pos.ts_open == base + i * 2 * _H4  # every other 4h bar starts a trade
        assert pos.ts_close == pos.ts_open + _H4  # held exactly one bar
        assert pos.closing == "MARKET"
    # the exit bar evaluates nothing (R4): the re-entry fills one bar after the EXIT fill
    for prev, nxt in zip(positions[:-1], positions[1:], strict=True):
        assert prev.ts_close is not None
        assert nxt.ts_open == prev.ts_close + _H4
    assert (_utc(positions[-1].ts_open), positions[-1].px_open, positions[-1].ts_close) == (
        "01-04 20:00:59.999",
        15520.5,
        None,
    )


# ---------------------------------------------------------------------------
# R5: 1m immediate mode, N=3
# ---------------------------------------------------------------------------


def test_1m_screening_holds_n_bars(catalog: Path) -> None:
    metrics, positions, _, _ = _run(
        catalog, _dsl("1m", name="nbar_r5", n=3), end="2024-01-02"
    )

    assert metrics.total_trades == 288
    assert len(positions) == 288
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_open, first.px_close) == (
        "01-01 00:03:59.999",
        "01-01 00:06:59.999",
        10003.5,
        10006.4,
    )
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_open, second.px_close) == (
        "01-01 00:08:59.999",
        "01-01 00:11:59.999",
        10008.6,
        10011.4,
    )
    # every closed trade but the run-end one is held exactly 3 minutes
    for pos in positions[:-1]:
        assert pos.ts_close is not None
        assert pos.ts_close - pos.ts_open == 3 * 60_000_000_000
        assert pos.closing == "MARKET"
    # the last position's exit is due after the run: on_stop closes it at the last bar
    last = positions[-1]
    assert (_utc(last.ts_open), _utc(last.ts_close), last.px_close) == (
        "01-01 23:58:59.999",
        "01-01 23:59:59.999",
        11439.5,
    )


# ---------------------------------------------------------------------------
# R6: 1m N=1
# ---------------------------------------------------------------------------


def test_1m_n1(catalog: Path) -> None:
    metrics, positions, _, _ = _run(catalog, _dsl("1m", name="nbar_r6", n=1), end="2024-01-02")

    assert metrics.total_trades > 100
    assert (_utc(positions[0].ts_open), _utc(positions[0].ts_close), positions[0].px_open) == (
        "01-01 00:03:59.999",
        "01-01 00:04:59.999",
        10003.5,
    )
    assert positions[0].px_close == 10004.4
    assert positions[0].ts_close is not None
    assert positions[0].ts_close - positions[0].ts_open == 60_000_000_000
    assert (_utc(positions[1].ts_open), positions[1].px_open) == ("01-01 00:06:59.999", 10006.6)
    assert all(pos.ts_close is not None for pos in positions)  # N=1 always exits inside the run


# ---------------------------------------------------------------------------
# R7: a pending delayed exit is cleared by the time exit on the same bar
# ---------------------------------------------------------------------------


def test_time_exit_clears_pending_delayed_exit(catalog: Path) -> None:
    """The 2nd entry is 00:08:59.999: a NOT-cleared pending exit delays it to 00:09:59.999."""
    metrics, positions, _, _ = _run(
        catalog,
        _dsl("1m", name="nbar_r7", n=3, exit_condition="close > 10005"),
        end="2024-01-02",
    )

    assert metrics.total_trades == 359
    assert len(positions) == 359
    first, second = positions[0], positions[1]
    # first close is the TIME exit (bar 3), not the indicator exit queued for bar 4
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 00:03:59.999",
        "01-01 00:06:59.999",
        10006.4,
    )
    assert (_utc(second.ts_open), second.px_open) == ("01-01 00:08:59.999", 10008.6)
    # ...and the second position's OWN indicator exit fires two bars later
    assert (_utc(second.ts_close), second.px_close) == ("01-01 00:10:59.999", 10010.4)


# ---------------------------------------------------------------------------
# R8: an indicator exit earlier than N wins; a same-bar exit is sent once
# ---------------------------------------------------------------------------


def test_indicator_exit_before_n_wins(catalog: Path) -> None:
    metrics, positions, closes, _ = _run(
        catalog, _dsl("4h", name="nbar_r8", n=12, exit_condition="ema > 12000")
    )

    assert metrics.total_trades == 8
    assert len(positions) == 8
    first = positions[0]
    assert (_utc(first.ts_close), first.px_close) == ("01-02 16:00:59.999", 12400.4)
    assert first.ts_close is not None
    assert first.ts_close - first.ts_open == 7 * _H4  # 7 < 12: the indicator exit won
    assert first.closing == "MARKET"
    # one filled close per position: the time exit never double-sends on the exit bar
    assert [(c.symbol, c.ts) for c in closes] == [
        (p.symbol, p.ts_close) for p in positions if p.ts_close is not None
    ]


def test_same_bar_indicator_exit_sends_one_exit(catalog: Path) -> None:
    """``close > 10006`` first fires on the 3rd bar -- the same bar as the time exit.

    The time exit returns first (no deferred indicator exit is queued), so the first
    close fills at bar 3's close (00:06:59.999); the deferred exit would fill at
    00:07:59.999. One exit per bar: exactly one filled close at that instant.
    """
    metrics, positions, closes, _ = _run(
        catalog,
        _dsl("1m", name="nbar_r8_same_bar", n=3, exit_condition="close > 10006"),
        end="2024-01-02",
    )

    assert metrics.total_trades == 359
    first = positions[0]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 00:03:59.999",
        "01-01 00:06:59.999",
        10006.4,
    )
    assert [c.ts for c in closes if c.ts == _ns("01-01 00:06:59.999")] == [_ns("01-01 00:06:59.999")]
    assert _utc(positions[1].ts_open) == "01-01 00:08:59.999"


# ---------------------------------------------------------------------------
# R9: SL/TP first -- a short's untouched 10 % SL never fires
# ---------------------------------------------------------------------------


def test_short_time_exit(catalog: Path) -> None:
    metrics, positions, closes, orders = _run(
        catalog, _dsl("4h", name="nbar_r9", n=2, side="short"), end="2024-01-03"
    )

    assert metrics.total_trades == 3
    # every entry is a SELL: never a reverse (long) position
    assert [o.side for o in _entries(orders)] == ["SELL"] * len(positions)
    first = positions[0]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 12:00:59.999",
        "01-01 20:00:59.999",
        11200.6,
    )
    # the 10 % SL is never reached on the rising toy: the MARKET time exit closed it
    assert {p.closing for p in positions} == {"MARKET"}
    _assert_one_filled_close_per_position(positions, closes)


# ---------------------------------------------------------------------------
# R10: a missing primary bar is not counted (the count is bars DELIVERED)
# ---------------------------------------------------------------------------


def test_missing_bar_not_counted(tmp_path: Path) -> None:
    cat = _write_gapped_catalog(
        tmp_path / "catalog", {"BTCUSDT": {5}}, n_min=1440, timeframes=("1m",)
    )
    metrics, positions, _, _ = _run(cat, _dsl("1m", name="nbar_r10", n=3), end="2024-01-02")

    assert metrics.total_trades == 288
    first = positions[0]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_open, first.px_close) == (
        "01-01 00:03:59.999",
        "01-01 00:07:59.999",
        10003.5,
        10007.4,
    )
    # minute 5 is missing: 3 DELIVERED bars span 4 minutes
    assert first.ts_close is not None
    assert first.ts_close - first.ts_open == 4 * 60_000_000_000


# ---------------------------------------------------------------------------
# R14: per-instrument counters (BTC and ETH alone in one multi-symbol run)
# ---------------------------------------------------------------------------


def test_multi_symbol_counts_per_instrument(catalog: Path) -> None:
    metrics, positions, closes, _ = _run(
        catalog, _dsl("4h", name="nbar_r14", n=12), symbols=("BTCUSDT", "ETHUSDT")
    )

    assert metrics.total_trades == 4
    by_symbol = _by_symbol(positions)
    btc, eth = by_symbol["BTCUSDT"], by_symbol["ETHUSDT"]
    assert [(_utc(p.ts_open), _utc(p.ts_close)) for p in btc] == [
        ("01-01 12:00:59.999", "01-03 12:00:59.999"),
        ("01-03 16:00:59.999", "OPEN"),
    ]
    assert [(_utc(p.ts_open), _utc(p.ts_close)) for p in eth] == [
        ("01-01 12:00:59.999", "01-03 12:00:59.999"),
        ("01-03 16:00:59.999", "OPEN"),
    ]
    # each instrument counts its own bars: same 12-bar holds, ETH prices from ETH's path
    assert (btc[0].px_open, btc[0].px_close) == (10720.5, 13600.4)
    assert (eth[0].px_open, eth[0].px_close) == (21441.01, 27200.99)
    assert {(c.symbol, _utc(c.ts)) for c in closes} == {
        ("BTCUSDT", "01-03 12:00:59.999"),
        ("ETHUSDT", "01-03 12:00:59.999"),
    }


# ---------------------------------------------------------------------------
# R15: the count is NOT a sweepable config field (fail closed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [12, None])
def test_sweep_key_rejected(catalog: Path, n: int | None) -> None:
    dsl = _dsl("4h", name=f"nbar_r15_{n or 'off'}", n=n)
    runner = NTScreeningRunner(dsl, ["BTCUSDT"], _START, _END, catalog_path=str(catalog))

    with pytest.raises(UnknownStrategyParamError, match="max_bars_in_trade"):
        runner({"max_bars_in_trade": 5})


# ---------------------------------------------------------------------------
# R17: the fast tier refuses the field and falls back to NT when allowed
# ---------------------------------------------------------------------------


def test_fast_tier_falls_back_on_max_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe_quant.discovery import genome as genome_mod
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.fast_tier import fallbacks
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

    chromosome = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, C.GT, 50.0)],
        exit_genes=[],
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        direction=Direction.LONG,
    )
    monkeypatch.setattr(genome_mod, "chromosome_to_dsl", lambda _: {"max_bars_in_trade": 12})

    def _no_runner(*_: Any, **__: Any) -> None:
        raise AssertionError("_make_runner reached with max_bars_in_trade set")

    monkeypatch.setattr(FastBacktestFn, "_make_runner", _no_runner)
    fn = FastBacktestFn(["BTCUSDT"], "1m", _START, "2024-01-02")
    with pytest.raises(FastTierUnsupported, match="max_bars_in_trade time exit"):
        fn._run_fast(chromosome, _START, "2024-01-02", None)

    fallbacks.clear()
    monkeypatch.setattr(
        NTBacktestFn, "_run_single", lambda self, c, s, e, symbols=None: {"total_trades": 7}
    )
    fallback_fn = FastBacktestFn(["BTCUSDT"], "1m", _START, "2024-01-02", nt_fallback=True)
    assert fallback_fn._run_single(chromosome, _START, "2024-01-02") == {"total_trades": 7}
    assert fallbacks.summary() == {
        "total": 1,
        "by_reason": {"max_bars_in_trade time exit": 1},
    }


# ---------------------------------------------------------------------------
# R9b (a): SL hit before N, outbox mode (4h trade ticks) -- fresh count after
# ---------------------------------------------------------------------------


def test_sl_before_n_outbox_keeps_a_fresh_count(catalog: Path) -> None:
    metrics, positions, closes, orders = _run(
        catalog, _dsl("4h", name="nbar_r9b_a", n=12, side="short", sl=1.0), end="2024-01-03"
    )

    assert metrics.total_trades == 9
    assert [o.side for o in _entries(orders)] == ["SELL"] * len(positions)  # no reverse
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 12:00:59.999",
        "01-01 15:59:59.999",
        10827.8,
    )
    assert first.closing == "STOP_MARKET"  # the SL, not the time exit
    # the next entry starts a FRESH count: its own SL 10 bars later, not an instant exit
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_close) == (
        "01-01 16:00:59.999",
        "01-01 19:59:59.999",
        11070.1,
    )
    assert second.closing == "STOP_MARKET"
    assert all(p.closing == "STOP_MARKET" for p in positions)
    # exactly one filled close per position: no second (time) exit landed anywhere
    assert [(c.symbol, c.ts) for c in closes] == [
        (p.symbol, p.ts_close) for p in positions if p.ts_close is not None
    ]


# ---------------------------------------------------------------------------
# R9b (a): SL hit before N, 1m immediate mode -- fresh count after
# ---------------------------------------------------------------------------


def test_sl_before_n_immediate_keeps_a_fresh_count(catalog: Path) -> None:
    metrics, positions, closes, orders = _run(
        catalog,
        _dsl("1m", name="nbar_r9b_a1m", n=12, side="short", sl=0.1),
        end="2024-01-02",
    )

    assert metrics.total_trades == 125
    assert [o.side for o in _entries(orders)] == ["SELL"] * len(positions)  # no reverse
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 00:03:59.999",
        "01-01 00:13:59.999",
        10013.6,
    )
    assert first.closing == "STOP_MARKET"  # the SL, not the time exit
    # fresh count: the 2nd trade survives its own 10 bars instead of exiting on bar 1
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_close) == (
        "01-01 00:14:59.999",
        "01-01 00:24:59.999",
        10024.5,
    )
    assert second.closing == "STOP_MARKET"
    assert second.ts_close is not None and second.ts_close - second.ts_open == 10 * 60_000_000_000
    _assert_one_filled_close_per_position(positions, closes)


# ---------------------------------------------------------------------------
# R9b (b): TP hit before N, outbox mode (4h trade ticks) -- fresh count after
# ---------------------------------------------------------------------------


def test_tp_before_n_outbox_keeps_a_fresh_count(catalog: Path) -> None:
    metrics, positions, closes, orders = _run(
        catalog, _dsl("4h", name="nbar_r9b_b", n=12, side="long", tp=1.0), end="2024-01-03"
    )

    assert metrics.total_trades == 9
    assert [o.side for o in _entries(orders)] == ["BUY"] * len(positions)  # no reverse
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 12:00:59.999",
        "01-01 15:59:59.999",
        10827.6,
    )
    assert first.closing == "LIMIT"  # the take-profit, not the time exit
    assert first.ts_close is not None and first.ts_close - first.ts_open < 12 * _H4  # 1 < 12
    # the next entry starts a FRESH count: its own TP on its first bar, not an instant exit
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_close) == (
        "01-01 16:00:59.999",
        "01-01 19:59:59.999",
        11070.1,
    )
    assert second.closing == "LIMIT"
    # every protective close precedes the release of a time exit: none was submitted
    assert all(p.closing == "LIMIT" for p in positions)
    assert [o for o in orders if o.status == "REJECTED"] == []
    _assert_one_filled_close_per_position(positions, closes)


# ---------------------------------------------------------------------------
# R9b (b): TP hit before N, 1m immediate mode -- fresh count after
# ---------------------------------------------------------------------------


def test_tp_before_n_immediate_keeps_a_fresh_count(catalog: Path) -> None:
    metrics, positions, closes, orders = _run(
        catalog,
        _dsl("1m", name="nbar_r9b_b1m", n=12, side="long", tp=0.1),
        end="2024-01-02",
    )

    assert metrics.total_trades == 124
    assert [o.side for o in _entries(orders)] == ["BUY"] * len(positions)  # no reverse
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 00:03:59.999",
        "01-01 00:13:59.999",
        10013.4,
    )
    assert first.closing == "LIMIT"  # the take-profit, not the time exit
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_close) == (
        "01-01 00:14:59.999",
        "01-01 00:24:59.999",
        10024.5,
    )
    assert second.closing == "LIMIT"
    assert second.ts_close is not None and second.ts_close - second.ts_open == 10 * 60_000_000_000
    _assert_one_filled_close_per_position(positions, closes)


# ---------------------------------------------------------------------------
# R9b (c): a protective order touched ON the time exit's release tick wins
# ---------------------------------------------------------------------------


def test_protective_fill_on_the_release_tick_wins(catalog: Path) -> None:
    """TP 26.863 % sits between the 12th bar's high and the release tick's price.

    The market exit queued on the 12th bar is submitted on the release tick, after
    the matching engine filled the resting TP: it is reduce-only and rejected, so
    the position closes at the TP price (13600.2), not at the market fill (13600.4).
    """
    metrics, positions, closes, orders = _run(
        catalog, _dsl("4h", name="nbar_r9b_c", n=12, tp=26.863), end="2024-01-05"
    )

    assert metrics.total_trades == 2
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_close), first.px_close) == ("01-03 12:00:59.999", 13600.2)
    assert first.closing == "LIMIT"
    # exactly one filled close: the rejected market exit never filled
    assert closes == [_Close("BTCUSDT", _ns("01-03 12:00:59.999"), 13600.2)]
    # ...but it WAS submitted on the release tick and rejected as reduce-only
    assert [
        (o.type, o.filled, _utc(o.ts_last), o.status)
        for o in orders
        if o.type == "MARKET" and o.filled == 0.0
    ] == [("MARKET", 0.0, "01-03 12:00:59.999", "REJECTED")]
    # the next position gets a fresh count (a stale 12 would exit it on the first bar)
    assert (_utc(second.ts_open), second.ts_close) == ("01-03 16:00:59.999", None)
    assert [o.side for o in _entries(orders)] == ["BUY", "BUY"]


def test_protective_fill_on_the_release_tick_wins_immediate(catalog: Path) -> None:
    """TP 0.1 % is touched on the 10th bar's close tick -- the tick the time exit releases on.

    The engine fills the resting TP before ``on_bar`` counts the bar, so the time exit
    finds a flat position and submits nothing: one close, at the TP price, as the LIMIT
    (a market exit would have been submitted at that same instant). The toy's absolute
    price rise makes later trades miss the TP on bar 10; those do exit by the MARKET time
    exit, so only the first (collision) trade's reason is pinned here.
    """
    metrics, positions, closes, orders = _run(
        catalog, _dsl("1m", name="nbar_r9b_c1m", n=10, tp=0.1), end="2024-01-02"
    )

    assert metrics.total_trades == 124
    assert [o.side for o in _entries(orders)] == ["BUY"] * len(positions)  # no reverse
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 00:03:59.999",
        "01-01 00:13:59.999",
        10013.4,
    )
    assert first.closing == "LIMIT"  # the TP, not a market time exit on the same tick
    assert first.ts_close is not None and first.ts_close - first.ts_open == 10 * 60_000_000_000
    # nothing was submitted by the time exit at the collision tick, filled or rejected
    assert [
        o
        for o in orders
        if o.reduce_only and o.type == "MARKET" and o.ts_last == _ns("01-01 00:13:59.999")
    ] == []
    # the next position counts from zero: its own TP 10 bars in, not a stale immediate exit
    assert (_utc(second.ts_open), _utc(second.ts_close), second.px_close) == (
        "01-01 00:14:59.999",
        "01-01 00:24:59.999",
        10024.5,
    )
    assert second.closing == "LIMIT"
    _assert_one_filled_close_per_position(positions, closes)


# ---------------------------------------------------------------------------
# R10b: the release feed is starved while primary bars continue
# ---------------------------------------------------------------------------


def test_starved_release_waits_for_the_next_own_tick(tmp_path: Path) -> None:
    """1m bars at the 12:00 boundary are missing, so no tick releases the queued exit.

    The exit queued on the 01-03 08:00 bar (the 12th) waits for the next OWN tick at
    16:00:59.999; every extra exit queued on the starved bars is rejected there, so
    exactly one close fills and no stale exit survives to close the next position.
    If the feed never resumed, the queued exit would never release and the position
    would stay OPEN at the run end (marked by the extraction layer, R16b); the run
    window's coverage check refuses a feed that stops more than one bar before the
    end, so that variant cannot be constructed here.
    """
    cat = _write_gapped_catalog(
        tmp_path / "catalog",
        {"BTCUSDT": {_STARVED_BOUNDARY, _STARVED_BOUNDARY + 1}},
    )
    metrics, positions, closes, orders = _run(cat, _dsl("4h", name="nbar_r10b", n=12))

    assert metrics.total_trades == 2
    first, second = positions[0], positions[1]
    assert (_utc(first.ts_open), _utc(first.ts_close), first.px_close) == (
        "01-01 12:00:59.999",
        "01-03 16:00:59.999",
        13840.4,
    )
    assert first.closing == "MARKET"
    # exactly one filled close at the resumed release (all extra exits were rejected)
    assert closes == [_Close("BTCUSDT", _ns("01-03 16:00:59.999"), 13840.4)]
    # the exit queued on the starved bars was submitted there and rejected, once
    assert [
        (o.type, o.filled, _utc(o.ts_last), o.status)
        for o in orders
        if o.type == "MARKET" and o.filled == 0.0
    ] == [("MARKET", 0.0, "01-03 16:00:59.999", "REJECTED")]
    # no stale queued exit closes the next position
    assert (_utc(second.ts_open), second.px_open, second.ts_close) == (
        "01-03 20:00:59.999",
        14080.6,
        None,
    )


def test_starved_release_ignores_another_symbols_tick(tmp_path: Path) -> None:
    """ETH's 12:00 tick releases ETH's queued exit; BTC's stalled one waits for BTC."""
    cat = _write_gapped_catalog(
        tmp_path / "catalog",
        {"BTCUSDT": {_STARVED_BOUNDARY, _STARVED_BOUNDARY + 1}},
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    metrics, positions, closes, _ = _run(
        cat, _dsl("4h", name="nbar_r10b_pair", n=12), symbols=("BTCUSDT", "ETHUSDT")
    )

    assert metrics.total_trades == 4
    by_symbol = _by_symbol(positions)
    assert [(_utc(p.ts_open), _utc(p.ts_close), p.px_close) for p in by_symbol["ETHUSDT"]] == [
        ("01-01 12:00:59.999", "01-03 12:00:59.999", 27200.99),
        ("01-03 16:00:59.999", "OPEN", None),
    ]
    assert [(_utc(p.ts_open), _utc(p.ts_close), p.px_close) for p in by_symbol["BTCUSDT"]] == [
        ("01-01 12:00:59.999", "01-03 16:00:59.999", 13840.4),
        ("01-03 20:00:59.999", "OPEN", None),
    ]
    assert [(c.symbol, _utc(c.ts)) for c in closes] == [
        ("ETHUSDT", "01-03 12:00:59.999"),
        ("BTCUSDT", "01-03 16:00:59.999"),
    ]
