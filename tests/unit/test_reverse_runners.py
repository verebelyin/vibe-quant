"""``position_mode: reverse`` on the real runners (design-v2 § 3.1, § 6 R2, R2a-R2c, R4i, R8a, R8b).

Toy catalog: ``_write_catalog`` (rising +0.01 %/min, BTC 10 000 / ETH 20 000, 1 USDT/min).
The strategy is long while ``close < 10100`` and has ONE short-signal bar (``10100 <= close < 10110``:
the 15m bar that closes at minute 104 is the only one, at 10104.5), so there is exactly one signal
flip while the long is open. Everything is read from the ENGINE CACHE (order fills, open positions), never from
serialized results.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from tests.unit.test_audit_metrics_funding import empty_archive
from tests.unit.test_max_bars_exit_parity import _validation
from tests.unit.test_screening_fill_mode import _START, _write_catalog
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    from pathlib import Path

_MIN_NS = 60_000_000_000
_MS_NS = 1_000_000
_MIN_MS = _MIN_NS // _MS_NS
_DAY0_NS = 1_704_067_200_000_000_000
_END = "2024-01-02"

# 15m bar k covers minutes 15k..15k+14; its close is base*(1 + 1e-4*(15k+14.5)).
# k=6 closes at 10104.5 (first >= 10100) -> signal bar ends at minute 105.
_SIGNAL_BAR_END_NS = _DAY0_NS + 105 * _MIN_NS - _MS_NS
# screening releases on the first 1m fill tick after the signal bar's ts_init (minute 105's close)
_RELEASE_TICK_NS = _SIGNAL_BAR_END_NS + _MIN_NS


class Fill(NamedTuple):
    symbol: str
    side: str
    reduce_only: bool
    type: str
    qty: float
    px: float
    ts: int
    status: str
    commission: float


class Run(NamedTuple):
    orders: list[Fill]
    open_positions: list[tuple[str, str, float]]  # (symbol, side, qty)
    strategies: list[Any]
    ledger: Any
    metrics: Any
    open_pos: list[Any]


def reverse_dsl(
    name: str,
    *,
    tf: str = "15m",
    reverse: bool | None = True,
    long: tuple[str, ...] = ("close < 10100",),
    short: tuple[str, ...] = ("close >= 10100", "close < 10110"),
    **over: Any,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": tf,
        "indicators": {},
        "entry_conditions": {"long": list(long), "short": list(short)},
        "stop_loss": {"type": "fixed_pct", "percent": 20.0},
        "take_profit": {"type": "fixed_pct", "percent": 90.0},
    }
    if reverse:
        d["position_mode"] = "reverse"
    d.update(over)
    return d


def run_screening(
    catalog: Path,
    dsl: dict[str, Any],
    *,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    end: str = _END,
    max_margin_pct: float | None = None,
    risk_per_trade: float | None = None,
    funding_archive: Path | None = None,
) -> Run:
    """The public screening entry point; the cache is read from its extraction hook."""
    seen: list[Run] = []
    orig = NTScreeningRunner._extract_metrics

    def spy(
        self: NTScreeningRunner,
        params: dict[str, float | int],
        bt_result: Any,
        engine: Any,
        start_time: float,
        starting_balance: float = 1000.0,
        cost_ledger: Any = None,
    ) -> Any:
        cache = engine.kernel.cache
        orders = sorted(
            (
                Fill(
                    symbol=str(o.instrument_id).split("-")[0],
                    side=o.side.name,
                    reduce_only=bool(o.is_reduce_only),
                    type=o.order_type.name,
                    qty=float(o.filled_qty),
                    px=float(o.avg_px) if o.filled_qty else 0.0,
                    ts=int(o.ts_last),
                    status=o.status.name,
                    commission=sum(float(c) for c in o.commissions()),
                )
                for o in cache.orders()
            ),
            key=lambda f: (f.ts, f.symbol, f.type),
        )
        opened = [
            (str(p.instrument_id).split("-")[0], p.side.name, float(p.quantity))
            for p in cache.positions_open()
        ]
        metrics = orig(
            self, params, bt_result, engine, start_time,
            starting_balance=starting_balance, cost_ledger=cost_ledger,
        )
        seen.append(
            Run(
                orders, opened, list(engine.kernel.trader.strategies()), cost_ledger, metrics,
                list(cache.positions_open()),
            )
        )
        return metrics

    sizing: dict[str, Any] = {"max_margin_pct": max_margin_pct}
    if risk_per_trade is not None:
        sizing["risk_per_trade"] = risk_per_trade

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(NTScreeningRunner, "_extract_metrics", spy)
        NTScreeningRunner(
            dsl,
            list(symbols),
            _START,
            end,
            catalog_path=str(catalog),
            funding_archive_path=str(funding_archive) if funding_archive else None,
            launch=LaunchSettings(sizing=SizingSettings(**sizing)),
        )({})
    return seen[0]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """BTC and ETH on the SAME toy prices (each written as the first symbol of its own call)."""
    root = tmp_path_factory.mktemp("catalog")
    tfs = ("1m", "15m", "1h", "4h")
    _write_catalog(root, ("BTCUSDT",), timeframes=tfs)
    return _write_catalog(root, ("ETHUSDT",), timeframes=tfs)


def entries(run: Run, symbol: str = "BTCUSDT") -> list[Fill]:
    return [f for f in run.orders if f.symbol == symbol and not f.reduce_only and f.qty > 0]


def market_exits(run: Run, symbol: str = "BTCUSDT") -> list[Fill]:
    return [
        f for f in run.orders
        if f.symbol == symbol and f.reduce_only and f.type == "MARKET" and f.qty > 0
    ]


def test_reverse_entry_fills_on_exit_release_tick_screening_15m(catalog: Path) -> None:
    """R2: screening 15m outbox. The exit and the opposite entry fill on the SAME release tick."""
    run = run_screening(catalog, reverse_dsl("rev_r2"))

    long_entry, short_entry = entries(run)
    (exit_fill,) = market_exits(run)
    assert (long_entry.side, short_entry.side, exit_fill.side) == ("BUY", "SELL", "SELL")
    assert exit_fill.ts == short_entry.ts == _RELEASE_TICK_NS, (exit_fill.ts, short_entry.ts)
    # same tick price (1m close of minute 105) within the 1-tick slippage draw of each fill
    tick_px = 10_000.0 * (1 + 1e-4 * 105.5)
    assert exit_fill.px == pytest.approx(tick_px, abs=0.2)
    assert short_entry.px == pytest.approx(tick_px, abs=0.2)
    assert run.open_positions == [("BTCUSDT", "SHORT", short_entry.qty)]


def test_legacy_without_position_mode_never_reverses(catalog: Path) -> None:
    """Probe of the old behaviour: no ``position_mode`` -> the short signal is ignored while long."""
    run = run_screening(catalog, reverse_dsl("rev_legacy", reverse=None))

    (long_entry,) = entries(run)
    assert long_entry.side == "BUY"
    assert market_exits(run) == []
    assert run.open_positions == [("BTCUSDT", "LONG", long_entry.qty)]


def test_validation_own_1m_release_fills_the_reversal_on_the_same_detail_bar(
    catalog: Path, tmp_path: Path
) -> None:
    """R2a: validation releases on the instrument's own next 1m detail bar; the entry rides it."""
    fills, _ = _validation(tmp_path, catalog, reverse_dsl("rev_r2a"))

    orders = [o for o in fills.orders if o.filled > 0]
    long_entry, short_entry = [o for o in orders if not o.reduce_only]
    (exit_fill,) = [o for o in orders if o.reduce_only and o.type == "MARKET"]
    assert (long_entry.side, short_entry.side, exit_fill.side) == ("BUY", "SELL", "SELL")
    assert exit_fill.ts_last == short_entry.ts_last == _RELEASE_TICK_NS


def test_one_minute_defer_sends_the_exit_on_the_next_bar_and_the_entry_rides_it(catalog: Path) -> None:
    """R2b/R6: 1m screening defers the signal one bar; at dispatch the state is opposite -> reversal.

    The short signal is bar 100 (close 10100.5); the dispatch bar is 101 whose close is the
    send instant (no outbox on 1m), so the exit and the entry fill on bar 101's own ts_init.
    """
    dsl = reverse_dsl("rev_r2b", tf="1m", short=("close >= 10100", "close < 10101"))
    run = run_screening(catalog, dsl)

    long_entry, short_entry = entries(run)
    # 1m immediate mode closes what is still open when the data ends (on_stop): not ours
    exit_fill, end_of_run_close = market_exits(run)
    dispatch_bar_ts = _DAY0_NS + 102 * _MIN_NS - _MS_NS
    assert (long_entry.side, short_entry.side) == ("BUY", "SELL")
    assert exit_fill.ts == short_entry.ts == dispatch_bar_ts
    assert (end_of_run_close.side, end_of_run_close.qty) == ("BUY", short_entry.qty)


def test_new_side_is_sized_on_the_signal_bar_with_its_own_stop(catalog: Path) -> None:
    """R3: risk 20 / (short stop 15% of the signal close) rounds down to 0.013; the long (20% stop) is 0.009."""
    dsl = reverse_dsl("rev_r3", stop_loss_short={"type": "fixed_pct", "percent": 15.0})
    run = run_screening(catalog, dsl)

    long_entry, short_entry = entries(run)
    assert (long_entry.qty, short_entry.qty) == (0.009, 0.013)
    (exit_fill,) = market_exits(run)
    assert exit_fill.qty == long_entry.qty  # the exit closes the OLD size, not the new one


def test_entry_refused_by_the_margin_limit_ends_flat(catalog: Path) -> None:
    """R4/D1: the 0.013 short needs 13.1 margin; 1.1% of 1000 allows 11 -> flat, one refusal."""
    dsl = reverse_dsl("rev_r4", stop_loss_short={"type": "fixed_pct", "percent": 15.0})
    run = run_screening(catalog, dsl, max_margin_pct=0.011)

    (long_entry,) = entries(run)  # the 9.0-margin long fit; the reversal's short never left
    (exit_fill,) = market_exits(run)
    assert exit_fill.ts == _RELEASE_TICK_NS
    assert run.open_positions == []
    assert [s._margin_refusals for s in run.strategies] == [1]
    assert long_entry.qty == 0.009


def test_both_entry_conditions_true_while_long_reverses(catalog: Path) -> None:
    """R4h: with the long condition always true, the held long still yields to the short signal
    (only the opposite side is evaluated), and the next bar's long signal reverses back."""
    run = run_screening(catalog, reverse_dsl("rev_r4h", long=("close > 0",)))

    sides = [e.side for e in entries(run)]
    assert sides == ["BUY", "SELL", "BUY"], sides
    assert entries(run)[1].ts == _RELEASE_TICK_NS
    assert len(market_exits(run)) == 2


@pytest.mark.parametrize("order", [("BTCUSDT", "ETHUSDT"), ("ETHUSDT", "BTCUSDT")])
def test_two_symbols_reversing_on_the_same_tick_count_each_others_positions(
    catalog: Path, order: tuple[str, str]
) -> None:
    """R4i: both symbols reverse at 01:45:59.999. The new 0.039 short needs 39.4 margin, the limit is
    44 (4.4% of 1000), and the other symbol's still-open 9.0 long pushes it to 48.4. The engine
    handles BTC's tick first (whatever order the strategies were added in): BTC's reversal sees
    ETH's long and is refused (flat); ETH's sees BTC flat and goes through."""
    dsl = reverse_dsl("rev_r4i", stop_loss_short={"type": "fixed_pct", "percent": 5.0})
    run = run_screening(catalog, dsl, symbols=order, max_margin_pct=0.044)

    assert [(e.side, e.qty) for e in entries(run, "BTCUSDT")] == [("BUY", 0.009)]
    assert [(e.side, e.qty) for e in entries(run, "ETHUSDT")] == [("BUY", 0.009), ("SELL", 0.039)]
    for symbol in order:  # both exits ran: the refused symbol closed its long and stayed flat
        assert [x.ts for x in market_exits(run, symbol)][0] == _RELEASE_TICK_NS
    refusals = {s.instrument_id.symbol.value.split("-")[0]: s._margin_refusals for s in run.strategies}
    assert refusals == {"BTCUSDT": 1, "ETHUSDT": 0}


_DAY0_MS = _DAY0_NS // _MS_NS
# The 15m bar that closes at minute 104 is the only short signal (see the module docstring); its
# release tick is minute 105's own close, 01:45:59.999.
_REVERSAL_NS = _DAY0_NS + 106 * _MIN_NS - _MS_NS


def _flat_catalog(root: Path) -> Path:
    """Five days of flat 1m bars: 10 100.0 through minute 89, then 10 480.0 (one short signal).

    A flat price makes BOTH sides' entry price literal: the long enters at exactly 10 100.0 on
    the bar-0 release tick, so 0.01 x 10 100 = 101.0 USDT notional.
    """
    cm = CatalogManager(root)
    inst = create_instrument("BTCUSDT")
    cm.write_instrument(inst)
    klines: list[Any] = [
        {
            "open_time": _DAY0_MS + i * _MIN_MS,
            "open": px, "high": px, "low": px, "close": px,
            "volume": 10.0,
            "close_time": _DAY0_MS + i * _MIN_MS + 59_999,
        }
        for i in range(5 * 1440)
        for px in [10_480.0 if i >= 90 else 10_100.0]
    ]
    bars_1m = klines_to_bars(klines, inst.id, get_bar_type("BTCUSDT", "1m"),
                             inst.size_precision, inst.price_precision)
    cm.write_bars(bars_1m)
    for tf, minutes in (("15m", 15), ("1h", 60), ("4h", 240)):
        cm.write_bars(aggregate_bars(bars_1m, get_bar_type("BTCUSDT", tf), minutes,
                                     inst.size_precision, inst.price_precision))
    return root


def _funding_archive(path: Path, ts_ns: int) -> Path:
    """A real archive holding ONE settlement exactly at ``ts_ns`` (rate 0.0001, 8h interval)."""
    archive = RawDataArchive(path)
    try:
        archive.conn.commit()
    finally:
        archive.close()
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate, source) "
        "VALUES (?,?,?,?)",
        ("BTCUSDT", ts_ns // _MS_NS, 0.0001, "test"),
    )
    conn.commit()
    conn.close()
    return path


def test_a_settlement_on_the_reversal_instant_is_paid_by_the_old_side_only(tmp_path: Path) -> None:
    """R8a: the settlement timestamp EQUALS both the exit and the new entry timestamp. Under the
    (entry, exit] rule the long pays it (0.0001 x 101.0 USDT notional = 0.0101) and the short,
    entered on that same instant, does not -- even accrued 10 minutes on. Moving the window to
    [entry, exit) flips both numbers."""
    from vibe_quant.trade_ledger import accrue_position_funding

    run = run_screening(
        _flat_catalog(tmp_path / "catalog"),
        reverse_dsl("rev_r8a", long=("close < 10100.5",), short=("close >= 10470", "close < 10490")),
        risk_per_trade=0.0202,  # 20.2 USDT risk / 20% stop = 0.010 qty at 10 100
        funding_archive=_funding_archive(tmp_path / "f.db", _REVERSAL_NS),
    )

    long_entry, short_entry = entries(run)
    assert (long_entry.qty, long_entry.px) == (0.01, 10_100.0)
    assert long_entry.qty * long_entry.px == pytest.approx(101.0)  # the literal notional
    assert short_entry.ts == _REVERSAL_NS
    ((key, long_funding),) = run.ledger.finalized.items()  # only the long is closed
    assert long_funding == pytest.approx(0.0101, rel=1e-12)  # settlement == exit is INCLUDED
    (short_pos,) = run.open_pos
    assert short_pos.ts_opened == _REVERSAL_NS
    # settled at the entry instant, the short pays nothing even after 10 minutes of holding
    assert accrue_position_funding(
        run.ledger.model.funding, short_pos, exit_ns=_REVERSAL_NS + 10 * _MIN_NS
    ).total == 0.0
    # the sizing ledger's view at the reversal instant is the long's cost and nothing more
    assert run.ledger.paid_through(_REVERSAL_NS, run.open_pos) == pytest.approx(0.0101, rel=1e-12)


def test_a_reversal_pays_two_taker_commissions_and_slips_each_fill(catalog: Path, tmp_path: Path) -> None:
    """R8b: the exit and the opposite entry each pay the taker rate on their own notional; both
    SELL fills sit at the 10105.5 tick or one tick worse (never better)."""
    run = run_screening(catalog, reverse_dsl("rev_r8b"), funding_archive=empty_archive(tmp_path / "f.db"))

    from vibe_quant.fees import FeeSchedule

    taker = float(FeeSchedule().taker_rate)
    (exit_fill,) = market_exits(run)
    _, short_entry = entries(run)
    tick_px = 10_000.0 * (1 + 1e-4 * 105.5)
    for fill in (exit_fill, short_entry):
        assert fill.px in (pytest.approx(tick_px, abs=1e-6), pytest.approx(tick_px - 0.1, abs=1e-6))
        assert fill.commission == pytest.approx(fill.qty * fill.px * taker, rel=1e-9)
        assert fill.commission > 0
