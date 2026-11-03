"""Strategies size and margin-check on the costed balance (design-costs.md T2, vibe-quant-q3eq5.2).

NT books commissions only: funding and (validation) slippage are charged AFTER the run. The
runner installs a :class:`~vibe_quant.modeled_costs.CostLedger` on every backtest strategy and the
generated sizing subtracts what it has paid through the sizing instant. These tests run REAL
NautilusTrader screening runs on a toy catalog (flat 100 USDT, then a jump) with a temp funding
archive at +0.0001 per 8h, so the first trade's funding is a hand count of settlements.
"""

from __future__ import annotations

import math
import types
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.objects import Price, Quantity

from tests.unit.test_audit_dsl_runtime import _BAR, _Acct, _sizer
from tests.unit.test_backtest_margin_limit import _compiled, _fake_self
from tests.unit.test_screening_funding_exit_timing import interval_archive
from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.errors import DataUnavailableError
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation.exit_timing import MissingExitBars
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    from pathlib import Path

_MIN_MS = 60_000
_H_MS = 3_600_000
_DAY0_MS = 1_704_067_200_000  # 2024-01-01T00:00Z
_START, _END = "2024-01-01", "2024-01-06"
_N_MIN = 5 * 1440
RATE = 0.0001


def _write_catalog(
    root: Path,
    price_at: Any,
    timeframes: tuple[str, ...] = ("1m", "1h"),
    n_min: int = _N_MIN,
    symbols: dict[str, Any] | None = None,
    marks: bool = False,
) -> Path:
    """1m bars (flat OHLC = ``price_at(minute)``) plus aggregated coarser bars, per symbol.

    ``symbols`` maps a symbol to its own price function (default: BTCUSDT on ``price_at``).
    """
    cm = CatalogManager(root)
    for symbol, fn in (symbols or {"BTCUSDT": price_at}).items():
        inst = create_instrument(symbol)
        cm.write_instrument(inst)
        klines: list[Any] = []
        for i in range(n_min):
            px = fn(i)
            klines.append(
                {"open_time": _DAY0_MS + i * _MIN_MS, "open": px, "high": px, "low": px,
                 "close": px, "volume": 10.0, "close_time": _DAY0_MS + i * _MIN_MS + 59_999}
            )
        bars_1m = klines_to_bars(klines, inst.id, get_bar_type(symbol, "1m"),
                                 inst.size_precision, inst.price_precision)
        cm.write_bars(bars_1m)
        if marks:  # MARK 1m bars for the liquidation overlay, same prices as LAST
            from nautilus_trader.model.data import Bar
            from nautilus_trader.model.enums import PriceType
            from nautilus_trader.model.objects import Price as NtPrice

            mark_type = get_bar_type(symbol, "1m", price_type=PriceType.MARK)
            cm.write_bars([
                Bar(mark_type, NtPrice(fn(i), inst.price_precision), NtPrice(fn(i), inst.price_precision),
                    NtPrice(fn(i), inst.price_precision), NtPrice(fn(i), inst.price_precision),
                    Quantity(1, 0), (_DAY0_MS + i * _MIN_MS) * 1_000_000,
                    (_DAY0_MS + i * _MIN_MS + 59_999) * 1_000_000)
                for i in range(n_min)
            ])
        for tf, minutes in (("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1440)):
            if tf in timeframes:
                cm.write_bars(aggregate_bars(bars_1m, get_bar_type(symbol, tf), minutes,
                                             inst.size_precision, inst.price_precision))
    return root


def _jump(minute: int, before: float = 100.0, after: float = 120.0) -> Any:
    return lambda i: before if i < minute else after


def _dsl(tf: str = "1h", tp_pct: float = 10.0) -> dict[str, Any]:
    """Always-long: every flat bar re-enters; a 10% take profit closes a cycle."""
    return {
        "name": f"costed_{tf}",
        "timeframe": tf,
        "indicators": {"ema": {"type": "EMA", "period": 3}},
        "entry_conditions": {"long": ["ema > 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": 50.0},
        "take_profit": {"type": "fixed_pct", "percent": tp_pct},
    }


def _archive(tmp_path: Path, rate: float = RATE, symbols: tuple[str, ...] = ("BTCUSDT",)) -> Path:
    rows = [(_DAY0_MS + t * _H_MS, 8) for t in range(-24, 24 * 14, 8)]
    return interval_archive(tmp_path / "funding.db", {s: rows for s in symbols}, rate=rate)


def _launch(**sizing: Any) -> LaunchSettings:
    return LaunchSettings(sizing=SizingSettings(max_position_pct=1.0, risk_per_trade=0.5, **sizing))


class _Run:
    """One real screening run plus what the engine saw (captured through ``_extract_metrics``)."""

    def __init__(self) -> None:
        self.metrics: Any = None
        self.fills: list[tuple[str, str, float, float, int]] = []  # id, side, qty, px, ts_event
        self.positions: list[Any] = []
        self.balances: list[tuple[int, float]] = []  # (ts_event, USDT total) per account event
        self.strategies: list[Any] = []
        self.ledgers: list[Any] = []
        self.skips: list[dict[str, int]] = []  # per strategy, read before the engine reset


def run_screening(
    monkeypatch: pytest.MonkeyPatch,
    catalog: Path,
    archive: Path,
    dsl: dict[str, Any],
    *,
    launch: LaunchSettings | None = None,
    symbols: list[str] | None = None,
    start: str = _START,
    end: str = _END,
    **kwargs: Any,
) -> _Run:
    seen = _Run()
    orig = NTScreeningRunner._extract_metrics

    def _spy(self: NTScreeningRunner, params: Any, bt_result: Any, engine: Any,
             *a: Any, **k: Any) -> Any:
        cache = engine.kernel.cache
        for order in sorted(cache.orders(), key=lambda o: o.ts_init):
            for ev in order.events:
                if isinstance(ev, OrderFilled):
                    seen.fills.append((str(order.instrument_id), ev.order_side.name,
                                       float(ev.last_qty), float(ev.last_px), int(ev.ts_event)))
        seen.fills.sort(key=lambda f: f[4])
        seen.positions = sorted(
            [*cache.positions(), *cache.position_snapshots()], key=lambda p: p.ts_opened
        )
        account = cache.accounts()[0]
        seen.balances = [
            (int(e.ts_event), float(b.total.as_double()))
            for e in account.events
            for b in e.balances
        ]
        seen.strategies = list(engine.trader.strategies())
        seen.ledgers.append(k.get("cost_ledger"))
        seen.skips = [dict(getattr(s, "_sizing_skips", {})) for s in seen.strategies]
        return orig(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(NTScreeningRunner, "_extract_metrics", _spy)
    runner = NTScreeningRunner(
        dsl, symbols or ["BTCUSDT"], start, end, catalog_path=str(catalog),
        funding_archive_path=str(archive), launch=launch or _launch(), **kwargs,
    )
    seen.metrics = runner({})
    return seen


def _settlements_between(entry_ns: int, exit_ns: int) -> int:
    """Hand count of 8h settlements (00/08/16 UTC) in ``(entry_ns, exit_ns]``."""
    step = 8 * 3_600 * 1_000_000_000
    return exit_ns // step - entry_ns // step


def test_screening_entry_sized_on_costed_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: the second entry is sized on NT balance minus the first trade's funding."""
    catalog = _write_catalog(tmp_path / "cat", _jump(30 * 60))  # jump at Jan-2 06:00
    run = run_screening(monkeypatch, catalog, _archive(tmp_path), _dsl())
    buys = [f for f in run.fills if f[1] == "BUY"]
    sells = [f for f in run.fills if f[1] == "SELL"]
    assert len(buys) >= 2 and len(sells) >= 1
    first, second = buys[0], buys[1]
    exit_ts = sells[0][4]
    # Jan-1 03:00:59.999 -> Jan-2 06:00:59.999: settlements 08:00, 16:00, 00:00 (3 x 0.0001 x 1000)
    assert _settlements_between(first[4], exit_ts) == 3
    paid = 3 * RATE * first[2] * first[3]
    assert paid == pytest.approx(0.3)
    nt_balance = 1000.0 + run.positions[0].realized_pnl.as_double()
    assert nt_balance == pytest.approx(1098.2802)
    # sized on the bar close 120.0 in 0.001 steps: costed balance 1097.9802, NT balance 1098.2802
    assert second[2] == 9.149
    # outbox mode returns early from on_stop, yet the ledger was told the run stopped first
    assert run.ledgers[0].stopped_ns is not None
    assert math.floor(nt_balance / 120.0 * 1000) / 1000 == 9.152  # the pre-fix size


# ---------------------------------------------------------------- 1d refined exit (R5/R6/R7/R7b)

_DAY = 1440
_END_1D = "2024-01-12"
_TP_MINUTE = 5 * _DAY + 720  # Jan-6 12:00: the day's 1m bar that reaches the 110 take profit


def run_1d(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> _Run:
    catalog = _write_catalog(
        tmp_path / "cat", _jump(_TP_MINUTE), timeframes=("1m", "1d"), n_min=11 * _DAY
    )
    return run_screening(
        monkeypatch, catalog, _archive(tmp_path), _dsl("1d"), start=_START, end=_END_1D, **kwargs
    )


def test_close_finalized_before_same_bar_sizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6 + R5: the 1d take profit fills at the Jan-6 23:59:59.999 boundary and the next entry
    is sized on that SAME instant, so the close must already be finalized -- with funding to
    the refined 1m touch (Jan-6 12:00:59.999: 7 settlements, 0.7), not to the boundary (8)."""
    run = run_1d(monkeypatch, tmp_path)
    buys = [f for f in run.fills if f[1] == "BUY"]
    sells = [f for f in run.fills if f[1] == "SELL"]
    assert sells[0][4] == 1_704_585_599_999_000_000  # Jan-6 23:59:59.999, the boundary
    assert buys[1][4] - sells[0][4] == 60_000_000_000  # the next fill tick
    nt_balance = 1000.0 + run.positions[0].realized_pnl.as_double()
    assert nt_balance == pytest.approx(1098.2802)
    # costed 1098.2802 - 0.7 = 1097.5802 -> 9.146 (boundary funding 0.8 would size 9.145; the
    # unfinalized close, i.e. NT's balance alone, 9.152)
    assert buys[1][2] == 9.146
    assert run.ledgers[0].finalized[next(iter(run.ledgers[0].finalized))] == pytest.approx(0.7)


def test_missing_exit_bars_fails_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R7: a refined 1d exit with no 1m LAST window for it fails the run with MissingExitBars
    (a DataUnavailableError, raised from the strategy's on_event), never a -inf score."""
    monkeypatch.setattr("vibe_quant.validation.extraction.load_bar_windows", lambda *a, **k: None)
    with pytest.raises(MissingExitBars) as err:
        run_1d(monkeypatch, tmp_path)
    assert isinstance(err.value, DataUnavailableError)


def test_other_refinement_error_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R7b: any other failure inside the refinement is the R10b wrap, same text as post-hoc."""

    def boom(*_a: Any, **_k: Any) -> None:
        raise ValueError("order SL-1 is not in the engine cache")

    monkeypatch.setattr("vibe_quant.screening.liquidation_metrics.resting_exit_inputs", boom)
    with pytest.raises(DataUnavailableError) as err:
        run_1d(monkeypatch, tmp_path)
    assert str(err.value) == (
        "funding exit refinement (1d): ValueError: order SL-1 is not in the engine cache"
    )


# ------------------------------------------------------------------ R10: costed <= 0 skips

def test_costed_nonpositive_skips_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R10: funding at 0.5 per settlement (3 x 500 = 1500 on 1000) leaves 98.28 - 1500 < 0:
    every later entry is skipped (counted under ``no_equity``), the open run is untouched
    (no bust, no stop) and NT still holds a balance of 1098.28."""
    catalog = _write_catalog(tmp_path / "cat", _jump(30 * 60))
    run = run_screening(monkeypatch, catalog, _archive(tmp_path, rate=0.5), _dsl())
    assert [f[1] for f in run.fills] == ["BUY", "SELL"]
    (skips,) = run.skips
    assert set(skips) == {"no_equity"}
    assert skips["no_equity"] >= 10
    nt_balance = 1000.0 + run.positions[0].realized_pnl.as_double()
    assert nt_balance == pytest.approx(1098.2802)  # NT alone would have re-entered


def test_costed_nonpositive_log_text() -> None:
    """R10: the skip line keeps its text with the COSTED value (1000 - 1012.5 = -12.5)."""
    fake = _sizer(_Acct({USDT: 1000.0}))
    seen: list[tuple[int, Any]] = []

    def paid_through(now_ns: int, open_positions: Any) -> float:
        seen.append((now_ns, open_positions))
        return 1012.5

    fake._cost_ledger = types.SimpleNamespace(paid_through=paid_through)
    fake.cache.positions_open = lambda venue=None: ["open-pos"]
    bar = types.SimpleNamespace(close=Price.from_str("60000.0"), ts_init=123)
    assert fake._calculate_position_size(bar, True) is None
    assert fake.log.warnings == ["Sizing: no positive USDT equity (-12.5) -- entry skipped"]
    assert seen == [(123, ["open-pos"])]


def test_costed_equity_zero_edge() -> None:
    """R10 edge: costed exactly 0 is skipped as ``no_equity`` (``<= 0``); a cent above it is
    not (it is rejected later, by the instrument minimum, under another reason)."""
    bar = types.SimpleNamespace(close=Price.from_str("1.0"), ts_init=1)
    reasons = {}
    for paid in (1000.0, 999.99):
        fake = _sizer(_Acct({USDT: 1000.0}), max_pos=1.0, sl_pct=50.0, risk=0.5)
        fake._cost_ledger = types.SimpleNamespace(paid_through=lambda *_a, p=paid: p)
        fake.cache.positions_open = lambda venue=None: []
        assert fake._calculate_position_size(bar, True) is None
        reasons[paid] = set(fake._sizing_skips)
    assert reasons == {1000.0: {"no_equity"}, 999.99: {"min_notional"}}


# ---------------------------------------------------------------- R16: paper has no ledger

def test_paper_strategy_has_no_cost_ledger() -> None:
    """R16: no ledger -> equity is the venue balance exactly (the venue already books funding
    and slippage); only ``engine_run`` (every NT backtest engine) and the fast-tier replay
    (``fills.py``, which replaces the engine) install one -- never a paper/live path."""
    import ast
    from pathlib import Path as _P

    stub = _sizer(_Acct({USDT: 1050.0}))  # no _cost_ledger attribute at all
    assert getattr(stub, "_cost_ledger", None) is None
    assert stub._calculate_position_size(_BAR, True) == Quantity.from_str("0.008")  # 525/60000

    root = _P(__file__).resolve().parents[2] / "vibe_quant"
    assigners = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            targets = node.targets if isinstance(node, ast.Assign) else []
            if any(isinstance(t, ast.Attribute) and t.attr == "_cost_ledger" for t in targets):
                assigners.append(path.name)
    assert assigners == ["engine_run.py", "fills.py"]


# ------------------------------------------------- R17 zero costs, R18 overlay, R19 identity

def _entries(run: _Run, symbol: str = "BTCUSDT-PERP.BINANCE") -> list[tuple[Any, ...]]:
    return [f for f in run.fills if f[1] == "BUY" and f[0] == symbol]


def test_zero_costs_bit_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R17: funding archived at rate 0.0 charges nothing, so the costed run is bit-identical to
    the pre-ledger tree (these literals were produced by main ce87a87f, fills and metrics)."""
    catalog = _write_catalog(tmp_path / "cat", _jump(30 * 60))
    run = run_screening(monkeypatch, catalog, _archive(tmp_path, rate=0.0), _dsl())
    assert run.fills == [
        ("BTCUSDT-PERP.BINANCE", "BUY", 10.0, 100.0, 1704078059999000000),
        ("BTCUSDT-PERP.BINANCE", "SELL", 10.0, 109.9, 1704175259999000000),
        ("BTCUSDT-PERP.BINANCE", "BUY", 9.152, 120.1, 1704178859999000000),
    ]
    m = run.metrics
    assert (m.sharpe_ratio, m.total_trades, m.total_return) == (
        7.739837154601049, 2, 0.09626630239999995
    )
    assert (m.profit_factor, m.max_drawdown, m.total_funding, m.total_fees) == (
        48.80099166909108, 0.0018336828798334555, 0.0, 1.8184976
    )


def test_overlay_on_sizes_on_costed_balance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R18: liquidation mode on keeps the NT path sized on the costed balance and reconciles on
    pass 1 (a mismatch would raise LiquidationIntegrityError)."""
    catalog = _write_catalog(tmp_path / "cat", _jump(30 * 60), marks=True)
    launch = LaunchSettings(
        sizing=SizingSettings(max_position_pct=1.0, risk_per_trade=0.5),
        liquidation=LiquidationSettings(mode="isolated"),
    )
    run = run_screening(monkeypatch, catalog, _archive(tmp_path), _dsl(), launch=launch)
    assert [f[2] for f in _entries(run)] == [10.0, 9.149]
    assert run.metrics.liquidation_count == 0


def _staircase(minute: int) -> float:
    """+20% every 6 hours: every 10% take profit closes a cycle and the next entry follows."""
    return round(100.0 * 1.2 ** (minute // 360), 1)


def test_ledger_identity_flat_single_symbol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R19: over many cycles every entry is sized on 1000 + sum(prior net PnL), net PnL being
    NT's realized PnL (commissions in) minus the hand-counted funding (settlements x rate x
    entry notional); the closes are what the ledger finalized, and the costed balance equals the
    post-hoc ledger within Money's 1e-8 rounding."""
    catalog = _write_catalog(tmp_path / "cat", _staircase)
    run = run_screening(monkeypatch, catalog, _archive(tmp_path), _dsl())
    closed = [p for p in run.positions if p.is_closed]
    assert len(closed) >= 8
    buys = _entries(run)
    balance = 1000.0
    for k, buy in enumerate(buys):
        close = _staircase((buy[4] - _DAY0_MS * 1_000_000) // 60_000_000_000 - 1)
        assert buy[2] == math.floor(balance / close * 1000 + 1e-9) / 1000, k
        if k < len(closed):
            pos = closed[k]
            funding = (
                _settlements_between(int(pos.ts_opened), int(pos.ts_closed))
                * RATE * float(pos.peak_qty) * float(pos.avg_px_open)
            )
            balance += pos.realized_pnl.as_double() - funding
    assert sum(run.ledgers[0].finalized.values()) == pytest.approx(
        sum(
            _settlements_between(int(p.ts_opened), int(p.ts_closed))
            * RATE * float(p.peak_qty) * float(p.avg_px_open)
            for p in closed
        ),
        abs=1e-6,
    )


# ------------------------------------------------------------------ R11 margin on costed equity

def _margin_fake(paid: float) -> Any:
    fake = _fake_self(_compiled()[0], open_orders=[], inflight=[])
    seen: list[tuple[int, Any]] = []
    flat = types.SimpleNamespace(  # zero quantity: no margin of its own
        instrument_id=fake.instrument_id,
        quantity=Quantity.from_str("0.000"),
        avg_px_open=Price.from_str("100.0"),
    )
    fake.open_positions = [flat]
    fake.cache.positions_open = lambda venue=None: fake.open_positions
    fake.clock = types.SimpleNamespace(timestamp_ns=lambda: 777)

    def paid_through(now_ns: int, open_positions: Any) -> float:
        seen.append((now_ns, open_positions))
        return paid

    fake._cost_ledger = types.SimpleNamespace(paid_through=paid_through)
    fake.paid_calls = seen
    return fake


def test_margin_limit_reads_costed_equity() -> None:
    """R11: equity = portfolio equity 1000 - paid 800 = 200, read at the release clock; at
    leverage 10 and a 0.5 limit, 100 of margin (10 @ 100) is exactly allowed and 10.01 refused
    (on NT's 1000 both would pass). The open positions are the cache's."""
    from nautilus_trader.model.objects import Quantity as Q

    fake = _margin_fake(800.0)
    assert fake._entry_margin_ok(Q.from_str("10.00"), 100.0) is True, fake.warnings
    assert fake._entry_margin_ok(Q.from_str("10.01"), 100.0) is False
    assert fake._margin_refusals == 1
    assert fake.paid_calls == [(777, fake.open_positions)] * 2
    # without a ledger the same refused size passes on NT's 1000
    bare = _fake_self(_compiled()[0], open_orders=[], inflight=[])
    assert bare._entry_margin_ok(Q.from_str("10.01"), 100.0) is True


def test_margin_check_costs_are_read_at_release_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R11 in a real run: with the margin limit on, the ledger is also asked at the venue-call
    (fill-tick) instant -- :00:59.999 -- not only at the bar closes (:59:59.999) sizing uses."""
    from vibe_quant.modeled_costs import CostLedger

    def residues(launch: LaunchSettings, sub: str) -> set[int]:
        times: list[int] = []
        orig = CostLedger.paid_through

        def spy(self: CostLedger, now_ns: int, open_positions: Any) -> float:
            times.append(now_ns)
            return orig(self, now_ns, open_positions)

        monkeypatch.setattr(CostLedger, "paid_through", spy)
        catalog = _write_catalog(tmp_path / sub, _jump(30 * 60))
        run_screening(monkeypatch, catalog, _archive(tmp_path), _dsl(), launch=launch)
        monkeypatch.setattr(CostLedger, "paid_through", orig)
        return {t % 3_600_000_000_000 for t in times}

    off = residues(_launch(), "off")
    on = residues(_launch(max_margin_pct=0.9), "on")
    assert off == {3_599_999_000_000}
    assert on == {3_599_999_000_000, 59_999_000_000}


# ---------------------------------------------------------------------- R20 fresh ledgers

def test_each_window_has_its_own_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R20: a later window (WFA / CV / eval window) is its own engine with its own EMPTY ledger:
    nothing is carried over from the earlier window, and its first entry is sized on 1000."""
    catalog = _write_catalog(tmp_path / "cat", _jump(30 * 60))
    archive = _archive(tmp_path)
    first = run_screening(monkeypatch, catalog, archive, _dsl(), start="2024-01-01", end="2024-01-04")
    later = run_screening(monkeypatch, catalog, archive, _dsl(), start="2024-01-03", end="2024-01-06")
    assert first.ledgers[0] is not later.ledgers[0]
    assert len(first.ledgers[0].finalized) == 1
    assert later.ledgers[0].finalized == {}  # the price stays at 120: no closed cycle
    assert [f[2] for f in _entries(later)] == [8.333]  # 1000 / 120.0 bar close (fresh account)


# --------------------------------------------- multi-symbol funding and the equal-ts note (R24)

def test_other_symbols_open_funding_counts_while_one_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Note (4): BTC stays open (flat 100) while ETH cycles; ETH's second entry is sized on the
    shared NT balance minus ETH's closed funding (5 settlements x 0.0001 x 1000 = 0.5) AND
    BTC's open funding to the sizing bar (the same 5 settlements: 0.5)."""
    catalog = _write_catalog(
        tmp_path / "cat",
        None,
        symbols={"BTCUSDT": lambda i: 100.0, "ETHUSDT": _jump(42 * 60)},  # ETH jumps Jan-2 18:00
    )
    run = run_screening(
        monkeypatch, catalog, _archive(tmp_path, symbols=("BTCUSDT", "ETHUSDT")), _dsl(),
        symbols=["BTCUSDT", "ETHUSDT"],
    )
    eth = "ETHUSDT-PERP.BINANCE"
    eth_buys = _entries(run, eth)
    (btc_buy,) = _entries(run)
    eth1 = next(p for p in run.positions if "ETHUSDT" in str(p.instrument_id) and p.is_closed)
    btc_fee = sum(abs(c.as_double()) for c in next(
        p for p in run.positions if "BTCUSDT" in str(p.instrument_id)).commissions())
    assert [f[1] for f in run.fills if f[0] == eth][:2] == ["BUY", "SELL"]
    nt_balance = 1000.0 + eth1.realized_pnl.as_double() - btc_fee
    paid = 0.5 + 0.5
    assert btc_buy[2] == 10.0
    assert eth_buys[1][2] == math.floor((nt_balance - paid) / 120.0 * 1000) / 1000
    assert eth_buys[1][2] < math.floor((nt_balance - 0.5) / 120.0 * 1000) / 1000  # BTC's counts


def test_equal_instant_sizing_inherits_same_ts_order_sensitivity(tmp_path: Path) -> None:
    """R24 (documented limit): at one instant T, Y's sizing counts X's position as OPEN (funded
    through T, 2 settlements = 0.2) if X's close was not yet processed, or as FINALIZED (funded
    to its exit, 1 settlement = 0.1) if it was -- exactly the order sensitivity NT's own
    balance already has at equal timestamps."""
    from nautilus_trader.model.enums import OrderSide

    from tests.unit.fake_engine import FakePosition
    from vibe_quant.modeled_costs import CostLedger, CostModel
    from vibe_quant.validation.funding import FundingCalculator

    archive = _archive(tmp_path)
    ns = 1_000_000
    entry = (_DAY0_MS + 59_999) * ns  # Jan-1 00:00:59.999
    exit_ = (_DAY0_MS + 12 * _H_MS + 59_999) * ns  # 12:00:59.999: 08:00 settled
    now = (_DAY0_MS + 24 * _H_MS - 1) * ns  # 23:59:59.999: 16:00 settled too
    pos = FakePosition.closed(
        "X", "BTCUSDT-PERP.BINANCE", entry_ns=entry, exit_ns=exit_, entry_price=100.0,
        exit_price=100.0, qty=10.0, entry_side=OrderSide.BUY,
    )
    model = CostModel(FundingCalculator(str(archive)), frozenset(), "1h", 3_600_000_000_000, None)
    unprocessed = CostLedger(model, None)
    assert unprocessed.paid_through(now, [pos]) == pytest.approx(0.2)
    processed = CostLedger(model, None)
    processed.on_position_closed(pos)
    assert processed.paid_through(now, []) == pytest.approx(0.1)


def test_stop_closes_are_allowed_only_after_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13 in a real run: a 1m strategy (no outbox) has NT's ``on_stop`` close its open position
    at the run end. The ledger never finalizes that close, which reconcile allows only because
    ``stopped`` ran first -- before the outbox early return, which 4h/1d runs take."""
    catalog = _write_catalog(tmp_path / "cat", lambda i: 100.0, timeframes=("1m",), n_min=2 * _DAY)
    run = run_screening(
        monkeypatch, catalog, _archive(tmp_path), _dsl("1m", tp_pct=90.0),
        start="2024-01-01", end="2024-01-03",
    )
    (ledger,) = run.ledgers
    assert ledger.stopped_ns is not None
    assert run.metrics.sharpe_ratio != float("-inf")
    assert [p.is_closed for p in run.positions] == [True]  # closed by on_stop, never finalized
    assert ledger.finalized == {}
