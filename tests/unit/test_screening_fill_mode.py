"""Screening releases orders through the command outbox on fill ticks (vibe-quant-yul7u.10).

Mode keyed on the finest LOADED timeframe:
- 1m strategy -> unchanged (execution_delay_probability 1.0, no ticks, no outbox)
- coarser strategy that loads 1m (indicator / additional TF) -> outbox "bar" on its own 1m
- coarser -> outbox "trade_tick" on the resolved fill-tick set (one TradeTick config per symbol)
Outbox modes run with delay 0 and no LatencyModel; end-of-run open positions are marked
to market like validation; symbol order never changes results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.data import TradeTick

from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.data.fill_ticks import FillTickSet, fill_tick_dir_signature
from vibe_quant.screening.nt_runner import (
    FillTickCatalogModifiedError,
    MissingBarDataError,
    NTScreeningRunner,
    UnknownStrategyParamError,
    resolve_strategy_params,
    screening_fill_mode,
)

if TYPE_CHECKING:
    from nautilus_trader.config import BacktestRunConfig

_MIN_MS = 60_000
_DAY0_MS = 1_704_067_200_000  # 2024-01-01T00:00Z
_START, _END = "2024-01-01", "2024-01-05"
_START_NS, _END_NS = 1_704_067_200_000_000_000, 1_704_412_800_000_000_000  # _START, _END
_N_MIN = 5 * 1440  # 1m bars 2024-01-01 .. 2024-01-05 23:59


def _dsl(tf: str = "4h", name: str = "fill_mode", ind_tf: str | None = None) -> dict[str, Any]:
    """Always-long strategy with out-of-reach SL/TP (rising market): one entry, open at end.

    SL 10%: risk 2% of the 1000 USDT screening balance -> 200 USDT notional, above
    BTC min_notional 50 (a 50% SL sized 40 USDT and skipped every entry).
    """
    ema: dict[str, Any] = {"type": "EMA", "period": 3}
    if ind_tf:
        ema["timeframe"] = ind_tf
    d: dict[str, Any] = {
        "name": f"{name}_{tf}_{ind_tf or 'x'}",
        "timeframe": tf,
        "indicators": {"ema": ema},
        "entry_conditions": {"long": ["ema > 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": 10.0},
        "take_profit": {"type": "fixed_pct", "percent": 90.0},
    }
    if ind_tf:
        d["additional_timeframes"] = [ind_tf]
    return d


def _write_catalog(root: Path, symbols: tuple[str, ...] = ("BTCUSDT",), n_min: int = _N_MIN,
                   timeframes: tuple[str, ...] = ("1m", "4h")) -> Path:
    """Rising synthetic 1m bars (+0.01%/min, >= 10 price ticks) plus aggregated coarser bars."""
    cm = CatalogManager(root)
    for k, symbol in enumerate(symbols):
        inst = create_instrument(symbol)
        cm.write_instrument(inst)
        base = 10_000.0 * (k + 1)
        klines: list[Any] = [
            {"open_time": _DAY0_MS + i * _MIN_MS, "open": base * (1 + 1e-4 * i),
             "high": base * (1 + 1e-4 * (i + 1)), "low": base * (1 + 1e-4 * (i - 1)),
             "close": base * (1 + 1e-4 * (i + 0.5)), "volume": 10.0,
             "close_time": _DAY0_MS + i * _MIN_MS + 59_999}
            for i in range(n_min)
        ]
        bars_1m = klines_to_bars(klines, inst.id, get_bar_type(symbol, "1m"),
                                 inst.size_precision, inst.price_precision)
        if "1m" in timeframes:
            cm.write_bars(bars_1m)
        for tf, minutes in (("15m", 15), ("1h", 60), ("4h", 240)):
            if tf in timeframes:
                cm.write_bars(aggregate_bars(bars_1m, get_bar_type(symbol, tf), minutes,
                                             inst.size_precision, inst.price_precision))
    return root


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    return _write_catalog(tmp_path / "catalog", ("BTCUSDT", "ETHUSDT"))


def _runner(catalog: Path, dsl: dict[str, Any], symbols: list[str] | None = None,
            **kw: Any) -> NTScreeningRunner:
    return NTScreeningRunner(dsl, symbols or ["BTCUSDT"], _START, _END,
                             catalog_path=str(catalog), **kw)


def _config(runner: NTScreeningRunner) -> BacktestRunConfig:
    run_config, _ = runner._build_run_config({})
    assert run_config is not None
    return run_config


def _strategy_cfgs(run_config: BacktestRunConfig) -> list[dict[str, Any]]:
    assert run_config.engine is not None
    return [dict(s.config) for s in run_config.engine.strategies]


def _tick_cfgs(run_config: BacktestRunConfig) -> list[Any]:
    return [d for d in run_config.data if "TradeTick" in str(d.data_cls)]


# ---------------------------------------------------------------------------
# 1. mode table
# ---------------------------------------------------------------------------


class TestModeTable:
    def test_pure_mode_function(self) -> None:
        assert screening_fill_mode("4h", {"4h"}) == ("trade_tick", 0.0, "4h")
        assert screening_fill_mode("4h", {"4h", "1d"}) == ("trade_tick", 0.0, "4h")
        assert screening_fill_mode("4h", {"4h", "1h"}) == ("trade_tick", 0.0, "1h")
        assert screening_fill_mode("4h", {"4h", "1m"}) == ("bar", 0.0, None)
        assert screening_fill_mode("1m", {"1m"}) == ("", 1.0, None)
        assert screening_fill_mode("1m", {"1m", "4h"}) == ("", 1.0, None)

    def test_4h_only_is_trade_tick_on_resolved_ticks(self, catalog: Path) -> None:
        run_config = _config(_runner(catalog, _dsl("4h")))
        (cfg,) = _strategy_cfgs(run_config)
        assert cfg["command_release"] == "trade_tick"
        assert cfg["execution_delay_probability"] == 0.0
        assert "command_release_bar_type" not in cfg or not cfg["command_release_bar_type"]
        (tick,) = _tick_cfgs(run_config)
        assert Path(tick.catalog_path).parts[-3:-1] == ("4h", "BTCUSDT")
        assert str(tick.instrument_id) == "BTCUSDT-PERP.BINANCE"
        assert (tick.start_time, tick.end_time) == (_START, _END)

    def test_4h_with_1m_indicator_is_bar_on_own_1m(self, catalog: Path) -> None:
        run_config = _config(_runner(catalog, _dsl("4h", ind_tf="1m")))
        (cfg,) = _strategy_cfgs(run_config)
        assert cfg["command_release"] == "bar"
        assert cfg["command_release_bar_type"] == "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
        assert cfg["execution_delay_probability"] == 0.0
        assert _tick_cfgs(run_config) == []

    def test_1m_strategy_unchanged(self, catalog: Path) -> None:
        run_config = _config(_runner(catalog, _dsl("1m")))
        (cfg,) = _strategy_cfgs(run_config)
        assert cfg["execution_delay_probability"] == 1.0
        assert not cfg.get("command_release")
        assert _tick_cfgs(run_config) == []


# ---------------------------------------------------------------------------
# 2. TradeTick configs use the data_cls CLASS
# ---------------------------------------------------------------------------


def test_trade_tick_configs_use_class_object(catalog: Path) -> None:
    run_config = _config(_runner(catalog, _dsl("4h"), ["ETHUSDT", "BTCUSDT"]))
    ticks = _tick_cfgs(run_config)
    assert len(ticks) == 2
    assert all(t.data_cls is TradeTick for t in ticks)
    assert [str(t.instrument_id) for t in ticks] == ["BTCUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE"]


# ---------------------------------------------------------------------------
# 3. missing / partial ticks raise before the run config is built
# ---------------------------------------------------------------------------


@pytest.fixture
def no_run_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import nautilus_trader.config as ntc

    def _boom(**_: Any) -> None:
        raise AssertionError("BacktestRunConfig built before the fill-tick check")

    monkeypatch.setattr(ntc, "BacktestRunConfig", _boom)


@pytest.mark.usefixtures("no_run_config")
class TestMissingTicks:
    def test_no_1m_data_raises(self, tmp_path: Path) -> None:
        cat = _write_catalog(tmp_path / "catalog", timeframes=("4h",))
        with pytest.raises(MissingBarDataError, match="1m"):
            _runner(cat, _dsl("4h"))({})

    def test_partial_1m_data_raises(self, tmp_path: Path) -> None:
        cat = _write_catalog(tmp_path / "catalog", n_min=2 * 1440, timeframes=("1m",))
        _write_catalog(tmp_path / "catalog", timeframes=("4h",))  # full 4h, half 1m
        with pytest.raises(MissingBarDataError, match="do not cover"):
            _runner(cat, _dsl("4h"))({})

    def test_given_set_missing_a_symbol_raises(self, catalog: Path) -> None:
        ticks = FillTickSet(timeframe="4h", paths={"BTCUSDT": str(catalog)},
                            missing_boundaries={"BTCUSDT": 0}, start_ns=_START_NS, end_ns=_END_NS)
        with pytest.raises(MissingBarDataError, match="ETHUSDT"):
            _runner(catalog, _dsl("4h"), ["BTCUSDT", "ETHUSDT"], fill_ticks=ticks)({})

    def test_given_set_wrong_timeframe_raises(self, catalog: Path) -> None:
        ticks = FillTickSet(timeframe="1h", paths={"BTCUSDT": str(catalog)},
                            missing_boundaries={"BTCUSDT": 0}, start_ns=_START_NS, end_ns=_END_NS)
        with pytest.raises(MissingBarDataError, match="1h"):
            _runner(catalog, _dsl("4h"), fill_ticks=ticks)({})


@pytest.mark.usefixtures("no_run_config")
@pytest.mark.parametrize(
    ("start_ns", "end_ns"),
    [
        (None, None),  # window unknown
        (_START_NS + 1, _END_NS),  # resolved for a later start
        (_START_NS, _END_NS - 1),  # resolved for an earlier end
        (_END_NS, _END_NS + 10**15),  # a different window entirely
    ],
)
def test_given_set_for_another_window_raises(
    catalog: Path, start_ns: int | None, end_ns: int | None
) -> None:
    ticks = FillTickSet(timeframe="4h", paths={"BTCUSDT": str(catalog)},
                        missing_boundaries={"BTCUSDT": 0}, start_ns=start_ns, end_ns=end_ns)
    with pytest.raises(MissingBarDataError, match="window"):
        _runner(catalog, _dsl("4h"), fill_ticks=ticks)({})


def test_parent_resolved_set_for_the_run_window_is_accepted(catalog: Path) -> None:
    from vibe_quant.data.fill_ticks import resolve_fill_ticks

    ticks = resolve_fill_ticks(["BTCUSDT"], "4h", _START, _END, catalog)
    assert (ticks.start_ns, ticks.end_ns) == (_START_NS, _END_NS)
    (tick,) = _tick_cfgs(_config(_runner(catalog, _dsl("4h"), fill_ticks=ticks)))
    assert tick.catalog_path == ticks.paths["BTCUSDT"]


def test_given_set_is_used_without_resolving(catalog: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vibe_quant.data.fill_ticks as ft

    def _no_resolve(*_: Any, **__: Any) -> None:
        raise AssertionError("resolve_fill_ticks called although a set was given")

    monkeypatch.setattr(ft, "resolve_fill_ticks", _no_resolve)
    # a set resolved for a wider window covers the run
    ticks = FillTickSet(timeframe="4h", paths={"BTCUSDT": "/x/BTC"}, missing_boundaries={"BTCUSDT": 3},
                        start_ns=_START_NS - 1, end_ns=_END_NS + 1)
    run_config = _config(_runner(catalog, _dsl("4h"), fill_ticks=ticks))
    (tick,) = _tick_cfgs(run_config)
    assert tick.catalog_path == "/x/BTC"


# ---------------------------------------------------------------------------
# 4. delay 0 and no latency in outbox modes; 6. raise_exception
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ind_tf", [None, "1m"])
def test_outbox_modes_have_no_delay_no_latency_and_raise(catalog: Path, ind_tf: str | None) -> None:
    run_config = _config(_runner(catalog, _dsl("4h", ind_tf=ind_tf)))
    (cfg,) = _strategy_cfgs(run_config)
    assert cfg["command_release"] in ("trade_tick", "bar")
    assert cfg["execution_delay_probability"] == 0.0
    (venue,) = run_config.venues
    assert venue.latency_model is None
    assert run_config.raise_exception is True


def test_1m_mode_raises_exceptions_too(catalog: Path) -> None:
    run_config = _config(_runner(catalog, _dsl("1m")))
    assert run_config.raise_exception is True


# ---------------------------------------------------------------------------
# 5. command_release keys are runner-owned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["command_release", "command_release_bar_type"])
def test_command_release_keys_rejected(key: str) -> None:
    fields = ("instrument_id", "ema_period", "command_release", "command_release_bar_type")
    assert resolve_strategy_params({"ema.period": 5}, fields) == {"ema_period": 5}
    with pytest.raises(UnknownStrategyParamError, match=key):
        resolve_strategy_params({key: "bar"}, fields)


# ---------------------------------------------------------------------------
# 7. open position at end of run counts in return/PnL; tick dirs untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tf", "ind_tf"), [("4h", None), ("4h", "1m"), ("1m", None)])
def test_open_position_at_end_is_marked(catalog: Path, tf: str, ind_tf: str | None) -> None:
    runner = _runner(catalog, _dsl(tf, ind_tf=ind_tf))
    _, tick_dirs = runner._build_run_config({})
    before = {d: fill_tick_dir_signature(d) for d in tick_dirs}
    metrics = runner({})
    assert metrics.sharpe_ratio != float("-inf")
    assert metrics.total_trades == 1
    # Rising market, one long never closed: unmarked it was only the entry fee (< 0)
    assert metrics.total_return > 0.01
    assert metrics.profit_factor > 1.0
    assert {d: fill_tick_dir_signature(d) for d in tick_dirs} == before


def _funding_archive(path: Path, rate: float = 0.0003) -> Path:
    """Constant funding for BTC/ETH at every 8h settlement 2024-01-01 .. 01-07."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for i in range(7 * 3):
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
                (symbol, _DAY0_MS + i * 8 * 3_600_000, rate),
            )
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize("ind_tf", [None, "1m"])
def test_open_position_mark_matches_validation_extraction(
    catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ind_tf: str | None
) -> None:
    """Same engine through screening's _extract_metrics and validation's extract_results:
    the open-position mark (exit fee in fees and DD cash, funding up to the mark, mark
    cash event on the Sharpe series) must agree on every headline number."""
    from vibe_quant.validation.extraction import extract_results
    from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

    clear_rate_cache()
    archive = _funding_archive(tmp_path / "raw.db")
    validation: list[Any] = []
    orig = NTScreeningRunner._extract_metrics

    def _spy(self: NTScreeningRunner, params: Any, bt_result: Any, engine: Any,
             *a: Any, **k: Any) -> Any:
        from vibe_quant.validation.extraction import finest_timeframe

        validation.append(extract_results(
            1, "s", bt_result, engine, self._venue_config(),
            funding_calculator=FundingCalculator(archive),
            run_start_date=self._start_date, run_end_date=self._end_date,
            execution_timeframe=finest_timeframe(self._all_timeframes),
            catalog_path=self._resolved_catalog_path,
        ))
        return orig(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(NTScreeningRunner, "_extract_metrics", _spy)
    m = _runner(catalog, _dsl("4h", name="parity", ind_tf=ind_tf), ["BTCUSDT", "ETHUSDT"],
                funding_archive_path=str(archive))({})
    (v,) = validation
    assert [t.exit_reason for t in v.trades] == ["end_of_data", "end_of_data"]
    assert v.total_slippage == 0.0  # screening fill model: no SPEC slippage on either side
    assert m.total_funding != 0.0 and m.funding_fallback_settlements == 0
    assert m.total_trades == v.total_trades == 2
    assert m.total_fees == pytest.approx(v.total_fees, rel=1e-12)
    assert m.total_funding == pytest.approx(v.total_funding, rel=1e-12)
    assert m.total_return == pytest.approx(v.total_return, rel=1e-12)
    assert m.profit_factor == pytest.approx(v.profit_factor, rel=1e-12)
    assert m.sharpe_ratio == pytest.approx(v.sharpe_ratio, rel=1e-12)
    assert m.max_drawdown == pytest.approx(v.max_drawdown, rel=1e-12)
    assert m.max_drawdown > 0.0


def test_tick_dir_modified_during_run_raises(catalog: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nautilus_trader.backtest.node import BacktestNode

    runner = _runner(catalog, _dsl("4h"))
    _, (tick_dir,) = runner._build_run_config({})
    orig = BacktestNode.dispose

    def _dispose(self: Any) -> None:
        orig(self)  # type: ignore[no-untyped-call]
        (Path(tick_dir) / "stray.parquet").write_bytes(b"x")

    monkeypatch.setattr(BacktestNode, "dispose", _dispose)
    try:
        with pytest.raises(FillTickCatalogModifiedError):
            runner({})
    finally:
        (Path(tick_dir) / "stray.parquet").unlink(missing_ok=True)


def test_each_symbol_fills_on_its_own_first_tick_any_order(
    catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entry fills at the symbol's OWN 1m close at boundary T (tick 1, stamped T+59.999s),
    in both symbol orders; the engine saw 2 fill ticks per 4h boundary per symbol."""
    from nautilus_trader.model.identifiers import InstrumentId

    seen: dict[str, tuple[int, float, int]] = {}
    orig = NTScreeningRunner._extract_metrics

    def _spy(self: NTScreeningRunner, params: Any, bt_result: Any, engine: Any,
             *a: Any, **k: Any) -> Any:
        cache = engine.kernel.cache
        for order in cache.orders():
            if order.order_type.name == "MARKET" and order.filled_qty.as_double() > 0:
                iid = str(order.instrument_id)
                n_ticks = len(cache.trade_ticks(InstrumentId.from_str(iid)))
                seen.setdefault(iid, (order.ts_last, float(order.avg_px), n_ticks))
        return orig(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(NTScreeningRunner, "_extract_metrics", _spy)
    results = []
    for symbols in (["BTCUSDT", "ETHUSDT"], ["ETHUSDT", "BTCUSDT"]):
        seen.clear()
        m = _runner(catalog, _dsl("4h", name="own_tick"), symbols)({})
        results.append((m.sharpe_ratio, m.total_return, m.profit_factor, m.max_drawdown))
        assert set(seen) == {"BTCUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE"}
        for k, (iid, (ts, px, n_ticks)) in enumerate(sorted(seen.items())):
            boundary = ts - 59_999_000_000
            assert boundary % (4 * 3_600_000_000_000) == 0, iid
            minute = (boundary // 1_000_000 - _DAY0_MS) // _MIN_MS
            own_close = 10_000.0 * (k + 1) * (1 + 1e-4 * (minute + 0.5))
            tick = 0.1 if iid.startswith("BTC") else 0.01
            assert abs(px - own_close) <= tick + 1e-9, (iid, px, own_close)
            # 5 days of 4h boundaries, 2 ticks each (the window end is exclusive of ticks)
            assert 2 * 6 * 4 <= n_ticks <= 2 * 6 * 5, (iid, n_ticks)
    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# 8. symbol order never changes results (real data: needs the shared FillModel RNG)
# ---------------------------------------------------------------------------

_REAL_CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog"


@pytest.mark.skipif(not (_REAL_CATALOG / "data" / "bar").is_dir(), reason="needs data/catalog")
def test_symbol_order_bit_identical() -> None:
    dsl = {
        "name": "fill_mode_order",
        "timeframe": "1h",
        "indicators": {"rsi": {"type": "RSI", "period": 7}},
        "entry_conditions": {"long": ["rsi < 35"], "short": ["rsi > 65"]},
        "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 45"]},
        "stop_loss": {"type": "fixed_pct", "percent": 1.5},
        "take_profit": {"type": "fixed_pct", "percent": 2.0},
    }

    def run(symbols: list[str]) -> tuple[float, ...]:
        m = NTScreeningRunner(dsl, symbols, "2024-01-01", "2024-02-15",
                              catalog_path=str(_REAL_CATALOG))({})
        return (m.sharpe_ratio, m.total_trades, m.total_return, m.profit_factor, m.max_drawdown)

    a = run(["BTCUSDT", "ETHUSDT"])
    assert a[1] > 50
    assert run(["ETHUSDT", "BTCUSDT"]) == a
