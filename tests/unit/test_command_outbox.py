"""vibe-quant-yul7u.9: per-instrument command outbox in the generated strategy.

With ``command_release`` set, every submit/cancel the generated strategy issues
is queued and only sent when the strategy receives its OWN next datum with
``ts_init`` > queue ts: its own trade tick ("trade_tick") or its own detail bar
("bar"). This replaces NT's venue-wide latency release, which let symbol A's
data event release symbol B's order at B's stale book (= the signal close).
``command_release=""`` (live, paper, 1m strategies) sends immediately.

Direct BacktestEngine runs on synthetic 5m/4h bars + 1m bars/trade ticks.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType, TradeTick
from nautilus_trader.model.enums import AccountType, AggressorSide, OmsType, OrderType
from nautilus_trader.model.identifiers import TradeId, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from tests.unit.test_audit_paper_node import fake_tradingnode  # noqa: F401  (fixture)
from vibe_quant.data.catalog import create_instrument
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl import templates
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard

if TYPE_CHECKING:
    from pathlib import Path

    from tests.unit.test_audit_paper_node import _FakeTradingNode

_BTC = "BTCUSDT-PERP.BINANCE"
_ETH = "ETHUSDT-PERP.BINANCE"
_S = 1_000_000_000
_M = 60 * _S
_M5 = 5 * _M
_VOL = "100000.000"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _bar(iid: str, spec: str, step: int, i: int, c: float, *, prec: int = 1) -> Bar:
    """Flat-ish bar ``i`` of length ``step``; ts_init = bar close (like the catalog)."""
    fmt = f"{{:.{prec}f}}"
    return Bar(
        bar_type=BarType.from_str(f"{iid}-{spec}-LAST-EXTERNAL"),
        open=Price.from_str(fmt.format(c)),
        high=Price.from_str(fmt.format(c)),
        low=Price.from_str(fmt.format(c)),
        close=Price.from_str(fmt.format(c)),
        volume=Quantity.from_str(_VOL),
        ts_event=i * step,
        ts_init=(i + 1) * step,
    )


def _tick(iid: str, ts: int, px: float, *, prec: int = 1) -> TradeTick:
    # Huge size: tick size is L1 liquidity (board gotcha 18dd00ed46ebd050).
    return TradeTick(
        instrument_id=create_instrument(iid.split("-")[0]).id,
        price=Price.from_str(f"{px:.{prec}f}"),
        size=Quantity.from_str("1000000.000"),
        aggressor_side=AggressorSide.BUYER,
        trade_id=TradeId(f"T-{iid[:3]}-{ts}"),
        ts_event=ts,
        ts_init=ts,
    )


def _engine(symbols: tuple[str, ...] = ("BTCUSDT",)) -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
        use_reduce_only=True,
    )
    for sym in symbols:
        engine.add_instrument(create_instrument(sym))
    return engine


def _dsl(name: str, **overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "5m",
        "indicators": {"sma": {"type": "SMA", "period": 2}},
        "entry_conditions": {"long": ["close > 50"]},
        "exit_conditions": {"long": ["close < 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 20.0},
        "take_profit": {"type": "fixed_pct", "percent": 40.0},
    }
    d.update(overrides)
    return d


def _compiled(d: dict[str, Any]) -> tuple[type, type]:
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(d))
    cn = _to_class_name(d["name"])
    return getattr(module, f"{cn}Strategy"), getattr(module, f"{cn}Config")


def _probe_cls(strategy_cls: type) -> type:
    class Probe(strategy_cls):  # type: ignore[misc, valid-type]
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.tick_calls = 0
            self.pending_after_bar: list[bool] = []

        def on_trade_tick(self, tick: TradeTick) -> None:
            self.tick_calls += 1
            super().on_trade_tick(tick)

        def on_bar(self, bar: Bar) -> None:
            super().on_bar(bar)
            if bar.bar_type == self.primary_bar_type:
                self.pending_after_bar.append(self._has_pending_entry())

    return Probe


def _entry_fills(engine: BacktestEngine, iid: str) -> list[tuple[float, int]]:
    """(avg px, ts) of every filled non-reduce-only order on ``iid``."""
    return [
        (float(o.avg_px), o.ts_last)
        for o in engine.cache.orders()
        if str(o.instrument_id) == iid and not o.is_reduce_only and o.is_closed and o.filled_qty
    ]


# Signal on 5m bar 1 (SMA(2) ready, close > 50) -> queued at its ts_init = 2*_M5.
_T_SIGNAL = 2 * _M5


def _run(
    *,
    mode: str,
    data: list[Any],
    symbols: tuple[str, ...] = ("BTCUSDT",),
    name: str = "outbox",
    release_spec: str = "1-MINUTE",
    dsl: dict[str, Any] | None = None,
) -> tuple[BacktestEngine, dict[str, Any]]:
    strategy_cls, config_cls = _compiled(dsl or _dsl(name))
    probe = _probe_cls(strategy_cls)
    engine = _engine(symbols)
    engine.add_data(data)
    strats: dict[str, Any] = {}
    for sym in symbols:
        iid = f"{sym}-PERP.BINANCE"
        kw: dict[str, Any] = {"instrument_id": iid, "command_release": mode}
        if mode == "bar":
            kw["command_release_bar_type"] = f"{iid}-{release_spec}-LAST-EXTERNAL"
        strat = probe(config_cls(**kw))
        engine.add_strategy(strat)
        strats[iid] = strat
    engine.run()
    return engine, strats


def _strategy_bars(iid: str, closes: list[float], *, prec: int = 1) -> list[Bar]:
    return [_bar(iid, "5-MINUTE", _M5, i, c, prec=prec) for i, c in enumerate(closes)]


# ---------------------------------------------------------------------------
# 1. Another instrument's datum never releases this strategy's queue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("btc_first", [True, False], ids=["btc_tick_first", "eth_tick_first"])
def test_other_instrument_tick_does_not_release_queue(btc_first: bool) -> None:
    btc_tick_ts = _T_SIGNAL + (1 if btc_first else 2) * _S
    eth_tick_ts = _T_SIGNAL + (2 if btc_first else 1) * _S
    data: list[Any] = (
        _strategy_bars(_BTC, [100.0, 100.0, 100.0])
        + _strategy_bars(_ETH, [200.0, 200.0, 200.0], prec=2)
        + [_tick(_BTC, btc_tick_ts, 105.0), _tick(_ETH, eth_tick_ts, 205.0, prec=2)]
    )
    engine, _ = _run(mode="trade_tick", data=data, symbols=("BTCUSDT", "ETHUSDT"))
    # Each entry fills at ITS OWN tick, never at the stale signal close (100/200)
    # that a release by the other symbol's earlier tick would give.
    assert _entry_fills(engine, _BTC) == [(105.0, btc_tick_ts)]
    assert _entry_fills(engine, _ETH) == [(205.0, eth_tick_ts)]
    engine.dispose()


# ---------------------------------------------------------------------------
# 2. Own next datum with a LATER ts releases; the same ts does not
# ---------------------------------------------------------------------------


def test_own_later_tick_releases_same_ts_tick_does_not() -> None:
    data: list[Any] = _strategy_bars(_BTC, [100.0, 100.0, 100.0]) + [
        _tick(_BTC, _T_SIGNAL, 103.0),  # same ts as the signal bar: must not release
        _tick(_BTC, _T_SIGNAL + _S, 105.0),
    ]
    engine, strats = _run(mode="trade_tick", data=data)
    assert strats[_BTC].tick_calls == 2  # the same-ts tick did reach on_trade_tick
    assert _entry_fills(engine, _BTC) == [(105.0, _T_SIGNAL + _S)]
    engine.dispose()


def test_own_later_detail_bar_releases_same_ts_bar_does_not() -> None:
    one_m = [
        _bar(_BTC, "1-MINUTE", _M, 9, 103.0),  # ts_init == _T_SIGNAL: must not release
        _bar(_BTC, "1-MINUTE", _M, 10, 105.0),  # first 1m bar after the signal
        _bar(_BTC, "1-MINUTE", _M, 11, 106.0),
    ]
    assert one_m[0].ts_init == _T_SIGNAL
    # strategy bars first: equal-ts data keeps insertion order, so the same-ts 1m
    # bar reaches the strategy AFTER the signal bar
    data: list[Any] = _strategy_bars(_BTC, [100.0, 100.0, 100.0]) + one_m
    engine, _ = _run(mode="bar", data=data)
    assert _entry_fills(engine, _BTC) == [(105.0, one_m[1].ts_init)]
    engine.dispose()


# ---------------------------------------------------------------------------
# 3. End-of-run (on_stop) commands are never sent in outbox mode
# ---------------------------------------------------------------------------


def test_on_stop_commands_never_sent_in_outbox_mode() -> None:
    data: list[Any] = _strategy_bars(_BTC, [100.0, 100.0, 100.0]) + [
        _tick(_BTC, _T_SIGNAL + _S, 105.0),  # releases the entry
        _tick(_BTC, _T_SIGNAL + 2 * _S, 105.0),  # releases SL/TP
    ]
    engine, _ = _run(mode="trade_tick", data=data)
    positions = engine.cache.positions()
    assert len(positions) == 1 and positions[0].is_open  # no end-of-run close sent
    protective = [o for o in engine.cache.orders() if o.is_reduce_only]
    assert {o.order_type for o in protective} == {OrderType.STOP_MARKET, OrderType.LIMIT}
    assert all(o.is_open for o in protective)  # no end-of-run cancel sent
    engine.dispose()


def test_queued_entry_without_later_datum_is_never_sent() -> None:
    engine, strats = _run(mode="trade_tick", data=_strategy_bars(_BTC, [100.0, 100.0]))
    assert engine.cache.orders() == []
    assert len(strats[_BTC]._outbox) == 1  # still queued at the end of the run
    engine.dispose()


# ---------------------------------------------------------------------------
# 4. command_release="" sends immediately and subscribes to nothing extra
# ---------------------------------------------------------------------------


def test_mode_off_sends_immediately_and_subscribes_no_ticks() -> None:
    data: list[Any] = _strategy_bars(_BTC, [100.0, 100.0, 100.0]) + [_tick(_BTC, _T_SIGNAL + _S, 105.0)]
    engine, strats = _run(mode="", data=data)
    entries = [o for o in engine.cache.orders() if not o.is_reduce_only]
    assert [o.ts_init for o in entries] == [_T_SIGNAL]  # submitted on the signal bar
    assert _entry_fills(engine, _BTC) == [(100.0, _T_SIGNAL)]  # at the signal close
    assert strats[_BTC].tick_calls == 0
    assert engine.kernel.data_engine.subscribed_trade_ticks() == []
    assert [str(b) for b in engine.kernel.data_engine.subscribed_bars()] == [
        f"{_BTC}-5-MINUTE-LAST-EXTERNAL"
    ]
    # today's on_stop: cancel SL/TP and close the position
    assert all(p.is_closed for p in engine.cache.positions())
    assert not engine.cache.orders_open()
    assert strats[_BTC]._outbox == []
    engine.dispose()


# ---------------------------------------------------------------------------
# 5. _has_pending_entry sees queued entries
# ---------------------------------------------------------------------------


def test_has_pending_entry_sees_queued_entries() -> None:
    # signals on bars 1 AND 2; the first own tick only arrives after bar 2
    data: list[Any] = _strategy_bars(_BTC, [100.0, 100.0, 100.0, 100.0]) + [
        _tick(_BTC, 3 * _M5 + _S, 105.0)
    ]
    engine, strats = _run(mode="trade_tick", data=data)
    assert strats[_BTC].pending_after_bar[1:3] == [True, True]
    entries = [o for o in engine.cache.orders() if not o.is_reduce_only]
    assert len(entries) == 1, [str(o) for o in entries]  # no 2nd entry from bar 2
    engine.dispose()


# ---------------------------------------------------------------------------
# 6. Paper strips both backtest-only fields
# ---------------------------------------------------------------------------

_PAPER_DSL = {
    "name": "outbox_paper",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "exit_conditions": {"long": []},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


async def test_paper_strips_command_release_fields(
    tmp_path: Path,
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    from vibe_quant.paper.config import BinanceTestnetConfig, PaperTradingConfig
    from vibe_quant.paper.node import PaperTradingNode

    sm = StateManager(tmp_path / "state.db")
    sid = sm.create_strategy("outbox_paper", _PAPER_DSL)
    rid = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={
            "rsi_14_period": 21,
            "command_release": "bar",
            "command_release_bar_type": f"{_BTC}-1-MINUTE-LAST-EXTERNAL",
        },
    )
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    node = PaperTradingNode(
        PaperTradingConfig(
            trader_id="PAPER-008",
            binance=BinanceTestnetConfig("key", "secret", testnet=True),
            strategy_id=sid,
            validation_run_id=rid,
            db_path=tmp_path / "state.db",
            logs_path=tmp_path / "logs",
        )
    )
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.strategy_params == {"rsi_14_period": 21}
        strat: Any = node._compiled_strategies[0]
        assert strat.config.rsi_14_period == 21
        assert strat.config.command_release == ""
        assert strat.config.command_release_bar_type == ""
    finally:
        await node._shutdown()


# ---------------------------------------------------------------------------
# 7. No generated venue call bypasses the outbox helper
# ---------------------------------------------------------------------------

# Every NT Strategy method that sends a venue command.
_VENUE_METHODS = "|".join(
    (
        "submit_order",
        "submit_order_list",
        "modify_order",
        "cancel_order",
        "cancel_orders",
        "cancel_all_orders",
        "cancel_gtd_expiry",
        "query_order",
        "close_position",
        "close_all_positions",
    )
)
_DIRECT_CALL = re.compile(rf"self\.({_VENUE_METHODS})\(")
_ANY_REF = re.compile(rf"self\.({_VENUE_METHODS})\b")


def _method_of(lines: list[str], idx: int) -> str:
    for j in range(idx, -1, -1):
        m = re.match(r"\s*def (\w+)\(", lines[j])
        if m:
            return m.group(1)
    return ""


@pytest.mark.parametrize(
    "source",
    ["generated", "templates"],
)
def test_venue_calls_only_through_outbox_helper(source: str) -> None:
    if source == "generated":
        src = StrategyCompiler().compile(
            validate_strategy_dict(
                _dsl(
                    "outbox_rg",
                    stop_loss={"type": "atr_trailing", "atr_multiplier": 2.0, "indicator": "atr"},
                    indicators={"sma": {"type": "SMA", "period": 2}, "atr": {"type": "ATR", "period": 3}},
                )
            )
        )
    else:
        src = "\n".join(
            "\n".join(v) for k, v in vars(templates).items() if k.endswith("_LINES") and isinstance(v, tuple)
        )
    lines = src.splitlines()
    direct = [(i, lines[i].strip()) for i, ln in enumerate(lines) if _DIRECT_CALL.search(ln)]
    # The only direct venue calls: the command_release="" path of on_stop.
    assert {_method_of(lines, i) for i, _ in direct} <= {"on_stop"}, direct
    assert sorted(s for _, s in direct) == [
        "self.cancel_all_orders(self.instrument_id)",
        "self.close_all_positions(self.instrument_id)",
    ]
    # Every other reference hands the bound method to the outbox helper (or is
    # the queued-entry check in _has_pending_entry).
    for i, ln in enumerate(lines):
        if not _ANY_REF.search(ln) or _DIRECT_CALL.search(ln):
            continue
        method = _method_of(lines, i)
        assert "self._send_command(self." in ln or method == "_has_pending_entry", (method, ln)
    sends = sum(ln.count("self._send_command(") for ln in lines)
    assert sends >= 11, sends  # 7 submit + 2 cancel + 2 cancel_all call sites


# ---------------------------------------------------------------------------
# 8. 1m pandas-path indicator on a 4h strategy still gets every 1m bar
# ---------------------------------------------------------------------------


def test_bar_mode_feeds_every_1m_bar_to_pandas_indicator() -> None:
    h4 = 4 * 60 * _M
    dsl = _dsl(
        "outbox_pta",
        timeframe="4h",
        additional_timeframes=["1m"],
        indicators={
            "sma": {"type": "SMA", "period": 2},
            "adx_1m": {"type": "ADX", "period": 3, "timeframe": "1m"},
        },
    )
    n_1m = 2 * 240 + 5
    data: list[Any] = [_bar(_BTC, "4-HOUR", h4, i, 100.0) for i in range(2)] + [
        _bar(_BTC, "1-MINUTE", _M, i, 100.0 + (i % 7) * 0.1) for i in range(n_1m)
    ]
    engine, strats = _run(mode="bar", data=data, dsl=dsl)
    strat = strats[_BTC]
    assert strat._command_release_bar_type == strat.bar_type_1m
    assert len(strat._pta_bufs["1m"]["close"]) == n_1m
    # and the 1m bars also released the 4h signal's entry
    assert len(_entry_fills(engine, _BTC)) == 1
    engine.dispose()


# ---------------------------------------------------------------------------
# on_start guards: misconfigured outbox fails at start, never silently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"command_release": "bar", "command_release_bar_type": f"{_ETH}-1-MINUTE-LAST-EXTERNAL"}, "is not for"),
        ({"command_release": "bar", "command_release_bar_type": ""}, "needs command_release_bar_type"),
        ({"command_release": "trade_ticks"}, "Unknown command_release"),
        ({"command_release": "trade_tick", "execution_delay_probability": 1.0}, "execution_delay_probability"),
        ({"command_release": "bar", "command_release_bar_type": f"{_BTC}-1-MINUTE-LAST-EXTERNAL",
          "execution_delay_probability": 0.3}, "execution_delay_probability"),
    ],
    ids=["wrong_instrument_bar_type", "empty_bar_type", "unknown_mode", "delay_with_tick", "delay_with_bar"],
)
def test_misconfigured_outbox_raises_at_start(overrides: dict[str, Any], match: str) -> None:
    strategy_cls, config_cls = _compiled(_dsl("outbox_guard"))
    engine = _engine(("BTCUSDT", "ETHUSDT"))
    engine.add_data(_strategy_bars(_BTC, [100.0, 100.0, 100.0]))
    strat = strategy_cls(config_cls(instrument_id=_BTC, **overrides))
    engine.add_strategy(strat)
    with pytest.raises(ValueError, match=match):
        strat.on_start()
    engine.dispose()


def test_valid_outbox_config_starts() -> None:
    strategy_cls, config_cls = _compiled(_dsl("outbox_guard_ok"))
    engine = _engine()
    strat = strategy_cls(config_cls(instrument_id=_BTC, command_release="bar",
                                    command_release_bar_type=f"{_BTC}-1-MINUTE-LAST-EXTERNAL"))
    engine.add_strategy(strat)
    strat.on_start()
    assert str(strat._command_release_bar_type) == f"{_BTC}-1-MINUTE-LAST-EXTERNAL"
    engine.dispose()


# ---------------------------------------------------------------------------
# on_reset clears the queue
# ---------------------------------------------------------------------------


def test_on_reset_clears_queue() -> None:
    engine, strats = _run(mode="trade_tick", data=_strategy_bars(_BTC, [100.0, 100.0]))
    strat = strats[_BTC]
    assert len(strat._outbox) == 1
    strat.on_reset()
    assert strat._outbox == []
    engine.dispose()
