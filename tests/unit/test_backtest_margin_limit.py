"""Backtest margin limit in the generated strategy (bead vibe-quant-t8nrv, task T5).

With ``max_margin_pct`` set, every ENTRY is checked at its venue-call instant against
``sum(open + pending + new margin) <= max_margin_pct * equity`` (the one rule of
:mod:`vibe_quant.margin_limit`; a backtest uses the run leverage as ``margin_leverage``).
Off (``None``, the default) the generated strategy reads nothing and is byte-identical.

Rows R36-R40. R37 runs a REAL two-symbol NT engine, so it doubles as the probe that equity
and the other strategy's positions are visible at outbox release time.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path  # noqa: TC003
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from tests.unit import test_validation_sizing as vs
from tests.unit.test_validation_sizing import (  # noqa: F401  (autouse fixture)
    _offline,
)
from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.validation.results import ValidationResult  # noqa: TC001
from vibe_quant.validation.runner import ValidationRunner

_BTC = "BTCUSDT-PERP.BINANCE"
_ETH = "ETHUSDT-PERP.BINANCE"
_S = 1_000_000_000
_M = 60 * _S
_M5 = 5 * _M
_VOL = "100000.000"
_USDT = Currency.from_str("USDT")

# Signal on 5m bar 1 (SMA(2) ready, close > 50): the 5m bar closes at 2*_M5 = minute 10, the
# first own 1m bar after it closes at minute 11.
_T_SIGNAL = 2 * _M5
_T_RELEASE = 11 * _M


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _bar(iid: str, spec: str, step: int, i: int, c: float, *, prec: int) -> Bar:
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


def _engine(symbols: tuple[str, ...], *, balance: int, leverage: int) -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(balance, _USDT)],
        default_leverage=Decimal(leverage),
        bar_execution=True,
        use_reduce_only=True,
    )
    for sym in symbols:
        engine.add_instrument(create_instrument(sym))
    return engine


def _dsl() -> dict[str, Any]:
    # 2% stop: raw size = 2% risk / 2% stop = 100% of equity, so the 50% cap binds.
    return {
        "name": "margin_probe",
        "timeframe": "5m",
        "indicators": {"sma": {"type": "SMA", "period": 2}},
        "entry_conditions": {"long": ["close > 50"]},
        "exit_conditions": {"long": ["close < 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def _compiled() -> tuple[type, type]:
    d = _dsl()
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(d))
    cn = _to_class_name(d["name"])
    return getattr(module, f"{cn}Strategy"), getattr(module, f"{cn}Config")


def _recording(strategy_cls: type) -> type:
    """Subclass logging every margin check as (clock ns, qty, price, verdict)."""

    class Recording(strategy_cls):  # type: ignore[misc, valid-type]
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.margin_calls: list[tuple[int, float, float, bool]] = []

        def _entry_margin_ok(self, qty: Any, price: float, exclude_id: Any = None) -> bool:
            ok = super()._entry_margin_ok(qty, price, exclude_id)
            self.margin_calls.append((self.clock.timestamp_ns(), float(qty), price, ok))
            return ok

    return Recording


def _run(
    *,
    margin: float | None,
    cap: float = 0.5,
    leverage: int = 1,
    balance: int = 10_000,
    mode: str = "bar",
    entry_order: str = "market",
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
) -> tuple[BacktestEngine, dict[str, Any]]:
    strategy_cls, config_cls = _compiled()
    recording = _recording(strategy_cls)
    engine = _engine(symbols, balance=balance, leverage=leverage)
    px = {"BTCUSDT": (_BTC, 100.0, 1), "ETHUSDT": (_ETH, 200.0, 2)}
    data: list[Bar] = []
    for sym in symbols:
        iid, price, prec = px[sym]
        data += [_bar(iid, "5-MINUTE", _M5, i, price, prec=prec) for i in range(3)]
        if mode == "bar":
            data += [_bar(iid, "1-MINUTE", _M, i, price, prec=prec) for i in range(15)]
    engine.add_data(data)
    strats: dict[str, Any] = {}
    for sym in symbols:
        iid = px[sym][0]
        kw: dict[str, Any] = {
            "instrument_id": iid,
            "command_release": mode if mode == "bar" else "",
            "max_position_pct": cap,
            "entry_order": entry_order,
        }
        if mode == "bar":
            kw["command_release_bar_type"] = f"{iid}-1-MINUTE-LAST-EXTERNAL"
        if margin is not None:
            kw["max_margin_pct"] = margin
            kw["margin_leverage"] = float(leverage)
        strat = recording(config_cls(**kw))
        engine.add_strategy(strat)
        strats[iid] = strat
    engine.run()
    return engine, strats


def _filled_entries(engine: BacktestEngine) -> list[Any]:
    return [o for o in engine.cache.orders() if not o.is_reduce_only and o.filled_qty]


def _refusals(strats: dict[str, Any]) -> int:
    return sum(s._margin_refusals for s in strats.values())


@pytest.fixture
def disposed() -> Any:
    created: list[BacktestEngine] = []
    yield created
    for engine in created:
        engine.dispose()


# ---------------------------------------------------------------------------
# R37 / R37b: the real two-symbol engine (also the NT probe)
# ---------------------------------------------------------------------------


def test_second_symbol_refused_at_1x(disposed: list[BacktestEngine]) -> None:
    """R37: lev 1, cap 0.5, cushion 0.5 -> the first entry lands exactly ON the limit
    (equality allowed), the second symbol's entry is refused while the first is open."""
    engine, strats = _run(margin=0.5)
    disposed.append(engine)
    fills = _filled_entries(engine)
    assert len(fills) == 1, [str(o.instrument_id) for o in fills]
    assert _refusals(strats) >= 1
    refused = [s for s in strats.values() if s._margin_refusals]
    assert len(refused) == 1
    # the refused symbol never opened a position or left an order behind
    assert not [o for o in engine.cache.orders(instrument_id=InstrumentId.from_str(
        next(k for k, v in strats.items() if v is refused[0])))]
    assert [(ok) for (_t, _q, _p, ok) in refused[0].margin_calls] == [False]


def test_cushion_1_matches_off(disposed: list[BacktestEngine]) -> None:
    """R37b: a limit that never binds produces the trade list of the off run."""
    off_engine, off_strats = _run(margin=None, cap=0.4)
    on_engine, on_strats = _run(margin=1.0, cap=0.4)
    disposed.extend([off_engine, on_engine])

    def trades(engine: BacktestEngine) -> list[tuple[str, str, float, int]]:
        return sorted(
            (str(o.instrument_id), str(o.side), float(o.quantity), o.ts_last)
            for o in engine.cache.orders()
        )

    assert len(_filled_entries(off_engine)) == 2
    assert trades(on_engine) == trades(off_engine)
    assert _refusals(on_strats) == 0
    assert _refusals(off_strats) == 0


# ---------------------------------------------------------------------------
# R36: off reads nothing
# ---------------------------------------------------------------------------


def test_off_path_reads_nothing() -> None:
    """R36: off, ``_entry_margin_ok`` returns True without touching the cache, the
    portfolio, the instrument or the log (a bare config-only ``self``)."""
    strategy_cls, _ = _compiled()
    bare = SimpleNamespace(config=SimpleNamespace(max_margin_pct=None))
    assert strategy_cls._entry_margin_ok(bare, Quantity.from_str("1.000"), 100.0) is True
    # and an off engine run never even counts a refusal
    assert not hasattr(bare, "_margin_refusals")


def test_off_source_has_no_extra_state_when_unconfigured() -> None:
    """R36: the config fields default to the off values."""
    _, config_cls = _compiled()
    cfg = config_cls(instrument_id=_BTC)
    assert cfg.max_margin_pct is None
    assert cfg.margin_leverage == 10.0


# ---------------------------------------------------------------------------
# R37c: pending set is unioned by client_order_id
# ---------------------------------------------------------------------------


class _Order:
    def __init__(self, cid: str, instrument_id: InstrumentId, qty: str, px: str) -> None:
        self.client_order_id = cid
        self.instrument_id = instrument_id
        self.is_reduce_only = False
        self.leaves_qty = Quantity.from_str(qty)
        self.price = Price.from_str(px)


class _Pending:
    """A pending market entry: no ``price`` attribute, so the check must read a last close."""

    def __init__(self, cid: str, instrument_id: InstrumentId, qty: str) -> None:
        self.client_order_id = cid
        self.instrument_id = instrument_id
        self.is_reduce_only = False
        self.leaves_qty = Quantity.from_str(qty)


def _fake_self(
    strategy_cls: type,
    *,
    open_orders: list[Any],
    inflight: list[Any],
    last_price: Any = None,
) -> Any:
    usdt = _USDT
    inst_a = create_instrument("BTCUSDT")
    inst_b = create_instrument("ETHUSDT")
    instruments = {inst_a.id: inst_a, inst_b.id: inst_b}
    cache = SimpleNamespace(
        positions_open=lambda venue=None: [],
        orders_open=lambda venue=None: list(open_orders),
        orders_inflight=lambda venue=None: list(inflight),
        instrument=lambda iid: instruments[iid],
        price=lambda iid, pt: None if last_price is None else last_price(iid),
    )
    fake = SimpleNamespace(
        config=SimpleNamespace(max_margin_pct=0.5, margin_leverage=10.0),
        instrument=inst_b,
        instrument_id=inst_b.id,
        cache=cache,
        portfolio=SimpleNamespace(equity=lambda venue: {usdt: Money(1000, usdt)}),
        log=SimpleNamespace(warning=lambda msg: fake.warnings.append(msg)),
        warnings=[],
        _margin_refusals=0,
    )
    for name in ("_entry_margin_ok",):
        setattr(fake, name, MethodType(getattr(strategy_cls, name), fake))
    return fake


def test_template_pending_counted_once() -> None:
    """R37c: a PENDING_UPDATE post-only entry on A sits in BOTH NT order sets; at L10,
    equity 1000, cushion 0.5 a new 2000 entry on B adds 200 + 200 = 400 <= 500 (allowed),
    not the 600 a concatenation of the two sets would read (refused)."""
    strategy_cls, _ = _compiled()
    btc = create_instrument("BTCUSDT").id
    pending = _Order("O-1", btc, "20.000", "100.0")
    fake = _fake_self(strategy_cls, open_orders=[pending], inflight=[pending])
    assert fake._entry_margin_ok(Quantity.from_str("10.00"), 200.0) is True, fake.warnings
    assert fake._margin_refusals == 0


def test_template_pending_distinct_orders_both_counted() -> None:
    """R37c twin: two DIFFERENT pending orders (open + inflight) are both margin."""
    strategy_cls, _ = _compiled()
    btc = create_instrument("BTCUSDT").id
    one = _Order("O-1", btc, "20.000", "100.0")
    two = _Order("O-2", btc, "20.000", "100.0")
    fake = _fake_self(strategy_cls, open_orders=[one], inflight=[two])
    # 200 + 200 + 200 = 600 > 500
    assert fake._entry_margin_ok(Quantity.from_str("10.00"), 200.0) is False
    assert fake._margin_refusals == 1
    assert "Margin: entry refused" in fake.warnings[0]


def test_template_pending_equality_edge() -> None:
    """The limit is inclusive: pending 200 + new 300 == 500 passes, one more tick refuses."""
    strategy_cls, _ = _compiled()
    btc = create_instrument("BTCUSDT").id
    one = _Order("O-1", btc, "20.000", "100.0")
    fake = _fake_self(strategy_cls, open_orders=[one], inflight=[])
    assert fake._entry_margin_ok(Quantity.from_str("15.00"), 200.0) is True
    assert fake._entry_margin_ok(Quantity.from_str("15.01"), 200.0) is False


def test_template_pending_unpriced_is_refused() -> None:
    """R37c: a pending market entry (no ``price``) on another instrument whose last close is
    unknown has no reference price at all: the entry is refused, not silently skipped."""
    strategy_cls, _ = _compiled()
    btc = create_instrument("BTCUSDT").id
    fake = _fake_self(strategy_cls, open_orders=[_Pending("O-1", btc, "20.000")], inflight=[])
    assert fake._entry_margin_ok(Quantity.from_str("15.00"), 200.0) is False
    assert fake._margin_refusals == 1
    assert "no reference price for pending" in fake.warnings[0]


def test_template_pending_unpriced_counted_at_last_close() -> None:
    """R37c twin: the same pending market entry HAS a last close, so it is counted at that
    price: pending 20 @ 100 -> 200, so new 15 @ 200 (300) == the 500 limit passes and one
    more tick refuses (a skipped pending would pass both)."""
    strategy_cls, _ = _compiled()
    btc = create_instrument("BTCUSDT").id
    fake = _fake_self(
        strategy_cls,
        open_orders=[_Pending("O-1", btc, "20.000")],
        inflight=[],
        last_price=lambda iid: Price.from_str("100.0"),
    )
    assert fake._entry_margin_ok(Quantity.from_str("15.00"), 200.0) is True
    assert fake._entry_margin_ok(Quantity.from_str("15.01"), 200.0) is False
    assert fake._margin_refusals == 1


# ---------------------------------------------------------------------------
# R38: the check instant
# ---------------------------------------------------------------------------


def test_check_at_release_not_signal(disposed: list[BacktestEngine]) -> None:
    """R38: queued market entries are checked at the outbox RELEASE (own 1m bar), never at
    the signal; a refused queued entry leaves the outbox and sends nothing."""
    engine, strats = _run(margin=0.01)  # limit 100 < any entry: all refused
    disposed.append(engine)
    for strat in strats.values():
        # one check, at the release; the bar-3 signal (minute 15) has no later datum
        assert [t for (t, _q, _p, _ok) in strat.margin_calls] == [_T_RELEASE]
        assert [ok for (_t, _q, _p, ok) in strat.margin_calls] == [False]
        assert strat._margin_refusals == 1
        # the refused command left the outbox: only the still-unreleased bar-3 intent waits
        assert [ts for (ts, _fn, _args) in strat._outbox] == [3 * _M5]
    assert engine.cache.orders() == []


def test_check_at_immediate_send(disposed: list[BacktestEngine]) -> None:
    """R38: with no outbox the check runs at the signal bar itself (the send instant)."""
    engine, strats = _run(margin=0.01, mode="immediate")
    disposed.append(engine)
    for strat in strats.values():
        # checked at each signal bar's own close (bar-1 signal, then bar-2's), sent or not
        assert [t for (t, _q, _p, _ok) in strat.margin_calls] == [_T_SIGNAL, 3 * _M5]
        assert strat._margin_refusals == 2
    assert engine.cache.orders() == []


def test_check_at_post_only_release(disposed: list[BacktestEngine]) -> None:
    """R38: a post-only intent is checked when it is priced, at the release datum, with the
    LIMIT price (one tick inside the release close)."""
    engine, strats = _run(margin=0.01, entry_order="post_only", symbols=("BTCUSDT",))
    disposed.append(engine)
    strat = strats[_BTC]
    assert [(t, p) for (t, _q, p, _ok) in strat.margin_calls] == [(_T_RELEASE, pytest.approx(99.9))]
    assert [ts for (ts, _fn, _args) in strat._outbox] == [3 * _M5]
    assert engine.cache.orders() == []


# ---------------------------------------------------------------------------
# R39: the overlay is post-hoc
# ---------------------------------------------------------------------------


def test_overlay_post_hoc_documented() -> None:
    """R39: the in-engine check reads NT's own equity and run leverage; the generated
    strategy knows nothing of the liquidation overlay (which only truncates ledgers after
    the run), and the screening config carries the RUN leverage whatever the overlay says."""
    src = StrategyCompiler().compile(validate_strategy_dict(_dsl()))
    assert "portfolio.equity(" in src
    assert "margin_leverage" in src
    assert "liquidation" not in src.lower()

    from vibe_quant.sizing_settings import SizingSettings

    for leverage in (5.0, 20.0):
        params = SizingSettings(max_margin_pct=0.5).strategy_params(leverage)
        assert params == {"max_margin_pct": 0.5, "margin_leverage": leverage}


# ---------------------------------------------------------------------------
# R40: validation notes
# ---------------------------------------------------------------------------


class _EngineNode(vs._CapturingNode):
    """The capturing node with an engine exposing the strategies' refusal counters."""

    def get_engine(self, _id: object) -> SimpleNamespace:
        base = super().get_engine(_id)
        strategies = [SimpleNamespace(_margin_refusals=2), SimpleNamespace(_margin_refusals=1)]
        return SimpleNamespace(
            get_result=base.get_result, trader=SimpleNamespace(strategies=lambda: strategies)
        )


def _validation_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parameters: dict[str, object]
) -> tuple[ValidationResult, dict[str, Any]]:
    import nautilus_trader.backtest.node as nt_node

    monkeypatch.setattr(nt_node, "BacktestNode", _EngineNode)
    run_id = vs._launch(tmp_path, parameters)
    stored: list[ValidationResult] = []
    monkeypatch.setattr(
        ValidationRunner, "_store_results", lambda self, rid, result: stored.append(result)
    )
    runner = vs._runner(tmp_path)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()
    (cfg,) = vs._strategy_configs()
    return stored[0], dict(cfg.config)


def test_validation_notes_margin_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R40: ``notes.margin_limit`` = the cushion + the summed refused entries, only when on;
    the strategy config carries the run leverage as ``margin_leverage``."""
    result, cfg = _validation_result(
        tmp_path, monkeypatch, {"max_margin_pct": 0.5, "leverage": 5}
    )
    assert result.notes["margin_limit"] == {"max_margin_pct": 0.5, "refused_entries": 3}
    assert json.loads(json.dumps(result.notes))["margin_limit"]["refused_entries"] == 3
    assert cfg["max_margin_pct"] == 0.5
    assert cfg["margin_leverage"] == 5.0


def test_validation_off_has_no_margin_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R40/R36: key absent -> no notes entry and no margin keys in the strategy config."""
    result, cfg = _validation_result(tmp_path, monkeypatch, {"leverage": 5})
    assert "margin_limit" not in result.notes
    assert "max_margin_pct" not in cfg
    assert "margin_leverage" not in cfg


def test_aggregate_walk_forward_merges_margin_limit_notes() -> None:
    """R40: the walk-forward aggregate sums ``refused_entries`` across windows (2 + 1 = 3) and
    keeps the shared ``max_margin_pct``; no window on -> no note."""

    def window(refused: int) -> ValidationResult:
        result = ValidationResult()
        result.notes["margin_limit"] = {"max_margin_pct": 0.5, "refused_entries": refused}
        return result

    agg = ValidationRunner._aggregate_walk_forward_results(
        run_id=1, strategy_name="s", window_results=[window(2), window(1)]
    )
    assert agg.notes["margin_limit"] == {"max_margin_pct": 0.5, "refused_entries": 3}

    off = ValidationRunner._aggregate_walk_forward_results(
        run_id=1, strategy_name="s", window_results=[ValidationResult(), ValidationResult()]
    )
    assert "margin_limit" not in off.notes
