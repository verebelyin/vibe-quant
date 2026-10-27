"""Unowned (external) positions are reported once after a pre-start halt (oug0a).

A guard halted before the strategies exist flattens an empty book. NT reconciles
the account's open positions - including ones no guarded strategy owns - before
``on_start``, so the pre-start halt's flatten never saw them and the warning
listing them never fired. ``on_start`` re-reads the book and reports the unowned
positions once, deduped against the list the pre-start flatten already reported
and skipped when an escalation inside ``on_start`` flattens (and warns) itself.

Contract rows R14-R20 (bead vibe-quant-oug0a): R14 red on base. The real NT
kernel harness is ``tests/unit/test_audit_paper_guard.py``; the external
position stands in for NT's reconciliation before the strategies start.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from nautilus_trader.model.enums import OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, PositionId, StrategyId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.test_kit.stubs.execution import TestExecStubs

import tests.unit.test_audit_paper_guard as harness
from tests.unit.test_audit_paper_node import DSL, _config
from tests.unit.test_paper_halt_restore import _count_flatten
from vibe_quant.alerts.telegram import TelegramBot, TelegramConfig
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.paper.guard import (
    RESTORED_HALT_PREFIX,
    UNOWNED_PREFIX,
    GuardState,
    HaltReason,
    RestoredHalt,
    RiskState,
)
from vibe_quant.paper.journal import FakeJournal
from vibe_quant.paper.node import PaperTradingNode, _GuardBridge

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_SYMBOL = "BTCUSDT"
_INSTRUMENT = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
_BARS = 5
_SIGNAL_MSG = "halt via SIGUSR1"
_ISO = "2026-01-05T21:00:00+00:00"
_EXT_POSITION = "P-EXT-1 (strategy EXTERNAL, BTCUSDT-PERP.BINANCE)"
_EXT_WARNING = (
    "positions not owned by any guarded strategy were NOT closed: "
    "P-EXT-1 (strategy EXTERNAL, BTCUSDT-PERP.BINANCE)"
)
_EXT_2_WARNING = (
    "positions not owned by any guarded strategy were NOT closed: "
    "P-EXT-1 (strategy EXTERNAL, BTCUSDT-PERP.BINANCE), "
    "P-EXT-2 (strategy EXTERNAL, BTCUSDT-PERP.BINANCE)"
)


@pytest.fixture(autouse=True)
def _dispose_engines() -> Iterator[None]:
    """Dispose every engine this module built, even when a test fails."""
    yield
    while harness._ENGINES:
        engine = harness._ENGINES.pop()
        retain_log_guard(engine)
        engine.dispose()


def _add_external(engine: Any, position_id: str = "P-EXT-1") -> None:
    """Add an open position owned by no guarded strategy (NT reconciliation)."""
    instrument = engine.cache.instrument(_INSTRUMENT)
    order = TestExecStubs.market_order(
        instrument=instrument,
        order_side=OrderSide.BUY,
        quantity=Quantity.from_str("0.010"),
    )
    fill = TestEventStubs.order_filled(
        order,
        instrument=instrument,
        strategy_id=StrategyId("EXTERNAL"),
        position_id=PositionId(position_id),
        last_px=Price.from_str("100.0"),
    )
    engine.cache.add_position(Position(instrument=instrument, fill=fill), OmsType.NETTING)


def _bars() -> list[Any]:
    return harness._flat_bars(get_bar_type(_SYMBOL, "1m"), [100.0] * _BARS)


def _restored_error() -> RestoredHalt:
    return RestoredHalt(
        reason=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}venue down",
        checkpoint_time=_ISO,
    )


# --------------------------------------------------------------- R14


def test_prestart_halt_reports_reconciled_unowned_position() -> None:
    """R14: halted before the run, the position reconciled after the halt is
    reported once by ``on_start`` (base: no warning at all) and stays open."""
    h = harness.build(symbols=(_SYMBOL,))
    assert h.guard.halt(HaltReason.SIGNAL, _SIGNAL_MSG) is True
    _add_external(h.engine)
    assert h.listener.warnings == [], "nothing open when the pre-start halt flattened"

    harness.run(h, _bars())

    assert h.listener.warnings == [_EXT_WARNING]
    assert [str(p.id) for p in h.engine.cache.positions_open()] == ["P-EXT-1"]
    assert h.guard.guard_state == GuardState.HALTED
    assert h.guard.halt_reason == HaltReason.SIGNAL


# --------------------------------------------------------------- R15


def test_unowned_reported_once_when_seen_before_start() -> None:
    """R15: a position already open at the pre-start halt is warned once by that
    halt's flatten; ``on_start`` must not repeat it."""
    h = harness.build(symbols=(_SYMBOL,))
    _add_external(h.engine)
    assert h.guard.halt(HaltReason.SIGNAL, _SIGNAL_MSG) is True
    assert h.listener.warnings == [_EXT_WARNING], "the pre-start flatten's own warning"

    harness.run(h, _bars())

    assert h.listener.warnings == [_EXT_WARNING], "deduped against the pre-start list"


def test_changed_unowned_list_is_reported_with_every_position() -> None:
    """R14 hostile probe: a listed position before the halt plus a different one
    after it - one warning, listing both (the list differs from the pre-start one)."""
    h = harness.build(symbols=(_SYMBOL,))
    _add_external(h.engine, "P-EXT-1")
    assert h.guard.halt(HaltReason.SIGNAL, _SIGNAL_MSG) is True
    _add_external(h.engine, "P-EXT-2")

    harness.run(h, _bars())

    assert h.listener.warnings == [_EXT_WARNING, _EXT_2_WARNING]


# --------------------------------------------------------------- R16


def test_replay_over_restored_error_reports_unowned() -> None:
    """R16: a SIGNAL halt replayed over a restored ERROR halt flattens once
    (R8a) and still reports the position reconciled after it, once."""
    h = harness.build(symbols=(_SYMBOL,), restored_halt=_restored_error())
    calls = _count_flatten(h.guard)
    assert h.guard.halt(HaltReason.SIGNAL, _SIGNAL_MSG) is True
    _add_external(h.engine)

    harness.run(h, _bars())

    assert len(calls) == 1, "the replay flattened; on_start must not flatten again"
    assert h.guard.halt_reason == HaltReason.SIGNAL
    assert h.listener.warnings == [_EXT_WARNING]


# --------------------------------------------------------------- R17


def test_restored_halt_first_tick_warns_once() -> None:
    """With no pre-start halt the restored halt's deferred first-tick flatten is the
    only reporter, and it reports the unowned positions once: the flatten's own
    warning, never a second ``restored halt flatten:`` echo of it (vibe-quant-v4fi4)."""
    h = harness.build(symbols=(_SYMBOL,), restored_halt=_restored_error())
    _add_external(h.engine)

    harness.run(h, _bars())

    assert h.guard.halt_reason == HaltReason.ERROR
    assert h.listener.warnings == [_EXT_WARNING]


# --------------------------------------------------------------- R18


def test_escalation_in_on_start_warns_once() -> None:
    """R18: a drawdown breach inside ``on_start`` escalates the pre-start SIGNAL
    halt; its flatten is the only warning. Dropping the ``_started`` reset is
    not observable here: the escalation's flatten records the same unowned list
    ``on_start`` re-reads, so the list dedupe still suppresses a second report."""
    risk = RiskState(
        high_water_mark=Decimal("10600"),
        day=date(2026, 1, 5),
        day_start_equity=Decimal("10000"),
    )
    h = harness.build(
        symbols=(_SYMBOL,),
        limits={"max_drawdown_pct": Decimal("0.05"), "max_daily_loss_pct": Decimal("0.5")},
        risk_state=risk,
    )
    assert h.guard.halt(HaltReason.SIGNAL, _SIGNAL_MSG) is True
    _add_external(h.engine)

    harness.run(h, _bars())

    assert h.guard.guard_state == GuardState.HALTED
    assert h.guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert h.listener.warnings == [_EXT_WARNING], "the escalation flatten's own warning only"


# --------------------------------------------------------------- R19


def test_resume_before_start_reports_nothing() -> None:
    """R19: resuming the pre-start halt before the run leaves nothing to report."""
    h = harness.build(symbols=(_SYMBOL,))
    assert h.guard.halt(HaltReason.SIGNAL, _SIGNAL_MSG) is True
    _add_external(h.engine)
    assert h.guard.resume() == (True, "resumed from halt")
    assert h.guard.guard_state == GuardState.RUNNING

    harness.run(h, _bars())

    assert h.guard.guard_state == GuardState.RUNNING
    assert h.listener.warnings == []


# --------------------------------------------------------------- R20


@pytest.mark.parametrize(
    ("exc_type", "expected_log"),
    [
        (RuntimeError, "unowned-position report failed (RuntimeError)"),
        (OSError, "unowned-position report failed (OSError)"),
    ],
)
def test_report_failure_is_class_only(
    caplog: pytest.LogCaptureFixture,
    exc_type: type[Exception],
    expected_log: str,
) -> None:
    """R20: a raising ``on_warning`` in the report does not break ``on_start``;
    the ERROR log names the class only and carries no exception text/``exc_info``.
    Two unrelated classes pin the forwarded class name."""
    h = harness.build(symbols=(_SYMBOL,))
    assert h.guard.halt(HaltReason.SIGNAL, _SIGNAL_MSG) is True

    def boom(message: str) -> None:
        raise exc_type("telegram token ZZLEAK9")

    h.listener.on_warning = boom  # type: ignore[method-assign]
    _add_external(h.engine)

    with caplog.at_level(logging.ERROR):
        harness.run(h, _bars())

    assert h.guard.guard_state == GuardState.HALTED, "on_start completed"
    errors = [r for r in caplog.records if "unowned-position report failed" in r.getMessage()]
    assert [r.getMessage() for r in errors] == [expected_log]
    assert errors[0].exc_info is None, "no exc_info on the report failure"
    assert "ZZLEAK9" not in caplog.text
    assert [str(p.id) for p in h.engine.cache.positions_open()] == ["P-EXT-1"]


# ------------------------------------- R21: close-all operator alerts (7fylz)

_STRATEGY_FAILURE = "RuntimeError: distinct strategy exit failure"


@pytest.fixture()
def _seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run for a bare paper node."""
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    rid = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
    )
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    return {"db": db, "sid": sid, "rid": rid, "tmp": tmp_path}


def _close_all_through_node(
    h: harness.Harness,
    seeded: dict[str, Any],
    *,
    break_exit: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[Any, ...]], Any, list[dict[str, object]]]:
    """Run the real guard's close-all via the public node entry, returning the node's
    alerts and the SIGNAL payloads it wrote (the journal keeps the full error list)."""
    if break_exit:

        def boom(self: Any) -> None:
            raise RuntimeError("distinct strategy exit failure")

        monkeypatch.setattr(harness.NaiveStrategy, "market_exit", boom)
    node = PaperTradingNode(_config(seeded))
    node._journal = FakeJournal()
    node._guard = h.guard
    h.guard._listener = _GuardBridge(node)
    results: list[Any] = []
    try:
        harness.run(
            h,
            _bars(),
            actions=[
                (
                    harness.START + timedelta(seconds=30),
                    lambda: results.append(node.close_all_positions()),
                )
            ],
        )
        alerts = [c for c in node._journal.calls if c[0] == "alert"]
        payloads = [
            c[2]
            for c in node._journal.calls
            if c[0] == "write"
            and c[1] == EventType.SIGNAL
            and c[2]["action"] == "close_all_positions"
        ]
        return alerts, results[0], payloads
    finally:
        node._state_manager.close()


def _close_all_alerts(alerts: list[tuple[Any, ...]]) -> list[str]:
    return [str(c[2]) for c in alerts if str(c[2]).startswith("Close-all problems: ")]


def test_close_all_unowned_position_alerts_once(
    _seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R21: an external position on close-all is the guard's one WARNING alert;
    the node's composed close-all alert must not repeat its text - and the SIGNAL
    event still records the unowned error (vibe-quant-7fylz)."""
    h = harness.build(symbols=(_SYMBOL,))
    _add_external(h.engine)

    alerts, result, payloads = _close_all_through_node(
        h, _seeded, break_exit=False, monkeypatch=monkeypatch
    )

    assert sum(_EXT_WARNING in str(c[2]) for c in alerts) == 1
    assert alerts == [("alert", "error", f"WARNING: {_EXT_WARNING}", True)]
    assert _close_all_alerts(alerts) == []
    assert result.errors == [_EXT_WARNING]
    assert payloads[0]["unowned_positions"] == [_EXT_POSITION]
    assert payloads[0]["errors"] == [_EXT_WARNING]


def test_close_all_strategy_error_alerts_once_without_unowned_text(
    _seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R21: a guarded strategy that fails to flatten still reaches the Close-all
    problems alert, exactly once, and no unowned text appears anywhere."""
    h = harness.build(symbols=(_SYMBOL,))
    strategy_error = f"{h.strategies[0].id}: {_STRATEGY_FAILURE}"

    alerts, result, payloads = _close_all_through_node(
        h, _seeded, break_exit=True, monkeypatch=monkeypatch
    )

    assert _close_all_alerts(alerts) == [f"Close-all problems: {strategy_error}"]
    assert sum("positions not owned by any guarded strategy" in str(c[2]) for c in alerts) == 0
    assert result.errors == [strategy_error]
    assert payloads[0]["errors"] == [strategy_error]


def test_close_all_strategy_error_and_unowned_each_alert_once(
    _seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R21: both together - the unowned text and the strategy error each appear in
    exactly one composed operator alert, and the unowned text is not echoed into
    the Close-all problems alert."""
    h = harness.build(symbols=(_SYMBOL,))
    _add_external(h.engine)
    strategy_error = f"{h.strategies[0].id}: {_STRATEGY_FAILURE}"

    alerts, result, payloads = _close_all_through_node(
        h, _seeded, break_exit=True, monkeypatch=monkeypatch
    )

    assert sum(_EXT_WARNING in str(c[2]) for c in alerts) == 1
    assert sum(strategy_error in str(c[2]) for c in alerts) == 1
    assert _close_all_alerts(alerts) == [f"Close-all problems: {strategy_error}"]
    assert result.errors == [strategy_error, _EXT_WARNING]
    assert payloads[0]["errors"] == [strategy_error, _EXT_WARNING]


@pytest.mark.parametrize("prior_error", [False, True], ids=["clean_limiter", "after_error_alert"])
@pytest.mark.parametrize("break_exit", [False, True], ids=["unowned_only", "with_strategy_error"])
async def test_close_all_unowned_text_reaches_telegram_once(
    _seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, prior_error: bool, break_exit: bool
) -> None:
    """R21/R1 (vibe-quant-7fylz): through the real journal and bot (HTTP mocked), the
    unowned-position text reaches Telegram exactly once - also right after another
    ERROR alert, which the 60 s ERROR rate limit would drop unless the alert is
    critical - and a strategy failure reaches it exactly once, never twice."""
    sent: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["text"])
        return httpx.Response(200, json={"ok": True})

    bot = TelegramBot(TelegramConfig(bot_token="test-token", chat_id="42"))

    async def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(record))

    monkeypatch.setattr(bot, "_get_client", client)
    if break_exit:

        def boom(self: Any) -> None:
            raise RuntimeError("distinct strategy exit failure")

        monkeypatch.setattr(harness.NaiveStrategy, "market_exit", boom)
    h = harness.build(symbols=(_SYMBOL,))
    _add_external(h.engine)
    node = PaperTradingNode(_config(_seeded))
    node._journal.attach(None, bot, "test")
    node._guard = h.guard
    h.guard._listener = _GuardBridge(node)
    results: list[Any] = []
    try:
        if prior_error:
            assert await bot.send_error("earlier unrelated failure", bypass_rate_limit=True)
            sent.clear()
        harness.run(
            h,
            _bars(),
            actions=[
                (
                    harness.START + timedelta(seconds=30),
                    lambda: results.append(node.close_all_positions()),
                )
            ],
        )
        await node._journal.wait_alerts(timeout=None)
        assert results[0].unowned_positions == [_EXT_POSITION]
        assert sum(_EXT_WARNING in text for text in sent) == 1, sent
        assert sum("distinct strategy exit failure" in text for text in sent) == int(break_exit), (
            sent
        )
    finally:
        node._state_manager.close()
        await bot.close()


def test_unowned_prefix_is_public_and_guard_text_uses_it() -> None:
    """R2: the prefix is a public guard constant the node filters on, not a private import."""
    assert UNOWNED_PREFIX == "positions not owned by any guarded strategy were NOT closed: "
    assert _EXT_WARNING.startswith(UNOWNED_PREFIX)
