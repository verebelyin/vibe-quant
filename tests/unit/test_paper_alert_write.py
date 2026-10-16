"""Alert delivery when the paper event writer raises (bead vibe-quant-oledr + k8oei).

Every event-write-then-alert pair in ``vibe_quant/paper/node.py`` routes through
:meth:`PaperTradingNode._write_then_alert`: a persistent writer failure (disk full,
permissions) must not swallow the operator's alert, and the write's own exception
still propagates unchanged afterwards (the caller keeps its retry semantics). The
terminal ``_liquidation_loop`` log and ``_shutdown`` are class-only, so neither a
fetch error's ``__context__`` chain nor a traceback (with a signed URL / api key)
can leak.

Contract rows (brief cc-fu-paperalert; chief ruling R8 adds the 7 remaining pairs):

* R1  ``halt()`` pre-guard branch
* R2  ``_on_guard_warning``
* R3  ``start()`` failure path (the ORIGINAL exception propagates, not the write's)
* R4  ``_liquidation_loop`` failure log is class-only
* R5  ``_shutdown`` stop-event write is caught (class-only) and shutdown continues
* R6  ``_shutdown`` ``_event_writer.close()`` raising is caught the same way
* R7  with a healthy writer every site still writes before it alerts
* R8  the 7 further pairs: _on_error_alert, both _check_connectivity sites,
       both _on_guard_state_change branches, _on_order_denied, close_all_positions
* R9  each converted site pins its full event tuple (type, payload, timestamp) and a
       healthy guard transition still reaches its trailing checkpoint (write -> alert
       -> checkpoint), so a dropped timestamp/payload or an early return fails a test.
* R10 (bykze) a guard state transition persists its checkpoint even when the event
       write raises: the ordered trace is ``write[, alert] -> checkpoint`` and only then
       does the write's own error propagate - on the HALT branch, the RESUMED branch and
       a plain (no-alert) change - so a disk-full writer cannot cost the halt its
       checkpoint. The real checkpoint row shows the halted state and reason.
* R11 (bykze) a checkpoint that itself fails never replaces the write's error:
       ``_save_checkpoint_now`` swallows its own failure (the write's OSError still
       propagates).
* R4b (bykze) a healthy plain (non-alerting) guard transition also reaches its
       trailing checkpoint: ``write -> checkpoint`` is pinned by a test, so a mutant
       that skips the checkpoint on that branch dies (Sol SF1).
* R12 (s3pqx, brief row R8) a failing ``_save_checkpoint_now`` logs its own error
       class-only - the store's text, the active write error's text/``__context__`` and
       any traceback stay out of the log - and the write's OSError still propagates.
* R13 (otrm3) a raising ``_commands.close()`` in ``_shutdown`` is caught the same way as
       R6's writer close (class only): the trading engine is still stopped, the event
       writer still flushed and closed, and both handles end as ``None`` - while the
       healthy order (``commands.close`` before ``stop_async``) is unchanged.
* R14 (ott8r) the ``_control_loop`` per-iteration log and the ``process_commands``
       per-command log are class only: the active exception's ``__context__`` chain
       (a fetch error's signed URL / api key text) and any traceback stay out of the
       log, and both loops keep running.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import DSL, _config
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.errors import ErrorCategory, ErrorContext
from vibe_quant.paper.guard import CloseAllResult, GuardState, HaltReason
from vibe_quant.paper.node import NodeState, PaperTradingNode
from vibe_quant.paper.persistence import PaperCommand, StatePersistence

if TYPE_CHECKING:
    from pathlib import Path

#: Leak marker: contains none of the redacted credential names, so a mutant that
#: drops a name cannot hide it (impl-sb-watchlog-2).
_LEAK = "ZZLEAK9"
_TRADER_ID = "PAPER-007"

#: ``node._ns_to_dt(1_700_000_000_000_000_000)`` — the denial event's ``ts_init``.
_DENIED_TS = datetime.fromtimestamp(1_700_000_000, tz=UTC)


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run (liquidation off, leverage 5)."""
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


def _node(seeded: dict[str, Any]) -> PaperTradingNode:
    return PaperTradingNode(_config(seeded))


class _Spy:
    """Records every event write and alert in ONE ordered list.

    Both hooks append to ``order``: separate spies prove counts, not relative order,
    so a helper mutated to alert-then-write would pass a count-only test. ``order``
    is the thing under test (R7). ``writes`` records the full ``(event_type, data,
    timestamp)`` tuple, so a dropped timestamp or a wrong payload fails a test (R9),
    even when the event count and order are unchanged. ``write_error`` makes the
    write raise after the attempt is recorded (a persistent disk-full), tracing the
    raising path too.
    """

    def __init__(self, *, write_error: OSError | None = None) -> None:
        self.order: list[str] = []
        self.alerts: list[tuple[str, str, bool]] = []
        self.writes: list[tuple[EventType, dict[str, object], Any]] = []
        self.error = write_error

    def install(self, monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> None:
        monkeypatch.setattr(node, "_write_event", self.write)
        monkeypatch.setattr(node, "_send_telegram_alert", self.alert)

    def write(self, event_type: EventType, data: dict[str, object], timestamp: Any = None) -> None:
        self.order.append("write")
        self.writes.append((event_type, data, timestamp))
        if self.error is not None:
            raise self.error

    def alert(self, kind: str, message: str, *, critical: bool = False) -> None:
        self.order.append("alert")
        self.alerts.append((kind, message, critical))


def _fail_write(monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> tuple[OSError, _Spy]:
    error = OSError("disk full")
    spy = _Spy(write_error=error)
    spy.install(monkeypatch, node)
    return error, spy


def _trace_checkpoint(monkeypatch: pytest.MonkeyPatch, spy: _Spy, node: PaperTradingNode) -> None:
    """Record the checkpoint in the same ordered trace, so an early return shows (R9)."""
    monkeypatch.setattr(node, "_save_checkpoint_now", lambda: spy.order.append("checkpoint"))


# ------------------------------------------------------------------- R1 halt()


def test_halt_pre_guard_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7: ``halt()`` before a guard exists writes the RISK_CHECK, then alerts."""
    node = _node(seeded)
    spy = _Spy()
    spy.install(monkeypatch, node)

    node.halt(HaltReason.MANUAL, "boom")

    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (
            EventType.RISK_CHECK,
            {"action": "halt", "reason": "manual", "message": "boom", "passed": False},
            None,
        )
    ]
    assert spy.alerts == [("circuit_breaker", "HALT (manual): boom", True)]


def test_halt_pre_guard_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: the OSError propagates after the alert; the status is set before the write."""
    node = _node(seeded)
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node.halt(HaltReason.MANUAL, "boom")

    assert caught.value is error, "the write's own exception object is re-raised"
    assert spy.order == ["write", "alert"], "the alert follows the attempted write"
    assert spy.writes == [
        (
            EventType.RISK_CHECK,
            {"action": "halt", "reason": "manual", "message": "boom", "passed": False},
            None,
        )
    ]
    assert spy.alerts == [("circuit_breaker", "HALT (manual): boom", True)]
    assert node.status.state is NodeState.HALTED
    assert node.status.error_message == "boom"


# --------------------------------------------------------- R2 _on_guard_warning


def test_on_guard_warning_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7: the guard warning is written first, then sent."""
    node = _node(seeded)
    spy = _Spy()
    spy.install(monkeypatch, node)

    node._on_guard_warning("careful")

    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (EventType.RISK_CHECK, {"action": "warning", "message": "careful"}, None)
    ]
    assert spy.alerts == [("error", "WARNING: careful", False)]


def test_on_guard_warning_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: a failing write cannot drop the warning; the OSError still propagates."""
    node = _node(seeded)
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node._on_guard_warning("careful")

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (EventType.RISK_CHECK, {"action": "warning", "message": "careful"}, None)
    ]
    assert spy.alerts == [("error", "WARNING: careful", False)]


# ---------------------------------------------------- R3 start() failure path


async def test_start_failure_alerts_and_raises_the_original_error(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R3: the CRITICAL alert goes out, the ORIGINAL error propagates, log is class-only."""
    node = _node(seeded)
    boom = RuntimeError(f"boom {_LEAK}")

    async def _raise_initialize() -> None:
        raise boom

    async def _noop_shutdown() -> None:
        return None

    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    monkeypatch.setattr(node, "_initialize", _raise_initialize)
    monkeypatch.setattr(node, "_shutdown", _noop_shutdown)
    _error, spy = _fail_write(monkeypatch, node)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as caught:
        await node.start()

    assert caught.value is boom, "start() raises the original failure, not the write's OSError"
    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (
            EventType.RISK_CHECK,
            {"action": "error", "error": f"boom {_LEAK}", "passed": False},
            None,
        )
    ]
    assert spy.alerts == [("error", f"Paper node failed: boom {_LEAK}", True)]
    assert "paper event write failed" in caplog.text
    assert "OSError" in caplog.text
    assert _LEAK not in caplog.text, "the original error's text is never logged"
    assert "Traceback" not in caplog.text
    assert node.status.state is NodeState.ERROR


# --------------------------------------------- R4 _liquidation_loop failure log


async def test_liquidation_loop_failure_log_is_class_only(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R4: the handler logs the class only and still sends the CRITICAL alert."""
    node = _node(seeded)
    spy = _Spy()
    monkeypatch.setattr(node, "_send_telegram_alert", spy.alert)

    def broken() -> None:
        try:
            raise RuntimeError(f"fetch failed {_LEAK}")
        except RuntimeError:
            raise RuntimeError(f"calendar broke {_LEAK}")  # noqa: B904  # keep __context__

    monkeypatch.setattr(node, "_build_liquidation_watcher", broken)

    with caplog.at_level(logging.DEBUG):
        await node._liquidation_loop()

    assert "liquidation watch task failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert _LEAK not in caplog.text, "the __context__ chain must not print"
    assert "Traceback" not in caplog.text
    assert spy.alerts == [
        (
            "error",
            "liquidation watch failed: RuntimeError; exchange liquidations are NOT being reported",
            True,
        )
    ]


# --------------------------------------------------------------- R5/R6 shutdown


class _FakePersistence:
    def __init__(self) -> None:
        self.saved = 0
        self.stopped = 0
        self.closed = 0

    def save_checkpoint(self, checkpoint: object) -> None:
        self.saved += 1

    async def stop_periodic_checkpointing(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        self.closed += 1


class _FakeTradingNodeStop:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop_async(self) -> None:
        self.stopped += 1


class _FakeStateManager:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _FakeTelegramClose:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class _FakeWriter:
    """Event writer whose ``close`` raises (R6)."""

    def __init__(self, *, close_error: OSError | None = None) -> None:
        self.events: list[object] = []
        self.flushed = 0
        self.closed = 0
        self._close_error = close_error

    def write(self, event: object) -> None:
        self.events.append(event)

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1
        if self._close_error is not None:
            raise self._close_error


async def _hang() -> None:
    await asyncio.Event().wait()


async def test_shutdown_stop_event_write_failure_is_class_only_and_shutdown_continues(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R5: a raising stop-event write is logged class-only; the rest of shutdown runs."""
    node = _node(seeded)
    error = OSError(f"disk {_LEAK}")

    def raise_write(event_type: EventType, data: dict[str, object], timestamp: Any = None) -> None:
        raise error

    monkeypatch.setattr(node, "_write_event", raise_write)
    node._event_writer = None
    persistence = _FakePersistence()
    node._persistence = persistence  # type: ignore[assignment]
    trading = _FakeTradingNodeStop()
    node._trading_node = trading  # type: ignore[assignment]
    sm = _FakeStateManager()
    node._state_manager = sm  # type: ignore[assignment]
    node._control_task = asyncio.create_task(_hang())
    node._liquidation_task = asyncio.create_task(_hang())
    control, liquidation = node._control_task, node._liquidation_task

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(node._shutdown(), 5)

    assert "paper stop event write failed" in caplog.text
    assert "OSError" in caplog.text
    assert _LEAK not in caplog.text
    assert "Traceback" not in caplog.text
    assert control.done() and liquidation.done(), "both loops are stopped"
    assert node._control_task is None
    assert node._liquidation_task is None
    assert persistence.saved == 1, "the final checkpoint is still attempted"
    assert persistence.stopped == 1
    assert persistence.closed == 1
    assert node._persistence is None
    assert trading.stopped == 1, "the trading node is still stopped"
    assert sm.closed == 1, "the state manager is still closed"


async def test_shutdown_event_writer_close_failure_is_class_only_and_shutdown_continues(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R6: a raising ``close()`` is logged class-only; the writer is dropped and shutdown ends."""
    node = _node(seeded)
    error = OSError(f"disk {_LEAK}")
    writer = _FakeWriter(close_error=error)
    node._event_writer = writer  # type: ignore[assignment]
    sm = _FakeStateManager()
    node._state_manager = sm  # type: ignore[assignment]
    telegram = _FakeTelegramClose()
    node._telegram = telegram  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(node._shutdown(), 5)

    assert "paper event writer close failed" in caplog.text
    assert "OSError" in caplog.text
    assert _LEAK not in caplog.text
    assert "Traceback" not in caplog.text
    assert writer.closed == 1
    assert node._event_writer is None, "the writer is dropped even though close failed"
    assert telegram.closed == 1, "the telegram client is still closed"
    assert sm.closed == 1, "the state manager is still closed"


# ------------------------------------ R8: _on_error_alert (site 1)


def _error_context(error: BaseException) -> ErrorContext:
    return ErrorContext(
        error=error,
        category=ErrorCategory.FATAL,
        operation="submit_order",
        symbol="BTCUSDT-PERP.BINANCE",
    )


def _error_alert_payload() -> dict[str, object]:
    return {
        "action": "alert",
        "alert_type": "fatal_error",
        "error_category": "fatal",
        "error_type": "RuntimeError",
        "error_message": "venue down",
        "operation": "submit_order",
        "symbol": "BTCUSDT-PERP.BINANCE",
        "retry_count": 0,
        "passed": False,
    }


def test_on_error_alert_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7/R8: the ErrorHandler alert writes the RISK_CHECK first."""
    node = _node(seeded)
    spy = _Spy()
    spy.install(monkeypatch, node)

    node._on_error_alert("fatal_error", _error_context(RuntimeError("venue down")))

    assert spy.order == ["write", "alert"]
    assert spy.writes == [(EventType.RISK_CHECK, _error_alert_payload(), None)]
    assert spy.alerts == [
        ("error", "fatal_error: venue down\nOp: submit_order\nSymbol: BTCUSDT-PERP.BINANCE", False)
    ]


def test_on_error_alert_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: the ErrorHandler alert is sent even when the write raises; OSError propagates."""
    node = _node(seeded)
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node._on_error_alert("fatal_error", _error_context(RuntimeError("venue down")))

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    assert spy.writes == [(EventType.RISK_CHECK, _error_alert_payload(), None)]
    assert spy.alerts == [
        ("error", "fatal_error: venue down\nOp: submit_order\nSymbol: BTCUSDT-PERP.BINANCE", False)
    ]


# -------------------------------- R8: _check_connectivity late-connect (site 2)


class _Engine:
    def __init__(self, connected: bool) -> None:
        self.connected = connected

    def check_connected(self) -> bool:
        return self.connected


class _Kernel:
    def __init__(self, connected: bool) -> None:
        self.data_engine = _Engine(connected)
        self.exec_engine = _Engine(connected)


class _ConnectivityNode:
    def __init__(self, connected: bool) -> None:
        self.kernel = _Kernel(connected)


def test_check_connectivity_late_connect_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7/R8: the after-timeout connect reports once, non-critically."""
    node = _node(seeded)
    node._trading_node = _ConnectivityNode(connected=True)  # type: ignore[assignment]
    node._connected = None
    node._startup_timeout_reported = True
    node._status.error_message = None
    spy = _Spy()
    spy.install(monkeypatch, node)

    node._check_connectivity()

    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (
            EventType.RISK_CHECK,
            {"action": "venue_connected_after_startup_timeout", "passed": True},
            None,
        )
    ]
    assert spy.alerts == [
        ("error", f"paper {_TRADER_ID} venue connected after startup timeout", False)
    ]


def test_check_connectivity_late_connect_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: the late-connect recovery is not lost when the write raises."""
    node = _node(seeded)
    node._trading_node = _ConnectivityNode(connected=True)  # type: ignore[assignment]
    node._connected = None
    node._startup_timeout_reported = True
    node._status.error_message = None
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node._check_connectivity()

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (
            EventType.RISK_CHECK,
            {"action": "venue_connected_after_startup_timeout", "passed": True},
            None,
        )
    ]
    assert spy.alerts == [
        ("error", f"paper {_TRADER_ID} venue connected after startup timeout", False)
    ]


# ----------------------------------- R8: _check_connectivity reconnected (site 3)


def test_check_connectivity_reconnected_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7/R8: a restored connection writes the SIGNAL, then alerts."""
    node = _node(seeded)
    node._trading_node = _ConnectivityNode(connected=True)  # type: ignore[assignment]
    node._connected = False
    spy = _Spy()
    spy.install(monkeypatch, node)

    node._check_connectivity()

    assert spy.order == ["write", "alert"]
    assert spy.writes == [(EventType.SIGNAL, {"action": "reconnected"}, None)]
    assert spy.alerts == [("error", "venue connection restored", False)]


def test_check_connectivity_reconnected_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: the reconnect alert is sent even when the write raises."""
    node = _node(seeded)
    node._trading_node = _ConnectivityNode(connected=True)  # type: ignore[assignment]
    node._connected = False
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node._check_connectivity()

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    assert spy.writes == [(EventType.SIGNAL, {"action": "reconnected"}, None)]
    assert spy.alerts == [("error", "venue connection restored", False)]


# -------------------------------- R8: _on_guard_state_change HALT (site 4)


def test_on_guard_state_change_halt_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7/R8/R9: a guard halt writes the RISK_CHECK, alerts, then checkpoints."""
    node = _node(seeded)
    spy = _Spy()
    spy.install(monkeypatch, node)
    _trace_checkpoint(monkeypatch, spy, node)

    node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "drawdown", {"x": 1})

    assert spy.order == ["write", "alert", "checkpoint"]
    assert spy.writes == [
        (
            EventType.RISK_CHECK,
            {
                "action": "halt",
                "check_type": "manual",
                "reason": "manual",
                "message": "drawdown",
                "passed": False,
                "x": 1,
            },
            None,
        )
    ]
    assert spy.alerts == [("circuit_breaker", "HALT (manual): drawdown", True)]


def test_on_guard_state_change_halt_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8/R10: the CRITICAL halt alert is sent and the halt is checkpointed when the
    write raises; the write's own OSError is what propagates afterwards."""
    node = _node(seeded)
    error, spy = _fail_write(monkeypatch, node)
    _trace_checkpoint(monkeypatch, spy, node)

    with pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "drawdown", {})

    assert caught.value is error
    assert spy.order == ["write", "alert", "checkpoint"]
    assert spy.writes == [
        (
            EventType.RISK_CHECK,
            {
                "action": "halt",
                "check_type": "manual",
                "reason": "manual",
                "message": "drawdown",
                "passed": False,
            },
            None,
        )
    ]
    assert spy.alerts == [("circuit_breaker", "HALT (manual): drawdown", True)]


# ----------------------------- R8: _on_guard_state_change RESUMED (site 5)


def test_on_guard_state_change_resumed_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7/R8/R9: a halt -> running transition writes, alerts, then checkpoints."""
    node = _node(seeded)
    node._status.state = NodeState.HALTED
    spy = _Spy()
    spy.install(monkeypatch, node)
    _trace_checkpoint(monkeypatch, spy, node)

    node._on_guard_state_change(GuardState.RUNNING, None, "back", {})

    assert spy.order == ["write", "alert", "checkpoint"]
    assert spy.writes == [
        (
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": "halted",
                "to_state": "running",
                "message": "back",
            },
            None,
        )
    ]
    assert spy.alerts == [("circuit_breaker", "RESUMED: back", True)]


def test_on_guard_state_change_resumed_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8/R10: the RESUMED alert is sent and the transition checkpointed when the
    write raises; the write's own OSError is what propagates afterwards."""
    node = _node(seeded)
    node._status.state = NodeState.HALTED
    error, spy = _fail_write(monkeypatch, node)
    _trace_checkpoint(monkeypatch, spy, node)

    with pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.RUNNING, None, "back", {})

    assert caught.value is error
    assert spy.order == ["write", "alert", "checkpoint"]
    assert spy.writes == [
        (
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": "halted",
                "to_state": "running",
                "message": "back",
            },
            None,
        )
    ]
    assert spy.alerts == [("circuit_breaker", "RESUMED: back", True)]


def test_on_guard_state_change_without_resume_writes_without_alerting(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4b/R8: a plain state change alerts today, so it must not start alerting now -
    and it still reaches its trailing checkpoint (Sol SF1)."""
    node = _node(seeded)
    node._status.state = NodeState.RUNNING
    spy = _Spy()
    spy.install(monkeypatch, node)
    _trace_checkpoint(monkeypatch, spy, node)

    node._on_guard_state_change(GuardState.PAUSED, None, "paused", {})

    assert spy.order == ["write", "checkpoint"]
    assert spy.writes == [
        (
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": "running",
                "to_state": "paused",
                "message": "paused",
            },
            None,
        )
    ]
    assert spy.alerts == []


def test_on_guard_state_change_plain_change_checkpoints_when_the_write_fails(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3/R10: a plain (no alert) change still checkpoints when the write raises."""
    node = _node(seeded)
    node._status.state = NodeState.RUNNING
    error, spy = _fail_write(monkeypatch, node)
    _trace_checkpoint(monkeypatch, spy, node)

    with pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.PAUSED, None, "paused", {})

    assert caught.value is error
    assert spy.order == ["write", "checkpoint"]
    assert spy.alerts == []


# ------------------------------------ R10/R11: the checkpoint really lands


def test_on_guard_state_change_persists_the_halt_when_the_write_fails(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5/R10: the halt is a REAL checkpoint row even though the RISK_CHECK write failed."""
    node = _node(seeded)
    persistence = StatePersistence(db_path=seeded["db"], trader_id=node._config.trader_id)
    node._persistence = persistence
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "drawdown", {})

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    try:
        checkpoint = persistence.load_latest_checkpoint()
        assert checkpoint is not None
        assert checkpoint.node_status["state"] == "halted"
        assert checkpoint.node_status["halt_reason"] == "manual"
        assert checkpoint.node_status["error_message"] == "drawdown"
    finally:
        persistence.close()


class _RaisingPersistence:
    """A checkpoint store that records the attempt and then fails (R11)."""

    def __init__(self, message: str = "checkpoint store is down") -> None:
        self.calls = 0
        self.message = message

    def save_checkpoint(self, checkpoint: object) -> int:
        self.calls += 1
        raise RuntimeError(self.message)


def test_checkpoint_failure_does_not_replace_the_write_error(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R11: the checkpoint is attempted, fails, is swallowed, and the write's OSError
    (not the checkpoint's error) still propagates."""
    node = _node(seeded)
    persistence = _RaisingPersistence()
    node._persistence = persistence  # type: ignore[assignment]
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "drawdown", {})

    assert caught.value is error
    assert persistence.calls == 1
    assert spy.order == ["write", "alert"]


def test_save_checkpoint_now_failure_log_is_class_only(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R12/R8: a failing checkpoint is logged class-only. The helper now always runs
    under an active write error (the bykze finally), so neither the store's own text nor
    the write error's text/``__context__`` may print; the write's OSError still wins."""
    node = _node(seeded)
    node._persistence = _RaisingPersistence(f"checkpoint store {_LEAK}")  # type: ignore[assignment]
    error = OSError(f"disk {_LEAK}")
    spy = _Spy(write_error=error)
    spy.install(monkeypatch, node)

    with caplog.at_level(logging.DEBUG), pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "drawdown", {})

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    assert "RuntimeError" in caplog.text
    assert _LEAK not in caplog.text, "neither the store error nor the write error may print"
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


# ----------------------------------------- R8: _on_order_denied (site 6)


def _denied_event() -> Any:
    return SimpleNamespace(
        client_order_id="C-1",
        instrument_id="BTCUSDT-PERP.BINANCE",
        strategy_id="S-1",
        reason="min notional",
        ts_init=1_700_000_000_000_000_000,
    )


def _denied_payload(*, by_gate: bool) -> dict[str, object]:
    return {
        "action": "denied",
        "by_guard": by_gate,
        "order_id": "C-1",
        "symbol": "BTCUSDT-PERP.BINANCE",
        "strategy_id": "S-1",
        "reason": "min notional",
    }


def test_on_order_denied_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7/R8/R9: a risk-engine denial writes the ORDER (with its ts_init), then alerts."""
    node = _node(seeded)
    spy = _Spy()
    spy.install(monkeypatch, node)

    node._on_order_denied(_denied_event(), by_gate=False)

    assert spy.order == ["write", "alert"]
    assert spy.writes == [(EventType.ORDER, _denied_payload(by_gate=False), _DENIED_TS)]
    assert spy.alerts == [
        ("error", "Order denied by risk engine: min notional\nBTCUSDT-PERP.BINANCE", False)
    ]


def test_on_order_denied_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: the denial alert is sent even when the write raises."""
    node = _node(seeded)
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node._on_order_denied(_denied_event(), by_gate=False)

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    assert spy.writes == [(EventType.ORDER, _denied_payload(by_gate=False), _DENIED_TS)]
    assert spy.alerts == [
        ("error", "Order denied by risk engine: min notional\nBTCUSDT-PERP.BINANCE", False)
    ]


def test_on_order_denied_by_gate_writes_without_alerting(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8/R9: a gate denial alerts nothing and still writes the ORDER with its ts_init."""
    node = _node(seeded)
    spy = _Spy()
    spy.install(monkeypatch, node)

    node._on_order_denied(_denied_event(), by_gate=True)

    assert spy.order == ["write"]
    assert spy.writes == [(EventType.ORDER, _denied_payload(by_gate=True), _DENIED_TS)]
    assert spy.alerts == []


# ------------------------------------ R8: close_all_positions (site 7)


class _CloseGuard:
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors

    def close_all(self) -> CloseAllResult:
        return CloseAllResult(targeted_positions=["P-1"], errors=list(self.errors))


def _close_all_payload(errors: list[str]) -> dict[str, object]:
    return {
        "action": "close_all_positions",
        "targeted_positions": ["P-1"],
        "unowned_positions": [],
        "errors": errors,
    }


def test_close_all_positions_writes_before_it_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7/R8: close-all errors write the SIGNAL, then the CRITICAL alert."""
    node = _node(seeded)
    node._guard = _CloseGuard(["S-1: venue down"])  # type: ignore[assignment]
    spy = _Spy()
    spy.install(monkeypatch, node)

    result = node.close_all_positions()

    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (EventType.SIGNAL, _close_all_payload(["S-1: venue down"]), None)
    ]
    assert spy.alerts == [("error", "Close-all problems: S-1: venue down", True)]
    assert result.errors == ["S-1: venue down"]


def test_close_all_positions_alert_survives_a_failing_write(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: the close-all error alert is sent even when the write raises."""
    node = _node(seeded)
    node._guard = _CloseGuard(["S-1: venue down"])  # type: ignore[assignment]
    error, spy = _fail_write(monkeypatch, node)

    with pytest.raises(OSError) as caught:
        node.close_all_positions()

    assert caught.value is error
    assert spy.order == ["write", "alert"]
    assert spy.writes == [
        (EventType.SIGNAL, _close_all_payload(["S-1: venue down"]), None)
    ]
    assert spy.alerts == [("error", "Close-all problems: S-1: venue down", True)]


def test_close_all_positions_without_errors_writes_without_alerting(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8: a clean close-all alerts nothing today, so it must not start alerting now."""
    node = _node(seeded)
    node._guard = _CloseGuard([])  # type: ignore[assignment]
    spy = _Spy()
    spy.install(monkeypatch, node)

    node.close_all_positions()

    assert spy.order == ["write"]
    assert spy.writes == [(EventType.SIGNAL, _close_all_payload([]), None)]
    assert spy.alerts == []


# --------------------------------- R13: _shutdown command-queue close (otrm3)


class _FakeCommandQueue:
    """Command queue whose ``close`` can raise (otrm3) and which records the
    claim/fail/complete calls (ott8r)."""

    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.closed = 0
        self.pending: list[PaperCommand] = []
        self.claimed: list[str] = []
        self.fails: list[tuple[int, str]] = []
        self.completes: list[int] = []
        self._close_error = close_error

    def close(self) -> None:
        self.closed += 1
        if self._close_error is not None:
            raise self._close_error

    def claim_pending(self, trader_id: str) -> list[PaperCommand]:
        self.claimed.append(trader_id)
        return list(self.pending)

    def fail(self, command_id: int, error: str, result: object = None) -> None:
        self.fails.append((command_id, error))

    def complete(self, command_id: int, result: object) -> None:
        self.completes.append(command_id)


def _command(command_id: int, command: str) -> PaperCommand:
    return PaperCommand(
        id=command_id,
        trader_id=_TRADER_ID,
        command=command,
        payload={},
        status="running",
        result=None,
        error=None,
    )


async def test_shutdown_command_queue_close_failure_is_class_only_and_shutdown_continues(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R13/otrm3: a raising ``_commands.close()`` is logged class-only and must not skip
    stopping the trading engine, flushing the writer or dropping either handle."""
    node = _node(seeded)
    error = RuntimeError(f"{_LEAK} close")
    commands = _FakeCommandQueue(close_error=error)
    node._commands = commands  # type: ignore[assignment]
    writer = _FakeWriter()
    node._event_writer = writer  # type: ignore[assignment]
    persistence = _FakePersistence()
    node._persistence = persistence  # type: ignore[assignment]
    trading = _FakeTradingNodeStop()
    node._trading_node = trading  # type: ignore[assignment]
    node._state_manager = _FakeStateManager()  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(node._shutdown(), 5)

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1, "exactly one ERROR, and no traceback logger.exception"
    assert errors[0].getMessage() == (
        "paper command queue close failed (RuntimeError); continuing shutdown"
    )
    assert errors[0].exc_info is None
    assert "RuntimeError" in caplog.text
    assert _LEAK not in caplog.text
    assert "Traceback" not in caplog.text
    assert commands.closed == 1
    assert trading.stopped == 1, "the trading engine is still stopped"
    assert writer.flushed == 1, "the writer is still flushed"
    assert writer.closed == 1, "the writer is still closed"
    assert persistence.saved == 1 and persistence.closed == 1
    assert node._commands is None
    assert node._trading_node is None
    assert node._event_writer is None


async def test_shutdown_closes_the_command_queue_before_stopping_the_trading_node(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13/otrm3: the healthy order is unchanged - ``commands.close`` once, then
    ``stop_async``, in one ordered trace."""
    node = _node(seeded)
    order: list[str] = []

    class _OrderedCommands(_FakeCommandQueue):
        def close(self) -> None:
            order.append("commands.close")
            super().close()

    class _OrderedTrading(_FakeTradingNodeStop):
        async def stop_async(self) -> None:
            order.append("stop_async")
            await super().stop_async()

    commands = _OrderedCommands()
    node._commands = commands  # type: ignore[assignment]
    node._event_writer = None
    node._persistence = None
    trading = _OrderedTrading()
    node._trading_node = trading  # type: ignore[assignment]
    node._state_manager = _FakeStateManager()  # type: ignore[assignment]

    await asyncio.wait_for(node._shutdown(), 5)

    assert order == ["commands.close", "stop_async"]
    assert commands.closed == 1
    assert trading.stopped == 1
    assert node._commands is None
    assert node._trading_node is None


# ------------------------------- R14: _control_loop class-only log (ott8r)


async def test_control_loop_iteration_failure_log_is_class_only_and_the_loop_keeps_running(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R14/ott8r: the ``_control_loop`` handler logs the class only, so neither the
    exception's text nor its ``__context__`` chain prints, and the loop iterates again."""
    node = _node(seeded)
    monkeypatch.setattr(node._config, "control_poll_interval", 0.0)
    monkeypatch.setattr(node, "_check_kill_switch", lambda: None)
    monkeypatch.setattr(node, "_check_connectivity", lambda: None)
    iterations = 0

    async def flaky() -> None:
        nonlocal iterations
        iterations += 1
        if iterations == 1:
            try:
                raise RuntimeError(f"fetch failed {_LEAK}")
            except RuntimeError:
                raise OSError(f"disk {_LEAK}")  # noqa: B904  # keep __context__
        node._shutdown_event.set()

    monkeypatch.setattr(node, "process_commands", flaky)

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(node._control_loop(), 5)

    assert iterations == 2, "the loop survived the failure and ran another iteration"
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage() == "paper control loop iteration failed (OSError)"
    assert errors[0].exc_info is None
    assert "OSError" in caplog.text
    assert _LEAK not in caplog.text, "the __context__ chain must not print"
    assert "Traceback" not in caplog.text


# ----------------------------- R14: process_commands class-only log (ott8r)


async def test_process_commands_failure_log_is_class_only_and_the_loop_continues(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R14/ott8r: a failing command logs the class (plus its name) only, still records
    today's ``fail`` text and moves on to the next command."""
    node = _node(seeded)
    commands = _FakeCommandQueue()
    commands.pending = [_command(1, "halt"), _command(2, "resume")]
    node._commands = commands  # type: ignore[assignment]
    executed: list[int] = []

    async def execute(cmd: PaperCommand) -> tuple[bool, dict[str, Any], str | None]:
        executed.append(cmd.id)
        if cmd.id == 1:
            try:
                raise RuntimeError(f"fetch failed {_LEAK}")
            except RuntimeError:
                raise RuntimeError(f"{_LEAK} cmd")  # noqa: B904  # keep __context__
        return True, {"ok": True}, None

    monkeypatch.setattr(node, "_execute_command", execute)

    with caplog.at_level(logging.ERROR):
        await node.process_commands()

    assert executed == [1, 2], "the loop continues with the next command"
    assert commands.fails == [(1, f"RuntimeError: {_LEAK} cmd")], "today's fail text stands"
    assert commands.completes == [2]
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage() == "paper command halt failed (RuntimeError)"
    assert errors[0].exc_info is None
    assert _LEAK not in caplog.text, "neither the command error nor its __context__ may print"
    assert "Traceback" not in caplog.text
