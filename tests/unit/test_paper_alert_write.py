"""Alert delivery when the paper event writer raises (bead vibe-quant-oledr + k8oei).

Every event-write-then-alert pair in ``vibe_quant/paper/node.py`` routes through
``journal.record(..., alert=...)``: a persistent writer failure (disk full,
permissions) must not swallow the operator's alert, and the write's own exception
still propagates unchanged afterwards (the caller keeps its retry semantics). The
terminal ``_liquidation_loop`` log and ``_shutdown`` are class-only, so neither a
fetch error's ``__context__`` chain nor a traceback (with a signed URL / api key)
can leak.

What this file pins (one test per outcome the node itself owns):

* R7/R8/R9  the ALERTING SITES: each one writes its full event tuple (type, payload,
  timestamp) and then sends its alert - one table test, so a site that drops a field,
  alerts before it writes or starts/stops alerting fails here.
* R2/R8  the same table under a failing writer: the caller still sends its alert and the
  write's own OSError propagates (a caller that splits record(...) from alert(...) loses
  the alert here; the journal's own test cannot see that composition).
* R16 (neuvu.1) a replayed start-up halt keeps write -> alert when the submit raises;
  the OSError then leaves ``start()`` and the halt still reaches the checkpoint.
* R3  ``start()`` raises the ORIGINAL failure, not the write's, and logs class only.
* R4  ``_liquidation_loop``'s failure log is class only and still alerts critically.
* R5/R6/R13 (k8oei, otrm3) a raising stop-event write, writer close and command-queue
  close are class only and shutdown continues.
* R10/R11/R12 (bykze, s3pqx) a halt, a resume and a plain pause each land their real
  checkpoint row, even when the transition's write failed; a failing checkpoint store
  never masks the write's error and logs class only.
* R14 (ott8r) ``process_commands`` logs the class only, stores a fixed fail text and
  moves on to the next command.

The journal owns the write-then-alert mechanism itself (``test_paper_journal.py``:
``test_record_writes_then_alerts``, ``test_record_alerts_even_when_the_submit_raises_
then_reraises``), so this file no longer repeats it per site. The HALT/RESUMED
payload and text pairs are pinned end to end by
``test_paper_journal_transcript.py::test_full_session_transcript``; the close-all
composition by ``test_paper_guard_unowned_report.py``; the connectivity pair by
``test_paper_startup_and_estimate.py`` / ``test_paper_recovery_alert_delivery.py``;
the denied-order branches by ``test_paper_entry_check_alert.py``; the control-loop
log by ``test_paper_event_log_node.py``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import DSL, _config, fake_tradingnode  # noqa: F401
from tests.unit.test_paper_halt_restore import _UnitGuard
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.errors import ErrorCategory, ErrorContext
from vibe_quant.paper.guard import CloseAllResult, GuardState, HaltReason
from vibe_quant.paper.journal import FakeJournal
from vibe_quant.paper.node import NodeState, PaperTradingNode
from vibe_quant.paper.persistence import PaperCommand, StatePersistence

if TYPE_CHECKING:
    from collections.abc import Callable
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


class _FailingJournal(FakeJournal):
    """A ``FakeJournal`` whose event write raises once ``armed`` (a disk-full writer).

    The journal is the node's public write/alert edge (``FakeJournal`` is its
    documented test double), so the trace is the real ``record``/``record_durable``
    path: the attempted write is recorded, then the alert, then the error propagates
    unchanged (R1/irdig). ``armed`` lets a start-up test break the writer only after
    the node's own start-up writes, which must still land.
    """

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error
        self.armed = False

    def _write_event(
        self,
        event_type: EventType,
        data: dict[str, object],
        timestamp: datetime | None = None,
    ) -> None:
        self.calls.append(("write", event_type, data, timestamp))
        if self.armed:
            raise self.error
        return None


def _failing(
    seeded: dict[str, Any], *, armed: bool = True, message: str = "disk full"
) -> tuple[PaperTradingNode, _FailingJournal]:
    journal = _FailingJournal(OSError(message))
    journal.armed = armed
    node = _node(seeded)
    node._journal = journal
    return node, journal


def _traced(seeded: dict[str, Any]) -> tuple[PaperTradingNode, FakeJournal]:
    journal = FakeJournal()
    node = _node(seeded)
    node._journal = journal
    return node, journal


# ----------------------------------------------------- R7/R8/R9: the alerting sites


def _error_context(error: BaseException) -> ErrorContext:
    return ErrorContext(
        error=error,
        category=ErrorCategory.FATAL,
        operation="submit_order",
        symbol="BTCUSDT-PERP.BINANCE",
    )


_ERROR_ALERT_PAYLOAD: dict[str, object] = {
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


def _close_all_payload(errors: list[str], *, targeted: list[str]) -> dict[str, object]:
    return {
        "action": "close_all_positions",
        "targeted_positions": targeted,
        "unowned_positions": [],
        "errors": errors,
    }


class _CloseGuard:
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors

    def close_all(self) -> CloseAllResult:
        return CloseAllResult(targeted_positions=["P-1"], errors=list(self.errors))


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


def _connectivity(*, late: bool) -> Callable[[PaperTradingNode], None]:
    def prime(node: PaperTradingNode) -> None:
        node._trading_node = _ConnectivityNode(connected=True)  # type: ignore[assignment]
        node._connected = None if late else False
        node._startup_timeout_reported = late
        node._status.error_message = None

    return prime


def _running(node: PaperTradingNode) -> None:
    node._status.state = NodeState.RUNNING


def _pause(node: PaperTradingNode) -> None:
    node._on_guard_state_change(GuardState.PAUSED, None, "paused", {})


def _with_close_guard(errors: list[str] | None) -> Callable[[PaperTradingNode], None]:
    def prime(node: PaperTradingNode) -> None:
        node._guard = None if errors is None else _CloseGuard(errors)  # type: ignore[assignment]

    return prime


_LATE_CONNECT_PAYLOAD: dict[str, object] = {
    "action": "venue_connected_after_startup_timeout",
    "passed": True,
}

_ALERT_SITES = [
    pytest.param(
        lambda node: None,
        lambda node: node._on_guard_warning("careful"),
        [
            ("write", EventType.RISK_CHECK, {"action": "warning", "message": "careful"}, None),
            ("alert", "error", "WARNING: careful", False),
        ],
        id="guard-warning",
    ),
    pytest.param(
        lambda node: None,
        lambda node: node._on_error_alert(
            "fatal_error", _error_context(RuntimeError("venue down"))
        ),
        [
            ("write", EventType.RISK_CHECK, _ERROR_ALERT_PAYLOAD, None),
            (
                "alert",
                "error",
                "fatal_error: venue down\nOp: submit_order\nSymbol: BTCUSDT-PERP.BINANCE",
                False,
            ),
        ],
        id="error-handler-alert",
    ),
    pytest.param(
        _connectivity(late=False),
        lambda node: node._check_connectivity(),
        [
            ("write", EventType.SIGNAL, {"action": "reconnected"}, None),
            ("alert", "error", "venue connection restored", True),
        ],
        id="venue-reconnected",
    ),
    pytest.param(
        _connectivity(late=True),
        lambda node: node._check_connectivity(),
        [
            ("write", EventType.RISK_CHECK, _LATE_CONNECT_PAYLOAD, None),
            ("alert", "error", f"paper {_TRADER_ID} venue connected after startup timeout", True),
        ],
        id="late-connect",
    ),
    pytest.param(
        lambda node: None,
        lambda node: node._on_order_denied(_denied_event(), by_gate=False),
        [
            ("write", EventType.ORDER, _denied_payload(by_gate=False), _DENIED_TS),
            (
                "alert",
                "error",
                "Order denied by risk engine: min notional\nBTCUSDT-PERP.BINANCE",
                False,
            ),
        ],
        id="risk-engine-denial",
    ),
    pytest.param(
        lambda node: None,
        lambda node: node._on_order_denied(_denied_event(), by_gate=True),
        [("write", EventType.ORDER, _denied_payload(by_gate=True), _DENIED_TS)],
        id="gate-denial-no-alert",
    ),
    pytest.param(
        _running,
        _pause,
        [
            (
                "write",
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": "running",
                    "to_state": "paused",
                    "message": "paused",
                },
                None,
            )
        ],
        id="paused-no-alert",
    ),
    pytest.param(
        _with_close_guard([]),
        lambda node: node.close_all_positions(),
        [("write", EventType.SIGNAL, _close_all_payload([], targeted=["P-1"]), None)],
        id="close-all-clean-no-alert",
    ),
    pytest.param(
        _with_close_guard(["S-1: venue down"]),
        lambda node: node.close_all_positions(),
        [
            (
                "write",
                EventType.SIGNAL,
                _close_all_payload(["S-1: venue down"], targeted=["P-1"]),
                None,
            ),
            ("alert", "error", "Close-all problems: S-1: venue down", True),
        ],
        id="close-all-errors",
    ),
    pytest.param(
        _with_close_guard(None),
        lambda node: node.close_all_positions(),
        [
            (
                "write",
                EventType.SIGNAL,
                _close_all_payload(["trading node not running"], targeted=[]),
                None,
            ),
            ("alert", "error", "Close-all problems: trading node not running", True),
        ],
        id="close-all-without-a-guard",
    ),
]


@pytest.mark.parametrize(("setup", "call", "expected"), _ALERT_SITES)
def test_each_alerting_site_writes_its_event_then_alerts(
    seeded: dict[str, Any],
    setup: Callable[[PaperTradingNode], None],
    call: Callable[[PaperTradingNode], None],
    expected: list[tuple[Any, ...]],
) -> None:
    """R7/R8/R9: one write (full tuple), then the alert - or no alert where none applies.

    ``FakeJournal.calls`` is one ordered list, so a site mutated to alert before it
    writes, to drop a payload field/timestamp, or to start (or stop) alerting fails
    here.
    """
    node, journal = _traced(seeded)
    setup(node)
    call(node)

    assert journal.calls == expected


@pytest.mark.parametrize(("setup", "call", "expected"), _ALERT_SITES)
def test_each_alerting_site_keeps_its_alert_when_the_write_fails(
    seeded: dict[str, Any],
    setup: Callable[[PaperTradingNode], None],
    call: Callable[[PaperTradingNode], None],
    expected: list[tuple[Any, ...]],
) -> None:
    """R2/R8: a failing event write cannot cost the site its alert (the CALLER's
    composition, which the journal's own test cannot see): the same attempted write and
    alert trace as the healthy run, then the write's own OSError propagates unchanged -
    even at the sites that never alert."""
    node, journal = _failing(seeded)
    setup(node)

    with pytest.raises(OSError) as caught:
        call(node)

    assert caught.value is journal.error
    assert journal.calls == expected


# ------------------------------------------------ R16: the replayed start-up halt


async def test_startup_halt_replay_alerts_despite_failing_write(
    seeded: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    fake_tradingnode: Any,  # noqa: F811
) -> None:
    """R16 (neuvu.1): the replayed start-up halt keeps write -> alert even when the
    event submit raises; the OSError then leaves ``start()``, which ends ERROR keeping
    halt_reason signal, and the checkpoint still says so."""
    node, journal = _failing(seeded, armed=False)
    node.halt(HaltReason.SIGNAL, "halt via SIGUSR1")  # what the SIGUSR1 handler does
    monkeypatch.setattr(node_module, "TradingGuard", _UnitGuard)
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)

    async def telegram() -> None:
        journal.armed = True  # from the guard's build on, every event write fails
        return None

    monkeypatch.setattr(node_module, "require_working_telegram", telegram)

    with pytest.raises(OSError) as caught:
        await node.start()

    assert caught.value is journal.error
    halt_calls = [
        call
        for call in journal.calls
        if call[0] == "alert" or (call[0] == "write" and call[2].get("action") == "halt")
    ]
    assert halt_calls[:2] == [
        (
            "write",
            EventType.RISK_CHECK,
            {
                "action": "halt",
                "check_type": "signal",
                "reason": "signal",
                "message": "halt via SIGUSR1",
                "passed": False,
            },
            None,
        ),
        ("alert", "circuit_breaker", "HALT (signal): halt via SIGUSR1", True),
    ], "the replayed halt writes before it alerts, and the alert is still sent"
    assert node.status.state is NodeState.ERROR
    assert node.status.halt_reason is HaltReason.SIGNAL

    persistence = StatePersistence(db_path=seeded["db"], trader_id=_TRADER_ID)
    try:
        checkpoint = persistence.load_latest_checkpoint()
    finally:
        persistence.close()
    assert checkpoint is not None
    assert checkpoint.node_status["halt_reason"] == "signal", "the halt still reaches the store"


# ---------------------------------------------------- R3: start() failure path


async def test_start_failure_alerts_and_raises_the_original_error(
    seeded: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_tradingnode: Any,  # noqa: F811
) -> None:
    """R3: the CRITICAL alert goes out, the ORIGINAL error propagates, log is class-only."""
    node, journal = _failing(seeded, armed=False)
    boom = RuntimeError(f"boom {_LEAK}")
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    monkeypatch.setattr(node_module, "TradingGuard", _UnitGuard)

    async def run_async(_self: Any) -> None:
        journal.armed = True  # the trading loop fails while the event writer is down
        raise boom

    monkeypatch.setattr(fake_tradingnode, "run_async", run_async)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as caught:
        await node.start()

    assert caught.value is boom, "start() raises the original failure, not the write's OSError"
    failure_calls = [
        call
        for call in journal.calls
        if call[0] == "alert" or (call[0] == "write" and call[2].get("action") == "error")
    ]
    assert failure_calls == [
        (
            "write",
            EventType.RISK_CHECK,
            {"action": "error", "error": f"boom {_LEAK}", "passed": False},
            None,
        ),
        ("alert", "error", f"Paper node failed: boom {_LEAK}", True),
    ], "the failure is written first and the alert is sent anyway"
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
    node, journal = _traced(seeded)

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
    assert journal.calls == [
        (
            "write",
            EventType.RISK_CHECK,
            {
                "action": "liquidation_watch_failed",
                "message": "liquidation watch failed: RuntimeError",
                "passed": False,
            },
            None,
        ),
        (
            "alert",
            "error",
            "liquidation watch failed: RuntimeError; exchange liquidations are NOT being reported",
            True,
        ),
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
    seeded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """R5: a raising stop-event write is logged class-only; the rest of shutdown runs."""
    node, journal = _failing(seeded, message=f"disk {_LEAK}")
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
    assert [call[0] for call in journal.calls] == ["write"], "the stop event was attempted once"
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
    seeded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """R6: a raising ``close()`` is logged class-only; the writer is dropped and shutdown ends."""
    node = _node(seeded)
    writer = _FakeWriter(close_error=OSError(f"disk {_LEAK}"))
    node._journal.writer = writer  # type: ignore[assignment]
    sm = _FakeStateManager()
    node._state_manager = sm  # type: ignore[assignment]
    telegram = _FakeTelegramClose()
    node._journal.telegram = telegram  # type: ignore[assignment]

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(node._shutdown(), 5)

    assert "paper event writer close failed" in caplog.text
    assert "OSError" in caplog.text
    assert _LEAK not in caplog.text
    assert "Traceback" not in caplog.text
    assert writer.closed == 1
    assert node._journal.writer is None, "the writer is dropped even though close failed"
    assert telegram.closed == 1, "the telegram client is still closed"
    assert sm.closed == 1, "the state manager is still closed"


# ------------------------------------ R10/R11/R12: the checkpoint really lands


class _RaisingPersistence:
    """A checkpoint store that records the attempt and then fails (R11)."""

    def __init__(self, message: str = "checkpoint store is down") -> None:
        self.calls = 0
        self.message = message

    def save_checkpoint(self, checkpoint: object) -> int:
        self.calls += 1
        raise RuntimeError(self.message)


_TRANSITIONS = [
    pytest.param(
        NodeState.RUNNING,
        GuardState.HALTED,
        HaltReason.MANUAL,
        "drawdown",
        ["write", "alert"],
        ("halted", "manual"),
        id="halt",
    ),
    pytest.param(
        NodeState.HALTED,
        GuardState.RUNNING,
        None,
        "back",
        ["write", "alert"],
        ("running", None),
        id="resumed",
    ),
    pytest.param(
        NodeState.RUNNING,
        GuardState.PAUSED,
        None,
        "paused",
        ["write"],
        ("paused", None),
        id="plain-pause",
    ),
]


@pytest.mark.parametrize("write_fails", [False, True], ids=["write-ok", "write-fails"])
@pytest.mark.parametrize(("prev", "state", "reason", "message", "kinds", "persisted"), _TRANSITIONS)
def test_on_guard_state_change_persists_the_transition_even_when_the_write_fails(
    seeded: dict[str, Any],
    write_fails: bool,
    prev: NodeState,
    state: GuardState,
    reason: HaltReason | None,
    message: str,
    kinds: list[str],
    persisted: tuple[str, str | None],
) -> None:
    """R10: a halt, a resume and a plain pause are each a REAL checkpoint row - the new
    state, the halt reason and the message - whether or not the event write failed, so
    a crash before the next save restores the latest guard state, not an older one."""
    journal: FakeJournal = _FailingJournal(OSError("disk full")) if write_fails else FakeJournal()
    node = _node(seeded)
    node._journal = journal
    node._status.state = prev
    persistence = StatePersistence(db_path=seeded["db"], trader_id=node._config.trader_id)
    node._persistence = persistence
    if isinstance(journal, _FailingJournal):
        journal.armed = True
        with pytest.raises(OSError) as caught:
            node._on_guard_state_change(state, reason, message, {})
        assert caught.value is journal.error
    else:
        node._on_guard_state_change(state, reason, message, {})

    assert [call[0] for call in journal.calls] == kinds
    node._db_worker.submit(lambda: None).result(5)  # the save is queued on the FIFO worker
    try:
        checkpoint = persistence.load_latest_checkpoint()
        assert checkpoint is not None
        assert checkpoint.node_status["state"] == persisted[0]
        assert checkpoint.node_status["halt_reason"] == persisted[1]
        assert checkpoint.node_status["error_message"] == message
    finally:
        persistence.close()


def test_checkpoint_failure_never_masks_the_write_error_and_logs_class_only(
    seeded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """R11/R12: the checkpoint is attempted, fails and is swallowed - its own text, the
    write error's text/``__context__`` and any traceback stay out of the class-only log,
    and the write's OSError is what still propagates."""
    node, journal = _failing(seeded, message=f"disk {_LEAK}")
    persistence = _RaisingPersistence(f"checkpoint store {_LEAK}")
    node._persistence = persistence  # type: ignore[assignment]

    with caplog.at_level(logging.DEBUG), pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "drawdown", {})

    node._db_worker.submit(lambda: None).result(5)  # the save is queued on the FIFO worker
    assert caught.value is journal.error
    assert persistence.calls == 1, "the checkpoint was attempted"
    assert [call[0] for call in journal.calls] == ["write", "alert"]
    assert "RuntimeError" in caplog.text
    assert _LEAK not in caplog.text, "neither the store error nor the write error may print"
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


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
    seeded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """R13/otrm3: a raising ``_commands.close()`` is logged class-only and must not skip
    stopping the trading engine, flushing the writer or dropping either handle."""
    node = _node(seeded)
    commands = _FakeCommandQueue(close_error=RuntimeError(f"{_LEAK} close"))
    node._commands = commands  # type: ignore[assignment]
    writer = _FakeWriter()
    node._journal.writer = writer  # type: ignore[assignment]
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
    assert node._journal.writer is None


# ----------------------------- R14: process_commands class-only log (ott8r)


async def test_process_commands_failure_log_is_class_only_and_the_loop_continues(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R14/ott8r: a failing command logs the class (plus its name) only, records a
    class-only ``fail`` text (nothing from the exception) and moves on to the next
    command."""
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
    assert commands.fails == [(1, "RuntimeError: command failed; see the paper node log")], (
        "the stored error is the class plus fixed text, never the exception text"
    )
    assert _LEAK not in str(commands.fails), "the exception text must not reach the command row"
    assert commands.completes == [2]
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage() == "paper command halt failed (RuntimeError)"
    assert errors[0].exc_info is None
    assert _LEAK not in caplog.text, "neither the command error nor its __context__ may print"
    assert "Traceback" not in caplog.text
