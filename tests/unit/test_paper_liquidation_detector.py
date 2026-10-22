"""Paper node wiring for the forceOrders liquidation detector (T6 / P3b).

Bead vibe-quant-2gyk1.1.11. Contract rows (design r2, job 20261015-honest-safe) plus
the chief's round-1 rulings R47e (``mode`` vocabulary) and R50c (restore boundary):

* R47  each new force order -> exactly ONE base ``LIQUIDATION`` event
       (``source='binance_force_orders'``, ``loss=None``, int ``ts_ns``, ``mode``
       ``'isolated'``/``'cross'``) and one CRITICAL Telegram.
* R47b a missing/raising event writer makes ``on_liquidation`` raise, so the watcher
       fails that symbol's cycle (cursor unchanged) and delivers the row next cycle.
* R49  REPORT ONLY: no submit, cancel, close or halt; strategy and guard untouched.
* R50  ``node_status['liquidation_watch']`` carries ``watcher.state()``; a fresh start
       (no checkpoint at all) puts every traded symbol at the node start ms.
* R50b/R50c  a prior checkpoint restores the per-symbol cursors it stores and puts every
       traded symbol it omits at the CHECKPOINT timestamp ms (read as UTC); ``seen`` is
       restored.
* R52  a degraded detector warns ONCE at three failures (RISK_CHECK
       ``liquidation_watch_degraded`` plus a NON-critical Telegram) and the first
       success after that reports ``liquidation_watch_recovered`` the same way.
* R52b at most one delivery per key per process, at least one across a restart.
* R53  shutdown sets the stop event, cancels the task and suppresses CancelledError.

No network and no wall-clock sleeping: the fetch is a fake and the clock is injected.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import DSL, _config
from tests.unit.test_paper_risk_inherit import _risk
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper import lineage
from vibe_quant.paper.config import account_id
from vibe_quant.paper.force_orders import ForceOrder
from vibe_quant.paper.node import HaltReason, NodeState, PaperTradingNode
from vibe_quant.paper.persistence import StatePersistence

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from vibe_quant.logging.events import Event

_TRADER_ID = "PAPER-007"
_RESTART_ID = "PAPER-008"
#: Two traded symbols, because R50 says "EVERY traded symbol".
_SYMBOLS = ("BTCUSDT", "ETHUSDT")
#: Injected node start: 2026-01-02T03:00:00Z.
_START = datetime(2026, 1, 2, 3, 0, 0, tzinfo=UTC)
_START_MS = 1_767_322_800_000
#: A checkpoint row two hours earlier (``StatePersistence`` stores ``datetime('now')``
#: as naive UTC text: persistence.py:39).
_CHECKPOINT_STAMP = "2026-01-02 02:00:00"
_CHECKPOINT_MS = 1_767_319_200_000
_ROW_MS = _START_MS - 60_000
_CLIENT_ORDER_ID = "autoclose-1596107620040000020"
_ORDER_ID = 1596107620040000020
_OVERLAP_MS = 300_000


class _Clock:
    """Injected ``_now``: time only moves when the test moves it."""

    def __init__(self, now: datetime = _START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _Alerts:
    """Records every ``PaperTradingNode._send_telegram_alert`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, kind: str, message: str, *, critical: bool = False) -> None:
        self.calls.append((kind, message, critical))


class _Trace:
    """One ordered log of the side effects a contract orders (SF1 / R1 irdig).

    The write and the alert are recorded into ONE list: separate spies prove counts,
    not relative order, so a helper mutated from write-then-alert to alert-then-write
    still passes a count-only test. ``order`` is the thing under test.
    """

    def __init__(self, alerts: _Alerts) -> None:
        self.order: list[str] = []
        self.alerts = alerts

    def write(self, event: Event) -> None:
        self.order.append("write")

    def alert(self, kind: str, message: str, *, critical: bool = False) -> None:
        self.order.append("alert")
        self.alerts(kind, message, critical=critical)


class _Spy:
    """A money-touching method that records its calls instead of running."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


def _spy_methods(
    monkeypatch: pytest.MonkeyPatch, target: object, names: Sequence[str]
) -> dict[str, _Spy]:
    """Replace each named method with a recording spy."""
    spies: dict[str, _Spy] = {}
    for name in names:
        assert hasattr(target, name), f"{target!r} has no {name} to spy on"
        spy = _Spy(name)
        monkeypatch.setattr(target, name, spy)
        spies[name] = spy
    return spies


class _FakeFetch:
    """Stand-in for P3a1's ``Fetch``: records the window, replays rows or raises."""

    def __init__(
        self,
        orders: Mapping[str, Sequence[ForceOrder]] | None = None,
        errors: Mapping[str, Exception] | None = None,
    ) -> None:
        self.requests: list[tuple[str, int, int]] = []
        self.orders: dict[str, Sequence[ForceOrder]] = dict(orders or {})
        self.errors: dict[str, Exception] = dict(errors or {})

    async def __call__(self, symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        self.requests.append((symbol, start_ms, end_ms))
        error = self.errors.get(symbol)
        if error is not None:
            raise error
        return [
            order
            for order in self.orders.get(symbol, ())
            if start_ms <= order.update_time_ms <= end_ms
        ]


class _FetchFactory:
    """Stands in for ``build_binance_fetch``: records the credentials it is handed."""

    def __init__(self, fetch: _FakeFetch) -> None:
        self.fetch = fetch
        self.credentials: list[tuple[str, str, str, bool]] = []

    def __call__(
        self, api_key: str, api_secret: str, account_type: str, testnet: bool
    ) -> _FakeFetch:
        self.credentials.append((api_key, api_secret, account_type, testnet))
        return self.fetch


class _FakeKernel:
    class _RiskEngine:
        def execute(self, command: object) -> None:
            pass

    def __init__(self) -> None:
        self.risk_engine = self._RiskEngine()


class _FakeTrader:
    def __init__(self) -> None:
        self.actors: list[object] = []
        self.strategies: list[object] = []

    def add_actor(self, actor: object) -> None:
        self.actors.append(actor)

    def add_strategy(self, strategy: object) -> None:
        self.strategies.append(strategy)


class _FakeTradingNode:
    """Stand-in for ``nautilus_trader.live.node.TradingNode`` (no venue, no connect)."""

    instances: list[_FakeTradingNode] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.kernel = _FakeKernel()
        self.trader = _FakeTrader()
        _FakeTradingNode.instances.append(self)

    def add_data_client_factory(self, name: str, factory: object) -> None:
        pass

    def add_exec_client_factory(self, name: str, factory: object) -> None:
        pass

    def build(self) -> None:
        pass

    async def run_async(self) -> None:
        return

    async def stop_async(self) -> None:
        return


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


@dataclass
class _Harness:
    """One initialized node plus the seams the test observes it through."""

    node: PaperTradingNode
    clock: _Clock
    alerts: _Alerts
    fetch: _FakeFetch
    factory: _FetchFactory
    log: Path

    def events(self, event_type: EventType) -> list[dict[str, Any]]:
        """The logged events of one type, in write order."""
        if self.node._event_writer is not None:
            self.node._event_log.barrier().result(10)  # lines are appended on the event worker
            self.node._event_writer.flush()
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        entries = [json.loads(line) for line in lines if line.strip()]
        return [entry for entry in entries if entry["event"] == event_type.value]

    def actions(self, action: str) -> list[dict[str, Any]]:
        """The RISK_CHECK payloads carrying one ``action``."""
        return [
            entry["data"]
            for entry in self.events(EventType.RISK_CHECK)
            if entry["data"].get("action") == action
        ]


def _order(
    *,
    symbol: str = "BTCUSDT",
    order_id: int = _ORDER_ID,
    update_time_ms: int = _ROW_MS,
    executed_qty: str = "0.4",
    side: str = "SELL",
    avg_price: str = "2500.5",
    client_order_id: str = _CLIENT_ORDER_ID,
    status: str = "FILLED",
) -> ForceOrder:
    """One decoded forceOrders row (the shape P3a1 hands the watcher)."""
    return ForceOrder.from_binance(
        {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": client_order_id,
            "side": side,
            "status": status,
            "avgPrice": avg_price,
            "executedQty": executed_qty,
            "updateTime": update_time_ms,
            "autoCloseType": "LIQUIDATION",
        }
    )


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed 2-symbol validation run (liquidation off, leverage 5)."""
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    rid = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=list(_SYMBOLS),
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
    )
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    return {"db": db, "sid": sid, "rid": rid, "tmp": tmp_path}


def _validation_run(seeded: dict[str, Any], parameters: dict[str, Any]) -> int:
    sm = StateManager(seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seeded["sid"],
            run_mode="validation",
            symbols=list(_SYMBOLS),
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters=parameters,
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid


def _write_checkpoint(
    seeded: dict[str, Any], *, trader_id: str, stamp: str, node_status: dict[str, Any]
) -> None:
    """Insert one checkpoint row exactly as ``StatePersistence`` stores it."""
    persistence = StatePersistence(db_path=seeded["db"], trader_id=trader_id)
    try:
        persistence.conn.execute(
            """INSERT INTO paper_trading_checkpoints
               (trader_id, positions, orders, balance, node_status, created_at)
               VALUES (?, '{}', '{}', '{}', ?, ?)""",
            (trader_id, json.dumps(node_status), stamp),
        )
        persistence.conn.commit()
    finally:
        persistence.close()


async def _harness(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    *,
    orders: Mapping[str, Sequence[ForceOrder]] | None = None,
    errors: Mapping[str, Exception] | None = None,
    clock: _Clock | None = None,
    alerts: _Alerts | None = None,
    trader_id: str = _TRADER_ID,
    logs_dir: str = "logs",
    validation_run_id: int | None = None,
    start_fresh: bool = False,
) -> _Harness:
    """An initialized node whose watch fetcher and clock are injected seams."""
    fetch = _FakeFetch(orders, errors)
    factory = _FetchFactory(fetch)
    clock = clock if clock is not None else _Clock()
    alerts = alerts if alerts is not None else _Alerts()
    overrides: dict[str, Any] = {}
    if validation_run_id is not None:
        overrides["validation_run_id"] = validation_run_id
    if start_fresh:
        overrides["start_fresh"] = True
    node = PaperTradingNode(
        _config(
            seeded, trader_id=trader_id, logs_path=seeded["tmp"] / logs_dir, **overrides
        )
    )
    monkeypatch.setattr(node, "_now", clock)
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    monkeypatch.setattr(node_module, "build_binance_fetch", factory)
    await node._initialize()
    return _Harness(
        node=node,
        clock=clock,
        alerts=alerts,
        fetch=fetch,
        factory=factory,
        log=seeded["tmp"] / logs_dir / f"{trader_id}.jsonl",
    )


async def _until(predicate: Callable[[], bool], *, timeout_s: float = 10.0) -> None:
    """Yield to the event loop until ``predicate`` holds.

    The liquidation report waits for the event-log worker (ogd37), so spinning a fixed
    number of zero-sleep cycles can end before the thread answers: bound it by time.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


# ------------------------------------------------------------------------- R47


async def test_force_order_writes_liquidation_event_and_critical_alert_without_orders(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R47: one base LIQUIDATION event (full payload) and one CRITICAL alert, no orders."""
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [_order()]})
    node = harness.node
    try:
        node._start_liquidation_task()
        task = node._liquidation_task
        assert task is not None, "the watch task must exist beside the control loop"
        await _until(lambda: len(harness.alerts.calls) == 1)
        node._liquidation_stop.set()
        await asyncio.wait_for(task, 5)
    finally:
        await node._shutdown()

    assert harness.factory.credentials == [("key", "secret", "USDT_FUTURES", True)], (
        "the fetch is built from the node's own (testnet) Binance config"
    )
    assert harness.fetch.requests == [
        ("BTCUSDT", _START_MS - _OVERLAP_MS, _START_MS),
        ("ETHUSDT", _START_MS - _OVERLAP_MS, _START_MS),
    ]

    events = harness.events(EventType.LIQUIDATION)
    assert len(events) == 1, f"exactly one LIQUIDATION event, got {events}"
    assert isinstance(events[0]["data"]["ts_ns"], int), "R47: ts_ns is an int, never a float"
    assert events[0]["data"] == {
        "source": "binance_force_orders",
        "position_id": "BTCUSDT-PERP.BINANCE",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "auto_close_type": "LIQUIDATION",
        "liquidation_price": 2500.5,
        "quantity": 0.4,
        "loss": None,
        "ts_ns": _ROW_MS * 1_000_000,
        "mode": "cross",
        "venue_order_id": str(_ORDER_ID),
        "client_order_id": _CLIENT_ORDER_ID,
        "status": "FILLED",
    }

    assert len(harness.alerts.calls) == 1, "one CRITICAL Telegram per force order"
    kind, message, critical = harness.alerts.calls[0]
    assert kind == "error"
    assert critical is True, "the alert bypasses the per-type rate limit"
    assert "BTCUSDT" in message
    assert "LIQUIDATION" in message

    assert harness.events(EventType.ORDER) == [], "report only: no order is ever submitted"
    assert harness.actions("halt") == [], "report only: a liquidation never halts the node"


# ------------------------------------------------------------------------- R49


async def test_watcher_never_submits_cancels_closes_or_halts(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R49: a row on each symbol reports and moves nothing - strategy/guard untouched."""
    orders = {
        "BTCUSDT": [_order()],
        "ETHUSDT": [_order(symbol="ETHUSDT", order_id=99, update_time_ms=_START_MS - 30_000)],
    }
    harness = await _harness(monkeypatch, seeded, orders=orders)
    node = harness.node
    guard = node.guard
    assert guard is not None
    money_names = (
        "submit_order",
        "cancel_order",
        "cancel_all_orders",
        "modify_order",
        "close_position",
        "close_all_positions",
    )
    guard_spies = _spy_methods(
        monkeypatch, guard, ("halt", "pause", "resume", "close_all", "evaluate_risk")
    )
    strategy_spies = [
        _spy_methods(monkeypatch, strategy, money_names)
        for strategy in node._compiled_strategies
    ]

    watcher = node._build_liquidation_watcher()
    await watcher.poll_once()

    assert len(harness.events(EventType.LIQUIDATION)) == 2, "both symbols still report"
    assert all(not spy.calls for spy in guard_spies.values()), guard_spies
    assert all(
        not spy.calls for spies in strategy_spies for spy in spies.values()
    ), strategy_spies
    assert harness.events(EventType.ORDER) == []
    assert harness.events(EventType.FILL) == []
    assert harness.events(EventType.POSITION_CLOSE) == []
    await node._shutdown()


@pytest.mark.parametrize(
    ("parameters", "margin_type", "mode"),
    [
        ({"liquidation_mode": "isolated", "leverage": 10}, "ISOLATED", "isolated"),
        ({"liquidation_mode": "cross", "leverage": 5}, "CROSSED", "cross"),
    ],
)
async def test_mode_follows_the_resolved_margin_type(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    parameters: dict[str, Any],
    margin_type: str,
    mode: str,
) -> None:
    """R47e: ``mode`` is the event vocabulary ('isolated'/'cross'), never 'crossed'."""
    rid = _validation_run(seeded, parameters)
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [_order()]},
                             validation_run_id=rid)
    assert harness.node.resolved_config is not None
    assert harness.node.resolved_config.margin_type == margin_type
    watcher = harness.node._build_liquidation_watcher()
    await watcher.poll_once()

    payload = harness.events(EventType.LIQUIDATION)[0]["data"]
    assert payload["mode"] == mode
    await harness.node._shutdown()


# ------------------------------------------------------------------------ R47b


async def test_missing_event_writer_raises_so_the_cycle_retries(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R47b: a missing writer raises; the cycle keeps the cursor and the next one retries."""
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [_order()]})
    node = harness.node
    watcher = node._build_liquidation_watcher()
    writer = node._event_writer
    assert writer is not None

    node._event_writer = None
    with pytest.raises(RuntimeError, match="event writer"):
        await node._on_exchange_liquidation(_order())

    await watcher.poll_once()
    assert harness.events(EventType.LIQUIDATION) == [], "no writer: nothing was reported"
    assert watcher.state()["cursors"]["BTCUSDT"] == _START_MS, "R47b: cursor unchanged"
    assert watcher.state()["seen"] == {}

    node._event_writer = writer
    harness.clock.advance(61)  # exactly the first backoff (60 s) plus a second
    await watcher.poll_once()
    assert len(harness.events(EventType.LIQUIDATION)) == 1, "the row is delivered on retry"
    await node._shutdown()


# ------------------------------------------------------------------------- R50


async def test_fresh_start_cursors_are_the_node_start_for_every_traded_symbol(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R50/R50b: with no checkpoint at all every traded symbol starts at the node clock."""
    harness = await _harness(monkeypatch, seeded)
    watcher = harness.node._build_liquidation_watcher()

    assert watcher.state() == {
        "cursors": {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS},
        "seen": {},
    }
    await harness.node._shutdown()


async def test_checkpoint_records_the_watch_state(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R50: ``node_status['liquidation_watch']`` is the watcher's own state."""
    orders = {
        "BTCUSDT": [_order()],
        "ETHUSDT": [_order(symbol="ETHUSDT", order_id=99, update_time_ms=_START_MS - 30_000)],
    }
    harness = await _harness(monkeypatch, seeded, orders=orders)
    watcher = harness.node._build_liquidation_watcher()
    await watcher.poll_once()

    status = harness.node._capture_checkpoint().node_status
    assert status["liquidation_watch"] == {
        "cursors": {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS},
        "seen": {
            f"BTCUSDT:{_ORDER_ID}": _ROW_MS,
            "ETHUSDT:99": _START_MS - 30_000,
        },
    }
    await harness.node._shutdown()


async def test_restore_uses_checkpoint_cursors_and_timestamp_for_missing_symbols(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R50b/R50c: stored cursors win; an omitted traded symbol gets the checkpoint stamp."""
    _write_checkpoint(
        seeded,
        trader_id=_RESTART_ID,
        stamp=_CHECKPOINT_STAMP,
        node_status={
            "liquidation_watch": {
                "cursors": {"BTCUSDT": 1_767_321_000_000, "STALEUSDT": 1},
                "seen": {"BTCUSDT:7": 1_767_320_000_000, "ETHUSDT:8": 1},
            }
        },
    )
    harness = await _harness(
        monkeypatch, seeded, trader_id=_RESTART_ID, logs_dir="logs2"
    )
    assert harness.node._restored_watch_ms == _CHECKPOINT_MS
    watcher = harness.node._build_liquidation_watcher()

    assert watcher.state() == {
        "cursors": {
            "BTCUSDT": 1_767_321_000_000,
            "ETHUSDT": _CHECKPOINT_MS,
            "STALEUSDT": 1,
        },
        "seen": {"BTCUSDT:7": 1_767_320_000_000},
    }
    await harness.node._shutdown()


@pytest.mark.parametrize("host_tz", ["Europe/Budapest", "America/New_York", "UTC"])
def test_checkpoint_timestamp_is_read_as_utc(
    monkeypatch: pytest.MonkeyPatch, host_tz: str
) -> None:
    """R50b: the checkpoint row's timestamp is naive UTC text (persistence.py:39).

    The host zone is forced, so a ``.timestamp()`` regression fails on any machine.
    """
    monkeypatch.setenv("TZ", host_tz)
    time.tzset()
    try:
        naive = datetime(2026, 1, 2, 2, 0, 0)
        assert lineage.checkpoint_ms(naive) == _CHECKPOINT_MS
        assert lineage.checkpoint_ms(naive.replace(tzinfo=UTC)) == _CHECKPOINT_MS
    finally:
        monkeypatch.undo()
        time.tzset()


# ------------------------------------------------------------------------ R50d

_STORED_WATCH = {
    "cursors": {"BTCUSDT": 1_767_321_000_000},
    "seen": {"BTCUSDT:7": 1_767_320_000_000},
}


def test_restored_watch_state_keeps_only_int_entries() -> None:
    """R50: a stored watch map round-trips through JSON; only ints are cursors.

    Pins today's filter (vibe-quant-rabmo.1c, moved verbatim from the node): ``True``
    is an ``int`` in Python but never a cursor, and a float / str / None entry is
    dropped instead of reaching the watcher. A missing or non-dict payload is empty.
    """
    assert lineage.restored_watch_state(
        {
            "cursors": {
                "BTCUSDT": 1_767_321_000_000,
                "BOOLUSDT": True,
                "FLOATUSDT": 1.5,
                "STRUSDT": "8",
                "NONEUSDT": None,
            },
            "seen": {"BTCUSDT:7": 1_767_320_000_000, "BOOLUSDT:8": False},
        }
    ) == {
        "cursors": {"BTCUSDT": 1_767_321_000_000},
        "seen": {"BTCUSDT:7": 1_767_320_000_000},
    }
    assert lineage.restored_watch_state(None) == {"cursors": {}, "seen": {}}
    assert lineage.restored_watch_state({"cursors": "junk", "seen": 5}) == {
        "cursors": {},
        "seen": {},
    }


def _latest_watch(harness: _Harness) -> Any:
    persistence = harness.node._persistence
    assert persistence is not None
    checkpoint = persistence.load_latest_checkpoint(_RESTART_ID)
    assert checkpoint is not None
    return checkpoint.node_status.get("liquidation_watch")


async def test_init_checkpoint_keeps_the_restored_watch_state(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R50d: the checkpoint ``_initialize`` writes (no watcher yet) keeps the stored state."""
    _write_checkpoint(
        seeded, trader_id=_RESTART_ID, stamp=_CHECKPOINT_STAMP,
        node_status={"liquidation_watch": _STORED_WATCH},
    )
    harness = await _harness(monkeypatch, seeded, trader_id=_RESTART_ID, logs_dir="logs2")

    assert harness.node._liquidation_watcher is None, "no watcher has been built yet"
    assert _latest_watch(harness) == {
        # the traded symbol the stored state omits keeps its downtime gap (R50c)
        "cursors": {"BTCUSDT": 1_767_321_000_000, "ETHUSDT": _CHECKPOINT_MS},
        "seen": {"BTCUSDT:7": 1_767_320_000_000},
    }
    await harness.node._shutdown()


async def test_crash_loop_before_the_watcher_starts_never_loses_the_watch_state(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R50d: start -> early failure (shutdown, no watcher) -> restart still scans the gap."""
    _write_checkpoint(
        seeded, trader_id=_RESTART_ID, stamp=_CHECKPOINT_STAMP,
        node_status={"liquidation_watch": _STORED_WATCH},
    )
    first = await _harness(monkeypatch, seeded, trader_id=_RESTART_ID, logs_dir="logs2")
    await first.node._shutdown()  # the final checkpoint is also written without a watcher

    second = await _harness(monkeypatch, seeded, trader_id=_RESTART_ID, logs_dir="logs3")
    watcher = second.node._build_liquidation_watcher()

    assert watcher.state() == {
        "cursors": {"BTCUSDT": 1_767_321_000_000, "ETHUSDT": _CHECKPOINT_MS},
        "seen": {"BTCUSDT:7": 1_767_320_000_000},
    }
    await second.node._shutdown()


async def test_init_checkpoint_after_a_checkpoint_without_watch_state_keeps_its_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R50d/R50c: a prior checkpoint with no watch state still bounds the downtime gap."""
    _write_checkpoint(
        seeded, trader_id=_RESTART_ID, stamp=_CHECKPOINT_STAMP, node_status={"state": "running"}
    )
    first = await _harness(monkeypatch, seeded, trader_id=_RESTART_ID, logs_dir="logs2")
    await first.node._shutdown()

    second = await _harness(monkeypatch, seeded, trader_id=_RESTART_ID, logs_dir="logs3")
    watcher = second.node._build_liquidation_watcher()

    assert watcher.state()["cursors"] == {
        "BTCUSDT": _CHECKPOINT_MS,
        "ETHUSDT": _CHECKPOINT_MS,
    }, "the init checkpoint must not move the gap start to the restart time"
    await second.node._shutdown()


async def test_first_ever_start_checkpoint_carries_no_watch_state(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R50: with nothing to restore, the init checkpoint does not invent watch state."""
    harness = await _harness(monkeypatch, seeded, trader_id=_RESTART_ID)
    assert _latest_watch(harness) is None
    await harness.node._shutdown()


# ------------------------------------------------------------------------ gzf8x


_HALT_STAMP = "2026-01-02 00:30:00"
_HALT_STAMP_MS = 1_767_313_800_000


def _newest_checkpoint_cursors(seeded: dict[str, Any]) -> Any:
    """R1: the watch cursors of the newest row (persistence picks ``ORDER BY id DESC``)."""
    conn = sqlite3.connect(seeded["db"])
    try:
        row = conn.execute(
            "SELECT node_status FROM paper_trading_checkpoints ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return json.loads(row[0])["liquidation_watch"]["cursors"]


async def _started_session(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    logs_dir: str,
) -> tuple[PaperTradingNode, _FakeFetch, _Alerts]:
    """Run the real ``start()`` until the watcher has polled both traded symbols."""
    fetch = _FakeFetch({"BTCUSDT": [_order(update_time_ms=_CHECKPOINT_MS - 3_600_000)]})
    alerts = _Alerts()
    node = PaperTradingNode(
        _config(seeded, trader_id=_TRADER_ID, logs_path=seeded["tmp"] / logs_dir)
    )
    monkeypatch.setattr(node, "_now", _Clock())
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    monkeypatch.setattr(node_module, "build_binance_fetch", _FetchFactory(fetch))

    async def two_requests(_self: _FakeTradingNode) -> None:
        # qe6ye: symbols are fetched concurrently, so wait for the cycle to FINISH
        # (both cursors advanced), not merely for both requests to be sent.
        for _ in range(400):
            watcher = node._liquidation_watcher
            if (
                len(fetch.requests) >= 2
                and watcher is not None
                and set(watcher.state()["cursors"].values()) == {_START_MS}
            ):
                break
            await asyncio.sleep(0.01)

    monkeypatch.setattr(fake_tradingnode, "run_async", two_requests)
    await node.start()
    return node, fetch, alerts


@pytest.mark.parametrize("halt", ["error", None])
async def test_halted_error_shutdown_checkpoint_keeps_the_polled_watch(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    halt: str | None,
) -> None:
    """gzf8x R1-R3: a session that stops while halted-ERROR saves the POLLED watch.

    The shutdown checkpoint must carry the cursors the watcher reached, not the
    pre-poll copy restored at init, so the next start sends no duplicate alert.
    """
    node_status: dict[str, Any] = {
        "account": account_id(_config(seeded).binance),
        "risk": _risk(high_water_mark="10000"),
        "liquidation_watch": {"cursors": {"BTCUSDT": _HALT_STAMP_MS}, "seen": {}},
    }
    if halt is not None:
        node_status.update(
            {"halt_reason": halt, "state": "halted", "error_message": "boom"}
        )
    _write_checkpoint(
        seeded, trader_id=_TRADER_ID, stamp=_HALT_STAMP, node_status=node_status
    )

    first, _, first_alerts = await _started_session(
        monkeypatch, seeded, fake_tradingnode, "l1"
    )
    exchange = [call for call in first_alerts.calls if call[2] and "EXCHANGE" in call[1]]
    assert len(exchange) == 1, "run1 reports the liquidation it polled"
    assert first.status.state is NodeState.STOPPED
    assert first.status.halt_reason is (HaltReason.ERROR if halt is not None else None)
    assert _newest_checkpoint_cursors(seeded) == {
        "BTCUSDT": _START_MS,
        "ETHUSDT": _START_MS,
    }, "the final row carries the polled cursors, not the pre-poll copy"

    _, second_fetch, second_alerts = await _started_session(
        monkeypatch, seeded, fake_tradingnode, "l2"
    )
    assert [
        call for call in second_alerts.calls if call[2] and "EXCHANGE" in call[1]
    ] == [], "no duplicate alert on the next start"
    assert second_fetch.requests[0] == (
        "BTCUSDT",
        _START_MS - _OVERLAP_MS,
        _START_MS,
    ), "run2 starts from run1's polled cursor (minus the overlap)"


# ------------------------------------------------------------------------ R52g


async def test_watcher_build_failure_reports_one_risk_check_and_one_critical_alert(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R52g: a dead detector is loud (RISK_CHECK + ONE critical alert); the node survives."""
    harness = await _harness(monkeypatch, seeded)
    node = harness.node

    def broken(api_key: str, api_secret: str, account_type: str, testnet: bool) -> Any:
        raise ImportError(f"cannot import signer for key={api_key} secret={api_secret}")

    monkeypatch.setattr(node_module, "build_binance_fetch", broken)
    node._start_liquidation_task()
    task = node._liquidation_task
    assert task is not None
    await asyncio.wait_for(task, 5)

    assert task.exception() is None, "the failure never propagates out of the task"
    failed = harness.actions("liquidation_watch_failed")
    assert len(failed) == 1
    assert failed[0]["passed"] is False
    assert "ImportError" in failed[0]["message"]
    assert harness.alerts.calls == [("error", harness.alerts.calls[0][1], True)], (
        "exactly one alert, and it is CRITICAL"
    )
    assert "ImportError" in harness.alerts.calls[0][1]
    for text in (failed[0]["message"], harness.alerts.calls[0][1]):
        assert "secret" not in text and "key=" not in text, f"credentials leaked: {text}"
    assert node._liquidation_watcher is None
    assert node.status.state.value != "error", "report only: the node keeps running"
    await node._shutdown()


async def test_watcher_run_failure_reports_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R52g: a failure after the build (inside ``run``) is reported identically."""
    harness = await _harness(monkeypatch, seeded)
    node = harness.node

    async def broken_run(self: Any, stop: asyncio.Event) -> None:
        raise RuntimeError("loop exploded")

    monkeypatch.setattr(node_module.LiquidationWatcher, "run", broken_run)
    node._start_liquidation_task()
    task = node._liquidation_task
    assert task is not None
    await asyncio.wait_for(task, 5)

    assert [a["message"] for a in harness.actions("liquidation_watch_failed")] == [
        "liquidation watch failed: RuntimeError"
    ]
    assert [(c[0], c[2]) for c in harness.alerts.calls] == [("error", True)]
    await node._shutdown()


async def test_shutdown_alone_stops_the_watch_task(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R53: ``_shutdown`` itself stops the task, not only ``_run_loop``'s finally."""
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [_order()]})
    node = harness.node
    node._start_liquidation_task()
    task = node._liquidation_task
    assert task is not None
    await _until(lambda: len(harness.alerts.calls) == 1)

    await asyncio.wait_for(node._shutdown(), 5)

    assert task.done()
    assert node._liquidation_task is None


# ------------------------------------------------------------------------ R52b


async def test_key_is_delivered_once_per_process_and_again_after_a_restart(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R52b: at most once per process, at least once across a restart."""
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [_order()]})
    watcher = harness.node._build_liquidation_watcher()
    await watcher.poll_once()
    await watcher.poll_once()
    assert len(harness.events(EventType.LIQUIDATION)) == 1, "once within the process"
    assert len(harness.alerts.calls) == 1
    await harness.node._shutdown()

    # The checkpoint predates the row and does not name it in ``seen``: a restart after a
    # crash between the write and that checkpoint delivers it again (at-least-once).
    _write_checkpoint(
        seeded,
        trader_id=_RESTART_ID,
        stamp=_CHECKPOINT_STAMP,
        node_status={"liquidation_watch": {"cursors": {"BTCUSDT": _CHECKPOINT_MS}, "seen": {}}},
    )
    restarted = await _harness(
        monkeypatch, seeded, orders={"BTCUSDT": [_order()]}, trader_id=_RESTART_ID,
        logs_dir="logs2",
    )
    restarted_watcher = restarted.node._build_liquidation_watcher()
    await restarted_watcher.poll_once()

    assert len(restarted.events(EventType.LIQUIDATION)) == 1, "re-alerted after the restart"
    assert len(restarted.alerts.calls) == 1
    await restarted.node._shutdown()


# -------------------------------------------------------------------------- R52


async def test_degraded_then_recovered_report_risk_checks_and_non_critical_alerts(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R52: the third consecutive failure warns once; the next success reports recovery."""
    harness = await _harness(monkeypatch, seeded, errors={"BTCUSDT": ConnectionError("boom")})
    trace, _error, _restore = _trace_side_effects(monkeypatch, harness)
    watcher = harness.node._build_liquidation_watcher()

    for _ in range(3):
        await watcher.poll_once()
        harness.clock.advance(700)  # past the capped per-symbol backoff

    assert harness.actions("liquidation_watch_degraded") == [
        {
            "action": "liquidation_watch_degraded",
            "message": "liquidation detector degraded: BTCUSDT ConnectionError",
            "passed": False,
        }
    ], "one degraded warning, at exactly three failures"
    assert harness.alerts.calls == [
        ("error", "WARNING: liquidation detector degraded: BTCUSDT ConnectionError", False)
    ], "a degraded detector is not a critical alert"
    assert trace.order == ["write", "alert"], (
        "SF1: on the success path the degraded warning is written before it is sent"
    )

    harness.fetch.errors.clear()
    await watcher.poll_once()

    assert harness.actions("liquidation_watch_recovered") == [
        {
            "action": "liquidation_watch_recovered",
            "symbol": "BTCUSDT",
            "failures": 3,
            "passed": True,
        }
    ]
    assert harness.alerts.calls[-1] == (
        "error",
        "liquidation detector recovered: BTCUSDT after 3 failures",
        False,
    ), "recovery is not a critical alert either"
    assert trace.order == ["write", "alert", "write", "alert"], (
        "SF1: the recovery write also precedes its alert"
    )
    await harness.node._shutdown()


# -------------------------------------------------------------------------- R53


async def test_shutdown_stops_and_cancels_the_watch_task(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R53: ``_stop_liquidation_loop`` sets the stop event, cancels and never hangs."""
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [_order()]})
    node = harness.node
    node._start_liquidation_task()
    task = node._liquidation_task
    assert task is not None
    await _until(lambda: len(harness.alerts.calls) == 1)

    await asyncio.wait_for(node._stop_liquidation_loop(), 5)

    assert node._liquidation_stop.is_set(), "shutdown sets the stop event"
    assert node._liquidation_task is None
    assert task.done(), "the watch task is not left running"
    assert task.cancelled(), "shutdown cancels it (CancelledError is suppressed)"
    await node._shutdown()


async def test_run_loop_starts_the_watch_task_beside_the_control_loop(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R47/R53: ``_run_loop`` starts the watch task and its shutdown cancels it."""
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_now", _Clock())
    monkeypatch.setattr(node, "_send_telegram_alert", _Alerts())
    started = asyncio.Event()
    cancelled: list[bool] = []

    async def recorder() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(node, "_liquidation_loop", recorder)
    await node._initialize()
    try:
        await node._run_loop()
        assert started.is_set(), "the watch task starts beside the control loop"
        assert cancelled == [True], "shutting the loop down cancels the watch task"
        assert node._liquidation_task is None
    finally:
        await node._shutdown()


# -------------------------------------------------- R1 (irdig) failing event write


def _trace_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    harness: _Harness,
    *,
    fail_write: bool = False,
) -> tuple[_Trace, OSError | None, Callable[[], None]]:
    """Record every event write and alert in ONE ordered trace (SF1 / R1 irdig).

    Both hooks append to ``trace.order``, so the write-before-alert contract is
    asserted as an order, not as two counts. ``fail_write=True`` makes the writer
    raise after recording the "write" (a persistent disk-full), tracing the raising
    path too: the alert must still follow the attempted write.

    Returns the trace, the error every write raises (``None`` when writes succeed),
    and a ``restore`` that puts the real writer and alert spy back, so a test can pin
    the R47b retry (and shut the node down).
    """
    writer = harness.node._event_writer
    assert writer is not None
    original_write = writer.write
    error = OSError("disk full") if fail_write else None
    trace = _Trace(harness.alerts)

    def write(event: Event) -> None:
        trace.write(event)
        if error is not None:
            raise error
        original_write(event)

    monkeypatch.setattr(writer, "write", write)
    monkeypatch.setattr(harness.node, "_send_telegram_alert", trace.alert)

    def restore() -> None:
        monkeypatch.setattr(writer, "write", original_write)
        monkeypatch.setattr(harness.node, "_send_telegram_alert", harness.alerts)

    return trace, error, restore


async def test_exchange_liquidation_traces_write_before_alert(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """SF1: on the success path the write is recorded before the alert, in one trace."""
    harness = await _harness(monkeypatch, seeded)
    trace, _error, _restore = _trace_side_effects(monkeypatch, harness)

    await harness.node._on_exchange_liquidation(_order())

    assert trace.order == ["write", "alert"], (
        "the helper must write the event before it sends the alert"
    )
    await harness.node._shutdown()


async def test_liquidation_alert_survives_a_failing_write_and_the_row_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R1: a failing write still alerts and re-raises; R47b retry is unchanged."""
    order = _order()
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [order]})
    node = harness.node
    trace, error, restore = _trace_side_effects(monkeypatch, harness, fail_write=True)
    assert error is not None
    watcher = node._build_liquidation_watcher()

    with pytest.raises(OSError) as caught:
        await node._on_exchange_liquidation(order)

    assert caught.value is error, "the write's own exception object is re-raised"
    assert trace.order == ["write", "alert"], (
        "the attempted write is recorded before the alert, even though it raised"
    )
    assert harness.alerts.calls == [
        (
            "error",
            f"EXCHANGE {order.auto_close_type} {order.symbol} {order.side} "
            f"qty {order.executed_qty} @ {order.avg_price} (order {order.order_id}); "
            "paper position state may be stale; NO paper action taken",
            True,
        )
    ], "the CRITICAL alert is sent with the unchanged text despite the failed write"

    # R47b: the raise fails that symbol's cycle, so the cursor stays put ...
    await watcher.poll_once()
    assert harness.events(EventType.LIQUIDATION) == [], "nothing was recorded"
    assert watcher.state()["cursors"]["BTCUSDT"] == _START_MS, "R47b: cursor unchanged"
    assert harness.alerts.calls == [harness.alerts.calls[0]] * 2, (
        "the retry alerts again: a duplicate alert is accepted, a lost one is not"
    )
    assert trace.order == ["write", "alert"] * 2, "the retry writes before it alerts too"

    # ... and the row is delivered once the writer works again.
    restore()
    harness.clock.advance(61)  # exactly the first backoff (60 s) plus a second
    await watcher.poll_once()
    assert len(harness.events(EventType.LIQUIDATION)) == 1, "delivered on retry"
    assert len(harness.alerts.calls) == 3
    # The trace only covers the two raising attempts above: ``restore`` put the real
    # writer and alert spy back, so the successful retry is recorded by them, not here.
    assert trace.order == ["write", "alert"] * 2
    await node._shutdown()


async def test_degraded_alert_survives_a_failing_event_write(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R1: the degraded warning is not critical, and a failing write cannot drop it."""
    harness = await _harness(monkeypatch, seeded)
    trace, error, restore = _trace_side_effects(monkeypatch, harness, fail_write=True)
    assert error is not None

    with pytest.raises(OSError) as caught:
        await harness.node._on_liquidation_warning(
            "liquidation detector degraded: BTCUSDT ConnectionError"
        )

    assert caught.value is error
    assert harness.alerts.calls == [
        ("error", "WARNING: liquidation detector degraded: BTCUSDT ConnectionError", False)
    ]
    assert trace.order == ["write", "alert"], (
        "the degraded warning is written first, then sent, even when the write raised"
    )
    restore()
    await harness.node._shutdown()


async def test_recovered_alert_survives_a_failing_event_write(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R1: the recovery report is not critical either, and survives a failing write."""
    harness = await _harness(monkeypatch, seeded)
    trace, error, restore = _trace_side_effects(monkeypatch, harness, fail_write=True)
    assert error is not None

    with pytest.raises(OSError) as caught:
        await harness.node._on_liquidation_recovered("BTCUSDT", 3)

    assert caught.value is error
    assert harness.alerts.calls == [
        ("error", "liquidation detector recovered: BTCUSDT after 3 failures", False)
    ], "a recovery is a non-critical alert"
    assert trace.order == ["write", "alert"], "the recovery is written before it is sent"
    restore()
    await harness.node._shutdown()


async def test_watch_failure_alert_survives_a_failing_event_write(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R1: the task's terminal handler never raises, even when the write does."""
    harness = await _harness(monkeypatch, seeded)
    node = harness.node
    trace, _error, restore = _trace_side_effects(monkeypatch, harness, fail_write=True)

    def broken(api_key: str, api_secret: str, account_type: str, testnet: bool) -> Any:
        raise ImportError("cannot import signer")

    monkeypatch.setattr(node_module, "build_binance_fetch", broken)
    node._start_liquidation_task()
    task = node._liquidation_task
    assert task is not None
    await asyncio.wait_for(task, 5)

    assert task.exception() is None, "a failing write never escapes the watch task"
    assert harness.actions("liquidation_watch_failed") == [], "that write failed"
    assert harness.alerts.calls == [
        (
            "error",
            "liquidation watch failed: ImportError; "
            "exchange liquidations are NOT being reported",
            True,
        )
    ], "the CRITICAL alert is sent even though the RISK_CHECK could not be written"
    assert trace.order == ["write", "alert"], (
        "the terminal handler attempts the write before the critical alert"
    )
    restore()
    await node._shutdown()


_LEAK = "signature=ZZLEAK9 apiKey=QQLEAK7"


async def test_failing_write_log_carries_the_class_only_not_the_fetch_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R1b: a write failing inside on_warning while a fetch error is active never prints it."""
    harness = await _harness(monkeypatch, seeded)
    _trace_side_effects(monkeypatch, harness, fail_write=True)

    with pytest.raises(OSError):  # noqa: PT012
        try:
            raise RuntimeError(f"GET /fapi failed {_LEAK}")
        except RuntimeError:
            await harness.node._on_liquidation_warning("liquidation detector degraded: BTCUSDT")

    assert "ZZLEAK9" not in caplog.text
    assert "QQLEAK7" not in caplog.text
    assert "Traceback" not in caplog.text
    assert "OSError" in caplog.text, "the write failure is still logged, by class"
    assert len(harness.alerts.calls) == 1, "the alert still goes out"


async def test_watch_failed_write_failure_log_carries_the_class_only(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R1b: the terminal watch-failed handler logs a failing write by class, no traceback."""
    harness = await _harness(monkeypatch, seeded)
    node = harness.node
    writer = node._event_writer
    assert writer is not None

    def write(event: Event) -> None:
        raise OSError(f"disk {_LEAK}")

    monkeypatch.setattr(writer, "write", write)

    def broken(api_key: str, api_secret: str, account_type: str, testnet: bool) -> Any:
        raise ImportError("cannot import signer")

    monkeypatch.setattr(node_module, "build_binance_fetch", broken)
    node._start_liquidation_task()
    task = node._liquidation_task
    assert task is not None
    await asyncio.wait_for(task, 5)

    assert "ZZLEAK9" not in caplog.text
    assert "QQLEAK7" not in caplog.text
    assert "paper event write failed" in caplog.text
    assert "OSError" in caplog.text
    assert len(harness.alerts.calls) == 1
    monkeypatch.undo()
    await node._shutdown()
