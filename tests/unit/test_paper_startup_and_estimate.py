"""Paper startup-connect timeout and estimated liquidation price (T6 / P2b).

Bead vibe-quant-2gyk1.1.11. Contract rows (design r2, job 20261015-honest-safe):

* R42  a refused margin change never starts a strategy. Pinned on a
       real NT ``TradingNode`` in ``test_paper_margin_refusal_real_node.py``, not here:
       real NT LOGS the failed connect and idles, it never raises.
* R42b more than 120 s with ``_connected is None``: ONCE a RISK_CHECK
       ``{action: startup_connect_timeout, timeout_s: 120, margin_type, passed: False}``,
       exactly one critical Telegram (bypassing the rate limit), and
       ``status.error_message`` set. The timeout is measured from
       ``status.started_at``. A later connect sends one NON-critical
       "venue connected after startup timeout" (chief ruling R42c: a RISK_CHECK
       ``{action: venue_connected_after_startup_timeout, passed: true}`` plus a
       non-critical Telegram; no SIGNAL event).
* R44  an ISOLATED position open reports RISK_CHECK ``liquidation_estimate`` with
       ``est_liquidation_price`` from the shared ``liquidation.isolated_liq_price``
       rule (never re-derived here) and its ``isolated_margin`` (chief ruling R44b:
       ``basis`` names the method as a string). Report only: no orders, no halt.
* R45  CROSSED reports a null price with the wallet reason; tier errors report null
       with ``str(exc)``; the report never raises.
* R45b (chief ruling) the WHOLE estimate is guarded: any exception (OSError, a bare
       ValueError, KeyError) is logged and reported as null + reason, the position
       open is still logged and ``guard.evaluate_risk`` still runs.
* R42d (chief ruling) recovery clears ``error_message`` only if it is still the
       timeout message; the timeout never overwrites an existing (halt) message.
* R42e (chief ruling) the connect probe runs first; a first tick that sees the venue
       connected after > 120 s sends neither the timeout alert nor the recovery.
* R11 (s3pqx) a connect-probe failure logs ``RuntimeError`` by class only: no exception
       text, no traceback, and no ``__context__`` chain (a raw fetch error's signed URL /
       api key would otherwise print). The startup-timeout path still runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.model.identifiers import InstrumentId

import vibe_quant.liquidation.tiers as tiers_module
import vibe_quant.paper.guard as guard_module
from tests.unit.test_audit_paper_node import DSL, _config
from vibe_quant.db.state_manager import StateManager
from vibe_quant.liquidation.tiers import TierTableExceeded, UnknownSymbol, isolated_liq_price
from vibe_quant.liquidation.tiers import load_tier_tables as _load_tier_tables
from vibe_quant.logging.events import EventType
from vibe_quant.paper.node import NodeState, PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

_TRADER_ID = "PAPER-007"
#: Leak marker: contains none of the redacted credential names, so a mutant that
#: drops a name cannot hide it (impl-sb-watchlog-2).
_LEAK = "ZZLEAK9"
#: R42b fixes the threshold at 120 s; the payload carries it too.
_TIMEOUT_S = 120
#: R44b: the estimate's basis is this string, with the tier-table version spliced in.
_BASIS = (
    "isolated formula at entry: entry_price, leverage, Binance tier MMR (table {version}); "
    "excludes fees, funding and later margin changes"
)
#: R45: a crossed position's LP depends on the whole account, so it is not estimated.
_CROSS_REASON = "cross LP depends on the account wallet and other positions; not estimated"

#: An isolated 10x validated run, and the probe position (BTC 0.01 @ 60000).
_ISOLATED_10X = {"liquidation_mode": "isolated", "leverage": 10}


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


def _timeout_message() -> str:
    return (
        f"paper {_TRADER_ID} not connected after {_TIMEOUT_S} s; venue setup refused "
        "(e.g. margin type change with open orders or position); strategy NOT trading"
    )


def _recovery_message() -> str:
    return f"paper {_TRADER_ID} venue connected after startup timeout"


def _basis() -> str:
    return _BASIS.format(version=_load_tier_tables().version)


# ----------------------------------------------------------------- fake NT node


class _Engine:
    """One data/exec engine: ``check_connected`` answers (or raises) as scripted."""

    def __init__(self) -> None:
        self.connected = False
        self.error: Exception | None = None

    def check_connected(self) -> bool:
        if self.error is not None:
            raise self.error
        return self.connected


class _Kernel:
    class _RiskEngine:
        def execute(self, command: object) -> None:
            pass

    def __init__(self) -> None:
        self.risk_engine = self._RiskEngine()
        self.data_engine = _Engine()
        self.exec_engine = _Engine()


class _Trader:
    def __init__(self) -> None:
        self.actors: list[object] = []
        self.strategies: list[object] = []
        self.start_calls = 0

    def add_actor(self, actor: object) -> None:
        self.actors.append(actor)

    def add_strategy(self, strategy: object) -> None:
        self.strategies.append(strategy)

    def start(self) -> None:
        self.start_calls += 1


class _FakeTradingNode:
    """Stand-in for ``nautilus_trader.live.node.TradingNode``.

    Connecting is not the paper node's job - this only pins the ordering. R42 (a refused
    margin change never starts a strategy) is pinned on the real NT ``TradingNode`` in
    ``test_paper_margin_refusal_real_node.py``.
    """

    instances: list[_FakeTradingNode] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.kernel = _Kernel()
        self.trader = _Trader()
        self.built = False
        _FakeTradingNode.instances.append(self)

    def add_data_client_factory(self, name: str, factory: object) -> None:
        pass

    def add_exec_client_factory(self, name: str, factory: object) -> None:
        pass

    def build(self) -> None:
        self.built = True

    async def run_async(self) -> None:
        self.trader.start()

    async def stop_async(self) -> None:
        return


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


class _Alerts:
    """Records every ``PaperTradingNode._send_telegram_alert`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, kind: str, message: str, *, critical: bool = False) -> None:
        self.calls.append((kind, message, critical))


class _Telegram:
    """TelegramBot stand-in recording the ``bypass_rate_limit`` flag."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        self.calls.append(("error", message, bypass_rate_limit))
        return True

    async def send_circuit_breaker(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        self.calls.append(("circuit_breaker", message, bypass_rate_limit))
        return True


class _Clock:
    """Injected clock (R42b: the timeout is measured from ``status.started_at``)."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


# ------------------------------------------------------------------- helpers


def _validation_run(seeded: dict[str, Any], parameters: dict[str, Any]) -> int:
    sm = StateManager(seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seeded["sid"],
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters=parameters,
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid


async def _initialized(seeded: dict[str, Any], **kw: Any) -> PaperTradingNode:
    node = PaperTradingNode(_config(seeded, **kw))
    await node._initialize()
    return node


def _log(node: PaperTradingNode, seeded: dict[str, Any]) -> list[dict[str, Any]]:
    assert node._event_writer is not None
    node._event_log.barrier().result(10)  # lines are appended on the event worker
    node._event_writer.flush()
    path = seeded["tmp"] / "logs" / f"{_TRADER_ID}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _actions(node: PaperTradingNode, seeded: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [e["data"] for e in _log(node, seeded) if e["data"].get("action") == action]


def _orders(node: PaperTradingNode, seeded: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in _log(node, seeded) if e["event"] == EventType.ORDER.value]


def _position_event(
    side: PositionSide = PositionSide.LONG,
    *,
    symbol: str = "BTCUSDT",
    quantity: float = 0.01,
    entry_price: float = 60000.0,
    position_id: str = "P-1",
) -> Any:
    return SimpleNamespace(
        position_id=position_id,
        instrument_id=InstrumentId.from_str(f"{symbol}-PERP.BINANCE"),
        side=side,
        quantity=quantity,
        avg_px_open=entry_price,
        ts_event=1_713_365_200_000_000_000,
    )


def _estimate_payload(node: PaperTradingNode, seeded: dict[str, Any]) -> dict[str, Any]:
    payloads = _actions(node, seeded, "liquidation_estimate")
    assert len(payloads) == 1, f"expected exactly one estimate, got {payloads}"
    return payloads[0]


# ------------------------------------------------------------------------ R42b


async def test_startup_connect_timeout_alerts_once_critical(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b: >120 s with ``_connected is None`` -> one RISK_CHECK + one critical alert."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    try:
        node._check_connectivity()

        assert _actions(node, seeded, "startup_connect_timeout") == [
            {
                "action": "startup_connect_timeout",
                "timeout_s": _TIMEOUT_S,
                "margin_type": "CROSSED",
                "passed": False,
            }
        ]
        assert alerts.calls == [("error", _timeout_message(), True)]
        assert node.status.error_message == _timeout_message()

        node._check_connectivity()  # the next control tick must not duplicate it
        assert len(_actions(node, seeded, "startup_connect_timeout")) == 1
        assert alerts.calls == [("error", _timeout_message(), True)]

        # report only: no strategy was started and no order exists
        assert fake_tradingnode.instances[0].trader.start_calls == 0
        assert _orders(node, seeded) == []
    finally:
        await node._shutdown()


async def test_startup_connect_timeout_is_silent_at_exactly_120s(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b edge: exactly 120 s is silent, one microsecond more fires (``<`` vs ``<=``)."""
    node = await _initialized(seeded)
    clock = _Clock(datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC))
    monkeypatch.setattr(node, "_now", clock)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = clock.now
    try:
        clock.now += timedelta(seconds=_TIMEOUT_S)
        node._check_connectivity()
        assert _actions(node, seeded, "startup_connect_timeout") == []
        assert alerts.calls == []
        assert node.status.error_message is None

        clock.now += timedelta(microseconds=1)
        node._check_connectivity()
        assert len(_actions(node, seeded, "startup_connect_timeout")) == 1
        assert alerts.calls == [("error", _timeout_message(), True)]
    finally:
        await node._shutdown()


async def test_startup_connect_timeout_measures_from_status_started_at(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b: the elapsed time is measured from ``status.started_at``, not from the check."""
    node = await _initialized(seeded)
    clock = _Clock(datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC))
    monkeypatch.setattr(node, "_now", clock)
    monkeypatch.setattr(node, "_send_telegram_alert", _Alerts())
    try:
        # The node has been up for 10 minutes, but the current start is 119 s old.
        node.status.started_at = clock.now - timedelta(seconds=_TIMEOUT_S - 1)
        node._check_connectivity()
        assert _actions(node, seeded, "startup_connect_timeout") == []

        clock.now += timedelta(seconds=2)
        assert node._connected is None
        node._check_connectivity()
        assert len(_actions(node, seeded, "startup_connect_timeout")) == 1
    finally:
        await node._shutdown()


async def test_later_connect_after_startup_timeout_recovers_once_non_critical(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b/R42c: a later connect reports once, non-critically, and then stays quiet."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    try:
        node._check_connectivity()
        assert len(_actions(node, seeded, "startup_connect_timeout")) == 1
        alerts.calls.clear()

        kernel = fake_tradingnode.instances[0].kernel
        kernel.data_engine.connected = True
        kernel.exec_engine.connected = True
        node._check_connectivity()

        assert _actions(node, seeded, "venue_connected_after_startup_timeout") == [
            {"action": "venue_connected_after_startup_timeout", "passed": True}
        ]
        assert alerts.calls == [("error", _recovery_message(), False)]
        assert node._connected is True

        node._check_connectivity()
        assert len(_actions(node, seeded, "venue_connected_after_startup_timeout")) == 1
        assert alerts.calls == [("error", _recovery_message(), False)]
    finally:
        await node._shutdown()


async def test_startup_timeout_alert_bypasses_the_rate_limit(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b: the critical flag reaches ``send_error(bypass_rate_limit=True)``."""
    node = await _initialized(seeded)
    telegram = _Telegram()
    node._telegram = telegram
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    try:
        node._check_connectivity()
        for _ in range(5):
            await asyncio.sleep(0)
        assert telegram.calls == [("error", _timeout_message(), True)]
    finally:
        await node._shutdown()


# ------------------------------------------------------------------- R44 / R45


async def test_isolated_long_position_open_reports_the_estimated_liquidation_price(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """R44/R44b: BTC 0.01 @ 60000, 10x isolated -> long 54216.867469879515, margin 60."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    try:
        node._on_position_opened(_position_event(PositionSide.LONG))

        assert _estimate_payload(node, seeded) == {
            "action": "liquidation_estimate",
            "position_id": "P-1",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.01,
            "entry_price": 60000.0,
            "leverage": 10,
            "margin_type": "ISOLATED",
            "isolated_margin": 60.0,
            "est_liquidation_price": 54216.867469879515,
            "tier_table_version": _load_tier_tables().version,
            "basis": _basis(),
            "passed": True,
        }
        # report only: no order, no halt
        assert _orders(node, seeded) == []
        assert node.status.state is not NodeState.HALTED
    finally:
        await node._shutdown()


async def test_isolated_short_position_open_reports_the_estimated_liquidation_price(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """R44: the same probe short -> 65737.05179282869 (the sign comes from the position)."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    try:
        node._on_position_opened(_position_event(PositionSide.SHORT))

        payload = _estimate_payload(node, seeded)
        assert payload["side"] == "SHORT"
        assert payload["est_liquidation_price"] == 65737.05179282869
        assert payload["isolated_margin"] == 60.0
        assert payload["margin_type"] == "ISOLATED"
        assert payload["basis"] == _basis()
        assert _orders(node, seeded) == []
    finally:
        await node._shutdown()


async def test_crossed_position_open_reports_null_with_a_reason(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """R45: a CROSSED position reports a null price, that reason, and never raises."""
    node = await _initialized(seeded)  # liquidation off -> CROSSED
    try:
        node._on_position_opened(_position_event(PositionSide.LONG))

        assert _estimate_payload(node, seeded) == {
            "action": "liquidation_estimate",
            "position_id": "P-1",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.01,
            "entry_price": 60000.0,
            "leverage": 5,
            "margin_type": "CROSSED",
            "isolated_margin": None,
            "est_liquidation_price": None,
            "tier_table_version": None,
            "basis": None,
            "reason": _CROSS_REASON,
            "passed": True,
        }
        assert node.status.state is not NodeState.HALTED
    finally:
        await node._shutdown()


async def test_unknown_symbol_reports_null_with_the_exception_text(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """R45: an unknown symbol gives a null price plus ``str(exc)``, never a raise."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    try:
        node._on_position_opened(_position_event(PositionSide.LONG, symbol="NOPEUSDT"))

        payload = _estimate_payload(node, seeded)
        assert payload["est_liquidation_price"] is None
        assert payload["reason"] == str(UnknownSymbol("NOPEUSDT"))
        assert payload["symbol"] == "NOPEUSDT"
        assert payload["isolated_margin"] == 60.0
        assert payload["tier_table_version"] == _load_tier_tables().version
        assert payload["basis"] is None
        assert payload["passed"] is True
        assert _orders(node, seeded) == []
    finally:
        await node._shutdown()


async def test_tier_table_exceeded_reports_null_with_the_exception_text(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """R45: a notional beyond the last tier cap is reported, not traded or raised."""
    quantity, entry_price = 30000.0, 60000.0
    btc = _load_tier_tables()["BTCUSDT"]
    with pytest.raises(TierTableExceeded) as excinfo:
        isolated_liq_price(1, quantity, entry_price, quantity * entry_price / 10.0, btc)

    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    try:
        node._on_position_opened(
            _position_event(PositionSide.LONG, quantity=quantity, entry_price=entry_price)
        )

        payload = _estimate_payload(node, seeded)
        assert payload["est_liquidation_price"] is None
        assert payload["reason"] == str(excinfo.value)
        assert "last cap" in payload["reason"]
        assert payload["passed"] is True
        assert node.status.state is not NodeState.HALTED
        assert _orders(node, seeded) == []
    finally:
        await node._shutdown()


# ------------------------------------------------------------------------ R42d/e


def _connect_venue(fake_tradingnode: type[_FakeTradingNode], *, connected: bool = True) -> None:
    kernel = fake_tradingnode.instances[0].kernel
    kernel.data_engine.connected = connected
    kernel.exec_engine.connected = connected


async def test_first_connect_before_timeout_sends_no_recovery_message(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42c/S3 (M1): an ordinary first connect says nothing; only a late one recovers."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=5)
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()

        assert node._connected is True
        assert _actions(node, seeded, "venue_connected_after_startup_timeout") == []
        assert _actions(node, seeded, "startup_connect_timeout") == []
        assert alerts.calls == []
    finally:
        await node._shutdown()


async def test_isolated_node_timeout_payload_carries_the_isolated_margin_type(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b/S3 (M2): the timeout payload reports the node's resolved margin type."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    monkeypatch.setattr(node, "_send_telegram_alert", _Alerts())
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    try:
        node._check_connectivity()

        assert _actions(node, seeded, "startup_connect_timeout") == [
            {
                "action": "startup_connect_timeout",
                "timeout_s": _TIMEOUT_S,
                "margin_type": "ISOLATED",
                "passed": False,
            }
        ]
    finally:
        await node._shutdown()


async def test_connected_node_is_silent_after_the_timeout_window_elapses(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b/S3 (M3): connected before 120 s, then 200 s elapse -> no event, no alert."""
    node = await _initialized(seeded)
    clock = _Clock(datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC))
    monkeypatch.setattr(node, "_now", clock)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = clock.now
    try:
        clock.now += timedelta(seconds=10)
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        assert node._connected is True

        clock.now += timedelta(seconds=200)
        node._check_connectivity()

        assert _actions(node, seeded, "startup_connect_timeout") == []
        assert _actions(node, seeded, "venue_connected_after_startup_timeout") == []
        assert alerts.calls == []
        assert node.status.error_message is None
    finally:
        await node._shutdown()


async def test_first_tick_seeing_a_connected_venue_after_120s_sends_no_alert(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42e: the probe runs first; a stalled loop that finds the venue connected is silent."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=200)
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()

        assert node._connected is True
        assert _actions(node, seeded, "startup_connect_timeout") == []
        assert _actions(node, seeded, "venue_connected_after_startup_timeout") == []
        assert alerts.calls == []
        assert node.status.error_message is None
    finally:
        await node._shutdown()


async def test_failing_connect_probe_after_120s_still_reports_the_timeout(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42e: a probe that raises counts as not connected (a stuck connect breaks it too)."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    fake_tradingnode.instances[0].kernel.data_engine.error = RuntimeError("probe broke")
    try:
        node._check_connectivity()

        assert len(_actions(node, seeded, "startup_connect_timeout")) == 1
        assert alerts.calls == [("error", _timeout_message(), True)]
    finally:
        await node._shutdown()


def _chained_probe_error() -> RuntimeError:
    """A ``RuntimeError`` raised while another ``RuntimeError`` is active (R11).

    Both carry the leak marker, so any ``logger.exception`` in the probe's handler
    prints it twice through the ``__context__`` chain.
    """
    try:
        try:
            raise RuntimeError(f"fetch failed {_LEAK}")
        except RuntimeError:
            raise RuntimeError(f"probe broke {_LEAK}")  # noqa: B904  # keep __context__
    except RuntimeError as err:
        return err


async def test_failing_connect_probe_log_is_class_only(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R11: the probe-failure log names the class only, with no chain text or traceback."""
    node = await _initialized(seeded)
    monkeypatch.setattr(node, "_send_telegram_alert", _Alerts())
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    error = _chained_probe_error()
    assert error.__context__ is not None
    fake_tradingnode.instances[0].kernel.data_engine.error = error
    try:
        with caplog.at_level(logging.ERROR, logger="vibe_quant.paper.node"):
            node._check_connectivity()

        records = [r for r in caplog.records if r.name == "vibe_quant.paper.node"]
        assert any("RuntimeError" in r.getMessage() for r in records)
        assert all(_LEAK not in r.getMessage() for r in records)
        assert all(r.exc_info is None and not r.exc_text for r in records)
        assert _LEAK not in caplog.text, "the __context__ chain must not print"
        assert "Traceback" not in caplog.text
        # R42e: a probe that raises is still "not connected", so the timeout fires.
        assert len(_actions(node, seeded, "startup_connect_timeout")) == 1
    finally:
        await node._shutdown()


async def test_late_connect_clears_the_timeout_error_message(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42d: after a late connect the "strategy NOT trading" message is gone."""
    node = await _initialized(seeded)
    monkeypatch.setattr(node, "_send_telegram_alert", _Alerts())
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    try:
        node._check_connectivity()
        assert node.status.error_message == _timeout_message()

        _connect_venue(fake_tradingnode)
        node._check_connectivity()

        assert node._connected is True
        assert node.status.error_message is None
    finally:
        await node._shutdown()


async def test_late_connect_keeps_a_later_halt_message(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42d: a halt message set after the timeout is never wiped by the recovery."""
    node = await _initialized(seeded)
    monkeypatch.setattr(node, "_send_telegram_alert", _Alerts())
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    try:
        node._check_connectivity()
        node.status.error_message = "halted: max drawdown"

        _connect_venue(fake_tradingnode)
        node._check_connectivity()

        assert len(_actions(node, seeded, "venue_connected_after_startup_timeout")) == 1
        assert node.status.error_message == "halted: max drawdown"
    finally:
        await node._shutdown()


async def test_timeout_does_not_overwrite_an_existing_halt_message(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42d nit: the timeout sets ``error_message`` only when it is empty."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    node.status.error_message = "halted: kill switch"
    try:
        node._check_connectivity()

        assert node.status.error_message == "halted: kill switch"
        assert len(_actions(node, seeded, "startup_connect_timeout")) == 1
        assert alerts.calls == [("error", _timeout_message(), True)]
    finally:
        await node._shutdown()


# ------------------------------------------------------------------------ R45b


class _Boom(Exception):
    """Not a tier-table exception: only the blanket R45b guard can catch it."""


def _drive_through_guard(
    node: PaperTradingNode, monkeypatch: pytest.MonkeyPatch, event: Any
) -> list[str]:
    """Deliver ``event`` through the real guard -> bridge -> node path (guard.py:474)."""
    guard = node._guard
    assert guard is not None
    calls: list[str] = []
    monkeypatch.setattr(guard_module, "PositionOpened", type(event))
    monkeypatch.setattr(guard, "_is_guarded_strategy", lambda _sid: True)
    monkeypatch.setattr(guard, "evaluate_risk", lambda: calls.append("evaluate_risk"))
    guard._on_position_event(event)
    return calls


def _guarded_event(side: PositionSide = PositionSide.LONG, **kw: Any) -> Any:
    class _Opened(SimpleNamespace):
        pass

    event = _position_event(side, **kw)
    return _Opened(**vars(event), strategy_id="S-1")


def _open_logged(node: PaperTradingNode, seeded: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in _log(node, seeded) if e["event"] == EventType.POSITION_OPEN.value]


async def test_oserror_loading_tier_tables_is_reported_not_raised(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R45b: OSError from ``load_tier_tables`` -> null + reason; open logged; risk re-evaluated."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))

    def _raise() -> None:
        raise OSError("tier table file gone")

    monkeypatch.setattr(tiers_module, "load_tier_tables", _raise)
    try:
        calls = _drive_through_guard(node, monkeypatch, _guarded_event())

        assert calls == ["evaluate_risk"]
        assert len(_open_logged(node, seeded)) == 1
        payload = _estimate_payload(node, seeded)
        assert payload["est_liquidation_price"] is None
        assert payload["reason"] == "tier table file gone"
        assert payload["passed"] is True
        assert node.status.state is not NodeState.HALTED
        assert _orders(node, seeded) == []
    finally:
        await node._shutdown()


async def test_bare_valueerror_from_the_liquidation_rule_is_reported_not_raised(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R45b: a bare ValueError from ``isolated_liq_price`` -> null + reason, no raise."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))

    def _raise(*_a: object, **_k: object) -> float:
        raise ValueError("qty must be > 0")

    monkeypatch.setattr(tiers_module, "isolated_liq_price", _raise)
    try:
        calls = _drive_through_guard(node, monkeypatch, _guarded_event())

        assert calls == ["evaluate_risk"]
        assert len(_open_logged(node, seeded)) == 1
        payload = _estimate_payload(node, seeded)
        assert payload["est_liquidation_price"] is None
        assert payload["reason"] == "qty must be > 0"
        assert payload["isolated_margin"] == 60.0
    finally:
        await node._shutdown()


async def test_zero_quantity_position_is_reported_not_raised(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R45b: the real rule's bare ValueError for quantity 0 does not escape."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    try:
        calls = _drive_through_guard(node, monkeypatch, _guarded_event(quantity=0.0))

        assert calls == ["evaluate_risk"]
        payload = _estimate_payload(node, seeded)
        assert payload["est_liquidation_price"] is None
        assert "qty" in str(payload["reason"])
    finally:
        await node._shutdown()


async def test_flat_side_position_is_reported_not_raised(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R45b: a ``FLAT`` side (KeyError in the sign map) does not escape."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    try:
        calls = _drive_through_guard(node, monkeypatch, _guarded_event(PositionSide.FLAT))

        assert calls == ["evaluate_risk"]
        assert len(_open_logged(node, seeded)) == 1
        payload = _estimate_payload(node, seeded)
        assert payload["side"] == "FLAT"
        assert payload["est_liquidation_price"] is None
        assert "FLAT" in str(payload["reason"])
    finally:
        await node._shutdown()


async def test_failing_estimate_event_write_does_not_break_position_handling(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R45b: even the best-effort RISK_CHECK write is guarded."""
    node = await _initialized(seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X))
    real_write = node._journal.record

    def _write(
        event_type: EventType,
        data: dict[str, object],
        *,
        alert: Any = None,
        timestamp: Any = None,
    ) -> Any:
        if event_type is EventType.RISK_CHECK:
            raise _Boom("event log closed")
        return real_write(event_type, data, alert=alert, timestamp=timestamp)

    monkeypatch.setattr(node._journal, "record", _write)
    try:
        calls = _drive_through_guard(node, monkeypatch, _guarded_event())

        assert calls == ["evaluate_risk"]
        assert len(_open_logged(node, seeded)) == 1
    finally:
        await node._shutdown()


async def test_startup_timeout_alert_survives_a_failing_event_write(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1c: a failing RISK_CHECK write cannot silence the only refused-margin signal."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    writer = node._event_writer
    assert writer is not None
    original_write = writer.write

    def write(event: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(writer, "write", write)
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=_TIMEOUT_S + 1)
    try:
        node._check_connectivity()  # the write failure never escapes the probe
        assert alerts.calls == [("error", _timeout_message(), True)]

        node._check_connectivity()  # latch stays set: one report per session
        assert alerts.calls == [("error", _timeout_message(), True)]
        # the failing append runs on the event worker: wait for it before the restore
        node._event_log.barrier().result(10)
        assert node._event_log.health().failed == 1
    finally:
        monkeypatch.setattr(writer, "write", original_write)
        await node._shutdown()
