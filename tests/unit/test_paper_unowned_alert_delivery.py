"""The unowned-position text reaches Telegram exactly once on every listener path.

Kill switch, manual halt, restored halt first tick, pre-start halt on_start report and
close-all, each fresh and right after another ERROR alert (the 60 s ERROR limit), with
a real SessionJournal + TelegramBot (MockTransport) and a real TradingGuard on a
BacktestEngine, through node.halt / node.close_all_positions.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

import tests.unit.test_audit_paper_guard as harness
from tests.unit.test_audit_paper_node import DSL, _config
from tests.unit.test_paper_guard_unowned_report import _add_external, _restored_error
from vibe_quant.alerts.telegram import TelegramBot, TelegramConfig
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper.guard import HaltReason
from vibe_quant.paper.node import PaperTradingNode, _GuardBridge

MARK = "NOT closed"
N_BARS = 30

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
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


def _go(seeded: dict[str, Any], path: str, prior: bool) -> list[str]:
    sent: list[str] = []

    def _rec(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content)["text"])
        return httpx.Response(200, json={"ok": True})

    async def main() -> None:
        bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

        async def _client() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(_rec))

        bot._get_client = _client  # type: ignore[method-assign]
        node = PaperTradingNode(_config(seeded))
        node._journal.attach(None, bot, "s")
        node._journal._write_event = lambda *a, **k: None  # type: ignore[method-assign]
        h = harness.build(
            symbols=("BTCUSDT",), restored_halt=_restored_error() if path == "restore" else None
        )
        node._guard = h.guard
        h.guard._listener = _GuardBridge(node)
        if prior:
            node._on_guard_warning("margin cushion low on ETHUSDT")
            await node._journal.wait_alerts(None)
        acts: list[tuple[Any, Callable[[], None]]] = []
        at = harness.START + timedelta(seconds=30)
        if path == "prestart":
            node.halt(HaltReason.SIGNAL, "halt via SIGUSR1")  # before start: nothing open
            _add_external(h.engine)  # NT reconciles it after the halt
        else:
            _add_external(h.engine)
        if path == "kill":
            acts.append(
                (at, lambda: node.halt(HaltReason.KILL_SWITCH, "system kill switch: probe"))
            )
            # a later, less severe halt must not re-flatten / re-alert
            acts.append((at + timedelta(minutes=3), lambda: node.halt(HaltReason.MANUAL, "again")))
        elif path == "manual":
            acts.append((at, lambda: node.halt(HaltReason.MANUAL, "manual via API")))
            acts.append((at + timedelta(minutes=3), lambda: node.halt(HaltReason.MANUAL, "again")))
        elif path == "close_all":
            acts.append((at, lambda: node.close_all_positions()))
        bars = harness._flat_bars(get_bar_type("BTCUSDT", "1m"), [100.0] * N_BARS)
        harness.run(h, bars, actions=acts)
        await node._journal.wait_alerts(None)
        node._state_manager.close()

    asyncio.run(main())
    return sent


@pytest.mark.parametrize("prior", [False, True], ids=["fresh", "after_error_alert"])
@pytest.mark.parametrize("path", ["kill", "manual", "restore", "prestart", "close_all"])
def test_unowned_delivered_exactly_once(seeded: dict[str, Any], path: str, prior: bool) -> None:
    sent = _go(seeded, path, prior)
    unowned = [t for t in sent if MARK in t]
    print(f"\n[{path}/{'prior' if prior else 'fresh'}] SENT={len(sent)} UNOWNED={len(unowned)}")
    for t in sent:
        print("  MSG:", t.replace("\n", " | ")[:150])
    assert len(unowned) == 1, f"{path}: {len(unowned)} unowned alerts over {N_BARS} bars"
