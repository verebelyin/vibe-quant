"""A new paper session inherits the last session's risk state on the same account.

Contract rows R1-R24 of vibe-quant-3fqw9 (core). The risk bookkeeping (high water
mark, UTC-day baseline, losing streak) and an ERROR halt carry over from the newest
checkpoint of ANOTHER trader on the same venue + testnet/live + API-key fingerprint;
the trader's own checkpoint is never discarded, "start fresh" only skips the
cross-session lookup. Every row runs through the real ``_initialize`` path.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_halt_restore import (
    _EmptyCache,
    _FakeClock,
    _full_risk,
    _SettablePortfolio,
    seeded,  # noqa: F401  (fixture)
)
from vibe_quant.paper.config import BinanceTestnetConfig, account_id
from vibe_quant.paper.guard import (
    RESTORED_HALT_PREFIX,
    GuardState,
    HaltReason,
    RiskState,
    TradingGuard,
)
from vibe_quant.paper.node import NodeState, PaperTradingNode, inherited_halt_message
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from pathlib import Path

ACCOUNT_A = "binance:testnet:891a21f36b950681"
ACCOUNT_B = "binance:testnet:2b3782037b86aff4"
LIMITS_NOTE = "limits re-baselined at current equity"


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


# ------------------------------------------------------------------ helpers


def _risk(**overrides: object) -> dict[str, object]:
    """R4's risk dict: hwm 10600, day today, dse 10400, 3 losses, last 10300."""
    return _full_risk(
        day=datetime.now(UTC).date().isoformat(),
        day_start_equity="10400",
        last_equity="10300",
        **overrides,
    )


def _put(
    db: Path,
    trader_id: str,
    *,
    account: str | None = ACCOUNT_A,
    risk: dict[str, object] | None = None,
    halt: str | None = None,
    message: str | None = None,
    state: str = "stopped",
    extra: dict[str, object] | None = None,
) -> int:
    """Store one checkpoint row (ids ascend in call order); returns its id."""
    status: dict[str, object] = {"state": state, "trader_id": trader_id}
    if account is not None:
        status["account"] = account
    if risk is not None:
        status["risk"] = risk
    if halt is not None:
        status["halt_reason"] = halt
    if message is not None:
        status["error_message"] = message
    status.update(extra or {})
    persistence = StatePersistence(db_path=db, trader_id=trader_id)
    try:
        return persistence.save_checkpoint(StateCheckpoint(trader_id=trader_id, node_status=status))
    finally:
        persistence.close()


def _sql(db: Path, sql: str, *args: object) -> None:
    con = sqlite3.connect(db)
    with con:
        con.execute(sql, args)
    con.close()


def _latest(db: Path, trader_id: str) -> StateCheckpoint:
    persistence = StatePersistence(db_path=db, trader_id=trader_id)
    try:
        latest = persistence.load_latest_checkpoint()
    finally:
        persistence.close()
    assert latest is not None
    return latest


def _node(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    *,
    trader: str = "PAPER-007",
    key: str = "KEY-A",
    testnet: bool = True,
    **kw: Any,
) -> tuple[PaperTradingNode, list[dict[str, object]]]:
    """A node on ``key`` whose event payloads land in a list."""
    events: list[dict[str, object]] = []
    node = PaperTradingNode(
        _config(
            seeded,
            trader_id=trader,
            binance=BinanceTestnetConfig(key, "secret", testnet=testnet),
            **kw,
        )
    )

    def write(event_type: Any, data: dict[str, object], timestamp: Any = None) -> None:
        events.append(data)

    monkeypatch.setattr(node, "_write_event", write)
    return node, events


def _restores(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [e for e in events if e.get("action") == "restore"]


_RESTORE_KEYS = {
    "action",
    "checkpoint_time",
    "prior_state",
    "prior_halt_reason",
    "prior_positions",
    "prior_orders",
    "restored_risk",
    "restored_halt",
    "risk_problems",
    "risk_source_trader_id",
    "risk_source_checkpoint_time",
}


# ----------------------------------------------------------- R1 account id


def test_account_id_vectors() -> None:
    """R1: venue + env + sha256(api_key)[:16]; neither key nor secret appears."""
    testnet = account_id(BinanceTestnetConfig("KEY-A", "SECRET-A", testnet=True))
    live = account_id(BinanceTestnetConfig("KEY-A", "SECRET-A", testnet=False))
    assert testnet == "binance:testnet:891a21f36b950681"
    assert live == "binance:live:891a21f36b950681"
    assert account_id(BinanceTestnetConfig("KEY-B", "S", testnet=True)) == ACCOUNT_B
    for text in (testnet, live):
        assert "KEY-A" not in text
        assert "SECRET-A" not in text


# --------------------------------------------------------- first test (T2)


async def test_new_trader_inherits_last_session_risk_on_same_account(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4: PAPER-007 starts on the account PAPER-041 last used and inherits its risk."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is not None
        restored = node._restored_risk.to_dict()
        assert restored["high_water_mark"] == "10600"
        assert restored["consecutive_losses"] == 3
        assert restored["day_start_equity"] == "10400"
        assert restored["last_equity"] == "10300"
        [event] = _restores(events)
        assert event["risk_source_trader_id"] == "PAPER-041"
        assert event["checkpoint_time"] is None
        assert event["prior_halt_reason"] is None
        assert event["prior_positions"] == []
        assert event["restored_halt"] is None
        assert node._restored_halt is None
        assert node.status.state != NodeState.HALTED
    finally:
        await node._shutdown()


# ---------------------------------------------------- R2 / R3 / R23 / R24


async def test_every_checkpoint_carries_account(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: initial, immediate and final checkpoints all record the account."""
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    assert _latest(seeded["db"], "PAPER-007").node_status["account"] == ACCOUNT_A
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints")
    node._save_checkpoint_now()
    await node._db_worker.run(lambda: None)
    assert _latest(seeded["db"], "PAPER-007").node_status["account"] == ACCOUNT_A
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints")
    await node._shutdown()
    assert _latest(seeded["db"], "PAPER-007").node_status["account"] == ACCOUNT_A
    assert "KEY-A" not in repr(_latest(seeded["db"], "PAPER-007").node_status)


async def test_no_prior_session_starts_fresh_and_warns(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R3: an empty table -> no restore event, one warning, start event audit keys."""
    node, events = _node(monkeypatch, seeded)
    with caplog.at_level(logging.WARNING):
        await node._initialize()
    try:
        assert node._restored_risk is None
        assert _restores(events) == []
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "no prior session" in r.getMessage()
        ]
        assert warnings == [
            "paper PAPER-007: no prior session for account "
            f"{ACCOUNT_A}; limits start at current equity"
        ]
        [start] = [e for e in events if e.get("action") == "start"]
        assert start["start_fresh"] is False
        assert start["account"] == ACCOUNT_A
        assert start["trader_id"] == "PAPER-007"
        assert node.status.state != NodeState.HALTED
    finally:
        await node._shutdown()


async def test_start_event_records_start_fresh_and_account(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R23: the start event carries start_fresh (bool) and the account."""
    node, events = _node(monkeypatch, seeded, start_fresh=True)
    await node._initialize()
    try:
        [start] = [e for e in events if e.get("action") == "start"]
        assert start["start_fresh"] is True
        assert start["account"] == ACCOUNT_A
        assert {"strategy_id", "validation_run_id", "symbols", "testnet", "leverage"} <= set(start)
    finally:
        await node._shutdown()


# --------------------------------------------------- R5-R10 which row wins


async def test_other_key_not_inherited(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R5: a row on account B is invisible to a KEY-A node."""
    _put(seeded["db"], "PAPER-041", account=ACCOUNT_B, risk=_risk())
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is None
        assert _restores(events) == []
    finally:
        await node._shutdown()


async def test_live_and_testnet_never_mix(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6: the same key on live is a different account from the same key on testnet."""
    _put(seeded["db"], "PAPER-041", account="binance:live:891a21f36b950681", risk=_risk())
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is None
        assert _restores(events) == []
    finally:
        await node._shutdown()


async def test_legacy_row_not_inherited(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7: a row written before the account key existed is never matched."""
    _put(seeded["db"], "MHZ1-SID82", account=None, risk=_risk(), extra={"testnet": True})
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is None
        assert _restores(events) == []
    finally:
        await node._shutdown()


async def test_newest_account_row_wins(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R8: of two other sessions on the account the newer row is the source."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    _put(seeded["db"], "PAPER-042", risk=_risk(high_water_mark="11000"))
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "11000"
        assert _restores(events)[0]["risk_source_trader_id"] == "PAPER-042"
    finally:
        await node._shutdown()


async def test_newer_account_row_beats_own_risk_but_not_own_halt(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9: risk from the newer PAPER-041 row; the halt still comes from the own row."""
    own = _put(seeded["db"], "PAPER-007", risk=_risk(high_water_mark="10000"), halt="manual",
               message="M", state="halted")
    _sql(seeded["db"], "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
         "2026-01-01T00:00:00", own)
    _put(seeded["db"], "PAPER-041", risk=_risk(high_water_mark="11000"))
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "11000"
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert node.status.error_message == f"{RESTORED_HALT_PREFIX}M"
        [event] = _restores(events)
        assert event["risk_source_trader_id"] == "PAPER-041"
        assert event["checkpoint_time"] == "2026-01-01T00:00:00"
        assert event["prior_halt_reason"] == "manual"
        assert event["risk_source_checkpoint_time"] != "2026-01-01T00:00:00"
    finally:
        await node._shutdown()


async def test_own_newer_than_account_row_unchanged(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10: when the own row is the newest, today's path: own risk, own source."""
    _put(seeded["db"], "PAPER-041", risk=_risk(high_water_mark="11000"))
    _put(seeded["db"], "PAPER-007", risk=_risk(high_water_mark="10000"))
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10000"
        assert _restores(events)[0]["risk_source_trader_id"] == "PAPER-007"
    finally:
        await node._shutdown()


# ------------------------------------------------------------ R11 start fresh


async def test_start_fresh_keeps_own_risk_and_halt_but_looks_up_the_watch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R11: start_fresh keeps the own risk and halt; the account lookup runs for the watch.

    1ghod: the liquidation watch is report-only, so it is still inherited from the
    newer account row -- but the risk bookkeeping and the halt stay the own row's.
    """
    _put(seeded["db"], "PAPER-007", risk=_risk(high_water_mark="10000"), halt="manual",
         message="M", state="halted")
    _put(seeded["db"], "PAPER-041", risk=_risk(high_water_mark="11000"),
         extra={"liquidation_watch": {"cursors": {"BTCUSDT": 5}, "seen": {}}})
    calls: list[tuple[str, str]] = []
    original = StatePersistence.load_newer_account_checkpoint

    def spy(self: StatePersistence, account: str, trader_id: str) -> StateCheckpoint | None:
        calls.append((account, trader_id))
        return original(self, account, trader_id)

    monkeypatch.setattr(StatePersistence, "load_newer_account_checkpoint", spy)
    node, events = _node(monkeypatch, seeded, start_fresh=True)
    await node._initialize()
    try:
        assert calls == [(ACCOUNT_A, "PAPER-007")]
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10000"
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert _restores(events)[0]["risk_source_trader_id"] == "PAPER-007"
        assert node._restored_watch == {"cursors": {"BTCUSDT": 5}, "seen": {}}
    finally:
        await node._shutdown()


async def test_start_fresh_ignores_account_rows(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R11b: start_fresh with no own row -> blank limits, no restore event, no warning."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    node, events = _node(monkeypatch, seeded, start_fresh=True)
    with caplog.at_level(logging.WARNING):
        await node._initialize()
    try:
        assert node._restored_risk is None
        assert _restores(events) == []
        assert "no prior session" not in caplog.text
        assert node.status.state != NodeState.HALTED
        [start] = [e for e in events if e.get("action") == "start"]
        assert start["start_fresh"] is True
    finally:
        await node._shutdown()


# ------------------------------------------------ R12-R14e fail-closed rows


async def test_inherited_partial_risk_fails_closed(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R12: an unreadable inherited risk state halts ERROR with the re-baseline note."""
    _put(seeded["db"], "PAPER-041", risk=_risk(high_water_mark="10600x"))
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == (
            f"{RESTORED_HALT_PREFIX}unreadable risk state in checkpoint "
            f"(high_water_mark='10600x'); {LIMITS_NOTE}; "
            "owned and claimed positions flattened"
        )
        [event] = _restores(events)
        assert event["risk_problems"] == ["high_water_mark='10600x'"]
        assert event["risk_source_trader_id"] == "PAPER-041"
    finally:
        await node._shutdown()


@pytest.mark.parametrize("newest_malformed", [False, True])
async def test_unrelated_malformed_row_is_skipped(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    newest_malformed: bool,
) -> None:
    """R13/R13b: a malformed node_status row is skipped whatever its position."""
    if newest_malformed:
        _put(seeded["db"], "PAPER-041", risk=_risk())
        bad = _put(seeded["db"], "PAPER-040")
    else:
        bad = _put(seeded["db"], "PAPER-040")
        _put(seeded["db"], "PAPER-041", risk=_risk())
    _sql(seeded["db"], "UPDATE paper_trading_checkpoints SET node_status = ? WHERE id = ?",
         "not json", bad)
    node, events = _node(monkeypatch, seeded)
    with caplog.at_level(logging.ERROR):
        await node._initialize()
    try:
        assert node.status.state != NodeState.HALTED
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert _restores(events)[0]["risk_source_trader_id"] == "PAPER-041"
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    finally:
        await node._shutdown()


async def test_chosen_row_unparseable_fails_closed(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R13c: the chosen account row has a bad created_at -> HALTED ERROR, class-only log."""
    row = _put(seeded["db"], "PAPER-041", risk=_risk())
    _sql(seeded["db"], "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
         "garbage", row)
    node, events = _node(monkeypatch, seeded)
    with caplog.at_level(logging.ERROR):
        await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == (
            f"{RESTORED_HALT_PREFIX}checkpoint unreadable (ValueError); {LIMITS_NOTE}"
        )
        assert node._restored_risk == RiskState()
        assert _restores(events) == [
            {
                "action": "restore",
                "checkpoint_time": None,
                "prior_state": None,
                "prior_halt_reason": None,
                "prior_positions": [],
                "prior_orders": [],
                "restored_risk": RiskState().to_dict(),
                "restored_halt": "error",
                "risk_problems": [],
                "risk_source_trader_id": None,
                "risk_source_checkpoint_time": None,
            }
        ]
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert [r.getMessage() for r in errors] == [
            "paper PAPER-007: checkpoint unreadable (ValueError); halting"
        ]
        assert errors[0].exc_info is None
    finally:
        await node._shutdown()


async def test_restore_failure_log_leaks_no_chained_marker(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The account lookup raising (chained ZZLEAK9 from ZZINNER4) leaks into no sink."""
    def boom(self: StatePersistence, account: str, trader_id: str) -> StateCheckpoint | None:
        raise RuntimeError("lookup ZZLEAK9") from ValueError("ZZINNER4")

    monkeypatch.setattr(StatePersistence, "load_newer_account_checkpoint", boom)
    node, events = _node(monkeypatch, seeded)
    with caplog.at_level(logging.DEBUG):
        await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.error_message == (
            f"{RESTORED_HALT_PREFIX}checkpoint unreadable (RuntimeError); {LIMITS_NOTE}"
        )
        for sink in (caplog.text, repr(events), node.status.error_message):
            assert "ZZLEAK9" not in sink
            assert "ZZINNER4" not in sink
    finally:
        await node._shutdown()


async def test_inherited_error_halt_restored(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14: an ERROR halt of the last session comes back HALTED, operator-resumable."""
    _put(seeded["db"], "PAPER-041", risk=_risk(), halt="error", message="X", state="halted")
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == "inherited from PAPER-041: X"
        assert _restores(events)[0]["restored_halt"] == "error"
        assert node._restored_halt is not None
        assert node._restored_halt.reason in {HaltReason.ERROR}
    finally:
        await node._shutdown()


@pytest.mark.parametrize("reason", ["manual", "signal", "kill_switch"])
async def test_operator_halts_not_inherited(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    """R14b: MANUAL / SIGNAL / KILL_SWITCH halts do not cross sessions; risk does."""
    _put(seeded["db"], "PAPER-041", risk=_risk(), halt=reason, message="M", state="halted")
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node.status.state != NodeState.HALTED
        assert node._restored_halt is None
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert _restores(events)[0]["restored_halt"] is None
    finally:
        await node._shutdown()


async def test_inherited_unknown_halt_fails_closed(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14c: an unrecognised halt_reason counts as ERROR and is inherited."""
    _put(seeded["db"], "PAPER-041", risk=_risk(), halt="bogus", state="halted")
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == (
            "inherited from PAPER-041: unrecognised prior halt_reason 'bogus'"
        )
    finally:
        await node._shutdown()


async def test_own_halt_beats_inherited_error(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14d (a): the own MANUAL halt wins; (b) without one the inherited ERROR is used."""
    _put(seeded["db"], "PAPER-007", risk=_risk(high_water_mark="10000"), halt="manual",
         message="M", state="halted")
    _put(seeded["db"], "PAPER-041", risk=_risk(), halt="error", message="X", state="halted")
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node.status.halt_reason == HaltReason.MANUAL
        assert node.status.error_message == f"{RESTORED_HALT_PREFIX}M"
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
    finally:
        await node._shutdown()
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints")
    _put(seeded["db"], "PAPER-007", risk=_risk(high_water_mark="10000"))
    _put(seeded["db"], "PAPER-041", risk=_risk(), halt="error", message="X", state="halted")
    node_b, _events_b = _node(monkeypatch, seeded)
    await node_b._initialize()
    try:
        assert node_b.status.halt_reason == HaltReason.ERROR
        assert node_b.status.error_message == "inherited from PAPER-041: X"
        assert node_b._restored_risk is not None
        assert node_b._restored_risk.to_dict()["high_water_mark"] == "10600"
    finally:
        await node_b._shutdown()


async def test_inherited_halt_prefix_does_not_grow(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14e: a chain of prefixes collapses to one."""
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_risk(),
        halt="error",
        message="inherited from PAPER-040: restored after restart: X",
        state="halted",
    )
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node.status.error_message == "inherited from PAPER-041: X"
    finally:
        await node._shutdown()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("X", "inherited from PAPER-041: X"),
        (f"{RESTORED_HALT_PREFIX}X", "inherited from PAPER-041: X"),
        ("inherited from PAPER-040: X", "inherited from PAPER-041: X"),
        ("inherited from PAPER-040: restored after restart: X", "inherited from PAPER-041: X"),
        ("restored after restart: inherited from PAPER-039: restored after restart: X",
         "inherited from PAPER-041: X"),
        ("inherited from PAPER-040 X", "inherited from PAPER-041: inherited from PAPER-040 X"),
    ],
)
def test_inherited_halt_message_strips_both_prefixes(message: str, expected: str) -> None:
    assert inherited_halt_message("PAPER-041", message) == expected


# --------------------------------------------------------------------- R15


_INHERIT_HINT = (
    " (limits inherited from PAPER-041; start fresh skips that trader id's risk state; "
    "this trader id's own saved limits and halt still carry)"
)


def _wire_guard(monkeypatch: pytest.MonkeyPatch, equity: str) -> _SettablePortfolio:
    """Give the node's real (unregistered) guard a portfolio, clock and cache."""
    portfolio = _SettablePortfolio(equity)
    monkeypatch.setattr(TradingGuard, "portfolio", property(lambda self: portfolio))
    monkeypatch.setattr(TradingGuard, "clock", property(lambda self: _FakeClock()))
    monkeypatch.setattr(TradingGuard, "cache", property(lambda self: _EmptyCache()))
    return portfolio


async def test_inherited_hwm_retrips_drawdown(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R15 / P-R8: the node's own guard carries the inherited peak 10000: RUNNING at
    8501, MAX_DRAWDOWN at 8500 (15%), with the inherited-limits hint in the message."""
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_full_risk(
            high_water_mark="10000", day="2026-01-01", day_start_equity="10000",
            consecutive_losses=0, last_equity="10000",
        ),
    )
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        guard = node._guard
        assert guard is not None
        assert guard.risk_state is node._restored_risk
        portfolio = _wire_guard(monkeypatch, "8501")
        guard.evaluate_risk()
        assert guard.guard_state == GuardState.RUNNING
        assert node.status.state != NodeState.HALTED
        portfolio.equity_value = Decimal("8500")
        guard.evaluate_risk()
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.MAX_DRAWDOWN
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MAX_DRAWDOWN
        expected = (
            "drawdown 15.00% >= limit 15.00% (equity 8500, high water mark 10000)"
            + _INHERIT_HINT
        )
        assert guard._message == expected
        assert node.status.error_message == expected
    finally:
        await node._shutdown()


async def test_inherited_streak_halts_with_hint_and_own_streak_does_not(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P-R8: a losing-streak halt on inherited risk names the source; on the node's own
    risk it does not."""
    streak = _full_risk(
        high_water_mark="10000", day="2026-01-01", day_start_equity="10000",
        consecutive_losses=10, last_equity="10000",
    )
    expected = "10 consecutive losing positions >= limit 10"
    _put(seeded["db"], "PAPER-041", risk=streak)
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._guard is not None
        _wire_guard(monkeypatch, "10000")
        node._guard.evaluate_risk()
        assert node._guard.halt_reason == HaltReason.MAX_CONSECUTIVE_LOSSES
        assert node._guard._message == expected + _INHERIT_HINT
    finally:
        await node._shutdown()
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints")
    _put(seeded["db"], "PAPER-007", risk=streak)
    own, _events2 = _node(monkeypatch, seeded)
    await own._initialize()
    try:
        assert own._guard is not None
        own._guard.evaluate_risk()
        assert own._guard.halt_reason == HaltReason.MAX_CONSECUTIVE_LOSSES
        assert own._guard._message == expected
    finally:
        await own._shutdown()


async def test_inherited_risk_logs_warning_naming_source(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P-R8: inheriting from another trader is a WARNING; an own restore is not."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    node, _events = _node(monkeypatch, seeded)
    with caplog.at_level(logging.INFO):
        await node._initialize()
    await node._shutdown()
    [record] = [r for r in caplog.records if "inherited risk state" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "paper PAPER-007: inherited risk state from PAPER-041 (high water mark 10600, "
        "3 consecutive losses); a limit breach halts at the first equity read; "
        "start fresh skips this other trader id's risk state; "
        "this trader id's own saved limits and halt still carry"
    )
    caplog.clear()
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints")
    _put(seeded["db"], "PAPER-041", risk=_risk())
    node2, _events2 = _node(monkeypatch, seeded, trader="PAPER-041")
    with caplog.at_level(logging.INFO):
        await node2._initialize()
    await node2._shutdown()
    assert not [r for r in caplog.records if "inherited risk state" in r.getMessage()]


# ------------------------------------------------- P-R7 foreign own checkpoint


async def test_same_trader_other_environment_carries_nothing(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P-R7: trader X on testnet, then trader X on live (same key): no risk, no halt."""
    _put(seeded["db"], "PAPER-007", risk=_risk(), halt="manual", message="M", state="halted")
    node, events = _node(monkeypatch, seeded, testnet=False)
    with caplog.at_level(logging.WARNING):
        await node._initialize()
    try:
        assert node._restored_risk is None
        assert node._restored_halt is None
        assert node.status.state != NodeState.HALTED
        assert _restores(events) == []
        skipped = [r.getMessage() for r in caplog.records if "ignored" in r.getMessage()]
        assert skipped == [
            f"paper PAPER-007: own checkpoint is for account {ACCOUNT_A}, not "
            "binance:live:891a21f36b950681; ignored"
        ]
        assert "KEY-A" not in caplog.text
    finally:
        await node._shutdown()


ACCOUNT_LIVE = "binance:live:891a21f36b950681"
_WATCH = {"cursors": {"BTCUSDT": 1700000000000}, "seen": {}}


async def _session(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    *,
    testnet: bool,
    risk: dict[str, object] | None = None,
    halt: str | None = None,
    start_fresh: bool = False,
) -> PaperTradingNode:
    """Seed one own PAPER-007 row for the environment, start the node, return it
    (the caller asserts, then shuts it down)."""
    if risk is not None:
        _put(
            seeded["db"],
            "PAPER-007",
            account=ACCOUNT_A if testnet else ACCOUNT_LIVE,
            risk=risk,
            halt=halt,
            message="M" if halt else None,
            state="halted" if halt else "stopped",
            extra={"liquidation_watch": _WATCH},
        )
    node, _events = _node(monkeypatch, seeded, testnet=testnet, start_fresh=start_fresh)
    await node._initialize()
    return node


@pytest.mark.parametrize("start_fresh", [False, True])
@pytest.mark.parametrize("halt", ["manual", "error"])
async def test_back_on_testnet_restores_testnet_halt_and_risk(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    halt: str,
    start_fresh: bool,
) -> None:
    """P-R7b: X testnet halted -> X live -> X testnet restores the testnet row (halt,
    risk, watch), with or without start_fresh (which skips only the cross-trader lookup)."""
    _put(seeded["db"], "PAPER-007", account=ACCOUNT_A, risk=_risk(), halt=halt,
         message="M", state="halted", extra={"liquidation_watch": _WATCH})
    live = await _session(monkeypatch, seeded, testnet=False)
    try:
        assert live._restored_risk is None
        assert live._restored_halt is None
    finally:
        await live._shutdown()
    back = await _session(monkeypatch, seeded, testnet=True, start_fresh=start_fresh)
    try:
        assert back._restored_risk is not None
        assert back._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert back.status.state == NodeState.HALTED
        assert back.status.halt_reason == HaltReason(halt)
        assert back.status.error_message == f"{RESTORED_HALT_PREFIX}M"
        assert back._restored_watch == {"cursors": {"BTCUSDT": 1700000000000}, "seen": {}}
        assert back._guard is not None
        assert back._guard.risk_state is back._restored_risk
        assert back._guard.guard_state == GuardState.HALTED
    finally:
        await back._shutdown()


@pytest.mark.parametrize("start_fresh", [False, True])
@pytest.mark.parametrize("halt", ["manual", "error"])
async def test_back_on_live_restores_live_halt_and_risk(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    halt: str,
    start_fresh: bool,
) -> None:
    """P-R7b mirror: X live halted -> X testnet -> X live restores the live row."""
    _put(seeded["db"], "PAPER-007", account=ACCOUNT_LIVE, risk=_risk(high_water_mark="12000"),
         halt=halt, message="M", state="halted", extra={"liquidation_watch": _WATCH})
    testnet = await _session(monkeypatch, seeded, testnet=True)
    try:
        assert testnet._restored_risk is None
        assert testnet._restored_halt is None
    finally:
        await testnet._shutdown()
    back = await _session(monkeypatch, seeded, testnet=False, start_fresh=start_fresh)
    try:
        assert back._restored_risk is not None
        assert back._restored_risk.to_dict()["high_water_mark"] == "12000"
        assert back._restored_watch == {"cursors": {"BTCUSDT": 1700000000000}, "seen": {}}
        assert back.status.state == NodeState.HALTED
        assert back.status.halt_reason == HaltReason(halt)
        assert back._guard is not None
        assert back._guard.risk_state is back._restored_risk
    finally:
        await back._shutdown()


async def test_legacy_own_row_without_account_is_restored(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-account own row (no key) still restores for the same trader."""
    _put(seeded["db"], "PAPER-007", account=None, risk=_risk(), halt="manual",
         message="M", state="halted")
    node = await _session(monkeypatch, seeded, testnet=True)
    try:
        assert node.status.halt_reason == HaltReason.MANUAL
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
    finally:
        await node._shutdown()


@pytest.mark.parametrize(
    ("halt", "reason"),
    [("error", HaltReason.ERROR), ("manual", HaltReason.MANUAL),
     ("kill_switch", HaltReason.KILL_SWITCH), ("signal", HaltReason.SIGNAL)],
)
async def test_key_rotation_carries_the_own_risk_and_halt(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    halt: str,
    reason: HaltReason,
) -> None:
    """qa42q: an own KEY-B row into a KEY-A node (same env: a key rotation) is the
    trader's own row, so its risk and halt carry and a rotation cannot re-arm a
    halted trader."""
    _put(seeded["db"], "PAPER-007", account=ACCOUNT_B, risk=_risk(), halt=halt,
         message="X", state="halted")
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert node._restored_halt is not None
        assert node._restored_halt.reason == reason
        assert node.status.state == NodeState.HALTED
        assert _restores(events)[0]["risk_source_trader_id"] == "PAPER-007"
    finally:
        await node._shutdown()


async def test_key_rotation_carries_the_risk_so_a_drawdown_retrips(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """qa42q: a risk halt is not restored as a halt; it re-trips from the carried
    bookkeeping, so the rotated row's high water mark must survive the rotation."""
    _put(seeded["db"], "PAPER-007", account=ACCOUNT_B,
         risk=_risk(high_water_mark="20000"), halt="max_drawdown", state="halted")
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "20000"
        assert node._restored_halt is None
    finally:
        await node._shutdown()


async def test_other_trader_row_of_another_key_carries_nothing(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """qa42q boundary: only the trader's OWN row crosses a key rotation; another
    trader's KEY-B row stays on its account."""
    _put(seeded["db"], "PAPER-041", account=ACCOUNT_B, risk=_risk(), halt="error",
         message="X", state="halted")
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is None
        assert node._restored_halt is None
        assert node.status.state != NodeState.HALTED
        assert _restores(events) == []
    finally:
        await node._shutdown()


async def test_foreign_own_row_does_not_hide_account_rows(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P-R7: the own-id threshold ignores a foreign own row, so an older KEY-A row of
    another trader is still inherited."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    _put(seeded["db"], "PAPER-007", account=ACCOUNT_B, risk=_risk(high_water_mark="9000"))
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert _restores(events)[0]["risk_source_trader_id"] == "PAPER-041"
    finally:
        await node._shutdown()


# ----------------------------------------------------------------- R18 / R19


async def test_two_restarts_chain(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R18: 008 inherits 041; 008 again uses its own row; a new 009 inherits from 008."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    first, _e1 = _node(monkeypatch, seeded, trader="PAPER-008")
    await first._initialize()
    try:
        assert first._restored_risk is not None
        assert first._restored_risk.to_dict()["high_water_mark"] == "10600"
        initial = _latest(seeded["db"], "PAPER-008").node_status
        assert initial["account"] == ACCOUNT_A
        risk = initial["risk"]
        assert isinstance(risk, dict)
        assert risk["high_water_mark"] == "10600"
        assert risk["day_start_equity"] == "10400"
        assert risk["consecutive_losses"] == 3
    finally:
        await first._shutdown()

    again, events_again = _node(monkeypatch, seeded, trader="PAPER-008")
    await again._initialize()
    try:
        assert again._restored_risk is not None
        assert again._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert _restores(events_again)[0]["risk_source_trader_id"] == "PAPER-008"
    finally:
        await again._shutdown()

    third, events_third = _node(monkeypatch, seeded, trader="PAPER-009")
    await third._initialize()
    try:
        assert third._restored_risk is not None
        assert third._restored_risk.to_dict()["high_water_mark"] == "10600"
        assert _restores(events_third)[0]["risk_source_trader_id"] == "PAPER-008"
    finally:
        await third._shutdown()


async def test_failure_checkpoint_then_new_session(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R19: a failed restore's checkpoint is inherited by the next new session as ERROR."""
    row = _put(seeded["db"], "PAPER-041", risk=_risk())
    _sql(seeded["db"], "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
         "garbage", row)
    first, _e1 = _node(monkeypatch, seeded, trader="PAPER-008")
    await first._initialize()
    await first._shutdown()
    failure = _latest(seeded["db"], "PAPER-008").node_status
    assert failure["halt_reason"] == "error"
    assert failure["account"] == ACCOUNT_A
    assert failure["error_message"] == (
        f"{RESTORED_HALT_PREFIX}checkpoint unreadable (ValueError); {LIMITS_NOTE}"
    )
    # the garbage row is the second session's problem no more: drop it
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints WHERE id = ?", row)

    second, _e2 = _node(monkeypatch, seeded, trader="PAPER-009")
    await second._initialize()
    try:
        assert second.status.state == NodeState.HALTED
        assert second.status.halt_reason == HaltReason.ERROR
        assert second.status.error_message == (
            f"inherited from PAPER-008: checkpoint unreadable (ValueError); {LIMITS_NOTE}"
        )
        assert second._restored_risk == RiskState()
    finally:
        await second._shutdown()


# --------------------------------------------------------------------- R24


async def test_restore_event_keys_uniform(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R24: own, inherited and except-path restore events share one key set."""
    # own-only
    _put(seeded["db"], "PAPER-007", risk=_risk(), account=None)
    node, events = _node(monkeypatch, seeded)
    await node._initialize()
    await node._shutdown()
    [own_event] = _restores(events)
    assert set(own_event) == _RESTORE_KEYS
    assert own_event["risk_source_checkpoint_time"] == own_event["checkpoint_time"]
    # inherited
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints")
    _put(seeded["db"], "PAPER-041", risk=_risk())
    node2, events2 = _node(monkeypatch, seeded)
    await node2._initialize()
    await node2._shutdown()
    [inherited_event] = _restores(events2)
    assert set(inherited_event) == _RESTORE_KEYS
    assert inherited_event["risk_source_checkpoint_time"] is not None
    # except path
    _sql(seeded["db"], "DELETE FROM paper_trading_checkpoints")
    row = _put(seeded["db"], "PAPER-041", risk=_risk())
    _sql(seeded["db"], "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
         "garbage", row)
    node3, events3 = _node(monkeypatch, seeded)
    await node3._initialize()
    await node3._shutdown()
    [failed_event] = _restores(events3)
    assert set(failed_event) == _RESTORE_KEYS
    assert failed_event["risk_source_trader_id"] is None
