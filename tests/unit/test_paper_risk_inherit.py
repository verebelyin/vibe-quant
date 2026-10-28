"""A new paper session inherits the last session's risk state on the same account.

Contract rows R1-R24 of vibe-quant-3fqw9 (core). The risk bookkeeping (high water
mark, UTC-day baseline, losing streak) and an ERROR halt carry over from the newest
checkpoint of ANOTHER trader on the same venue + testnet/live + API-key fingerprint;
the trader's own checkpoint is never discarded, "start fresh" only skips the
cross-session lookup.

The DB-seed branches are one table over the node's restore edge
(``PaperTradingNode._restore_from_checkpoint``, the seam of
``test_paper_lineage_table``), so each branch costs a restore instead of a full node
start; the full ``_initialize`` path keeps the tests whose outcome IS the start
(account key written, start event, operator warnings) and the guard's re-trip.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
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
from vibe_quant.paper.journal import FakeJournal
from vibe_quant.paper.node import NodeState, PaperTradingNode, inherited_halt_message
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

ACCOUNT_A = "binance:testnet:891a21f36b950681"
ACCOUNT_B = "binance:testnet:2b3782037b86aff4"
ACCOUNT_LIVE = "binance:live:891a21f36b950681"
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
) -> tuple[PaperTradingNode, list[tuple[Any, ...]]]:
    """A node on ``key`` whose journal records every write into one ordered list."""
    fake = FakeJournal()
    node = PaperTradingNode(
        _config(
            seeded,
            trader_id=trader,
            binance=BinanceTestnetConfig(key, "secret", testnet=testnet),
            **kw,
        )
    )
    node._journal = fake
    return node, fake.calls


def _restore_only(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    *,
    trader: str = "PAPER-007",
    testnet: bool = True,
    start_fresh: bool = False,
) -> tuple[PaperTradingNode, list[tuple[Any, ...]]]:
    """The node's restore edge alone: ``_restore_from_checkpoint``, no node start."""
    node, events = _node(
        monkeypatch, seeded, trader=trader, testnet=testnet, start_fresh=start_fresh
    )
    node._persistence = StatePersistence(seeded["db"], trader_id=trader)
    try:
        node._restore_from_checkpoint()
    finally:
        node._persistence.close()
        node._persistence = None
    return node, events


def _payloads(calls: list[tuple[Any, ...]]) -> list[dict[str, object]]:
    return [call[2] for call in calls if call[0] == "write"]


def _restores(calls: list[tuple[Any, ...]]) -> list[dict[str, object]]:
    return [e for e in _payloads(calls) if e.get("action") == "restore"]


# ------------------------------------------------------------- restore rows


@dataclass(frozen=True)
class _RestoreRow:
    """One restore branch: the DB seed and what the restore must leave behind."""

    name: str
    seed: Callable[[Path], None]
    risk: dict[str, object] | None = None  # subset of to_dict; None = nothing restored
    source: str | None = None  # the restore event's risk_source_trader_id
    halt: HaltReason | None = None
    message: str | None = None  # the exact restored message / status.error_message
    watch_cursors: dict[str, int] | None = None  # None: _restored_watch must stay None
    event: bool = True
    start_fresh: bool = False
    trader: str = "PAPER-007"
    testnet: bool = True


def _newest_other_row_wins(db: Path) -> None:
    _put(db, "PAPER-041", risk=_risk())
    _put(db, "PAPER-042", risk=_risk(high_water_mark="11000"))


def _other_risk_beats_own_but_not_own_halt(db: Path) -> None:
    own = _put(
        db,
        "PAPER-007",
        risk=_risk(high_water_mark="10000"),
        halt="manual",
        message="M",
        state="halted",
    )
    _sql(
        db,
        "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
        "2026-01-01T00:00:00",
        own,
    )
    _put(db, "PAPER-041", risk=_risk(high_water_mark="11000"))


def _own_newest_row(db: Path) -> None:
    _put(db, "PAPER-041", risk=_risk(high_water_mark="11000"))
    _put(db, "PAPER-007", risk=_risk(high_water_mark="10000"))


def _other_trader_legacy_row(db: Path) -> None:
    _put(db, "MHZ1-SID82", account=None, risk=_risk(), extra={"testnet": True})


def _other_row_only(db: Path) -> None:
    _put(db, "PAPER-041", risk=_risk())


def _garbled_inherited_risk(db: Path) -> None:
    _put(db, "PAPER-041", risk=_risk(high_water_mark="10600x"))


def _malformed_other_row_newest(db: Path) -> None:
    _put(db, "PAPER-041", risk=_risk())
    bad = _put(db, "PAPER-040")
    _sql(
        db,
        "UPDATE paper_trading_checkpoints SET node_status = ? WHERE id = ?",
        "not json",
        bad,
    )


def _other_halt(reason: str) -> Callable[[Path], None]:
    def seed(db: Path) -> None:
        _put(db, "PAPER-041", risk=_risk(), halt=reason, message="X", state="halted")

    return seed


def _own_row_then_inherited_error(db: Path) -> None:
    _put(db, "PAPER-007", risk=_risk(high_water_mark="10000"))
    _put(db, "PAPER-041", risk=_risk(), halt="error", message="X", state="halted")


def _own_manual_halt_then_newer_inherited_error(db: Path) -> None:
    _put(
        db,
        "PAPER-007",
        risk=_risk(high_water_mark="10000"),
        halt="manual",
        message="M",
        state="halted",
    )
    _put(db, "PAPER-041", risk=_risk(), halt="error", message="X", state="halted")


def _testnet_halt_under_a_newer_live_row(db: Path) -> None:
    _put(
        db,
        "PAPER-007",
        account=ACCOUNT_A,
        risk=_risk(),
        halt="manual",
        message="M",
        state="halted",
    )
    _put(db, "PAPER-007", account=ACCOUNT_LIVE, risk=_risk(high_water_mark="12000"))


def _live_halt_under_a_newer_testnet_row(db: Path) -> None:
    _put(
        db,
        "PAPER-007",
        account=ACCOUNT_LIVE,
        risk=_risk(high_water_mark="12000"),
        halt="error",
        message="M",
        state="halted",
    )
    _put(db, "PAPER-007", account=ACCOUNT_A, risk=_risk())
    # the restore reads the live row: the newer testnet row is another environment


def _rotation(halt: str) -> Callable[[Path], None]:
    def seed(db: Path) -> None:
        _put(db, "PAPER-007", account=ACCOUNT_B, risk=_risk(), halt=halt, message="X", state="halted")

    return seed


def _rotation_of_a_risk_halt(db: Path) -> None:
    _put(
        db,
        "PAPER-007",
        account=ACCOUNT_B,
        risk=_risk(high_water_mark="20000"),
        halt="max_drawdown",
        state="halted",
    )


def _other_trader_on_another_key(db: Path) -> None:
    _put(db, "PAPER-041", account=ACCOUNT_B, risk=_risk(), halt="error", message="X", state="halted")


def _foreign_own_row_under_an_account_row(db: Path) -> None:
    _put(db, "PAPER-041", risk=_risk())
    _put(db, "PAPER-007", account=ACCOUNT_B, risk=_risk(high_water_mark="9000"))


_ROWS: list[_RestoreRow] = [
    _RestoreRow(
        "r8_newest_other_row_wins",
        _newest_other_row_wins,
        risk={"high_water_mark": "11000", "consecutive_losses": 3},
        source="PAPER-042",
        watch_cursors={},
    ),
    _RestoreRow(
        "r9_other_risk_beats_own_but_not_own_halt",
        _other_risk_beats_own_but_not_own_halt,
        risk={"high_water_mark": "11000"},
        source="PAPER-041",
        halt=HaltReason.MANUAL,
        message=f"{RESTORED_HALT_PREFIX}M",
        watch_cursors={},
    ),
    _RestoreRow(
        "r10_own_newest_row_unchanged",
        _own_newest_row,
        risk={"high_water_mark": "10000"},
        source="PAPER-007",
        watch_cursors={},
    ),
    _RestoreRow("r7_other_trader_legacy_row_not_matched", _other_trader_legacy_row, event=False),
    _RestoreRow(
        "r11b_start_fresh_ignores_other_rows",
        _other_row_only,
        event=False,
        start_fresh=True,
        watch_cursors={},
    ),
    _RestoreRow(
        "r12_inherited_partial_risk_halts",
        _garbled_inherited_risk,
        risk={"high_water_mark": None, "consecutive_losses": 3},
        source="PAPER-041",
        halt=HaltReason.ERROR,
        message=(
            f"{RESTORED_HALT_PREFIX}unreadable risk state in checkpoint "
            "(high_water_mark='10600x'); limits re-baselined at current equity; "
            "owned and claimed positions flattened"
        ),
        watch_cursors={},
    ),
    _RestoreRow(
        "r13_malformed_other_row_skipped",
        _malformed_other_row_newest,
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        watch_cursors={},
    ),
    _RestoreRow(
        "r14_inherited_error_halt",
        _other_halt("error"),
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        halt=HaltReason.ERROR,
        message="inherited from PAPER-041: X",
        watch_cursors={},
    ),
    _RestoreRow(
        "r14b_manual_not_inherited",
        _other_halt("manual"),
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        watch_cursors={},
    ),
    _RestoreRow(
        "r14b_signal_not_inherited",
        _other_halt("signal"),
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        watch_cursors={},
    ),
    _RestoreRow(
        "r14b_kill_switch_not_inherited",
        _other_halt("kill_switch"),
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        watch_cursors={},
    ),
    _RestoreRow(
        "r14c_unknown_inherited_halt_fails_closed",
        _other_halt("bogus"),
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        halt=HaltReason.ERROR,
        message="inherited from PAPER-041: unrecognised prior halt_reason 'bogus'",
        watch_cursors={},
    ),
    _RestoreRow(
        "r14d_own_row_then_inherited_error",
        _own_row_then_inherited_error,
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        halt=HaltReason.ERROR,
        message="inherited from PAPER-041: X",
        watch_cursors={},
    ),
    _RestoreRow(
        "r14d_own_manual_halt_beats_newer_inherited_error",
        _own_manual_halt_then_newer_inherited_error,
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        halt=HaltReason.MANUAL,
        message=f"{RESTORED_HALT_PREFIX}M",
        watch_cursors={},
    ),
    _RestoreRow(
        "p_r7b_testnet_row_under_a_newer_live_row",
        _testnet_halt_under_a_newer_live_row,
        risk={"high_water_mark": "10600"},
        source="PAPER-007",
        halt=HaltReason.MANUAL,
        message=f"{RESTORED_HALT_PREFIX}M",
        watch_cursors={},
    ),
    _RestoreRow(
        "p_r7b_live_row_under_a_newer_testnet_row",
        _live_halt_under_a_newer_testnet_row,
        risk={"high_water_mark": "12000"},
        source="PAPER-007",
        halt=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}M",
        watch_cursors={},
        testnet=False,
    ),
    _RestoreRow(
        "qa42q_key_rotation_error",
        _rotation("error"),
        risk={"high_water_mark": "10600"},
        source="PAPER-007",
        halt=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}X",
    ),
    _RestoreRow(
        "qa42q_key_rotation_manual",
        _rotation("manual"),
        risk={"high_water_mark": "10600"},
        source="PAPER-007",
        halt=HaltReason.MANUAL,
        message=f"{RESTORED_HALT_PREFIX}X",
    ),
    _RestoreRow(
        "qa42q_key_rotation_kill_switch",
        _rotation("kill_switch"),
        risk={"high_water_mark": "10600"},
        source="PAPER-007",
        halt=HaltReason.KILL_SWITCH,
        message=f"{RESTORED_HALT_PREFIX}X",
    ),
    _RestoreRow(
        "qa42q_key_rotation_signal",
        _rotation("signal"),
        risk={"high_water_mark": "10600"},
        source="PAPER-007",
        halt=HaltReason.SIGNAL,
        message=f"{RESTORED_HALT_PREFIX}X",
    ),
    _RestoreRow(
        "qa42q_rotation_carries_the_risk_of_a_risk_halt",
        _rotation_of_a_risk_halt,
        risk={"high_water_mark": "20000"},
        source="PAPER-007",
    ),
    _RestoreRow(
        "qa42q_other_trader_on_another_key_carries_nothing",
        _other_trader_on_another_key,
        event=False,
    ),
    _RestoreRow(
        "p_r7_foreign_own_row_does_not_hide_an_account_row",
        _foreign_own_row_under_an_account_row,
        risk={"high_water_mark": "10600"},
        source="PAPER-041",
        watch_cursors={},
    ),
]


def _check_restored(
    node: PaperTradingNode, events: list[tuple[Any, ...]], row: _RestoreRow
) -> None:
    if row.risk is None:
        assert node._restored_risk is None, row.name
    else:
        assert node._restored_risk is not None, row.name
        restored = node._restored_risk.to_dict()
        assert {key: restored[key] for key in row.risk} == row.risk, row.name
    # the hint names ANOTHER trader; an own row carries its risk without one
    other = row.source if row.source not in (None, row.trader) else None
    assert node._risk_inherited_from == other, row.name
    halt = node._restored_halt
    assert (halt.reason if halt is not None else None) is row.halt, row.name
    assert node.status.halt_reason is row.halt, row.name
    expected_state = NodeState.HALTED if row.halt is not None else NodeState.INITIALIZING
    assert node.status.state == expected_state, row.name
    if row.message is not None:
        assert halt is not None and halt.message == row.message, row.name
        assert node.status.error_message == row.message, row.name
    if row.watch_cursors is None:
        assert node._restored_watch is None, row.name
    else:
        assert node._restored_watch is not None, row.name
        assert node._restored_watch["cursors"] == row.watch_cursors, row.name
    if row.event:
        [event] = _restores(events)
        assert event["risk_source_trader_id"] == row.source, row.name
    else:
        assert _restores(events) == [], row.name


@pytest.mark.parametrize("row", _ROWS, ids=[row.name for row in _ROWS])
def test_restore_branch_rows(
    row: _RestoreRow,
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every R4-R14e / P-R7 / qa42q branch through the node's restore edge."""
    row.seed(seeded["db"])

    node, events = _restore_only(
        monkeypatch,
        seeded,
        trader=row.trader,
        testnet=row.testnet,
        start_fresh=row.start_fresh,
    )
    try:
        _check_restored(node, events, row)
    finally:
        node._state_manager.close()


# ----------------------------------------------------------- R1 account id


def test_account_id_vectors() -> None:
    """R1: venue + env + sha256(api_key)[:16]; neither key nor secret appears."""
    testnet = account_id(BinanceTestnetConfig("KEY-A", "SECRET-A", testnet=True))
    live = account_id(BinanceTestnetConfig("KEY-A", "SECRET-A", testnet=False))
    assert testnet == ACCOUNT_A
    assert live == ACCOUNT_LIVE
    assert account_id(BinanceTestnetConfig("KEY-B", "S", testnet=True)) == ACCOUNT_B
    for text in (testnet, live):
        assert "KEY-A" not in text
        assert "SECRET-A" not in text


# --------------------------------------------------------- first test (T2)


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
        [start] = [e for e in _payloads(events) if e.get("action") == "start"]
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
        [start] = [e for e in _payloads(events) if e.get("action") == "start"]
        assert start["start_fresh"] is True
        assert start["account"] == ACCOUNT_A
        assert {"strategy_id", "validation_run_id", "symbols", "testnet", "leverage"} <= set(start)
    finally:
        await node._shutdown()


# --------------------------------------------------------------- R14e text


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("X", "inherited from PAPER-041: X"),
        ("inherited from PAPER-040: restored after restart: X", "inherited from PAPER-041: X"),
        (
            "restored after restart: inherited from PAPER-039: restored after restart: X",
            "inherited from PAPER-041: X",
        ),
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
            high_water_mark="10000",
            day="2026-01-01",
            day_start_equity="10000",
            consecutive_losses=0,
            last_equity="10000",
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
            "drawdown 15.00% >= limit 15.00% (equity 8500, high water mark 10000)" + _INHERIT_HINT
        )
        assert guard.message == expected
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
        high_water_mark="10000",
        day="2026-01-01",
        day_start_equity="10000",
        consecutive_losses=10,
        last_equity="10000",
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
        assert node._guard.message == expected + _INHERIT_HINT
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
        assert own._guard.message == expected
    finally:
        await own._shutdown()


async def test_inherited_risk_logs_warning_naming_source(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P-R8: inheriting from another trader is a WARNING naming the source."""
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


# ------------------------------------------------- P-R7 foreign own checkpoint


async def test_same_trader_other_environment_carries_nothing(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P-R7: trader X on testnet, then trader X on live (same key): no risk, no halt,
    and the ignored row is named for the operator."""
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
            f"{ACCOUNT_LIVE}; ignored"
        ]
        assert "KEY-A" not in caplog.text
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
    _sql(
        seeded["db"],
        "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
        "garbage",
        row,
    )
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
