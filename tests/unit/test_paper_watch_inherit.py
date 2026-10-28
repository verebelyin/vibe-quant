"""vibe-quant-1zh0m: the liquidation watch follows the risk source row.

A new paper session on an account inherits the risk bookkeeping from the newest
checkpoint of another trader on that account (3fqw9), but the liquidation watch
state (forceOrders cursors + dedupe keys) was only restored from the trader's OWN
row. So a new trader id started scanning at node start and never saw the venue
liquidations that happened between the two sessions.

The watch comes from the SAME source row as ``RiskState``, MERGED with the own
row: the newer row wins for a symbol both name, a symbol only the own row names
keeps its own cursor, and the gap boundary is the older of the two timestamps (so
a symbol neither row names still rescans from before the gap). ``start_fresh``
skips only the risk/halt inheritance: the account lookup still runs and the watch
is still inherited and merged with the own row (vibe-quant-1ghod).

The row mappings are one table over the node's restore edge (the
``test_paper_lineage_table`` seam); the tests kept below assert the ALERT that a
missed cursor would swallow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_halt_restore import seeded  # noqa: F401  (fixture)
from tests.unit.test_paper_liquidation_detector import (
    _CHECKPOINT_MS,
    _CHECKPOINT_STAMP,
    _OVERLAP_MS,
    _START_MS,
    _TRADER_ID,
    _harness,
    _order,
    _spy_methods,
    _write_checkpoint,
)
from tests.unit.test_paper_liquidation_detector import (
    fake_tradingnode as detector_fake_tradingnode,  # noqa: F401  (fixture)
)
from tests.unit.test_paper_liquidation_detector import (
    seeded as detector_seeded,  # noqa: F401  (fixture)
)
from tests.unit.test_paper_risk_inherit import (
    _latest,
    _node,
    _put,
    _restore_only,
    _risk,
    _sql,
    fake_tradingnode,  # noqa: F401  (fixture)
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.config import account_id
from vibe_quant.paper.guard import HaltReason
from vibe_quant.paper.node import NodeState

if TYPE_CHECKING:
    from pathlib import Path

#: Own row 00:30Z and the newer other-session row 02:00Z (_CHECKPOINT_STAMP).
_OWN_STAMP = "2026-01-02 00:30:00"
_OWN_MS = _CHECKPOINT_MS - 5_400_000
#: 01:00Z: between the own row (00:30) and the other row (02:00).
_GAP_MS = _CHECKPOINT_MS - 3_600_000


# ------------------------------------------------------ inherited watch rows


@dataclass(frozen=True)
class _WatchRow:
    """One watch-merge branch: the rows seeded and what the restore must yield."""

    name: str
    seed: Any
    cursors: dict[str, int]
    seen: dict[str, int]
    watch_ms: int


def _own_then_newer_disjoint(db: Path) -> None:
    """Own row 00:30 names BTC; the newer row 02:00 names ETH."""
    _put(
        db,
        _TRADER_ID,
        risk=_risk(),
        extra={
            "liquidation_watch": {"cursors": {"BTCUSDT": _OWN_MS}, "seen": {"BTCUSDT:1": _OWN_MS}}
        },
    )
    _sql(db, "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = 1", _OWN_STAMP)
    _put(
        db,
        "PAPER-041",
        risk=_risk(),
        extra={
            "liquidation_watch": {
                "cursors": {"ETHUSDT": _CHECKPOINT_MS},
                "seen": {"ETHUSDT:2": _CHECKPOINT_MS},
            }
        },
    )
    _sql(db, "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = 2", _CHECKPOINT_STAMP)


def _newer_row_without_watch_key(db: Path) -> None:
    """A pre-watcher row still bounds the gap through its timestamp (R50c)."""
    _put(db, "PAPER-041", risk=_risk())
    _sql(db, "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = 1", _CHECKPOINT_STAMP)


_WATCH_ROWS: list[_WatchRow] = [
    _WatchRow(
        "own_symbol_kept_and_source_added",
        _own_then_newer_disjoint,
        cursors={"BTCUSDT": _OWN_MS, "ETHUSDT": _CHECKPOINT_MS},
        seen={"BTCUSDT:1": _OWN_MS, "ETHUSDT:2": _CHECKPOINT_MS},
        watch_ms=_OWN_MS,
    ),
    _WatchRow(
        "newer_row_without_watch_key",
        _newer_row_without_watch_key,
        cursors={},
        seen={},
        watch_ms=_CHECKPOINT_MS,
    ),
]


@pytest.mark.parametrize("row", _WATCH_ROWS, ids=[row.name for row in _WATCH_ROWS])
def test_inherited_watch_merge_rows(
    row: _WatchRow,
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symbol only the own row names keeps its cursor; the gap is the older stamp."""
    row.seed(seeded["db"])

    node, _events = _restore_only(monkeypatch, seeded)
    try:
        assert node._restored_watch == {"cursors": row.cursors, "seen": row.seen}
        assert node._restored_watch_ms == row.watch_ms
    finally:
        node._state_manager.close()


async def test_inherited_watch_survives_into_the_new_sessions_own_checkpoint(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chain: PAPER-041 -> PAPER-007's init checkpoint (no watcher yet) -> PAPER-009."""
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_risk(),
        extra={"liquidation_watch": {"cursors": {"BTCUSDT": _OWN_MS}, "seen": {}}},
    )
    first, _events = _node(monkeypatch, seeded)
    await first._initialize()
    try:
        watch = _latest(seeded["db"], "PAPER-007").node_status["liquidation_watch"]
        assert isinstance(watch, dict)
        assert watch["cursors"]["BTCUSDT"] == _OWN_MS
    finally:
        await first._shutdown()

    third, _events = _node(monkeypatch, seeded, trader="PAPER-009")
    await third._initialize()
    try:
        assert third._restored_watch is not None
        assert third._restored_watch["cursors"]["BTCUSDT"] == _OWN_MS
    finally:
        await third._shutdown()


# ------------------------------------------------------------ alert paths


async def test_gap_liquidation_is_alerted_after_inheriting(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A force order between the two sessions reaches the alert path."""
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-041",
        stamp=_CHECKPOINT_STAMP,
        node_status={
            "account": account_id(_config(detector_seeded).binance),
            "liquidation_watch": {
                "cursors": {"BTCUSDT": _CHECKPOINT_MS},
                "seen": {},
            },
        },
    )
    harness = await _harness(
        monkeypatch,
        detector_seeded,
        orders={"BTCUSDT": [_order(update_time_ms=_CHECKPOINT_MS + 60_000)]},
        trader_id=_TRADER_ID,
    )
    try:
        watcher = harness.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert len(harness.events(EventType.LIQUIDATION)) == 1
        assert len(harness.alerts.calls) == 1
    finally:
        await harness.node._shutdown()


async def test_own_symbol_unwatched_by_newer_other_session_is_still_watched(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Own row 00:30 names BTC only; the newer row 02:00 names ETH only.

    The own BTC cursor must survive the inheritance: falling back to the other
    row's timestamp (02:00) would skip the 01:00 venue liquidation entirely.
    """
    acct = account_id(_config(detector_seeded).binance)
    _write_checkpoint(
        detector_seeded,
        trader_id=_TRADER_ID,
        stamp=_OWN_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {"cursors": {"BTCUSDT": _OWN_MS}, "seen": {}},
        },
    )
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-041",
        stamp=_CHECKPOINT_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {"cursors": {"ETHUSDT": _CHECKPOINT_MS}, "seen": {}},
        },
    )
    harness = await _harness(
        monkeypatch,
        detector_seeded,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
        trader_id=_TRADER_ID,
    )
    try:
        assert harness.node._restored_watch is not None
        assert harness.node._restored_watch["cursors"]["BTCUSDT"] == _OWN_MS
        assert harness.node._restored_watch_ms == _OWN_MS

        watcher = harness.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert len(harness.events(EventType.LIQUIDATION)) == 1
        assert len(harness.alerts.calls) == 1
    finally:
        await harness.node._shutdown()


# ------------------------------------------------- 0m3tf disjoint-symbol chain


def _run_with_symbols(state_seeded: dict[str, Any], symbols: list[str]) -> int:
    """A completed validation run over ``symbols`` (the detector runner, parameterised)."""
    sm = StateManager(state_seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=state_seeded["sid"],
            run_mode="validation",
            symbols=symbols,
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid


async def test_disjoint_chain_middle_session_alerts_the_untraded_gap_once(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1cvw6/N1: A(BTC) -> B(ETH) -> C(BTC): B watches the gap and alerts it once."""
    acct = account_id(_config(detector_seeded).binance)
    btc_run = _run_with_symbols(detector_seeded, ["BTCUSDT"])
    eth_run = _run_with_symbols(detector_seeded, ["ETHUSDT"])
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-001",
        stamp=_OWN_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {"cursors": {"BTCUSDT": _OWN_MS}, "seen": {}},
        },
    )

    middle = await _harness(
        monkeypatch,
        detector_seeded,
        trader_id="PAPER-002",
        validation_run_id=eth_run,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
    )
    try:
        watcher = middle.node._build_liquidation_watcher()
        await watcher.poll_once()
        assert len(middle.events(EventType.LIQUIDATION)) == 1, "B watches the untraded BTC gap"
        assert len(middle.alerts.calls) == 1
        status = middle.node._capture_checkpoint().node_status
    finally:
        await middle.node._shutdown()
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-002",
        stamp=_CHECKPOINT_STAMP,
        node_status=status,
    )

    middle_watch = status["liquidation_watch"]
    assert middle_watch["cursors"] == {"ETHUSDT": _START_MS, "BTCUSDT": _START_MS}

    last = await _harness(
        monkeypatch,
        detector_seeded,
        trader_id="PAPER-003",
        validation_run_id=btc_run,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
    )
    try:
        watcher = last.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert last.events(EventType.LIQUIDATION) == []
        assert last.alerts.calls == []
    finally:
        await last.node._shutdown()


async def test_disjoint_chain_carries_the_seen_key_and_does_not_re_alert(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0m3tf/N1b: a BTC order A already alerted is not delivered again by C."""
    acct = account_id(_config(detector_seeded).binance)
    btc_run = _run_with_symbols(detector_seeded, ["BTCUSDT"])
    eth_run = _run_with_symbols(detector_seeded, ["ETHUSDT"])
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-001",
        stamp=_OWN_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {
                "cursors": {"BTCUSDT": _OWN_MS},
                "seen": {"BTCUSDT:7": _GAP_MS},
            },
        },
    )

    middle = await _harness(
        monkeypatch,
        detector_seeded,
        trader_id="PAPER-002",
        validation_run_id=eth_run,
    )
    try:
        watcher = middle.node._build_liquidation_watcher()
        await watcher.poll_once()
        status = middle.node._capture_checkpoint().node_status
    finally:
        await middle.node._shutdown()
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-002",
        stamp=_CHECKPOINT_STAMP,
        node_status=status,
    )

    last = await _harness(
        monkeypatch,
        detector_seeded,
        trader_id="PAPER-003",
        validation_run_id=btc_run,
        orders={"BTCUSDT": [_order(order_id=7, update_time_ms=_GAP_MS)]},
    )
    try:
        watcher = last.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert last.events(EventType.LIQUIDATION) == []
        assert last.alerts.calls == []
    finally:
        await last.node._shutdown()


@pytest.mark.parametrize(("carried", "warns"), [(11, False), (12, True)])
async def test_watch_over_the_request_budget_warns_at_build(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    carried: int,
    warns: bool,
) -> None:
    """altqb: ETHUSDT + N carried symbols; above 240 weight/min (10% of the 2400/min
    IP cap) the watcher build logs one WARNING with the numbers; at 240 it is quiet."""
    acct = account_id(_config(detector_seeded).binance)
    eth_run = _run_with_symbols(detector_seeded, ["ETHUSDT"])
    cursors = {f"C{i}USDT": _OWN_MS for i in range(carried)}
    _write_checkpoint(
        detector_seeded,
        trader_id=_TRADER_ID,
        stamp=_OWN_STAMP,
        node_status={"account": acct, "liquidation_watch": {"cursors": cursors, "seen": {}}},
    )
    harness = await _harness(monkeypatch, detector_seeded, orders={}, validation_run_id=eth_run)
    try:
        with caplog.at_level(logging.WARNING):
            harness.node._build_liquidation_watcher()
        hits = [r.getMessage() for r in caplog.records if "request weight" in r.getMessage()]
        if warns:
            assert hits == [
                f"paper {_TRADER_ID}: liquidation watch request weight 260/min over the "
                "240/min budget (13 symbols x 20 per 60 s; Binance IP cap 2400/min)"
            ]
        else:
            assert hits == []
    finally:
        await harness.node._shutdown()


async def test_halted_error_session_still_watches_untraded_symbols(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """toqng/R13: a session restored HALTED (ERROR) still polls a carried untraded
    symbol and reports its liquidation; the halt is untouched."""
    acct = account_id(_config(detector_seeded).binance)
    eth_run = _run_with_symbols(detector_seeded, ["ETHUSDT"])
    _write_checkpoint(
        detector_seeded,
        trader_id=_TRADER_ID,
        stamp=_OWN_STAMP,
        node_status={
            "account": acct,
            "state": "halted",
            "halt_reason": "error",
            "error_message": "boom",
            "liquidation_watch": {"cursors": {"BTCUSDT": _OWN_MS}, "seen": {}},
        },
    )
    harness = await _harness(
        monkeypatch,
        detector_seeded,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
        validation_run_id=eth_run,
    )
    try:
        assert harness.node.status.state == NodeState.HALTED
        assert harness.node.status.halt_reason == HaltReason.ERROR
        watcher = harness.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert [symbol for symbol, _, _ in harness.fetch.requests] == ["ETHUSDT", "BTCUSDT"]
        events = harness.events(EventType.LIQUIDATION)
        assert [event["data"]["symbol"] for event in events] == ["BTCUSDT"]
        assert harness.node.status.state == NodeState.HALTED
    finally:
        await harness.node._shutdown()


async def test_untraded_symbol_liquidation_is_reported_without_touching_strategy_or_guard(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1cvw6/R5: a liquidation on a restored untraded symbol reports; nothing else moves."""
    acct = account_id(_config(detector_seeded).binance)
    eth_run = _run_with_symbols(detector_seeded, ["ETHUSDT"])
    _write_checkpoint(
        detector_seeded,
        trader_id=_TRADER_ID,
        stamp=_OWN_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {"cursors": {"BTCUSDT": _OWN_MS}, "seen": {}},
        },
    )

    harness = await _harness(
        monkeypatch,
        detector_seeded,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
        validation_run_id=eth_run,
    )
    guard = harness.node.guard
    assert guard is not None
    guard_spies = _spy_methods(
        monkeypatch, guard, ("halt", "pause", "resume", "close_all", "evaluate_risk")
    )

    watcher = harness.node._build_liquidation_watcher()
    await watcher.poll_once()

    try:
        assert harness.fetch.requests == [
            ("ETHUSDT", _OWN_MS - _OVERLAP_MS, _START_MS),
            ("BTCUSDT", _OWN_MS - _OVERLAP_MS, _START_MS),
        ]
        events = harness.events(EventType.LIQUIDATION)
        assert len(events) == 1, f"exactly one LIQUIDATION event, got {events}"
        assert events[0]["data"]["symbol"] == "BTCUSDT"
        assert events[0]["data"]["position_id"] == "BTCUSDT-PERP.BINANCE"
        assert events[0]["data"]["mode"] == "cross"
        assert harness.alerts.calls == [
            (
                "error",
                "EXCHANGE LIQUIDATION BTCUSDT SELL qty 0.4 @ 2500.5 "
                "(order 1596107620040000020); not traded this session (position "
                "left by an earlier session); NO paper action taken",
                True,
            )
        ]
        assert all(not spy.calls for spy in guard_spies.values()), guard_spies
        assert harness.events(EventType.ORDER) == []
        assert harness.node._capture_checkpoint().node_status["liquidation_watch"] == {
            "cursors": {"ETHUSDT": 1_767_322_800_000, "BTCUSDT": 1_767_322_800_000},
            "seen": {},
        }
    finally:
        await harness.node._shutdown()
