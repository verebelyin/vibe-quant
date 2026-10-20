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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    ACCOUNT_A,
    ACCOUNT_B,
    LIMITS_NOTE,
    _latest,
    _node,
    _put,
    _restores,
    _risk,
    _sql,
    fake_tradingnode,  # noqa: F401  (fixture)
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.config import account_id
from vibe_quant.paper.guard import RESTORED_HALT_PREFIX, HaltReason
from vibe_quant.paper.node import NodeState, PaperTradingNode
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    import pytest


_WATCH_CURSOR = 1_700_000_000_000
#: Own row 00:30Z and the newer other-session row 02:00Z (_CHECKPOINT_STAMP).
_OWN_STAMP = "2026-01-02 00:30:00"
_OWN_MS = _CHECKPOINT_MS - 5_400_000
#: 01:00Z: between the own row (00:30) and the other row (02:00).
_GAP_MS = _CHECKPOINT_MS - 3_600_000


def _watch(cursor: int) -> dict[str, object]:
    return {"cursors": {"BTCUSDT": cursor}, "seen": {"BTCUSDT:7": cursor}}


async def test_new_trader_inherits_liquidation_watch_from_the_newest_account_row(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new trader's watch is the other trader's, bounded by ITS timestamp."""
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_risk(),
        extra={"liquidation_watch": _watch(_WATCH_CURSOR)},
    )
    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_watch == _watch(_WATCH_CURSOR)
        assert node._restored_watch_ms == PaperTradingNode._checkpoint_ms(
            _latest(seeded["db"], "PAPER-041").timestamp
        )
    finally:
        await node._shutdown()


async def test_inherited_watch_survives_into_the_new_sessions_own_checkpoint(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chain: PAPER-041 -> PAPER-007's init checkpoint -> PAPER-009 inherits it."""
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_risk(),
        extra={"liquidation_watch": _watch(_WATCH_CURSOR)},
    )
    first, _events = _node(monkeypatch, seeded)
    await first._initialize()
    try:
        own = _latest(seeded["db"], "PAPER-007")
        watch = own.node_status["liquidation_watch"]
        assert isinstance(watch, dict)
        assert watch["cursors"]["BTCUSDT"] == _WATCH_CURSOR
    finally:
        await first._shutdown()

    third, _events = _node(monkeypatch, seeded, trader="PAPER-009")
    await third._initialize()
    try:
        assert third._restored_watch is not None
        assert third._restored_watch["cursors"]["BTCUSDT"] == _WATCH_CURSOR
    finally:
        await third._shutdown()


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


async def test_own_older_row_keeps_its_symbols_disjoint_from_the_source(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disjoint symbol sets: the union, bounded by the own (earlier) timestamp."""
    acct = account_id(_config(detector_seeded).binance)
    _write_checkpoint(
        detector_seeded,
        trader_id=_TRADER_ID,
        stamp=_OWN_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {
                "cursors": {"BTCUSDT": _OWN_MS},
                "seen": {"BTCUSDT:1": _OWN_MS},
            },
        },
    )
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-041",
        stamp=_CHECKPOINT_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {
                "cursors": {"ETHUSDT": _CHECKPOINT_MS},
                "seen": {"ETHUSDT:2": _CHECKPOINT_MS},
            },
        },
    )
    harness = await _harness(monkeypatch, detector_seeded, trader_id=_TRADER_ID)
    try:
        assert harness.node._restored_watch == {
            "cursors": {"BTCUSDT": _OWN_MS, "ETHUSDT": _CHECKPOINT_MS},
            "seen": {"BTCUSDT:1": _OWN_MS, "ETHUSDT:2": _CHECKPOINT_MS},
        }
        assert harness.node._restored_watch_ms == _OWN_MS
    finally:
        await harness.node._shutdown()


async def test_own_older_row_keeps_its_own_symbol_and_the_source_wins_on_shared(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping sets: the newer row's cursor wins for a symbol both name."""
    acct = account_id(_config(detector_seeded).binance)
    _write_checkpoint(
        detector_seeded,
        trader_id=_TRADER_ID,
        stamp=_OWN_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {
                "cursors": {"BTCUSDT": _OWN_MS, "ETHUSDT": _OWN_MS},
                "seen": {"BTCUSDT:1": _OWN_MS, "ETHUSDT:1": _OWN_MS},
            },
        },
    )
    _write_checkpoint(
        detector_seeded,
        trader_id="PAPER-041",
        stamp=_CHECKPOINT_STAMP,
        node_status={
            "account": acct,
            "liquidation_watch": {
                "cursors": {"ETHUSDT": _CHECKPOINT_MS},
                "seen": {"ETHUSDT:2": _CHECKPOINT_MS},
            },
        },
    )
    harness = await _harness(monkeypatch, detector_seeded, trader_id=_TRADER_ID)
    try:
        assert harness.node._restored_watch == {
            "cursors": {"BTCUSDT": _OWN_MS, "ETHUSDT": _CHECKPOINT_MS},
            "seen": {
                "BTCUSDT:1": _OWN_MS,
                "ETHUSDT:1": _OWN_MS,
                "ETHUSDT:2": _CHECKPOINT_MS,
            },
        }
    finally:
        await harness.node._shutdown()


async def test_newer_other_trader_row_beats_older_own_watch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The own older row must not rescan from earlier (duplicate alerts)."""
    _put(seeded["db"], "PAPER-007", risk=_risk(), extra={"liquidation_watch": _watch(1)})
    _put(seeded["db"], "PAPER-041", risk=_risk(), extra={"liquidation_watch": _watch(2)})

    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_watch is not None
        assert node._restored_watch["cursors"]["BTCUSDT"] == 2
    finally:
        await node._shutdown()


async def test_own_newest_row_still_supplies_its_watch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchanged behaviour: with no newer account row, the own watch is used."""
    _put(seeded["db"], "PAPER-041", risk=_risk(), extra={"liquidation_watch": _watch(2)})
    _put(seeded["db"], "PAPER-007", risk=_risk(), extra={"liquidation_watch": _watch(1)})

    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_watch is not None
        assert node._restored_watch["cursors"]["BTCUSDT"] == 1
    finally:
        await node._shutdown()


async def test_start_fresh_inherits_the_watch_but_not_the_risk(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1ghod/S1: start_fresh resets the risk but still inherits the newer row's watch."""
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_risk(),
        extra={"liquidation_watch": _watch(_WATCH_CURSOR)},
    )
    row = _latest(seeded["db"], "PAPER-041")

    node, events = _node(monkeypatch, seeded, start_fresh=True)
    with caplog.at_level(logging.WARNING):
        await node._initialize()
    try:
        assert node._restored_watch == _watch(_WATCH_CURSOR)
        assert node._restored_watch_ms == PaperTradingNode._checkpoint_ms(row.timestamp)
        assert node._restored_risk is None
        assert _restores(events) == []
        assert "no prior session" not in caplog.text
        assert node.status.state != NodeState.HALTED
    finally:
        await node._shutdown()


async def test_inherited_row_without_watch_key_starts_at_its_timestamp(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-watcher row still bounds the gap through its timestamp (R50c)."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    row = _latest(seeded["db"], "PAPER-041")

    node, _events = _node(monkeypatch, seeded)
    await node._initialize()
    try:
        assert node._restored_watch == {"cursors": {}, "seen": {}}
        assert node._restored_watch_ms == PaperTradingNode._checkpoint_ms(row.timestamp)
    finally:
        await node._shutdown()


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


async def test_own_older_row_merges_a_carrying_sources_untraded_cursor(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0m3tf/N2: the source already carries BTC; the merge keeps it and the own bound."""
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
            "liquidation_watch": {
                "cursors": {"ETHUSDT": _CHECKPOINT_MS, "BTCUSDT": _OWN_MS},
                "seen": {},
            },
        },
    )

    harness = await _harness(monkeypatch, detector_seeded, trader_id=_TRADER_ID)
    try:
        assert harness.node._restored_watch == {
            "cursors": {"BTCUSDT": _OWN_MS, "ETHUSDT": _CHECKPOINT_MS},
            "seen": {},
        }
        assert harness.node._restored_watch_ms == _OWN_MS
    finally:
        await harness.node._shutdown()


# ------------------------------------------- 1ghod start_fresh: watch only


async def test_start_fresh_merges_the_own_and_newer_rows_watch(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1ghod/S2: start_fresh + disjoint rows: the union, bounded by the own timestamp."""
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
        trader_id=_TRADER_ID,
        start_fresh=True,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
    )
    try:
        assert harness.node._restored_watch == {
            "cursors": {"BTCUSDT": _OWN_MS, "ETHUSDT": _CHECKPOINT_MS},
            "seen": {},
        }
        assert harness.node._restored_watch_ms == _OWN_MS

        watcher = harness.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert len(harness.events(EventType.LIQUIDATION)) == 1
        assert len(harness.alerts.calls) == 1
    finally:
        await harness.node._shutdown()


async def test_start_fresh_keeps_the_own_risk_and_halt_but_merges_the_watch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1ghod/S3: risk and halt stay the own row's; the watch is the union of both."""
    _put(
        seeded["db"],
        "PAPER-007",
        risk=_risk(high_water_mark="10000"),
        halt="manual",
        message="M",
        state="halted",
        extra={"liquidation_watch": {"cursors": {"BTCUSDT": 1_000}, "seen": {}}},
    )
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_risk(high_water_mark="11000"),
        halt="error",
        message="boom",
        extra={"liquidation_watch": {"cursors": {"ETHUSDT": 2_000}, "seen": {}}},
    )

    node, events = _node(monkeypatch, seeded, start_fresh=True)
    with caplog.at_level(logging.WARNING):
        await node._initialize()
    try:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10000"
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert _restores(events)[0]["risk_source_trader_id"] == "PAPER-007"
        assert node._risk_inherited_from is None
        assert "inherited risk state" not in caplog.text
        assert node._restored_watch == {
            "cursors": {"BTCUSDT": 1_000, "ETHUSDT": 2_000},
            "seen": {},
        }
    finally:
        await node._shutdown()


async def test_start_fresh_own_row_only_looks_up_the_newer_row_exactly_once(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1ghod/S4: the lookup runs under start_fresh; the own row supplies the watch."""
    _put(
        seeded["db"],
        "PAPER-007",
        risk=_risk(high_water_mark="10000"),
        extra={"liquidation_watch": _watch(3)},
    )
    row = _latest(seeded["db"], "PAPER-007")
    calls: list[tuple[str, str]] = []
    original = StatePersistence.load_newer_account_checkpoint

    def spy(self: StatePersistence, account: str, trader_id: str) -> StateCheckpoint | None:
        calls.append((account, trader_id))
        return original(self, account, trader_id)

    monkeypatch.setattr(StatePersistence, "load_newer_account_checkpoint", spy)
    node, _events = _node(monkeypatch, seeded, start_fresh=True)
    await node._initialize()
    try:
        assert calls == [(ACCOUNT_A, "PAPER-007")]
        assert node._restored_watch == _watch(3)
        assert node._restored_watch_ms == PaperTradingNode._checkpoint_ms(row.timestamp)
    finally:
        await node._shutdown()


async def test_start_fresh_newer_row_without_watch_key_bounds_the_gap(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1ghod/S5: a pre-watcher newer row still bounds the gap through its timestamp."""
    _put(seeded["db"], "PAPER-041", risk=_risk())
    row = _latest(seeded["db"], "PAPER-041")

    node, _events = _node(monkeypatch, seeded, start_fresh=True)
    await node._initialize()
    try:
        assert node._restored_watch == {"cursors": {}, "seen": {}}
        assert node._restored_watch_ms == PaperTradingNode._checkpoint_ms(row.timestamp)
    finally:
        await node._shutdown()


async def test_start_fresh_other_account_row_supplies_no_watch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1ghod/S6: a newer row of ANOTHER account is invisible to the watch too."""
    _put(
        seeded["db"],
        "PAPER-041",
        account=ACCOUNT_B,
        risk=_risk(),
        extra={"liquidation_watch": _watch(9)},
    )

    node, _events = _node(monkeypatch, seeded, start_fresh=True)
    await node._initialize()
    try:
        assert node._restored_watch is None
        assert node._restored_watch_ms == 0
    finally:
        await node._shutdown()


async def test_start_fresh_unreadable_newer_row_still_fails_closed(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1ghod/S7: the always-on lookup parses strictly, so start_fresh halts ERROR too."""
    row = _put(seeded["db"], "PAPER-041", risk=_risk())
    _sql(
        seeded["db"],
        "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
        "garbage",
        row,
    )

    node, _events = _node(monkeypatch, seeded, start_fresh=True)
    with caplog.at_level(logging.ERROR):
        await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == (
            f"{RESTORED_HALT_PREFIX}checkpoint unreadable (ValueError); {LIMITS_NOTE}"
        )
        errors = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert [record.getMessage() for record in errors] == [
            "paper PAPER-007: checkpoint unreadable (ValueError); halting"
        ]
        assert errors[0].exc_info is None
    finally:
        await node._shutdown()


async def test_start_fresh_unreadable_newer_row_keeps_the_own_watch(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W3/W5: start_fresh still halts ERROR, but the own watch survives the lookup.

    An unreadable newer row (a garbage ``created_at`` parses strictly and raises) must
    not discard the readable own cursors: the halt is preserved AND the liquidation
    between the two sessions is still alerted. A duplicate is acceptable, a miss is not.
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
        stamp="garbage",
        node_status={
            "account": acct,
            "liquidation_watch": {"cursors": {"BTCUSDT": _OWN_MS}, "seen": {}},
        },
    )

    harness = await _harness(
        monkeypatch,
        detector_seeded,
        trader_id=_TRADER_ID,
        start_fresh=True,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
    )
    try:
        assert harness.node.status.state == NodeState.HALTED
        assert harness.node.status.halt_reason == HaltReason.ERROR
        assert harness.node._restored_watch == {
            "cursors": {"BTCUSDT": _OWN_MS},
            "seen": {},
        }
        assert harness.node._restored_watch_ms == _OWN_MS

        watcher = harness.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert len(harness.events(EventType.LIQUIDATION)) == 1
        assert len(harness.alerts.calls) == 1
    finally:
        await harness.node._shutdown()


async def test_restart_unreadable_newer_row_keeps_the_own_watch(
    detector_seeded: dict[str, Any],  # noqa: F811
    detector_fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W5: the same hole exists on a plain restart; the own watch must survive too."""
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
        stamp="garbage",
        node_status={"account": acct},
    )

    harness = await _harness(
        monkeypatch,
        detector_seeded,
        trader_id=_TRADER_ID,
        orders={"BTCUSDT": [_order(update_time_ms=_GAP_MS)]},
    )
    try:
        assert harness.node.status.halt_reason == HaltReason.ERROR
        assert harness.node._restored_watch == {
            "cursors": {"BTCUSDT": _OWN_MS},
            "seen": {},
        }
        assert harness.node._restored_watch_ms == _OWN_MS

        watcher = harness.node._build_liquidation_watcher()
        await watcher.poll_once()

        assert len(harness.events(EventType.LIQUIDATION)) == 1
    finally:
        await harness.node._shutdown()
