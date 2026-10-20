"""vibe-quant-1zh0m: the liquidation watch follows the risk source row.

A new paper session on an account inherits the risk bookkeeping from the newest
checkpoint of another trader on that account (3fqw9), but the liquidation watch
state (forceOrders cursors + dedupe keys) was only restored from the trader's OWN
row. So a new trader id started scanning at node start and never saw the venue
liquidations that happened between the two sessions.

The watch now comes from the SAME source row as ``RiskState``, MERGED with the own
row: the newer row wins for a symbol both name, a symbol only the own row names
keeps its own cursor, and the gap boundary is the older of the two timestamps (so
a symbol neither row names still rescans from before the gap). ``start_fresh``
still skips the cross-session lookup (the source is then the own row, unchanged).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_halt_restore import seeded  # noqa: F401  (fixture)
from tests.unit.test_paper_liquidation_detector import (
    _CHECKPOINT_MS,
    _CHECKPOINT_STAMP,
    _TRADER_ID,
    _harness,
    _order,
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
    _risk,
    fake_tradingnode,  # noqa: F401  (fixture)
)
from vibe_quant.logging.events import EventType
from vibe_quant.paper.config import account_id
from vibe_quant.paper.node import PaperTradingNode

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


async def test_start_fresh_does_not_inherit_the_watch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R11: start_fresh skips the account lookup, so there is no watch to inherit."""
    _put(
        seeded["db"],
        "PAPER-041",
        risk=_risk(),
        extra={"liquidation_watch": _watch(_WATCH_CURSOR)},
    )

    node, _events = _node(monkeypatch, seeded, start_fresh=True)
    await node._initialize()
    try:
        assert node._restored_watch is None
        assert node._restored_watch_ms == 0
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
