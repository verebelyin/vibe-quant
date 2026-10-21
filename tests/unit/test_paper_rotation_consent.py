"""The LIVE consent gate and the restore read the same row with one rotation rule.

Rows R21-R40 of design L3 (vibe-quant-neuvu.2, ruling Q9=A): a key rotation by an
EXISTING trader id (same trader_id, new LIVE key in the same venue + env, with a
prior own row on the old key) carries its risk state and halt (the restore does
since qa42q), so the consent gate must not ask for ``confirm_fresh_live``. A new
trader id on a new key has nothing to carry, so it still asks. The invariant R36 is
``gate == (the restore found a risk source)``, walked over every seedable row table.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_halt_restore import seeded  # noqa: F401  (fixture)
from tests.unit.test_paper_risk_inherit import _put, _risk
from vibe_quant.paper.config import BinanceTestnetConfig, account_env, account_id, is_key_rotation
from vibe_quant.paper.guard import HaltReason
from vibe_quant.paper.node import NodeState, PaperTradingNode
from vibe_quant.paper.persistence import StatePersistence

if TYPE_CHECKING:
    from pathlib import Path

TRADER = "LIVE-001"
K_NEW = account_id(BinanceTestnetConfig("new-key", "s", testnet=False))
K_OLD = account_id(BinanceTestnetConfig("old-key", "s", testnet=False))
T_OLD = account_id(BinanceTestnetConfig("old-key", "s", testnet=True))


def test_account_vectors() -> None:
    """Pin the account ids the table below is written in."""
    assert K_NEW.startswith("binance:live:")
    assert K_OLD.startswith("binance:live:")
    assert T_OLD.startswith("binance:testnet:")
    assert len({K_NEW, K_OLD}) == 2


@dataclass
class Row:
    """One checkpoint row: ``raw`` bypasses the JSON encoder (invalid / non-object)."""

    trader: str
    account: str | None = None
    halt: str | None = None
    hwm: str = "10600"
    raw: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class Case:
    rows: list[Row]
    gate: bool
    source: str | None  # risk_source_trader_id of the restore event (None = nothing carried)
    halt: HaltReason | None = None
    hwm: str | None = None
    r36: bool = True  # False: excluded from the agreement invariant (G10)
    error_has: str | None = None
    warning_has: str | None = None


def _manual(account: str, *, hwm: str = "10600", trader: str = TRADER) -> Row:
    return Row(trader, account, halt="manual", hwm=hwm)


CASES: dict[str, Case] = {
    # R21: nothing saved -> consent, "no prior session"
    "G1": Case([], gate=False, source=None),
    # R22: own row on the new key
    "G2": Case([Row(TRADER, K_NEW)], gate=True, source=TRADER, hwm="10600"),
    # R23: own row on the OLD live key, halted manual -> carries, gate True (red today)
    "G3": Case(
        [_manual(K_OLD)], gate=True, source=TRADER, halt=HaltReason.MANUAL, hwm="10600"
    ),
    # R25: own old-key row, then an own testnet row (newest): nothing carries
    "G4": Case(
        [_manual(K_OLD), Row(TRADER, T_OLD)],
        gate=False,
        source=None,
        warning_has="not " + K_NEW + "; ignored",
    ),
    # R26: own testnet row only
    "G5": Case([Row(TRADER, T_OLD)], gate=False, source=None),
    # R27: another trader's old-key row only
    "G6": Case([Row("PAPER-041", K_OLD)], gate=False, source=None),
    # R28: another trader on the new key -> inherited
    "G7": Case([Row("PAPER-041", K_NEW)], gate=True, source="PAPER-041", hwm="10600"),
    # R29: own legacy row without an account key
    "G8": Case([Row(TRADER, None)], gate=True, source=TRADER, hwm="10600"),
    # R30: own newest row is not JSON -> fail-closed halt, consent stays truthful
    "G9": Case(
        [Row(TRADER, raw="{not json")],
        gate=False,
        source=None,
        halt=HaltReason.ERROR,
        error_has="checkpoint unreadable (JSONDecodeError)",
    ),
    # R31: own newest row is JSON but not an object: legacy branch, fail-closed halt
    "G10": Case(
        [Row(TRADER, raw="[1]")],
        gate=True,
        source=None,
        halt=HaltReason.ERROR,
        error_has="checkpoint unreadable (AttributeError)",
        r36=False,
    ),
    # R32: own newest row with a non-string account
    "G11": Case(
        [Row(TRADER, raw='{"account": 5}')],
        gate=False,
        source=None,
        halt=HaltReason.ERROR,
        error_has="checkpoint unreadable (TypeError)",
    ),
    # R33: own old-key row, then another trader on the new key (newer): risk from the
    # other trader, the own (rotated) halt wins
    "G12": Case(
        [_manual(K_OLD, hwm="9000"), Row("PAPER-041", K_NEW, hwm="12000")],
        gate=True,
        source="PAPER-041",
        halt=HaltReason.MANUAL,
        hwm="12000",
    ),
}


def _seed(db: Path, rows: list[Row]) -> None:
    for row in rows:
        if row.raw is not None:
            store = StatePersistence(db, trader_id=row.trader)  # creates the table
            try:
                store.conn.execute(
                    "INSERT INTO paper_trading_checkpoints "
                    "(trader_id, positions, orders, balance, node_status) "
                    "VALUES (?, '{}', '{}', '{}', ?)",
                    (row.trader, row.raw),
                )
                store.conn.commit()
            finally:
                store.close()
            continue
        _put(
            db,
            row.trader,
            account=row.account,
            risk=_risk(high_water_mark=row.hwm),
            halt=row.halt,
            message="halt via API" if row.halt else None,
            state="halted" if row.halt else "stopped",
        )


def _restore(
    seeded: dict[str, Any],  # noqa: F811
) -> tuple[PaperTradingNode, list[dict[str, object]], StatePersistence]:
    """A LIVE node on the NEW key with its persistence set, restore events captured."""
    node = PaperTradingNode(
        _config(
            seeded,
            trader_id=TRADER,
            binance=BinanceTestnetConfig("new-key", "s", testnet=False),
        )
    )
    persistence = StatePersistence(seeded["db"], trader_id=TRADER)
    node._persistence = persistence
    events: list[dict[str, object]] = []
    node._write_event = lambda t, d, timestamp=None: events.append(d)  # type: ignore[assignment]
    node._restore_from_checkpoint()
    return node, events, persistence


@pytest.mark.parametrize("name", list(CASES))
def test_gate_and_restore_agree_on_risk_carry(
    name: str,
    seeded: dict[str, Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R21-R33 + R36: the gate says "consent needed" exactly when nothing carries."""
    case = CASES[name]
    _seed(seeded["db"], case.rows)

    with caplog.at_level(logging.WARNING):
        node, events, persistence = _restore(seeded)
    try:
        gate = persistence.has_account_checkpoint(K_NEW, TRADER)
        sources = [e["risk_source_trader_id"] for e in events if e.get("action") == "restore"]
        carried = any(s is not None for s in sources)

        assert gate is case.gate
        assert (sources[0] if sources else None) == case.source
        if case.r36:
            assert gate == carried
        halt = node._restored_halt.reason if node._restored_halt is not None else None
        assert halt == case.halt
        if case.hwm is not None:
            assert node._restored_risk is not None
            assert node._restored_risk.to_dict()["high_water_mark"] == case.hwm
        if case.source is None:
            assert not carried
        if case.error_has is not None:
            assert node.status.state == NodeState.HALTED
            assert node.status.halt_reason == HaltReason.ERROR
            assert node.status.error_message is not None
            assert case.error_has in node.status.error_message
        if case.warning_has is not None:
            assert case.warning_has in caplog.text
    finally:
        persistence.close()
        node._state_manager.close()


def test_rotation_carries_halt_into_the_node_state(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R23: the rotated halt arrives HALTED/MANUAL with the carried message."""
    _seed(seeded["db"], CASES["G3"].rows)
    node, _events, persistence = _restore(seeded)
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert persistence.has_account_checkpoint(K_NEW, TRADER) is True
    finally:
        persistence.close()
        node._state_manager.close()


# ------------------------------------------------------------------ rule itself


@pytest.mark.parametrize(
    ("newest", "account", "expected"),
    [
        (None, K_NEW, False),
        (K_NEW, K_NEW, False),  # same account: not a rotation
        (K_OLD, K_NEW, True),  # same venue + env, other key
        (T_OLD, K_NEW, False),  # other env
        ("binance:live:aaaa", "binance:live:bbbb", True),
        ("other:live:aaaa", "binance:live:bbbb", False),  # other venue
    ],
)
def test_is_key_rotation_rule(newest: object, account: str, expected: bool) -> None:
    assert is_key_rotation(newest, account) is expected


@pytest.mark.parametrize("bad", [5, 1.5, True, ["binance:live:a"], {"a": 1}])
def test_is_key_rotation_rejects_a_non_string_account(bad: object) -> None:
    with pytest.raises(TypeError):
        is_key_rotation(bad, K_NEW)


def test_account_env_is_venue_and_env() -> None:
    assert account_env("binance:testnet:891a21f36b950681") == "binance:testnet"
    assert account_env(K_NEW) == "binance:live"


def test_raw_insert_helper_matches_a_real_row(seeded: dict[str, Any]) -> None:  # noqa: F811
    """Guard the table above: the raw rows really are in the table the gate reads."""
    _seed(seeded["db"], CASES["G9"].rows + CASES["G10"].rows)
    con = sqlite3.connect(seeded["db"])
    try:
        stored = con.execute(
            "SELECT node_status FROM paper_trading_checkpoints ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    assert stored == [("{not json",), ("[1]",)]
