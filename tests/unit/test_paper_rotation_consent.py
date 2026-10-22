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
from tests.unit.test_paper_halt_restore import (  # noqa: F401  (fixtures / helpers)
    _latest_checkpoint,
    _raises_venue_build,
    fake_tradingnode,
    seeded,
)
from tests.unit.test_paper_risk_inherit import _put, _risk
from vibe_quant.paper import lineage
from vibe_quant.paper.config import BinanceTestnetConfig, account_id
from vibe_quant.paper.guard import GuardState, HaltReason
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
    r36: bool = True  # False: excluded from the agreement invariant (G10, G17, G18)
    error_has: str | None = None
    message: str | None = None  # asserted == node.status.error_message
    status: NodeState | None = None  # asserted == node.status.state
    warning_has: tuple[str, ...] | None = None
    warning_absent: str | None = None
    start_fresh: bool = False


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
    # R4 (D1=A, ae6ah): an older own live-key row carries even under a newer testnet
    # row; the restore logs both "ignored" and the rotation.
    "G4": Case(
        [_manual(K_OLD), Row(TRADER, T_OLD)],
        gate=True,
        source=TRADER,
        halt=HaltReason.MANUAL,
        hwm="10600",
        warning_has=("not " + K_NEW + "; ignored", "key rotation from account"),
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
    # R1 (ae6ah): rotating back A -> B -> A restores the OLDER key's newer row.
    "G13": Case(
        [Row(TRADER, K_NEW, hwm="10000"), _manual(K_OLD)],
        gate=True,
        source=TRADER,
        halt=HaltReason.MANUAL,
        hwm="10600",
        message="restored after restart: halt via API",
        status=NodeState.HALTED,
        warning_has=("key rotation from account " + K_OLD + " to " + K_NEW,),
    ),
    # R2: the same rotation with a non-halted old-key row: risk carries, no halt.
    "G14": Case(
        [Row(TRADER, K_NEW, hwm="10000"), Row(TRADER, K_OLD)],
        gate=True,
        source=TRADER,
        halt=None,
        hwm="10600",
        warning_has=("key rotation from account " + K_OLD + " to " + K_NEW,),
    ),
    # R3: the own newest row wins over an older same-env row: no rotation.
    "G15": Case(
        [_manual(K_OLD), Row(TRADER, K_NEW, hwm="10000")],
        gate=True,
        source=TRADER,
        halt=None,
        hwm="10000",
        warning_absent="key rotation from account",
    ),
    # R7: another trader's newer row on THIS account supplies the risk, the own
    # (rotated) halt still wins (today's precedence, D5).
    "G16": Case(
        [
            Row(TRADER, K_NEW, hwm="10000"),
            Row("PAPER-041", K_NEW, hwm="12000"),
            _manual(K_OLD),
        ],
        gate=True,
        source="PAPER-041",
        halt=HaltReason.MANUAL,
        hwm="12000",
    ),
    # R7-fresh: start_fresh skips the other trader's risk, the own rotation carries.
    "G16-fresh": Case(
        [
            Row(TRADER, K_NEW, hwm="10000"),
            Row("PAPER-041", K_NEW, hwm="12000"),
            _manual(K_OLD),
        ],
        gate=True,
        source=TRADER,
        halt=HaltReason.MANUAL,
        hwm="10600",
        start_fresh=True,
    ),
    # R8 (D8): a newer own row with a non-string account fails closed; the gate still
    # sees the older own row on this key.
    "G17": Case(
        [Row(TRADER, K_NEW, hwm="10000"), Row(TRADER, raw='{"account": 5}')],
        gate=True,
        source=None,
        halt=HaltReason.ERROR,
        error_has="checkpoint unreadable (TypeError)",
        r36=False,
    ),
    # R9: an unparseable newest row fails closed; the gate reads the older row.
    "G18": Case(
        [Row(TRADER, K_NEW), Row(TRADER, raw="{not json")],
        gate=True,
        source=None,
        halt=HaltReason.ERROR,
        error_has="checkpoint unreadable (JSONDecodeError)",
        r36=False,
    ),
    # R10: the newest own row is a legacy row (no account): it is the source.
    "G19": Case(
        [_manual(K_OLD), Row(TRADER, None, hwm="11000")],
        gate=True,
        source=TRADER,
        halt=None,
        hwm="11000",
        warning_absent="key rotation from account",
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
    *,
    binance: BinanceTestnetConfig | None = None,
    start_fresh: bool = False,
) -> tuple[PaperTradingNode, list[dict[str, object]], StatePersistence]:
    """A LIVE node on the NEW key with its persistence set, restore events captured."""
    node = PaperTradingNode(
        _config(
            seeded,
            trader_id=TRADER,
            binance=binance or BinanceTestnetConfig("new-key", "s", testnet=False),
            start_fresh=start_fresh,
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
    """R1-R11: the gate says "consent needed" exactly when nothing carries."""
    case = CASES[name]
    _seed(seeded["db"], case.rows)

    with caplog.at_level(logging.WARNING):
        node, events, persistence = _restore(seeded, start_fresh=case.start_fresh)
    try:
        gate = lineage.has_prior_state(persistence, K_NEW, TRADER)
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
        if case.message is not None:
            assert node.status.error_message == case.message
        if case.status is not None:
            assert node.status.state == case.status
        for text in case.warning_has or ():
            assert text in caplog.text
        if case.warning_absent is not None:
            assert case.warning_absent not in caplog.text
    finally:
        persistence.close()
        node._state_manager.close()


def test_testnet_start_ignores_live_rows(
    seeded: dict[str, Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R5: a testnet start never mixes the live env's rows into its risk."""
    _seed(seeded["db"], [Row(TRADER, T_OLD, hwm="10000"), _manual(K_OLD)])

    with caplog.at_level(logging.WARNING):
        node, events, persistence = _restore(
            seeded, binance=BinanceTestnetConfig("old-key", "s", testnet=True)
        )
    try:
        assert node._restored_halt is None
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict()["high_water_mark"] == "10000"
        assert "key rotation" not in caplog.text
        sources = [e["risk_source_trader_id"] for e in events if e.get("action") == "restore"]
        assert sources == [TRADER]
    finally:
        persistence.close()
        node._state_manager.close()


async def test_rotate_back_survives_two_failed_restarts(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6: a failed restart checkpoints the rotated risk on the NEW key; restart 2
    restores that own row (env-latest, no rotation) and restart 3 comes up
    HALTED/MANUAL with the carried high water mark."""
    _seed(seeded["db"], CASES["G13"].rows)
    config = _config(
        seeded,
        trader_id=TRADER,
        binance=BinanceTestnetConfig("new-key", "s", testnet=False),
    )

    def gate() -> bool:
        persistence = StatePersistence(seeded["db"], trader_id=TRADER)
        try:
            return lineage.has_prior_state(persistence, K_NEW, TRADER)
        finally:
            persistence.close()

    assert gate() is True
    for restart in (1, 2):
        node = PaperTradingNode(config)
        monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
        monkeypatch.setattr(node, "_create_live_trading_node", _raises_venue_build)
        with pytest.raises(RuntimeError, match="venue build failed"):
            await node.start()
        assert gate() is True, restart
        latest = _latest_checkpoint(seeded["db"], trader_id=TRADER)
        assert latest is not None
        assert latest.node_status["account"] == K_NEW
        assert latest.node_status["halt_reason"] == "manual"
        assert latest.node_status["risk"]["high_water_mark"] == "10600"
        if restart == 2:
            assert node._restored_risk is not None
            assert node._restored_risk.to_dict()["high_water_mark"] == "10600"
            assert node._restored_halt is not None
            assert node._restored_halt.reason is HaltReason.MANUAL

    node3 = PaperTradingNode(config)
    await node3._initialize()
    try:
        assert gate() is True
        assert node3._guard is not None
        assert node3._guard.guard_state is GuardState.HALTED
        assert node3._guard.halt_reason is HaltReason.MANUAL
        assert node3._restored_risk is not None
        assert node3._restored_risk.to_dict()["high_water_mark"] == "10600"
    finally:
        await node3._shutdown()



def test_rotation_carries_halt_into_the_node_state(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R23: the rotated halt arrives HALTED/MANUAL with the carried message."""
    _seed(seeded["db"], CASES["G3"].rows)
    node, _events, persistence = _restore(seeded)
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert lineage.has_prior_state(persistence, K_NEW, TRADER) is True
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
    assert lineage.is_key_rotation(newest, account) is expected


@pytest.mark.parametrize("bad", [5, 1.5, True, ["binance:live:a"], {"a": 1}])
def test_is_key_rotation_rejects_a_non_string_account(bad: object) -> None:
    with pytest.raises(TypeError):
        lineage.is_key_rotation(bad, K_NEW)


def test_account_env_is_venue_and_env() -> None:
    assert lineage.account_env("binance:testnet:891a21f36b950681") == "binance:testnet"
    assert lineage.account_env(K_NEW) == "binance:live"


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
