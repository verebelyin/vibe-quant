"""Checkpoint lineage lookups live in one module (vibe-quant-rabmo.1, T9).

The restore and the LIVE consent gate ask the same questions of the same rows:
which of this trader's checkpoints belongs to this account, which row is the
newest in the venue + env (a key rotation), which row of another trader is newer,
and whether anything carries at all. This table pins the module that answers them
over the store connection (same SQL, same return semantics) and the two rotated
rows' helpers (``account_env``, ``is_key_rotation``) it now owns.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from vibe_quant.paper import lineage
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

TRADER = "OWN-001"
OTHER = "OTHER-001"
ACCOUNT_A = "binance:live:aaaa"
ACCOUNT_B = "binance:live:bbbb"
ACCOUNT_T = "binance:testnet:aaaa"


@pytest.fixture
def store(tmp_path: Path) -> Generator[StatePersistence]:
    p = StatePersistence(tmp_path / "lineage.db", trader_id=TRADER)
    yield p
    p.close()


def _save(store: StatePersistence, trader_id: str, node_status: dict[str, object]) -> None:
    store.save_checkpoint(StateCheckpoint(trader_id=trader_id, node_status=node_status))


def _save_raw(store: StatePersistence, trader_id: str, node_status: str) -> None:
    store.conn.execute(
        """INSERT INTO paper_trading_checkpoints
           (trader_id, positions, orders, balance, node_status)
           VALUES (?, '{}', '{}', '{}', ?)""",
        (trader_id, node_status),
    )
    store.conn.commit()


def test_module_owns_the_six_lookups() -> None:
    """One module, six functions: three reads, the gate, and the rotation helpers."""
    for name in (
        "latest_own",
        "latest_in_env",
        "newer_other",
        "has_prior_state",
        "account_env",
        "is_key_rotation",
    ):
        assert callable(getattr(lineage, name, None)), name


def test_every_lookup_takes_the_store_first() -> None:
    """The connection is the caller's: no hidden store, no module-level connection."""
    for name in ("latest_own", "latest_in_env", "newer_other", "has_prior_state"):
        first = next(iter(inspect.signature(getattr(lineage, name)).parameters))
        assert first == "store", name


def test_store_no_longer_owns_the_lineage_lookups() -> None:
    """The move is complete: one copy of each rule, in lineage."""
    for name in (
        "load_latest_account_checkpoint",
        "load_latest_env_checkpoint",
        "load_newer_account_checkpoint",
        "has_account_checkpoint",
    ):
        assert not hasattr(StatePersistence, name), name


def test_row_to_checkpoint_is_public() -> None:
    """R2 F8: one parser, public, so lineage does not carry a second copy."""
    assert callable(StatePersistence.row_to_checkpoint)
    assert not hasattr(StatePersistence, "_row_to_checkpoint")


@pytest.mark.parametrize(
    ("account", "expected"),
    [
        ("binance:live:aaaa", "binance:live"),
        ("binance:testnet:891a21f36b950681", "binance:testnet"),
        ("other:live:aaaa", "other:live"),
        ("no-colon", ""),
    ],
)
def test_account_env_is_the_venue_and_env(account: str, expected: str) -> None:
    assert lineage.account_env(account) == expected


@pytest.mark.parametrize(
    ("newest", "account", "expected"),
    [
        (ACCOUNT_B, ACCOUNT_A, True),
        (ACCOUNT_A, ACCOUNT_A, False),
        (ACCOUNT_T, ACCOUNT_A, False),
        ("other:live:aaaa", ACCOUNT_A, False),
        ("binance:live:aaaa", "binance:live:bbbb", True),
        (None, ACCOUNT_A, False),
    ],
)
def test_is_key_rotation_is_another_key_of_the_same_env(
    newest: object, account: str, expected: bool
) -> None:
    assert lineage.is_key_rotation(newest, account) is expected


@pytest.mark.parametrize("bad", [5, 1.5, True, ["binance:live:a"], {"a": 1}])
def test_is_key_rotation_rejects_a_non_string_account(bad: object) -> None:
    """The restore fails closed on a corrupt row; the gate treats it as "no rotation"."""
    with pytest.raises(TypeError):
        lineage.is_key_rotation(bad, ACCOUNT_A)


def test_latest_own_returns_this_accounts_newest_row(store: StatePersistence) -> None:
    """Two unrelated inputs: a legacy row and an account row, newest wins."""
    _save(store, TRADER, {"state": "legacy"})
    _save(store, TRADER, {"account": ACCOUNT_A, "state": "account"})
    own = lineage.latest_own(store, TRADER, ACCOUNT_A)
    assert own is not None
    assert own.node_status == {"account": ACCOUNT_A, "state": "account"}
    assert own.trader_id == TRADER


def test_latest_own_skips_another_account(store: StatePersistence) -> None:
    _save(store, TRADER, {"account": ACCOUNT_A, "state": "own"})
    _save(store, TRADER, {"account": ACCOUNT_T, "state": "testnet"})
    own = lineage.latest_own(store, TRADER, ACCOUNT_A)
    assert own is not None
    assert own.node_status == {"account": ACCOUNT_A, "state": "own"}


def test_latest_in_env_returns_the_newest_key_of_the_env(store: StatePersistence) -> None:
    _save(store, TRADER, {"account": ACCOUNT_A})
    _save(store, TRADER, {"account": ACCOUNT_B})
    latest = lineage.latest_in_env(store, TRADER, ACCOUNT_A)
    assert latest is not None
    assert latest.node_status == {"account": ACCOUNT_B}


def test_latest_in_env_never_crosses_the_env(store: StatePersistence) -> None:
    _save(store, TRADER, {"account": ACCOUNT_T})
    assert lineage.latest_in_env(store, TRADER, ACCOUNT_A) is None
    _save(store, TRADER, {"account": ACCOUNT_A})
    latest = lineage.latest_in_env(store, TRADER, ACCOUNT_A)
    assert latest is not None
    assert latest.node_status == {"account": ACCOUNT_A}


def test_newer_other_returns_a_newer_row_of_another_trader(store: StatePersistence) -> None:
    _save(store, TRADER, {"account": ACCOUNT_A, "state": "own"})
    _save(store, OTHER, {"account": ACCOUNT_A, "state": "other"})
    newer = lineage.newer_other(store, ACCOUNT_A, TRADER)
    assert newer is not None
    assert newer.trader_id == OTHER
    assert newer.node_status == {"account": ACCOUNT_A, "state": "other"}


def test_newer_other_ignores_older_and_other_accounts(store: StatePersistence) -> None:
    _save(store, OTHER, {"account": ACCOUNT_A, "state": "older"})
    _save(store, TRADER, {"account": ACCOUNT_A, "state": "own"})
    _save(store, OTHER, {"account": ACCOUNT_B, "state": "another-account"})
    assert lineage.newer_other(store, ACCOUNT_A, TRADER) is None


def test_has_prior_state_is_false_on_an_empty_store(store: StatePersistence) -> None:
    assert lineage.has_prior_state(store, ACCOUNT_A, TRADER) is False


def test_has_prior_state_counts_another_trader_on_the_account(store: StatePersistence) -> None:
    _save(store, OTHER, {"account": ACCOUNT_A})
    assert lineage.has_prior_state(store, ACCOUNT_A, TRADER) is True


def test_has_prior_state_counts_the_own_legacy_row(store: StatePersistence) -> None:
    _save(store, TRADER, {"state": "legacy"})
    assert lineage.has_prior_state(store, ACCOUNT_A, TRADER) is True


def test_has_prior_state_counts_an_own_key_rotation(store: StatePersistence) -> None:
    """R23: the same trader id on another key of the env carries its risk and halt."""
    _save(store, TRADER, {"account": ACCOUNT_B})
    assert lineage.has_prior_state(store, ACCOUNT_A, TRADER) is True


def test_has_prior_state_ignores_another_env_and_another_traders_rotation(
    store: StatePersistence,
) -> None:
    _save(store, OTHER, {"account": ACCOUNT_B})
    _save(store, TRADER, {"account": ACCOUNT_T})
    assert lineage.has_prior_state(store, ACCOUNT_A, TRADER) is False


@pytest.mark.parametrize("raw", ['{"account": 5}', '{"account": ["binance:live:bbbb"]}'])
def test_has_prior_state_never_raises_for_a_non_string_account(
    store: StatePersistence, raw: str
) -> None:
    """R32: a corrupt account value answers "no rotation" instead of raising."""
    _save_raw(store, TRADER, raw)
    assert lineage.has_prior_state(store, ACCOUNT_A, TRADER) is False


@pytest.mark.parametrize("raw", ["{not json"])
def test_has_prior_state_never_raises_for_an_unreadable_row(
    store: StatePersistence, raw: str
) -> None:
    """R30: an unparseable newest own row cannot prove a rotation."""
    _save(store, TRADER, {"account": ACCOUNT_B})
    _save_raw(store, TRADER, raw)
    assert lineage.has_prior_state(store, ACCOUNT_A, TRADER) is False


@pytest.mark.parametrize(
    ("trader_id", "node_status", "expected"),
    [
        (TRADER, {"account": ACCOUNT_A, "state": "running"}, "running"),
        (OTHER, {"account": ACCOUNT_B, "state": "halted"}, "halted"),
    ],
)
def test_latest_own_parses_the_whole_row_through_the_public_parser(
    store: StatePersistence, trader_id: str, node_status: dict[str, object], expected: str
) -> None:
    """The store's one parser fills every field the restore reads, twice, literally."""
    _save(store, trader_id, node_status)
    own = lineage.latest_own(store, trader_id, str(node_status["account"]))
    assert own is not None
    assert own.trader_id == trader_id
    assert own.positions == {}
    assert own.orders == {}
    assert own.balance == {}
    assert own.node_status["state"] == expected
    assert isinstance(own.timestamp, datetime)
