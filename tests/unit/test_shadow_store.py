"""Shadow forward tracker store: schema v22 tables, slot assignment, per-row txns, lock.

Design: data/swarm/20261122-edge/designs/arch-c7-2.md (§ Schema, R37, R38 store half).
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import init_schema
from vibe_quant.db.state_manager import (
    ShadowDuplicate,
    ShadowUpdateRunning,
    StateManager,
    shadow_update_lock,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPO = Path(__file__).resolve().parents[2]
_PREREG = "Pre-registered: forward Sharpe > 1 on BTC 4h after costs."  # >= 40 chars
_SHADOW_TABLES = ("shadow_strategies", "shadow_days", "shadow_events")


def _row(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": "rsi-btc",
        "source_kind": "strategy",
        "source_ref": {"strategy_id": 7},
        "strategy_id": 7,
        "dsl_json": '{"name": "rsi-btc"}',
        # one DSL per name: rows that differ only by name are different strategies
        "dsl_sha256": hashlib.sha256(str(over.get("name", "rsi-btc")).encode()).hexdigest(),
        "compiler_hash": "c0ffee",
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "parameters": {"leverage": 1},
        "frozen_identity": {"venue": {"taker_fee": 0.0005}},
        "enrolled_at": "2026-11-22T18:00:00.000000+00:00",
        "first_day": "2026-11-23",
        "run_start": "2026-05-27",
        "family": "F1",
        "pre_registration": _PREREG,
        "max_dd_limit": 0.2,
        "engine_epoch": 1,
        "engine_epoch_last": 1,
    }
    row.update(over)
    return row


@pytest.fixture
def sm(tmp_path: Path) -> Iterator[StateManager]:
    mgr = StateManager(tmp_path / "state.db")
    yield mgr
    mgr.close()


def _snapshot(conn: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    return {
        t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2")]
        for t in _SHADOW_TABLES
    }


# --- schema: fresh and v21 -> v22 ---------------------------------------------------


def _shadow_ddl(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE name LIKE 'shadow_%' AND sql IS NOT NULL"
        )
    }


def test_v21_db_migrates_to_shadow_tables(tmp_path: Path) -> None:
    fresh = get_connection(tmp_path / "fresh.db")
    init_schema(fresh)
    fresh_ddl = _shadow_ddl(fresh)
    fresh.close()
    assert set(_SHADOW_TABLES) <= set(fresh_ddl)

    # v21 = today's schema without the shadow tables, holding user data.
    db = tmp_path / "v21.db"
    old = get_connection(db)
    init_schema(old)
    for table in reversed(_SHADOW_TABLES):
        old.execute(f"DROP TABLE {table}")
    old.execute("INSERT INTO strategies (name, dsl_config) VALUES ('kept', '{\"a\": 1}')")
    old.commit()
    assert _shadow_ddl(old) == {}
    old.close()

    mgr = StateManager(db)
    try:
        assert _shadow_ddl(mgr.conn) == fresh_ddl
        kept = mgr.conn.execute("SELECT name, dsl_config FROM strategies").fetchall()
        assert [tuple(r) for r in kept] == [("kept", '{"a": 1}')]
        sid = mgr.insert_shadow_strategy(_row())
        assert [r["id"] for r in mgr.list_shadow_strategies()] == [sid]
    finally:
        mgr.close()


# --- CHECK / UNIQUE constraints -----------------------------------------------------


def test_slot_unique(sm: StateManager) -> None:
    ids = [sm.insert_shadow_strategy(_row(name=f"s{i}")) for i in range(6)]
    test_id = sm.insert_shadow_strategy(_row(name="backdated", is_test=1))
    other = sm.insert_shadow_strategy(_row(name="other-family", family="F2"))
    slots = {r["id"]: r["slot"] for r in sm.list_shadow_strategies()}
    assert [slots[i] for i in ids] == [1, 2, 3, 4, 5, None]
    assert slots[test_id] is None
    assert slots[other] == 1

    # A spent slot is never handed out again: retiring slot 2 leaves it taken.
    sm.commit_shadow_update(ids[1], new_days=[], fields={"state": "retired"}, events=[])
    after = sm.insert_shadow_strategy(_row(name="after-retire"))
    assert {r["id"]: r["slot"] for r in sm.list_shadow_strategies()}[after] is None

    conn = sm.conn
    cols = (
        "name, source_kind, dsl_json, dsl_sha256, compiler_hash, symbols, timeframe, "
        "parameters, frozen_identity, enrolled_at, first_day, run_start, family, slot, "
        "pre_registration, max_dd_limit, engine_epoch, engine_epoch_last"
    )
    base = ("x", "strategy", "{}", "h", "c", "[]", "4h", "{}", "{}", "t", "d", "d")
    tail = (_PREREG, 0.2, 1, 1)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        conn.execute(
            f"INSERT INTO shadow_strategies ({cols}) VALUES ({','.join('?' * 18)})",
            (*base, "F1", 3, *tail),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(
            f"INSERT INTO shadow_strategies ({cols}) VALUES ({','.join('?' * 18)})",
            (*base, "F9", 6, *tail),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(
            f"INSERT INTO shadow_strategies ({cols}) VALUES ({','.join('?' * 18)})",
            (*base, "F9", 0, *tail),
        )
    conn.rollback()
    with pytest.raises(ValueError, match="slot"):
        sm.insert_shadow_strategy(_row(slot=1))


_SAME = {"dsl_sha256": "cd" * 32, "symbols": ["ETHUSDT", "BTCUSDT"],
         "parameters": {"leverage": 1, "risk_per_trade": 0.5}}  # fmt: skip
_SAME_REORDERED = {"dsl_sha256": "cd" * 32, "symbols": ["BTCUSDT", "ETHUSDT"],
                   "parameters": {"risk_per_trade": 0.5, "leverage": 1}}  # fmt: skip


@pytest.mark.parametrize(
    ("state", "blocks"),
    [("tracking", True), ("stalled", True), ("passed", True), ("futile", True),
     ("retired", False)],
)  # fmt: skip
def test_duplicate_inputs_refused_inside_the_insert(
    sm: StateManager, state: str, blocks: bool
) -> None:
    """clmmc.27: same DSL hash + symbols + parameters as a non-retired non-test row."""
    first = sm.insert_shadow_strategy(_row(name="first", **_SAME))
    sm.commit_shadow_update(first, new_days=[], fields={"state": state}, events=[])
    if blocks:
        refusal = f"already tracked as shadow #{first} (first, {state})"
        with pytest.raises(ShadowDuplicate, match=re.escape(refusal)):
            sm.insert_shadow_strategy(_row(name="second", **_SAME_REORDERED))
        assert sm.shadow_duplicate("cd" * 32, ["BTCUSDT", "ETHUSDT"],
                                   {"risk_per_trade": 0.5, "leverage": 1}) == refusal  # fmt: skip
        assert [r["id"] for r in sm.list_shadow_strategies()] == [first]
    else:
        second = sm.insert_shadow_strategy(_row(name="second", **_SAME_REORDERED))
        assert {r["id"]: r["slot"] for r in sm.list_shadow_strategies()} == {first: 1, second: 2}


def test_duplicate_check_skips_test_rows_and_other_inputs(sm: StateManager) -> None:
    first = sm.insert_shadow_strategy(_row(name="first", **_SAME))
    test_a = sm.insert_shadow_strategy(_row(name="test-a", is_test=1, **_SAME))  # not blocked
    test_b = sm.insert_shadow_strategy(_row(name="test-b", is_test=1, **_SAME))
    other = sm.insert_shadow_strategy(
        _row(name="other", **{**_SAME, "parameters": {"leverage": 2, "risk_per_trade": 0.5}})
    )
    eth = sm.insert_shadow_strategy(_row(name="eth", **{**_SAME, "symbols": ["ETHUSDT"]}))
    slots = {r["id"]: r["slot"] for r in sm.list_shadow_strategies()}
    assert slots == {first: 1, test_a: None, test_b: None, other: 2, eth: 3}
    sm.commit_shadow_update(first, new_days=[], fields={"state": "retired"}, events=[])
    # a test row never blocks: with the only real twin retired, a real row is admitted
    again = sm.insert_shadow_strategy(_row(name="again", **_SAME))
    assert {r["id"]: r["slot"] for r in sm.list_shadow_strategies()}[again] == 4


def test_duplicate_key_folds_number_spellings_but_not_bools(sm: StateManager) -> None:
    """S1: 1 and 1.0 are one parameter; True is not 1; the fold reaches nested values.

    The browser sends JSON numbers without a fraction (1), a Python client the DB's float
    (1.0): one strategy, one evidence slot. A bool stays a bool (True != 1).
    """
    sha = "cd" * 32
    int_row = sm.insert_shadow_strategy(
        _row(name="int", dsl_sha256=sha, parameters={"max_position_pct": 1, "risk_per_trade": 0.5})
    )
    float_row = sm.insert_shadow_strategy(
        _row(name="float", dsl_sha256=sha, parameters={"max_position_pct": 2.0, "risk_per_trade": 0.5})
    )
    bool_row = sm.insert_shadow_strategy(_row(name="bool", dsl_sha256=sha, parameters={"pct": True}))
    edge_row = sm.insert_shadow_strategy(
        _row(name="edge", dsl_sha256=sha, parameters={"max_position_pct": 1.5})
    )
    nested_row = sm.insert_shadow_strategy(
        _row(
            name="nested",
            dsl_sha256=sha,
            parameters={"risk": {"max_position_pct": 1, "sizes": [1, 2.0]}},
        )
    )

    def refused(shadow_id: int, name: str, parameters: dict[str, object]) -> None:
        assert sm.shadow_duplicate(sha, ["BTCUSDT"], parameters) == (
            f"already tracked as shadow #{shadow_id} ({name}, tracking)"
        )
        with pytest.raises(ShadowDuplicate, match=re.escape(f"#{shadow_id} ({name}, tracking)")):
            sm.insert_shadow_strategy(_row(name="twin", dsl_sha256=sha, parameters=parameters))

    # int 1 -> integral float 1.0, and the reverse (a row stored as 2.0 -> queried as 2)
    refused(int_row, "int", {"max_position_pct": 1.0, "risk_per_trade": 0.5})
    refused(float_row, "float", {"max_position_pct": 2, "risk_per_trade": 0.5})
    # a non-integral float matches itself, never a neighbour
    refused(edge_row, "edge", {"max_position_pct": 1.5})
    assert sm.shadow_duplicate(sha, ["BTCUSDT"], {"max_position_pct": 1.5000001}) is None
    # bool is not the int: True matches itself, never 1
    assert sm.shadow_duplicate(sha, ["BTCUSDT"], {"pct": 1}) is None
    refused(bool_row, "bool", {"pct": True})
    # nested dict values (and list elements) fold too
    refused(nested_row, "nested", {"risk": {"max_position_pct": 1.0, "sizes": [1.0, 2]}})
    assert [r["id"] for r in sm.list_shadow_strategies()] == [
        int_row,
        float_row,
        bool_row,
        edge_row,
        nested_row,
    ]


def test_test_row_takes_no_slot(sm: StateManager) -> None:
    test_id = sm.insert_shadow_strategy(_row(name="backdated", is_test=1))
    real = sm.insert_shadow_strategy(_row(name="real"))
    slots = {r["id"]: r["slot"] for r in sm.list_shadow_strategies()}
    assert slots == {test_id: None, real: 1}


@pytest.mark.parametrize(
    ("over", "ok"),
    [
        ({"pre_registration": "x" * 40}, True),
        ({"pre_registration": "  " + "x" * 39 + "   "}, False),
        ({"max_dd_limit": 1.0}, True),
        ({"max_dd_limit": 0.0}, False),
        ({"max_dd_limit": 1.5}, False),
        ({"source_kind": "candidate"}, True),
        ({"source_kind": "paper"}, False),
        ({"state": "stalled"}, True),
        ({"state": "running"}, False),
    ],
)
def test_strategy_checks(sm: StateManager, over: dict[str, object], ok: bool) -> None:
    if ok:
        sm.insert_shadow_strategy(_row(**over))
        assert len(sm.list_shadow_strategies()) == 1
    else:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            sm.insert_shadow_strategy(_row(**over))
        assert sm.list_shadow_strategies() == []


def test_list_and_days_round_trip(sm: StateManager) -> None:
    a = sm.insert_shadow_strategy(_row(name="a", symbols=["BTCUSDT", "ETHUSDT"]))
    b = sm.insert_shadow_strategy(_row(name="b", parameters={"rsi": 14}))
    sm.commit_shadow_update(
        a,
        new_days=[(20415, 0.0125), (20416, -0.0)],
        fields={"committed_through": "2026-11-24", "stats": {"e_value": 1.0, "blocks": 0}},
        events=[{"kind": "note", "message": "first update"}],
    )
    sm.commit_shadow_update(b, new_days=[(20415, -0.003)], fields={}, events=[])
    rows = {r["name"]: r for r in sm.list_shadow_strategies()}
    assert rows["a"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert rows["a"]["stats"] == {"e_value": 1.0, "blocks": 0}
    assert rows["a"]["committed_through"] == "2026-11-24"
    assert rows["b"]["parameters"] == {"rsi": 14}
    assert rows["b"]["stats"] is None
    assert sm.get_shadow_days(a) == [(20415, 0.0125), (20416, -0.0)]
    assert sm.get_shadow_days(b) == [(20415, -0.003)]
    with pytest.raises(ValueError, match="frozen_identity"):
        sm.commit_shadow_update(a, new_days=[], fields={"frozen_identity": {}}, events=[])


# --- all-or-nothing per-row transaction ---------------------------------------------


@pytest.mark.parametrize(
    ("new_days", "fields", "events", "error"),
    [
        # duplicate stored day (rerun over a committed day)
        ([(20416, 0.01), (20415, 0.02)], {"state": "stalled"}, [], sqlite3.IntegrityError),
        # NaN return is stored as NULL -> NOT NULL fails after the first day went in
        ([(20416, 0.01), (20417, float("nan"))], {"state": "stalled"}, [], sqlite3.IntegrityError),
        # last statement (an event) violates its CHECK after days + fields went in
        (
            [(20416, 0.01)],
            {"state": "futile"},
            [{"kind": "state_change", "message": "ok"}, {"kind": "bogus", "message": "x"}],
            sqlite3.IntegrityError,
        ),
        # invalid state CHECK on the UPDATE after the days
        ([(20416, 0.01)], {"state": "running"}, [], sqlite3.IntegrityError),
    ],
)
def test_commit_is_all_or_nothing(
    sm: StateManager,
    new_days: list[tuple[int, float]],
    fields: dict[str, object],
    events: list[dict[str, object]],
    error: type[Exception],
) -> None:
    sid = sm.insert_shadow_strategy(_row())
    sm.commit_shadow_update(
        sid, new_days=[(20415, 0.005)], fields={"committed_through": "2026-11-23"}, events=[]
    )
    before = _snapshot(sm.conn)
    with pytest.raises(error):
        sm.commit_shadow_update(sid, new_days=new_days, fields=fields, events=events)
    assert _snapshot(sm.conn) == before
    # the connection is usable afterwards (no transaction left open)
    sm.commit_shadow_update(sid, new_days=[(20416, 0.01)], fields={}, events=[])
    assert sm.get_shadow_days(sid) == [(20415, 0.005), (20416, 0.01)]


def test_nan_in_json_column_refused(sm: StateManager) -> None:
    sid = sm.insert_shadow_strategy(_row())
    before = _snapshot(sm.conn)
    with pytest.raises(ValueError, match="JSON compliant"):
        sm.commit_shadow_update(sid, new_days=[], fields={"stats": {"x": float("nan")}}, events=[])
    assert _snapshot(sm.conn) == before
    with pytest.raises(ValueError, match="JSON compliant"):
        sm.insert_shadow_strategy(_row(name="nan", parameters={"x": float("nan")}))
    assert _snapshot(sm.conn) == before


def test_commit_unknown_row_writes_nothing(sm: StateManager) -> None:
    sid = sm.insert_shadow_strategy(_row())
    before = _snapshot(sm.conn)
    with pytest.raises(LookupError, match=str(sid + 1)):
        sm.commit_shadow_update(
            sid + 1, new_days=[], fields={}, events=[{"kind": "note", "message": "x"}]
        )
    assert _snapshot(sm.conn) == before


# --- R38 store half: SIGKILL inside row 2's transaction -----------------------------

_KILL_CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from vibe_quant.db.state_manager import StateManager

    sm = StateManager(Path(sys.argv[1]))
    a, b = int(sys.argv[2]), int(sys.argv[3])
    sm.commit_shadow_update(
        a, new_days=[(20415, 0.01), (20416, 0.02)],
        fields={"committed_through": "2026-11-24"},
        events=[{"kind": "note", "message": "row 1 done"}],
    )
    def trace(sql):
        if "shadow_events" in sql:
            print("inside", flush=True)
            import signal
            signal.pause()
    sm.conn.set_trace_callback(trace)
    sm.commit_shadow_update(
        b, new_days=[(20415, -0.01), (20416, 0.03)],
        fields={"committed_through": "2026-11-24", "state": "stalled"},
        events=[{"kind": "state_change", "from_state": "tracking", "to_state": "stalled",
                 "message": "row 2"}],
    )
    print("committed (BAD)", flush=True)
    """
)


def test_kill_between_rows(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    sm = StateManager(db)
    try:
        a = sm.insert_shadow_strategy(_row(name="row1"))
        b = sm.insert_shadow_strategy(_row(name="row2"))
    finally:
        sm.close()
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    child = subprocess.Popen(
        [sys.executable, "-c", _KILL_CHILD, str(db), str(a), str(b)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "inside"
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    assert child.returncode == -signal.SIGKILL

    sm = StateManager(db)
    try:
        rows = {r["name"]: r for r in sm.list_shadow_strategies()}
        assert rows["row1"]["committed_through"] == "2026-11-24"
        assert sm.get_shadow_days(a) == [(20415, 0.01), (20416, 0.02)]
        # row 2: nothing of the killed transaction survived
        assert rows["row2"]["committed_through"] is None
        assert rows["row2"]["state"] == "tracking"
        assert sm.get_shadow_days(b) == []
        msgs = [r[0] for r in sm.conn.execute("SELECT message FROM shadow_events ORDER BY id")]
        assert msgs == ["row 1 done"]
        # rerun: row 2 processed once; replaying row 1's days cannot duplicate them
        sm.commit_shadow_update(
            b,
            new_days=[(20415, -0.01), (20416, 0.03)],
            fields={"committed_through": "2026-11-24"},
            events=[],
        )
        with pytest.raises(sqlite3.IntegrityError):
            sm.commit_shadow_update(a, new_days=[(20416, 0.02)], fields={}, events=[])
        dup = sm.conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM shadow_days GROUP BY shadow_id, day "
            "HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        assert dup == 0
        assert sm.get_shadow_days(b) == [(20415, -0.01), (20416, 0.03)]
    finally:
        sm.close()


# --- alerts / seen --------------------------------------------------------------------


def test_alert_and_seen_bookkeeping(sm: StateManager) -> None:
    real = sm.insert_shadow_strategy(_row(name="real"))
    test = sm.insert_shadow_strategy(_row(name="bt", is_test=1))
    sm.commit_shadow_update(
        real,
        new_days=[],
        fields={"state": "stalled"},
        events=[
            {
                "kind": "state_change",
                "from_state": "tracking",
                "to_state": "stalled",
                "message": "data gap at 2026-11-20 (BTCUSDT)",
            },
            {"kind": "note", "message": "no new data since 2026-11-19"},
        ],
    )
    sm.commit_shadow_update(
        test,
        new_days=[],
        fields={"state": "passed"},
        events=[
            {
                "kind": "state_change",
                "from_state": "tracking",
                "to_state": "passed",
                "message": "passed",
            }
        ],
    )
    pending = sm.pending_shadow_alerts()
    assert [(p["name"], p["is_test"], p["message"]) for p in pending] == [
        ("real", 0, "data gap at 2026-11-20 (BTCUSDT)"),
        ("bt", 1, "passed"),
    ]
    assert sm.mark_shadow_alert_sent([pending[0]["id"]]) == 1
    assert sm.mark_shadow_alert_sent([pending[0]["id"]]) == 0  # stamped once
    assert [p["name"] for p in sm.pending_shadow_alerts()] == ["bt"]
    assert sm.mark_shadow_events_seen() == 2  # state changes only; notes never count
    assert sm.mark_shadow_events_seen() == 0


# --- R37: lock resolves aliases; flock released by the kernel on SIGKILL --------------

_LOCK_CHILD = textwrap.dedent(
    """
    import signal, sys
    from pathlib import Path
    from vibe_quant.db.state_manager import shadow_update_lock

    with shadow_update_lock(Path(sys.argv[1])):
        print("held", flush=True)
        signal.pause()
    """
)


def test_lock_alias(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    db.touch()
    alias = tmp_path / "alias.db"
    alias.symlink_to(db)
    lock_file = Path(str(db.resolve()) + ".shadow.lock")
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    child = subprocess.Popen(
        [sys.executable, "-c", _LOCK_CHILD, str(db)], stdout=subprocess.PIPE, text=True, env=env
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        inode = lock_file.stat().st_ino
        with (
            pytest.raises(ShadowUpdateRunning, match="shadow update already running"),
            shadow_update_lock(alias),
        ):
            pass
        with pytest.raises(ShadowUpdateRunning), shadow_update_lock(db):
            pass
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    with shadow_update_lock(alias):
        assert lock_file.stat().st_ino == inode
    assert lock_file.exists()  # never unlinked
    assert lock_file.stat().st_ino == inode
    assert not Path(str(alias) + ".shadow.lock").exists()
