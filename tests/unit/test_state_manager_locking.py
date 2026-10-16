"""Locking invariants for StateManager's shared sqlite connection.

``StateManager`` opens ONE ``sqlite3.Connection`` (``check_same_thread=False``)
that is shared by the FastAPI threadpool, the extraction worker/drain threads
and the scrape subprocesses. Every read therefore races every write on the same
cursor state; an interleaved read surfaces as
``sqlite3.InterfaceError: bad parameter or other API misuse`` and then wedges the
connection for all later callers (a 20-30 s stall per query).

The invariant enforced here: every *read* of ``self.conn`` / ``self._conn`` is
serialized by ``with self._write_lock:``.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import vibe_quant.db.state_manager as _state_manager_module

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager

_SRC_PATH = Path(_state_manager_module.__file__)

# ``conn`` creates the connection under its own lock and ``close`` tears the
# object down; both are excluded. Methods ending in ``_locked`` are the
# documented "caller already holds the write lock" helpers, so their own bodies
# are exempt -- but every *call site* of them must itself hold the lock
# (checked by ``test_locked_helper_call_sites_hold_the_write_lock``).
_EXEMPT_METHODS = {"conn", "close"}


def _is_write_lock_with(node: ast.With) -> bool:
    """True for ``with self._write_lock:`` (possibly multi-item)."""
    return any(
        isinstance(item.context_expr, ast.Attribute)
        and item.context_expr.attr == "_write_lock"
        and isinstance(item.context_expr.value, ast.Name)
        and item.context_expr.value.id == "self"
        for item in node.items
    )


def _is_self_conn_access(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in {"conn", "_conn"}
    )


def _scan_method(method: ast.FunctionDef) -> tuple[list[int], list[int]]:
    """Return (unlocked conn reads, unlocked ``*_locked`` calls) line numbers.

    A method whose name ends in ``_locked`` is treated as already inside the
    write lock (its callers hold it), so both lists stay empty for it.
    """
    conn_reads: list[int] = []
    locked_calls: list[int] = []
    baseline = method.name.endswith("_locked")

    def walk(node: ast.AST, inside_lock: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_inside = inside_lock
            if isinstance(child, ast.With) and _is_write_lock_with(child):
                child_inside = True
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.ctx, ast.Load)
                and _is_self_conn_access(child)
                and not child_inside
            ):
                conn_reads.append(child.lineno)
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "self"
                and child.func.attr.endswith("_locked")
                and not child_inside
            ):
                locked_calls.append(child.lineno)
            walk(child, child_inside)

    walk(method, baseline)
    return conn_reads, locked_calls


def _state_manager_methods(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "StateManager"
    )
    return {
        node.name: node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _analyze(source: str) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    methods = _state_manager_methods(ast.parse(source))
    conn_offenders: dict[str, list[int]] = {}
    call_offenders: dict[str, list[int]] = {}
    for name, method in methods.items():
        if name in _EXEMPT_METHODS:
            continue
        conn_reads, locked_calls = _scan_method(method)
        if conn_reads:
            conn_offenders[name] = conn_reads
        if locked_calls:
            call_offenders[name] = locked_calls
    return conn_offenders, call_offenders


def test_every_conn_read_is_serialized_by_write_lock() -> None:
    conn_offenders, _ = _analyze(_SRC_PATH.read_text())
    assert conn_offenders == {}, (
        "StateManager methods read the shared sqlite connection without holding "
        f"self._write_lock: {conn_offenders}"
    )


def test_locked_helper_call_sites_hold_the_write_lock() -> None:
    _, call_offenders = _analyze(_SRC_PATH.read_text())
    assert call_offenders == {}, (
        "helper methods ending in _locked are called without the write lock "
        f"being held: {call_offenders}"
    )


_GOOD = """
class StateManager:
    def get_x(self):
        with self._write_lock:
            return self.conn.execute("SELECT 1").fetchone()

    def _helper_locked(self):
        return self.conn.execute("SELECT 1").fetchone()

    def public(self):
        with self._write_lock:
            return self._helper_locked()
"""

_BAD_CONN = """
class StateManager:
    def get_x(self):
        return self.conn.execute("SELECT 1").fetchone()
"""

_BAD_CALL = """
class StateManager:
    def _helper_locked(self):
        return self.conn.execute("SELECT 1").fetchone()

    def public(self):
        return self._helper_locked()
"""


def test_analyzer_flags_unlocked_connection_read() -> None:
    conn_offenders, _ = _analyze(_BAD_CONN)
    assert set(conn_offenders) == {"get_x"}
    assert conn_offenders["get_x"]


def test_analyzer_flags_locked_helper_called_unlocked() -> None:
    _, call_offenders = _analyze(_BAD_CALL)
    assert set(call_offenders) == {"public"}
    assert call_offenders["public"]


def test_analyzer_accepts_correct_locking() -> None:
    assert _analyze(_GOOD) == ({}, {})


def _seed_jobs(sm: StateManager, n_jobs: int) -> int:
    item_id = sm.create_research_item(
        source="reddit",
        external_id="stress-1",
        url="https://example.com/1",
        title="stress",
        body="body",
        author="author",
        posted_at=None,
        score=1,
    )
    for _ in range(n_jobs):
        sm.enqueue_extraction_job(item_id)
    return item_id


def test_concurrent_readers_and_writer_do_not_corrupt_connection(
    tmp_state_manager: StateManager,
) -> None:
    sm = tmp_state_manager
    n_jobs = 40
    item_id = _seed_jobs(sm, n_jobs)

    errors: list[Exception] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            for _ in range(200):
                sm.list_extraction_queue()
                sm.list_extraction_jobs()
                sm.get_research_item(item_id)
                sm.count_active_extraction_jobs()
        except Exception as exc:  # pragma: no cover - only on the bug
            errors.append(exc)

    def writer() -> None:
        try:
            while not stop.is_set():
                job = sm.claim_next_extraction_job()
                if job is None:
                    return
                sm.complete_extraction_job(int(job["id"]), status="done")
        except Exception as exc:  # pragma: no cover - only on the bug
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(8)]
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    for thread in readers:
        thread.start()
    for thread in readers:
        thread.join()
    stop.set()
    writer_thread.join()

    assert errors == [], errors
    assert sm.count_active_extraction_jobs() == 0
    assert len(sm.list_extraction_jobs(status="done")) == n_jobs


def test_close_waits_for_an_in_flight_connection_use(tmp_state_manager: StateManager) -> None:
    """``close()`` must not tear the connection down under a thread that is mid-query.

    Closing a ``check_same_thread=False`` connection while another thread is
    inside ``connection.execute`` segfaults CPython's sqlite3 (0x38 NULL deref in
    ``_pysqlite_query_execute``). A reader owns ``_write_lock`` for the whole
    query, so ``close()`` has to queue behind it.
    """
    sm = tmp_state_manager
    sm.create_strategy("closing", {"x": 1})

    sm._write_lock.acquire()  # stand-in for a reader that is mid-query
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (sm.close(), closed.set()))
    closer.start()
    try:
        assert not closed.wait(timeout=0.3), "close() ran while a query held the write lock"
        assert sm._conn is not None
    finally:
        sm._write_lock.release()
    closer.join(timeout=5)
    assert closed.is_set()
    assert sm._conn is None


def test_close_waits_for_connection_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``close()`` must also queue behind ``conn`` while it initializes.

    ``conn`` opens a connection and runs ``init_schema`` under ``_lock``
    (publishing ``self._conn`` only afterwards); a ``close()`` that takes only
    ``_write_lock`` can close that half-built connection and leave the opening
    thread with ``conn`` returning ``None``. An ``init_schema`` wrapper that pauses after the connection exists
    makes the race deterministic instead of probabilistic.
    """
    sm = _state_manager_module.StateManager(tmp_path / "init-race.db")
    real_init_schema = _state_manager_module.init_schema
    in_init = threading.Event()
    release = threading.Event()
    initialized: list[sqlite3.Connection] = []

    def slow_init_schema(conn: sqlite3.Connection) -> None:
        real_init_schema(conn)
        initialized.append(conn)
        in_init.set()
        assert release.wait(timeout=10), "init_schema was never released"

    monkeypatch.setattr(_state_manager_module, "init_schema", slow_init_schema)

    opened: list[sqlite3.Connection] = []
    opener = threading.Thread(target=lambda: opened.append(sm.conn))
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (sm.close(), closed.set()))

    opener.start()
    try:
        assert in_init.wait(timeout=5), "conn never reached init_schema"
        closer.start()
        assert not closed.wait(timeout=0.3), "close() ran during connection initialization"
        assert sm._conn is None  # published only after init_schema succeeds
        initialized[0].execute("SELECT 1")  # still open: close() did not touch it
    finally:
        release.set()

    opener.join(timeout=5)
    closer.join(timeout=5)
    assert not opener.is_alive(), "conn did not finish after init_schema was released"
    assert closed.is_set(), "close() did not run after initialization finished"
    assert opened == initialized and opened, (
        "conn returned a connection that was never initialized"
    )
    assert sm._conn is None
    assert sm.list_strategies() == []  # reopens lazily through conn
    sm.close()


_CLOSE_STRESS_SCRIPT = """
import sys, tempfile, threading, time
from pathlib import Path
from vibe_quant.db.state_manager import StateManager

sm = StateManager(Path(tempfile.mkdtemp()) / "stress.db")
sm.create_strategy("s", {"x": 1})
stop = threading.Event()
lock = threading.Lock()
errors = []
ok = 0
closes = 0

def reader():
    global ok
    while not stop.is_set():
        try:
            strategy = sm.get_strategy(1)
            if strategy is None or strategy["name"] != "s" or strategy["dsl_config"] != {"x": 1}:
                raise AssertionError(f"get_strategy(1) -> {strategy!r}")
            strategies = sm.list_strategies()
            if [s["name"] for s in strategies] != ["s"]:
                raise AssertionError(f"list_strategies() -> {strategies!r}")
        except Exception as exc:
            errors.append(exc)
            return
        with lock:
            ok += 1

threads = [threading.Thread(target=reader) for _ in range(4)]
for t in threads:
    t.start()
end = time.monotonic() + 2.0
while time.monotonic() < end:
    sm.close()
    closes += 1
stop.set()
for t in threads:
    t.join()
if errors:
    print(f"reader_failure_count={len(errors)} first={errors[0]!r}", file=sys.stderr)
    sys.exit(2)
print(f"survived ok={ok} closes={closes}")
"""


def test_close_racing_readers_does_not_crash_the_interpreter() -> None:
    """Hammer ``close()`` against reader threads in a child process.

    Unfixed, the child dies with SIGSEGV (returncode -11) within ~0.1 s; a child
    is used so the crash cannot take the pytest/xdist worker down with it.

    The readers also assert their query results, so a build where queries fail
    (or raise) cannot report "survived": any reader exception exits the child
    non-zero, and the parent additionally requires that queries actually
    succeeded (``ok > 0``) and that ``close()`` actually ran (``closes > 0``).
    """
    result = subprocess.run(
        [sys.executable, "-c", _CLOSE_STRESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-600:])
    assert result.stdout.startswith("survived "), result.stdout
    fields = dict(part.split("=", 1) for part in result.stdout.split()[1:])
    assert int(fields["ok"]) > 0, result.stdout
    assert int(fields["closes"]) > 0, result.stdout


# --- Modules that used to reach into ``state.conn`` from outside StateManager ---

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_MODULES = (
    "vibe_quant/screening/replay_drift.py",
    "vibe_quant/validation/consistency.py",
)


def _conn_attribute_lines(source: str) -> list[int]:
    """Line numbers of every ``<anything>.conn`` / ``<anything>._conn`` access."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr in {"conn", "_conn"}
    ]


def test_state_consumers_never_touch_the_shared_connection() -> None:
    offenders = {
        rel: lines
        for rel in _EXTERNAL_MODULES
        if (lines := _conn_attribute_lines((_REPO_ROOT / rel).read_text()))
    }
    assert offenders == {}, (
        "these modules read/write StateManager's shared sqlite connection outside "
        f"its write lock; add a locked StateManager method instead: {offenders}"
    )


def test_conn_attribute_scanner_flags_external_access() -> None:
    assert _conn_attribute_lines("def f(state):\n    return state.conn.execute('x')\n") == [2]
    assert _conn_attribute_lines("def f(state):\n    return state.get_strategy(1)\n") == []


# --- New locked methods return exactly what the old inline queries did ---


def _seed_screening_run(sm: StateManager, strategy_id: int, status: str, rows: int) -> int:
    run_id = sm.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={},
    )
    sm.update_backtest_run_status(run_id, status)
    for i in range(rows):
        sm.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
            " VALUES (?, ?, ?, ?)",
            (run_id, f'{{"p": {i}}}', 1.0 + i, 10 + i),
        )
    sm.conn.commit()
    return run_id


def test_latest_screening_run_matches_old_inline_query(tmp_state_manager: StateManager) -> None:
    sm = tmp_state_manager
    sid = sm.create_strategy("s1", {"name": "s1"})
    other = sm.create_strategy("s2", {"name": "s2"})
    _seed_screening_run(sm, sid, "completed", 1)
    newest = _seed_screening_run(sm, sid, "completed", 2)
    _seed_screening_run(sm, sid, "failed", 3)  # newer but not completed
    _seed_screening_run(sm, sid, "completed", 0)  # newer, completed, no sweep rows
    _seed_screening_run(sm, other, "completed", 1)

    old = sm.conn.execute(
        """
        SELECT br.id, br.start_date, br.end_date FROM backtest_runs br
        WHERE br.strategy_id = ? AND br.run_mode = 'screening'
              AND br.status = 'completed'
              AND EXISTS (SELECT 1 FROM sweep_results sr WHERE sr.run_id = br.id)
        ORDER BY br.id DESC LIMIT 1
        """,
        (sid,),
    ).fetchone()
    new = sm.get_latest_screening_run_with_sweeps(sid)
    assert new == (newest, "2024-01-01", "2025-01-01")
    assert new == tuple(old)
    assert sm.get_latest_screening_run_with_sweeps(9999) is None


def test_sweep_reference_rows_match_old_inline_query(tmp_state_manager: StateManager) -> None:
    sm = tmp_state_manager
    sid = sm.create_strategy("s1", {"name": "s1"})
    run_id = _seed_screening_run(sm, sid, "completed", 3)
    sm.conn.execute(
        "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
        " VALUES (?, '{}', NULL, NULL)",
        (run_id,),
    )
    sm.conn.commit()
    old = [
        tuple(r)
        for r in sm.conn.execute(
            "SELECT parameters, sharpe_ratio, total_trades FROM sweep_results "
            "WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    ]
    new = sm.list_sweep_reference_rows(run_id)
    assert new == old
    assert new[0] == ('{"p": 0}', 1.0, 10)
    assert new[-1] == ("{}", None, None)
    assert sm.list_sweep_reference_rows(9999) == []


def test_strategy_dsl_config_raw_returns_stored_column(tmp_state_manager: StateManager) -> None:
    sm = tmp_state_manager
    sid = sm.create_strategy("s1", {"name": "s1", "size": 0.5})
    assert sm.get_strategy_dsl_config_raw(sid) == '{"name": "s1", "size": 0.5}'
    assert sm.get_strategy_dsl_config_raw(9999) is None


def _seed_discovery_run(sm: StateManager, notes: str | None, status: str = "completed") -> int:
    run_id = sm.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-06-01",
        parameters={},
    )
    sm.update_backtest_run_status(run_id, status)
    if notes is not None:
        sm.conn.execute(
            "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)", (run_id, notes)
        )
        sm.conn.commit()
    return run_id


def test_completed_discovery_notes_match_old_inline_query(
    tmp_state_manager: StateManager,
) -> None:
    sm = tmp_state_manager
    first = _seed_discovery_run(sm, '{"top_strategies": ["abc123"]}')
    _seed_discovery_run(sm, '{"top_strategies": ["zzz"]}', status="failed")
    _seed_discovery_run(sm, None)  # completed but no result row
    last = _seed_discovery_run(sm, '{"top_strategies": ["abc123", "other"]}')

    base = """
        SELECT br.id, res.notes, br.start_date, br.end_date, br.symbols
        FROM backtest_runs br
        JOIN backtest_results res ON res.run_id = br.id
        WHERE br.run_mode = 'discovery' AND br.status = 'completed'
    """
    old_all = [tuple(r) for r in sm.conn.execute(base + " ORDER BY br.id DESC")]
    new_all = sm.list_completed_discovery_notes()
    assert new_all == old_all
    assert [r[0] for r in new_all] == [last, first]

    old_like = [
        tuple(r)
        for r in sm.conn.execute(
            base + " AND res.notes LIKE ? ORDER BY br.id DESC", ("%other%",)
        )
    ]
    new_like = sm.list_completed_discovery_notes("%other%")
    assert new_like == old_like
    assert [r[0] for r in new_like] == [last]


def _drift_payload() -> dict[str, object]:
    return {"flagged": True, "trade_ratio": 0.5}


def test_merge_replay_drift_payload_prefers_result_notes(
    tmp_state_manager: StateManager,
) -> None:
    sm = tmp_state_manager
    run_id = sm.create_backtest_run(
        strategy_id=None, run_mode="screening", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2025-01-01", end_date="2025-06-01", parameters={"keep": 1},
    )
    sm.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, '{"data_window": "x"}'),
    )
    sm.conn.commit()
    sm.merge_replay_drift_payload(run_id, _drift_payload())
    notes = json.loads(sm.get_backtest_result(run_id)["notes"])  # type: ignore[index]
    assert notes == {"data_window": "x", "replay_drift": _drift_payload()}
    # parameters untouched when the notes row took the payload
    assert sm.get_backtest_run(run_id)["parameters"] == {"keep": 1}  # type: ignore[index]


def test_merge_replay_drift_payload_falls_back_to_run_parameters(
    tmp_state_manager: StateManager,
) -> None:
    sm = tmp_state_manager
    run_id = sm.create_backtest_run(
        strategy_id=None, run_mode="screening", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2025-01-01", end_date="2025-06-01", parameters={"keep": 1},
    )
    sm.merge_replay_drift_payload(run_id, _drift_payload())
    assert sm.get_backtest_run(run_id)["parameters"] == {  # type: ignore[index]
        "keep": 1,
        "replay_drift": _drift_payload(),
    }


def test_merge_replay_drift_payload_replaces_garbage_notes(
    tmp_state_manager: StateManager,
) -> None:
    sm = tmp_state_manager
    run_id = sm.create_backtest_run(
        strategy_id=None, run_mode="screening", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2025-01-01", end_date="2025-06-01", parameters={},
    )
    sm.conn.execute("INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)", (run_id, "[1]"))
    sm.conn.commit()
    sm.merge_replay_drift_payload(run_id, _drift_payload())
    notes = json.loads(sm.get_backtest_result(run_id)["notes"])  # type: ignore[index]
    assert notes == {"replay_drift": _drift_payload()}


def _screening_run(sm: StateManager) -> int:
    return sm.create_backtest_run(
        strategy_id=None, run_mode="screening", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2025-01-01", end_date="2025-06-01", parameters={"keep": 1},
    )


def _read_committed(db_path: Path, sql: str, run_id: int) -> str:
    """Read through an independent connection: sees only committed data."""
    other = sqlite3.connect(db_path)
    try:
        return str(other.execute(sql, (run_id,)).fetchone()[0])
    finally:
        other.close()


def test_merge_replay_drift_payload_notes_branch_is_committed(
    tmp_state_manager: StateManager, tmp_path: Path
) -> None:
    sm = tmp_state_manager
    run_id = _screening_run(sm)
    sm.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, '{"data_window": "x"}'),
    )
    sm.conn.commit()
    sm.merge_replay_drift_payload(run_id, _drift_payload())
    raw = _read_committed(
        tmp_path / "state.db", "SELECT notes FROM backtest_results WHERE run_id = ?", run_id
    )
    assert json.loads(raw) == {"data_window": "x", "replay_drift": _drift_payload()}


def test_merge_replay_drift_payload_parameters_branch_is_committed(
    tmp_state_manager: StateManager, tmp_path: Path
) -> None:
    sm = tmp_state_manager
    run_id = _screening_run(sm)
    sm.merge_replay_drift_payload(run_id, _drift_payload())
    raw = _read_committed(
        tmp_path / "state.db", "SELECT parameters FROM backtest_runs WHERE id = ?", run_id
    )
    assert json.loads(raw) == {"keep": 1, "replay_drift": _drift_payload()}


@pytest.mark.parametrize("empty_notes", [None, ""])
def test_merge_replay_drift_payload_empty_result_notes_fall_back_to_parameters(
    tmp_state_manager: StateManager, empty_notes: str | None
) -> None:
    sm = tmp_state_manager
    run_id = _screening_run(sm)
    sm.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)", (run_id, empty_notes)
    )
    sm.conn.commit()
    sm.merge_replay_drift_payload(run_id, _drift_payload())
    assert sm.get_backtest_run(run_id)["parameters"] == {  # type: ignore[index]
        "keep": 1,
        "replay_drift": _drift_payload(),
    }
    assert sm.get_backtest_result(run_id)["notes"] == empty_notes  # type: ignore[index]
