"""Audit e70tl.15/.16: result persistence (re-run replace, migration, user notes)."""

from __future__ import annotations

import itertools
import sqlite3
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import SCHEMA_SQL
from vibe_quant.db.state_manager import StateManager

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

ZERO_TRADE_ERR = "Validation produced 0 trades — likely missing/empty data"


def _trade(i: int, pnl: float = 1.0) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT-PERP",
        "direction": "LONG",
        "leverage": 1,
        "entry_time": f"2025-01-{i + 1:02d}T00:00:00+00:00",
        "exit_time": f"2025-01-{i + 1:02d}T01:00:00+00:00",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "quantity": 1.0,
        "net_pnl": pnl,
    }


@pytest.fixture
def mgr(tmp_path: Path) -> Iterator[StateManager]:
    m = StateManager(tmp_path / "state.db")
    yield m
    m.close()


_NAMES = itertools.count()


def _run(mgr: StateManager, mode: str = "validation") -> int:
    sid = mgr.create_strategy(f"s_{mode}_{next(_NAMES)}", {"name": "s"})
    return mgr.create_backtest_run(sid, mode, ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {})


class TestRerunReplaces:
    def test_rerun_keeps_one_result_row_and_replaces_trades(self, mgr: StateManager) -> None:
        run_id = _run(mgr)
        mgr.save_backtest_result(run_id, {"total_trades": 0, "sharpe_ratio": None})
        mgr.save_backtest_result(
            run_id,
            {"total_trades": 3, "sharpe_ratio": 2.5},
            trades=[_trade(0), _trade(1), _trade(2)],
        )
        mgr.save_backtest_result(
            run_id, {"total_trades": 2, "sharpe_ratio": 1.5}, trades=[_trade(0), _trade(1)]
        )

        n_rows = mgr.conn.execute(
            "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert n_rows == 1
        result = mgr.get_backtest_result(run_id)
        assert result is not None
        assert result["total_trades"] == 2
        assert result["sharpe_ratio"] == 1.5
        assert len(mgr.get_trades(run_id)) == 2

    def test_runner_style_two_call_save_does_not_append(self, mgr: StateManager) -> None:
        """save_backtest_result + save_trades_batch per attempt (validation runner)."""
        run_id = _run(mgr)
        for _ in range(3):
            mgr.save_backtest_result(run_id, {"total_trades": 4})
            mgr.save_trades_batch(run_id, [_trade(i) for i in range(4)])
        assert len(mgr.get_trades(run_id)) == 4

    def test_rerun_with_zero_trades_clears_previous_trades(self, mgr: StateManager) -> None:
        run_id = _run(mgr)
        mgr.save_backtest_result(run_id, {"total_trades": 2}, trades=[_trade(0), _trade(1)])
        mgr.save_backtest_result(run_id, {"total_trades": 0})
        assert mgr.get_trades(run_id) == []

    def test_other_runs_untouched(self, mgr: StateManager) -> None:
        a, b = _run(mgr), _run(mgr)
        mgr.save_backtest_result(a, {"total_trades": 1}, trades=[_trade(0)])
        mgr.save_backtest_result(b, {"total_trades": 2}, trades=[_trade(0), _trade(1)])
        mgr.save_backtest_result(a, {"total_trades": 1}, trades=[_trade(5)])
        assert len(mgr.get_trades(b)) == 2
        assert [t["entry_time"][:10] for t in mgr.get_trades(a)] == ["2025-01-06"]

    def test_failed_trade_insert_rolls_back_result(self, mgr: StateManager) -> None:
        run_id = _run(mgr)
        mgr.save_backtest_result(run_id, {"total_trades": 1, "sharpe_ratio": 1.0}, [_trade(0)])
        bad = [_trade(0), {"symbol": "X"}]  # inconsistent keys → rejected before writing
        with pytest.raises(ValueError, match="Inconsistent trade keys"):
            mgr.save_backtest_result(run_id, {"total_trades": 2, "sharpe_ratio": 9.0}, bad)
        result = mgr.get_backtest_result(run_id)
        assert result is not None and result["sharpe_ratio"] == 1.0
        assert len(mgr.get_trades(run_id)) == 1

    def test_summary_has_no_duplicate_run_ids(self, mgr: StateManager) -> None:
        run_id = _run(mgr)
        mgr.save_backtest_result(run_id, {"total_trades": 0})
        mgr.save_backtest_result(run_id, {"total_trades": 66, "sharpe_ratio": 2.3})
        rows = [r for r in mgr.list_runs_with_results() if r["run_id"] == run_id]
        assert len(rows) == 1
        assert rows[0]["total_trades"] == 66
        listed = [r for r in mgr.list_backtest_results() if r["run_id"] == run_id]
        assert len(listed) == 1

    def test_unique_index_rejects_raw_duplicate_insert(self, mgr: StateManager) -> None:
        run_id = _run(mgr)
        mgr.save_backtest_result(run_id, {"total_trades": 1})
        with pytest.raises(sqlite3.IntegrityError):
            mgr.conn.execute("INSERT INTO backtest_results (run_id) VALUES (?)", (run_id,))


class TestUserNotes:
    def test_user_notes_do_not_touch_machine_notes(self, mgr: StateManager) -> None:
        run_id = _run(mgr, "discovery")
        payload = '{"type": "discovery", "top_strategies": [{"dsl": {}}]}'
        mgr.save_backtest_result(run_id, {"total_trades": 1, "notes": payload})
        assert mgr.update_result_user_notes(run_id, "looks overfit")
        row = mgr.get_backtest_result(run_id)
        assert row is not None
        assert row["notes"] == payload
        assert row["user_notes"] == "looks overfit"

    def test_user_notes_survive_rerun(self, mgr: StateManager) -> None:
        run_id = _run(mgr)
        mgr.save_backtest_result(run_id, {"total_trades": 1})
        mgr.update_result_user_notes(run_id, "keep me")
        mgr.save_backtest_result(run_id, {"total_trades": 2})
        row = mgr.get_backtest_result(run_id)
        assert row is not None and row["user_notes"] == "keep me"

    def test_user_notes_without_result_returns_false(self, mgr: StateManager) -> None:
        assert mgr.update_result_user_notes(_run(mgr), "x") is False


# ---------------------------------------------------------------------------
# Migration v16 on a legacy (pre-fix) database
# ---------------------------------------------------------------------------


def _legacy_db(path: Path) -> sqlite3.Connection:
    """DB as written by the pre-fix code: no unique index, no user_notes, user_version 0."""
    conn = get_connection(path)
    conn.executescript(SCHEMA_SQL)
    conn.execute("ALTER TABLE backtest_results DROP COLUMN user_notes")
    conn.execute("ALTER TABLE background_jobs DROP COLUMN pid_start_time")
    conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (1, 's', '{}')")
    conn.commit()
    return conn


def _legacy_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str = "completed",
    error: str | None = None,
    mode: str = "validation",
) -> None:
    conn.execute(
        """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status, error_message)
           VALUES (?, 1, ?, '["BTCUSDT"]', '4h', '2025-01-01', '2025-06-01', '{}', ?, ?)""",
        (run_id, mode, status, error),
    )


def _legacy_result(
    conn: sqlite3.Connection, run_id: int, trades: int, sharpe: float | None, notes: str | None = None
) -> None:
    conn.execute(
        "INSERT INTO backtest_results (run_id, total_trades, sharpe_ratio, notes) VALUES (?, ?, ?, ?)",
        (run_id, trades, sharpe, notes),
    )


def _legacy_trades(conn: sqlite3.Connection, run_id: int, n: int, first_id: int) -> None:
    for i in range(n):
        conn.execute(
            """INSERT INTO trades (id, run_id, symbol, direction, entry_time, entry_price,
                   quantity, net_pnl) VALUES (?, ?, 'BTCUSDT-PERP', 'LONG', ?, 100, 1, ?)""",
            (first_id + i, run_id, f"2025-02-{i + 1:02d}T00:00:00", float(i)),
        )


@pytest.fixture
def migrated(tmp_path: Path) -> Iterator[StateManager]:
    path = tmp_path / "legacy.db"
    conn = _legacy_db(path)
    # 337-style: 0-trade attempt then a 66-trade attempt; status completed + stale 0-trade error.
    _legacy_run(conn, 337, error=ZERO_TRADE_ERR)
    _legacy_result(conn, 337, 0, None)
    _legacy_result(conn, 337, 6, 2.31687880086055)
    _legacy_trades(conn, 337, 6, first_id=100)
    # 368-style: single 0-trade attempt, flipped to completed by mark_completed().
    _legacy_run(conn, 368, error=ZERO_TRADE_ERR)
    _legacy_result(conn, 368, 0, None)
    conn.execute(
        "INSERT INTO background_jobs (run_id, pid, job_type, status) "
        "VALUES (368, 1, 'validation', 'completed')"
    )
    # 459-style: three identical attempts, blocks separated by gaps AND contiguous blocks.
    _legacy_run(conn, 459)
    for _ in range(3):
        _legacy_result(conn, 459, 5, -0.88)
    _legacy_trades(conn, 459, 5, first_id=200)
    _legacy_trades(conn, 459, 5, first_id=205)  # contiguous: boundary = entry_time reset
    _legacy_trades(conn, 459, 5, first_id=300)  # id gap
    # 380-style: one result row but trades saved twice.
    _legacy_run(conn, 380)
    _legacy_result(conn, 380, 4, 1.1)
    _legacy_trades(conn, 380, 4, first_id=400)
    _legacy_trades(conn, 380, 4, first_id=410)
    # Latest attempt had 0 trades, an older one had trades → all trades stale.
    _legacy_run(conn, 500)
    _legacy_result(conn, 500, 3, 1.0)
    _legacy_trades(conn, 500, 3, first_id=500)
    _legacy_result(conn, 500, 0, None)
    # Untouched normal run + a run with fewer trade rows than total_trades (open pos).
    _legacy_run(conn, 870)
    _legacy_result(conn, 870, 3, 0.58)
    _legacy_trades(conn, 870, 3, first_id=600)
    _legacy_run(conn, 30)
    _legacy_result(conn, 30, 1, 0.1)
    # User text typed into the machine notes column of a validation run.
    _legacy_run(conn, 901)
    _legacy_result(conn, 901, 1, 0.1, notes="my manual note")
    # Discovery notes (incl. legacy non-JSON machine text) stay in notes.
    _legacy_run(conn, 902, mode="discovery")
    _legacy_result(conn, 902, 1, 0.1, notes="discovery: generations=5, evaluated=100")
    _legacy_run(conn, 903)
    _legacy_result(conn, 903, 1, 0.1, notes='{"consistency": {"flags": []}}')
    conn.commit()
    conn.close()

    mgr = StateManager(path)
    yield mgr
    mgr.close()


class TestMigrationV16:
    def test_337_latest_attempt_wins_and_error_cleared(self, migrated: StateManager) -> None:
        res = migrated.get_backtest_result(337)
        assert res is not None
        assert res["total_trades"] == 6
        assert res["sharpe_ratio"] == 2.31687880086055
        assert len(migrated.get_trades(337)) == 6
        run = migrated.get_backtest_run(337)
        assert run is not None
        assert run["status"] == "completed"
        assert run["error_message"] is None

    def test_368_zero_trade_run_marked_failed(self, migrated: StateManager) -> None:
        run = migrated.get_backtest_run(368)
        assert run is not None
        assert run["status"] == "failed"
        assert run["error_message"] == ZERO_TRADE_ERR
        job = migrated.get_job(368)
        assert job is not None
        assert job["status"] == "failed"
        assert job["error_message"] == ZERO_TRADE_ERR

    def test_459_triplicated_trades_reduced_to_latest_attempt(
        self, migrated: StateManager
    ) -> None:
        trades = migrated.get_trades(459)
        assert len(trades) == 5
        assert sorted(t["id"] for t in trades) == [300, 301, 302, 303, 304]
        n = migrated.conn.execute(
            "SELECT COUNT(*) FROM backtest_results WHERE run_id = 459"
        ).fetchone()[0]
        assert n == 1

    def test_380_duplicated_trades_without_duplicate_result(
        self, migrated: StateManager
    ) -> None:
        assert sorted(t["id"] for t in migrated.get_trades(380)) == [410, 411, 412, 413]

    def test_latest_zero_trade_attempt_drops_stale_trades(self, migrated: StateManager) -> None:
        assert migrated.get_trades(500) == []
        res = migrated.get_backtest_result(500)
        assert res is not None and res["total_trades"] == 0

    def test_clean_runs_untouched(self, migrated: StateManager) -> None:
        assert len(migrated.get_trades(870)) == 3
        res = migrated.get_backtest_result(30)
        assert res is not None and res["total_trades"] == 1

    def test_user_text_moved_to_user_notes(self, migrated: StateManager) -> None:
        res = migrated.get_backtest_result(901)
        assert res is not None
        assert res["user_notes"] == "my manual note"
        assert res["notes"] is None
        disc = migrated.get_backtest_result(902)
        assert disc is not None
        assert disc["notes"] == "discovery: generations=5, evaluated=100"
        assert disc["user_notes"] is None
        val = migrated.get_backtest_result(903)
        assert val is not None and val["notes"] == '{"consistency": {"flags": []}}'

    def test_summary_has_no_duplicates(self, migrated: StateManager) -> None:
        ids = [r["run_id"] for r in migrated.list_runs_with_results()]
        assert len(ids) == len(set(ids))

    def test_idempotent_and_versioned(self, migrated: StateManager, tmp_path: Path) -> None:
        assert migrated.conn.execute("PRAGMA user_version").fetchone()[0] == 16
        migrated.close()
        again = StateManager(tmp_path / "legacy.db")
        try:
            assert len(again.get_trades(459)) == 5
            assert len(again.get_trades(337)) == 6
        finally:
            again.close()
