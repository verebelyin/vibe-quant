"""``sweep_results.exit_timing`` exists on a fresh DB and on a migrated old one.

The column (design D1b) is TEXT holding the JSON
of the runner's ``exit_timing`` notes (``{refined, kept_bar_close, by_symbol}``),
is part of ``_SWEEP_RESULTS_COLUMNS`` (so the StateManager batch insert accepts
it) and the idempotent migration in ``_migrate_add_columns`` adds it to a
database created before it existed. Rows written without the key stay NULL (a
mode-off run and every older row).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import SCHEMA_SQL, init_schema
from vibe_quant.db.state_manager import _SWEEP_RESULTS_COLUMNS, StateManager

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path


def _columns(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row[1]): str(row[2] or "").upper()
        for row in conn.execute("PRAGMA table_info(sweep_results)")
    }


def test_fresh_db_has_exit_timing_text(tmp_path: Path) -> None:
    """A brand-new database carries the column."""
    conn = get_connection(tmp_path / "fresh.db")
    init_schema(conn)
    columns = _columns(conn)
    conn.close()
    assert columns.get("exit_timing") == "TEXT"


def test_migration_adds_exit_timing_to_old_db(tmp_path: Path) -> None:
    """E9: a database created before the column gets it from the migration."""
    conn = get_connection(tmp_path / "old.db")
    conn.executescript(SCHEMA_SQL)
    conn.execute("ALTER TABLE sweep_results DROP COLUMN exit_timing")
    conn.commit()
    assert "exit_timing" not in _columns(conn)

    init_schema(conn)
    columns = _columns(conn)
    conn.close()
    assert columns.get("exit_timing") == "TEXT"


def test_migration_preserves_a_legacy_row_and_leaves_exit_timing_sql_null(tmp_path: Path) -> None:
    """E9: a row written before the column keeps its values and reads SQL NULL afterwards.

    ``ALTER TABLE ... ADD COLUMN exit_timing TEXT DEFAULT '{}'`` would backfill every
    legacy row with non-NULL ``'{}'``, mis-labelling a mode-off/older row as a zero-valued
    mode-on one; the migration must add the bare ``TEXT`` column (NULL default).
    """
    conn = get_connection(tmp_path / "legacy.db")
    conn.executescript(SCHEMA_SQL)
    conn.execute("ALTER TABLE sweep_results DROP COLUMN exit_timing")
    conn.execute(
        "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades) "
        "VALUES (NULL, '{}', 1.25, 7)"
    )
    conn.commit()
    assert "exit_timing" not in _columns(conn)

    init_schema(conn)
    init_schema(conn)  # idempotent: the second run must not touch the row either

    row = conn.execute(
        "SELECT sharpe_ratio, total_trades, exit_timing, exit_timing IS NULL FROM sweep_results"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 1.25  # the legacy row's existing fields are preserved
    assert row[1] == 7
    assert row[2] is None  # SQL NULL, not the JSON text '{}'
    assert row[3] == 1  # SELECT exit_timing IS NULL


def test_sweep_results_columns_include_exit_timing() -> None:
    """The batch write allow-list accepts the column (it is a hard error otherwise)."""
    assert "exit_timing" in _SWEEP_RESULTS_COLUMNS


def test_rows_without_the_key_read_null_and_a_json_value_round_trips(tmp_path: Path) -> None:
    """A row without the key stays NULL; a JSON string decodes on read."""
    manager = StateManager(tmp_path / "sm.db")
    try:
        strategy_id = manager.create_strategy(name="exit_timing", dsl_config={})
        run_id = manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )
        manager.save_sweep_results_batch(
            run_id,
            [
                {"parameters": {}, "sharpe_ratio": 1.0, "exit_timing": None},
                {
                    "parameters": {},
                    "sharpe_ratio": 2.0,
                    "exit_timing": '{"by_symbol": {}, "kept_bar_close": 0, "refined": 0}',
                },
            ],
        )
        rows = manager.get_sweep_results(run_id)
    finally:
        manager.close()

    # get_sweep_results sorts by Sharpe DESC, so the 2.0 row comes first.
    assert rows[0]["exit_timing"] == {"by_symbol": {}, "kept_bar_close": 0, "refined": 0}
    assert rows[1]["exit_timing"] is None
