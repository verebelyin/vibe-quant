"""``sweep_results.starting_balance`` exists on a fresh DB and on a migrated old one.

bead vibe-quant-zpbmo, task Z1: the column is REAL, is part of
``_SWEEP_RESULTS_COLUMNS`` (so the StateManager insert accepts it), and the
idempotent migration in ``_migrate_add_columns`` adds it to a database created
before it existed. Rows written without the key stay NULL (an older screening run).
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


def test_fresh_db_has_starting_balance_real(tmp_path: Path) -> None:
    """A brand-new database carries the column."""
    conn = get_connection(tmp_path / "fresh.db")
    init_schema(conn)
    columns = _columns(conn)
    conn.close()
    assert columns.get("starting_balance") == "REAL"


def test_migration_adds_starting_balance_to_old_db(tmp_path: Path) -> None:
    """A database created before the column gets it from the migration."""
    conn = get_connection(tmp_path / "old.db")
    conn.executescript(SCHEMA_SQL)
    conn.execute("ALTER TABLE sweep_results DROP COLUMN starting_balance")
    conn.commit()
    assert "starting_balance" not in _columns(conn)

    init_schema(conn)
    columns = _columns(conn)
    conn.close()
    assert columns.get("starting_balance") == "REAL"


def test_sweep_results_columns_include_starting_balance() -> None:
    """The write allow-list accepts the column (it is a hard error otherwise)."""
    assert "starting_balance" in _SWEEP_RESULTS_COLUMNS


def _create_run(manager: StateManager, name: str) -> int:
    strategy_id = manager.create_strategy(name=name, dsl_config={})
    return manager.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )


def test_saving_rows_with_and_without_the_balance(tmp_path: Path) -> None:
    """A written row keeps its float value; a row without the key reads NULL."""
    manager = StateManager(tmp_path / "sm.db")
    try:
        run_id = _create_run(manager, "sweep_balance")
        manager.save_sweep_results_batch(
            run_id,
            [
                {"parameters": {}, "sharpe_ratio": 1.0, "starting_balance": 10000.0},
                {"parameters": {}, "sharpe_ratio": 2.0, "starting_balance": -999.0},
            ],
        )
        rows = manager.get_sweep_results(run_id)

        legacy_id = _create_run(manager, "sweep_balance_legacy")
        manager.save_sweep_results_batch(legacy_id, [{"parameters": {}, "sharpe_ratio": 1.0}])
        legacy = manager.get_sweep_results(legacy_id)
    finally:
        manager.close()

    # get_sweep_results sorts by Sharpe DESC, so the -999 sentinel row comes first.
    assert [row["starting_balance"] for row in rows] == [-999.0, 10000.0]
    assert legacy[0]["starting_balance"] is None
