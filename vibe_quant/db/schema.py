"""SQLite schema definitions for vibe-quant state database."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

SCHEMA_SQL = """
-- Strategy definitions (DSL configs)
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    dsl_config JSON NOT NULL,
    strategy_type TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    is_active BOOLEAN DEFAULT 1,
    version INTEGER DEFAULT 1
);

-- Position sizing configurations (separate from strategies)
CREATE TABLE IF NOT EXISTS sizing_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    method TEXT NOT NULL,
    config JSON NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Risk management configurations (separate from strategies)
CREATE TABLE IF NOT EXISTS risk_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    strategy_level JSON NOT NULL,
    portfolio_level JSON NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Backtest runs (both screening and validation)
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER REFERENCES strategies(id),
    sizing_config_id INTEGER REFERENCES sizing_configs(id),
    risk_config_id INTEGER REFERENCES risk_configs(id),
    run_mode TEXT NOT NULL,
    symbols JSON NOT NULL,
    timeframe TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    parameters JSON NOT NULL,
    latency_preset TEXT,
    status TEXT DEFAULT 'pending',
    pid INTEGER,
    heartbeat_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Backtest results (one row per completed run)
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    total_return REAL,
    cagr REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    max_drawdown REAL,
    max_drawdown_duration_days INTEGER,
    volatility_annual REAL,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate REAL,
    profit_factor REAL,
    avg_win REAL,
    avg_loss REAL,
    largest_win REAL,
    largest_loss REAL,
    avg_trade_duration_hours REAL,
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    total_fees REAL,
    total_funding REAL,
    total_slippage REAL,
    deflated_sharpe REAL,
    walk_forward_efficiency REAL,
    purged_kfold_mean_sharpe REAL,
    execution_time_seconds REAL,
    starting_balance REAL,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Individual trades (for detailed analysis)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    leverage INTEGER DEFAULT 1,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    entry_fee REAL,
    exit_fee REAL,
    funding_fees REAL,
    slippage_cost REAL,
    gross_pnl REAL,
    net_pnl REAL,
    roi_percent REAL,
    exit_reason TEXT
);

-- Sweep results (bulk storage for parameter sweeps from screening)
CREATE TABLE IF NOT EXISTS sweep_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    parameters JSON NOT NULL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    total_return REAL,
    profit_factor REAL,
    win_rate REAL,
    total_trades INTEGER,
    total_fees REAL,
    total_funding REAL,
    is_pareto_optimal BOOLEAN DEFAULT 0,
    passed_deflated_sharpe BOOLEAN,
    passed_walk_forward BOOLEAN,
    passed_purged_kfold BOOLEAN
);

-- Background job tracking for process management
CREATE TABLE IF NOT EXISTS background_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER UNIQUE REFERENCES backtest_runs(id) ON DELETE CASCADE,
    pid INTEGER NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT DEFAULT 'running',
    heartbeat_at TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    log_file TEXT,
    error_message TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_results_run ON backtest_results(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_sweep_results_run ON sweep_results(run_id);
CREATE INDEX IF NOT EXISTS idx_sweep_results_pareto ON sweep_results(is_pareto_optimal);
CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON background_jobs(status);
"""


def _migrate_add_columns(conn: sqlite3.Connection) -> None:
    """Add columns that may be missing from older databases."""
    import contextlib

    migrations = [
        ("backtest_results", "starting_balance", "REAL"),
        ("backtest_results", "notes", "TEXT"),
    ]
    for table, column, col_type in migrations:
        with contextlib.suppress(Exception):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema.

    Args:
        conn: SQLite connection with WAL mode enabled.
    """
    conn.executescript(SCHEMA_SQL)
    _migrate_add_columns(conn)
    conn.commit()
