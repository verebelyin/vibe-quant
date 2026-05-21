"""SQLite schema definitions for vibe-quant state database."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

# Bump when adding new migrations to _migrate_add_columns
SCHEMA_VERSION: int = 11

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
    skewness REAL,
    kurtosis REAL,
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
    execution_time_seconds REAL,
    skewness REAL,
    kurtosis REAL,
    is_pareto_optimal BOOLEAN DEFAULT 0,
    passed_deflated_sharpe BOOLEAN,
    passed_walk_forward BOOLEAN,
    passed_purged_kfold BOOLEAN
);

-- Background job tracking for process management
CREATE TABLE IF NOT EXISTS background_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER UNIQUE,
    pid INTEGER NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT DEFAULT 'running',
    heartbeat_at TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    log_file TEXT,
    error_message TEXT
);

-- Screening-to-validation consistency checks
CREATE TABLE IF NOT EXISTS consistency_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    screening_run_id INTEGER NOT NULL,
    validation_run_id INTEGER NOT NULL,
    screening_sharpe REAL NOT NULL,
    validation_sharpe REAL NOT NULL,
    sharpe_degradation REAL NOT NULL,
    screening_return REAL NOT NULL,
    validation_return REAL NOT NULL,
    return_degradation REAL NOT NULL,
    is_execution_sensitive INTEGER NOT NULL,
    parameters TEXT NOT NULL,
    checked_at TEXT NOT NULL
);

-- System state (singleton row, id=1). Persistent kill-switch + halt metadata.
-- One row, updated in place; never deleted.
CREATE TABLE IF NOT EXISTS system_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    kill_switch INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    killed_at TEXT,
    killed_by TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO system_state (id, kill_switch) VALUES (1, 0);

-- External research pipeline (Reddit, arxiv, ...). Source-agnostic items + LLM extractions.
CREATE TABLE IF NOT EXISTS research_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    body TEXT,
    author TEXT,
    posted_at TEXT,
    score INTEGER,
    extras_json TEXT,
    fetched_at TEXT DEFAULT (datetime('now')),
    extraction_status TEXT DEFAULT 'pending'
        CHECK (extraction_status IN ('pending', 'queued', 'running', 'extracted', 'failed', 'skipped')),
    UNIQUE(source, external_id)
);

CREATE TABLE IF NOT EXISTS research_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_item_id INTEGER NOT NULL REFERENCES research_items(id),
    extracted_at TEXT DEFAULT (datetime('now')),
    llm_model TEXT,
    confidence REAL,
    rationale TEXT,
    raw_response TEXT,
    dsl_yaml TEXT,
    parsed_dsl_json TEXT,
    parse_error TEXT,
    proposed_indicators_json TEXT,
    strategy_id INTEGER REFERENCES strategies(id),
    status TEXT DEFAULT 'parsed'
        CHECK (status IN ('parsed', 'failed', 'skipped', 'promoted', 'rejected')),
    screen_sharpe REAL,
    screen_status TEXT,
    screen_run_id INTEGER REFERENCES backtest_runs(id),
    screen_pf REAL,
    screen_max_dd REAL,
    screen_return REAL,
    screen_trades INTEGER,
    screen_error TEXT,
    screen_completed_at TEXT
);

-- Research per-source settings (one row per source). Currently stores the
-- subreddit list for the reddit source so users can edit it from the UI
-- without restarting the backend. Falls back to env vars when no row exists.
CREATE TABLE IF NOT EXISTS research_settings (
    source TEXT PRIMARY KEY,
    subreddits_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Persistent extraction queue. One row per enqueued extraction job; the
-- worker process atomically claims jobs in id order. Replaces FastAPI
-- BackgroundTasks (which dies with the process and leaves no audit trail).
CREATE TABLE IF NOT EXISTS research_extraction_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_item_id INTEGER NOT NULL REFERENCES research_items(id),
    background_job_id INTEGER REFERENCES background_jobs(id),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    queued_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    last_error TEXT,
    heartbeat_at TEXT
);

CREATE TABLE IF NOT EXISTS research_scrape_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    items_fetched INTEGER DEFAULT 0,
    items_new INTEGER DEFAULT 0,
    items_extracted INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'killed')),
    error_message TEXT,
    pid INTEGER,
    heartbeat_at TEXT,
    config_json TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_results_run ON backtest_results(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_sweep_results_run ON sweep_results(run_id);
CREATE INDEX IF NOT EXISTS idx_sweep_results_pareto ON sweep_results(is_pareto_optimal);
CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON background_jobs(status);
CREATE INDEX IF NOT EXISTS idx_research_items_source_status ON research_items(source, extraction_status);
CREATE INDEX IF NOT EXISTS idx_research_items_posted ON research_items(posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_extractions_item ON research_extractions(research_item_id);
CREATE INDEX IF NOT EXISTS idx_research_extractions_status ON research_extractions(status);
CREATE INDEX IF NOT EXISTS idx_research_scrape_runs_status ON research_scrape_runs(status);
CREATE INDEX IF NOT EXISTS idx_research_extraction_jobs_status_id
    ON research_extraction_jobs(status, id);
CREATE INDEX IF NOT EXISTS idx_research_extraction_jobs_item
    ON research_extraction_jobs(research_item_id);
"""


def _migrate_add_columns(conn: sqlite3.Connection) -> None:
    """Add columns that may be missing from older databases.

    Each migration is idempotent (ALTER TABLE ADD COLUMN fails silently
    if column already exists). When adding new migrations, bump SCHEMA_VERSION.
    """
    migrations = [
        ("backtest_results", "starting_balance", "REAL"),
        ("backtest_results", "notes", "TEXT"),
        ("background_jobs", "error_message", "TEXT"),
        ("sweep_results", "execution_time_seconds", "REAL"),
        ("sweep_results", "skewness", "REAL"),
        ("sweep_results", "kurtosis", "REAL"),
        ("backtest_results", "skewness", "REAL"),
        ("backtest_results", "kurtosis", "REAL"),
        ("research_extractions", "proposed_indicators_json", "TEXT"),
        ("research_extractions", "screen_sharpe", "REAL"),
        ("research_extractions", "screen_status", "TEXT"),
        ("research_extractions", "screen_run_id", "INTEGER"),
        ("research_extractions", "screen_pf", "REAL"),
        ("research_extractions", "screen_max_dd", "REAL"),
        ("research_extractions", "screen_return", "REAL"),
        ("research_extractions", "screen_trades", "INTEGER"),
        ("research_extractions", "screen_error", "TEXT"),
        ("research_extractions", "screen_completed_at", "TEXT"),
        ("research_extraction_jobs", "attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("research_extraction_jobs", "max_attempts", "INTEGER NOT NULL DEFAULT 3"),
        ("research_extraction_jobs", "last_error", "TEXT"),
        ("research_extraction_jobs", "heartbeat_at", "TEXT"),
    ]
    applied = 0
    for table, column, col_type in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            applied += 1
            logger.info("Applied migration: %s.%s (%s)", table, column, col_type)
        except Exception:  # noqa: BLE001
            pass  # Column already exists
    if applied:
        logger.info("Applied %d schema migration(s) (current version: %d)", applied, SCHEMA_VERSION)


def _migrate_research_items_allow_queued(conn: sqlite3.Connection) -> None:
    """Rebuild research_items so its extraction_status CHECK includes 'queued'.

    SQLite cannot alter a CHECK constraint in place, so for existing DBs
    that predate the queue (schema v9 and earlier) we copy the table.
    No-op on fresh DBs where CREATE TABLE already encodes the new CHECK.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='research_items'"
    ).fetchone()
    if not row:
        return
    table_sql = row[0] or ""
    if "'queued'" in table_sql:
        # Live table already has CHECK; drop any orphan rebuild table from a
        # previously interrupted run so we leave the DB tidy.
        conn.execute("DROP TABLE IF EXISTS research_items_new")
        conn.commit()
        return

    logger.info("Rebuilding research_items to allow extraction_status='queued'")
    # FK enforcement must be disabled across DROP TABLE research_items, since
    # research_extractions and research_extraction_jobs hold FKs to it. Per
    # SQLite docs, PRAGMA foreign_keys is a no-op inside a transaction, so set
    # it before any BEGIN. Also clear any leftover rebuild table from a prior
    # failed run.
    conn.commit()  # ensure no implicit txn is open
    prev_fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP TABLE IF EXISTS research_items_new")
        conn.commit()
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE research_items_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                body TEXT,
                author TEXT,
                posted_at TEXT,
                score INTEGER,
                extras_json TEXT,
                fetched_at TEXT DEFAULT (datetime('now')),
                extraction_status TEXT DEFAULT 'pending'
                    CHECK (extraction_status IN
                        ('pending', 'queued', 'running', 'extracted', 'failed', 'skipped')),
                UNIQUE(source, external_id)
            );
            INSERT INTO research_items_new
                (id, source, external_id, url, title, body, author, posted_at,
                 score, extras_json, fetched_at, extraction_status)
            SELECT id, source, external_id, url, title, body, author, posted_at,
                   score, extras_json, fetched_at, extraction_status
            FROM research_items;
            DROP TABLE research_items;
            ALTER TABLE research_items_new RENAME TO research_items;
            COMMIT;
            """
        )
    finally:
        conn.execute(f"PRAGMA foreign_keys = {'ON' if prev_fk else 'OFF'}")


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema.

    Args:
        conn: SQLite connection with WAL mode enabled.
    """
    conn.executescript(SCHEMA_SQL)
    _migrate_add_columns(conn)
    _migrate_research_items_allow_queued(conn)
    conn.commit()
