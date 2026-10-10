"""SQLite schema definitions for vibe-quant state database."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

# Bump when adding new migrations to _migrate_add_columns. One-time data
# migrations are additionally gated by ``PRAGMA user_version`` markers.
SCHEMA_VERSION: int = 17

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
    -- Machine-written JSON (discovery payload, consistency flags, compiler
    -- version). Never user-editable: the Notes panel writes user_notes.
    notes TEXT,
    user_notes TEXT,
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
    error_message TEXT,
    -- Process identity (OS start time of `pid`) so a recycled PID is never
    -- mistaken for the job's process (signalled or reported alive).
    pid_start_time TEXT
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
    evidence_level TEXT CHECK (
        evidence_level IS NULL OR evidence_level IN ('live_traded', 'backtested', 'idea_only')
    ),
    completeness REAL CHECK (
        completeness IS NULL OR (completeness >= 0.0 AND completeness <= 1.0)
    ),
    rationale TEXT,
    raw_response TEXT,
    prompt TEXT,
    dsl_yaml TEXT,
    parsed_dsl_json TEXT,
    parse_error TEXT,
    proposed_indicators_json TEXT,
    risk_management_json TEXT,
    notable_parameters_json TEXT,
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

-- One row per (extraction, idx) into proposed_indicators_json once the
-- user clicks "Scaffold plugin". Caches the codegen outcome so re-clicks
-- don't re-burn LLM tokens (force=1 bypasses). Slice 1 only writes/reads;
-- slices 2-3 populate plugin_path/test_path/commit_sha on success.
CREATE TABLE IF NOT EXISTS research_indicator_scaffolds (
    extraction_id INTEGER NOT NULL REFERENCES research_extractions(id),
    idx INTEGER NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('ok', 'codegen_failed', 'test_failed', 'failed')),
    plugin_path TEXT,
    test_path TEXT,
    commit_sha TEXT,
    error TEXT,
    test_output TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (extraction_id, idx)
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
CREATE INDEX IF NOT EXISTS idx_research_indicator_scaffolds_extraction
    ON research_indicator_scaffolds(extraction_id);
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
        ("research_extractions", "prompt", "TEXT"),
        (
            "research_extractions",
            "evidence_level",
            "TEXT CHECK (evidence_level IS NULL OR "
            "evidence_level IN ('live_traded', 'backtested', 'idea_only'))",
        ),
        (
            "research_extractions",
            "completeness",
            "REAL CHECK (completeness IS NULL OR "
            "(completeness >= 0.0 AND completeness <= 1.0))",
        ),
        ("research_extractions", "screen_sharpe", "REAL"),
        ("research_extractions", "screen_status", "TEXT"),
        ("research_extractions", "screen_run_id", "INTEGER"),
        ("research_extractions", "screen_pf", "REAL"),
        ("research_extractions", "screen_max_dd", "REAL"),
        ("research_extractions", "screen_return", "REAL"),
        ("research_extractions", "screen_trades", "INTEGER"),
        ("research_extractions", "screen_error", "TEXT"),
        ("research_extractions", "screen_completed_at", "TEXT"),
        ("research_extractions", "risk_management_json", "TEXT"),
        ("research_extractions", "notable_parameters_json", "TEXT"),
        ("research_extraction_jobs", "attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("research_extraction_jobs", "max_attempts", "INTEGER NOT NULL DEFAULT 3"),
        ("research_extraction_jobs", "last_error", "TEXT"),
        ("research_extraction_jobs", "heartbeat_at", "TEXT"),
        # Set by POST /extraction-jobs/{id}/cancel for a RUNNING job; the
        # worker polls it (~1s) and interrupts only that job's subprocess.
        # Status stays 'running' until the worker finalizes it as 'cancelled'.
        ("research_extraction_jobs", "cancel_requested_at", "TEXT"),
        ("backtest_results", "user_notes", "TEXT"),
        ("background_jobs", "pid_start_time", "TEXT"),
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


ZERO_TRADE_ERROR_PATTERN = "%produced 0 trades%"
# PRAGMA user_version marker for _migrate_v16_dedupe_results (one-time data fix).
_USER_VERSION_V16 = 16


def _trade_attempt_segments(rows: list[sqlite3.Row]) -> list[list[int]]:
    """Split one run's trade rows (ordered by id) into per-attempt segments.

    Each attempt inserted its trades in a single ``executemany`` transaction,
    so one attempt's ids are contiguous and its entry times non-decreasing.
    A new attempt starts at an id gap or where entry_time jumps backwards.
    """
    segments: list[list[int]] = []
    prev_id: int | None = None
    prev_entry: str | None = None
    for row in rows:
        trade_id = int(row[0])
        entry = str(row[1])
        new_attempt = (
            prev_id is None
            or trade_id != prev_id + 1
            or (prev_entry is not None and entry < prev_entry)
        )
        if new_attempt:
            segments.append([])
        segments[-1].append(trade_id)
        prev_id, prev_entry = trade_id, entry
    return segments


def _dedupe_run_trades(
    conn: sqlite3.Connection, run_id: int, n_results: int, kept_trades: int | None
) -> None:
    """Keep only the latest attempt's trades for ``run_id`` (see migration v16)."""
    rows = conn.execute(
        "SELECT id, entry_time FROM trades WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    if not rows or kept_trades is None:
        return
    if kept_trades == 0:
        if n_results > 1:
            # Latest attempt produced no trades: every stored trade is stale.
            conn.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
            logger.info("Migration v16: run %d dropped %d stale trades", run_id, len(rows))
        else:
            logger.warning(
                "Migration v16: run %d has %d trades but result says 0; left untouched",
                run_id,
                len(rows),
            )
        return
    if len(rows) <= kept_trades:
        return
    last = _trade_attempt_segments(rows)[-1]
    if len(last) != kept_trades:
        logger.warning(
            "Migration v16: run %d has %d trades for %d expected and no clean attempt "
            "boundary; left untouched",
            run_id,
            len(rows),
            kept_trades,
        )
        return
    conn.execute(
        "DELETE FROM trades WHERE run_id = ? AND (id < ? OR id > ?)",
        (run_id, last[0], last[-1]),
    )
    logger.info(
        "Migration v16: run %d kept latest attempt's %d trades (dropped %d)",
        run_id,
        len(last),
        len(rows) - len(last),
    )


def _migrate_v16_dedupe_results(conn: sqlite3.Connection) -> None:
    """One-time data migration: one result row per run, latest attempt wins.

    Re-running a run used to append a second ``backtest_results`` row and a
    second batch of trades (reads then picked an arbitrary row / summed all
    batches). Also re-marks runs left ``completed`` with the runner's
    "produced 0 trades" error (mark_completed() used to overwrite ``failed``),
    moves user-typed notes out of the machine ``notes`` column, and adds a
    UNIQUE index on ``backtest_results.run_id``.

    Gated by ``PRAGMA user_version`` and run inside ``BEGIN IMMEDIATE`` so
    concurrent processes opening the DB apply it exactly once.
    """
    if conn.execute("PRAGMA user_version").fetchone()[0] >= _USER_VERSION_V16:
        return
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("PRAGMA user_version").fetchone()[0] >= _USER_VERSION_V16:
            conn.rollback()
            return

        plan = conn.execute(
            """SELECT r.run_id, r.n_results, r.kept_id, k.total_trades AS kept_trades,
                      COALESCE(t.n_trades, 0) AS n_trades
               FROM (SELECT run_id, COUNT(*) AS n_results, MAX(id) AS kept_id
                     FROM backtest_results WHERE run_id IS NOT NULL GROUP BY run_id) r
               JOIN backtest_results k ON k.id = r.kept_id
               LEFT JOIN (SELECT run_id, COUNT(*) AS n_trades FROM trades GROUP BY run_id) t
                    ON t.run_id = r.run_id
               WHERE r.n_results > 1 OR COALESCE(t.n_trades, 0) > COALESCE(k.total_trades, 0)"""
        ).fetchall()
        for run_id, n_results, kept_id, kept_trades, _n_trades in plan:
            if n_results > 1:
                # Preserve notes written on an older row if the kept row has none.
                conn.execute(
                    """UPDATE backtest_results SET notes = (
                           SELECT notes FROM backtest_results
                           WHERE run_id = ? AND notes IS NOT NULL ORDER BY id DESC LIMIT 1)
                       WHERE id = ? AND notes IS NULL""",
                    (run_id, kept_id),
                )
                conn.execute(
                    "DELETE FROM backtest_results WHERE run_id = ? AND id != ?",
                    (run_id, kept_id),
                )
                logger.info(
                    "Migration v16: run %d kept result row %d (dropped %d older)",
                    run_id,
                    kept_id,
                    n_results - 1,
                )
            _dedupe_run_trades(conn, int(run_id), int(n_results), kept_trades)

        # 0-trade validations: runner marked failed, mark_completed() flipped it back.
        stuck = conn.execute(
            """SELECT r.id, br.total_trades, r.error_message
               FROM backtest_runs r
               LEFT JOIN backtest_results br ON br.run_id = r.id
               WHERE r.status = 'completed' AND r.error_message LIKE ?""",
            (ZERO_TRADE_ERROR_PATTERN,),
        ).fetchall()
        for run_id, total_trades, error in stuck:
            if total_trades is not None and total_trades > 0:
                # A later attempt succeeded; the error belongs to the stale attempt.
                conn.execute("UPDATE backtest_runs SET error_message = NULL WHERE id = ?", (run_id,))
            else:
                conn.execute("UPDATE backtest_runs SET status = 'failed' WHERE id = ?", (run_id,))
                conn.execute(
                    """UPDATE background_jobs SET status = 'failed', error_message = ?
                       WHERE run_id = ? AND status = 'completed'""",
                    (error, run_id),
                )
            logger.info("Migration v16: run %d status repaired", run_id)

        # User text typed into the machine notes column (non-discovery runs only;
        # discovery notes are always machine output, even legacy non-JSON ones).
        conn.execute(
            """UPDATE backtest_results SET user_notes = notes, notes = NULL
               WHERE notes IS NOT NULL AND user_notes IS NULL
                 AND (CASE WHEN json_valid(notes) THEN json_type(notes) END)
                     IS NOT 'object'
                 AND run_id IN (SELECT id FROM backtest_runs WHERE run_mode != 'discovery')"""
        )

        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_backtest_results_run_id "
            "ON backtest_results(run_id)"
        )
        conn.execute(f"PRAGMA user_version = {_USER_VERSION_V16}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema.

    Args:
        conn: SQLite connection with WAL mode enabled.
    """
    conn.executescript(SCHEMA_SQL)
    _migrate_add_columns(conn)
    _migrate_research_items_allow_queued(conn)
    conn.commit()
    _migrate_v16_dedupe_results(conn)
