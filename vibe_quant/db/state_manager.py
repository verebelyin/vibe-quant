"""StateManager class providing CRUD operations for vibe-quant state database."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import TYPE_CHECKING, Any

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import (
    SYSTEM_STATE_RESEEDED_BY,
    SYSTEM_STATE_RESEEDED_REASON,
    SYSTEM_STATE_UNLOCK_ACK_BY,
    SYSTEM_STATE_UNRESEEDABLE_MSG,
    SystemStateMissingError,
    init_schema,
    reseed_system_state,
)
from vibe_quant.sizing_settings import SizingConfigError, recorded_sizing

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path


def _recorded_sizing_params(parameters_json: object) -> dict[str, float] | None:
    """The sizing params a run recorded, or None when the stored column is unreadable.

    A run that recorded no sizing key reads as the defaults (``{}``). A column that is not
    valid JSON, is not an object, or holds an invalid sizing value is unreadable: callers
    must skip that run rather than compare it at the defaults (vibe-quant-2gyk1.2 S29c).
    """
    raw = parameters_json
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    try:
        return recorded_sizing(raw).to_params()
    except SizingConfigError:
        return None


class DuplicateResearchItem(Exception):
    """Raised when a research_item with (source, external_id) already exists."""


# Type alias for JSON-like dict structures from database
JsonDict = dict[str, Any]

# Column whitelists per table (must match schema.py definitions)
_BACKTEST_RESULTS_COLUMNS: frozenset[str] = frozenset(
    {
        "run_id",
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "max_drawdown_duration_days",
        "volatility_annual",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "largest_win",
        "largest_loss",
        "avg_trade_duration_hours",
        "max_consecutive_wins",
        "max_consecutive_losses",
        "total_fees",
        "total_funding",
        "total_slippage",
        "deflated_sharpe",
        "walk_forward_efficiency",
        "purged_kfold_mean_sharpe",
        "execution_time_seconds",
        "starting_balance",
        "skewness",
        "kurtosis",
        "notes",
    }
)

_TRADES_COLUMNS: frozenset[str] = frozenset(
    {
        "run_id",
        "symbol",
        "direction",
        "leverage",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "entry_fee",
        "exit_fee",
        "funding_fees",
        "slippage_cost",
        "gross_pnl",
        "net_pnl",
        "roi_percent",
        "exit_reason",
    }
)

_SWEEP_RESULTS_COLUMNS: frozenset[str] = frozenset(
    {
        "run_id",
        "parameters",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "total_return",
        "profit_factor",
        "win_rate",
        "total_trades",
        "total_fees",
        "total_funding",
        "execution_time_seconds",
        "skewness",
        "kurtosis",
        "mtm_sharpe_daily",
        "mtm_skewness",
        "mtm_kurtosis",
        "mtm_days",
        "starting_balance",
        "is_pareto_optimal",
        "passed_deflated_sharpe",
        "passed_walk_forward",
        "passed_purged_kfold",
        "liquidation_count",
        "liquidation_forfeited",
        "exit_timing",
    }
)


def _validate_columns(columns: list[str], allowed: frozenset[str], table: str) -> None:
    """Validate column names against whitelist. Raises ValueError on unknown columns."""
    bad = set(columns) - allowed
    if bad:
        raise ValueError(f"Unknown columns for {table}: {sorted(bad)}")


def _trade_insert(run_id: int, trades: Sequence[JsonDict]) -> tuple[str, list[list[Any]]]:
    """Validated (INSERT sql, rows) for writing ``trades`` of ``run_id``.

    Columns come from the first trade; every trade must have the same keys in
    the same order (a misaligned positional insert would corrupt the table).
    """
    if not trades:
        return "", []
    keys = list(trades[0].keys())
    columns = ["run_id", *keys]
    _validate_columns(columns, _TRADES_COLUMNS, "trades")
    rows: list[list[Any]] = []
    for trade in trades:
        if list(trade.keys()) != keys:
            raise ValueError(f"Inconsistent trade keys: {list(trade.keys())} vs {keys}")
        rows.append([run_id, *trade.values()])
    placeholders = ", ".join(["?"] * len(columns))
    return f"INSERT INTO trades ({', '.join(columns)}) VALUES ({placeholders})", rows


class StateManager:
    """Manager for vibe-quant SQLite state database.

    Provides CRUD operations for strategies, configs, backtest runs, and results.
    All connections use WAL mode for concurrent read/write access.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize StateManager.

        Args:
            db_path: Path to database file. Uses default if not specified.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection (thread-safe)."""
        with self._lock:
            if self._conn is None:
                # Publish only after init succeeds: a failed migration (e.g. a
                # held writer lock) must not leave a half-migrated cached
                # connection; the next access retries.
                conn = get_connection(self._db_path)
                try:
                    init_schema(conn)
                except BaseException:
                    conn.close()
                    raise
                self._conn = conn
            return self._conn

    @property
    def db_path(self) -> Path | None:
        """Configured database path (``None`` = the default); never opens a connection."""
        return self._db_path

    def close(self) -> None:
        """Close database connection.

        Waits for any in-flight query (they hold ``_write_lock``): closing a
        ``check_same_thread=False`` connection under a thread that is inside
        ``execute`` segfaults CPython's sqlite3. Lock order matches the query
        methods (``_write_lock`` then ``_lock`` via ``conn``).
        """
        with self._write_lock, self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --- Strategy CRUD ---

    def create_strategy(
        self,
        name: str,
        dsl_config: JsonDict,
        description: str | None = None,
        strategy_type: str | None = None,
    ) -> int:
        """Create a new strategy.

        Args:
            name: Unique strategy name.
            dsl_config: Strategy DSL configuration as dict.
            description: Optional description.
            strategy_type: Optional type (technical, statistical, composite).

        Returns:
            ID of created strategy.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO strategies (name, dsl_config, description, strategy_type)
                   VALUES (?, ?, ?, ?)""",
                (name, json.dumps(dsl_config), description, strategy_type),
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def get_strategy(self, strategy_id: int) -> JsonDict | None:
        """Get strategy by ID.

        Args:
            strategy_id: Strategy ID.

        Returns:
            Strategy dict or None if not found.
        """
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["dsl_config"] = json.loads(result["dsl_config"])
        return result

    def get_strategy_by_name(self, name: str) -> JsonDict | None:
        """Get strategy by name.

        Args:
            name: Strategy name.

        Returns:
            Strategy dict or None if not found.
        """
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM strategies WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["dsl_config"] = json.loads(result["dsl_config"])
        return result

    def list_strategies(self, active_only: bool = True) -> list[JsonDict]:
        """List all strategies.

        Args:
            active_only: If True, only return active strategies.

        Returns:
            List of strategy dicts.
        """
        query = "SELECT * FROM strategies"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY updated_at DESC"

        with self._write_lock:
            rows = self.conn.execute(query).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["dsl_config"] = json.loads(result["dsl_config"])
            results.append(result)
        return results

    def update_strategy(
        self,
        strategy_id: int,
        name: str | None = None,
        dsl_config: JsonDict | None = None,
        description: str | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Update strategy fields.

        Args:
            strategy_id: Strategy ID.
            name: New name (optional).
            dsl_config: New DSL config (optional).
            description: New description (optional).
            is_active: New active status (optional).
        """
        updates = ["updated_at = datetime('now')"]
        params: list[Any] = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if dsl_config is not None:
            updates.append("dsl_config = ?")
            params.append(json.dumps(dsl_config))
            updates.append("version = version + 1")

        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if is_active is not None:
            updates.append("is_active = ?")
            params.append(is_active)

        params.append(strategy_id)
        with self._write_lock:
            self.conn.execute(f"UPDATE strategies SET {', '.join(updates)} WHERE id = ?", params)
            self.conn.commit()

    # --- Sizing Config CRUD ---

    def create_sizing_config(self, name: str, method: str, config: JsonDict) -> int:
        """Create a sizing configuration.

        Args:
            name: Unique config name.
            method: Sizing method (fixed_fractional, kelly, atr).
            config: Method-specific parameters.

        Returns:
            ID of created config.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                "INSERT INTO sizing_configs (name, method, config) VALUES (?, ?, ?)",
                (name, method, json.dumps(config)),
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def get_sizing_config(self, config_id: int) -> JsonDict | None:
        """Get sizing config by ID."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM sizing_configs WHERE id = ?", (config_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["config"] = json.loads(result["config"])
        return result

    def list_sizing_configs(self) -> list[JsonDict]:
        """List all sizing configs."""
        with self._write_lock:
            rows = self.conn.execute(
                "SELECT * FROM sizing_configs ORDER BY name"
            ).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["config"] = json.loads(result["config"])
            results.append(result)
        return results

    def update_sizing_config(
        self, config_id: int, name: str, method: str, config: JsonDict
    ) -> None:
        """Update an existing sizing configuration."""
        with self._write_lock:
            self.conn.execute(
                "UPDATE sizing_configs SET name = ?, method = ?, config = ? WHERE id = ?",
                (name, method, json.dumps(config), config_id),
            )
            self.conn.commit()

    def delete_sizing_config(self, config_id: int) -> None:
        """Delete a sizing configuration by ID."""
        with self._write_lock:
            self.conn.execute("DELETE FROM sizing_configs WHERE id = ?", (config_id,))
            self.conn.commit()

    # --- Risk Config CRUD ---

    def create_risk_config(
        self, name: str, strategy_level: JsonDict, portfolio_level: JsonDict
    ) -> int:
        """Create a risk configuration.

        Args:
            name: Unique config name.
            strategy_level: Strategy-level risk parameters.
            portfolio_level: Portfolio-level risk parameters.

        Returns:
            ID of created config.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO risk_configs (name, strategy_level, portfolio_level)
                   VALUES (?, ?, ?)""",
                (name, json.dumps(strategy_level), json.dumps(portfolio_level)),
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def get_risk_config(self, config_id: int) -> JsonDict | None:
        """Get risk config by ID."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM risk_configs WHERE id = ?", (config_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["strategy_level"] = json.loads(result["strategy_level"])
        result["portfolio_level"] = json.loads(result["portfolio_level"])
        return result

    def list_risk_configs(self) -> list[JsonDict]:
        """List all risk configs."""
        with self._write_lock:
            rows = self.conn.execute("SELECT * FROM risk_configs ORDER BY name").fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["strategy_level"] = json.loads(result["strategy_level"])
            result["portfolio_level"] = json.loads(result["portfolio_level"])
            results.append(result)
        return results

    def update_risk_config(
        self, config_id: int, name: str, strategy_level: JsonDict, portfolio_level: JsonDict
    ) -> None:
        """Update an existing risk configuration."""
        with self._write_lock:
            self.conn.execute(
                "UPDATE risk_configs SET name = ?, strategy_level = ?, portfolio_level = ? WHERE id = ?",
                (name, json.dumps(strategy_level), json.dumps(portfolio_level), config_id),
            )
            self.conn.commit()

    def delete_risk_config(self, config_id: int) -> None:
        """Delete a risk configuration by ID."""
        with self._write_lock:
            self.conn.execute("DELETE FROM risk_configs WHERE id = ?", (config_id,))
            self.conn.commit()

    # --- System state (kill switch) ---

    def _select_system_state_locked(self) -> JsonDict | None:
        """The singleton row as a dict, or None when id=1 is absent.

        Caller holds ``_write_lock`` (the name keeps this true under the
        structural lock test).
        """
        row = self.conn.execute(
            "SELECT kill_switch, reason, killed_at, killed_by, updated_at "
            "FROM system_state WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "kill_switch": bool(row[0]),
            "reason": row[1],
            "killed_at": row[2],
            "killed_by": row[3],
            "updated_at": row[4],
        }

    def get_system_state(self) -> JsonDict:
        """Read the singleton system_state row, re-creating it ENGAGED if absent.

        A missing row is never answered with disengaged defaults: the row is
        re-created engaged (with ``SYSTEM_STATE_RESEEDED_REASON``), committed and
        re-selected, so every caller -- the paper API refusal, the node poll, the
        header status endpoint -- sees the halt (vibe-quant-tdjop).

        Returns:
            Dict with keys: kill_switch (bool), reason (str|None),
            killed_at (str|None), killed_by (str|None), updated_at (str).

        Raises:
            SystemStateMissingError: If the row is absent and re-creating it
                fails (the INSERT raised a ``sqlite3.Error``, or the row is still
                not readable afterwards).
        """
        with self._write_lock:
            return self._read_or_reseed_system_state_locked()

    def _read_or_reseed_system_state_locked(self) -> JsonDict:
        """Read, or re-create engaged and re-read. Caller holds ``_write_lock``."""
        state = self._select_system_state_locked()
        if state is not None:
            return state
        try:
            reseed_system_state(self.conn, engaged=True)
            self.conn.commit()
            state = self._select_system_state_locked()
        except sqlite3.Error as exc:
            raise SystemStateMissingError(SYSTEM_STATE_UNRESEEDABLE_MSG) from exc
        if state is None:
            raise SystemStateMissingError(SYSTEM_STATE_UNRESEEDABLE_MSG)
        return state

    def set_kill_switch(self, reason: str, killed_by: str | None = None) -> None:
        """Engage the kill switch. Idempotent - re-setting updates reason.

        An upsert: a missing row is inserted, not silently skipped (the old
        UPDATE-only form was a no-op on a deleted row, so the kill command could
        report success while nothing was persisted).
        """
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO system_state
                       (id, kill_switch, reason, killed_at, killed_by, updated_at)
                   VALUES (1, 1, ?, datetime('now'), ?, datetime('now'))
                   ON CONFLICT(id) DO UPDATE SET
                       kill_switch = 1,
                       reason = excluded.reason,
                       killed_at = excluded.killed_at,
                       killed_by = excluded.killed_by,
                       updated_at = excluded.updated_at""",
                (reason, killed_by),
            )
            self.conn.commit()

    def clear_kill_switch(self, cleared_by: str | None = None) -> None:
        """Release the kill switch. Logs who cleared it in ``killed_by``.

        On a missing row the release cannot succeed: the row is re-created
        ENGAGED and ``SystemStateMissingError`` is raised telling the operator to
        unlock again -- an unlock must never leave the switch fail-open.
        """
        with self._write_lock:
            row = self.conn.execute(
                "SELECT reason, killed_by FROM system_state WHERE id = 1"
            ).fetchone()
            # Missing row, or the row a lazy re-creation made (this manager's
            # schema init, another connection's, or a cached read): the unlock
            # is refused once so the operator sees the halt before releasing it.
            if row is None or (
                row[0] == SYSTEM_STATE_RESEEDED_REASON and row[1] == SYSTEM_STATE_RESEEDED_BY
            ):
                if row is None:
                    reseed_system_state(self.conn, engaged=True)
                self.conn.execute(
                    "UPDATE system_state SET killed_by = ? WHERE id = 1",
                    (SYSTEM_STATE_UNLOCK_ACK_BY,),
                )
                self.conn.commit()
                raise SystemStateMissingError(
                    "system_state row id=1 was missing; re-created engaged - unlock again"
                )
            self.conn.execute(
                "UPDATE system_state SET kill_switch = 0, reason = NULL, "
                "killed_at = NULL, killed_by = ?, updated_at = datetime('now') "
                "WHERE id = 1",
                (cleared_by,),
            )
            self.conn.commit()

    # --- Research per-source settings ---

    def get_research_subreddits(self, source: str) -> list[str] | None:
        """Return the saved subreddit list for ``source``, or None if unset."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT subreddits_json FROM research_settings WHERE source = ?",
                (source,),
            ).fetchone()
        if row is None:
            return None
        parsed = json.loads(row[0])
        if not isinstance(parsed, list):
            return None
        return [str(s) for s in parsed]

    def set_research_subreddits(self, source: str, subreddits: list[str]) -> None:
        """Upsert the subreddit list for ``source``."""
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO research_settings (source, subreddits_json, updated_at) "
                "VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(source) DO UPDATE SET "
                "subreddits_json = excluded.subreddits_json, "
                "updated_at = datetime('now')",
                (source, json.dumps(subreddits)),
            )
            self.conn.commit()

    def clear_research_subreddits(self, source: str) -> None:
        """Drop the saved row for ``source`` so callers fall back to env defaults."""
        with self._write_lock:
            self.conn.execute(
                "DELETE FROM research_settings WHERE source = ?", (source,)
            )
            self.conn.commit()

    # --- Backtest Run CRUD ---

    def create_backtest_run(
        self,
        strategy_id: int | None,
        run_mode: str,
        symbols: Sequence[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        parameters: JsonDict,
        sizing_config_id: int | None = None,
        risk_config_id: int | None = None,
        latency_preset: str | None = None,
    ) -> int:
        """Create a backtest run record.

        Args:
            strategy_id: Strategy ID (None for discovery runs).
            run_mode: 'screening' or 'validation'.
            symbols: List of symbols to backtest.
            timeframe: Primary timeframe.
            start_date: Start date (ISO format).
            end_date: End date (ISO format).
            parameters: Strategy parameters for this run.
            sizing_config_id: Optional sizing config ID.
            risk_config_id: Optional risk config ID.
            latency_preset: Optional latency preset name.

        Returns:
            ID of created run.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO backtest_runs
                   (strategy_id, sizing_config_id, risk_config_id, run_mode, symbols,
                    timeframe, start_date, end_date, parameters, latency_preset)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    strategy_id,
                    sizing_config_id,
                    risk_config_id,
                    run_mode,
                    json.dumps(list(symbols)),
                    timeframe,
                    start_date,
                    end_date,
                    json.dumps(parameters),
                    latency_preset,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def get_backtest_run(self, run_id: int) -> JsonDict | None:
        """Get backtest run by ID."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["symbols"] = json.loads(result["symbols"])
        result["parameters"] = json.loads(result["parameters"])
        return result

    def update_backtest_run_status(
        self,
        run_id: int,
        status: str,
        pid: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update backtest run status.

        Args:
            run_id: Run ID.
            status: New status (pending, running, completed, failed).
            pid: Process ID if running.
            error_message: Error message if failed.
        """
        updates = ["status = ?"]
        params: list[Any] = [status]

        if status == "running":
            updates.append("started_at = datetime('now')")
            if pid is not None:
                updates.append("pid = ?")
                params.append(pid)
        elif status in ("completed", "failed", "killed", "cancelled"):
            updates.append("completed_at = datetime('now')")
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

        params.append(run_id)
        with self._write_lock:
            self.conn.execute(
                f"UPDATE backtest_runs SET {', '.join(updates)} WHERE id = ?", params
            )
            self.conn.commit()

    def update_heartbeat(self, run_id: int) -> None:
        """Update heartbeat timestamp for a running backtest."""
        with self._write_lock:
            self.conn.execute(
                "UPDATE backtest_runs SET heartbeat_at = datetime('now') WHERE id = ?",
                (run_id,),
            )
            self.conn.commit()

    def list_backtest_runs(
        self, strategy_id: int | None = None, status: str | None = None
    ) -> list[JsonDict]:
        """List backtest runs with optional filters."""
        query = "SELECT * FROM backtest_runs WHERE 1=1"
        params: list[Any] = []

        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        with self._write_lock:
            rows = self.conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["symbols"] = json.loads(result["symbols"])
            result["parameters"] = json.loads(result["parameters"])
            results.append(result)
        return results

    # --- Backtest Results CRUD ---

    def save_backtest_result(
        self,
        run_id: int,
        metrics: JsonDict,
        trades: Sequence[JsonDict] | None = None,
    ) -> int:
        """Save the result of one attempt of a run, replacing any prior attempt.

        A run has exactly one result row (UNIQUE run_id). Re-running a run is a
        new attempt: its previous result row AND all its trades are deleted and
        the new row (+ ``trades``) inserted in one transaction. User notes
        (``user_notes``) survive re-runs.

        Args:
            run_id: Backtest run ID.
            metrics: Dict of metric names to values.
            trades: This attempt's trades (None/empty → the run has no trades).

        Returns:
            ID of created result.
        """
        columns = ["run_id"] + list(metrics.keys())
        _validate_columns(columns, _BACKTEST_RESULTS_COLUMNS, "backtest_results")
        placeholders = ", ".join(["?"] * len(columns))
        values = [run_id] + list(metrics.values())
        trade_sql, trade_rows = _trade_insert(run_id, trades or [])

        with self._write_lock:
            conn = self.conn
            try:
                prev = conn.execute(
                    "SELECT user_notes FROM backtest_results WHERE run_id = ? "
                    "AND user_notes IS NOT NULL ORDER BY id DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
                conn.execute("DELETE FROM backtest_results WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
                cursor = conn.execute(
                    f"INSERT INTO backtest_results ({', '.join(columns)}) VALUES ({placeholders})",
                    values,
                )
                result_id = cursor.lastrowid or 0
                if prev is not None:
                    conn.execute(
                        "UPDATE backtest_results SET user_notes = ? WHERE id = ?",
                        (prev[0], result_id),
                    )
                if trade_rows:
                    conn.executemany(trade_sql, trade_rows)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            return result_id

    def get_backtest_result(self, run_id: int) -> JsonDict | None:
        """Get the (latest) backtest result for a run."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM backtest_results WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def merge_replay_drift_payload(self, run_id: int, payload: JsonDict) -> None:
        """Persist a replay-drift payload: into the result's notes JSON if it has
        notes, else into ``backtest_runs.parameters``."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT notes FROM backtest_results WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None and row[0]:
                try:
                    notes = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    notes = {}
                if not isinstance(notes, dict):
                    notes = {}
                notes["replay_drift"] = payload
                self.conn.execute(
                    "UPDATE backtest_results SET notes = ? WHERE run_id = ?",
                    (json.dumps(notes), run_id),
                )
                self.conn.commit()
                return

            row2 = self.conn.execute(
                "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
            params: JsonDict = {}
            if row2 and row2[0]:
                try:
                    parsed = json.loads(row2[0])
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                if isinstance(parsed, dict):
                    params = parsed
            params["replay_drift"] = payload
            self.conn.execute(
                "UPDATE backtest_runs SET parameters = ? WHERE id = ?",
                (json.dumps(params), run_id),
            )
            self.conn.commit()

    def get_latest_screening_run_with_sweeps(
        self,
        strategy_id: int,
        *,
        window: tuple[str, str] | None = None,
        symbols: list[str] | None = None,
        sizing: Mapping[str, float] | None = None,
    ) -> tuple[int, str, str] | None:
        """Newest completed screening run of a strategy that has sweep rows.

        ``window`` / ``symbols`` (order-insensitive), when given, restrict it to
        runs over exactly that date range / symbol set. ``sizing`` restricts it to runs
        whose recorded sizing params equal it (``to_params()`` of the run's settings);
        a run whose stored sizing is unreadable is skipped, never a reason to raise.

        Returns ``(id, start_date, end_date)`` or None.
        """
        with self._write_lock:
            rows = self.conn.execute(
                """
                SELECT br.id, br.start_date, br.end_date, br.symbols, br.parameters
                FROM backtest_runs br
                WHERE br.strategy_id = ? AND br.run_mode = 'screening'
                      AND br.status = 'completed'
                      AND EXISTS (SELECT 1 FROM sweep_results sr WHERE sr.run_id = br.id)
                ORDER BY br.id DESC
                """,
                (strategy_id,),
            ).fetchall()
        wanted_sizing = None if sizing is None else dict(sizing)
        for run_id, start, end, run_symbols, run_parameters in rows:
            if window is not None and (start, end) != tuple(window):
                continue
            if symbols is not None and sorted(json.loads(run_symbols or "[]")) != sorted(symbols):
                continue
            if wanted_sizing is not None and (
                _recorded_sizing_params(run_parameters) != wanted_sizing
            ):
                continue
            return (run_id, start, end)
        return None

    def list_sweep_reference_rows(
        self, run_id: int
    ) -> list[tuple[str, float | None, int | None]]:
        """``(parameters, sharpe_ratio, total_trades)`` of a run's sweep rows, by id."""
        with self._write_lock:
            rows = self.conn.execute(
                "SELECT parameters, sharpe_ratio, total_trades FROM sweep_results "
                "WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
            return [(r[0], r[1], r[2]) for r in rows]

    def get_strategy_dsl_config_raw(self, strategy_id: int) -> object:
        """Stored ``strategies.dsl_config`` column as-is (unparsed); None if no such row."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT dsl_config FROM strategies WHERE id = ?", (strategy_id,)
            ).fetchone()
            return None if row is None else row[0]

    def list_completed_discovery_notes(
        self, notes_like: str | None = None
    ) -> list[tuple[int, str, str, str, str]]:
        """``(run_id, notes, start_date, end_date, symbols)`` of completed discovery
        runs that have a result row, newest first; optionally ``notes LIKE`` a pattern."""
        sql = """
            SELECT br.id, res.notes, br.start_date, br.end_date, br.symbols
            FROM backtest_runs br
            JOIN backtest_results res ON res.run_id = br.id
            WHERE br.run_mode = 'discovery' AND br.status = 'completed'
        """
        params: tuple[str, ...] = ()
        if notes_like is not None:
            sql += " AND res.notes LIKE ?"
            params = (notes_like,)
        with self._write_lock:
            rows = self.conn.execute(sql + " ORDER BY br.id DESC", params).fetchall()
            return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def list_backtest_results(
        self, strategy_id: int | None = None, limit: int | None = None
    ) -> list[JsonDict]:
        """List backtest results with optional filters.

        Joins backtest_results with backtest_runs to allow filtering by
        strategy_id and to include run metadata in results.

        Args:
            strategy_id: Filter by strategy ID (optional).
            limit: Maximum number of results to return (optional).

        Returns:
            List of backtest result dicts, ordered by run creation time descending.
        """
        query = """
            SELECT br.*, r.strategy_id, r.run_mode, r.symbols, r.timeframe,
                   r.start_date, r.end_date, r.status, r.created_at AS run_created_at
            FROM backtest_results br
            JOIN backtest_runs r ON br.run_id = r.id
            WHERE br.id = (SELECT MAX(id) FROM backtest_results WHERE run_id = r.id)
        """
        params: list[Any] = []

        if strategy_id is not None:
            query += " AND r.strategy_id = ?"
            params.append(strategy_id)

        query += " ORDER BY r.created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._write_lock:
            rows = self.conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            # Parse JSON fields from the joined run data
            if "symbols" in result and result["symbols"] is not None:
                result["symbols"] = json.loads(result["symbols"])
            results.append(result)
        return results

    def update_result_notes(self, run_id: int, notes: str) -> None:
        """Overwrite the machine ``notes`` JSON of a run's result.

        Pipeline-internal (discovery payload, consistency flags, compiler
        version). User-typed notes go through :meth:`update_result_user_notes`.
        """
        with self._write_lock:
            self.conn.execute(
                "UPDATE backtest_results SET notes = ? WHERE run_id = ?",
                (notes, run_id),
            )
            self.conn.commit()

    def update_result_user_notes(self, run_id: int, user_notes: str) -> bool:
        """Set the user's free-text notes on a run's result.

        Returns:
            False if the run has no result row.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                "UPDATE backtest_results SET user_notes = ? WHERE run_id = ?",
                (user_notes, run_id),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    # --- Trade CRUD ---

    def save_trade(self, run_id: int, trade: JsonDict) -> int:
        """Save a single trade record.

        Args:
            run_id: Backtest run ID.
            trade: Trade data dict.

        Returns:
            ID of created trade.
        """
        trade_data = {"run_id": run_id, **trade}
        columns = list(trade_data.keys())
        _validate_columns(columns, _TRADES_COLUMNS, "trades")
        placeholders = ", ".join(["?"] * len(columns))

        with self._write_lock:
            cursor = self.conn.execute(
                f"INSERT INTO trades ({', '.join(columns)}) VALUES ({placeholders})",
                list(trade_data.values()),
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def save_trades_batch(self, run_id: int, trades: Sequence[JsonDict]) -> None:
        """Replace all trades of a run with ``trades`` (one transaction).

        A batch is the complete trade list of one attempt; saving again
        (re-run) must replace, never append duplicates.

        Args:
            run_id: Backtest run ID.
            trades: List of trade data dicts.
        """
        sql, rows = _trade_insert(run_id, trades)
        with self._write_lock:
            conn = self.conn
            try:
                conn.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
                if rows:
                    conn.executemany(sql, rows)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def get_trades(self, run_id: int) -> list[JsonDict]:
        """Get all trades for a backtest run."""
        with self._write_lock:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY entry_time", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # --- Sweep Results CRUD ---

    def save_sweep_result(self, run_id: int, result: JsonDict) -> int:
        """Save a single sweep result.

        Args:
            run_id: Backtest run ID.
            result: Sweep result dict including parameters and metrics.

        Returns:
            ID of created result.
        """
        result_data = {"run_id": run_id, **result}
        if "parameters" in result_data:
            result_data["parameters"] = json.dumps(result_data["parameters"])

        columns = list(result_data.keys())
        _validate_columns(columns, _SWEEP_RESULTS_COLUMNS, "sweep_results")
        placeholders = ", ".join(["?"] * len(columns))

        with self._write_lock:
            cursor = self.conn.execute(
                f"INSERT INTO sweep_results ({', '.join(columns)}) VALUES ({placeholders})",
                list(result_data.values()),
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def save_sweep_results_batch(self, run_id: int, results: Sequence[JsonDict]) -> None:
        """Save multiple sweep results in a batch.

        Deletes any existing sweep_results for run_id first to prevent
        duplicates on re-run.
        """
        if not results:
            return

        processed = []
        for r in results:
            result_data = {"run_id": run_id, **r}
            if "parameters" in result_data:
                result_data["parameters"] = json.dumps(result_data["parameters"])
            processed.append(result_data)

        columns = list(processed[0].keys())
        _validate_columns(columns, _SWEEP_RESULTS_COLUMNS, "sweep_results")
        placeholders = ", ".join(["?"] * len(columns))

        with self._write_lock:
            self.conn.execute(
                "DELETE FROM sweep_results WHERE run_id = ?", (run_id,)
            )
            self.conn.executemany(
                f"INSERT INTO sweep_results ({', '.join(columns)}) VALUES ({placeholders})",
                [list(r.values()) for r in processed],
            )
            self.conn.commit()

    def get_sweep_results(self, run_id: int, pareto_only: bool = False) -> list[JsonDict]:
        """Get sweep results for a backtest run.

        Args:
            run_id: Backtest run ID.
            pareto_only: If True, only return Pareto-optimal results.

        Returns:
            List of sweep result dicts.
        """
        query = "SELECT * FROM sweep_results WHERE run_id = ?"
        if pareto_only:
            query += " AND is_pareto_optimal = 1"
        query += " ORDER BY sharpe_ratio DESC"

        with self._write_lock:
            rows = self.conn.execute(query, (run_id,)).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["parameters"] = json.loads(result["parameters"])
            # .get: hand-rolled sweep_results fixtures (7 test files) lack the column.
            if result.get("exit_timing") is not None:
                result["exit_timing"] = json.loads(result["exit_timing"])
            results.append(result)
        return results

    def mark_pareto_optimal(self, result_ids: Sequence[int]) -> None:
        """Mark sweep results as Pareto optimal.

        Args:
            result_ids: IDs of results to mark as Pareto optimal.
        """
        if not result_ids:
            return
        placeholders = ", ".join(["?"] * len(result_ids))
        with self._write_lock:
            self.conn.execute(
                f"UPDATE sweep_results SET is_pareto_optimal = 1 WHERE id IN ({placeholders})",
                list(result_ids),
            )
            self.conn.commit()

    # --- Background Jobs CRUD ---
    # NOTE: Job operations are also implemented in jobs/manager.py (BacktestJobManager).
    # StateManager provides low-level CRUD; BacktestJobManager adds subprocess lifecycle,
    # heartbeat, and process management. Both operate on the same background_jobs table.
    # Callers should prefer BacktestJobManager for new code.

    def register_job(
        self, run_id: int, pid: int, job_type: str, log_file: str | None = None
    ) -> int:
        """Register a background job.

        Args:
            run_id: Associated backtest run ID.
            pid: Process ID.
            job_type: Job type (screening, validation, data_update).
            log_file: Optional log file path.

        Returns:
            ID of created job.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO background_jobs (run_id, pid, job_type, log_file)
                   VALUES (?, ?, ?, ?)""",
                (run_id, pid, job_type, log_file),
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def get_job(self, run_id: int) -> JsonDict | None:
        """Get job by run ID."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM background_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_running_jobs(self) -> list[JsonDict]:
        """Get all running jobs."""
        with self._write_lock:
            rows = self.conn.execute(
                "SELECT * FROM background_jobs WHERE status = 'running'"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_job_status(self, run_id: int, status: str, error: str | None = None) -> None:
        """Update job status.

        Args:
            run_id: Run ID.
            status: New status (running, completed, failed, killed).
            error: Optional error message.
        """
        with self._write_lock:
            if status in ("completed", "failed", "killed"):
                self.conn.execute(
                    """UPDATE background_jobs
                       SET status = ?, completed_at = datetime('now'), error_message = ?
                       WHERE run_id = ?""",
                    (status, error, run_id),
                )
            else:
                self.conn.execute(
                    "UPDATE background_jobs SET status = ? WHERE run_id = ?",
                    (status, run_id),
                )
            self.conn.commit()

    def list_runs_with_results(
        self,
        strategy_id: int | None = None,
        run_mode: str | None = None,
        status: str | None = None,
    ) -> list[JsonDict]:
        """List runs joined with strategy name + key result metrics.

        LEFT JOINs so pending/running/failed runs still appear.
        """
        query = """
            SELECT
                r.id AS run_id,
                r.strategy_id,
                s.name AS strategy_name,
                r.run_mode,
                r.symbols,
                r.timeframe,
                r.status,
                r.created_at,
                r.completed_at,
                COALESCE(br.total_return, sw.total_return) AS total_return,
                COALESCE(br.sharpe_ratio, sw.sharpe_ratio) AS sharpe_ratio,
                COALESCE(br.max_drawdown, sw.max_drawdown) AS max_drawdown,
                COALESCE(br.total_trades, sw.total_trades) AS total_trades,
                br.winning_trades,
                br.losing_trades,
                COALESCE(br.win_rate, sw.win_rate) AS win_rate,
                COALESCE(br.profit_factor, sw.profit_factor) AS profit_factor
            FROM backtest_runs r
            LEFT JOIN strategies s ON r.strategy_id = s.id
            LEFT JOIN backtest_results br ON br.id = (
                SELECT MAX(id) FROM backtest_results WHERE run_id = r.id
            )
            LEFT JOIN (
                SELECT run_id,
                       total_return, sharpe_ratio, max_drawdown,
                       total_trades, win_rate, profit_factor,
                       ROW_NUMBER() OVER (
                           PARTITION BY run_id ORDER BY sharpe_ratio DESC
                       ) AS rn
                FROM sweep_results
            ) sw ON r.id = sw.run_id AND sw.rn = 1
            WHERE 1=1
        """
        params: list[Any] = []

        if strategy_id is not None:
            query += " AND r.strategy_id = ?"
            params.append(strategy_id)
        if run_mode is not None:
            query += " AND r.run_mode = ?"
            params.append(run_mode)
        if status is not None:
            query += " AND r.status = ?"
            params.append(status)

        query += " ORDER BY r.created_at DESC"

        with self._write_lock:
            rows = self.conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            if result.get("symbols") is not None:
                result["symbols"] = json.loads(result["symbols"])
            results.append(result)
        return results

    def update_job_heartbeat(self, run_id: int) -> None:
        """Update job heartbeat timestamp."""
        with self._write_lock:
            self.conn.execute(
                """UPDATE background_jobs SET heartbeat_at = datetime('now')
                   WHERE run_id = ?""",
                (run_id,),
            )
            self.conn.commit()

    # --- Research pipeline ---

    def create_research_item(
        self,
        *,
        source: str,
        external_id: str,
        url: str,
        title: str | None,
        body: str | None,
        author: str | None,
        posted_at: str | None,
        score: int | None,
        extras: JsonDict | None = None,
    ) -> int:
        """Insert a research_item. Raises DuplicateResearchItem if (source, external_id) exists."""
        with self._write_lock:
            try:
                cursor = self.conn.execute(
                    """INSERT INTO research_items
                       (source, external_id, url, title, body, author, posted_at, score, extras_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        source,
                        external_id,
                        url,
                        title,
                        body,
                        author,
                        posted_at,
                        score,
                        json.dumps(extras) if extras is not None else None,
                    ),
                )
                self.conn.commit()
                assert cursor.lastrowid is not None
                return cursor.lastrowid
            except sqlite3.IntegrityError as e:
                raise DuplicateResearchItem(
                    f"research_item already exists: source={source} external_id={external_id}"
                ) from e

    def get_research_item(self, item_id: int) -> JsonDict | None:
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM research_items WHERE id = ?", (item_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("extras_json") is not None:
            d["extras"] = json.loads(d["extras_json"])
        else:
            d["extras"] = {}
        return d

    def list_research_items(
        self,
        *,
        source: str | None = None,
        status: str | None = None,
        sort: str = "newest_scraped",
        limit: int = 50,
        offset: int = 0,
        hide_low_trade: bool = False,
        q: str | None = None,
    ) -> list[JsonDict]:
        order_clauses = {
            "newest_scraped": "i.fetched_at DESC, i.id DESC",
            "newest_posted": "i.posted_at DESC, i.id DESC",
            "highest_score": "i.score IS NULL, i.score DESC, i.id DESC",
            "highest_confidence": "latest_confidence IS NULL, latest_confidence DESC, i.id DESC",
            "credibility": "credibility_score IS NULL, credibility_score DESC, i.id DESC",
            "screen_sharpe": "max_screen_sharpe IS NULL, max_screen_sharpe DESC, i.id DESC",
        }
        order_by = order_clauses.get(sort, order_clauses["newest_scraped"])
        # Project only list-view columns. body and extras_json (which holds
        # the comments array) can be MBs per row — fetch them in get_research_item only.
        query = (
            "SELECT i.id, i.source, i.external_id, i.url, i.title, i.author,"
            " i.posted_at, i.score, i.fetched_at, i.extraction_status,"
            " json_extract(i.extras_json, '$.num_comments') AS num_comments, ("
            " SELECT confidence FROM research_extractions e"
            " WHERE e.research_item_id = i.id"
            " ORDER BY e.extracted_at DESC, e.id DESC LIMIT 1"
            ") AS latest_confidence, ("
            # Tier weight *3 vs completeness range [-1,1] (width 2) keeps evidence
            # tiers non-overlapping: evidence level always dominates completeness.
            " SELECT MAX((CASE e.evidence_level"
            " WHEN 'live_traded' THEN 3 WHEN 'backtested' THEN 2"
            " WHEN 'idea_only' THEN 1 END) * 3 + COALESCE(e.completeness, -1))"
            " FROM research_extractions e WHERE e.research_item_id = i.id"
            ") AS credibility_score, ("
            " SELECT MAX(screen_sharpe) FROM research_extractions e"
            " WHERE e.research_item_id = i.id"
            ") AS max_screen_sharpe FROM research_items i WHERE 1=1"
        )
        params: list[Any] = []
        if source is not None:
            query += " AND i.source = ?"
            params.append(source)
        if status is not None:
            query += " AND i.extraction_status = ?"
            params.append(status)
        if hide_low_trade:
            # Hide items where ≥1 extraction has trade-count data AND no extraction reached MIN_TRADES.
            # Items with no extractions, or no extraction with screen_trades populated, are NOT hidden.
            query += (
                " AND NOT ("
                "  EXISTS (SELECT 1 FROM research_extractions e"
                "    WHERE e.research_item_id = i.id AND e.screen_trades IS NOT NULL)"
                "  AND NOT EXISTS (SELECT 1 FROM research_extractions e"
                "    WHERE e.research_item_id = i.id AND e.screen_trades >= 50)"
                " )"
            )
        if q:
            query += " AND LOWER(i.title) LIKE ?"
            params.append(f"%{q.lower()}%")
        query += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._write_lock:
            rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def update_research_item_status(self, item_id: int, status: str) -> bool:
        with self._write_lock:
            cursor = self.conn.execute(
                "UPDATE research_items SET extraction_status = ? WHERE id = ?",
                (status, item_id),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def create_extraction(
        self,
        *,
        research_item_id: int,
        status: str,
        llm_model: str | None,
        confidence: float | None,
        rationale: str | None,
        raw_response: str,
        dsl_yaml: str | None,
        parsed_dsl_json: str | None,
        parse_error: str | None,
        evidence_level: str | None = None,
        completeness: float | None = None,
        proposed_indicators_json: str | None = None,
        prompt: str | None = None,
        risk_management_json: str | None = None,
        notable_parameters_json: str | None = None,
    ) -> int:
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO research_extractions
                   (research_item_id, status, llm_model, confidence, rationale,
                    raw_response, dsl_yaml, parsed_dsl_json, parse_error,
                    proposed_indicators_json, prompt, evidence_level, completeness,
                    risk_management_json, notable_parameters_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    research_item_id,
                    status,
                    llm_model,
                    confidence,
                    rationale,
                    raw_response,
                    dsl_yaml,
                    parsed_dsl_json,
                    parse_error,
                    proposed_indicators_json,
                    prompt,
                    evidence_level,
                    completeness,
                    risk_management_json,
                    notable_parameters_json,
                ),
            )
            self.conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def list_extractions_for_item(self, item_id: int) -> list[JsonDict]:
        with self._write_lock:
            rows = self.conn.execute(
                """SELECT * FROM research_extractions
                   WHERE research_item_id = ?
                   ORDER BY extracted_at DESC, id DESC""",
                (item_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_extraction(self, extraction_id: int) -> JsonDict | None:
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM research_extractions WHERE id = ?", (extraction_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_extraction_status(
        self,
        extraction_id: int,
        status: str,
        strategy_id: int | None = None,
    ) -> bool:
        with self._write_lock:
            if strategy_id is not None:
                cursor = self.conn.execute(
                    "UPDATE research_extractions SET status = ?, strategy_id = ? WHERE id = ?",
                    (status, strategy_id, extraction_id),
                )
            else:
                cursor = self.conn.execute(
                    "UPDATE research_extractions SET status = ? WHERE id = ?",
                    (status, extraction_id),
                )
            self.conn.commit()
            return cursor.rowcount > 0

    def update_extraction_screen_results(
        self,
        extraction_id: int,
        *,
        screen_sharpe: float | None,
        screen_status: str,
        screen_run_id: int | None,
        screen_pf: float | None = None,
        screen_max_dd: float | None = None,
        screen_return: float | None = None,
        screen_trades: int | None = None,
        screen_error: str | None = None,
        screen_completed_at: str | None = None,
    ) -> bool:
        with self._write_lock:
            cursor = self.conn.execute(
                """UPDATE research_extractions
                   SET screen_sharpe = ?, screen_status = ?, screen_run_id = ?,
                       screen_pf = ?, screen_max_dd = ?, screen_return = ?,
                       screen_trades = ?, screen_error = ?, screen_completed_at = ?
                   WHERE id = ?""",
                (
                    screen_sharpe,
                    screen_status,
                    screen_run_id,
                    screen_pf,
                    screen_max_dd,
                    screen_return,
                    screen_trades,
                    screen_error,
                    screen_completed_at,
                    extraction_id,
                ),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def get_indicator_scaffold(
        self, extraction_id: int, idx: int
    ) -> JsonDict | None:
        """Return the cached scaffold outcome for (extraction_id, idx) or None."""
        with self._write_lock:
            row = self.conn.execute(
                """SELECT * FROM research_indicator_scaffolds
                   WHERE extraction_id = ? AND idx = ?""",
                (extraction_id, idx),
            ).fetchone()
        return dict(row) if row else None

    def list_indicator_scaffolds(self, extraction_id: int) -> list[JsonDict]:
        """All scaffold rows for an extraction, indexed by idx ascending."""
        with self._write_lock:
            rows = self.conn.execute(
                """SELECT * FROM research_indicator_scaffolds
                   WHERE extraction_id = ?
                   ORDER BY idx""",
                (extraction_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_indicator_scaffold(
        self,
        *,
        extraction_id: int,
        idx: int,
        status: str,
        plugin_path: str | None = None,
        test_path: str | None = None,
        commit_sha: str | None = None,
        error: str | None = None,
        test_output: str | None = None,
    ) -> JsonDict:
        """Insert or replace the scaffold row for (extraction_id, idx)."""
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO research_indicator_scaffolds
                       (extraction_id, idx, status, plugin_path, test_path,
                        commit_sha, error, test_output, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(extraction_id, idx) DO UPDATE SET
                       status = excluded.status,
                       plugin_path = excluded.plugin_path,
                       test_path = excluded.test_path,
                       commit_sha = excluded.commit_sha,
                       error = excluded.error,
                       test_output = excluded.test_output,
                       updated_at = datetime('now')""",
                (
                    extraction_id,
                    idx,
                    status,
                    plugin_path,
                    test_path,
                    commit_sha,
                    error,
                    test_output,
                ),
            )
            self.conn.commit()
        row = self.get_indicator_scaffold(extraction_id, idx)
        assert row is not None
        return row

    def find_scaffold_provenance_by_plugin_path(
        self, plugin_path_suffix: str
    ) -> JsonDict | None:
        """Return provenance (extraction_id, item_id, item_url) for a plugin file.

        Used by the promote-indicator endpoint to attach the originating
        research-item URL to the ``bd remember`` fact. Matches on
        ``plugin_path LIKE '%<suffix>'`` so both repo-relative and
        absolute storage of the path resolve to the same row.

        Returns ``None`` when no scaffold row matches — promote still
        runs (the file exists), just without source-URL provenance.
        """
        with self._write_lock:
            row = self.conn.execute(
                """SELECT
                       s.extraction_id    AS extraction_id,
                       e.research_item_id AS research_item_id,
                       i.url              AS item_url
                   FROM research_indicator_scaffolds s
                   JOIN research_extractions e ON e.id = s.extraction_id
                   JOIN research_items       i ON i.id = e.research_item_id
                   WHERE s.status = 'ok'
                     AND s.plugin_path LIKE ?
                   ORDER BY s.updated_at DESC
                   LIMIT 1""",
                (f"%{plugin_path_suffix}",),
            ).fetchone()
        return dict(row) if row else None

    def delete_indicator_scaffold(self, extraction_id: int, idx: int) -> bool:
        """Drop the cached scaffold row. Returns True if a row was deleted."""
        with self._write_lock:
            cursor = self.conn.execute(
                """DELETE FROM research_indicator_scaffolds
                   WHERE extraction_id = ? AND idx = ?""",
                (extraction_id, idx),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def create_scrape_run(
        self,
        *,
        source: str,
        pid: int | None,
        config: JsonDict | None = None,
    ) -> int:
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO research_scrape_runs (source, pid, config_json, heartbeat_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (source, pid, json.dumps(config) if config else None),
            )
            self.conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def get_scrape_run(self, run_id: int) -> JsonDict | None:
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM research_scrape_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def latest_scrape_run(self, source: str) -> JsonDict | None:
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM research_scrape_runs WHERE source = ? ORDER BY id DESC LIMIT 1",
                (source,),
            ).fetchone()
        return dict(row) if row else None

    def adopt_scrape_run(self, run_id: int, pid: int) -> bool:
        """Set pid + heartbeat on an existing pending scrape_run row.

        Used when the API pre-creates the row and a subprocess takes it
        over. Status remains 'running' (the schema default).
        """
        with self._write_lock:
            cursor = self.conn.execute(
                """UPDATE research_scrape_runs
                   SET pid = ?, status = 'running',
                       started_at = COALESCE(started_at, datetime('now')),
                       heartbeat_at = datetime('now')
                   WHERE id = ?""",
                (pid, run_id),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def list_active_scrape_runs(self, source: str | None = None) -> list[JsonDict]:
        """Return scrape_run rows currently in status='running'.

        Pass `source` to filter to a single source.
        """
        with self._write_lock:
            if source is None:
                rows = self.conn.execute(
                    "SELECT * FROM research_scrape_runs WHERE status = 'running' ORDER BY id DESC"
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM research_scrape_runs WHERE status = 'running' AND source = ? ORDER BY id DESC",
                    (source,),
                ).fetchall()
        return [dict(r) for r in rows]

    def count_research_items(
        self,
        *,
        source: str | None = None,
        status: str | None = None,
        hide_low_trade: bool = False,
        q: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) AS c FROM research_items i WHERE 1=1"
        params: list[Any] = []
        if source is not None:
            query += " AND i.source = ?"
            params.append(source)
        if status is not None:
            query += " AND i.extraction_status = ?"
            params.append(status)
        if hide_low_trade:
            query += (
                " AND NOT ("
                "  EXISTS (SELECT 1 FROM research_extractions e"
                "    WHERE e.research_item_id = i.id AND e.screen_trades IS NOT NULL)"
                "  AND NOT EXISTS (SELECT 1 FROM research_extractions e"
                "    WHERE e.research_item_id = i.id AND e.screen_trades >= 50)"
                " )"
            )
        if q:
            query += " AND LOWER(i.title) LIKE ?"
            params.append(f"%{q.lower()}%")
        with self._write_lock:
            row = self.conn.execute(query, params).fetchone()
        return int(row["c"]) if row else 0

    def update_scrape_run_heartbeat(self, run_id: int) -> bool:
        with self._write_lock:
            cursor = self.conn.execute(
                "UPDATE research_scrape_runs SET heartbeat_at = datetime('now') WHERE id = ?",
                (run_id,),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def increment_scrape_run_counters(
        self,
        run_id: int,
        *,
        fetched: int = 0,
        new: int = 0,
        extracted: int = 0,
        failed: int = 0,
    ) -> None:
        with self._write_lock:
            self.conn.execute(
                """UPDATE research_scrape_runs SET
                       items_fetched = items_fetched + ?,
                       items_new = items_new + ?,
                       items_extracted = items_extracted + ?,
                       items_failed = items_failed + ?
                   WHERE id = ?""",
                (fetched, new, extracted, failed, run_id),
            )
            self.conn.commit()

    def complete_scrape_run(
        self,
        run_id: int,
        *,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        with self._write_lock:
            cursor = self.conn.execute(
                """UPDATE research_scrape_runs
                   SET status = ?, completed_at = datetime('now'), error_message = ?
                   WHERE id = ?""",
                (status, error_message, run_id),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    # ---------- extraction queue ----------

    def enqueue_extraction_job(
        self,
        research_item_id: int,
        *,
        max_attempts: int = 3,
    ) -> int:
        """Insert a queued extraction job and mark the item as queued.

        Returns the new job id. Caller should ensure the item exists and
        is not already running (the router checks this).
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO research_extraction_jobs
                       (research_item_id, status, queued_at, max_attempts)
                   VALUES (?, 'queued', datetime('now'), ?)""",
                (research_item_id, max_attempts),
            )
            self.conn.execute(
                "UPDATE research_items SET extraction_status = 'queued' WHERE id = ?",
                (research_item_id,),
            )
            self.conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def claim_next_extraction_job(self) -> JsonDict | None:
        """Atomically claim the oldest queued job.

        Uses UPDATE … RETURNING in a single statement so two workers cannot
        both grab the same row. Returns None when the queue is empty.
        Sets started_at on first claim and bumps heartbeat_at on every claim.
        Also propagates the running status to the parent research_items row so
        the UI badge transitions queued → running.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                """UPDATE research_extraction_jobs
                   SET status = 'running',
                       started_at = COALESCE(started_at, datetime('now')),
                       heartbeat_at = datetime('now')
                   WHERE id = (
                       SELECT id FROM research_extraction_jobs
                       WHERE status = 'queued'
                       ORDER BY id ASC
                       LIMIT 1
                   )
                   RETURNING *"""
            )
            row = cursor.fetchone()
            if row is not None:
                self.conn.execute(
                    "UPDATE research_items SET extraction_status = 'running' WHERE id = ?",
                    (int(row["research_item_id"]),),
                )
            self.conn.commit()
            return dict(row) if row else None

    def heartbeat_extraction_job(self, job_id: int) -> bool:
        """Bump heartbeat_at for a running job. Returns False if not running."""
        with self._write_lock:
            cursor = self.conn.execute(
                """UPDATE research_extraction_jobs
                   SET heartbeat_at = datetime('now')
                   WHERE id = ? AND status = 'running'""",
                (job_id,),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def fail_extraction_job(
        self,
        job_id: int,
        error_message: str,
    ) -> JsonDict:
        """Record a failure: increment attempts, set last_error.

        If attempts < max_attempts → re-queue (status='queued', clear
        started_at so the next claim resets it; item status back to 'queued').
        Otherwise mark status='failed' and the item 'failed'.

        Returns the post-update row dict. Raises if the job does not exist.
        """
        with self._write_lock:
            self.conn.execute(
                """UPDATE research_extraction_jobs
                   SET attempts = attempts + 1,
                       last_error = ?,
                       error_message = ?
                   WHERE id = ?""",
                (error_message, error_message, job_id),
            )
            row = self.conn.execute(
                "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                self.conn.commit()
                raise ValueError(f"extraction job {job_id} not found")
            job = dict(row)
            item_id = int(job["research_item_id"])
            if int(job["attempts"]) < int(job["max_attempts"]):
                self.conn.execute(
                    """UPDATE research_extraction_jobs
                       SET status = 'queued', started_at = NULL,
                           heartbeat_at = NULL, completed_at = NULL
                       WHERE id = ?""",
                    (job_id,),
                )
                self.conn.execute(
                    "UPDATE research_items SET extraction_status = 'queued' WHERE id = ?",
                    (item_id,),
                )
                final_status = "queued"
            else:
                self.conn.execute(
                    """UPDATE research_extraction_jobs
                       SET status = 'failed', completed_at = datetime('now')
                       WHERE id = ?""",
                    (job_id,),
                )
                self.conn.execute(
                    "UPDATE research_items SET extraction_status = 'failed' WHERE id = ?",
                    (item_id,),
                )
                final_status = "failed"
            self.conn.commit()
            job["status"] = final_status
            return job

    def cancel_queued_extraction_job(
        self,
        job_id: int,
    ) -> JsonDict | None:
        """Cancel a queued extraction job and restore the parent item's status.

        Only acts on jobs in status='queued'. Returns the updated job row on
        success, None if the job doesn't exist, or raises ValueError if the
        job is in a state that cannot be cancelled (running/done/failed).

        Item status is reset to the latest non-running snapshot we can
        compute: if a previous extraction exists, mirror its rolled-up
        status; otherwise 'pending'.
        """
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            job = dict(row)
            status = str(job["status"])
            if status == "cancelled":
                return job  # idempotent
            if status != "queued":
                raise ValueError(
                    f"job {job_id} cannot be cancelled from status={status!r}"
                )
            if not self._mark_cancelled_locked(
                job_id, int(job["research_item_id"]), error_message=None
            ):
                # Cross-process race: the job left 'queued' under us.
                raise ValueError(
                    f"job {job_id} cannot be cancelled from status != 'queued'"
                )
            self.conn.commit()
            job["status"] = "cancelled"
            return job

    def request_cancel_extraction_job(self, job_id: int) -> JsonDict | None:
        """Request cancellation of a queued or running extraction job.

        - queued    -> cancelled immediately (same as cancel_queued_extraction_job).
        - running   -> stamps ``cancel_requested_at``; the worker interrupts this
          job's subprocess and finalizes the row as 'cancelled'.
        - cancelled -> idempotent no-op.
        - done/failed -> raises ValueError (already has a terminal outcome).

        Every UPDATE carries a ``status`` predicate + rowcount check, so a
        cross-process transition between the SELECT and the UPDATE (e.g. the
        API cancelling a job a worker just claimed) can never stamp or
        overwrite the wrong state; on such a race the loop re-reads and
        re-dispatches. Returns the post-update row, or None if the job does
        not exist.
        """
        with self._write_lock:
            for _ in range(4):
                row = self.conn.execute(
                    "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    return None
                job = dict(row)
                status = str(job["status"])
                if status == "cancelled":
                    return job  # idempotent
                if status in ("done", "failed"):
                    raise ValueError(
                        f"job {job_id} cannot be cancelled from status={status!r}"
                    )
                if status == "queued":
                    if self._mark_cancelled_locked(
                        job_id, int(job["research_item_id"]), error_message=None
                    ):
                        self.conn.commit()
                        job["status"] = "cancelled"
                        return job
                    continue  # raced: re-read
                # Running: flag it. COALESCE keeps the first request time.
                if not self._stamp_cancel_requested_locked(job_id):
                    continue  # left 'running' under us: re-read
                self.conn.commit()
                updated = self.conn.execute(
                    "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                return dict(updated) if updated else job
            # Another process keeps flipping the row; report the freshest read.
            row = self.conn.execute(
                "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return dict(row) if row else None

    def finish_cancelled_extraction_job(
        self,
        job_id: int,
        *,
        error_message: str,
    ) -> JsonDict:
        """Finalize a running job as 'cancelled' after its subprocess was interrupted.

        A user cancel is not a failure: ``attempts`` is left untouched (no
        retry, no increment) and the parent item's ``extraction_status`` is
        restored so the item can be re-extracted. Idempotent for a job already
        'cancelled'; raises ValueError if the row is missing or already
        'done'/'failed' — a cancel must never overwrite a real outcome.

        Returns the post-update row.
        """
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"extraction job {job_id} not found")
            job = dict(row)
            status = str(job["status"])
            if status == "cancelled":
                return job  # idempotent
            if status in ("done", "failed"):
                raise ValueError(
                    f"job {job_id} cannot be cancelled from status={status!r}"
                )
            if not self._mark_cancelled_locked(
                job_id, int(job["research_item_id"]), error_message=error_message
            ):
                raise ValueError(
                    f"job {job_id} changed state before the cancel could be applied"
                )
            self.conn.commit()
            job["status"] = "cancelled"
            job["error_message"] = error_message
            return job

    def _stamp_cancel_requested_locked(self, job_id: int) -> bool:
        """Stamp ``cancel_requested_at`` for a RUNNING job. Caller holds the lock.

        Returns False when the job is no longer 'running' — the guard that
        stops a cross-process race from stamping a finished job. COALESCE keeps
        the first request time.
        """
        cursor = self.conn.execute(
            """UPDATE research_extraction_jobs
               SET cancel_requested_at = COALESCE(
                       cancel_requested_at, datetime('now'))
               WHERE id = ? AND status = 'running'""",
            (job_id,),
        )
        return cursor.rowcount > 0

    def _mark_cancelled_locked(
        self,
        job_id: int,
        item_id: int,
        *,
        error_message: str | None,
    ) -> bool:
        """Set a job row to 'cancelled' + restore its item. Caller holds the lock.

        The UPDATE is guarded by status so a cross-process transition into a
        real terminal state ('done'/'failed') is never overwritten. Returns
        False — touching neither the job nor the item — when the guard misses.
        """
        cursor = self.conn.execute(
            """UPDATE research_extraction_jobs
               SET status = 'cancelled',
                   completed_at = datetime('now'),
                   error_message = COALESCE(?, error_message, 'cancelled by user')
               WHERE id = ? AND status IN ('queued', 'running')""",
            (error_message, job_id),
        )
        if cursor.rowcount == 0:
            return False
        self._restore_item_after_cancel_locked(item_id)
        return True

    def _restore_item_after_cancel_locked(self, item_id: int) -> None:
        """Roll the parent item out of queued/running after a cancel.

        Mirrors the latest extraction's rolled-up status when one exists,
        otherwise 'pending'. Caller holds the write lock.
        """
        latest_ext = self.conn.execute(
            """SELECT status FROM research_extractions
               WHERE research_item_id = ?
               ORDER BY extracted_at DESC, id DESC LIMIT 1""",
            (item_id,),
        ).fetchone()
        if latest_ext is None:
            new_item_status = "pending"
        else:
            ext_status = str(latest_ext["status"])
            new_item_status = {
                "parsed": "extracted",
                "promoted": "extracted",
                "rejected": "extracted",
                "failed": "failed",
                "skipped": "skipped",
            }.get(ext_status, "pending")
        self.conn.execute(
            "UPDATE research_items SET extraction_status = ? WHERE id = ?",
            (new_item_status, item_id),
        )

    def list_extraction_queue(
        self,
        *,
        statuses: Iterable[str] = ("queued", "running"),
        limit: int = 200,
    ) -> list[JsonDict]:
        """List extraction-job rows joined with item title + url for display.

        Default returns active (queued + running) jobs ordered by queued_at.
        """
        status_list = list(statuses)
        if not status_list:
            return []
        placeholders = ",".join("?" for _ in status_list)
        with self._write_lock:
            rows = self.conn.execute(
                f"""SELECT j.id, j.research_item_id, j.status, j.queued_at,
                       j.started_at, j.completed_at, j.attempts, j.max_attempts,
                       j.last_error, j.error_message, j.heartbeat_at,
                       i.title AS item_title, i.url AS item_url,
                       i.source AS item_source
                FROM research_extraction_jobs j
                JOIN research_items i ON j.research_item_id = i.id
                WHERE j.status IN ({placeholders})
                ORDER BY j.queued_at ASC, j.id ASC
                LIMIT ?""",
                (*status_list, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_active_extraction_jobs(self) -> int:
        """Number of jobs currently in status IN ('queued', 'running')."""
        with self._write_lock:
            row = self.conn.execute(
                """SELECT COUNT(*) AS c FROM research_extraction_jobs
               WHERE status IN ('queued', 'running')"""
            ).fetchone()
        return int(row["c"]) if row else 0

    def sweep_stuck_extraction_jobs(
        self,
        threshold_seconds: int,
        *,
        exclude_job_id: int | None = None,
        exclude_job_ids: Iterable[int] | None = None,
    ) -> list[JsonDict]:
        """Find running jobs whose heartbeat has gone stale and reset them.

        Each stuck job is treated as a failure (`attempts++`, last_error set);
        retried if attempts<max_attempts, otherwise final-failed. Returns the
        list of affected job rows (post-update).

        `exclude_job_id` and `exclude_job_ids` let a worker skip its own
        in-flight jobs so it never sweeps itself just because the heartbeat
        thread hasn't fired yet. Both can be combined.
        """
        threshold = int(threshold_seconds)
        excluded: set[int] = set()
        if exclude_job_id is not None:
            excluded.add(exclude_job_id)
        if exclude_job_ids is not None:
            excluded.update(int(j) for j in exclude_job_ids)
        sql = (
            "SELECT id FROM research_extraction_jobs "
            "WHERE status = 'running' "
            "AND (heartbeat_at IS NULL "
            "     OR (julianday('now') - julianday(heartbeat_at)) * 86400 > ?)"
        )
        params: list[object] = [threshold]
        if excluded:
            placeholders = ",".join("?" for _ in excluded)
            sql += f" AND id NOT IN ({placeholders})"
            params.extend(excluded)
        with self._write_lock:
            rows = self.conn.execute(sql, tuple(params)).fetchall()
        out: list[JsonDict] = []
        for r in rows:
            out.append(
                self.fail_extraction_job(
                    int(r["id"]),
                    f"stuck: no heartbeat for >{threshold}s",
                )
            )
        return out

    def reset_orphan_running_items(self) -> int:
        """Reset items left at extraction_status='running' from the old
        BackgroundTasks path (no row in research_extraction_jobs).

        Returns the number of items reset to 'pending'. Idempotent — items
        that already have a queue row are NOT touched.
        """
        with self._write_lock:
            cursor = self.conn.execute(
                """UPDATE research_items
                   SET extraction_status = 'pending'
                   WHERE extraction_status = 'running'
                     AND id NOT IN (
                         SELECT research_item_id FROM research_extraction_jobs
                     )"""
            )
            self.conn.commit()
            return cursor.rowcount

    def complete_extraction_job(
        self,
        job_id: int,
        *,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        """Mark a job done/failed/cancelled. `status` must be one of those three."""
        if status not in ("done", "failed", "cancelled"):
            raise ValueError(f"invalid terminal job status: {status}")
        with self._write_lock:
            cursor = self.conn.execute(
                """UPDATE research_extraction_jobs
                   SET status = ?, completed_at = datetime('now'), error_message = ?
                   WHERE id = ?""",
                (status, error_message, job_id),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def get_extraction_job(self, job_id: int) -> JsonDict | None:
        # The worker watcher (1 s/job), the API cancel poll (0.25 s) and the
        # drain threads share this connection. An unlocked SELECT here can
        # interleave with a writer and corrupt the connection
        # (sqlite3.InterfaceError), so the read must take the write lock.
        with self._write_lock:
            row = self.conn.execute(
                "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def extraction_job_cancel_requested(self, job_id: int) -> bool:
        """Cheap locked poll: has a cancel been requested for this job?

        The per-job watcher calls this ~1 s while a drain thread writes; the
        read must be serialized with those writers (shared connection)."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT cancel_requested_at FROM research_extraction_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return bool(row and row["cancel_requested_at"])

    def list_extraction_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[JsonDict]:
        with self._write_lock:
            if status is None:
                rows = self.conn.execute(
                    "SELECT * FROM research_extraction_jobs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM research_extraction_jobs WHERE status = ? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    # --- Locked queries for API routers / discovery (bd vibe-quant-w55cl) ---
    # These used to be inline ``state.conn.execute(...)`` calls outside
    # ``_write_lock``; rows are materialised inside the lock.

    def get_database_file(self) -> str:
        """Filesystem path of the main database (``""`` for in-memory)."""
        with self._write_lock:
            row = self.conn.execute("PRAGMA database_list").fetchone()
            return str(row["file"]) if row is not None else ""

    def list_table_names(self) -> list[str]:
        """Names of all tables in ``sqlite_master`` (storage order)."""
        with self._write_lock:
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor]

    def count_table_rows(self, known_tables: frozenset[str]) -> dict[str, int]:
        """Row count per existing table whose name is in ``known_tables``.

        ``known_tables`` is a hard allow-list: names are interpolated into SQL.
        """
        with self._write_lock:
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor if row[0] in known_tables]
            counts: dict[str, int] = {}
            for table in tables:
                count_cursor = self.conn.execute(f"SELECT COUNT(*) FROM [{table}]")  # noqa: S608
                counts[table] = count_cursor.fetchone()[0]
            return counts

    def list_recent_discovery_result_notes(self, limit: int = 10) -> list[str]:
        """``backtest_results.notes`` of the newest completed discovery runs.

        Skips NULL/empty notes; newest run first.
        """
        with self._write_lock:
            rows = self.conn.execute(
                """SELECT br.notes FROM backtest_runs r
                   JOIN backtest_results br ON br.run_id = r.id
                   WHERE r.run_mode='discovery' AND r.status='completed'
                     AND br.notes IS NOT NULL AND br.notes != ''
                   ORDER BY r.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [row[0] for row in rows]

    def list_active_strategy_dsl_rows(self) -> list[tuple[int, str, object]]:
        """``(id, name, dsl_config)`` of active strategies (raw column), by id."""
        with self._write_lock:
            rows = self.conn.execute(
                "SELECT id, name, dsl_config FROM strategies "
                "WHERE is_active IS NULL OR is_active = 1 ORDER BY id"
            ).fetchall()
            return [(row[0], row[1], row[2]) for row in rows]

    def strategy_name_exists(self, name: str) -> bool:
        """True if any strategy row (active or not) already uses ``name``."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT 1 FROM strategies WHERE name = ?", (name,)
            ).fetchone()
            return row is not None

    def get_strategy_id_by_name(self, name: str) -> int | None:
        """Id of the strategy named ``name``, else None."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT id FROM strategies WHERE name = ?", (name,)
            ).fetchone()
            return None if row is None else int(row[0])

    def get_latest_extraction_job_for_item(self, item_id: int) -> JsonDict | None:
        """Most recent ``research_extraction_jobs`` row for a research item."""
        with self._write_lock:
            row = self.conn.execute(
                """SELECT * FROM research_extraction_jobs
                   WHERE research_item_id = ?
                   ORDER BY id DESC LIMIT 1""",
                (item_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_result_notes_raw(self, run_id: int) -> str | None:
        """Raw ``backtest_results.notes`` of a run; None if no result row."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT notes FROM backtest_results WHERE run_id = ?", (run_id,)
            ).fetchone()
            return None if row is None else row[0]

    def get_backtest_run_status(self, run_id: int) -> str | None:
        """``backtest_runs.status`` of a run; None if no such run."""
        with self._write_lock:
            row = self.conn.execute(
                "SELECT status FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return None if row is None else row[0]

    def list_campaign_oos_rows(
        self, run_mode: str, campaign_id: str
    ) -> list[tuple[int, str | None, float | None, float | None, float | None, int | None]]:
        """``(run_id, parameters, sharpe, total_return, max_drawdown, total_trades)``
        of a campaign's runs in ``run_mode`` (left-joined result), by run id."""
        with self._write_lock:
            rows = self.conn.execute(
                """
                SELECT br.id, br.parameters, r.sharpe_ratio, r.total_return,
                       r.max_drawdown, r.total_trades
                FROM backtest_runs br
                LEFT JOIN backtest_results r ON r.run_id = br.id
                WHERE br.run_mode = ?
                  AND json_extract(br.parameters, '$.campaign_id') = ?
                ORDER BY br.id
                """,
                (run_mode, campaign_id),
            ).fetchall()
            return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def list_prior_discovery_notes(
        self, before_run_id: int, timeframe: str
    ) -> list[tuple[int, str | None, str, str]]:
        """``(run_id, notes, start_date, end_date)`` of discovery runs with a result
        row, ``run_id < before_run_id`` and the same ``timeframe``."""
        with self._write_lock:
            rows = self.conn.execute(
                "SELECT b.id, r.notes, b.start_date, b.end_date "
                "FROM backtest_results r JOIN backtest_runs b ON b.id = r.run_id "
                "WHERE b.run_mode = ? AND b.id < ? AND b.timeframe = ?",
                ("discovery", before_run_id, timeframe),
            ).fetchall()
            return [(r[0], r[1], r[2], r[3]) for r in rows]
