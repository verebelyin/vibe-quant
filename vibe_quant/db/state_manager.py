"""StateManager class providing CRUD operations for vibe-quant state database."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import TYPE_CHECKING, Any

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import init_schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


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
        "is_pareto_optimal",
        "passed_deflated_sharpe",
        "passed_walk_forward",
        "passed_purged_kfold",
    }
)


def _validate_columns(columns: list[str], allowed: frozenset[str], table: str) -> None:
    """Validate column names against whitelist. Raises ValueError on unknown columns."""
    bad = set(columns) - allowed
    if bad:
        raise ValueError(f"Unknown columns for {table}: {sorted(bad)}")


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
                self._conn = get_connection(self._db_path)
                init_schema(self._conn)
            return self._conn

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
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
        cursor = self.conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
        row = cursor.fetchone()
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
        cursor = self.conn.execute("SELECT * FROM strategies WHERE name = ?", (name,))
        row = cursor.fetchone()
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

        cursor = self.conn.execute(query)
        results = []
        for row in cursor:
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
        cursor = self.conn.execute("SELECT * FROM sizing_configs WHERE id = ?", (config_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        result["config"] = json.loads(result["config"])
        return result

    def list_sizing_configs(self) -> list[JsonDict]:
        """List all sizing configs."""
        cursor = self.conn.execute("SELECT * FROM sizing_configs ORDER BY name")
        results = []
        for row in cursor:
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
        cursor = self.conn.execute("SELECT * FROM risk_configs WHERE id = ?", (config_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        result["strategy_level"] = json.loads(result["strategy_level"])
        result["portfolio_level"] = json.loads(result["portfolio_level"])
        return result

    def list_risk_configs(self) -> list[JsonDict]:
        """List all risk configs."""
        cursor = self.conn.execute("SELECT * FROM risk_configs ORDER BY name")
        results = []
        for row in cursor:
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

    def get_system_state(self) -> JsonDict:
        """Read the singleton system_state row. Always returns a dict.

        Returns:
            Dict with keys: kill_switch (bool), reason (str|None),
            killed_at (str|None), killed_by (str|None), updated_at (str).
        """
        row = self.conn.execute(
            "SELECT kill_switch, reason, killed_at, killed_by, updated_at "
            "FROM system_state WHERE id = 1"
        ).fetchone()
        if row is None:
            # Defensive: row should have been inserted by init_schema.
            return {
                "kill_switch": False,
                "reason": None,
                "killed_at": None,
                "killed_by": None,
                "updated_at": None,
            }
        return {
            "kill_switch": bool(row[0]),
            "reason": row[1],
            "killed_at": row[2],
            "killed_by": row[3],
            "updated_at": row[4],
        }

    def set_kill_switch(self, reason: str, killed_by: str | None = None) -> None:
        """Engage the kill switch. Idempotent — re-setting updates reason."""
        with self._write_lock:
            self.conn.execute(
                "UPDATE system_state SET kill_switch = 1, reason = ?, "
                "killed_at = datetime('now'), killed_by = ?, updated_at = datetime('now') "
                "WHERE id = 1",
                (reason, killed_by),
            )
            self.conn.commit()

    def clear_kill_switch(self, cleared_by: str | None = None) -> None:
        """Release the kill switch. Logs who cleared it in ``killed_by``."""
        with self._write_lock:
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
        cursor = self.conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
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

        cursor = self.conn.execute(query, params)
        results = []
        for row in cursor:
            result = dict(row)
            result["symbols"] = json.loads(result["symbols"])
            result["parameters"] = json.loads(result["parameters"])
            results.append(result)
        return results

    # --- Backtest Results CRUD ---

    def save_backtest_result(self, run_id: int, metrics: JsonDict) -> int:
        """Save backtest results.

        Args:
            run_id: Backtest run ID.
            metrics: Dict of metric names to values.

        Returns:
            ID of created result.
        """
        columns = ["run_id"] + list(metrics.keys())
        _validate_columns(columns, _BACKTEST_RESULTS_COLUMNS, "backtest_results")
        placeholders = ", ".join(["?"] * len(columns))
        values = [run_id] + list(metrics.values())

        with self._write_lock:
            cursor = self.conn.execute(
                f"INSERT INTO backtest_results ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            self.conn.commit()
            return cursor.lastrowid or 0

    def get_backtest_result(self, run_id: int) -> JsonDict | None:
        """Get backtest result for a run."""
        cursor = self.conn.execute("SELECT * FROM backtest_results WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

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
            WHERE 1=1
        """
        params: list[Any] = []

        if strategy_id is not None:
            query += " AND r.strategy_id = ?"
            params.append(strategy_id)

        query += " ORDER BY r.created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        results = []
        for row in cursor:
            result = dict(row)
            # Parse JSON fields from the joined run data
            if "symbols" in result and result["symbols"] is not None:
                result["symbols"] = json.loads(result["symbols"])
            results.append(result)
        return results

    def update_result_notes(self, run_id: int, notes: str) -> None:
        """Update notes/annotations for a backtest result.

        Args:
            run_id: Backtest run ID.
            notes: Notes text to store.
        """
        with self._write_lock:
            self.conn.execute(
                "UPDATE backtest_results SET notes = ? WHERE run_id = ?",
                (notes, run_id),
            )
            self.conn.commit()

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
        """Save multiple trades in a batch.

        Args:
            run_id: Backtest run ID.
            trades: List of trade data dicts.
        """
        if not trades:
            return

        # Get columns from first trade
        columns = ["run_id"] + list(trades[0].keys())
        _validate_columns(columns, _TRADES_COLUMNS, "trades")
        placeholders = ", ".join(["?"] * len(columns))

        with self._write_lock:
            self.conn.executemany(
                f"INSERT INTO trades ({', '.join(columns)}) VALUES ({placeholders})",
                [[run_id] + list(t.values()) for t in trades],
            )
            self.conn.commit()

    def get_trades(self, run_id: int) -> list[JsonDict]:
        """Get all trades for a backtest run."""
        cursor = self.conn.execute(
            "SELECT * FROM trades WHERE run_id = ? ORDER BY entry_time", (run_id,)
        )
        return [dict(row) for row in cursor]

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

        cursor = self.conn.execute(query, (run_id,))
        results = []
        for row in cursor:
            result = dict(row)
            result["parameters"] = json.loads(result["parameters"])
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
        cursor = self.conn.execute("SELECT * FROM background_jobs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_running_jobs(self) -> list[JsonDict]:
        """Get all running jobs."""
        cursor = self.conn.execute("SELECT * FROM background_jobs WHERE status = 'running'")
        return [dict(row) for row in cursor]

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
            LEFT JOIN backtest_results br ON r.id = br.run_id
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

        cursor = self.conn.execute(query, params)
        results = []
        for row in cursor:
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
    ) -> int:
        with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO research_extractions
                   (research_item_id, status, llm_model, confidence, rationale,
                    raw_response, dsl_yaml, parsed_dsl_json, parse_error,
                    proposed_indicators_json, prompt, evidence_level, completeness)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                ),
            )
            self.conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def list_extractions_for_item(self, item_id: int) -> list[JsonDict]:
        rows = self.conn.execute(
            """SELECT * FROM research_extractions
               WHERE research_item_id = ?
               ORDER BY extracted_at DESC, id DESC""",
            (item_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_extraction(self, extraction_id: int) -> JsonDict | None:
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
        row = self.conn.execute(
            """SELECT * FROM research_indicator_scaffolds
               WHERE extraction_id = ? AND idx = ?""",
            (extraction_id, idx),
        ).fetchone()
        return dict(row) if row else None

    def list_indicator_scaffolds(self, extraction_id: int) -> list[JsonDict]:
        """All scaffold rows for an extraction, indexed by idx ascending."""
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
        row = self.conn.execute(
            "SELECT * FROM research_scrape_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def latest_scrape_run(self, source: str) -> JsonDict | None:
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
            self.conn.execute(
                """UPDATE research_extraction_jobs
                   SET status = 'cancelled',
                       completed_at = datetime('now'),
                       error_message = COALESCE(error_message, 'cancelled by user')
                   WHERE id = ?""",
                (job_id,),
            )
            item_id = int(job["research_item_id"])
            # Restore item to a sensible non-running state. If there are
            # prior extractions, derive item status from the latest one;
            # otherwise default to 'pending'.
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
            self.conn.commit()
            job["status"] = "cancelled"
            return job

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
        row = self.conn.execute(
            "SELECT * FROM research_extraction_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_extraction_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[JsonDict]:
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
