"""StateManager class providing CRUD operations for vibe-quant state database."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import init_schema

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence
    from pathlib import Path

# Type alias for JSON-like dict structures from database
JsonDict = dict[str, Any]

# Column whitelists per table (must match schema.py definitions)
_BACKTEST_RESULTS_COLUMNS: frozenset[str] = frozenset({
    "run_id", "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "max_drawdown", "max_drawdown_duration_days",
    "volatility_annual", "total_trades", "winning_trades", "losing_trades",
    "win_rate", "profit_factor", "avg_win", "avg_loss", "largest_win",
    "largest_loss", "avg_trade_duration_hours", "max_consecutive_wins",
    "max_consecutive_losses", "total_fees", "total_funding", "total_slippage",
    "deflated_sharpe", "walk_forward_efficiency", "purged_kfold_mean_sharpe",
    "execution_time_seconds", "starting_balance", "notes",
})

_TRADES_COLUMNS: frozenset[str] = frozenset({
    "run_id", "symbol", "direction", "leverage", "entry_time", "exit_time",
    "entry_price", "exit_price", "quantity", "entry_fee", "exit_fee",
    "funding_fees", "slippage_cost", "gross_pnl", "net_pnl", "roi_percent",
    "exit_reason",
})

_SWEEP_RESULTS_COLUMNS: frozenset[str] = frozenset({
    "run_id", "parameters", "sharpe_ratio", "sortino_ratio", "max_drawdown",
    "total_return", "profit_factor", "win_rate", "total_trades", "total_fees",
    "total_funding", "execution_time_seconds", "is_pareto_optimal",
    "passed_deflated_sharpe", "passed_walk_forward", "passed_purged_kfold",
})


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

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            init_schema(self._conn)
        return self._conn

    def close(self) -> None:
        """Close database connection."""
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
        cursor = self.conn.execute(
            "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
        )
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
        cursor = self.conn.execute(
            "SELECT * FROM strategies WHERE name = ?", (name,)
        )
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
        dsl_config: JsonDict | None = None,
        description: str | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Update strategy fields.

        Args:
            strategy_id: Strategy ID.
            dsl_config: New DSL config (optional).
            description: New description (optional).
            is_active: New active status (optional).
        """
        updates = ["updated_at = datetime('now')"]
        params: list[Any] = []

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
        self.conn.execute(
            f"UPDATE strategies SET {', '.join(updates)} WHERE id = ?", params
        )
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
        cursor = self.conn.execute(
            "INSERT INTO sizing_configs (name, method, config) VALUES (?, ?, ?)",
            (name, method, json.dumps(config)),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def get_sizing_config(self, config_id: int) -> JsonDict | None:
        """Get sizing config by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM sizing_configs WHERE id = ?", (config_id,)
        )
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

    def update_sizing_config(self, config_id: int, name: str, method: str, config: JsonDict) -> None:
        """Update an existing sizing configuration."""
        self.conn.execute(
            "UPDATE sizing_configs SET name = ?, method = ?, config = ? WHERE id = ?",
            (name, method, json.dumps(config), config_id),
        )
        self.conn.commit()

    def delete_sizing_config(self, config_id: int) -> None:
        """Delete a sizing configuration by ID."""
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
        cursor = self.conn.execute(
            """INSERT INTO risk_configs (name, strategy_level, portfolio_level)
               VALUES (?, ?, ?)""",
            (name, json.dumps(strategy_level), json.dumps(portfolio_level)),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def get_risk_config(self, config_id: int) -> JsonDict | None:
        """Get risk config by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM risk_configs WHERE id = ?", (config_id,)
        )
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
        self.conn.execute(
            "UPDATE risk_configs SET name = ?, strategy_level = ?, portfolio_level = ? WHERE id = ?",
            (name, json.dumps(strategy_level), json.dumps(portfolio_level), config_id),
        )
        self.conn.commit()

    def delete_risk_config(self, config_id: int) -> None:
        """Delete a risk configuration by ID."""
        self.conn.execute("DELETE FROM risk_configs WHERE id = ?", (config_id,))
        self.conn.commit()

    # --- Backtest Run CRUD ---

    def create_backtest_run(
        self,
        strategy_id: int,
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
            strategy_id: Strategy ID.
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
        cursor = self.conn.execute(
            "SELECT * FROM backtest_runs WHERE id = ?", (run_id,)
        )
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
        elif status in ("completed", "failed"):
            updates.append("completed_at = datetime('now')")
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

        params.append(run_id)
        self.conn.execute(
            f"UPDATE backtest_runs SET {', '.join(updates)} WHERE id = ?", params
        )
        self.conn.commit()

    def update_heartbeat(self, run_id: int) -> None:
        """Update heartbeat timestamp for a running backtest."""
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

        cursor = self.conn.execute(
            f"INSERT INTO backtest_results ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def get_backtest_result(self, run_id: int) -> JsonDict | None:
        """Get backtest result for a run."""
        cursor = self.conn.execute(
            "SELECT * FROM backtest_results WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_result_notes(self, run_id: int, notes: str) -> None:
        """Update notes/annotations for a backtest result.

        Args:
            run_id: Backtest run ID.
            notes: Notes text to store.
        """
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

        cursor = self.conn.execute(
            f"INSERT INTO sweep_results ({', '.join(columns)}) VALUES ({placeholders})",
            list(result_data.values()),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def save_sweep_results_batch(self, run_id: int, results: Sequence[JsonDict]) -> None:
        """Save multiple sweep results in a batch."""
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

        self.conn.executemany(
            f"INSERT INTO sweep_results ({', '.join(columns)}) VALUES ({placeholders})",
            [list(r.values()) for r in processed],
        )
        self.conn.commit()

    def get_sweep_results(
        self, run_id: int, pareto_only: bool = False
    ) -> list[JsonDict]:
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
        self.conn.execute(
            f"UPDATE sweep_results SET is_pareto_optimal = 1 WHERE id IN ({placeholders})",
            list(result_ids),
        )
        self.conn.commit()

    # --- Background Jobs CRUD ---

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
        cursor = self.conn.execute(
            """INSERT INTO background_jobs (run_id, pid, job_type, log_file)
               VALUES (?, ?, ?, ?)""",
            (run_id, pid, job_type, log_file),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def get_job(self, run_id: int) -> JsonDict | None:
        """Get job by run ID."""
        cursor = self.conn.execute(
            "SELECT * FROM background_jobs WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_running_jobs(self) -> list[JsonDict]:
        """Get all running jobs."""
        cursor = self.conn.execute(
            "SELECT * FROM background_jobs WHERE status = 'running'"
        )
        return [dict(row) for row in cursor]

    def update_job_status(
        self, run_id: int, status: str, error: str | None = None
    ) -> None:
        """Update job status.

        Args:
            run_id: Run ID.
            status: New status (running, completed, failed, killed).
            error: Optional error message.
        """
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

    def update_job_heartbeat(self, run_id: int) -> None:
        """Update job heartbeat timestamp."""
        self.conn.execute(
            """UPDATE background_jobs SET heartbeat_at = datetime('now')
               WHERE run_id = ?""",
            (run_id,),
        )
        self.conn.commit()
