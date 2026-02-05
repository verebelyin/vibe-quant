"""Screening-to-validation consistency checker.

Compares screening results against validation backtests to identify
candidates that significantly degraded (execution-sensitive strategies).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from vibe_quant.db.connection import DEFAULT_DB_PATH


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    """Result of consistency check between screening and validation.

    Attributes:
        strategy_name: Strategy name.
        screening_run_id: Screening sweep run ID.
        validation_run_id: Validation backtest run ID.
        screening_sharpe: Sharpe ratio from screening.
        validation_sharpe: Sharpe ratio from validation.
        sharpe_degradation: (screening - validation) / screening as percentage.
        screening_return: Total return from screening.
        validation_return: Total return from validation.
        return_degradation: (screening - validation) / screening as percentage.
        is_execution_sensitive: True if degradation > 50%.
        parameters: Strategy parameters JSON.
        checked_at: Timestamp of check.
    """

    strategy_name: str
    screening_run_id: int
    validation_run_id: int
    screening_sharpe: float
    validation_sharpe: float
    sharpe_degradation: float
    screening_return: float
    validation_return: float
    return_degradation: float
    is_execution_sensitive: bool
    parameters: str
    checked_at: str


class ConsistencyChecker:
    """Compares screening vs validation results for consistency.

    Identifies candidates that improved/degraded and flags
    execution-sensitive strategies (>50% degradation).
    """

    DEGRADATION_THRESHOLD = 0.50  # 50% degradation threshold

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize consistency checker.

        Args:
            db_path: Path to SQLite database. Defaults to DEFAULT_DB_PATH.
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._ensure_tables()
        return self._conn

    def _ensure_tables(self) -> None:
        """Create consistency_checks table if not exists."""
        self.conn.execute("""
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
            )
        """)
        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def check_consistency(
        self,
        screening_run_id: int,
        validation_run_id: int,
    ) -> ConsistencyResult:
        """Compare screening vs validation results.

        Args:
            screening_run_id: ID from sweep_results table.
            validation_run_id: ID from backtest_results table.

        Returns:
            ConsistencyResult with comparison metrics.

        Raises:
            ValueError: If run IDs not found.
        """
        # Get screening result
        row = self.conn.execute(
            """
            SELECT s.name AS strategy_name, sr.sharpe_ratio, sr.total_return, sr.parameters
            FROM sweep_results sr
            JOIN backtest_runs br ON sr.run_id = br.id
            JOIN strategies s ON br.strategy_id = s.id
            WHERE sr.run_id = ?
            """,
            (screening_run_id,),
        ).fetchone()

        if row is None:
            msg = f"Screening run {screening_run_id} not found"
            raise ValueError(msg)

        screening = {
            "name": row["strategy_name"],
            "sharpe": row["sharpe_ratio"],
            "return": row["total_return"],
            "params": row["parameters"],
        }

        # Get validation result
        row = self.conn.execute(
            """
            SELECT s.name AS strategy_name, res.sharpe_ratio, res.total_return
            FROM backtest_results res
            JOIN backtest_runs br ON res.run_id = br.id
            JOIN strategies s ON br.strategy_id = s.id
            WHERE res.run_id = ?
            """,
            (validation_run_id,),
        ).fetchone()

        if row is None:
            msg = f"Validation run {validation_run_id} not found"
            raise ValueError(msg)

        validation = {
            "name": row["strategy_name"],
            "sharpe": row["sharpe_ratio"],
            "return": row["total_return"],
        }

        # Calculate degradations (positive = degraded, negative = improved)
        sharpe_deg = self._calc_degradation(screening["sharpe"], validation["sharpe"])
        return_deg = self._calc_degradation(screening["return"], validation["return"])

        # Flag as execution-sensitive if degradation exceeds threshold
        is_sensitive = (
            sharpe_deg > self.DEGRADATION_THRESHOLD
            or return_deg > self.DEGRADATION_THRESHOLD
        )

        result = ConsistencyResult(
            strategy_name=screening["name"],
            screening_run_id=screening_run_id,
            validation_run_id=validation_run_id,
            screening_sharpe=screening["sharpe"],
            validation_sharpe=validation["sharpe"],
            sharpe_degradation=sharpe_deg,
            screening_return=screening["return"],
            validation_return=validation["return"],
            return_degradation=return_deg,
            is_execution_sensitive=is_sensitive,
            parameters=screening["params"],
            checked_at=datetime.now().isoformat(),
        )

        # Store result
        self._save_result(result)

        return result

    def _calc_degradation(self, screening: float, validation: float) -> float:
        """Calculate degradation percentage.

        Args:
            screening: Screening metric value.
            validation: Validation metric value.

        Returns:
            Degradation as decimal (0.5 = 50% degradation).
            Positive = degraded, negative = improved.
        """
        if screening == 0:
            return 0.0 if validation == 0 else -1.0 if validation > 0 else 1.0
        return (screening - validation) / abs(screening)

    def _save_result(self, result: ConsistencyResult) -> None:
        """Save consistency result to database."""
        self.conn.execute(
            """
            INSERT INTO consistency_checks (
                strategy_name, screening_run_id, validation_run_id,
                screening_sharpe, validation_sharpe, sharpe_degradation,
                screening_return, validation_return, return_degradation,
                is_execution_sensitive, parameters, checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.strategy_name,
                result.screening_run_id,
                result.validation_run_id,
                result.screening_sharpe,
                result.validation_sharpe,
                result.sharpe_degradation,
                result.screening_return,
                result.validation_return,
                result.return_degradation,
                1 if result.is_execution_sensitive else 0,
                result.parameters,
                result.checked_at,
            ),
        )
        self.conn.commit()

    def check_batch(
        self,
        pairs: list[tuple[int, int]],
    ) -> list[ConsistencyResult]:
        """Check multiple screening/validation pairs.

        Args:
            pairs: List of (screening_run_id, validation_run_id) tuples.

        Returns:
            List of ConsistencyResult objects.
        """
        return [
            self.check_consistency(screening_id, validation_id)
            for screening_id, validation_id in pairs
        ]

    def get_execution_sensitive(
        self,
        limit: int = 100,
    ) -> list[ConsistencyResult]:
        """Get strategies flagged as execution-sensitive.

        Args:
            limit: Maximum results to return.

        Returns:
            List of ConsistencyResult for sensitive strategies.
        """
        rows = self.conn.execute(
            """
            SELECT * FROM consistency_checks
            WHERE is_execution_sensitive = 1
            ORDER BY sharpe_degradation DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            ConsistencyResult(
                strategy_name=row["strategy_name"],
                screening_run_id=row["screening_run_id"],
                validation_run_id=row["validation_run_id"],
                screening_sharpe=row["screening_sharpe"],
                validation_sharpe=row["validation_sharpe"],
                sharpe_degradation=row["sharpe_degradation"],
                screening_return=row["screening_return"],
                validation_return=row["validation_return"],
                return_degradation=row["return_degradation"],
                is_execution_sensitive=bool(row["is_execution_sensitive"]),
                parameters=row["parameters"],
                checked_at=row["checked_at"],
            )
            for row in rows
        ]

    def get_improved(self, limit: int = 100) -> list[ConsistencyResult]:
        """Get strategies that improved from screening to validation.

        Args:
            limit: Maximum results to return.

        Returns:
            List of ConsistencyResult where validation outperformed screening.
        """
        rows = self.conn.execute(
            """
            SELECT * FROM consistency_checks
            WHERE sharpe_degradation < 0
            ORDER BY sharpe_degradation ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            ConsistencyResult(
                strategy_name=row["strategy_name"],
                screening_run_id=row["screening_run_id"],
                validation_run_id=row["validation_run_id"],
                screening_sharpe=row["screening_sharpe"],
                validation_sharpe=row["validation_sharpe"],
                sharpe_degradation=row["sharpe_degradation"],
                screening_return=row["screening_return"],
                validation_return=row["validation_return"],
                return_degradation=row["return_degradation"],
                is_execution_sensitive=bool(row["is_execution_sensitive"]),
                parameters=row["parameters"],
                checked_at=row["checked_at"],
            )
            for row in rows
        ]

    def generate_report(self, checks: list[ConsistencyResult]) -> str:
        """Generate a text report from consistency checks.

        Args:
            checks: List of consistency results.

        Returns:
            Formatted report string.
        """
        if not checks:
            return "No consistency checks to report."

        lines = [
            "=" * 70,
            "SCREENING-TO-VALIDATION CONSISTENCY REPORT",
            "=" * 70,
            "",
        ]

        sensitive = [c for c in checks if c.is_execution_sensitive]
        improved = [c for c in checks if c.sharpe_degradation < 0]
        degraded = [c for c in checks if c.sharpe_degradation > 0 and not c.is_execution_sensitive]

        lines.append(f"Total checked: {len(checks)}")
        lines.append(f"Execution-sensitive (>50% degradation): {len(sensitive)}")
        lines.append(f"Degraded (<50%): {len(degraded)}")
        lines.append(f"Improved: {len(improved)}")
        lines.append("")

        if sensitive:
            lines.append("-" * 70)
            lines.append("EXECUTION-SENSITIVE STRATEGIES (>50% degradation):")
            lines.append("-" * 70)
            for c in sensitive:
                lines.append(f"  {c.strategy_name}")
                lines.append(f"    Sharpe: {c.screening_sharpe:.2f} -> {c.validation_sharpe:.2f} ({c.sharpe_degradation:+.1%})")
                lines.append(f"    Return: {c.screening_return:.2f}% -> {c.validation_return:.2f}% ({c.return_degradation:+.1%})")
            lines.append("")

        if improved:
            lines.append("-" * 70)
            lines.append("IMPROVED STRATEGIES (validation > screening):")
            lines.append("-" * 70)
            for c in improved[:10]:  # Top 10
                lines.append(f"  {c.strategy_name}")
                lines.append(f"    Sharpe: {c.screening_sharpe:.2f} -> {c.validation_sharpe:.2f} ({c.sharpe_degradation:+.1%})")
            lines.append("")

        return "\n".join(lines)


def check_consistency(
    screening_run_id: int,
    validation_run_id: int,
    db_path: str | Path | None = None,
) -> ConsistencyResult:
    """Convenience function to check single pair.

    Args:
        screening_run_id: ID from sweep_results.
        validation_run_id: ID from backtest_results.
        db_path: Optional database path.

    Returns:
        ConsistencyResult object.
    """
    checker = ConsistencyChecker(db_path)
    try:
        return checker.check_consistency(screening_run_id, validation_run_id)
    finally:
        checker.close()
