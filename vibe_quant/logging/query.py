"""Query functions for event logs using DuckDB.

Provides efficient querying of JSONL event logs with DuckDB. Supports filtering
by event type and returns results as dicts or pandas DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from vibe_quant.logging.events import EventType


def _get_log_path(run_id: str, base_path: Path | str = "logs/events") -> Path:
    """Get path to log file for a run.

    Args:
        run_id: Backtest run identifier.
        base_path: Directory containing event logs.

    Returns:
        Path to the JSONL file.
    """
    return Path(base_path) / f"{run_id}.jsonl"


def query_events(
    run_id: str,
    event_type: EventType | None = None,
    base_path: Path | str = "logs/events",
) -> list[dict[str, object]]:
    """Query events from a run's log file.

    Uses DuckDB to efficiently read and filter JSONL files.

    Args:
        run_id: Backtest run identifier.
        event_type: Optional filter for specific event type.
        base_path: Directory containing event logs.

    Returns:
        List of event dictionaries.

    Raises:
        FileNotFoundError: If log file doesn't exist.
    """
    import duckdb

    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    # Build query
    query = f"SELECT * FROM read_json_auto('{log_path}')"

    if event_type is not None:
        query += f" WHERE event_type = '{event_type.value}'"

    query += " ORDER BY timestamp"

    # Execute and convert to list of dicts
    result = duckdb.query(query)
    df = result.fetchdf()

    # Convert DataFrame rows to dicts
    records: list[dict[str, object]] = df.to_dict(orient="records")  # type: ignore[assignment]
    return records


def query_events_df(
    run_id: str,
    event_type: EventType | None = None,
    base_path: Path | str = "logs/events",
) -> pd.DataFrame:
    """Query events as a pandas DataFrame.

    Uses DuckDB for efficient JSONL reading, returns DataFrame for analysis.

    Args:
        run_id: Backtest run identifier.
        event_type: Optional filter for specific event type.
        base_path: Directory containing event logs.

    Returns:
        DataFrame with event data.

    Raises:
        FileNotFoundError: If log file doesn't exist.
    """
    import duckdb

    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    # Build query
    query = f"SELECT * FROM read_json_auto('{log_path}')"

    if event_type is not None:
        query += f" WHERE event_type = '{event_type.value}'"

    query += " ORDER BY timestamp"

    # Execute and return DataFrame
    result = duckdb.query(query)
    return result.fetchdf()


def count_events_by_type(
    run_id: str,
    base_path: Path | str = "logs/events",
) -> dict[str, int]:
    """Count events by type for a run.

    Args:
        run_id: Backtest run identifier.
        base_path: Directory containing event logs.

    Returns:
        Dict mapping event type to count.

    Raises:
        FileNotFoundError: If log file doesn't exist.
    """
    import duckdb

    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    query = f"""
        SELECT event_type, COUNT(*) as count
        FROM read_json_auto('{log_path}')
        GROUP BY event_type
        ORDER BY event_type
    """

    result = duckdb.query(query)
    df = result.fetchdf()

    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        counts[str(row["event_type"])] = int(row["count"])

    return counts


def get_run_summary(
    run_id: str,
    base_path: Path | str = "logs/events",
) -> dict[str, object]:
    """Get summary statistics for a run.

    Args:
        run_id: Backtest run identifier.
        base_path: Directory containing event logs.

    Returns:
        Dict with summary stats (event counts, time range, etc.).

    Raises:
        FileNotFoundError: If log file doesn't exist.
    """
    import duckdb

    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    query = f"""
        SELECT
            COUNT(*) as total_events,
            MIN(timestamp) as first_event,
            MAX(timestamp) as last_event,
            COUNT(DISTINCT event_type) as event_type_count
        FROM read_json_auto('{log_path}')
    """

    result = duckdb.query(query)
    df = result.fetchdf()

    if df.empty:
        return {
            "run_id": run_id,
            "total_events": 0,
            "first_event": None,
            "last_event": None,
            "event_type_count": 0,
            "events_by_type": {},
        }

    row = df.iloc[0]
    return {
        "run_id": run_id,
        "total_events": int(row["total_events"]),
        "first_event": str(row["first_event"]) if row["first_event"] else None,
        "last_event": str(row["last_event"]) if row["last_event"] else None,
        "event_type_count": int(row["event_type_count"]),
        "events_by_type": count_events_by_type(run_id, base_path),
    }


def list_runs(base_path: Path | str = "logs/events") -> list[str]:
    """List all run IDs with event logs.

    Args:
        base_path: Directory containing event logs.

    Returns:
        List of run IDs.
    """
    base = Path(base_path)
    if not base.exists():
        return []

    return [p.stem for p in base.glob("*.jsonl")]


__all__ = [
    "query_events",
    "query_events_df",
    "count_events_by_type",
    "get_run_summary",
    "list_runs",
]
