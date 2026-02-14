"""Query functions for event logs using DuckDB.

Provides efficient querying of JSONL event logs with DuckDB. Supports filtering
by event type and returns results as dicts or pandas DataFrames.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from vibe_quant.logging.events import EventType

_SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_BASE_PATH = _PROJECT_ROOT / "logs" / "events"


def _validate_run_id(run_id: str) -> None:
    """Validate run_id is safe for use in file paths and queries."""
    if not _SAFE_RUN_ID.match(run_id):
        msg = f"Invalid run_id: must be alphanumeric/hyphens/underscores, got '{run_id}'"
        raise ValueError(msg)


def _get_log_path(run_id: str, base_path: Path | str | None = None) -> Path:
    """Get path to log file for a run.

    Args:
        run_id: Backtest run identifier.
        base_path: Directory containing event logs.

    Returns:
        Path to the JSONL file.
    """
    resolved = Path(base_path) if base_path is not None else _DEFAULT_BASE_PATH
    return resolved / f"{run_id}.jsonl"


def query_events(
    run_id: str,
    event_type: EventType | None = None,
    base_path: Path | str | None = None,
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

    _validate_run_id(run_id)
    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    # Build parameterized query
    conn = duckdb.connect()
    try:
        if event_type is not None:
            query = "SELECT * FROM read_json_auto(?) WHERE event = ? ORDER BY ts"
            result = conn.execute(query, [str(log_path), event_type.value])
        else:
            query = "SELECT * FROM read_json_auto(?) ORDER BY ts"
            result = conn.execute(query, [str(log_path)])

        df = result.fetchdf()
    finally:
        conn.close()

    records: list[dict[str, object]] = df.to_dict(orient="records")  # type: ignore[assignment]
    return records


def query_events_df(
    run_id: str,
    event_type: EventType | None = None,
    base_path: Path | str | None = None,
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

    _validate_run_id(run_id)
    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    conn = duckdb.connect()
    try:
        if event_type is not None:
            query = "SELECT * FROM read_json_auto(?) WHERE event = ? ORDER BY ts"
            result = conn.execute(query, [str(log_path), event_type.value])
        else:
            query = "SELECT * FROM read_json_auto(?) ORDER BY ts"
            result = conn.execute(query, [str(log_path)])
        return result.fetchdf()
    finally:
        conn.close()


def count_events_by_type(
    run_id: str,
    base_path: Path | str | None = None,
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

    _validate_run_id(run_id)
    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    conn = duckdb.connect()
    try:
        query = """
            SELECT event, COUNT(*) as count
            FROM read_json_auto(?)
            GROUP BY event
            ORDER BY event
        """
        result = conn.execute(query, [str(log_path)])
        df = result.fetchdf()
    finally:
        conn.close()

    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        counts[str(row["event"])] = int(row["count"])

    return counts


def get_run_summary(
    run_id: str,
    base_path: Path | str | None = None,
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

    _validate_run_id(run_id)
    log_path = _get_log_path(run_id, base_path)

    if not log_path.exists():
        msg = f"Event log not found: {log_path}"
        raise FileNotFoundError(msg)

    conn = duckdb.connect()
    try:
        query = """
            SELECT
                COUNT(*) as total_events,
                MIN(ts) as first_event,
                MAX(ts) as last_event,
                COUNT(DISTINCT event) as event_type_count
            FROM read_json_auto(?)
        """
        result = conn.execute(query, [str(log_path)])
        df = result.fetchdf()
    finally:
        conn.close()

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


def list_runs(base_path: Path | str | None = None) -> list[str]:
    """List all run IDs with event logs.

    Args:
        base_path: Directory containing event logs.

    Returns:
        List of run IDs.
    """
    base = Path(base_path) if base_path is not None else _DEFAULT_BASE_PATH
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
