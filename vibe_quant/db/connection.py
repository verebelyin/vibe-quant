"""SQLite connection factory with WAL mode enabled by default."""

import sqlite3
from pathlib import Path
from typing import Final

DEFAULT_DB_PATH: Final = Path("data/state/vibe_quant.db")
BUSY_TIMEOUT_MS: Final = 5000


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get SQLite connection with WAL mode and busy timeout configured.

    Args:
        db_path: Path to database file. Defaults to data/state/vibe_quant.db

    Returns:
        Configured SQLite connection with WAL mode enabled.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Enable WAL mode and set busy timeout
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn
