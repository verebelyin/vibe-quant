"""SQLite connection factory with WAL mode enabled by default."""

import os
import sqlite3
import sys
from pathlib import Path
from typing import Final

# Process-wide state DB. Override with VIBE_QUANT_DB when starting the backend:
# the env var is inherited by every job subprocess, so the API, job rows and
# results all use the same DB (a runtime switch could only ever swap part of it).
DEFAULT_DB_PATH: Final = Path(os.environ.get("VIBE_QUANT_DB") or "data/state/vibe_quant.db")
BUSY_TIMEOUT_MS: Final = 5000
# The real state DB of the checkout this package lives in (or a scratch copy's symlink to it).
_REAL_DB: Final = Path(__file__).resolve().parents[2] / "data" / "state" / "vibe_quant.db"


class RealStateDBUnderTestError(RuntimeError):
    """A test (or a job a test launched) tried to open the real state DB."""


def _under_test() -> bool:
    return "pytest" in sys.modules or bool(os.environ.get("VIBE_QUANT_PYTEST"))


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get SQLite connection with WAL mode and busy timeout configured.

    Args:
        db_path: Path to database file. Defaults to data/state/vibe_quant.db

    Returns:
        Configured SQLite connection with WAL mode enabled.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if _under_test() and os.path.realpath(db_path) == os.path.realpath(_REAL_DB):
        # 2026-10-19: a review scratch copy's data/state symlink slipped past the conftest
        # VIBE_QUANT_DB pin and a test inserted run 957 into the real DB.
        raise RealStateDBUnderTestError(f"tests must not open the real state DB: {db_path}")

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Enable WAL mode and set busy timeout
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn
