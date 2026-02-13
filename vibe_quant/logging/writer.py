"""Event writer for JSONL event logs.

Thread-safe writer that appends events to JSONL files for backtest event logging.
Supports context manager protocol for automatic resource cleanup.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

    from vibe_quant.logging.events import Event


class EventWriter:
    """Thread-safe JSONL event writer.

    Writes events to `logs/events/{run_id}.jsonl` files. Uses a lock for
    thread-safe writes and supports context manager protocol.

    Example:
        with EventWriter(run_id="abc123") as writer:
            writer.write(event)

    Attributes:
        run_id: Unique identifier for the backtest run.
        base_path: Directory for event logs.
    """

    def __init__(
        self,
        run_id: str,
        base_path: Path | str = "logs/events",
    ) -> None:
        """Initialize EventWriter.

        Args:
            run_id: Unique backtest run identifier.
            base_path: Directory for event log files.
        """
        self.run_id = run_id
        self.base_path = Path(base_path)
        self._lock = threading.Lock()
        self._file: open | None = None  # type: ignore[valid-type]
        self._closed = False

        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def file_path(self) -> Path:
        """Get path to the JSONL file for this run."""
        return self.base_path / f"{self.run_id}.jsonl"

    def _ensure_open(self) -> None:
        """Ensure file is open for writing."""
        if self._file is None:
            # Line buffering (buffering=1) ensures each write is flushed on newline,
            # preventing data loss if process crashes before close()
            self._file = open(self.file_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def write(self, event: Event) -> None:
        """Write event to JSONL file.

        Thread-safe. Serializes event to JSON and appends to file with newline.

        Args:
            event: Event to write.

        Raises:
            RuntimeError: If writer has been closed.
        """
        if self._closed:
            msg = "EventWriter has been closed"
            raise RuntimeError(msg)

        with self._lock:
            self._ensure_open()
            line = json.dumps(event.to_dict(), separators=(",", ":"))
            assert self._file is not None  # for type checker
            self._file.write(line + "\n")

    def write_many(self, events: list[Event]) -> None:
        """Write multiple events atomically.

        Thread-safe. Writes all events in a single lock acquisition.

        Args:
            events: List of events to write.

        Raises:
            RuntimeError: If writer has been closed.
        """
        if self._closed:
            msg = "EventWriter has been closed"
            raise RuntimeError(msg)

        with self._lock:
            self._ensure_open()
            assert self._file is not None  # for type checker
            for event in events:
                line = json.dumps(event.to_dict(), separators=(",", ":"))
                self._file.write(line + "\n")

    def flush(self) -> None:
        """Flush buffered writes to disk.

        Thread-safe. Ensures all writes are persisted.
        """
        with self._lock:
            if self._file is not None:
                self._file.flush()

    def close(self) -> None:
        """Close the file handle.

        Thread-safe. Safe to call multiple times.
        """
        with self._lock:
            self._closed = True
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> EventWriter:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing file."""
        self.close()


__all__ = ["EventWriter"]
