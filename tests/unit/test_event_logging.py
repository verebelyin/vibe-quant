"""Tests for structured event logging."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestEventTypes:
    """Tests for Event and EventType."""

    def test_event_from_dict(self) -> None:
        """Event.from_dict deserializes correctly."""
        from vibe_quant.logging import Event, EventType

        d = {
            "timestamp": "2024-01-15T10:00:00+00:00",
            "event_type": "ORDER",
            "run_id": "xyz789",
            "strategy_name": "my_strat",
            "data": {"order_id": "O-001"},
        }

        event = Event.from_dict(d)

        assert event.event_type == EventType.ORDER
        assert event.run_id == "xyz789"
        assert event.strategy_name == "my_strat"
        assert event.data["order_id"] == "O-001"


class TestEventQuery:
    """Tests for query functions."""

    def test_query_events(self, tmp_path: Path) -> None:
        """Query events returns list of dicts."""
        from vibe_quant.logging import Event, EventType, EventWriter, query_events

        # Write test events
        with EventWriter(run_id="query_test", base_path=tmp_path) as writer:
            for i in range(3):
                event = Event(
                    timestamp=datetime(2024, 1, 15, 10, i, 0, tzinfo=UTC),
                    event_type=EventType.SIGNAL if i % 2 == 0 else EventType.ORDER,
                    run_id="query_test",
                    strategy_name="strat",
                    data={"seq": i},
                )
                writer.write(event)

        # Query all events
        events = query_events("query_test", base_path=tmp_path)
        assert len(events) == 3

        # Query filtered
        signals = query_events("query_test", event_type=EventType.SIGNAL, base_path=tmp_path)
        assert len(signals) == 2

    def test_query_events_df(self, tmp_path: Path) -> None:
        """Query events returns DataFrame."""
        from vibe_quant.logging import Event, EventType, EventWriter, query_events_df

        with EventWriter(run_id="df_test", base_path=tmp_path) as writer:
            for i in range(5):
                event = Event(
                    timestamp=datetime(2024, 1, 15, 10, i, 0, tzinfo=UTC),
                    event_type=EventType.FILL,
                    run_id="df_test",
                    strategy_name="strat",
                    data={"price": 42000 + i * 10},
                )
                writer.write(event)

        df = query_events_df("df_test", base_path=tmp_path)

        assert len(df) == 5
        assert "event" in df.columns
        assert "ts" in df.columns

    def test_query_nonexistent_raises(self, tmp_path: Path) -> None:
        """Query nonexistent file raises FileNotFoundError."""
        from vibe_quant.logging import query_events

        with pytest.raises(FileNotFoundError):
            query_events("nonexistent", base_path=tmp_path)

    def test_count_events_by_type(self, tmp_path: Path) -> None:
        """Count events by type."""
        from vibe_quant.logging import (
            Event,
            EventType,
            EventWriter,
            count_events_by_type,
        )

        with EventWriter(run_id="count_test", base_path=tmp_path) as writer:
            for event_type in [EventType.SIGNAL, EventType.SIGNAL, EventType.ORDER]:
                event = Event(
                    timestamp=datetime.now(UTC),
                    event_type=event_type,
                    run_id="count_test",
                    strategy_name="strat",
                    data={},
                )
                writer.write(event)

        counts = count_events_by_type("count_test", base_path=tmp_path)

        assert counts["SIGNAL"] == 2
        assert counts["ORDER"] == 1

    def test_get_run_summary(self, tmp_path: Path) -> None:
        """Get run summary."""
        from vibe_quant.logging import Event, EventType, EventWriter, get_run_summary

        with EventWriter(run_id="summary_test", base_path=tmp_path) as writer:
            for i, et in enumerate([EventType.SIGNAL, EventType.ORDER, EventType.FILL]):
                event = Event(
                    timestamp=datetime(2024, 1, 15, 10, i, 0, tzinfo=UTC),
                    event_type=et,
                    run_id="summary_test",
                    strategy_name="strat",
                    data={},
                )
                writer.write(event)

        summary = get_run_summary("summary_test", base_path=tmp_path)

        assert summary["run_id"] == "summary_test"
        assert summary["total_events"] == 3
        assert summary["event_type_count"] == 3
