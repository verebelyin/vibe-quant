"""Tests for structured event logging."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestEventTypes:
    """Tests for Event and EventType."""

    def test_event_to_dict(self) -> None:
        """Event.to_dict serializes correctly."""
        from vibe_quant.logging import Event, EventType

        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        event = Event(
            timestamp=ts,
            event_type=EventType.SIGNAL,
            run_id="abc123",
            strategy_name="test_strat",
            data={"indicator": "rsi", "value": 28.5},
        )

        d = event.to_dict()

        assert d["ts"] == "2024-01-15T10:00:00+00:00"
        assert d["event"] == "SIGNAL"
        assert d["run_id"] == "abc123"
        assert d["strategy"] == "test_strat"
        assert d["data"]["indicator"] == "rsi"
        assert d["data"]["value"] == 28.5

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

    def test_signal_event(self) -> None:
        """SignalEvent populates data dict."""
        from vibe_quant.logging import EventType, SignalEvent

        event = SignalEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.SIGNAL,  # Will be overwritten
            run_id="run1",
            strategy_name="strat1",
            indicator="rsi",
            value=25.0,
            condition="rsi < 30",
            side="long",
        )

        assert event.event_type == EventType.SIGNAL
        assert event.data["indicator"] == "rsi"
        assert event.data["value"] == 25.0
        assert event.data["condition"] == "rsi < 30"
        assert event.data["side"] == "long"

    def test_order_event(self) -> None:
        """OrderEvent populates data dict."""
        from vibe_quant.logging import EventType, OrderEvent

        event = OrderEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.ORDER,
            run_id="run1",
            strategy_name="strat1",
            order_id="O-001",
            side="BUY",
            quantity=0.1,
            price=42000.0,
            order_type="LIMIT",
            reason="signal",
        )

        assert event.event_type == EventType.ORDER
        assert event.data["order_id"] == "O-001"
        assert event.data["price"] == 42000.0

    def test_fill_event(self) -> None:
        """FillEvent populates data dict."""
        from vibe_quant.logging import EventType, FillEvent

        event = FillEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.FILL,
            run_id="run1",
            strategy_name="strat1",
            order_id="O-001",
            fill_price=42010.0,
            quantity=0.1,
            fees=4.2,
            slippage=10.0,
        )

        assert event.event_type == EventType.FILL
        assert event.data["fill_price"] == 42010.0
        assert event.data["fees"] == 4.2

    def test_position_events(self) -> None:
        """Position open/close events populate data."""
        from vibe_quant.logging import EventType, PositionCloseEvent, PositionOpenEvent

        open_event = PositionOpenEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.POSITION_OPEN,
            run_id="run1",
            strategy_name="strat1",
            position_id="P-001",
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.1,
            entry_price=42000.0,
            leverage=10,
        )

        assert open_event.event_type == EventType.POSITION_OPEN
        assert open_event.data["leverage"] == 10

        close_event = PositionCloseEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.POSITION_CLOSE,
            run_id="run1",
            strategy_name="strat1",
            position_id="P-001",
            symbol="BTCUSDT",
            exit_price=43000.0,
            gross_pnl=100.0,
            net_pnl=95.0,
            exit_reason="take_profit",
        )

        assert close_event.event_type == EventType.POSITION_CLOSE
        assert close_event.data["net_pnl"] == 95.0

    def test_risk_check_event(self) -> None:
        """RiskCheckEvent populates data dict."""
        from vibe_quant.logging import EventType, RiskCheckEvent

        event = RiskCheckEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.RISK_CHECK,
            run_id="run1",
            strategy_name="strat1",
            check_type="drawdown",
            metric="current_drawdown_pct",
            current_value=0.12,
            threshold=0.15,
            action="",
            passed=True,
        )

        assert event.event_type == EventType.RISK_CHECK
        assert event.data["passed"] is True

    def test_funding_event(self) -> None:
        """FundingEvent populates data dict."""
        from vibe_quant.logging import EventType, FundingEvent

        event = FundingEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.FUNDING,
            run_id="run1",
            strategy_name="strat1",
            symbol="BTCUSDT",
            funding_rate=0.0001,
            funding_payment=-4.2,
            position_value=42000.0,
        )

        assert event.event_type == EventType.FUNDING
        assert event.data["funding_payment"] == -4.2

    def test_liquidation_event(self) -> None:
        """LiquidationEvent populates data dict."""
        from vibe_quant.logging import EventType, LiquidationEvent

        event = LiquidationEvent(
            timestamp=datetime.now(UTC),
            event_type=EventType.LIQUIDATION,
            run_id="run1",
            strategy_name="strat1",
            position_id="P-001",
            symbol="BTCUSDT",
            liquidation_price=40000.0,
            quantity=0.1,
            loss=420.0,
        )

        assert event.event_type == EventType.LIQUIDATION
        assert event.data["loss"] == 420.0

    def test_create_event_factory(self) -> None:
        """create_event factory creates events correctly."""
        from vibe_quant.logging import EventType, create_event

        event = create_event(
            event_type=EventType.SIGNAL,
            run_id="run1",
            strategy_name="strat1",
            data={"indicator": "rsi", "value": 30},
        )

        assert event.event_type == EventType.SIGNAL
        assert event.run_id == "run1"
        assert event.data["indicator"] == "rsi"


class TestEventWriter:
    """Tests for EventWriter."""

    def test_write_single_event(self, tmp_path: Path) -> None:
        """Write single event to JSONL."""
        from vibe_quant.logging import Event, EventType, EventWriter

        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        event = Event(
            timestamp=ts,
            event_type=EventType.SIGNAL,
            run_id="test_run",
            strategy_name="test_strat",
            data={"value": 42},
        )

        with EventWriter(run_id="test_run", base_path=tmp_path) as writer:
            writer.write(event)

        log_path = tmp_path / "test_run.jsonl"
        assert log_path.exists()

        with open(log_path) as f:
            line = f.readline()
            d = json.loads(line)
            assert d["event"] == "SIGNAL"
            assert d["data"]["value"] == 42

    def test_write_many_events(self, tmp_path: Path) -> None:
        """Write multiple events atomically."""
        from vibe_quant.logging import Event, EventType, EventWriter

        events = [
            Event(
                timestamp=datetime.now(UTC),
                event_type=EventType.SIGNAL,
                run_id="test_run",
                strategy_name="strat",
                data={"n": i},
            )
            for i in range(5)
        ]

        with EventWriter(run_id="test_run", base_path=tmp_path) as writer:
            writer.write_many(events)

        log_path = tmp_path / "test_run.jsonl"
        with open(log_path) as f:
            lines = f.readlines()
            assert len(lines) == 5

    def test_flush(self, tmp_path: Path) -> None:
        """Flush ensures data is written."""
        from vibe_quant.logging import Event, EventType, EventWriter

        event = Event(
            timestamp=datetime.now(UTC),
            event_type=EventType.ORDER,
            run_id="flush_test",
            strategy_name="strat",
            data={},
        )

        writer = EventWriter(run_id="flush_test", base_path=tmp_path)
        writer.write(event)
        writer.flush()

        # File should have content before close
        log_path = tmp_path / "flush_test.jsonl"
        assert log_path.exists()
        assert log_path.stat().st_size > 0

        writer.close()

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        """Write after close raises RuntimeError."""
        from vibe_quant.logging import Event, EventType, EventWriter

        writer = EventWriter(run_id="closed_test", base_path=tmp_path)
        writer.close()

        event = Event(
            timestamp=datetime.now(UTC),
            event_type=EventType.FILL,
            run_id="closed_test",
            strategy_name="strat",
            data={},
        )

        with pytest.raises(RuntimeError, match="has been closed"):
            writer.write(event)

    def test_thread_safety(self, tmp_path: Path) -> None:
        """Concurrent writes don't corrupt file."""
        from vibe_quant.logging import Event, EventType, EventWriter

        writer = EventWriter(run_id="threaded", base_path=tmp_path)
        num_threads = 10
        events_per_thread = 100

        def write_events(thread_id: int) -> None:
            for i in range(events_per_thread):
                event = Event(
                    timestamp=datetime.now(UTC),
                    event_type=EventType.SIGNAL,
                    run_id="threaded",
                    strategy_name="strat",
                    data={"thread": thread_id, "seq": i},
                )
                writer.write(event)

        threads = [
            threading.Thread(target=write_events, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        writer.close()

        # Verify all events written
        log_path = tmp_path / "threaded.jsonl"
        with open(log_path) as f:
            lines = f.readlines()
            assert len(lines) == num_threads * events_per_thread

            # Verify each line is valid JSON
            for line in lines:
                d = json.loads(line)
                assert "thread" in d["data"]


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
        signals = query_events(
            "query_test", event_type=EventType.SIGNAL, base_path=tmp_path
        )
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

    def test_list_runs(self, tmp_path: Path) -> None:
        """List runs returns all run IDs."""
        from vibe_quant.logging import Event, EventType, EventWriter, list_runs

        for run_id in ["run_a", "run_b", "run_c"]:
            with EventWriter(run_id=run_id, base_path=tmp_path) as writer:
                event = Event(
                    timestamp=datetime.now(UTC),
                    event_type=EventType.SIGNAL,
                    run_id=run_id,
                    strategy_name="strat",
                    data={},
                )
                writer.write(event)

        runs = list_runs(base_path=tmp_path)

        assert set(runs) == {"run_a", "run_b", "run_c"}

    def test_list_runs_empty(self, tmp_path: Path) -> None:
        """List runs on empty/nonexistent dir returns empty list."""
        from vibe_quant.logging import list_runs

        runs = list_runs(base_path=tmp_path / "nonexistent")
        assert runs == []
