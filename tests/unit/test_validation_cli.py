"""Tests for validation CLI entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.validation import __main__ as validation_cli

if TYPE_CHECKING:
    from pathlib import Path


class _FakeManager:
    def __init__(self) -> None:
        self.completed_calls: list[tuple[int, str | None]] = []
        self.closed = False

    def mark_completed(self, run_id: int, error: str | None = None) -> None:
        self.completed_calls.append((run_id, error))

    def close(self) -> None:
        self.closed = True


def test_main_unpacks_run_with_heartbeat_tuple(tmp_path: Path, monkeypatch) -> None:
    """CLI should unpack (manager, stop_fn) and call both manager.close and stop_fn."""
    fake_manager = _FakeManager()
    stop_called = {"value": False}

    def fake_stop() -> None:
        stop_called["value"] = True

    class FakeRunner:
        def __init__(self, db_path: Path) -> None:
            self.db_path = db_path
            self.closed = False

        def run(self, run_id: int):  # noqa: ANN001
            class Result:
                sharpe_ratio = 1.23
                total_return = 0.05

            assert run_id == 7
            return Result()

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "vibe_quant.jobs.manager.run_with_heartbeat",
        lambda run_id, db_path: (fake_manager, fake_stop),
    )
    monkeypatch.setattr("vibe_quant.validation.runner.ValidationRunner", FakeRunner)
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--run-id", "7", "--db", str(tmp_path / "state.db")],
    )

    assert validation_cli.main() == 0
    assert fake_manager.completed_calls == [(7, None)]
    assert fake_manager.closed is True
    assert stop_called["value"] is True
