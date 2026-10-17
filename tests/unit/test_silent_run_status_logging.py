"""A failed run-status write must not vanish (exception-hygiene follow-up).

Each ``__main__`` writes "failed" for the run and records the job as done. When
that write itself fails, the run would stay ``running`` forever with nothing in
the log; the handler now logs an ERROR. The stdout print and the exit code are
unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest


class _BoomState:
    """StateManager double whose run reads/writes raise."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_backtest_run(self, run_id: int) -> object:
        raise RuntimeError("db locked")

    def update_backtest_run_status(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("db locked")

    def close(self) -> None:
        pass


class _BoomManager:
    """Job-manager double whose completion write raises."""

    def mark_completed(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("db locked")

    def close(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    import vibe_quant.db.state_manager as state_module
    import vibe_quant.jobs.manager as jobs_module

    monkeypatch.setattr(state_module, "StateManager", _BoomState)
    monkeypatch.setattr(
        jobs_module,
        "run_with_heartbeat",
        lambda run_id, db_path: (_BoomManager(), lambda: None),
    )


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _assert_logged(caplog: pytest.LogCaptureFixture, run_id: int) -> None:
    assert any(
        "could not record run" in m and f"run {run_id}" in m and "RuntimeError" in m
        for m in _messages(caplog)
    ), _messages(caplog)


def test_screening_run_status_write_failure_logs_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import vibe_quant.screening.__main__ as screening_main

    _install(monkeypatch)

    with caplog.at_level(logging.ERROR, logger="vibe_quant.screening.__main__"):
        code = screening_main.cmd_run(argparse.Namespace(run_id=7, db=None))

    assert code == 1
    _assert_logged(caplog, 7)


def test_discovery_run_status_write_failure_logs_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import vibe_quant.discovery.__main__ as discovery_main

    _install(monkeypatch)
    ns = SimpleNamespace(run_id=7, db=None)
    monkeypatch.setattr(
        discovery_main, "build_parser", lambda: SimpleNamespace(parse_args=lambda: ns)
    )
    monkeypatch.setattr(discovery_main, "check_launch_leverage", lambda parser, args: None)

    with caplog.at_level(logging.ERROR, logger="vibe_quant.discovery.__main__"):
        code = discovery_main.main()

    assert code == 1
    _assert_logged(caplog, 7)


def test_validation_run_status_write_failure_logs_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import vibe_quant.validation.__main__ as validation_main
    import vibe_quant.validation.runner as runner_module

    class _Runner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self, run_id: int, detail_timeframe: str | None = None) -> object:
            raise RuntimeError("boom")

        def close(self) -> None:
            pass

    _install(monkeypatch)
    monkeypatch.setattr(runner_module, "ValidationRunner", _Runner)
    monkeypatch.setattr(sys, "argv", ["prog", "--run-id", "7"])

    with caplog.at_level(logging.ERROR, logger="vibe_quant.validation.__main__"):
        code = validation_main.main()

    assert code == 1
    _assert_logged(caplog, 7)
