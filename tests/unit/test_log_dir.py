"""Tests for vibe_quant.utils.log_dir and the job-log leak fix (vibe-quant-yul7u.21)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.routers.discovery import _read_progress_file
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.utils import log_dir

if TYPE_CHECKING:
    import pytest


def test_log_dir_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "job-logs"
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(target))

    assert log_dir() == target
    assert target.is_dir()


def test_log_dir_defaults_to_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIBE_QUANT_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert log_dir() == Path("logs")
    assert (tmp_path / "logs").is_dir()


def test_log_dir_empty_env_falls_back_to_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An empty override must not resolve to Path("."); it means "unset".
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", "")
    monkeypatch.chdir(tmp_path)

    assert log_dir() == Path("logs")


def test_log_dir_create_false_does_not_mkdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "not-created"
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(target))

    assert log_dir(create=False) == target
    assert not target.exists()


def test_read_progress_file_does_not_create_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A GET path must not create the log dir as a side effect of reading.
    target = tmp_path / "reader-logs"
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(target))

    assert _read_progress_file(12345) is None
    assert not target.exists()


async def test_api_launch_writes_job_log_under_env_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Launching screening + validation via the API writes logs under env dir.

    subprocess.Popen is patched, but BacktestJobManager.start_job is REAL — it
    opens the log file — so the assertion only passes if the router resolves the
    log path through ``log_dir()`` (a hardcoded ``logs/...`` would leak into the
    repo / cwd instead).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "joblogs"))

    db = tmp_path / "api.db"
    app = create_app()
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.catalog_manager = CatalogManager()
    app.state.ws_manager = ConnectionManager()

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            sid = state.create_strategy("guard", {"name": "guard"})
            body = {
                "strategy_id": sid,
                "symbols": ["BTCUSDT"],
                "timeframe": "4h",
                "start_date": "2025-01-01",
                "end_date": "2025-06-01",
                "parameters": {"leverage": 5, "initial_balance": 1000},
            }
            with patch("subprocess.Popen") as popen:
                popen.return_value.pid = 4242
                screening = await ac.post("/api/backtest/screening", json=body)
                validation = await ac.post("/api/backtest/validation", json=body)
    finally:
        jobs.close()
        state.close()

    assert screening.status_code == 201, screening.text
    assert validation.status_code == 201, validation.text

    joblogs = tmp_path / "joblogs"
    assert list(joblogs.glob("screening_*.log")), "screening job log missing from env dir"
    assert list(joblogs.glob("validation_*.log")), "validation job log missing from env dir"
    assert not (tmp_path / "logs").exists(), "repo logs/ must not be created"


def test_event_default_dirs_follow_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Writer + query default event dirs both resolve under the env dir.

    Before this, ``EventWriter`` and ``query`` defaulted to the absolute
    project-root ``logs/events`` and ignored ``VIBE_QUANT_LOG_DIR`` while
    ``ValidationRunner`` honored it — so the writer and the reader disagreed
    whenever the env var was set.
    """
    from vibe_quant.logging import query as query_module
    from vibe_quant.logging import writer as writer_module

    target = tmp_path / "event-logs"
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(target))
    monkeypatch.chdir(tmp_path)

    writer = writer_module.EventWriter(run_id="env-default")
    try:
        assert writer.base_path == target / "events"
    finally:
        writer.close()

    assert query_module._get_log_path("env-default").parent == target / "events"


def test_event_default_dirs_unset_keep_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the env var unset the default stays the absolute project-root path."""
    from vibe_quant.logging import query as query_module
    from vibe_quant.logging import writer as writer_module

    monkeypatch.delenv("VIBE_QUANT_LOG_DIR", raising=False)
    # A different cwd must not turn the default into a relative path.
    monkeypatch.chdir(tmp_path)

    expected = writer_module.EventWriter._DEFAULT_BASE_PATH
    assert expected.is_absolute()
    assert writer_module._default_base_path() == expected
    assert query_module._default_base_path() == expected
