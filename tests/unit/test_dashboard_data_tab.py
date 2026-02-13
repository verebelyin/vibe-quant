"""Tests for dashboard data management tab."""

from __future__ import annotations

from typing import Any


def test_module_import() -> None:
    """Test that data_management module can be imported without errors."""
    from vibe_quant.dashboard.pages import data_management

    assert data_management is not None


def test_render_function_exists() -> None:
    """Test that render() function exists and is callable."""
    from vibe_quant.dashboard.pages.data_management import render

    assert callable(render)


def test_format_bytes() -> None:
    """Test format_bytes helper function."""
    from vibe_quant.dashboard.utils import format_bytes

    assert format_bytes(0) == "0.0 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(1024 * 1024) == "1.0 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
    assert format_bytes(500) == "500.0 B"


def test_get_storage_usage_no_data() -> None:
    """Test _get_storage_usage returns zeros when no data exists."""
    from vibe_quant.dashboard.pages.data_management import _get_storage_usage

    # Should not raise even if paths don't exist
    usage = _get_storage_usage()
    assert "archive" in usage
    assert "catalog" in usage
    assert isinstance(usage["archive"], int)
    assert isinstance(usage["catalog"], int)


def test_helper_functions_exist() -> None:
    """Test that all helper functions exist."""
    from vibe_quant.dashboard.pages import data_management

    assert hasattr(data_management, "format_bytes")
    assert hasattr(data_management, "_get_storage_usage")
    assert hasattr(data_management, "_get_data_status")
    assert hasattr(data_management, "_verify_data")
    assert hasattr(data_management, "render_storage_metrics")
    assert hasattr(data_management, "render_data_coverage")
    assert hasattr(data_management, "render_symbol_management")
    assert hasattr(data_management, "render_data_actions")
    assert hasattr(data_management, "render_data_quality")


def test_render_functions_callable() -> None:
    """Test that render functions are callable."""
    from vibe_quant.dashboard.pages.data_management import (
        render,
        render_data_actions,
        render_data_coverage,
        render_data_quality,
        render_storage_metrics,
        render_symbol_management,
    )

    assert callable(render)
    assert callable(render_storage_metrics)
    assert callable(render_data_coverage)
    assert callable(render_symbol_management)
    assert callable(render_data_actions)
    assert callable(render_data_quality)


def test_run_subprocess_with_progress_handles_popen_error(monkeypatch: Any) -> None:
    """Subprocess spawn failures should be surfaced with deterministic error status."""
    from vibe_quant.dashboard.pages import data_management

    updates: list[dict[str, str]] = []
    unregister_calls = {"count": 0}

    class _FakeLogArea:
        def code(self, _text: str) -> None:
            return None

    class _FakeStatus:
        def empty(self) -> _FakeLogArea:
            return _FakeLogArea()

        def update(self, *, label: str, state: str) -> None:
            updates.append({"label": label, "state": state})

    def _raise_popen(*_args: object, **_kwargs: object) -> Any:
        raise OSError("spawn failed")

    monkeypatch.setattr(data_management.st, "status", lambda *_args, **_kwargs: _FakeStatus())
    monkeypatch.setattr(data_management.subprocess, "Popen", _raise_popen)
    monkeypatch.setattr(
        data_management,
        "_unregister_subprocess",
        lambda: unregister_calls.__setitem__("count", unregister_calls["count"] + 1),
    )

    data_management._run_subprocess_with_progress(
        cmd=["/no/such/cmd"],
        label="Running...",
        success_label="Done",
        fail_label="Failed",
    )

    assert unregister_calls["count"] == 1
    assert updates[-1]["state"] == "error"
    assert "Failed: OSError: spawn failed" in updates[-1]["label"]
