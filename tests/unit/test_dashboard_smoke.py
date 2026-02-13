"""Smoke tests for dashboard page wiring and critical render path."""

from __future__ import annotations

import importlib
from typing import Any


def test_dashboard_page_modules_import_with_render_entrypoints() -> None:
    """All dashboard page modules import and expose expected render entrypoints."""
    page_modules = [
        ("vibe_quant.dashboard.pages.strategy_management", "render_strategy_management_tab"),
        ("vibe_quant.dashboard.pages.discovery", "render_discovery_tab"),
        ("vibe_quant.dashboard.pages.backtest_launch", "render_backtest_launch_tab"),
        ("vibe_quant.dashboard.pages.results_analysis", "render_results_tab"),
        ("vibe_quant.dashboard.pages.paper_trading", "render_paper_trading_tab"),
        ("vibe_quant.dashboard.pages.data_management", "render"),
        ("vibe_quant.dashboard.pages.settings", "render_settings_tab"),
    ]

    for module_name, render_name in page_modules:
        module = importlib.import_module(module_name)
        render_fn = getattr(module, render_name)
        assert callable(render_fn), f"{module_name}.{render_name} must be callable"


def test_dashboard_page_modules_do_not_render_on_import(monkeypatch: Any) -> None:
    """Page modules must not execute Streamlit render calls during import/reload."""
    import streamlit as st

    def _fail(name: str) -> Any:
        def _raiser(*args: Any, **kwargs: Any) -> None:
            raise AssertionError(f"Unexpected streamlit.{name} call at import time")

        return _raiser

    monkeypatch.setattr(st, "title", _fail("title"))
    monkeypatch.setattr(st, "header", _fail("header"))

    page_modules = [
        "vibe_quant.dashboard.pages.strategy_management",
        "vibe_quant.dashboard.pages.discovery",
        "vibe_quant.dashboard.pages.backtest_launch",
        "vibe_quant.dashboard.pages.results_analysis",
        "vibe_quant.dashboard.pages.paper_trading",
        "vibe_quant.dashboard.pages.data_management",
        "vibe_quant.dashboard.pages.settings",
    ]

    for module_name in page_modules:
        module = importlib.import_module(module_name)
        importlib.reload(module)


def test_dashboard_app_main_registers_all_pages(monkeypatch: Any) -> None:
    """app.main should register all dashboard pages with Streamlit navigation."""
    from vibe_quant.dashboard import app

    seen_page_callables: list[Any] = []
    seen_page_groups: dict[str, list[Any]] = {}
    page_config: dict[str, Any] = {}
    html_payloads: list[str] = []
    run_called = {"value": False}

    class _FakeNav:
        def run(self) -> None:
            run_called["value"] = True

    def fake_page(page_fn: Any, **kwargs: Any) -> dict[str, Any]:
        seen_page_callables.append(page_fn)
        return {"page": page_fn, **kwargs}

    def fake_navigation(pages: dict[str, list[Any]]) -> _FakeNav:
        seen_page_groups.update(pages)
        return _FakeNav()

    monkeypatch.setattr(app.st, "set_page_config", lambda **kwargs: page_config.update(kwargs))
    monkeypatch.setattr(app.st, "Page", fake_page)
    monkeypatch.setattr(app.st, "navigation", fake_navigation)
    monkeypatch.setattr(app.st, "html", lambda payload: html_payloads.append(payload))

    app.main()

    assert page_config["page_title"] == "vibe-quant Dashboard"
    assert run_called["value"] is True
    assert set(seen_page_groups) == {"Strategies", "Backtesting", "Trading", "System"}

    expected_entrypoints = {
        "render_strategy_management_tab",
        "render_discovery_tab",
        "render_backtest_launch_tab",
        "render_results_tab",
        "render_paper_trading_tab",
        "render",
        "render_settings_tab",
    }
    assert {fn.__name__ for fn in seen_page_callables} == expected_entrypoints
    assert html_payloads, "st.html keyboard-handler injection should be rendered"


def test_dashboard_app_main_handles_page_runtime_error(monkeypatch: Any) -> None:
    """app.main should surface page errors instead of crashing server."""
    from vibe_quant.dashboard import app

    captured_errors: list[str] = []
    captured_info: list[str] = []

    class _FailingNav:
        def run(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(app.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(app.st, "Page", lambda path, **kwargs: {"path": path, **kwargs})
    monkeypatch.setattr(app.st, "navigation", lambda pages: _FailingNav())
    monkeypatch.setattr(app.st, "html", lambda payload: None)
    monkeypatch.setattr(app.st, "error", lambda message: captured_errors.append(message))
    monkeypatch.setattr(app.st, "info", lambda message: captured_info.append(message))

    app.main()

    assert captured_errors and "Page error: boom" in captured_errors[0]
    assert captured_info and "Try refreshing the page" in captured_info[0]
