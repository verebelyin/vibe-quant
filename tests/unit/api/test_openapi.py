"""OpenAPI spec validation and snapshot tests."""

from __future__ import annotations

from vibe_quant.api.app import create_app

EXPECTED_PATH_COUNT = 67
EXPECTED_SCHEMA_COUNT = 53

REQUIRED_PATHS = [
    "/health",
    "/api/strategies",
    "/api/backtest/screening",
    "/api/backtest/validation",
    "/api/backtest/jobs",
    "/api/results/runs",
    "/api/results/compare",
    "/api/data/status",
    "/api/data/coverage",
    "/api/data/ingest",
    "/api/discovery/launch",
    "/api/discovery/jobs",
    "/api/paper/start",
    "/api/paper/status",
    "/api/settings/sizing",
    "/api/settings/risk",
    "/api/settings/system-info",
    "/api/settings/database",
]


def test_openapi_spec_generates() -> None:
    app = create_app()
    spec = app.openapi()
    assert spec["info"]["title"] == "vibe-quant API"
    assert "paths" in spec


def test_openapi_path_count() -> None:
    app = create_app()
    spec = app.openapi()
    assert len(spec["paths"]) == EXPECTED_PATH_COUNT


def test_openapi_schema_count() -> None:
    app = create_app()
    spec = app.openapi()
    schemas = spec.get("components", {}).get("schemas", {})
    assert len(schemas) == EXPECTED_SCHEMA_COUNT


def test_openapi_required_paths_present() -> None:
    app = create_app()
    spec = app.openapi()
    paths = set(spec["paths"].keys())
    for path in REQUIRED_PATHS:
        assert path in paths, f"Missing required path: {path}"


def test_openapi_health_endpoint() -> None:
    app = create_app()
    spec = app.openapi()
    health = spec["paths"]["/health"]
    assert "get" in health


def test_openapi_ws_endpoints_excluded() -> None:
    """WebSocket endpoints should not appear in OpenAPI spec."""
    app = create_app()
    spec = app.openapi()
    for path in spec["paths"]:
        assert not path.startswith("/ws/"), f"WS endpoint in OpenAPI: {path}"
