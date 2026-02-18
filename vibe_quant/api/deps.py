"""FastAPI dependency injection for vibe-quant."""

from __future__ import annotations

from fastapi import Request  # noqa: TCH002

from vibe_quant.api.ws.manager import ConnectionManager  # noqa: TCH001
from vibe_quant.data.catalog import CatalogManager  # noqa: TCH001
from vibe_quant.db.state_manager import StateManager  # noqa: TCH001
from vibe_quant.jobs.manager import BacktestJobManager  # noqa: TCH001

__all__ = ["get_catalog_manager", "get_job_manager", "get_state_manager", "get_ws_manager"]


def get_state_manager(request: Request) -> StateManager:
    return request.app.state.state_manager  # type: ignore[no-any-return]


def get_job_manager(request: Request) -> BacktestJobManager:
    return request.app.state.job_manager  # type: ignore[no-any-return]


def get_catalog_manager(request: Request) -> CatalogManager:
    return request.app.state.catalog_manager  # type: ignore[no-any-return]


def get_ws_manager(request: Request) -> ConnectionManager:
    return request.app.state.ws_manager  # type: ignore[no-any-return]
