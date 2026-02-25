"""FastAPI application factory for vibe-quant."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vibe_quant import __version__
from vibe_quant.api.routers.backtest import router as backtest_router
from vibe_quant.api.routers.data import router as data_router
from vibe_quant.api.routers.discovery import router as discovery_router
from vibe_quant.api.routers.internal import router as internal_router
from vibe_quant.api.routers.paper_trading import router as paper_trading_router
from vibe_quant.api.routers.results import router as results_router
from vibe_quant.api.routers.settings import router as settings_router
from vibe_quant.api.routers.strategies import router as strategies_router
from vibe_quant.api.sse.progress import router as sse_progress_router
from vibe_quant.api.ws.discovery import router as ws_discovery_router
from vibe_quant.api.ws.jobs import router as ws_jobs_router
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.api.ws.trading import router as ws_trading_router
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["create_app"]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state_mgr = StateManager()
    job_mgr = BacktestJobManager()
    catalog_mgr = CatalogManager()

    ws_mgr = ConnectionManager()

    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.catalog_manager = catalog_mgr
    app.state.ws_manager = ws_mgr

    await ws_mgr.start()
    try:
        yield
    finally:
        await ws_mgr.stop()
        job_mgr.close()
        state_mgr.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="vibe-quant API",
        version=__version__,
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(backtest_router)
    app.include_router(data_router)
    app.include_router(discovery_router)
    app.include_router(internal_router)
    app.include_router(paper_trading_router)
    app.include_router(results_router)
    app.include_router(settings_router)
    app.include_router(strategies_router)
    app.include_router(sse_progress_router)
    app.include_router(ws_discovery_router)
    app.include_router(ws_jobs_router)
    app.include_router(ws_trading_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
