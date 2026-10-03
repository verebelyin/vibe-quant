"""FastAPI application factory for vibe-quant."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vibe_quant import __version__
from vibe_quant.api.routers.backtest import router as backtest_router
from vibe_quant.api.routers.data import router as data_router
from vibe_quant.api.routers.discovery import router as discovery_router
from vibe_quant.api.routers.indicators import router as indicators_router
from vibe_quant.api.routers.internal import router as internal_router
from vibe_quant.api.routers.paper_trading import router as paper_trading_router
from vibe_quant.api.routers.reconciliation import router as reconciliation_router
from vibe_quant.api.routers.research import router as research_router
from vibe_quant.api.routers.results import router as results_router
from vibe_quant.api.routers.settings import router as settings_router
from vibe_quant.api.routers.strategies import router as strategies_router
from vibe_quant.api.routers.system import router as system_router
from vibe_quant.api.sse.progress import router as sse_progress_router
from vibe_quant.api.ws.discovery import router as ws_discovery_router
from vibe_quant.api.ws.jobs import router as ws_jobs_router
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.api.ws.trading import router as ws_trading_router
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Not CORS-safelisted → a cross-site page cannot send it without a preflight,
# which CORSMiddleware rejects for foreign origins.
CSRF_HEADER = "x-requested-with"


async def _csrf_guard(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject browser-originated state-changing requests a foreign page could forge.

    Cross-site "simple" requests (HTML form posts, ``fetch`` with ``no-cors``)
    skip the CORS preflight, so bodyless POSTs like ``/jobs/cleanup-stale`` or
    ``/indicators/reload`` could be triggered from any website. Browsers always
    attach ``Origin``/``Sec-Fetch-Site`` to such requests; when present we
    require a JSON body type or the custom ``X-Requested-With`` header — both
    force a preflight cross-origin. Non-browser clients (CLI, tests) are
    unaffected.
    """
    if request.method in _UNSAFE_METHODS and (
        "origin" in request.headers or "sec-fetch-site" in request.headers
    ):
        content_type = request.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json" and CSRF_HEADER not in request.headers:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Cross-site request rejected: send Content-Type: "
                    "application/json or an X-Requested-With header"
                },
            )
    return await call_next(request)


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

    # Jobs that died while the backend was down would otherwise show
    # 'running' forever; live ones keep running (they heartbeat themselves).
    reconciled = job_mgr.reconcile_jobs()
    if reconciled:
        logger.warning("startup: marked %d dead 'running' job(s) failed", reconciled)

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

    # Registered before CORS so CORS stays outermost (preflights answered first).
    app.middleware("http")(_csrf_guard)
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
    app.include_router(indicators_router)
    app.include_router(internal_router)
    app.include_router(paper_trading_router)
    app.include_router(reconciliation_router)
    app.include_router(research_router)
    app.include_router(results_router)
    app.include_router(settings_router)
    app.include_router(strategies_router)
    app.include_router(system_router)
    app.include_router(sse_progress_router)
    app.include_router(ws_discovery_router)
    app.include_router(ws_jobs_router)
    app.include_router(ws_trading_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
