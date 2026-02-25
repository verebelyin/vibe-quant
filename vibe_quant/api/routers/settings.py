"""Settings router: sizing configs, risk configs, latency presets, system info, database."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from vibe_quant.api.deps import get_state_manager
from vibe_quant.api.schemas.settings import (
    DatabaseInfoResponse,
    DatabaseSwitchRequest,
    LatencyPreset,
    RiskConfigCreate,
    RiskConfigResponse,
    RiskConfigUpdate,
    SizingConfigCreate,
    SizingConfigResponse,
    SizingConfigUpdate,
    SystemInfoResponse,
)
from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]

_LATENCY_PRESETS: list[LatencyPreset] = [
    LatencyPreset(name="co_located", description="Co-located server (1ms)", base_latency_ms=1),
    LatencyPreset(name="domestic", description="Domestic / same-region (20ms)", base_latency_ms=20),
    LatencyPreset(name="international", description="International / cross-region (100ms)", base_latency_ms=100),
    LatencyPreset(name="retail", description="Retail home connection (200ms)", base_latency_ms=200),
]

# ---------------------------------------------------------------------------
# Sizing configs
# ---------------------------------------------------------------------------


@router.get("/sizing", response_model=list[SizingConfigResponse])
async def list_sizing_configs(mgr: StateMgr) -> list[SizingConfigResponse]:
    rows = mgr.list_sizing_configs()
    return [SizingConfigResponse(**r) for r in rows]


@router.post("/sizing", response_model=SizingConfigResponse, status_code=201)
async def create_sizing_config(body: SizingConfigCreate, mgr: StateMgr) -> SizingConfigResponse:
    config_id = mgr.create_sizing_config(name=body.name, method=body.method, config=body.config)
    row = mgr.get_sizing_config(config_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Failed to create sizing config")
    return SizingConfigResponse(**row)


@router.get("/sizing/{config_id}", response_model=SizingConfigResponse)
async def get_sizing_config(config_id: int, mgr: StateMgr) -> SizingConfigResponse:
    row = mgr.get_sizing_config(config_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Sizing config not found")
    return SizingConfigResponse(**row)


@router.put("/sizing/{config_id}", response_model=SizingConfigResponse)
async def update_sizing_config(
    config_id: int, body: SizingConfigUpdate, mgr: StateMgr
) -> SizingConfigResponse:
    existing = mgr.get_sizing_config(config_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Sizing config not found")
    mgr.update_sizing_config(
        config_id,
        name=body.name if body.name is not None else existing["name"],
        method=body.method if body.method is not None else existing["method"],
        config=body.config if body.config is not None else existing["config"],
    )
    row = mgr.get_sizing_config(config_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Sizing config disappeared after update")
    return SizingConfigResponse(**row)


@router.delete("/sizing/{config_id}", status_code=204)
async def delete_sizing_config(config_id: int, mgr: StateMgr) -> Response:
    existing = mgr.get_sizing_config(config_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Sizing config not found")
    mgr.delete_sizing_config(config_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Risk configs
# ---------------------------------------------------------------------------


@router.get("/risk", response_model=list[RiskConfigResponse])
async def list_risk_configs(mgr: StateMgr) -> list[RiskConfigResponse]:
    rows = mgr.list_risk_configs()
    return [RiskConfigResponse(**r) for r in rows]


@router.post("/risk", response_model=RiskConfigResponse, status_code=201)
async def create_risk_config(body: RiskConfigCreate, mgr: StateMgr) -> RiskConfigResponse:
    config_id = mgr.create_risk_config(
        name=body.name,
        strategy_level=body.strategy_level,
        portfolio_level=body.portfolio_level,
    )
    row = mgr.get_risk_config(config_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Failed to create risk config")
    return RiskConfigResponse(**row)


@router.get("/risk/{config_id}", response_model=RiskConfigResponse)
async def get_risk_config(config_id: int, mgr: StateMgr) -> RiskConfigResponse:
    row = mgr.get_risk_config(config_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Risk config not found")
    return RiskConfigResponse(**row)


@router.put("/risk/{config_id}", response_model=RiskConfigResponse)
async def update_risk_config(
    config_id: int, body: RiskConfigUpdate, mgr: StateMgr
) -> RiskConfigResponse:
    existing = mgr.get_risk_config(config_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Risk config not found")
    mgr.update_risk_config(
        config_id,
        name=body.name if body.name is not None else existing["name"],
        strategy_level=(
            body.strategy_level if body.strategy_level is not None else existing["strategy_level"]
        ),
        portfolio_level=(
            body.portfolio_level
            if body.portfolio_level is not None
            else existing["portfolio_level"]
        ),
    )
    row = mgr.get_risk_config(config_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Risk config disappeared after update")
    return RiskConfigResponse(**row)


@router.delete("/risk/{config_id}", status_code=204)
async def delete_risk_config(config_id: int, mgr: StateMgr) -> Response:
    existing = mgr.get_risk_config(config_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Risk config not found")
    mgr.delete_risk_config(config_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Latency presets
# ---------------------------------------------------------------------------


@router.get("/latency-presets", response_model=list[LatencyPreset])
async def list_latency_presets() -> list[LatencyPreset]:
    return _LATENCY_PRESETS


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def _get_nt_version() -> str:
    try:
        import nautilus_trader  # noqa: PLC0415

        return nautilus_trader.__version__  # type: ignore[no-any-return]
    except (ImportError, AttributeError):
        return "not installed"


def _get_table_counts(mgr: StateManager) -> dict[str, int]:
    cursor = mgr.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor]
    counts: dict[str, int] = {}
    for table in tables:
        count_cursor = mgr.conn.execute(f"SELECT COUNT(*) FROM [{table}]")  # noqa: S608
        counts[table] = count_cursor.fetchone()[0]
    return counts


def _file_size(path: str | Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


@router.get("/system-info", response_model=SystemInfoResponse)
async def get_system_info(mgr: StateMgr) -> SystemInfoResponse:
    db_path = mgr._db_path or Path("data/state/vibe_quant.db")  # noqa: SLF001
    catalog_path = Path("data/catalog")
    catalog_size = 0
    if catalog_path.exists():
        catalog_size = sum(f.stat().st_size for f in catalog_path.rglob("*") if f.is_file())
    return SystemInfoResponse(
        nt_version=_get_nt_version(),
        python_version=sys.version,
        catalog_size_bytes=catalog_size,
        db_size_bytes=_file_size(db_path),
        table_counts=_get_table_counts(mgr),
    )


# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------


@router.get("/database", response_model=DatabaseInfoResponse)
async def get_database_info(mgr: StateMgr) -> DatabaseInfoResponse:
    db_path = mgr._db_path or Path("data/state/vibe_quant.db")  # noqa: SLF001
    cursor = mgr.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor]
    return DatabaseInfoResponse(path=str(db_path), tables=tables)


@router.put("/database", response_model=DatabaseInfoResponse)
async def switch_database(body: DatabaseSwitchRequest, request: Request) -> DatabaseInfoResponse:
    new_path = Path(body.path)
    if not new_path.suffix == ".db":
        raise HTTPException(status_code=400, detail="Database path must end in .db")

    # Prevent path traversal — resolve to absolute and enforce allowed directory
    allowed_dir = Path("data/state").resolve()
    try:
        resolved = new_path.resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc
    if not str(resolved).startswith(str(allowed_dir) + os.sep) and resolved != allowed_dir:
        raise HTTPException(
            status_code=400,
            detail="Database path must be within data/state/ directory",
        )

    if not new_path.parent.exists():
        raise HTTPException(status_code=400, detail="Parent directory does not exist")

    old_mgr: StateManager = request.app.state.state_manager
    old_mgr.close()

    new_mgr = StateManager(db_path=new_path)
    # Force connection init
    _ = new_mgr.conn
    request.app.state.state_manager = new_mgr

    cursor = new_mgr.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor]
    return DatabaseInfoResponse(path=str(new_path), tables=tables)
