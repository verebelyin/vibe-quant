"""Strategy CRUD router."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from vibe_quant.api.deps import get_state_manager
from vibe_quant.api.schemas.strategy import (
    StrategyCreate,
    StrategyListResponse,
    StrategyResponse,
    StrategyUpdate,
    ValidationResult,
)
from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    mgr: StateMgr,
    active_only: bool = True,
) -> StrategyListResponse:
    rows = mgr.list_strategies(active_only=active_only)
    return StrategyListResponse(
        strategies=[StrategyResponse(**r) for r in rows],
    )


@router.get("/templates")
async def list_templates() -> list[dict[str, object]]:
    # TODO: wire up DSL template registry when available
    return []


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: int, mgr: StateMgr) -> StrategyResponse:
    row = mgr.get_strategy(strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return StrategyResponse(**row)


@router.post("", response_model=StrategyResponse, status_code=201)
async def create_strategy(body: StrategyCreate, mgr: StateMgr) -> StrategyResponse:
    strategy_id = mgr.create_strategy(
        name=body.name,
        dsl_config=body.dsl_config,
        description=body.description,
        strategy_type=body.strategy_type,
    )
    row = mgr.get_strategy(strategy_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Failed to create strategy")
    return StrategyResponse(**row)


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: int,
    body: StrategyUpdate,
    mgr: StateMgr,
) -> StrategyResponse:
    existing = mgr.get_strategy(strategy_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    mgr.update_strategy(
        strategy_id,
        dsl_config=body.dsl_config,
        description=body.description,
        is_active=body.is_active,
    )
    row = mgr.get_strategy(strategy_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Strategy disappeared after update")
    return StrategyResponse(**row)


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(strategy_id: int, mgr: StateMgr) -> Response:
    existing = mgr.get_strategy(strategy_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    mgr.update_strategy(strategy_id, is_active=False)
    return Response(status_code=204)


@router.post("/{strategy_id}/validate", response_model=ValidationResult)
async def validate_strategy(strategy_id: int, mgr: StateMgr) -> ValidationResult:
    row = mgr.get_strategy(strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    from vibe_quant.dsl.parser import DSLValidationError, validate_strategy_dict
    from vibe_quant.dsl.translator import translate_dsl_config

    try:
        translated = translate_dsl_config(
            row["dsl_config"], strategy_name=str(row["name"])
        )
        validate_strategy_dict(translated)
        return ValidationResult(valid=True, errors=[])
    except DSLValidationError as exc:
        return ValidationResult(valid=False, errors=exc.details if exc.details else [str(exc)])
    except Exception as exc:
        return ValidationResult(valid=False, errors=[str(exc)])
