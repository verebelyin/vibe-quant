"""Strategy domain schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrategyCreate(BaseModel):
    name: str
    dsl_config: dict[str, object]
    description: str | None = None
    strategy_type: str | None = None


class StrategyUpdate(BaseModel):
    dsl_config: dict[str, object] | None = None
    description: str | None = None
    is_active: bool | None = None


class StrategyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    dsl_config: dict[str, object]
    strategy_type: str | None
    created_at: str
    updated_at: str
    is_active: bool
    version: int


class StrategyListResponse(BaseModel):
    strategies: list[StrategyResponse]


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]
