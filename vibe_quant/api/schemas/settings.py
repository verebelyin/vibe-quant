"""Settings domain schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SizingConfigCreate(BaseModel):
    name: str
    method: str
    config: dict[str, object]


class SizingConfigUpdate(BaseModel):
    name: str | None = None
    method: str | None = None
    config: dict[str, object] | None = None


class SizingConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    method: str
    config: dict[str, object]
    created_at: str


class RiskConfigCreate(BaseModel):
    name: str
    strategy_level: dict[str, object]
    portfolio_level: dict[str, object]


class RiskConfigUpdate(BaseModel):
    name: str | None = None
    strategy_level: dict[str, object] | None = None
    portfolio_level: dict[str, object] | None = None


class RiskConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    strategy_level: dict[str, object]
    portfolio_level: dict[str, object]
    created_at: str


class LatencyPreset(BaseModel):
    name: str
    description: str
    base_latency_ms: int


class SystemInfoResponse(BaseModel):
    nt_version: str
    python_version: str
    catalog_size_bytes: int
    db_size_bytes: int
    table_counts: dict[str, int]


class DatabaseInfoResponse(BaseModel):
    path: str
    tables: list[str]


class DatabaseSwitchRequest(BaseModel):
    path: str
