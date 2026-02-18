"""Backtest domain schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BacktestLaunchRequest(BaseModel):
    strategy_id: int
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str
    parameters: dict[str, object]
    sizing_config_id: int | None = None
    risk_config_id: int | None = None
    latency_preset: str | None = None
    overfitting_filters: dict[str, bool] | None = None


class BacktestRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: int
    run_mode: str
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str
    parameters: dict[str, object]
    status: str
    started_at: str | None
    completed_at: str | None
    error_message: str | None
    created_at: str


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    pid: int | None
    job_type: str
    status: str
    heartbeat_at: str | None
    started_at: str | None
    completed_at: str | None
    is_stale: bool


class HeartbeatRequest(BaseModel):
    run_id: int


class TradesBatchRequest(BaseModel):
    trades: list[dict[str, object]]


class SweepResultsBatchRequest(BaseModel):
    results: list[dict[str, object]]


class ParetoMarkRequest(BaseModel):
    result_ids: list[int]


class CoverageCheckRequest(BaseModel):
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str


class CoverageCheckResponse(BaseModel):
    coverage: dict[str, object]
