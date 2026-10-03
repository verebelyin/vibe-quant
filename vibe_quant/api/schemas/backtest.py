"""Backtest domain schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from vibe_quant.api.schemas._validators import KNOWN_TIMEFRAMES, check_date_range, clean_symbols


def _positive_number(params: dict[str, object], key: str, upper: float | None = None) -> None:
    if key not in params or params[key] is None:
        return
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameters.{key} must be a number, got {value!r}")
    if value <= 0 or (upper is not None and value > upper):
        bound = f"in (0, {upper:g}]" if upper is not None else "> 0"
        raise ValueError(f"parameters.{key} must be {bound}, got {value!r}")


class BacktestLaunchRequest(BaseModel):
    # Unknown fields are rejected: the API used to accept sizing_config_id /
    # risk_config_id / overfitting_filters and silently never apply them.
    model_config = ConfigDict(extra="forbid")

    strategy_id: int
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str
    parameters: dict[str, object]
    latency_preset: str | None = None

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        return clean_symbols(v)

    @field_validator("timeframe")
    @classmethod
    def _timeframe(cls, v: str) -> str:
        if v not in KNOWN_TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {sorted(KNOWN_TIMEFRAMES)}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _ranges(self) -> BacktestLaunchRequest:
        check_date_range(self.start_date, self.end_date, allow_equal=False)
        # Runners fall back to 10x / 1000 USDT on non-positive values — reject instead.
        _positive_number(self.parameters, "leverage", upper=125)
        _positive_number(self.parameters, "initial_balance")
        return self


class BacktestRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: int | None = None
    run_mode: str
    symbols: list[str]
    timeframe: str
    start_date: str | None = None
    end_date: str | None = None
    parameters: dict[str, object] | None = None
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
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

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        return clean_symbols(v)

    @model_validator(mode="after")
    def _ranges(self) -> CoverageCheckRequest:
        check_date_range(self.start_date, self.end_date, allow_equal=True)
        return self


class CoverageCheckResponse(BaseModel):
    coverage: dict[str, object]
