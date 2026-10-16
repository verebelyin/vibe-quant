"""Backtest domain schemas."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from vibe_quant.api.schemas._validators import KNOWN_TIMEFRAMES, check_date_range, clean_symbols
from vibe_quant.api.schemas.finite import reject_non_finite
from vibe_quant.balance_settings import BalanceConfigError, parse_initial_balance
from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings
from vibe_quant.sizing_settings import SIZING_FIELDS, SizingConfigError, SizingSettings


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
    # Charge the BNB-discounted (x0.9) fee schedule. Stored into the run
    # parameters only when true, so default runs' stored params are unchanged
    # (screening and discovery accept it too, bd vibe-quant-2gyk1.4 D3-2).
    bnb_discount: StrictBool = False
    # Post-only maker entries are validation-only and opt-in: the entry is a
    # limit priced at the outbox release. Screening refuses a non-market
    # entry_order, and the three fields are stored in the run parameters only
    # when entry_order == "post_only" (D3-6).
    entry_order: Literal["market", "post_only"] = "market"
    maker_offset_ticks: int = Field(1, ge=1)
    maker_entry_timeout_bars: int = Field(3, ge=1)
    # Run-level liquidation mode. Stored into the run parameters only when on,
    # so default runs' stored params are unchanged (E1). The run's existing
    # parameters.leverage is the venue/liquidation leverage (vibe-quant-2gyk1.1.10).
    liquidation_mode: Literal["off", "isolated", "cross"] = "off"

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

    @model_validator(mode="before")
    @classmethod
    def _reject_non_finite(cls, data: object) -> object:
        # JSON's bare NaN/Infinity/-Infinity are not JSON values; a 422 naming the
        # path is the only sensible answer (a NaN parameter used to launch at 201,
        # and a NaN leverage made FastAPI's own 422 un-serialisable -> 500).
        reject_non_finite(data)
        return data

    @model_validator(mode="after")
    def _ranges(self) -> BacktestLaunchRequest:
        check_date_range(self.start_date, self.end_date, allow_equal=False)
        # The mode is a top-level field: a nested parameters.liquidation_mode
        # would be silently ignored, so reject it loudly (E4).
        if "liquidation_mode" in self.parameters:
            msg = (
                "parameters.liquidation_mode is not allowed; "
                "use the top-level liquidation_mode field"
            )
            raise ValueError(msg)
        # Runners fall back to 10x / 1000 USDT on non-positive values — reject instead.
        # Leverage goes through the ONE shared rule (whole number in [1, 125]) in EVERY
        # mode (vibe-quant-6wq6w), so the API, CLI and stored params agree.
        if "leverage" in self.parameters:
            try:
                LiquidationSettings(leverage=cast("float", self.parameters["leverage"]))
            except LiquidationConfigError as exc:
                raise ValueError(f"parameters.leverage: {exc}") from exc
        # The starting balance goes through its own ONE shared rule (whole number in
        # [1, 1000000000], vibe-quant-zpbmo): a value the runner would ignore or reject
        # mid-engine is a 422 here, before any run row exists.
        try:
            parse_initial_balance(self.parameters)
        except BalanceConfigError as exc:
            raise ValueError(f"parameters.initial_balance: {exc}") from exc
        # Both sizing settings go through the ONE shared rule
        # (vibe-quant-2gyk1.2): a string, a bool, an explicit null or an
        # out-of-range number used to launch a run that sized at the default,
        # and a cap above the run's margin leverage would be unmarginable.
        # Checked in EVERY mode, before any run row exists; the resolved
        # leverage is the run's own `parameters.leverage` (10.0 by default).
        try:
            SizingSettings.from_params(self.parameters).check_leverage(
                LiquidationSettings.from_params(self.parameters).leverage
            )
        except SizingConfigError as exc:
            raise ValueError(f"parameters: {exc}") from exc
        # `parameters.sweep` is a free dict of strategy parameters: a sizing key
        # there would size every sweep row differently and is not a strategy
        # field at all, so refuse it here as well as at run time (S14).
        sweep = self.parameters.get("sweep")
        if isinstance(sweep, dict):
            for key in SIZING_FIELDS:
                if key in sweep:
                    msg = (
                        f"parameters.sweep: {key} is a run-level sizing setting "
                        f"(parameters.{key}), not a sweep parameter"
                    )
                    raise ValueError(msg)
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
