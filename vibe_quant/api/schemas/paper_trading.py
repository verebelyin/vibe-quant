"""Paper trading domain schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperStartRequest(BaseModel):
    """Start a paper session.

    Prefer ``validation_run_id``: the session then trades that run's exact
    strategy parameters, symbols and leverage. Without it, ``strategy_id`` and
    ``symbols`` are required and the compiled DSL defaults are traded.

    Fractions, not percent: ``risk_per_trade=0.02`` is 2%. Credentials are
    never accepted here (env only); unknown fields are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    strategy_id: int | None = None
    validation_run_id: int | None = None
    symbols: list[str] | None = None
    testnet: bool = True
    confirm_live: bool = Field(
        default=False,
        description="Must be true together with testnet=false to trade real funds.",
    )
    trader_id: str | None = None
    sizing_method: Literal["fixed_fractional"] | None = None
    max_leverage: float | None = Field(default=None, ge=1, le=125)
    max_position_pct: float | None = Field(default=None, gt=0, le=125)
    risk_per_trade: float | None = Field(default=None, gt=0, le=0.5)
    max_drawdown_pct: float | None = Field(default=None, gt=0, le=1)
    max_daily_loss_pct: float | None = Field(default=None, gt=0, le=1)
    max_consecutive_losses: int | None = Field(default=None, ge=1)
    max_position_count: int | None = Field(default=None, ge=1)


class PaperRestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trader_id: str
    confirm_live: bool = False


class PaperStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    state: str
    pnl_metrics: dict[str, object] | None = None
    trades_count: int
    run_id: int | None = None
    trader_id: str | None = None
    testnet: bool | None = None
    halt_reason: str | None = None
    message: str | None = None


class PaperPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    direction: str
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


class PaperOrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    status: str | None = None


class CheckpointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: str
    state: str
    halt_reason: str | None = None
    error_message: str | None = None
