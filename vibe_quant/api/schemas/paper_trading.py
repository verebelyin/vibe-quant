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
    confirm_fresh_live: bool = Field(
        default=False,
        strict=True,  # "true" / 1 / null are 422, not coerced: only a JSON bool flips it
        description="Must be true together with testnet=false AND start_fresh=true: a live "
        "start fresh resets the real-money drawdown, daily-loss and losing-streak limits.",
    )
    trader_id: str | None = None
    start_fresh: bool = Field(
        default=False,
        strict=True,  # "true" / 1 / null are 422, not coerced: only a JSON bool flips it
        description="Skip inheriting the last session's peak, daily-loss and losing-streak "
        "limits for this account (default: continue from that session's risk state).",
    )
    trade_alerts: bool = Field(
        default=False,
        strict=True,  # "true" / 1 / null are 422, not coerced: only a JSON bool flips it
        description="Message on every entry and exit (needs Telegram configured).",
    )
    sizing_method: Literal["fixed_fractional"] | None = None
    max_leverage: float | None = Field(default=None, ge=1, le=125)
    max_position_pct: float | None = Field(default=None, gt=0, le=125)
    max_margin_pct: float | None = Field(
        default=None,
        gt=0,
        le=1,
        strict=True,  # a boolean is not a number (lax mode turns true into 1.0 = 100%)
        description="Margin cushion: refuse entries beyond this fraction of equity "
        "(default: the validation run's value, else 0.5).",
    )
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
    snapshot_errors: list[str] = Field(default_factory=list)


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
