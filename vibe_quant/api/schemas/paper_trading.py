"""Paper trading domain schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PaperStartRequest(BaseModel):
    strategy_id: int
    testnet: bool = False
    trader_id: str | None = None
    sizing_method: str | None = None
    max_leverage: float | None = None
    max_position_pct: float | None = None
    risk_per_trade: float | None = None
    max_drawdown_pct: float | None = None
    max_daily_loss_pct: float | None = None
    max_consecutive_losses: int | None = None
    max_position_count: int | None = None


class PaperStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    state: str
    pnl_metrics: dict[str, object] | None = None
    trades_count: int


class PaperPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    direction: str
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


class CheckpointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: str
    state: str
    halt_reason: str | None = None
    error_message: str | None = None
