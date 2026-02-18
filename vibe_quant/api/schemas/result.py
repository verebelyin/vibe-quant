"""Result domain schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vibe_quant.api.schemas.backtest import BacktestRunResponse  # noqa: TCH001


class BacktestResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    total_return: float | None
    cagr: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    max_drawdown: float | None
    max_drawdown_duration_days: int | None
    volatility_annual: float | None
    total_trades: int | None
    winning_trades: int | None
    losing_trades: int | None
    win_rate: float | None
    profit_factor: float | None
    avg_win: float | None
    avg_loss: float | None
    largest_win: float | None
    largest_loss: float | None
    avg_trade_duration_hours: float | None
    max_consecutive_wins: int | None
    max_consecutive_losses: int | None
    total_fees: float | None
    total_funding: float | None
    total_slippage: float | None
    deflated_sharpe: float | None
    walk_forward_efficiency: float | None
    purged_kfold_mean_sharpe: float | None
    execution_time_seconds: float | None
    starting_balance: float | None
    notes: str | None
    created_at: str | None


class TradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int | None
    symbol: str
    direction: str
    leverage: int
    entry_time: str
    exit_time: str | None
    entry_price: float
    exit_price: float | None
    quantity: float
    entry_fee: float | None
    exit_fee: float | None
    funding_fees: float | None
    slippage_cost: float | None
    gross_pnl: float | None
    net_pnl: float | None
    roi_percent: float | None
    exit_reason: str | None


class SweepResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int | None
    parameters: dict[str, object]
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown: float | None
    total_return: float | None
    profit_factor: float | None
    win_rate: float | None
    total_trades: int | None
    total_fees: float | None
    total_funding: float | None
    execution_time_seconds: float | None
    is_pareto_optimal: bool
    passed_deflated_sharpe: bool | None
    passed_walk_forward: bool | None
    passed_purged_kfold: bool | None


class RunListResponse(BaseModel):
    runs: list[BacktestRunResponse]


class EquityCurvePoint(BaseModel):
    timestamp: str
    equity: float


class DrawdownPoint(BaseModel):
    timestamp: str
    drawdown: float


class MonthlyReturn(BaseModel):
    year: int
    month: int
    return_pct: float


class ComparisonResponse(BaseModel):
    runs: list[BacktestResultResponse]


class NotesUpdateRequest(BaseModel):
    notes: str
