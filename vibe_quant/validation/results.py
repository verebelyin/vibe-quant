"""Validation result data classes.

Dataclasses for validation backtest results and individual trade records.
These are used by :class:`~vibe_quant.validation.runner.ValidationRunner`
and consumed by dashboard pages, the CLI, and downstream pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vibe_quant.metrics import PerformanceMetrics


@dataclass
class TradeRecord:
    """Individual trade record for storage.

    Attributes:
        symbol: Instrument symbol.
        direction: 'LONG' or 'SHORT'.
        leverage: Leverage used.
        entry_time: Entry timestamp ISO format.
        exit_time: Exit timestamp ISO format.
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Trade quantity.
        entry_fee: Entry transaction fee.
        exit_fee: Exit transaction fee.
        funding_fees: Total funding paid/received.
        slippage_cost: Slippage cost.
        gross_pnl: PnL before fees.
        net_pnl: PnL after fees.
        roi_percent: Return on investment percent.
        exit_reason: Why position was closed.
    """

    symbol: str
    direction: str
    leverage: int
    entry_time: str
    exit_time: str | None
    entry_price: float
    exit_price: float | None
    quantity: float
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    funding_fees: float = 0.0
    slippage_cost: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    roi_percent: float = 0.0
    exit_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for database storage."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "leverage": self.leverage,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "funding_fees": self.funding_fees,
            "slippage_cost": self.slippage_cost,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "roi_percent": self.roi_percent,
            "exit_reason": self.exit_reason,
        }


@dataclass
class ValidationResult(PerformanceMetrics):
    """Results from a validation backtest run.

    Extends :class:`~vibe_quant.metrics.PerformanceMetrics` with
    validation-specific fields (trade detail, extended analytics).
    """

    run_id: int = 0
    strategy_name: str = ""
    cagr: float = 0.0
    calmar_ratio: float = 0.0
    volatility_annual: float = 0.0
    max_drawdown_duration_days: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration_hours: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_slippage: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)

    starting_balance: float = 100000.0

    def to_metrics_dict(self) -> dict[str, object]:
        """Convert to metrics dictionary for database storage."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "cagr": self.cagr,
            "calmar_ratio": self.calmar_ratio,
            "volatility_annual": self.volatility_annual,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "total_slippage": self.total_slippage,
            "execution_time_seconds": self.execution_time_seconds,
            "starting_balance": self.starting_balance,
        }
