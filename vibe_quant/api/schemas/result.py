"""Result domain schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from vibe_quant.api.schemas.backtest import BacktestRunResponse  # noqa: TCH001
from vibe_quant.bankroll.kelly import DEFAULT_TARGET_VOL
from vibe_quant.bankroll.report import (
    DEFAULT_DD_LIMIT,
    DEFAULT_MULTIPLES,
    DEFAULT_USER_CAP,
    default_workers,
)
from vibe_quant.metrics import SHARPE_DISPLAY_DESCRIPTION, sharpe_to_display


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
    # Machine JSON written by the pipelines — never shown/edited as user notes.
    notes: str | None
    created_at: str | None
    user_notes: str | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    bootstrap_sharpe_lower: float | None = None
    bootstrap_sharpe_upper: float | None = None
    bootstrap_ci_level: float | None = None
    bootstrap_min_sharpe: float | None = None
    wfa_sharpe_consistency: float | None = None
    wfa_consistency: float | None = None
    wfa_passed: bool | None = None
    cross_window_passed: bool | None = None
    cross_window_results: list[dict[str, object]] | None = None
    cross_regime_results: list[dict[str, object]] | None = None
    random_short_baseline_pct: float | None = None
    # The effective cross-window Sharpe limit the run was gated with, in the STORED
    # (√252) unit ``cross_window_results[*].passed`` was computed from (R8).
    cross_window_min_sharpe: float | None = None

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def sharpe_display(self) -> float | None:
        return sharpe_to_display(self.sharpe_ratio)

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def sortino_display(self) -> float | None:
        return sharpe_to_display(self.sortino_ratio)

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def purged_kfold_mean_sharpe_display(self) -> float | None:
        return sharpe_to_display(self.purged_kfold_mean_sharpe)

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def cross_window_min_sharpe_display(self) -> float | None:
        return sharpe_to_display(self.cross_window_min_sharpe)


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


class ExitTimingSummary(BaseModel):
    """Refined-exit counts of a mode-on screening sweep row (bead vibe-quant-3r9wj)."""

    refined: int
    kept_bar_close: int
    by_symbol: dict[str, dict[str, int]]


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
    # Opt-in liquidation summary, written by the screening sweep only when the
    # run's liquidation mode is on; NULL on mode-off and older rows (D1).
    liquidation_count: int | None = None
    liquidation_forfeited: float | None = None
    # Refined-exit counts, written by the screening sweep only when the run's
    # liquidation mode is on; NULL on mode-off and older rows (D1b).
    exit_timing: ExitTimingSummary | None = None

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def sharpe_display(self) -> float | None:
        return sharpe_to_display(self.sharpe_ratio)

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def sortino_display(self) -> float | None:
        return sharpe_to_display(self.sortino_ratio)


class RunSummaryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    strategy_id: int | None = None
    strategy_name: str | None = None
    run_mode: str
    symbols: list[str]
    timeframe: str
    status: str
    total_return: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    total_trades: int | None = None
    winning_trades: int | None = None
    losing_trades: int | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    created_at: str | None = None
    completed_at: str | None = None

    @computed_field(description=SHARPE_DISPLAY_DESCRIPTION)  # type: ignore[prop-decorator]
    @property
    def sharpe_display(self) -> float | None:
        return sharpe_to_display(self.sharpe_ratio)


class RunSummaryResponse(BaseModel):
    runs: list[RunSummaryItem]


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
    """User free-text notes (stored in ``backtest_results.user_notes``)."""

    notes: str


class FrozenDslResponse(BaseModel):
    """The DSL a completed validation run executed (shadow enrolment, vibe-quant-clmmc.27).

    ``dsl_json`` is the canonical (translated) DSL carrying the strategy's own name, so a
    re-translation under another name keeps it; it hashes to ``dsl_sha256``, the value the
    run recorded in ``notes.validation_input``.
    """

    run_id: int
    strategy_id: int
    strategy_name: str
    dsl_json: str
    dsl_sha256: str


# --- Bankroll sizing report (bead vibe-quant-2gyk1.3) ---


class BankrollReportRequest(BaseModel):
    """Launch a bankroll sizing report on a completed validation run.

    ``strict`` keeps a bool or a numeric string out of every number, and non-finite
    floats are refused. ``multiples`` are the requested size multiples k of the run's
    validated exposure cap (1.0 is always added by the report).
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    user_cap: float = Field(default=DEFAULT_USER_CAP, gt=0, le=10)
    dd_limit: float = Field(default=DEFAULT_DD_LIMIT, gt=0, lt=1)
    target_vol: float = Field(default=DEFAULT_TARGET_VOL, gt=0, le=5)
    multiples: list[float] = Field(
        default_factory=lambda: list(DEFAULT_MULTIPLES), min_length=1, max_length=9
    )
    # None = the parent's own liquidation mode when it is on, otherwise cross.
    liquidation_mode: Literal["isolated", "cross"] | None = None
    workers: int = Field(default_factory=default_workers, ge=1, le=4)

    @field_validator("multiples")
    @classmethod
    def _multiples_domain(cls, value: list[float]) -> list[float]:
        if any(not 0 < k <= 10 for k in value):
            raise ValueError("every multiple must be in (0, 10]")
        if len(set(value)) != len(value):
            raise ValueError("multiples must be unique")
        return value


class BankrollReportLaunchResponse(BaseModel):
    report_id: int
    job_id: int
    # Upper estimate from the parent's own run time; None when the parent has none.
    estimated_seconds: int | None


class BankrollReportResponse(BaseModel):
    """The latest bankroll report of a run, read in one snapshot with its freshness.

    ``report`` is the published report JSON (None until completed). A ``stale``
    report's ``recommendation`` is masked to None; ``stale_reasons`` say why.
    """

    id: int
    run_id: int
    job_id: int
    status: Literal["running", "completed", "failed", "stale_parent"]
    error_message: str | None
    created_at: str
    completed_at: str | None
    request: dict[str, object]
    report: dict[str, object] | None
    freshness: Literal["fresh", "stale"]
    stale_reasons: list[str]
