"""Discovery domain schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DiscoveryLaunchRequest(BaseModel):
    population: int = 20
    generations: int = 15
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_count: int = 2
    tournament_size: int = 3
    convergence_generations: int = 10
    symbols: list[str] = ["BTCUSDT"]
    timeframes: list[str] = ["4h"]
    indicator_pool: list[str] | None = None
    direction: str | None = None  # "long", "short", "both", or None (random)
    start_date: str | None = None
    end_date: str | None = None
    train_test_split: float = 0.0  # 0=disabled, 0.5=50/50 train/holdout split
    cross_window_months: list[int] | None = None  # e.g. [1, 2] for +1mo, +2mo shifted windows
    cross_window_min_sharpe: float = 0.5  # min Sharpe on shifted windows
    num_seeds: int = 1  # >1 enables multi-seed ensemble


class DiscoveryJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    status: str
    started_at: str | None
    completed_at: str | None = None
    progress: dict[str, object] | None = None
    symbols: list[str] | None = None
    timeframe: str | None = None
    generations: int | None = None
    population: int | None = None
    strategies_found: int | None = None
    error_message: str | None = None


class DiscoveryResultResponse(BaseModel):
    strategies: list[dict[str, object]]


class PromoteResponse(BaseModel):
    strategy_id: int
    run_id: int
    name: str
    mode: str


class ReplayResponse(BaseModel):
    replay_run_id: int
    original_run_id: int
