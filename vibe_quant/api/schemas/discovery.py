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
    start_date: str | None = None
    end_date: str | None = None


class DiscoveryJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    status: str
    started_at: str | None
    progress: dict[str, object] | None = None


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
