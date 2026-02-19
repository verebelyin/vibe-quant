"""Discovery domain schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DiscoveryLaunchRequest(BaseModel):
    population: int
    generations: int
    mutation_rate: float
    crossover_rate: float = 0.8
    elite_count: int = 2
    tournament_size: int = 3
    convergence_generations: int = 10
    symbols: list[str]
    timeframes: list[str]
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
