"""Screening pipeline with parallel parameter sweeps."""

from vibe_quant.screening.consistency import (
    ConsistencyChecker,
    ConsistencyResult,
    check_consistency,
)
from vibe_quant.screening.grid import (
    build_parameter_grid,
    compute_pareto_front,
    filter_by_metrics,
    rank_by_sharpe,
)
from vibe_quant.screening.pipeline import (
    ScreeningPipeline,
    create_screening_pipeline,
)
from vibe_quant.screening.types import (
    BacktestMetrics,
    BacktestRunner,
    MetricFilters,
    ScreeningResult,
)

__all__ = [
    "BacktestMetrics",
    "BacktestRunner",
    "MetricFilters",
    "ScreeningPipeline",
    "ScreeningResult",
    "build_parameter_grid",
    "compute_pareto_front",
    "create_screening_pipeline",
    "filter_by_metrics",
    "rank_by_sharpe",
    # Consistency
    "ConsistencyChecker",
    "ConsistencyResult",
    "check_consistency",
]
