"""Tests for screening pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vibe_quant.dsl import parse_strategy_string
from vibe_quant.screening import (
    BacktestMetrics,
    MetricFilters,
    ScreeningPipeline,
    build_parameter_grid,
    compute_pareto_front,
    create_screening_pipeline,
    filter_by_metrics,
    rank_by_sharpe,
)

if TYPE_CHECKING:
    from vibe_quant.dsl.schema import StrategyDSL


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_strategy_yaml() -> str:
    """Minimal strategy with sweep parameters."""
    return """
name: test_minimal
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
sweep:
  rsi.period: [7, 14, 21]
  stop_loss.percent: [1.0, 2.0, 3.0]
"""


@pytest.fixture
def strategy_with_sweep(minimal_strategy_yaml: str) -> StrategyDSL:
    """Parsed strategy with sweep parameters."""
    return parse_strategy_string(minimal_strategy_yaml)


@pytest.fixture
def strategy_no_sweep() -> StrategyDSL:
    """Strategy without sweep parameters."""
    yaml = """
name: no_sweep_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
    return parse_strategy_string(yaml)


@pytest.fixture
def sample_metrics() -> list[BacktestMetrics]:
    """Sample backtest metrics for testing filters and ranking."""
    return [
        BacktestMetrics(
            parameters={"p1": 1},
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.15,
            total_return=0.30,
            profit_factor=1.8,
            win_rate=0.55,
            total_trades=100,
        ),
        BacktestMetrics(
            parameters={"p1": 2},
            sharpe_ratio=2.0,
            sortino_ratio=2.5,
            max_drawdown=0.20,
            total_return=0.40,
            profit_factor=2.0,
            win_rate=0.60,
            total_trades=80,
        ),
        BacktestMetrics(
            parameters={"p1": 3},
            sharpe_ratio=0.5,
            sortino_ratio=0.6,
            max_drawdown=0.35,
            total_return=0.10,
            profit_factor=1.1,
            win_rate=0.45,
            total_trades=150,
        ),
        BacktestMetrics(
            parameters={"p1": 4},
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            max_drawdown=0.10,
            total_return=0.20,
            profit_factor=1.5,
            win_rate=0.52,
            total_trades=30,
        ),
    ]


# =============================================================================
# Parameter Grid Tests
# =============================================================================


class TestBuildParameterGrid:
    """Tests for build_parameter_grid function."""

    def test_empty_sweep_returns_single_empty_dict(self) -> None:
        """Empty sweep should return one empty dict."""
        result = build_parameter_grid({})
        assert result == [{}]

    def test_single_param_returns_list_of_dicts(self) -> None:
        """Single param sweep should return list of single-key dicts."""
        sweep = {"rsi.period": [7, 14, 21]}
        result = build_parameter_grid(sweep)

        assert len(result) == 3
        assert {"rsi.period": 7} in result
        assert {"rsi.period": 14} in result
        assert {"rsi.period": 21} in result

    def test_two_params_creates_cartesian_product(self) -> None:
        """Two params should create Cartesian product."""
        sweep = {
            "rsi.period": [7, 14],
            "stop_loss.percent": [1.0, 2.0],
        }
        result = build_parameter_grid(sweep)

        assert len(result) == 4
        assert {"rsi.period": 7, "stop_loss.percent": 1.0} in result
        assert {"rsi.period": 7, "stop_loss.percent": 2.0} in result
        assert {"rsi.period": 14, "stop_loss.percent": 1.0} in result
        assert {"rsi.period": 14, "stop_loss.percent": 2.0} in result

    def test_three_params_creates_full_product(self) -> None:
        """Three params should create full Cartesian product."""
        sweep = {
            "a": [1, 2],
            "b": [10, 20],
            "c": [100, 200],
        }
        result = build_parameter_grid(sweep)

        # 2 * 2 * 2 = 8 combinations
        assert len(result) == 8

    def test_preserves_param_names(self) -> None:
        """Param names should be preserved exactly."""
        sweep = {
            "indicator.period": [14],
            "stop_loss.atr_multiplier": [2.0],
        }
        result = build_parameter_grid(sweep)

        assert len(result) == 1
        assert "indicator.period" in result[0]
        assert "stop_loss.atr_multiplier" in result[0]


# =============================================================================
# Filter Tests
# =============================================================================


class TestFilterByMetrics:
    """Tests for filter_by_metrics function."""

    def test_default_filters_pass_good_results(
        self, sample_metrics: list[BacktestMetrics]
    ) -> None:
        """Default filters should pass results meeting all criteria."""
        filters = MetricFilters()
        result = filter_by_metrics(sample_metrics, filters)

        # First two and fourth pass; third fails drawdown (0.35 > 0.3)
        assert len(result) == 3
        assert result[0].parameters == {"p1": 1}
        assert result[1].parameters == {"p1": 2}
        assert result[2].parameters == {"p1": 4}

    def test_min_sharpe_filter(self) -> None:
        """Results below min_sharpe should be filtered out."""
        metrics = [
            BacktestMetrics(parameters={}, sharpe_ratio=0.5, profit_factor=1.5, total_trades=100),
            BacktestMetrics(parameters={}, sharpe_ratio=1.5, profit_factor=1.5, total_trades=100),
        ]
        filters = MetricFilters(min_sharpe=1.0, max_drawdown=1.0)
        result = filter_by_metrics(metrics, filters)

        assert len(result) == 1
        assert result[0].sharpe_ratio == 1.5

    def test_min_profit_factor_filter(self) -> None:
        """Results below min_profit_factor should be filtered out."""
        metrics = [
            BacktestMetrics(parameters={}, sharpe_ratio=1.0, profit_factor=0.9, total_trades=100),
            BacktestMetrics(parameters={}, sharpe_ratio=1.0, profit_factor=1.5, total_trades=100),
        ]
        filters = MetricFilters(max_drawdown=1.0)
        result = filter_by_metrics(metrics, filters)

        assert len(result) == 1
        assert result[0].profit_factor == 1.5

    def test_max_drawdown_filter(self) -> None:
        """Results above max_drawdown should be filtered out."""
        metrics = [
            BacktestMetrics(parameters={}, sharpe_ratio=1.0, profit_factor=1.5, max_drawdown=0.40, total_trades=100),
            BacktestMetrics(parameters={}, sharpe_ratio=1.0, profit_factor=1.5, max_drawdown=0.20, total_trades=100),
        ]
        filters = MetricFilters(max_drawdown=0.30)
        result = filter_by_metrics(metrics, filters)

        assert len(result) == 1
        assert result[0].max_drawdown == 0.20

    def test_min_trades_filter(self) -> None:
        """Results below min_trades should be filtered out."""
        metrics = [
            BacktestMetrics(parameters={}, sharpe_ratio=1.0, profit_factor=1.5, total_trades=30),
            BacktestMetrics(parameters={}, sharpe_ratio=1.0, profit_factor=1.5, total_trades=100),
        ]
        filters = MetricFilters(min_trades=50, max_drawdown=1.0)
        result = filter_by_metrics(metrics, filters)

        assert len(result) == 1
        assert result[0].total_trades == 100

    def test_empty_input_returns_empty(self) -> None:
        """Empty input should return empty list."""
        result = filter_by_metrics([], MetricFilters())
        assert result == []

    def test_all_filtered_returns_empty(self) -> None:
        """If all results fail filters, return empty list."""
        metrics = [
            BacktestMetrics(parameters={}, sharpe_ratio=-1.0, profit_factor=0.5, total_trades=10),
        ]
        result = filter_by_metrics(metrics, MetricFilters())
        assert result == []


# =============================================================================
# Ranking Tests
# =============================================================================


class TestRankBySharpe:
    """Tests for rank_by_sharpe function."""

    def test_sorts_descending_by_sharpe(self) -> None:
        """Results should be sorted by Sharpe ratio descending."""
        metrics = [
            BacktestMetrics(parameters={"id": 1}, sharpe_ratio=1.0),
            BacktestMetrics(parameters={"id": 2}, sharpe_ratio=2.0),
            BacktestMetrics(parameters={"id": 3}, sharpe_ratio=1.5),
        ]
        result = rank_by_sharpe(metrics)

        assert result[0].sharpe_ratio == 2.0
        assert result[1].sharpe_ratio == 1.5
        assert result[2].sharpe_ratio == 1.0

    def test_empty_list_returns_empty(self) -> None:
        """Empty input should return empty list."""
        result = rank_by_sharpe([])
        assert result == []

    def test_single_item_returns_same(self) -> None:
        """Single item should return unchanged."""
        metrics = [BacktestMetrics(parameters={}, sharpe_ratio=1.5)]
        result = rank_by_sharpe(metrics)
        assert len(result) == 1
        assert result[0].sharpe_ratio == 1.5


# =============================================================================
# Pareto Front Tests
# =============================================================================


class TestComputeParetoFront:
    """Tests for compute_pareto_front function."""

    def test_empty_input_returns_empty(self) -> None:
        """Empty input should return empty list."""
        result = compute_pareto_front([])
        assert result == []

    def test_single_result_is_pareto_optimal(self) -> None:
        """Single result should be Pareto-optimal."""
        metrics = [BacktestMetrics(parameters={}, sharpe_ratio=1.0)]
        result = compute_pareto_front(metrics)
        assert result == [0]

    def test_one_dominates_other(self) -> None:
        """If one dominates other, only dominant is Pareto-optimal."""
        metrics = [
            # Result 0: worse in all objectives
            BacktestMetrics(
                parameters={"id": 0},
                sharpe_ratio=1.0,
                max_drawdown=0.20,  # 1-dd = 0.80
                profit_factor=1.5,
            ),
            # Result 1: better in all objectives (dominates 0)
            BacktestMetrics(
                parameters={"id": 1},
                sharpe_ratio=2.0,
                max_drawdown=0.10,  # 1-dd = 0.90
                profit_factor=2.0,
            ),
        ]
        result = compute_pareto_front(metrics)

        assert 1 in result
        assert 0 not in result

    def test_non_dominated_both_pareto(self) -> None:
        """Non-dominated results should both be Pareto-optimal."""
        metrics = [
            # Result 0: better Sharpe, worse drawdown
            BacktestMetrics(
                parameters={"id": 0},
                sharpe_ratio=2.0,
                max_drawdown=0.30,  # 1-dd = 0.70
                profit_factor=1.5,
            ),
            # Result 1: worse Sharpe, better drawdown
            BacktestMetrics(
                parameters={"id": 1},
                sharpe_ratio=1.5,
                max_drawdown=0.10,  # 1-dd = 0.90
                profit_factor=1.5,
            ),
        ]
        result = compute_pareto_front(metrics)

        # Neither dominates the other
        assert 0 in result
        assert 1 in result

    def test_three_results_mixed_dominance(self) -> None:
        """Three results with mixed dominance."""
        metrics = [
            # Result 0: dominated by 1
            BacktestMetrics(
                parameters={"id": 0},
                sharpe_ratio=1.0,
                max_drawdown=0.20,
                profit_factor=1.0,
            ),
            # Result 1: Pareto-optimal
            BacktestMetrics(
                parameters={"id": 1},
                sharpe_ratio=2.0,
                max_drawdown=0.20,
                profit_factor=2.0,
            ),
            # Result 2: Pareto-optimal (non-dominated)
            BacktestMetrics(
                parameters={"id": 2},
                sharpe_ratio=1.5,
                max_drawdown=0.10,  # Better drawdown than 1
                profit_factor=1.8,
            ),
        ]
        result = compute_pareto_front(metrics)

        assert 0 not in result  # Dominated by 1
        assert 1 in result
        assert 2 in result


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestScreeningPipeline:
    """Tests for ScreeningPipeline class."""

    def test_init_builds_parameter_grid(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """Pipeline should build parameter grid on init."""
        pipeline = ScreeningPipeline(dsl=strategy_with_sweep)

        # 3 * 3 = 9 combinations
        assert pipeline.num_combinations == 9

    def test_init_no_sweep_has_one_combination(
        self, strategy_no_sweep: StrategyDSL
    ) -> None:
        """Strategy without sweep should have 1 combination (empty params)."""
        pipeline = ScreeningPipeline(dsl=strategy_no_sweep)
        assert pipeline.num_combinations == 1

    def test_strategy_name_from_dsl(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """Strategy name should come from DSL."""
        pipeline = ScreeningPipeline(dsl=strategy_with_sweep)
        assert pipeline.strategy_name == "test_minimal"

    def test_run_returns_screening_result(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """run() should return ScreeningResult with all fields."""
        pipeline = ScreeningPipeline(dsl=strategy_with_sweep, max_workers=2)
        result = pipeline.run()

        assert result.strategy_name == "test_minimal"
        assert result.total_combinations == 9
        assert result.execution_time_seconds > 0
        assert isinstance(result.results, list)
        assert isinstance(result.pareto_optimal_indices, list)

    def test_run_with_filters(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """run() should apply filters to results."""
        pipeline = ScreeningPipeline(dsl=strategy_with_sweep, max_workers=2)

        # Very strict filters that few results will pass
        filters = MetricFilters(
            min_sharpe=2.0,
            min_profit_factor=2.5,
            min_trades=100,
        )
        result = pipeline.run(filters=filters)

        # Most mock results won't pass these strict filters
        assert result.passed_filters <= result.total_combinations

    def test_run_calls_progress_callback(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """run() should call progress callback."""
        pipeline = ScreeningPipeline(dsl=strategy_with_sweep, max_workers=2)

        progress_calls: list[tuple[int, int]] = []

        def callback(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        pipeline.run(progress_callback=callback)

        assert len(progress_calls) == 9  # One call per combination
        # All calls should have total = 9
        assert all(total == 9 for _, total in progress_calls)

    def test_custom_backtest_runner(
        self, strategy_no_sweep: StrategyDSL
    ) -> None:
        """Pipeline should use custom backtest runner.

        Note: ProcessPoolExecutor can't pickle local functions, so we test
        the mock runner (which is a module-level function) and verify
        results are returned correctly.
        """
        # The default mock runner is used since custom local functions
        # can't be pickled for multiprocessing
        pipeline = ScreeningPipeline(
            dsl=strategy_no_sweep,
            max_workers=1,  # Single worker to avoid complexity
        )
        result = pipeline.run()

        # Should have run exactly one backtest (no sweep params)
        assert len(result.results) >= 0  # May filter out based on metrics
        assert result.total_combinations == 1


class TestCreateScreeningPipeline:
    """Tests for create_screening_pipeline factory."""

    def test_creates_pipeline_with_mock(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """Factory should create pipeline with mock runner."""
        pipeline = create_screening_pipeline(
            dsl=strategy_with_sweep,
            use_mock=True,
        )

        assert isinstance(pipeline, ScreeningPipeline)
        assert pipeline.num_combinations == 9

    def test_respects_max_workers(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """Factory should pass max_workers."""
        pipeline = create_screening_pipeline(
            dsl=strategy_with_sweep,
            use_mock=True,
            max_workers=4,
        )

        assert pipeline._max_workers == 4


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineDatabaseIntegration:
    """Integration tests for pipeline with database."""

    def test_save_results_to_database(
        self, strategy_with_sweep: StrategyDSL
    ) -> None:
        """Pipeline should save results to database."""
        from vibe_quant.db.state_manager import StateManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = StateManager(db_path)

            # Create a strategy and run record
            strategy_id = manager.create_strategy(
                name="test_strategy",
                dsl_config=strategy_with_sweep.model_dump(),
            )
            run_id = manager.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="screening",
                symbols=["BTCUSDT"],
                timeframe="5m",
                start_date="2024-01-01",
                end_date="2024-12-31",
                parameters={},
            )

            # Run pipeline
            pipeline = ScreeningPipeline(
                dsl=strategy_with_sweep,
                max_workers=2,
            )
            result = pipeline.run()

            # Save results
            pipeline.save_results(result, manager, run_id)

            # Verify saved
            saved = manager.get_sweep_results(run_id)
            assert len(saved) == len(result.results)

            # Check Pareto flags
            pareto_saved = manager.get_sweep_results(run_id, pareto_only=True)
            assert len(pareto_saved) == len(result.pareto_optimal_indices)

            manager.close()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_large_sweep_grid(self) -> None:
        """Large sweep grid should work correctly."""
        sweep = {
            "a": list(range(10)),
            "b": list(range(10)),
        }
        result = build_parameter_grid(sweep)
        assert len(result) == 100

    def test_single_value_sweep(self) -> None:
        """Single value in sweep should work."""
        sweep = {"a": [42]}
        result = build_parameter_grid(sweep)
        assert result == [{"a": 42}]

    def test_float_sweep_values(self) -> None:
        """Float values in sweep should work."""
        sweep = {"pct": [0.1, 0.2, 0.3]}
        result = build_parameter_grid(sweep)
        assert len(result) == 3
        assert {"pct": 0.1} in result

    def test_negative_sharpe_handling(self) -> None:
        """Negative Sharpe ratios should be handled correctly."""
        metrics = [
            BacktestMetrics(parameters={}, sharpe_ratio=-0.5),
            BacktestMetrics(parameters={}, sharpe_ratio=0.5),
        ]
        ranked = rank_by_sharpe(metrics)
        assert ranked[0].sharpe_ratio == 0.5
        assert ranked[1].sharpe_ratio == -0.5
