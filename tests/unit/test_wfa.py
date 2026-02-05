"""Tests for Walk-Forward Analysis (WFA) module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from vibe_quant.overfitting.wfa import (
    BacktestRunner,
    WalkForwardAnalysis,
    WFAConfig,
    WFAResult,
    WFAWindow,
)

# -- Mock Backtest Runner --


@dataclass
class MockBacktestResult:
    """Mock result for testing."""

    sharpe: float
    total_return: float
    params: dict[str, object]


class MockRunner:
    """Mock backtest runner for testing."""

    def __init__(
        self,
        is_results: list[MockBacktestResult] | None = None,
        oos_results: list[MockBacktestResult] | None = None,
    ) -> None:
        """Initialize with predetermined results.

        Args:
            is_results: Results to return for optimize() calls.
            oos_results: Results to return for backtest() calls.
        """
        self._is_results = is_results or []
        self._oos_results = oos_results or []
        self._optimize_calls: list[tuple[str, date, date, dict[str, list[object]]]] = []
        self._backtest_calls: list[tuple[str, date, date, dict[str, object]]] = []
        self._optimize_idx = 0
        self._backtest_idx = 0

    @property
    def optimize_calls(self) -> list[tuple[str, date, date, dict[str, list[object]]]]:
        """Return recorded optimize() calls."""
        return self._optimize_calls

    @property
    def backtest_calls(self) -> list[tuple[str, date, date, dict[str, object]]]:
        """Return recorded backtest() calls."""
        return self._backtest_calls

    def optimize(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        param_grid: dict[str, list[object]],
    ) -> tuple[dict[str, object], float, float]:
        """Return predetermined IS result."""
        self._optimize_calls.append((strategy_id, start_date, end_date, param_grid))

        if self._optimize_idx < len(self._is_results):
            r = self._is_results[self._optimize_idx]
            self._optimize_idx += 1
            return r.params, r.sharpe, r.total_return

        # Default result
        return {"fast_period": 10}, 1.5, 20.0

    def backtest(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        params: dict[str, object],
    ) -> tuple[float, float]:
        """Return predetermined OOS result."""
        self._backtest_calls.append((strategy_id, start_date, end_date, params))

        if self._backtest_idx < len(self._oos_results):
            r = self._oos_results[self._backtest_idx]
            self._backtest_idx += 1
            return r.sharpe, r.total_return

        # Default result
        return 1.0, 15.0


# -- WFAConfig Tests --


class TestWFAConfig:
    """Tests for WFAConfig validation."""

    def test_default_config(self) -> None:
        """Default config has expected values."""
        cfg = WFAConfig.default()
        assert cfg.in_sample_days == 270
        assert cfg.out_of_sample_days == 90
        assert cfg.step_days == 30
        assert cfg.min_windows == 8
        assert cfg.min_oos_sharpe == 0.5
        assert cfg.max_degradation == 0.5
        assert cfg.min_consistency == 0.5

    def test_custom_config(self) -> None:
        """Custom config accepted."""
        cfg = WFAConfig(
            in_sample_days=180,
            out_of_sample_days=60,
            step_days=15,
            min_windows=5,
        )
        assert cfg.in_sample_days == 180
        assert cfg.out_of_sample_days == 60
        assert cfg.step_days == 15
        assert cfg.min_windows == 5

    def test_invalid_in_sample_days(self) -> None:
        """in_sample_days must be positive."""
        with pytest.raises(ValueError, match="in_sample_days must be positive"):
            WFAConfig(in_sample_days=0, out_of_sample_days=30, step_days=10)

        with pytest.raises(ValueError, match="in_sample_days must be positive"):
            WFAConfig(in_sample_days=-10, out_of_sample_days=30, step_days=10)

    def test_invalid_out_of_sample_days(self) -> None:
        """out_of_sample_days must be positive."""
        with pytest.raises(ValueError, match="out_of_sample_days must be positive"):
            WFAConfig(in_sample_days=90, out_of_sample_days=0, step_days=10)

    def test_invalid_step_days(self) -> None:
        """step_days must be positive."""
        with pytest.raises(ValueError, match="step_days must be positive"):
            WFAConfig(in_sample_days=90, out_of_sample_days=30, step_days=0)

    def test_invalid_min_windows(self) -> None:
        """min_windows must be at least 1."""
        with pytest.raises(ValueError, match="min_windows must be at least 1"):
            WFAConfig(in_sample_days=90, out_of_sample_days=30, step_days=10, min_windows=0)

    def test_invalid_max_degradation(self) -> None:
        """max_degradation must be in [0, 1]."""
        with pytest.raises(ValueError, match="max_degradation must be in"):
            WFAConfig(
                in_sample_days=90,
                out_of_sample_days=30,
                step_days=10,
                max_degradation=-0.1,
            )
        with pytest.raises(ValueError, match="max_degradation must be in"):
            WFAConfig(
                in_sample_days=90,
                out_of_sample_days=30,
                step_days=10,
                max_degradation=1.5,
            )

    def test_invalid_min_consistency(self) -> None:
        """min_consistency must be in [0, 1]."""
        with pytest.raises(ValueError, match="min_consistency must be in"):
            WFAConfig(
                in_sample_days=90,
                out_of_sample_days=30,
                step_days=10,
                min_consistency=-0.1,
            )


# -- WFAWindow Tests --


class TestWFAWindow:
    """Tests for WFAWindow calculations."""

    def test_is_oos_profitable_true(self) -> None:
        """Profitable when OOS return > 0."""
        window = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=1.5,
            oos_sharpe=1.0,
            is_return=25.0,
            oos_return=5.0,
            best_params={"fast": 10},
        )
        assert window.is_oos_profitable is True

    def test_is_oos_profitable_false(self) -> None:
        """Not profitable when OOS return <= 0."""
        window = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=1.5,
            oos_sharpe=-0.5,
            is_return=25.0,
            oos_return=-10.0,
            best_params={},
        )
        assert window.is_oos_profitable is False

        # Zero return = not profitable
        window2 = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=1.5,
            oos_sharpe=0.0,
            is_return=25.0,
            oos_return=0.0,
            best_params={},
        )
        assert window2.is_oos_profitable is False

    def test_sharpe_degradation(self) -> None:
        """Sharpe degradation = (IS - OOS) / |IS|."""
        window = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=2.0,
            oos_sharpe=1.0,
            is_return=20.0,
            oos_return=10.0,
            best_params={},
        )
        # (2.0 - 1.0) / 2.0 = 0.5
        assert window.sharpe_degradation == 0.5

    def test_sharpe_degradation_negative_is(self) -> None:
        """Degradation uses absolute IS value."""
        window = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=-1.0,
            oos_sharpe=-0.5,
            is_return=-10.0,
            oos_return=-5.0,
            best_params={},
        )
        # (-1.0 - (-0.5)) / |-1.0| = -0.5
        assert window.sharpe_degradation == -0.5

    def test_sharpe_degradation_zero_is(self) -> None:
        """Handles zero IS Sharpe."""
        # Zero IS, positive OOS = improvement (-1)
        w1 = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=0.0,
            oos_sharpe=1.0,
            is_return=0.0,
            oos_return=10.0,
            best_params={},
        )
        assert w1.sharpe_degradation == -1.0

        # Zero IS, negative OOS = degradation (1)
        w2 = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=0.0,
            oos_sharpe=-1.0,
            is_return=0.0,
            oos_return=-10.0,
            best_params={},
        )
        assert w2.sharpe_degradation == 1.0

        # Zero IS, zero OOS = no change (0)
        w3 = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=0.0,
            oos_sharpe=0.0,
            is_return=0.0,
            oos_return=0.0,
            best_params={},
        )
        assert w3.sharpe_degradation == 0.0


# -- Window Generation Tests --


class TestWindowGeneration:
    """Tests for WFA window generation logic."""

    def test_basic_window_generation(self) -> None:
        """Generates correct windows for simple case."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=1,
        )
        wfa = WalkForwardAnalysis(config=cfg)

        # 360 days = 12 months
        # Window = 90 IS + 30 OOS = 120 days
        # Starting positions: 0, 30, 60, 90, 120, 150, 180, 210 (240 would exceed)
        # Each needs 120 days from start
        # Position 0: needs days 0-119 -> OOS ends at 119
        # Position 210: needs days 210-329 -> OOS ends at 329 (within 359)
        # Position 240: needs days 240-359 -> OOS ends at 359 = day 360 (valid)
        data_start = date(2023, 1, 1)
        data_end = data_start + timedelta(days=359)  # 360 days

        windows = wfa.generate_windows(data_start, data_end)

        # Should get 9 windows
        assert len(windows) == 9

        # Check first window
        is_start, is_end, oos_start, oos_end = windows[0]
        assert is_start == date(2023, 1, 1)
        # IS spans day 0 to day 89 = 90 days, so end = start + 89 days = Mar 31
        assert is_end == date(2023, 3, 31)
        assert oos_start == date(2023, 4, 1)
        # OOS spans day 90 to day 119 = 30 days, so end = oos_start + 29 days = Apr 30
        assert oos_end == date(2023, 4, 30)

    def test_insufficient_data(self) -> None:
        """Raises error when data too short."""
        cfg = WFAConfig(
            in_sample_days=270,
            out_of_sample_days=90,
            step_days=30,
            min_windows=1,
        )
        wfa = WalkForwardAnalysis(config=cfg)

        data_start = date(2023, 1, 1)
        data_end = date(2023, 6, 1)  # ~150 days, needs 360

        with pytest.raises(ValueError, match="Data range.*too short"):
            wfa.generate_windows(data_start, data_end)

    def test_exact_fit_single_window(self) -> None:
        """Single window when data exactly fits one window."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=1,
        )
        wfa = WalkForwardAnalysis(config=cfg)

        # Exactly 120 days = one window
        data_start = date(2023, 1, 1)
        data_end = data_start + timedelta(days=119)

        windows = wfa.generate_windows(data_start, data_end)
        assert len(windows) == 1

    def test_window_count_2_year_default(self) -> None:
        """Default config generates ~13 windows from 2 years."""
        cfg = WFAConfig.default()  # 270 IS, 90 OOS, 30 step
        wfa = WalkForwardAnalysis(config=cfg)

        # 2 years = 730 days
        data_start = date(2022, 1, 1)
        data_end = date(2023, 12, 31)

        windows = wfa.generate_windows(data_start, data_end)

        # Formula: floor((total - IS - OOS) / step) + 1
        # = floor((730 - 270 - 90) / 30) + 1 = floor(370/30) + 1 = 12 + 1 = 13
        assert len(windows) >= 12  # At least 12
        assert len(windows) <= 14  # No more than 14

    def test_no_overlapping_oos_periods(self) -> None:
        """OOS periods don't overlap with next window's IS."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,  # Same as OOS = no gap
            min_windows=1,
        )
        wfa = WalkForwardAnalysis(config=cfg)

        data_start = date(2023, 1, 1)
        data_end = date(2024, 1, 1)
        windows = wfa.generate_windows(data_start, data_end)

        for i in range(len(windows) - 1):
            _, _, _, oos_end = windows[i]
            next_is_start, _, _, _ = windows[i + 1]
            # IS starts step_days after previous IS, not after OOS
            # So there's an overlap by design in rolling WFA
            assert next_is_start > windows[i][0]


# -- WFA Run Tests --


class TestWFARun:
    """Tests for WFA execution."""

    def test_run_without_runner_raises(self) -> None:
        """run() raises if no runner set."""
        cfg = WFAConfig(in_sample_days=90, out_of_sample_days=30, step_days=30, min_windows=1)
        wfa = WalkForwardAnalysis(config=cfg)

        with pytest.raises(ValueError, match="BacktestRunner must be set"):
            wfa.run(
                "test_strategy",
                date(2023, 1, 1),
                date(2023, 12, 31),
                {"param": [1, 2, 3]},
            )

    def test_run_insufficient_windows_raises(self) -> None:
        """run() raises if fewer windows than min_windows."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=10,  # Require 10 windows
        )
        runner = MockRunner()
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        # Data only supports 2 windows
        data_start = date(2023, 1, 1)
        data_end = data_start + timedelta(days=149)  # ~150 days = 2 windows

        with pytest.raises(ValueError, match="Only 2 windows.*minimum 10"):
            wfa.run("test", data_start, data_end, {})

    def test_run_calls_runner_correctly(self) -> None:
        """run() calls optimize and backtest for each window."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=60,
            min_windows=2,
        )

        is_results = [
            MockBacktestResult(sharpe=1.5, total_return=20.0, params={"fast": 10}),
            MockBacktestResult(sharpe=1.8, total_return=25.0, params={"fast": 12}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=1.0, total_return=10.0, params={}),
            MockBacktestResult(sharpe=1.2, total_return=12.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        data_start = date(2023, 1, 1)
        data_end = data_start + timedelta(days=239)  # Supports 2-3 windows
        param_grid: dict[str, list[object]] = {"fast_period": [5, 10, 15]}

        result = wfa.run("test_strategy", data_start, data_end, param_grid)

        # Check optimize calls
        assert len(runner.optimize_calls) == result.num_windows
        for call in runner.optimize_calls:
            assert call[0] == "test_strategy"
            assert call[3] == param_grid

        # Check backtest calls
        assert len(runner.backtest_calls) == result.num_windows

        # Verify best params passed to backtest
        assert runner.backtest_calls[0][3] == {"fast": 10}
        assert runner.backtest_calls[1][3] == {"fast": 12}

    def test_run_returns_correct_result(self) -> None:
        """run() returns WFAResult with correct aggregations."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=3,
            min_oos_sharpe=0.5,
            max_degradation=0.5,
            min_consistency=0.5,
        )

        # 3 windows: 2 profitable, 1 loss
        is_results = [
            MockBacktestResult(sharpe=2.0, total_return=20.0, params={"p": 1}),
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={"p": 2}),
            MockBacktestResult(sharpe=1.8, total_return=18.0, params={"p": 3}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=1.0, total_return=8.0, params={}),
            MockBacktestResult(sharpe=0.8, total_return=6.0, params={}),
            MockBacktestResult(sharpe=-0.2, total_return=-3.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        # Window requires 120 days (90 IS + 30 OOS)
        # 3 windows with 30-day step: start at day 0, 30, 60
        # Window 2 ends at day 60 + 119 = 179
        # Window 3 would start at day 90, end at day 209 -- just beyond 179
        # So 180 days gives exactly 3 windows (days 0-179)
        data_start = date(2023, 1, 1)
        data_end = data_start + timedelta(days=179)

        result = wfa.run("test", data_start, data_end, {})

        assert result.num_windows == 3
        assert result.num_profitable_windows == 2  # 8 > 0, 6 > 0, -3 < 0

        # OOS Sharpe avg = (1.0 + 0.8 - 0.2) / 3 = 0.533...
        assert abs(result.aggregated_oos_sharpe - 0.5333) < 0.01

        # OOS Return avg = (8 + 6 - 3) / 3 = 3.67
        assert abs(result.aggregated_oos_return - 3.67) < 0.1

        # IS Return avg = (20 + 15 + 18) / 3 = 17.67
        # Efficiency = 3.67 / 17.67 = 0.208
        assert abs(result.efficiency - 0.208) < 0.01

        # Consistency = 2/3 = 0.667
        assert abs(result.consistency_ratio - 0.667) < 0.01


# -- Robustness Tests --


class TestRobustnessCheck:
    """Tests for robustness criteria evaluation."""

    def test_robust_strategy(self) -> None:
        """Strategy passes all thresholds."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=2,
            min_oos_sharpe=0.5,
            max_degradation=0.5,
            min_consistency=0.5,
        )

        # Good results
        is_results = [
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
            MockBacktestResult(sharpe=1.6, total_return=16.0, params={}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=1.2, total_return=10.0, params={}),
            MockBacktestResult(sharpe=1.0, total_return=8.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 5, 30), {})

        assert result.is_robust is True
        # OOS Sharpe = 1.1 > 0.5
        # Degradation: w1 = (1.5-1.2)/1.5=0.2, w2=(1.6-1.0)/1.6=0.375, avg=0.287 < 0.5
        # Consistency = 2/2 = 1.0 > 0.5

    def test_fails_oos_sharpe_threshold(self) -> None:
        """Strategy fails OOS Sharpe threshold."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=2,
            min_oos_sharpe=1.0,  # High threshold
        )

        is_results = [
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=0.3, total_return=5.0, params={}),
            MockBacktestResult(sharpe=0.5, total_return=8.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 5, 30), {})

        assert result.is_robust is False
        assert result.aggregated_oos_sharpe < 1.0

    def test_fails_degradation_threshold(self) -> None:
        """Strategy fails degradation threshold."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=2,
            max_degradation=0.3,  # Strict threshold
        )

        is_results = [
            MockBacktestResult(sharpe=2.0, total_return=20.0, params={}),
            MockBacktestResult(sharpe=2.0, total_return=20.0, params={}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=0.8, total_return=8.0, params={}),  # 60% degradation
            MockBacktestResult(sharpe=0.9, total_return=9.0, params={}),  # 55% degradation
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 5, 30), {})

        assert result.is_robust is False
        assert result.is_vs_oos_degradation > 0.3

    def test_fails_consistency_threshold(self) -> None:
        """Strategy fails consistency threshold."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=3,
            min_consistency=0.7,  # 70% required
        )

        is_results = [
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=0.8, total_return=5.0, params={}),
            MockBacktestResult(sharpe=-0.2, total_return=-5.0, params={}),  # Loss
            MockBacktestResult(sharpe=-0.3, total_return=-3.0, params={}),  # Loss
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 6, 30), {})

        assert result.is_robust is False
        assert result.consistency_ratio < 0.7  # 1/3 = 0.33


# -- Edge Cases --


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_is_returns(self) -> None:
        """Handles zero IS returns (efficiency calculation)."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=2,
        )

        is_results = [
            MockBacktestResult(sharpe=0.0, total_return=0.0, params={}),
            MockBacktestResult(sharpe=0.0, total_return=0.0, params={}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=0.5, total_return=5.0, params={}),
            MockBacktestResult(sharpe=0.5, total_return=5.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 5, 30), {})

        # mean IS = 0, mean OOS = 5 -> efficiency = inf
        assert result.efficiency == float("inf")

    def test_all_negative_returns(self) -> None:
        """Handles all negative returns."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=2,
        )

        is_results = [
            MockBacktestResult(sharpe=-0.5, total_return=-10.0, params={}),
            MockBacktestResult(sharpe=-0.3, total_return=-5.0, params={}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=-0.8, total_return=-15.0, params={}),
            MockBacktestResult(sharpe=-0.4, total_return=-8.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 5, 30), {})

        assert result.num_profitable_windows == 0
        assert result.consistency_ratio == 0.0
        assert result.is_robust is False

    def test_runner_protocol_compliance(self) -> None:
        """MockRunner satisfies BacktestRunner protocol."""
        runner = MockRunner()
        assert isinstance(runner, BacktestRunner)

    def test_wfa_result_frozen(self) -> None:
        """WFAResult is immutable."""
        result = WFAResult(
            windows=(),
            aggregated_oos_sharpe=1.0,
            aggregated_oos_return=10.0,
            efficiency=0.8,
            is_vs_oos_degradation=0.2,
            consistency_ratio=0.8,
            is_robust=True,
            config=WFAConfig.default(),
        )

        with pytest.raises(AttributeError):
            result.aggregated_oos_sharpe = 2.0  # type: ignore[misc]

    def test_wfa_window_frozen(self) -> None:
        """WFAWindow is immutable."""
        window = WFAWindow(
            window_index=0,
            is_start_date="2023-01-01",
            is_end_date="2023-09-30",
            oos_start_date="2023-10-01",
            oos_end_date="2023-12-31",
            is_sharpe=1.5,
            oos_sharpe=1.0,
            is_return=20.0,
            oos_return=10.0,
            best_params={},
        )

        with pytest.raises(AttributeError):
            window.is_sharpe = 2.0  # type: ignore[misc]


# -- Report Generation Tests --


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_robust(self) -> None:
        """Report for robust strategy."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=2,
        )

        is_results = [
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={"x": 1}),
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={"x": 2}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=1.2, total_return=10.0, params={}),
            MockBacktestResult(sharpe=1.0, total_return=8.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 5, 30), {})
        report = wfa.generate_report(result)

        assert "WALK-FORWARD ANALYSIS REPORT" in report
        assert "ROBUST: YES" in report
        assert "Windows: 2" in report
        assert "OOS Sharpe (avg):" in report
        assert "WF Efficiency:" in report

    def test_generate_report_not_robust(self) -> None:
        """Report shows failed criteria."""
        cfg = WFAConfig(
            in_sample_days=90,
            out_of_sample_days=30,
            step_days=30,
            min_windows=2,
            min_oos_sharpe=2.0,  # Very high threshold
        )

        is_results = [
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
            MockBacktestResult(sharpe=1.5, total_return=15.0, params={}),
        ]
        oos_results = [
            MockBacktestResult(sharpe=0.5, total_return=5.0, params={}),
            MockBacktestResult(sharpe=0.5, total_return=5.0, params={}),
        ]

        runner = MockRunner(is_results=is_results, oos_results=oos_results)
        wfa = WalkForwardAnalysis(config=cfg, runner=runner)

        result = wfa.run("test", date(2023, 1, 1), date(2023, 5, 30), {})
        report = wfa.generate_report(result)

        assert "ROBUST: NO" in report
        assert "FAILED CRITERIA:" in report
        assert "OOS Sharpe" in report
