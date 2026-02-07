"""Walk-Forward Analysis (WFA) for overfitting prevention.

Tests strategy robustness by optimizing on in-sample periods
and validating on out-of-sample periods, rolling forward through time.

Configuration (from SPEC.md for 2-year data):
- Training window: 9 months (270 days)
- Test window: 3 months (90 days)
- Step size: 1 month (30 days)
- Produces ~13 windows from 24 months

Filter criteria:
- Walk-forward efficiency > 0.5 (mean OOS / mean IS return)
- > 50% of OOS windows profitable
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class WFAConfig:
    """Configuration for Walk-Forward Analysis.

    Attributes:
        in_sample_days: Number of days for training/optimization.
        out_of_sample_days: Number of days for testing.
        step_days: Number of days to roll forward between windows.
        min_windows: Minimum number of windows required.
        min_oos_sharpe: Minimum aggregated OOS Sharpe to pass.
        max_degradation: Maximum allowed IS vs OOS degradation (0.5 = 50%).
        min_consistency: Minimum fraction of profitable OOS windows.
        min_efficiency: Minimum walk-forward efficiency (mean OOS / mean IS return).
    """

    in_sample_days: int
    out_of_sample_days: int
    step_days: int
    min_windows: int = 8
    min_oos_sharpe: float = 0.5
    max_degradation: float = 0.5
    min_consistency: float = 0.5
    min_efficiency: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.in_sample_days <= 0:
            raise ValueError("in_sample_days must be positive")
        if self.out_of_sample_days <= 0:
            raise ValueError("out_of_sample_days must be positive")
        if self.step_days <= 0:
            raise ValueError("step_days must be positive")
        if self.min_windows < 1:
            raise ValueError("min_windows must be at least 1")
        if not (0 <= self.max_degradation <= 1):
            raise ValueError("max_degradation must be in [0, 1]")
        if not (0 <= self.min_consistency <= 1):
            raise ValueError("min_consistency must be in [0, 1]")
        if self.min_efficiency < 0:
            raise ValueError("min_efficiency must be non-negative")

    @classmethod
    def default(cls) -> WFAConfig:
        """Return default config (9m IS / 3m OOS / 1m step)."""
        return cls(
            in_sample_days=270,  # 9 months
            out_of_sample_days=90,  # 3 months
            step_days=30,  # 1 month
            min_windows=8,
            min_oos_sharpe=0.5,
            max_degradation=0.5,
            min_consistency=0.5,
            min_efficiency=0.5,
        )


@dataclass(frozen=True, slots=True)
class WFAWindow:
    """Individual window result in Walk-Forward Analysis.

    Attributes:
        window_index: Zero-based index of this window.
        is_start_date: In-sample period start date (ISO format).
        is_end_date: In-sample period end date (ISO format).
        oos_start_date: Out-of-sample period start date (ISO format).
        oos_end_date: Out-of-sample period end date (ISO format).
        is_sharpe: Sharpe ratio from in-sample optimization.
        oos_sharpe: Sharpe ratio on out-of-sample period.
        is_return: Total return from in-sample (percentage).
        oos_return: Total return on out-of-sample (percentage).
        best_params: Optimized parameters from IS period.
    """

    window_index: int
    is_start_date: str
    is_end_date: str
    oos_start_date: str
    oos_end_date: str
    is_sharpe: float
    oos_sharpe: float
    is_return: float
    oos_return: float
    best_params: dict[str, object]

    @property
    def is_oos_profitable(self) -> bool:
        """Whether OOS period was profitable."""
        return self.oos_return > 0

    @property
    def sharpe_degradation(self) -> float:
        """Sharpe degradation: (IS - OOS) / |IS|."""
        if self.is_sharpe == 0:
            return 0.0 if self.oos_sharpe == 0 else -1.0 if self.oos_sharpe > 0 else 1.0
        return (self.is_sharpe - self.oos_sharpe) / abs(self.is_sharpe)

    @property
    def return_degradation(self) -> float:
        """Return degradation: (IS - OOS) / |IS|."""
        if self.is_return == 0:
            return 0.0 if self.oos_return == 0 else -1.0 if self.oos_return > 0 else 1.0
        return (self.is_return - self.oos_return) / abs(self.is_return)


@dataclass(frozen=True, slots=True)
class WFAResult:
    """Aggregated Walk-Forward Analysis result.

    Attributes:
        windows: Individual window results.
        aggregated_oos_sharpe: Combined Sharpe across all OOS periods.
        aggregated_oos_return: Combined return across all OOS periods.
        efficiency: mean(OOS_return) / mean(IS_return).
        is_vs_oos_degradation: Average IS vs OOS Sharpe degradation.
        consistency_ratio: Fraction of profitable OOS windows.
        is_robust: Whether strategy passes all thresholds.
        config: WFA configuration used.
    """

    windows: tuple[WFAWindow, ...]
    aggregated_oos_sharpe: float
    aggregated_oos_return: float
    efficiency: float
    is_vs_oos_degradation: float
    consistency_ratio: float
    is_robust: bool
    config: WFAConfig

    @property
    def num_windows(self) -> int:
        """Number of windows analyzed."""
        return len(self.windows)

    @property
    def num_profitable_windows(self) -> int:
        """Number of OOS windows that were profitable."""
        return sum(1 for w in self.windows if w.is_oos_profitable)


@runtime_checkable
class BacktestRunner(Protocol):
    """Protocol for backtest execution.

    Implementers must provide optimize() and backtest() methods.
    This is mocked for testing - real impl will use NT screening mode.
    """

    def optimize(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        param_grid: dict[str, list[object]],
    ) -> tuple[dict[str, object], float, float]:
        """Optimize strategy on date range.

        Args:
            strategy_id: Strategy identifier.
            start_date: Optimization period start.
            end_date: Optimization period end.
            param_grid: Parameter grid to search.

        Returns:
            Tuple of (best_params, sharpe, total_return).
        """
        ...

    def backtest(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        params: dict[str, object],
    ) -> tuple[float, float]:
        """Run backtest with fixed parameters.

        Args:
            strategy_id: Strategy identifier.
            start_date: Backtest period start.
            end_date: Backtest period end.
            params: Fixed strategy parameters.

        Returns:
            Tuple of (sharpe, total_return).
        """
        ...


class BaseBacktestRunner(ABC):
    """Abstract base for backtest runners."""

    @abstractmethod
    def optimize(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        param_grid: dict[str, list[object]],
    ) -> tuple[dict[str, object], float, float]:
        """Optimize strategy on date range."""
        ...

    @abstractmethod
    def backtest(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        params: dict[str, object],
    ) -> tuple[float, float]:
        """Run backtest with fixed parameters."""
        ...


class WalkForwardAnalysis:
    """Walk-Forward Analysis executor.

    Splits data into rolling IS/OOS windows, optimizes on IS,
    tests on OOS, and aggregates results.
    """

    def __init__(
        self,
        config: WFAConfig | None = None,
        runner: BacktestRunner | None = None,
    ) -> None:
        """Initialize WFA.

        Args:
            config: WFA configuration. Uses default if None.
            runner: Backtest runner. Must be provided before run().
        """
        self._config = config or WFAConfig.default()
        self._runner = runner

    @property
    def config(self) -> WFAConfig:
        """Return WFA configuration."""
        return self._config

    @property
    def runner(self) -> BacktestRunner | None:
        """Return backtest runner."""
        return self._runner

    @runner.setter
    def runner(self, runner: BacktestRunner) -> None:
        """Set backtest runner."""
        self._runner = runner

    def generate_windows(
        self,
        data_start: date,
        data_end: date,
    ) -> list[tuple[date, date, date, date]]:
        """Generate IS/OOS window date ranges.

        Args:
            data_start: First available data date.
            data_end: Last available data date.

        Returns:
            List of (is_start, is_end, oos_start, oos_end) tuples.

        Raises:
            ValueError: If data range too short for any windows.
        """
        windows: list[tuple[date, date, date, date]] = []

        # Window length = IS + OOS
        window_length = self._config.in_sample_days + self._config.out_of_sample_days
        total_days = (data_end - data_start).days + 1

        if total_days < window_length:
            msg = (
                f"Data range ({total_days} days) too short for window "
                f"({window_length} days = {self._config.in_sample_days} IS + "
                f"{self._config.out_of_sample_days} OOS)"
            )
            raise ValueError(msg)

        # Generate windows by stepping forward
        current_is_start = data_start

        while True:
            is_end = current_is_start + timedelta(days=self._config.in_sample_days - 1)
            oos_start = is_end + timedelta(days=1)
            oos_end = oos_start + timedelta(days=self._config.out_of_sample_days - 1)

            # Stop if OOS would exceed data range
            if oos_end > data_end:
                break

            windows.append((current_is_start, is_end, oos_start, oos_end))

            # Step forward
            current_is_start = current_is_start + timedelta(days=self._config.step_days)

        return windows

    def run(
        self,
        strategy_id: str,
        data_start: date,
        data_end: date,
        param_grid: dict[str, list[object]],
    ) -> WFAResult:
        """Execute Walk-Forward Analysis.

        Args:
            strategy_id: Strategy identifier.
            data_start: First available data date.
            data_end: Last available data date.
            param_grid: Parameter grid for optimization.

        Returns:
            WFAResult with all windows and aggregated metrics.

        Raises:
            ValueError: If insufficient windows or no runner.
        """
        if self._runner is None:
            raise ValueError("BacktestRunner must be set before calling run()")

        # Generate windows
        window_dates = self.generate_windows(data_start, data_end)

        if len(window_dates) < self._config.min_windows:
            msg = (
                f"Only {len(window_dates)} windows generated, "
                f"minimum {self._config.min_windows} required"
            )
            raise ValueError(msg)

        # Execute each window
        windows: list[WFAWindow] = []
        for idx, (is_start, is_end, oos_start, oos_end) in enumerate(window_dates):
            # Optimize on IS
            best_params, is_sharpe, is_return = self._runner.optimize(
                strategy_id, is_start, is_end, param_grid
            )

            # Test on OOS with optimized params
            oos_sharpe, oos_return = self._runner.backtest(
                strategy_id, oos_start, oos_end, best_params
            )

            window = WFAWindow(
                window_index=idx,
                is_start_date=is_start.isoformat(),
                is_end_date=is_end.isoformat(),
                oos_start_date=oos_start.isoformat(),
                oos_end_date=oos_end.isoformat(),
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                is_return=is_return,
                oos_return=oos_return,
                best_params=best_params,
            )
            windows.append(window)

        return self._aggregate_results(windows)

    def _aggregate_results(self, windows: list[WFAWindow]) -> WFAResult:
        """Aggregate window results into final WFAResult.

        Uses a single pass over windows to accumulate all metrics,
        avoiding 5+ separate list iterations.

        Args:
            windows: List of individual window results.

        Returns:
            WFAResult with aggregated metrics.
        """
        if not windows:
            return WFAResult(
                windows=(),
                aggregated_oos_sharpe=0.0,
                aggregated_oos_return=0.0,
                efficiency=0.0,
                is_vs_oos_degradation=0.0,
                consistency_ratio=0.0,
                is_robust=False,
                config=self._config,
            )

        # Single-pass accumulation of all metrics
        n = len(windows)
        inv_n = 1.0 / n
        sum_oos_sharpe = 0.0
        sum_oos_return = 0.0
        sum_is_return = 0.0
        sum_degradation = 0.0
        profitable_count = 0

        for w in windows:
            sum_oos_sharpe += w.oos_sharpe
            sum_oos_return += w.oos_return
            sum_is_return += w.is_return
            sum_degradation += w.sharpe_degradation
            if w.oos_return > 0:
                profitable_count += 1

        aggregated_oos_sharpe = sum_oos_sharpe * inv_n
        aggregated_oos_return = sum_oos_return * inv_n

        # Efficiency: mean(OOS_return) / mean(IS_return)
        mean_is_return = sum_is_return * inv_n
        if mean_is_return == 0:
            efficiency = 0.0 if aggregated_oos_return == 0 else float("inf")
        else:
            efficiency = aggregated_oos_return / mean_is_return

        avg_degradation = sum_degradation * inv_n
        consistency = profitable_count * inv_n

        # Robustness check (SPEC Section 8: efficiency > 0.5 AND > 50% OOS profitable)
        is_robust = (
            aggregated_oos_sharpe >= self._config.min_oos_sharpe
            and avg_degradation <= self._config.max_degradation
            and consistency >= self._config.min_consistency
            and efficiency >= self._config.min_efficiency
        )

        return WFAResult(
            windows=tuple(windows),
            aggregated_oos_sharpe=aggregated_oos_sharpe,
            aggregated_oos_return=aggregated_oos_return,
            efficiency=efficiency,
            is_vs_oos_degradation=avg_degradation,
            consistency_ratio=consistency,
            is_robust=is_robust,
            config=self._config,
        )

    def generate_report(self, result: WFAResult) -> str:
        """Generate text report from WFA result.

        Args:
            result: WFA result to report.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 70,
            "WALK-FORWARD ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Windows: {result.num_windows}",
            f"Config: IS={self._config.in_sample_days}d, "
            f"OOS={self._config.out_of_sample_days}d, "
            f"Step={self._config.step_days}d",
            "",
            "-" * 70,
            "AGGREGATED METRICS",
            "-" * 70,
            f"OOS Sharpe (avg): {result.aggregated_oos_sharpe:.3f}",
            f"OOS Return (avg): {result.aggregated_oos_return:.2f}%",
            f"WF Efficiency: {result.efficiency:.3f}",
            f"IS vs OOS Degradation: {result.is_vs_oos_degradation:.1%}",
            f"Consistency: {result.consistency_ratio:.1%} "
            f"({result.num_profitable_windows}/{result.num_windows} profitable)",
            "",
            "-" * 70,
            f"ROBUST: {'YES' if result.is_robust else 'NO'}",
            "-" * 70,
        ]

        if not result.is_robust:
            lines.append("FAILED CRITERIA:")
            if result.aggregated_oos_sharpe < self._config.min_oos_sharpe:
                lines.append(
                    f"  - OOS Sharpe {result.aggregated_oos_sharpe:.3f} < "
                    f"{self._config.min_oos_sharpe}"
                )
            if result.is_vs_oos_degradation > self._config.max_degradation:
                lines.append(
                    f"  - Degradation {result.is_vs_oos_degradation:.1%} > "
                    f"{self._config.max_degradation:.1%}"
                )
            if result.consistency_ratio < self._config.min_consistency:
                lines.append(
                    f"  - Consistency {result.consistency_ratio:.1%} < "
                    f"{self._config.min_consistency:.1%}"
                )
            if result.efficiency < self._config.min_efficiency:
                lines.append(
                    f"  - Efficiency {result.efficiency:.3f} < "
                    f"{self._config.min_efficiency}"
                )
            lines.append("")

        # Window details
        lines.append("-" * 70)
        lines.append("WINDOW DETAILS")
        lines.append("-" * 70)

        for w in result.windows:
            lines.append(
                f"[{w.window_index}] IS: {w.is_start_date} to {w.is_end_date}"
            )
            lines.append(
                f"    OOS: {w.oos_start_date} to {w.oos_end_date}"
            )
            lines.append(
                f"    Sharpe: {w.is_sharpe:.3f} -> {w.oos_sharpe:.3f} "
                f"({w.sharpe_degradation:+.1%})"
            )
            lines.append(
                f"    Return: {w.is_return:.2f}% -> {w.oos_return:.2f}% "
                f"({'PROFIT' if w.is_oos_profitable else 'LOSS'})"
            )

        return "\n".join(lines)
