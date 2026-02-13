"""Purged K-Fold Cross-Validation for financial time series.

Implements train/test splitting with purge and embargo gaps to prevent information
leakage due to autocorrelation. Based on López de Prado's "Advances in Financial
Machine Learning".

Standard K-Fold CV leaks information in time series due to overlapping features
(e.g., rolling indicators). Purged K-Fold adds gaps:
- Purge: removes samples immediately after train set
- Embargo: removes samples immediately before test set

Example with purge_pct=0.01, embargo_pct=0.01, n_samples=1000:
    Train: [0..189] | Purge: [190..199] | Embargo: [200..209] | Test: [210..399]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator


def max_indicator_lookback(dsl_config: dict[str, object]) -> int:
    """Extract the maximum indicator lookback period from a DSL config dict.

    Inspects all indicators' period, slow_period, and signal_period fields
    and returns the largest value found. Returns 0 if no lookback is found.

    Args:
        dsl_config: Strategy DSL as dict (e.g. from StrategyDSL.model_dump()).

    Returns:
        Maximum lookback in bars across all indicators.
    """
    max_lb = 0
    indicators = dsl_config.get("indicators", {})
    if not isinstance(indicators, dict):
        return 0
    for _name, ind in indicators.items():
        if not isinstance(ind, dict):
            continue
        for key in ("period", "slow_period", "signal_period"):
            val = ind.get(key)
            if isinstance(val, int) and val > max_lb:
                max_lb = val
    return max_lb


class BacktestRunner(Protocol):
    """Protocol for backtest execution.

    Implementations should run a backtest on the given indices and return metrics.
    """

    def run(self, train_indices: list[int], test_indices: list[int]) -> FoldResult:
        """Run backtest on train/test split.

        Args:
            train_indices: Indices for training period.
            test_indices: Indices for test period.

        Returns:
            FoldResult with metrics from the backtest.
        """
        ...


@dataclass(frozen=True)
class FoldResult:
    """Result from a single cross-validation fold.

    Attributes:
        fold_index: Zero-based index of this fold.
        train_size: Number of samples in training set.
        test_size: Number of samples in test set.
        train_sharpe: Sharpe ratio on training data (in-sample).
        test_sharpe: Sharpe ratio on test data (out-of-sample).
        train_return: Total return on training data.
        test_return: Total return on test data.
    """

    fold_index: int
    train_size: int
    test_size: int
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float


@dataclass(frozen=True)
class CVConfig:
    """Configuration for Purged K-Fold Cross-Validation.

    Attributes:
        n_splits: Number of folds (K).
        purge_pct: Percentage of total samples to remove after train set.
            If indicator_lookback_bars is set, this is auto-calculated.
        embargo_pct: Percentage of total samples to remove before test set.
        indicator_lookback_bars: Max indicator lookback in bars (e.g., 50 for
            EMA-50). Per SPEC Section 8, the purge period should equal the
            max indicator lookback. If set, overrides purge_pct when
            n_samples is known.
    """

    n_splits: int = 5
    purge_pct: float = 0.01
    embargo_pct: float = 0.01
    indicator_lookback_bars: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_splits < 2:
            msg = f"n_splits must be >= 2, got {self.n_splits}"
            raise ValueError(msg)
        if not 0.0 <= self.purge_pct < 1.0:
            msg = f"purge_pct must be in [0, 1), got {self.purge_pct}"
            raise ValueError(msg)
        if not 0.0 <= self.embargo_pct < 1.0:
            msg = f"embargo_pct must be in [0, 1), got {self.embargo_pct}"
            raise ValueError(msg)


@dataclass(frozen=True)
class CVResult:
    """Aggregated results from Purged K-Fold Cross-Validation.

    Attributes:
        fold_results: Results from each individual fold.
        mean_oos_sharpe: Mean out-of-sample Sharpe across folds.
        std_oos_sharpe: Standard deviation of out-of-sample Sharpe.
        mean_oos_return: Mean out-of-sample return across folds.
        is_robust: True if mean OOS Sharpe > threshold AND std OOS Sharpe < max.
    """

    fold_results: list[FoldResult]
    mean_oos_sharpe: float
    std_oos_sharpe: float
    mean_oos_return: float
    is_robust: bool


class PurgedKFold:
    """K-Fold time series splitter with purge and embargo gaps.

    Prevents information leakage in time series cross-validation by adding
    gaps between train and test sets. Unlike sklearn's TimeSeriesSplit which
    uses expanding windows, this uses proper K-Fold with temporal ordering.

    Args:
        n_splits: Number of folds (default 5).
        purge_pct: Fraction of data to purge after train set (default 0.01).
        embargo_pct: Fraction of data to embargo before test set (default 0.01).

    Example:
        >>> kfold = PurgedKFold(n_splits=5, purge_pct=0.01, embargo_pct=0.01)
        >>> for train_idx, test_idx in kfold.split(1000):
        ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
        indicator_lookback_bars: int | None = None,
    ) -> None:
        """Initialize PurgedKFold.

        Args:
            n_splits: Number of folds.
            purge_pct: Fraction of data to purge after train.
            embargo_pct: Fraction of data to embargo before test.
            indicator_lookback_bars: Max indicator lookback in bars. Per SPEC
                Section 8, purge period should equal max indicator lookback.
                If set, overrides purge_pct when n_samples is known in split().

        Raises:
            ValueError: If parameters are invalid.
        """
        if n_splits < 2:
            msg = f"n_splits must be >= 2, got {n_splits}"
            raise ValueError(msg)
        if not 0.0 <= purge_pct < 1.0:
            msg = f"purge_pct must be in [0, 1), got {purge_pct}"
            raise ValueError(msg)
        if not 0.0 <= embargo_pct < 1.0:
            msg = f"embargo_pct must be in [0, 1), got {embargo_pct}"
            raise ValueError(msg)

        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.indicator_lookback_bars = indicator_lookback_bars

    def split(self, n_samples: int) -> Iterator[tuple[list[int], list[int]]]:
        """Generate train/test indices for each fold.

        For each fold, the test set is one of the K equal partitions.
        The train set is all other partitions, with purge and embargo gaps
        applied to prevent leakage.

        Uses range slicing and list pre-allocation instead of repeated
        extend operations for better memory efficiency.

        Args:
            n_samples: Total number of samples in the dataset.

        Yields:
            Tuple of (train_indices, test_indices) for each fold.

        Raises:
            ValueError: If n_samples is too small for the configuration.
        """
        # SPEC Section 8: purge period = max indicator lookback
        if self.indicator_lookback_bars is not None and self.indicator_lookback_bars > 0:
            purge_len = self.indicator_lookback_bars
            computed_pct = purge_len / n_samples if n_samples > 0 else 0.0
            if self.purge_pct != 0.01 and abs(computed_pct - self.purge_pct) > 1e-6:
                logger.warning(
                    "indicator_lookback_bars=%d overrides purge_pct=%.4f "
                    "(effective purge=%.4f of %d samples)",
                    self.indicator_lookback_bars,
                    self.purge_pct,
                    computed_pct,
                    n_samples,
                )
        else:
            purge_len = int(n_samples * self.purge_pct)
        embargo_len = int(n_samples * self.embargo_pct)
        total_gap = purge_len + embargo_len

        # Minimum samples needed per fold after gaps
        min_samples_per_fold = 10
        if n_samples < self.n_splits * min_samples_per_fold + total_gap * (
            self.n_splits - 1
        ):
            msg = (
                f"Not enough samples ({n_samples}) for {self.n_splits} folds "
                f"with purge={purge_len} and embargo={embargo_len}. "
                f"Need at least {self.n_splits * min_samples_per_fold + total_gap * (self.n_splits - 1)}."
            )
            raise ValueError(msg)

        # Pre-compute fold boundaries once (avoid repeated multiplication)
        fold_size = n_samples // self.n_splits
        fold_starts = [i * fold_size for i in range(self.n_splits)]
        fold_ends = [fold_starts[i + 1] if i < self.n_splits - 1 else n_samples
                     for i in range(self.n_splits)]

        for fold_idx in range(self.n_splits):
            # Test set is the current fold
            test_indices = list(range(fold_starts[fold_idx], fold_ends[fold_idx]))

            # Build train set from all other folds, applying purge/embargo
            # Collect ranges first, then build list in one shot
            train_ranges: list[range] = []

            for train_fold_idx in range(self.n_splits):
                if train_fold_idx == fold_idx:
                    continue

                train_fold_start = fold_starts[train_fold_idx]
                train_fold_end = fold_ends[train_fold_idx]

                # Apply purge if train fold immediately precedes test fold
                if train_fold_idx == fold_idx - 1:
                    purge_start = max(train_fold_start, train_fold_end - purge_len)
                    train_fold_end = purge_start

                # Apply embargo if train fold immediately follows test fold
                if train_fold_idx == fold_idx + 1:
                    embargo_end = min(train_fold_end, train_fold_start + embargo_len)
                    train_fold_start = embargo_end

                # Only add if we have samples left after purge/embargo
                if train_fold_start < train_fold_end:
                    train_ranges.append(range(train_fold_start, train_fold_end))

            # Build train indices from collected ranges
            total_train = sum(len(r) for r in train_ranges)
            train_indices: list[int] = [0] * total_train
            offset = 0
            for r in train_ranges:
                rlen = len(r)
                train_indices[offset:offset + rlen] = r
                offset += rlen

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


@dataclass
class PurgedKFoldCV:
    """Cross-validation runner using Purged K-Fold splits.

    Runs backtests on each fold and aggregates results to assess strategy
    robustness across different time periods.

    SPEC Section 8 robustness criteria:
        - Mean OOS Sharpe > min_oos_sharpe (default 0.5)
        - std(OOS Sharpe) < max_oos_sharpe_std (default 1.0)

    Args:
        config: CV configuration (n_splits, purge_pct, embargo_pct).
        robustness_threshold: Legacy threshold (kept for backwards compatibility).
        min_oos_sharpe: Minimum mean OOS Sharpe ratio (default 0.5).
        max_oos_sharpe_std: Maximum std of OOS Sharpe ratios (default 1.0).
    """

    config: CVConfig
    robustness_threshold: float = 0.5
    min_oos_sharpe: float = 0.5
    max_oos_sharpe_std: float = 1.0
    _kfold: PurgedKFold = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal K-Fold splitter."""
        self._kfold = PurgedKFold(
            n_splits=self.config.n_splits,
            purge_pct=self.config.purge_pct,
            embargo_pct=self.config.embargo_pct,
            indicator_lookback_bars=self.config.indicator_lookback_bars,
        )

    def run(self, n_samples: int, runner: BacktestRunner) -> CVResult:
        """Run cross-validation using provided backtest runner.

        Args:
            n_samples: Total number of samples in dataset.
            runner: BacktestRunner implementation to execute backtests.

        Returns:
            CVResult with aggregated metrics and robustness assessment.
        """
        fold_results: list[FoldResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(self._kfold.split(n_samples)):
            result = runner.run(train_idx, test_idx)
            # Create new FoldResult with correct fold_index
            fold_result = FoldResult(
                fold_index=fold_idx,
                train_size=result.train_size,
                test_size=result.test_size,
                train_sharpe=result.train_sharpe,
                test_sharpe=result.test_sharpe,
                train_return=result.train_return,
                test_return=result.test_return,
            )
            fold_results.append(fold_result)

        return self._aggregate_results(fold_results)

    def run_with_results(
        self, n_samples: int, fold_results: list[FoldResult]
    ) -> CVResult:
        """Aggregate pre-computed fold results.

        Use when fold results are computed externally (e.g., parallel execution).

        Args:
            n_samples: Total number of samples (for validation).
            fold_results: Pre-computed results for each fold.

        Returns:
            CVResult with aggregated metrics.

        Raises:
            ValueError: If number of results doesn't match n_splits.
        """
        if len(fold_results) != self.config.n_splits:
            msg = (
                f"Expected {self.config.n_splits} fold results, "
                f"got {len(fold_results)}"
            )
            raise ValueError(msg)

        return self._aggregate_results(fold_results)

    def _aggregate_results(self, fold_results: list[FoldResult]) -> CVResult:
        """Aggregate fold results into CVResult.

        Uses single-pass computation (sum and sum-of-squares) for mean and
        variance to avoid multiple list iterations.

        Args:
            fold_results: Results from each fold.

        Returns:
            Aggregated CVResult.
        """
        if not fold_results:
            return CVResult(
                fold_results=[],
                mean_oos_sharpe=0.0,
                std_oos_sharpe=0.0,
                mean_oos_return=0.0,
                is_robust=False,
            )

        n = len(fold_results)
        inv_n = 1.0 / n

        # Single-pass Welford's online algorithm for numerically stable variance
        sum_return = 0.0
        mean_sharpe = 0.0
        m2_sharpe = 0.0  # sum of squared deviations from running mean

        for i, r in enumerate(fold_results):
            s = r.test_sharpe
            sum_return += r.test_return
            # Welford update
            k = i + 1
            delta = s - mean_sharpe
            mean_sharpe += delta / k
            delta2 = s - mean_sharpe
            m2_sharpe += delta * delta2

        mean_return = sum_return * inv_n

        # Bessel's correction: sample variance = M2 / (n-1)
        if n > 1:
            variance = m2_sharpe / (n - 1)
            std_sharpe = variance**0.5 if variance > 0.0 else 0.0
        else:
            std_sharpe = 0.0

        is_robust = (mean_sharpe > self.min_oos_sharpe) and (
            std_sharpe < self.max_oos_sharpe_std
        )

        return CVResult(
            fold_results=list(fold_results),
            mean_oos_sharpe=mean_sharpe,
            std_oos_sharpe=std_sharpe,
            mean_oos_return=mean_return,
            is_robust=is_robust,
        )

    def get_splits(self, n_samples: int) -> list[tuple[list[int], list[int]]]:
        """Get all train/test splits without running backtests.

        Useful for inspecting splits or running backtests in parallel.

        Args:
            n_samples: Total number of samples.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        return list(self._kfold.split(n_samples))
