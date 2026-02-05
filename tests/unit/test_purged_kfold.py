"""Unit tests for Purged K-Fold Cross-Validation."""

from __future__ import annotations

import pytest

from vibe_quant.overfitting.purged_kfold import (
    CVConfig,
    CVResult,
    FoldResult,
    PurgedKFold,
    PurgedKFoldCV,
)

# --- PurgedKFold Tests ---


class TestPurgedKFoldInit:
    """Tests for PurgedKFold initialization."""

    def test_valid_defaults(self) -> None:
        """Default parameters create valid splitter."""
        kfold = PurgedKFold()
        assert kfold.n_splits == 5
        assert kfold.purge_pct == 0.01
        assert kfold.embargo_pct == 0.01

    def test_custom_params(self) -> None:
        """Custom parameters accepted."""
        kfold = PurgedKFold(n_splits=10, purge_pct=0.05, embargo_pct=0.02)
        assert kfold.n_splits == 10
        assert kfold.purge_pct == 0.05
        assert kfold.embargo_pct == 0.02

    def test_n_splits_one_raises(self) -> None:
        """n_splits < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedKFold(n_splits=1)

    def test_n_splits_zero_raises(self) -> None:
        """n_splits = 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedKFold(n_splits=0)

    def test_negative_purge_pct_raises(self) -> None:
        """Negative purge_pct raises ValueError."""
        with pytest.raises(ValueError, match="purge_pct must be in"):
            PurgedKFold(purge_pct=-0.01)

    def test_purge_pct_one_raises(self) -> None:
        """purge_pct >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="purge_pct must be in"):
            PurgedKFold(purge_pct=1.0)

    def test_negative_embargo_pct_raises(self) -> None:
        """Negative embargo_pct raises ValueError."""
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            PurgedKFold(embargo_pct=-0.01)

    def test_embargo_pct_one_raises(self) -> None:
        """embargo_pct >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            PurgedKFold(embargo_pct=1.0)

    def test_zero_purge_embargo_valid(self) -> None:
        """Zero purge and embargo is valid (standard K-Fold)."""
        kfold = PurgedKFold(purge_pct=0.0, embargo_pct=0.0)
        assert kfold.purge_pct == 0.0
        assert kfold.embargo_pct == 0.0


class TestPurgedKFoldSplit:
    """Tests for PurgedKFold.split() index generation."""

    def test_correct_number_of_splits(self) -> None:
        """split() yields correct number of folds."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        splits = list(kfold.split(100))
        assert len(splits) == 5

    def test_all_samples_covered_in_test(self) -> None:
        """Each sample appears in exactly one test set."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        n_samples = 100
        all_test_indices: set[int] = set()

        for _, test_idx in kfold.split(n_samples):
            for idx in test_idx:
                assert idx not in all_test_indices, f"Index {idx} in multiple test sets"
                all_test_indices.add(idx)

        assert all_test_indices == set(range(n_samples))

    def test_no_overlap_between_train_test(self) -> None:
        """Train and test sets do not overlap within a fold."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        for train_idx, test_idx in kfold.split(100):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_train_indices_less_than_test_with_gap(self) -> None:
        """With purge/embargo, train indices < test indices (no lookahead)."""
        kfold = PurgedKFold(n_splits=3, purge_pct=0.05, embargo_pct=0.05)
        n_samples = 1000

        for train_idx, test_idx in kfold.split(n_samples):
            # Get train indices that are less than min test index
            min_test = min(test_idx)
            max_test = max(test_idx)

            # Train indices should either be:
            # 1. Less than min_test - purge_gap (before test)
            # 2. Greater than max_test + embargo_gap (after test)
            for t_idx in train_idx:
                # Either before test (with purge gap) or after test (with embargo gap)
                assert (
                    t_idx < min_test or t_idx > max_test
                ), f"Train index {t_idx} overlaps test range [{min_test}, {max_test}]"

    def test_purge_gap_applied(self) -> None:
        """Purge creates gap between train and following test set."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.05, embargo_pct=0.0)
        n_samples = 1000
        purge_len = int(n_samples * 0.05)  # 50 samples

        # Fold 1: test is [200, 400), train before should end at 200 - purge
        splits = list(kfold.split(n_samples))

        # Check fold 1 (second fold)
        train_idx, test_idx = splits[1]
        test_start = min(test_idx)

        # Train indices from fold 0 should end before test_start - purge
        train_before_test = [t for t in train_idx if t < test_start]
        if train_before_test:
            max_train_before = max(train_before_test)
            gap = test_start - max_train_before - 1
            assert gap >= purge_len - 1, f"Purge gap too small: {gap} < {purge_len}"

    def test_embargo_gap_applied(self) -> None:
        """Embargo creates gap between test and following train set."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.05)
        n_samples = 1000
        embargo_len = int(n_samples * 0.05)  # 50 samples

        splits = list(kfold.split(n_samples))

        # Check fold 0 (first fold)
        train_idx, test_idx = splits[0]
        test_end = max(test_idx)

        # Train indices from fold 1 should start after test_end + embargo
        train_after_test = [t for t in train_idx if t > test_end]
        if train_after_test:
            min_train_after = min(train_after_test)
            gap = min_train_after - test_end - 1
            assert gap >= embargo_len - 1, f"Embargo gap too small: {gap} < {embargo_len}"

    def test_too_few_samples_raises(self) -> None:
        """Too few samples for configuration raises ValueError."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.1, embargo_pct=0.1)
        with pytest.raises(ValueError, match="Not enough samples"):
            list(kfold.split(50))  # Way too small

    def test_minimum_samples_accepted(self) -> None:
        """Minimum viable samples work."""
        # 5 folds, 10 samples each = 50, plus gaps
        kfold = PurgedKFold(n_splits=5, purge_pct=0.01, embargo_pct=0.01)
        # Should work with enough samples
        splits = list(kfold.split(100))
        assert len(splits) == 5

    def test_get_n_splits(self) -> None:
        """get_n_splits returns n_splits."""
        kfold = PurgedKFold(n_splits=7)
        assert kfold.get_n_splits() == 7

    def test_uneven_fold_sizes(self) -> None:
        """Handles samples not evenly divisible by n_splits."""
        kfold = PurgedKFold(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        n_samples = 100  # Not divisible by 3

        splits = list(kfold.split(n_samples))
        all_test = []
        for _, test_idx in splits:
            all_test.extend(test_idx)

        # All samples should be in some test set
        assert sorted(all_test) == list(range(n_samples))


class TestPurgedKFoldNoLeakage:
    """Tests specifically for information leakage prevention."""

    def test_train_before_test_has_purge(self) -> None:
        """Train data before test is purged at end."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.02, embargo_pct=0.0)
        n_samples = 1000
        purge_len = int(n_samples * 0.02)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(n_samples)):
            if fold_idx == 0:
                continue  # First fold has no train before test

            test_start = min(test_idx)
            train_before = [t for t in train_idx if t < test_start]

            if train_before:
                max_train_before = max(train_before)
                # Gap should be at least purge_len
                assert (
                    test_start - max_train_before > purge_len
                ), f"Fold {fold_idx}: insufficient purge gap"

    def test_train_after_test_has_embargo(self) -> None:
        """Train data after test is embargoed at start."""
        kfold = PurgedKFold(n_splits=5, purge_pct=0.0, embargo_pct=0.02)
        n_samples = 1000
        embargo_len = int(n_samples * 0.02)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(n_samples)):
            if fold_idx == kfold.n_splits - 1:
                continue  # Last fold has no train after test

            test_end = max(test_idx)
            train_after = [t for t in train_idx if t > test_end]

            if train_after:
                min_train_after = min(train_after)
                # Gap should be at least embargo_len
                assert (
                    min_train_after - test_end > embargo_len
                ), f"Fold {fold_idx}: insufficient embargo gap"


# --- CVConfig Tests ---


class TestCVConfig:
    """Tests for CVConfig dataclass."""

    def test_valid_defaults(self) -> None:
        """Default config is valid."""
        config = CVConfig()
        assert config.n_splits == 5
        assert config.purge_pct == 0.01
        assert config.embargo_pct == 0.01

    def test_n_splits_one_raises(self) -> None:
        """n_splits < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            CVConfig(n_splits=1)

    def test_invalid_purge_raises(self) -> None:
        """Invalid purge_pct raises ValueError."""
        with pytest.raises(ValueError, match="purge_pct must be in"):
            CVConfig(purge_pct=1.5)

    def test_invalid_embargo_raises(self) -> None:
        """Invalid embargo_pct raises ValueError."""
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            CVConfig(embargo_pct=-0.1)


# --- FoldResult Tests ---


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_creation(self) -> None:
        """FoldResult creates correctly."""
        result = FoldResult(
            fold_index=0,
            train_size=800,
            test_size=200,
            train_sharpe=1.5,
            test_sharpe=0.8,
            train_return=0.25,
            test_return=0.10,
        )
        assert result.fold_index == 0
        assert result.train_size == 800
        assert result.test_size == 200
        assert result.train_sharpe == 1.5
        assert result.test_sharpe == 0.8
        assert result.train_return == 0.25
        assert result.test_return == 0.10

    def test_frozen(self) -> None:
        """FoldResult is immutable."""
        result = FoldResult(
            fold_index=0,
            train_size=800,
            test_size=200,
            train_sharpe=1.5,
            test_sharpe=0.8,
            train_return=0.25,
            test_return=0.10,
        )
        with pytest.raises(AttributeError):
            result.fold_index = 1  # type: ignore[misc]


# --- CVResult Tests ---


class TestCVResult:
    """Tests for CVResult dataclass."""

    def test_creation(self) -> None:
        """CVResult creates correctly."""
        fold_results = [
            FoldResult(0, 800, 200, 1.5, 0.8, 0.25, 0.10),
            FoldResult(1, 800, 200, 1.2, 0.6, 0.20, 0.08),
        ]
        result = CVResult(
            fold_results=fold_results,
            mean_oos_sharpe=0.7,
            std_oos_sharpe=0.1,
            mean_oos_return=0.09,
            is_robust=True,
        )
        assert len(result.fold_results) == 2
        assert result.mean_oos_sharpe == 0.7
        assert result.is_robust is True


# --- PurgedKFoldCV Tests ---


class MockBacktestRunner:
    """Mock backtest runner for testing."""

    def __init__(self, sharpes: list[float], returns: list[float]) -> None:
        """Initialize with predetermined results."""
        self.sharpes = sharpes
        self.returns = returns
        self.call_count = 0

    def run(self, train_indices: list[int], test_indices: list[int]) -> FoldResult:
        """Return predetermined result for fold."""
        idx = self.call_count
        self.call_count += 1
        return FoldResult(
            fold_index=idx,
            train_size=len(train_indices),
            test_size=len(test_indices),
            train_sharpe=self.sharpes[idx] + 0.5,  # IS higher than OOS
            test_sharpe=self.sharpes[idx],
            train_return=self.returns[idx] + 0.1,
            test_return=self.returns[idx],
        )


class TestPurgedKFoldCV:
    """Tests for PurgedKFoldCV runner."""

    def test_runs_all_folds(self) -> None:
        """CV runs backtest for each fold."""
        config = CVConfig(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config)
        runner = MockBacktestRunner(
            sharpes=[0.8, 0.6, 0.7, 0.5, 0.9],
            returns=[0.10, 0.08, 0.09, 0.07, 0.12],
        )

        result = cv.run(n_samples=100, runner=runner)

        assert runner.call_count == 5
        assert len(result.fold_results) == 5

    def test_mean_sharpe_calculation(self) -> None:
        """Mean OOS Sharpe calculated correctly."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config)
        sharpes = [0.6, 0.9, 1.2]  # Mean = 0.9
        runner = MockBacktestRunner(sharpes=sharpes, returns=[0.1, 0.1, 0.1])

        result = cv.run(n_samples=90, runner=runner)

        assert abs(result.mean_oos_sharpe - 0.9) < 0.001

    def test_std_sharpe_calculation(self) -> None:
        """Std OOS Sharpe calculated with Bessel correction."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config)
        sharpes = [0.6, 0.9, 1.2]  # Std = 0.3 with n-1
        runner = MockBacktestRunner(sharpes=sharpes, returns=[0.1, 0.1, 0.1])

        result = cv.run(n_samples=90, runner=runner)

        # Variance = ((0.6-0.9)^2 + (0.9-0.9)^2 + (1.2-0.9)^2) / 2 = 0.09
        # Std = 0.3
        assert abs(result.std_oos_sharpe - 0.3) < 0.001

    def test_is_robust_true(self) -> None:
        """Strategy marked robust when mean - std > threshold."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config, robustness_threshold=0.5)
        # Mean = 1.0, Std = ~0.0 -> 1.0 - 0.0 > 0.5
        sharpes = [1.0, 1.0, 1.0]
        runner = MockBacktestRunner(sharpes=sharpes, returns=[0.1, 0.1, 0.1])

        result = cv.run(n_samples=90, runner=runner)

        assert result.is_robust is True

    def test_is_robust_false_low_mean(self) -> None:
        """Strategy not robust when mean too low."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config, robustness_threshold=0.5)
        sharpes = [0.3, 0.3, 0.3]  # Mean = 0.3 < 0.5
        runner = MockBacktestRunner(sharpes=sharpes, returns=[0.1, 0.1, 0.1])

        result = cv.run(n_samples=90, runner=runner)

        assert result.is_robust is False

    def test_is_robust_false_high_std(self) -> None:
        """Strategy not robust when std too high."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config, robustness_threshold=0.5)
        sharpes = [0.0, 0.9, 1.8]  # Mean = 0.9, Std = 0.9 -> 0.9 - 0.9 = 0
        runner = MockBacktestRunner(sharpes=sharpes, returns=[0.1, 0.1, 0.1])

        result = cv.run(n_samples=90, runner=runner)

        assert result.is_robust is False

    def test_run_with_results(self) -> None:
        """run_with_results aggregates pre-computed results."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config)

        fold_results = [
            FoldResult(0, 60, 30, 1.5, 0.8, 0.2, 0.1),
            FoldResult(1, 60, 30, 1.3, 0.7, 0.18, 0.09),
            FoldResult(2, 60, 30, 1.4, 0.9, 0.19, 0.11),
        ]

        result = cv.run_with_results(n_samples=90, fold_results=fold_results)

        assert len(result.fold_results) == 3
        # Mean = (0.8 + 0.7 + 0.9) / 3 = 0.8
        assert abs(result.mean_oos_sharpe - 0.8) < 0.001

    def test_run_with_results_wrong_count_raises(self) -> None:
        """run_with_results raises if fold count doesn't match."""
        config = CVConfig(n_splits=5, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config)

        fold_results = [
            FoldResult(0, 60, 30, 1.5, 0.8, 0.2, 0.1),
            FoldResult(1, 60, 30, 1.3, 0.7, 0.18, 0.09),
        ]  # Only 2, expected 5

        with pytest.raises(ValueError, match="Expected 5 fold results"):
            cv.run_with_results(n_samples=100, fold_results=fold_results)

    def test_get_splits(self) -> None:
        """get_splits returns all train/test splits."""
        config = CVConfig(n_splits=5, purge_pct=0.01, embargo_pct=0.01)
        cv = PurgedKFoldCV(config=config)

        splits = cv.get_splits(n_samples=1000)

        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_single_fold_std_is_zero(self) -> None:
        """Single fold results in zero std."""
        config = CVConfig(n_splits=2, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config)

        # Manually create single result scenario via run_with_results
        # (Can't really test single fold since n_splits >= 2)
        fold_results = [
            FoldResult(0, 50, 50, 1.5, 0.8, 0.2, 0.1),
            FoldResult(1, 50, 50, 1.5, 0.8, 0.2, 0.1),  # Same as first
        ]
        result = cv.run_with_results(n_samples=100, fold_results=fold_results)

        assert result.std_oos_sharpe == 0.0


class TestPurgedKFoldCVEdgeCases:
    """Edge case tests for PurgedKFoldCV."""

    def test_empty_fold_results(self) -> None:
        """Empty fold results return zero values."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config)

        # Test internal aggregation with empty list
        result = cv._aggregate_results([])

        assert result.mean_oos_sharpe == 0.0
        assert result.std_oos_sharpe == 0.0
        assert result.is_robust is False

    def test_negative_sharpes(self) -> None:
        """Handles negative Sharpe ratios."""
        config = CVConfig(n_splits=3, purge_pct=0.0, embargo_pct=0.0)
        cv = PurgedKFoldCV(config=config, robustness_threshold=0.0)
        sharpes = [-0.5, -0.3, -0.4]
        runner = MockBacktestRunner(sharpes=sharpes, returns=[-0.1, -0.05, -0.08])

        result = cv.run(n_samples=90, runner=runner)

        assert result.mean_oos_sharpe < 0
        assert result.is_robust is False

    def test_large_n_samples(self) -> None:
        """Handles large sample count."""
        config = CVConfig(n_splits=10, purge_pct=0.01, embargo_pct=0.01)
        cv = PurgedKFoldCV(config=config)

        splits = cv.get_splits(n_samples=100000)

        assert len(splits) == 10
        total_test = sum(len(test) for _, test in splits)
        assert total_test == 100000


# --- Integration Tests ---


class TestPurgedKFoldIntegration:
    """Integration tests for full CV workflow."""

    def test_full_workflow(self) -> None:
        """Full workflow: config -> CV -> run -> result."""
        # Configure
        config = CVConfig(n_splits=5, purge_pct=0.02, embargo_pct=0.02)

        # Create CV runner
        cv = PurgedKFoldCV(config=config, robustness_threshold=0.5)

        # Mock runner with realistic-ish results
        runner = MockBacktestRunner(
            sharpes=[0.8, 0.7, 0.9, 0.6, 0.85],  # Decent OOS performance
            returns=[0.12, 0.10, 0.14, 0.08, 0.11],
        )

        # Run CV
        result = cv.run(n_samples=1000, runner=runner)

        # Verify structure
        assert len(result.fold_results) == 5
        assert result.mean_oos_sharpe > 0
        assert result.std_oos_sharpe >= 0
        assert isinstance(result.is_robust, bool)

        # Verify fold details
        for i, fold in enumerate(result.fold_results):
            assert fold.fold_index == i
            assert fold.train_size > 0
            assert fold.test_size > 0

    def test_import_from_package(self) -> None:
        """Classes importable from overfitting package."""
        from vibe_quant.overfitting import (
            CVConfig,
            CVResult,
            FoldResult,
            PurgedKFold,
            PurgedKFoldCV,
        )

        assert CVConfig is not None
        assert CVResult is not None
        assert FoldResult is not None
        assert PurgedKFold is not None
        assert PurgedKFoldCV is not None
