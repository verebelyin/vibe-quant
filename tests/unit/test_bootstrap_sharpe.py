"""Tests for bootstrap Sharpe ratio confidence interval."""

from __future__ import annotations

import numpy as np

from vibe_quant.overfitting.bootstrap_sharpe import (
    BootstrapResult,
    _sharpe_from_returns,
    bootstrap_sharpe_ci,
)


class TestSharpeFromReturns:
    def test_positive_returns(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sharpe = _sharpe_from_returns(returns)
        assert sharpe > 0

    def test_zero_std(self):
        returns = np.array([1.0, 1.0, 1.0, 1.0])
        sharpe = _sharpe_from_returns(returns)
        assert sharpe == 0.0

    def test_single_return(self):
        returns = np.array([5.0])
        sharpe = _sharpe_from_returns(returns)
        assert sharpe == 0.0

    def test_empty_returns(self):
        returns = np.array([])
        sharpe = _sharpe_from_returns(returns)
        assert sharpe == 0.0

    def test_negative_mean(self):
        returns = np.array([-1.0, -2.0, -3.0, -1.5, -2.5])
        sharpe = _sharpe_from_returns(returns)
        assert sharpe < 0


class TestBootstrapSharpeCi:
    def test_basic_positive(self):
        """Strong positive returns should produce positive CI."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 1.0, 100)  # Positive mean
        result = bootstrap_sharpe_ci(returns, n_bootstrap=1000, seed=42)

        assert isinstance(result, BootstrapResult)
        assert result.n_trades == 100
        assert result.n_bootstrap == 1000
        assert result.ci_lower < result.ci_upper
        assert result.observed_sharpe > 0

    def test_few_trades_warns(self):
        """< 5 trades should return unreliable result."""
        returns = np.array([1.0, 2.0, 3.0])
        result = bootstrap_sharpe_ci(returns, n_bootstrap=100)

        assert result.n_trades == 3
        assert result.n_bootstrap == 0  # Skipped
        assert result.passed is False

    def test_strong_signal_passes(self):
        """Very strong signal with many trades should pass min_sharpe=1.0."""
        rng = np.random.default_rng(123)
        returns = rng.normal(2.0, 1.0, 200)  # Strong positive, 200 trades
        result = bootstrap_sharpe_ci(returns, min_sharpe=1.0, n_bootstrap=5000, seed=123)

        assert result.passed is True
        assert result.ci_lower > 1.0

    def test_weak_signal_fails(self):
        """Weak signal with few trades should fail min_sharpe=1.0."""
        rng = np.random.default_rng(456)
        returns = rng.normal(0.05, 1.0, 20)  # Weak positive, only 20 trades
        result = bootstrap_sharpe_ci(returns, min_sharpe=1.0, n_bootstrap=5000, seed=456)

        assert result.passed is False
        assert result.ci_lower < 1.0

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        returns = np.random.default_rng(0).normal(0.5, 1.0, 50)
        r1 = bootstrap_sharpe_ci(returns, n_bootstrap=500, seed=99)
        r2 = bootstrap_sharpe_ci(returns, n_bootstrap=500, seed=99)

        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_list_input(self):
        """Should accept list as well as ndarray."""
        returns_list = [0.5, 1.0, -0.3, 0.8, 0.2, 1.1, -0.1, 0.6, 0.4, 0.9]
        result = bootstrap_sharpe_ci(returns_list, n_bootstrap=100, seed=42)
        assert result.n_trades == 10
        assert result.n_bootstrap == 100

    def test_ci_level_custom(self):
        """Custom CI level should be reflected in result."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 1.0, 100)
        result_90 = bootstrap_sharpe_ci(returns, ci_level=0.90, n_bootstrap=1000, seed=42)
        result_99 = bootstrap_sharpe_ci(returns, ci_level=0.99, n_bootstrap=1000, seed=42)

        assert result_90.ci_level == 0.90
        assert result_99.ci_level == 0.99
        # 99% CI should be wider than 90%
        width_90 = result_90.ci_upper - result_90.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower
        assert width_99 > width_90

    def test_bootstrap_sharpes_array(self):
        """Result should contain full bootstrap distribution."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 1.0, 50)
        result = bootstrap_sharpe_ci(returns, n_bootstrap=500, seed=42)

        assert len(result.bootstrap_sharpes) == 500
        assert result.bootstrap_sharpes.dtype == np.float64
