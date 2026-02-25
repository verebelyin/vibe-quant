"""Tests for vibe_quant.discovery.guardrails."""

from __future__ import annotations

from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.guardrails import (
    GuardrailConfig,
    GuardrailResult,
    apply_discovery_dsr,
    apply_guardrails,
    check_complexity,
    check_min_trades,
)
from vibe_quant.overfitting.dsr import DSRResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fitness(
    *,
    sharpe: float = 1.5,
    max_dd: float = 0.1,
    pf: float = 2.0,
    trades: int = 100,
) -> FitnessResult:
    return FitnessResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        total_trades=trades,
        total_return=0.2,
        complexity_penalty=0.0,
        overtrade_penalty=0.0,
        raw_score=0.7,
        adjusted_score=0.7,
        passed_filters=True,
        filter_results={},
    )


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestGuardrailConfig:
    def test_defaults(self) -> None:
        cfg = GuardrailConfig()
        assert cfg.min_trades == 50
        assert cfg.max_complexity == 8
        assert cfg.require_dsr is True
        assert cfg.dsr_significance_level == 0.05
        assert cfg.require_wfa is True
        assert cfg.wfa_min_efficiency == 0.5
        assert cfg.require_purged_kfold is False

    def test_custom_values(self) -> None:
        cfg = GuardrailConfig(min_trades=100, max_complexity=6)
        assert cfg.min_trades == 100
        assert cfg.max_complexity == 6


# ---------------------------------------------------------------------------
# Min trade filter
# ---------------------------------------------------------------------------


class TestMinTrades:
    def test_pass(self) -> None:
        passed, reason = check_min_trades(100, 50)
        assert passed is True
        assert reason is None

    def test_exact_threshold(self) -> None:
        passed, reason = check_min_trades(50, 50)
        assert passed is True
        assert reason is None

    def test_fail(self) -> None:
        passed, reason = check_min_trades(30, 50)
        assert passed is False
        assert reason is not None
        assert "30" in reason
        assert "50" in reason


# ---------------------------------------------------------------------------
# Complexity guard
# ---------------------------------------------------------------------------


class TestComplexity:
    def test_pass(self) -> None:
        passed, reason = check_complexity(6, 8)
        assert passed is True
        assert reason is None

    def test_exact_threshold(self) -> None:
        passed, reason = check_complexity(8, 8)
        assert passed is True
        assert reason is None

    def test_fail(self) -> None:
        passed, reason = check_complexity(10, 8)
        assert passed is False
        assert reason is not None
        assert "10" in reason
        assert "8" in reason


# ---------------------------------------------------------------------------
# DSR correction
# ---------------------------------------------------------------------------


class TestDiscoveryDSR:
    def test_significant_with_few_trials(self) -> None:
        """Strong Sharpe + few trials -> significant."""
        result = apply_discovery_dsr(
            observed_sharpe=2.5,
            num_trials=5,
            num_observations=500,
        )
        assert isinstance(result, DSRResult)
        assert result.is_significant is True
        assert result.p_value < 0.05

    def test_insignificant_with_many_trials(self) -> None:
        """Mediocre Sharpe + many trials -> not significant."""
        result = apply_discovery_dsr(
            observed_sharpe=0.8,
            num_trials=5000,
            num_observations=252,
        )
        assert isinstance(result, DSRResult)
        assert result.is_significant is False
        assert result.num_trials == 5000

    def test_custom_significance_level(self) -> None:
        result = apply_discovery_dsr(
            observed_sharpe=2.0,
            num_trials=10,
            num_observations=500,
            significance_level=0.01,
        )
        assert isinstance(result, DSRResult)
        # With 10 trials and Sharpe=2, should still be significant at 1%
        assert result.is_significant is True


# ---------------------------------------------------------------------------
# Combined guardrails
# ---------------------------------------------------------------------------


class TestApplyGuardrails:
    def test_all_pass_no_optional(self) -> None:
        """All basic checks pass, optional checks disabled."""
        cfg = GuardrailConfig(
            require_dsr=False,
            require_wfa=False,
            require_purged_kfold=False,
        )
        fitness = _make_fitness(trades=100)
        result = apply_guardrails(fitness, num_genes=4, config=cfg)

        assert result.passed is True
        assert result.min_trades_passed is True
        assert result.complexity_passed is True
        assert result.dsr_passed is None
        assert result.wfa_passed is None
        assert result.kfold_passed is None
        assert result.reasons == []

    def test_min_trades_fails(self) -> None:
        cfg = GuardrailConfig(
            require_dsr=False,
            require_wfa=False,
        )
        fitness = _make_fitness(trades=10)
        result = apply_guardrails(fitness, num_genes=3, config=cfg)

        assert result.passed is False
        assert result.min_trades_passed is False
        assert result.complexity_passed is True
        assert len(result.reasons) == 1

    def test_complexity_fails(self) -> None:
        cfg = GuardrailConfig(
            require_dsr=False,
            require_wfa=False,
            max_complexity=4,
        )
        fitness = _make_fitness(trades=100)
        result = apply_guardrails(fitness, num_genes=6, config=cfg)

        assert result.passed is False
        assert result.min_trades_passed is True
        assert result.complexity_passed is False
        assert len(result.reasons) == 1

    def test_dsr_pass(self) -> None:
        """DSR enabled + strong Sharpe + few trials -> pass."""
        cfg = GuardrailConfig(
            require_dsr=True,
            require_wfa=False,
        )
        fitness = _make_fitness(sharpe=3.0, trades=200)
        result = apply_guardrails(
            fitness,
            num_genes=4,
            config=cfg,
            num_trials=10,
            num_observations=500,
        )

        assert result.dsr_passed is True
        assert result.dsr_result is not None
        assert result.passed is True

    def test_dsr_fail(self) -> None:
        """DSR enabled + weak Sharpe + many trials -> fail."""
        cfg = GuardrailConfig(
            require_dsr=True,
            require_wfa=False,
        )
        fitness = _make_fitness(sharpe=0.5, trades=100)
        result = apply_guardrails(
            fitness,
            num_genes=4,
            config=cfg,
            num_trials=5000,
            num_observations=252,
        )

        assert result.dsr_passed is False
        assert result.passed is False
        assert any("DSR" in r for r in result.reasons)

    def test_wfa_required_but_not_provided(self) -> None:
        """WFA required but no WFA instance -> fail."""
        cfg = GuardrailConfig(
            require_dsr=False,
            require_wfa=True,
        )
        fitness = _make_fitness(trades=100)
        result = apply_guardrails(fitness, num_genes=4, config=cfg)

        assert result.wfa_passed is False
        assert result.passed is False
        assert any("WFA" in r for r in result.reasons)

    def test_kfold_required_but_not_provided(self) -> None:
        """K-Fold required but no CV instance -> fail."""
        cfg = GuardrailConfig(
            require_dsr=False,
            require_wfa=False,
            require_purged_kfold=True,
        )
        fitness = _make_fitness(trades=100)
        result = apply_guardrails(fitness, num_genes=4, config=cfg)

        assert result.kfold_passed is False
        assert result.passed is False

    def test_disabled_checks_skipped(self) -> None:
        """Disabled checks return None, not False."""
        cfg = GuardrailConfig(
            require_dsr=False,
            require_wfa=False,
            require_purged_kfold=False,
        )
        fitness = _make_fitness(trades=100)
        result = apply_guardrails(fitness, num_genes=4, config=cfg)

        assert result.dsr_passed is None
        assert result.wfa_passed is None
        assert result.kfold_passed is None
        assert result.dsr_result is None
        assert result.wfa_result is None
        assert result.kfold_result is None

    def test_multiple_failures(self) -> None:
        """Both trades and complexity fail -> multiple reasons."""
        cfg = GuardrailConfig(
            min_trades=100,
            max_complexity=3,
            require_dsr=False,
            require_wfa=False,
        )
        fitness = _make_fitness(trades=20)
        result = apply_guardrails(fitness, num_genes=5, config=cfg)

        assert result.passed is False
        assert result.min_trades_passed is False
        assert result.complexity_passed is False
        assert len(result.reasons) == 2


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------


class TestGuardrailResult:
    def test_dataclass_fields(self) -> None:
        r = GuardrailResult(
            passed=True,
            min_trades_passed=True,
            complexity_passed=True,
        )
        assert r.passed is True
        assert r.dsr_passed is None
        assert r.wfa_passed is None
        assert r.kfold_passed is None
        assert r.reasons == []
