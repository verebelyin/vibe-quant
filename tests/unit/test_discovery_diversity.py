"""Tests for population diversity metrics and interventions."""

import math

import pytest

from vibe_quant.discovery.diversity import (
    population_entropy,
    should_inject_immigrants,
    inject_random_immigrants,
)
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)


def _gene(ind: str = "RSI") -> StrategyGene:
    params = {"period": 14.0}
    if ind == "MACD":
        params = {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 9.0}
    elif ind == "STOCH":
        params = {"k_period": 14.0, "d_period": 3.0}
    return StrategyGene(
        indicator_type=ind, parameters=params,
        condition=ConditionType.GT, threshold=50.0,
    )


def _chrom(indicator: str = "RSI", direction: Direction = Direction.LONG) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_gene(indicator)],
        exit_genes=[_gene(indicator)],
        stop_loss_pct=2.0, take_profit_pct=5.0, direction=direction,
    )


class TestPopulationEntropy:
    """Tests for Shannon entropy of population."""

    def test_monoculture_zero_entropy(self) -> None:
        """All-identical population has zero entropy."""
        pop = [_chrom("RSI", Direction.LONG) for _ in range(10)]
        ent = population_entropy(pop)
        assert ent == pytest.approx(0.0, abs=1e-6)

    def test_diverse_population_high_entropy(self) -> None:
        """Population with varied indicators has high entropy."""
        indicators = ["RSI", "ATR", "CCI", "MFI", "ADX", "STOCH", "WILLR", "ROC"]
        pop = [_chrom(indicators[i % len(indicators)]) for i in range(16)]
        ent = population_entropy(pop)
        assert ent > 0.3

    def test_entropy_between_0_and_1(self) -> None:
        """Entropy is normalized to [0, 1]."""
        pop = [_chrom("RSI")] * 5 + [_chrom("CCI")] * 5
        ent = population_entropy(pop)
        assert 0.0 <= ent <= 1.0

    def test_single_chromosome_zero_entropy(self) -> None:
        """Single individual has zero entropy (no variation)."""
        pop = [_chrom()]
        ent = population_entropy(pop)
        assert ent == pytest.approx(0.0, abs=1e-6)


class TestShouldInjectImmigrants:
    """Tests for injection trigger."""

    def test_low_entropy_triggers_injection(self) -> None:
        assert should_inject_immigrants(0.1, threshold=0.3)

    def test_high_entropy_no_injection(self) -> None:
        assert not should_inject_immigrants(0.8, threshold=0.3)

    def test_exactly_at_threshold_no_injection(self) -> None:
        assert not should_inject_immigrants(0.3, threshold=0.3)


class TestInjectRandomImmigrants:
    """Tests for random immigrant injection."""

    def test_replaces_bottom_n(self) -> None:
        pop = [_chrom("RSI") for _ in range(10)]
        scores = [float(i) for i in range(10)]
        new_pop = inject_random_immigrants(pop, scores, fraction=0.2)
        assert len(new_pop) == 10

    def test_preserves_population_size(self) -> None:
        pop = [_chrom() for _ in range(20)]
        scores = [0.5] * 20
        new_pop = inject_random_immigrants(pop, scores, fraction=0.1)
        assert len(new_pop) == 20

    def test_at_least_one_immigrant(self) -> None:
        pop = [_chrom() for _ in range(5)]
        scores = [0.5] * 5
        new_pop = inject_random_immigrants(pop, scores, fraction=0.01)
        assert len(new_pop) == 5
