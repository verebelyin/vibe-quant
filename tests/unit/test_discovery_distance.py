"""Tests for chromosome Gower distance metric."""

import pytest

from vibe_quant.discovery.distance import gene_distance, chromosome_distance
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)


def _gene(ind: str = "RSI", period: int = 14, condition: ConditionType = ConditionType.GT,
          threshold: float = 50.0, sub_value: str | None = None) -> StrategyGene:
    """Helper to create a gene with minimal boilerplate."""
    params: dict[str, float] = {"period": float(period)}
    if ind == "MACD":
        params = {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 9.0}
    elif ind == "STOCH":
        params = {"k_period": float(period), "d_period": 3.0}
    return StrategyGene(
        indicator_type=ind, parameters=params, condition=condition,
        threshold=threshold, sub_value=sub_value,
    )


def _chrom(entry_genes: list[StrategyGene], exit_genes: list[StrategyGene],
           sl: float = 2.0, tp: float = 5.0,
           direction: Direction = Direction.LONG) -> StrategyChromosome:
    """Helper to create a chromosome."""
    return StrategyChromosome(
        entry_genes=entry_genes, exit_genes=exit_genes,
        stop_loss_pct=sl, take_profit_pct=tp, direction=direction,
    )


class TestGeneDistance:
    """Tests for single gene distance."""

    def test_identical_genes_zero_distance(self) -> None:
        g = _gene("RSI", 14, ConditionType.GT, 50.0)
        assert gene_distance(g, g) == pytest.approx(0.0)

    def test_different_indicator_high_distance(self) -> None:
        a = _gene("RSI", 14, ConditionType.GT, 50.0)
        b = _gene("MACD", 12, ConditionType.GT, 0.001)
        d = gene_distance(a, b)
        assert d > 0.3
        assert d <= 1.0

    def test_same_indicator_different_params(self) -> None:
        a = _gene("RSI", 5, ConditionType.GT, 30.0)
        b = _gene("RSI", 50, ConditionType.GT, 70.0)
        d = gene_distance(a, b)
        assert 0.0 < d < 0.7

    def test_different_condition_only(self) -> None:
        a = _gene("RSI", 14, ConditionType.GT, 50.0)
        b = _gene("RSI", 14, ConditionType.LT, 50.0)
        d = gene_distance(a, b)
        assert 0.0 < d < 0.5

    def test_symmetry(self) -> None:
        a = _gene("RSI", 14, ConditionType.GT, 50.0)
        b = _gene("ATR", 20, ConditionType.LT, 0.01)
        assert gene_distance(a, b) == pytest.approx(gene_distance(b, a))

    def test_distance_bounded_0_1(self) -> None:
        a = _gene("RSI", 5, ConditionType.GT, 25.0)
        b = _gene("CCI", 50, ConditionType.CROSSES_BELOW, -200.0)
        d = gene_distance(a, b)
        assert 0.0 <= d <= 1.0


class TestChromosomeDistance:
    """Tests for full chromosome distance."""

    def test_identical_chromosomes_zero(self) -> None:
        c = _chrom([_gene()], [_gene("ATR", 14, ConditionType.LT, 0.01)])
        assert chromosome_distance(c, c) == pytest.approx(0.0)

    def test_completely_different_chromosomes(self) -> None:
        a = _chrom(
            [_gene("RSI", 14, ConditionType.GT, 50.0)],
            [_gene("ATR", 14, ConditionType.LT, 0.01)],
            sl=1.0, tp=1.0, direction=Direction.LONG,
        )
        b = _chrom(
            [_gene("CCI", 50, ConditionType.CROSSES_BELOW, -200.0),
             _gene("STOCH", 21, ConditionType.LTE, 20.0)],
            [_gene("MFI", 30, ConditionType.GT, 80.0)],
            sl=10.0, tp=20.0, direction=Direction.SHORT,
        )
        d = chromosome_distance(a, b)
        assert d > 0.5

    def test_same_structure_different_params(self) -> None:
        a = _chrom(
            [_gene("RSI", 10, ConditionType.GT, 30.0)],
            [_gene("ATR", 10, ConditionType.LT, 0.005)],
        )
        b = _chrom(
            [_gene("RSI", 40, ConditionType.GT, 70.0)],
            [_gene("ATR", 25, ConditionType.LT, 0.025)],
        )
        d = chromosome_distance(a, b)
        assert 0.0 < d < 0.5

    def test_different_gene_count_penalized(self) -> None:
        a = _chrom([_gene()], [_gene("ATR")])
        b = _chrom(
            [_gene(), _gene("MACD"), _gene("CCI")],
            [_gene("ATR")],
        )
        d = chromosome_distance(a, b)
        assert d > 0.1

    def test_direction_mismatch_adds_distance(self) -> None:
        a = _chrom([_gene()], [_gene("ATR")], direction=Direction.LONG)
        b = _chrom([_gene()], [_gene("ATR")], direction=Direction.SHORT)
        d = chromosome_distance(a, b)
        assert d > 0.0

    def test_symmetry(self) -> None:
        a = _chrom([_gene("RSI")], [_gene("ATR")])
        b = _chrom([_gene("CCI"), _gene("STOCH")], [_gene("MFI")])
        assert chromosome_distance(a, b) == pytest.approx(chromosome_distance(b, a))

    def test_distance_bounded_0_1(self) -> None:
        a = _chrom([_gene("RSI")], [_gene("ATR")], sl=0.5, tp=0.5, direction=Direction.LONG)
        b = _chrom(
            [_gene("CCI"), _gene("STOCH"), _gene("MFI"), _gene("ADX"), _gene("WILLR")],
            [_gene("ROC"), _gene("MACD"), _gene("RSI")],
            sl=10.0, tp=20.0, direction=Direction.SHORT,
        )
        d = chromosome_distance(a, b)
        assert 0.0 <= d <= 1.0
