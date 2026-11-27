"""Tests for chromosome Gower distance metric."""

import pytest

from vibe_quant.discovery.distance import chromosome_distance, gene_distance
from vibe_quant.discovery.exit_genes import ExitBlock, ExitRules
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)


def _gene(
    ind: str = "RSI",
    period: int = 14,
    condition: ConditionType = ConditionType.GT,
    threshold: float = 50.0,
    sub_value: str | None = None,
) -> StrategyGene:
    """Helper to create a gene with minimal boilerplate."""
    params: dict[str, float] = {"period": float(period)}
    if ind == "MACD":
        params = {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 9.0}
    elif ind == "STOCH":
        params = {"period_k": float(period), "period_d": 3.0}
    return StrategyGene(
        indicator_type=ind,
        parameters=params,
        condition=condition,
        threshold=threshold,
        sub_value=sub_value,
    )


def _chrom(
    entry_genes: list[StrategyGene],
    exit_genes: list[StrategyGene],
    sl: float = 2.0,
    tp: float = 5.0,
    direction: Direction = Direction.LONG,
) -> StrategyChromosome:
    """Helper to create a chromosome."""
    return StrategyChromosome(
        entry_genes=entry_genes,
        exit_genes=exit_genes,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        direction=direction,
    )


class TestGeneDistance:
    """Tests for single gene distance."""

    def test_identical_genes_zero_distance(self) -> None:
        g = _gene("RSI", 14, ConditionType.GT, 50.0)
        assert gene_distance(g, g) == pytest.approx(0.0)

    def test_identical_out_of_pool_genes_zero_distance(self) -> None:
        # ATR left the GA pool (NATR replaced it); legacy warm-start genes
        # must still compare as identical to themselves.
        a = _gene("ATR", 14, ConditionType.GT, 40.0)
        b = _gene("ATR", 14, ConditionType.GT, 40.0)
        assert gene_distance(a, b) == pytest.approx(0.0)
        assert gene_distance(a, _gene("ATR", 20, ConditionType.GT, 40.0)) > 0.0

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
        # NATR replaced ATR in the GA pool (regression: vibe-quant-e70tl.14)
        c = _chrom([_gene()], [_gene("NATR", 14, ConditionType.LT, 1.0)])
        assert chromosome_distance(c, c) == pytest.approx(0.0)

    def test_completely_different_chromosomes(self) -> None:
        a = _chrom(
            [_gene("RSI", 14, ConditionType.GT, 50.0)],
            [_gene("ATR", 14, ConditionType.LT, 0.01)],
            sl=1.0,
            tp=1.0,
            direction=Direction.LONG,
        )
        b = _chrom(
            [
                _gene("CCI", 50, ConditionType.CROSSES_BELOW, -200.0),
                _gene("STOCH", 21, ConditionType.LTE, 20.0),
            ],
            [_gene("MFI", 30, ConditionType.GT, 80.0)],
            sl=10.0,
            tp=20.0,
            direction=Direction.SHORT,
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
            sl=10.0,
            tp=20.0,
            direction=Direction.SHORT,
        )
        d = chromosome_distance(a, b)
        assert 0.0 <= d <= 1.0


def _ext(
    block: ExitBlock,
    short: ExitBlock | None = None,
    max_bars: int | None = None,
    direction: Direction = Direction.LONG,
) -> StrategyChromosome:
    """Same genes everywhere, so only the exit rules can move the distance."""
    chrom = _chrom([_gene()], [_gene("ATR")], sl=2.0, tp=3.0, direction=direction)
    chrom.exit_rules = ExitRules(block, short, max_bars)
    return chrom


_TOTAL_WEIGHT = 5.0  # genes 3 + direction 1 + SL 0.5 + TP 0.5


class TestExitRulesDistance:
    def test_legacy_pair_is_todays_float(self) -> None:
        a = _chrom([_gene()], [_gene("ATR")], sl=2.0, tp=5.0)
        b = _chrom([_gene()], [_gene("ATR")], sl=6.0, tp=15.0)
        expected = (0.5 * (4.0 / 9.5) + 0.5 * (10.0 / 19.5)) / _TOTAL_WEIGHT
        assert chromosome_distance(a, b) == expected

    def test_same_rules_distance_zero(self) -> None:
        block = ExitBlock("atr_trailing", 2.5, "risk_reward", 2.0, 14)
        assert chromosome_distance(_ext(block, max_bars=12), _ext(block, max_bars=12)) == 0.0

    def test_stop_type_differs_only(self) -> None:
        a = _ext(ExitBlock("fixed_pct", 2.0, "fixed_pct", 4.0, None))
        b = _ext(ExitBlock("safety", None, "fixed_pct", 4.0, None))
        assert chromosome_distance(a, b) == pytest.approx(0.5 / _TOTAL_WEIGHT, abs=1e-15)

    def test_time_exit_one_off_moves_the_target_slot(self) -> None:
        block = ExitBlock("fixed_pct", 2.0, "fixed_pct", 4.0, None)
        # time distance 1.0 averaged with target distance 0.0 -> 0.5, weight 0.5
        assert chromosome_distance(_ext(block), _ext(block, max_bars=12)) == pytest.approx(
            0.5 * 0.5 / _TOTAL_WEIGHT, abs=1e-15
        )

    def test_same_type_values_are_range_normalised(self) -> None:
        a = _ext(ExitBlock("fixed_pct", 2.0, "atr_fixed", 2.0, 14), max_bars=10)
        b = _ext(ExitBlock("fixed_pct", 6.0, "atr_fixed", 6.0, 14), max_bars=40)
        stop = 4.0 / 9.5  # STOP_PCT span
        target = (4.0 / 9.0 + 30.0 / 40.0) / 2  # TARGET_ATR_MULT span 1..10; |10-40| / 40
        expected = (0.5 * stop + 0.5 * target) / _TOTAL_WEIGHT
        assert chromosome_distance(a, b) == pytest.approx(expected, abs=1e-15)

    def test_both_averages_long_and_short(self) -> None:
        fixed = ExitBlock("fixed_pct", 2.0, "fixed_pct", 4.0, None)
        trailing = ExitBlock("atr_trailing", 2.0, "fixed_pct", 4.0, 14)
        a = _ext(fixed, fixed, direction=Direction.BOTH)
        b = _ext(fixed, trailing, direction=Direction.BOTH)
        # only the short stop differs by type: (0 + 1) / 2 on the stop slot
        assert chromosome_distance(a, b) == pytest.approx(0.5 * 0.5 / _TOTAL_WEIGHT, abs=1e-15)

    def test_symmetry_and_bound(self) -> None:
        a = _ext(ExitBlock("atr_fixed", 1.0, "none", None, 7), max_bars=3)
        b = _ext(ExitBlock("safety", None, "atr_fixed", 10.0, 30))
        assert chromosome_distance(a, b) == chromosome_distance(b, a)
        assert 0.0 < chromosome_distance(a, b) <= 1.0
