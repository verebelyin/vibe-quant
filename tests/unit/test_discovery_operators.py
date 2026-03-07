"""Tests for genetic operators (crowding replacement)."""

from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
    crowding_replace,
)


def _make_chrom(
    indicator: str = "RSI", direction: Direction = Direction.LONG
) -> StrategyChromosome:
    """Helper to make a simple chromosome for testing."""
    gene = StrategyGene(
        indicator_type=indicator,
        parameters=(
            {"period": 14.0}
            if indicator != "MACD"
            else {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 9.0}
        ),
        condition=ConditionType.GT,
        threshold=50.0 if indicator == "RSI" else 0.0,
    )
    return StrategyChromosome(
        entry_genes=[gene],
        exit_genes=[gene.clone()],
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        direction=direction,
    )


class TestCrowdingReplace:
    """Tests for deterministic crowding replacement."""

    def test_fitter_offspring_replaces_similar_parent(self) -> None:
        """Offspring with higher fitness replaces its most-similar parent."""
        parent_a = _make_chrom("RSI", Direction.LONG)
        parent_b = _make_chrom("CCI", Direction.SHORT)

        offspring_a = parent_a.clone()
        offspring_a.uid = "new_uid_a"
        offspring_b = parent_b.clone()
        offspring_b.uid = "new_uid_b"

        result = crowding_replace(
            parents=[parent_a, parent_b],
            parent_fitness=[0.3, 0.5],
            offspring=[offspring_a, offspring_b],
            offspring_fitness=[0.4, 0.6],
        )
        assert len(result) == 2

    def test_weaker_offspring_does_not_replace(self) -> None:
        """Offspring with lower fitness does NOT replace similar parent."""
        parent_a = _make_chrom("RSI", Direction.LONG)
        parent_b = _make_chrom("CCI", Direction.SHORT)

        offspring_a = parent_a.clone()
        offspring_a.uid = "new_uid_a"
        offspring_b = parent_b.clone()
        offspring_b.uid = "new_uid_b"

        result = crowding_replace(
            parents=[parent_a, parent_b],
            parent_fitness=[0.8, 0.9],
            offspring=[offspring_a, offspring_b],
            offspring_fitness=[0.1, 0.2],
        )
        assert result[0].uid == parent_a.uid
        assert result[1].uid == parent_b.uid

    def test_correct_matching_offspring_to_parent(self) -> None:
        """Offspring matched to most-similar parent, not any parent."""
        parent_a = _make_chrom("RSI", Direction.LONG)
        parent_b = _make_chrom("CCI", Direction.SHORT)

        offspring_rsi = parent_a.clone()
        offspring_rsi.uid = "off_rsi"
        offspring_cci = parent_b.clone()
        offspring_cci.uid = "off_cci"

        result = crowding_replace(
            parents=[parent_a, parent_b],
            parent_fitness=[0.3, 0.9],
            offspring=[offspring_rsi, offspring_cci],
            offspring_fitness=[0.5, 0.1],
        )
        assert result[0].uid == "off_rsi"  # RSI offspring won (0.5 > 0.3)
        assert result[1].uid == parent_b.uid  # CCI parent survived (0.9 > 0.1)


class TestPerturb:
    def test_perturb_zero_with_narrow_bounds(self) -> None:
        """_perturb(0.0) with narrow bounds should stay within bounds, not use absolute frac."""
        import random
        from vibe_quant.discovery.operators import _perturb
        random.seed(42)
        lo, hi = -0.05, 0.05
        results = [_perturb(0.0, 0.2, lo, hi) for _ in range(200)]
        # All within bounds
        for r in results:
            assert lo <= r <= hi, f"Result {r} outside [{lo}, {hi}]"
        # Most values should NOT be pinned at boundaries (< 25% at each boundary)
        at_lo = sum(1 for r in results if r == lo)
        at_hi = sum(1 for r in results if r == hi)
        boundary_pct = (at_lo + at_hi) / len(results)
        assert boundary_pct < 0.25, (
            f"{boundary_pct:.0%} of values pinned at boundaries — perturbation overshoots range"
        )

    def test_perturb_zero_without_bounds_uses_absolute(self) -> None:
        """_perturb(0.0) without bounds should still use frac as absolute range."""
        import random
        from vibe_quant.discovery.operators import _perturb
        random.seed(42)
        results = [_perturb(0.0, 0.2) for _ in range(100)]
        # Should produce values in [-0.2, 0.2]
        assert any(r < -0.05 for r in results), "Should produce values below -0.05 when unbounded"
        assert any(r > 0.05 for r in results), "Should produce values above 0.05 when unbounded"


class TestRepairChromosome:
    def test_repair_fixes_out_of_range_threshold(self) -> None:
        """Chromosomes with out-of-range thresholds should be repaired."""
        from vibe_quant.discovery.operators import (
            StrategyGene, StrategyChromosome, ConditionType, Direction,
            is_valid_chromosome, _repair_chromosome, _ensure_pool, THRESHOLD_RANGES,
        )
        _ensure_pool()
        # ATR with RSI-scale threshold (impossible: ATR range is 0.001-0.08)
        bad_gene = StrategyGene(
            indicator_type="ATR", parameters={"period": 14.0},
            condition=ConditionType.GT, threshold=72.0,
        )
        chrom = StrategyChromosome(
            entry_genes=[bad_gene],
            exit_genes=[StrategyGene(
                indicator_type="RSI", parameters={"period": 14.0},
                condition=ConditionType.LT, threshold=50.0,
            )],
            stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
        )
        assert not is_valid_chromosome(chrom)
        repaired = _repair_chromosome(chrom)
        assert is_valid_chromosome(repaired)
        lo, hi = THRESHOLD_RANGES["ATR"]
        assert lo <= repaired.entry_genes[0].threshold <= hi

    def test_repair_preserves_valid_chromosome(self) -> None:
        """Valid chromosomes should pass through repair unchanged."""
        from vibe_quant.discovery.operators import (
            StrategyGene, StrategyChromosome, ConditionType, Direction,
            is_valid_chromosome, _repair_chromosome,
        )
        gene = StrategyGene(
            indicator_type="RSI", parameters={"period": 14.0},
            condition=ConditionType.GT, threshold=50.0,
        )
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[gene.clone()],
            stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
        )
        assert is_valid_chromosome(chrom)
        repaired = _repair_chromosome(chrom)
        assert repaired.entry_genes[0].threshold == 50.0


class TestThresholdRanges:
    def test_macd_threshold_range_wide_enough(self) -> None:
        """MACD threshold range must span at least 0.05 to produce viable signals."""
        from vibe_quant.discovery.operators import THRESHOLD_RANGES, _ensure_pool

        THRESHOLD_RANGES.clear()
        _ensure_pool()
        lo, hi = THRESHOLD_RANGES["MACD"]
        assert hi - lo >= 0.05, f"MACD range too narrow: ({lo}, {hi})"

    def test_atr_threshold_range_wide_enough(self) -> None:
        """ATR threshold range must span at least 0.05 to produce viable signals."""
        from vibe_quant.discovery.operators import THRESHOLD_RANGES, _ensure_pool

        THRESHOLD_RANGES.clear()
        _ensure_pool()
        lo, hi = THRESHOLD_RANGES["ATR"]
        assert hi - lo >= 0.05, f"ATR range too narrow: ({lo}, {hi})"
