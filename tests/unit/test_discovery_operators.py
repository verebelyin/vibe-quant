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
