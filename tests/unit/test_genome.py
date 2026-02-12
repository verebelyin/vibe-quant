"""Tests for strategy genome representation."""

from __future__ import annotations

import random

import pytest

from vibe_quant.discovery.genome import (
    INDICATOR_POOL,
    VALID_CONDITIONS,
    VALID_DIRECTIONS,
    StrategyChromosome,
    StrategyGene,
    chromosome_to_dsl,
    generate_random_chromosome,
    validate_chromosome,
)
from vibe_quant.dsl.schema import StrategyDSL

# =============================================================================
# Gene creation
# =============================================================================


class TestStrategyGene:
    def test_create_basic(self) -> None:
        gene = StrategyGene(
            indicator_type="RSI",
            parameters={"period": 14},
            condition="crosses_below",
            threshold=30.0,
        )
        assert gene.indicator_type == "RSI"
        assert gene.parameters == {"period": 14}
        assert gene.condition == "crosses_below"
        assert gene.threshold == 30.0

    def test_frozen(self) -> None:
        gene = StrategyGene("EMA", {"period": 20}, "greater_than", 0.0)
        with pytest.raises(AttributeError):
            gene.threshold = 50.0  # type: ignore[misc]

    def test_all_indicators_representable(self) -> None:
        """Every pool indicator can be instantiated as a gene."""
        for name, ind_def in INDICATOR_POOL.items():
            params: dict[str, int | float] = {
                k: int(lo) for k, (lo, _hi) in ind_def.param_ranges.items()
            }
            gene = StrategyGene(name, params, "greater_than", 0.0)
            assert gene.indicator_type == name


# =============================================================================
# Chromosome creation
# =============================================================================


class TestStrategyChromosome:
    def test_create_minimal(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        assert len(chrom.entry_genes) == 1
        assert len(chrom.exit_genes) == 1
        assert chrom.direction == "long"
        assert len(chrom.uid) == 12

    def test_uid_unique(self) -> None:
        uids = {
            StrategyChromosome(
                entry_genes=[StrategyGene("RSI", {"period": 14}, "less_than", 30.0)],
                exit_genes=[StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)],
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
            ).uid
            for _ in range(100)
        }
        assert len(uids) == 100


# =============================================================================
# Random generation
# =============================================================================


class TestRandomGeneration:
    def test_generates_valid_structure(self) -> None:
        rng = random.Random(42)
        chrom = generate_random_chromosome(rng)
        assert 1 <= len(chrom.entry_genes) <= 5
        assert 1 <= len(chrom.exit_genes) <= 3
        assert 0.5 <= chrom.stop_loss_pct <= 10.0
        assert 0.5 <= chrom.take_profit_pct <= 20.0
        assert chrom.direction in VALID_DIRECTIONS

    def test_1000_samples_all_valid(self) -> None:
        rng = random.Random(123)
        for _ in range(1000):
            chrom = generate_random_chromosome(rng)
            errors = validate_chromosome(chrom)
            assert errors == [], f"Invalid chromosome: {errors}"

    def test_deterministic_with_seed(self) -> None:
        a = generate_random_chromosome(random.Random(99))
        b = generate_random_chromosome(random.Random(99))
        assert a.entry_genes == b.entry_genes
        assert a.exit_genes == b.exit_genes
        assert a.stop_loss_pct == b.stop_loss_pct

    def test_genes_use_pool_indicators(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            chrom = generate_random_chromosome(rng)
            for gene in chrom.entry_genes + chrom.exit_genes:
                assert gene.indicator_type in INDICATOR_POOL
                assert gene.condition in VALID_CONDITIONS


# =============================================================================
# Validation
# =============================================================================


class TestValidation:
    def _make_gene(
        self,
        ind: str = "RSI",
        params: dict[str, int | float] | None = None,
        cond: str = "less_than",
        thr: float = 30.0,
    ) -> StrategyGene:
        if params is None:
            params = {"period": 14}
        return StrategyGene(ind, params, cond, thr)

    def test_valid_chromosome_passes(self) -> None:
        chrom = StrategyChromosome(
            entry_genes=[self._make_gene()],
            exit_genes=[self._make_gene(cond="greater_than", thr=70.0)],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        assert validate_chromosome(chrom) == []

    def test_no_entry_genes(self) -> None:
        chrom = StrategyChromosome(
            entry_genes=[],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("entry" in e.lower() for e in errors)

    def test_no_exit_genes(self) -> None:
        chrom = StrategyChromosome(
            entry_genes=[self._make_gene()],
            exit_genes=[],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("exit" in e.lower() for e in errors)

    def test_too_many_entry_genes(self) -> None:
        chrom = StrategyChromosome(
            entry_genes=[self._make_gene() for _ in range(6)],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("5" in e and "entry" in e.lower() for e in errors)

    def test_unknown_indicator(self) -> None:
        gene = StrategyGene("UNKNOWN_IND", {"period": 10}, "less_than", 50.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("unknown" in e.lower() for e in errors)

    def test_param_out_of_range(self) -> None:
        gene = StrategyGene("RSI", {"period": 999}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("out of range" in e for e in errors)

    def test_missing_param(self) -> None:
        gene = StrategyGene("RSI", {}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("missing" in e.lower() for e in errors)

    def test_invalid_condition(self) -> None:
        gene = StrategyGene("RSI", {"period": 14}, "INVALID_OP", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("condition" in e.lower() for e in errors)

    def test_sl_out_of_range(self) -> None:
        chrom = StrategyChromosome(
            entry_genes=[self._make_gene()],
            exit_genes=[self._make_gene()],
            stop_loss_pct=15.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("stop_loss" in e for e in errors)

    def test_tp_out_of_range(self) -> None:
        chrom = StrategyChromosome(
            entry_genes=[self._make_gene()],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=25.0,
        )
        errors = validate_chromosome(chrom)
        assert any("take_profit" in e for e in errors)

    def test_invalid_direction(self) -> None:
        chrom = StrategyChromosome(
            entry_genes=[self._make_gene()],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
            direction="sideways",
        )
        errors = validate_chromosome(chrom)
        assert any("direction" in e for e in errors)

    def test_unexpected_param(self) -> None:
        gene = StrategyGene("RSI", {"period": 14, "bogus": 99}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[self._make_gene()],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        errors = validate_chromosome(chrom)
        assert any("unexpected" in e.lower() for e in errors)


# =============================================================================
# DSL conversion
# =============================================================================


class TestDSLConversion:
    def test_basic_conversion_valid_dsl(self) -> None:
        g_entry = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        g_exit = StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)
        chrom = StrategyChromosome(
            entry_genes=[g_entry],
            exit_genes=[g_exit],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="long",
        )
        dsl_dict = chromosome_to_dsl(chrom)
        # Should parse without errors
        strategy = StrategyDSL(**dsl_dict)
        assert strategy.name.startswith("genome_")
        assert "rsi_entry_0" in strategy.indicators

    def test_both_direction_populates_long_and_short(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="both",
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert len(strategy.entry_conditions.long) > 0
        assert len(strategy.entry_conditions.short) > 0

    def test_short_only(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="short",
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert len(strategy.entry_conditions.short) > 0
        assert len(strategy.entry_conditions.long) == 0

    def test_macd_gene_conversion(self) -> None:
        g = StrategyGene(
            "MACD",
            {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "crosses_above",
            0.0,
        )
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        macd_cfg = strategy.indicators["macd_entry_0"]
        assert macd_cfg.type == "MACD"
        assert macd_cfg.fast_period == 12

    def test_bbands_gene_conversion(self) -> None:
        g = StrategyGene("BBANDS", {"period": 20, "std_dev": 2.0}, "less_than", 0.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        bb_cfg = strategy.indicators["bbands_entry_0"]
        assert bb_cfg.type == "BBANDS"
        assert bb_cfg.std_dev == 2.0

    def test_stoch_gene_conversion(self) -> None:
        g = StrategyGene("STOCH", {"k_period": 14, "d_period": 3}, "less_than", 20.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert "stoch_entry_0" in strategy.indicators

    def test_sl_tp_converted_to_percent(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        assert dsl_dict["stop_loss"]["percent"] == 5.0
        assert dsl_dict["take_profit"]["percent"] == 10.0

    def test_random_chromosomes_produce_valid_dsl(self) -> None:
        """100 random chromosomes all produce valid StrategyDSL."""
        rng = random.Random(42)
        for _ in range(100):
            chrom = generate_random_chromosome(rng)
            dsl_dict = chromosome_to_dsl(chrom)
            strategy = StrategyDSL(**dsl_dict)
            assert strategy.name.startswith("genome_")

    def test_multi_gene_chromosome(self) -> None:
        genes = [
            StrategyGene("RSI", {"period": 14}, "less_than", 30.0),
            StrategyGene("EMA", {"period": 50}, "crosses_above", 0.0),
            StrategyGene("ATR", {"period": 14}, "greater_than", 0.01),
        ]
        chrom = StrategyChromosome(
            entry_genes=genes,
            exit_genes=[genes[0]],
            stop_loss_pct=3.0,
            take_profit_pct=6.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert len(strategy.indicators) == 4  # 3 entry + 1 exit
