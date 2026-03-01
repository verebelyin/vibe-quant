"""Tests for strategy genome representation."""

from __future__ import annotations

import random

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
from vibe_quant.discovery.operators import THRESHOLD_RANGES, is_valid_chromosome, mutate
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

    def test_mutable(self) -> None:
        gene = StrategyGene("EMA", {"period": 20}, "greater_than", 0.0)
        gene.threshold = 50.0
        assert gene.threshold == 50.0

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
        assert chrom.direction.value == "long"
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
        assert chrom.direction.value in VALID_DIRECTIONS

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
                assert gene.condition.value in VALID_CONDITIONS

    def test_generated_chromosome_is_compatible_with_mutation_operators(self) -> None:
        """Genome output should be directly mutable by discovery operators."""
        chrom = generate_random_chromosome(random.Random(2026))
        mutated = mutate(chrom, mutation_rate=0.4)
        assert isinstance(mutated, StrategyChromosome)
        assert is_valid_chromosome(mutated)


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

    def test_threshold_in_range_for_random_chromosomes(self) -> None:
        """All random chromosomes should have thresholds within valid ranges."""
        rng = random.Random(42)
        for _ in range(200):
            chrom = generate_random_chromosome(rng)
            for gene in chrom.entry_genes + chrom.exit_genes:
                if gene.indicator_type in THRESHOLD_RANGES:
                    tlo, thi = THRESHOLD_RANGES[gene.indicator_type]
                    assert tlo <= gene.threshold <= thi, (
                        f"{gene.indicator_type} threshold {gene.threshold} outside [{tlo}, {thi}]"
                    )

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


# =============================================================================
# Threshold validation
# =============================================================================


class TestThresholdValidation:
    def test_threshold_out_of_range_detected(self) -> None:
        """ATR with threshold=72 (RSI range) should fail validation."""
        gene = StrategyGene("ATR", {"period": 14}, "greater_than", 72.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert any("threshold" in e.lower() for e in errors)

    def test_threshold_in_range_passes(self) -> None:
        """ATR with valid threshold should pass."""
        gene = StrategyGene("ATR", {"period": 14}, "greater_than", 0.015)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert errors == []

    def test_rsi_threshold_out_of_range(self) -> None:
        """RSI with threshold=150 (>100) should fail."""
        gene = StrategyGene("RSI", {"period": 14}, "greater_than", 150.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert any("threshold" in e.lower() for e in errors)


# =============================================================================
# Per-direction SL/TP genome tests
# =============================================================================


class TestPerDirectionSLTPGenome:
    def test_chromosome_to_dsl_emits_per_direction(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="both",
            stop_loss_long_pct=1.09,
            stop_loss_short_pct=8.29,
            take_profit_long_pct=17.13,
            take_profit_short_pct=13.06,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        assert dsl_dict["stop_loss_long"]["percent"] == 1.09
        assert dsl_dict["stop_loss_short"]["percent"] == 8.29
        assert dsl_dict["take_profit_long"]["percent"] == 17.13
        assert dsl_dict["take_profit_short"]["percent"] == 13.06
        assert dsl_dict["stop_loss"]["percent"] == 2.0

    def test_no_per_direction_for_single_direction(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="long",
        )
        dsl_dict = chromosome_to_dsl(chrom)
        assert "stop_loss_long" not in dsl_dict

    def test_per_direction_dsl_validates(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="both",
            stop_loss_long_pct=1.09,
            stop_loss_short_pct=8.29,
            take_profit_long_pct=17.13,
            take_profit_short_pct=13.06,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert strategy.stop_loss_long.percent == 1.09

    def test_validate_per_direction_sl_range(self) -> None:
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="both",
            stop_loss_long_pct=15.0,
        )
        errors = validate_chromosome(chrom)
        assert any("stop_loss_long" in e for e in errors)
