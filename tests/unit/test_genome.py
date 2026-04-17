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


# =============================================================================
# BBANDS/DONCHIAN normalized sub-values
# =============================================================================


class TestBBandsDonchianGenome:
    def test_bbands_in_indicator_pool(self) -> None:
        assert "BBANDS" in INDICATOR_POOL
        assert INDICATOR_POOL["BBANDS"].default_threshold_range == (0.0, 1.0)

    def test_donchian_in_indicator_pool(self) -> None:
        assert "DONCHIAN" in INDICATOR_POOL
        assert INDICATOR_POOL["DONCHIAN"].default_threshold_range == (0.0, 1.0)

    def test_bbands_gene_with_percent_b(self) -> None:
        g = StrategyGene(
            "BBANDS", {"period": 20, "std_dev": 2.0}, ">", 0.8, sub_value="percent_b"
        )
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "<", 30.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert "bbands_entry_0" in strategy.indicators
        # Condition should reference percent_b sub-value
        long_conds = strategy.entry_conditions.long
        assert any("percent_b" in c for c in long_conds)

    def test_bbands_gene_with_bandwidth(self) -> None:
        g = StrategyGene(
            "BBANDS", {"period": 20, "std_dev": 2.0}, "<", 0.05, sub_value="bandwidth"
        )
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "<", 30.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        long_conds = strategy.entry_conditions.long
        assert any("bandwidth" in c for c in long_conds)

    def test_donchian_gene_with_position(self) -> None:
        g = StrategyGene(
            "DONCHIAN", {"period": 20}, ">", 0.9, sub_value="position"
        )
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "<", 30.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert "donchian_entry_0" in strategy.indicators
        long_conds = strategy.entry_conditions.long
        assert any("position" in c for c in long_conds)

    def test_bbands_validation_passes(self) -> None:
        g = StrategyGene(
            "BBANDS", {"period": 20, "std_dev": 2.0}, ">", 0.8, sub_value="percent_b"
        )
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "<", 30.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert errors == []

    def test_donchian_validation_passes(self) -> None:
        g = StrategyGene(
            "DONCHIAN", {"period": 20}, ">", 0.5, sub_value="position"
        )
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "<", 30.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert errors == []

    def test_random_chromosomes_with_bbands_donchian(self) -> None:
        """Random chromosomes can include BBANDS/DONCHIAN and remain valid."""
        rng = random.Random(42)
        bbands_seen = False
        donchian_seen = False
        for _ in range(500):
            chrom = generate_random_chromosome(rng)
            errors = validate_chromosome(chrom)
            assert errors == [], f"Invalid chromosome: {errors}"
            for gene in chrom.entry_genes + chrom.exit_genes:
                if gene.indicator_type == "BBANDS":
                    bbands_seen = True
                    assert gene.sub_value in ("percent_b", "bandwidth")
                elif gene.indicator_type == "DONCHIAN":
                    donchian_seen = True
                    assert gene.sub_value == "position"
        assert bbands_seen, "BBANDS not seen in 500 random chromosomes"
        assert donchian_seen, "DONCHIAN not seen in 500 random chromosomes"


# =============================================================================
# P6: dynamic indicator pool built from the registry
# =============================================================================


class TestDynamicIndicatorPool:
    """The GA pool is now built from indicator_registry.all_specs()."""

    def test_build_indicator_pool_from_registry(self) -> None:
        """Every spec with threshold_range AND param_ranges lands in the pool."""
        from vibe_quant.discovery.genome import build_indicator_pool
        from vibe_quant.dsl.indicators import indicator_registry

        pool = build_indicator_pool()
        assert pool, "build_indicator_pool returned an empty pool"

        # Every entry must correspond to a real spec that satisfies the
        # enrollment criteria.
        for name, ind_def in pool.items():
            spec = indicator_registry.get(name)
            assert spec is not None, f"{name} in pool but not in registry"
            assert spec.threshold_range is not None
            assert spec.param_ranges
            # IndicatorDef copies spec fields verbatim.
            assert ind_def.default_threshold_range == spec.threshold_range
            assert ind_def.dsl_type == spec.name

    def test_price_relative_indicators_excluded(self) -> None:
        """EMA/SMA/WMA/DEMA/TEMA set threshold_range=None and stay out."""
        from vibe_quant.discovery.genome import build_indicator_pool

        pool = build_indicator_pool()
        for price_relative in ("EMA", "SMA", "WMA", "DEMA", "TEMA"):
            assert price_relative not in pool, (
                f"{price_relative} is price-relative and must not auto-enroll "
                f"in the GA pool — threshold=0 against an absolute price "
                f"produces no trades."
            )

    def test_custom_plugin_with_threshold_range_auto_enrolls(self) -> None:
        """Register a throwaway spec with threshold_range + param_ranges and
        observe that it shows up in build_indicator_pool() immediately."""
        from vibe_quant.discovery.genome import build_indicator_pool
        from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

        def _noop_compute(df, params):  # noqa: ARG001
            return df["close"]

        spec = IndicatorSpec(
            name="TESTGA_ENROLLED",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 10},
            param_schema={"period": int},
            compute_fn=_noop_compute,
            param_ranges={"period": (5.0, 30.0)},
            threshold_range=(10.0, 90.0),
        )
        indicator_registry.register_spec(spec)
        try:
            pool = build_indicator_pool()
            assert "TESTGA_ENROLLED" in pool
            entry = pool["TESTGA_ENROLLED"]
            assert entry.param_ranges == {"period": (5.0, 30.0)}
            assert entry.default_threshold_range == (10.0, 90.0)
        finally:
            indicator_registry._indicators.pop(  # type: ignore[attr-defined]
                "TESTGA_ENROLLED", None
            )

    def test_custom_plugin_without_threshold_range_excluded(self) -> None:
        """A plugin with param_ranges but threshold_range=None is skipped.

        This is the same rule EMA/SMA/etc. rely on: price-relative
        indicators have no sensible numeric threshold, so the GA would
        only generate broken conditions. threshold_range=None acts as
        the explicit opt-out knob for plugin authors."""
        from vibe_quant.discovery.genome import build_indicator_pool
        from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

        def _noop_compute(df, params):  # noqa: ARG001
            return df["close"]

        spec = IndicatorSpec(
            name="TESTGA_EXCLUDED",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 10},
            param_schema={"period": int},
            compute_fn=_noop_compute,
            param_ranges={"period": (5.0, 30.0)},
            threshold_range=None,
        )
        indicator_registry.register_spec(spec)
        try:
            pool = build_indicator_pool()
            assert "TESTGA_EXCLUDED" not in pool
        finally:
            indicator_registry._indicators.pop(  # type: ignore[attr-defined]
                "TESTGA_EXCLUDED", None
            )


# =============================================================================
# MA pool
# =============================================================================


class TestMAPool:
    """MA_POOL enrolls ma_kind=True plugins; INDICATOR_POOL still excludes them."""

    def test_kama_vidya_frama_in_ma_pool(self) -> None:
        from vibe_quant.discovery.genome import MA_POOL

        assert "KAMA" in MA_POOL
        assert "VIDYA" in MA_POOL
        assert "FRAMA" in MA_POOL

    def test_ma_pool_entries_have_param_ranges(self) -> None:
        from vibe_quant.discovery.genome import MA_POOL

        for name, defn in MA_POOL.items():
            assert defn.name == name
            assert defn.param_ranges, f"{name} has empty param_ranges"

    def test_mas_not_in_indicator_pool(self) -> None:
        """MAs must stay out of INDICATOR_POOL — scalar thresholds make no
        sense against raw price levels."""
        for name in ("KAMA", "VIDYA", "FRAMA"):
            assert name not in INDICATOR_POOL, f"{name} leaked into INDICATOR_POOL"

    def test_ma_kind_false_excludes_from_ma_pool(self) -> None:
        """A plugin without ma_kind=True should not appear in MA_POOL."""
        from vibe_quant.discovery.genome import build_ma_pool
        from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

        def _noop_compute(df, params):  # noqa: ARG001
            return df["close"]

        spec = IndicatorSpec(
            name="TESTMA_EXCLUDED",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 10},
            param_schema={"period": int},
            compute_fn=_noop_compute,
            param_ranges={"period": (5.0, 30.0)},
            threshold_range=None,
            # ma_kind omitted → defaults to False
        )
        indicator_registry.register_spec(spec)
        try:
            pool = build_ma_pool()
            assert "TESTMA_EXCLUDED" not in pool
        finally:
            indicator_registry._indicators.pop(  # type: ignore[attr-defined]
                "TESTMA_EXCLUDED", None
            )


# =============================================================================
# MA gene chromosomes
# =============================================================================


class TestMAChromosome:
    """MA genes — DSL emission, roundtrip serialization, validation, compile."""

    @staticmethod
    def _make_ma_chrom(ma_kind: str = "KAMA") -> StrategyChromosome:
        from vibe_quant.discovery.operators import (
            ConditionType,
            Direction,
            PriceVsMAConditionGene,
        )
        from vibe_quant.discovery.operators import (
            StrategyChromosome as _Chrom,
        )
        from vibe_quant.discovery.operators import (
            StrategyGene as _Gene,
        )

        return _Chrom(
            entry_genes=[
                _Gene(
                    indicator_type="RSI",
                    parameters={"period": 14},
                    condition=ConditionType.LT,
                    threshold=30.0,
                )
            ],
            exit_genes=[
                _Gene(
                    indicator_type="RSI",
                    parameters={"period": 14},
                    condition=ConditionType.GT,
                    threshold=70.0,
                )
            ],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction=Direction.LONG,
            ma_entry_genes=[
                PriceVsMAConditionGene(
                    indicator_type=ma_kind,
                    parameters={"period": 20.0},
                    op=ConditionType.GT,
                )
            ],
        )

    def test_chromosome_to_dsl_emits_ma_indicator_and_close_condition(self) -> None:
        chrom = self._make_ma_chrom("KAMA")
        dsl = chromosome_to_dsl(chrom)

        ma_names = [k for k in dsl["indicators"] if k.startswith("kama_ma_entry_")]
        assert len(ma_names) == 1
        ma_name = ma_names[0]
        assert dsl["indicators"][ma_name]["type"] == "KAMA"
        assert dsl["indicators"][ma_name]["period"] == 20

        # Condition uses `close` as the left operand, MA as the right
        assert f"close > {ma_name}" in dsl["entry_conditions"]["long"]

    def test_chromosome_to_dsl_parses_as_strategy(self) -> None:
        """Emitted DSL dict must round-trip through StrategyDSL validation."""
        chrom = self._make_ma_chrom("KAMA")
        dsl = chromosome_to_dsl(chrom)
        dsl["timeframe"] = "5m"
        StrategyDSL(**dsl)  # raises on invalid

    def test_ma_gene_roundtrip_serialization(self) -> None:
        from vibe_quant.discovery.genome import (
            chromosome_to_serializable,
            serializable_to_chromosome,
        )

        original = self._make_ma_chrom("VIDYA")
        d = chromosome_to_serializable(original)
        assert "ma_entry_genes" in d
        restored = serializable_to_chromosome(d)

        assert len(restored.ma_entry_genes) == 1
        g0, r0 = original.ma_entry_genes[0], restored.ma_entry_genes[0]
        assert r0.indicator_type == g0.indicator_type
        assert r0.op == g0.op
        assert r0.parameters == g0.parameters

    def test_validate_chromosome_accepts_ma_gene(self) -> None:
        chrom = self._make_ma_chrom("KAMA")
        assert validate_chromosome(chrom) == []
        assert is_valid_chromosome(chrom)

    def test_validate_chromosome_rejects_unknown_ma_indicator(self) -> None:
        from vibe_quant.discovery.operators import ConditionType, PriceVsMAConditionGene

        chrom = self._make_ma_chrom("KAMA")
        chrom.ma_entry_genes[0] = PriceVsMAConditionGene(
            indicator_type="DOES_NOT_EXIST",
            parameters={"period": 20.0},
            op=ConditionType.GT,
        )
        errors = validate_chromosome(chrom)
        assert any("DOES_NOT_EXIST" in e for e in errors)

    def test_ma_exit_gene_emits_close_condition_in_exit(self) -> None:
        from vibe_quant.discovery.operators import (
            ConditionType,
            PriceVsMAConditionGene,
        )

        chrom = self._make_ma_chrom("FRAMA")
        chrom.ma_entry_genes = []
        chrom.ma_exit_genes = [
            PriceVsMAConditionGene(
                indicator_type="FRAMA",
                parameters={"period": 16.0},
                op=ConditionType.LT,
            )
        ]
        dsl = chromosome_to_dsl(chrom)
        assert any(
            "close < frama_ma_exit_0" in c
            for c in dsl["exit_conditions"]["long"]
        )

    def test_clone_preserves_ma_genes_independently(self) -> None:
        original = self._make_ma_chrom("KAMA")
        clone = original.clone()

        assert len(clone.ma_entry_genes) == 1
        # Mutating the clone must not touch the original
        clone.ma_entry_genes[0].parameters["period"] = 99.0
        assert original.ma_entry_genes[0].parameters["period"] == 20.0

    def test_compile_ma_chromosome_to_module(self) -> None:
        """End-to-end: MA-gene DSL must compile to a loadable strategy module."""
        from vibe_quant.dsl.compiler import StrategyCompiler

        chrom = self._make_ma_chrom("KAMA")
        dsl_dict = chromosome_to_dsl(chrom)
        dsl_dict["timeframe"] = "5m"
        dsl = StrategyDSL(**dsl_dict)

        compiler = StrategyCompiler()
        module = compiler.compile_to_module(dsl)

        # Strategy class must exist — kama compute_fn path means
        # `_update_pta_indicators` should be wired in.
        class_name = "".join(w.capitalize() for w in dsl.name.split("_"))
        assert hasattr(module, f"{class_name}Strategy")
        assert hasattr(module, f"{class_name}Config")

    def test_mutate_preserves_ma_pool_membership(self) -> None:
        """Hot mutation of an MA-bearing chromosome keeps indicator in MA_POOL."""
        from vibe_quant.discovery.genome import MA_POOL

        original = self._make_ma_chrom("KAMA")
        random.seed(42)
        for _ in range(50):
            mutated = mutate(original, mutation_rate=1.0)
            for g in mutated.ma_entry_genes + mutated.ma_exit_genes:
                assert g.indicator_type in MA_POOL

    def test_crossover_produces_valid_ma_counts(self) -> None:
        """Crossover of two MA-bearing parents must stay within cap."""
        from vibe_quant.discovery.operators import MAX_MA_ENTRY_GENES, crossover

        parent_a = self._make_ma_chrom("KAMA")
        parent_b = self._make_ma_chrom("VIDYA")
        random.seed(7)
        child_a, child_b = crossover(parent_a, parent_b)
        assert len(child_a.ma_entry_genes) <= MAX_MA_ENTRY_GENES
        assert len(child_b.ma_entry_genes) <= MAX_MA_ENTRY_GENES
