"""Tests for genetic operators (discovery module)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from vibe_quant.discovery.operators import (
    INDICATOR_POOL,
    MAX_ENTRY_GENES,
    MAX_EXIT_GENES,
    MIN_ENTRY_GENES,
    MIN_EXIT_GENES,
    SL_RANGE,
    TP_RANGE,
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
    THRESHOLD_RANGES,
    apply_elitism,
    crossover,
    initialize_population,
    is_valid_chromosome,
    mutate,
    tournament_select,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gene(
    indicator: str = "RSI",
    threshold: float = 30.0,
    condition: ConditionType = ConditionType.LT,
) -> StrategyGene:
    return StrategyGene(
        indicator_type=indicator,
        parameters={"period": 14.0},
        condition=condition,
        threshold=threshold,
    )


def _make_chromosome(
    n_entry: int = 2,
    n_exit: int = 1,
    sl: float = 2.0,
    tp: float = 4.0,
    direction: Direction = Direction.LONG,
) -> StrategyChromosome:
    # Use thresholds within RSI valid range (25-75)
    entry_thresholds = [25.0 + i * 5.0 for i in range(n_entry)]
    exit_thresholds = [70.0 + i * 2.0 for i in range(n_exit)]
    return StrategyChromosome(
        entry_genes=[_make_gene(threshold=t) for t in entry_thresholds],
        exit_genes=[_make_gene(threshold=t) for t in exit_thresholds],
        stop_loss_pct=sl,
        take_profit_pct=tp,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# is_valid_chromosome
# ---------------------------------------------------------------------------


class TestIsValidChromosome:
    def test_valid(self) -> None:
        chrom = _make_chromosome()
        assert is_valid_chromosome(chrom)

    def test_too_many_entry_genes(self) -> None:
        chrom = _make_chromosome(n_entry=MAX_ENTRY_GENES + 1)
        assert not is_valid_chromosome(chrom)

    def test_too_few_entry_genes(self) -> None:
        chrom = _make_chromosome(n_entry=0)
        assert not is_valid_chromosome(chrom)

    def test_sl_out_of_range(self) -> None:
        chrom = _make_chromosome(sl=0.01)
        assert not is_valid_chromosome(chrom)

    def test_invalid_indicator(self) -> None:
        chrom = _make_chromosome()
        chrom.entry_genes[0].indicator_type = "INVALID"
        assert not is_valid_chromosome(chrom)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


class TestCrossover:
    def test_produces_two_offspring(self) -> None:
        a = _make_chromosome(n_entry=3, n_exit=2)
        b = _make_chromosome(n_entry=2, n_exit=1)
        c1, c2 = crossover(a, b)
        assert isinstance(c1, StrategyChromosome)
        assert isinstance(c2, StrategyChromosome)

    def test_offspring_are_valid(self) -> None:
        random.seed(42)
        a = _make_chromosome(n_entry=3, n_exit=2)
        b = _make_chromosome(n_entry=2, n_exit=1)
        for _ in range(50):
            c1, c2 = crossover(a, b)
            assert is_valid_chromosome(c1), (
                f"child1 invalid: {len(c1.entry_genes)} entry, {len(c1.exit_genes)} exit"
            )
            assert is_valid_chromosome(c2), (
                f"child2 invalid: {len(c2.entry_genes)} entry, {len(c2.exit_genes)} exit"
            )

    def test_offspring_entry_gene_count_in_bounds(self) -> None:
        random.seed(0)
        a = _make_chromosome(n_entry=5, n_exit=3)
        b = _make_chromosome(n_entry=1, n_exit=1)
        for _ in range(100):
            c1, c2 = crossover(a, b)
            assert MIN_ENTRY_GENES <= len(c1.entry_genes) <= MAX_ENTRY_GENES
            assert MIN_EXIT_GENES <= len(c1.exit_genes) <= MAX_EXIT_GENES
            assert MIN_ENTRY_GENES <= len(c2.entry_genes) <= MAX_ENTRY_GENES
            assert MIN_EXIT_GENES <= len(c2.exit_genes) <= MAX_EXIT_GENES

    def test_statistical_mixing(self) -> None:
        """Over many trials, offspring should contain genes from both parents."""
        random.seed(7)
        # Make parents with distinguishable indicators
        a = StrategyChromosome(
            entry_genes=[_make_gene("RSI"), _make_gene("RSI")],
            exit_genes=[_make_gene("RSI")],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        b = StrategyChromosome(
            entry_genes=[_make_gene("EMA"), _make_gene("EMA")],
            exit_genes=[_make_gene("EMA")],
            stop_loss_pct=3.0,
            take_profit_pct=6.0,
        )
        indicators_seen: Counter[str] = Counter()
        for _ in range(200):
            c1, c2 = crossover(a, b)
            for g in c1.entry_genes + c2.entry_genes:
                indicators_seen[g.indicator_type] += 1

        # Both RSI and EMA should appear
        assert indicators_seen["RSI"] > 0, "Expected RSI genes from parent A"
        assert indicators_seen["EMA"] > 0, "Expected EMA genes from parent B"

    def test_sl_tp_from_parents(self) -> None:
        """SL/TP values should come from one parent or the other."""
        random.seed(99)
        a = _make_chromosome(sl=1.0, tp=2.0)
        b = _make_chromosome(sl=5.0, tp=10.0)
        sl_vals = set()
        tp_vals = set()
        for _ in range(100):
            c1, _ = crossover(a, b)
            sl_vals.add(c1.stop_loss_pct)
            tp_vals.add(c1.take_profit_pct)
        assert sl_vals <= {1.0, 5.0}
        assert tp_vals <= {2.0, 10.0}

    def test_does_not_modify_parents(self) -> None:
        a = _make_chromosome(n_entry=2)
        b = _make_chromosome(n_entry=3)
        a_entry_len = len(a.entry_genes)
        b_entry_len = len(b.entry_genes)
        crossover(a, b)
        assert len(a.entry_genes) == a_entry_len
        assert len(b.entry_genes) == b_entry_len


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


class TestMutation:
    def test_mutation_returns_new_chromosome(self) -> None:
        chrom = _make_chromosome()
        mutated = mutate(chrom, mutation_rate=1.0)
        assert mutated is not chrom

    def test_high_rate_changes_at_least_one_gene(self) -> None:
        """With rate=1.0, expect at least one gene differs."""
        random.seed(12)
        chrom = _make_chromosome(n_entry=3, n_exit=2)
        changed_count = 0
        for _ in range(50):
            mutated = mutate(chrom, mutation_rate=1.0)
            orig_indicators = [g.indicator_type for g in chrom.entry_genes + chrom.exit_genes]
            new_indicators = [g.indicator_type for g in mutated.entry_genes + mutated.exit_genes]
            orig_thresholds = [g.threshold for g in chrom.entry_genes + chrom.exit_genes]
            new_thresholds = [g.threshold for g in mutated.entry_genes + mutated.exit_genes]
            orig_conditions = [g.condition for g in chrom.entry_genes + chrom.exit_genes]
            new_conditions = [g.condition for g in mutated.entry_genes + mutated.exit_genes]
            if (
                orig_indicators != new_indicators
                or orig_thresholds != new_thresholds
                or orig_conditions != new_conditions
            ):
                changed_count += 1
        # At high mutation rate, nearly all should change
        assert changed_count > 40, f"Only {changed_count}/50 changed at rate=1.0"

    def test_respects_entry_gene_constraints(self) -> None:
        random.seed(0)
        for _ in range(200):
            chrom = _make_chromosome(n_entry=1, n_exit=1)
            mutated = mutate(chrom, mutation_rate=0.5)
            assert MIN_ENTRY_GENES <= len(mutated.entry_genes) <= MAX_ENTRY_GENES
            assert MIN_EXIT_GENES <= len(mutated.exit_genes) <= MAX_EXIT_GENES

    def test_respects_sl_tp_range(self) -> None:
        random.seed(42)
        for _ in range(200):
            chrom = _make_chromosome(sl=SL_RANGE[0], tp=TP_RANGE[0])
            mutated = mutate(chrom, mutation_rate=1.0)
            assert SL_RANGE[0] <= mutated.stop_loss_pct <= SL_RANGE[1]
            assert TP_RANGE[0] <= mutated.take_profit_pct <= TP_RANGE[1]

    def test_zero_rate_preserves_genes(self) -> None:
        """mutation_rate=0 should not change genes (though structural mutation has
        rate*0.3=0 probability)."""
        random.seed(0)
        chrom = _make_chromosome(n_entry=2, n_exit=1)
        mutated = mutate(chrom, mutation_rate=0.0)
        assert len(mutated.entry_genes) == len(chrom.entry_genes)
        for orig, new in zip(chrom.entry_genes, mutated.entry_genes, strict=True):
            assert orig.indicator_type == new.indicator_type
            assert orig.threshold == new.threshold

    def test_mutated_is_valid(self) -> None:
        random.seed(7)
        for _ in range(100):
            chrom = _make_chromosome()
            mutated = mutate(chrom, mutation_rate=0.3)
            assert is_valid_chromosome(mutated)

    def test_does_not_modify_original(self) -> None:
        chrom = _make_chromosome()
        original_sl = chrom.stop_loss_pct
        original_tp = chrom.take_profit_pct
        original_entries = [(g.indicator_type, g.threshold) for g in chrom.entry_genes]
        mutate(chrom, mutation_rate=1.0)
        assert chrom.stop_loss_pct == original_sl
        assert chrom.take_profit_pct == original_tp
        for i, (ind, thr) in enumerate(original_entries):
            assert chrom.entry_genes[i].indicator_type == ind
            assert chrom.entry_genes[i].threshold == thr


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------


class TestTournamentSelect:
    def test_picks_best_in_tournament(self) -> None:
        """With tournament_size == population size, always picks the best."""
        population = [_make_chromosome() for _ in range(5)]
        scores = [1.0, 5.0, 3.0, 2.0, 4.0]
        # tournament_size = 5 => always picks index 1 (score 5.0)
        for _ in range(20):
            winner = tournament_select(population, scores, tournament_size=5)
            assert winner is population[1]

    def test_statistical_bias_toward_best(self) -> None:
        """Best individual should be selected most often."""
        random.seed(42)
        # Use distinguishable sl values so we can identify winners
        population = [_make_chromosome(sl=float(i + 1)) for i in range(10)]
        scores = list(range(10))  # index 9 has highest score
        id_to_idx = {id(c): i for i, c in enumerate(population)}
        wins: Counter[int] = Counter()
        for _ in range(1000):
            winner = tournament_select(population, scores, tournament_size=3)
            wins[id_to_idx[id(winner)]] += 1
        # Index 9 (best) should win most
        assert wins[9] == max(wins.values())

    def test_empty_population_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            tournament_select([], [], tournament_size=3)

    def test_mismatched_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="size"):
            tournament_select([_make_chromosome()], [1.0, 2.0], tournament_size=1)

    def test_tournament_size_clamped(self) -> None:
        """Tournament size > population should still work."""
        population = [_make_chromosome(), _make_chromosome()]
        scores = [1.0, 2.0]
        winner = tournament_select(population, scores, tournament_size=100)
        assert winner is population[1]


# ---------------------------------------------------------------------------
# Elitism
# ---------------------------------------------------------------------------


class TestElitism:
    def test_returns_top_n(self) -> None:
        population = [_make_chromosome(sl=float(i)) for i in range(1, 6)]
        scores = [10.0, 50.0, 30.0, 20.0, 40.0]
        elites = apply_elitism(population, scores, elite_count=2)
        assert len(elites) == 2
        # Best is index 1 (50.0), second is index 4 (40.0)
        assert elites[0].stop_loss_pct == population[1].stop_loss_pct
        assert elites[1].stop_loss_pct == population[4].stop_loss_pct

    def test_elites_are_cloned(self) -> None:
        population = [_make_chromosome()]
        scores = [1.0]
        elites = apply_elitism(population, scores, elite_count=1)
        assert elites[0] is not population[0]
        assert elites[0].stop_loss_pct == population[0].stop_loss_pct

    def test_elite_count_clamped(self) -> None:
        """elite_count > population returns all."""
        population = [_make_chromosome(), _make_chromosome()]
        scores = [1.0, 2.0]
        elites = apply_elitism(population, scores, elite_count=10)
        assert len(elites) == 2

    def test_empty_population_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            apply_elitism([], [], elite_count=1)

    def test_preserves_original_population(self) -> None:
        population = [_make_chromosome(sl=1.5)]
        scores = [1.0]
        elites = apply_elitism(population, scores, elite_count=1)
        # Modify elite -- original should be unaffected
        elites[0].stop_loss_pct = 999.0
        assert population[0].stop_loss_pct == 1.5


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------


class TestInitializePopulation:
    def test_correct_size(self) -> None:
        pop = initialize_population(size=20)
        assert len(pop) == 20

    def test_all_valid(self) -> None:
        random.seed(42)
        pop = initialize_population(size=100)
        for i, chrom in enumerate(pop):
            assert is_valid_chromosome(chrom), f"Chromosome {i} invalid"

    def test_diversity(self) -> None:
        """Population should contain varied indicator types."""
        random.seed(0)
        pop = initialize_population(size=50)
        all_indicators: set[str] = set()
        for chrom in pop:
            for gene in chrom.entry_genes + chrom.exit_genes:
                all_indicators.add(gene.indicator_type)
        # With 50 chromosomes, expect at least a few distinct indicators
        assert len(all_indicators) >= 5

    def test_zero_size(self) -> None:
        pop = initialize_population(size=0)
        assert pop == []

    def test_gene_counts_in_bounds(self) -> None:
        random.seed(7)
        pop = initialize_population(size=100)
        for chrom in pop:
            assert MIN_ENTRY_GENES <= len(chrom.entry_genes) <= MAX_ENTRY_GENES
            assert MIN_EXIT_GENES <= len(chrom.exit_genes) <= MAX_EXIT_GENES

    def test_indicator_types_from_pool(self) -> None:
        random.seed(42)
        pop = initialize_population(size=30)
        for chrom in pop:
            for gene in chrom.entry_genes + chrom.exit_genes:
                assert gene.indicator_type in INDICATOR_POOL


# ---------------------------------------------------------------------------
# Mutation threshold reset on indicator swap
# ---------------------------------------------------------------------------


class TestMutateThresholdReset:
    def test_indicator_swap_resets_threshold_to_valid_range(self) -> None:
        """When mutation swaps indicator type, threshold must be in new indicator's range."""
        random.seed(42)
        # Create gene with RSI threshold (25-75 range)
        gene = _make_gene("RSI", threshold=72.0, condition=ConditionType.GT)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[_make_gene()],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )

        # Mutate many times — after indicator swap, threshold should be in new range
        for _ in range(500):
            mutated = mutate(chrom, mutation_rate=1.0)
            for g in mutated.entry_genes:
                if g.indicator_type in THRESHOLD_RANGES:
                    lo, hi = THRESHOLD_RANGES[g.indicator_type]
                    assert lo <= g.threshold <= hi, (
                        f"{g.indicator_type} threshold {g.threshold} outside [{lo}, {hi}]"
                    )


# ---------------------------------------------------------------------------
# Per-direction SL/TP operators
# ---------------------------------------------------------------------------


class TestPerDirectionSLTPOperators:
    def test_crossover_preserves_per_direction(self) -> None:
        random.seed(42)
        a = _make_chromosome(direction=Direction.BOTH)
        a.stop_loss_long_pct = 1.0
        a.stop_loss_short_pct = 5.0
        b = _make_chromosome(direction=Direction.BOTH)
        b.stop_loss_long_pct = 2.0
        b.stop_loss_short_pct = 8.0
        sl_long_vals: set[float] = set()
        for _ in range(100):
            c1, _ = crossover(a, b)
            if c1.stop_loss_long_pct is not None:
                sl_long_vals.add(c1.stop_loss_long_pct)
        assert sl_long_vals <= {1.0, 2.0}

    def test_mutation_perturbs_per_direction(self) -> None:
        random.seed(42)
        chrom = _make_chromosome(direction=Direction.BOTH)
        chrom.stop_loss_long_pct = 2.0
        chrom.stop_loss_short_pct = 5.0
        chrom.take_profit_long_pct = 8.0
        chrom.take_profit_short_pct = 12.0
        changed = {k: False for k in ["sl_long", "sl_short", "tp_long", "tp_short"]}
        for _ in range(100):
            m = mutate(chrom, mutation_rate=1.0)
            if m.stop_loss_long_pct != 2.0:
                changed["sl_long"] = True
            if m.stop_loss_short_pct != 5.0:
                changed["sl_short"] = True
            if m.take_profit_long_pct != 8.0:
                changed["tp_long"] = True
            if m.take_profit_short_pct != 12.0:
                changed["tp_short"] = True
        assert all(changed.values()), f"Not all per-direction values mutated: {changed}"

    def test_random_both_has_per_direction(self) -> None:
        random.seed(42)
        pop = initialize_population(size=50, direction_constraint=Direction.BOTH)
        has_per_dir = sum(1 for c in pop if c.stop_loss_long_pct is not None)
        assert has_per_dir == 50

    def test_random_long_no_per_direction(self) -> None:
        random.seed(42)
        pop = initialize_population(size=20, direction_constraint=Direction.LONG)
        has_per_dir = sum(1 for c in pop if c.stop_loss_long_pct is not None)
        assert has_per_dir == 0
