"""T-cplx (vibe-quant-dktr4): ``max_complexity`` is 10 scalar genes, GA-enforced.

The GA must not PRODUCE a genome with more than ``MAX_SCALAR_GENES`` scalar-threshold
genes (entry + exit + short_entry + short_exit; MA genes are not counted) nor keep
one: init, mutation, crossover/repair, warm-start seeds and diversity immigrants all
trim deterministically (no RNG draw). The final guardrail at 10 stays as a
fail-closed backstop, so no top candidate is rejected for complexity.

Single-direction genomes are untouched: the per-list caps (5 + 3) already keep them
at <= 8, so nothing here can change a long/short run's RNG stream (R16).
"""

from __future__ import annotations

import random

import pytest

from tests.unit.test_discovery_pipeline import _make_config
from vibe_quant.discovery.mock_backtest import mock_backtest
from vibe_quant.discovery.operators import (
    MAX_SCALAR_GENES,
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
    _enforce_complexity_cap,
    _random_chromosome,
    crossover,
    initialize_population,
    is_valid_chromosome,
    mutate,
    scalar_gene_count,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline

# ---------------------------------------------------------------------------
# helpers (local copies: no cross-file private test helper imports)
# ---------------------------------------------------------------------------


def _rsi(condition: ConditionType, threshold: float, period: float = 14.0) -> StrategyGene:
    return StrategyGene("RSI", {"period": period}, condition, threshold)


def _both(n_entry: int, n_exit: int, n_short_entry: int, n_short_exit: int) -> StrategyChromosome:
    """A both chromosome with given long/short gene counts (RSI14, thresholds valid)."""
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0 + i) for i in range(n_entry)],
        exit_genes=[_rsi(ConditionType.GT, 70.0 + i) for i in range(n_exit)],
        short_entry_genes=[_rsi(ConditionType.GT, 70.0 + i) for i in range(n_short_entry)],
        short_exit_genes=[_rsi(ConditionType.LT, 30.0 + i) for i in range(n_short_exit)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
    )


def _long(n_entry: int = 3, n_exit: int = 2) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0 + i) for i in range(n_entry)],
        exit_genes=[_rsi(ConditionType.GT, 70.0 + i) for i in range(n_exit)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


def _shape(chrom: StrategyChromosome) -> tuple[int, int, int, int]:
    return (
        len(chrom.entry_genes),
        len(chrom.exit_genes),
        len(chrom.short_entry_genes),
        len(chrom.short_exit_genes),
    )


# ---------------------------------------------------------------------------
# Pipeline: a seeded both-direction run never evaluates or rejects over-cap
# ---------------------------------------------------------------------------


def test_seeded_both_run_never_evaluates_or_rejects_over_cap() -> None:
    state = random.getstate()
    try:
        random.seed(6)
        pipe = DiscoveryPipeline(
            _make_config(direction="both", population_size=6, max_generations=2),
            mock_backtest,
        )
        result = pipe.run()
    finally:
        random.setstate(state)

    assert pipe._all_scored
    assert max(scalar_gene_count(c) for c, _ in pipe._all_scored) <= MAX_SCALAR_GENES

    reasons = [
        str(reason)
        for rejection in result.guardrail_rejections
        for reason in rejection["reasons"]  # type: ignore[attr-defined]
    ]
    assert not [r for r in reasons if r.startswith("Too complex")], reasons


# ---------------------------------------------------------------------------
# Shape rows: deterministic tail trim, exact outputs, zero RNG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        # (entry, exit, short_entry, short_exit) -> (expected shape, genes)
        ((2, 2, 2, 2), (2, 2, 2, 2)),  # B8: 8, no-op
        ((3, 2, 2, 2), (3, 2, 2, 2)),  # B9: 9, no-op
        ((3, 2, 3, 2), (3, 2, 3, 2)),  # B10: 10, no-op
        ((5, 3, 1, 1), (5, 3, 1, 1)),  # B10b: 10, no-op
        ((3, 3, 3, 2), (2, 3, 3, 2)),  # B11a: entry tail (tie -> entry first)
        ((3, 2, 3, 3), (2, 2, 3, 3)),  # B11b
        ((5, 3, 2, 1), (4, 3, 2, 1)),  # B11c
        ((4, 2, 5, 3), (2, 2, 3, 3)),  # B14
        ((5, 3, 5, 3), (2, 2, 3, 3)),  # B16
    ],
)
def test_enforce_complexity_cap_trim_walk(
    shape: tuple[int, int, int, int], expected: tuple[int, int, int, int]
) -> None:
    chrom = _both(*shape)
    before = scalar_gene_count(chrom)
    source = tuple(
        [gene.clone() for gene in genes]
        for genes in (
            chrom.entry_genes,
            chrom.exit_genes,
            chrom.short_entry_genes,
            chrom.short_exit_genes,
        )
    )

    state = random.getstate()
    _enforce_complexity_cap(chrom)
    assert random.getstate() == state, "the cap helper must not draw RNG (R1)"

    assert _shape(chrom) == expected
    expected_genes = sum(expected)
    assert scalar_gene_count(chrom) == expected_genes
    if before > MAX_SCALAR_GENES:
        assert expected_genes == MAX_SCALAR_GENES

    # The trim must cut TAILS: each retained list is its source list's head.
    # A head cut (``pop(0)``) retains the same shape but the wrong conditions.
    for kept, original in zip(
        (
            chrom.entry_genes,
            chrom.exit_genes,
            chrom.short_entry_genes,
            chrom.short_exit_genes,
        ),
        source,
        strict=True,
    ):
        assert kept == original[: len(kept)]


def test_enforce_complexity_cap_never_empties_a_list() -> None:
    chrom = _both(5, 3, 5, 3)
    _enforce_complexity_cap(chrom)
    assert min(_shape(chrom)) >= 1


# ---------------------------------------------------------------------------
# Producer rows: init (I1/I2/I3), mutation (M1/M2), crossover (X1/X2)
# ---------------------------------------------------------------------------


def test_i1_random_both_chromosome_respects_cap() -> None:
    state = random.getstate()
    try:
        random.seed(11)
        for _ in range(500):
            chrom = _random_chromosome(Direction.BOTH)
            assert scalar_gene_count(chrom) <= MAX_SCALAR_GENES
            assert is_valid_chromosome(chrom)
    finally:
        random.setstate(state)


def test_i2_initialize_population_both_respects_cap() -> None:
    state = random.getstate()
    try:
        random.seed(7)
        population = initialize_population(30, Direction.BOTH)
    finally:
        random.setstate(state)
    assert len(population) == 30
    assert max(scalar_gene_count(c) for c in population) <= MAX_SCALAR_GENES


def test_i3_warm_start_seed_is_trimmed_not_skipped() -> None:
    seed = _both(5, 3, 5, 3)
    seed_uid = seed.uid
    # Only one seed slot (size 4 // 2): if it were rejected the call would raise
    # ValueError ("none of 1 seed chromosome(s) is valid") instead.
    population = initialize_population(4, Direction.BOTH, [seed])
    admitted = population[0]
    assert admitted.uid != seed_uid
    assert admitted.direction is Direction.BOTH
    assert _shape(admitted) == (2, 2, 3, 3)
    assert is_valid_chromosome(admitted)


def test_m1_mutate_both_respects_cap_over_seeds() -> None:
    state = random.getstate()
    try:
        for child_seed in range(40):
            random.seed(child_seed)
            child = mutate(_both(3, 2, 3, 2), mutation_rate=1.0)
            assert scalar_gene_count(child) <= MAX_SCALAR_GENES
            assert is_valid_chromosome(child)
    finally:
        random.setstate(state)


def test_m2_mutate_add_to_short_exit_trims_the_longest_tie_first() -> None:
    # The shape a short_exit add produces: (3,2,3,3) = 11. The entry tail goes
    # (ties -> entry first), not the gene that just grew.
    chrom = _both(3, 2, 3, 3)
    _enforce_complexity_cap(chrom)
    assert _shape(chrom) == (2, 2, 3, 3)


def test_x1_crossover_of_two_both_parents_respects_cap() -> None:
    state = random.getstate()
    try:
        for cross_seed in range(40):
            random.seed(cross_seed)
            child_a, child_b = crossover(_both(5, 3, 5, 3), _both(5, 3, 5, 3))
            for child in (child_a, child_b):
                assert scalar_gene_count(child) <= MAX_SCALAR_GENES
                assert is_valid_chromosome(child)
    finally:
        random.setstate(state)


def test_x2_crossover_long_and_both_mixed_child_rules() -> None:
    state = random.getstate()
    try:
        for cross_seed in range(40):
            random.seed(cross_seed)
            child_a, child_b = crossover(_long(), _both(5, 3, 5, 3))
            for child in (child_a, child_b):
                assert scalar_gene_count(child) <= MAX_SCALAR_GENES
                assert is_valid_chromosome(child)
                if child.direction is not Direction.BOTH:
                    assert _shape(child)[2:] == (0, 0)
    finally:
        random.setstate(state)


def test_both_ga_loop_respects_cap() -> None:
    """The long/short digest loop, run for BOTH: every member stays <= 10."""
    state = random.getstate()
    try:
        random.seed(7)
        pop = initialize_population(30, Direction.BOTH)
        for _ in range(20):
            nxt: list[StrategyChromosome] = []
            for i in range(0, len(pop), 2):
                child_a, child_b = crossover(pop[i], pop[i + 1])
                nxt.append(mutate(child_a, 0.1))
                nxt.append(mutate(child_b, 0.1))
            pop = nxt[:30]
    finally:
        random.setstate(state)
    assert max(scalar_gene_count(c) for c in pop) <= MAX_SCALAR_GENES


# ---------------------------------------------------------------------------
# Validity backstop and the guardrail config sync (G4)
# ---------------------------------------------------------------------------


def test_is_valid_chromosome_rejects_over_cap() -> None:
    assert is_valid_chromosome(_both(3, 2, 3, 2)) is True  # 10

    over = _both(3, 3, 3, 2)  # 11: list caps are all satisfied
    assert scalar_gene_count(over) == 11
    assert is_valid_chromosome(over) is False


def test_g4_guardrail_default_syncs_with_operators_constant() -> None:
    from vibe_quant.discovery.guardrails import GuardrailConfig

    assert MAX_SCALAR_GENES == 10
    assert GuardrailConfig().max_complexity == MAX_SCALAR_GENES
