"""Population diversity metrics and interventions for GA discovery.

Monitors Shannon entropy across indicator types, directions, and conditions.
Injects random immigrants when diversity drops below threshold.
"""

from __future__ import annotations

import math
from collections import Counter

from vibe_quant.discovery.operators import (
    StrategyChromosome,
    _random_chromosome,
)


def _shannon_entropy(counts: Counter[str]) -> float:
    """Compute Shannon entropy from a frequency counter.

    Returns entropy in bits. Returns 0 for empty/single-value counters.
    """
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def population_entropy(population: list[StrategyChromosome]) -> float:
    """Compute normalized population entropy across indicator/direction/condition loci.

    Returns a value in [0, 1] where 0 = monoculture, 1 = max diversity.
    Averages normalized Shannon entropy across three loci:
    - Indicator types used (entry + exit genes)
    - Direction (long/short/both)
    - Condition types (all genes)
    """
    if len(population) <= 1:
        return 0.0

    ind_counter: Counter[str] = Counter()
    cond_counter: Counter[str] = Counter()
    dir_counter: Counter[str] = Counter()

    for chrom in population:
        dir_val = chrom.direction.value if hasattr(chrom.direction, "value") else str(chrom.direction)
        dir_counter[dir_val] += 1
        for gene in chrom.entry_genes + chrom.exit_genes:
            ind_counter[gene.indicator_type] += 1
            cond_val = gene.condition.value if hasattr(gene.condition, "value") else str(gene.condition)
            cond_counter[cond_val] += 1

    from vibe_quant.discovery.operators import ConditionType, Direction, _ensure_pool, _INDICATOR_NAMES
    _ensure_pool()

    n_indicators = max(len(_INDICATOR_NAMES), 1)
    n_directions = len(Direction)
    n_conditions = len(ConditionType)

    max_ind = math.log2(n_indicators) if n_indicators > 1 else 1.0
    max_dir = math.log2(n_directions) if n_directions > 1 else 1.0
    max_cond = math.log2(n_conditions) if n_conditions > 1 else 1.0

    norm_ind = _shannon_entropy(ind_counter) / max_ind if max_ind > 0 else 0.0
    norm_dir = _shannon_entropy(dir_counter) / max_dir if max_dir > 0 else 0.0
    norm_cond = _shannon_entropy(cond_counter) / max_cond if max_cond > 0 else 0.0

    return (norm_ind + norm_dir + norm_cond) / 3.0


def should_inject_immigrants(entropy: float, threshold: float = 0.3) -> bool:
    """Check if entropy is low enough to trigger immigrant injection."""
    return entropy < threshold


def inject_random_immigrants(
    population: list[StrategyChromosome],
    fitness_scores: list[float],
    fraction: float = 0.1,
    direction_constraint: object | None = None,
) -> list[StrategyChromosome]:
    """Replace worst individuals with random immigrants.

    Args:
        population: Current population.
        fitness_scores: Parallel fitness scores.
        fraction: Fraction of population to replace (e.g. 0.1 = 10%).
        direction_constraint: Direction constraint for new chromosomes.

    Returns:
        New population with immigrants replacing worst individuals.
    """
    n_replace = max(1, int(len(population) * fraction))

    indexed = sorted(enumerate(fitness_scores), key=lambda x: x[1])
    worst_indices = {idx for idx, _ in indexed[:n_replace]}

    new_pop: list[StrategyChromosome] = []
    for i, chrom in enumerate(population):
        if i in worst_indices:
            new_pop.append(_random_chromosome(direction_constraint=direction_constraint))
        else:
            new_pop.append(chrom)

    return new_pop
