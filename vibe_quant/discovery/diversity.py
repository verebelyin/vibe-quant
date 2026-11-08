"""Population diversity metrics and interventions for GA discovery.

Monitors Shannon entropy across indicator types, directions, and conditions.
Injects random immigrants when diversity drops below threshold.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING

from vibe_quant.discovery.genome import genome_kind
from vibe_quant.discovery.operators import (
    Direction,
    StrategyChromosome,
    _random_chromosome,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from vibe_quant.discovery.param_genome import ParamGenome


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


def _param_population_entropy(population: Sequence[ParamGenome]) -> float:
    """Normalized Shannon entropy of the values at each free param, averaged.

    Per free param: entropy of the population's values over ``log2(min(n, n_values))``
    (the most a population of ``n`` can show), so a population of distinct values
    scores 1 and identical genomes score 0. Fixed params carry no diversity.
    """
    free = population[0].port.free_params()
    if len(population) <= 1 or not free:
        return 0.0
    total = 0.0
    for param in free:
        n_values = len(param.choices) if param.kind == "choice" else param.n_steps + 1
        ceiling = math.log2(min(len(population), n_values))
        if ceiling <= 0:
            continue
        counts: Counter[str] = Counter(repr(genome.values[param.name]) for genome in population)
        total += _shannon_entropy(counts) / ceiling
    return total / len(free)


def population_entropy(population: list[StrategyChromosome]) -> float:
    """Compute normalized population entropy across indicator/direction/condition loci.

    Returns a value in [0, 1] where 0 = monoculture, 1 = max diversity.
    Averages normalized Shannon entropy across three loci:
    - Indicator types used (entry + exit genes, both sides)
    - Direction (long/short/both)
    - Condition types (all genes, both sides)
    """
    if len(population) <= 1:
        return 0.0
    if genome_kind(population[0]) == "param":
        return _param_population_entropy(population)  # type: ignore[arg-type]

    ind_counter: Counter[str] = Counter()
    cond_counter: Counter[str] = Counter()
    dir_counter: Counter[str] = Counter()

    for chrom in population:
        dir_val = chrom.direction.value if hasattr(chrom.direction, "value") else str(chrom.direction)
        dir_counter[dir_val] += 1
        # Short-side genes count too (inydc T2, R19); they are empty for a
        # non-BOTH chromosome, so a single-direction population is unchanged.
        for gene in (
            chrom.entry_genes + chrom.exit_genes + chrom.short_entry_genes + chrom.short_exit_genes
        ):
            ind_counter[gene.indicator_type] += 1
            cond_val = gene.condition.value if hasattr(gene.condition, "value") else str(gene.condition)
            cond_counter[cond_val] += 1

    from vibe_quant.discovery.operators import (  # noqa: I001
        ConditionType,
        Direction,
        _ensure_pool,
        _INDICATOR_NAMES,
    )
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


def immigrant_count(population_size: int, fraction: float) -> int:
    """Number of immigrants for ``fraction``: 0 when fraction <= 0, else >= 1."""
    if fraction <= 0.0 or population_size <= 0:
        return 0
    return max(1, int(population_size * fraction))


def inject_random_immigrants(
    population: list[StrategyChromosome],
    fitness_scores: Sequence[float] | None,
    fraction: float = 0.1,
    direction_constraint: Direction | None = None,
    protected: Collection[int] = (),
    random_fn: Callable[[], StrategyChromosome] | None = None,
) -> list[StrategyChromosome]:
    """Replace individuals with random immigrants.

    Args:
        population: Current population.
        fitness_scores: Fitness scores PARALLEL to ``population`` -- the worst
            are replaced. Pass ``None`` when the population is not evaluated
            yet (e.g. freshly evolved offspring): random members are replaced
            instead. Scores of a different population must never be used --
            indexing a new population with the previous generation's scores
            replaced arbitrary members, including the elite.
        fraction: Fraction of population to replace (e.g. 0.1 = 10%).
            ``0`` disables injection.
        direction_constraint: Direction constraint for new chromosomes.
        protected: Indices that must never be replaced (elites).
        random_fn: Draws one immigrant (the run's ``GenomeOps.random``); ``None`` =
            a random legacy chromosome under ``direction_constraint``.

    Returns:
        New population with immigrants replacing the chosen individuals.

    Raises:
        ValueError: If ``fitness_scores`` is not parallel to ``population``.
    """
    if fitness_scores is not None and len(fitness_scores) != len(population):
        msg = (
            f"fitness_scores ({len(fitness_scores)}) not parallel to "
            f"population ({len(population)})"
        )
        raise ValueError(msg)

    protected_set = set(protected)
    candidates = [i for i in range(len(population)) if i not in protected_set]
    n_replace = min(immigrant_count(len(population), fraction), len(candidates))
    if n_replace == 0:
        return list(population)

    if fitness_scores is not None:
        scores = fitness_scores
        candidates.sort(key=lambda i: scores[i])
        replace = set(candidates[:n_replace])
    else:
        replace = set(random.sample(candidates, n_replace))

    draw = random_fn or (lambda: _random_chromosome(direction_constraint=direction_constraint))
    return [draw() if i in replace else chrom for i, chrom in enumerate(population)]
