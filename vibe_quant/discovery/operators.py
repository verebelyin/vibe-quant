"""Genetic operators for strategy discovery.

Provides crossover, mutation, tournament selection, elitism, and population
initialization for evolving trading strategy chromosomes.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Genome types (stub -- will be replaced by genome.py when available)
# ---------------------------------------------------------------------------

# Indicator pool: maps indicator type -> {param_name: (min, max)}
INDICATOR_POOL: dict[str, dict[str, tuple[float, float]]] = {
    "RSI": {"period": (5, 50)},
    "EMA": {"period": (5, 200)},
    "SMA": {"period": (5, 200)},
    "WMA": {"period": (5, 200)},
    "DEMA": {"period": (5, 200)},
    "TEMA": {"period": (5, 200)},
    "MACD": {"fast_period": (5, 30), "slow_period": (15, 60), "signal_period": (3, 20)},
    "STOCH": {"period_k": (5, 30), "period_d": (2, 10)},
    "CCI": {"period": (10, 50)},
    "WILLR": {"period": (5, 30)},
    "ROC": {"period": (5, 30)},
    "ATR": {"period": (5, 50)},
    "BBANDS": {"period": (10, 50), "std_dev": (1.0, 4.0)},
    "KC": {"period": (10, 50), "atr_multiplier": (0.5, 5.0)},
    "DONCHIAN": {"period": (10, 50)},
    "MFI": {"period": (5, 50)},
}

_INDICATOR_NAMES = list(INDICATOR_POOL.keys())


class ConditionType(Enum):
    """Condition types for gene evaluation."""

    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


_CONDITION_TYPES = list(ConditionType)

# Complementary pairs for mutation swaps
_CONDITION_COMPLEMENTS: dict[ConditionType, ConditionType] = {
    ConditionType.GT: ConditionType.LT,
    ConditionType.LT: ConditionType.GT,
    ConditionType.GTE: ConditionType.LTE,
    ConditionType.LTE: ConditionType.GTE,
    ConditionType.CROSSES_ABOVE: ConditionType.CROSSES_BELOW,
    ConditionType.CROSSES_BELOW: ConditionType.CROSSES_ABOVE,
}


class Direction(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"
    BOTH = "both"


@dataclass(slots=True)
class StrategyGene:
    """Single gene: one indicator condition.

    Attributes:
        indicator_type: Indicator name from INDICATOR_POOL.
        parameters: Indicator parameter values.
        condition: Comparison condition type.
        threshold: Threshold value for the condition.
    """

    indicator_type: str
    parameters: dict[str, float]
    condition: ConditionType
    threshold: float

    def clone(self) -> StrategyGene:
        """Deep-copy this gene."""
        return StrategyGene(
            indicator_type=self.indicator_type,
            parameters=dict(self.parameters),
            condition=self.condition,
            threshold=self.threshold,
        )


# Constraints
MIN_ENTRY_GENES = 1
MAX_ENTRY_GENES = 5
MIN_EXIT_GENES = 1
MAX_EXIT_GENES = 3
SL_RANGE = (0.5, 10.0)  # stop-loss % range
TP_RANGE = (0.5, 20.0)  # take-profit % range


@dataclass(slots=True)
class StrategyChromosome:
    """Full chromosome encoding a strategy.

    Attributes:
        entry_genes: Genes for entry conditions (1-5).
        exit_genes: Genes for exit conditions (1-3).
        stop_loss_pct: Stop-loss percentage.
        take_profit_pct: Take-profit percentage.
        direction: Trade direction.
    """

    entry_genes: list[StrategyGene]
    exit_genes: list[StrategyGene]
    stop_loss_pct: float
    take_profit_pct: float
    direction: Direction = field(default=Direction.LONG)

    def clone(self) -> StrategyChromosome:
        """Deep-copy this chromosome."""
        return StrategyChromosome(
            entry_genes=[g.clone() for g in self.entry_genes],
            exit_genes=[g.clone() for g in self.exit_genes],
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            direction=self.direction,
        )


type Chromosome = StrategyChromosome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_params(indicator_type: str) -> dict[str, float]:
    """Generate random parameters for an indicator type."""
    ranges = INDICATOR_POOL[indicator_type]
    params: dict[str, float] = {}
    for name, (lo, hi) in ranges.items():
        if isinstance(lo, float):
            params[name] = round(random.uniform(lo, hi), 4)
        else:
            params[name] = float(random.randint(int(lo), int(hi)))
    return params


def _random_gene() -> StrategyGene:
    """Generate a single random gene."""
    ind = random.choice(_INDICATOR_NAMES)
    return StrategyGene(
        indicator_type=ind,
        parameters=_random_params(ind),
        condition=random.choice(_CONDITION_TYPES),
        threshold=round(random.uniform(0, 100), 4),
    )


def _perturb(value: float, frac: float = 0.2, lo: float | None = None, hi: float | None = None) -> float:
    """Perturb a value by +/- frac fraction. Optionally clamp to [lo, hi].

    Optimized with early exit for zero-value case and inlined clamping.
    """
    if value == 0.0:
        # When value is exactly 0, use frac as absolute perturbation range
        # so genes (e.g., thresholds) can mutate away from zero
        result = random.uniform(-frac, frac)
    else:
        delta = value * frac
        result = value + random.uniform(-delta, delta)
    if lo is not None and result < lo:
        result = lo
    if hi is not None and result > hi:
        result = hi
    return round(result, 4)


def _clamp_genes(genes: list[StrategyGene], min_count: int, max_count: int) -> list[StrategyGene]:
    """Ensure gene list length is within [min_count, max_count]."""
    while len(genes) < min_count:
        genes.append(_random_gene())
    if len(genes) > max_count:
        genes = genes[:max_count]
    return genes


def is_valid_chromosome(chrom: Chromosome) -> bool:
    """Check chromosome satisfies all constraints."""
    if not (MIN_ENTRY_GENES <= len(chrom.entry_genes) <= MAX_ENTRY_GENES):
        return False
    if not (MIN_EXIT_GENES <= len(chrom.exit_genes) <= MAX_EXIT_GENES):
        return False
    if not (SL_RANGE[0] <= chrom.stop_loss_pct <= SL_RANGE[1]):
        return False
    if not (TP_RANGE[0] <= chrom.take_profit_pct <= TP_RANGE[1]):
        return False
    for gene in chrom.entry_genes + chrom.exit_genes:
        if gene.indicator_type not in INDICATOR_POOL:
            return False
    return True


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------


def crossover(parent_a: Chromosome, parent_b: Chromosome) -> tuple[Chromosome, Chromosome]:
    """Uniform crossover producing two offspring.

    For each gene position, randomly pick from parent A or B.
    Entry and exit genes are crossed independently.
    SL/TP: randomly picked from one parent or the other per child.

    Args:
        parent_a: First parent chromosome.
        parent_b: Second parent chromosome.

    Returns:
        Tuple of two offspring chromosomes.
    """
    child_a_entries = _crossover_genes(
        parent_a.entry_genes, parent_b.entry_genes, MIN_ENTRY_GENES, MAX_ENTRY_GENES
    )
    child_b_entries = _crossover_genes(
        parent_b.entry_genes, parent_a.entry_genes, MIN_ENTRY_GENES, MAX_ENTRY_GENES
    )
    child_a_exits = _crossover_genes(
        parent_a.exit_genes, parent_b.exit_genes, MIN_EXIT_GENES, MAX_EXIT_GENES
    )
    child_b_exits = _crossover_genes(
        parent_b.exit_genes, parent_a.exit_genes, MIN_EXIT_GENES, MAX_EXIT_GENES
    )

    # SL/TP: random pick per child
    sl_a = parent_a.stop_loss_pct if random.random() < 0.5 else parent_b.stop_loss_pct
    tp_a = parent_a.take_profit_pct if random.random() < 0.5 else parent_b.take_profit_pct
    sl_b = parent_b.stop_loss_pct if random.random() < 0.5 else parent_a.stop_loss_pct
    tp_b = parent_b.take_profit_pct if random.random() < 0.5 else parent_a.take_profit_pct

    dir_a = parent_a.direction if random.random() < 0.5 else parent_b.direction
    dir_b = parent_b.direction if random.random() < 0.5 else parent_a.direction

    child_a = StrategyChromosome(
        entry_genes=child_a_entries,
        exit_genes=child_a_exits,
        stop_loss_pct=sl_a,
        take_profit_pct=tp_a,
        direction=dir_a,
    )
    child_b = StrategyChromosome(
        entry_genes=child_b_entries,
        exit_genes=child_b_exits,
        stop_loss_pct=sl_b,
        take_profit_pct=tp_b,
        direction=dir_b,
    )
    return child_a, child_b


def _crossover_genes(
    genes_a: list[StrategyGene],
    genes_b: list[StrategyGene],
    min_count: int,
    max_count: int,
) -> list[StrategyGene]:
    """Uniform crossover on gene lists of potentially different length.

    Iterates over the max-length of the two parents. At each position,
    randomly picks from whichever parent has a gene at that index; when
    both have one, coin-flip.
    """
    max_len = max(len(genes_a), len(genes_b))
    result: list[StrategyGene] = []
    for i in range(max_len):
        has_a = i < len(genes_a)
        has_b = i < len(genes_b)
        if has_a and has_b:
            chosen = genes_a[i] if random.random() < 0.5 else genes_b[i]
        elif has_a:
            # Include with 50% probability to allow trimming
            if random.random() < 0.5:
                chosen = genes_a[i]
            else:
                continue
        else:
            if random.random() < 0.5:
                chosen = genes_b[i]  # type: ignore[index]
            else:
                continue
        result.append(chosen.clone())
    return _clamp_genes(result, min_count, max_count)


def mutate(chromosome: Chromosome, mutation_rate: float = 0.1) -> Chromosome:
    """Mutate a chromosome in-place-style (returns new chromosome).

    For each gene, with probability mutation_rate:
      - Swap indicator type
      - Perturb parameter (+/-20%)
      - Flip condition type
      - Perturb threshold (+/-20%)

    With lower probability (mutation_rate * 0.3):
      - Add or remove a gene (respecting constraints)

    With small probability (mutation_rate * 0.5):
      - Mutate SL/TP values (+/-20%)

    Args:
        chromosome: Chromosome to mutate.
        mutation_rate: Per-gene mutation probability [0, 1].

    Returns:
        New mutated chromosome.
    """
    chrom = chromosome.clone()

    # Mutate entry genes
    chrom.entry_genes = _mutate_genes(
        chrom.entry_genes, mutation_rate, MIN_ENTRY_GENES, MAX_ENTRY_GENES
    )

    # Mutate exit genes
    chrom.exit_genes = _mutate_genes(
        chrom.exit_genes, mutation_rate, MIN_EXIT_GENES, MAX_EXIT_GENES
    )

    # Mutate SL/TP
    if random.random() < mutation_rate * 0.5:
        chrom.stop_loss_pct = _perturb(chrom.stop_loss_pct, 0.2, SL_RANGE[0], SL_RANGE[1])
    if random.random() < mutation_rate * 0.5:
        chrom.take_profit_pct = _perturb(chrom.take_profit_pct, 0.2, TP_RANGE[0], TP_RANGE[1])

    return chrom


def _mutate_genes(
    genes: list[StrategyGene],
    rate: float,
    min_count: int,
    max_count: int,
) -> list[StrategyGene]:
    """Mutate a list of genes."""
    for gene in genes:
        if random.random() < rate:
            _mutate_single_gene(gene)

    # Structural mutation: add/remove gene
    if random.random() < rate * 0.3:
        if len(genes) < max_count and random.random() < 0.5:
            genes.append(_random_gene())
        elif len(genes) > min_count:
            genes.pop(random.randrange(len(genes)))

    return _clamp_genes(genes, min_count, max_count)


def _mutate_single_gene(gene: StrategyGene) -> None:
    """Mutate a single gene in place. Picks one mutation type at random."""
    mutation_type = random.randint(0, 3)

    if mutation_type == 0:
        # Swap indicator type
        new_ind = random.choice(_INDICATOR_NAMES)
        gene.indicator_type = new_ind
        gene.parameters = _random_params(new_ind)

    elif mutation_type == 1:
        # Perturb parameters
        ranges = INDICATOR_POOL.get(gene.indicator_type, {})
        for pname, val in list(gene.parameters.items()):
            if pname in ranges:
                lo, hi = ranges[pname]
                gene.parameters[pname] = _perturb(val, 0.2, lo, hi)
            else:
                gene.parameters[pname] = _perturb(val, 0.2)

    elif mutation_type == 2:
        # Flip condition
        gene.condition = _CONDITION_COMPLEMENTS.get(gene.condition, random.choice(_CONDITION_TYPES))

    else:
        # Perturb threshold
        gene.threshold = _perturb(gene.threshold, 0.2, 0.0, None)


def tournament_select(
    population: list[Chromosome],
    fitness_scores: Sequence[float],
    tournament_size: int = 3,
) -> Chromosome:
    """Tournament selection: pick best from a random subset.

    Args:
        population: List of chromosomes.
        fitness_scores: Parallel list of fitness values (higher is better).
        tournament_size: Number of contenders per tournament.

    Returns:
        Chromosome with the highest fitness among contenders.

    Raises:
        ValueError: If population is empty or sizes mismatch.
    """
    if not population:
        msg = "Population is empty"
        raise ValueError(msg)
    if len(population) != len(fitness_scores):
        msg = f"Population size ({len(population)}) != fitness size ({len(fitness_scores)})"
        raise ValueError(msg)
    tournament_size = min(tournament_size, len(population))
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda i: fitness_scores[i])
    return population[best_idx]


def apply_elitism(
    population: list[Chromosome],
    fitness_scores: Sequence[float],
    elite_count: int = 2,
) -> list[Chromosome]:
    """Return top elite_count individuals unchanged.

    Uses heapq.nlargest for O(n log k) instead of O(n log n) full sort
    when elite_count << population size (typical: 2 elites from 50+ pop).

    Args:
        population: List of chromosomes.
        fitness_scores: Parallel list of fitness values (higher is better).
        elite_count: Number of elites to preserve.

    Returns:
        List of elite chromosomes (cloned).

    Raises:
        ValueError: If population is empty or sizes mismatch.
    """
    if not population:
        msg = "Population is empty"
        raise ValueError(msg)
    if len(population) != len(fitness_scores):
        msg = f"Population size ({len(population)}) != fitness size ({len(fitness_scores)})"
        raise ValueError(msg)
    elite_count = min(elite_count, len(population))
    # heapq.nlargest is O(n log k) vs sorted O(n log n)
    top_indices = heapq.nlargest(elite_count, range(len(population)), key=lambda i: fitness_scores[i])
    return [population[i].clone() for i in top_indices]


def initialize_population(size: int = 50) -> list[Chromosome]:
    """Generate a population of random valid chromosomes.

    Args:
        size: Number of chromosomes to generate.

    Returns:
        List of valid random chromosomes.
    """
    population: list[Chromosome] = []
    for _ in range(size):
        population.append(_random_chromosome())
    return population


def _random_chromosome() -> Chromosome:
    """Generate a single random valid chromosome."""
    n_entry = random.randint(MIN_ENTRY_GENES, MAX_ENTRY_GENES)
    n_exit = random.randint(MIN_EXIT_GENES, MAX_EXIT_GENES)
    return StrategyChromosome(
        entry_genes=[_random_gene() for _ in range(n_entry)],
        exit_genes=[_random_gene() for _ in range(n_exit)],
        stop_loss_pct=round(random.uniform(*SL_RANGE), 4),
        take_profit_pct=round(random.uniform(*TP_RANGE), 4),
        direction=random.choice(list(Direction)),
    )
