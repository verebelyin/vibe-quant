"""Genetic operators for strategy discovery.

Provides crossover, mutation, tournament selection, elitism, and population
initialization for evolving trading strategy chromosomes.
"""

from __future__ import annotations

import heapq
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Indicator pool -- canonical source is genome.INDICATOR_POOL (IndicatorDef).
# This flat dict is derived at import time for operators that only need
# param ranges. Extra indicators not in genome are appended below.
# ---------------------------------------------------------------------------


def _build_indicator_pool() -> dict[str, dict[str, tuple[float, float]]]:
    """Build flat indicator pool from genome's IndicatorDef objects + extras."""
    # Lazy import to avoid circular dependency (genome imports from operators)
    from vibe_quant.discovery.genome import INDICATOR_POOL as _GENOME_POOL

    pool: dict[str, dict[str, tuple[float, float]]] = {
        name: dict(ind_def.param_ranges) for name, ind_def in _GENOME_POOL.items()
    }
    # Extra oscillator indicators with bounded output ranges.
    # Price-relative indicators (SMA, WMA, DEMA, TEMA, BBANDS, KC, DONCHIAN)
    # are excluded because they require indicator-vs-indicator comparison
    # (not indicator-vs-threshold) to produce meaningful conditions.
    _extras: dict[str, dict[str, tuple[float, float]]] = {
        "CCI": {"period": (10, 50)},
        "WILLR": {"period": (5, 30)},
        "ROC": {"period": (5, 30)},
    }
    for name, ranges in _extras.items():
        if name not in pool:
            pool[name] = ranges
    return pool


# Deferred initialization to break circular import
INDICATOR_POOL: dict[str, dict[str, tuple[float, float]]] = {}
_INDICATOR_NAMES: list[str] = []


def _ensure_pool() -> None:
    """Populate INDICATOR_POOL and THRESHOLD_RANGES on first use (breaks circular import)."""
    global _INDICATOR_NAMES  # noqa: PLW0603
    if not INDICATOR_POOL:
        INDICATOR_POOL.update(_build_indicator_pool())
        _INDICATOR_NAMES.extend(INDICATOR_POOL.keys())
    if not THRESHOLD_RANGES:
        THRESHOLD_RANGES.update(_build_threshold_ranges())


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
    sub_value: str | None = None  # e.g. "signal", "histogram" for MACD

    def clone(self) -> StrategyGene:
        """Deep-copy this gene."""
        return StrategyGene(
            indicator_type=self.indicator_type,
            parameters=dict(self.parameters),
            condition=self.condition,
            threshold=self.threshold,
            sub_value=self.sub_value,
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
    stop_loss_long_pct: float | None = field(default=None)
    stop_loss_short_pct: float | None = field(default=None)
    take_profit_long_pct: float | None = field(default=None)
    take_profit_short_pct: float | None = field(default=None)
    time_filters: dict[str, object] = field(default_factory=dict)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def clone(self) -> StrategyChromosome:
        """Deep-copy this chromosome."""
        return StrategyChromosome(
            entry_genes=[g.clone() for g in self.entry_genes],
            exit_genes=[g.clone() for g in self.exit_genes],
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            direction=self.direction,
            stop_loss_long_pct=self.stop_loss_long_pct,
            stop_loss_short_pct=self.stop_loss_short_pct,
            take_profit_long_pct=self.take_profit_long_pct,
            take_profit_short_pct=self.take_profit_short_pct,
            time_filters=dict(self.time_filters),
            uid=self.uid,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_params(indicator_type: str) -> dict[str, float]:
    """Generate random parameters for an indicator type."""
    _ensure_pool()
    ranges = INDICATOR_POOL[indicator_type]
    params: dict[str, float] = {}
    for name, (lo, hi) in ranges.items():
        if isinstance(lo, float):
            params[name] = round(random.uniform(lo, hi), 4)
        else:
            params[name] = float(random.randint(int(lo), int(hi)))
    _enforce_param_constraints(indicator_type, params)
    return params


def _enforce_param_constraints(indicator_type: str, params: dict[str, float]) -> None:
    """Enforce cross-parameter constraints (e.g. MACD fast < slow)."""
    if indicator_type == "MACD":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        if fast >= slow:
            # Swap so fast < slow, with minimum gap of 1
            params["fast_period"] = min(fast, slow)
            params["slow_period"] = max(fast, slow) + 1


# Indicator-specific threshold ranges to avoid impossible conditions
# that produce 0 trades (the #1 cause of wasted evals).
# Ranges for INDICATOR_POOL indicators are derived from genome.IndicatorDef;
# extras (CCI, WILLR, etc.) are defined here directly.
def _build_threshold_ranges() -> dict[str, tuple[float, float]]:
    """Build threshold ranges: derive from INDICATOR_POOL + manual extras."""
    from vibe_quant.discovery.genome import INDICATOR_POOL as _GENOME_POOL

    ranges = {name: defn.default_threshold_range for name, defn in _GENOME_POOL.items()}
    # Extra oscillators not in INDICATOR_POOL (no IndicatorDef)
    ranges.update({
        "CCI": (-100.0, 100.0),
        "WILLR": (-80.0, -20.0),
        "MFI": (20.0, 80.0),
        "ROC": (-5.0, 5.0),
    })
    return ranges


# Deferred initialization (same pattern as INDICATOR_POOL) to break circular import
THRESHOLD_RANGES: dict[str, tuple[float, float]] = {}


_PRICE_RELATIVE_INDICATORS = frozenset(
    {
        "EMA",
        "BBANDS",
    }
)


def _random_gene() -> StrategyGene:
    """Generate a single random gene with sensible thresholds.

    Uses indicator-specific threshold ranges to avoid conditions
    that are too restrictive to trigger trades.
    """
    _ensure_pool()
    ind = random.choice(_INDICATOR_NAMES)

    # Use indicator-specific threshold ranges
    if ind in _PRICE_RELATIVE_INDICATORS:
        # Price-relative indicators: use 0 threshold (compared to price)
        threshold = 0.0
        # Prefer crosses_above/crosses_below for price-relative indicators
        condition = random.choice(
            [
                ConditionType.CROSSES_ABOVE,
                ConditionType.CROSSES_BELOW,
                ConditionType.GT,
                ConditionType.LT,
            ]
        )
    elif ind in THRESHOLD_RANGES:
        lo, hi = THRESHOLD_RANGES[ind]
        threshold = round(random.uniform(lo, hi), 4)
        condition = random.choice(_CONDITION_TYPES)
    else:
        threshold = round(random.uniform(0, 100), 4)
        condition = random.choice(_CONDITION_TYPES)

    return StrategyGene(
        indicator_type=ind,
        parameters=_random_params(ind),
        condition=condition,
        threshold=threshold,
    )


def _perturb(
    value: float, frac: float = 0.2, lo: float | None = None, hi: float | None = None
) -> float:
    """Perturb a value by +/- frac fraction. Optionally clamp to [lo, hi].

    When value is exactly 0.0, ``frac`` is used as an absolute range
    [-frac, frac] so that genes (e.g., thresholds) can mutate away from
    zero. Without this, zero-valued genes would be permanently stuck
    since ``0 * frac = 0``.
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


def is_valid_chromosome(chrom: StrategyChromosome) -> bool:
    """Check chromosome satisfies all constraints."""
    _ensure_pool()
    if not (MIN_ENTRY_GENES <= len(chrom.entry_genes) <= MAX_ENTRY_GENES):
        return False
    if not (MIN_EXIT_GENES <= len(chrom.exit_genes) <= MAX_EXIT_GENES):
        return False
    if not (SL_RANGE[0] <= chrom.stop_loss_pct <= SL_RANGE[1]):
        return False
    if not (TP_RANGE[0] <= chrom.take_profit_pct <= TP_RANGE[1]):
        return False
    # Per-direction SL/TP range checks
    for attr, valid_range in [
        ("stop_loss_long_pct", SL_RANGE),
        ("stop_loss_short_pct", SL_RANGE),
        ("take_profit_long_pct", TP_RANGE),
        ("take_profit_short_pct", TP_RANGE),
    ]:
        val = getattr(chrom, attr)
        if val is not None and not (valid_range[0] <= val <= valid_range[1]):
            return False

    for gene in chrom.entry_genes + chrom.exit_genes:
        if gene.indicator_type not in INDICATOR_POOL:
            return False
        if gene.indicator_type == "MACD":
            fast = gene.parameters.get("fast_period", 12)
            slow = gene.parameters.get("slow_period", 26)
            if fast >= slow:
                return False
        # Check threshold is in valid range for indicator
        if gene.indicator_type in THRESHOLD_RANGES:
            tlo, thi = THRESHOLD_RANGES[gene.indicator_type]
            if not (tlo <= gene.threshold <= thi):
                return False
    return True


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------


def crossover(
    parent_a: StrategyChromosome, parent_b: StrategyChromosome
) -> tuple[StrategyChromosome, StrategyChromosome]:
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

    # Per-direction SL/TP: random pick per child
    sl_long_a = parent_a.stop_loss_long_pct if random.random() < 0.5 else parent_b.stop_loss_long_pct
    sl_short_a = parent_a.stop_loss_short_pct if random.random() < 0.5 else parent_b.stop_loss_short_pct
    tp_long_a = parent_a.take_profit_long_pct if random.random() < 0.5 else parent_b.take_profit_long_pct
    tp_short_a = parent_a.take_profit_short_pct if random.random() < 0.5 else parent_b.take_profit_short_pct
    sl_long_b = parent_b.stop_loss_long_pct if random.random() < 0.5 else parent_a.stop_loss_long_pct
    sl_short_b = parent_b.stop_loss_short_pct if random.random() < 0.5 else parent_a.stop_loss_short_pct
    tp_long_b = parent_b.take_profit_long_pct if random.random() < 0.5 else parent_a.take_profit_long_pct
    tp_short_b = parent_b.take_profit_short_pct if random.random() < 0.5 else parent_a.take_profit_short_pct

    child_a = StrategyChromosome(
        entry_genes=child_a_entries,
        exit_genes=child_a_exits,
        stop_loss_pct=sl_a,
        take_profit_pct=tp_a,
        direction=dir_a,
        stop_loss_long_pct=sl_long_a,
        stop_loss_short_pct=sl_short_a,
        take_profit_long_pct=tp_long_a,
        take_profit_short_pct=tp_short_a,
    )
    child_b = StrategyChromosome(
        entry_genes=child_b_entries,
        exit_genes=child_b_exits,
        stop_loss_pct=sl_b,
        take_profit_pct=tp_b,
        direction=dir_b,
        stop_loss_long_pct=sl_long_b,
        stop_loss_short_pct=sl_short_b,
        take_profit_long_pct=tp_long_b,
        take_profit_short_pct=tp_short_b,
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


def mutate(chromosome: StrategyChromosome, mutation_rate: float = 0.1) -> StrategyChromosome:
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
        chromosome: StrategyChromosome to mutate.
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

    # Mutate per-direction SL/TP (only if set)
    if chrom.stop_loss_long_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.stop_loss_long_pct = _perturb(chrom.stop_loss_long_pct, 0.2, SL_RANGE[0], SL_RANGE[1])
    if chrom.stop_loss_short_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.stop_loss_short_pct = _perturb(chrom.stop_loss_short_pct, 0.2, SL_RANGE[0], SL_RANGE[1])
    if chrom.take_profit_long_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.take_profit_long_pct = _perturb(chrom.take_profit_long_pct, 0.2, TP_RANGE[0], TP_RANGE[1])
    if chrom.take_profit_short_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.take_profit_short_pct = _perturb(chrom.take_profit_short_pct, 0.2, TP_RANGE[0], TP_RANGE[1])

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
        _ensure_pool()
        new_ind = random.choice(_INDICATOR_NAMES)
        gene.indicator_type = new_ind
        gene.parameters = _random_params(new_ind)
        # Reset threshold to valid range for new indicator
        if new_ind in THRESHOLD_RANGES:
            tlo, thi = THRESHOLD_RANGES[new_ind]
            gene.threshold = round(random.uniform(tlo, thi), 4)
        elif new_ind in _PRICE_RELATIVE_INDICATORS:
            gene.threshold = 0.0

    elif mutation_type == 1:
        # Perturb parameters
        ranges = INDICATOR_POOL.get(gene.indicator_type, {})
        for pname, val in list(gene.parameters.items()):
            if pname in ranges:
                lo, hi = ranges[pname]
                gene.parameters[pname] = _perturb(val, 0.2, lo, hi)
            else:
                gene.parameters[pname] = _perturb(val, 0.2)
        _enforce_param_constraints(gene.indicator_type, gene.parameters)

    elif mutation_type == 2:
        # Flip condition
        gene.condition = _CONDITION_COMPLEMENTS.get(gene.condition, random.choice(_CONDITION_TYPES))

    else:
        # Perturb threshold, clamped to indicator-specific range if known
        if gene.indicator_type in THRESHOLD_RANGES:
            tlo, thi = THRESHOLD_RANGES[gene.indicator_type]
            gene.threshold = _perturb(gene.threshold, 0.2, tlo, thi)
        else:
            gene.threshold = _perturb(gene.threshold, 0.2)


def tournament_select(
    population: list[StrategyChromosome],
    fitness_scores: Sequence[float],
    tournament_size: int = 3,
) -> StrategyChromosome:
    """Tournament selection: pick best from a random subset.

    Args:
        population: List of chromosomes.
        fitness_scores: Parallel list of fitness values (higher is better).
        tournament_size: Number of contenders per tournament.

    Returns:
        StrategyChromosome with the highest fitness among contenders.

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
    population: list[StrategyChromosome],
    fitness_scores: Sequence[float],
    elite_count: int = 2,
) -> list[StrategyChromosome]:
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
    top_indices = heapq.nlargest(
        elite_count, range(len(population)), key=lambda i: fitness_scores[i]
    )
    return [population[i].clone() for i in top_indices]


def initialize_population(
    size: int = 50, direction_constraint: Direction | None = None
) -> list[StrategyChromosome]:
    """Generate a population of random valid chromosomes.

    Args:
        size: Number of chromosomes to generate.
        direction_constraint: If set, all chromosomes use this direction.

    Returns:
        List of valid random chromosomes.
    """
    population: list[StrategyChromosome] = []
    for _ in range(size):
        population.append(_random_chromosome(direction_constraint=direction_constraint))
    return population


def _random_chromosome(
    direction_constraint: Direction | None = None,
) -> StrategyChromosome:
    """Generate a single random valid chromosome.

    Biased toward simpler strategies (1-2 entry genes, 1 exit gene)
    to reduce the 60-78% of genomes that produce 0 trades due to
    overly restrictive multi-condition entries.
    """
    # Weighted toward fewer genes: 1 gene=50%, 2=30%, 3=15%, 4-5=5%
    n_entry = random.choices(
        [1, 2, 3, random.randint(4, MAX_ENTRY_GENES)], weights=[50, 30, 15, 5]
    )[0]
    n_exit = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
    direction = direction_constraint if direction_constraint is not None else random.choice(list(Direction))
    chrom = StrategyChromosome(
        entry_genes=[_random_gene() for _ in range(n_entry)],
        exit_genes=[_random_gene() for _ in range(n_exit)],
        stop_loss_pct=round(random.uniform(*SL_RANGE), 4),
        take_profit_pct=round(random.uniform(*TP_RANGE), 4),
        direction=direction,
    )
    if chrom.direction == Direction.BOTH:
        chrom.stop_loss_long_pct = round(random.uniform(*SL_RANGE), 4)
        chrom.stop_loss_short_pct = round(random.uniform(*SL_RANGE), 4)
        chrom.take_profit_long_pct = round(random.uniform(*TP_RANGE), 4)
        chrom.take_profit_short_pct = round(random.uniform(*TP_RANGE), 4)
    return chrom
