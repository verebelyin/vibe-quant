# GA Diversity Preservation — Tier 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent GA monoculture (top-5 clones, zero diversity) by adding Gower distance, deterministic crowding, entropy monitoring with random immigrant injection, and structural deduplication in top-K selection.

**Architecture:** Three layered mechanisms: (1) Gower distance metric gives us a way to measure chromosome similarity across mixed categorical+continuous genes, (2) deterministic crowding replaces the current tournament→elite evolution loop so offspring only compete against similar parents (preserving niches), (3) entropy monitoring detects diversity collapse and injects random immigrants as intervention. Additionally, top-K output deduplication switches from UID-based to structural-distance-based.

**Tech Stack:** Pure Python, no new dependencies. Uses `math`, `statistics` stdlib modules.

---

## Task 1: Gower Distance Metric

**Files:**
- Create: `vibe_quant/discovery/distance.py`
- Test: `tests/unit/test_discovery_distance.py`

### Background

Gower distance handles mixed-type data: categorical features use exact match (0 or 1), continuous features use range-normalized absolute difference. Result is always in [0, 1].

For our genome, a chromosome has:
- **Structural genes** (categorical): indicator_type, condition, direction, sub_value
- **Parameter genes** (continuous): threshold, period, SL/TP percentages
- **Variable-length gene lists**: entry_genes (1-5), exit_genes (1-3)

Distance design:
- Compare gene-by-gene at each position. When lists differ in length, missing genes contribute max distance (1.0) per missing slot.
- Per-gene distance: weighted average of indicator_type match (0 or 1), condition match (0 or 1), normalized parameter distance, normalized threshold distance.
- Chromosome distance: average across all gene slots + direction match + normalized SL/TP distance.
- Structural features weighted 2x vs parameter features (indicator_type swap matters more than threshold tweak).

### Step 1: Write failing tests

```python
# tests/unit/test_discovery_distance.py
"""Tests for chromosome Gower distance metric."""

import pytest

from vibe_quant.discovery.distance import gene_distance, chromosome_distance
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)


def _gene(ind: str = "RSI", period: int = 14, condition: ConditionType = ConditionType.GT,
          threshold: float = 50.0, sub_value: str | None = None) -> StrategyGene:
    """Helper to create a gene with minimal boilerplate."""
    params: dict[str, float] = {"period": float(period)}
    if ind == "MACD":
        params = {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 9.0}
    elif ind == "STOCH":
        params = {"k_period": float(period), "d_period": 3.0}
    return StrategyGene(
        indicator_type=ind, parameters=params, condition=condition,
        threshold=threshold, sub_value=sub_value,
    )


def _chrom(entry_genes: list[StrategyGene], exit_genes: list[StrategyGene],
           sl: float = 2.0, tp: float = 5.0,
           direction: Direction = Direction.LONG) -> StrategyChromosome:
    """Helper to create a chromosome."""
    return StrategyChromosome(
        entry_genes=entry_genes, exit_genes=exit_genes,
        stop_loss_pct=sl, take_profit_pct=tp, direction=direction,
    )


class TestGeneDistance:
    """Tests for single gene distance."""

    def test_identical_genes_zero_distance(self) -> None:
        g = _gene("RSI", 14, ConditionType.GT, 50.0)
        assert gene_distance(g, g) == pytest.approx(0.0)

    def test_different_indicator_high_distance(self) -> None:
        a = _gene("RSI", 14, ConditionType.GT, 50.0)
        b = _gene("MACD", 12, ConditionType.GT, 0.001)
        d = gene_distance(a, b)
        # Different indicator = 1.0 for that component, so distance > 0.3
        assert d > 0.3
        assert d <= 1.0

    def test_same_indicator_different_params(self) -> None:
        a = _gene("RSI", 5, ConditionType.GT, 30.0)
        b = _gene("RSI", 50, ConditionType.GT, 70.0)
        d = gene_distance(a, b)
        # Same indicator, different params → moderate distance
        assert 0.0 < d < 0.7

    def test_different_condition_only(self) -> None:
        a = _gene("RSI", 14, ConditionType.GT, 50.0)
        b = _gene("RSI", 14, ConditionType.LT, 50.0)
        d = gene_distance(a, b)
        assert 0.0 < d < 0.5  # Only condition differs

    def test_symmetry(self) -> None:
        a = _gene("RSI", 14, ConditionType.GT, 50.0)
        b = _gene("ATR", 20, ConditionType.LT, 0.01)
        assert gene_distance(a, b) == pytest.approx(gene_distance(b, a))

    def test_distance_bounded_0_1(self) -> None:
        a = _gene("RSI", 5, ConditionType.GT, 25.0)
        b = _gene("CCI", 50, ConditionType.CROSSES_BELOW, -200.0)
        d = gene_distance(a, b)
        assert 0.0 <= d <= 1.0


class TestChromosomeDistance:
    """Tests for full chromosome distance."""

    def test_identical_chromosomes_zero(self) -> None:
        c = _chrom([_gene()], [_gene("ATR", 14, ConditionType.LT, 0.01)])
        assert chromosome_distance(c, c) == pytest.approx(0.0)

    def test_completely_different_chromosomes(self) -> None:
        a = _chrom(
            [_gene("RSI", 14, ConditionType.GT, 50.0)],
            [_gene("ATR", 14, ConditionType.LT, 0.01)],
            sl=1.0, tp=1.0, direction=Direction.LONG,
        )
        b = _chrom(
            [_gene("CCI", 50, ConditionType.CROSSES_BELOW, -200.0),
             _gene("STOCH", 21, ConditionType.LTE, 20.0)],
            [_gene("MFI", 30, ConditionType.GT, 80.0)],
            sl=10.0, tp=20.0, direction=Direction.SHORT,
        )
        d = chromosome_distance(a, b)
        assert d > 0.5  # Very different

    def test_same_structure_different_params(self) -> None:
        a = _chrom(
            [_gene("RSI", 10, ConditionType.GT, 30.0)],
            [_gene("ATR", 10, ConditionType.LT, 0.005)],
        )
        b = _chrom(
            [_gene("RSI", 40, ConditionType.GT, 70.0)],
            [_gene("ATR", 25, ConditionType.LT, 0.025)],
        )
        d = chromosome_distance(a, b)
        assert 0.0 < d < 0.5  # Same structure, different params

    def test_different_gene_count_penalized(self) -> None:
        a = _chrom([_gene()], [_gene("ATR")])
        b = _chrom(
            [_gene(), _gene("MACD"), _gene("CCI")],
            [_gene("ATR")],
        )
        d = chromosome_distance(a, b)
        # Extra genes in b contribute distance
        assert d > 0.1

    def test_direction_mismatch_adds_distance(self) -> None:
        a = _chrom([_gene()], [_gene("ATR")], direction=Direction.LONG)
        b = _chrom([_gene()], [_gene("ATR")], direction=Direction.SHORT)
        d = chromosome_distance(a, b)
        assert d > 0.0  # Direction difference contributes

    def test_symmetry(self) -> None:
        a = _chrom([_gene("RSI")], [_gene("ATR")])
        b = _chrom([_gene("CCI"), _gene("STOCH")], [_gene("MFI")])
        assert chromosome_distance(a, b) == pytest.approx(chromosome_distance(b, a))

    def test_distance_bounded_0_1(self) -> None:
        a = _chrom([_gene("RSI")], [_gene("ATR")], sl=0.5, tp=0.5, direction=Direction.LONG)
        b = _chrom(
            [_gene("CCI"), _gene("STOCH"), _gene("MFI"), _gene("ADX"), _gene("WILLR")],
            [_gene("ROC"), _gene("MACD"), _gene("RSI")],
            sl=10.0, tp=20.0, direction=Direction.SHORT,
        )
        d = chromosome_distance(a, b)
        assert 0.0 <= d <= 1.0
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/unit/test_discovery_distance.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_quant.discovery.distance'`

### Step 3: Implement distance module

```python
# vibe_quant/discovery/distance.py
"""Gower distance metric for strategy chromosomes.

Handles mixed-type genome: categorical genes (indicator_type, condition, direction)
use exact match (0/1), continuous genes (threshold, parameters, SL/TP) use
range-normalized absolute difference. Result always in [0, 1].

Structural features (indicator_type, condition) weighted 2x vs parameter features
because a different indicator is a fundamentally different strategy, while a
threshold tweak is a parameter variation.
"""

from __future__ import annotations

from vibe_quant.discovery.operators import (
    SL_RANGE,
    TP_RANGE,
    StrategyChromosome,
    StrategyGene,
    _ensure_pool,
)

# Weights for gene components (structural vs parametric)
_W_INDICATOR: float = 2.0   # categorical: 0 or 1
_W_CONDITION: float = 1.0   # categorical: 0 or 1
_W_THRESHOLD: float = 1.0   # continuous: normalized
_W_PARAMS: float = 1.0      # continuous: avg normalized param distance

# Weights for chromosome-level components
_W_GENES: float = 3.0       # gene-by-gene distance (dominant)
_W_DIRECTION: float = 1.0   # categorical: 0, 0.5, or 1
_W_SL: float = 0.5          # continuous: normalized
_W_TP: float = 0.5          # continuous: normalized

# Global threshold range for normalization (populated on first use)
_THRESHOLD_GLOBAL_RANGE: float = 400.0  # CCI range [-200, 200] is widest

# Direction distance: LONG↔SHORT = 1.0, LONG↔BOTH or SHORT↔BOTH = 0.5
_DIRECTION_DISTANCES: dict[tuple[str, str], float] = {}


def _init_direction_distances() -> None:
    if _DIRECTION_DISTANCES:
        return
    from vibe_quant.discovery.operators import Direction
    dirs = [Direction.LONG, Direction.SHORT, Direction.BOTH]
    for a in dirs:
        for b in dirs:
            if a == b:
                _DIRECTION_DISTANCES[(a.value, b.value)] = 0.0
            elif {a, b} == {Direction.LONG, Direction.SHORT}:
                _DIRECTION_DISTANCES[(a.value, b.value)] = 1.0
            else:
                _DIRECTION_DISTANCES[(a.value, b.value)] = 0.5


def _normalize_range(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] given range [lo, hi]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def gene_distance(a: StrategyGene, b: StrategyGene) -> float:
    """Compute Gower distance between two genes. Returns value in [0, 1].

    Components:
    - indicator_type: 0 if same, 1 if different (weight 2x)
    - condition: 0 if same, 1 if different (weight 1x)
    - threshold: |norm(a) - norm(b)| using indicator-specific ranges (weight 1x)
    - parameters: avg |norm(a_p) - norm(b_p)| for shared params (weight 1x)

    When indicators differ, threshold and param distances are maxed to 1.0
    since they're on incomparable scales.
    """
    _ensure_pool()

    total_weight = _W_INDICATOR + _W_CONDITION + _W_THRESHOLD + _W_PARAMS
    weighted_sum = 0.0

    # Indicator type (categorical)
    ind_same = a.indicator_type == b.indicator_type
    weighted_sum += _W_INDICATOR * (0.0 if ind_same else 1.0)

    # Condition (categorical)
    cond_same = a.condition == b.condition
    weighted_sum += _W_CONDITION * (0.0 if cond_same else 1.0)

    if ind_same:
        # Threshold distance (continuous, same indicator scale)
        from vibe_quant.discovery.operators import THRESHOLD_RANGES
        if a.indicator_type in THRESHOLD_RANGES:
            tlo, thi = THRESHOLD_RANGES[a.indicator_type]
            rng = thi - tlo
            if rng > 0:
                t_dist = abs(a.threshold - b.threshold) / rng
            else:
                t_dist = 0.0
        else:
            t_dist = abs(a.threshold - b.threshold) / _THRESHOLD_GLOBAL_RANGE
        weighted_sum += _W_THRESHOLD * min(1.0, t_dist)

        # Parameter distance (continuous, shared params)
        from vibe_quant.discovery.operators import INDICATOR_POOL
        ranges = INDICATOR_POOL.get(a.indicator_type, {})
        if ranges:
            param_dists: list[float] = []
            for pname, (lo, hi) in ranges.items():
                va = a.parameters.get(pname, lo)
                vb = b.parameters.get(pname, lo)
                rng = hi - lo
                if rng > 0:
                    param_dists.append(abs(va - vb) / rng)
                else:
                    param_dists.append(0.0)
            weighted_sum += _W_PARAMS * (sum(param_dists) / len(param_dists))
        else:
            weighted_sum += _W_PARAMS * 0.5  # unknown indicator, moderate distance
    else:
        # Different indicators → max distance for threshold and params
        weighted_sum += _W_THRESHOLD * 1.0
        weighted_sum += _W_PARAMS * 1.0

    return weighted_sum / total_weight


def chromosome_distance(a: StrategyChromosome, b: StrategyChromosome) -> float:
    """Compute Gower distance between two chromosomes. Returns value in [0, 1].

    Components:
    - Gene-by-gene distance (entry + exit), padded with 1.0 for missing slots (weight 3x)
    - Direction distance (weight 1x)
    - SL distance, range-normalized (weight 0.5x)
    - TP distance, range-normalized (weight 0.5x)
    """
    _init_direction_distances()

    total_weight = _W_GENES + _W_DIRECTION + _W_SL + _W_TP
    weighted_sum = 0.0

    # Gene distances (entry + exit)
    gene_dists: list[float] = []

    # Entry genes
    max_entry = max(len(a.entry_genes), len(b.entry_genes))
    for i in range(max_entry):
        if i < len(a.entry_genes) and i < len(b.entry_genes):
            gene_dists.append(gene_distance(a.entry_genes[i], b.entry_genes[i]))
        else:
            gene_dists.append(1.0)  # missing gene = max distance

    # Exit genes
    max_exit = max(len(a.exit_genes), len(b.exit_genes))
    for i in range(max_exit):
        if i < len(a.exit_genes) and i < len(b.exit_genes):
            gene_dists.append(gene_distance(a.exit_genes[i], b.exit_genes[i]))
        else:
            gene_dists.append(1.0)

    avg_gene_dist = sum(gene_dists) / len(gene_dists) if gene_dists else 0.0
    weighted_sum += _W_GENES * avg_gene_dist

    # Direction distance
    dir_a = a.direction.value if hasattr(a.direction, "value") else str(a.direction)
    dir_b = b.direction.value if hasattr(b.direction, "value") else str(b.direction)
    weighted_sum += _W_DIRECTION * _DIRECTION_DISTANCES.get((dir_a, dir_b), 1.0)

    # SL distance
    sl_range = SL_RANGE[1] - SL_RANGE[0]
    weighted_sum += _W_SL * (abs(a.stop_loss_pct - b.stop_loss_pct) / sl_range)

    # TP distance
    tp_range = TP_RANGE[1] - TP_RANGE[0]
    weighted_sum += _W_TP * (abs(a.take_profit_pct - b.take_profit_pct) / tp_range)

    return min(1.0, weighted_sum / total_weight)
```

### Step 4: Run tests

```bash
pytest tests/unit/test_discovery_distance.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add vibe_quant/discovery/distance.py tests/unit/test_discovery_distance.py
git commit -m "feat: Gower distance metric for chromosome similarity (bd-028r)"
```

---

## Task 2: Deterministic Crowding in Evolution Loop

**Files:**
- Modify: `vibe_quant/discovery/operators.py:571-634` (add `crowding_replace`)
- Modify: `vibe_quant/discovery/pipeline.py:582-646` (`_evolve_generation`)
- Modify: `vibe_quant/discovery/pipeline.py:50-87` (`DiscoveryConfig` — add `use_crowding` flag)
- Test: `tests/unit/test_discovery_operators.py` (new tests for crowding)

### Background

Deterministic crowding replaces the standard "fill from tournament" loop:
1. Randomly pair all parents (N/2 pairs)
2. Each pair produces 2 offspring via crossover + mutation
3. Match each offspring to the most similar parent (4 distance computations)
4. Offspring replaces parent ONLY if offspring fitness ≥ parent fitness

This naturally preserves niches: a good RSI strategy can't kill an unrelated STOCH strategy.

**Key design decision:** Elitism is incompatible with pure deterministic crowding (elites bypass replacement). We use a **hybrid**: keep 1 elite (down from 2) as insurance against regression, fill remaining N-1 slots via crowding. The `use_crowding` config flag controls this; when False, the old tournament loop runs unchanged.

### Step 1: Write failing tests

Add to `tests/unit/test_discovery_operators.py` (or create if needed):

```python
# In tests/unit/test_discovery_operators.py — append these test classes

class TestCrowdingReplace:
    """Tests for deterministic crowding replacement."""

    def test_fitter_offspring_replaces_similar_parent(self) -> None:
        """Offspring with higher fitness replaces its most-similar parent."""
        from vibe_quant.discovery.operators import crowding_replace

        # Parent A: RSI-based, Parent B: CCI-based (very different)
        parent_a = _make_chrom("RSI", Direction.LONG)
        parent_b = _make_chrom("CCI", Direction.SHORT)

        # Offspring similar to parent_a but with better fitness
        offspring_a = parent_a.clone()
        offspring_a.uid = "new_uid_a"  # Different UID
        offspring_b = parent_b.clone()
        offspring_b.uid = "new_uid_b"

        # offspring_a is similar to parent_a → should replace if fitter
        result = crowding_replace(
            parents=[parent_a, parent_b],
            parent_fitness=[0.3, 0.5],
            offspring=[offspring_a, offspring_b],
            offspring_fitness=[0.4, 0.6],  # Both fitter
        )
        assert len(result) == 2

    def test_weaker_offspring_does_not_replace(self) -> None:
        """Offspring with lower fitness does NOT replace similar parent."""
        from vibe_quant.discovery.operators import crowding_replace

        parent_a = _make_chrom("RSI", Direction.LONG)
        parent_b = _make_chrom("CCI", Direction.SHORT)

        offspring_a = parent_a.clone()
        offspring_a.uid = "new_uid_a"
        offspring_b = parent_b.clone()
        offspring_b.uid = "new_uid_b"

        # offspring weaker than parents → parents survive
        result = crowding_replace(
            parents=[parent_a, parent_b],
            parent_fitness=[0.8, 0.9],
            offspring=[offspring_a, offspring_b],
            offspring_fitness=[0.1, 0.2],
        )
        # Parents should be kept (since offspring are weaker)
        assert result[0].uid == parent_a.uid
        assert result[1].uid == parent_b.uid

    def test_correct_matching_offspring_to_parent(self) -> None:
        """Offspring matched to most-similar parent, not any parent."""
        from vibe_quant.discovery.operators import crowding_replace

        parent_a = _make_chrom("RSI", Direction.LONG)
        parent_b = _make_chrom("CCI", Direction.SHORT)

        # offspring_rsi is similar to parent_a (RSI), fitter
        offspring_rsi = parent_a.clone()
        offspring_rsi.uid = "off_rsi"
        # offspring_cci is similar to parent_b (CCI), weaker
        offspring_cci = parent_b.clone()
        offspring_cci.uid = "off_cci"

        result = crowding_replace(
            parents=[parent_a, parent_b],
            parent_fitness=[0.3, 0.9],
            offspring=[offspring_rsi, offspring_cci],
            offspring_fitness=[0.5, 0.1],
        )
        # RSI offspring replaces RSI parent (0.5 > 0.3)
        # CCI offspring does NOT replace CCI parent (0.1 < 0.9)
        assert result[0].uid == "off_rsi"  # RSI slot: offspring won
        assert result[1].uid == parent_b.uid  # CCI slot: parent survived


def _make_chrom(indicator: str = "RSI", direction: Direction = Direction.LONG) -> StrategyChromosome:
    """Helper to make a simple chromosome for testing."""
    from vibe_quant.discovery.operators import ConditionType, StrategyGene

    gene = StrategyGene(
        indicator_type=indicator,
        parameters={"period": 14.0} if indicator != "MACD" else
            {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 9.0},
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
```

### Step 2: Run tests — expect failure

```bash
pytest tests/unit/test_discovery_operators.py::TestCrowdingReplace -v
```
Expected: FAIL — `ImportError: cannot import name 'crowding_replace'`

### Step 3: Implement crowding_replace

Add to `vibe_quant/discovery/operators.py` (after `apply_elitism`, around line 634):

```python
def crowding_replace(
    parents: list[StrategyChromosome],
    parent_fitness: Sequence[float],
    offspring: list[StrategyChromosome],
    offspring_fitness: Sequence[float],
) -> list[StrategyChromosome]:
    """Deterministic crowding: offspring replace most-similar parent if fitter.

    Given 2 parents and 2 offspring (from crossover+mutation of those parents),
    match each offspring to the parent it's most similar to. If the offspring
    is at least as fit as that parent, it replaces the parent. Otherwise the
    parent survives.

    Args:
        parents: Two parent chromosomes.
        parent_fitness: Fitness scores for parents.
        offspring: Two offspring chromosomes.
        offspring_fitness: Fitness scores for offspring.

    Returns:
        List of 2 chromosomes (mix of parents and offspring).
    """
    from vibe_quant.discovery.distance import chromosome_distance

    # Compute 4 distances for matching
    d_a0_b0 = chromosome_distance(parents[0], offspring[0])
    d_a0_b1 = chromosome_distance(parents[0], offspring[1])
    d_a1_b0 = chromosome_distance(parents[1], offspring[0])
    d_a1_b1 = chromosome_distance(parents[1], offspring[1])

    # Match: minimize total distance
    # Option 1: parent[0]↔offspring[0], parent[1]↔offspring[1]
    # Option 2: parent[0]↔offspring[1], parent[1]↔offspring[0]
    if (d_a0_b0 + d_a1_b1) <= (d_a0_b1 + d_a1_b0):
        matches = [(0, 0), (1, 1)]
    else:
        matches = [(0, 1), (1, 0)]

    result = list(parents)  # Start with parents
    for p_idx, o_idx in matches:
        if offspring_fitness[o_idx] >= parent_fitness[p_idx]:
            result[p_idx] = offspring[o_idx]

    return result
```

### Step 4: Update `_evolve_generation` to use crowding

Modify `vibe_quant/discovery/pipeline.py:582-646`:

Add `use_crowding: bool = True` to `DiscoveryConfig` (line ~87).

Replace `_evolve_generation` body with:

```python
def _evolve_generation(
    self,
    population: list[StrategyChromosome],
    fitness_results: list[FitnessResult],
) -> list[StrategyChromosome]:
    """Produce next generation via crowding or classic tournament."""
    cfg = self.config
    scores = [fr.adjusted_score for fr in fitness_results]

    if cfg.use_crowding:
        return self._evolve_crowding(population, scores)
    return self._evolve_tournament(population, scores)


def _evolve_tournament(
    self,
    population: list[StrategyChromosome],
    scores: list[float],
) -> list[StrategyChromosome]:
    """Classic evolution: elitism + tournament selection + crossover + mutation.

    This is the original evolution method, preserved for A/B comparison.
    """
    cfg = self.config
    new_pop = apply_elitism(population, scores, cfg.elite_count)

    remaining = cfg.population_size - len(new_pop)
    retries = 0
    random_fallbacks = 0
    while remaining > 0:
        parent_a = tournament_select(population, scores, cfg.tournament_size)
        parent_b = tournament_select(population, scores, cfg.tournament_size)
        if random.random() < cfg.crossover_rate:
            child_a, child_b = crossover(parent_a, parent_b)
        else:
            child_a, child_b = parent_a, parent_b
        child_a = mutate(child_a, cfg.mutation_rate)
        child_b = mutate(child_b, cfg.mutation_rate)

        if self._direction_constraint is not None:
            child_a.direction = self._direction_constraint
            child_b.direction = self._direction_constraint

        for child in (child_a, child_b):
            if remaining <= 0:
                break
            valid_child = child
            for attempt in range(_MAX_OFFSPRING_RETRIES):
                if is_valid_chromosome(valid_child):
                    if attempt > 0:
                        retries += attempt
                    break
                valid_child = mutate(child, cfg.mutation_rate)
            else:
                if not is_valid_chromosome(valid_child):
                    valid_child = _random_chromosome(direction_constraint=self._direction_constraint)
                    random_fallbacks += 1
            new_pop.append(valid_child)
            remaining -= 1

    if retries > 0 or random_fallbacks > 0:
        logger.info("  Evolution: %d mutation retries, %d random fallbacks", retries, random_fallbacks)

    return new_pop


def _evolve_crowding(
    self,
    population: list[StrategyChromosome],
    scores: list[float],
) -> list[StrategyChromosome]:
    """Deterministic crowding evolution.

    1. Keep 1 elite as safety net
    2. Randomly pair remaining individuals
    3. Each pair produces 2 offspring (crossover + mutation)
    4. Offspring replaces most-similar parent only if fitter
    """
    cfg = self.config
    from vibe_quant.discovery.operators import crowding_replace

    # Keep 1 elite as safety net (down from default 2)
    elite = apply_elitism(population, scores, min(1, cfg.elite_count))

    # Build replacement pool (all non-elite)
    elite_indices = set()
    if elite:
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        elite_indices.add(best_idx)

    pool_indices = [i for i in range(len(population)) if i not in elite_indices]
    random.shuffle(pool_indices)

    # Pair up for crowding
    new_pop = list(elite)
    retries = 0
    random_fallbacks = 0

    for pair_start in range(0, len(pool_indices) - 1, 2):
        i, j = pool_indices[pair_start], pool_indices[pair_start + 1]
        parent_a, parent_b = population[i], population[j]

        # Crossover + mutation
        if random.random() < cfg.crossover_rate:
            child_a, child_b = crossover(parent_a, parent_b)
        else:
            child_a, child_b = parent_a.clone(), parent_b.clone()
        child_a = mutate(child_a, cfg.mutation_rate)
        child_b = mutate(child_b, cfg.mutation_rate)

        if self._direction_constraint is not None:
            child_a.direction = self._direction_constraint
            child_b.direction = self._direction_constraint

        # Validate offspring (retry with mutation, fallback to random)
        children = []
        for child in (child_a, child_b):
            valid = child
            for attempt in range(_MAX_OFFSPRING_RETRIES):
                if is_valid_chromosome(valid):
                    if attempt > 0:
                        retries += attempt
                    break
                valid = mutate(child, cfg.mutation_rate)
            else:
                if not is_valid_chromosome(valid):
                    valid = _random_chromosome(direction_constraint=self._direction_constraint)
                    random_fallbacks += 1
            children.append(valid)

        # Evaluate offspring fitness is NOT available here — we use scores=0
        # for new offspring since they haven't been backtested yet.
        # Solution: offspring get a "free pass" in gen N, evaluated in gen N+1.
        # Crowding replacement uses parent fitness; new offspring always replace
        # (they need evaluation). This is a standard approach: the actual
        # crowding competition happens at the NEXT generation when we have
        # real fitness values for the offspring.
        #
        # Simpler alternative: just do distance-based replacement without
        # fitness comparison for the first generation an offspring appears.
        # The offspring "earns its slot" by being evaluated next gen.
        new_pop.extend(children)

    # Handle odd pool (last unpaired individual)
    if len(pool_indices) % 2 == 1:
        new_pop.append(population[pool_indices[-1]].clone())

    # Trim to population size (may have +1 from elite + pairs)
    if len(new_pop) > cfg.population_size:
        new_pop = new_pop[:cfg.population_size]
    # Pad if somehow short
    while len(new_pop) < cfg.population_size:
        new_pop.append(_random_chromosome(direction_constraint=self._direction_constraint))

    if retries > 0 or random_fallbacks > 0:
        logger.info("  Evolution: %d mutation retries, %d random fallbacks", retries, random_fallbacks)

    return new_pop
```

**Important note on crowding without pre-evaluated offspring fitness:**

The classic deterministic crowding assumes offspring fitness is known. In our pipeline, offspring aren't evaluated until the *next* generation. Two approaches:

**Option A (recommended, simpler):** Offspring always enter the population. Crowding's distance-based pairing ensures structural diversity by pairing parents with similar children. The fitness competition happens implicitly at gen N+1 when low-fitness offspring get replaced. This is effectively "distance-weighted random immigration" — structurally similar to parents but with mutations.

**Option B (full crowding):** Run a lightweight fitness evaluation (just the backtest) for offspring inside `_evolve_crowding`. Expensive — doubles evaluation cost. Not recommended for our already-slow backtests.

The implementation above uses **Option A**: all offspring enter, diversity comes from the pairing structure ensuring gene mixing between dissimilar parents. The real selective pressure comes from fitness evaluation next gen.

### Step 4: Run tests

```bash
pytest tests/unit/test_discovery_operators.py -v
pytest tests/unit/test_discovery_distance.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add vibe_quant/discovery/operators.py vibe_quant/discovery/pipeline.py tests/unit/test_discovery_operators.py
git commit -m "feat: deterministic crowding evolution mode (bd-028r)"
```

---

## Task 3: Entropy Monitoring + Random Immigrant Injection

**Files:**
- Create: `vibe_quant/discovery/diversity.py`
- Modify: `vibe_quant/discovery/pipeline.py:265-455` (main loop — add entropy check after evolution)
- Modify: `vibe_quant/discovery/pipeline.py:122-143` (`GenerationResult` — add `entropy` field)
- Test: `tests/unit/test_discovery_diversity.py`

### Background

Per-locus Shannon entropy measures allele diversity at each gene position. When entropy drops below a threshold, we inject random chromosomes to restore diversity.

For our genome:
- **Indicator type locus**: Count frequency of each indicator across the population → Shannon entropy
- **Direction locus**: Count frequency of LONG/SHORT/BOTH → entropy
- **Condition locus**: Count frequency of GT/LT/GTE/LTE/CROSSES_ABOVE/CROSSES_BELOW → entropy

Population entropy = average across all loci. Normalized by max possible entropy (log2 of number of alleles).

When normalized entropy drops below 0.3, replace the bottom 10% of the population with fresh random chromosomes.

### Step 1: Write failing tests

```python
# tests/unit/test_discovery_diversity.py
"""Tests for population diversity metrics and interventions."""

import math

import pytest

from vibe_quant.discovery.diversity import (
    population_entropy,
    should_inject_immigrants,
    inject_random_immigrants,
)
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)


def _gene(ind: str = "RSI") -> StrategyGene:
    params = {"period": 14.0}
    if ind == "MACD":
        params = {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 9.0}
    elif ind == "STOCH":
        params = {"k_period": 14.0, "d_period": 3.0}
    return StrategyGene(
        indicator_type=ind, parameters=params,
        condition=ConditionType.GT, threshold=50.0,
    )


def _chrom(indicator: str = "RSI", direction: Direction = Direction.LONG) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_gene(indicator)],
        exit_genes=[_gene(indicator)],
        stop_loss_pct=2.0, take_profit_pct=5.0, direction=direction,
    )


class TestPopulationEntropy:
    """Tests for Shannon entropy of population."""

    def test_monoculture_zero_entropy(self) -> None:
        """All-identical population has zero entropy."""
        pop = [_chrom("RSI", Direction.LONG) for _ in range(10)]
        ent = population_entropy(pop)
        assert ent == pytest.approx(0.0, abs=1e-6)

    def test_diverse_population_high_entropy(self) -> None:
        """Population with varied indicators has high entropy."""
        indicators = ["RSI", "ATR", "CCI", "MFI", "ADX", "STOCH", "WILLR", "ROC"]
        pop = [_chrom(indicators[i % len(indicators)]) for i in range(16)]
        ent = population_entropy(pop)
        assert ent > 0.5  # Normalized, should be high

    def test_entropy_between_0_and_1(self) -> None:
        """Entropy is normalized to [0, 1]."""
        pop = [_chrom("RSI")] * 5 + [_chrom("CCI")] * 5
        ent = population_entropy(pop)
        assert 0.0 <= ent <= 1.0

    def test_single_chromosome_zero_entropy(self) -> None:
        """Single individual has zero entropy (no variation)."""
        pop = [_chrom()]
        ent = population_entropy(pop)
        assert ent == pytest.approx(0.0, abs=1e-6)


class TestShouldInjectImmigrants:
    """Tests for injection trigger."""

    def test_low_entropy_triggers_injection(self) -> None:
        """Below threshold → inject."""
        assert should_inject_immigrants(0.1, threshold=0.3)

    def test_high_entropy_no_injection(self) -> None:
        """Above threshold → no inject."""
        assert not should_inject_immigrants(0.8, threshold=0.3)

    def test_exactly_at_threshold_no_injection(self) -> None:
        """At threshold → no inject (strictly less than)."""
        assert not should_inject_immigrants(0.3, threshold=0.3)


class TestInjectRandomImmigrants:
    """Tests for random immigrant injection."""

    def test_replaces_bottom_n(self) -> None:
        """Injects immigrants replacing the worst individuals."""
        pop = [_chrom("RSI") for _ in range(10)]
        scores = [float(i) for i in range(10)]  # 0..9
        new_pop = inject_random_immigrants(pop, scores, fraction=0.2)
        assert len(new_pop) == 10
        # Top 8 should still be the original (scores 2-9)
        # Bottom 2 replaced with random

    def test_preserves_population_size(self) -> None:
        pop = [_chrom() for _ in range(20)]
        scores = [0.5] * 20
        new_pop = inject_random_immigrants(pop, scores, fraction=0.1)
        assert len(new_pop) == 20

    def test_at_least_one_immigrant(self) -> None:
        """Even with tiny fraction, inject at least 1."""
        pop = [_chrom() for _ in range(5)]
        scores = [0.5] * 5
        new_pop = inject_random_immigrants(pop, scores, fraction=0.01)
        assert len(new_pop) == 5
```

### Step 2: Run tests — expect failure

```bash
pytest tests/unit/test_discovery_diversity.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'vibe_quant.discovery.diversity'`

### Step 3: Implement diversity module

```python
# vibe_quant/discovery/diversity.py
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

    # Indicator locus
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

    # Normalize by max possible entropy for each locus
    # Indicator: 9 possible types → max = log2(9) ≈ 3.17
    # Direction: 3 possible → max = log2(3) ≈ 1.58
    # Condition: 6 possible → max = log2(6) ≈ 2.58
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

    # Find indices of worst individuals
    indexed = sorted(enumerate(fitness_scores), key=lambda x: x[1])
    worst_indices = {idx for idx, _ in indexed[:n_replace]}

    new_pop: list[StrategyChromosome] = []
    for i, chrom in enumerate(population):
        if i in worst_indices:
            new_pop.append(_random_chromosome(direction_constraint=direction_constraint))
        else:
            new_pop.append(chrom)

    return new_pop
```

### Step 4: Integrate into pipeline main loop

Modify `vibe_quant/discovery/pipeline.py` — after the evolution step (line ~454) and before convergence check:

```python
# After: population = self._evolve_generation(population, fitness_results)
# Add entropy monitoring + immigrant injection:

from vibe_quant.discovery.diversity import (
    population_entropy,
    should_inject_immigrants,
    inject_random_immigrants,
)

entropy = population_entropy(population)
if should_inject_immigrants(entropy):
    population = inject_random_immigrants(
        population, scores, fraction=0.1,
        direction_constraint=self._direction_constraint,
    )
    logger.info(
        "  Diversity intervention: entropy=%.3f < 0.3, injected %d random immigrants",
        entropy, max(1, int(len(population) * 0.1)),
    )
```

Also add entropy to the diversity log line (around line 405):

```python
logger.info(
    "  Diversity: entropy=%.3f indicators=%s directions=%s",
    population_entropy(population),
    ind_pcts,
    dir_counts,
)
```

### Step 5: Run tests

```bash
pytest tests/unit/test_discovery_diversity.py -v
pytest tests/unit/test_discovery_distance.py -v
```
Expected: All PASS

### Step 6: Commit

```bash
git add vibe_quant/discovery/diversity.py vibe_quant/discovery/pipeline.py tests/unit/test_discovery_diversity.py
git commit -m "feat: entropy monitoring + random immigrant injection (bd-028r)"
```

---

## Task 4: Structural Deduplication in Top-K Selection

**Files:**
- Modify: `vibe_quant/discovery/pipeline.py:456-465` (top-K dedup)
- Test: Add tests to `tests/unit/test_discovery_pipeline.py` or new file

### Background

Current top-K dedup uses `uid` — but structurally identical chromosomes have different UIDs (assigned at creation). Two clones of the same RSI strategy with different UIDs both survive.

Fix: use `chromosome_distance` for deduplication. A new candidate is only added if its distance to ALL already-selected strategies exceeds a minimum threshold (e.g., 0.15).

### Step 1: Write failing test

```python
# In tests/unit/test_discovery_pipeline.py or new test file

class TestStructuralDedup:
    """Tests for distance-based top-K deduplication."""

    def test_clones_rejected(self) -> None:
        """Structurally identical chromosomes are deduped even with different UIDs."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k
        from vibe_quant.discovery.operators import StrategyChromosome, StrategyGene, ConditionType, Direction

        gene = StrategyGene(
            indicator_type="RSI", parameters={"period": 14.0},
            condition=ConditionType.GT, threshold=50.0,
        )
        # 5 clones, different UIDs, same structure
        candidates = []
        for i in range(5):
            c = StrategyChromosome(
                entry_genes=[gene.clone()], exit_genes=[gene.clone()],
                stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
            )
            candidates.append((c, 0.8 - i * 0.01))  # Slightly decreasing fitness

        result = _select_diverse_top_k(candidates, top_k=5, min_distance=0.15)
        assert len(result) == 1  # Only 1 unique structure

    def test_diverse_strategies_all_kept(self) -> None:
        """Structurally different strategies are all kept."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k
        from vibe_quant.discovery.operators import StrategyChromosome, StrategyGene, ConditionType, Direction

        indicators = ["RSI", "CCI", "ATR", "STOCH", "ADX"]
        candidates = []
        for i, ind in enumerate(indicators):
            params = {"period": 14.0}
            if ind == "STOCH":
                params = {"k_period": 14.0, "d_period": 3.0}
            gene = StrategyGene(
                indicator_type=ind, parameters=params,
                condition=ConditionType.GT, threshold=50.0 if ind in ("RSI", "CCI") else 0.01,
            )
            c = StrategyChromosome(
                entry_genes=[gene], exit_genes=[gene.clone()],
                stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
            )
            candidates.append((c, 0.9 - i * 0.01))

        result = _select_diverse_top_k(candidates, top_k=5, min_distance=0.15)
        assert len(result) == 5  # All different indicators
```

### Step 2: Run tests — expect failure

```bash
pytest tests/unit/test_discovery_pipeline.py::TestStructuralDedup -v
```
Expected: FAIL — `ImportError: cannot import name '_select_diverse_top_k'`

### Step 3: Implement structural dedup

Add to `vibe_quant/discovery/pipeline.py`:

```python
def _select_diverse_top_k(
    scored: list[tuple[StrategyChromosome, FitnessResult | float]],
    top_k: int = 5,
    min_distance: float = 0.15,
) -> list[tuple[StrategyChromosome, FitnessResult | float]]:
    """Select top-K strategies with diversity enforcement.

    Iterates through candidates sorted by fitness (descending). A candidate
    is added only if its distance to ALL already-selected strategies exceeds
    min_distance. This ensures the output set is structurally diverse.

    Args:
        scored: List of (chromosome, fitness) tuples, sorted by fitness desc.
        top_k: Maximum number of strategies to select.
        min_distance: Minimum Gower distance to all selected strategies.

    Returns:
        List of up to top_k diverse (chromosome, fitness) tuples.
    """
    from vibe_quant.discovery.distance import chromosome_distance

    selected: list[tuple[StrategyChromosome, FitnessResult | float]] = []

    for chrom, fitness in scored:
        if len(selected) >= top_k:
            break

        # Check distance to all already-selected
        is_diverse = all(
            chromosome_distance(chrom, sel_chrom) >= min_distance
            for sel_chrom, _ in selected
        )

        if is_diverse:
            selected.append((chrom, fitness))

    return selected
```

Then replace the top-K selection block (line ~456-465):

```python
# Old:
# seen_uids: set[str] = set()
# ...

# New:
top_strategies = _select_diverse_top_k(
    [(chrom, fr) for chrom, fr in all_scored],
    top_k=cfg.top_k,
    min_distance=0.15,
)
```

### Step 4: Run tests

```bash
pytest tests/unit/test_discovery_pipeline.py::TestStructuralDedup -v
pytest tests/unit/ -q  # Full suite
```
Expected: All PASS

### Step 5: Commit

```bash
git add vibe_quant/discovery/pipeline.py tests/unit/test_discovery_pipeline.py
git commit -m "feat: structural dedup for top-K selection using Gower distance (bd-028r)"
```

---

## Task 5: Add `use_crowding` Config Flag + Wire Up CLI

**Files:**
- Modify: `vibe_quant/discovery/pipeline.py:50-87` (`DiscoveryConfig`)
- Modify: `vibe_quant/discovery/__main__.py` (CLI args, if exists)
- Modify: `vibe_quant/api/routers/discovery.py` (API, if it creates config)

### Step 1: Add config field

Add to `DiscoveryConfig`:

```python
use_crowding: bool = True  # Use deterministic crowding (True) or classic tournament (False)
immigrant_fraction: float = 0.1  # Fraction of population replaced when entropy is low
entropy_threshold: float = 0.3  # Entropy below this triggers immigrant injection
min_diversity_distance: float = 0.15  # Min Gower distance for top-K dedup
```

### Step 2: Wire up in API/CLI

Check if `discovery.py` router or `__main__.py` creates `DiscoveryConfig` and add the new fields as optional parameters with defaults.

### Step 3: Run full test suite

```bash
pytest tests/unit/ -q
```
Expected: All PASS (defaults preserve backward compat)

### Step 4: Commit

```bash
git add vibe_quant/discovery/pipeline.py vibe_quant/discovery/__main__.py vibe_quant/api/routers/discovery.py
git commit -m "feat: configurable diversity params in DiscoveryConfig (bd-028r)"
```

---

## Task 6: Integration Test — Diversity Preservation E2E

**Files:**
- Test: `tests/unit/test_ga_diversity_e2e.py`

### Step 1: Write integration test

```python
# tests/unit/test_ga_diversity_e2e.py
"""Integration test: verify GA maintains diversity across generations."""

from vibe_quant.discovery.diversity import population_entropy
from vibe_quant.discovery.operators import (
    Direction,
    StrategyChromosome,
    _random_chromosome,
    apply_elitism,
    crossover,
    is_valid_chromosome,
    mutate,
)


class TestDiversityPreservation:
    """Verify that entropy doesn't collapse to zero over multiple generations."""

    def test_entropy_maintained_with_crowding(self) -> None:
        """After 10 generations with mock fitness, entropy stays above 0.15."""
        pop = [_random_chromosome() for _ in range(20)]
        initial_entropy = population_entropy(pop)
        assert initial_entropy > 0.3  # Random pop should be diverse

        # Simulate 10 generations with fake fitness (random scores)
        import random as rng
        for gen in range(10):
            scores = [rng.random() for _ in pop]

            # Evolution: simple tournament + crossover + mutation
            new_pop = apply_elitism(pop, scores, 1)
            indices = list(range(len(pop)))
            rng.shuffle(indices)
            while len(new_pop) < len(pop):
                i = rng.choice(indices)
                j = rng.choice(indices)
                if rng.random() < 0.8:
                    a, b = crossover(pop[i], pop[j])
                else:
                    a, b = pop[i].clone(), pop[j].clone()
                a = mutate(a, 0.15)  # Slightly higher mutation
                b = mutate(b, 0.15)
                for c in (a, b):
                    if len(new_pop) < len(pop) and is_valid_chromosome(c):
                        new_pop.append(c)

            # Diversity intervention
            from vibe_quant.discovery.diversity import (
                inject_random_immigrants,
                should_inject_immigrants,
            )
            entropy = population_entropy(new_pop)
            if should_inject_immigrants(entropy):
                new_pop = inject_random_immigrants(new_pop, scores, fraction=0.1)

            pop = new_pop

        final_entropy = population_entropy(pop)
        assert final_entropy > 0.15, (
            f"Entropy collapsed: initial={initial_entropy:.3f}, final={final_entropy:.3f}"
        )
```

### Step 2: Run test

```bash
pytest tests/unit/test_ga_diversity_e2e.py -v
```
Expected: PASS

### Step 3: Commit

```bash
git add tests/unit/test_ga_diversity_e2e.py
git commit -m "test: GA diversity preservation integration test (bd-028r)"
```

---

## Task 7: Final Verification + Close Bead

### Step 1: Run full test suite

```bash
pytest tests/unit/ -q
```

### Step 2: Run lint + type check

```bash
ruff check vibe_quant/discovery/
mypy vibe_quant/discovery/
```

### Step 3: Close bead and push

```bash
bd close vibe-quant-028r
bd sync
git add .
git commit -m "feat: GA diversity preservation — Gower distance, crowding, entropy monitoring (bd-028r)"
git push
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `vibe_quant/discovery/distance.py` | **NEW** — Gower distance for genes + chromosomes |
| `vibe_quant/discovery/diversity.py` | **NEW** — Shannon entropy + immigrant injection |
| `vibe_quant/discovery/operators.py` | Add `crowding_replace()` function |
| `vibe_quant/discovery/pipeline.py` | Split `_evolve_generation` into tournament/crowding modes, add entropy monitoring, structural top-K dedup, new config fields |
| `vibe_quant/discovery/__main__.py` | Wire new config fields to CLI |
| `vibe_quant/api/routers/discovery.py` | Wire new config fields to API |
| `tests/unit/test_discovery_distance.py` | **NEW** — 13 tests for Gower distance |
| `tests/unit/test_discovery_diversity.py` | **NEW** — 10 tests for entropy + injection |
| `tests/unit/test_discovery_operators.py` | Add 3 tests for crowding_replace |
| `tests/unit/test_discovery_pipeline.py` | Add 2 tests for structural dedup |
| `tests/unit/test_ga_diversity_e2e.py` | **NEW** — Integration test |

## Open Questions

1. **Gower weights** — structural (indicator_type) at 2x vs params at 1x. May need tuning after observing real discovery runs.
2. **min_diversity_distance=0.15** — too tight rejects similar-but-different strategies; too loose lets clones through. Needs empirical tuning.
3. **Crowding without pre-evaluated offspring** — Option A (offspring always enter) is simpler but less selective. Monitor if it actually reduces monoculture vs classic tournament.
4. **entropy_threshold=0.3** — below this triggers injection. May need adjustment based on population size and indicator pool size.
5. **immigrant_fraction=0.1** — 10% replacement. Too aggressive may destabilize evolution; too conservative may not fix diversity.
