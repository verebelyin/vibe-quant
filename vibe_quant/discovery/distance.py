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

# Direction distance: LONG<->SHORT = 1.0, LONG<->BOTH or SHORT<->BOTH = 0.5
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
        # Different indicators -> max distance for threshold and params
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
