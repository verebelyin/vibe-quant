"""Strategy genome representation for genetic discovery.

Encodes trading strategies as chromosomes (lists of indicator/condition genes)
for evolutionary search. Supports random generation, validation, and conversion
to the StrategyDSL format.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)

# ---------------------------------------------------------------------------
# Condition and direction constants
# ---------------------------------------------------------------------------

_LEGACY_CONDITION_ALIASES: dict[str, ConditionType] = {
    "crosses_above": ConditionType.CROSSES_ABOVE,
    "crosses_below": ConditionType.CROSSES_BELOW,
    "greater_than": ConditionType.GT,
    "less_than": ConditionType.LT,
}

VALID_CONDITIONS: frozenset[str] = frozenset({
    *_LEGACY_CONDITION_ALIASES,
    *(cond.value for cond in ConditionType),
})

VALID_DIRECTIONS: frozenset[str] = frozenset({
    *(direction.value for direction in Direction),
})

# Maps genome condition names to DSL operator syntax
_CONDITION_TO_DSL_OP: dict[str, str] = {
    "crosses_above": "crosses_above",
    "crosses_below": "crosses_below",
    "greater_than": ">",
    "less_than": "<",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
}

# ---------------------------------------------------------------------------
# Indicator pool: available indicators with valid parameter ranges
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndicatorDef:
    """Definition of an indicator available for genome construction.

    Attributes:
        name: Canonical indicator name (uppercase).
        param_ranges: Mapping of param name -> (min, max) inclusive.
        default_threshold_range: (min, max) for a sensible threshold when
            the indicator is used in a condition against a numeric value.
        dsl_type: Indicator type string for the DSL schema.
    """

    name: str
    param_ranges: dict[str, tuple[float, float]]
    default_threshold_range: tuple[float, float]
    dsl_type: str


INDICATOR_POOL: dict[str, IndicatorDef] = {
    "RSI": IndicatorDef(
        name="RSI",
        param_ranges={"period": (5, 50)},
        default_threshold_range=(20.0, 80.0),
        dsl_type="RSI",
    ),
    # EMA excluded: price-relative indicator, threshold=0 produces no trades
    # Requires indicator-vs-indicator comparison (not yet supported)
    "MACD": IndicatorDef(
        name="MACD",
        param_ranges={
            "fast_period": (8, 21),
            "slow_period": (21, 50),
            "signal_period": (5, 13),
        },
        default_threshold_range=(-0.01, 0.01),
        dsl_type="MACD",
    ),
    # BBANDS excluded: price-relative indicator, threshold=0 produces no trades
    "ATR": IndicatorDef(
        name="ATR",
        param_ranges={"period": (5, 30)},
        default_threshold_range=(0.001, 0.05),
        dsl_type="ATR",
    ),
    "STOCH": IndicatorDef(
        name="STOCH",
        param_ranges={"k_period": (5, 21), "d_period": (3, 9)},
        default_threshold_range=(20.0, 80.0),
        dsl_type="STOCH",
    ),
}

def _normalize_condition(condition: ConditionType | str) -> ConditionType | None:
    """Normalize legacy string conditions to canonical enum values."""
    if isinstance(condition, ConditionType):
        return condition
    if condition in _LEGACY_CONDITION_ALIASES:
        return _LEGACY_CONDITION_ALIASES[condition]
    try:
        return ConditionType(condition)
    except ValueError:
        return None


def _normalize_direction(direction: Direction | str) -> Direction | None:
    """Normalize direction string to Direction enum."""
    if isinstance(direction, Direction):
        return direction
    try:
        return Direction(direction)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Random generation helpers
# ---------------------------------------------------------------------------


def _random_gene(rng: random.Random | None = None) -> StrategyGene:
    """Generate a single random gene from the indicator pool."""
    r = rng or random
    ind_name = r.choice(list(INDICATOR_POOL))
    ind_def = INDICATOR_POOL[ind_name]

    # Sample parameters
    params: dict[str, int | float] = {}
    for pname, (lo, hi) in ind_def.param_ranges.items():
        # Heuristic: if both bounds are ints and >= 1, treat as int
        if lo == int(lo) and hi == int(hi) and lo >= 1:
            params[pname] = r.randint(int(lo), int(hi))
        else:
            params[pname] = round(r.uniform(lo, hi), 4)

    # Enforce MACD fast < slow
    if ind_name == "MACD":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        if fast >= slow:
            params["fast_period"] = min(fast, slow)
            params["slow_period"] = max(fast, slow) + 1

    condition = r.choice(list(ConditionType))

    # Threshold
    tlo, thi = ind_def.default_threshold_range
    threshold = tlo if tlo == thi else round(r.uniform(tlo, thi), 4)

    # MACD can use signal or histogram sub-values
    sub_value = None
    if ind_name == "MACD":
        sub_value = r.choice([None, "signal", "histogram"])

    return StrategyGene(
        indicator_type=ind_name,
        parameters=params,
        condition=condition,
        threshold=threshold,
        sub_value=sub_value,
    )


def generate_random_chromosome(
    rng: random.Random | None = None,
) -> StrategyChromosome:
    """Generate a structurally valid random chromosome.

    Args:
        rng: Optional Random instance for reproducibility.

    Returns:
        A valid StrategyChromosome with random genes.
    """
    r = rng if rng is not None else random.Random()

    n_entry = r.randint(1, 5)
    n_exit = r.randint(1, 3)

    entry_genes = [_random_gene(r) for _ in range(n_entry)]
    exit_genes = [_random_gene(r) for _ in range(n_exit)]

    sl = round(r.uniform(0.5, 10.0), 4)  # Percentage (0.5% to 10%)
    tp = round(r.uniform(0.5, 20.0), 4)  # Percentage (0.5% to 20%)

    direction = r.choice(list(Direction))

    return StrategyChromosome(
        entry_genes=entry_genes,
        exit_genes=exit_genes,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_chromosome(chrom: StrategyChromosome) -> list[str]:
    """Validate a chromosome, returning a list of error strings (empty = valid).

    Checks:
    - At least 1 entry gene, at least 1 exit gene
    - Entry genes count 1-5, exit genes count 1-5
    - All indicators from INDICATOR_POOL
    - Parameters within valid ranges
    - Condition is valid
    - SL/TP within valid ranges
    - Direction is valid
    """
    errors: list[str] = []

    # Gene count checks
    if len(chrom.entry_genes) < 1:
        errors.append("Must have at least 1 entry gene")
    if len(chrom.entry_genes) > 5:
        errors.append(f"Max 5 entry genes, got {len(chrom.entry_genes)}")
    if len(chrom.exit_genes) < 1:
        errors.append("Must have at least 1 exit gene")
    if len(chrom.exit_genes) > 5:
        errors.append(f"Max 5 exit genes, got {len(chrom.exit_genes)}")

    # Validate each gene
    for label, genes in [("entry", chrom.entry_genes), ("exit", chrom.exit_genes)]:
        for i, gene in enumerate(genes):
            prefix = f"{label}[{i}]"

            if gene.indicator_type not in INDICATOR_POOL:
                errors.append(
                    f"{prefix}: unknown indicator '{gene.indicator_type}'"
                )
                continue

            ind_def = INDICATOR_POOL[gene.indicator_type]

            # Validate condition
            normalized_condition = _normalize_condition(gene.condition)
            if normalized_condition is None:
                errors.append(f"{prefix}: invalid condition '{gene.condition}'")

            # Validate parameters are within range
            for pname, (lo, hi) in ind_def.param_ranges.items():
                val = gene.parameters.get(pname)
                if val is None:
                    errors.append(f"{prefix}: missing param '{pname}'")
                elif not (lo <= val <= hi):
                    errors.append(
                        f"{prefix}: param '{pname}'={val} out of range [{lo}, {hi}]"
                    )

            # Check for unexpected params
            for pname in gene.parameters:
                if pname not in ind_def.param_ranges:
                    errors.append(f"{prefix}: unexpected param '{pname}'")

            # MACD: fast_period must be < slow_period
            if gene.indicator_type == "MACD":
                fast = gene.parameters.get("fast_period", 12)
                slow = gene.parameters.get("slow_period", 26)
                if fast >= slow:
                    errors.append(
                        f"{prefix}: MACD fast_period ({fast}) must be < slow_period ({slow})"
                    )

    # SL/TP ranges (percentage values, e.g. 2.0 = 2%)
    if not (0.5 <= chrom.stop_loss_pct <= 10.0):
        errors.append(
            f"stop_loss_pct={chrom.stop_loss_pct} out of range [0.5, 10.0]"
        )
    if not (0.5 <= chrom.take_profit_pct <= 20.0):
        errors.append(
            f"take_profit_pct={chrom.take_profit_pct} out of range [0.5, 20.0]"
        )

    # Direction
    if _normalize_direction(chrom.direction) is None:
        errors.append(f"invalid direction '{chrom.direction}'")

    return errors


# ---------------------------------------------------------------------------
# DSL conversion
# ---------------------------------------------------------------------------


def _gene_indicator_name(gene: StrategyGene, idx: int, prefix: str) -> str:
    """Deterministic indicator name for a gene."""
    return f"{gene.indicator_type.lower()}_{prefix}_{idx}"


def _gene_to_indicator_config(gene: StrategyGene) -> dict[str, object]:
    """Build DSL IndicatorConfig dict from a gene."""
    cfg: dict[str, object] = {"type": gene.indicator_type}

    if gene.indicator_type == "MACD":
        cfg["fast_period"] = int(gene.parameters.get("fast_period", 12))
        cfg["slow_period"] = int(gene.parameters.get("slow_period", 26))
        cfg["signal_period"] = int(gene.parameters.get("signal_period", 9))
    elif gene.indicator_type == "STOCH":
        cfg["period"] = int(gene.parameters.get("k_period", 14))
        cfg["d_period"] = int(gene.parameters.get("d_period", 3))
    elif gene.indicator_type == "BBANDS":
        cfg["period"] = int(gene.parameters.get("period", 20))
        cfg["std_dev"] = float(gene.parameters.get("std_dev", 2.0))
    elif gene.indicator_type == "ATR":
        cfg["period"] = int(gene.parameters.get("period", 14))
    else:
        # RSI, EMA, etc. -- single "period" param
        cfg["period"] = int(gene.parameters.get("period", 14))

    return cfg


def _gene_to_condition_str(gene: StrategyGene, indicator_name: str) -> str:
    """Build DSL condition string from a gene.

    Uses the DSL operator syntax: crosses_above, crosses_below, >, <.
    """
    normalized_condition = _normalize_condition(gene.condition)
    if normalized_condition is None:
        msg = f"Unsupported condition: {gene.condition}"
        raise ValueError(msg)
    op = _CONDITION_TO_DSL_OP[normalized_condition.value]
    threshold = gene.threshold
    # Format threshold: drop trailing zeros but keep at least one decimal
    thr_str = str(int(threshold)) if threshold == int(threshold) else f"{threshold:g}"
    # MACD sub-value access (signal, histogram)
    ref = f"{indicator_name}.{gene.sub_value}" if gene.sub_value else indicator_name
    return f"{ref} {op} {thr_str}"


def chromosome_to_dsl(chrom: StrategyChromosome) -> dict[str, object]:
    """Convert a chromosome to a StrategyDSL-compatible YAML dict.

    The output dict can be passed to ``StrategyDSL(**d)`` for validation
    or serialized to YAML.

    Args:
        chrom: Validated chromosome.

    Returns:
        Dict compatible with vibe_quant.dsl.schema.StrategyDSL.
    """
    indicators: dict[str, object] = {}
    entry_long: list[str] = []
    entry_short: list[str] = []
    exit_long: list[str] = []
    exit_short: list[str] = []

    direction = _normalize_direction(chrom.direction)
    if direction is None:
        msg = f"Unsupported direction: {chrom.direction}"
        raise ValueError(msg)

    # Build indicators + conditions from entry genes
    for i, gene in enumerate(chrom.entry_genes):
        ind_name = _gene_indicator_name(gene, i, "entry")
        indicators[ind_name] = _gene_to_indicator_config(gene)
        cond_str = _gene_to_condition_str(gene, ind_name)
        if direction in (Direction.LONG, Direction.BOTH):
            entry_long.append(cond_str)
        if direction in (Direction.SHORT, Direction.BOTH):
            entry_short.append(cond_str)

    # Build indicators + conditions from exit genes
    for i, gene in enumerate(chrom.exit_genes):
        ind_name = _gene_indicator_name(gene, i, "exit")
        indicators[ind_name] = _gene_to_indicator_config(gene)
        cond_str = _gene_to_condition_str(gene, ind_name)
        if direction in (Direction.LONG, Direction.BOTH):
            exit_long.append(cond_str)
        if direction in (Direction.SHORT, Direction.BOTH):
            exit_short.append(cond_str)

    # Entry conditions must have at least one side populated
    entry_conditions: dict[str, list[str]] = {}
    if entry_long:
        entry_conditions["long"] = entry_long
    if entry_short:
        entry_conditions["short"] = entry_short

    exit_conditions: dict[str, list[str]] = {}
    if exit_long:
        exit_conditions["long"] = exit_long
    if exit_short:
        exit_conditions["short"] = exit_short

    # Stop loss / take profit as fixed_pct (already in percentage)
    sl_pct = round(chrom.stop_loss_pct, 2)
    tp_pct = round(chrom.take_profit_pct, 2)

    dsl: dict[str, object] = {
        "name": f"genome_{chrom.uid}",
        "timeframe": "5m",
        "indicators": indicators,
        "entry_conditions": entry_conditions,
        "exit_conditions": exit_conditions,
        "stop_loss": {"type": "fixed_pct", "percent": sl_pct},
        "take_profit": {"type": "fixed_pct", "percent": tp_pct},
    }

    # Time filters (optional)
    if chrom.time_filters:
        dsl["time_filters"] = chrom.time_filters

    return dsl
