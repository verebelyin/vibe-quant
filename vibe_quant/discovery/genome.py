"""Strategy genome representation for genetic discovery.

Encodes trading strategies as chromosomes (lists of indicator/condition genes)
for evolutionary search. Supports random generation, validation, and conversion
to the StrategyDSL format.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from vibe_quant.discovery.operators import (
    THRESHOLD_RANGES,
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
    _ensure_pool,
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

VALID_CONDITIONS: frozenset[str] = frozenset(
    {
        *_LEGACY_CONDITION_ALIASES,
        *(cond.value for cond in ConditionType),
    }
)

VALID_DIRECTIONS: frozenset[str] = frozenset(
    {
        *(direction.value for direction in Direction),
    }
)

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


def build_indicator_pool() -> dict[str, IndicatorDef]:
    """Assemble the GA indicator pool from the live ``indicator_registry``.

    Rules:

    - Any spec with a non-None ``threshold_range`` AND a non-empty
      ``param_ranges`` dict is auto-enrolled.
    - Specs that leave ``threshold_range=None`` (EMA/SMA/WMA/DEMA/TEMA and
      other price-relative indicators) are excluded — threshold=0 against
      an absolute price is never a meaningful condition.
    - Param bounds are copied verbatim from the spec; the GA's
      ``_random_gene`` handles the int-vs-float heuristic.

    Plugins dropped into ``vibe_quant/dsl/plugins/`` slot in here by
    simply declaring the two fields on their spec — no edits to this
    file. ``build_indicator_pool`` is called at import time so callers
    that read ``INDICATOR_POOL`` as a plain dict keep working. If a
    plugin is registered AFTER module import (unit tests), call
    ``build_indicator_pool()`` again to pick it up.

    Returns:
        Dict keyed by indicator name → ``IndicatorDef``.
    """
    # Local import deferred to avoid a circular import at package init
    # time: discovery/operators imports discovery/genome.
    from vibe_quant.dsl.indicators import indicator_registry

    pool: dict[str, IndicatorDef] = {}
    for spec in indicator_registry.all_specs():
        if spec.threshold_range is None or not spec.param_ranges:
            continue
        pool[spec.name] = IndicatorDef(
            name=spec.name,
            param_ranges=dict(spec.param_ranges),
            default_threshold_range=spec.threshold_range,
            dsl_type=spec.name,
        )
    return pool


# Module-level pool materialized from the registry at import time. Kept
# as a plain dict (not a @property / callable) to preserve the legacy
# consumer API where code reads ``INDICATOR_POOL`` directly.
INDICATOR_POOL: dict[str, IndicatorDef] = build_indicator_pool()


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

    # Sub-values for multi-output indicators
    sub_value = None
    if ind_name == "MACD":
        sub_value = r.choice([None, "signal", "histogram"])
    elif ind_name == "BBANDS":
        sub_value = r.choice(["percent_b", "bandwidth"])
        if sub_value == "bandwidth":
            tlo, thi = 0.0, 0.2
            threshold = round(r.uniform(tlo, thi), 4)
    elif ind_name == "DONCHIAN":
        sub_value = "position"

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
    _ensure_pool()  # ensure THRESHOLD_RANGES is populated
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
                errors.append(f"{prefix}: unknown indicator '{gene.indicator_type}'")
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
                    errors.append(f"{prefix}: param '{pname}'={val} out of range [{lo}, {hi}]")

            # Check for unexpected params
            for pname in gene.parameters:
                if pname not in ind_def.param_ranges:
                    errors.append(f"{prefix}: unexpected param '{pname}'")

            # Validate threshold is within indicator's expected range
            if gene.indicator_type in THRESHOLD_RANGES:
                tlo, thi = THRESHOLD_RANGES[gene.indicator_type]
                if not (tlo <= gene.threshold <= thi):
                    errors.append(
                        f"{prefix}: threshold {gene.threshold} outside "
                        f"[{tlo}, {thi}] for {gene.indicator_type}"
                    )

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
        errors.append(f"stop_loss_pct={chrom.stop_loss_pct} out of range [0.5, 10.0]")
    if not (0.5 <= chrom.take_profit_pct <= 20.0):
        errors.append(f"take_profit_pct={chrom.take_profit_pct} out of range [0.5, 20.0]")

    # Per-direction SL/TP ranges
    for attr, label, valid_range in [
        ("stop_loss_long_pct", "stop_loss_long_pct", (0.5, 10.0)),
        ("stop_loss_short_pct", "stop_loss_short_pct", (0.5, 10.0)),
        ("take_profit_long_pct", "take_profit_long_pct", (0.5, 20.0)),
        ("take_profit_short_pct", "take_profit_short_pct", (0.5, 20.0)),
    ]:
        val = getattr(chrom, attr, None)
        if val is not None and not (valid_range[0] <= val <= valid_range[1]):
            errors.append(f"{label}={val} out of range [{valid_range[0]}, {valid_range[1]}]")

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
    elif gene.indicator_type == "DONCHIAN":
        cfg["period"] = int(gene.parameters.get("period", 20))
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

    # Per-direction SL/TP overrides (only for BOTH direction)
    if direction == Direction.BOTH:
        if chrom.stop_loss_long_pct is not None:
            dsl["stop_loss_long"] = {"type": "fixed_pct", "percent": round(chrom.stop_loss_long_pct, 2)}
        if chrom.stop_loss_short_pct is not None:
            dsl["stop_loss_short"] = {"type": "fixed_pct", "percent": round(chrom.stop_loss_short_pct, 2)}
        if chrom.take_profit_long_pct is not None:
            dsl["take_profit_long"] = {"type": "fixed_pct", "percent": round(chrom.take_profit_long_pct, 2)}
        if chrom.take_profit_short_pct is not None:
            dsl["take_profit_short"] = {"type": "fixed_pct", "percent": round(chrom.take_profit_short_pct, 2)}

    # Time filters (optional)
    if chrom.time_filters:
        dsl["time_filters"] = chrom.time_filters

    return dsl


def chromosome_to_serializable(chrom: StrategyChromosome) -> dict[str, object]:
    """Serialize chromosome to a JSON-safe dict for storage and reconstruction.

    Unlike chromosome_to_dsl (which produces DSL format for backtesting),
    this preserves the exact gene structure for warm-start seeding.
    """

    def _gene_dict(g: StrategyGene) -> dict[str, object]:
        return {
            "indicator_type": g.indicator_type,
            "parameters": dict(g.parameters),
            "condition": g.condition.value,
            "threshold": g.threshold,
            "sub_value": g.sub_value,
        }

    d: dict[str, object] = {
        "entry_genes": [_gene_dict(g) for g in chrom.entry_genes],
        "exit_genes": [_gene_dict(g) for g in chrom.exit_genes],
        "stop_loss_pct": chrom.stop_loss_pct,
        "take_profit_pct": chrom.take_profit_pct,
        "direction": chrom.direction.value if hasattr(chrom.direction, "value") else str(chrom.direction),
    }
    if chrom.stop_loss_long_pct is not None:
        d["stop_loss_long_pct"] = chrom.stop_loss_long_pct
    if chrom.stop_loss_short_pct is not None:
        d["stop_loss_short_pct"] = chrom.stop_loss_short_pct
    if chrom.take_profit_long_pct is not None:
        d["take_profit_long_pct"] = chrom.take_profit_long_pct
    if chrom.take_profit_short_pct is not None:
        d["take_profit_short_pct"] = chrom.take_profit_short_pct
    if chrom.time_filters:
        d["time_filters"] = dict(chrom.time_filters)
    return d


def serializable_to_chromosome(d: dict[str, object]) -> StrategyChromosome:
    """Reconstruct a StrategyChromosome from a serialized dict.

    Inverse of chromosome_to_serializable.
    """

    def _parse_gene(g: dict[str, object]) -> StrategyGene:
        params_raw = g.get("parameters", {})
        params: dict[str, float] = {
            str(k): float(v) for k, v in params_raw.items()  # type: ignore[union-attr]
        }
        return StrategyGene(
            indicator_type=str(g["indicator_type"]),
            parameters=params,
            condition=ConditionType(str(g["condition"])),
            threshold=float(g["threshold"]),  # type: ignore[arg-type]
            sub_value=str(g["sub_value"]) if g.get("sub_value") else None,
        )

    entry_genes_raw = d.get("entry_genes", [])
    exit_genes_raw = d.get("exit_genes", [])
    entry_genes = [_parse_gene(g) for g in entry_genes_raw]  # type: ignore[union-attr]
    exit_genes = [_parse_gene(g) for g in exit_genes_raw]  # type: ignore[union-attr]

    direction_raw = d.get("direction", "long")
    direction = Direction(str(direction_raw))

    tf_raw = d.get("time_filters")
    time_filters: dict[str, object] = dict(tf_raw) if tf_raw and isinstance(tf_raw, dict) else {}

    return StrategyChromosome(
        entry_genes=entry_genes,
        exit_genes=exit_genes,
        stop_loss_pct=float(d.get("stop_loss_pct", 2.0)),  # type: ignore[arg-type]
        take_profit_pct=float(d.get("take_profit_pct", 3.0)),  # type: ignore[arg-type]
        direction=direction,
        stop_loss_long_pct=float(d["stop_loss_long_pct"]) if d.get("stop_loss_long_pct") is not None else None,  # type: ignore[arg-type]
        stop_loss_short_pct=float(d["stop_loss_short_pct"]) if d.get("stop_loss_short_pct") is not None else None,  # type: ignore[arg-type]
        take_profit_long_pct=float(d["take_profit_long_pct"]) if d.get("take_profit_long_pct") is not None else None,  # type: ignore[arg-type]
        take_profit_short_pct=float(d["take_profit_short_pct"]) if d.get("take_profit_short_pct") is not None else None,  # type: ignore[arg-type]
        time_filters=time_filters,
    )
