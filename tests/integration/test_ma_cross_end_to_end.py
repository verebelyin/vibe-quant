"""End-to-end proof that ma_fast-vs-ma_slow cross genes survive every
stage of the GA→DSL→compile pipeline.

Covers bd-fuaj (Phase 2 MA crosses): a ``PriceVsMAConditionGene`` with
``parameters_slow`` must convert to a DSL dict that validates, compiles
to Python source, imports as a module, and (after mutation/round-trip)
continues to compile. Each test is independent.
"""

from __future__ import annotations

import random

import pytest

from vibe_quant.discovery.genome import (
    MA_POOL,
    StrategyChromosome,
    chromosome_to_dsl,
    chromosome_to_serializable,
    serializable_to_chromosome,
)
from vibe_quant.discovery.operators import (
    _MA_CONDITION_TYPES,
    ConditionType,
    Direction,
    PriceVsMAConditionGene,
    StrategyGene,
    _repair_chromosome,
    crossover,
    is_valid_chromosome,
    mutate,
)
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict


def _make_cross_chrom(
    ma_kind: str = "KAMA",
    period_fast: float = 10.0,
    period_slow: float = 30.0,
) -> StrategyChromosome:
    """Build a minimal chromosome whose single MA entry gene is a cross."""
    return StrategyChromosome(
        entry_genes=[
            StrategyGene(
                indicator_type="RSI",
                parameters={"period": 14},
                condition=ConditionType.LT,
                threshold=30.0,
            )
        ],
        exit_genes=[
            StrategyGene(
                indicator_type="RSI",
                parameters={"period": 14},
                condition=ConditionType.GT,
                threshold=70.0,
            )
        ],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
        ma_entry_genes=[
            PriceVsMAConditionGene(
                indicator_type=ma_kind,
                parameters={"period": period_fast},
                op=ConditionType.GT,
                parameters_slow={"period": period_slow},
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. DSL emission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ma_kind", sorted(MA_POOL.keys()))
def test_cross_chrom_emits_two_indicators_for_every_ma_kind(ma_kind: str) -> None:
    """Every MA in the pool must emit a well-formed two-leg DSL."""
    chrom = _make_cross_chrom(ma_kind, 8.0, 24.0)
    dsl = chromosome_to_dsl(chrom)

    indicators = dsl["indicators"]
    assert isinstance(indicators, dict)
    fast_keys = [k for k in indicators if k.endswith("_fast")]
    slow_keys = [k for k in indicators if k.endswith("_slow")]
    assert len(fast_keys) == 1, f"expected 1 fast leg, got {fast_keys}"
    assert len(slow_keys) == 1, f"expected 1 slow leg, got {slow_keys}"

    fast_cfg = indicators[fast_keys[0]]
    slow_cfg = indicators[slow_keys[0]]
    assert isinstance(fast_cfg, dict) and isinstance(slow_cfg, dict)
    assert fast_cfg["type"] == ma_kind
    assert slow_cfg["type"] == ma_kind
    assert fast_cfg["period"] == 8
    assert slow_cfg["period"] == 24

    entry_long = dsl["entry_conditions"]["long"]  # type: ignore[index]
    assert f"{fast_keys[0]} > {slow_keys[0]}" in entry_long


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ma_kind", sorted(MA_POOL.keys()))
def test_cross_dsl_validates_against_schema(ma_kind: str) -> None:
    chrom = _make_cross_chrom(ma_kind, 10.0, 30.0)
    dsl_dict = chromosome_to_dsl(chrom)
    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = f"e2e_cross_{ma_kind.lower()}"
    dsl = validate_strategy_dict(dsl_dict)
    assert dsl.name == f"e2e_cross_{ma_kind.lower()}"
    assert f"{ma_kind.lower()}_ma_entry_0_fast" in dsl.indicators
    assert f"{ma_kind.lower()}_ma_entry_0_slow" in dsl.indicators


# ---------------------------------------------------------------------------
# 3. Source compile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ma_kind", sorted(MA_POOL.keys()))
def test_cross_dsl_compiles_to_source_with_both_legs(ma_kind: str) -> None:
    chrom = _make_cross_chrom(ma_kind, 10.0, 30.0)
    dsl_dict = chromosome_to_dsl(chrom)
    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = f"e2e_cross_{ma_kind.lower()}"
    dsl = validate_strategy_dict(dsl_dict)

    source = StrategyCompiler().compile(dsl)
    fast_name = f"{ma_kind.lower()}_ma_entry_0_fast"
    slow_name = f"{ma_kind.lower()}_ma_entry_0_slow"
    # Both legs are declared as config fields and referenced in the
    # generated condition code.
    assert f"{fast_name}_period" in source
    assert f"{slow_name}_period" in source
    assert f"{fast_name} > {slow_name}" in source


# ---------------------------------------------------------------------------
# 4. Module import
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ma_kind", sorted(MA_POOL.keys()))
def test_cross_dsl_compiles_to_importable_module(ma_kind: str) -> None:
    chrom = _make_cross_chrom(ma_kind, 10.0, 30.0)
    dsl_dict = chromosome_to_dsl(chrom)
    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = f"e2e_cross_import_{ma_kind.lower()}"
    dsl = validate_strategy_dict(dsl_dict)

    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    strategy_cls = getattr(module, f"{camel}Strategy", None)
    config_cls = getattr(module, f"{camel}Config", None)
    assert strategy_cls is not None
    assert config_cls is not None


# ---------------------------------------------------------------------------
# 5. Round-trip serialization
# ---------------------------------------------------------------------------


def test_cross_chrom_roundtrip_serialization_still_compiles() -> None:
    """Serialize → deserialize → recompile must produce the same DSL."""
    original = _make_cross_chrom("VIDYA", 8.0, 24.0)
    d = chromosome_to_serializable(original)
    restored = serializable_to_chromosome(d)

    dsl_a = chromosome_to_dsl(original)
    dsl_b = chromosome_to_dsl(restored)
    # Name is a content-hash that varies; compare the structural payload.
    dsl_a.pop("name", None)
    dsl_b.pop("name", None)
    assert dsl_a == dsl_b

    dsl_a["timeframe"] = "5m"
    dsl_a["name"] = "e2e_rt"
    StrategyCompiler().compile_to_module(validate_strategy_dict(dsl_a))


# ---------------------------------------------------------------------------
# 6. Repair keeps it compilable
# ---------------------------------------------------------------------------


def test_repair_keeps_degenerate_cross_compilable() -> None:
    """A chromosome with swapped periods must repair cleanly and still compile."""
    chrom = _make_cross_chrom("KAMA", 30.0, 10.0)
    assert not is_valid_chromosome(chrom)
    repaired = _repair_chromosome(chrom)
    assert is_valid_chromosome(repaired)

    dsl_dict = chromosome_to_dsl(repaired)
    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = "e2e_repaired"
    StrategyCompiler().compile_to_module(validate_strategy_dict(dsl_dict))


# ---------------------------------------------------------------------------
# 7. Mutation fuzz — every mutant must still compile
# ---------------------------------------------------------------------------


def test_mutation_fuzz_every_mutant_compiles() -> None:
    """100 random mutations of a cross seed must all compile end-to-end."""
    random.seed(123)
    seed = _make_cross_chrom("KAMA", 10.0, 30.0)
    compiler = StrategyCompiler()

    for i in range(100):
        mutant = mutate(seed, mutation_rate=1.0)
        if not is_valid_chromosome(mutant):
            mutant = _repair_chromosome(mutant)
        dsl_dict = chromosome_to_dsl(mutant)
        dsl_dict["timeframe"] = "5m"
        dsl_dict["name"] = f"e2e_fuzz_{i}"
        dsl = validate_strategy_dict(dsl_dict)
        source = compiler.compile(dsl)
        assert "class " in source


# ---------------------------------------------------------------------------
# Edge case 1 — cross genes in BOTH entry and exit slots of one chromosome
# ---------------------------------------------------------------------------


def test_entry_and_exit_cross_coexist() -> None:
    """An entry-side cross and an exit-side cross must not collide on names."""
    chrom = _make_cross_chrom("KAMA", 10.0, 30.0)
    chrom.ma_exit_genes = [
        PriceVsMAConditionGene(
            indicator_type="VIDYA",
            parameters={"period": 7.0},
            op=ConditionType.LT,
            parameters_slow={"period": 21.0},
        )
    ]
    assert is_valid_chromosome(chrom)

    dsl_dict = chromosome_to_dsl(chrom)
    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = "e2e_entry_exit_cross"
    dsl = validate_strategy_dict(dsl_dict)

    indicator_names = list(dsl.indicators.keys())
    # All 4 MA legs present, all uniquely named
    assert "kama_ma_entry_0_fast" in indicator_names
    assert "kama_ma_entry_0_slow" in indicator_names
    assert "vidya_ma_exit_0_fast" in indicator_names
    assert "vidya_ma_exit_0_slow" in indicator_names
    assert len(indicator_names) == len(set(indicator_names))

    StrategyCompiler().compile_to_module(dsl)


# ---------------------------------------------------------------------------
# Edge case 2 — boundary periods (pool min/max and tight adjacency)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ma_kind", "p_fast", "p_slow"),
    [
        ("KAMA", 5.0, 50.0),     # pool min vs pool max
        ("FRAMA", 6.0, 50.0),    # FRAMA's own min (6) vs max (50)
        ("VIDYA", 5.0, 6.0),     # tight adjacency (min, min+1)
        ("KAMA", 49.0, 50.0),    # tight adjacency at the top
    ],
)
def test_boundary_periods_compile(ma_kind: str, p_fast: float, p_slow: float) -> None:
    chrom = _make_cross_chrom(ma_kind, p_fast, p_slow)
    assert is_valid_chromosome(chrom)

    dsl_dict = chromosome_to_dsl(chrom)
    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = f"e2e_boundary_{ma_kind.lower()}_{int(p_fast)}_{int(p_slow)}"
    StrategyCompiler().compile_to_module(validate_strategy_dict(dsl_dict))


# ---------------------------------------------------------------------------
# Edge case 3 — direction=BOTH emits cross to both long and short entries
# ---------------------------------------------------------------------------


def test_direction_both_emits_cross_to_long_and_short() -> None:
    chrom = _make_cross_chrom("KAMA", 10.0, 30.0)
    chrom.direction = Direction.BOTH

    dsl_dict = chromosome_to_dsl(chrom)
    long_cond = dsl_dict["entry_conditions"]["long"]  # type: ignore[index]
    short_cond = dsl_dict["entry_conditions"]["short"]  # type: ignore[index]

    cross_expr = "kama_ma_entry_0_fast > kama_ma_entry_0_slow"
    assert cross_expr in long_cond
    assert cross_expr in short_cond

    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = "e2e_direction_both"
    StrategyCompiler().compile_to_module(validate_strategy_dict(dsl_dict))


# ---------------------------------------------------------------------------
# Edge case 4 — every valid cross op compiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", _MA_CONDITION_TYPES, ids=lambda c: c.value)
def test_every_cross_op_compiles(op: ConditionType) -> None:
    chrom = _make_cross_chrom("KAMA", 10.0, 30.0)
    chrom.ma_entry_genes[0].op = op

    dsl_dict = chromosome_to_dsl(chrom)
    # Emitted condition must use the op's DSL spelling
    entry_long = dsl_dict["entry_conditions"]["long"]  # type: ignore[index]
    assert any("kama_ma_entry_0_fast" in c and "kama_ma_entry_0_slow" in c for c in entry_long)

    dsl_dict["timeframe"] = "5m"
    dsl_dict["name"] = f"e2e_op_{op.value.replace('>', 'gt').replace('<', 'lt')}"
    StrategyCompiler().compile_to_module(validate_strategy_dict(dsl_dict))


# ---------------------------------------------------------------------------
# Edge case 5 — crossover of two cross parents preserves invariant + compiles
# ---------------------------------------------------------------------------


def test_crossover_of_two_cross_parents_compiles() -> None:
    """50 random crossovers of two cross-shaped parents must each compile."""
    random.seed(321)
    parent_a = _make_cross_chrom("KAMA", 8.0, 28.0)
    parent_b = _make_cross_chrom("VIDYA", 12.0, 40.0)
    compiler = StrategyCompiler()

    for i in range(50):
        child_a, child_b = crossover(parent_a, parent_b)
        for label, child in (("a", child_a), ("b", child_b)):
            if not is_valid_chromosome(child):
                child = _repair_chromosome(child)
            for g in child.ma_entry_genes + child.ma_exit_genes:
                if g.parameters_slow is not None:
                    p_fast = g.parameters.get("period")
                    p_slow = g.parameters_slow.get("period")
                    if p_fast is not None and p_slow is not None:
                        assert p_fast < p_slow, (
                            f"iter {i} child {label}: invariant broken "
                            f"fast={p_fast} slow={p_slow}"
                        )
            dsl_dict = chromosome_to_dsl(child)
            dsl_dict["timeframe"] = "5m"
            dsl_dict["name"] = f"e2e_xo_{i}_{label}"
            compiler.compile(validate_strategy_dict(dsl_dict))
