"""inydc T1: direction=both owns its short-side genes (representation/DSL/serialization).

``chromosome_to_dsl`` used to copy the long entry/exit conditions into the short
side, and the compiler's long-first tie meant a BOTH strategy never took a short
trade. T1 gives BOTH chromosomes their own ``short_*`` genes.

This module also owns the step-0 comparability fixture:
``tests/fixtures/discovery/single_direction_digests.json`` pins the byte-level
behaviour that must NOT move for direction=long/short (R1) and the RNG
neutrality of a single-direction GA run (R16), plus the distance floats over
legacy BOTH chromosomes with empty short lists (R7c). ``build_single_direction_digests``
is the generator; re-run it to regenerate the fixture.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from vibe_quant.discovery.distance import chromosome_distance
from vibe_quant.discovery.genome import (
    chromosome_to_dsl,
    chromosome_to_serializable,
    generate_random_chromosome,
    serializable_to_chromosome,
    validate_chromosome,
)
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    PriceVsMAConditionGene,
    StrategyChromosome,
    StrategyGene,
    _random_chromosome,
    _repair_chromosome,
    crossover,
    initialize_population,
    mutate,
)
from vibe_quant.dsl.schema import StrategyDSL

DIGEST_SEED = 7
COUNT = 200
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "discovery" / "single_direction_digests.json"
)


# ---------------------------------------------------------------------------
# step-0 fixture generator
# ---------------------------------------------------------------------------


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _population_digest(direction: Direction) -> str:
    """R16: a single-direction population after 20 crossover+mutate rounds."""
    random.seed(DIGEST_SEED)
    pop = initialize_population(30, direction)
    for _ in range(20):
        nxt: list[StrategyChromosome] = []
        for i in range(0, len(pop), 2):
            child_a, child_b = crossover(pop[i], pop[i + 1])
            nxt.append(mutate(child_a, 0.1))
            nxt.append(mutate(child_b, 0.1))
        pop = nxt[:30]
    return _sha([chromosome_to_serializable(c) for c in pop])


def _non_both_distance_reprs() -> list[str]:
    random.seed(DIGEST_SEED)
    out: list[str] = []
    for _ in range(COUNT):
        a = _random_chromosome(random.choice([Direction.LONG, Direction.SHORT]))
        b = _random_chromosome(random.choice([Direction.LONG, Direction.SHORT]))
        out.append(repr(chromosome_distance(a, b)))
    return out


def _legacy_both(direction: Direction) -> StrategyChromosome:
    """A BOTH chromosome in the legacy shape: no short_entry_genes key (R7)."""
    d = chromosome_to_serializable(_random_chromosome(direction))
    d["direction"] = "both"
    return serializable_to_chromosome(d)


def _legacy_both_distance_reprs() -> list[str]:
    random.seed(DIGEST_SEED)
    out: list[str] = []
    for _ in range(50):
        a = _legacy_both(random.choice([Direction.LONG, Direction.SHORT]))
        b = _legacy_both(random.choice([Direction.LONG, Direction.SHORT]))
        out.append(repr(chromosome_distance(a, b)))
    return out


def _generate_random_serialized(seed: int) -> dict[str, object]:
    return chromosome_to_serializable(generate_random_chromosome(random.Random(seed)))


def build_single_direction_digests() -> dict[str, object]:
    """Regenerate every fixture digest on the current code. Deterministic."""
    random.seed(DIGEST_SEED)

    # (a) chromosome_to_dsl for 200 long + 200 short chromosomes (R1)
    dsl_digests: list[str] = []
    for direction in (Direction.LONG, Direction.SHORT):
        for _ in range(COUNT):
            dsl = chromosome_to_dsl(_random_chromosome(direction))
            dsl.pop("name", None)
            dsl_digests.append(_sha(dsl))

    return {
        "dsl_single_direction": _sha(dsl_digests),
        "population_long": _population_digest(Direction.LONG),
        "population_short": _population_digest(Direction.SHORT),
        "distance_non_both": _sha(_non_both_distance_reprs()),
        "distance_legacy_both": _sha(_legacy_both_distance_reprs()),
        # R1b: generate_random_chromosome is byte-identical for long/short.
        "random_generation_long": _generate_random_serialized(0),
        "random_generation_short": _generate_random_serialized(1),
    }


def test_single_direction_digests_match_fixture() -> None:
    """R1/R1b/R7c/R16: long+short behaviour is byte-identical to the step-0 fixture."""
    want = json.loads(FIXTURE_PATH.read_text())
    assert build_single_direction_digests() == want


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rsi(condition: ConditionType, threshold: float, period: float = 14.0) -> StrategyGene:
    return StrategyGene("RSI", {"period": period}, condition, threshold)


def _long_chrom() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0)],
        exit_genes=[_rsi(ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


def _both_chrom() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0)],
        exit_genes=[_rsi(ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
        short_entry_genes=[_rsi(ConditionType.GT, 70.0)],
        short_exit_genes=[_rsi(ConditionType.LT, 30.0)],
    )


# ---------------------------------------------------------------------------
# R3: both DSL uses the short genes
# ---------------------------------------------------------------------------


def test_both_dsl_uses_short_genes() -> None:
    chrom = _both_chrom()
    dsl = chromosome_to_dsl(chrom)

    assert dsl["entry_conditions"] == {
        "long": ["rsi_entry_0 < 30"],
        "short": ["rsi_short_entry_0 > 70"],
    }
    assert dsl["exit_conditions"] == {
        "long": ["rsi_exit_0 > 70"],
        "short": ["rsi_short_exit_0 < 30"],
    }
    assert list(dsl["indicators"]) == [
        "rsi_entry_0",
        "rsi_exit_0",
        "rsi_short_entry_0",
        "rsi_short_exit_0",
    ]
    StrategyDSL(**{**dsl, "timeframe": "5m"})


def test_both_short_ma_genes_get_their_own_indicator_names() -> None:
    from vibe_quant.discovery.operators import PriceVsMAConditionGene

    chrom = _both_chrom()
    chrom.short_ma_entry_genes = [
        PriceVsMAConditionGene(
            indicator_type="KAMA",
            parameters={"period": 10.0},
            op=ConditionType.GT,
            parameters_slow={"period": 30.0},
        )
    ]
    dsl = chromosome_to_dsl(chrom)
    indicators = dsl["indicators"]
    assert "kama_ma_short_entry_0_fast" in indicators
    assert "kama_ma_short_entry_0_slow" in indicators
    assert (
        "kama_ma_short_entry_0_fast > kama_ma_short_entry_0_slow"
        in dsl["entry_conditions"]["short"]  # type: ignore[index]
    )
    # long side untouched and no name collides
    assert len(indicators) == len(set(indicators))
    assert "kama_ma_entry_0_fast" not in indicators


@pytest.mark.parametrize("slot", ["entry", "exit"])
@pytest.mark.parametrize(
    ("slow", "expected"),
    [
        (None, "close < ema_ma_short_{slot}_0"),
        ({"period": 50.0}, "ema_ma_short_{slot}_0_fast < ema_ma_short_{slot}_0_slow"),
    ],
    ids=["close_vs_ma", "fast_vs_slow"],
)
def test_both_short_ma_gene_lands_on_short_side_only(
    slot: str, slow: dict[str, float] | None, expected: str
) -> None:
    chrom = _both_chrom()
    long_ma = PriceVsMAConditionGene(
        indicator_type="SMA", parameters={"period": 20.0}, op=ConditionType.GT
    )
    short_ma = PriceVsMAConditionGene(
        indicator_type="EMA",
        parameters={"period": 10.0},
        op=ConditionType.LT,
        parameters_slow=slow,
    )
    setattr(chrom, f"ma_{slot}_genes", [long_ma])
    setattr(chrom, f"short_ma_{slot}_genes", [short_ma])

    dsl = chromosome_to_dsl(chrom)

    conds = dsl[f"{slot}_conditions"]
    rsi_long = "rsi_entry_0 < 30" if slot == "entry" else "rsi_exit_0 > 70"
    rsi_short = "rsi_short_entry_0 > 70" if slot == "entry" else "rsi_short_exit_0 < 30"
    assert conds == {  # type: ignore[comparison-overlap]
        "long": [rsi_long, f"close > sma_ma_{slot}_0"],
        "short": [rsi_short, expected.format(slot=slot)],
    }
    # the other slot is untouched by this MA gene
    other = "exit" if slot == "entry" else "entry"
    assert all("ema_ma" not in c for side in dsl[f"{other}_conditions"].values() for c in side)  # type: ignore[attr-defined]
    StrategyDSL(**{**dsl, "timeframe": "5m"})


# ---------------------------------------------------------------------------
# R4 / R5: DSL direction-vs-short-gene guards
# ---------------------------------------------------------------------------


def test_dsl_both_without_short_genes_raises() -> None:
    chrom = StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0)],
        exit_genes=[_rsi(ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
    )
    with pytest.raises(
        ValueError,
        match=r"direction=both needs short_entry_genes and short_exit_genes \(got 0 entry, 0 exit\)",
    ):
        chromosome_to_dsl(chrom)


def test_dsl_single_direction_with_short_genes_raises() -> None:
    chrom = _long_chrom()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 70.0)]
    with pytest.raises(
        ValueError, match=r"short-side genes are only valid for direction=both \(got long\)"
    ):
        chromosome_to_dsl(chrom)

    short = _long_chrom()
    short.direction = Direction.SHORT
    short.short_exit_genes = [_rsi(ConditionType.LT, 30.0)]
    with pytest.raises(
        ValueError, match=r"short-side genes are only valid for direction=both \(got short\)"
    ):
        chromosome_to_dsl(short)


# ---------------------------------------------------------------------------
# R1b: generate_random_chromosome stays byte-identical for long/short
# ---------------------------------------------------------------------------


def test_generate_random_chromosome_single_direction_byte_identical() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    long_chrom = generate_random_chromosome(random.Random(0))
    short_chrom = generate_random_chromosome(random.Random(1))
    assert long_chrom.direction is Direction.LONG
    assert short_chrom.direction is Direction.SHORT
    assert chromosome_to_serializable(long_chrom) == fixture["random_generation_long"]
    assert chromosome_to_serializable(short_chrom) == fixture["random_generation_short"]


def test_generate_random_chromosome_both_draws_short_genes() -> None:
    seed = next(
        s
        for s in range(60)
        if generate_random_chromosome(random.Random(s)).direction is Direction.BOTH
    )
    chrom = generate_random_chromosome(random.Random(seed))
    assert 1 <= len(chrom.short_entry_genes) <= 5
    assert 1 <= len(chrom.short_exit_genes) <= 3
    assert validate_chromosome(chrom) == []
    StrategyDSL(**{**chromosome_to_dsl(chrom), "timeframe": "5m"})


# ---------------------------------------------------------------------------
# R6: serialization round trip
# ---------------------------------------------------------------------------


def test_both_round_trip_preserves_short_genes_and_dsl() -> None:
    original = _both_chrom()
    blob = chromosome_to_serializable(original)
    assert "short_entry_genes" in blob and "short_exit_genes" in blob
    assert "short_ma_entry_genes" not in blob  # empty short MA lists are omitted

    restored = serializable_to_chromosome(blob)
    assert restored.short_entry_genes == original.short_entry_genes
    assert restored.short_exit_genes == original.short_exit_genes
    assert chromosome_to_serializable(restored) == blob

    a = chromosome_to_dsl(original)
    b = chromosome_to_dsl(restored)
    a.pop("name", None)
    b.pop("name", None)
    assert a == b


def test_both_serialization_writes_non_empty_short_ma_lists() -> None:
    from vibe_quant.discovery.operators import PriceVsMAConditionGene

    chrom = _both_chrom()
    chrom.short_ma_exit_genes = [
        PriceVsMAConditionGene(
            indicator_type="VIDYA", parameters={"period": 8.0}, op=ConditionType.LT
        )
    ]
    blob = chromosome_to_serializable(chrom)
    assert blob["short_ma_exit_genes"]
    assert "short_ma_entry_genes" not in blob
    restored = serializable_to_chromosome(blob)
    assert restored.short_ma_exit_genes[0].indicator_type == "VIDYA"


def test_single_direction_serialization_gains_no_short_keys() -> None:
    for direction in (Direction.LONG, Direction.SHORT):
        chrom = _long_chrom()
        chrom.direction = direction
        blob = chromosome_to_serializable(chrom)
        assert not [k for k in blob if k.startswith("short")]


# ---------------------------------------------------------------------------
# R8 / R9 / R10 / R11: serializable_to_chromosome guards
# ---------------------------------------------------------------------------


def _blob(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "entry_genes": [
            {
                "indicator_type": "RSI",
                "parameters": {"period": 14.0},
                "condition": "<",
                "threshold": 30.0,
                "sub_value": None,
            }
        ],
        "exit_genes": [
            {
                "indicator_type": "RSI",
                "parameters": {"period": 14.0},
                "condition": ">",
                "threshold": 70.0,
                "sub_value": None,
            }
        ],
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "direction": "long",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("value", "type_name"),
    [(None, "NoneType"), ("nope", "str"), ({"a": 1}, "dict"), (3, "int")],
)
def test_r8_r9_short_entry_genes_must_be_a_list(value: object, type_name: str) -> None:
    with pytest.raises(
        ValueError, match=rf"short_entry_genes must be a list, got {type_name}"
    ):
        serializable_to_chromosome(_blob(direction="short", short_entry_genes=value))


def test_r10_both_with_empty_short_entry_raises() -> None:
    with pytest.raises(ValueError, match="direction=both chromosome has no short_entry_genes"):
        serializable_to_chromosome(
            _blob(direction="both", short_entry_genes=[], short_exit_genes=[{}])
        )


def test_r10_both_with_empty_short_exit_raises() -> None:
    entry = _blob()["entry_genes"]
    with pytest.raises(ValueError, match="direction=both chromosome has no short_exit_genes"):
        serializable_to_chromosome(
            _blob(direction="both", short_entry_genes=entry, short_exit_genes=[])
        )


def test_r11_long_with_short_key_raises_even_when_empty() -> None:
    with pytest.raises(
        ValueError, match=r"short_entry_genes is only valid for direction=both \(got long\)"
    ):
        serializable_to_chromosome(_blob(direction="long", short_entry_genes=[]))


def test_r11_short_with_short_ma_key_raises() -> None:
    with pytest.raises(
        ValueError, match=r"short_ma_entry_genes is only valid for direction=both \(got short\)"
    ):
        serializable_to_chromosome(
            _blob(direction="short", short_ma_entry_genes=[{}])
        )


def test_r8_precedence_over_r11() -> None:
    """A type error is reported before the direction check, even for a long chromosome."""
    with pytest.raises(ValueError, match="short_entry_genes must be a list, got NoneType"):
        serializable_to_chromosome(_blob(direction="long", short_entry_genes=None))


def test_legacy_both_without_short_keys_stays_both() -> None:
    """R7 ruling: generic deserialization keeps a legacy BOTH as BOTH (seed.py converts)."""
    chrom = serializable_to_chromosome(_blob(direction="both"))
    assert chrom.direction is Direction.BOTH
    assert chrom.short_entry_genes == [] and chrom.short_exit_genes == []


# ---------------------------------------------------------------------------
# R17 / R17b: validate_chromosome
# ---------------------------------------------------------------------------


def test_r17_both_without_short_genes_is_invalid() -> None:
    chrom = StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0)],
        exit_genes=[_rsi(ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
    )
    errors = validate_chromosome(chrom)
    assert any("short entry" in e for e in errors)
    assert any("short exit" in e for e in errors)


def test_r17_single_direction_with_short_gene_is_invalid() -> None:
    chrom = _long_chrom()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 70.0)]
    errors = validate_chromosome(chrom)
    assert any("only valid for direction=both" in e for e in errors)


def test_r17_out_of_range_short_threshold_is_invalid() -> None:
    chrom = _both_chrom()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 150.0)]
    errors = validate_chromosome(chrom)
    assert any("threshold" in e for e in errors)


def test_r17_valid_both_chromosome_passes() -> None:
    assert validate_chromosome(_both_chrom()) == []


def test_r17b_out_of_pool_short_indicator_is_invalid() -> None:
    chrom = _both_chrom()
    chrom.short_entry_genes = [StrategyGene("NOT_AN_IND", {"period": 14.0}, ConditionType.GT, 1.0)]
    errors = validate_chromosome(chrom)
    assert any("NOT_AN_IND" in e for e in errors)


def test_r17b_short_macd_fast_ge_slow_is_invalid() -> None:
    chrom = _both_chrom()
    chrom.short_entry_genes = [
        StrategyGene(
            "MACD",
            {"fast_period": 30.0, "slow_period": 10.0, "signal_period": 9.0},
            ConditionType.GT,
            0.0,
        )
    ]
    errors = validate_chromosome(chrom)
    assert any("fast_period" in e for e in errors)


def test_r17b_short_ma_out_of_pool_is_invalid() -> None:
    from vibe_quant.discovery.operators import PriceVsMAConditionGene

    chrom = _both_chrom()
    chrom.short_ma_entry_genes = [
        PriceVsMAConditionGene(
            indicator_type="NOPE", parameters={"period": 10.0}, op=ConditionType.GT
        )
    ]
    errors = validate_chromosome(chrom)
    assert any("NOPE" in e for e in errors)


# ---------------------------------------------------------------------------
# R16b: mock-backtest seeding stays base-identical for a long chromosome
# ---------------------------------------------------------------------------


def test_r16b_mock_backtest_long_sharpe_unchanged() -> None:
    from vibe_quant.discovery.mock_backtest import mock_backtest

    random.seed(1)
    chrom = _random_chromosome(Direction.LONG)
    assert mock_backtest(chrom)["sharpe_ratio"] == 1.547214633099186


# ---------------------------------------------------------------------------
# scalar_gene_count helper (T1; consumed by fitness/pipeline in T3)
# ---------------------------------------------------------------------------


def test_scalar_gene_count_sums_both_sides() -> None:
    from vibe_quant.discovery.operators import scalar_gene_count

    assert scalar_gene_count(_long_chrom()) == 2
    both = _both_chrom()
    assert scalar_gene_count(both) == 4
    both.entry_genes.append(_rsi(ConditionType.LT, 35.0))
    both.short_exit_genes.append(_rsi(ConditionType.LT, 25.0))
    assert scalar_gene_count(both) == 6


def test_clone_copies_short_lists_independently() -> None:
    original = _both_chrom()
    clone = original.clone()
    assert clone.uid == original.uid
    clone.short_entry_genes[0].threshold = 99.0
    assert original.short_entry_genes[0].threshold == 70.0


def test_clone_short_gene_count() -> None:
    original = _both_chrom()
    clone = original.clone()
    assert clone.short_entry_genes == original.short_entry_genes
    assert clone.short_exit_genes == original.short_exit_genes
    assert clone.short_ma_entry_genes == [] and clone.short_ma_exit_genes == []


# ---------------------------------------------------------------------------
# R14: crossover+repair reconciles the short side with the child's direction
# ---------------------------------------------------------------------------


def test_r14_repair_clears_short_side_on_non_both() -> None:
    """A repaired LONG/SHORT chromosome owns no short-side genes."""
    chrom = _long_chrom()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 70.0)]
    chrom.short_exit_genes = [_rsi(ConditionType.LT, 30.0)]
    chrom.short_ma_entry_genes = [
        PriceVsMAConditionGene(
            indicator_type="KAMA", parameters={"period": 10.0}, op=ConditionType.GT
        )
    ]
    repaired = _repair_chromosome(chrom)
    assert repaired is chrom
    assert repaired.short_entry_genes == []
    assert repaired.short_exit_genes == []
    assert repaired.short_ma_entry_genes == []
    assert repaired.short_ma_exit_genes == []


def test_r14_repair_fills_and_caps_short_side_on_both() -> None:
    """BOTH is topped up to 1 entry / 1 exit and capped at 5 / 3 / one MA per slot."""
    random.seed(3)
    empty = _both_chrom()
    empty.short_entry_genes = []
    empty.short_exit_genes = []
    _repair_chromosome(empty)
    assert len(empty.short_entry_genes) == 1
    assert len(empty.short_exit_genes) == 1

    oversized = _both_chrom()
    oversized.short_entry_genes = [_rsi(ConditionType.GT, 70.0) for _ in range(7)]
    oversized.short_exit_genes = [_rsi(ConditionType.LT, 30.0) for _ in range(5)]
    oversized.short_ma_entry_genes = [
        PriceVsMAConditionGene(
            indicator_type="KAMA", parameters={"period": 10.0}, op=ConditionType.GT
        )
        for _ in range(2)
    ]
    oversized.short_ma_exit_genes = [
        PriceVsMAConditionGene(
            indicator_type="VIDYA", parameters={"period": 8.0}, op=ConditionType.LT
        )
        for _ in range(2)
    ]
    _repair_chromosome(oversized)
    assert len(oversized.short_entry_genes) == 5
    assert len(oversized.short_exit_genes) == 3
    assert len(oversized.short_ma_entry_genes) == 1
    assert len(oversized.short_ma_exit_genes) == 1


def test_r14_seeded_crossover_repair_short_side_bounds() -> None:
    """500 seeded crossovers: non-both children own no short genes, BOTH is in bounds."""
    random.seed(11)
    parents = [
        _both_chrom(),
        _long_chrom(),
        _both_chrom(),
        _long_chrom(),
    ]
    for i in range(500):
        a = parents[i % len(parents)]
        b = parents[(i + 1) % len(parents)]
        for child in crossover(a, b):
            if child.direction is Direction.BOTH:
                assert 1 <= len(child.short_entry_genes) <= 5
                assert 1 <= len(child.short_exit_genes) <= 3
                assert len(child.short_ma_entry_genes) <= 1
                assert len(child.short_ma_exit_genes) <= 1
            else:
                assert child.short_entry_genes == []
                assert child.short_exit_genes == []
                assert child.short_ma_entry_genes == []
                assert child.short_ma_exit_genes == []
