"""inydc T2: genetic operators / validity / distance / diversity on the short side.

T1 gave ``direction=both`` chromosomes their own ``short_entry_genes`` /
``short_exit_genes`` / ``short_ma_*`` lists and the DSL/serialization for them.
T2 makes the rest of the discovery layer actually use those lists:

- ``crossover`` crosses each short list (min 0, so no RNG draw when both parents
  carry none -- the long/short RNG stream must not move, R16);
- ``mutate`` mutates the short lists only for BOTH and only AFTER every existing
  draw (mutating an empty list draws RNG, so the direction guard is load-bearing);
- ``is_valid_chromosome`` enforces the short-side rules (R17/R17b);
- ``initialize_population`` reconciles seeded chromosomes before validating them;
- ``chromosome_distance`` pads missing short slots at gene distance 1.0 (R18);
- ``population_entropy`` counts short genes in the indicator/condition counters (R19).

The fixture digest test in ``test_genome_both_sides.py`` is the acceptance test
for R16 and must stay green unmodified; ``test_single_direction_rng_neutral``
here re-derives the same digest from a full crossover+mutate run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import pytest

from vibe_quant.discovery.distance import chromosome_distance
from vibe_quant.discovery.diversity import population_entropy
from vibe_quant.discovery.genome import chromosome_to_serializable
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    PriceVsMAConditionGene,
    StrategyChromosome,
    StrategyGene,
    _ensure_pool,
    _random_chromosome,
    crossover,
    initialize_population,
    is_valid_chromosome,
    mutate,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "discovery" / "single_direction_digests.json"
)
DIGEST_SEED = 7


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rsi(condition: ConditionType, threshold: float, period: float = 14.0) -> StrategyGene:
    return StrategyGene("RSI", {"period": period}, condition, threshold)


def _ma(
    indicator: str = "SMA",
    period: float = 20.0,
    op: ConditionType = ConditionType.GT,
    slow: dict[str, float] | None = None,
) -> PriceVsMAConditionGene:
    return PriceVsMAConditionGene(indicator, {"period": period}, op, slow)


def _long() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0)],
        exit_genes=[_rsi(ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


def _both() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0)],
        exit_genes=[_rsi(ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
        short_entry_genes=[_rsi(ConditionType.GT, 70.0)],
        short_exit_genes=[_rsi(ConditionType.LT, 30.0)],
    )


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# R16: the long/short RNG stream is unchanged by the short-side work
# ---------------------------------------------------------------------------


def _population_digest(direction: Direction) -> str:
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


def test_single_direction_rng_neutral() -> None:
    """R16: crossover+mutate over 20 rounds draws exactly the fixture RNG."""
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert _population_digest(Direction.LONG) == fixture["population_long"]
    assert _population_digest(Direction.SHORT) == fixture["population_short"]


# ---------------------------------------------------------------------------
# R14: mixed-parent crossover/mutation reconciles the short side with direction
# ---------------------------------------------------------------------------


def test_crossover_short_lists_bounds() -> None:
    """R14: 500 seeded crossovers -- non-BOTH owns no short genes, BOTH is in bounds."""
    random.seed(11)
    parents = [_both(), _long(), _both(), _long()]
    seen_both = seen_single = 0
    for i in range(500):
        a = parents[i % len(parents)]
        b = parents[(i + 1) % len(parents)]
        for child in crossover(a, b):
            if child.direction is Direction.BOTH:
                seen_both += 1
                assert 1 <= len(child.short_entry_genes) <= 5
                assert 1 <= len(child.short_exit_genes) <= 3
                assert len(child.short_ma_entry_genes) <= 1
                assert len(child.short_ma_exit_genes) <= 1
                assert is_valid_chromosome(child)
            else:
                seen_single += 1
                assert child.short_entry_genes == []
                assert child.short_exit_genes == []
                assert child.short_ma_entry_genes == []
                assert child.short_ma_exit_genes == []
    assert seen_both and seen_single


def test_crossover_mutate_short_lists_bounds() -> None:
    """R14: crossover THEN mutate keeps each short list in bounds for BOTH children."""
    random.seed(12)
    parents = [_both(), _long()]
    for i in range(500):
        a = parents[i % 2]
        b = parents[(i + 1) % 2]
        for child in crossover(a, b):
            child = mutate(child, 0.5)
            if child.direction is Direction.BOTH:
                assert 1 <= len(child.short_entry_genes) <= 5
                assert 1 <= len(child.short_exit_genes) <= 3
                assert len(child.short_ma_entry_genes) <= 1
                assert len(child.short_ma_exit_genes) <= 1
                assert is_valid_chromosome(child)
            else:
                assert child.short_entry_genes == []
                assert child.short_exit_genes == []
                assert child.short_ma_entry_genes == []
                assert child.short_ma_exit_genes == []


def test_crossover_inherits_short_genes_from_parents() -> None:
    """A BOTH x BOTH crossover takes each short gene from one of its parents."""
    random.seed(5)
    a = _both()
    a.short_entry_genes = [_rsi(ConditionType.GT, 70.0)]
    a.short_exit_genes = [_rsi(ConditionType.LT, 30.0)]
    b = _both()
    b.short_entry_genes = [_rsi(ConditionType.GT, 20.0)]
    b.short_exit_genes = [_rsi(ConditionType.LT, 80.0)]

    entry_thresholds: set[float] = set()
    for _ in range(200):
        for child in crossover(a, b):
            assert child.direction is Direction.BOTH
            # Each parent has exactly one short slot, so crossover fills exactly
            # one -- no repair draw can add a fresh gene here.
            assert len(child.short_entry_genes) == 1
            assert len(child.short_exit_genes) == 1
            entry_thresholds.add(child.short_entry_genes[0].threshold)
            assert child.short_exit_genes[0].threshold in {30.0, 80.0}
    assert entry_thresholds == {70.0, 20.0}


def test_crossover_crosses_short_ma_genes() -> None:
    random.seed(6)
    a = _both()
    a.short_ma_entry_genes = [_ma("KAMA", 20.0)]
    b = _both()
    b.short_ma_entry_genes = [_ma("VIDYA", 10.0)]
    seen: set[str | None] = set()
    for _ in range(100):
        for child in crossover(a, b):
            assert child.direction is Direction.BOTH
            assert len(child.short_ma_entry_genes) == 1
            seen.add(child.short_ma_entry_genes[0].indicator_type)
    assert seen == {"KAMA", "VIDYA"}


# ---------------------------------------------------------------------------
# mutate: short lists only for BOTH, after every existing draw
# ---------------------------------------------------------------------------


def test_mutate_skips_short_lists_for_non_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long/short chromosome mutates exactly two gene lists (its own two)."""
    import vibe_quant.discovery.operators as ops

    calls: list[int] = []
    original = ops._mutate_genes

    def spy(genes: list[StrategyGene], rate: float, min_count: int, max_count: int):
        calls.append(len(genes))
        return original(genes, rate, min_count, max_count)

    monkeypatch.setattr(ops, "_mutate_genes", spy)

    random.seed(1)
    mutate(_long(), 1.0)
    assert calls == [1, 1]  # entry + exit only

    calls.clear()
    random.seed(1)
    mutate(_both(), 1.0)
    assert calls == [1, 1, 1, 1]  # entry + exit + short entry + short exit


def test_mutate_long_keeps_short_lists_empty() -> None:
    random.seed(3)
    for _ in range(50):
        m = mutate(_long(), 1.0)
        assert m.short_entry_genes == []
        assert m.short_exit_genes == []
        assert m.short_ma_entry_genes == []
        assert m.short_ma_exit_genes == []


def test_mutate_fills_empty_both_short_lists() -> None:
    """A BOTH chromosome with empty short lists is refilled (pipeline _make_valid)."""
    random.seed(2)
    chrom = _both()
    chrom.short_entry_genes = []
    chrom.short_exit_genes = []
    m = mutate(chrom, 1.0)
    assert 1 <= len(m.short_entry_genes) <= 5
    assert 1 <= len(m.short_exit_genes) <= 3
    assert is_valid_chromosome(m)


def test_mutate_both_short_lists_change_and_stay_bounded() -> None:
    random.seed(9)
    chrom = _both()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 70.0) for _ in range(3)]
    chrom.short_exit_genes = [_rsi(ConditionType.LT, 30.0) for _ in range(2)]
    baseline = json.dumps(
        chromosome_to_serializable(chrom)["short_entry_genes"], sort_keys=True
    )
    seen: set[str] = set()
    for _ in range(100):
        m = mutate(chrom, 1.0)
        assert 1 <= len(m.short_entry_genes) <= 5
        assert 1 <= len(m.short_exit_genes) <= 3
        assert is_valid_chromosome(m)
        blob = chromosome_to_serializable(m)
        seen.add(json.dumps(blob["short_entry_genes"], sort_keys=True))
    assert baseline not in seen  # short entries actually mutated across trials


# ---------------------------------------------------------------------------
# R17 / R17b: is_valid_chromosome
# ---------------------------------------------------------------------------


def test_is_valid_both_without_short_genes_is_false() -> None:
    chrom = _both()
    chrom.short_entry_genes = []
    assert not is_valid_chromosome(chrom)
    chrom = _both()
    chrom.short_exit_genes = []
    assert not is_valid_chromosome(chrom)


def test_is_valid_both_with_short_genes_is_true() -> None:
    assert is_valid_chromosome(_both())
    chrom = _both()
    chrom.short_ma_entry_genes = [_ma("KAMA", 10.0, slow={"period": 30.0})]
    assert is_valid_chromosome(chrom)


def test_is_valid_single_direction_with_short_genes_is_false() -> None:
    chrom = _long()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 70.0)]
    assert not is_valid_chromosome(chrom)

    chrom = _long()
    chrom.short_ma_exit_genes = [_ma()]
    assert not is_valid_chromosome(chrom)

    short = _long()
    short.direction = Direction.SHORT
    short.short_exit_genes = [_rsi(ConditionType.LT, 30.0)]
    assert not is_valid_chromosome(short)


def test_is_valid_out_of_range_short_threshold_is_false() -> None:
    chrom = _both()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 150.0)]
    assert not is_valid_chromosome(chrom)


def test_is_valid_out_of_pool_short_indicator_is_false() -> None:
    chrom = _both()
    chrom.short_entry_genes = [
        StrategyGene("NOT_AN_IND", {"period": 14.0}, ConditionType.GT, 1.0)
    ]
    assert not is_valid_chromosome(chrom)


def test_is_valid_short_macd_fast_ge_slow_is_false() -> None:
    chrom = _both()
    chrom.short_entry_genes = [
        StrategyGene(
            "MACD",
            {"fast_period": 30.0, "slow_period": 10.0, "signal_period": 9.0},
            ConditionType.GT,
            0.0,
        )
    ]
    assert not is_valid_chromosome(chrom)


def test_is_valid_short_ma_out_of_pool_is_false() -> None:
    chrom = _both()
    chrom.short_ma_entry_genes = [_ma("NOPE")]
    assert not is_valid_chromosome(chrom)


def test_is_valid_short_ma_cross_fast_ge_slow_is_false() -> None:
    chrom = _both()
    chrom.short_ma_exit_genes = [_ma("KAMA", 50.0, slow={"period": 10.0})]
    assert not is_valid_chromosome(chrom)


def test_is_valid_short_lists_over_cap_are_false() -> None:
    chrom = _both()
    chrom.short_entry_genes = [_rsi(ConditionType.GT, 70.0)] * 6
    assert not is_valid_chromosome(chrom)

    chrom = _both()
    chrom.short_exit_genes = [_rsi(ConditionType.LT, 30.0)] * 4
    assert not is_valid_chromosome(chrom)

    chrom = _both()
    chrom.short_ma_entry_genes = [_ma("KAMA"), _ma("KAMA")]
    assert not is_valid_chromosome(chrom)


# ---------------------------------------------------------------------------
# R18: chromosome_distance pads missing short slots like missing long slots
# ---------------------------------------------------------------------------


def test_r18_both_vs_long_same_long_genes_is_exactly_0_4() -> None:
    long_ = _long()
    both = _both()
    assert chromosome_distance(both, long_) == 0.4
    assert chromosome_distance(long_, both) == 0.4


def test_r18_missing_short_slots_padded_at_gene_distance_1_0() -> None:
    """BOTH(1 short entry, 1 short exit) vs BOTH(no short genes): 2 padded slots.

    gene_dists = [entry 0, exit 0, short_entry 1.0, short_exit 1.0] -> avg 0.5,
    weighted 3.0 * 0.5 = 1.5, total weight 5.0 -> 0.3 exactly.
    """
    full = _both()
    empty = _both()
    empty.short_entry_genes = []
    empty.short_exit_genes = []
    assert chromosome_distance(full, empty) == 0.3


def test_r18_short_genes_contribute_distance() -> None:
    a = _both()
    b = _both()
    b.short_entry_genes = [_rsi(ConditionType.GT, 20.0)]
    assert chromosome_distance(a, b) > 0.0
    assert chromosome_distance(a, a.clone()) == 0.0


def test_r18_non_both_distance_matches_fixture() -> None:
    """R18/R7c: single-direction pairs keep the step-0 distance floats."""
    random.seed(DIGEST_SEED)
    reprs: list[str] = []
    for _ in range(200):
        a = _random_chromosome(random.choice([Direction.LONG, Direction.SHORT]))
        b = _random_chromosome(random.choice([Direction.LONG, Direction.SHORT]))
        reprs.append(repr(chromosome_distance(a, b)))
    fixture = json.loads(FIXTURE_PATH.read_text())
    assert _sha(reprs) == fixture["distance_non_both"]


# ---------------------------------------------------------------------------
# R19: population_entropy counts short genes
# ---------------------------------------------------------------------------


def test_r19_entropy_counts_short_genes() -> None:
    a = _both()
    a.short_entry_genes = []
    a.short_exit_genes = []
    b = _both()
    b.short_entry_genes = []
    b.short_exit_genes = []
    before = population_entropy([a, b])

    c = _both()
    c.short_entry_genes = [StrategyGene("CCI", {"period": 20.0}, ConditionType.GT, 100.0)]
    c.short_exit_genes = [StrategyGene("CCI", {"period": 20.0}, ConditionType.LT, -100.0)]
    after = population_entropy([c, b])

    assert before != after
    assert after > before


def test_r19_non_both_entropy_unchanged() -> None:
    """Two LONG chromosomes, RSI vs CCI, all conditions equal: exactly 1 bit of
    indicator entropy / log2(n_indicators) / 3 (the short lists are empty)."""
    _ensure_pool()
    from vibe_quant.discovery.operators import _INDICATOR_NAMES

    rsi = _long()
    rsi.exit_genes = [_rsi(ConditionType.LT, 70.0)]
    cci = _long()
    cci.entry_genes = [StrategyGene("CCI", {"period": 20.0}, ConditionType.LT, -100.0)]
    cci.exit_genes = [StrategyGene("CCI", {"period": 20.0}, ConditionType.LT, 100.0)]

    n = max(len(_INDICATOR_NAMES), 1)
    expected = (1.0 / math.log2(n)) / 3.0
    assert population_entropy([rsi, cci]) == pytest.approx(expected)
