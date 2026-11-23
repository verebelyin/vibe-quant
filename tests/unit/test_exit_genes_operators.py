"""GA exit genes T2: draw, mutate, cross, validity and penalty of an extended genome.

``exit_rules is None`` (the legacy genome) must draw no RNG and move no float, so a
default run stays bit-identical; an extended genome stays valid and renderable through
every operator, and its legacy SL/TP fields stay pinned.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from vibe_quant.discovery.distance import chromosome_distance
from vibe_quant.discovery.exit_genes import (
    ATR_PERIOD,
    STOP_TYPES,
    TARGET_TYPES,
    ExitBlock,
    ExitRules,
    canonical_block,
    max_bars_range,
    mutate_rules,
    random_rules,
    rules_from_legacy,
    rules_problems,
)
from vibe_quant.discovery.fitness import chromosome_sl_tp_penalty
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
    _random_chromosome,
    crossover,
    initialize_population,
    is_valid_chromosome,
    mutate,
)
from vibe_quant.dsl.schema import StrategyDSL

TF = "4h"
ALL_DIRECTIONS = (Direction.LONG, Direction.SHORT, Direction.BOTH)


def _gene(period: float = 14.0) -> StrategyGene:
    return StrategyGene("RSI", {"period": period}, ConditionType.LT, 30.0)


def _block(
    stop: str, stop_v: float | None, target: str, target_v: float | None, atr: int | None = None
) -> ExitBlock:
    return canonical_block(ExitBlock(stop, stop_v, target, target_v, atr))  # type: ignore[arg-type]


def _chrom(
    direction: Direction, rules: ExitRules | None, *, sl: float = 2.0, tp: float = 3.0
) -> StrategyChromosome:
    both = direction == Direction.BOTH
    return StrategyChromosome(
        entry_genes=[_gene()],
        exit_genes=[_gene(10.0)],
        stop_loss_pct=sl,
        take_profit_pct=tp,
        direction=direction,
        short_entry_genes=[_gene(9.0)] if both else [],
        short_exit_genes=[_gene(8.0)] if both else [],
        exit_rules=rules,
    )


def _assert_pinned(chrom: StrategyChromosome) -> None:
    assert chrom.stop_loss_pct == 2.0
    assert chrom.take_profit_pct == 3.0
    assert chrom.stop_loss_long_pct is None
    assert chrom.stop_loss_short_pct is None
    assert chrom.take_profit_long_pct is None
    assert chrom.take_profit_short_pct is None


# ---------------------------------------------------------------------------
# R2: the legacy genome space draws nothing new
# ---------------------------------------------------------------------------


def test_legacy_genome_draws_nothing_new() -> None:
    """Guard: seeded legacy calls leave the global RNG exactly where main's code left it.

    The literals were measured on the base before the exit-gene hooks existed; a hook that
    draws on the legacy path moves the next ``getrandbits`` and the float sums.
    """
    random.seed(11)
    pop = initialize_population(6, None)
    pop += [_random_chromosome(Direction.BOTH), _random_chromosome(Direction.SHORT)]
    kids = [mutate(c, 0.5) for c in pop]
    crossover(pop[1], pop[6])
    crossover(pop[0], pop[7])
    distances = [chromosome_distance(p, q) for p, q in zip(pop, kids, strict=True)]
    penalties = [chromosome_sl_tp_penalty(p) for p in pop]

    assert random.getrandbits(64) == 11284078870858183469
    assert sum(distances) == 0.7090200888720918
    assert sum(penalties) == 0.12387998056365401
    assert all(c.exit_rules is None for c in pop + kids)


# ---------------------------------------------------------------------------
# Draw order (design 2.2): a seeded extended run replays
# ---------------------------------------------------------------------------


_STOP_RANGE = {"fixed_pct": (0.5, 10.0), "atr_fixed": (1.0, 6.0), "atr_trailing": (1.0, 6.0)}
_TARGET_RANGE = {"fixed_pct": (0.5, 20.0), "atr_fixed": (1.0, 10.0), "risk_reward": (1.0, 5.0)}


def _replay_block(ref: random.Random) -> ExitBlock:
    """One block drawn by hand in the documented order: stop, target, ATR period."""
    stop_type = ref.choice(STOP_TYPES)
    stop_v = None if stop_type == "safety" else round(ref.uniform(*_STOP_RANGE[stop_type]), 2)
    target_type = ref.choice(TARGET_TYPES)
    target_v = None if target_type == "none" else round(ref.uniform(*_TARGET_RANGE[target_type]), 2)
    needs_atr = stop_type in ("atr_fixed", "atr_trailing") or target_type == "atr_fixed"
    atr = ref.randint(*ATR_PERIOD) if needs_atr else None
    return canonical_block(ExitBlock(stop_type, stop_v, target_type, target_v, atr))


def _replay_rules(ref: random.Random, *, both: bool) -> ExitRules:
    long = _replay_block(ref)
    short = _replay_block(ref) if both else None
    max_bars = ref.randint(*max_bars_range(TF)) if ref.random() < 0.3 else None
    return ExitRules(long, short, max_bars)


@pytest.mark.simple_target(
    reason="pins the RNG draw order a seeded extended run replays; callers only see validity"
)
@pytest.mark.parametrize("direction", ALL_DIRECTIONS)
def test_random_rules_draw_order_is_pinned(direction: Direction) -> None:
    """Replaying the documented draw order by hand gives the same rules for 60 seeds."""
    for seed in range(60):
        random.seed(seed)
        got = random_rules(direction, TF)
        assert got == _replay_rules(random.Random(seed), both=direction == Direction.BOTH)


# ---------------------------------------------------------------------------
# R4c: legacy fields of an extended genome are dead and pinned
# ---------------------------------------------------------------------------


def test_legacy_fields_pinned() -> None:
    rules = ExitRules(_block("atr_trailing", 2.5, "risk_reward", 2.0, 14), None, 12)
    a = _chrom(Direction.LONG, rules, sl=1.0, tp=9.0)
    b = _chrom(Direction.LONG, rules, sl=7.5, tp=15.0)
    a.uid = b.uid = "same"
    assert chromosome_to_dsl(a) == chromosome_to_dsl(b)

    random.seed(3)
    produced: list[StrategyChromosome] = []
    for direction in ALL_DIRECTIONS:
        produced.append(_random_chromosome(direction, exit_timeframe=TF))
    produced.extend(mutate(c, 1.0, exit_timeframe=TF) for c in list(produced))
    produced.extend(initialize_population(8, None, exit_timeframe=TF))
    seeds = [
        _chrom(Direction.BOTH, None, sl=1.5, tp=8.0),
        _chrom(Direction.LONG, None, sl=4.0, tp=6.0),
    ]
    seeds[0].stop_loss_short_pct = 3.0
    seeds[0].take_profit_long_pct = 12.0
    produced.extend(initialize_population(8, None, seeds, exit_timeframe=TF)[:2])
    for _ in range(40):
        c1, c2 = crossover(produced[0], produced[3])
        produced.extend([c1, c2])
    assert all(c.exit_rules is not None for c in produced)
    for c in produced:
        _assert_pinned(c)


# ---------------------------------------------------------------------------
# R7: validity
# ---------------------------------------------------------------------------

_GOOD = _block("atr_fixed", 2.0, "fixed_pct", 4.0, 14)


@pytest.mark.parametrize(
    ("direction", "rules", "needle"),
    [
        (Direction.BOTH, ExitRules(_GOOD, None, None), "exit_rules.short: required"),
        (Direction.LONG, ExitRules(_GOOD, _GOOD, None), "only valid for direction=both"),
        (
            Direction.LONG,
            ExitRules(ExitBlock("fixed_pct", 10.5, "none", None, None), None, None),
            "exit_rules.long.stop_value: 10.5 outside [0.5, 10.0]",
        ),
        (
            Direction.LONG,
            ExitRules(ExitBlock("fixed_pct", 2.0, "risk_reward", 5.5, None), None, None),
            "exit_rules.long.target_value: 5.5 outside [1.0, 5.0]",
        ),
        (
            Direction.LONG,
            ExitRules(ExitBlock("atr_fixed", 2.0, "none", None, 31), None, None),
            "atr_period: 31 outside [7, 30]",
        ),
        (
            Direction.LONG,
            ExitRules(ExitBlock("fixed_pct", 2.0, "none", None, 14), None, None),
            "atr_period: only valid with an ATR",
        ),
        (Direction.LONG, ExitRules(_GOOD, None, 0), "exit_rules.max_bars: 0 outside [1, 100000]"),
        (
            Direction.LONG,
            ExitRules(_GOOD, None, 100_001),
            "exit_rules.max_bars: 100001 outside [1, 100000]",
        ),
        (
            Direction.LONG,
            ExitRules(ExitBlock("safety", None, "risk_reward", 2.0, None), None, None),
            "not canonical",
        ),
    ],
)
def test_validity_rules(direction: Direction, rules: ExitRules, needle: str) -> None:
    problems = rules_problems(rules, direction)
    assert any(needle in p for p in problems), problems
    assert not is_valid_chromosome(_chrom(direction, rules))


@pytest.mark.parametrize("max_bars", [1, 100_000, None])
def test_validity_accepts_edges(max_bars: int | None) -> None:
    rules = ExitRules(_GOOD, None, max_bars)
    assert rules_problems(rules, Direction.LONG) == []
    assert is_valid_chromosome(_chrom(Direction.LONG, rules))


def test_valid_both_genome_is_valid() -> None:
    assert is_valid_chromosome(_chrom(Direction.BOTH, ExitRules(_GOOD, _GOOD, 12)))


# ---------------------------------------------------------------------------
# R8: operators keep every child valid and renderable
# ---------------------------------------------------------------------------


def _assert_child_ok(child: StrategyChromosome) -> None:
    assert child.exit_rules is not None
    assert is_valid_chromosome(child), rules_problems(child.exit_rules, child.direction)
    StrategyDSL(**chromosome_to_dsl(child))  # type: ignore[arg-type]
    _assert_pinned(child)
    rules = child.exit_rules
    assert (rules.short is not None) == (child.direction == Direction.BOTH)
    if rules.max_bars is not None:
        lo, hi = max_bars_range(TF)
        assert lo <= rules.max_bars <= hi
    for block in (rules.long, rules.short):
        if block is not None and block.atr_period is not None:
            assert ATR_PERIOD[0] <= block.atr_period <= ATR_PERIOD[1]


def test_mutate_and_cross_keep_parser_valid() -> None:
    random.seed(7)
    switched_stop = switched_target = timed = untimed = 0
    population = [
        _random_chromosome(d, exit_timeframe=TF) for d in ALL_DIRECTIONS for _ in range(4)
    ]
    for i in range(2000):
        parent = population[i % len(population)]
        child = mutate(parent, 0.3, exit_timeframe=TF)
        _assert_child_ok(child)
        assert child.uid != parent.uid
        assert parent.exit_rules is not None and child.exit_rules is not None
        switched_stop += child.exit_rules.long.stop_type != parent.exit_rules.long.stop_type
        switched_target += child.exit_rules.long.target_type != parent.exit_rules.long.target_type
        timed += child.exit_rules.max_bars is not None
        untimed += child.exit_rules.max_bars is None
        population[i % len(population)] = child

    for _ in range(1000):
        a, b = random.sample(population, 2)
        for child in crossover(a, b):
            _assert_child_ok(child)
            assert child.exit_rules is not None
            parents = [p.exit_rules for p in (a, b) if p.exit_rules is not None]
            assert any(child.exit_rules.long is p.long for p in parents)
            if child.exit_rules.short is not None:
                assert any(
                    child.exit_rules.short is p.short or child.exit_rules.short is p.long
                    for p in parents
                )
    assert switched_stop >= 1
    assert switched_target >= 1
    assert timed >= 1
    assert untimed >= 1


@pytest.mark.parametrize(
    "rules",
    [
        # every value on its upper edge, time exit on its lower edge
        ExitRules(_block("atr_trailing", 6.0, "atr_fixed", 10.0, 30), None, 3),
        # every value on its lower edge, time exit on its upper edge
        ExitRules(_block("atr_fixed", 1.0, "atr_fixed", 1.0, 7), None, 120),
    ],
)
def test_mutate_clamps_perturbed_values_into_range(rules: ExitRules) -> None:
    random.seed(1)
    moved = 0
    for _ in range(500):
        out = mutate_rules(rules, 1.0, TF)
        assert rules_problems(out, Direction.LONG) == []
        assert out.max_bars is None or 3 <= out.max_bars <= 120
        moved += out != rules
    assert moved > 100


def test_mutate_extended_without_timeframe_is_refused() -> None:
    chrom = _random_chromosome(Direction.LONG, exit_timeframe=TF)
    with pytest.raises(ValueError, match="exit_timeframe"):
        mutate(chrom, 0.3)


def test_crossover_of_mixed_exit_spaces_is_refused() -> None:
    extended = _random_chromosome(Direction.LONG, exit_timeframe=TF)
    legacy = _chrom(Direction.LONG, None)
    with pytest.raises(ValueError, match="mixed exit gene spaces"):
        crossover(extended, legacy)
    with pytest.raises(ValueError, match="mixed exit gene spaces"):
        crossover(legacy, extended)


def test_crossover_to_both_copies_long_block_to_short() -> None:
    """A BOTH child of two LONG parents has no short block to inherit: short = long, no RNG."""
    ra = ExitRules(_block("atr_trailing", 2.0, "none", None, 14), None, 12)
    rb = ExitRules(_block("fixed_pct", 3.0, "fixed_pct", 6.0), None, None)
    a, b = _chrom(Direction.BOTH, ra), _chrom(Direction.BOTH, rb)
    a.exit_rules = dataclasses.replace(ra, short=None)
    b.exit_rules = dataclasses.replace(rb, short=None)
    random.seed(2)
    saw_both = 0
    for _ in range(200):
        for child in crossover(a, b):
            assert child.exit_rules is not None
            if child.direction == Direction.BOTH:
                saw_both += 1
                assert child.exit_rules.short is child.exit_rules.long
            else:
                assert child.exit_rules.short is None
    assert saw_both


def test_canonical_risk_reward_under_safety_stays_none_through_operators() -> None:
    """Over many draws a safety stop never carries a risk_reward target."""
    random.seed(8)
    pool = [random_rules(Direction.BOTH, TF) for _ in range(600)]
    pool += [mutate_rules(r, 1.0, TF) for r in pool[:300]]
    blocks = [b for r in pool for b in (r.long, r.short) if b is not None]
    assert any(b.stop_type == "safety" for b in blocks)
    assert not any(b.stop_type == "safety" and b.target_type == "risk_reward" for b in blocks)


def test_rules_from_legacy_keeps_each_sides_effective_exit() -> None:
    plain = _chrom(Direction.SHORT, None, sl=4.0123, tp=6.0)
    rules = rules_from_legacy(plain)
    assert rules.long == ExitBlock("fixed_pct", 4.01, "fixed_pct", 6.0, None)
    assert rules.short is None
    assert rules.max_bars is None

    both = _chrom(Direction.BOTH, None, sl=4.0, tp=6.0)
    both.stop_loss_short_pct = 7.0
    both.take_profit_long_pct = 12.0
    rules = rules_from_legacy(both)
    assert rules.long == ExitBlock("fixed_pct", 4.0, "fixed_pct", 12.0, None)
    assert rules.short == ExitBlock("fixed_pct", 7.0, "fixed_pct", 6.0, None)


# ---------------------------------------------------------------------------
# R9: the penalty reads the rendered DSL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        # safety 50 % / target 5 % = ratio 10 -> 0.02 * (10 - 5)
        (ExitBlock("safety", None, "fixed_pct", 5.0, None), 0.1),
        # ratio 25 -> uncapped 0.4, capped
        (ExitBlock("safety", None, "fixed_pct", 2.0, None), 0.15),
        # ratio 2.5 -> within threshold
        (ExitBlock("safety", None, "fixed_pct", 20.0, None), 0.0),
        # 10 % stop / 100 % "none" target -> ratio 0.1
        (ExitBlock("fixed_pct", 10.0, "none", None, None), 0.0),
        # no percent ratio for an ATR stop, whatever the target
        (ExitBlock("atr_trailing", 2.0, "fixed_pct", 0.5, 14), 0.0),
        (ExitBlock("atr_fixed", 1.0, "risk_reward", 1.0, 7), 0.0),
    ],
)
def test_penalty_reads_rendered_dsl(block: ExitBlock, expected: float) -> None:
    chrom = _chrom(Direction.LONG, ExitRules(block, None, None), sl=1.0, tp=19.0)
    assert chromosome_sl_tp_penalty(chrom) == pytest.approx(expected, abs=1e-12)


def test_penalty_of_both_genome_scores_the_worse_side() -> None:
    fine = ExitBlock("fixed_pct", 2.0, "fixed_pct", 4.0, None)
    bad = ExitBlock("safety", None, "fixed_pct", 5.0, None)
    chrom = _chrom(Direction.BOTH, ExitRules(fine, bad, None))
    assert chromosome_sl_tp_penalty(chrom) == pytest.approx(0.1, abs=1e-12)


def test_penalty_of_legacy_genome_is_unchanged() -> None:
    chrom = _chrom(Direction.LONG, None, sl=7.0, tp=0.7)
    # 7 / 0.7 = 10 -> 0.02 * (10 - 5)
    assert chromosome_sl_tp_penalty(chrom) == pytest.approx(0.1, abs=1e-12)
