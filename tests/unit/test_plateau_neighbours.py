"""Plateau neighbours (fitness v2): one deterministic +-1-step neighbour per gene coordinate.

The generator must never draw the GA's global RNG: it builds candidates in a fixed
coordinate order and, when there are more than eight, picks eight with a
``random.Random`` seeded from the chromosome's own DSL key.

Extended genomes: ``exit_rules`` is not a plateau dimension, so every neighbour keeps
its parent's rules object unchanged (``exit_genes`` are copied, never stepped).
"""

from __future__ import annotations

import hashlib
import random

import pytest

from vibe_quant.discovery.exit_genes import ExitRules, rules_from_legacy
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    PriceVsMAConditionGene,
    StrategyChromosome,
    StrategyGene,
    _plateau_candidates,
    _random_chromosome,
    plateau_neighbours,
)
from vibe_quant.dsl.identity import dsl_body_key

PLANTED_KEY = (
    '{"entry_conditions":{"long":["rsi_entry_0 crosses_below 30"]},'
    '"exit_conditions":{"long":["rsi_exit_0 crosses_above 60"]},'
    '"indicators":{"rsi_entry_0":{"period":14,"type":"RSI"},'
    '"rsi_exit_0":{"period":14,"type":"RSI"}},'
    '"stop_loss":{"percent":4.0,"type":"fixed_pct"},'
    '"take_profit":{"percent":8.0,"type":"fixed_pct"},"timeframe":"5m"}'
)


def _key(chrom: StrategyChromosome) -> str:
    return dsl_body_key(chromosome_to_dsl(chrom))


def _rsi(period: float, condition: ConditionType, threshold: float) -> StrategyGene:
    return StrategyGene("RSI", {"period": period}, condition, threshold)


def _planted() -> StrategyChromosome:
    """RSI14 crosses_below 30 entry / RSI14 crosses_above 60 exit, SL 4, TP 8, LONG."""
    return StrategyChromosome(
        entry_genes=[_rsi(14.0, ConditionType.CROSSES_BELOW, 30.0)],
        exit_genes=[_rsi(14.0, ConditionType.CROSSES_ABOVE, 60.0)],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.LONG,
    )


def _coords(
    cands: list[StrategyChromosome],
) -> list[tuple[float, float, float, float, float, float]]:
    """(entry period, entry threshold, exit period, exit threshold, SL, TP) per candidate."""
    return [
        (
            c.entry_genes[0].parameters["period"],
            c.entry_genes[0].threshold,
            c.exit_genes[0].parameters["period"],
            c.exit_genes[0].threshold,
            c.stop_loss_pct,
            c.take_profit_pct,
        )
        for c in cands
    ]


def test_candidate_order_is_exact() -> None:
    """R17: 12 candidates in coordinate order, +-1 step of 5% of each range."""
    cands = _plateau_candidates(_planted(), _key)

    assert _coords(cands) == [
        (12.0, 30.0, 14.0, 60.0, 4.0, 8.0),  # entry period -2
        (16.0, 30.0, 14.0, 60.0, 4.0, 8.0),  # entry period +2
        (14.0, 27.5, 14.0, 60.0, 4.0, 8.0),  # entry threshold -2.5
        (14.0, 32.5, 14.0, 60.0, 4.0, 8.0),  # entry threshold +2.5
        (14.0, 30.0, 12.0, 60.0, 4.0, 8.0),  # exit period -2
        (14.0, 30.0, 16.0, 60.0, 4.0, 8.0),  # exit period +2
        (14.0, 30.0, 14.0, 57.5, 4.0, 8.0),  # exit threshold -2.5
        (14.0, 30.0, 14.0, 62.5, 4.0, 8.0),  # exit threshold +2.5
        (14.0, 30.0, 14.0, 60.0, 3.525, 8.0),  # SL -0.475
        (14.0, 30.0, 14.0, 60.0, 4.475, 8.0),  # SL +0.475
        (14.0, 30.0, 14.0, 60.0, 4.0, 7.025),  # TP -0.975
        (14.0, 30.0, 14.0, 60.0, 4.0, 8.975),  # TP +0.975
    ]
    assert len({_key(c) for c in cands}) == 12


def test_default_four_are_the_design_order_truncated() -> None:
    """Q20=B: the default returns the first four of the seeded eight, pinned."""
    chrom = _planted()
    assert _key(chrom) == PLANTED_KEY

    got = plateau_neighbours(chrom, _key)

    assert _coords(got) == [
        (14.0, 30.0, 14.0, 60.0, 4.0, 7.025),
        (14.0, 30.0, 12.0, 60.0, 4.0, 8.0),
        (14.0, 30.0, 14.0, 60.0, 3.525, 8.0),
        (16.0, 30.0, 14.0, 60.0, 4.0, 8.0),
    ]
    assert [c.uid for c in got] == [
        "5b6c9a04c322",
        "549cee24c9ea",
        "14c7b7a1757b",
        "2cd4c405a05c",
    ]


def test_eight_neighbours_pinned() -> None:
    """R17: 12 candidates -> sample of 8 fixed by sha256 of the own key."""
    got = plateau_neighbours(_planted(), _key, max_neighbours=8)

    assert _coords(got) == [
        (14.0, 30.0, 14.0, 60.0, 4.0, 7.025),
        (14.0, 30.0, 12.0, 60.0, 4.0, 8.0),
        (14.0, 30.0, 14.0, 60.0, 3.525, 8.0),
        (16.0, 30.0, 14.0, 60.0, 4.0, 8.0),
        (14.0, 27.5, 14.0, 60.0, 4.0, 8.0),
        (12.0, 30.0, 14.0, 60.0, 4.0, 8.0),
        (14.0, 30.0, 14.0, 62.5, 4.0, 8.0),
        (14.0, 30.0, 16.0, 60.0, 4.0, 8.0),
    ]


def test_never_more_than_eight() -> None:
    assert len(plateau_neighbours(_planted(), _key, max_neighbours=20)) == 8
    assert plateau_neighbours(_planted(), _key, max_neighbours=0) == []


def test_neighbours_are_deterministic() -> None:
    first = [(c.uid, _key(c)) for c in plateau_neighbours(_planted(), _key, max_neighbours=8)]
    second = [(c.uid, _key(c)) for c in plateau_neighbours(_planted(), _key, max_neighbours=8)]
    assert first == second


def test_global_rng_state_untouched() -> None:
    """No global RNG draw: the GA's seeded stream must not move."""
    chrom = _planted()  # builds its uid from the global RNG; outside the measured region
    random.seed(20261123)
    before = random.getstate()

    plateau_neighbours(chrom, _key, max_neighbours=8)

    assert random.getstate() == before


def test_step_out_of_range_is_skipped_never_clamped() -> None:
    """RSI period 49: +2 would be 51 > 50, so only 47 exists -- never 50 or 51."""
    chrom = StrategyChromosome(
        entry_genes=[_rsi(49.0, ConditionType.LT, 30.0)],
        exit_genes=[_rsi(14.0, ConditionType.GT, 60.0)],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.LONG,
    )

    cands = _plateau_candidates(chrom, _key)

    assert [c.entry_genes[0].parameters["period"] for c in cands].count(47.0) == 1
    assert all(c.entry_genes[0].parameters["period"] != 51.0 for c in cands)
    assert all(c.entry_genes[0].parameters["period"] != 50.0 for c in cands)


def test_macd_fast_step_across_slow_is_skipped() -> None:
    """MACD fast 20 / slow 21: fast +1 = 21 is invalid, so it is not a neighbour."""
    chrom = StrategyChromosome(
        entry_genes=[
            StrategyGene(
                "MACD",
                {"fast_period": 20.0, "slow_period": 21.0, "signal_period": 9.0},
                ConditionType.GT,
                0.0,
            )
        ],
        exit_genes=[_rsi(14.0, ConditionType.GT, 60.0)],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.LONG,
    )

    cands = _plateau_candidates(chrom, _key)

    assert all(
        c.entry_genes[0].parameters["fast_period"] < c.entry_genes[0].parameters["slow_period"]
        for c in cands
    )
    assert all(c.entry_genes[0].parameters["fast_period"] != 21.0 for c in cands)


def test_ma_cross_step_reordered_is_skipped() -> None:
    """A step that canonicalization reorders (fast >= slow) moves another coordinate: skip."""
    chrom = StrategyChromosome(
        entry_genes=[_rsi(14.0, ConditionType.LT, 30.0)],
        exit_genes=[_rsi(14.0, ConditionType.GT, 60.0)],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.LONG,
        ma_entry_genes=[
            PriceVsMAConditionGene(
                "KAMA", {"period": 10.0}, ConditionType.GT, parameters_slow={"period": 11.0}
            )
        ],
    )

    cands = _plateau_candidates(chrom, _key)

    for cand in cands:
        gene = cand.ma_entry_genes[0]
        assert gene.parameters["period"] < gene.parameters_slow["period"]
        assert gene.parameters["period"] != 12.0


def test_dead_param_step_dropped_as_own_dsl() -> None:
    """STOCH period_d is dead without sub_value='d': stepping it yields the own DSL, dropped."""

    def stoch(sub_value: str | None) -> StrategyChromosome:
        return StrategyChromosome(
            entry_genes=[
                StrategyGene(
                    "STOCH",
                    {"period_k": 14.0, "period_d": 3.0},
                    ConditionType.LT,
                    30.0,
                    sub_value,
                )
            ],
            exit_genes=[_rsi(14.0, ConditionType.GT, 60.0)],
            stop_loss_pct=4.0,
            take_profit_pct=8.0,
            direction=Direction.LONG,
        )

    dead = _plateau_candidates(stoch(None), _key)
    live = _plateau_candidates(stoch("d"), _key)

    assert all(_key(c) != _key(stoch(None)) for c in dead)
    assert all(c.entry_genes[0].parameters["period_d"] == 3.0 for c in dead)
    assert len(live) == len(dead) + 1  # only period_d +1 = 4 is in range


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
def test_exit_rules_copied_unchanged(direction: Direction) -> None:
    """exit_rules is not a plateau dimension: every neighbour keeps its parent's rules."""
    rules: ExitRules = rules_from_legacy(_planted())
    chrom = _planted()
    chrom.direction = direction
    chrom.exit_rules = rules

    neighbours = plateau_neighbours(chrom, _key, max_neighbours=8)

    assert neighbours
    for neighbour in neighbours:
        assert neighbour.exit_rules is rules
    assert all(neighbour.uid != chrom.uid for neighbour in neighbours)


def test_uid_is_sha256_of_neighbour_key() -> None:
    """The uid is forwarded from key_fn, over two unrelated genomes."""
    planted = _planted()
    other = StrategyChromosome(
        entry_genes=[_rsi(20.0, ConditionType.GT, 70.0)],
        exit_genes=[_rsi(8.0, ConditionType.LT, 40.0)],
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        direction=Direction.SHORT,
    )

    for chrom in (planted, other):
        for neighbour in plateau_neighbours(chrom, _key, max_neighbours=8):
            assert neighbour.uid == hashlib.sha256(_key(neighbour).encode()).hexdigest()[:12]


def test_stops_respect_the_run_range() -> None:
    """R15b: a capped run never steps a stop past the cap; the step itself is unchanged."""
    random.seed(0)
    chrom = _random_chromosome(direction_constraint=Direction.SHORT)
    chrom.stop_loss_pct = 8.46

    default = [n.stop_loss_pct for n in plateau_neighbours(chrom, _key, max_neighbours=8)]
    assert 8.935 in default  # cap + one 0.475 step (5% of the SL_RANGE span)

    capped = [
        n.stop_loss_pct
        for n in plateau_neighbours(chrom, _key, max_neighbours=8, sl_range=(0.5, 8.46))
    ]
    assert max(capped) <= 8.46
    assert 8.935 not in capped
    assert 7.985 in capped  # cap - one step, kept

    # A range that excludes nothing returns today's list, byte for byte.
    wider = [
        n.stop_loss_pct
        for n in plateau_neighbours(chrom, _key, max_neighbours=8, sl_range=(0.5, 8.98))
    ]
    assert wider == default
