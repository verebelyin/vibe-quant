"""Per-run stop range through the GA operators (arch-c29-1 T1, no behaviour change).

``sl_range`` is a keyword-only per-run override of ``SL_RANGE``: every operator that
draws, clamps, repairs or validates a stop takes it, so a capped run draws stops in
``(SL_RANGE[0], cap)`` while an uncapped caller (``sl_range`` omitted) draws exactly
as before -- same values, same number of RNG calls. ``stop_loss_max`` is the config
knob the pipeline forwards; ``None`` (the default) leaves the range unchanged.
"""

from __future__ import annotations

import logging
import random

import pytest

from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import (
    SL_RANGE,
    TP_RANGE,
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
from vibe_quant.discovery.pipeline import LegacyGenomeOps

CAP = (0.5, 9.08)
BTC_CAP = 9.08


def _chrom(*, stop: float = 4.0, direction: Direction = Direction.SHORT) -> StrategyChromosome:
    """RSI14 crosses_below 30 / RSI14 crosses_above 60; SL/TP as given."""
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.CROSSES_BELOW, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.CROSSES_ABOVE, 60.0)],
        stop_loss_pct=stop,
        take_profit_pct=8.0,
        direction=direction,
    )


def _stops(chrom: StrategyChromosome) -> list[float]:
    """Every stop field the genome carries (generic + per-direction)."""
    return [
        value
        for value in (
            chrom.stop_loss_pct,
            chrom.stop_loss_long_pct,
            chrom.stop_loss_short_pct,
        )
        if value is not None
    ]


# ---------------------------------------------------------------------------
# R12 / R12b: the config knob and its validation
# ---------------------------------------------------------------------------


def test_config_validation() -> None:
    """R12: None / 9.08 pass; the endpoints and anything outside are refused."""
    from vibe_quant.discovery.pipeline import DiscoveryConfig

    assert DiscoveryConfig(stop_loss_max=None).stop_loss_max is None
    assert DiscoveryConfig(stop_loss_max=9.08).stop_loss_max == BTC_CAP
    for bad in (0.5, 10.0, 0.4, 11.0):
        with pytest.raises(ValueError, match=r"stop_loss_max must be in \(0\.5, 10\.0\)"):
            DiscoveryConfig(stop_loss_max=bad)


def test_config_refuses_cap_with_extended_exits() -> None:
    """R12b: an extended-exit run renders its own stops, so a stop cap is a contradiction."""
    from vibe_quant.discovery.pipeline import DiscoveryConfig

    with pytest.raises(ValueError, match="exit_gene_space 'extended' excludes stop_loss_max"):
        DiscoveryConfig(exit_gene_space="extended", stop_loss_max=BTC_CAP)


# ---------------------------------------------------------------------------
# R13: draws stay inside the cap and consume the same RNG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_draws_inside_cap_same_rng_count(seed: int) -> None:
    """R13: 1000 draws per direction are in range; the RNG state is the uncapped one's."""
    for direction in (Direction.LONG, Direction.SHORT, Direction.BOTH):
        random.seed(seed)
        capped = [
            _random_chromosome(direction_constraint=direction, sl_range=CAP) for _ in range(1000)
        ]
        capped_state = random.getstate()
        random.seed(seed)
        uncapped = [_random_chromosome(direction_constraint=direction) for _ in range(1000)]
        assert random.getstate() == capped_state

        drawn = [stop for chrom in capped for stop in _stops(chrom)]
        assert min(drawn) >= CAP[0]
        assert max(drawn) <= CAP[1]
        # Rescaled, not clamped: the whole cap is searched, so a high stop is drawn.
        assert max(drawn) > 9.0
        for chrom in capped:
            assert TP_RANGE[0] <= chrom.take_profit_pct <= TP_RANGE[1]
            for attr in ("take_profit_long_pct", "take_profit_short_pct"):
                value = getattr(chrom, attr)
                if value is not None:
                    assert TP_RANGE[0] <= value <= TP_RANGE[1]
        # The uncapped run really drew past the cap (teeth: the ranges differ).
        assert max(stop for chrom in uncapped for stop in _stops(chrom)) > 9.5


# ---------------------------------------------------------------------------
# R14 / R15: mutate and crossover stay in range via the repair clamp
# ---------------------------------------------------------------------------


def test_mutate_clamps() -> None:
    """R14: 500 mutations of a genome at the cap stay at or below it; same RNG."""
    random.seed(11)
    base = _random_chromosome(direction_constraint=Direction.BOTH)
    base.stop_loss_pct = BTC_CAP
    base.stop_loss_long_pct = BTC_CAP
    base.stop_loss_short_pct = BTC_CAP

    random.seed(7)
    capped = [mutate(base, 1.0, sl_range=CAP) for _ in range(500)]
    capped_state = random.getstate()
    random.seed(7)
    for _ in range(500):
        mutate(base, 1.0)
    assert random.getstate() == capped_state

    for chrom in capped:
        assert _stops(chrom)
        for stop in _stops(chrom):
            assert CAP[0] <= stop <= CAP[1]


def test_crossover_repairs() -> None:
    """R15: children of parents at 10.0 / 9.5 (+ per-direction 9.7 / 9.9) are repaired; same RNG."""
    parent_a = _chrom(stop=10.0)
    parent_a.stop_loss_long_pct = 9.9
    parent_a.stop_loss_short_pct = 9.7
    parent_b = _chrom(stop=9.5)
    parent_b.stop_loss_long_pct = 9.7
    parent_b.stop_loss_short_pct = 9.9

    random.seed(3)
    children = crossover(parent_a, parent_b, sl_range=CAP)
    capped_state = random.getstate()
    random.seed(3)
    crossover(parent_a, parent_b)
    assert random.getstate() == capped_state

    for child in children:
        assert _stops(child)
        for stop in _stops(child):
            assert CAP[0] <= stop <= CAP[1]
        # Both parents are above the cap on both per-direction fields, so every pick this
        # child could inherit is out of range: only the repair can put it on the cap.
        assert child.stop_loss_long_pct == BTC_CAP
        assert child.stop_loss_short_pct == BTC_CAP


# ---------------------------------------------------------------------------
# R16: validity is judged against the run range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["stop_loss_pct", "stop_loss_short_pct"])
def test_validity(field: str) -> None:
    """R16: 9.5 is valid by default and invalid at a 9.08 cap; 9.08 is the edge."""
    chrom = _chrom(stop=4.0)
    setattr(chrom, field, 9.5)
    assert is_valid_chromosome(chrom) is True
    assert is_valid_chromosome(chrom, sl_range=CAP) is False

    setattr(chrom, field, BTC_CAP)
    assert is_valid_chromosome(chrom, sl_range=CAP) is True


# ---------------------------------------------------------------------------
# R17 / R17b: warm-start seeds are clamped, never dropped
# ---------------------------------------------------------------------------


def test_seed_clamped(caplog: pytest.LogCaptureFixture) -> None:
    """R17: a seed at 10.0 / 9.9 / 9.7 is kept with every stop at 9.08 and logged once."""
    seed = _chrom(stop=10.0)
    seed.stop_loss_long_pct = 9.9
    seed.stop_loss_short_pct = 9.7

    with caplog.at_level(logging.WARNING):
        random.seed(1)
        population = initialize_population(4, seed_chromosomes=[seed], sl_range=CAP)

    assert population[0].stop_loss_pct == BTC_CAP
    assert population[0].stop_loss_long_pct == BTC_CAP
    assert population[0].stop_loss_short_pct == BTC_CAP
    clamped = [r for r in caplog.records if "clamped the stop of" in r.getMessage()]
    assert len(clamped) == 1
    assert "Warm-start: clamped the stop of 1 seed(s) to the 9.08% stop cap" in (
        clamped[0].getMessage()
    )


def test_seed_clamped_default_range(caplog: pytest.LogCaptureFixture) -> None:
    """R17b/R17b': the default range changes nothing: 10.0 kept, an impossible 10.5 skipped.

    A seed above the GA's own 10 % ceiling was never reachable; main skips it as invalid, so
    the population is main's for the same seed. Only a capped run clamps (``test_seed_clamped``).
    """
    in_range = _chrom(stop=10.0)
    with caplog.at_level(logging.WARNING):
        random.seed(1)
        population = initialize_population(4, seed_chromosomes=[in_range])
    assert population[0].stop_loss_pct == SL_RANGE[1]
    assert not [r for r in caplog.records if "clamped the stop of" in r.getMessage()]

    edge = _chrom(stop=BTC_CAP)
    with caplog.at_level(logging.WARNING):
        random.seed(1)
        population = initialize_population(4, seed_chromosomes=[edge], sl_range=CAP)
    assert population[0].stop_loss_pct == BTC_CAP
    assert not [r for r in caplog.records if "clamped the stop of" in r.getMessage()]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        random.seed(0)
        seeds = [_chrom(stop=10.0), _chrom(stop=10.5)]  # a genome's uid draws from the RNG too
        with_impossible = initialize_population(6, seed_chromosomes=seeds)
    # Captured from the pre-cap operators (86ec7125) with the same seed and seeds: the skipped
    # seed still consumed its uid draw, so the five random fills follow this exact stream.
    # Re-pinned when the context pool gained OI_CHANGE / LS_RATIO_Z, then DVOL_Z (the fills
    # draw from it).
    assert [c.stop_loss_pct for c in with_impossible] == [
        10.0, 7.4334, 8.7422, 9.6565, 7.1436, 8.5282,
    ]  # fmt: skip
    assert "skipped 1 seed(s)" in caplog.text
    assert "clamped the stop of" not in caplog.text


# ---------------------------------------------------------------------------
# R19: the rendered DSL can never sit above the cap
# ---------------------------------------------------------------------------


def test_render_never_above_cap() -> None:
    """R19: a 9.0799 stop rounds up to 9.08, never above it."""
    dsl = chromosome_to_dsl(_chrom(stop=9.0799))

    assert dsl["stop_loss"]["percent"] == BTC_CAP
    assert dsl["stop_loss"]["percent"] <= BTC_CAP


# ---------------------------------------------------------------------------
# The wiring: the ops surface forwards the run's range to every operator
# ---------------------------------------------------------------------------


@pytest.mark.simple_target(
    reason="pins the exact kwargs the ops surface forwards (R18: an uncapped run passes none)"
)
def test_ops_forward_the_run_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """An uncapped ``LegacyGenomeOps`` calls every operator exactly as before (R18)."""
    import vibe_quant.discovery.pipeline as pipeline_module

    seen: list[tuple[str, dict[str, object]]] = []

    def record(name: str):
        def wrapper(*args: object, **kwargs: object) -> object:
            seen.append((name, kwargs))
            return None

        return wrapper

    monkeypatch.setattr(pipeline_module, "_random_chromosome", record("random"))
    monkeypatch.setattr(pipeline_module, "mutate", record("mutate"))
    monkeypatch.setattr(pipeline_module, "crossover", record("crossover"))
    monkeypatch.setattr(pipeline_module, "is_valid_chromosome", record("is_valid"))
    monkeypatch.setattr(pipeline_module, "initialize_population", record("initialize"))

    uncapped = LegacyGenomeOps(lambda: Direction.LONG)
    capped = LegacyGenomeOps(lambda: Direction.LONG, stop_loss_max=BTC_CAP)
    extended = LegacyGenomeOps(lambda: Direction.LONG, "4h", BTC_CAP)
    for ops in (uncapped, capped, extended):
        ops.random()
        ops.mutate(_chrom(), 0.1)
        ops.crossover(_chrom(), _chrom())
        ops.is_valid(_chrom())
        ops.initialize(2)

    assert capped.sl_range == CAP
    assert uncapped.sl_range == SL_RANGE
    assert seen[0][1] == {"direction_constraint": Direction.LONG}
    assert seen[1][1] == {}
    assert seen[2][1] == {}
    assert seen[3][1] == {}
    assert seen[4][1] == {"direction_constraint": Direction.LONG, "seed_chromosomes": None}
    assert seen[5][1] == {"direction_constraint": Direction.LONG, "sl_range": CAP}
    assert seen[6][1] == {"sl_range": CAP}
    assert seen[7][1] == {"sl_range": CAP}
    assert seen[8][1] == {"sl_range": CAP}
    assert seen[9][1] == {
        "direction_constraint": Direction.LONG,
        "seed_chromosomes": None,
        "sl_range": CAP,
    }
    # An extended run: exit_timeframe reaches random/mutate/initialize ONLY.
    assert seen[10][1] == {
        "direction_constraint": Direction.LONG,
        "exit_timeframe": "4h",
        "sl_range": CAP,
    }
    assert seen[11][1] == {"exit_timeframe": "4h", "sl_range": CAP}
    assert seen[12][1] == {"sl_range": CAP}
    assert seen[13][1] == {"sl_range": CAP}
    assert seen[14][1] == {
        "direction_constraint": Direction.LONG,
        "seed_chromosomes": None,
        "exit_timeframe": "4h",
        "sl_range": CAP,
    }
