"""Reproducibility guard: the screening fill model must be seeded.

NautilusTrader's ``FillModel`` is unseeded by default. With ``prob_slippage > 0``
(screening uses 0.5) that makes every fill's 1-tick slippage a fresh random draw,
so two *identical* screening backtests -- and the GA fitness evals that share the
exact same code path -- disagree in the low-order digits run-to-run. That was the
secondary cause behind the run-812 VIDYA replay drift (bd vibe-quant-1gvyc): the
dominant cause was the multi-window fitness aggregate vs single-range replay metric
mismatch, but on top of it the unseeded fill model added genuine non-determinism.

The fix pins ``random_seed`` on the screening fill model (``SCREENING_FILL_SEED``)
and propagates it all the way into the ``ImportableFillModelConfig`` that
``BacktestVenueConfig`` hands to the engine. These tests lock that wiring so a
regression that drops the seed (anywhere along the path) flips them red.
"""

from __future__ import annotations

from vibe_quant.validation.fill_model import (
    ScreeningFillModelConfig,
    create_screening_fill_model,
)
from vibe_quant.validation.venue import (
    SCREENING_FILL_SEED,
    create_backtest_venue_config,
    create_venue_config_for_screening,
)


def _slip_sequence(seed: int | None, n: int = 32) -> list[bool]:
    """Draw ``n`` slippage decisions from a freshly-built screening fill model."""
    model = create_screening_fill_model(
        ScreeningFillModelConfig(prob_slippage=0.5, random_seed=seed)
    )
    return [model.is_slipped() for _ in range(n)]


def test_seeded_fill_model_is_reproducible() -> None:
    """Same seed -> identical slippage draws; this is the guarantee we lock."""
    assert _slip_sequence(SCREENING_FILL_SEED) == _slip_sequence(SCREENING_FILL_SEED)


def test_seed_actually_drives_the_rng() -> None:
    """Different seeds diverge -- proves the seed reaches the RNG, not a no-op.

    With ``prob_slippage=0.5`` over 32 draws the chance two seeds coincide on every
    draw is ~2**-32, so this is not flaky.
    """
    assert _slip_sequence(1) != _slip_sequence(2)


def test_config_default_seed_is_none() -> None:
    """Direct constructors are unchanged (backward compatible) -- only the screening
    venue opts into a fixed seed."""
    assert ScreeningFillModelConfig().random_seed is None


def test_screening_venue_config_is_seeded() -> None:
    """The screening venue pins the fill seed by default."""
    venue = create_venue_config_for_screening()
    assert isinstance(venue.fill_config, ScreeningFillModelConfig)
    assert venue.fill_config.random_seed == SCREENING_FILL_SEED


def test_seed_propagates_into_importable_fill_config() -> None:
    """End-to-end: the seed reaches the dict NT uses to build the engine FillModel.

    ``create_backtest_venue_config`` is the exact public path ``NTScreeningRunner``
    takes, so asserting on its ``ImportableFillModelConfig.config`` locks the whole
    production wiring (config field -> venue config -> importable config)."""
    bt_venue = create_backtest_venue_config(create_venue_config_for_screening())
    assert bt_venue.fill_model is not None
    assert bt_venue.fill_model.config["random_seed"] == SCREENING_FILL_SEED
