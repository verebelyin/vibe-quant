"""Legacy GA machinery under a fixed seed: populations and RNG stream are pinned.

A seeded legacy run must keep producing the same genomes AND leave the global
``random`` generator in the same state, so routing genome access through
type-dispatched accessors cannot add or drop a draw on the legacy path. The
literal hashes were captured on the base before any accessor work.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery.genome import chromosome_to_serializable
from vibe_quant.discovery.mock_backtest import mock_backtest
from vibe_quant.discovery.operators import (
    Direction,
    StrategyChromosome,
    _ensure_pool,
    _random_chromosome,
    initialize_population,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

if TYPE_CHECKING:
    from collections.abc import Iterator

GOLDEN: dict[str, tuple[str, str]] = {
    "tournament": (
        "19a6c61763e64181fbd318a3a49f11a29533e1ccafcd0f7e1fc4a203fa3d3394",
        "15dc42be4528ad1dfb02550f628c7949c9cac6cf345710b702d24208b799011f",
    ),
    "crowding": (
        "0bc623fa6df1bfe6d2757636ea89d818042477644264c65cf0322391ca9ea68b",
        "d87adeff8bc70108845b7e528a4f3ea2870e41b52247248c3128d08ab4c99f66",
    ),
    "immigrants": (
        "7d41b9105c23eccd59115daae0e12bc447060a8beb78b90c567d34a5d981fbb2",
        "173dd81afbe0f7c461995bbe31598d74c7087d7229fc6d81c991c2efa54070eb",
    ),
    "fallback": (
        "6fe5962b6cc7df293b1a1e017feafa04f5d77fda7ce51bbfa627238d2b06f002",
        "16d930a98ffd24e7b66cc72514b13add6ba0b678e16a0780f77df976a76a81c9",
    ),
    "warm_start": (
        "38f287c93fb8107bad1ccbf132841735559d39383e8c2e2928c2ebf49b99d356",
        "933a1de65d9b3cffd2f8fa364c9b60d835c28c03af049105d34dd91cc2cb3ead",
    ),
}


@pytest.fixture(autouse=True)
def _restore_global_random() -> Iterator[None]:
    _ensure_pool()  # lazy pool build must not happen inside a seeded section
    saved = random.getstate()
    try:
        yield
    finally:
        random.setstate(saved)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _population_hash(populations: list[list[StrategyChromosome]]) -> str:
    payload = [
        [{"uid": c.uid, **chromosome_to_serializable(c)} for c in population]
        for population in populations
    ]
    return _digest(json.dumps(payload, sort_keys=True, default=str))


def _rng_hash() -> str:
    return _digest(repr(random.getstate()))


def _config(**overrides: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "population_size": 8,
        "max_generations": 4,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "elite_count": 2,
        "tournament_size": 3,
        "convergence_generations": 50,
        "top_k": 3,
        "min_trades": 50,
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "max_workers": None,
        "require_dsr": False,
        "eval_windows": 1,
        "entropy_threshold": 0.0,
    }
    base.update(overrides)
    return DiscoveryConfig(**base)


def _seeded_run(
    config: DiscoveryConfig,
    seeds: list[StrategyChromosome] | None = None,
) -> tuple[str, str]:
    """Run the pipeline seeded; hash every generation's population and the RNG."""
    populations: list[list[StrategyChromosome]] = []
    pipe = DiscoveryPipeline(config, mock_backtest, seed_chromosomes=seeds)
    real_evolve = pipe._evolve_generation

    def capture(population: list[StrategyChromosome], fitness_results: Any) -> Any:
        populations.append([c.clone() for c in population])
        return real_evolve(population, fitness_results)

    pipe._evolve_generation = capture  # type: ignore[method-assign]
    random.seed(5)
    result = pipe.run()
    populations.append([g.best_chromosome for g in result.generations])
    return _population_hash(populations), _rng_hash()


def test_tournament_run_is_pinned() -> None:
    assert _seeded_run(_config(use_crowding=False)) == GOLDEN["tournament"]


def test_crowding_run_is_pinned() -> None:
    assert _seeded_run(_config(use_crowding=True)) == GOLDEN["crowding"]


def test_immigrant_injection_is_pinned() -> None:
    random.seed(11)
    monoculture = [_random_chromosome(direction_constraint=Direction.LONG) for _ in range(8)]
    for chrom in monoculture:
        chrom.entry_genes = monoculture[0].entry_genes[:1]
    pipe = DiscoveryPipeline(_config(entropy_threshold=1.1, immigrant_fraction=0.5), mock_backtest)
    known: list[Any] = [None] * len(monoculture)
    new_pop, new_known = pipe._maybe_inject_immigrants(monoculture, known)
    assert (_population_hash([new_pop]), _rng_hash()) == GOLDEN["immigrants"]
    assert sum(a is not b for a, b in zip(monoculture, new_pop, strict=True)) == 4
    assert len(new_known) == len(new_pop)


def test_random_fallback_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    import vibe_quant.discovery.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "is_valid_chromosome", lambda _c: False)
    pipe = DiscoveryPipeline(_config(), mock_backtest)
    random.seed(23)
    parents = [_random_chromosome(direction_constraint=Direction.SHORT) for _ in range(2)]
    pipe._direction_constraint = Direction.SHORT
    children, retries, fallbacks = pipe._breed(parents[0], parents[1])
    assert (retries, fallbacks) == (20, 2)
    assert (_population_hash([children]), _rng_hash()) == GOLDEN["fallback"]


def test_warm_start_is_pinned() -> None:
    random.seed(3)
    seeds = [_random_chromosome(direction_constraint=Direction.LONG) for _ in range(4)]
    population = initialize_population(
        8, direction_constraint=Direction.LONG, seed_chromosomes=seeds
    )
    start_pop, start_rng = _population_hash([population]), _rng_hash()
    run_pop, run_rng = _seeded_run(_config(use_crowding=True, direction="long"), seeds)
    assert (_digest(start_pop + run_pop), _digest(start_rng + run_rng)) == GOLDEN["warm_start"]
