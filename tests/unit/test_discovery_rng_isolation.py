"""Global-random isolation for discovery fitness evaluation.

NautilusTrader's ``FillModel`` constructor reseeds the GLOBAL ``random`` module
(``nautilus_trader/backtest/models/fill.pyx``), so an in-process NT backtest
during a sequential GA evaluation resets the GA's seeded RNG
(vibe-quant-0bwe2, vibe-quant-qei3p).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery import fitness
from vibe_quant.discovery.fitness import evaluate_population, preserve_global_random
from vibe_quant.discovery.operators import StrategyChromosome, initialize_population
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def _restore_entry_global_random() -> Iterator[None]:
    """Hand the process-global ``random`` back to the next test untouched."""
    entry = random.getstate()
    try:
        yield
    finally:
        random.setstate(entry)


# ---------------------------------------------------------------------------
# Backtest functions (module level: picklable by the spawn start method)
# ---------------------------------------------------------------------------


def _mock_metrics(chrom: StrategyChromosome) -> dict[str, Any]:
    n_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
    total_return = 0.5 + n_genes * 0.05
    total_trades = 150
    r = random.Random(n_genes)
    mean_ret = total_return / total_trades
    trade_returns = tuple(r.gauss(mean_ret, mean_ret * 0.3) for _ in range(total_trades))
    return {
        "sharpe_ratio": 2.0 + n_genes * 0.1,
        "max_drawdown": 0.08,
        "profit_factor": 2.0,
        "total_trades": total_trades,
        "total_return": total_return,
        "trade_returns": trade_returns,
    }


def _pure_bt(chrom: StrategyChromosome) -> dict[str, Any]:
    """Deterministic metrics; its own ``Random`` only, no global draws."""
    return _mock_metrics(chrom)


def _reseeding_bt(chrom: StrategyChromosome) -> dict[str, Any]:
    """Mimics NT's FillModel: reseeds the GLOBAL random, then returns metrics."""
    random.seed(42)
    random.random()
    return _mock_metrics(chrom)


def _raising_bt(chrom: StrategyChromosome) -> dict[str, Any]:
    """Reseeds the global random and then fails like missing data does."""
    random.seed(7)
    random.random()
    raise DataUnavailableError("no bars")


# ---------------------------------------------------------------------------
# The context manager itself
# ---------------------------------------------------------------------------


def test_preserve_global_random_restores_state() -> None:
    random.seed(1234)
    before = random.getstate()
    with preserve_global_random():
        random.seed(42)
        random.random()
    assert random.getstate() == before


def test_preserve_global_random_restores_state_on_exception() -> None:
    random.seed(99)
    before = random.getstate()
    with pytest.raises(RuntimeError, match="boom"), preserve_global_random():
        random.seed(1)
        raise RuntimeError("boom")
    assert random.getstate() == before


# ---------------------------------------------------------------------------
# Contract rows R26-R29
# ---------------------------------------------------------------------------


def test_sequential_eval_restores_global_random() -> None:
    """R26: a reseeding backtest fn cannot leak out of a sequential batch."""
    chroms = initialize_population(2)
    random.seed(5)
    state = random.getstate()

    results = evaluate_population(chroms, _reseeding_bt, max_workers=None)

    assert len(results) == 2
    assert random.getstate() == state


def test_restore_on_data_unavailable() -> None:
    """R27: the ``finally`` restores the state when the batch raises."""
    chroms = initialize_population(2)
    random.seed(11)
    state = random.getstate()

    with pytest.raises(DataUnavailableError):
        evaluate_population(chroms, _raising_bt, max_workers=None)

    assert random.getstate() == state


def test_parallel_fallback_restores_global_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """R28: the ``_parallel_broken`` fallback runs the same guarded tail."""
    monkeypatch.setattr(fitness, "_parallel_broken", True)
    chroms = initialize_population(2)
    random.seed(13)
    state = random.getstate()

    results = evaluate_population(chroms, _reseeding_bt, max_workers=2)

    assert len(results) == 2
    assert random.getstate() == state


def _make_config(**overrides: Any) -> DiscoveryConfig:
    defaults: dict[str, Any] = {
        "population_size": 6,
        "max_generations": 3,
        "mutation_rate": 0.1,
        "elite_count": 1,
        "tournament_size": 3,
        "convergence_generations": 1,
        "top_k": 2,
        "min_trades": 50,
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "max_workers": 1,
        "require_dsr": False,
    }
    defaults.update(overrides)
    return DiscoveryConfig(**defaults)


def _seeded_run(
    backtest_fn: Callable[[StrategyChromosome], dict[str, Any]],
    max_workers: int | None,
) -> tuple[list[str], list[str]]:
    """One seeded run's observable trajectory: cache order + per-gen best uid."""
    random.seed(5)
    pipe = DiscoveryPipeline(_make_config(max_workers=max_workers), backtest_fn)
    result = pipe.run()
    best_uids = [generation.best_chromosome.uid for generation in result.generations]
    return list(pipe._fitness_cache), best_uids


def test_seeded_genomes_equal_across_worker_counts() -> None:
    """R29: reseeding NT backtests cannot change a seeded sequential trajectory."""
    saved = random.getstate()
    try:
        pure = _seeded_run(_pure_bt, None)
        reseeded_sequential = _seeded_run(_reseeding_bt, None)
        reseeded_parallel = _seeded_run(_reseeding_bt, 2)
    finally:
        random.setstate(saved)

    assert pure[0]
    assert reseeded_sequential == pure
    assert reseeded_parallel == pure


# ---------------------------------------------------------------------------
# Isolation of the tests themselves
# ---------------------------------------------------------------------------


_OBSERVED_TESTS = (
    "test_preserve_global_random_restores_state",
    "test_preserve_global_random_restores_state_on_exception",
    "test_sequential_eval_restores_global_random",
    "test_restore_on_data_unavailable",
    "test_parallel_fallback_restores_global_random",
)


class _GlobalRandomObserver:
    """Records every test whose session leaves the process-global ``random`` state moved."""

    def __init__(self) -> None:
        self.leaked: list[str] = []

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> Any:
        entry = random.getstate()
        result = yield
        if random.getstate() != entry:
            self.leaked.append(item.name)
        return result


def test_module_tests_restore_entry_global_random() -> None:
    """Each test below must hand the process-global ``random`` back as it received it."""
    node_ids = [f"{Path(__file__)}::{name}" for name in _OBSERVED_TESTS]
    observer = _GlobalRandomObserver()
    exit_code = pytest.main(
        ["-q", "-o", "addopts=", "-p", "no:cacheprovider", *node_ids],
        plugins=[observer],
    )
    assert exit_code == pytest.ExitCode.OK
    assert observer.leaked == []
