"""The GA indicator pool must not leak out of a test that runs the pipeline.

``DiscoveryPipeline._apply_indicator_pool_filter`` filters the module globals
``operators.INDICATOR_POOL`` / ``_INDICATOR_NAMES`` in place, and ``_ensure_pool``
only rebuilds an EMPTY pool -- so before the autouse
``_restore_ga_indicator_pool`` fixture in tests/unit/conftest.py, any test that ran
the pipeline left the context specs (BTC_TREND, ...) deleted for every later test
on the same xdist worker (bead vibe-quant-73khn).

The two tests below are an ordered pair inside one module: the first drives the
real ``run()`` path and records the pool it filtered, the second asserts the next
test starts from that same, unfiltered pool.
"""

from __future__ import annotations

from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
from vibe_quant.discovery.genome import build_indicator_pool
from vibe_quant.discovery.operators import (
    _INDICATOR_NAMES,
    INDICATOR_POOL,
    _ensure_pool,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline

#: Sorted pool keys the filtering test built, i.e. the state the next test must see.
_POOL_BEFORE_FILTER: list[str] | None = None


def test_pipeline_run_leaves_the_shared_pool_filtered() -> None:
    """The documented leak: run() mutates the module globals for good."""
    global _POOL_BEFORE_FILTER

    _ensure_pool()
    assert "BTC_TREND" in INDICATOR_POOL, "needs_context spec missing from the real pool"
    _POOL_BEFORE_FILTER = sorted(INDICATOR_POOL)

    cfg = _make_config(population_size=6, max_generations=2, elite_count=1)
    DiscoveryPipeline(cfg, _mock_backtest).run()

    assert "BTC_TREND" not in INDICATOR_POOL, "the pool filter no longer mutates in place"
    assert "BTC_TREND" not in _INDICATOR_NAMES


def test_pool_and_names_are_unfiltered_for_the_next_test() -> None:
    """The regression: the next test starts from the unfiltered pool again.

    ``_ensure_pool`` is called first because the pool is a lazily-built cache --
    a test that starts from an unbuilt pool must still end up with the full one,
    never with whatever the previous test's ``run()`` left behind.
    """
    assert _POOL_BEFORE_FILTER is not None, (
        "ordered pair -- run this module, not a single test, so the leak test has run"
    )
    # BTC_TREND is a needs_context spec, i.e. exactly a name the default pool filter drops.
    assert "BTC_TREND" in build_indicator_pool(include_context=True)

    # Checked BEFORE ``_ensure_pool()``: that call only EXTENDS an EMPTY names list, so it
    # repairs nothing here. The two module globals are filtered independently, so a fixture
    # that restores ``INDICATOR_POOL`` but not ``_INDICATOR_NAMES`` (or re-extends the saved
    # names on top of the filtered list) leaves them disagreeing and this is what catches it.
    assert sorted(_INDICATOR_NAMES) == sorted(INDICATOR_POOL), (
        "_INDICATOR_NAMES not restored alongside INDICATOR_POOL"
    )

    _ensure_pool()

    assert "BTC_TREND" in INDICATOR_POOL, "INDICATOR_POOL leaked between tests"
    assert "BTC_TREND" in _INDICATOR_NAMES, "_INDICATOR_NAMES leaked between tests"
    assert sorted(INDICATOR_POOL) == _POOL_BEFORE_FILTER, "pool not restored to its prior state"
