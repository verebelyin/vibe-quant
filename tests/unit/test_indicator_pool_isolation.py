"""The GA indicator pool must not leak out of a test that runs the pipeline.

``DiscoveryPipeline._apply_indicator_pool_filter`` filters the module globals
``operators.INDICATOR_POOL`` / ``_INDICATOR_NAMES`` in place, and ``_ensure_pool``
only rebuilds an EMPTY pool -- so before the autouse
``_restore_ga_indicator_pool`` fixture in tests/unit/conftest.py, any test that ran
the pipeline left the context specs (BTC_TREND, ...) deleted for every later test
on the same xdist worker (bead vibe-quant-73khn).

One self-contained test drives the fixture's own body (``restore_ga_indicator_pool``)
around the real ``run()`` path: an ordered pair of tests broke when xdist worksteal
split the module across workers (vibe-quant-1xhri).
"""

from __future__ import annotations

from tests.unit.conftest import restore_ga_indicator_pool
from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
from vibe_quant.discovery.genome import build_indicator_pool
from vibe_quant.discovery.operators import (
    _INDICATOR_NAMES,
    INDICATOR_POOL,
    _ensure_pool,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline


def test_pipeline_run_filters_the_pool_and_the_restore_undoes_it() -> None:
    _ensure_pool()
    assert "BTC_TREND" in INDICATOR_POOL, "needs_context spec missing from the real pool"
    assert "BTC_TREND" in build_indicator_pool(include_context=True)
    before = sorted(INDICATOR_POOL)

    with restore_ga_indicator_pool():
        cfg = _make_config(population_size=6, max_generations=2, elite_count=1)
        DiscoveryPipeline(cfg, _mock_backtest).run()
        # the documented leak: run() mutates the module globals in place
        assert "BTC_TREND" not in INDICATOR_POOL, "the pool filter no longer mutates in place"
        assert "BTC_TREND" not in _INDICATOR_NAMES

    # Checked BEFORE ``_ensure_pool()``: that call only EXTENDS an EMPTY names list, so it
    # repairs nothing here. The two globals are filtered independently, so a restore that
    # fixes ``INDICATOR_POOL`` but not ``_INDICATOR_NAMES`` (or re-extends the saved names on
    # top of the filtered list) leaves them disagreeing, and this is what catches it.
    assert sorted(_INDICATOR_NAMES) == sorted(INDICATOR_POOL), (
        "_INDICATOR_NAMES not restored alongside INDICATOR_POOL"
    )
    _ensure_pool()
    assert "BTC_TREND" in INDICATOR_POOL, "INDICATOR_POOL leaked out of the run"
    assert "BTC_TREND" in _INDICATOR_NAMES, "_INDICATOR_NAMES leaked out of the run"
    assert sorted(INDICATOR_POOL) == before, "pool not restored to its prior state"


def test_restore_runs_even_when_the_body_raises() -> None:
    _ensure_pool()
    before = sorted(INDICATOR_POOL)
    try:
        with restore_ga_indicator_pool():
            INDICATOR_POOL.pop("BTC_TREND")
            raise RuntimeError("test body failed")
    except RuntimeError:
        pass
    assert sorted(INDICATOR_POOL) == before
