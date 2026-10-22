"""The shared engine-run lifecycle: one backtest node, built/run/torn down once.

Bead vibe-quant-rabmo.4. Screening (``NTScreeningRunner._run_backtest``) and
validation (``ValidationRunner._run_backtest``) built the same node, registered the
same kind of statistics on it and tore it down the same way; the per-tier differences
are the values in :class:`EnginePolicy` and what the consumer does with the run
(metrics, margin-refusal reads, the engine-not-found decision).

Deliberately free of module-level ``nautilus_trader`` imports (R-iso): the API imports
:func:`loaded_timeframes` from here, and a mode-off run must stay light. The engine
stack is imported inside the functions that use it.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from nautilus_trader.config import BacktestDataConfig

    from vibe_quant.validation.venue import VenueConfig


@dataclass(frozen=True)
class EnginePolicy:
    """The two per-tier engine values: portfolio statistics and NT log verbosity."""

    statistics: tuple[str, ...]
    log_level_env: str
    log_level_default: str


#: Screening registers four statistics. MaxDrawdown excluded on this tier.
SCREENING_ENGINE = EnginePolicy(
    ("SharpeRatio", "SortinoRatio", "WinRate", "ProfitFactor"),
    "VIBE_QUANT_NT_LOG_LEVEL_SCREENING",
    "WARNING",
)

#: Validation registers nine (MaxDrawdown included: NT >= 1.223 has it).
VALIDATION_ENGINE = EnginePolicy(
    (
        "SharpeRatio",
        "SortinoRatio",
        "WinRate",
        "ProfitFactor",
        "Expectancy",
        "LongRatio",
        "AvgWinner",
        "AvgLoser",
        "MaxDrawdown",
    ),
    "VIBE_QUANT_NT_LOG_LEVEL",
    "INFO",
)


@dataclass(frozen=True)
class EngineSpec:
    """Everything one run needs, already decided by the tier (no loops here)."""

    strategy_configs: tuple[Any, ...]
    data_configs: tuple[Any, ...]
    venue: VenueConfig
    start: str
    end: str
    policy: EnginePolicy
    cleanup_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class EngineRun:
    """What the consumer gets: the engine and its result, or ``None``s for no engine."""

    engine: Any | None
    bt_result: Any | None
    run_config_id: str


def bar_data_config(catalog_str: str, bar_type: str, start: str, end: str) -> BacktestDataConfig:
    """One bar backtest data config over ``[start, end]`` for a single bar type."""
    from nautilus_trader.config import BacktestDataConfig
    from nautilus_trader.model.data import Bar

    # NT 1.226+: pass data_cls as the CLASS, not the import
    # string — BacktestDataConfig.query compares `data_cls is Bar`
    # so a string silently disables bar-type narrowing and loads
    # every bar timeframe in the catalog (~300x the needed data).
    return BacktestDataConfig(
        catalog_path=catalog_str,
        data_cls=Bar,
        bar_types=[bar_type],
        start_time=start,
        end_time=end,
    )


def loaded_timeframes(dsl_dict: dict[str, Any]) -> set[str]:
    """Every bar timeframe a DSL loads: primary + ``additional_timeframes`` + indicator ones.

    The fill mode's tick timeframe is the FINEST loaded timeframe
    (:func:`screening_fill_mode`), so ``NTScreeningRunner._compile`` and the
    parent-side resolvers (``screening.pipeline.resolve_parent_fill_ticks``)
    must derive this set the same way. If they diverge, the parent resolves
    ticks for a coarser timeframe and the runner rejects the set
    (``MissingBarDataError``) at run time (vibe-quant-yul7u.12).
    """
    loaded = {str(dsl_dict["timeframe"])}
    loaded.update(dsl_dict.get("additional_timeframes") or [])
    for ind in (dsl_dict.get("indicators") or {}).values():
        tf = ind.get("timeframe") if isinstance(ind, dict) else getattr(ind, "timeframe", None)
        if tf:
            loaded.add(str(tf))
    return loaded


def _statistics(policy: EnginePolicy) -> list[Any]:
    """One instance per statistic class, in the policy's order.

    Resolved by name from the pyo3 module: an unknown name is a typo in a policy
    constant, and must fail loudly rather than silently shrink the statistic set.
    """
    import nautilus_trader.core.nautilus_pyo3 as nautilus_pyo3

    return [getattr(nautilus_pyo3, name)() for name in policy.statistics]


def run_engine[R](
    spec: EngineSpec,
    consume: Callable[[EngineRun], R],
    *,
    prepare: Callable[[Any], object] | None = None,
) -> R:
    """Build and run one backtest node, hand its result to ``consume``, then tear down.

    Order is load-bearing: the node is constructed BEFORE the ``try`` (a constructor
    error has no teardown to run), statistics are registered after ``node.build()``,
    ``prepare`` runs between registration and ``node.run()`` (screening installs its
    PTA precompute that way), and ``consume`` runs after the engine is read. Teardown
    always runs: engines are reset (guarded) and disposed, then every
    ``spec.cleanup_dirs`` entry is cleaned — a plain ``finally``, so a teardown error
    replaces an earlier one, exactly as the two tiers did.

    The consumer decides what a ``None`` engine means (screening records -inf metrics,
    validation raises its "engine not found" error); the refresh of ``engine.trader``
    and every refusal read stay there too.
    """
    from nautilus_trader.backtest.node import BacktestNode
    from nautilus_trader.config import (
        BacktestEngineConfig,
        BacktestRunConfig,
        LoggingConfig,
    )

    from vibe_quant.data.catalog import cleanup_epoch_parquet
    from vibe_quant.nt_compat import retain_log_guard
    from vibe_quant.validation.venue import create_backtest_venue_config

    engine_config = BacktestEngineConfig(
        strategies=list(spec.strategy_configs),
        run_analysis=True,
        logging=LoggingConfig(
            log_level=os.environ.get(spec.policy.log_level_env, spec.policy.log_level_default)
        ),
    )

    # raise_exception: node.build() otherwise swallows engine-build errors
    # (get_engine() -> None, "engine not found").
    run_config = BacktestRunConfig(
        engine=engine_config,
        venues=[create_backtest_venue_config(spec.venue)],
        data=list(spec.data_configs),
        start=spec.start,
        end=spec.end,
        dispose_on_completion=False,
        raise_exception=True,
    )

    node = BacktestNode(configs=[run_config])
    try:
        node.build()

        stats = _statistics(spec.policy)
        for engine in node.get_engines():
            analyzer = engine.kernel.portfolio.analyzer
            for stat in stats:
                analyzer.register_statistic(stat)

        if prepare is not None:
            prepare(node)

        node.run()

        engine = node.get_engine(run_config.id)
        bt_result = engine.get_result() if engine is not None else None
        return consume(EngineRun(engine=engine, bt_result=bt_result, run_config_id=run_config.id))
    finally:
        # Reset engines before dispose to avoid
        # InvalidStateTrigger('RUNNING -> DISPOSE')
        for eng in node.get_engines():
            retain_log_guard(eng)
            with contextlib.suppress(Exception):
                eng.reset()
        node.dispose()  # type: ignore[no-untyped-call]

        # NT writes corrupt epoch-timestamp parquet on dispose(), into
        # every catalog the run read -- the fill-tick catalogs included
        for directory in spec.cleanup_dirs:
            cleanup_epoch_parquet(directory)
