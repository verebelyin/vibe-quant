"""Launch-time data prechecks: fill ticks + context-indicator aux data (vibe-quant-b5xfj).

A launch whose strategy timeframe fills on ticks, or whose indicator pool names a
context indicator (``FUNDING``, ``BTC_TREND``, ...), needs data covering the run
window. Without it the run row and the subprocess are created first and the run
fails minutes later inside the runner:

* ``MissingBarDataError: fill ticks for BNBUSDT 1d do not cover the run window``
  (discovery 1d, run 923);
* ``AuxDataUnavailableError: BTCUSDT daily bars end before the requested window``
  (discovery 4h, run 924).

These sync helpers answer 422 BEFORE ``create_backtest_run``; routers run them in
a worker thread (``asyncio.to_thread``) so the parquet read never blocks the
event loop. Same shape as :mod:`vibe_quant.api.mark_precheck`.

Each checker mirrors the window its own runtime resolves:

* screening and discovery hand the request dates straight to
  ``resolve_fill_ticks`` / ``aux_data.preflight``, so the whole requested span
  (holdout included) is checked;
* the fill-tick timeframe is the one ``screening_fill_mode`` derives from the
  run's loaded timeframes (discovery chromosomes compile to single-timeframe
  DSLs; a screening DSL loads its primary + additional + indicator timeframes).
  A 1m strategy releases on its own bars and uses no ticks at all;
* a screening check runs against the EFFECTIVE DSL the subprocess will parse —
  ``parameters.dsl_override`` when the launch carried one, else the stored
  ``dsl_config`` (``screening/__main__.py`` picks the override the same way).

The data layer is imported lazily: a launch that needs neither ticks nor context
aux data imports nothing new. A screening DSL the runner itself cannot parse is
not a data problem — the check is skipped (logged at INFO) and the run fails at
runtime exactly as today (M14).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException

# Re-exported: tests patch this name to point the precheck at a tmp catalog.
from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH as DEFAULT_CATALOG_PATH
from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Appended to a fill-tick 422: the tick sets are built from the catalog's 1m
#: bars, so the missing piece is ingested data.
_FILL_TICK_HINT = "(python -m vibe_quant.data ingest --symbols SYMBOL --start YYYY-MM-DD)"


def precheck_discovery_data(
    timeframe: str,
    symbols: Sequence[str],
    indicator_pool: Sequence[str] | None,
    start_date: str,
    end_date: str,
) -> None:
    """422 unless a discovery launch's fill ticks and context aux data cover the window.

    The window is the full requested span (every GA stage, holdout included, scans
    a sub-window of it). ``indicator_pool`` None means the GA default pool, which
    excludes every ``needs_context`` spec, so the aux check is then a no-op.
    """
    _check_fill_ticks(timeframe, {timeframe}, symbols, start_date, end_date)
    _check_aux_data(_pool_names(indicator_pool), symbols, start_date, end_date)


def precheck_screening_data(
    dsl_config: dict[str, object],
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    strategy_name: str = "strategy",
    parameters: dict[str, object] | None = None,
) -> None:
    """422 unless a screening launch's fill ticks and context aux data cover the window.

    The DSL checked is the EFFECTIVE one the screening subprocess will run, picked
    exactly as ``vibe_quant/screening/__main__.py`` picks it: a non-empty
    ``parameters["dsl_override"]`` used RAW when present, otherwise the stored
    ``dsl_config`` translated for the strategy (``translate_dsl_config``); either
    way validated through ``StrategyDSL``. The launch accepts an override into the
    run's parameters, so checking only the stored config would resolve the wrong
    tick timeframe (both a missed 422 and a spurious one). The loaded timeframes
    come from ``loaded_timeframes`` -- the same helper
    ``NTScreeningRunner._compile`` uses, so parent and runner agree on the
    fill-tick timeframe.
    """
    from vibe_quant.dsl.schema import StrategyDSL
    from vibe_quant.dsl.translator import translate_dsl_config
    from vibe_quant.screening.nt_runner import loaded_timeframes

    override = parameters.get("dsl_override") if isinstance(parameters, dict) else None
    try:
        raw_config: dict[str, object]
        if isinstance(override, dict) and override:
            raw_config = override
        else:
            raw_config = translate_dsl_config(dsl_config, strategy_name=strategy_name)
        dsl_dict = StrategyDSL.model_validate(raw_config).model_dump()
    except (ValueError, TypeError, KeyError) as exc:
        # Pydantic's ValidationError is a ValueError: a strategy problem, not a
        # data one (the run fails at compile time exactly as before).
        logger.info("data precheck skipped: strategy DSL does not parse (%s)", exc)
        return

    _check_fill_ticks(
        str(dsl_dict["timeframe"]), loaded_timeframes(dsl_dict), symbols, start_date, end_date
    )
    _check_aux_data(_dsl_indicator_types(dsl_dict), symbols, start_date, end_date)


def _pool_names(indicator_pool: Sequence[str] | None) -> list[str]:
    """The pool the GA will actually use: the request's, or the context-free default."""
    if indicator_pool is not None:
        return list(indicator_pool)
    from vibe_quant.discovery.genome import build_indicator_pool

    return sorted(build_indicator_pool())


def _dsl_indicator_types(dsl_dict: dict[str, object]) -> list[str]:
    """The indicator types a strategy uses, as the screening runner preflights them."""
    indicators = dsl_dict.get("indicators")
    if not isinstance(indicators, dict):
        return []
    return [
        str(spec.get("type", "")) for spec in indicators.values() if isinstance(spec, dict)
    ]


def _check_fill_ticks(
    timeframe: str,
    loaded: set[str],
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> None:
    """422 unless the fill ticks a run at ``timeframe`` needs cover the window."""
    from vibe_quant.data.fill_ticks import resolve_fill_ticks
    from vibe_quant.screening.nt_runner import screening_fill_mode

    tick_tf = screening_fill_mode(timeframe, loaded).tick_timeframe
    if tick_tf is None:  # a 1m strategy fills on its own bars; no tick set is built
        return
    try:
        resolve_fill_ticks(symbols, tick_tf, start_date, end_date, DEFAULT_CATALOG_PATH)
    except DataUnavailableError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{exc} {_FILL_TICK_HINT}",
        ) from exc


def _check_aux_data(
    indicator_types: list[str],
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> None:
    """422 unless the aux data of every context indicator covers the window.

    Configured like the runners' own parent-side preflight: the archive already
    configured for this process (default when none), and this process's catalog.
    """
    from vibe_quant.dsl import aux_data

    aux_data.configure(aux_data.archive_path(), DEFAULT_CATALOG_PATH)
    try:
        aux_data.preflight(indicator_types, list(symbols), start_date, end_date)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
