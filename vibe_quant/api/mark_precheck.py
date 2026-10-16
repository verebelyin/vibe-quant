"""MARK-coverage precheck for mode-on launches (vibe-quant-592lt).

A launch with ``liquidation_mode`` on needs MARK 1m bars covering the whole
window the backtest will scan. Without them the run row and the subprocess are
created first and the run fails later inside ``apply_liquidation ->
require_mark_coverage`` — minutes of GA/backtest work lost to a data gap. These
sync helpers answer 422 BEFORE ``create_backtest_run``; routers run them in a
worker thread (``asyncio.to_thread``) so the parquet read never blocks the
event loop.

Each checker mirrors the window its own runtime scans:

* screening/discovery pass the run's start/end dates straight to
  ``apply_liquidation`` (``screening/nt_runner.py``), so the window is
  ``[date_to_ns(start), date_to_ns(end))`` of the request;
* validation clamps the request to the 1m coverage it will actually run on
  (:func:`~vibe_quant.validation.runner.clamp_run_window`), so a request end
  past the data is not a false 422 — the MARK window shrinks with it.

Coverage checking is stricter than the runtime in one way: it requires MARK
bars for EVERY requested symbol in request order, not only the symbols that
traded (a launch that would succeed only because a symbol never traded is
refused instead of silently unchecked).

``mark_bars`` is imported lazily: a mode-off launch imports nothing new (M15).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from fastapi import HTTPException

# Re-exported: tests patch this name to point the precheck at a tmp catalog.
from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH as DEFAULT_CATALOG_PATH
from vibe_quant.validation.extraction import date_to_ns

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

# One cold MARK decode peaks ~1.1 GB (measured 0.294 s / 1144 MB per symbol), so
# concurrent API prechecks run the coverage step one at a time.
_COVERAGE_LOCK = threading.Lock()

_DOWNLOAD_HINT = "(python -m vibe_quant.data download-mark --start YYYY-MM-DD --catalog)"


def check_mark_coverage(
    catalog_path: str | Path,
    symbols: Sequence[str],
    start_ns: int,
    end_ns: int,
) -> None:
    """Check MARK 1m coverage, importing ``mark_bars`` only when a run needs it.

    A thin module-level wrapper around
    :func:`vibe_quant.data.mark_bars.check_mark_coverage` (same rules, same
    first-failure exception) so the data-layer import stays off the mode-off
    launch path and the precheck's coverage step has one patch point.
    """
    from vibe_quant.data import mark_bars

    mark_bars.check_mark_coverage(catalog_path, symbols, start_ns, end_ns)


def precheck_screening(
    mode: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> None:
    """422 unless MARK 1m bars cover the screening window ``[start, end)``.

    The window is exactly what ``nt_runner`` hands ``apply_liquidation``; the
    end is exclusive, so the last required bar opens at ``end - 60 s``.
    """
    if mode == "off":
        return
    _precheck_window(mode, symbols, date_to_ns(start_date), date_to_ns(end_date))


def precheck_validation(
    mode: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    strategy_dsl_config: dict[str, object],
    strategy_name: str,
    parameters: dict[str, object],
) -> None:
    """422 unless MARK 1m bars cover the window validation will actually run on.

    The request window is resolved through the runner's own
    :func:`~vibe_quant.validation.runner.resolve_detail_timeframe` /
    :func:`~vibe_quant.validation.runner.clamp_run_window`, so the checked
    window is the subprocess's clamped one (whole UTC days the 1m data covers).
    A ``ValidationRunnerError`` from either -- no 1m bars for a symbol, no full
    day of overlap -- is the run's own failure, answered as 422 here.

    A DSL the runner itself cannot parse is not a MARK problem: the precheck is
    skipped (logged at INFO) and the run fails at runtime exactly as today.
    """
    if mode == "off":
        return
    from vibe_quant.validation.runner import (
        ValidationRunner,
        ValidationRunnerError,
        clamp_run_window,
        resolve_detail_timeframe,
    )

    run_config: dict[str, object] = {
        "symbols": list(symbols),
        "start_date": start_date,
        "end_date": end_date,
        "parameters": parameters,
    }
    # Parsed exactly as the runner does (frontend-format translation included).
    runner = ValidationRunner.__new__(ValidationRunner)
    try:
        dsl = runner._validate_dsl(strategy_dsl_config, strategy_name=strategy_name)
    except ValidationRunnerError as exc:
        logger.info("MARK precheck skipped: strategy DSL does not parse (%s)", exc)
        return

    try:
        detail = resolve_detail_timeframe(
            run_config, dsl.timeframe, None, catalog_path=DEFAULT_CATALOG_PATH
        )
        clamped, _note = clamp_run_window(
            run_config, detail or dsl.timeframe, catalog_path=DEFAULT_CATALOG_PATH
        )
    except ValidationRunnerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    clamped_start = clamped.get("start_date")
    clamped_end = clamped.get("end_date")
    _precheck_window(
        mode,
        symbols,
        date_to_ns(clamped_start if isinstance(clamped_start, str) else None),
        date_to_ns(clamped_end if isinstance(clamped_end, str) else None),
    )


def precheck_discovery(
    mode: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> None:
    """422 unless MARK 1m bars cover the full discovery span ``[start, end)``.

    Every stage (train/holdout/eval/cross/WFA) scans a sub-window of the
    resolved request dates, so the whole span is the union the run touches.
    """
    if mode == "off":
        return
    _precheck_window(mode, symbols, date_to_ns(start_date), date_to_ns(end_date))


def _precheck_window(
    mode: str,
    symbols: Sequence[str],
    start_ns: int | None,
    end_ns: int | None,
) -> None:
    """Run the coverage check and turn its first failure into the launch 422."""
    if start_ns is None or end_ns is None:  # unparseable date: not a MARK problem
        return
    from vibe_quant.data import mark_bars

    try:
        with _COVERAGE_LOCK:
            check_mark_coverage(DEFAULT_CATALOG_PATH, symbols, start_ns, end_ns)
    except mark_bars.MarkDataMissing as exc:
        raise HTTPException(
            status_code=422,
            detail=f"liquidation_mode={mode!r} needs MARK 1m bars: {exc} {_DOWNLOAD_HINT}",
        ) from exc
