"""Auto-screen each parsed extraction with the screening metric set.

Runs synchronously inside the scrape subprocess (no separate worker, no
queue). Failures and timeouts are swallowed and recorded on the extraction
row so the scrape loop never aborts mid-run.
"""

from __future__ import annotations

import contextlib
import json
import logging
import signal
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.screening.types import BacktestMetrics

logger = logging.getLogger(__name__)

# v1 defaults (locked via bd-l685 design decisions: BTC-only, 6mo lookback)
# Bare symbol — NTScreeningRunner appends "-PERP.BINANCE" itself
# (the old "BTCUSDT-PERP.BINANCE" value double-suffixed the instrument ID,
# so every auto-screen ran against a nonexistent instrument: 0 trades).
DEFAULT_SYMBOLS: list[str] = ["BTCUSDT"]
DEFAULT_LOOKBACK_DAYS: int = 180
DEFAULT_TIMEFRAME: str = "1h"
DEFAULT_TIMEOUT_SECONDS: int = 300


class _ScreenTimeout(Exception):
    """Raised when a single screening run exceeds DEFAULT_TIMEOUT_SECONDS."""


def _default_window(timeframe: str) -> tuple[str, str] | None:
    """Lookback window clamped to catalog coverage.

    A wall-clock window silently runs past the end of downloaded data
    (0 trades, reported as success). Clamp the end to the last catalog bar;
    return None when there is no data at all so the caller can fail loudly.
    """
    today = datetime.now(tz=UTC).date()
    end = today
    try:
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH, CatalogManager

        date_range = CatalogManager(DEFAULT_CATALOG_PATH).get_bar_date_range(
            DEFAULT_SYMBOLS[0], timeframe
        )
        if date_range is None:
            return None
        end = min(today, date_range[1].date())
    except Exception:  # noqa: BLE001 — clamping is best-effort
        logger.warning("auto_screen: could not read catalog coverage, using today")
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start.isoformat(), end.isoformat()


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def auto_screen_extraction(
    sm: StateManager,
    extraction_id: int,
    parsed_dsl_json: str,
) -> None:
    """Run one screening backtest for a freshly parsed extraction.

    Creates a `backtest_runs` row tagged with
    `parameters.auto_screen_source.extraction_id = <id>` and writes the
    full metric set (Sharpe, PF, Max DD, Return, Trades) + completion
    timestamp back onto the extraction row. Any exception is logged and
    recorded as `screen_status='failed'` with an error string — never
    propagates.
    """
    try:
        dsl_dict = _normalize_dsl(parsed_dsl_json)
    except Exception as e:  # noqa: BLE001
        msg = f"DSL invalid: {type(e).__name__}: {e}"
        logger.warning("auto_screen: %s (extraction %s)", msg, extraction_id)
        sm.update_extraction_screen_results(
            extraction_id,
            screen_sharpe=None,
            screen_status="failed",
            screen_run_id=None,
            screen_error=msg,
            screen_completed_at=_now_iso(),
        )
        return

    tf_value = dsl_dict.get("timeframe", DEFAULT_TIMEFRAME)
    timeframe = str(tf_value) if tf_value is not None else DEFAULT_TIMEFRAME

    window = _default_window(timeframe)
    if window is None:
        msg = f"no {timeframe} catalog data for {DEFAULT_SYMBOLS[0]}"
        logger.warning("auto_screen: %s (extraction %s)", msg, extraction_id)
        sm.update_extraction_screen_results(
            extraction_id,
            screen_sharpe=None,
            screen_status="failed",
            screen_run_id=None,
            screen_error=msg,
            screen_completed_at=_now_iso(),
        )
        return
    start_date, end_date = window

    run_id = sm.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=DEFAULT_SYMBOLS,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        parameters={"auto_screen_source": {"extraction_id": extraction_id}},
    )

    try:
        metrics = _run_single_metrics(dsl_dict, start_date, end_date)
    except _ScreenTimeout:
        logger.warning(
            "auto_screen: timeout for extraction %s (run %d)", extraction_id, run_id
        )
        sm.update_backtest_run_status(run_id, "failed", error_message="auto-screen timeout")
        sm.update_extraction_screen_results(
            extraction_id,
            screen_sharpe=None,
            screen_status="failed",
            screen_run_id=run_id,
            screen_error="timeout",
            screen_completed_at=_now_iso(),
        )
        return
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        logger.warning(
            "auto_screen: runner failed for extraction %s (run %d): %s",
            extraction_id,
            run_id,
            msg,
        )
        sm.update_backtest_run_status(run_id, "failed", error_message=msg)
        sm.update_extraction_screen_results(
            extraction_id,
            screen_sharpe=None,
            screen_status="failed",
            screen_run_id=run_id,
            screen_error=msg,
            screen_completed_at=_now_iso(),
        )
        return

    # Close out the run row — auto-screen previously left these 'pending'
    # forever, polluting Results Analysis.
    sm.update_backtest_run_status(run_id, "completed")
    sm.update_extraction_screen_results(
        extraction_id,
        screen_sharpe=_finite_or_none(metrics.sharpe_ratio),
        screen_status="done",
        screen_run_id=run_id,
        screen_pf=_finite_or_none(metrics.profit_factor),
        screen_max_dd=_finite_or_none(metrics.max_drawdown),
        screen_return=_finite_or_none(metrics.total_return),
        screen_trades=int(metrics.total_trades) if metrics.total_trades is not None else None,
        screen_completed_at=_now_iso(),
    )


def _normalize_dsl(parsed_dsl_json: str) -> dict[str, object]:
    """Parse extraction JSON, normalize via translator, validate as StrategyDSL.

    Returns the canonical dict (model_dump) ready for the screening runner.
    """
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.dsl.translator import translate_dsl_config

    raw = json.loads(parsed_dsl_json)
    if not isinstance(raw, dict):
        raise TypeError("parsed_dsl_json must decode to an object")
    canonical = translate_dsl_config(raw, strategy_name=str(raw.get("name", "auto")))
    dsl = validate_strategy_dict(canonical)
    return dsl.model_dump()


def _run_single_metrics(
    dsl_dict: dict[str, object], start_date: str, end_date: str
) -> BacktestMetrics:
    """Instantiate NTScreeningRunner and run one backtest with default params.

    Wall-clock-bounded via SIGALRM; if it doesn't return within
    DEFAULT_TIMEOUT_SECONDS, raises :class:`_ScreenTimeout`.
    """
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    runner = NTScreeningRunner(
        dsl_dict=dsl_dict,
        symbols=DEFAULT_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
    )

    with _alarm(DEFAULT_TIMEOUT_SECONDS):
        return runner({})


@contextlib.contextmanager
def _alarm(seconds: int) -> Iterator[None]:
    """SIGALRM-based wall-clock guard. Only effective in the main thread."""

    def _handler(_signum: int, _frame: object) -> None:
        raise _ScreenTimeout(f"screening exceeded {seconds}s")

    try:
        prev = signal.signal(signal.SIGALRM, _handler)
    except (ValueError, AttributeError):
        # Not on a unix main thread — fall through without protection.
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        with contextlib.suppress(ValueError, TypeError):
            signal.signal(signal.SIGALRM, prev)


def _finite_or_none(value: float | int | None) -> float | None:
    """NTScreeningRunner uses ±inf as sentinels for runner errors; surface as None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if v in (float("inf"), float("-inf")):
        return None
    return v
