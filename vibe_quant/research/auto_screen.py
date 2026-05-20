"""Auto-screen each parsed extraction with a single Sharpe.

Runs synchronously inside the scrape subprocess (no separate worker, no
queue). Failures are swallowed and recorded as `screen_status='failed'` on
the extraction row so the scrape loop never aborts mid-run.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

# v1 defaults (locked via bd-l685 design decisions: BTC-only, 6mo lookback)
DEFAULT_SYMBOLS: list[str] = ["BTCUSDT-PERP.BINANCE"]
DEFAULT_LOOKBACK_DAYS: int = 180
DEFAULT_TIMEFRAME: str = "1h"


def _default_window() -> tuple[str, str]:
    today = datetime.now(tz=UTC).date()
    start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start.isoformat(), today.isoformat()


def auto_screen_extraction(
    sm: StateManager,
    extraction_id: int,
    parsed_dsl_json: str,
) -> None:
    """Run one screening backtest for a freshly parsed extraction.

    Creates a `backtest_runs` row tagged with
    `parameters.auto_screen_source.extraction_id = <id>` and writes Sharpe +
    status back onto the extraction row. Any exception is logged and
    recorded as `screen_status='failed'` — never propagates.
    """
    start_date, end_date = _default_window()
    try:
        dsl_dict = _normalize_dsl(parsed_dsl_json)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "auto_screen: DSL invalid for extraction %s: %s", extraction_id, e
        )
        sm.update_extraction_screen_results(
            extraction_id,
            screen_sharpe=None,
            screen_status="failed",
            screen_run_id=None,
        )
        return

    tf_value = dsl_dict.get("timeframe", DEFAULT_TIMEFRAME)
    timeframe = str(tf_value) if tf_value is not None else DEFAULT_TIMEFRAME
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
        sharpe = _run_single_sharpe(dsl_dict, start_date, end_date)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "auto_screen: runner failed for extraction %s (run %d): %s",
            extraction_id,
            run_id,
            e,
        )
        sm.update_extraction_screen_results(
            extraction_id,
            screen_sharpe=None,
            screen_status="failed",
            screen_run_id=run_id,
        )
        return

    sm.update_extraction_screen_results(
        extraction_id,
        screen_sharpe=sharpe,
        screen_status="done",
        screen_run_id=run_id,
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


def _run_single_sharpe(
    dsl_dict: dict[str, object], start_date: str, end_date: str
) -> float | None:
    """Instantiate NTScreeningRunner and run one backtest with default params."""
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    runner = NTScreeningRunner(
        dsl_dict=dsl_dict,
        symbols=DEFAULT_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
    )
    metrics = runner({})
    sharpe = metrics.sharpe_ratio
    if sharpe is None:
        return None
    # NTScreeningRunner uses -inf as a sentinel for runner errors; surface
    # that as None rather than persisting a non-finite float.
    try:
        if not (sharpe == sharpe and sharpe not in (float("inf"), float("-inf"))):
            return None
    except TypeError:
        return None
    return float(sharpe)
