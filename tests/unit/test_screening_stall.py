"""Hung-worker handling in the parallel screening sweep (vibe-quant-4gvwt).

The old ``as_completed`` + ``future.result(timeout=300)`` pattern never
timed out (``as_completed`` only yields already-done futures), so one hung
NT backtest stalled the whole sweep forever.
"""

from __future__ import annotations

import time

from vibe_quant.dsl import parse_strategy_string
from vibe_quant.screening.pipeline import ScreeningPipeline
from vibe_quant.screening.types import BacktestMetrics

STRATEGY_YAML = """
name: stall_test
timeframe: 4h
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
sweep:
  rsi.period: [7, 14, 21]
"""


def _sleeping_runner(params: dict[str, float | int]) -> BacktestMetrics:
    """Picklable runner that hangs far longer than the stall timeout."""
    time.sleep(30.0)
    return BacktestMetrics(parameters=params, sharpe_ratio=1.0)


def _fast_runner(params: dict[str, float | int]) -> BacktestMetrics:
    return BacktestMetrics(parameters=params, sharpe_ratio=1.0, total_trades=100)


def test_stalled_workers_yield_sentinels_and_pipeline_returns() -> None:
    dsl = parse_strategy_string(STRATEGY_YAML)
    pipeline = ScreeningPipeline(
        dsl=dsl,
        backtest_runner=_sleeping_runner,
        max_workers=2,
        stall_timeout_s=1.0,
    )

    started = time.monotonic()
    result = pipeline.run(apply_dsr=False)
    elapsed = time.monotonic() - started

    # All combos accounted for, all sentinel, and we did not wait 30s
    assert result.total_combinations == 3
    assert len(result.results) == 3
    assert all(r.sharpe_ratio == -999.0 for r in result.results)
    assert elapsed < 20.0


def test_fast_runner_unaffected_by_stall_timeout() -> None:
    dsl = parse_strategy_string(STRATEGY_YAML)
    pipeline = ScreeningPipeline(
        dsl=dsl,
        backtest_runner=_fast_runner,
        max_workers=2,
        stall_timeout_s=30.0,
    )
    result = pipeline.run(apply_dsr=False)
    assert len(result.results) == 3
    assert all(r.sharpe_ratio == 1.0 for r in result.results)
