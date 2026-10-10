"""Real-runner wiring: ``NTScreeningRunner.last_daily_returns`` -> DSR inputs
(bd vibe-quant-yul7u.30).

Runs an actual ``NTBacktestFn._run_single(collect_daily_returns=True)`` over a
tiny real window and proves the dense daily series it returns reproduces the
Sharpe the runner itself reported -- the same equality ``test_dsr.py::
test_daily_sharpe_inputs_matches_nt_sharpe`` pins at the metrics level.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.overfitting.dsr import daily_sharpe_inputs
from vibe_quant.utils import compute_day_count

_ANN = math.sqrt(252.0)
_START, _END = "2024-01-01", "2024-03-01"


def _chrom(**kw: Any) -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    base: dict[str, Any] = {
        # RSI(14) < 50 long entry / > 60 exit: trades on the Jan-Feb 2024 4h
        # bars; SL 10% keeps BTC notional above the exchange minimum (a 2% SL
        # sized the position below min_notional and skipped every entry).
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 50.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 60.0, None)],
        "stop_loss_pct": 10.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


def _btcusdt_catalog_available() -> bool:
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

    bar_dir = DEFAULT_CATALOG_PATH / "data" / "bar"
    if not bar_dir.is_dir():
        return False
    return any("BTCUSDT-PERP.BINANCE" in d.name for d in bar_dir.iterdir())


def test_run_single_daily_returns_match_reported_sharpe() -> None:
    if not _btcusdt_catalog_available():
        pytest.skip("BTCUSDT catalog data not available")
    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", _START, _END, collect_daily_returns=True,
    )  # fmt: skip
    out = fn._run_single(_chrom(), _START, _END)

    series = out["daily_returns"]
    assert series is not None and len(series) > 0
    assert int(out["total_trades"]) > 0  # the series is not a veiled flat run

    # dense series covers every day; NT/DSR consume the same days minus day 0
    day_count = compute_day_count(_START, _END)
    assert day_count is not None and len(series) == day_count
    trimmed = series.without_first_day()
    assert len(trimmed) == day_count - 1

    sr, _skew, _kurt = daily_sharpe_inputs(trimmed.values)
    assert sr * _ANN == pytest.approx(float(out["sharpe_ratio"]), rel=1e-9)
