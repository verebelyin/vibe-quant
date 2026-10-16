"""Real-runner wiring: the ``NTBacktestFn`` MTM series -> runner MTM statistics.

Runs an actual ``NTBacktestFn`` over a tiny real window and proves the dense
mark-to-market daily series it RETURNS (``_run_single`` ->
``_chain_window_series``, the path discovery consumes) IS the runner's own
``last_mtm_daily_returns`` with exactly day 0 dropped: the same series whose
statistics the runner captures into ``mtm_sharpe_daily``/``mtm_skewness``/
``mtm_kurtosis``/``mtm_days``. Comparing them pins the forwarding path, which
``test_dsr.py::test_daily_sharpe_inputs_matches_nt_sharpe`` (the metric alone)
and ``test_overfitting_mtm_dsr.py::
test_runner_stores_mtm_stats_for_every_run_equal_to_series_inputs`` (calls
``_make_runner`` directly) both miss.

Was the realized-series wiring (bd vibe-quant-yul7u.30); re-pointed at the MTM
series when the realized one was removed (bd vibe-quant-f9zzq).
"""

from __future__ import annotations

from typing import Any

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.overfitting.dsr import daily_sharpe_inputs
from vibe_quant.utils import compute_day_count

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


def test_returned_mtm_series_matches_captured_runner_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _btcusdt_catalog_available():
        pytest.skip("BTCUSDT catalog data not available")
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    # Capture the runner's own MTM stats: _run_single discards the metrics it
    # computes them on, so spy on the call it uses to reach the runner.
    captured: dict[str, Any] = {}
    real_call = NTScreeningRunner.__call__

    def spy_call(self: Any, params: Any) -> Any:
        metrics = real_call(self, params)
        captured["metrics"] = metrics
        captured["runner"] = self
        return metrics

    monkeypatch.setattr(NTScreeningRunner, "__call__", spy_call)

    fn = NTBacktestFn(["BTCUSDT"], "4h", _START, _END, collect_daily_returns=True)
    out = fn(_chrom())

    series = out["mtm_daily_returns"]
    assert series is not None and len(series) > 0
    assert int(out["total_trades"]) > 0  # the series is not a veiled flat run

    metrics = captured["metrics"]
    full = captured["runner"].last_mtm_daily_returns
    assert full is not None

    # The runner's stats come from the UNTRIMMED series minus day 0, so the day
    # grid the returned series lands on must be that exact single trim.
    day_count = compute_day_count(_START, _END)
    assert day_count is not None
    assert len(full) == day_count
    assert metrics.mtm_days == day_count - 1
    assert len(series) == day_count - 1
    assert series.first_day == full.first_day + 1  # exactly one day-0 trim
    assert list(series.values) == list(full.values[1:])

    # ... and the stats the runner captured are exactly what the returned series
    # yields, so the forwarding path cannot drop or shift a different day.
    assert daily_sharpe_inputs(series.values) == (
        metrics.mtm_sharpe_daily, metrics.mtm_skewness, metrics.mtm_kurtosis,
    )  # fmt: skip
