"""SANITY metric warnings must not fire for deliberately synthetic results.

Discovery early-exit results and failed-backtest results carry forced metrics
(e.g. sharpe=-1.0 with 0 trades) and are expected to look inconsistent;
_sanity_check_metrics should only flag real backtest output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import _evaluate_single, evaluate_population
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)

if TYPE_CHECKING:
    import pytest


def _make_chromosome() -> StrategyChromosome:
    gene = StrategyGene(
        indicator_type="RSI",
        parameters={"period": 14.0},
        condition=ConditionType.CROSSES_ABOVE,
        threshold=30.0,
    )
    return StrategyChromosome(
        entry_genes=[gene],
        exit_genes=[gene],
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        direction=Direction.LONG,
    )


def _sanity_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "SANITY" in r.getMessage()]


def _run(bt: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
        return bt

    with caplog.at_level(logging.WARNING, logger="vibe_quant.discovery.fitness"):
        evaluate_population([_make_chromosome()], bt_fn)


class TestSanityCheckSkipsSyntheticResults:
    def test_early_exit_result_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        _run(
            {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "total_return": 0.0,
                "early_exit": 0,
            },
            caplog,
        )
        assert _sanity_warnings(caplog) == []

    def test_early_exit_symbol_result_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        _run(
            {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "total_return": 0.0,
                "early_exit_symbol": 1,
            },
            caplog,
        )
        assert _sanity_warnings(caplog) == []

    def test_error_result_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        _run(
            {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "error": "RuntimeError: nt died",
            },
            caplog,
        )
        assert _sanity_warnings(caplog) == []

    def test_real_inconsistent_metrics_still_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        _run(
            {
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.2,
                "profit_factor": 0.0,
                "total_trades": 0,
                "total_return": 0.0,
            },
            caplog,
        )
        warnings = _sanity_warnings(caplog)
        assert len(warnings) == 1
        assert "0 trades but sharpe=1.5000" in warnings[0]


def _evaluate(bt: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="vibe_quant.discovery.fitness"):
        _evaluate_single(_make_chromosome(), lambda _: bt)


def _real_symbol(sharpe: float, ret: float, trades: int) -> dict[str, Any]:
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": trades,
        "total_return": ret,
    }


class TestWorstSymbolAggregateSkipsSynthetic:
    def test_aggregate_with_early_exit_symbol_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        agg = NTBacktestFn._aggregate_symbols(
            [
                {
                    "sharpe_ratio": -1.0,
                    "max_drawdown": 1.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "total_return": 0.0,
                    "window_trades": (0,),
                    "early_exit": 0,
                },
                _real_symbol(1.2, 0.2, 50),
            ],
            ["A", "B"],
        )
        _evaluate(agg, caplog)
        assert _sanity_warnings(caplog) == []

    def test_aggregate_with_error_symbol_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        agg = NTBacktestFn._aggregate_symbols(
            [
                {
                    "sharpe_ratio": -1.0,
                    "max_drawdown": 1.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "total_return": 0.0,
                    "error": "RuntimeError: nt died",
                },
                _real_symbol(1.2, 0.2, 50),
            ],
            ["A", "B"],
        )
        _evaluate(agg, caplog)
        assert _sanity_warnings(caplog) == []

    def test_aggregate_of_real_inconsistent_symbols_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        bad = {
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.2,
            "profit_factor": 0.0,
            "total_trades": 0,
            "total_return": 0.0,
        }
        agg = NTBacktestFn._aggregate_symbols([dict(bad), dict(bad)], ["A", "B"])
        _evaluate(agg, caplog)
        warnings = _sanity_warnings(caplog)
        # 1 top-level aggregate + 2 per-symbol recursive evaluations all warn.
        assert len(warnings) == 3
        assert all("0 trades but sharpe=1.5000" in w for w in warnings)
