"""The train gate that admits a genome to the ``--gate-pool all`` pool (vibe-quant-clmmc.2).

``passes_train_gate`` is the rule that also decides whether a genome keeps its DSR
series: portfolio = trades >= min_trades and return > 0; worst = every symbol
profitable with >= min_trades trades.
"""

from __future__ import annotations

from typing import Any

import pytest

from vibe_quant.discovery.fitness import FitnessResult, passes_train_gate


def _fit(trades: int, ret: float, **kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": trades, "total_return": ret, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.5,
        "adjusted_score": 0.5, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(kw)
    return FitnessResult(**base)


@pytest.mark.simple_target(
    reason="kill-zone gate rule: decides which genomes reach the champion gates (A3 edges)"
)
@pytest.mark.parametrize(
    ("trades", "ret", "expected"),
    [(50, 1e-9, True), (49, 0.2, False), (80, 0.0, False), (80, float("nan"), False)],
)
def test_portfolio_edges(trades: int, ret: float, expected: bool) -> None:
    assert passes_train_gate(_fit(trades, ret), 50) is expected


@pytest.mark.simple_target(
    reason="kill-zone gate rule: decides which genomes reach the champion gates (A3 edges)"
)
@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"BTCUSDT": (0.3, 80), "ETHUSDT": (1e-9, 50)}, True),
        ({"BTCUSDT": (0.3, 80), "ETHUSDT": (0.0, 80)}, False),
        ({"BTCUSDT": (0.3, 80), "ETHUSDT": (0.2, 49)}, False),
    ],
)
def test_worst_mode_needs_every_symbol(stats: dict[str, tuple[float, int]], expected: bool) -> None:
    # the run-level numbers pass: only the per-symbol rule decides
    assert passes_train_gate(_fit(200, 0.5, symbol_stats=stats), 50) is expected
