"""Discovery pool transport and the CLI's two result-save sites.

The fitness pool task ships the fast-tier fallback snapshot back with its
result; these tests pin the argument forwarding and the error-row handling
around that transport, plus what ``run.main`` persists on the zero-champion and
the champion save sites.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery import fitness
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.errors import DataUnavailableError

from .test_fitness import _make_chromosome

if TYPE_CHECKING:
    from pathlib import Path

_METRICS: dict[str, float | int] = {
    "sharpe_ratio": 1.0,
    "max_drawdown": 0.1,
    "profit_factor": 1.5,
    "total_trades": 60,
    "total_return": 0.2,
    "skewness": 0.1,
    "kurtosis": 3.0,
    "trade_returns": (0.01, -0.01),
}


def test_parallel_data_unavailable_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitness, "_parallel_broken", False)

    def no_data(_chrom: Any) -> dict[str, float | int]:
        raise DataUnavailableError("catalog missing")

    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(DataUnavailableError):
        fitness.evaluate_population(
            [_make_chromosome()],
            no_data,
            max_workers=2,
            executor=pool,
            min_trades=10,
            timeframe="4h",
        )


_MAIN_BASE = [
    "--seed", "7", "--population-size", "6", "--max-generations", "2",
    "--max-workers", "-1", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
    "--timeframe", "4h", "--symbols", "BTCUSDT",
]  # fmt: skip
_MAIN_NO_GATES = ["--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate"]


def test_cli_saves_both_the_no_champion_and_the_champion_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run persists a notes row on BOTH save sites, mock or real backtests.

    The same mock backtests stand in for NT so the run is hermetic: a real
    NTBacktestFn construction fails the test, and the patched factory counts
    the headline / WFA branches it served.
    """
    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery.mock_backtest import mock_backtest

    real_make_cells = discovery_run._make_cell_fns
    nt_calls: list[str] = []

    def mock_cell_fns(**kwargs: Any) -> Any:
        return real_make_cells(**{**kwargs, "use_mock": True})

    def mock_nt_factory(**kwargs: Any) -> Any:
        nt_calls.append(str(kwargs["start_date"]))
        return mock_backtest

    def no_real_nt(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("real NTBacktestFn built by the CLI save-site test")

    monkeypatch.setattr(discovery_run, "_make_cell_fns", mock_cell_fns)
    monkeypatch.setattr(discovery_run, "_make_nt_backtest_fn", mock_nt_factory)
    monkeypatch.setattr(discovery_run, "_check_data_available", lambda symbols: True)
    monkeypatch.setattr(NTBacktestFn, "__init__", no_real_nt)

    def saved(name: str, extra: list[str], *, mock: bool) -> dict[str, Any]:
        db = tmp_path / name
        state = StateManager(db)
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="discovery",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2025-01-01",
            parameters={},
        )
        state.close()
        argv = ["prog", *_MAIN_BASE, *extra, "--run-id", str(run_id), "--db", str(db)]
        if mock:
            argv.append("--mock")
        monkeypatch.setattr(sys, "argv", argv)
        assert cli.main() == 0
        state = StateManager(db)
        row = state.get_backtest_result(run_id)
        state.close()
        assert row is not None
        return json.loads(row["notes"])

    for mock in (True, False):
        nt_calls.clear()
        plain = saved(f"plain_{mock}.db", [], mock=mock)
        assert plain.get("top_strategies") == []  # zero-champion save site
        assert plain["outcome"] == "no_viable_strategies"
        champions = saved(f"champions_{mock}.db", _MAIN_NO_GATES, mock=mock)
        assert len(champions["top_strategies"]) == 5  # champion save site
        if mock:
            assert nt_calls == []
        else:
            # The non-mock run reached the full-range / WFA factories, and both
            # were served by the mock (no_real_nt never fired).
            assert nt_calls


def _generic_fn(chrom: Any) -> dict[str, float | int]:
    """1 entry gene: 30 trades (passes min_trades=10, fails the default 50); 2: 500."""
    trades = {1: 30, 2: 500}[len(chrom.entry_genes)]
    return {**_METRICS, "total_trades": trades}


def _generic_filters(chrom: Any, _bt: dict[str, float | int]) -> dict[str, bool]:
    return {"genes": len(chrom.entry_genes) == 1}


def test_generic_pool_path_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fitness, "_parallel_broken", False)
    chroms = [_make_chromosome(1, 1), _make_chromosome(2, 1)]
    kwargs: dict[str, Any] = {"min_trades": 10, "timeframe": "1m"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = fitness.evaluate_population(
            chroms, _generic_fn, _generic_filters, max_workers=2, executor=pool, **kwargs
        )

    sequential = fitness.evaluate_population(
        chroms, _generic_fn, _generic_filters, max_workers=None, **kwargs
    )
    assert parallel == sequential
    # The forwarded values reached the worker: min_trades=10 (not 50) lets 30 trades
    # pass; "1m" (not None) keeps 500 trades under the overtrade threshold.
    assert [r.total_trades for r in parallel] == [30, 500]
    assert parallel[0].passed_filters is True
    assert parallel[1].overtrade_penalty == 0.0
    assert parallel[1].filter_results == {"genes": False}


def test_generic_pool_failed_task_gets_an_error_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fitness, "_parallel_broken", False)
    real = fitness._evaluate_single

    def boom_on_two_genes(chrom: Any, *args: Any, **kwargs: Any) -> Any:
        if len(chrom.entry_genes) == 2:
            raise RuntimeError("worker exploded")
        return real(chrom, *args, **kwargs)

    monkeypatch.setattr(fitness, "_evaluate_single", boom_on_two_genes)
    chroms = [_make_chromosome(1, 1), _make_chromosome(2, 1)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        out = fitness.evaluate_population(
            chroms, _generic_fn, max_workers=2, executor=pool, min_trades=10, timeframe="1m"
        )

    assert out[0].total_trades == 30  # the chromosome that finished is untouched
    assert out[1].error == "RuntimeError: worker exploded"
