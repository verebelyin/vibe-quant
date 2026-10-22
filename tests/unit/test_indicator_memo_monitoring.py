"""Indicator-memo monitoring: pool transport, persisted notes and log lines.

Covers design rows R16c and R26-R30 (vibe-quant-6n0n9 T3).
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery import fitness, pipeline
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.operators import Direction, StrategyChromosome
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
    _ind_memo_generation_line,
)
from vibe_quant.dsl import indicator_memo
from vibe_quant.errors import DataUnavailableError

from .test_fitness import _make_chromosome, _make_gene

if TYPE_CHECKING:
    from collections.abc import Iterator
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

_SYMBOL_TRADES = {"BTCUSDT": 40, "ETHUSDT": 30}


@pytest.fixture(autouse=True)
def _clean_memo(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO", "1")
    monkeypatch.delenv("VIBE_QUANT_IND_MEMO_MAX_MB", raising=False)
    indicator_memo.refresh_enabled()
    indicator_memo.clear()
    yield
    indicator_memo.clear()


def _snap(proc_id: str, hits: int, misses: int) -> dict[str, Any]:
    snap = dict(indicator_memo.snapshot())
    snap["proc_id"] = proc_id
    snap["hits"] = hits
    snap["misses"] = misses
    return snap


def _line_summary(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "enabled": True,
        "max_mb": 64,
        "processes": 1,
        "hits": 0,
        "misses": 0,
        "hit_rate": None,
        "bypassed": 0,
        "evictions": 0,
        "switch_offs": 0,
        "off_processes": 0,
        "parent_sessions": 0,
        "entries_max": 0,
        "est_mb_max": 0.0,
        "rss_growth_mb_max": None,
        "peak_rss_mb_max_worker": None,
        "peak_rss_mb_parent": 0.0,
    }
    base.update(over)
    return base


class SplitFn(NTBacktestFn):
    """Two-symbol worst-mode fn: one ``_run_single`` per symbol, no engine."""

    def __init__(self) -> None:
        super().__init__(
            ["BTCUSDT", "ETHUSDT"],
            "4h",
            "2024-01-01",
            "2024-02-01",
            symbol_agg="worst",
            min_trades=10,
        )

    def _run_single(self, chromosome, start_date, end_date, symbols=None):  # type: ignore[no-untyped-def]
        assert symbols is not None and len(symbols) == 1
        return {**_METRICS, "total_trades": _SYMBOL_TRADES[symbols[0]]}


class JudgeRecorder(NTBacktestFn):
    """Records the judge flag every evaluation sees; returns zero metrics."""

    def __init__(self) -> None:
        super().__init__(["BTCUSDT"], "4h", "2024-01-01", "2024-02-01")
        self.judges: list[bool] = []

    def __call__(self, chromosome) -> dict[str, float | int]:  # type: ignore[no-untyped-def]
        self.judges.append(self.ind_memo_judge)
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "total_return": -1.0,
        }


def test_record_worker_last_snapshot_per_proc_wins() -> None:
    indicator_memo.clear()
    indicator_memo.record_worker(_snap("a", 5, 5))
    indicator_memo.record_worker(_snap("a", 9, 6))
    indicator_memo.record_worker(_snap("b", 1, 3))

    summary = indicator_memo.run_summary()
    assert summary["hits"] == 10
    assert summary["misses"] == 9
    assert summary["hit_rate"] == 0.5263
    assert summary["processes"] == 3

    own_id = str(indicator_memo.snapshot()["proc_id"])
    indicator_memo.record_worker(_snap(own_id, 100, 0))
    assert indicator_memo.run_summary()["hits"] == 10


def test_run_summary_no_calls() -> None:
    indicator_memo.clear()
    summary = indicator_memo.run_summary()
    assert summary["hit_rate"] is None
    assert summary["peak_rss_mb_max_worker"] is None
    assert summary["rss_growth_mb_max"] is None
    assert summary["processes"] == 1


def test_parallel_paths_record_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fitness, "_parallel_broken", False)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(indicator_memo, "record_worker", recorded.append)
    chroms = [_make_chromosome(), _make_chromosome(2, 2)]
    fn = SplitFn()

    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = fitness.evaluate_population(
            chroms, fn, max_workers=2, executor=pool, min_trades=10, timeframe="4h"
        )

    assert len(recorded) == len(chroms) * len(fn.symbols)
    assert all(snap["proc_id"] == indicator_memo.snapshot()["proc_id"] for snap in recorded)

    sequential = fitness.evaluate_population(
        chroms, fn, max_workers=None, min_trades=10, timeframe="4h"
    )
    assert parallel == sequential


def test_failed_task_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fitness, "_parallel_broken", False)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(indicator_memo, "record_worker", recorded.append)

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
    assert recorded == []


def _distinct_population(base_period: int) -> list[Any]:
    """Four chromosomes with DSL keys no other population here (or random) shares."""
    return [
        StrategyChromosome(
            entry_genes=[_make_gene(period=base_period + i)],
            exit_genes=[_make_gene(period=base_period + i)],
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            direction=Direction.LONG,
        )
        for i in range(4)
    ]


def _pipeline_cfg(generations: int) -> DiscoveryConfig:
    return DiscoveryConfig(
        population_size=4,
        max_generations=generations,
        max_workers=1,
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-02-01",
        require_dsr=False,
        require_buy_hold=False,
        require_bootstrap_ci=False,
        eval_windows=1,
    )


def _deterministic_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    fn: NTBacktestFn,
    gate_fn: NTBacktestFn | None = None,
    generations: int = 2,
) -> DiscoveryPipeline:
    """A pipeline whose generations never depend on the global RNG state.

    Generation 1 and the offspring are four distinct unseen genomes each, so
    every generation evaluates four new backtests whatever ``random`` holds.
    """
    monkeypatch.setattr(DiscoveryPipeline, "_apply_indicator_pool_filter", lambda self: None)
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_aux_data", lambda self: None)
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_bars", lambda self: None)
    monkeypatch.setattr(
        pipeline, "initialize_population", lambda *_a, **_k: _distinct_population(100)
    )
    monkeypatch.setattr(
        DiscoveryPipeline,
        "_evolve_generation",
        lambda self, _pop, _fit: (_distinct_population(200), [None] * 4),
    )
    gates = gate_fn if gate_fn is not None else JudgeRecorder()
    return DiscoveryPipeline(
        _pipeline_cfg(generations), fn, backtest_fn_factory=lambda _start, _end: gates
    )


def test_pipeline_judges_from_generation_two(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = JudgeRecorder()
    gate_fn = JudgeRecorder()

    _deterministic_pipeline(monkeypatch, fn, gate_fn).run()

    assert fn.judges == [False] * 4 + [True] * 4
    assert gate_fn.ind_memo_judge is True


_MAIN_BASE = [
    "--seed", "7", "--population-size", "6", "--max-generations", "2",
    "--max-workers", "-1", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
    "--timeframe", "4h", "--symbols", "BTCUSDT",
]  # fmt: skip
_MAIN_NO_GATES = ["--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate"]


def test_attach_ind_memo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery import record
    from vibe_quant.discovery.mock_backtest import mock_backtest

    summary = {"hits": 3, "misses": 1}
    monkeypatch.setattr(indicator_memo, "run_summary", lambda: summary)

    notes: dict[str, Any] = {"top_strategies": ["x"]}
    record.attach_ind_memo(notes, use_mock=False)
    assert notes == {"top_strategies": ["x"], "ind_memo": summary}

    mock_notes: dict[str, Any] = {"top_strategies": []}
    record.attach_ind_memo(mock_notes, use_mock=True)
    assert mock_notes == {"top_strategies": []}

    # Both save sites (zero- and five-champion) carry the memo only on a real
    # run. The same mock backtests stand in for NT so the run is hermetic: a
    # real NTBacktestFn construction fails the test, and the patched factory
    # counts the headline (:1479) / WFA (:1556) branches it served.
    real_make_cells = cli._make_cell_fns
    nt_calls: list[str] = []

    def mock_cell_fns(**kwargs: Any) -> Any:
        return real_make_cells(**{**kwargs, "use_mock": True})

    def mock_nt_factory(**kwargs: Any) -> Any:
        nt_calls.append(str(kwargs["start_date"]))
        return mock_backtest

    def no_real_nt(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("real NTBacktestFn built by the ind_memo notes test")

    monkeypatch.setattr(cli, "_make_cell_fns", mock_cell_fns)
    monkeypatch.setattr(cli, "_make_nt_backtest_fn", mock_nt_factory)
    monkeypatch.setattr(cli, "_check_data_available", lambda symbols: True)
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
        for stored in (plain, champions):
            if mock:
                assert "ind_memo" not in stored
            else:
                assert stored["ind_memo"] == summary
        if mock:
            assert nt_calls == []
        else:
            # The non-mock run reached the full-range / WFA factories, and both
            # were served by the mock (no_real_nt never fired).
            assert nt_calls


def test_generation_line() -> None:
    prev = _line_summary(hits=10, misses=10, entries_max=5, est_mb_max=0.7)
    cur = _line_summary(
        hits=30,
        misses=14,
        entries_max=42,
        est_mb_max=6.3,
        rss_growth_mb_max=7.5,
        evictions=2,
        off_processes=1,
        processes=3,
        peak_rss_mb_max_worker=44.2,
        peak_rss_mb_parent=340.0,
    )
    assert _ind_memo_generation_line(prev, cur) == (
        "  IndMemo: gen hits=20 misses=4 rate=0.83 | run rate=0.68 "
        "entries_max=42 est_mb_max=6.3 rss_growth_mb_max=7.5 evictions=2 "
        "off=1/3 procs peak_rss_max_worker=44.2 MB parent_rss=340 MB"
    )


def test_generation_line_skipped_for_non_nt_fn(caplog: pytest.LogCaptureFixture) -> None:
    cfg = DiscoveryConfig(population_size=2, max_generations=1, max_workers=1, elite_count=1)
    plain = DiscoveryPipeline(cfg, lambda chrom: {})
    nt = DiscoveryPipeline(cfg, JudgeRecorder())
    summary = _line_summary(hits=1, misses=1, est_mb_max=0.1)

    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        plain._log_ind_memo_generation(None, summary)
        assert "IndMemo" not in caplog.text
        caplog.clear()
        nt._log_ind_memo_generation(None, summary)
        assert "IndMemo: gen hits=1 misses=1 rate=0.50" in caplog.text


def test_run_end_cap_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = DiscoveryConfig(population_size=2, max_generations=1, max_workers=1, elite_count=1)
    run = DiscoveryPipeline(cfg, JudgeRecorder())
    summary = _line_summary(hits=5, misses=5, hit_rate=0.5, evictions=5, max_mb=64)
    monkeypatch.setattr(indicator_memo, "run_summary", lambda: summary)

    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        run._log_ind_memo_run_end()
    assert "IndMemo run summary: " in caplog.text
    assert (
        "IndMemo cap reached: 5 evictions, hit rate 0.50; "
        "consider VIBE_QUANT_IND_MEMO_MAX_MB=128"
    ) in caplog.text

    caplog.clear()
    monkeypatch.setattr(
        indicator_memo, "run_summary", lambda: _line_summary(hits=5, misses=5, hit_rate=0.5)
    )
    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        run._log_ind_memo_run_end()
    assert "IndMemo run summary: " in caplog.text
    assert "cap reached" not in caplog.text


def test_run_end_cap_warning_names_the_workers_cap(caplog: pytest.LogCaptureFixture) -> None:
    """D12: workers ran with 8 MB and evicted; the parent never opened a session."""
    worker = dict(indicator_memo.snapshot())
    worker.update(proc_id="worker-1", max_bytes=8_000_000, evictions=3, hits=2, misses=2)
    indicator_memo.record_worker(worker)
    cfg = DiscoveryConfig(population_size=2, max_generations=1, max_workers=1, elite_count=1)
    run = DiscoveryPipeline(cfg, JudgeRecorder())

    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        run._log_ind_memo_run_end()

    assert '"max_mb": 8' in caplog.text
    assert (
        "IndMemo cap reached: 3 evictions, hit rate 0.50; "
        "consider VIBE_QUANT_IND_MEMO_MAX_MB=16"
    ) in caplog.text


def test_run_logs_memo_lines_through_the_real_call_sites(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run = _deterministic_pipeline(monkeypatch, JudgeRecorder())

    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        run.run()

    assert caplog.text.count("IndMemo: gen hits=") == 2  # one per generation
    assert caplog.text.count("IndMemo run summary: ") == 1
    assert "IndMemo cap reached" not in caplog.text

    caplog.clear()
    monkeypatch.setattr(
        indicator_memo,
        "run_summary",
        lambda: _line_summary(hits=5, misses=5, hit_rate=0.5, evictions=5, max_mb=64),
    )
    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        _deterministic_pipeline(monkeypatch, JudgeRecorder()).run()
    assert caplog.text.count("IndMemo run summary: ") == 1
    assert (
        "IndMemo cap reached: 5 evictions, hit rate 0.50; "
        "consider VIBE_QUANT_IND_MEMO_MAX_MB=128"
    ) in caplog.text


def _generic_fn(chrom: Any) -> dict[str, float | int]:
    """1 entry gene: 30 trades (passes min_trades=10, fails the default 50); 2: 500."""
    trades = {1: 30, 2: 500}[len(chrom.entry_genes)]
    return {**_METRICS, "total_trades": trades}


def _generic_filters(chrom: Any, _bt: dict[str, float | int]) -> dict[str, bool]:
    return {"genes": len(chrom.entry_genes) == 1}


def test_generic_pool_path_forwards_args_and_records_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitness, "_parallel_broken", False)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(indicator_memo, "record_worker", recorded.append)
    chroms = [_make_chromosome(1, 1), _make_chromosome(2, 1)]
    kwargs: dict[str, Any] = {"min_trades": 10, "timeframe": "1m"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = fitness.evaluate_population(
            chroms, _generic_fn, _generic_filters, max_workers=2, executor=pool, **kwargs
        )

    assert len(recorded) == len(chroms)
    assert all(snap["proc_id"] == indicator_memo.snapshot()["proc_id"] for snap in recorded)
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


def test_generic_pool_failed_task_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fitness, "_parallel_broken", False)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(indicator_memo, "record_worker", recorded.append)
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

    assert len(recorded) == 1  # only the chromosome that finished
    assert out[0].total_trades == 30
    assert out[1].error == "RuntimeError: worker exploded"
