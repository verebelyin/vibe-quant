"""Early exit of multi-window discovery eval."""

from __future__ import annotations

import random
from array import array
from types import SimpleNamespace
from typing import Any

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import _evaluate_single
from vibe_quant.metrics import DailyReturns

NAN = float("nan")
WINDOWS = [("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-06-30")]


def _m(ret: float, trades: int = 50, sharpe: float = 1.0, pf: float = 1.5) -> dict[str, Any]:
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": 0.1,
        "profit_factor": pf,
        "total_trades": trades,
        "total_return": ret,
        "skewness": 0.0,
        "kurtosis": 3.0,
        "trade_returns": (0.01,),
    }


def _fn(
    monkeypatch: Any, metrics: list[Any], min_trades: int = 60
) -> tuple[NTBacktestFn, list[int]]:
    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2025-06-30", windows=WINDOWS, min_trades=min_trades
    )
    calls: list[int] = []

    def fake(self: Any, chrom: Any, ws: str, we: str) -> dict[str, Any]:
        calls.append(1)
        m = metrics[len(calls) - 1]
        if isinstance(m, Exception):
            raise m
        return m

    monkeypatch.setattr(NTBacktestFn, "_run_single", fake)
    return fn, calls


CHROM = SimpleNamespace(
    uid="x",
    entry_genes=[],
    exit_genes=[],
    short_entry_genes=[],
    short_exit_genes=[],
    stop_loss_pct=1.0,
    take_profit_pct=2.0,
    direction="long",
)


def test_stops_after_window_with_nonpositive_return(monkeypatch: Any) -> None:
    fn, calls = _fn(monkeypatch, [_m(-0.1), _m(0.2), _m(0.2)])
    out = fn(CHROM)  # type: ignore[arg-type]
    assert len(calls) == 1
    assert out["early_exit"] == 0
    assert out["total_return"] == -0.1  # real return kept for progress display
    assert out["window_trades"] == (50,)
    assert out["total_trades"] == 50


def test_stops_after_window_below_trade_gate(monkeypatch: Any) -> None:
    # gate = max(1, 60 // 6) = 10
    fn, calls = _fn(monkeypatch, [_m(0.2), _m(0.2, trades=3), _m(0.2)])
    out = fn(CHROM)  # type: ignore[arg-type]
    assert len(calls) == 2
    assert out["early_exit"] == 1
    assert out["window_trades"] == (50, 3)


def test_nan_return_counts_as_nonpositive(monkeypatch: Any) -> None:
    fn, calls = _fn(monkeypatch, [_m(NAN), _m(0.2), _m(0.2)])
    out = fn(CHROM)  # type: ignore[arg-type]
    assert len(calls) == 1
    assert out["early_exit"] == 0


def test_all_windows_run_when_each_passes(monkeypatch: Any) -> None:
    ms = [_m(0.1), _m(0.2, sharpe=2.0), _m(0.3)]
    fn, calls = _fn(monkeypatch, ms)
    out = fn(CHROM)  # type: ignore[arg-type]
    assert len(calls) == 3
    assert "early_exit" not in out
    assert out == NTBacktestFn._aggregate_multi_window(ms, 60)


def test_fitness_identical_to_full_eval(monkeypatch: Any) -> None:
    rng = random.Random(7)
    for _ in range(200):
        ms = [
            _m(
                rng.choice([-0.2, 0.0, NAN, 0.05, 0.3]),
                trades=rng.choice([0, 3, 9, 10, 40]),
                sharpe=rng.uniform(-1, 3),
                pf=rng.choice([0.5, 1.5, NAN]),
            )
            for _ in range(3)
        ]
        fn, calls = _fn(monkeypatch, ms)
        early = fn(CHROM)  # type: ignore[arg-type]
        full = NTBacktestFn._aggregate_multi_window(ms, 60)
        e = _evaluate_single(CHROM, lambda c, r=early: r, min_trades=60)  # type: ignore[arg-type]
        f = _evaluate_single(CHROM, lambda c, r=full: r, min_trades=60)  # type: ignore[arg-type]
        assert e.adjusted_score == f.adjusted_score
        w0 = ms[0]
        w0_fails = not (float(w0["total_return"]) > 0) or w0["total_trades"] < 10
        if w0_fails:
            assert len(calls) == 1
            assert early["early_exit"] == 0


def test_error_window_still_returns_error(monkeypatch: Any) -> None:
    fn, calls = _fn(monkeypatch, [_m(0.2), RuntimeError("boom"), _m(0.2)])
    out = fn(CHROM)  # type: ignore[arg-type]
    assert "RuntimeError: boom" in str(out["error"])
    assert "early_exit" not in out
    assert len(calls) == 2


TWO = [("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")]


def _two_window_fn(
    monkeypatch: Any, metrics: list[Any], *, full: bool, collect: bool = False
) -> tuple[NTBacktestFn, list[int]]:
    fn = NTBacktestFn(
        ["BTCUSDT"],
        "4h",
        "2024-01-01",
        "2024-12-31",
        windows=TWO,
        min_trades=60,
        collect_daily_returns=collect,
    )
    fn.full_windows = full
    calls: list[int] = []

    def fake(self: Any, chrom: Any, ws: str, we: str) -> dict[str, Any]:
        calls.append(1)
        return dict(metrics[len(calls) - 1])

    monkeypatch.setattr(NTBacktestFn, "_run_single", fake)
    return fn, calls


def test_full_windows_runs_every_window(monkeypatch: Any) -> None:
    full_fn, full_calls = _two_window_fn(monkeypatch, [_m(-0.01), _m(0.03)], full=True)
    out = full_fn(CHROM)  # type: ignore[arg-type]
    assert len(full_calls) == 2
    assert "early_exit" not in out
    assert out["window_returns"] == (-0.01, 0.03)
    assert out["window_trades"] == (50, 50)
    # a different pair of returns reaches the output unchanged
    other, _ = _two_window_fn(monkeypatch, [_m(0.2), _m(-0.4)], full=True)
    assert other(CHROM)["window_returns"] == (0.2, -0.4)  # type: ignore[arg-type]
    # default (False): today's early exit
    default_fn, default_calls = _two_window_fn(monkeypatch, [_m(-0.01), _m(0.03)], full=False)
    early = default_fn(CHROM)  # type: ignore[arg-type]
    assert len(default_calls) == 1
    assert early["early_exit"] == 0
    assert "window_returns" not in early
    assert NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-12-31").full_windows is False


def test_full_windows_runs_past_a_trade_gate_stop(monkeypatch: Any) -> None:
    # window 0 has 3 trades (< max(1, 60 // 4) = 15): v1 stops, full_windows aggregates the failure
    fn, calls = _two_window_fn(monkeypatch, [_m(0.2, trades=3), _m(0.1)], full=True)
    out = fn(CHROM)  # type: ignore[arg-type]
    assert len(calls) == 2
    assert "early_exit" not in out
    assert out["window_trades"] == (3, 50)
    assert out["window_returns"] == (0.2, 0.1)
    assert out["total_return"] == 0.0  # the per-window floor failure is unchanged


def test_full_windows_chains_series_on_the_floor_failure_branch(monkeypatch: Any) -> None:
    def with_series(ret: float, trades: int, first_day: int, values: list[float]) -> dict[str, Any]:
        return {
            **_m(ret, trades=trades),
            "mtm_daily_returns": DailyReturns(first_day, array("d", values)),
        }

    fn, _ = _two_window_fn(
        monkeypatch,
        [with_series(0.2, 3, 100, [0.0, 0.01, 0.02]), with_series(-0.1, 50, 103, [0.03, -0.01])],
        full=True,
        collect=True,
    )
    out = fn(CHROM)  # type: ignore[arg-type]
    series = out["mtm_daily_returns"]
    assert isinstance(series, DailyReturns)
    assert list(series.values) == [0.01, 0.02, 0.03, -0.01]  # window 0 drops its day 0
    assert out["window_returns"] == (0.2, -0.1)


def test_aggregate_emits_window_returns_on_both_branches() -> None:
    nan_window = _m(NAN)
    normal = NTBacktestFn._aggregate_multi_window([_m(0.1), _m(0.3)], 60)
    assert normal["window_returns"] == (0.1, 0.3)
    assert normal["total_return"] == 0.1
    kept = NTBacktestFn._aggregate_multi_window([_m(0.1), nan_window, _m(-0.2)], 60)[
        "window_returns"
    ]
    assert isinstance(kept, tuple)
    assert kept[0] == 0.1 and kept[2] == -0.2 and kept[1] != kept[1]  # NaN kept, not coerced
    # floor-failure branch (window 1 has 3 trades < 15): raw returns, every other key as before
    failed = NTBacktestFn._aggregate_multi_window([_m(0.5), _m(0.4, trades=3)], 60)
    assert failed["window_returns"] == (0.5, 0.4)
    assert failed["total_return"] == 0.0
    assert failed["sharpe_ratio"] == -1.0
    assert failed["window_trades"] == (50, 3)
    assert failed["total_trades"] == 53
