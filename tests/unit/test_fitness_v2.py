"""Fitness v2: shrunk MTM t-stat scorer, capped loser band, full-window refusal rows."""

from __future__ import annotations

import math
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import (
    FitnessV2,
    _evaluate_single,
    evaluate_population,
    fitness_v2_own,
    passes_train_gate,
)
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.metrics import DailyReturns

from .test_fitness import _make_chromosome

K = 50
X = [0.01, -0.005, 0.002, 0.0, 0.003]


def _dr(values: list[float]) -> DailyReturns:
    return DailyReturns(19000, array("d", values))


def _own(
    values: list[float],
    trades: int = 51,
    windows: tuple[float, ...] | None = None,
    penalty: float = 0.0,
) -> tuple[float, float | None, float | None]:
    return fitness_v2_own(_dr(values), trades, K, windows, penalty)


def _seeded_series() -> tuple[list[float], list[float]]:
    """The reviewer's profitable (1 of 3 windows positive) and vol-drag loser series."""
    rng = np.random.default_rng(0)
    first = rng.normal(0.0, 0.01, 900)  # drawn to keep the stream aligned
    del first
    loser = rng.normal(0.0, 0.04, 900)
    loser = loser - loser.mean() + 0.0006
    profitable = rng.normal(0.0, 0.005, 900)
    profitable = profitable - profitable.mean() + 0.00002
    return profitable.tolist(), loser.tolist()


def test_t_stat_matches_numpy() -> None:
    own, t_stat, factor = _own(X)
    arr = np.asarray(X, dtype=np.float64)
    assert t_stat == np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(5)
    assert t_stat == 0.8233869695926183
    assert own == 0.5860532494377533
    assert factor == 1.0


def test_t_stat_is_independent_of_trades_and_penalty() -> None:
    other = [0.02, -0.01, 0.004, 0.001, -0.002, 0.006]
    assert _own(other, trades=51)[1] == _own(other, trades=400, penalty=0.2)[1]
    assert _own(other)[1] == 0.7781397750793556


def test_no_trade_cliff() -> None:
    at_51 = _own(X, trades=51)[0]
    at_49 = _own(X, trades=49)[0]
    assert at_51 == 0.5860532494377533
    assert at_49 == 0.5846373927513328
    assert 0.0 < at_49 < at_51


@pytest.mark.parametrize(
    "series",
    [
        None,
        [],
        [0.01],
        [0.0] * 30,
        [0.01] * 30,
        [0.01, float("nan"), 0.02],
        [0.01, float("inf"), 0.02],
        [1.7e308, 1.6e308, 1.7e308],  # finite values whose mean overflows: t is not finite
    ],
)
def test_degenerate_inputs_score_zero(series: list[float] | None) -> None:
    sr = None if series is None else _dr(series)
    assert fitness_v2_own(sr, 51, K, None, 0.0) == (0.0, None, None)


def test_zero_trades_score_zero_for_any_series() -> None:
    assert _own(X, trades=0) == (0.0, None, None)
    assert _own(X, trades=-3) == (0.0, None, None)
    assert _own(X, trades=1)[0] > 0.0


def test_loser_band_below_profitable() -> None:
    # mean > 0 but compound -0.19 (vol drag): a loser whatever its t-stat says
    drag = [0.5, -0.4, 0.5, -0.4]
    own_drag, t_drag, _ = _own(drag)
    assert t_drag is not None and t_drag > 0
    assert own_drag == 0.11523258713932803
    assert own_drag < 0.225 < _own(X)[0]
    assert _own([-v for v in X])[0] == 0.10084840953907734
    # compound exactly 0.0 is a loser, not a profitable genome
    flat = [1.0, -0.5]
    assert float(np.prod(1.0 + np.asarray(flat))) - 1.0 == 0.0
    assert _own(flat)[0] == 0.11723111882012084
    assert _own(flat)[0] < _own(X)[0]


def test_loser_band_cap_follows_window_count() -> None:
    drag = [0.5, -0.4, 0.5, -0.4]
    # factor (N-1)/N times the cap L = 0.5 * 0.45 / (N-1) stays below the profitable floor 0.45 * 0.5 / N
    for n in (2, 3, 5, 8):
        windows = tuple([0.01] * (n - 1) + [-0.01])
        loser = _own(drag, windows=windows)[0]
        floor = (1.0 / n) * 0.45 * 0.5
        assert loser < floor
        # ... and the loser is not squashed to 0 (still a gradient)
        assert loser > 0.0


def test_loser_below_profitable_across_window_factors() -> None:
    profitable, loser = _seeded_series()
    assert float(np.prod(1.0 + np.asarray(profitable))) - 1.0 > 0
    assert float(np.prod(1.0 + np.asarray(loser))) - 1.0 < 0
    own_p, t_p, f_p = _own(profitable, trades=100, windows=(0.02, -0.01, -0.005))
    own_l, t_l, f_l = _own(loser, trades=100, windows=(0.05, 0.04, -0.3))
    assert (own_p, t_p, f_p) == (0.17310825165318944, 0.12061004904429513, 1 / 3)
    assert (own_l, t_l, f_l) == (0.04025256054661289, 0.44120320021423554, 2 / 3)
    assert own_l < own_p
    # the loser's own t-stat is higher: only the band keeps it below
    assert t_l is not None and t_p is not None and t_l > t_p


def test_window_factor() -> None:
    own3, _, f3 = _own(X, windows=(0.01, 0.02, 0.03), penalty=0.02)
    own2, _, f2 = _own(X, windows=(0.01, -0.02, 0.03), penalty=0.02)
    assert own3 == 0.5743321844489983
    assert own2 == 0.38288812296599883
    assert (f3, f2) == (1.0, 2 / 3)
    assert own2 == (2 / 3) * own3
    # zero and NaN are non-positive; the floor is one window
    assert _own(X, windows=(0.0, 0.0, 0.0))[2] == 1 / 3
    assert _own(X, windows=(0.01, float("nan"), 0.03))[2] == 2 / 3
    assert _own(X, windows=(-1.0, -1.0, -1.0, -1.0))[2] == 1 / 4
    assert _own(X, windows=(1e-300, 1e-300))[2] == 1.0
    assert _own(X, windows=(1e-300, 0.0))[2] == 0.5
    # one window or none: no factor
    assert _own(X, windows=(-0.05,))[2] == 1.0
    assert _own(X, windows=())[2] == 1.0
    assert _own(X, windows=None)[2] == 1.0


def _series_for(chrom_trades: int, level: float) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    values = (rng.normal(level, 0.01, 200)).tolist()
    return {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": chrom_trades,
        "total_return": 0.1,
        "mtm_daily_returns": _dr(values),
    }


def test_penalties_multiply_at_caps() -> None:
    genes = [
        StrategyGene("RSI", {"period": 14.0}, ConditionType.CROSSES_ABOVE, 30.0) for _ in range(7)
    ]
    chrom = StrategyChromosome(
        entry_genes=genes[:4],
        exit_genes=genes[4:],
        stop_loss_pct=10.0,
        take_profit_pct=0.5,
        direction=Direction.LONG,
    )
    bt = _series_for(1000, 0.002)
    fr = _evaluate_single(chrom, lambda _c: dict(bt), None, K, "4h", fitness="v2")
    assert (fr.complexity_penalty, fr.overtrade_penalty, fr.sl_tp_penalty) == (0.1, 0.3, 0.15)
    clean = fitness_v2_own(bt["mtm_daily_returns"], 1000, K, None, 0.0)[0]
    penalty = fr.complexity_penalty + fr.overtrade_penalty + fr.sl_tp_penalty
    assert fr.adjusted_score == (1.0 - penalty) * clean
    assert fr.adjusted_score == pytest.approx(0.45 * clean, rel=1e-12)


def test_range_and_monotone() -> None:
    rng = np.random.default_rng(42)
    for _ in range(500):
        n = int(rng.integers(2, 400))
        scale = float(rng.choice([0.001, 0.01, 0.2, 0.99, 1e6]))
        values = rng.normal(0.0, scale, n)
        if scale >= 0.99:
            values = np.clip(values, -0.99, None)
        own, _t, _ = fitness_v2_own(
            _dr(values.tolist()), int(rng.integers(0, 5001)), K, None, float(rng.uniform(0, 0.55))
        )
        assert math.isfinite(own)
        assert 0.0 <= own <= 1.0

    # monotone in t at fixed trades / penalty / factor, inside each band
    profit: list[tuple[float, float]] = []
    lose: list[tuple[float, float]] = []
    for level in np.linspace(-0.004, 0.006, 60):
        base = np.random.default_rng(11).normal(0.0, 0.01, 250)
        values = (base - base.mean() + level).tolist()
        own, t_stat, _ = _own(values, trades=120, windows=(0.01, 0.02), penalty=0.1)
        assert t_stat is not None
        compound = float(np.prod(1.0 + np.asarray(values))) - 1.0
        (profit if compound > 0 else lose).append((t_stat, own))
    assert profit and lose
    for band in (profit, lose):
        owns = [o for _t, o in sorted(band)]
        assert owns == sorted(owns)
        assert len(set(owns)) == len(owns)
    assert max(o for _t, o in lose) < min(o for _t, o in profit)


# --- _evaluate_single wiring -------------------------------------------------------------


def _bt(
    series: list[float] | None,
    trades: int = 80,
    ret: float = 0.05,
    window_returns: tuple[float, ...] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": trades,
        "total_return": ret,
    }
    if series is not None:
        out["mtm_daily_returns"] = _dr(series)
    if window_returns is not None:
        out["window_returns"] = window_returns
    out.update(extra)
    return out


def _score(bt: dict[str, Any], chrom: StrategyChromosome | None = None, **kw: Any):  # type: ignore[no-untyped-def]
    return _evaluate_single(chrom or _make_chromosome(), lambda _c: bt, None, K, "4h", **kw)


def test_v2_forwards_series_trades_and_windows() -> None:
    # two unrelated inputs, literal expected values (1 entry + 1 exit gene: no complexity,
    # no overtrade, SL/TP ratio 0.5 -> no penalty)
    first = _score(_bt(X, trades=51), fitness="v2")
    assert first.adjusted_score == 0.5860532494377533
    assert first.fitness_v2 == FitnessV2(0.5860532494377533, 0.8233869695926183, 1.0)
    second = _score(_bt(X, trades=49, window_returns=(0.01, -0.02, 0.03)), fitness="v2")
    assert second.adjusted_score == (2 / 3) * 0.5846373927513328
    assert second.fitness_v2 is not None and second.fitness_v2.window_factor == 2 / 3
    other = [0.02, -0.01, 0.004, 0.001, -0.002, 0.006]
    third = _score(_bt(other, trades=100), fitness="v2")
    assert third.adjusted_score == _own(other, trades=100)[0]
    assert third.adjusted_score != first.adjusted_score


def test_v1_default_has_no_v2_fields_and_ignores_series() -> None:
    with_series = _score(_bt(X, window_returns=(0.01, 0.02)))
    without = _score(_bt(None))
    assert with_series.fitness_v2 is None and without.fitness_v2 is None
    assert with_series.mtm_daily_returns is not None  # a train-gate passer keeps its series
    assert replace(with_series, mtm_daily_returns=None) == without
    assert without == _score(_bt(None), fitness="v1")
    # v1 still zeroes a genome with 49 trades (the cliff v2 removes)
    assert _score(_bt(None, trades=49)).adjusted_score == 0.0


def test_unknown_fitness_is_refused() -> None:
    with pytest.raises(ValueError, match="fitness"):
        _score(_bt(X), fitness="v3")
    with pytest.raises(ValueError, match="fitness"):
        _score(_bt(X), fitness="V2")


def test_v2_refuses_early_exit_bt() -> None:
    bt = _bt(X, early_exit=0)
    fr = _score(bt, fitness="v2")
    assert fr.error == "FitnessV2Error: early exit under fitness v2"
    assert fr.adjusted_score == 0.0
    # v1 still scores an early-exit bt as today (no error)
    assert _score(bt).error is None


def test_v2_refuses_early_exit_in_worst_mode_symbol() -> None:
    sym_ok = _bt(None, trades=80, ret=0.1)
    sym_exit = _bt(None, trades=10, ret=-0.02, early_exit=0)
    bt = _bt(
        X,
        trades=10,
        symbol_results={"ETHUSDT": sym_ok, "BTCUSDT": sym_exit},
        symbol_mtm_daily={"ETHUSDT": _dr(X), "BTCUSDT": None},
        synthetic_symbols=["BTCUSDT"],
    )
    fr = _score(bt, fitness="v2")
    assert fr.error == "FitnessV2Error: early exit under fitness v2"
    assert fr.adjusted_score == 0.0
    assert fr.symbol_scores is None


def test_v2_missing_series_fails_loud() -> None:
    fr = _score(_bt(None, trades=80), fitness="v2")
    assert fr.error == "FitnessV2Error: no MTM series"
    assert fr.adjusted_score == 0.0
    # trades 0 and no series: a plain zero, no error
    quiet = _score(_bt(None, trades=0, ret=0.0), fitness="v2")
    assert quiet.error is None
    assert quiet.adjusted_score == 0.0
    # a bt that carries its own error (crashed backtest) keeps that error text
    crashed = _score(_bt(None, trades=0, ret=0.0, error="RuntimeError: boom"), fitness="v2")
    assert crashed.error == "RuntimeError: boom"
    # worst mode: a symbol with trades but no series fails the whole row
    worst = _bt(
        X,
        symbol_results={"BTCUSDT": _bt(None, trades=80), "ETHUSDT": _bt(None, trades=80)},
        symbol_mtm_daily={"BTCUSDT": _dr(X), "ETHUSDT": None},
    )
    assert _score(worst, fitness="v2").error == "FitnessV2Error: no MTM series"


def test_bankrupt_genome_scores_a_quiet_zero_not_a_missing_series_error() -> None:
    """Trades but no MTM series because the balance hit zero (return <= -1.0): v1 parity, a plain 0."""
    for ret in (-1.0, -1.5):
        fr = _score(_bt(None, trades=80, ret=ret), fitness="v2")
        assert fr.error is None
        assert fr.adjusted_score == 0.0
        assert fr.fitness_v2 == FitnessV2(0.0, None, None)
    # the edge: anything above -1.0 (or no return at all) still has a series to lose
    assert (
        _score(_bt(None, trades=80, ret=-0.9999), fitness="v2").error
        == "FitnessV2Error: no MTM series"
    )
    assert _score(_bt(None, trades=80, ret=float("nan")), fitness="v2").error == (
        "FitnessV2Error: no MTM series"
    )
    # worst mode: the bankrupt symbol scores 0, its sibling keeps its own score
    bt = _bt(
        X,
        symbol_results={
            "BTCUSDT": _bt(None, trades=80, ret=0.1),
            "ETHUSDT": _bt(None, trades=80, ret=-1.0),
        },
        symbol_mtm_daily={"BTCUSDT": _dr(X), "ETHUSDT": None},
    )
    worst = _score(bt, fitness="v2")
    assert worst.error is None
    assert worst.symbol_scores == {"BTCUSDT": _own(X, trades=80)[0], "ETHUSDT": 0.0}
    sym_ok = _bt(None, trades=80, ret=0.1)
    ok_ret = _bt(
        X, symbol_results={"BTCUSDT": sym_ok, "ETHUSDT": _bt(None, trades=80, ret=-0.9999)},
        symbol_mtm_daily={"BTCUSDT": _dr(X), "ETHUSDT": None},
    )  # fmt: skip
    assert _score(ok_ret, fitness="v2").error == "FitnessV2Error: no MTM series"


def test_worst_mode_soft_over_v2() -> None:
    first = [0.02, -0.01, 0.004, 0.001, -0.002, 0.006]
    second = [0.001, -0.002, 0.0015, 0.0005, 0.002, -0.0008]
    own_btc = _own(first, trades=80)[0]
    own_eth = _own(second, trades=60)[0]
    assert 0.0 < own_eth < own_btc
    bt = _bt(
        first,
        trades=60,
        symbol_results={
            "BTCUSDT": _bt(None, trades=80, ret=0.1),
            "ETHUSDT": _bt(None, trades=60, ret=0.02),
        },
        symbol_mtm_daily={"BTCUSDT": _dr(first), "ETHUSDT": _dr(second)},
    )
    fr = _score(bt, fitness="v2")
    assert fr.symbol_scores == {"BTCUSDT": own_btc, "ETHUSDT": own_eth}
    assert fr.adjusted_score == 0.5 * own_eth + 0.5 * ((own_btc + own_eth) / 2)
    assert fr.symbol_stats == {"BTCUSDT": (0.1, 80), "ETHUSDT": (0.02, 60)}
    # the reported t-stat / factor are the worst symbol's
    assert fr.fitness_v2 == FitnessV2(fr.adjusted_score, _own(second, trades=60)[1], 1.0)
    # a symbol's own window returns reach its factor
    windowed = _bt(
        first,
        trades=60,
        symbol_results={
            "BTCUSDT": _bt(None, trades=80, ret=0.1, window_returns=(0.1, 0.1)),
            "ETHUSDT": _bt(None, trades=60, ret=0.02, window_returns=(0.1, -0.1)),
        },
        symbol_mtm_daily={"BTCUSDT": _dr(first), "ETHUSDT": _dr(second)},
    )
    half = _score(windowed, fitness="v2")
    assert half.symbol_scores is not None
    assert half.symbol_scores["ETHUSDT"] == 0.5 * own_eth
    assert half.symbol_scores["BTCUSDT"] == own_btc


def test_pooled_v2() -> None:
    pooled = [0.02, -0.01, 0.004, 0.001, -0.002, 0.006]
    stats = {"BTCUSDT": (0.1, 40), "ETHUSDT": (0.05, 40), "SOLUSDT": (0.02, 40)}
    bt = _bt(pooled, trades=120, window_returns=(0.1, -0.1), pooled_symbol_stats=stats)
    fr = _score(bt, fitness="v2")
    assert fr.adjusted_score == _own(pooled, trades=120, windows=(0.1, -0.1))[0]
    assert fr.pooled_symbol_stats == stats
    # one coin under the floor max(5, 50 // 6) = 8 trades: own is 0 (the coin floor stays hard)
    thin = dict(stats, SOLUSDT=(0.02, 7))
    zero = _score(dict(bt, pooled_symbol_stats=thin), fitness="v2")
    assert zero.adjusted_score == 0.0
    assert zero.fitness_v2 == FitnessV2(0.0, None, None)
    # exactly at the floor it scores
    edge = dict(stats, SOLUSDT=(0.02, 8))
    assert _score(dict(bt, pooled_symbol_stats=edge), fitness="v2").adjusted_score > 0.0


def test_train_gate_unchanged() -> None:
    # windows (-0.01, +0.05), chained compound +0.04: v2 scores it, the train gate still fails it
    values = [0.02, -0.01, 0.004, 0.001, -0.002, 0.006]
    bt = _bt(values, trades=80, ret=-0.01, window_returns=(-0.01, 0.05))
    fr = _score(bt, fitness="v2")
    assert fr.adjusted_score > 0.0
    assert not passes_train_gate(fr, K)
    assert fr.mtm_daily_returns is None  # no DSR series kept for a non-candidate
    ok = _score(_bt(values, trades=80, ret=0.04, window_returns=(0.01, 0.05)), fitness="v2")
    assert passes_train_gate(ok, K)
    assert ok.mtm_daily_returns is not None


def test_fitness_flows_through_evaluate_population() -> None:
    chroms = [_make_chromosome(), _make_chromosome(2, 1)]
    bt = _bt(X, trades=51)
    sequential = evaluate_population(
        chroms, lambda _c: dict(bt), None, min_trades=K, timeframe="4h", fitness="v2"
    )
    assert [r.fitness_v2 is not None for r in sequential] == [True, True]
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = evaluate_population(
            chroms,
            lambda _c: dict(bt),
            None,
            max_workers=2,
            executor=pool,
            min_trades=K,
            timeframe="4h",
            fitness="v2",
        )
    assert parallel[0].adjusted_score == 0.5860532494377533
    assert [r.fitness_v2 is not None for r in parallel] == [True, True]
    default = evaluate_population(chroms, lambda _c: dict(bt), None, min_trades=K, timeframe="4h")
    assert [r.fitness_v2 for r in default] == [None, None]


class _SplitFn(NTBacktestFn):
    """Worst mode over two symbols, one task per (chromosome, symbol)."""

    def __init__(self) -> None:
        super().__init__(
            ["BTCUSDT", "ETHUSDT"],
            "4h",
            "2024-01-01",
            "2024-06-01",
            symbol_agg="worst",
            min_trades=K,
            collect_daily_returns=True,
        )

    def _run_single(self, chromosome, start_date, end_date, symbols=None):  # type: ignore[no-untyped-def]
        assert symbols is not None
        values = X if symbols[0] == "BTCUSDT" else [0.02, -0.01, 0.004, 0.001, -0.002, 0.006]
        out = _bt(None, trades=80, ret=0.1)
        out["mtm_daily_returns"] = _dr(values)
        return out


def test_fitness_flows_through_split_worst_mode() -> None:
    fn = _SplitFn()
    chrom = _make_chromosome()
    with ThreadPoolExecutor(max_workers=2) as pool:
        (fr,) = evaluate_population(
            [chrom],
            fn,
            None,
            max_workers=2,
            executor=pool,
            min_trades=K,
            timeframe="4h",
            fitness="v2",
        )
    assert fr.error is None
    # the runner's chained series drops the first day (the day vs the starting balance)
    assert fr.symbol_scores == {
        "BTCUSDT": _own(X[1:], trades=80)[0],
        "ETHUSDT": _own([-0.01, 0.004, 0.001, -0.002, 0.006], trades=80)[0],
    }
    assert fr == _evaluate_single(chrom, fn, None, K, "4h", fitness="v2")
