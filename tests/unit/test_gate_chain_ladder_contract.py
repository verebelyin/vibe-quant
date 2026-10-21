"""T3 (vibe-quant-rabmo.6) characterization: the validation ladder.

Pins behaviour of ``DiscoveryPipeline._apply_validation_gates`` and the three
``evaluate_*`` helpers BEFORE rabmo.6 extracts ``gate_chain.py``: buy-hold ->
cross-window -> WFA -> holdout, fail-closed reason texts, and the result lists.

Characterization only -- every expectation is the code's current behaviour.
ALL private access lives in the ADAPTER BLOCK below; the rows call nothing but
the adapter, so T8/T9b can rewrite the adapter (``GateChain``) without touching
a single expectation. Backtest fns go ONLY through the ``DiscoveryPipeline``
constructor: T8's shims read construction-time evidence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import (
    CrossWindowResult,
    DiscoveryConfig,
    DiscoveryPipeline,
    DiscoveryResult,
    HoldoutResult,
    WFARollingResult,
)
from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest import LogCaptureFixture

    BacktestFn = Callable[[StrategyChromosome], dict[str, float | int]]
    BacktestFactory = Callable[[str, str], BacktestFn]


# ---------------------------------------------------------------------------
# ADAPTER BLOCK -- the ONLY place this file names a private member. T9b flips it
# to ``pipe.gates.evaluate_*``; the rows below never change.
# ---------------------------------------------------------------------------


def _make_pipe(
    config: DiscoveryConfig,
    backtest_fn: BacktestFn,
    *,
    holdout_backtest_fn: BacktestFn | None = None,
    backtest_fn_factory: BacktestFactory | None = None,
    buy_hold_closes_fn: Callable[[list[str]], dict[str, dict[int, float]]] | None = None,
) -> DiscoveryPipeline:
    """Build the pipeline with every fn passed through the CONSTRUCTOR."""
    return DiscoveryPipeline(
        config,
        backtest_fn,
        holdout_backtest_fn=holdout_backtest_fn,
        backtest_fn_factory=backtest_fn_factory,
        buy_hold_closes_fn=buy_hold_closes_fn,
    )


def _validate(pipe: DiscoveryPipeline, top: list[tuple[StrategyChromosome, FitnessResult]]) -> DiscoveryResult:
    """Run the full ladder (Gate 1b buy-hold -> cross-window -> WFA -> holdout)."""
    return pipe._apply_validation_gates(
        top,
        generations=[],
        total_evaluated=10,
        converged=False,
        convergence_gen=None,
    )


def _xwin(
    pipe: DiscoveryPipeline, top: list[tuple[StrategyChromosome, FitnessResult]]
) -> tuple[list[CrossWindowResult], str | None]:
    return pipe._evaluate_cross_windows(top)


def _wfa(
    pipe: DiscoveryPipeline,
    top: list[tuple[StrategyChromosome, FitnessResult]],
    range_start: str,
    range_end: str,
) -> tuple[list[WFARollingResult], str | None]:
    return pipe._evaluate_wfa_rolling(top, range_start, range_end)


def _holdout(
    pipe: DiscoveryPipeline, top: list[tuple[StrategyChromosome, FitnessResult]]
) -> list[HoldoutResult]:
    return pipe._evaluate_holdout(top)


def _buy_hold(
    pipe: DiscoveryPipeline, top: list[tuple[StrategyChromosome, FitnessResult]]
) -> list[tuple[StrategyChromosome, FitnessResult]]:
    return pipe._apply_buy_hold_gate(top)


def _holdout_min_trades(pipe: DiscoveryPipeline) -> int:
    return pipe.holdout_min_trades()


def _break_window_ranges(pipe: DiscoveryPipeline, name: str, message: str) -> None:
    """Make the gate-time window builder raise ``DataUnavailableError(message)``."""

    def raiser(*_args: object, **_kwargs: object) -> None:
        raise DataUnavailableError(message)

    setattr(pipe, name, raiser)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gene(
    ind: str,
    params: dict[str, float],
    cond: ConditionType = ConditionType.LT,
    thr: float = 30.0,
) -> StrategyGene:
    return StrategyGene(ind, params, cond, thr)


def _chrom(stop_loss_pct: float = 2.0, direction: Direction = Direction.LONG) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_gene("RSI", {"period": 14.0})],
        exit_genes=[_gene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=4.0,
        direction=direction,
    )


def _fit(**kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.8,
        "total_trades": 150,
        "total_return": 0.4,
        "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0,
        "sl_tp_penalty": 0.0,
        "raw_score": 0.6,
        "adjusted_score": 0.6,
        "passed_filters": True,
        "filter_results": {},
    }
    base.update(kw)
    return FitnessResult(**base)


def _cfg(**kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "population_size": 6,
        "max_generations": 2,
        "elite_count": 1,
        "top_k": 5,
        "min_trades": 50,
        "max_workers": None,
        "timeframe": "1d",
        "symbols": ["BTCUSDT"],
        "start_date": "2024-01-01",
        "end_date": "2025-01-01",
        "require_bootstrap_ci": False,
        "require_dsr": False,
        # Buy-hold is opt-out here: most rows exercise a later gate.
        "require_buy_hold": False,
    }
    base.update(kw)
    return DiscoveryConfig(**base)


def _nt() -> NTBacktestFn:
    return NTBacktestFn(["BTCUSDT"], "1d", "2024-01-01", "2025-01-01")


def _bt(**kw: Any) -> dict[str, float | int]:
    base: dict[str, float | int] = {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": 20,
        "total_return": 0.1,
    }
    base.update(kw)
    return base


def _top(*specs: tuple[float, FitnessResult]) -> list[tuple[StrategyChromosome, FitnessResult]]:
    """Candidates in the given order; the stop-loss identifies each row."""
    return [(_chrom(sl), fit) for sl, fit in specs]


# ---------------------------------------------------------------------------
# R37b (first test)
# ---------------------------------------------------------------------------


def test_holdout_sharpe_tie_rejects() -> None:
    """Holdout Sharpe == min needs a STRICT beat (tie rejects)."""
    cfg = _cfg(
        start_date="2024-01-01",
        end_date="2024-07-01",
        train_test_split=0.75,
        holdout_start_date="2024-07-01",
        holdout_end_date="2024-09-01",
        holdout_min_sharpe=0.5,
        holdout_min_trades=6,
    )
    pipe = _make_pipe(
        cfg,
        lambda c: _bt(),
        holdout_backtest_fn=lambda c: _bt(sharpe_ratio=0.5, total_return=0.1, total_trades=10),
    )
    res = _validate(pipe, _top((2.0, _fit())))
    assert res.top_strategies == []
    (rejection,) = res.guardrail_rejections
    assert rejection["stage"] == "holdout"
    assert rejection["reasons"] == ["Holdout: sharpe 0.50 <= 0.50"]


# ---------------------------------------------------------------------------
# R30-R33: buy-and-hold opens the ladder
# ---------------------------------------------------------------------------


def _reasons(res: DiscoveryResult) -> list[list[str]]:
    return [list(cast("list[str]", r["reasons"])) for r in res.guardrail_rejections]


def _stages(res: DiscoveryResult) -> list[str]:
    return [str(r["stage"]) for r in res.guardrail_rejections]


def test_order() -> None:
    """R30: a buy-hold reject never reaches cross-window / WFA / holdout."""
    factory_calls: list[tuple[str, str]] = []
    holdout_calls: list[str] = []

    def factory(ws: str, we: str) -> BacktestFn:
        factory_calls.append((ws, we))
        return lambda c: _bt()

    def holdout(c: StrategyChromosome) -> dict[str, float | int]:
        holdout_calls.append(c.uid)
        return _bt(total_return=0.1, total_trades=10)

    cfg = _cfg(
        cross_window_months=[1],
        wfa_oos_step_days=30,
        train_test_split=0.75,
        holdout_start_date="2024-07-01",
        holdout_end_date="2024-09-01",
        require_buy_hold=True,
    )
    pipe = _make_pipe(
        cfg,
        _nt(),
        holdout_backtest_fn=holdout,
        backtest_fn_factory=factory,
        buy_hold_closes_fn=lambda symbols: {},
    )
    res = _validate(pipe, _top((2.0, _fit()), (3.0, _fit())))

    assert res.top_strategies == []
    assert _stages(res) == ["buy_hold", "buy_hold"]
    assert _reasons(res) == [["Buy-hold: no daily series"]] * 2
    assert factory_calls == []
    assert holdout_calls == []


def test_buy_hold_skip_rows() -> None:
    """R31: only an NT backtest fn is enforced; off = no gate record at all."""
    top = _top((1.0, _fit()), (2.0, _fit()))

    on = _validate(_make_pipe(_cfg(require_buy_hold=True), lambda c: _bt()), top)
    assert [c.uid for c, _ in on.top_strategies] == [c.uid for c, _ in top]
    assert on.buy_hold_gate == {"enforced": False, "reason": "non-NT backtest fn"}
    assert on.guardrail_rejections == []

    off = _validate(_make_pipe(_cfg(require_buy_hold=False), lambda c: _bt()), top)
    assert [c.uid for c, _ in off.top_strategies] == [c.uid for c, _ in top]
    assert off.buy_hold_gate is None


def test_buy_hold_closes_error_fails_closed() -> None:
    """R32: a benchmark that cannot be built rejects every survivor."""
    def boom(_symbols: list[str]) -> dict[str, dict[int, float]]:
        raise RuntimeError("x")

    pipe = _make_pipe(_cfg(require_buy_hold=True), _nt(), buy_hold_closes_fn=boom)
    res = _validate(pipe, _top((1.0, _fit()), (2.0, _fit())))

    assert res.top_strategies == []
    assert _stages(res) == ["buy_hold", "buy_hold"]
    assert _reasons(res) == [["Buy-hold: benchmark closes unavailable: x"]] * 2
    assert res.buy_hold_gate == {"enforced": True, "error": "x"}


def test_buy_hold_empty() -> None:
    """R33: nothing to check -> enforced with no record and no closes load."""
    calls: list[list[str]] = []

    def closes(symbols: list[str]) -> dict[str, dict[int, float]]:
        calls.append(symbols)
        return {}

    pipe = _make_pipe(_cfg(require_buy_hold=True), _nt(), buy_hold_closes_fn=closes)
    res = _validate(pipe, [])

    assert res.buy_hold_gate == {"enforced": True}
    assert calls == []


# ---------------------------------------------------------------------------
# R34-R35: cross-window
# ---------------------------------------------------------------------------


def test_cross_window_rows() -> None:
    """R34: factory missing / tie-pass / fail / crash-window / missing-data."""
    missing = _validate(
        _make_pipe(_cfg(cross_window_months=[1]), lambda c: _bt()),
        _top((1.0, _fit()), (2.0, _fit())),
    )
    assert _stages(missing) == ["cross_window", "cross_window"]
    assert _reasons(missing) == [["Cross-window: no backtest factory"]] * 2

    # sharpe == min and return > 0 is a pass (sharpes compare inclusive).
    tie_pass = _make_pipe(
        _cfg(cross_window_months=[1]),
        lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (lambda c: _bt(sharpe_ratio=0.5, total_return=0.1)),
    )
    results, error = _xwin(tie_pass, _top((1.0, _fit())))
    assert error is None
    assert (results[0].windows_passed, results[0].total_windows, results[0].passed) == (1, 1, True)

    # Return 0 is never a pass.
    zero_return = _make_pipe(
        _cfg(cross_window_months=[1]),
        lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (lambda c: _bt(sharpe_ratio=3.0, total_return=0.0)),
    )
    results, error = _xwin(zero_return, _top((1.0, _fit())))
    assert error is None
    assert (results[0].windows_passed, results[0].passed) == (0, False)

    # A crashing window is a FAILED_WINDOW, not a pass and not a crash.
    def crashing(ws: str, we: str) -> BacktestFn:
        raise RuntimeError("x")

    crashed = _make_pipe(_cfg(cross_window_months=[1]), lambda c: _bt(), backtest_fn_factory=crashing)
    results, error = _xwin(crashed, _top((1.0, _fit())))
    assert error is None
    assert results[0].passed is False
    failed = results[0].window_results[0]
    assert (failed.sharpe_ratio, failed.total_return, failed.total_trades) == (0.0, 0.0, 0)

    # DataUnavailableError is NOT a window failure: it re-raises.
    def missing_data(ws: str, we: str) -> BacktestFn:
        raise DataUnavailableError("d")

    broken = _make_pipe(_cfg(cross_window_months=[1]), lambda c: _bt(), backtest_fn_factory=missing_data)
    with pytest.raises(DataUnavailableError, match="^d$"):
        _xwin(broken, _top((1.0, _fit())))


def test_cross_window_range_error_rejects_once() -> None:
    """R34b: DataUnavailableError from the range builder is the ValueError branch."""
    pipe = _make_pipe(
        _cfg(cross_window_months=[1]),
        lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (lambda c: _bt()),
    )
    _break_window_ranges(pipe, "cross_window_ranges", "d")
    res = _validate(pipe, _top((1.0, _fit()), (2.0, _fit())))

    assert res.top_strategies == []
    assert len(res.guardrail_rejections) == 2
    assert _stages(res) == ["cross_window", "cross_window"]
    assert _reasons(res) == [["d"], ["d"]]


def test_cross_window_required() -> None:
    """R34c: cross_window_min_pass decides; the fail text names the requirement."""
    cfg = _cfg(cross_window_months=[1, 2], cross_window_min_pass=1)

    def one_of_two(ws: str, we: str) -> BacktestFn:
        ret = 0.1 if ws == "2024-02-01" else -0.1
        return lambda c: _bt(sharpe_ratio=1.0, total_return=ret)

    res = _validate(_make_pipe(cfg, lambda c: _bt(), backtest_fn_factory=one_of_two), _top((1.0, _fit())))
    assert [c.stop_loss_pct for c, _ in res.top_strategies] == [1.0]
    (cwr,) = res.cross_window_results
    assert (cwr.windows_passed, cwr.total_windows, cwr.required, cwr.passed) == (1, 2, 1, True)

    none_of_two = _make_pipe(
        cfg, lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (lambda c: _bt(sharpe_ratio=1.0, total_return=-0.1)),
    )
    res = _validate(none_of_two, _top((1.0, _fit())))
    assert _reasons(res) == [
        ["Cross-window: 0/2 shifted windows passed (need 1; sharpe>=0.5 and return>0)"]
    ]


def test_cross_window_empty(caplog: LogCaptureFixture) -> None:
    """R35: requested with no survivors is a no-op plus a warning."""
    with caplog.at_level(logging.WARNING):
        res = _validate(_make_pipe(_cfg(cross_window_months=[1]), lambda c: _bt()), [])

    assert res.guardrail_rejections == []
    assert res.cross_window_results == []
    assert "Cross-window requested but no strategies survived guardrails" in caplog.text


# ---------------------------------------------------------------------------
# R36: rolling WFA
# ---------------------------------------------------------------------------


def test_wfa_rows() -> None:
    """R36: factory missing / empty range / consistency tie / unprofitable."""
    missing = _validate(
        _make_pipe(_cfg(wfa_oos_step_days=30), lambda c: _bt()),
        _top((1.0, _fit())),
    )
    assert _reasons(missing) == [["WFA: no backtest_fn_factory configured"]]
    assert _stages(missing) == ["wfa"]

    short = _validate(
        _make_pipe(
            _cfg(wfa_oos_step_days=30, start_date="2024-01-01", end_date="2024-01-10"),
            lambda c: _bt(),
            backtest_fn_factory=lambda ws, we: (lambda c: _bt()),
        ),
        _top((1.0, _fit())),
    )
    assert _reasons(short) == [["WFA: range 2024-01-01..2024-01-10 too short for 30d windows"]]

    # consistency == min is a pass (>=).
    cfg = _cfg(
        wfa_oos_step_days=30,
        start_date="2024-01-01",
        end_date="2024-03-01",
        wfa_min_consistency=0.5,
    )

    def half(ws: str, we: str) -> BacktestFn:
        return lambda c: _bt(total_return=0.1 if ws == "2024-01-01" else -0.1)

    res = _validate(_make_pipe(cfg, lambda c: _bt(), backtest_fn_factory=half), _top((1.0, _fit())))
    assert [c.stop_loss_pct for c, _ in res.top_strategies] == [1.0]
    (wfa,) = res.wfa_results
    assert (wfa.windows_profitable, wfa.total_windows, wfa.consistency, wfa.passed) == (1, 2, 0.5, True)

    # A 0.0-return window is NOT profitable.
    flat = _make_pipe(
        cfg, lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (lambda c: _bt(total_return=0.0)),
    )
    res = _validate(flat, _top((1.0, _fit())))
    assert _reasons(res) == [["WFA: 0/2 rolling windows profitable (0% < 50%)"]]


def test_wfa_range_error() -> None:
    """R36b: DataUnavailableError at gate time rejects via the ValueError branch."""
    pipe = _make_pipe(
        _cfg(wfa_oos_step_days=30),
        lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (lambda c: _bt()),
    )
    _break_window_ranges(pipe, "wfa_window_ranges", "d")
    res = _validate(pipe, _top((1.0, _fit()), (2.0, _fit())))

    assert res.top_strategies == []
    assert _stages(res) == ["wfa", "wfa"]
    assert _reasons(res) == [["d"], ["d"]]


# ---------------------------------------------------------------------------
# R37-R40: holdout + result lists
# ---------------------------------------------------------------------------


def _holdout_cfg(**kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "start_date": "2024-01-01",
        "end_date": "2024-07-01",
        "train_test_split": 0.75,
        "holdout_start_date": "2024-07-01",
        "holdout_end_date": "2024-09-01",
        "holdout_min_trades": 6,
    }
    base.update(kw)
    return _cfg(**base)


def test_holdout_rows() -> None:
    """R37: missing dates / missing fn / empty metrics / NaN / crash / missing data."""
    nan = float("nan")
    top = _top((1.0, _fit()))

    no_dates = _validate(
        _make_pipe(_cfg(train_test_split=0.8), lambda c: _bt()), top
    )
    assert _reasons(no_dates) == [["Holdout: no holdout dates"]]
    assert no_dates.holdout_min_trades is None

    no_fn = _validate(_make_pipe(_holdout_cfg(), lambda c: _bt()), top)
    assert _reasons(no_fn) == [["Holdout: no holdout backtest"]]
    assert no_fn.holdout_min_trades == 6

    empty = _validate(
        _make_pipe(_holdout_cfg(), lambda c: _bt(), holdout_backtest_fn=lambda c: {}), top
    )
    assert empty.holdout_min_trades == 6
    assert _reasons(empty) == [
        [
            "Holdout: 0 trades < 6",
            "Holdout: return 0.00% <= 0",
            "Holdout: sharpe 0.00 <= 0.00",
        ]
    ]

    nan_sharpe = _validate(
        _make_pipe(
            _holdout_cfg(),
            lambda c: _bt(),
            holdout_backtest_fn=lambda c: {
                "sharpe_ratio": nan,
                "total_return": 0.2,
                "total_trades": 10,
            },
        ),
        top,
    )
    assert _reasons(nan_sharpe) == [["Holdout: sharpe 0.00 <= 0.00"]]

    def crashing(c: StrategyChromosome) -> dict[str, float | int]:
        raise RuntimeError("x")

    crashed = _validate(_make_pipe(_holdout_cfg(), lambda c: _bt(), holdout_backtest_fn=crashing), top)
    assert _reasons(crashed) == _reasons(empty)

    def missing_data(c: StrategyChromosome) -> dict[str, float | int]:
        raise DataUnavailableError("d")

    with pytest.raises(DataUnavailableError, match="^d$"):
        _validate(_make_pipe(_holdout_cfg(), lambda c: _bt(), holdout_backtest_fn=missing_data), top)

    split_zero = _validate(_make_pipe(_cfg(), lambda c: _bt()), top)
    assert [c.stop_loss_pct for c, _ in split_zero.top_strategies] == [1.0]
    assert split_zero.holdout_min_trades is None
    assert split_zero.holdout_results == []


def test_holdout_min_trades() -> None:
    """R38: explicit floor, half the train-gate rate, and the 1-trade floor."""
    explicit = _make_pipe(_cfg(holdout_min_trades=7), lambda c: _bt())
    assert _holdout_min_trades(explicit) == 7

    derived = _make_pipe(
        _cfg(
            train_test_split=0.75,
            start_date="2024-01-01",
            end_date="2024-05-25",  # 145 d
            holdout_start_date="2024-05-25",
            holdout_end_date="2024-07-01",  # 37 d
        ),
        lambda c: _bt(),
    )
    assert _holdout_min_trades(derived) == 6

    zero_days = _make_pipe(
        _cfg(
            train_test_split=0.75,
            holdout_start_date="2024-06-01",
            holdout_end_date="2024-06-01",
        ),
        lambda c: _bt(),
    )
    assert _holdout_min_trades(zero_days) == 1


def test_result_lists_follow_survivors() -> None:
    """R40: result lists carry ONLY final survivors, in survivor order."""
    cfg = _cfg(
        start_date="2024-01-01",
        end_date="2024-03-01",
        train_test_split=0.75,
        holdout_start_date="2024-03-01",
        holdout_end_date="2024-05-01",
        cross_window_months=[1],
        wfa_oos_step_days=30,
        wfa_min_consistency=0.5,
        holdout_min_trades=6,
    )

    def holdout(c: StrategyChromosome) -> dict[str, float | int]:
        ret = -0.01 if c.stop_loss_pct == 2.0 else 0.1
        return _bt(sharpe_ratio=1.0, total_return=ret, total_trades=10)

    c1, c2, c3 = _chrom(1.0), _chrom(2.0), _chrom(3.0)
    pipe = _make_pipe(
        cfg,
        lambda c: _bt(),
        holdout_backtest_fn=holdout,
        backtest_fn_factory=lambda ws, we: (lambda c: _bt(sharpe_ratio=1.0, total_return=0.1)),
    )
    res = _validate(pipe, [(c3, _fit()), (c1, _fit()), (c2, _fit())])

    assert [c.uid for c, _ in res.top_strategies] == [c3.uid, c1.uid]
    # all three passed cross-window and WFA; only the two survivors are listed
    assert [r.passed for r in res.cross_window_results] == [True, True]
    assert [r.passed for r in res.wfa_results] == [True, True]
    assert [h.total_return for h in res.holdout_results] == [0.1, 0.1]
    assert _stages(res) == ["holdout"]
    assert _reasons(res) == [["Holdout: return -1.00% <= 0"]]
