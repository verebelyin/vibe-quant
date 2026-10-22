"""Power test T1b: the REAL discovery gate chain, one candidate at a time.

``scripts/research/power/gatechain.py`` scores ONE candidate on EVERY real repo
gate (guardrail floor + per-N DSR + buy-hold + holdout + cross-window + WFA) and
reports the per-gate marginals plus the chain verdicts, including the
leave-one-out ``chain_recent_minus_bh`` and the leave-two-out
``chain_recent_minus_dsr_bh``. Every number asserted here is the contract's
literal from ``design-power.md`` §6.

Nothing here re-implements gate math: the module and these tests call the same
repo functions (``apply_discovery_dsr_returns``, ``apply_guardrails``,
``GateChain.screen``, ``_aggregate_multi_window`` + ``_evaluate_single``,
``buy_hold_gate``, ``GateChain.evaluate_cross_windows``,
``GateChain.evaluate_wfa_rolling``).
"""

from __future__ import annotations

import sys
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import gatechain  # noqa: E402

from vibe_quant.discovery import run_spec as discovery_cli  # noqa: E402
from vibe_quant.discovery.backtest_fn import NTBacktestFn  # noqa: E402
from vibe_quant.discovery.buyhold import buy_hold_daily, buy_hold_gate, daily_sharpe  # noqa: E402
from vibe_quant.discovery.fitness import FitnessResult, _const_backtest, _evaluate_single  # noqa: E402
from vibe_quant.discovery.gate_chain import (  # noqa: E402
    GateChain,
    GateEvidence,
    TrialBudget,
    holdout_reasons,
)
from vibe_quant.discovery.guardrails import GuardrailConfig, apply_guardrails  # noqa: E402
from vibe_quant.discovery.operators import (  # noqa: E402
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline, HoldoutResult  # noqa: E402
from vibe_quant.metrics import DailyReturns  # noqa: E402
from vibe_quant.overfitting.bootstrap_sharpe import bootstrap_sharpe_ci  # noqa: E402

#: R1's contract series: ``np.random.default_rng(11).normal(0.0005, 0.01, 875)``.
R1_VALUES = list(np.random.default_rng(11).normal(0.0005, 0.01, 875))

FIRST_DAY = 19_000
N_DAYS = 875


def _null_backtest(_chrom: object) -> dict[str, float | int]:
    return {}


def _chrom(direction: Direction = Direction.LONG) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=direction,
    )


def _fit(**kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": 60,
        "total_return": 0.3,
        "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0,
        "sl_tp_penalty": 0.0,
        "raw_score": 0.5,
        "adjusted_score": 0.5,
        "passed_filters": True,
        "filter_results": {},
        "trade_returns": tuple([0.01] * 60),
    }
    base.update(kw)
    return FitnessResult(**base)


def _series(daily_sharpe_val: float, n: int, seed: int, sigma: float = 0.01) -> DailyReturns:
    """Gaussian daily returns standardized to exactly ``daily_sharpe_val``."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    x = z * sigma + daily_sharpe_val * sigma
    return DailyReturns(FIRST_DAY, array("d", x.tobytes()))


def _closes(returns: list[float], first_day: int = FIRST_DAY) -> dict[str, dict[int, float]]:
    """``{BTCUSDT: {day: close}}`` whose 1d returns start on ``first_day``."""
    close = 100.0
    out: dict[int, float] = {first_day - 1: close}
    for i, ret in enumerate(returns):
        close *= 1.0 + ret
        out[first_day + i] = close
    return {"BTCUSDT": out}


def _flat_closes(n: int) -> dict[str, dict[int, float]]:
    """A near-zero-drift benchmark: alternating +-0.1% days."""
    return _closes([0.001 if i % 2 == 0 else -0.001 for i in range(n)])


def _strong_closes(n: int) -> dict[str, dict[int, float]]:
    """A ~0.2 daily Sharpe benchmark: +-0.1% days on a +0.02% drift."""
    return _closes([0.0002 + (0.001 if i % 2 == 0 else -0.001) for i in range(n)])


def _window_factory(
    metrics: dict[tuple[str, str], dict[str, float | int]],
    default: dict[str, float | int],
) -> Any:
    """A ``(start, end) -> chrom -> bt`` factory with per-window metrics."""

    def factory(start: str, end: str) -> Any:
        return _const_backtest(metrics.get((start, end), default))

    return factory


def _flat_window() -> dict[str, float | int]:
    return {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": 20,
        "total_return": 0.05,
    }


def _pass_holdout() -> dict[str, float | int]:
    return {
        "sharpe_ratio": 0.5,
        "max_drawdown": 0.1,
        "profit_factor": 1.2,
        "total_trades": 10,
        "total_return": 0.05,
    }


def test_dsr_verdict_uses_real_function() -> None:
    passed167, rec167 = gatechain.dsr_verdict(R1_VALUES, 167)
    assert rec167["sharpe_daily"] == pytest.approx(0.07112153153357098, rel=1e-12)
    assert rec167["p_value"] == pytest.approx(0.7266370087847579, rel=1e-12)
    assert passed167 is False

    passed1, rec1 = gatechain.dsr_verdict(R1_VALUES, 1)
    assert rec1["p_value"] == pytest.approx(0.01786782317136858, rel=1e-12)
    assert passed1 is True


# ---------------------------------------------------------------------------
# R3: the plan windows of W3 / W6
# ---------------------------------------------------------------------------


def test_settings_follow_plan_windows() -> None:
    w3 = gatechain.settings_for("W3")
    assert (w3.plan.train_start, w3.plan.train_end) == ("2023-06-01", "2025-10-24")
    assert (w3.plan.holdout_start, w3.plan.holdout_end) == ("2025-10-24", "2026-06-01")
    assert w3.plan.eval_windows == [
        ("2023-06-01", "2024-08-12"),
        ("2024-08-12", "2025-10-24"),
    ]
    assert w3.plan.cross_windows == [
        (1, "2023-07-01", "2025-10-24"),
        (2, "2023-08-01", "2025-10-24"),
    ]
    assert len(w3.plan.wfa_windows) == 7
    assert w3.config.timeframe == "4h"

    w6 = gatechain.settings_for("W6")
    assert (w6.plan.train_start, w6.plan.train_end) == ("2021-01-01", "2025-08-18")
    assert (w6.plan.holdout_start, w6.plan.holdout_end) == ("2025-08-18", "2026-10-15")
    assert w6.plan.cross_windows == []
    assert len(w6.plan.wfa_windows) == 14
    assert w6.config.timeframe == "1d"

    for settings in (w3, w6):
        cfg, plan = settings.config, settings.plan
        # the config follows the plan, and the REAL pipeline derives the same windows
        assert (cfg.start_date, cfg.end_date) == (plan.train_start, plan.train_end)
        assert (cfg.holdout_start_date, cfg.holdout_end_date) == (plan.holdout_start, plan.holdout_end)
        pipe = DiscoveryPipeline(cfg, backtest_fn=_null_backtest)
        assert pipe.cross_window_ranges() == plan.cross_windows
        assert pipe.wfa_window_ranges(cfg.start_date, cfg.end_date) == plan.wfa_windows
        assert pipe.holdout_min_trades() == 6


def test_unknown_plan_window_rejected() -> None:
    with pytest.raises(ValueError, match="unknown plan window"):
        gatechain.settings_for("W9")


# ---------------------------------------------------------------------------
# R10: the fitness floor (adjusted_score > 0)
# ---------------------------------------------------------------------------


def _floor_window(trades: int, ret: float) -> dict[str, float | int]:
    return {
        "total_trades": trades,
        "total_return": ret,
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
    }


def _floor_fitness(windows: list[dict[str, float | int]]) -> float:
    agg = NTBacktestFn._aggregate_multi_window(windows, min_trades=50)
    fit = _evaluate_single(_chrom(), _const_backtest(agg), None, 50, "4h")
    return fit.adjusted_score


def test_fitness_floor_cases() -> None:
    assert _floor_fitness([_floor_window(30, 0.1), _floor_window(30, 0.01)]) > 0
    assert _floor_fitness([_floor_window(30, 0.1), _floor_window(30, -0.01)]) == 0.0
    assert _floor_fitness([_floor_window(11, 0.1), _floor_window(49, 0.01)]) == 0.0
    assert _floor_fitness([_floor_window(24, 0.1), _floor_window(25, 0.01)]) == 0.0


# ---------------------------------------------------------------------------
# R29: post-hoc DSR from a stored dsr_record
# ---------------------------------------------------------------------------


def test_posthoc_dsr_from_record() -> None:
    _passed, rec167 = gatechain.dsr_verdict(R1_VALUES, 167)
    assert gatechain.posthoc_dsr_p(rec167, 167) == pytest.approx(0.7266370087847579, rel=1e-12)
    assert gatechain.posthoc_dsr_p(rec167, 1) == pytest.approx(0.01786782317136858, rel=1e-12)

    _passed1, rec1 = gatechain.dsr_verdict(R1_VALUES, 1)
    assert gatechain.posthoc_dsr_p(rec1, 1) == pytest.approx(0.01786782317136858, rel=1e-12)
    # the record's stored p at its own N is reproduced too
    assert gatechain.posthoc_dsr_p(rec1, 1) == pytest.approx(rec1["p_value"], rel=1e-12)


# ---------------------------------------------------------------------------
# R13 / R14 / R15 / R16 / R17: the individual gates, through the entry point
# ---------------------------------------------------------------------------


def _window(ret: float, sharpe: float, trades: int = 20) -> dict[str, float | int]:
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": trades,
        "total_return": ret,
    }


def _evaluate(
    settings: gatechain.ChainSettings,
    window_bt: Any,
    closes: dict[str, dict[int, float]],
    fit: FitnessResult | None = None,
    holdout_bt: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    return gatechain.evaluate_candidate(
        _chrom(),
        fit if fit is not None else _fit(),
        holdout_bt if holdout_bt is not None else _pass_holdout(),
        window_bt,
        closes,
        settings,
    )


def _w3(dsr_trials: tuple[int, ...] = (1,)) -> gatechain.ChainSettings:
    return replace(gatechain.settings_for("W3"), dsr_trials=dsr_trials)


def test_holdout_edges() -> None:
    settings = _w3()
    pipe = DiscoveryPipeline(settings.config, backtest_fn=_null_backtest)
    assert pipe.holdout_min_trades() == 6

    ok = HoldoutResult(
        sharpe_ratio=0.5, max_drawdown=0.1, profit_factor=1.2, total_trades=6, total_return=0.01
    )
    min_sharpe = settings.config.holdout_min_sharpe
    assert holdout_reasons(ok, 6, min_sharpe) == []
    assert "5 trades < 6" in holdout_reasons(replace(ok, total_trades=5), 6, min_sharpe)[0]
    assert "return" in holdout_reasons(replace(ok, total_return=0.0), 6, min_sharpe)[0]
    assert "sharpe" in holdout_reasons(replace(ok, sharpe_ratio=0.0), 6, min_sharpe)[0]
    assert "sharpe" in holdout_reasons(replace(ok, sharpe_ratio=-0.5), 6, min_sharpe)[0]

    # forwarded: the marginal flips with the holdout metrics, not the train ones
    factory = _window_factory({}, _flat_window())
    passed = _evaluate(settings, factory, _flat_closes(N_DAYS), holdout_bt=_pass_holdout())
    assert passed["marginals"]["holdout"] is True
    failed = _evaluate(
        settings, factory, _flat_closes(N_DAYS), holdout_bt={**_pass_holdout(), "sharpe_ratio": 0.0}
    )
    assert failed["marginals"]["holdout"] is False


def test_cross_window_edges() -> None:
    settings = _w3()
    windows = [(ws, we) for _, ws, we in settings.plan.cross_windows]
    assert len(windows) == 2  # both must pass (cross_window_min_pass is None)
    first, second = windows

    def cross(metrics: dict[tuple[str, str], dict[str, float | int]]) -> bool:
        factory = _window_factory(metrics, _flat_window())
        out = _evaluate(settings, factory, _flat_closes(N_DAYS))
        return bool(out["marginals"]["cross_window"])

    assert cross({first: _window(0.02, 0.5), second: _window(0.02, 0.5)}) is True  # == 0.5 passes
    assert cross({first: _window(0.0, 1.0), second: _window(0.02, 1.0)}) is False  # == 0 return fails
    assert cross({first: _window(0.02, 0.4999), second: _window(0.02, 1.0)}) is False
    assert cross({first: _window(0.02, 1.0), second: _window(-0.01, 1.0)}) is False


def test_wfa_consistency_edges() -> None:
    settings = _w3()
    windows = list(settings.plan.wfa_windows)
    assert len(windows) == 7
    win, loss, zero = _window(0.05, 1.0), _window(-0.01, 1.0), _window(0.0, 1.0)

    def wfa(per_window: list[dict[str, float | int]]) -> bool:
        factory = _window_factory(dict(zip(windows, per_window, strict=True)), _flat_window())
        out = _evaluate(settings, factory, _flat_closes(N_DAYS))
        return bool(out["marginals"]["wfa"])

    assert wfa([win] * 6 + [loss]) is True  # 6/7 = 0.857 >= 0.75
    assert wfa([win] * 5 + [loss] * 2) is False  # 5/7 = 0.714 < 0.75
    # a zero-return window is NOT profitable: 5 profitable + 1 zero + 1 loss stays 5/7
    assert wfa([win] * 5 + [zero, loss]) is False


def test_buy_hold_strict() -> None:
    n = 40
    closes = _closes([0.002 if i % 2 == 0 else -0.001 for i in range(n)])
    bh, error = buy_hold_daily(closes, ["BTCUSDT"], FIRST_DAY, n)
    assert error is None and bh is not None
    bar = daily_sharpe(bh)
    assert bar is not None

    equal = DailyReturns(FIRST_DAY, array("d", bh.tolist()))
    passed, record, reasons = buy_hold_gate(equal, "long", closes, ["BTCUSDT"])
    assert passed is False  # strict >: equal to the bar FAILS
    assert record["bar_daily"] == bar
    assert reasons and "Buy-hold" in reasons[0]

    # a constant shift lifts the mean (and the Sharpe) without touching the std
    stronger = array("d", [value + 0.0005 for value in bh.tolist()])
    passed_up, _, _ = buy_hold_gate(DailyReturns(FIRST_DAY, stronger), "long", closes, ["BTCUSDT"])
    assert passed_up is True


def test_bootstrap_floor() -> None:
    assert discovery_cli._default_bootstrap_min_sharpe("4h") == 0.0
    assert discovery_cli._default_bootstrap_min_sharpe("1d") == 0.0
    # the power-test settings carry the real floors, not GuardrailConfig's 1.0
    assert gatechain.settings_for("W3").config.bootstrap_min_sharpe == 0.0
    assert gatechain.settings_for("W6").config.bootstrap_min_sharpe == 0.0

    assert bootstrap_sharpe_ci([0.01] * 10, min_sharpe=0.0).ci_lower == 0.0

    cfg = GuardrailConfig(
        require_dsr=False,
        require_wfa=False,
        require_bootstrap_ci=True,
        bootstrap_min_sharpe=0.0,
    )
    enough = apply_guardrails(_fit(), 2, cfg, trade_returns=np.array([0.01] * 10))
    assert enough.bootstrap_passed is True  # ci_lower == 0.0 passes the 0.0 floor
    short = apply_guardrails(_fit(), 2, cfg, trade_returns=np.array([0.01] * 4))
    assert short.bootstrap_passed is False


# ---------------------------------------------------------------------------
# R11 / R12: the chain is an AND of the marginals, and matches the pipeline
# ---------------------------------------------------------------------------

GATE_NAMES = {
    "min_trades",
    "min_return",
    "complexity",
    "bootstrap",
    "buy_hold",
    "holdout",
    "cross_window",
    "wfa",
}


def test_chain_is_and_of_marginals() -> None:
    settings = _w3(dsr_trials=(1, 167))
    windows = _window_factory({}, _flat_window())
    flat, strong = _flat_closes(N_DAYS), _strong_closes(N_DAYS)

    # A clears every configured gate at both N; B clears DSR only at N=1; C fails
    # only the buy-and-hold bar; D fails only the bootstrap floor.
    a = _evaluate(settings, windows, flat, _fit(mtm_daily_returns=_series(2.0, N_DAYS, 1)))
    b = _evaluate(settings, windows, flat, _fit(mtm_daily_returns=_series(0.06, N_DAYS, 2)))
    c = _evaluate(settings, windows, strong, _fit(mtm_daily_returns=_series(0.06, N_DAYS, 2)))
    d = _evaluate(settings, windows, flat, _fit(trade_returns=tuple([0.01] * 4)))

    for out in (a, b, c, d):
        m = out["marginals"]
        assert set(m) == GATE_NAMES  # every gate is evaluated on every candidate
        assert all(isinstance(v, bool) for v in m.values())
        assert out["floor"] is True
        for n in (1, 167):
            assert out["chain_recent"][n] is bool(
                out["floor"]
                and m["min_trades"]
                and m["min_return"]
                and m["bootstrap"]
                and out["dsr"][n]
                and m["buy_hold"]
                and m["holdout"]
            )
            assert out["chain_recent_minus_bh"][n] is bool(
                out["floor"]
                and m["min_trades"]
                and m["min_return"]
                and m["bootstrap"]
                and out["dsr"][n]
                and m["holdout"]
            )
            assert out["chain_full"][n] is bool(
                out["chain_recent"][n] and m["cross_window"] and m["wfa"]
            )
        assert out["chain_recent_minus_dsr_bh"] is bool(
            out["floor"] and m["min_trades"] and m["min_return"] and m["bootstrap"] and m["holdout"]
        )

    assert all(a["marginals"].values()) and all(a["dsr"].values())
    assert a["chain_recent"] == {1: True, 167: True}
    assert a["chain_recent_minus_bh"] == {1: True, 167: True}
    assert a["chain_recent_minus_dsr_bh"] is True
    assert a["chain_full"] == {1: True, 167: True}

    # DSR binds at production N, so the leave-DSR-out reading survives it
    assert b["chain_recent"] == {1: True, 167: False}
    assert b["chain_recent_minus_dsr_bh"] is True

    # the buy-and-hold bar binds, so minus_bh survives where chain_recent does not
    assert c["marginals"]["buy_hold"] is False and c["dsr"][1] is True
    assert c["chain_recent"][1] is False and c["chain_recent_minus_bh"][1] is True

    assert d["marginals"]["bootstrap"] is False and d["chain_recent"][1] is False


def test_chain_rejects_when_floor_or_windows_fail() -> None:
    settings = _w3(dsr_trials=(1, 167))
    flat = _flat_closes(N_DAYS)
    passing = _fit(mtm_daily_returns=_series(2.0, N_DAYS, 11))
    windows = _window_factory({}, _flat_window())

    # the fixture this test leans on: every gate passes, so the only variable is
    # the one under test (the existing composition test never flips floor/cw/wfa)
    ok = _evaluate(settings, windows, flat, passing)
    assert ok["floor"] is True and all(ok["marginals"].values())
    assert ok["chain_recent"] == {1: True, 167: True} and ok["chain_full"] == {1: True, 167: True}

    # R10 at the floor: adjusted_score 0 or negative rejects EVERY chain, at both
    # N, even though every other gate (incl. cross-window and WFA) passes
    for score in (0.0, -0.5):
        out = _evaluate(
            settings, windows, flat, _fit(adjusted_score=score, mtm_daily_returns=_series(2.0, N_DAYS, 11))
        )
        m = out["marginals"]
        assert out["floor"] is False, score
        assert m["min_trades"] and m["min_return"] and m["bootstrap"] and m["buy_hold"]
        assert m["holdout"] and m["cross_window"] and m["wfa"], score
        assert out["dsr"] == {1: True, 167: True}
        assert out["chain_recent"] == {1: False, 167: False}
        assert out["chain_recent_minus_bh"] == {1: False, 167: False}
        assert out["chain_recent_minus_dsr_bh"] is False
        assert out["chain_full"] == {1: False, 167: False}

    # R14: only the shifted windows fail (sharpe 0.4 < 0.5): chain_recent still
    # passes at both N, but chain_full must reject there
    cw_bad = {(ws, we): _window(0.02, 0.4) for _, ws, we in settings.plan.cross_windows}
    cw = _evaluate(settings, _window_factory(cw_bad, _flat_window()), flat, passing)
    assert cw["marginals"]["cross_window"] is False and cw["marginals"]["wfa"] is True
    assert cw["chain_recent"] == {1: True, 167: True}
    assert cw["chain_recent_minus_bh"] == {1: True, 167: True}
    assert cw["chain_full"] == {1: False, 167: False}

    # R15: 3/7 profitable WFA windows fail; the cross-windows still pass
    wfa_bad = {w: _window(-0.01, 1.0) for w in list(settings.plan.wfa_windows)[3:]}
    wfa = _evaluate(settings, _window_factory(wfa_bad, _flat_window()), flat, passing)
    assert wfa["marginals"]["wfa"] is False and wfa["marginals"]["cross_window"] is True
    assert wfa["chain_recent"] == {1: True, 167: True}
    assert wfa["chain_full"] == {1: False, 167: False}


def test_min_return_equality_and_isolation() -> None:
    settings = _w3(dsr_trials=(1, 167))
    windows = _window_factory({}, _flat_window())
    flat = _flat_closes(N_DAYS)

    # the neutral min-return floor is >= 0.0: -1e-12 fails, 0.0 and +1e-12 pass,
    # asserted through the public entry at both N
    for ret, expected in ((-1e-12, False), (0.0, True), (1e-12, True)):
        out = _evaluate(
            settings, windows, flat, _fit(total_return=ret, mtm_daily_returns=_series(2.0, N_DAYS, 7))
        )
        assert out["marginals"]["min_return"] is expected, ret

    # isolation: min_return is read with the other guardrails neutralised, so a
    # zero return still passes when min_trades and bootstrap fail
    isolated = _evaluate(
        settings,
        windows,
        flat,
        _fit(
            total_return=0.0,
            total_trades=3,
            trade_returns=tuple([0.01] * 3),
            mtm_daily_returns=_series(2.0, N_DAYS, 8),
        ),
    )
    m = isolated["marginals"]
    assert m["min_return"] is True
    assert m["min_trades"] is False and m["bootstrap"] is False
    assert isolated["chain_recent"] == {1: False, 167: False}


def test_guardrails_match_pipeline() -> None:
    base = gatechain.settings_for("W3")
    chrom = _chrom()
    fixtures = {
        "pass": _fit(mtm_daily_returns=_series(2.0, N_DAYS, 3)),
        "weak_dsr": _fit(mtm_daily_returns=_series(0.06, N_DAYS, 4)),
        "negative_return": _fit(total_return=-0.1, mtm_daily_returns=_series(2.0, N_DAYS, 5)),
        "too_few_trades": _fit(total_trades=30, mtm_daily_returns=_series(2.0, N_DAYS, 6)),
        "zero_return": _fit(total_return=0.0, mtm_daily_returns=_series(2.0, N_DAYS, 7)),
        "tiny_negative_return": _fit(total_return=-1e-12, mtm_daily_returns=_series(2.0, N_DAYS, 8)),
        "tiny_positive_return": _fit(total_return=1e-12, mtm_daily_returns=_series(2.0, N_DAYS, 9)),
        "zero_return_few_trades": _fit(
            total_return=0.0,
            total_trades=3,
            trade_returns=tuple([0.01] * 3),
            mtm_daily_returns=_series(2.0, N_DAYS, 10),
        ),
    }
    windows = _window_factory({}, _flat_window())
    for n in (1, 167):
        for label, fit in fixtures.items():
            def prior_trials_fn(_start: str, _end: str, num_trials: int = n) -> int:
                return num_trials - 1

            cfg = replace(base.config, prior_trials_fn=prior_trials_fn)
            chain = GateChain(cfg, GateEvidence(_null_backtest, False))
            kept = chain.screen([(chrom, fit)], TrialBudget(1, {}))

            out = _evaluate(replace(base, dsr_trials=(n,)), windows, _flat_closes(N_DAYS), fit)
            m = out["marginals"]
            mine = bool(
                m["min_trades"] and m["min_return"] and m["bootstrap"] and out["dsr"][n]
            )
            assert mine is bool(kept), (label, n, mine, kept)


@pytest.mark.parametrize("window", ["W3", "W6"])
def test_gate_chain_settings_stay_on_the_legacy_dsr_rule(window: str) -> None:
    # DiscoveryConfig defaults to q95 since T4; the power study reproduces its legacy chain.
    assert gatechain.settings_for(window).config.dsr_rule == "emax"
