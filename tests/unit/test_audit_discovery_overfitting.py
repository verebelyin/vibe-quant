"""Audit 2026-10-02 regression tests: overfitting statistics (DSR / WFA / CV)
and screening ranking (vibe-quant-e70tl.9, vibe-quant-e70tl.10 + mediums).

Expected values are hand-computed from the definitions, not from the code.
"""

from __future__ import annotations

import json
import math
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio, deannualize_sharpe
from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.overfitting.types import FilterConfig
from vibe_quant.overfitting.wfa import WalkForwardAnalysis, WFAConfig
from vibe_quant.screening.grid import compute_pareto_front, filter_by_metrics, rank_by_sharpe
from vibe_quant.screening.pipeline import ScreeningPipeline
from vibe_quant.screening.types import BacktestMetrics, MetricFilters

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# e70tl.9 — DSR trial count / observations / single survivor / NaN trials
# ---------------------------------------------------------------------------


def _discovery_run(tmp_path: Path, *, sharpe: float, evaluated: int) -> tuple[Path, int]:
    db = tmp_path / "t.db"
    sm = StateManager(db)
    run_id = sm.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2026-03-17", {}
    )
    sm.save_backtest_result(
        run_id,
        {
            "sharpe_ratio": sharpe, "total_return": 0.4, "max_drawdown": 0.1,
            "profit_factor": 1.5, "total_trades": 120, "skewness": 0.0, "kurtosis": 3.0,
            "notes": json.dumps({
                "type": "discovery",
                "evaluated": evaluated,
                "train_dates": ["2024-01-01", "2025-09-05"],
            }),
        },
    )
    sm.close()
    return db, run_id


def test_dsr_on_ga_champion_uses_discovery_trials_and_train_days(tmp_path: Path) -> None:
    db, run_id = _discovery_run(tmp_path, sharpe=2.2, evaluated=6000)
    p = OverfittingPipeline(db)
    res = p.run(run_id=run_id, config=FilterConfig.dsr_only())
    p.close()
    c = res.candidates[0]
    assert c.dsr_result is not None
    assert c.dsr_result.num_trials == 6000
    train_days = (date(2025, 9, 5) - date(2024, 1, 1)).days  # 613
    assert c.dsr_result.num_observations == train_days
    ref = DeflatedSharpeRatio().calculate(
        observed_sharpe=deannualize_sharpe(2.2), num_trials=6000,
        num_observations=train_days, skewness=0.0, kurtosis=3.0,
    )
    assert c.dsr_result.p_value == ref.p_value
    # 6000 GA trials: a 2.2 Sharpe is not significant (with N=1 it "passed")
    assert c.passed_dsr is False


def test_dsr_observations_default_to_run_day_count(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    sm = StateManager(db)
    sid = sm.create_strategy("s", {"timeframe": "4h"})
    rid = sm.create_backtest_run(sid, "screening", ["BTCUSDT"], "4h", "2022-01-01", "2022-12-31", {})
    sm.save_sweep_results_batch(rid, [
        {"parameters": {"p": 1}, "sharpe_ratio": 1.0, "total_return": 0.1, "total_trades": 50},
    ])
    sm.close()
    p = OverfittingPipeline(db)
    res = p.run(run_id=rid, config=FilterConfig.dsr_only())
    p.close()
    assert res.candidates[0].dsr_result is not None
    assert res.candidates[0].dsr_result.num_observations == 364  # not the old 252


def test_overfitting_cli_forwards_total_trials_and_derived_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibe_quant.overfitting.__main__ import main

    seen: dict[str, Any] = {}

    def fake_run(self: OverfittingPipeline, **kwargs: Any) -> Any:
        seen.update(kwargs)
        from vibe_quant.overfitting.types import PipelineResult

        return PipelineResult(
            config=kwargs["config"], total_candidates=0, passed_dsr=0,
            passed_wfa=0, passed_cv=0, passed_all=0,
        )

    monkeypatch.setattr(OverfittingPipeline, "run", fake_run)
    db, run_id = _discovery_run(tmp_path, sharpe=2.2, evaluated=10)
    assert main(["run", "--run-id", str(run_id), "--db", str(db), "--filters", "dsr",
                 "--total-trials", "777"]) == 0
    assert seen["total_trials"] == 777
    assert seen["num_observations"] is None  # resolved from the run, not 252


def _metrics(sr: float, trades: int = 100) -> BacktestMetrics:
    return BacktestMetrics(
        parameters={}, sharpe_ratio=sr, profit_factor=1.5, max_drawdown=0.1,
        total_trades=trades, total_return=0.1,
    )


def _screen(results: list[BacktestMetrics], start: str = "2024-01-01",
            end: str = "2024-12-31") -> Any:
    from types import SimpleNamespace

    pipe = ScreeningPipeline.__new__(ScreeningPipeline)
    pipe.dsl = SimpleNamespace(name="x")  # type: ignore[assignment]
    pipe._max_workers = 1
    pipe._start_date, pipe._end_date = start, end
    pipe._param_grid = [{"i": i} for i in range(len(results))]
    it = iter(results)
    pipe._runner = lambda params: next(it)  # type: ignore[assignment,misc]
    pipe._stall_timeout_s = 600
    return pipe.run(filters=MetricFilters())


def test_screening_dsr_applied_to_single_survivor() -> None:
    """1000 combos, ONE survives hard filters with Sharpe 1.2 -> DSR fails it."""
    out = _screen([_metrics(-0.5) for _ in range(999)] + [_metrics(1.2)])
    assert out.passed_filters == 0
    ref = DeflatedSharpeRatio().calculate(deannualize_sharpe(1.2), 1000, 365)
    assert not ref.is_significant  # the hand reference agrees


def test_screening_dsr_verdicts_ignore_nan_inf_and_crash_sentinel() -> None:
    """One NaN + one -inf (+ one crash sentinel) combo -> same verdicts as without."""
    base = [_metrics(0.2, 5) for _ in range(196)] + [_metrics(4.5), _metrics(4.6), _metrics(4.7)]
    clean = _screen(list(base))
    noisy = _screen(list(base) + [_metrics(float("nan"), 3), _metrics(float("-inf"), 0)])
    crashed = _screen(list(base) + [_metrics(-999.0, 0)])
    assert clean.passed_filters == 3
    assert noisy.passed_filters == clean.passed_filters
    assert crashed.passed_filters == clean.passed_filters


# ---------------------------------------------------------------------------
# e70tl.10 — WFA efficiency / losing strategies / run dates / CV params
# ---------------------------------------------------------------------------


class _PathRunner:
    """WFA runner whose window return = sum of a daily P&L path over the window."""

    def __init__(self, start: date, pnl: Any) -> None:
        self.start, self.pnl = start, pnl

    def optimize(self, sid: str, s: date, e: date, grid: dict[str, list[object]]) -> Any:
        sr, ret = self.backtest(sid, s, e, {})
        return {}, sr, ret

    def backtest(self, sid: str, s: date, e: date, params: dict[str, object]) -> Any:
        a = (s - self.start).days
        n = (e - s).days + 1
        ret = sum(self.pnl(i) for i in range(a, a + n))
        return (1.0 if ret > 0 else -1.0), ret


def test_wfa_stationary_edge_efficiency_is_one_and_robust() -> None:
    start = date(2024, 1, 1)
    wfa = WalkForwardAnalysis(WFAConfig.default(), _PathRunner(start, lambda i: 0.0005))
    r = wfa.run("x", start, date(2025, 12, 31), {})
    assert r.efficiency == pytest.approx(1.0, abs=1e-12)
    assert r.is_robust is True


def test_wfa_negative_is_positive_oos_does_not_auto_pass() -> None:
    class _Neg:
        def optimize(self, sid: str, s: date, e: date, g: dict[str, list[object]]) -> Any:
            return {}, -0.5, -0.02

        def backtest(self, sid: str, s: date, e: date, p: dict[str, object]) -> Any:
            return 1.0, 0.01

    wfa = WalkForwardAnalysis(WFAConfig.default(), _Neg())
    r = wfa.run("x", date(2024, 1, 1), date(2025, 12, 31), {})
    assert r.consistency_ratio == 1.0
    assert r.efficiency == 0.0  # was 5.0
    assert r.is_robust is False


def test_wfa_overall_losing_strategy_fails() -> None:
    """Loses 32.4% in the first 270 days, +0.03%/day after: full period -18.57%."""
    start = date(2024, 1, 1)
    pnl = lambda i: -0.0012 if i < 270 else 0.0003  # noqa: E731
    full = sum(pnl(i) for i in range(731))
    assert full == pytest.approx(270 * -0.0012 + 461 * 0.0003, abs=1e-12)  # -0.1857
    r = WalkForwardAnalysis(WFAConfig.default(), _PathRunner(start, pnl)).run(
        "x", start, date(2025, 12, 31), {},
    )
    assert r.consistency_ratio == 1.0  # every OOS window is green ...
    assert r.is_robust is False  # ... but the strategy loses overall


def test_wfa_fails_when_concatenated_oos_loses() -> None:
    """3/4 OOS windows slightly green, one big loss: overall OOS negative -> fail."""
    class _BigLoss:
        calls = 0

        def optimize(self, sid: str, s: date, e: date, g: dict[str, list[object]]) -> Any:
            return {}, 1.0, 0.09  # 270d IS, 0.0333%/day

        def backtest(self, sid: str, s: date, e: date, p: dict[str, object]) -> Any:
            _BigLoss.calls += 1
            return (1.0, 0.04) if _BigLoss.calls % 4 else (-2.0, -0.20)

    cfg = WFAConfig(in_sample_days=270, out_of_sample_days=90, step_days=90, min_windows=4)
    r = WalkForwardAnalysis(cfg, _BigLoss()).run("x", date(2024, 1, 1), date(2025, 12, 31), {})
    assert r.num_windows == 5
    assert sum(w.oos_return for w in r.windows) < 0
    assert r.is_robust is False


def _seed_screening_run(tmp_path: Path, start: str, end: str) -> tuple[Path, int]:
    db = tmp_path / "t.db"
    sm = StateManager(db)
    sid = sm.create_strategy("s", {"timeframe": "4h", "indicators": {}})
    rid = sm.create_backtest_run(sid, "screening", ["BTCUSDT"], "4h", start, end, {})
    sm.save_sweep_results_batch(rid, [
        {"parameters": {"rsi_period": 14}, "sharpe_ratio": 2.0, "total_return": 0.3,
         "total_trades": 100},
        {"parameters": {"rsi_period": 2}, "sharpe_ratio": 1.9, "total_return": 0.3,
         "total_trades": 100},
    ])
    sm.close()
    return db, rid


def test_wfa_on_2022_run_uses_2022_dates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vibe_quant.screening.nt_runner as ntr
    from vibe_quant.overfitting.__main__ import main

    seen: list[tuple[str, str]] = []

    class _Rec:
        def __init__(self, dsl_dict: Any, symbols: Any, start_date: str, end_date: str,
                     catalog_path: Any = None) -> None:
            seen.append((start_date, end_date))

        def __call__(self, params: Any) -> Any:
            return type("M", (), {"sharpe_ratio": 1.0, "total_return": 0.01})()

    monkeypatch.setattr(ntr, "NTScreeningRunner", _Rec)
    db, rid = _seed_screening_run(tmp_path, "2022-01-01", "2022-12-31")
    assert main([
        "run", "--run-id", str(rid), "--db", str(db), "--filters", "wfa", "--real-wfa",
        "--wfa-is-days", "60", "--wfa-oos-days", "30", "--wfa-step-days", "60",
        "--wfa-min-windows", "2",
    ]) == 0
    assert seen
    assert min(s for s, _ in seen) == "2022-01-01"
    assert max(e for _, e in seen) <= "2022-12-31"


def test_cv_runner_forwards_each_candidates_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import vibe_quant.screening.nt_runner as ntr
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    calls: list[dict[str, Any]] = []

    class _Fake:
        def __init__(self, dsl_dict: Any, symbols: Any, start_date: str, end_date: str,
                     catalog_path: Any = None) -> None:
            pass

        def __call__(self, params: dict[str, Any]) -> Any:
            calls.append(dict(params))
            good = params.get("rsi_period", 14) == 14
            return type("M", (), {"sharpe_ratio": 2.0 if good else -1.5,
                                  "total_return": 0.1 if good else -0.1})()

    monkeypatch.setattr(ntr, "NTScreeningRunner", _Fake)
    db, rid = _seed_screening_run(tmp_path, "2024-01-01", "2025-12-31")
    cat = tmp_path / "catalog"
    bar = cat / "data" / "bar" / "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"
    bar.mkdir(parents=True)
    ts = pd.date_range("2024-01-01", "2025-12-31", freq="4h")
    pd.DataFrame({"ts_event": ts.astype("int64")}).to_parquet(bar / "x.parquet")

    cv_runner = NTPurgedKFoldRunner(rid, db, catalog_path=cat)
    p = OverfittingPipeline(db, cv_runner=cv_runner)
    res = p.run(rid, config=FilterConfig.cv_only(), n_samples=cv_runner.n_samples)
    p.close()

    by_period = {json.loads(c.parameters)["rsi_period"]: c for c in res.candidates}
    assert by_period[14].cv_result is not None and by_period[2].cv_result is not None
    assert by_period[14].cv_result.mean_oos_sharpe == pytest.approx(2.0)
    assert by_period[2].cv_result.mean_oos_sharpe == pytest.approx(-1.5)
    assert {c["rsi_period"] for c in calls} == {14, 2}


# ---------------------------------------------------------------------------
# Medium: grid NaN handling, replay drift two-sided
# ---------------------------------------------------------------------------


def test_nan_metrics_fail_hard_filters() -> None:
    nan_m = BacktestMetrics(sharpe_ratio=float("nan"), profit_factor=float("nan"),
                            max_drawdown=0.05, total_trades=60)
    nan_dd = BacktestMetrics(sharpe_ratio=2.0, profit_factor=2.0,
                             max_drawdown=float("nan"), total_trades=60)
    inf_pf = BacktestMetrics(sharpe_ratio=2.0, profit_factor=float("inf"),
                             max_drawdown=0.05, total_trades=60)
    f = MetricFilters(min_sharpe=1.0, min_profit_factor=1.5, min_trades=10)
    assert filter_by_metrics([nan_m, nan_dd, inf_pf], f) == [inf_pf]


def test_rank_by_sharpe_puts_nan_last_and_sorts_the_rest() -> None:
    ms = [BacktestMetrics(sharpe_ratio=s) for s in [0.5, float("nan"), 2.0, 1.0,
                                                    float("nan"), 3.0, 0.1]]
    ranked = [m.sharpe_ratio for m in rank_by_sharpe(ms)]
    assert ranked[:5] == [3.0, 2.0, 1.0, 0.5, 0.1]
    assert all(math.isnan(x) for x in ranked[5:])


def test_pareto_front_excludes_nan_rows() -> None:
    ms = [
        BacktestMetrics(sharpe_ratio=float("nan"), max_drawdown=0.9, profit_factor=float("nan")),
        BacktestMetrics(sharpe_ratio=2.0, max_drawdown=0.05, profit_factor=2.0),
    ]
    assert compute_pareto_front(ms) == [1]
    assert compute_pareto_front(ms[:1]) == []


@pytest.mark.parametrize(
    ("disc_sharpe", "scr_sharpe", "flagged"),
    [
        (1.0, 1.1, False),    # within band
        (1.0, 5.0, True),     # much "better" than discovery: still drift
        (-1.0, -2.0, True),   # same sign, twice as bad (old check: ratio 2 -> clean)
        (1.0, -1.0, True),    # sign flip
        (1.0, 0.85, False),   # inside [0.8, 1.25]
    ],
)
def test_replay_drift_two_sided(disc_sharpe: float, scr_sharpe: float, flagged: bool) -> None:
    from vibe_quant.screening.replay_drift import _build_drift_payload

    payload = _build_drift_payload(disc_sharpe, 100, scr_sharpe, 100)
    assert payload["flagged"] is flagged


def test_replay_drift_trades_two_sided() -> None:
    from vibe_quant.screening.replay_drift import _build_drift_payload

    assert _build_drift_payload(1.0, 100, 1.0, 150)["flagged"] is True
    assert _build_drift_payload(1.0, 100, 1.0, 105)["flagged"] is False
