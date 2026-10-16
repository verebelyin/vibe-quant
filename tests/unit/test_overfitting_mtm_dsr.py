"""Overfitting / screening DSR tests the mark-to-market daily statistics.

The realized-PnL Sharpe books a multi-day trade's whole PnL on its exit day and
overstates the Sharpe; a promoted discovery champion or a screening row must be
re-checked on the market-valued series (bd vibe-quant-68630). Expected values
come from ``DeflatedSharpeRatio.calculate`` called by hand with the inputs a
reader can see in each test.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import SCHEMA_SQL, init_schema
from vibe_quant.db.state_manager import StateManager
from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    daily_sharpe_inputs,
    deannualize_sharpe,
    mtm_dsr_calculate,
)
from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.overfitting.types import FilterConfig
from vibe_quant.screening.pipeline import ScreeningPipeline
from vibe_quant.screening.types import BacktestMetrics, MetricFilters

if TYPE_CHECKING:
    from pathlib import Path

MTM_COLUMNS = ("mtm_sharpe_daily", "mtm_skewness", "mtm_kurtosis", "mtm_days")
MTM_SR_ANN = 1.6  # market-valued Sharpe of the champion (annualized)
REALIZED_SR_ANN = 2.5  # realized-PnL Sharpe of the same champion (headline)
T_DAYS = 1264
_MTM_INPUT = "pooled_mtm_daily_returns"


def _record(**over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "input": _MTM_INPUT,
        "observations": T_DAYS,
        "sharpe_daily": deannualize_sharpe(MTM_SR_ANN),
        "skewness": 0.0,
        "kurtosis": 3.0,
    }
    rec.update(over)
    return rec


def _discovery_db(
    tmp_path: Path,
    *,
    evaluated: int,
    record: dict[str, Any] | None,
    symbols: list[str] | None = None,
    symbol_agg: str = "portfolio",
) -> tuple[Path, int]:
    """A discovery run whose headline is the REALIZED Sharpe 2.5 (as persisted)."""
    db = tmp_path / "disc.db"
    sm = StateManager(db)
    syms = symbols or ["BTCUSDT"]
    run_id = sm.create_backtest_run(None, "discovery", syms, "4h", "2022-01-01", "2026-03-17", {})
    champion: dict[str, Any] = {"sharpe": REALIZED_SR_ANN}
    if record is not None:
        champion["dsr"] = record
    sm.save_backtest_result(
        run_id,
        {
            "sharpe_ratio": REALIZED_SR_ANN, "total_return": 0.4, "max_drawdown": 0.1,
            "profit_factor": 1.5, "total_trades": 120, "skewness": 0.0, "kurtosis": 3.0,
            "notes": json.dumps({
                "type": "discovery", "evaluated": evaluated,
                "symbols": syms, "symbol_agg": symbol_agg,
                "train_dates": ["2022-01-01", "2026-03-17"],
                "top_strategies": [champion],
            }),
        },
    )  # fmt: skip
    sm.close()
    return db, run_id


def _run_dsr(db: Path, run_id: int, **kwargs: Any) -> Any:
    p = OverfittingPipeline(db)
    try:
        res = p.run(run_id=run_id, config=FilterConfig.dsr_only(), **kwargs)
        row = p.conn.execute(
            "SELECT passed_deflated_sharpe FROM sweep_results WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        p.close()
    return res.candidates[0], row[0]


# --- promotion / recheck regression (reviewer rev-dsr-mtm) ------------------


def test_promoted_champion_dsr_rechecked_on_mtm_not_realized(tmp_path: Path) -> None:
    """MTM SR 1.6 passes at N=10 but must be REJECTED at N=79, T=1264; the realized
    headline SR 2.5 would pass at N=79 (p ~ 0.0009) and persist a wrong verdict."""
    dsr = DeflatedSharpeRatio()
    sd = deannualize_sharpe(MTM_SR_ANN)
    p10 = dsr.calculate(sd, 10, T_DAYS, 0.0, 3.0).p_value
    p79 = dsr.calculate(sd, 79, T_DAYS, 0.0, 3.0).p_value
    realized_p79 = dsr.calculate(deannualize_sharpe(REALIZED_SR_ANN), 79, T_DAYS, 0.0, 3.0).p_value
    assert p10 < 0.05 < p79  # hand reference: the control passes, the real N rejects
    assert realized_p79 < 0.05  # the bug: realized SR passes at N=79

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    db_a, rid_a = _discovery_db(tmp_path / "a", evaluated=10, record=_record())
    cand_a, flag_a = _run_dsr(db_a, rid_a)
    assert cand_a.passed_dsr is True and flag_a == 1
    assert cand_a.dsr_result.p_value == pytest.approx(p10, rel=1e-12)

    db_b, rid_b = _discovery_db(tmp_path / "b", evaluated=79, record=_record())
    cand_b, flag_b = _run_dsr(db_b, rid_b)
    assert cand_b.dsr_result is not None
    assert cand_b.dsr_result.num_observations == T_DAYS
    assert cand_b.dsr_result.p_value == pytest.approx(p79, rel=1e-12)
    assert cand_b.passed_dsr is False
    assert flag_b == 0  # persisted verdict is the MTM one
    assert cand_b.dsr_fail_reason is not None and "not significant" in cand_b.dsr_fail_reason


def test_promotion_copies_mtm_stats_from_dsr_record(tmp_path: Path) -> None:
    db, rid = _discovery_db(
        tmp_path, evaluated=10, record=_record(skewness=-0.4, kurtosis=5.5)
    )
    p = OverfittingPipeline(db)
    try:
        (cand,) = p._load_candidates(rid)
        row = p.conn.execute("SELECT * FROM sweep_results WHERE run_id = ?", (rid,)).fetchone()
    finally:
        p.close()
    expect = (deannualize_sharpe(MTM_SR_ANN), -0.4, 5.5, T_DAYS)
    assert tuple(row[c] for c in MTM_COLUMNS) == expect
    assert tuple(cand[c] for c in MTM_COLUMNS) == expect
    assert row["sharpe_ratio"] == REALIZED_SR_ANN  # headline stays the replay's number


@pytest.mark.parametrize(
    "record",
    [
        None,  # no DSR record at all
        _record(input="pooled_daily_returns"),  # pre-MTM realized-series record
        _record(missing=True),  # discovery itself failed DSR closed
        _record(sharpe_daily=None),
        _record(observations=float("nan")),
        _record(kurtosis="3.0"),
    ],
)
def test_promoted_champion_without_trusted_mtm_record_fails_closed(
    tmp_path: Path, record: dict[str, Any] | None
) -> None:
    db, rid = _discovery_db(tmp_path, evaluated=10, record=record)
    cand, flag = _run_dsr(db, rid)
    assert cand.dsr_result is None
    assert cand.passed_dsr is False and flag == 0
    assert cand.passed_all is False
    assert cand.dsr_fail_reason is not None and "mark-to-market" in cand.dsr_fail_reason
    p = OverfittingPipeline(db)
    try:
        row = p.conn.execute("SELECT * FROM sweep_results WHERE run_id = ?", (rid,)).fetchone()
    finally:
        p.close()
    assert all(row[c] is None for c in MTM_COLUMNS)


def test_moments_clamped_like_discovery(tmp_path: Path) -> None:
    """Positive skew / thin tails are not trusted (skew <= 0, kurtosis >= 3)."""
    db, rid = _discovery_db(
        tmp_path, evaluated=79, record=_record(skewness=1.5, kurtosis=1.2)
    )
    cand, _ = _run_dsr(db, rid)
    ref = DeflatedSharpeRatio().calculate(deannualize_sharpe(MTM_SR_ANN), 79, T_DAYS, 0.0, 3.0)
    assert cand.dsr_result is not None
    assert cand.dsr_result.p_value == pytest.approx(ref.p_value, rel=1e-12)


def test_run_no_longer_accepts_num_observations(tmp_path: Path) -> None:
    """The dead DSR override is gone: an external caller passing it fails loud."""
    db, rid = _discovery_db(tmp_path, evaluated=79, record=_record())
    pipeline = OverfittingPipeline(db)
    try:
        with pytest.raises(TypeError, match="num_observations"):
            pipeline.run(run_id=rid, config=FilterConfig.dsr_only(), num_observations=5)
    finally:
        pipeline.close()


def test_pipeline_uses_the_public_migrate_add_columns() -> None:
    """pipeline.py must reach the helper through schema's public name, not the private one."""
    import vibe_quant.db.schema as schema
    import vibe_quant.overfitting.pipeline as pipeline_module

    assert schema.migrate_add_columns is not None
    assert pipeline_module.migrate_add_columns is schema.migrate_add_columns


def test_recheck_consumes_the_record_discovery_actually_writes(tmp_path: Path) -> None:
    """Contract with ``apply_discovery_dsr_returns``: its real record promotes to the
    same Sharpe/moments/T, and rechecking at the run's own N reproduces its p-value
    and per-symbol verdict (so the two gates cannot drift apart silently)."""
    import numpy as np

    from vibe_quant.discovery.guardrails import apply_discovery_dsr_returns

    rng = np.random.default_rng(7)
    pooled = rng.normal(0.0012, 0.01, size=600).tolist()
    weak = rng.normal(0.0, 0.01, size=600)
    strong = rng.normal(0.0025, 0.01, size=600)

    def stats(x: Any) -> dict[str, float]:
        sr, sk, ku = daily_sharpe_inputs(x)
        return {"sharpe": sr, "mean": float(np.mean(x)), "skewness": sk, "kurtosis": ku,
                "observations": len(x)}  # fmt: skip

    sym = {"BTCUSDT": stats(strong), "ETHUSDT": stats(weak)}
    result, record, reasons = apply_discovery_dsr_returns(pooled, 40, symbol_daily=sym)
    assert result is not None
    assert any("per-symbol floor: ETHUSDT" in r for r in reasons)  # discovery rejected it

    db, rid = _discovery_db(
        tmp_path, evaluated=40, record=record, symbols=["BTCUSDT", "ETHUSDT"], symbol_agg="worst"
    )
    cand, _ = _run_dsr(db, rid)
    sharpe, skew, kurt = daily_sharpe_inputs(pooled)
    assert cand.dsr_result is not None
    assert cand.dsr_result.num_observations == len(pooled)
    assert cand.dsr_result.observed_sharpe == sharpe
    assert cand.dsr_result.p_value == pytest.approx(result.p_value, rel=1e-12)
    assert cand.passed_dsr is False
    assert cand.dsr_fail_reason is not None and "ETHUSDT" in cand.dsr_fail_reason
    assert (skew, kurt) == (record["skewness"], record["kurtosis"])


# --- per-symbol floor ------------------------------------------------------


def _worst_record(psr_a: float | None, psr_b: float | None) -> dict[str, Any]:
    symbols: dict[str, Any] = {}
    if psr_a is not None:
        symbols["BTCUSDT"] = {"psr": psr_a, "observations": T_DAYS}
    if psr_b is not None:
        symbols["ETHUSDT"] = {"psr": psr_b, "observations": T_DAYS}
    return _record(symbol_psr_floor=0.8, symbols=symbols)


@pytest.mark.parametrize(
    ("psr_a", "psr_b", "ok", "needle"),
    [
        (0.95, 0.9, True, None),
        (0.95, 0.5, False, "ETHUSDT PSR(SR>0)=0.500 < 0.80"),
        (0.95, None, False, "no per-symbol PSR for ETHUSDT"),
    ],
)
def test_worst_mode_recheck_reapplies_persisted_symbol_floor(
    tmp_path: Path, psr_a: float, psr_b: float | None, ok: bool, needle: str | None
) -> None:
    """Pooled DSR passes (N=10); a symbol below the floor, or missing, still rejects."""
    db, rid = _discovery_db(
        tmp_path, evaluated=10, record=_worst_record(psr_a, psr_b),
        symbols=["BTCUSDT", "ETHUSDT"], symbol_agg="worst",
    )  # fmt: skip
    cand, flag = _run_dsr(db, rid)
    assert cand.dsr_result is not None and cand.dsr_result.is_significant  # pooled passes
    assert cand.passed_dsr is ok and flag == int(ok)
    if needle:
        assert cand.dsr_fail_reason is not None and needle in cand.dsr_fail_reason


def test_single_symbol_worst_mode_has_no_extra_floor(tmp_path: Path) -> None:
    db, rid = _discovery_db(tmp_path, evaluated=10, record=_record(), symbol_agg="worst")
    cand, _ = _run_dsr(db, rid)
    assert cand.passed_dsr is True


# --- screening rows (sweep_results) ----------------------------------------


def _screening_db(tmp_path: Path, rows: list[dict[str, Any]]) -> tuple[Path, int]:
    db = tmp_path / "scr.db"
    sm = StateManager(db)
    sid = sm.create_strategy("s", {"timeframe": "4h"})
    rid = sm.create_backtest_run(sid, "screening", ["BTCUSDT"], "4h", "2022-01-01", "2022-12-31", {})
    sm.save_sweep_results_batch(rid, rows)
    sm.close()
    return db, rid


def test_screening_row_dsr_uses_mtm_days_and_ignores_realized_sharpe(tmp_path: Path) -> None:
    """Realized Sharpe 5.0 would pass any N; the MTM stats (SR 0.3 ann, 200 days) do not,
    and T is the MTM series length, not the run's 364-day date range."""
    db, rid = _screening_db(tmp_path, [{
        "parameters": {"p": 1}, "sharpe_ratio": 5.0, "total_return": 0.1, "total_trades": 50,
        "mtm_sharpe_daily": deannualize_sharpe(0.3), "mtm_skewness": 0.0,
        "mtm_kurtosis": 3.0, "mtm_days": 200,
    }])  # fmt: skip
    cand, flag = _run_dsr(db, rid, total_trials=5)
    ref = DeflatedSharpeRatio().calculate(deannualize_sharpe(0.3), 5, 200, 0.0, 3.0)
    assert cand.dsr_result is not None
    assert cand.dsr_result.num_observations == 200
    assert cand.dsr_result.p_value == pytest.approx(ref.p_value, rel=1e-12)
    assert cand.passed_dsr is False and flag == 0


def test_screening_row_without_mtm_stats_fails_closed(tmp_path: Path) -> None:
    db, rid = _screening_db(
        tmp_path,
        [{"parameters": {"p": 1}, "sharpe_ratio": 5.0, "total_return": 0.1, "total_trades": 50}],
    )
    cand, flag = _run_dsr(db, rid)
    assert cand.dsr_result is None and cand.passed_dsr is False and flag == 0
    assert cand.dsr_fail_reason is not None and "mtm_days" in cand.dsr_fail_reason


# --- ScreeningPipeline.run DSR filter --------------------------------------


def _metrics(sr_ann: float | None, realized: float = 3.0) -> BacktestMetrics:
    m = BacktestMetrics(
        parameters={}, sharpe_ratio=realized, profit_factor=1.5, max_drawdown=0.1,
        total_trades=100, total_return=0.1,
    )  # fmt: skip
    if sr_ann is not None:
        m.mtm_sharpe_daily = deannualize_sharpe(sr_ann)
        m.mtm_skewness, m.mtm_kurtosis, m.mtm_days = 0.0, 3.0, 365
    return m


def _screen(results: list[BacktestMetrics]) -> Any:
    from types import SimpleNamespace

    pipe = ScreeningPipeline.__new__(ScreeningPipeline)
    pipe.dsl = SimpleNamespace(name="x")  # type: ignore[assignment]
    pipe._max_workers = 1
    pipe._start_date, pipe._end_date = "2024-01-01", "2024-12-31"
    pipe._param_grid = [{"i": i} for i in range(len(results))]
    it = iter(results)
    pipe._runner = lambda params: next(it)  # type: ignore[assignment,misc]
    pipe._stall_timeout_s = 600
    return pipe.run(filters=MetricFilters())


def test_screening_pipeline_dsr_filter_judges_mtm_not_realized() -> None:
    """Every row has realized Sharpe 3.0; only the one with a strong MTM Sharpe survives,
    and a row with no MTM stats fails closed instead of falling back to realized."""
    weak = [_metrics(0.2) for _ in range(30)]
    strong = _metrics(4.0)
    missing = _metrics(None)
    out = _screen([*weak, strong, missing])
    survivors = [r for r in out.results if r.mtm_sharpe_daily == strong.mtm_sharpe_daily]
    assert out.passed_filters == 1
    assert len(survivors) == 1
    ref = DeflatedSharpeRatio().calculate(deannualize_sharpe(0.2), 32, 365)
    assert not ref.is_significant  # hand reference: the weak rows fail at N=32


def test_screening_trial_variance_uses_mtm_sharpes_not_realized() -> None:
    """30 trials at MTM SR 0.2 ann + one at 2.0, ALL with realized Sharpe 3.0: the trial
    variance is the dispersion of the MTM Sharpes, so the champion is rejected
    (p ~ 0.056). Variance taken from the realized Sharpes is 0 and would pass it."""
    from vibe_quant.screening.pipeline import _trial_sharpe_variance

    weak = [_metrics(0.2, realized=3.0) for _ in range(30)]
    champion = _metrics(2.0, realized=3.0)
    trials = [*weak, champion]

    daily = [deannualize_sharpe(0.2)] * 30 + [deannualize_sharpe(2.0)]
    mean = sum(daily) / 31
    hand_var = sum((x - mean) ** 2 for x in daily) / 30
    assert _trial_sharpe_variance(trials) == pytest.approx(hand_var, rel=1e-12)
    assert hand_var > 0.0  # (realized Sharpes are all 3.0 -> variance 0)

    ref = DeflatedSharpeRatio().calculate(
        deannualize_sharpe(2.0), 31, 365, 0.0, 3.0, trials_sharpe_variance=hand_var
    )
    assert ref.p_value > 0.05  # hand reference: rejected with the MTM variance
    realized_var_ref = DeflatedSharpeRatio().calculate(
        deannualize_sharpe(2.0), 31, 365, 0.0, 3.0, trials_sharpe_variance=0.0
    )
    assert realized_var_ref.p_value < 0.05  # ... and would pass with the realized one

    out = _screen(trials)
    assert out.total_combinations == 31
    assert out.passed_filters == 0  # no survivor: the champion fails DSR at the MTM variance


# --- storage ----------------------------------------------------------------


def test_save_sweep_results_persists_mtm_columns(tmp_path: Path) -> None:
    db, rid = _screening_db(tmp_path, [{
        "parameters": {"p": 1}, "sharpe_ratio": 1.0,
        "mtm_sharpe_daily": 0.05, "mtm_skewness": -0.2, "mtm_kurtosis": 4.0, "mtm_days": 300,
    }])  # fmt: skip
    sm = StateManager(db)
    try:
        (row,) = sm.get_sweep_results(rid)
    finally:
        sm.close()
    assert (row["mtm_sharpe_daily"], row["mtm_skewness"], row["mtm_kurtosis"], row["mtm_days"]) == (
        0.05, -0.2, 4.0, 300,
    )  # fmt: skip


def test_migration_adds_mtm_columns_to_old_db(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = get_connection(path)
    conn.executescript(SCHEMA_SQL)
    for col in MTM_COLUMNS:
        conn.execute(f"ALTER TABLE sweep_results DROP COLUMN {col}")
    conn.commit()
    names = {r[1] for r in conn.execute("PRAGMA table_info(sweep_results)")}
    assert not names & set(MTM_COLUMNS)
    init_schema(conn)
    names = {r[1] for r in conn.execute("PRAGMA table_info(sweep_results)")}
    assert set(MTM_COLUMNS) <= names
    conn.close()


# --- helper edge cases ------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        (None, 0.0, 3.0, 100),
        (0.1, None, 3.0, 100),
        (0.1, 0.0, None, 100),
        (0.1, 0.0, 3.0, None),
        (float("nan"), 0.0, 3.0, 100),
        (0.1, float("inf"), 3.0, 100),
        (0.1, 0.0, 3.0, 1),
    ],
)
def test_mtm_dsr_calculate_fails_closed(args: tuple[Any, ...]) -> None:
    result, reason = mtm_dsr_calculate(DeflatedSharpeRatio(), *args, num_trials=10)
    assert result is None and reason is not None and reason.startswith("DSR input missing")


# --- the runner stores the stats (real engine, skipped without catalog data) -


def _btc_catalog_available() -> bool:
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

    bar_dir = DEFAULT_CATALOG_PATH / "data" / "bar"
    return bar_dir.is_dir() and any("BTCUSDT-PERP.BINANCE" in d.name for d in bar_dir.iterdir())


def test_runner_stores_mtm_stats_for_every_run_equal_to_series_inputs() -> None:
    """Default runs (no series collection) carry mtm_*; they equal
    daily_sharpe_inputs(MTM series minus day 0) and differ from the realized Sharpe."""
    if not _btc_catalog_available():
        pytest.skip("BTCUSDT catalog data not available")
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.discovery.genome import chromosome_to_dsl
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 50.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 60.0, None)],
        stop_loss_pct=10.0, take_profit_pct=4.0, direction=Direction.LONG,
    )  # fmt: skip
    start, end = "2024-01-01", "2024-03-01"
    dsl = chromosome_to_dsl(chrom)
    dsl["timeframe"] = "4h"

    collecting = NTBacktestFn(["BTCUSDT"], "4h", start, end, collect_daily_returns=True)
    runner = collecting._make_runner(dsl, start, end, None)
    with_series = runner({})
    series = runner.last_mtm_daily_returns
    assert series is not None and with_series.total_trades > 0
    sharpe, skew, kurt = daily_sharpe_inputs(series.without_first_day().values)
    assert with_series.mtm_days == len(series) - 1
    assert (with_series.mtm_sharpe_daily, with_series.mtm_skewness, with_series.mtm_kurtosis) == (
        sharpe, skew, kurt,
    )  # fmt: skip
    assert math.isfinite(sharpe) and sharpe != 0.0

    plain = NTBacktestFn(["BTCUSDT"], "4h", start, end)
    plain_runner = plain._make_runner(dsl, start, end, None)
    without_series = plain_runner({})
    assert plain_runner.last_mtm_daily_returns is None  # series gate unchanged
    assert (
        without_series.mtm_sharpe_daily, without_series.mtm_skewness,
        without_series.mtm_kurtosis, without_series.mtm_days,
    ) == (sharpe, skew, kurt, len(series) - 1)  # fmt: skip
    assert without_series.sharpe_ratio == with_series.sharpe_ratio  # realized unchanged


def test_screening_save_results_round_trips_mtm_stats_into_recheck(tmp_path: Path) -> None:
    """ScreeningPipeline.save_results must persist all four MTM fields (non-default
    values), and the overfitting recheck must then test exactly those."""
    from vibe_quant.screening.types import ScreeningResult

    db = tmp_path / "rt.db"
    sm = StateManager(db)
    sid = sm.create_strategy("s", {"timeframe": "4h"})
    rid = sm.create_backtest_run(sid, "screening", ["BTCUSDT"], "4h", "2022-01-01", "2022-12-31", {})
    m = BacktestMetrics(
        parameters={"p": 1}, sharpe_ratio=9.0, profit_factor=1.5, max_drawdown=0.1,
        total_trades=50, total_return=0.1,
    )  # fmt: skip
    m.mtm_sharpe_daily, m.mtm_skewness, m.mtm_kurtosis, m.mtm_days = 0.0711, -0.37, 5.25, 287
    result = ScreeningResult(
        strategy_name="s", total_combinations=1, passed_filters=1,
        execution_time_seconds=0.0, results=[m], pareto_optimal_indices=[0],
    )  # fmt: skip
    pipeline = ScreeningPipeline.__new__(ScreeningPipeline)
    # save_results reads the run's liquidation settings AND its starting balance from
    # the instance; this test bypasses __init__, so set both defaults explicitly.
    from vibe_quant.liquidation_settings import LiquidationSettings

    pipeline._liquidation = LiquidationSettings()
    pipeline._starting_balance = 1000
    pipeline.save_results(result, sm, rid)
    (row,) = sm.get_sweep_results(rid)
    sm.close()
    assert (row["mtm_sharpe_daily"], row["mtm_skewness"], row["mtm_kurtosis"], row["mtm_days"]) == (
        0.0711, -0.37, 5.25, 287,
    )  # fmt: skip

    cand, _ = _run_dsr(db, rid, total_trials=3)
    ref = DeflatedSharpeRatio().calculate(0.0711, 3, 287, -0.37, 5.25)
    assert cand.dsr_result is not None
    assert cand.dsr_result.num_observations == 287
    assert cand.dsr_result.p_value == pytest.approx(ref.p_value, rel=1e-12)


def test_cli_dsr_recheck_of_pre_migration_db_promotes_and_judges(tmp_path: Path) -> None:
    """`python -m vibe_quant.overfitting` opens the DB with a raw connection (no
    init_schema): a DB from before the mtm_* columns must be migrated, not raise
    ``no column named mtm_sharpe_daily``; the champion's record is then promoted
    and judged (passes at N=10)."""
    from vibe_quant.overfitting.__main__ import main

    db, rid = _discovery_db(tmp_path, evaluated=10, record=_record())
    raw = sqlite3.connect(str(db))
    for col in MTM_COLUMNS:
        raw.execute(f"ALTER TABLE sweep_results DROP COLUMN {col}")
    raw.commit()
    raw.close()

    assert main(["run", "--run-id", str(rid), "--db", str(db), "--filters", "dsr"]) == 0

    raw = sqlite3.connect(str(db))
    raw.row_factory = sqlite3.Row
    row = raw.execute("SELECT * FROM sweep_results WHERE run_id = ?", (rid,)).fetchone()
    raw.close()
    assert row["mtm_days"] == T_DAYS
    assert row["mtm_sharpe_daily"] == deannualize_sharpe(MTM_SR_ANN)
    assert row["passed_deflated_sharpe"] == 1


def _old_schema_db(tmp_path: Path) -> tuple[Path, int]:
    db, rid = _discovery_db(tmp_path, evaluated=10, record=_record())
    raw = sqlite3.connect(str(db))
    for col in MTM_COLUMNS:
        raw.execute(f"ALTER TABLE sweep_results DROP COLUMN {col}")
    raw.commit()
    raw.close()
    return db, rid


@pytest.mark.parametrize("applied", [0, 3])
def test_incomplete_migration_fails_closed_and_next_call_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, applied: int
) -> None:
    """A busy fake for the shared migration helper: it leaves an mtm_* column
    missing (0 or 3 of 4 applied), so the connection is closed, ``_conn`` stays None
    and a clear RuntimeError is raised -- no half-migrated cached connection. Once
    the columns can be added, the next call retries and the recheck judges the champion."""
    import vibe_quant.overfitting.pipeline as pl

    db, rid = _old_schema_db(tmp_path)
    real = pl.migrate_add_columns
    seen: list[sqlite3.Connection] = []

    def busy(conn: sqlite3.Connection) -> None:
        seen.append(conn)
        for col in MTM_COLUMNS[1 : 1 + applied]:  # first column hits "database is locked"
            with contextlib.suppress(sqlite3.OperationalError):  # duplicate on 2nd call
                conn.execute(f"ALTER TABLE sweep_results ADD COLUMN {col} REAL")

    monkeypatch.setattr(pl, "migrate_add_columns", busy)
    p = OverfittingPipeline(db)
    with pytest.raises(RuntimeError, match=r"migration incomplete.*missing mtm_sharpe_daily"):
        p.conn  # noqa: B018
    assert p._conn is None
    with pytest.raises(sqlite3.ProgrammingError):  # the failed connection was closed
        seen[0].execute("SELECT 1")
    with pytest.raises(RuntimeError, match="migration incomplete"):  # still failing closed
        p.run(run_id=rid, config=FilterConfig.dsr_only())
    assert p._conn is None

    monkeypatch.setattr(pl, "migrate_add_columns", real)
    try:
        res = p.run(run_id=rid, config=FilterConfig.dsr_only())
    finally:
        p.close()
    assert res.candidates[0].passed_dsr is True  # champion promoted from its record, judged
