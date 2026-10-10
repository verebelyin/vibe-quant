"""Discovery window-config errors abort BEFORE the GA (vibe-quant-zhcq7 part 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery import config_check
from vibe_quant.discovery.config_check import (
    DiscoveryConfigError,
    cross_window_ranges_for,
    parse_cross_window_months,
    plan_windows,
    wfa_window_ranges_for,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.errors import DataUnavailableError
from vibe_quant.screening.nt_runner import MissingBarDataError
from vibe_quant.utils import split_date_range, split_into_windows

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome


class _Counting:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, chrom: StrategyChromosome) -> dict[str, Any]:
        self.calls += 1
        import random

        r = random.Random(7)
        return {
            "sharpe_ratio": 2.0, "max_drawdown": 0.08, "profit_factor": 2.0,
            "total_trades": 150, "total_return": 0.5,
            "trade_returns": tuple(r.gauss(0.0033, 0.001) for _ in range(150)),
        }


def _cfg(**overrides: Any) -> DiscoveryConfig:
    d: dict[str, Any] = {
        "population_size": 4, "max_generations": 2, "elite_count": 1,
        "tournament_size": 2, "convergence_generations": 1, "top_k": 1,
        "min_trades": 1, "max_workers": 1, "symbols": ["BTCUSDT"], "timeframe": "1h",
        "start_date": "2024-01-01", "end_date": "2024-02-15",
        "train_test_split": 0.5,
        "holdout_start_date": "2024-02-15", "holdout_end_date": "2024-04-01",
        "require_dsr": False,
    }
    d.update(overrides)
    return DiscoveryConfig(**d)


def _pipe(fn: _Counting, **overrides: Any) -> DiscoveryPipeline:
    return DiscoveryPipeline(
        _cfg(**overrides), fn, holdout_backtest_fn=fn,
        backtest_fn_factory=lambda s, e: fn,
    )


@pytest.fixture(autouse=True)
def _no_preflights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_aux_data", lambda self: None)


def test_too_short_cross_window_aborts_before_ga() -> None:
    fn = _Counting()
    with pytest.raises(DiscoveryConfigError, match="shorter than 7d"):
        _pipe(fn, cross_window_months=[2]).run()
    assert fn.calls == 0


def test_wfa_step_longer_than_train_aborts_before_ga() -> None:
    fn = _Counting()
    with pytest.raises(DiscoveryConfigError, match="too short for 90d"):
        _pipe(fn, wfa_oos_step_days=90).run()
    assert fn.calls == 0


@pytest.mark.parametrize("months", [[0], [-1], [1, 1]])
def test_bad_cross_window_months_rejected_before_ga(months: list[int]) -> None:
    fn = _Counting()
    with pytest.raises(DiscoveryConfigError):
        _pipe(fn, cross_window_months=months).run()
    assert fn.calls == 0


def test_no_factory_keeps_failing_closed_after_ga() -> None:
    fn = _Counting()
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[2]), fn, holdout_backtest_fn=fn,
    )
    result = pipe.run()  # must not raise
    assert fn.calls > 0
    assert result.top_strategies == []
    assert any(r["stage"] == "cross_window" for r in result.guardrail_rejections)


def test_several_bad_knobs_one_error_lists_all() -> None:
    with pytest.raises(DiscoveryConfigError) as ei:
        plan_windows(
            "2024-01-01", "2024-04-01", train_test_split=0.5, eval_windows=10,
            cross_window_months=[0, 5, 5], wfa_oos_step_days=90,
        )
    msg = str(ei.value)
    for needle in ("must be >= 1", "duplicates", "eval_windows", "shorter than 7d", "WFA"):
        assert needle in msg
    assert msg.count("; ") >= 4


def test_bad_split_collected_with_months() -> None:
    with pytest.raises(DiscoveryConfigError, match="must be >= 1.*; .*too short for split"):
        plan_windows(
            "2024-01-01", "2024-01-02", train_test_split=0.5, eval_windows=1,
            cross_window_months=[0], wfa_oos_step_days=0,
        )


def test_not_a_data_unavailable_error() -> None:
    assert issubclass(DiscoveryConfigError, ValueError)
    assert not issubclass(DiscoveryConfigError, DataUnavailableError)


def test_pure_ranges_match_pipeline_methods_and_legacy_dates() -> None:
    cfg = _cfg(
        start_date="2024-01-01", end_date="2024-10-19", train_test_split=0.8,
        holdout_start_date="2024-10-19", holdout_end_date="2025-01-01",
        cross_window_months=[1, 3], wfa_oos_step_days=30,
    )
    pipe = DiscoveryPipeline(cfg, backtest_fn=lambda c: {})
    cross = cross_window_ranges_for([1, 3], "2024-01-01", "2024-10-19", "2024-10-19")
    assert cross == pipe.cross_window_ranges()
    assert cross == [(1, "2024-02-01", "2024-10-19"), (3, "2024-04-01", "2024-10-19")]
    wfa = wfa_window_ranges_for(30, "2024-01-01", "2024-10-19", "2024-10-19")
    assert wfa == pipe.wfa_window_ranges("2024-01-01", "2024-10-19")
    assert wfa[0] == ("2024-01-01", "2024-01-31") and len(wfa) == 9


def test_plan_windows_uses_the_shared_split_functions() -> None:
    plan = plan_windows(
        "2024-01-01", "2025-01-01", train_test_split=0.8, eval_windows=3,
        cross_window_months=[1], wfa_oos_step_days=30,
    )
    ts, te, hs, he = split_date_range("2024-01-01", "2025-01-01", 0.8)
    assert (plan.train_start, plan.train_end, plan.holdout_start, plan.holdout_end) == (
        ts, te, hs, he,
    )
    assert plan.eval_windows == split_into_windows(ts, te, 3)


def test_plan_windows_splits_via_utils_not_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dates come from the (patched) shared split function, not a local copy."""
    monkeypatch.setattr(
        config_check, "split_date_range", lambda s, e, r: ("2024-01-01", "2024-06-01", "2024-06-01", "2025-01-01"),
    )
    plan = plan_windows(
        "2024-01-01", "2025-01-01", train_test_split=0.8, eval_windows=1,
        cross_window_months=[], wfa_oos_step_days=0,
    )
    assert plan.train_end == "2024-06-01"  # 0.8 would give 2024-10-20


@pytest.mark.parametrize(
    ("raw", "expected"), [("1,3,6", [1, 3, 6]), (" 2 , ,4", [2, 4]), ("", [])],
)
def test_parse_cross_window_months(raw: str, expected: list[int]) -> None:
    assert parse_cross_window_months(raw) == expected


def test_parse_cross_window_months_non_int() -> None:
    with pytest.raises(DiscoveryConfigError, match="'x'"):
        parse_cross_window_months("1,x")


def test_gate_range_data_unavailable_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = _Counting()
    pipe = _pipe(fn, cross_window_months=[1])

    def _boom() -> Any:
        raise MissingBarDataError("No catalog bars")

    monkeypatch.setattr(pipe, "cross_window_ranges", _boom)
    with pytest.raises(MissingBarDataError):
        pipe.run()
    assert fn.calls == 0


def test_wfa_overlap_in_gate_fails_closed() -> None:
    """_evaluate_wfa_rolling's range call is inside the fail-closed try."""
    fn = _Counting()
    pipe = _pipe(fn, wfa_oos_step_days=30)
    results, error = pipe._evaluate_wfa_rolling([], "2024-01-01", "2024-04-01")
    assert results == []
    assert error is not None and "overlaps the holdout" in error


def test_wfa_gate_range_data_unavailable_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = _Counting()
    pipe = _pipe(fn, wfa_oos_step_days=30)

    def _boom(*_a: Any) -> Any:
        raise MissingBarDataError("No catalog bars")

    monkeypatch.setattr(pipe, "wfa_window_ranges", _boom)
    with pytest.raises(MissingBarDataError):
        pipe.run()
    assert fn.calls == 0
