"""Unit tests for ``full_range_headline``.

The helper decides a champion's headline metrics: the result of one continuous
full-range backtest when discovery used multi-window fitness and/or a train/test
split (so the GA aggregate is a *different* statistic than a continuous replay),
or the GA fitness verbatim when no separate run was needed. Promotion/replay_drift
consume the ``full_range_*`` fields it emits, so this locks both branches.

regression: vibe-quant-rewru
"""

from __future__ import annotations

from vibe_quant.discovery.backtest_fn import full_range_headline

# A realistic single-backtest metrics dict (shape NTBacktestFn returns).
_FR = {
    "sharpe_ratio": 2.804,
    "max_drawdown": 0.0254,
    "profit_factor": 1.802,
    "total_trades": 46,
    "total_return": 0.171,
    "skewness": 0.1,
    "kurtosis": 3.2,
}

_FALLBACK = dict(
    fallback_sharpe=4.327,
    fallback_trades=70,
    fallback_max_dd=0.025,
    fallback_pf=2.2,
    fallback_return=0.30,
)


def test_uses_full_range_metrics_when_present() -> None:
    """A full-range backtest dict wins over the GA fallback values."""
    out = full_range_headline(_FR, **_FALLBACK)
    assert out["full_range_sharpe"] == 2.804
    assert out["full_range_trades"] == 46
    assert out["full_range_max_dd"] == 0.0254
    assert out["full_range_pf"] == 1.802
    assert out["full_range_return_pct"] == 0.171


def test_falls_back_to_fitness_when_none() -> None:
    """None metrics (single-window/no-split, or mock) -> echo the GA fitness."""
    out = full_range_headline(None, **_FALLBACK)
    assert out["full_range_sharpe"] == 4.327
    assert out["full_range_trades"] == 70
    assert out["full_range_max_dd"] == 0.025
    assert out["full_range_pf"] == 2.2
    assert out["full_range_return_pct"] == 0.30


def test_failure_metrics_missing_total_return_defaults_zero() -> None:
    """NTBacktestFn's failure dict omits total_return -> headline return is 0.0."""
    failure = {
        "sharpe_ratio": -1.0,
        "max_drawdown": 1.0,
        "profit_factor": 0.0,
        "total_trades": 0,
    }
    out = full_range_headline(failure, **_FALLBACK)
    assert out["full_range_return_pct"] == 0.0
    assert out["full_range_sharpe"] == -1.0
    assert out["full_range_trades"] == 0
