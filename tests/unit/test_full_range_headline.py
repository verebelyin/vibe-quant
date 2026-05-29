"""Unit tests for ``full_range_headline`` (bd vibe-quant-rewru).

The helper decides a champion's headline metrics: the result of one continuous
full-range backtest when discovery used multi-window fitness and/or a train/test
split (so the GA aggregate is a *different* statistic than a continuous replay),
or the GA fitness verbatim when no separate run was needed. Promotion/replay_drift
consume the ``full_range_*`` fields it emits, so this locks both branches.
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

_KEYS = {
    "full_range_sharpe",
    "full_range_trades",
    "full_range_max_dd",
    "full_range_pf",
    "full_range_return_pct",
}


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


def test_both_branches_emit_exactly_the_five_keys() -> None:
    """Field set is stable regardless of branch, so the entry shape is uniform."""
    assert set(full_range_headline(_FR, **_FALLBACK)) == _KEYS
    assert set(full_range_headline(None, **_FALLBACK)) == _KEYS


def test_coerces_numeric_types() -> None:
    """trades is int, the rest float -- even if inputs arrive as the other type."""
    out = full_range_headline(
        {
            "sharpe_ratio": 2,  # int in -> float out
            "max_drawdown": 0,
            "profit_factor": 1,
            "total_trades": 46.0,  # float in -> int out
            "total_return": 0,
        },
        **_FALLBACK,
    )
    assert isinstance(out["full_range_trades"], int) and out["full_range_trades"] == 46
    assert isinstance(out["full_range_sharpe"], float) and out["full_range_sharpe"] == 2.0

    # Fallback branch coerces too.
    out2 = full_range_headline(
        None,
        fallback_sharpe=3,
        fallback_trades=70.0,
        fallback_max_dd=0,
        fallback_pf=2,
        fallback_return=0,
    )
    assert isinstance(out2["full_range_trades"], int) and out2["full_range_trades"] == 70
    assert isinstance(out2["full_range_sharpe"], float)
