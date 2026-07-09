"""Contract + behavior tests for the RAMS plugin (Reddit extraction 447)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.plugins.rams import compute_rams


def _df(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": pd.Series(np.full(len(close), 100.0)),
        }
    )


def test_registered_and_ga_eligible() -> None:
    spec = indicator_registry.get("RAMS")
    assert spec is not None
    assert spec.threshold_range == (1.8, 3.2)
    assert spec.param_ranges == {"period": (20.0, 100.0)}
    assert spec.compute_fn is compute_rams


def test_flat_market_scores_near_neutral() -> None:
    """Flat close: macro=1, slope=0, proximity=1, rsi=0.5 → ~2.5."""
    close = pd.Series(np.full(400, 100.0))
    score = compute_rams(_df(close), {"period": 50})
    valid = score.dropna()
    assert len(valid) > 100
    assert abs(valid.iloc[-1] - 2.5) < 0.01


def test_uptrend_scores_higher_than_downtrend() -> None:
    n = 400
    up = pd.Series(100.0 + np.linspace(0, 60, n))
    down = pd.Series(160.0 - np.linspace(0, 60, n))
    s_up = compute_rams(_df(up), {"period": 50}).iloc[-1]
    s_down = compute_rams(_df(down), {"period": 50}).iloc[-1]
    assert s_up > 2.5
    assert s_down < 2.0
    assert s_up - s_down > 0.8


def test_causal_no_lookahead() -> None:
    """Score at bar t must not change when future bars are appended."""
    rng = np.random.RandomState(3)
    close = pd.Series(100.0 + np.cumsum(rng.randn(400) * 0.5))
    full = compute_rams(_df(close), {"period": 50})
    truncated = compute_rams(_df(close.iloc[:300].reset_index(drop=True)), {"period": 50})
    np.testing.assert_allclose(
        full.iloc[:300].to_numpy(), truncated.to_numpy(), rtol=0, atol=0
    )


def test_warmup_is_nan_then_valid() -> None:
    close = pd.Series(100.0 + np.arange(400, dtype=float) * 0.1)
    score = compute_rams(_df(close), {"period": 50})
    assert score.iloc[: _warmup() - 1].isna().all()
    assert score.iloc[_warmup() :].notna().all()


def _warmup() -> int:
    # macro_slow (200) dominates the rolling warmup
    return 200
