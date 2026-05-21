"""Auto-generated contract test for proposed indicator PRICE_POSITION.

Generated as part of the scaffold pipeline (bd-3p1k.1.3) — verifies the
synthesized compute_fn produces an output of the right shape and
isn't all-NaN past warmup. Re-run after editing the plugin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vibe_quant.dsl.indicators import indicator_registry, invoke_compute_fn


def _sample_ohlcv(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1000.0, 10000.0, n),
        },
        index=idx,
    )


def test_price_position_registered() -> None:
    spec = indicator_registry.get("PRICE_POSITION")
    assert spec is not None, "PRICE_POSITION plugin did not register"
    assert spec.compute_fn is not None


def test_price_position_contract_length_and_index() -> None:
    spec = indicator_registry.get("PRICE_POSITION")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, pd.Series)
    assert len(out) == len(df)
    assert out.index.equals(df.index)


def test_price_position_not_all_nan_past_warmup() -> None:
    spec = indicator_registry.get("PRICE_POSITION")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, pd.Series)
    # Past the second half of the series we expect at least one finite
    # value; a fully-NaN tail means the body computes nothing.
    tail = out.iloc[len(out) // 2 :]
    assert tail.notna().any(), "PRICE_POSITION produced all-NaN past warmup"
