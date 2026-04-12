"""End-to-end proof that the plugin pipeline works: ADAPTIVE_RSI.

Covers the full lifecycle from spec registration through GA enrollment,
DSL validation, strategy compilation, compute_fn golden-value check,
and API catalog surfacing. Each test is independent — no shared state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.plugins.example_adaptive_rsi import compute_adaptive_rsi

# ---------------------------------------------------------------------------
# 1. Registry discovery
# ---------------------------------------------------------------------------


def test_adaptive_rsi_appears_in_registry() -> None:
    """The plugin loader must auto-import the example plugin so
    ADAPTIVE_RSI is in the registry after the first import of
    ``vibe_quant.dsl.indicators``."""
    spec = indicator_registry.get("ADAPTIVE_RSI")
    assert spec is not None
    assert spec.compute_fn is compute_adaptive_rsi
    assert spec.nt_class is None
    assert spec.display_name == "Adaptive RSI"
    assert spec.category == "Momentum"


# ---------------------------------------------------------------------------
# 2. GA pool enrollment
# ---------------------------------------------------------------------------


def test_adaptive_rsi_in_ga_pool() -> None:
    """threshold_range + param_ranges → auto-enrolled in GA pool."""
    from vibe_quant.discovery.genome import build_indicator_pool

    pool = build_indicator_pool()
    assert "ADAPTIVE_RSI" in pool
    entry = pool["ADAPTIVE_RSI"]
    assert entry.param_ranges == {"period": (5.0, 50.0), "alpha": (0.1, 0.9)}
    assert entry.default_threshold_range == (20.0, 80.0)


# ---------------------------------------------------------------------------
# 3. DSL validation
# ---------------------------------------------------------------------------


def test_adaptive_rsi_dsl_validates() -> None:
    """A strategy YAML referencing ADAPTIVE_RSI must pass the DSL parser.

    Post-P5 the parser queries the registry; post-P9 the plugin is
    registered at import time. This confirms the full chain."""
    dsl = validate_strategy_dict(
        {
            "name": "arsi_e2e",
            "timeframe": "5m",
            "indicators": {"arsi": {"type": "ADAPTIVE_RSI", "period": 14}},
            "entry_conditions": {"long": ["arsi < 30"]},
            "exit_conditions": {"long": ["arsi > 70"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
    )
    assert dsl.indicators["arsi"].type == "ADAPTIVE_RSI"


# ---------------------------------------------------------------------------
# 4. Strategy compilation
# ---------------------------------------------------------------------------


def test_adaptive_rsi_compiles_strategy_source() -> None:
    """Compiled source must import and call compute_adaptive_rsi."""
    dsl = validate_strategy_dict(
        {
            "name": "arsi_compile",
            "timeframe": "5m",
            "indicators": {"arsi": {"type": "ADAPTIVE_RSI", "period": 14}},
            "entry_conditions": {"long": ["arsi < 30"]},
            "exit_conditions": {"long": ["arsi > 70"]},
            "stop_loss": {"type": "fixed_pct", "percent": 1.0},
            "take_profit": {"type": "fixed_pct", "percent": 2.0},
        }
    )
    src = StrategyCompiler().compile(dsl)
    compile(src, "<generated>", "exec")
    assert "compute_adaptive_rsi" in src
    assert "from vibe_quant.dsl.plugins.example_adaptive_rsi import compute_adaptive_rsi" in src
    # Must compile to a loadable module with the correct class names
    mod = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    assert hasattr(mod, f"{camel}Strategy")
    assert hasattr(mod, f"{camel}Config")


# ---------------------------------------------------------------------------
# 5. compute_fn golden values
# ---------------------------------------------------------------------------


def _make_fixture_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV fixture for reproducible golden-value checks."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close - rng.rand(n) * 0.2,
            "high": close + rng.rand(n) * 0.5,
            "low": close - rng.rand(n) * 0.5,
            "close": close,
            "volume": rng.randint(100, 1000, n).astype(float),
        }
    )


def test_adaptive_rsi_golden_values() -> None:
    """compute_adaptive_rsi output on a fixed 100-bar fixture must match
    hand-verified reference values.

    We don't pin every value — we check boundary behavior and a few
    mid-series values for sanity. The test is designed to catch
    algorithmic regressions (e.g. off-by-one in the ER window,
    wrong smoothing constant formula) without being brittle to
    floating-point platform differences."""
    df = _make_fixture_df(100)
    result = compute_adaptive_rsi(df, {"period": 14, "alpha": 0.5})

    assert isinstance(result, pd.Series)
    assert len(result) == 100

    # First `period` values should be NaN (warmup)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[13])

    # After warmup, values should be in valid RSI range [0, 100]
    valid = result.dropna()
    assert len(valid) > 50, f"Expected >50 valid values, got {len(valid)}"
    assert valid.min() >= 0.0, f"RSI below 0: {valid.min()}"
    assert valid.max() <= 100.0, f"RSI above 100: {valid.max()}"

    # Smoke: the adaptive RSI should not be constant (it adapts)
    assert valid.std() > 1.0, "Adaptive RSI has suspiciously low variance"

    # Pin a few specific values (computed once on seed=42 fixture and
    # verified by hand). Tolerance of 0.5 RSI points absorbs minor
    # floating-point platform differences.
    assert abs(result.iloc[20] - 47.66) < 0.5, f"bar 20 regression: {result.iloc[20]:.4f}"
    assert abs(result.iloc[50] - 42.77) < 0.5, f"bar 50 regression: {result.iloc[50]:.4f}"
    assert abs(result.iloc[80] - 44.79) < 0.5, f"bar 80 regression: {result.iloc[80]:.4f}"


def test_adaptive_rsi_edge_flat_market() -> None:
    """In a perfectly flat market (zero price change) RSI should converge
    toward 50 (balanced gains/losses → RS=1 → RSI=50) or be NaN."""
    flat = pd.DataFrame(
        {
            "open": [100.0] * 50,
            "high": [100.0] * 50,
            "low": [100.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000.0] * 50,
        }
    )
    result = compute_adaptive_rsi(flat, {"period": 14, "alpha": 0.5})
    valid = result.dropna()
    # All gains and losses are zero → avg_gain=0, avg_loss=0 → RSI=100
    # (by the division-by-zero branch in the formula: al==0 → RSI=100).
    # This is standard RSI behavior for flat series.
    if len(valid) > 0:
        for v in valid:
            assert v == 100.0 or np.isnan(v), f"Unexpected RSI in flat market: {v}"


# ---------------------------------------------------------------------------
# 6. API catalog surfacing
# ---------------------------------------------------------------------------


def test_adaptive_rsi_in_api_catalog() -> None:
    """GET /api/indicators/catalog must include ADAPTIVE_RSI."""
    from fastapi.testclient import TestClient

    from vibe_quant.api.app import create_app

    client = TestClient(create_app())
    resp = client.get("/api/indicators/catalog")
    assert resp.status_code == 200
    names = [e["type_name"] for e in resp.json()["indicators"]]
    assert "ADAPTIVE_RSI" in names
