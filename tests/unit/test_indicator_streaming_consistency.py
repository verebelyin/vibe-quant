"""Registry-driven no-look-ahead guard for every ``compute_fn`` indicator.

The compiler's per-bar dispatcher re-invokes an indicator's ``compute_fn`` on
the *growing* close/OHLCV buffer each bar and reads ``.iloc[-1]``. That path is
structurally causal (the buffer holds no future bars). The leak this guard
catches is the offline/online divergence class from the Reddit audit -- a
``compute_fn`` that peeks forward (full-series min-max normalize, a centered
window, an accidental ``.shift(-1)``, a ``dropna`` that misaligns rows). Such an
indicator's value *at bar t* would change as later bars arrive.

Invariant (generalizes the VIDYA buffer-stability test in
``test_plugins/test_adaptive_mas.py``): for any indicator, the value assigned to
the bar at label ``L`` must not change when the buffer is extended past ``L``::

    compute_fn(df[:t+1]).loc[L] == compute_fn(df[:t+1+k]).loc[L]   for all k >= 1

Comparison is by index *label*, not position, so it survives indicators that
drop warm-up rows (STOCH preserves a truncated 13..N index) and skips
forward-projected outputs that have no historical label (Ichimoku's senkou
spans live at future indices and are causal in the streaming buffer anyway).

The guard is registry-driven: a future plugin whose ``compute_fn`` peeks forward
fails automatically, with no edit here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl.indicators import indicator_registry

if TYPE_CHECKING:
    from collections.abc import Callable

# Indicators that ship a pure-Python compute path. nt_class-only specs stream
# natively through NT and have no compute_fn to exercise here.
_COMPUTE_FN_NAMES = sorted(
    spec.name for spec in indicator_registry.all_specs() if spec.compute_fn is not None
)

# Sampled bar labels + look-ahead horizons. Labels sit well past every built-in
# warm-up (Ichimoku senkou=52, PRICE_POSITION period=52) so values are live.
_LABEL_INDICES = (120, 180, 240)
_HORIZONS = (1, 10, 40)
_N_BARS = 320

# np.isclose tolerances: tight enough that any genuine forward-peek (which
# shifts values by orders of magnitude, as the leaky fixtures below show) trips
# the guard, loose enough to absorb ULP-level float noise across buffer lengths.
_RTOL = 1e-9
_ATOL = 1e-12


@pytest.fixture(scope="module")
def ohlcv_df() -> pd.DataFrame:
    """Deterministic OHLCV with a default RangeIndex (label == position)."""
    rng = np.random.RandomState(7)
    close = 100.0 + np.cumsum(rng.randn(_N_BARS) * 0.5)
    return pd.DataFrame(
        {
            "open": close - rng.rand(_N_BARS) * 0.2,
            "high": close + rng.rand(_N_BARS) * 0.5,
            "low": close - rng.rand(_N_BARS) * 0.5,
            "close": close,
            "volume": rng.randint(100, 1000, _N_BARS).astype(float),
        }
    )


def _output_keys(out: object) -> list[str | None]:
    return list(out.keys()) if isinstance(out, dict) else [None]


def _series(out: object, key: str | None) -> pd.Series:
    return out[key] if isinstance(out, dict) and key is not None else out  # type: ignore[index,return-value]


def _assert_compute_fn_causal(
    compute_fn: Callable[[pd.DataFrame, dict[str, object]], object],
    params: dict[str, object],
    df: pd.DataFrame,
) -> int:
    """Assert bar-``t`` values are invariant to future bars. Returns the number
    of (output, bar, horizon) comparisons actually made so callers can prove the
    indicator was genuinely exercised rather than silently all-skipped."""
    keys = _output_keys(compute_fn(df.iloc[: _LABEL_INDICES[-1] + 1].copy(), params))
    compared = 0
    for t in _LABEL_INDICES:
        label = df.index[t]
        short = compute_fn(df.iloc[: t + 1].copy(), params)
        for k in _HORIZONS:
            end = t + 1 + k
            if end > len(df):
                continue
            longer = compute_fn(df.iloc[:end].copy(), params)
            for key in keys:
                short_s = _series(short, key)
                long_s = _series(longer, key)
                # Skip outputs with no value at this historical label (warm-up
                # rows dropped, or forward-projected like Ichimoku spans).
                if label not in short_s.index or label not in long_s.index:
                    continue
                a = short_s.loc[label]
                b = long_s.loc[label]
                a_nan, b_nan = pd.isna(a), pd.isna(b)
                if a_nan and b_nan:
                    continue
                out = "value" if key is None else key
                assert not (a_nan or b_nan), (
                    f"{out} at bar {t} flips NaN<->value when {k} future bar(s) "
                    f"are appended (short={a!r} long={b!r}) -- look-ahead"
                )
                assert np.isclose(float(a), float(b), rtol=_RTOL, atol=_ATOL), (
                    f"{out} at bar {t} changed from {a!r} to {b!r} when {k} "
                    f"future bar(s) were appended -- indicator peeks forward"
                )
                compared += 1
    return compared


@pytest.mark.parametrize("name", _COMPUTE_FN_NAMES)
def test_compute_fn_has_no_look_ahead(name: str, ohlcv_df: pd.DataFrame) -> None:
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    compared = _assert_compute_fn_causal(
        spec.compute_fn, dict(spec.default_params), ohlcv_df
    )
    assert compared > 0, (
        f"{name}: no comparable historical-label outputs -- the causality "
        "assertion never ran (all outputs warm-up-NaN or forward-projected?)"
    )


def test_registry_compute_fn_indicators_are_discovered() -> None:
    """Sanity: the parametrization actually found the built-in compute_fn set,
    so a registry regression can't silently empty this guard."""
    assert len(_COMPUTE_FN_NAMES) >= 20
    for expected in ("RSI", "MACD", "STOCH", "BBANDS", "ICHIMOKU"):
        assert expected in _COMPUTE_FN_NAMES


# --- Negative controls: prove the guard actually bites -----------------------


def _leaky_full_series_normalize(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:  # noqa: ARG001
    """Min-max normalize over the WHOLE series -- value at bar t depends on the
    global min/max, which shift as future bars arrive."""
    c = df["close"]
    return (c - c.min()) / (c.max() - c.min())


def _leaky_next_bar_peek(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:  # noqa: ARG001
    """Classic unlagged signal: pull bar t+1's close onto bar t via shift(-1)."""
    return df["close"].shift(-1)


@pytest.mark.parametrize(
    "leaky_fn", [_leaky_full_series_normalize, _leaky_next_bar_peek]
)
def test_guard_rejects_forward_peeking_compute_fn(
    leaky_fn: Callable[[pd.DataFrame, dict[str, object]], pd.Series],
    ohlcv_df: pd.DataFrame,
) -> None:
    with pytest.raises(AssertionError):
        _assert_compute_fn_causal(leaky_fn, {}, ohlcv_df)
