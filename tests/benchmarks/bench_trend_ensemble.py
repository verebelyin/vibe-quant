"""Performance benchmark for the TREND_ENSEMBLE compute_fn.

Moved out of the unit gate (a wall-clock assert flakes on loaded machines).

Run with: python3.13 tests/benchmarks/bench_trend_ensemble.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from vibe_quant.dsl.plugins.trend_ensemble import compute_trend_ensemble


def _df(close: np.ndarray) -> pd.DataFrame:
    close_series = pd.Series(np.asarray(close, dtype=np.float64))
    return pd.DataFrame(
        {
            "open": close_series,
            "high": close_series + 1.0,
            "low": close_series - 1.0,
            "close": close_series,
            "volume": pd.Series(np.full(len(close_series), 100.0)),
        }
    )


def bench_compute_fn(iterations: int = 20) -> float:
    """Return average ms/call for a 2510-bar buffer with max_lookback=250.

    The pandas rolling/shift path measured ~2.2-2.5 ms/call; the scipy
    channel path brings that to ~0.7 ms, so 1.5 ms is a generous CI-noise
    margin for the old unit-gate threshold.
    """
    rng = np.random.RandomState(3)
    df = _df(100.0 + np.cumsum(rng.randn(2510)))
    params = {"max_lookback": 250}
    compute_trend_ensemble(df, params)  # warm-up: scipy import + caches

    start = time.perf_counter()
    for _ in range(iterations):
        compute_trend_ensemble(df, params)
    return (time.perf_counter() - start) / iterations * 1000.0


def main() -> None:
    ms_per_call = bench_compute_fn()
    print(f"TREND_ENSEMBLE compute_fn (2510 bars, max_lookback=250): {ms_per_call:.3f} ms/call")


if __name__ == "__main__":
    main()
