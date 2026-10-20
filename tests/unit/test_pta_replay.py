"""``vibe_quant.dsl.pta_replay``: runtime-equivalent compute_fn replay.

The generated strategy calls each ``compute_fn`` once per bar with a capped
rolling ``PtaBuffer`` window. ``pta_replay`` reproduces that stream from a full
history by driving the same buffer schedule and calling the compute_fn once per
run of bars that share a window start. These tests pin the replay to the real
generated code path (``pta_stream_harness.run_stream``) for every spec the
compiler routes through it, and pin the per-bar rules (lookback gate,
NaN-hold) individually.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.unit.pta_stream_harness import compile_indicators, random_walk_bars, run_stream
from vibe_quant.dsl.indicators import (
    IndicatorSpec,
    indicator_registry,
    pta_buffer_cap,
    pta_lookback,
)
from vibe_quant.dsl.pta_buffer import PtaBuffer
from vibe_quant.dsl.pta_replay import pta_window_starts, replay_pta_series


def _ohlcv(bars: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [b[5] for b in bars],
        }
    )


def _runtime(snapshots: list[dict[str, float]], key: str) -> np.ndarray:
    return np.array([snap.get(key, np.nan) for snap in snapshots], dtype=np.float64)


def _pta_candidates() -> list[str]:
    """Registry types that could reach the generated compute_fn buffer path."""
    names: list[str] = []
    for name in sorted(indicator_registry.list_indicators()):
        spec = indicator_registry.get(name)
        if spec is not None and spec.compute_fn is not None and not spec.needs_context:
            names.append(name)
    return names


_PTA_CANDIDATES = _pta_candidates()


def _compiled_or_skip(name: str) -> type:
    """Compile ``name`` alone; skip specs the compiler does not buffer."""
    try:
        _, cls = compile_indicators({"ind": {"type": name}})
    except Exception as exc:  # noqa: BLE001 — a registry spec may need aux context
        pytest.skip(f"{name}: does not compile as a PTA-path strategy ({type(exc).__name__})")
    if not hasattr(cls, "_feed_pta_buffer"):
        pytest.skip(f"{name}: compiles through the NT path, not the compute_fn buffer path")
    return cls


# ---------------------------------------------------------------------------
# a. The window schedule is PtaBuffer's own.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [0, 400, 420, 2500])
def test_window_starts_match_ptabuffer_len(cap: int) -> None:
    """``start_i == i + 1 - len(buffer)`` after every real PtaBuffer append."""
    n_bars = 5000
    buf = PtaBuffer(cap)
    expected = np.empty(n_bars, dtype=np.int64)
    for i in range(n_bars):
        buf.append(1.0, 2.0, 3.0, 4.0, 5.0)
        expected[i] = i + 1 - len(buf)

    np.testing.assert_array_equal(pta_window_starts(n_bars, cap), expected)


def test_window_starts_trim_schedule_exact_for_cap_400() -> None:
    """25% slack: the first trim lands at bar 500 and starts repeat until bar 600."""
    starts = pta_window_starts(700, 400)

    assert starts[498] == 0
    assert starts[499] == 0
    assert starts[500] == 101
    assert starts[501] == 101
    assert starts[600] == 101
    assert starts[601] == 202


def test_window_starts_cap_zero_never_trims() -> None:
    starts = pta_window_starts(1500, 0)
    np.testing.assert_array_equal(starts, np.zeros(1500, dtype=np.int64))


# ---------------------------------------------------------------------------
# b. Registry-wide parity with the real generated stream (the tripwire).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _PTA_CANDIDATES)
def test_replay_equals_runtime_stream_for_every_registry_compute_spec(name: str) -> None:
    """Every compute_fn spec the compiler buffers replays bitwise == the runtime.

    Covers the primary value under ``ind`` and every sub-output under
    ``ind_<key>`` (MACD, ICHIMOKU). A non-causal future plugin, or a change to
    the runtime trim/read semantics, turns this red.
    """
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    params = dict(spec.default_params)
    cls = _compiled_or_skip(name)

    bars = random_walk_bars(1500, seed=13)
    snapshots = run_stream(cls, bars, memo=False, specs={"ind": (name, params, "1h")})
    cap = pta_buffer_cap([pta_lookback(spec, params)], full_history=spec.requires_full_history)
    replayed = replay_pta_series(spec, _ohlcv(bars), params, cap)

    assert list(replayed) == list(spec.output_names)
    primary = spec.primary_output or spec.output_names[0]
    assert primary in replayed

    for output_name in spec.output_names:
        key = "ind" if output_name == primary else f"ind_{output_name}"
        runtime = _runtime(snapshots, key)
        differing = int(
            (~((replayed[output_name] == runtime) | (np.isnan(replayed[output_name]) & np.isnan(runtime)))).sum()
        )
        assert np.array_equal(replayed[output_name], runtime, equal_nan=True), (
            f"{name}.{output_name}: replay differs from the generated stream on "
            f"{differing}/{len(runtime)} bars (key {key})"
        )


def test_full_history_spec_replays_with_cap_zero() -> None:
    """A ``requires_full_history`` spec must replay as one bar-0 segment (cap 0)."""
    compiling: list[str] = []
    for name in _PTA_CANDIDATES:
        spec = indicator_registry.get(name)
        assert spec is not None
        if not spec.requires_full_history:
            continue
        try:
            _, cls = compile_indicators({"ind": {"type": name}})
        except Exception:  # noqa: BLE001 — not every spec compiles onto the buffer path
            continue
        if hasattr(cls, "_feed_pta_buffer"):
            compiling.append(name)

    if not compiling:
        pytest.skip("no requires_full_history spec compiles onto the compute_fn buffer path")

    bars = random_walk_bars(700, seed=17)
    for name in compiling:
        spec = indicator_registry.get(name)
        assert spec is not None
        params = dict(spec.default_params)
        cap = pta_buffer_cap([pta_lookback(spec, params)], full_history=True)
        assert cap == 0
        _, cls = compile_indicators({"ind": {"type": name}})
        snapshots = run_stream(cls, bars, memo=False, specs={"ind": (name, params, "1h")})
        replayed = replay_pta_series(spec, _ohlcv(bars), params, cap)
        np.testing.assert_array_equal(
            replayed[spec.output_names[0]], _runtime(snapshots, "ind"), err_msg=name
        )


# ---------------------------------------------------------------------------
# c. NaN-hold: a NaN after a valid value keeps the previous value (runtime rule).
# ---------------------------------------------------------------------------


def test_replay_holds_last_valid_value_when_compute_returns_nan() -> None:
    """Output is NaN on every 7th row; the replay must hold the last valid one."""

    def _holes(df: pd.DataFrame, _params: dict[str, object]) -> pd.Series:
        values = df["close"].to_numpy(dtype=np.float64).copy()
        values[np.arange(len(df)) % 7 == 6] = np.nan
        return pd.Series(values)

    name = "CHART_NAN_HOLD_PROBE"
    indicator_registry.register_spec(
        IndicatorSpec(
            name=name,
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 5},
            param_schema={},
            output_names=("value",),
            compute_fn=_holes,
        )
    )
    try:
        spec = indicator_registry.get(name)
        assert spec is not None and spec.compute_fn is not None
        params = dict(spec.default_params)
        bars = random_walk_bars(120, seed=5)

        replayed = replay_pta_series(spec, _ohlcv(bars), params, 400)["value"]

        lookback = pta_lookback(spec, params)
        expected = np.full(len(bars), np.nan)
        held = np.nan
        for i in range(len(bars)):
            if i + 1 >= lookback and i % 7 != 6:
                held = bars[i][4]
            expected[i] = held
        np.testing.assert_array_equal(replayed, expected)

        # Same answer from the real generated path when the probe spec compiles.
        try:
            _, cls = compile_indicators({"ind": {"type": name}})
        except Exception:  # noqa: BLE001 — hand-computed expectation above still holds
            return
        if not hasattr(cls, "_feed_pta_buffer"):
            return
        snapshots = run_stream(cls, bars, memo=False, specs={"ind": (name, params, "1h")})
        np.testing.assert_array_equal(replayed, _runtime(snapshots, "ind"))
    finally:
        indicator_registry.unregister(name)


# ---------------------------------------------------------------------------
# d. Lookback gate: bars before the runtime's warmup are NaN.
# ---------------------------------------------------------------------------


def test_replay_below_lookback_is_nan() -> None:
    spec = indicator_registry.get("KAMA")
    assert spec is not None
    params = {"period": 10}
    lookback = pta_lookback(spec, params)
    assert lookback > 1

    bars = random_walk_bars(120, seed=9)
    replayed = replay_pta_series(spec, _ohlcv(bars), params, 400)["value"]

    assert np.isnan(replayed[: lookback - 1]).all()
    assert np.isfinite(replayed[lookback - 1])
