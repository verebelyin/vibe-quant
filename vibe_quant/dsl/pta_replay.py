"""Runtime-equivalent replay of compute_fn indicators over a full history.

Generated strategies feed every ``compute_fn`` indicator a capped rolling
:class:`~vibe_quant.dsl.pta_buffer.PtaBuffer`: append one bar, trim once
``len > cap + cap // 4``, recompute, read the last value. A chart overlay that
computes over the whole request history diverges from that stream for
path-dependent indicators (ADAPTIVE_RSI, VIDYA, ...) and skips the runtime's
warmup gate and NaN-hold.

:func:`replay_pta_series` reproduces the stream exactly: it derives the window
start seen at every bar from a real ``PtaBuffer`` schedule
(:func:`pta_window_starts`), calls the compute_fn once per run of bars that
share a window start, and applies the generated ``_update_pta_indicators``
rules (lookback gate, keep the last non-NaN value). One compute call per
segment makes this O(bars) to schedule plus O(segments) compute instead of one
call per bar. It is bit-exact for causal compute_fns — the registry-wide
parity test is the guard against a non-causal future plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.indicators import IndicatorSpec, pta_lookback
from vibe_quant.dsl.pta_buffer import PtaBuffer

if TYPE_CHECKING:
    import pandas as pd

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def pta_window_starts(n_bars: int, cap: int) -> np.ndarray:
    """First history row of the buffer window the runtime sees at each bar.

    Drives a real :class:`PtaBuffer` with one append per bar and reads its
    length, so the trim schedule stays defined only by ``PtaBuffer.append``.
    ``cap == 0`` never trims, so every start is 0.
    """
    starts = np.empty(n_bars, dtype=np.int64)
    buf = PtaBuffer(cap)
    for i in range(n_bars):
        buf.append(0.0, 0.0, 0.0, 0.0, 0.0)
        starts[i] = i + 1 - len(buf)
    return starts


def replay_pta_series(
    spec: IndicatorSpec,
    ohlcv: pd.DataFrame,
    params: dict[str, object],
    cap: int,
) -> dict[str, np.ndarray]:
    """Replay the runtime's per-bar compute_fn stream over a whole history.

    Args:
        spec: Registry spec whose ``compute_fn`` the runtime calls.
        ohlcv: Rows with ``open, high, low, close, volume`` for the request.
        params: Effective params (spec defaults merged under request params).
        cap: Rolling-buffer cap the strategy would use for this indicator.

    Returns:
        One float64 array per entry of ``spec.output_names``, aligned with the
        ``ohlcv`` rows (NaN before the lookback gate and wherever the runtime
        has not produced a value yet).
    """
    compute_fn = spec.compute_fn
    if compute_fn is None:
        msg = f"Indicator {spec.name!r} has no compute_fn to replay"
        raise ValueError(msg)

    names = spec.output_names
    frame = ohlcv.loc[:, list(_OHLCV_COLUMNS)].astype("float64").reset_index(drop=True)
    n_bars = len(frame)
    output = {name: np.full(n_bars, np.nan, dtype=np.float64) for name in names}
    held = dict.fromkeys(names, np.nan)
    starts = pta_window_starts(n_bars, cap)
    lookback = pta_lookback(spec, params)

    i = 0
    while i < n_bars:
        start = int(starts[i])
        end = i
        while end + 1 < n_bars and int(starts[end + 1]) == start:
            end += 1
        window = frame.iloc[start : end + 1].reset_index(drop=True)
        segment = _segment_arrays(compute_fn(window, params), names, len(window))
        for bar in range(i, end + 1):
            pos = bar - start
            if pos + 1 < lookback:
                continue
            for name in names:
                value = float(segment[name][pos])
                if not np.isnan(value):
                    held[name] = value
                output[name][bar] = held[name]
        i = end + 1

    return output


def _segment_arrays(
    result: object, names: tuple[str, ...], n_rows: int
) -> dict[str, np.ndarray]:
    """Float64 per-output arrays for one compute call (malformed output → NaN)."""
    missing = np.full(n_rows, np.nan, dtype=np.float64)
    arrays: dict[str, np.ndarray] = dict.fromkeys(names, missing)

    raw: dict[str, object]
    if isinstance(result, dict):
        raw = {str(key): value for key, value in result.items()}
    elif len(names) == 1:
        raw = {names[0]: result}
    else:
        return arrays

    for name in names:
        value = raw.get(name)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (n_rows,):
            arrays[name] = array
    return arrays
