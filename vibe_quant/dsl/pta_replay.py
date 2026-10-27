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

A ``needs_context`` compute_fn flagged ``asof_context`` does not use that stream:
its value depends only on each row's bar close time, symbol, params and archive,
so :func:`asof_context_columns` computes it once over the whole frame and the
fast tier writes one column value per bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry, pta_lookback
from vibe_quant.dsl.pta_buffer import PtaBuffer

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class PtaReplayUnsupported(ValueError):
    """A compute_fn indicator that cannot be replayed (missing, no compute_fn, needs context)."""


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


def strategy_pta_columns(
    pta_params: dict[str, dict[str, object]],
    meta: dict[str, tuple[str, str]],
    caps: dict[str, int],
    tf: str,
    frame: pd.DataFrame,
    *,
    replay: Callable[
        [IndicatorSpec, pd.DataFrame, dict[str, object], int], dict[str, np.ndarray]
    ] = replay_pta_series,
) -> list[tuple[str, np.ndarray]]:
    """(``_pta_values`` key, held value per bar) for a strategy's compute_fn indicators on ``tf``.

    The one rule shared by the fast tier and the NT precompute seam. ``pta_params``
    and ``caps`` are the strategy's ``_pta_params`` / ``_pta_buffer_cap``, ``meta``
    the generated ``_PTA_META`` (indicator name -> (registry type, timeframe)),
    ``frame`` the OHLCV rows of ``tf``'s bars. Keys are exactly those the generated
    ``_update_pta_indicators`` writes: a single output under ``name``; a multi-output
    indicator's primary under ``name`` (when produced) and every output under
    ``name_<key>``. ``replay`` lets a caller memoise the per-indicator replay.

    Raises:
        PtaReplayUnsupported: an indicator of ``tf`` is unknown, has no compute_fn
            or needs aux context.
    """
    from vibe_quant.dsl.compiler import StrategyCompiler

    columns: list[tuple[str, np.ndarray]] = []
    for name, params in pta_params.items():
        indicator_type, indicator_tf = meta[name]
        if indicator_tf != tf:
            continue
        spec = indicator_registry.get(indicator_type)
        if spec is None or spec.compute_fn is None or spec.needs_context:
            raise PtaReplayUnsupported(f"compute_fn indicator {name} cannot be replayed")
        out = replay(spec, frame, params, caps[tf])
        names = spec.output_names
        if len(names) == 1:
            columns.append((name, out[names[0]]))
        else:
            primary = StrategyCompiler._effective_primary(spec)
            if primary in out:
                columns.append((name, out[primary]))
            columns.extend((f"{name}_{key}", out[key]) for key in names)
    return columns


def bar_close_ns(ts_init: np.ndarray, step_ns: int) -> np.ndarray:
    """Bar close times, int64: ``((ts_init + step//2) // step) * step``.

    The generated ``_feed_pta_buffer`` rounds every bar's ``ts_init`` to its
    timeframe boundary with exactly this rule before an as-of aux lookup
    (``bar_close_ns`` in that buffer's ``frame().attrs``), so a 1m bar whose
    ``ts_init`` is ``00:00:59.999`` looks up data closed at ``00:01:00.000``.
    """
    ts = np.asarray(ts_init, dtype=np.int64)
    step = np.int64(step_ns)
    return ((ts + step // 2) // step) * step


def asof_context_columns(
    pta_params: dict[str, dict[str, object]],
    meta: dict[str, tuple[str, str]],
    tf: str,
    frame: pd.DataFrame,
    close_ns: np.ndarray,
    instrument_id: object,
) -> list[tuple[str, np.ndarray, int]]:
    """(``_pta_values`` key, value per bar, first written row) for ``tf``'s as-of indicators.

    Every ``asof_context`` spec of ``tf`` gets ONE ``compute_fn`` call over the
    whole run frame instead of one per bar. The generated strategy calls it over
    a rolling buffer and keeps the last row: for a pointwise as-of lookup that is
    exactly ``column[k]`` at bar ``k``, so the whole-frame call is bit-identical
    (see ``IndicatorSpec.asof_context``).

    ``frame`` is the OHLCV of ``tf``'s bars and is never mutated: the call sees a
    shallow copy carrying ``attrs["bar_close_ns"] = close_ns`` (``bar_close_ns``
    of the run) and ``attrs["symbol"] = instrument_id`` (the same object the
    generated code puts there). Values are NOT held: a NaN is a real answer.
    ``first`` is ``pta_lookback(spec, params) - 1``; rows before it are never
    written by the generated feed either. Non-asof indicators are skipped.

    Raises:
        PtaReplayUnsupported: an indicator of ``tf`` is unknown, or its
            ``compute_fn`` does not return one value per bar.
    """
    columns: list[tuple[str, np.ndarray, int]] = []
    n_rows = len(frame)
    for name, params in pta_params.items():
        indicator_type, indicator_tf = meta[name]
        if indicator_tf != tf:
            continue
        spec = indicator_registry.get(indicator_type)
        compute_fn = spec.compute_fn if spec is not None else None
        if spec is None or compute_fn is None:
            raise PtaReplayUnsupported(f"compute_fn indicator {name} cannot be replayed")
        if not spec.asof_context:
            continue
        window = frame.copy(deep=False)
        window.attrs = {"bar_close_ns": close_ns, "symbol": instrument_id}
        values = _single_output(compute_fn(window, params), n_rows, name)
        columns.append((name, values, pta_lookback(spec, params) - 1))
    return columns


def _single_output(result: object, n_rows: int, name: str) -> np.ndarray:
    """Float64 per-bar values of one compute call (a malformed result is unsupported)."""
    array = np.asarray(result, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != n_rows:
        msg = (
            f"compute_fn indicator {name} returned shape {array.shape} "
            f"for {n_rows} bars (an as-of context spec must return one value per bar)"
        )
        raise PtaReplayUnsupported(msg)
    return array


class PtaPrecomputed:
    """Replayed compute_fn columns for one timeframe, fed bar by bar into ``_pta_values``.

    Installed on a generated strategy as ``strategy._pta_pre = {tf: PtaPrecomputed(...)}``
    (the runner's seam). Each bar must match the precompute's own index exactly
    (``ts_init`` and ``close``): the first bar that does not, or a bar past the end,
    fails it for good, and the generated feed then recomputes live from that bar on.
    There is no offset search: a shifted start changes the buffer windows.
    """

    def __init__(
        self, ts: list[int], close: list[float], columns: list[tuple[str, list[float]]]
    ) -> None:
        self.ts = ts
        self.close = close
        self.columns = columns
        self.n = len(ts)
        self.fed = 0
        self.failed_at: int | None = None

    def feed(self, values: dict[str, float], ts_init: int, close: float) -> bool:
        """Write this bar's values into ``values``; False means the live path owns the bar."""
        i = self.fed
        if self.failed_at is not None or i >= self.n or self.ts[i] != ts_init or self.close[i] != close:
            if self.failed_at is None:
                self.failed_at = i
            return False
        for key, column in self.columns:
            value = column[i]
            if value == value:  # a NaN never overwrites the held value (runtime NaN-hold)
                values[key] = value
        self.fed = i + 1
        return True
