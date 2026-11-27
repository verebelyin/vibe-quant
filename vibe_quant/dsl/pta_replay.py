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

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry, pta_lookback
from vibe_quant.dsl.pta_buffer import PtaBuffer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

# Cap-0 origin replays shared by every run of the process (vibe-quant-mjw5l): a sweep, a
# discovery population or a later run with another end replays a 1m origin history of
# millions of bars once per (bar files, indicator, params) instead of once per run. Key:
# (the caller's memo key -- catalog path, bar type, bar-file identity --, spec name, params);
# value: (the registry spec replayed -- a same-name replacement is a miss --, ts_init of the
# replayed history, read-only output arrays). A request whose history
# is a prefix of the stored one (``ts_init`` equal row for row: the end-slice proof) gets
# the stored arrays cut to its length, the value at bar T depending only on bars <= T (the
# causality the registry-wide replay parity test guards, and install_origin's seed and the
# precompute already rely on). A longer request replays and replaces the entry. Bound:
# _ORIGIN_SERIES_MAX_BYTES of arrays (outputs + ts_init) per process, least recently used
# leave, one entry over the cap is not stored. _ORIGIN_SERIES_LOCK guards every lookup,
# promotion, eviction and insertion (threads of one process); replays run outside it.
_ORIGIN_SERIES_MAX_BYTES = 512 * 1024 * 1024
_ORIGIN_SERIES: OrderedDict[
    tuple[object, ...], tuple[IndicatorSpec, np.ndarray, dict[str, np.ndarray]]
] = OrderedDict()
_ORIGIN_SERIES_LOCK = threading.Lock()


class PtaReplayUnsupported(ValueError):
    """A compute_fn indicator that cannot be replayed (missing, no compute_fn, needs context)."""


class PtaOriginError(ValueError):
    """The catalog-origin history does not end with the run's bars (the run must fail)."""


def pta_window_starts(n_bars: int, cap: int) -> np.ndarray:
    """First history row of the buffer window the runtime sees at each bar.

    Drives a real :class:`PtaBuffer` with one append per bar and reads its
    length, so the trim schedule stays defined only by ``PtaBuffer.append``.
    ``cap == 0`` never trims, so every start is 0 (no per-bar loop: a catalog-origin
    replay runs over millions of bars).
    """
    if not cap:
        return np.zeros(n_bars, dtype=np.int64)
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
    if not n_bars:
        return output
    starts = pta_window_starts(n_bars, cap)
    lookback = pta_lookback(spec, params)
    # Runs of bars that share a window start: one compute call each.
    breaks = (np.flatnonzero(np.diff(starts)) + 1).tolist()
    for i, stop in zip([0, *breaks], [*breaks, n_bars], strict=True):
        start = int(starts[i])
        if start == 0 and stop == n_bars:  # one window over every row (cap 0): no copy
            window = frame
        else:
            window = frame.iloc[start:stop].reset_index(drop=True)
        segment = _segment_arrays(compute_fn(window, params), names, len(window))
        first = max(i, start + lookback - 1)  # the runtime's lookback gate
        if first >= stop:
            continue
        for name in names:
            held[name] = _hold(
                segment[name][first - start : stop - start], held[name], output[name][first:stop]
            )

    return output


def _hold(values: np.ndarray, held: float, out: np.ndarray) -> float:
    """Write the runtime's NaN-hold of ``values`` into ``out``; return the value held after.

    Each row gets its own value when not NaN, else the last non-NaN value before it in
    ``values``, else ``held``. Values are copied, never recomputed (bit-exact).
    """
    rows = np.arange(len(values))
    last = np.where(np.isnan(values), -1, rows)
    np.maximum.accumulate(last, out=last)
    out[:] = np.where(last >= 0, values[np.maximum(last, 0)], held)
    return float(out[-1])


def _segment_arrays(result: object, names: tuple[str, ...], n_rows: int) -> dict[str, np.ndarray]:
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


def origin_offset(hist: Mapping[str, np.ndarray], run: Mapping[str, np.ndarray]) -> int:
    """Row of ``hist`` where the run's first bar sits: ``hist[k:]`` IS the run.

    ``hist`` is the bar type's catalog query from its first stored bar to the run's
    end, ``run`` the run's own query; both carry ``ts_init`` (int64) and ``close``.
    Exact equality on both columns, no tolerance: a backfill, a gap or a different
    end makes the run's history ambiguous, so the caller fails the run.

    Raises:
        PtaOriginError: the run is empty, or ``hist[k:]`` differs from ``run``.
    """
    hist_ts = np.asarray(hist["ts_init"], dtype=np.int64)
    run_ts = np.asarray(run["ts_init"], dtype=np.int64)
    if run_ts.shape[0] == 0:
        msg = "catalog-origin history: the run has no bars"
        raise PtaOriginError(msg)
    k = int(np.searchsorted(hist_ts, run_ts[0]))
    tail_ts = hist_ts[k:]
    tail_close = np.asarray(hist["close"], dtype=np.float64)[k:]
    run_close = np.asarray(run["close"], dtype=np.float64)
    if not (
        np.array_equal(tail_ts, run_ts)
        and tail_close.shape == run_close.shape
        and np.array_equal(tail_close.view(np.int64), run_close.view(np.int64))
    ):
        msg = "catalog-origin history does not end with the run's bars"
        raise PtaOriginError(msg)
    return k


def install_origin(
    strategy: Any,
    tf: str,
    hist: Mapping[str, np.ndarray],
    k: int,
    *,
    memo_key: tuple[object, ...] | None = None,
) -> None:
    """Prefill ``tf``'s compute_fn buffer with the ``k`` catalog bars before the run.

    The whole timeframe goes unbounded (cap 0), so every compute_fn indicator on it
    (an origin kind and its same-timeframe neighbours) computes from the bar type's
    first catalog bar. The pre-run bars only fill the buffer: they are never engine
    events, so they cannot trade. ``_pta_values`` gets the value each non-context
    indicator of ``tf`` holds after the prefill (the runtime's NaN-hold over the
    whole prefix), so the live per-bar path equals
    ``replay_pta_series(spec, hist, params, 0)[k:]`` from the first run bar on.
    Records ``strategy._pta_origin[tf] = (hist, k, replay)``: ``replay`` is the
    :class:`OriginReplay` of the whole ``hist`` that the seed read (row ``k - 1``,
    causal compute_fns) and the runner's precompute reuses, so one run replays the
    origin frame once per indicator. ``memo_key`` (catalog path, bar type, bar-file
    identity of the files ``hist`` was read from) shares those replays with later runs of
    the process (``_ORIGIN_SERIES``); None replays for this run only.

    Raises:
        PtaOriginError: the buffer already holds bars (install before the run).
    """
    old: PtaBuffer = strategy._pta_bufs[tf]
    if len(old):
        msg = f"catalog-origin install on {tf} after {len(old)} bars were buffered"
        raise PtaOriginError(msg)
    cols = {name: np.asarray(hist[name], dtype=np.float64) for name in _OHLCV_COLUMNS}
    close_ns = (
        bar_close_ns(np.asarray(hist["ts_init"])[:k], strategy._pta_step_ns[tf])
        if old.with_close_ns
        else None
    )
    buf = PtaBuffer(0, with_close_ns=old.with_close_ns)
    buf.extend(
        cols["open"][:k],
        cols["high"][:k],
        cols["low"][:k],
        cols["close"][:k],
        cols["volume"][:k],
        close_ns=close_ns,
    )
    strategy._pta_buffer_cap[tf] = 0
    strategy._pta_bufs[tf] = buf
    replay = OriginReplay(cols, np.asarray(hist["ts_init"], dtype=np.int64), memo_key)
    if k:
        _seed_held_values(strategy, tf, replay, k)
    origin = getattr(strategy, "_pta_origin", None)
    if origin is None:
        origin = {}
        strategy._pta_origin = origin
    origin[tf] = (hist, k, replay)


class OriginReplay:
    """The origin OHLCV frame of one (strategy, timeframe) and its memoised cap-0 replays.

    ``replay`` has :func:`replay_pta_series`'s signature (pass it to
    :func:`strategy_pta_columns`); a repeat call with the same spec, params and cap
    returns the stored arrays instead of replaying again. With a ``memo_key`` a cap-0
    replay of ``self.frame`` also goes through the process memo ``_ORIGIN_SERIES``.
    """

    def __init__(
        self,
        cols: Mapping[str, np.ndarray],
        ts_init: np.ndarray | None = None,
        memo_key: tuple[object, ...] | None = None,
    ) -> None:
        import pandas as pd

        self.frame = pd.DataFrame({name: cols[name] for name in _OHLCV_COLUMNS})
        self._done: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
        self._ts = ts_init
        self._memo_key = memo_key

    def replay(
        self, spec: IndicatorSpec, ohlcv: pd.DataFrame, params: dict[str, object], cap: int
    ) -> dict[str, np.ndarray]:
        """:func:`replay_pta_series` over ``ohlcv`` (always ``self.frame``), once per
        (spec, params, cap)."""
        params_key = tuple(sorted((k, repr(v)) for k, v in params.items()))
        key = (spec.name, params_key, cap)
        out = self._done.get(key)
        if out is None:
            if self._memo_key is None or self._ts is None or cap or ohlcv is not self.frame:
                out = replay_pta_series(spec, ohlcv, params, cap)
            else:
                shared_key = (self._memo_key, spec.name, params_key)
                out = self._shared(spec, params, shared_key, self._ts)
            self._done[key] = out
        return out

    def _shared(
        self,
        spec: IndicatorSpec,
        params: dict[str, object],
        key: tuple[object, ...],
        ts: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """The cap-0 replay of ``self.frame`` from ``_ORIGIN_SERIES``, or replayed and stored."""
        with _ORIGIN_SERIES_LOCK:
            served = _serve_stored(key, spec, ts)
        if served is not None:
            return served
        out = replay_pta_series(spec, self.frame, params, 0)
        for array in out.values():
            array.flags.writeable = False  # shared by every later run of the process
        nbytes = ts.nbytes + sum(array.nbytes for array in out.values())
        if nbytes > _ORIGIN_SERIES_MAX_BYTES:
            return out
        entry = (spec, ts.copy(), out)  # not a view of the history cache
        with _ORIGIN_SERIES_LOCK:
            if _serve_stored(key, spec, ts) is not None:  # another thread stored it meanwhile
                return out
            _ORIGIN_SERIES.pop(key, None)
            used = sum(_entry_bytes(stored) for stored in _ORIGIN_SERIES.values())
            while _ORIGIN_SERIES and used + nbytes > _ORIGIN_SERIES_MAX_BYTES:
                _, evicted = _ORIGIN_SERIES.popitem(last=False)
                used -= _entry_bytes(evicted)
            _ORIGIN_SERIES[key] = entry
        return out


def _entry_bytes(entry: tuple[IndicatorSpec, np.ndarray, dict[str, np.ndarray]]) -> int:
    return entry[1].nbytes + sum(array.nbytes for array in entry[2].values())


def _serve_stored(
    key: tuple[object, ...], spec: IndicatorSpec, ts: np.ndarray
) -> dict[str, np.ndarray] | None:
    """The stored replay cut to ``ts``'s length when it was replayed by this very ``spec``
    over a history ``ts`` is a prefix of (the end-slice proof), else None. Hold the lock."""
    stored = _ORIGIN_SERIES.get(key)
    if stored is None:
        return None
    stored_spec, stored_ts, arrays = stored
    n = len(ts)
    if stored_spec is not spec or len(stored_ts) < n or not np.array_equal(stored_ts[:n], ts):
        return None
    _ORIGIN_SERIES.move_to_end(key)
    return {name: array[:n] for name, array in arrays.items()}


def _seed_held_values(strategy: Any, tf: str, replay: OriginReplay, k: int) -> None:
    """Write the held value at prefill row ``k - 1`` of ``tf``'s non-context indicators."""
    kinds: dict[str, tuple[str, str]] = type(strategy)._PTA_KINDS
    meta = {}
    for name, (kind, ind_tf) in kinds.items():
        spec = indicator_registry.get(kind)
        if ind_tf == tf and spec is not None and not spec.needs_context:
            meta[name] = (kind, ind_tf)
    params = {name: strategy._pta_params[name] for name in meta}
    columns = strategy_pta_columns(params, meta, {tf: 0}, tf, replay.frame, replay=replay.replay)
    values: dict[str, float] = strategy._pta_values
    for key, column in columns:
        value = float(column[k - 1])
        if value == value:  # never produced (NaN) stays unset, as on the live path
            values[key] = value


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
        if (
            self.failed_at is not None
            or i >= self.n
            or self.ts[i] != ts_init
            or self.close[i] != close
        ):
            if self.failed_at is None:
                self.failed_at = i
            return False
        for key, column in self.columns:
            value = column[i]
            if value == value:  # a NaN never overwrites the held value (runtime NaN-hold)
                values[key] = value
        self.fed = i + 1
        return True
