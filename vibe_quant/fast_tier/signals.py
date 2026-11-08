"""Per-bar signal arrays from the generated strategy's own code, without an engine.

The compiled strategy class is instantiated through NT's ``StrategyFactory`` with the
config dict the screening runner builds, then re-classed to a subclass whose ``cache``
property, ``subscribe_bars`` and ``register_indicator_for_bars`` are stand-ins, so the
generated ``on_start`` creates its indicators exactly as it does in an engine. Each bar
then runs what NT runs before and inside ``on_bar``: every registered indicator's
``handle_bar``, the compute_fn buffer feed, ``_last_close``, ``_indicators_ready`` and,
on ready bars, the four condition checks and ``_update_prev_values``.

The checks are pure functions of indicator state and the previous values, and the
previous values are updated on EVERY ready bar in the engine too (the dispatch-bar early
return updates them as well), so evaluating every check on every ready bar yields the
exact booleans the strategy sees on the bars where it asks.

compute_fn (pandas) indicators are not recomputed per bar: their whole stream comes from
:func:`vibe_quant.dsl.pta_replay.replay_pta_series` (the runtime's buffer schedule,
lookback gate and NaN-hold, bit-exact for causal compute_fns) and is written into
``_pta_values`` under the keys the generated ``_update_pta_indicators`` uses. The replay is
a pure function of (indicator, params, buffer cap, bars), so its output is memoised per
process across genomes (GA children repeat their parents' params) in an LRU capped by
``VIBE_QUANT_FAST_TIER_MEMO_MAX_MB`` (default 256).

A strategy that loads several timeframes (or whose primary is not 1m) runs the MERGED path
(:func:`signals_for_runner`, :func:`_compute_signals_merged`): every timeframe's bars are events in
the engine's order (``ts_init``, ties by ``sorted(timeframes)``), each updates the indicators and
compute_fn columns of its own bar type, only primary events evaluate the checks, and
``record`` keeps the indicator readings a stop/target level makes after every event. The 1m-only
loop below it is untouched, so every fast-tier result stays bit-identical.

An indicator flagged ``asof_context`` (TAKER_IMBALANCE) skips that stream: its column is
one whole-frame call (:func:`vibe_quant.dsl.pta_replay.asof_context_columns`) and a NaN
in it is written as a value. Those columns are not memoised — one call is ~5 ms for two
weeks of 1m bars, and a cached column could outlive a preflight archive reload.
"""

from __future__ import annotations

import heapq
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.fast_tier.errors import FastTierUnsupported

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.instruments import Instrument

    from vibe_quant.screening.nt_runner import NTScreeningRunner

logger = logging.getLogger(__name__)

_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
_BARS: dict[tuple[str, str, str, str], list[Bar]] = {}
_INSTRUMENTS: dict[tuple[str, str], Instrument] = {}
# key -> (bars, value, bytes); keys carry id(bars) and the entry holds the bars list, so
# that id cannot be reused by another list while the entry lives
_MEMO: OrderedDict[tuple[object, ...], tuple[list[Bar], Any, int]] = OrderedDict()
_memo_bytes = 0
MEMO_MAX_MB_ENV = "VIBE_QUANT_FAST_TIER_MEMO_MAX_MB"


def _memo_cap_bytes() -> int:
    raw = os.environ.get(MEMO_MAX_MB_ENV) or "256"
    mb = int(raw)
    if mb < 1:
        raise ValueError(f"{MEMO_MAX_MB_ENV} must be an integer >= 1, got {raw!r}")
    return mb * 1024 * 1024


def _memo_get(key: tuple[object, ...]) -> Any:
    hit = _MEMO.get(key)
    if hit is None:
        return None
    _MEMO.move_to_end(key)
    return hit[1]


def _memo_put(key: tuple[object, ...], bars: list[Bar], value: Any, nbytes: int) -> None:
    """Store ``value``, evicting least-recently-used entries past the cap (never errors)."""
    global _memo_bytes
    cap = _memo_cap_bytes()
    if nbytes > cap:
        return
    while _MEMO and _memo_bytes + nbytes > cap:
        _, (_, _, evicted) = _MEMO.popitem(last=False)
        _memo_bytes -= evicted
    _MEMO[key] = (bars, value, nbytes)
    _memo_bytes += nbytes


def load_instrument(catalog_path: str, instrument_id: str) -> Instrument:
    """The catalog instrument (cached per process)."""
    key = (catalog_path, instrument_id)
    if key not in _INSTRUMENTS:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        found = ParquetDataCatalog(catalog_path).instruments(instrument_ids=[instrument_id])
        if not found:
            raise FastTierUnsupported(f"instrument {instrument_id} not in catalog {catalog_path}")
        _INSTRUMENTS[key] = found[0]
    return _INSTRUMENTS[key]


def load_bars(catalog_path: str, bar_type: str, start: str, end: str) -> list[Bar]:
    """Catalog bars of ``bar_type`` in [start, end) -- the query a BacktestDataConfig runs."""
    key = (catalog_path, bar_type, start, end)
    if key not in _BARS:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        _BARS[key] = list(
            ParquetDataCatalog(catalog_path).bars(bar_types=[bar_type], start=start, end=end)
        )
    return _BARS[key]


class _Silent:
    """Logger stand-in: the generated sizing code logs skipped entries."""

    def warning(self, *_a: object, **_k: object) -> None:
        return None

    info = error = debug = warning


@dataclass(frozen=True)
class EventLog:
    """What the strategy's indicators read after each data event of a merged run.

    One row per bar event of ANY loaded timeframe, in the order NT delivers them.
    ``values[name]`` is ``strategy._get_indicator_value(name)`` after that event (the
    readings a stop/target level or sizing makes), so a broker model can look up the value at
    a primary bar's own event (``primary`` row) or at any later instant (:meth:`row_at`).
    """

    ts: np.ndarray
    """``ts_init`` (ns) of each event."""
    primary: np.ndarray
    """Index into the primary bars for a primary event, -1 for another timeframe's."""
    values: dict[str, np.ndarray]

    def primary_rows(self) -> np.ndarray:
        """Event row of each primary bar (``rows[i]`` = row of primary bar ``i``)."""
        rows = np.full(int(self.primary.max()) + 1 if self.primary.size else 0, -1, dtype=np.int64)
        idx = np.flatnonzero(self.primary >= 0)
        rows[self.primary[idx]] = idx
        return rows

    def row_at(self, ts_ns: int) -> int:
        """Last event row with ``ts <= ts_ns`` (-1 before the first): the state at that instant."""
        return int(np.searchsorted(self.ts, ts_ns, side="right")) - 1


@dataclass(frozen=True)
class Signals:
    """Booleans per bar (False on bars where the strategy is not ready)."""

    ready: np.ndarray
    long_entry: np.ndarray
    short_entry: np.ndarray
    long_exit: np.ndarray
    short_exit: np.ndarray
    sizing_atr: np.ndarray | None = None
    """``sizing_mode="atr"``: the sizing ATR after each bar (NaN while it warms up)."""
    events: EventLog | None = None
    """Merged runs that asked to ``record`` indicator readings only."""


class _Cache:
    """The parts of ``self.cache`` the generated code reads outside position handling."""

    def __init__(self, instrument: Instrument) -> None:
        self._instrument = instrument
        self.balance: Any = None
        #: Every replayed position in creation order (the owner binds its own list).
        self.positions: list[Any] = []

    def instrument(self, _instrument_id: object) -> Instrument:
        return self._instrument

    def account_for_venue(self, _venue: object) -> _Account:
        return _Account(self)

    def positions_open(self, venue: object = None) -> list[Any]:
        """The open replayed positions (one venue: the replay's instrument).

        The generated sizing/margin code calls this with ``venue=``; the replay trades
        one instrument, so every replayed position is on that venue.
        """
        return [p for p in self.positions if p.is_open]


class _Account:
    def __init__(self, cache: _Cache) -> None:
        self._cache = cache

    def balance_total(self, _currency: object) -> Any:
        return self._cache.balance


def build_strategy(
    module_path: str,
    strategy_cls_name: str,
    config_cls_name: str,
    config: dict[str, Any],
    instrument: Instrument,
    *,
    any_timeframe: bool = False,
) -> tuple[Any, list[Any], _Cache]:
    """A started generated strategy, its registered NT indicators and its cache stand-in.

    ``any_timeframe`` accepts bar subscriptions of every timeframe (default: 1m only, the
    legacy fast tier). The (bar type string, indicator) pairs the strategy registered are
    kept in ``strategy._fast_routes`` for :func:`compute_signals`' merged path.
    """
    from nautilus_trader.config import ImportableStrategyConfig
    from nautilus_trader.trading.config import StrategyFactory

    strategy = StrategyFactory.create(
        ImportableStrategyConfig(
            strategy_path=f"{module_path}:{strategy_cls_name}",
            config_path=f"{module_path}:{config_cls_name}",
            config=config,
        )
    )
    cache = _Cache(instrument)
    indicators: list[Any] = []
    silent = _Silent()

    routes: list[tuple[str, Any]] = []

    def _subscribe(self: Any, bar_type: Any, *_a: object, **_k: object) -> None:
        if not any_timeframe and str(bar_type).split("-", 2)[-1] != "1-MINUTE-LAST-EXTERNAL":
            raise FastTierUnsupported(f"strategy subscribes {bar_type}: only 1m bars are replayed")

    def _register(self: Any, bar_type: Any, indicator: Any) -> None:
        _subscribe(self, bar_type)
        indicators.append(indicator)
        routes.append((str(bar_type), indicator))

    def _unsupported(self: Any, *_a: object, **_k: object) -> None:
        raise FastTierUnsupported("strategy subscribes trade ticks: the outbox is not replayed")

    sub = type(
        "FastTier" + type(strategy).__name__,
        (type(strategy),),
        {
            "cache": property(lambda self: cache),
            "log": property(lambda self: silent),
            "subscribe_bars": _subscribe,
            "subscribe_trade_ticks": _unsupported,
            "register_indicator_for_bars": _register,
            # no engine: never adopts a position on start
            "_sync_position_state": lambda self: None,
            # the indicators ran over every bar first: the replay sets the dispatch bar's
            # recorded sizing ATR (``Signals.sizing_atr``) before each sizing call
            "_sizing_atr_value": lambda self: self._fast_sizing_atr,
        },
    )
    strategy._fast_sizing_atr = None
    strategy._fast_routes = routes
    strategy.__class__ = sub
    strategy.on_start()
    return strategy, indicators, cache


def _pta_columns(
    strategy: Any, indicator_types: dict[str, str], bars: list[Bar]
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray, int]]]:
    """(held columns, as-of columns with their first written row) for compute_fn indicators.

    ``strategy._pta_params`` splits by ``IndicatorSpec.asof_context``: a non-asof
    indicator replays the runtime's rolling-buffer stream (the last value held over
    a NaN); an as-of indicator is one whole-frame call, and its column holds a NaN
    as a value (``compiler.py`` never holds a context NaN either).
    """
    pta_tfs = list(getattr(strategy, "_pta_bufs", {}) or {})
    if not pta_tfs:
        return [], []
    if pta_tfs != ["1m"]:
        raise FastTierUnsupported(f"compute_fn indicators on {pta_tfs}: only 1m is replayed")
    meta = {name: (indicator_types[name], "1m") for name in strategy._pta_params}
    return _tf_pta_columns(strategy, meta, "1m", bars)


def _pta_meta(
    strategy: Any, indicator_types: dict[str, str], default_tf: str
) -> dict[str, tuple[str, str]]:
    """indicator name -> (registry type, timeframe) of every compute_fn indicator.

    The generated ``_PTA_META`` when the strategy has one; a needs_context strategy emits
    none (it is one timeframe), so its indicators are on ``default_tf``.
    """
    meta = getattr(type(strategy), "_PTA_META", None)
    if meta:
        return {str(k): (str(v[0]), str(v[1])) for k, v in meta.items()}
    return {name: (indicator_types[name], default_tf) for name in strategy._pta_params}


def _tf_pta_columns(
    strategy: Any, meta: dict[str, tuple[str, str]], tf: str, bars: list[Bar]
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray, int]]]:
    """:func:`_pta_columns` for the compute_fn indicators of ONE timeframe ``tf`` (its ``bars``)."""
    import pandas as pd

    from vibe_quant.dsl.indicators import indicator_registry
    from vibe_quant.dsl.pta_replay import (
        PtaReplayUnsupported,
        asof_context_columns,
        replay_pta_series,
        strategy_pta_columns,
    )

    frame_key = ("frame", id(bars))
    frame = _memo_get(frame_key)
    if frame is None:
        frame = pd.DataFrame(
            {
                "open": [float(b.open) for b in bars],
                "high": [float(b.high) for b in bars],
                "low": [float(b.low) for b in bars],
                "close": [float(b.close) for b in bars],
                "volume": [float(b.volume) for b in bars],
            }
        )
        _memo_put(frame_key, bars, frame, int(frame.memory_usage(index=False).sum()))

    def memoised_replay(
        spec: Any, ohlcv: pd.DataFrame, params: dict[str, object], cap: int
    ) -> dict[str, np.ndarray]:
        key = ("replay", spec.name, tuple(sorted((k, repr(v)) for k, v in params.items())), cap, id(bars))
        out = _memo_get(key)
        if out is None:
            out = replay_pta_series(spec, ohlcv, params, cap)
            for arr in out.values():
                arr.flags.writeable = False  # shared across genomes
            _memo_put(key, bars, out, sum(arr.nbytes for arr in out.values()))
        return out  # type: ignore[no-any-return]

    asof_names = {
        name
        for name in strategy._pta_params
        if (spec := indicator_registry.get(meta[name][0])) is not None and spec.asof_context
    }
    plain_params = {n: p for n, p in strategy._pta_params.items() if n not in asof_names}
    asof_params = {n: p for n, p in strategy._pta_params.items() if n in asof_names}

    plain: list[tuple[str, np.ndarray]] = []
    asof: list[tuple[str, np.ndarray, int]] = []
    try:
        if plain_params:
            plain = strategy_pta_columns(
                plain_params,
                meta,
                strategy._pta_buffer_cap,
                tf,
                frame,
                replay=memoised_replay,
            )
        if asof_params:
            asof = asof_context_columns(
                asof_params,
                meta,
                tf,
                frame,
                _bar_close_ns(strategy, bars, tf),
                strategy.config.instrument_id,
            )
    except PtaReplayUnsupported as exc:
        raise FastTierUnsupported(str(exc)) from exc
    return plain, asof


def _bar_close_ns(strategy: Any, bars: list[Bar], tf: str = "1m") -> np.ndarray:
    """``bar_close_ns`` of the run's bars, memoised per ``id(bars)`` (like the frame)."""
    from vibe_quant.dsl.pta_replay import bar_close_ns

    step = int(strategy._pta_step_ns[tf])
    key = ("close_ns", id(bars), step)
    close = _memo_get(key)
    if close is None:
        ts_init = np.fromiter((int(b.ts_init) for b in bars), dtype=np.int64, count=len(bars))
        close = bar_close_ns(ts_init, step)
        _memo_put(key, bars, close, close.nbytes)
    return close  # type: ignore[no-any-return]


def compute_signals(
    strategy: Any,
    indicators: list[Any],
    bars: list[Bar],
    indicator_types: dict[str, str],
    *,
    extra_bars: Mapping[str, list[Bar]] | None = None,
    timeframe: str = "1m",
    record: Sequence[str] = (),
    coarse_first: bool = False,
) -> Signals:
    """Run the generated indicator/condition code over ``bars`` (see the module doc).

    ``indicator_types`` maps each DSL indicator name to its type (compute_fn replay).

    ``bars`` are the primary timeframe's (``timeframe``); ``extra_bars`` maps every other
    loaded timeframe to its bars. With either ``extra_bars`` or ``record`` the run is the
    MERGED one (:func:`_compute_signals_merged`: ``build_strategy(..., any_timeframe=True)``
    is required); otherwise it is the legacy 1m-only loop, unchanged. ``coarse_first`` merges a
    longer timeframe's bar BEFORE a shorter one's at the same ``ts_init`` (TradingView's rule)
    instead of the engine's data order.
    """
    if extra_bars or record or timeframe != "1m":
        return _compute_signals_merged(
            strategy, bars, indicator_types, extra_bars or {}, timeframe, tuple(record), coarse_first
        )
    n = len(bars)
    ready = np.zeros(n, dtype=bool)
    le = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    plain, asof = _pta_columns(strategy, indicator_types, bars)
    pta = [(key, col.tolist()) for key, col in plain]
    asof_pta = [(key, col.tolist(), first) for key, col, first in asof]
    pta_values: dict[str, float] = getattr(strategy, "_pta_values", {})
    check_le = getattr(strategy, "_check_long_entry", None)
    check_se = getattr(strategy, "_check_short_entry", None)
    check_lx = getattr(strategy, "_check_long_exit", None)
    check_sx = getattr(strategy, "_check_short_exit", None)
    is_ready = strategy._indicators_ready
    update_prev = strategy._update_prev_values
    atr: Any = getattr(strategy, "_sizing_atr", None)
    sizing_atr = None if atr is None else np.full(n, np.nan)
    for i, bar in enumerate(bars):
        for indicator in indicators:
            indicator.handle_bar(bar)
        if sizing_atr is not None and atr.initialized:
            sizing_atr[i] = float(atr.value)
        for key, col in pta:
            value = col[i]
            if value == value:  # not NaN: the runtime holds the last valid value
                pta_values[key] = value
        for key, col, first in asof_pta:
            if i >= first:  # a NaN is a real answer for aux data, never held
                pta_values[key] = col[i]
        strategy._last_close = float(bar.close)
        if not is_ready():
            continue
        ready[i] = True
        if check_le is not None:
            le[i] = check_le(bar)
        if check_se is not None:
            se[i] = check_se(bar)
        if check_lx is not None:
            lx[i] = check_lx(bar)
        if check_sx is not None:
            sx[i] = check_sx(bar)
        update_prev(bar)
    return Signals(ready, le, se, lx, sx, sizing_atr)


def _compute_signals_merged(
    strategy: Any,
    bars: list[Bar],
    indicator_types: dict[str, str],
    extra_bars: Mapping[str, list[Bar]],
    timeframe: str,
    record: tuple[str, ...],
    coarse_first: bool = False,
) -> Signals:
    """The signal booleans per PRIMARY bar of a strategy that loads several timeframes.

    Every bar of every timeframe is one event, in the order NT hands them to the strategy:
    by ``ts_init``, ties by the order the runners add their data (``sorted(timeframes)``,
    measured by ``tests/unit/test_mtf_midnight_order.py``: a ``1d`` bar and the day's final
    intraday bar share ``ts_init``, so ``1m``/``5m``/``1h``/``4h`` primaries see the new daily
    value on that final bar and ``15m`` one bar late). An event of timeframe ``tf`` updates the
    indicators registered on that bar type and the compute_fn columns of ``tf``; only a primary
    event sets ``_last_close`` and evaluates the checks, exactly as ``on_bar`` returns early for
    every other bar type. ``coarse_first`` ranks a longer timeframe's bar first at a tie (TradingView:
    a higher timeframe turns over on the period's final bar for every primary). ``record`` names
    indicator readings kept after every event (:class:`EventLog`).
    """
    tf_bars: dict[str, list[Bar]] = {timeframe: bars, **extra_bars}
    ranked = sorted(tf_bars, key=lambda tf: -_TF_MINUTES.get(tf, 0)) if coarse_first else sorted(tf_bars)
    order = {tf: k for k, tf in enumerate(ranked)}
    pta_tfs = list(getattr(strategy, "_pta_bufs", {}) or {})
    missing = [tf for tf in pta_tfs if tf not in tf_bars]
    if missing:
        raise FastTierUnsupported(f"compute_fn indicators on {missing}: those bars are not loaded")
    if getattr(strategy, "_pta_step_ns", None) and pta_tfs != [timeframe]:
        raise FastTierUnsupported("needs_context indicators replay one timeframe only")
    meta = _pta_meta(strategy, indicator_types, timeframe) if pta_tfs else {}
    pta: dict[str, tuple[list[tuple[str, list[float]]], list[tuple[str, list[float], int]]]] = {}
    for tf in pta_tfs:
        plain, asof = _tf_pta_columns(strategy, meta, tf, tf_bars[tf])
        pta[tf] = (
            [(key, col.tolist()) for key, col in plain],
            [(key, col.tolist(), first) for key, col, first in asof],
        )

    by_bar_type: dict[str, list[Any]] = {}
    for bar_type, indicator in getattr(strategy, "_fast_routes", ()):
        by_bar_type.setdefault(bar_type, []).append(indicator)
    primary_type = str(bars[0].bar_type) if bars else ""

    n = len(bars)
    ready = np.zeros(n, dtype=bool)
    le = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    pta_values: dict[str, float] = getattr(strategy, "_pta_values", {})
    check_le = getattr(strategy, "_check_long_entry", None)
    check_se = getattr(strategy, "_check_short_entry", None)
    check_lx = getattr(strategy, "_check_long_exit", None)
    check_sx = getattr(strategy, "_check_short_exit", None)
    atr: Any = getattr(strategy, "_sizing_atr", None)
    sizing_atr = None if atr is None else np.full(n, np.nan)

    def stream(tf: str) -> Any:
        k = order[tf]
        return ((int(b.ts_init), k, tf, i, b) for i, b in enumerate(tf_bars[tf]))

    ev_ts: list[int] = []
    ev_primary: list[int] = []
    ev_values: dict[str, list[float]] = {name: [] for name in record}
    for ts, _k, tf, i, bar in heapq.merge(*(stream(tf) for tf in tf_bars), key=lambda e: (e[0], e[1])):
        for indicator in by_bar_type.get(str(bar.bar_type), ()):
            indicator.handle_bar(bar)
        if tf in pta:
            plain_cols, asof_cols = pta[tf]
            for key, col in plain_cols:
                value = col[i]
                if value == value:  # not NaN: the runtime holds the last valid value
                    pta_values[key] = value
            for key, col, first in asof_cols:
                if i >= first:  # a NaN is a real answer for aux data, never held
                    pta_values[key] = col[i]
        is_primary = tf == timeframe and str(bar.bar_type) == primary_type
        if is_primary:
            if sizing_atr is not None and atr.initialized:
                sizing_atr[i] = float(atr.value)
            strategy._last_close = float(bar.close)
            if strategy._indicators_ready():
                ready[i] = True
                if check_le is not None:
                    le[i] = check_le(bar)
                if check_se is not None:
                    se[i] = check_se(bar)
                if check_lx is not None:
                    lx[i] = check_lx(bar)
                if check_sx is not None:
                    sx[i] = check_sx(bar)
                strategy._update_prev_values(bar)
        if record:
            ev_ts.append(ts)
            ev_primary.append(i if is_primary else -1)
            for name in record:
                ev_values[name].append(float(strategy._get_indicator_value(name)))
    events = None
    if record:
        events = EventLog(
            np.asarray(ev_ts, dtype=np.int64),
            np.asarray(ev_primary, dtype=np.int64),
            {name: np.asarray(vals, dtype=np.float64) for name, vals in ev_values.items()},
        )
    return Signals(ready, le, se, lx, sx, sizing_atr, events)


@dataclass(frozen=True)
class RunSignals:
    """A runner's strategy replayed over its catalog window (see :func:`signals_for_runner`)."""

    strategy: Any
    cache: _Cache
    instrument: Instrument
    timeframe: str
    """The strategy's primary timeframe; ``signals`` are aligned with ``bars[timeframe]``."""
    bars: dict[str, list[Bar]]
    signals: Signals


def level_indicator_names(strategy: Any) -> tuple[str, ...]:
    """The indicator readings a stop, target or signal level makes: every ``*_indicator`` config value."""
    config = strategy.config
    names = {
        str(getattr(config, field))
        for field in getattr(type(config), "__struct_fields__", ())
        if field.endswith("_indicator") and isinstance(getattr(config, field), str)
    }
    return tuple(sorted(names))


def signals_for_runner(
    runner: NTScreeningRunner, symbol: str, *, record_levels: bool = False, coarse_first: bool = False
) -> RunSignals:
    """Replay ``runner``'s compiled strategy over its catalog window on ``symbol``.

    Every loaded timeframe's bars come from the same catalog query the engine's data
    configs run; the strategy is the generated class built with the config the screening
    runner would give it (run-level sizing and overrides included), so the checks, sizing and
    level helpers are the engine's own code. ``record_levels`` keeps the readings of every
    ``*_indicator`` config field per event (:class:`EventLog`) for a broker model;
    ``coarse_first`` is :func:`compute_signals`' TradingView merge order.
    """
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION
    from vibe_quant.errors import DataUnavailableError
    from vibe_quant.screening.nt_runner import resolve_strategy_params

    runner._ensure_compiled()
    catalog = str(runner._resolved_catalog_path.resolve())
    instrument_id = f"{symbol}-PERP.BINANCE"
    instrument = load_instrument(catalog, instrument_id)
    timeframe = runner._timeframe
    bars: dict[str, list[Bar]] = {}
    for tf in sorted(runner._all_timeframes):
        if tf not in INTERVAL_TO_AGGREGATION:
            raise FastTierUnsupported(f"timeframe {tf} has no catalog bars")
        step, agg = INTERVAL_TO_AGGREGATION[tf]
        bar_type = f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"
        loaded = load_bars(catalog, bar_type, runner._start_date, runner._end_date)
        if not loaded:
            raise DataUnavailableError(
                f"no {tf} bars for {instrument_id} {runner._start_date}..{runner._end_date}"
            )
        bars[tf] = loaded

    params = resolve_strategy_params({}, runner._config_fields())
    params.update(runner._strategy_overrides)
    params.update(runner._sizing.strategy_params(runner._leverage))
    config = {"instrument_id": instrument_id, **params, "execution_delay_probability": 1.0}
    strategy, indicators, cache = build_strategy(
        runner._module_path,
        runner._strategy_cls_name,
        runner._config_cls_name,
        config,
        instrument,
        any_timeframe=True,
    )
    raw = runner._dsl_dict.get("indicators") or {}
    indicator_types = {str(k): str(v["type"]) for k, v in raw.items()} if isinstance(raw, dict) else {}
    record = level_indicator_names(strategy) if record_levels else ()
    signals = compute_signals(
        strategy,
        indicators,
        bars[timeframe],
        indicator_types,
        extra_bars={tf: b for tf, b in bars.items() if tf != timeframe},
        timeframe=timeframe,
        record=record,
        coarse_first=coarse_first,
    )
    return RunSignals(strategy, cache, instrument, timeframe, bars, signals)
