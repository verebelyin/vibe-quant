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
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.fast_tier.errors import FastTierUnsupported

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.instruments import Instrument

logger = logging.getLogger(__name__)

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
class Signals:
    """Booleans per bar (False on bars where the strategy is not ready)."""

    ready: np.ndarray
    long_entry: np.ndarray
    short_entry: np.ndarray
    long_exit: np.ndarray
    short_exit: np.ndarray


class _Cache:
    """The parts of ``self.cache`` the generated code reads outside position handling."""

    def __init__(self, instrument: Instrument) -> None:
        self._instrument = instrument
        self.balance: Any = None

    def instrument(self, _instrument_id: object) -> Instrument:
        return self._instrument

    def account_for_venue(self, _venue: object) -> _Account:
        return _Account(self)


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
) -> tuple[Any, list[Any], _Cache]:
    """A started generated strategy, its registered NT indicators and its cache stand-in."""
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

    def _subscribe(self: Any, bar_type: Any, *_a: object, **_k: object) -> None:
        if str(bar_type).split("-", 2)[-1] != "1-MINUTE-LAST-EXTERNAL":
            raise FastTierUnsupported(f"strategy subscribes {bar_type}: only 1m bars are replayed")

    def _register(self: Any, bar_type: Any, indicator: Any) -> None:
        _subscribe(self, bar_type)
        indicators.append(indicator)

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
        },
    )
    strategy.__class__ = sub
    strategy.on_start()
    return strategy, indicators, cache


def _pta_columns(
    strategy: Any, indicator_types: dict[str, str], bars: list[Bar]
) -> list[tuple[str, np.ndarray]]:
    """(``_pta_values`` key, held value per bar) for every compute_fn indicator output."""
    pta_tfs = list(getattr(strategy, "_pta_bufs", {}) or {})
    if not pta_tfs:
        return []
    if pta_tfs != ["1m"]:
        raise FastTierUnsupported(f"compute_fn indicators on {pta_tfs}: only 1m is replayed")
    import pandas as pd

    from vibe_quant.dsl.indicators import indicator_registry
    from vibe_quant.dsl.pta_replay import replay_pta_series

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
    cap = strategy._pta_buffer_cap["1m"]
    columns: list[tuple[str, np.ndarray]] = []
    for name, params in strategy._pta_params.items():
        spec = indicator_registry.get(indicator_types[name])
        if spec is None or spec.compute_fn is None or spec.needs_context:
            raise FastTierUnsupported(f"compute_fn indicator {name} cannot be replayed")
        key = ("replay", spec.name, tuple(sorted((k, repr(v)) for k, v in params.items())), cap, id(bars))
        out = _memo_get(key)
        if out is None:
            out = replay_pta_series(spec, frame, params, cap)
            for arr in out.values():
                arr.flags.writeable = False  # shared across genomes
            _memo_put(key, bars, out, sum(arr.nbytes for arr in out.values()))
        names = spec.output_names
        if len(names) == 1:
            columns.append((name, out[names[0]]))
        else:
            from vibe_quant.dsl.compiler import StrategyCompiler

            primary = StrategyCompiler._effective_primary(spec)
            if primary in out:
                columns.append((name, out[primary]))
            columns.extend((f"{name}_{key}", out[key]) for key in names)
    return columns


def compute_signals(
    strategy: Any, indicators: list[Any], bars: list[Bar], indicator_types: dict[str, str]
) -> Signals:
    """Run the generated indicator/condition code over ``bars`` (see the module doc).

    ``indicator_types`` maps each DSL indicator name to its type (compute_fn replay).
    """
    n = len(bars)
    ready = np.zeros(n, dtype=bool)
    le = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    pta = [(key, col.tolist()) for key, col in _pta_columns(strategy, indicator_types, bars)]
    pta_values: dict[str, float] = getattr(strategy, "_pta_values", {})
    check_le = getattr(strategy, "_check_long_entry", None)
    check_se = getattr(strategy, "_check_short_entry", None)
    check_lx = getattr(strategy, "_check_long_exit", None)
    check_sx = getattr(strategy, "_check_short_exit", None)
    is_ready = strategy._indicators_ready
    update_prev = strategy._update_prev_values
    for i, bar in enumerate(bars):
        for indicator in indicators:
            indicator.handle_bar(bar)
        for key, col in pta:
            value = col[i]
            if value == value:  # not NaN: the runtime holds the last valid value
                pta_values[key] = value
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
    return Signals(ready, le, se, lx, sx)
