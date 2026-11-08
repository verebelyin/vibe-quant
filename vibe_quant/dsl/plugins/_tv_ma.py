"""TradingView-exact moving-average kernels shared by the MA-family plugins.

Underscore prefix = not auto-registered as a plugin (``plugin_loader`` skips it).
``MA``, ``HULL``, ``MA_MACD``, ``AO`` and ``PRICE`` import from here; the
PMAX/SUPERTREND plugins import ``tv_ma`` too. Pure numpy, no pandas.

Public API::

    tv_ma(kind, src, length)        -> float64 array, NaN until ready
    tv_ma_raw(kind, src, length)    -> the raw Pine series (no readiness mask) a recursion reads
    tv_ma_ready(kind, length)       -> bars needed before the first finite value
    tv_hull(mode, src, length)      -> Pine "Hull Suite" Hma/Ehma/Thma switch
    tv_hull_ready(mode, length)
    tv_atr(high, low, close, n, m)  -> Pine ATR (rma/sma of the true range), NaN until ready
    tv_atr_ready(method, n)
    rolling_min / rolling_max       -> Pine ``lowest`` / ``highest`` over a window
    str_param(params, key, default) -> lower-cased string param
    source_array(df, name)          -> open/high/low/close/hl2/hlc3/ohlc4
    shift_back(a, offset)           -> Pine ``a[offset]`` (NaN for the first offset rows)

Every kernel transcribes the Pine v5 reference definition operation by
operation (same float64 additions in the same order), so it equals a plain
loop transcription bit for bit; ``tests/unit/test_plugins/test_tv_ma_plugins.py``
holds that loop reference and compares with zero tolerance. A result is NaN
wherever Pine's is ``na`` (a window with an ``na`` input, a recursion that has
not been seeded yet). EMA-built kinds (EMA, EHMA, DEMA, TEMA, T3, ZLEMA) seed with the first
finite source value (Pine v5 ``ta.ema``); WWMA and VAR seed with 0. The public :func:`tv_ma`
masks all of them to NaN until :func:`tv_ma_ready`; that mask only hides the exposed value.
A stateful consumer that recurses on the MA (PMAX stops) reads :func:`tv_ma_raw`, whose
EMA/WWMA/VAR/ZLEMA values exist from their first Pine-defined bar. RMA keeps its SMA seed.

``FULL_HISTORY_KINDS`` are recursive (their value at bar t depends on every
earlier bar), so a plugin that offers one cannot be computed on a trimmed
buffer; see ``requires_full_history`` on the specs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

KINDS: tuple[str, ...] = (
    "sma",
    "ema",
    "wma",
    "rma",
    "hma",
    "ehma",
    "thma",
    "dema",
    "tema",
    "t3",
    "tma",
    "wwma",
    "zlema",
    "var",
    "tsf",
    "hma_tv",
)
# Kinds built on ``ta.ema`` (first-source seed) or seeded with 0 (WWMA, VAR): finite from the
# first finite input, so ``tv_ma`` masks them to the declared warm-up (``tv_ma_ready``).
_EMA_FAMILY: frozenset[str] = frozenset({"ema", "ehma", "dema", "tema", "t3", "zlema"})
_MASKED_WARMUP: frozenset[str] = _EMA_FAMILY | {"wwma", "var"}
# Kinds whose value depends on all earlier bars (exponential/Wilder recursion).
FULL_HISTORY_KINDS: frozenset[str] = frozenset(
    {"ema", "rma", "ehma", "dema", "tema", "t3", "wwma", "zlema", "var"}
)
SOURCES: tuple[str, ...] = ("open", "high", "low", "close", "hl2", "hlc3", "ohlc4")
HULL_MODES: tuple[str, ...] = ("hma", "ehma", "thma", "hma_tv")
ATR_METHODS: tuple[str, ...] = ("rma", "rma_tr", "sma")

# 3commas T3 volume factor (06_3commas_bot.pine ``ab``).
_T3_AB = 0.7
# Pine ``sum(x, 9)`` window inside the PMAX VAR moving average.
_VAR_CMO_WINDOW = 9


def str_param(params: dict[str, object], key: str, default: str) -> str:
    """Lower-cased string param (``kind``/``src``/``mode``), ``default`` when absent or not a string."""
    val = params.get(key, default)
    return val.lower() if isinstance(val, str) else default


def _half(n: int) -> int:
    """Pine ``n / 2`` used as a series length: the float is truncated."""
    return n // 2


def _round_sqrt(n: int) -> int:
    """Pine ``round(sqrt(n))``: half rounds up (``sqrt`` of an integer is never x.5)."""
    return int(math.floor(math.sqrt(n) + 0.5))


def source_array(df: pd.DataFrame, name: str) -> np.ndarray:
    """Pine price source ``name`` as float64. Pine: ``hl2 = (high+low)/2``,
    ``hlc3 = (high+low+close)/3``, ``ohlc4 = (open+high+low+close)/4``."""
    key = name.lower()
    if key in ("open", "high", "low", "close"):
        return np.asarray(df[key].to_numpy(dtype=np.float64, copy=True), dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    if key == "hl2":
        return np.asarray((high + low) / 2, dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    if key == "hlc3":
        return np.asarray((high + low + close) / 3, dtype=np.float64)
    if key == "ohlc4":
        open_ = df["open"].to_numpy(dtype=np.float64)
        return np.asarray((open_ + high + low + close) / 4, dtype=np.float64)
    msg = f"unknown source {name!r}; valid: {', '.join(SOURCES)}"
    raise ValueError(msg)


def shift_back(a: np.ndarray, offset: int) -> np.ndarray:
    """Pine ``a[offset]``: the value ``offset`` bars ago, NaN for the first ``offset`` rows."""
    if offset < 0:
        msg = f"offset must be >= 0, got {offset}"
        raise ValueError(msg)
    if offset == 0:
        return a
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if offset < a.size:
        out[offset:] = a[:-offset]
    return out


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine v5 ``ta.sma`` reference: ``sum = 0.0; for i = 0 to y-1: sum := sum + x[i] / y``."""
    out = np.full(x.size, np.nan, dtype=np.float64)
    m = x.size - n + 1
    if m <= 0:
        return out
    acc = np.zeros(m, dtype=np.float64)
    for i in range(n):  # x[i] = i bars back from the window end
        acc = acc + x[n - 1 - i : n - 1 - i + m] / n
    out[n - 1 :] = acc
    return out


def _wma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine v5 ``ta.wma`` reference: ``weight = (y - i) * y; norm += weight; sum += x[i] * weight``."""
    out = np.full(x.size, np.nan, dtype=np.float64)
    m = x.size - n + 1
    if m <= 0:
        return out
    norm = 0.0
    acc = np.zeros(m, dtype=np.float64)
    for i in range(n):
        weight = (n - i) * n
        norm = norm + weight
        acc = acc + x[n - 1 - i : n - 1 - i + m] * weight
    out[n - 1 :] = acc / norm
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    """Pine v5 ``ta.ema`` reference (first-source seed, finite from the first finite input).

    ``sum := na(sum[1]) ? src : alpha * src + (1 - alpha) * nz(sum[1])``, ``alpha = 2 / (length + 1)``
    (TradingView v5 reference, ``ta.ema``). A ``na`` input makes the output ``na`` and the next
    bar re-seeds with its source. This is the raw recursion; :func:`tv_ma` masks the bars before
    :func:`tv_ma_ready`.
    """
    alpha = 2 / (n + 1)
    beta = 1 - alpha
    out = [math.nan] * x.size
    prev = math.nan
    for t, v in enumerate(x.tolist()):
        cur = v if prev != prev else alpha * v + beta * prev
        out[t] = cur
        prev = cur
    return np.array(out, dtype=np.float64)


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine v5 ``ta.rma`` reference (SMA seed, unlike EMA).

    ``sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])``;
    ``alpha = 1 / length``. A ``na`` input makes the output ``na`` and the next bar re-seeds
    with the SMA.
    """
    alpha = 1 / n
    seed = _sma(x, n).tolist()
    beta = 1 - alpha
    out = [math.nan] * x.size
    prev = math.nan
    for t, v in enumerate(x.tolist()):
        cur = seed[t] if prev != prev else alpha * v + beta * prev
        out[t] = cur
        prev = cur
    return np.array(out, dtype=np.float64)


def _first_finite(x: np.ndarray) -> int:
    hits = np.flatnonzero(np.isfinite(x))
    return int(hits[0]) if hits.size else x.size


def _wwma(x: np.ndarray, n: int) -> np.ndarray:
    """PMAX ``Wwma_Func``: ``WWMA := wwalpha*src + (1-wwalpha)*nz(WWMA[1])``, ``wwalpha = 1/length``.

    Pine seeds ``WWMA`` with 0 so it exists from bar 0. This is the raw recursion;
    :func:`tv_ma` masks the ``n - 1`` bars after the first finite input (not ready).
    """
    alpha = 1 / n
    beta = 1 - alpha
    xs = x.tolist()
    out = [math.nan] * len(xs)
    prev = 0.0
    for t, v in enumerate(xs):
        cur = alpha * v + beta * prev
        out[t] = cur
        prev = 0.0 if cur != cur else cur
    return np.array(out, dtype=np.float64)


def _var(x: np.ndarray, n: int) -> np.ndarray:
    """PMAX ``Var_Func`` (Chande VIDYA with a fixed 9-bar CMO window).

    ``vud1 = src>src[1] ? src-src[1] : 0``, ``vdd1`` the mirror, ``vUD = sum(vud1,9)``,
    ``vCMO = nz((vUD-vDD)/(vUD+vDD))``,
    ``VAR := nz(valpha*abs(vCMO)*src) + (1-valpha*abs(vCMO))*nz(VAR[1])``, ``valpha = 2/(length+1)``.
    A comparison with ``na`` is false, so the first bar gives 0 for both. This is the raw
    recursion; :func:`tv_ma` masks the bars before :func:`tv_ma_ready`.
    """
    size = x.size
    xs = x.tolist()
    valpha = 2 / (n + 1)
    vud1 = [0.0] * size
    vdd1 = [0.0] * size
    for t in range(1, size):
        a, b = xs[t], xs[t - 1]
        vud1[t] = a - b if a > b else 0.0
        vdd1[t] = b - a if a < b else 0.0
    out = [math.nan] * size
    prev = 0.0
    for t in range(size):
        cmo = 0.0
        if t >= _VAR_CMO_WINDOW - 1:
            ud = 0.0
            dd = 0.0
            for i in range(_VAR_CMO_WINDOW):
                ud = ud + vud1[t - i]
                dd = dd + vdd1[t - i]
            den = ud + dd
            cmo = (ud - dd) / den if den != 0.0 else 0.0
        k = valpha * abs(cmo)
        term = k * xs[t]
        term = 0.0 if term != term else term
        cur = term + (1 - k) * prev
        out[t] = cur
        prev = 0.0 if cur != cur else cur
    return np.array(out, dtype=np.float64)


def _linreg(x: np.ndarray, n: int, offset: int) -> np.ndarray:
    """Pine v5 ``ta.linreg(src, length, offset)``: least squares over the last ``length`` bars,
    ``intercept + slope * (length - 1 - offset)`` with x = 0 at the oldest bar."""
    out = np.full(x.size, np.nan, dtype=np.float64)
    m = x.size - n + 1
    if m <= 0:
        return out
    sx = 0
    sxx = 0
    sy = np.zeros(m, dtype=np.float64)
    sxy = np.zeros(m, dtype=np.float64)
    for k in range(n):  # k = bars since the window start
        y = x[k : k + m]
        sx = sx + k
        sxx = sxx + k * k
        sy = sy + y
        sxy = sxy + k * y
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    out[n - 1 :] = intercept + slope * (n - 1 - offset)
    return out


def _tsf(x: np.ndarray, n: int) -> np.ndarray:
    """PMAX ``Tsf_Func``: ``lrs = linreg(src,length,0) - linreg(src,length,1)``;
    ``TSF = linreg(src,length,0) + lrs``."""
    lrc = _linreg(x, n, 0)
    lrc1 = _linreg(x, n, 1)
    lrs = lrc - lrc1
    return np.asarray(_linreg(x, n, 0) + lrs, dtype=np.float64)


def _hma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ``hma``: ``wma(2*wma(src, length/2) - wma(src, length), round(sqrt(length)))``."""
    return _wma(2 * _wma(x, _half(n)) - _wma(x, n), _round_sqrt(n))


def _hma_tv(x: np.ndarray, n: int) -> np.ndarray:
    """Pine BUILT-IN ``hma``: ``wma(2*wma(src, length/2) - wma(src, length), floor(sqrt(length)))``.

    UNVERIFIED vs TradingView output (no export available): the built-in is believed to
    FLOOR the square root, while the Hull Suite script's own ``HMA`` (``hma`` above) rounds it.
    14_ichimoku_hull_macd.pine calls the built-in ``hma`` (lines 12-13, 32-33). The two
    differ whenever ``sqrt(length)`` has a fractional part >= 0.5 (length 14: floor 3, round 4).
    """
    return _wma(2 * _wma(x, _half(n)) - _wma(x, n), math.isqrt(n))


def _ehma(x: np.ndarray, n: int) -> np.ndarray:
    """Hull Suite ``EHMA``: ``ema(2*ema(src, length/2) - ema(src, length), round(sqrt(length)))``."""
    return _ema(2 * _ema(x, _half(n)) - _ema(x, n), _round_sqrt(n))


def _thma(x: np.ndarray, n: int) -> np.ndarray:
    """Hull Suite ``THMA``: ``wma(wma(src,length/3)*3 - wma(src,length/2) - wma(src,length), length)``."""
    return _wma(_wma(x, n // 3) * 3 - _wma(x, _half(n)) - _wma(x, n), n)


def _dema(x: np.ndarray, n: int) -> np.ndarray:
    """Pine v5 ``ta.dema``: ``2 * ema1 - ema2`` with ``ema2 = ema(ema1, length)``."""
    e1 = _ema(x, n)
    return np.asarray(2 * e1 - _ema(e1, n), dtype=np.float64)


def _tema(x: np.ndarray, n: int) -> np.ndarray:
    """Pine v5 ``ta.tema``: ``3 * (ema1 - ema2) + ema3``."""
    e1 = _ema(x, n)
    e2 = _ema(e1, n)
    e3 = _ema(e2, n)
    return np.asarray(3 * (e1 - e2) + e3, dtype=np.float64)


def _t3(x: np.ndarray, n: int) -> np.ndarray:
    """3commas ``T3`` (06_3commas_bot.pine): six chained EMAs, ``ab = 0.7``,
    ``ac1*axe6 + ac2*axe5 + ac3*axe4 + ac4*axe3``."""
    ab = _T3_AB
    ac1 = -ab * ab * ab
    ac2 = 3 * ab * ab + 3 * ab * ab * ab
    ac3 = -6 * ab * ab - 3 * ab - 3 * ab * ab * ab
    ac4 = 1 + 3 * ab + ab * ab * ab + 3 * ab * ab
    axe3 = _ema(_ema(_ema(x, n), n), n)
    axe4 = _ema(axe3, n)
    axe5 = _ema(axe4, n)
    axe6 = _ema(axe5, n)
    return ac1 * axe6 + ac2 * axe5 + ac3 * axe4 + ac4 * axe3


def _tma(x: np.ndarray, n: int) -> np.ndarray:
    """PMAX ``TMA``: ``sma(sma(src, ceil(length/2)), floor(length/2) + 1)``."""
    return _sma(_sma(x, math.ceil(n / 2)), n // 2 + 1)


def _zlema(x: np.ndarray, n: int) -> np.ndarray:
    """PMAX ``Zlema_Func``: ``zxLag = length/2==round(length/2) ? length/2 : (length-1)/2``
    (= ``length // 2``), ``ema(src + (src - src[zxLag]), length)``."""
    lag = n // 2
    return _ema(x + (x - shift_back(x, lag)), n)


_KERNELS = {
    "sma": _sma,
    "ema": _ema,
    "wma": _wma,
    "rma": _rma,
    "hma": _hma,
    "hma_tv": _hma_tv,
    "ehma": _ehma,
    "thma": _thma,
    "dema": _dema,
    "tema": _tema,
    "t3": _t3,
    "tma": _tma,
    "wwma": _wwma,
    "zlema": _zlema,
    "var": _var,
    "tsf": _tsf,
}


def _check_kind(kind: str) -> str:
    key = str(kind).lower()
    if key not in _KERNELS:
        msg = f"unknown MA kind {kind!r}; valid: {', '.join(KINDS)}"
        raise ValueError(msg)
    return key


def _min_length(kind: str) -> int:
    """Smallest ``length`` whose derived inner lengths are all >= 1 (and >= 2 for a regression)."""
    if kind in ("hma", "hma_tv", "ehma", "zlema"):
        return 2
    if kind == "thma":
        return 3
    if kind == "tsf":
        return 2
    return 1


def tv_ma_ready(kind: str, length: int) -> int:
    """Bars needed before ``tv_ma`` first returns a finite value on an all-finite source
    (``index of first finite + 1``). On a source whose first finite row is ``f`` the first
    finite output is ``f + tv_ma_ready - 1``."""
    key = _check_kind(kind)
    n = int(length)
    if n < _min_length(key):
        msg = f"MA kind {key!r} needs length >= {_min_length(key)}, got {n}"
        raise ValueError(msg)
    if key in ("sma", "ema", "wma", "rma", "tsf", "wwma"):
        return n
    if key in ("hma", "ehma"):
        return n + _round_sqrt(n) - 1
    if key == "hma_tv":
        return n + math.isqrt(n) - 1
    if key == "thma":
        return 2 * n - 1
    if key == "dema":
        return 2 * n - 1
    if key == "tema":
        return 3 * n - 2
    if key == "t3":
        return 6 * n - 5
    if key == "tma":
        return math.ceil(n / 2) + (n // 2 + 1) - 1
    if key == "zlema":
        return n // 2 + n
    return max(n, _VAR_CMO_WINDOW)  # var


def tv_ma(kind: str, src: np.ndarray, length: int) -> np.ndarray:
    """Moving average ``kind`` of ``src`` over ``length`` bars, NaN until ready.

    ``kind``: one of :data:`KINDS` (case-insensitive). ``src`` is a float64 array, NaN allowed.
    Raises ``ValueError`` for an unknown kind or a ``length`` too short for the kind.
    """
    key = _check_kind(kind)
    ready = tv_ma_ready(key, length)  # validates the length
    x = np.asarray(src, dtype=np.float64)
    out = _KERNELS[key](x, int(length))
    if key in _MASKED_WARMUP:  # recursion runs from the first finite value; hide the warm-up
        out[: min(_first_finite(x) + ready - 1, out.size)] = np.nan
    return out


def tv_ma_raw(kind: str, src: np.ndarray, length: int) -> np.ndarray:
    """Moving average ``kind`` exactly as the Pine script's series holds it: no warm-up mask.

    Equals :func:`tv_ma` for every kind whose window defines readiness (SMA, WMA, RMA, TMA, TSF, the
    HMA family). For EMA-built kinds, WWMA and VAR the recursion is defined from the first Pine
    bar, and a stateful reader (the PMAX stop ratchet) must see that history; mask only the
    exposed output with :func:`tv_ma_ready`.
    """
    key = _check_kind(kind)
    tv_ma_ready(key, length)  # validates the length
    return np.asarray(
        _KERNELS[key](np.asarray(src, dtype=np.float64), int(length)), dtype=np.float64
    )


def tv_hull_ready(mode: str, length: int) -> int:
    """Bars needed before :func:`tv_hull` is finite."""
    key = _check_hull_mode(mode)
    if key == "thma":
        return tv_ma_ready("thma", _half(int(length)))
    return tv_ma_ready(key, length)


def _check_hull_mode(mode: str) -> str:
    key = str(mode).lower()
    if key not in HULL_MODES:
        msg = f"unknown Hull mode {mode!r}; valid: {', '.join(HULL_MODES)}"
        raise ValueError(msg)
    return key


def tv_hull(mode: str, src: np.ndarray, length: int) -> np.ndarray:
    """Hull Suite ``Mode(modeSwitch, src, len)`` (07_hull_suite.pine): ``Hma``/``Ehma`` use
    ``length``; ``Thma`` is ``THMA(src, len/2)``."""
    key = _check_hull_mode(mode)
    n = int(length)
    return tv_ma(key, src, _half(n) if key == "thma" else n)


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ``lowest(x, n)``: minimum of the last ``n`` bars (itself included), NaN until ``n``
    bars exist or while a window holds a NaN. Pure comparison, no arithmetic."""
    return _rolling_extreme(x, n, np.min)


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ``highest(x, n)``: maximum of the last ``n`` bars, same NaN rules as :func:`rolling_min`."""
    return _rolling_extreme(x, n, np.max)


def _rolling_extreme(x: np.ndarray, n: int, fn: Callable[..., np.ndarray]) -> np.ndarray:
    if n < 1:
        msg = f"window length must be >= 1, got {n}"
        raise ValueError(msg)
    out = np.full(x.size, np.nan, dtype=np.float64)
    if x.size >= n:
        out[n - 1 :] = fn(np.lib.stride_tricks.sliding_window_view(x, n), axis=1)
    return out


def true_range(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, *, handle_na: bool
) -> np.ndarray:
    """Pine ``tr``: ``max(high - low, abs(high - close[1]), abs(low - close[1]))``.

    ``handle_na=True`` (the true range inside ``atr``/``ta.atr``) uses ``high - low`` on the
    first bar; ``False`` (the bare ``tr`` variable) leaves the first bar NaN, as the Pine
    reference documents for ``tr(handle_na)``. UNVERIFIED vs a TradingView export; it only
    moves the first ``sma(tr, n)`` / ``rma(tr, n)`` value by one bar.
    """
    prev = shift_back(close, 1)
    tr = np.maximum(np.maximum(high - low, np.abs(high - prev)), np.abs(low - prev))
    if handle_na and tr.size:
        tr[0] = high[0] - low[0]
    return np.asarray(tr, dtype=np.float64)


def _check_atr_method(method: str) -> str:
    key = str(method).lower()
    if key not in ATR_METHODS:
        msg = f"unknown ATR method {method!r}; valid: {', '.join(ATR_METHODS)}"
        raise ValueError(msg)
    return key


def tv_atr_ready(method: str, length: int) -> int:
    """Bars needed before :func:`tv_atr` is finite: ``rma`` (``atr(n)``) ``n``; ``rma_tr``
    (``rma(tr, n)``) and ``sma`` (``sma(tr, n)``) ``n + 1`` (the bare ``tr`` is NaN on bar 0)."""
    key = _check_atr_method(method)
    ready = tv_ma_ready("rma", int(length))  # validates the length
    return ready if key == "rma" else ready + 1


def tv_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int, method: str
) -> np.ndarray:
    """Average true range, NaN until ready.

    ``method``: ``rma`` = Pine ``atr(n)`` (Wilder RMA of ``tr(handle_na=true)``, 03/05 with
    ``changeATR`` true, 06 ``ta.atr``); ``rma_tr`` = ``rma(tr, n)`` (08 ``atrvalue``);
    ``sma`` = ``sma(tr, n)`` (03/05 with ``changeATR`` false).
    """
    key = _check_atr_method(method)
    tv_atr_ready(key, length)
    n = int(length)
    tr = true_range(high, low, close, handle_na=key == "rma")
    return _sma(tr, n) if key == "sma" else _rma(tr, n)
