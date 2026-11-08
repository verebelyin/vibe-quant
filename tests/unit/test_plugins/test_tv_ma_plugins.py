"""TradingView-exact MA-family plugins: PRICE, MA, HULL, MA_MACD, AO.

Money-grade proof (R50): for every plugin x kind x timeframe, the value the REAL generated
strategy stream produces after each bar == the vector replay the fast tier / NT precompute
seam uses == an INDEPENDENT full-history reference, with zero tolerance. The reference below
is a plain-Python-loop transcription of the Pine definitions; it imports nothing from
``_tv_ma`` so a kernel bug cannot cancel against itself.

Three buffer regimes:

* the strategy's real cap: specs that offer a recursive kind set ``requires_full_history``
  (cap 0, never trimmed) -- every kind, exact from the first ready bar;
* the harness' capped buffer (cap 400, trimmed when 25% over, >= 2 trims): the finite-lookback
  kinds must be exact on every bar, which is what makes their small lookback safe;
* R51: not ready -> NaN so a condition on it is false; R53: the GA pools are unchanged.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pta_stream_harness as harness  # noqa: E402
from pta_stream_harness import BarEvent, compile_indicators, compile_source, run_stream  # noqa: E402

from vibe_quant.dsl import invoke_compute_fn  # noqa: E402
from vibe_quant.dsl.indicators import indicator_registry, pta_buffer_cap, pta_lookback  # noqa: E402
from vibe_quant.dsl.plugins import _tv_ma  # noqa: E402
from vibe_quant.dsl.prefix_memo import MEMO  # noqa: E402
from vibe_quant.dsl.pta_replay import pta_window_starts, replay_pta_series  # noqa: E402

NAN = float("nan")
TFS = ("1m", "1h")
FINITE_KINDS = ("sma", "wma", "hma", "hma_tv", "thma", "tma", "tsf")
RECURSIVE_KINDS = ("ema", "rma", "ehma", "dema", "tema", "t3", "wwma", "zlema", "var")
ALL_KINDS = FINITE_KINDS + RECURSIVE_KINDS


@pytest.fixture(autouse=True)
def _clean_memo() -> object:
    MEMO.clear()
    yield
    MEMO.clear()


# ---------------------------------------------------------------------------
# Independent reference: plain loops over Python floats (NaN = Pine ``na``)
# ---------------------------------------------------------------------------


def r_sma(x: list[float], n: int) -> list[float]:
    """Pine v5 ta.sma: ``sum = 0.0; for i = 0 to y - 1: sum := sum + x[i] / y``."""
    out = [NAN] * len(x)
    for t in range(n - 1, len(x)):
        s = 0.0
        for i in range(n):
            s = s + x[t - i] / n
        out[t] = s
    return out


def r_wma(x: list[float], n: int) -> list[float]:
    """Pine v5 ta.wma: ``weight = (y - i) * y; norm += weight; sum += x[i] * weight``."""
    out = [NAN] * len(x)
    for t in range(n - 1, len(x)):
        norm = 0.0
        s = 0.0
        for i in range(n):
            weight = (n - i) * n
            norm = norm + weight
            s = s + x[t - i] * weight
        out[t] = s / norm
    return out


def r_ema(x: list[float], n: int) -> list[float]:
    """Pine v5 ta.ema (TradingView reference): ``sum := na(sum[1]) ? src : alpha * src + (1 - alpha) * nz(sum[1])``."""
    alpha = 2 / (n + 1)
    out: list[float] = []
    prev = NAN
    for v in x:
        cur = v if math.isnan(prev) else alpha * v + (1 - alpha) * prev
        out.append(cur)
        prev = cur
    return out


def r_rma(x: list[float], n: int) -> list[float]:
    """Pine v5 ta.rma: ``sum := na(sum[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(sum[1])``."""
    alpha = 1 / n
    seed = r_sma(x, n)
    out: list[float] = []
    prev = NAN
    for t, v in enumerate(x):
        cur = seed[t] if math.isnan(prev) else alpha * v + (1 - alpha) * prev
        out.append(cur)
        prev = cur
    return out


def _mask_warmup(x: list[float], out: list[float], ready: int) -> list[float]:
    """Declared warm-up: NaN until ``ready`` bars after the source's first finite row."""
    cut = _first_finite(x) + ready - 1
    return [NAN if i < cut else v for i, v in enumerate(out)]


def _zip(f: Any, *cols: list[float]) -> list[float]:
    return [f(*vals) for vals in zip(*cols, strict=True)]


def r_hma(x: list[float], n: int) -> list[float]:
    d = _zip(lambda a, b: 2 * a - b, r_wma(x, math.floor(n / 2)), r_wma(x, n))
    return r_wma(d, math.floor(math.sqrt(n) + 0.5))


def r_hma_tv(x: list[float], n: int) -> list[float]:
    """Pine built-in ``hma`` (UNVERIFIED vs TradingView): the square root is FLOORED."""
    d = _zip(lambda a, b: 2 * a - b, r_wma(x, math.floor(n / 2)), r_wma(x, n))
    return r_wma(d, math.floor(math.sqrt(n)))


def r_ehma(x: list[float], n: int) -> list[float]:
    d = _zip(lambda a, b: 2 * a - b, r_ema(x, math.floor(n / 2)), r_ema(x, n))
    return r_ema(d, math.floor(math.sqrt(n) + 0.5))


def r_thma(x: list[float], n: int) -> list[float]:
    d = _zip(
        lambda a, b, c: a * 3 - b - c,
        r_wma(x, math.floor(n / 3)),
        r_wma(x, math.floor(n / 2)),
        r_wma(x, n),
    )
    return r_wma(d, n)


def r_dema(x: list[float], n: int) -> list[float]:
    e1 = r_ema(x, n)
    return _zip(lambda a, b: 2 * a - b, e1, r_ema(e1, n))


def r_tema(x: list[float], n: int) -> list[float]:
    e1 = r_ema(x, n)
    e2 = r_ema(e1, n)
    e3 = r_ema(e2, n)
    return _zip(lambda a, b, c: 3 * (a - b) + c, e1, e2, e3)


def r_t3(x: list[float], n: int) -> list[float]:
    e = [r_ema(x, n)]
    for _ in range(5):
        e.append(r_ema(e[-1], n))
    ab = 0.7
    ac1 = -ab * ab * ab
    ac2 = 3 * ab * ab + 3 * ab * ab * ab
    ac3 = -6 * ab * ab - 3 * ab - 3 * ab * ab * ab
    ac4 = 1 + 3 * ab + ab * ab * ab + 3 * ab * ab
    return _zip(
        lambda a3, a4, a5, a6: ac1 * a6 + ac2 * a5 + ac3 * a4 + ac4 * a3, e[2], e[3], e[4], e[5]
    )


def r_tma(x: list[float], n: int) -> list[float]:
    return r_sma(r_sma(x, math.ceil(n / 2)), math.floor(n / 2) + 1)


def _first_finite(x: list[float]) -> int:
    return next((i for i, v in enumerate(x) if not math.isnan(v)), len(x))


def r_wwma_raw(x: list[float], n: int) -> list[float]:
    """PMAX ``Wwma_Func`` as Pine computes it: seeded with 0, defined from bar 0, no warm-up mask."""
    alpha = 1 / n
    out: list[float] = []
    prev = 0.0
    for v in x:
        cur = alpha * v + (1 - alpha) * prev
        out.append(cur)
        prev = 0.0 if math.isnan(cur) else cur
    return out


def r_wwma(x: list[float], n: int) -> list[float]:
    return [NAN if i < _first_finite(x) + n - 1 else v for i, v in enumerate(r_wwma_raw(x, n))]


def r_var_raw(x: list[float], n: int) -> list[float]:
    """PMAX ``Var_Func`` as Pine computes it: defined from bar 0, no warm-up mask."""
    valpha = 2 / (n + 1)
    size = len(x)
    vud1 = [0.0] * size
    vdd1 = [0.0] * size
    for t in range(1, size):
        vud1[t] = x[t] - x[t - 1] if x[t] > x[t - 1] else 0.0
        vdd1[t] = x[t - 1] - x[t] if x[t] < x[t - 1] else 0.0
    out: list[float] = []
    prev = 0.0
    for t in range(size):
        cmo = 0.0
        if t >= 8:
            ud = 0.0
            dd = 0.0
            for i in range(9):
                ud = ud + vud1[t - i]
                dd = dd + vdd1[t - i]
            cmo = (ud - dd) / (ud + dd) if ud + dd != 0 else 0.0
        k = valpha * abs(cmo)
        term = k * x[t]
        term = 0.0 if math.isnan(term) else term
        cur = term + (1 - k) * prev
        out.append(cur)
        prev = 0.0 if math.isnan(cur) else cur
    return out


def r_var(x: list[float], n: int) -> list[float]:
    ready = max(n, 9)
    return [NAN if i < _first_finite(x) + ready - 1 else v for i, v in enumerate(r_var_raw(x, n))]


def r_zlema(x: list[float], n: int) -> list[float]:
    lag = n // 2 if n / 2 == round(n / 2) else (n - 1) // 2
    zx = [x[t] + (x[t] - x[t - lag]) if t >= lag else NAN for t in range(len(x))]
    return r_ema(zx, n)


def _r_linreg(x: list[float], n: int, off: int) -> list[float]:
    """Pine ta.linreg: least squares, x = 0 at the window's oldest bar, ``intercept + slope * (n-1-off)``."""
    out = [NAN] * len(x)
    for t in range(n - 1, len(x)):
        sx = sy = sxy = sxx = 0.0
        for k in range(n):
            y = x[t - n + 1 + k]
            sx += k
            sy += y
            sxy += k * y
            sxx += k * k
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n
        out[t] = intercept + slope * (n - 1 - off)
    return out


def r_tsf(x: list[float], n: int) -> list[float]:
    lrc = _r_linreg(x, n, 0)
    lrc1 = _r_linreg(x, n, 1)
    return _zip(lambda a, b: a + (a - b), lrc, lrc1)


def _masked(raw: Any, ready: Any) -> Any:
    return lambda x, n: _mask_warmup(x, raw(x, n), ready(n))


def _round_sqrt(n: int) -> int:
    return math.floor(math.sqrt(n) + 0.5)


R_KINDS = {
    "sma": r_sma,
    "ema": _masked(r_ema, lambda n: n),
    "wma": r_wma,
    "rma": r_rma,
    "hma": r_hma,
    "hma_tv": r_hma_tv,
    "ehma": _masked(r_ehma, lambda n: n + _round_sqrt(n) - 1),
    "thma": r_thma,
    "dema": _masked(r_dema, lambda n: 2 * n - 1),
    "tema": _masked(r_tema, lambda n: 3 * n - 2),
    "t3": _masked(r_t3, lambda n: 6 * n - 5),
    "tma": r_tma,
    "wwma": r_wwma,
    "zlema": _masked(r_zlema, lambda n: n // 2 + n),
    "var": r_var,
    "tsf": r_tsf,
}


# The series Pine holds before the public warm-up mask: EMA-built kinds, WWMA and VAR exist from
# their first defined bar. A recursion reading an MA (MA_MACD signal, PMAX stops) reads these.
R_RAW_KINDS = {
    **R_KINDS,
    "ema": r_ema,
    "ehma": r_ehma,
    "dema": r_dema,
    "tema": r_tema,
    "t3": r_t3,
    "zlema": r_zlema,
    "wwma": r_wwma_raw,
    "var": r_var_raw,
}


def r_hull(mode: str, x: list[float], n: int) -> list[float]:
    """07_hull_suite.pine ``Mode``: Hma(len), Ehma(len), Thma(len/2)."""
    if mode == "thma":
        return r_thma(x, math.floor(n / 2))
    return R_KINDS[mode](x, n)


def r_source(bars: list[BarEvent], name: str) -> list[float]:
    out: list[float] = []
    for _tf, o, h, lo, c, _v in bars:
        out.append(
            {
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "hl2": (h + lo) / 2,
                "hlc3": (h + lo + c) / 3,
                "ohlc4": (o + h + lo + c) / 4,
            }[name]
        )
    return out


def r_lag(x: list[float], k: int) -> list[float]:
    return [x[t - k] if t >= k else NAN for t in range(len(x))]


def reference(type_: str, params: dict[str, Any], bars: list[BarEvent]) -> dict[str, list[float]]:
    """Pine-definition reference for one plugin, one list per output."""
    spec = indicator_registry.get(type_)
    assert spec is not None
    p = {**spec.default_params, **params}
    off = int(p["offset"])
    if type_ == "PRICE":
        return {"value": r_lag(r_source(bars, p["src"]), off)}
    if type_ == "MA":
        return {"value": r_lag(R_KINDS[p["kind"]](r_source(bars, p["src"]), p["period"]), off)}
    if type_ == "HULL":
        return {"value": r_lag(r_hull(p["mode"], r_source(bars, p["src"]), p["period"]), off)}
    if type_ == "AO":
        hl2 = r_source(bars, "hl2")
        d = _zip(lambda a, b: a - b, r_sma(hl2, p["fast_period"]), r_sma(hl2, p["slow_period"]))
        return {"value": r_lag(d, off)}
    assert type_ == "MA_MACD"
    ma = R_KINDS[p["kind"]]
    raw = R_RAW_KINDS[p["kind"]]
    x = r_source(bars, p["src"])
    macd = _zip(lambda a, b: a - b, ma(x, p["fast_period"]), ma(x, p["slow_period"]))
    # Pine: ``signal = ma(macd, n)`` recurses on the macd line from its first defined bar (the
    # raw MAs), not from the bar where the exposed macd is ready; only the output is masked.
    macd_raw = _zip(lambda a, b: a - b, raw(x, p["fast_period"]), raw(x, p["slow_period"]))
    exposed = ma(macd, p["signal_period"])  # NaN until macd ready + signal ready - 1
    signal = [
        v if not math.isnan(e) else NAN
        for v, e in zip(raw(macd_raw, p["signal_period"]), exposed, strict=True)
    ]
    hist = _zip(lambda a, b: a - b, macd, signal)
    return {
        "macd": r_lag(macd, off),
        "signal": r_lag(signal, off),
        "histogram": r_lag(hist, off),
    }


# ---------------------------------------------------------------------------
# Drivers: real generated-strategy stream, vector replay
# ---------------------------------------------------------------------------


def ohlc_bars(n: int, tf: str, seed: int = 11) -> list[BarEvent]:
    """Random-walk bars with a real open, high, low and close (all four sources differ)."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    open_ = np.concatenate(([100.0], close[:-1])) + rng.randn(n) * 0.1
    high = np.maximum(open_, close) + np.abs(rng.randn(n)) * 0.4
    low = np.minimum(open_, close) - np.abs(rng.randn(n)) * 0.4
    return [
        (tf, float(open_[i]), float(high[i]), float(low[i]), float(close[i]), 1.0) for i in range(n)
    ]


def _merged(type_: str, params: dict[str, Any]) -> dict[str, Any]:
    spec = indicator_registry.get(type_)
    assert spec is not None
    return {**spec.default_params, **params}


def real_cap(type_: str, params: dict[str, Any]) -> int:
    """The cap the compiled strategy gives this indicator's timeframe buffer."""
    spec = indicator_registry.get(type_)
    assert spec is not None
    lookback = pta_lookback(spec, _merged(type_, params))
    return pta_buffer_cap([lookback], full_history=spec.requires_full_history)


def n_for_two_trims(type_: str, params: dict[str, Any]) -> int:
    """Bars so a capped buffer (the harness' cap, never full history) trims at least twice."""
    spec = indicator_registry.get(type_)
    assert spec is not None
    cap = pta_buffer_cap([pta_lookback(spec, _merged(type_, params))], full_history=False)
    return cap + 2 * (cap // 4 + 1) + 6


def stream(
    type_: str,
    params: dict[str, Any],
    bars: list[BarEvent],
    tf: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trimmed: bool,
) -> dict[str, list[float]]:
    """Per-bar held values of every output, from the real generated ``_feed_pta_buffer`` path.

    ``trimmed=False`` uses the compiled strategy's own cap (0 = unbounded for full-history
    specs); ``trimmed=True`` forces a capped buffer (cap 400, trimmed at +25%) even for a full-history spec.
    """
    spec = indicator_registry.get(type_)
    assert spec is not None
    _, cls = compile_indicators({"x": {"type": type_, **params}}, timeframe=tf)
    merged = _merged(type_, params)
    # The harness mirrors the compiler (a full-history spec is unbounded); pin the cap explicitly:
    # the strategy's own for trimmed=False, the capped (never full-history) one for trimmed=True.
    cap = (
        pta_buffer_cap([pta_lookback(spec, merged)], full_history=False)
        if trimmed
        else real_cap(type_, params)
    )
    monkeypatch.setattr(harness, "stream_buffer_caps", lambda _specs: {tf: cap})
    snaps = run_stream(cls, bars, False, specs={"x": (type_, merged, tf)})
    outs = spec.output_names
    keys = {"value": "x"} if len(outs) == 1 else {o: f"x_{o}" for o in outs}
    return {o: [s.get(keys[o], NAN) for s in snaps] for o in outs}


def replay(
    type_: str, params: dict[str, Any], bars: list[BarEvent], cap: int
) -> dict[str, list[float]]:
    spec = indicator_registry.get(type_)
    assert spec is not None
    frame = pd.DataFrame(
        {
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [b[5] for b in bars],
        }
    )
    out = replay_pta_series(spec, frame, _merged(type_, params), cap)
    return {k: v.tolist() for k, v in out.items()}


def assert_identical(got: list[float], want: list[float], label: str) -> None:
    """Zero tolerance: same NaN positions and bit-identical floats."""
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(want, dtype=np.float64)
    assert a.shape == b.shape, f"{label}: {a.shape} != {b.shape}"
    bad = np.flatnonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))
    assert bad.size == 0, (
        f"{label}: {bad.size} of {a.size} bars differ, first at {bad[0]}: got {a[bad[0]]!r} want {b[bad[0]]!r}"
    )


def assert_stream_replay_reference(
    type_: str,
    params: dict[str, Any],
    tf: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trimmed: bool,
    n: int,
) -> None:
    spec = indicator_registry.get(type_)
    assert spec is not None
    bars = ohlc_bars(n, tf)
    want = reference(type_, params, bars)
    got = stream(type_, params, bars, tf, monkeypatch, trimmed=trimmed)
    cap = (
        pta_buffer_cap([pta_lookback(spec, _merged(type_, params))], full_history=False)
        if trimmed
        else real_cap(type_, params)
    )
    rep = replay(type_, params, bars, cap)
    if trimmed:
        starts = pta_window_starts(n, cap)
        assert len(set(starts.tolist())) >= 3, "fewer than two buffer trims in this run"
    for out in spec.output_names:
        assert any(not math.isnan(v) for v in want[out]), f"{type_}.{out}: reference never ready"
        assert_identical(got[out], want[out], f"{type_}.{out} stream vs reference")
        assert_identical(rep[out], want[out], f"{type_}.{out} replay vs reference")


# ---------------------------------------------------------------------------
# R50: runtime == replay == reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize(
    "mode,period",
    [
        ("hma", 55),
        ("ehma", 55),
        ("thma", 55),
        ("hma", 7),
        ("ehma", 7),
        ("thma", 21),
        ("hma_tv", 14),
        ("hma_tv", 7),
    ],
)
def test_hull_runtime_equals_reference_through_two_trims(
    mode: str, period: int, tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HULL on the strategy's real buffer (unbounded: ehma is recursive) is exact every bar."""
    params = {"period": period, "mode": mode}
    assert real_cap("HULL", params) == 0
    assert_stream_replay_reference(
        "HULL", params, tf, monkeypatch, trimmed=False, n=n_for_two_trims("HULL", params)
    )


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize(
    "mode,period",
    [("hma", 55), ("thma", 55), ("hma", 7), ("thma", 21), ("hma_tv", 14), ("hma_tv", 55)],
)
def test_hull_finite_modes_exact_through_trims(
    mode: str, period: int, tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hma/thma depend on their last ``lookback`` bars only: a trimmed buffer changes nothing."""
    assert_stream_replay_reference(
        "HULL",
        {"period": period, "mode": mode},
        tf,
        monkeypatch,
        trimmed=True,
        n=n_for_two_trims("HULL", {"period": period, "mode": mode}),
    )


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_ma_runtime_equals_reference_every_kind(
    kind: str, tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {"period": 9, "kind": kind, "src": "hlc3", "offset": 2}
    assert_stream_replay_reference(
        "MA", params, tf, monkeypatch, trimmed=False, n=n_for_two_trims("MA", params)
    )


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("kind", FINITE_KINDS)
def test_ma_finite_kinds_exact_through_trims(
    kind: str, tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {"period": 9, "kind": kind, "src": "ohlc4", "offset": 1}
    assert_stream_replay_reference(
        "MA", params, tf, monkeypatch, trimmed=True, n=n_for_two_trims("MA", params)
    )


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_ma_macd_runtime_equals_reference_every_kind(
    kind: str, tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {
        "kind": kind,
        "fast_period": 5,
        "slow_period": 9,
        "signal_period": 4,
        "src": "open",
        "offset": 1,
    }
    assert_stream_replay_reference(
        "MA_MACD", params, tf, monkeypatch, trimmed=False, n=n_for_two_trims("MA_MACD", params)
    )


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("kind", FINITE_KINDS)
def test_ma_macd_finite_kinds_exact_through_trims(
    kind: str, tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {"kind": kind, "fast_period": 5, "slow_period": 9, "signal_period": 4, "offset": 0}
    assert_stream_replay_reference(
        "MA_MACD", params, tf, monkeypatch, trimmed=True, n=n_for_two_trims("MA_MACD", params)
    )


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("params", [{}, {"fast_period": 3, "slow_period": 8, "offset": 3}])
def test_ao_exact_through_trims(
    params: dict[str, Any], tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert real_cap("AO", params) == 400  # finite lookback: the strategy trims it
    assert_stream_replay_reference(
        "AO", params, tf, monkeypatch, trimmed=False, n=n_for_two_trims("AO", params)
    )


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("src", ["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"])
def test_price_exact_through_trims(src: str, tf: str, monkeypatch: pytest.MonkeyPatch) -> None:
    params = {"src": src, "offset": 26}
    assert real_cap("PRICE", params) == 400
    assert_stream_replay_reference(
        "PRICE", params, tf, monkeypatch, trimmed=False, n=n_for_two_trims("PRICE", params)
    )


def test_recursive_specs_compile_with_an_unbounded_buffer() -> None:
    """The generated __init__ passes full_history=True for MA/HULL/MA_MACD (any kind) only."""
    expected = {"MA": True, "HULL": True, "MA_MACD": True, "AO": False, "PRICE": False}
    for type_, full in expected.items():
        src = compile_source({"x": {"type": type_}})
        assert f"full_history={full}" in src, type_
        assert real_cap(type_, {}) == (0 if full else 400), type_


# ---------------------------------------------------------------------------
# R51: not ready -> NaN
# ---------------------------------------------------------------------------


def _frame(bars: list[BarEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [b[5] for b in bars],
        }
    )


@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("length", [2, 7, 9])
def test_tv_ma_ready_is_the_first_finite_bar(kind: str, length: int) -> None:
    """``tv_ma_ready`` is exactly (index of first finite) + 1, and the value is finite from then on."""
    if kind == "thma" and length < 3:
        pytest.skip("thma needs length >= 3")
    x = np.array(r_source(ohlc_bars(120, "1h"), "close"))
    out = _tv_ma.tv_ma(kind, x, length)
    ready = _tv_ma.tv_ma_ready(kind, length)
    assert np.isnan(out[: ready - 1]).all()
    assert np.isfinite(out[ready - 1 :]).all()


RAW_RECURSIVE = {"ema": r_ema, "zlema": r_zlema, "wwma": r_wwma_raw, "var": r_var_raw}


@pytest.mark.parametrize("kind", sorted(RAW_RECURSIVE))
@pytest.mark.parametrize("length", [4, 20])
def test_tv_ma_raw_keeps_the_pine_values_the_public_ma_hides(kind: str, length: int) -> None:
    """EMA/ZLEMA/WWMA/VAR exist from their first Pine-defined bar; the public ``tv_ma`` NaNs the
    warm-up only. The raw series equals the independent loop on EVERY bar (zero tolerance) and
    the public one equals it from ``ready`` on."""
    x = r_source(ohlc_bars(80, "1h"), "close")
    raw = _tv_ma.tv_ma_raw(kind, np.array(x), length)
    want = RAW_RECURSIVE[kind](x, length)
    assert_identical(raw.tolist(), want, f"{kind}({length}) raw")
    first = next(i for i, v in enumerate(want) if not math.isnan(v))
    ready = _tv_ma.tv_ma_ready(kind, length)
    assert first < ready - 1, "fixture: the raw series must exist before the public warm-up ends"
    public = _tv_ma.tv_ma(kind, np.array(x), length)
    assert np.isnan(public[: ready - 1]).all()
    assert_identical(public[ready - 1 :].tolist(), want[ready - 1 :], f"{kind}({length}) public")


@pytest.mark.parametrize("kind", ["sma", "wma", "rma", "tma", "tsf", "hma", "thma"])
def test_tv_ma_raw_equals_public_for_window_kinds(kind: str) -> None:
    x = np.array(r_source(ohlc_bars(80, "1h"), "close"))
    assert_identical(_tv_ma.tv_ma_raw(kind, x, 9).tolist(), _tv_ma.tv_ma(kind, x, 9).tolist(), kind)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_tv_ma_on_a_leading_nan_source_is_ready_f_plus_ready_minus_one(kind: str) -> None:
    """A source whose first finite row is f: the first finite output is f + ready - 1
    (this is what lets MA_MACD feed its macd line into a signal MA)."""
    x = np.array(r_source(ohlc_bars(160, "1h"), "close"))
    x[:11] = np.nan
    out = _tv_ma.tv_ma(kind, x, 7)
    first = int(np.flatnonzero(np.isfinite(out))[0])
    assert first == 11 + _tv_ma.tv_ma_ready(kind, 7) - 1


@pytest.mark.parametrize(
    "type_,params",
    [
        ("PRICE", {"src": "hl2", "offset": 5}),
        ("MA", {"kind": "t3", "period": 4, "offset": 3}),
        ("MA", {"kind": "zlema", "period": 9}),
        ("HULL", {"mode": "thma", "period": 20, "offset": 2}),
        ("HULL", {"mode": "ehma", "period": 15}),
        ("MA_MACD", {"kind": "tema", "fast_period": 3, "slow_period": 5, "signal_period": 3}),
        ("AO", {"fast_period": 3, "slow_period": 6, "offset": 1}),
    ],
)
def test_not_ready_is_nan_and_the_gate_opens_on_the_first_finite_bar(
    type_: str, params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every output is NaN below the lookback and finite from it; the strategy stream stores
    no value until then (so a condition on the indicator is false)."""
    spec = indicator_registry.get(type_)
    assert spec is not None
    lookback = pta_lookback(spec, _merged(type_, params))
    bars = ohlc_bars(lookback + 30, "1h")
    df = _frame(bars)
    below = invoke_compute_fn(spec, df.iloc[: lookback - 1], _merged(type_, params))
    at = invoke_compute_fn(spec, df.iloc[:lookback], _merged(type_, params))
    below_d = below if isinstance(below, dict) else {"value": below}
    at_d = at if isinstance(at, dict) else {"value": at}
    for out in spec.output_names:
        assert below_d[out].isna().all(), f"{type_}.{out} finite below the lookback"
    primary = spec.output_names[0]  # the earliest-ready output (MA_MACD: macd; signal follows)
    assert np.isfinite(at_d[primary].iloc[-1]), f"{type_}.{primary} not finite at the lookback"
    if lookback > 1:
        assert np.isnan(at_d[primary].iloc[-2])
    snap = stream(type_, params, bars, "1h", monkeypatch, trimmed=False)
    want = reference(type_, params, bars)
    for out in spec.output_names:
        first = next(i for i, v in enumerate(snap[out]) if not math.isnan(v))
        assert first == _first_finite(want[out]), f"{type_}.{out} first stored value at bar {first}"
    assert (
        min(next(i for i, v in enumerate(snap[o]) if not math.isnan(v)) for o in spec.output_names)
        == lookback - 1
    )


# ---------------------------------------------------------------------------
# Literal values (hand-computed)
# ---------------------------------------------------------------------------


def _df(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    open_: list[float] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": open_ or close,
            "high": high or close,
            "low": low or close,
            "close": close,
            "volume": [1.0] * len(close),
        }
    )


def _run(type_: str, df: pd.DataFrame, params: dict[str, Any]) -> Any:
    spec = indicator_registry.get(type_)
    assert spec is not None
    return invoke_compute_fn(spec, df, _merged(type_, params))


def test_price_literal_values() -> None:
    close = [10.0, 11.0, 13.0, 12.0, 15.0]
    got = _run("PRICE", _df(close), {"offset": 2})
    assert got.isna().tolist() == [True, True, False, False, False]
    assert got.tolist()[2:] == [10.0, 11.0, 13.0]
    opens = [5.0, 6.0, 8.0, 7.0]
    got = _run("PRICE", _df([1.0] * 4, open_=opens), {"src": "open", "offset": 1})
    assert got.isna().tolist() == [True, False, False, False]
    assert got.tolist()[1:] == [5.0, 6.0, 8.0]
    hl2 = _run(
        "PRICE", _df([9.0] * 3, high=[12.0, 13.0, 15.0], low=[8.0, 9.0, 11.0]), {"src": "hl2"}
    )
    assert hl2.tolist() == [10.0, 11.0, 13.0]
    ohlc4 = _run(
        "PRICE",
        _df([4.0, 8.0], high=[6.0, 10.0], low=[2.0, 6.0], open_=[3.0, 7.0]),
        {"src": "ohlc4"},
    )
    assert ohlc4.tolist() == [3.75, 7.75]
    hlc3 = _run("PRICE", _df([4.0, 8.0], high=[6.0, 10.0], low=[2.0, 6.0]), {"src": "hlc3"})
    assert hlc3.tolist() == [4.0, 8.0]


def test_ma_literal_values() -> None:
    x = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
    wma = _run("MA", _df(x), {"kind": "wma", "period": 3}).tolist()
    assert np.isnan(wma[:2]).all()
    assert wma[2:] == [51 / 18, 93 / 18, 153 / 18, 231 / 18]
    # Pine v5 ta.ema seeds with the first source value: 1, 1.5, 2.75, 4.875, 7.9375, 11.96875
    ema = _run("MA", _df(x), {"kind": "ema", "period": 3}).tolist()
    assert np.isnan(ema[:2]).all()
    assert ema[2:] == [2.75, 4.875, 7.9375, 11.96875]
    # rma keeps the SMA seed: sma(1,2,4) = 7/3, then 1/3*src + 2/3*prev
    rma = _run("MA", _df(x), {"kind": "rma", "period": 3}).tolist()
    assert np.isnan(rma[:2]).all()
    assert rma[2] == 7 / 3
    tsf = _run("MA", _df([1.0, 3.0, 5.0, 7.0]), {"kind": "tsf", "period": 3}).tolist()
    assert np.isnan(tsf[:2]).all()
    assert tsf[2:] == pytest.approx([7.0, 9.0])
    tma = _run("MA", _df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), {"kind": "tma", "period": 4}).tolist()
    assert np.isnan(tma[:3]).all()
    assert tma[3:] == pytest.approx([2.5, 3.5, 4.5])
    wwma = _run("MA", _df([2.0, 4.0, 8.0, 16.0]), {"kind": "wwma", "period": 2}).tolist()
    assert np.isnan(wwma[0])
    assert wwma[1:] == [2.5, 5.25, 10.625]
    zlema = _run(
        "MA", _df([float(i) for i in range(1, 9)]), {"kind": "zlema", "period": 4}
    ).tolist()
    assert np.isnan(zlema[:5]).all()
    # zx = 2x - x[2 bars ago] = x + 2 from bar 2; ema(alpha 0.4) seeded at zx[2] = 5: 5, 5.4, 6.04, 6.824, ...
    assert zlema[5:] == pytest.approx([6.824, 7.6944, 8.61664])
    for kind in ("dema", "tema", "t3", "rma"):  # a constant series is its own average
        const = _run("MA", _df([5.0] * 60), {"kind": kind, "period": 4}).tolist()
        assert const[-1] == pytest.approx(5.0, abs=1e-12), kind


def test_hull_literal_values() -> None:
    x = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    got = _run("HULL", _df(x), {"mode": "hma", "period": 4}).tolist()
    assert np.isnan(got[:4]).all()
    assert got[4:] == pytest.approx([1265 / 90, 1265 / 45])
    shifted = _run("HULL", _df(x), {"mode": "hma", "period": 4, "offset": 1}).tolist()
    assert np.isnan(shifted[:5]).all()
    assert shifted[5] == pytest.approx(1265 / 90)
    # thma(period 8) = THMA(src, 4): wma(wma(x,1)*3 - wma(x,2) - wma(x,4), 4), first finite at bar 2*4-1
    thma = _run("HULL", _df([float(i * i) for i in range(1, 12)]), {"mode": "thma", "period": 8})
    assert thma.isna().tolist() == [True] * 6 + [False] * 5


def test_hma_tv_floors_the_square_root_where_hma_rounds_it() -> None:
    """UNVERIFIED vs TradingView (no export available): the built-in ``hma`` FLOORS ``sqrt(length)``.

    On a line x_t = t, wma(n) = t - (n-1)/3, so ``2*wma(7) - wma(14) = t + 1/3`` and the final
    ``wma(., m) = t + 1/3 - (m-1)/3``. Length 14: floor(sqrt) = 3 -> ``t - 1/3`` first finite at bar
    14 + 3 - 2 = 15; round(sqrt) = 4 -> ``t - 2/3`` first finite at bar 16.
    """
    line = [float(i) for i in range(20)]
    tv = _run("MA", _df(line), {"kind": "hma_tv", "period": 14}).tolist()
    assert np.isnan(tv[:15]).all()
    assert tv[15:] == pytest.approx([i - 1 / 3 for i in range(15, 20)], abs=1e-12)
    rounded = _run("MA", _df(line), {"kind": "hma", "period": 14}).tolist()
    assert np.isnan(rounded[:16]).all()
    assert rounded[16:] == pytest.approx([i - 2 / 3 for i in range(16, 20)], abs=1e-12)
    hull = _run("HULL", _df(line), {"mode": "hma_tv", "period": 14}).tolist()
    assert hull[15:] == tv[15:]


def test_hma_tv_equals_hma_when_floor_and_round_of_the_root_agree() -> None:
    x = np.array(r_source(ohlc_bars(120, "1h"), "close"))
    for n in (10, 16, 25, 36):  # sqrt 3.16, 4, 5, 6: floor == round
        assert np.array_equal(
            _tv_ma.tv_ma("hma_tv", x, n), _tv_ma.tv_ma("hma", x, n), equal_nan=True
        )
    for n in (7, 14, 20, 55):  # sqrt 2.65, 3.74, 4.47, 7.42: only 7 and 14 round up
        same = np.array_equal(
            _tv_ma.tv_ma("hma_tv", x, n), _tv_ma.tv_ma("hma", x, n), equal_nan=True
        )
        assert same == (math.floor(math.sqrt(n)) == math.floor(math.sqrt(n) + 0.5)), n


def test_ma_macd_literal_values() -> None:
    params = {"kind": "sma", "fast_period": 2, "slow_period": 3, "signal_period": 2}
    got = _run("MA_MACD", _df([1.0, 2.0, 4.0, 7.0, 11.0, 16.0]), params)
    assert got["macd"].tolist()[2:] == pytest.approx([2 / 3, 7 / 6, 5 / 3, 13 / 6])
    assert got["signal"].tolist()[3:] == pytest.approx([11 / 12, 17 / 12, 23 / 12])
    assert got["histogram"].tolist()[3:] == pytest.approx([0.25, 0.25, 0.25])
    assert got["signal"].isna().tolist()[:3] == [True] * 3
    other = _run("MA_MACD", _df([3.0, 1.0, 4.0, 1.0, 5.0, 9.0]), params)
    assert other["macd"].tolist()[2:] == pytest.approx([-1 / 6, 0.5, -1 / 3, 2.0])
    assert other["signal"].tolist()[3:] == pytest.approx([1 / 6, 1 / 12, 5 / 6])
    assert other["histogram"].tolist()[3:] == pytest.approx([1 / 3, -5 / 12, 7 / 6])
    lagged = _run("MA_MACD", _df([1.0, 2.0, 4.0, 7.0, 11.0, 16.0]), {**params, "offset": 1})
    assert lagged["histogram"].isna().tolist() == [True] * 4 + [False] * 2


def test_ma_macd_ema_signal_recurses_on_the_raw_macd_line() -> None:
    """kind ema, fast 2, slow 3, signal 2 on [1,2,4,7,11,16]. Pine's EMAs exist from bar 0, so the
    macd line is defined from bar 0 and ``ema(macd, 2)`` is seeded there (exact Fractions below);
    the exposed macd is ready at bar 2 and the exposed signal at bar 3. Seeding the signal at bar 2
    from the masked macd line (the old feed) gives a different bar-3 value."""
    from fractions import Fraction as Fr

    def ema(x: list[Fr], n: int) -> list[Fr]:
        a = Fr(2, n + 1)
        out = [x[0]]
        for v in x[1:]:
            out.append(a * v + (1 - a) * out[-1])
        return out

    close = [Fr(v) for v in (1, 2, 4, 7, 11, 16)]
    macd_raw = [f - s for f, s in zip(ema(close, 2), ema(close, 3), strict=True)]
    signal = ema(macd_raw, 2)
    masked_seed = ema(macd_raw[2:], 2)  # what a feed from the masked macd line produced
    assert signal[3] != masked_seed[1]  # the fixture tells the two feeds apart
    params = {"kind": "ema", "fast_period": 2, "slow_period": 3, "signal_period": 2}
    got = _run("MA_MACD", _df([1.0, 2.0, 4.0, 7.0, 11.0, 16.0]), params)
    assert got["macd"].isna().tolist() == [True, True] + [False] * 4
    assert got["macd"].tolist()[2:] == pytest.approx([float(v) for v in macd_raw[2:]], abs=1e-12)
    assert got["signal"].isna().tolist() == [True] * 3 + [False] * 3
    assert got["signal"].tolist()[3:] == pytest.approx([float(v) for v in signal[3:]], abs=1e-12)


def test_ao_literal_values() -> None:
    hl2 = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
    df = _df(hl2, high=[v + 1 for v in hl2], low=[v - 1 for v in hl2])
    got = _run("AO", df, {"fast_period": 2, "slow_period": 3}).tolist()
    assert np.isnan(got[:2]).all()
    assert got[2:] == pytest.approx([2 / 3, 7 / 6, 5 / 3, 13 / 6])
    prev = _run("AO", df, {"fast_period": 2, "slow_period": 3, "offset": 1}).tolist()
    assert np.isnan(prev[:3]).all()
    assert prev[3:] == pytest.approx([2 / 3, 7 / 6, 5 / 3])


# ---------------------------------------------------------------------------
# Kernel edges and parameter validation
# ---------------------------------------------------------------------------


def test_shift_back_edges() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _tv_ma.shift_back(a, 0).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert _tv_ma.shift_back(a, 1)[1:].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(_tv_ma.shift_back(a, 1)[0])
    assert np.isnan(_tv_ma.shift_back(a, 3)[:3]).all()
    assert _tv_ma.shift_back(a, 3)[3] == 1.0
    assert np.isnan(_tv_ma.shift_back(a, 4)).all()
    assert np.isnan(_tv_ma.shift_back(a, 9)).all()
    with pytest.raises(ValueError, match="offset"):
        _tv_ma.shift_back(a, -1)


@pytest.mark.parametrize(
    "type_,params,match",
    [
        ("MA", {"kind": "nope"}, "unknown MA kind"),
        ("MA", {"kind": "hma", "period": 1}, "length >= 2"),
        ("MA", {"kind": "thma", "period": 2}, "length >= 3"),
        ("MA", {"kind": "tsf", "period": 1}, "length >= 2"),
        ("MA", {"period": 0}, "length >= 1"),
        ("MA", {"src": "volume"}, "src"),
        ("MA", {"offset": -1}, "offset"),
        ("HULL", {"mode": "wma"}, "unknown Hull mode"),
        ("HULL", {"mode": "thma", "period": 5}, "length >= 3"),
        ("HULL", {"mode": "hma_tv", "period": 1}, "length >= 2"),
        ("MA_MACD", {"kind": "hma_tv", "fast_period": 1}, "length >= 2"),
        ("PRICE", {"src": "median"}, "src"),
        ("PRICE", {"offset": -2}, "offset"),
        ("AO", {"offset": -1}, "offset"),
        ("MA_MACD", {"kind": "zzz"}, "unknown MA kind"),
    ],
)
def test_bad_params_fail_loudly_at_lookback_time(
    type_: str, params: dict[str, Any], match: str
) -> None:
    spec = indicator_registry.get(type_)
    assert spec is not None
    with pytest.raises(ValueError, match=match):
        pta_lookback(spec, _merged(type_, params))


def test_kind_and_src_are_case_insensitive_and_compile_through_the_dsl() -> None:
    upper = _run("MA", _df([1.0, 2.0, 4.0, 7.0]), {"kind": "WMA", "period": 3, "src": "CLOSE"})
    lower = _run("MA", _df([1.0, 2.0, 4.0, 7.0]), {"kind": "wma", "period": 3, "src": "close"})
    assert upper.tolist()[2:] == lower.tolist()[2:] == [51 / 18, 93 / 18]
    module, cls = compile_indicators(
        {"x": {"type": "MA", "kind": "hma", "period": 21, "src": "hl2", "offset": 3}}
    )
    assert cls is not None and module is not None
    with pytest.raises(ValueError, match="unknown MA kind"):
        compile_indicators({"x": {"type": "MA", "kind": "bogus"}})


def test_hull_inner_length_truncates_odd_periods() -> None:
    """Pine ``length / 2`` is a truncated series length: period 7 -> 3 (round-half-even would give 4)."""
    x = np.array(r_source(ohlc_bars(80, "1h"), "close"))
    got = _tv_ma.tv_ma("hma", x, 7)
    assert_identical(got.tolist(), r_hma(x.tolist(), 7), "hma(7)")
    inner_floor = _tv_ma.tv_ma("wma", x, 3)
    inner_round = _tv_ma.tv_ma("wma", x, 4)
    wrong = _tv_ma.tv_ma("wma", 2 * inner_round - _tv_ma.tv_ma("wma", x, 7), 3)
    right = _tv_ma.tv_ma("wma", 2 * inner_floor - _tv_ma.tv_ma("wma", x, 7), 3)
    assert np.array_equal(got, right, equal_nan=True)
    assert not np.array_equal(got, wrong, equal_nan=True)


def test_ema_alpha_is_two_over_length_plus_one() -> None:
    """Last value of ema(3) on [3,3,3,9]: 0.5*9 + 0.5*3 = 6 (alpha 1/3 would give 5)."""
    got = _tv_ma.tv_ma("ema", np.array([3.0, 3.0, 3.0, 9.0]), 3)
    assert got[3] == 6.0
    rma = _tv_ma.tv_ma("rma", np.array([3.0, 3.0, 3.0, 9.0]), 3)
    assert rma[3] == 5.0


def test_ema_is_seeded_with_the_first_source_value_not_the_sma() -> None:
    """Pine v5 ta.ema: first value 1 (not sma 7/3); declared-ready bar 2 is 2.75, last 11.96875."""
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    got = _tv_ma.tv_ma("ema", x, 3)
    assert np.isnan(got[:2]).all()
    assert got[2:].tolist() == [2.75, 4.875, 7.9375, 11.96875]
    other = _tv_ma.tv_ma("ema", np.array([10.0, 12.0, 9.0, 15.0]), 2)  # alpha 2/3
    assert np.isnan(other[0])
    assert other[1:].tolist() == pytest.approx(
        [34 / 3, 9 * 2 / 3 + 34 / 9, 15 * 2 / 3 + (6 + 34 / 9) / 3], abs=1e-12
    )


def test_ema_family_masks_warmup_but_seeds_from_first_finite_value() -> None:
    """Leading-NaN source (MACD line into the signal MA): the recursion starts at the first
    finite value f and the output is NaN until f + ready - 1."""
    x = np.array([NAN, NAN, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    got = _tv_ma.tv_ma("ema", x, 3)
    assert np.isnan(got[:4]).all()
    assert got[4:].tolist() == [2.75, 4.875, 7.9375, 11.96875]
    dema = _tv_ma.tv_ma("dema", x, 2)  # ready 3 -> first finite at 2 + 2 = 4
    assert np.isnan(dema[:4]).all() and np.isfinite(dema[4:]).all()
    # e1 (alpha 2/3, raw from bar 2): 1, 1.6667, 3.2222; e2 seeded at e1[0]: 1, 1.4444, 2.6296
    e1 = [1.0, 2 / 3 * 2 + 1 / 3 * 1.0]
    e1.append(2 / 3 * 4 + 1 / 3 * e1[1])
    e2 = [1.0, 2 / 3 * e1[1] + 1 / 3 * 1.0]
    e2.append(2 / 3 * e1[2] + 1 / 3 * e2[1])
    assert dema[4] == pytest.approx(2 * e1[2] - e2[2], abs=1e-12)


def test_catalogue_metadata() -> None:
    """The new specs are in the editor catalogue with overlay/oscillator placement and the
    full-history flag only where a recursive kind exists."""
    placement = {
        "PRICE": "overlay",
        "MA": "overlay",
        "HULL": "overlay",
        "MA_MACD": "oscillator",
        "AO": "oscillator",
    }
    full = {"PRICE": False, "MA": True, "HULL": True, "MA_MACD": True, "AO": False}
    for name, where in placement.items():
        spec = indicator_registry.get(name)
        assert spec is not None
        assert spec.chart_placement == where
        assert spec.requires_full_history is full[name]
        assert spec.threshold_range is None
        assert spec.ma_kind is False
        assert spec.param_ranges == {}
    macd = indicator_registry.get("MA_MACD")
    assert macd is not None
    assert macd.output_names == ("macd", "signal", "histogram")
    assert macd.primary_output == "macd"
    assert re.fullmatch(r"[A-Z_]+", macd.name)
