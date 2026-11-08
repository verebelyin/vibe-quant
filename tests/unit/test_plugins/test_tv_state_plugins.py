"""TradingView-exact stateful plugins: SUPERTREND, PMAX, SIGNAL_LEVELS, RSI_DIV, ICHIMOKU_LEAD.

Money-grade proof (R50): for every plugin and params variant, on 1m and 1h, the value the REAL
generated strategy stream produces after each bar == the vector replay the fast tier / NT
precompute seam uses == an INDEPENDENT full-history reference, with zero tolerance. The
references below are plain-Python-loop transcriptions of the Pine scripts in series form
(``x[1]`` is the previous bar's final value, ``nz``, ``valuewhen``, ``barssince`` as functions
of the whole series); they import nothing from the plugin modules or ``_tv_ma``.

Buffer regimes: the ratchet and RSI plugins set ``requires_full_history`` (cap 0, never trimmed)
and are driven for more bars than two trims of the harness cap; ICHIMOKU_LEAD and the SMA-ATR
SIGNAL_LEVELS are finite-lookback and are also driven through the harness' capped buffer
(cap 400, trimmed at +25%, >= 2 trims). R51: not ready -> NaN. R52: RSI_DIV pulses return to
0.0 after the confirming bar (never NaN, so the runtime's last-value hold cannot latch them).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.unit.pta_stream_harness import BarEvent, compile_indicators
from tests.unit.test_plugins.test_tv_ma_plugins import (
    R_KINDS,
    TFS,
    _frame,
    _merged,
    assert_identical,
    n_for_two_trims,
    ohlc_bars,
    r_ema,
    r_lag,
    r_rma,
    r_sma,
    r_source,
    r_tma,
    r_tsf,
    r_var_raw,
    r_wma,
    r_wwma_raw,
    r_zlema,
    real_cap,
    replay,
    stream,
)
from vibe_quant.dsl import invoke_compute_fn
from vibe_quant.dsl.indicators import indicator_registry, pta_buffer_cap, pta_lookback
from vibe_quant.dsl.parser import DSLValidationError
from vibe_quant.dsl.prefix_memo import MEMO
from vibe_quant.dsl.pta_replay import pta_window_starts

NAN = float("nan")
PMAX_KINDS = ("sma", "ema", "wma", "tma", "var", "wwma", "zlema", "tsf")


@pytest.fixture(autouse=True)
def _clean_memo() -> object:
    MEMO.clear()
    yield
    MEMO.clear()


# ---------------------------------------------------------------------------
# Independent references (Pine series semantics, plain loops over Python floats)
# ---------------------------------------------------------------------------


def nz(v: float, repl: float = 0.0) -> float:
    return repl if math.isnan(v) else v


def prev(a: list[float], t: int) -> float:
    """Pine ``a[1]`` at bar ``t``."""
    return a[t - 1] if t > 0 else NAN


def r_tr(bars: list[BarEvent], handle_na: bool) -> list[float]:
    out: list[float] = []
    for i, (_tf, _o, h, lo, _c, _v) in enumerate(bars):
        if i == 0:
            out.append(h - lo if handle_na else NAN)
        else:
            pc = bars[i - 1][4]
            out.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    return out


def r_atr(bars: list[BarEvent], n: int, method: str) -> list[float]:
    """rma = ``atr(n)`` (tr with first-bar high-low); rma_tr = ``rma(tr, n)``; sma = ``sma(tr, n)``."""
    tr = r_tr(bars, method == "rma")
    return r_sma(tr, n) if method == "sma" else r_rma(tr, n)


def r_supertrend(bars: list[BarEvent], p: dict[str, Any]) -> dict[str, list[float]]:
    """03_supertrend_strategy.pine l.14-24, series form."""
    src = r_source(bars, p["src"])
    close = [b[4] for b in bars]
    atr = r_atr(bars, p["period"], p["atr_method"])
    mult = p["multiplier"]
    size = len(bars)
    up = [NAN] * size
    dn = [NAN] * size
    trend = [NAN] * size
    for t in range(size):
        up_raw = src[t] - mult * atr[t]
        up1 = nz(prev(up, t), up_raw)
        up[t] = max(up_raw, up1) if prev(close, t) > up1 else up_raw
        dn_raw = src[t] + mult * atr[t]
        dn1 = nz(prev(dn, t), dn_raw)
        dn[t] = min(dn_raw, dn1) if prev(close, t) < dn1 else dn_raw
        tr = nz(prev(trend, t), 1.0)
        if tr == -1 and close[t] > dn1:
            tr = 1.0
        elif tr == 1 and close[t] < up1:
            tr = -1.0
        trend[t] = tr
    ready = [not math.isnan(a) for a in atr]
    mask = [NAN] * size

    def keep(vals: list[float]) -> list[float]:
        return [v if ok else m for v, ok, m in zip(vals, ready, mask, strict=True)]

    line = [up[t] if trend[t] == 1 else dn[t] for t in range(size)]
    return {"value": keep(line), "dir": keep(trend), "up": keep(up), "dn": keep(dn)}


# ``getMA`` (05_pmax_explorer.pine l.45-80) as Pine evaluates it: the RAW series the stops recurse
# on. EMA/WWMA/VAR/ZLEMA exist from their first Pine-defined bar; the ``R_KINDS`` warm-up masks
# only hide the exposed ``mavg``/``value``/``dir`` (``ready`` below), never feed the ratchet.
R_RAW_KINDS = {
    "sma": r_sma,
    "ema": r_ema,
    "wma": r_wma,
    "tma": r_tma,
    "var": r_var_raw,
    "wwma": r_wwma_raw,
    "zlema": r_zlema,
    "tsf": r_tsf,
}


def r_pmax(bars: list[BarEvent], p: dict[str, Any]) -> dict[str, list[float]]:
    """05_pmax_explorer.pine ``Pmax_Func`` l.81-92 with ``MAvg = getMA`` l.45-80."""
    src = r_source(bars, p["src"])
    mavg = R_RAW_KINDS[p["kind"]](src, p["ma_period"])
    mavg_public = R_KINDS[p["kind"]](src, p["ma_period"])
    atr = r_atr(bars, p["period"], p["atr_method"])
    mult = p["multiplier"]
    size = len(bars)
    long_stop = [NAN] * size
    short_stop = [NAN] * size
    direction = [NAN] * size
    pmax = [NAN] * size
    for t in range(size):
        ls_raw = mavg[t] - mult * atr[t]
        ls_prev = nz(prev(long_stop, t), ls_raw)
        long_stop[t] = max(ls_raw, ls_prev) if mavg[t] > ls_prev else ls_raw
        ss_raw = mavg[t] + mult * atr[t]
        ss_prev = nz(prev(short_stop, t), ss_raw)
        short_stop[t] = min(ss_raw, ss_prev) if mavg[t] < ss_prev else ss_raw
        d = nz(prev(direction, t), 1.0) if t > 0 else 1.0
        if d == -1 and mavg[t] > ss_prev:
            d = 1.0
        elif d == 1 and mavg[t] < ls_prev:
            d = -1.0
        direction[t] = d
        pmax[t] = long_stop[t] if d == 1 else short_stop[t]
    ok = [not (math.isnan(m) or math.isnan(a)) for m, a in zip(mavg_public, atr, strict=True)]
    return {
        "value": [v if k else NAN for v, k in zip(pmax, ok, strict=True)],
        "mavg": mavg_public,
        "dir": [v if k else NAN for v, k in zip(direction, ok, strict=True)],
    }


def r_lowest(x: list[float], n: int) -> list[float]:
    return [min(x[t - n + 1 : t + 1]) if t >= n - 1 else NAN for t in range(len(x))]


def r_highest(x: list[float], n: int) -> list[float]:
    return [max(x[t - n + 1 : t + 1]) if t >= n - 1 else NAN for t in range(len(x))]


def r_signal_levels(bars: list[BarEvent], p: dict[str, Any]) -> dict[str, list[float]]:
    """06_3commas_bot.pine l.166, 170-171, 226-233 and 08_ao_stoch_rsi_atr.pine l.55-64."""
    high = [b[2] for b in bars]
    low = [b[3] for b in bars]
    close = [b[4] for b in bars]
    atr = r_atr(bars, p["atr_period"], p["atr_method"])
    lowest = r_lowest(low, p["lookback"])
    highest = r_highest(high, p["lookback"])
    out: dict[str, list[float]] = {
        k: [] for k in ("long_stop", "long_target", "short_stop", "short_target")
    }
    for t in range(len(bars)):
        ls = lowest[t] - atr[t] * p["risk_mult"]
        ss = highest[t] + atr[t] * p["risk_mult"]
        if p["target_mode"] == "rr":
            lt = close[t] + p["rr"] * (close[t] - ls)
            st = close[t] - p["rr"] * (ss - close[t])
        else:
            lt = close[t] + p["target_atr"] * atr[t]
            st = close[t] - p["target_atr"] * atr[t]
        out["long_stop"].append(ls)
        out["long_target"].append(lt)
        out["short_stop"].append(ss)
        out["short_target"].append(st)
    return out


def r_ichimoku_lead(bars: list[BarEvent], p: dict[str, Any]) -> dict[str, list[float]]:
    """14_ichimoku_hull_macd.pine l.23-27: ``donchian(len) = avg(lowest(len), highest(len))``."""
    high = [b[2] for b in bars]
    low = [b[3] for b in bars]

    def donchian(n: int) -> list[float]:
        lo = r_lowest(low, n)
        hi = r_highest(high, n)
        return [(a + b) / 2 for a, b in zip(lo, hi, strict=True)]

    conv = donchian(p["conversion_period"])
    base = donchian(p["base_period"])
    return {
        "lead1": [(a + b) / 2 for a, b in zip(conv, base, strict=True)],
        "lead2": donchian(p["span_period"]),
    }


def r_rsi(x: list[float], n: int) -> list[float]:
    """Pine ``rsi``: ``down == 0 ? 100 : up == 0 ? 0 : 100 - 100/(1 + up/down)``."""
    change = [NAN] + [x[t] - x[t - 1] for t in range(1, len(x))]
    gain = [max(c, 0.0) if not math.isnan(c) else NAN for c in change]
    loss = [-min(c, 0.0) if not math.isnan(c) else NAN for c in change]
    up = r_rma(gain, n)
    down = r_rma(loss, n)
    out: list[float] = []
    for u, d in zip(up, down, strict=True):
        if math.isnan(u) or math.isnan(d):
            out.append(NAN)
        elif d == 0:
            out.append(100.0)
        elif u == 0:
            out.append(0.0)
        else:
            out.append(100 - 100 / (1 + u / d))
    return out


def r_pivot(osc: list[float], t: int, left: int, right: int, *, low: bool) -> bool:
    """``pivotlow``/``pivothigh(osc, left, right)`` evaluated on bar ``t`` (centre ``t - right``)."""
    c = t - right
    if c - left < 0:
        return False
    window = osc[c - left : t + 1]
    if any(math.isnan(v) for v in window):
        return False
    lefts = window[:left]
    rights = window[left + 1 :]
    centre = window[left]
    if low:
        return all(v > centre for v in lefts) and all(v >= centre for v in rights)
    return all(v < centre for v in lefts) and all(v <= centre for v in rights)


def r_valuewhen(cond: list[bool], x: list[float], occurrence: int, t: int) -> float:
    hits = [i for i in range(t + 1) if cond[i]]
    return x[hits[-1 - occurrence]] if len(hits) > occurrence else NAN


def r_barssince(cond: list[bool], t: int) -> float:
    hits = [i for i in range(t + 1) if cond[i]]
    return float(t - hits[-1]) if hits else NAN


def r_rsi_div(bars: list[BarEvent], p: dict[str, Any]) -> dict[str, list[float]]:
    """13_rsi_divergence.pine l.38-62, 90-95, 118, 145-150, 177-182 in Pine series form."""
    osc = r_rsi(r_source(bars, p["src"]), p["period"])
    low = [b[3] for b in bars]
    high = [b[2] for b in bars]
    size = len(bars)
    left, right = p["lb_left"], p["lb_right"]
    pl = [r_pivot(osc, t, left, right, low=True) for t in range(size)]
    ph = [r_pivot(osc, t, left, right, low=False) for t in range(size)]
    osc_r, low_r, high_r = r_lag(osc, right), r_lag(low, right), r_lag(high, right)
    pl_prev = [False, *pl[:-1]]
    ph_prev = [False, *ph[:-1]]

    def in_range(cond_prev: list[bool], t: int) -> bool:
        bars_since = r_barssince(cond_prev, t)
        return not math.isnan(bars_since) and p["range_lower"] <= bars_since <= p["range_upper"]

    out = {
        k: [0.0] * size
        for k in ("bull", "hidden_bull", "bear", "hidden_bear", "long", "long_exit")
    }
    for t in range(size):
        if pl[t]:
            v_osc = r_valuewhen(pl, osc_r, 1, t)
            v_low = r_valuewhen(pl, low_r, 1, t)
            rng = in_range(pl_prev, t)
            if p["plot_bull"] and low_r[t] < v_low and osc_r[t] > v_osc and rng:
                out["bull"][t] = 1.0
            if p["plot_hidden_bull"] and low_r[t] > v_low and osc_r[t] < v_osc and rng:
                out["hidden_bull"][t] = 1.0
        if ph[t]:
            v_osc = r_valuewhen(ph, osc_r, 1, t)
            v_high = r_valuewhen(ph, high_r, 1, t)
            rng = in_range(ph_prev, t)
            if p["plot_bear"] and high_r[t] > v_high and osc_r[t] < v_osc and rng:
                out["bear"][t] = 1.0
            if p["plot_hidden_bear"] and high_r[t] < v_high and osc_r[t] > v_osc and rng:
                out["hidden_bear"][t] = 1.0
        out["long"][t] = 1.0 if out["bull"][t] or out["hidden_bull"][t] else 0.0
        # l.204: longCloseCondition = crossover(osc, takeProfitRSILevel) or bearCond
        cross = (
            t >= 1
            and not math.isnan(osc[t])
            and not math.isnan(osc[t - 1])
            and osc[t] > p["take_profit_level"]
            and osc[t - 1] <= p["take_profit_level"]
        )
        out["long_exit"][t] = 1.0 if cross or out["bear"][t] == 1.0 else 0.0
    return {"rsi": osc, **out}


REFERENCES = {
    "SUPERTREND": r_supertrend,
    "PMAX": r_pmax,
    "SIGNAL_LEVELS": r_signal_levels,
    "ICHIMOKU_LEAD": r_ichimoku_lead,
    "RSI_DIV": r_rsi_div,
}
PULSES = ("bull", "hidden_bull", "bear", "hidden_bear", "long", "long_exit")


def reference(type_: str, params: dict[str, Any], bars: list[BarEvent]) -> dict[str, list[float]]:
    """Pine-definition reference for one plugin; every output lagged by ``offset``."""
    p = _merged(type_, params)
    raw = REFERENCES[type_](bars, p)
    off = int(p["offset"])
    out: dict[str, list[float]] = {}
    for name, vals in raw.items():
        if type_ == "RSI_DIV" and name in PULSES:
            out[name] = [0.0] * off + vals[: len(vals) - off] if off else vals
        else:
            out[name] = r_lag(vals, off)
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def check_stream_replay_reference(
    type_: str,
    params: dict[str, Any],
    tf: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trimmed: bool,
    seed: int = 11,
) -> dict[str, list[float]]:
    """runtime stream == vector replay == reference on every bar, every output (zero tolerance)."""
    spec = indicator_registry.get(type_)
    assert spec is not None
    n = n_for_two_trims(type_, params)
    bars = ohlc_bars(n, tf, seed)
    want = reference(type_, params, bars)
    got = stream(type_, params, bars, tf, monkeypatch, trimmed=trimmed)
    cap = (
        pta_buffer_cap([pta_lookback(spec, _merged(type_, params))], full_history=False)
        if trimmed
        else real_cap(type_, params)
    )
    rep = replay(type_, params, bars, cap)
    if trimmed:
        assert len(set(pta_window_starts(n, cap).tolist())) >= 3, "fewer than two buffer trims"
    lookback = pta_lookback(spec, _merged(type_, params))
    for out in spec.output_names:
        gated = [NAN if i < lookback - 1 else v for i, v in enumerate(want[out])]
        assert any(not math.isnan(v) for v in gated), f"{type_}.{out}: reference never ready"
        assert_identical(got[out], _held(gated), f"{type_}.{out} stream vs reference")
        assert_identical(rep[out], _held(gated), f"{type_}.{out} replay vs reference")
    return want


def _held(values: list[float]) -> list[float]:
    """The runtime keeps the last non-NaN value of an output (NaN before the first)."""
    out: list[float] = []
    last = NAN
    for v in values:
        if not math.isnan(v):
            last = v
        out.append(last)
    return out


# ---------------------------------------------------------------------------
# R52 (first red test): RSI_DIV pulses return to 0.0 after the confirming bar
# ---------------------------------------------------------------------------


def test_rsi_div_pulse_resets_after_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the REAL strategy stream (which holds the last non-NaN value): the pulse is 1.0 on
    the confirming bar and 0.0 on the next bar and every bar until the next confirmation."""
    params = {"period": 5, "lb_left": 2, "lb_right": 2, "range_lower": 3, "range_upper": 40}
    n = n_for_two_trims("RSI_DIV", params)
    bars = ohlc_bars(n, "1h", seed=11)
    want = reference("RSI_DIV", params, bars)
    got = stream("RSI_DIV", params, bars, "1h", monkeypatch, trimmed=False)
    events = [t for t, v in enumerate(want["long"]) if v == 1.0]
    assert len(events) >= 2, "fixture must hold two confirmed divergences"
    for t in events:
        assert got["long"][t] == 1.0
        assert got["long"][t + 1] == 0.0, "pulse latched: not reset after the confirming bar"
    lookback = pta_lookback(indicator_registry.get("RSI_DIV"), _merged("RSI_DIV", params))
    for t in range(lookback - 1, n):
        if t not in events:
            assert got["long"][t] == 0.0
    assert all(not math.isnan(v) for v in got["bull"][lookback - 1 :])


# ---------------------------------------------------------------------------
# R50: runtime == replay == reference through trims
# ---------------------------------------------------------------------------

ST_PARAMS = [
    {},
    {"period": 7, "multiplier": 2.0, "src": "close", "atr_method": "sma", "offset": 2},
    {"period": 5, "multiplier": 1.5, "src": "hlc3", "atr_method": "rma_tr"},
]


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("params", ST_PARAMS)
def test_supertrend_runtime_equals_reference_through_two_trims(
    params: dict[str, Any], tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert real_cap("SUPERTREND", params) == 0  # the ratchet is never trimmed
    want = check_stream_replay_reference("SUPERTREND", params, tf, monkeypatch, trimmed=False)
    assert {1.0, -1.0} <= set(want["dir"]), "fixture must flip the trend both ways"


PMAX_CASES = [(k, "1h") for k in PMAX_KINDS] + [("ema", "1m"), ("var", "1m")]


@pytest.mark.parametrize("kind,tf", PMAX_CASES)
def test_pmax_runtime_equals_reference_every_kind(
    kind: str, tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {"kind": kind, "period": 7, "ma_period": 9, "multiplier": 2.0, "offset": 1}
    assert real_cap("PMAX", params) == 0
    want = check_stream_replay_reference("PMAX", params, tf, monkeypatch, trimmed=False)
    assert {1.0, -1.0} <= set(want["dir"]), "fixture must flip the trend both ways"


@pytest.mark.parametrize("tf", TFS)
def test_pmax_defaults_runtime_equals_reference(tf: str, monkeypatch: pytest.MonkeyPatch) -> None:
    check_stream_replay_reference("PMAX", {}, tf, monkeypatch, trimmed=False)


PMAX_RECURSIVE = ("ema", "wwma", "var", "zlema")
PMAX_UNEQUAL = [(k, 2, 20) for k in PMAX_RECURSIVE] + [(k, 20, 2) for k in PMAX_RECURSIVE]


@pytest.mark.parametrize("kind,atr_period,ma_period", PMAX_UNEQUAL)
def test_pmax_ratchets_on_the_raw_ma_not_the_masked_warmup(
    kind: str, atr_period: int, ma_period: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ATR ready long before the MA (and the reverse): the stops recurse on the raw Pine MA from
    its first defined bar, so every post-readiness bar equals the independent full-history reference."""
    params = {"kind": kind, "period": atr_period, "ma_period": ma_period, "multiplier": 1.0}
    check_stream_replay_reference("PMAX", params, "1h", monkeypatch, trimmed=False)


def test_pmax_literal_value_after_a_masked_ema_warmup() -> None:
    """period 2, ma_period 20, ema, multiplier 1, hl2 on ohlc_bars(100, 1h, seed 11): bar 19 is the
    first publicly ready EMA bar. Pine's stop ratchet has run since bar 1 (raw EMA, ATR seeded at
    bar 1) and reads 99.46391101969962 there; seeding it at bar 19 from the masked EMA gave
    97.59207994591077."""
    df = _frame(ohlc_bars(100, "1h", seed=11))
    got = _run("PMAX", df, {"period": 2, "ma_period": 20, "kind": "ema", "multiplier": 1.0})
    assert np.isnan(got["value"][:19]).all()
    assert got["value"][19] == 99.46391101969962


SL_PARAMS = [
    {},
    {
        "lookback": 3,
        "atr_period": 7,
        "atr_method": "rma_tr",
        "target_mode": "atr",
        "target_atr": 1.5,
        "risk_mult": 0.5,
        "offset": 1,
    },
    {"lookback": 1, "rr": 2.0, "atr_period": 5},
]


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("params", SL_PARAMS)
def test_signal_levels_runtime_equals_reference_through_two_trims(
    params: dict[str, Any], tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert real_cap("SIGNAL_LEVELS", params) == 0  # rma ATR is recursive
    check_stream_replay_reference("SIGNAL_LEVELS", params, tf, monkeypatch, trimmed=False)


@pytest.mark.parametrize("tf", TFS)
def test_signal_levels_sma_atr_is_exact_on_a_trimmed_buffer(
    tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``sma`` ATR every output depends on its last ``max(lookback, atr)+1`` bars only."""
    params = {"lookback": 4, "atr_period": 6, "atr_method": "sma", "rr": 1.5}
    check_stream_replay_reference("SIGNAL_LEVELS", params, tf, monkeypatch, trimmed=True)


ICH_PARAMS = [{}, {"conversion_period": 3, "base_period": 5, "span_period": 8, "offset": 2}]


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("params", ICH_PARAMS)
def test_ichimoku_lead_exact_through_trims(
    params: dict[str, Any], tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # finite lookback: the strategy trims it; cap = max(400, 10 x max(periods) + offset)
    lookback = max(
        _merged("ICHIMOKU_LEAD", params)[k]
        for k in ("conversion_period", "base_period", "span_period")
    ) + int(_merged("ICHIMOKU_LEAD", params)["offset"])
    assert real_cap("ICHIMOKU_LEAD", params) == max(400, 10 * lookback)
    assert real_cap("ICHIMOKU_LEAD", params) > 0
    check_stream_replay_reference("ICHIMOKU_LEAD", params, tf, monkeypatch, trimmed=False)
    check_stream_replay_reference("ICHIMOKU_LEAD", params, tf, monkeypatch, trimmed=True)


@pytest.mark.parametrize(
    "params",
    [
        {"conversion_period": 9, "base_period": 26, "span_period": 600},  # span > cap of lead1
        {"conversion_period": 5, "base_period": 700, "span_period": 9, "offset": 3},  # base > span
    ],
)
def test_ichimoku_lead_buffer_covers_the_longest_window(
    params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The buffer cap derives from the lookback; a window longer than the other output's must still
    fit, or the longer output never sees its bars and stays NaN in the runtime."""
    lookback = pta_lookback(
        indicator_registry.get("ICHIMOKU_LEAD"), _merged("ICHIMOKU_LEAD", params)
    )
    assert lookback == max(
        params["conversion_period"], params["base_period"], params["span_period"]
    ) + params.get("offset", 0)
    assert real_cap("ICHIMOKU_LEAD", params) > lookback + 25 * lookback // 100 - 1
    bars = ohlc_bars(lookback + 200, "1h")
    want = reference("ICHIMOKU_LEAD", params, bars)
    got = stream("ICHIMOKU_LEAD", params, bars, "1h", monkeypatch, trimmed=False)
    for out in ("lead1", "lead2"):
        assert math.isfinite(want[out][-1]), out
        assert got[out][-1] == want[out][-1], f"{out} runtime vs full history"


RSI_PARAMS = [
    {},
    {
        "period": 5,
        "lb_left": 2,
        "lb_right": 2,
        "range_lower": 3,
        "range_upper": 40,
        "plot_hidden_bear": 1,
        "src": "hlc3",
        "offset": 1,
    },
]


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize(
    "params",
    [
        {"period": 5, "lb_left": 2, "lb_right": 2, "take_profit_level": 60.0},
        {"period": 4, "take_profit_level": 55.0, "offset": 1},
    ],
)
def test_rsi_div_long_exit_pulse_matches_reference_through_two_trims(
    params: dict[str, Any], tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-13: ``long_exit`` = crossover(rsi, take_profit_level) OR bear, as a 1.0/0.0 pulse.

    The R50 driver compares every output of the spec, so this pins the new pulse against the
    independent Pine-series reference (which transcribes l.204) on the real strategy stream and
    the vector replay, through both buffer regimes.
    """
    assert real_cap("RSI_DIV", params) == 0
    want = check_stream_replay_reference("RSI_DIV", params, tf, monkeypatch, trimmed=False)
    assert sum(want["long_exit"]) >= 1, "the fixture must fire the exit pulse"


@pytest.mark.parametrize("tf", TFS)
@pytest.mark.parametrize("params", RSI_PARAMS)
def test_rsi_div_runtime_equals_reference_through_two_trims(
    params: dict[str, Any], tf: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert real_cap("RSI_DIV", params) == 0
    want = check_stream_replay_reference("RSI_DIV", params, tf, monkeypatch, trimmed=False)
    assert sum(want["bull"]) + sum(want["hidden_bull"]) + sum(want["bear"]) >= 1, "no divergence"


def test_rsi_div_fixture_exercises_every_pulse_kind() -> None:
    """The R50 fixtures must hit each pulse output at least once, or a dead branch would pass."""
    params = {
        "period": 5,
        "lb_left": 2,
        "lb_right": 2,
        "range_lower": 3,
        "range_upper": 40,
        "plot_hidden_bear": 1,
    }
    totals = dict.fromkeys(PULSES, 0.0)
    for seed in range(3, 8):
        want = reference("RSI_DIV", params, ohlc_bars(700, "1h", seed))
        for k in PULSES:
            totals[k] += sum(want[k])
    assert all(v >= 1 for v in totals.values()), totals


# ---------------------------------------------------------------------------
# R51: not ready -> NaN (RSI_DIV pulses: 0.0)
# ---------------------------------------------------------------------------

READY_CASES = [
    ("SUPERTREND", {"period": 4}),
    ("SUPERTREND", {"period": 4, "atr_method": "sma", "offset": 2}),
    ("PMAX", {"period": 3, "ma_period": 6, "kind": "wma"}),
    ("PMAX", {"period": 6, "ma_period": 3, "kind": "var"}),
    ("SIGNAL_LEVELS", {"lookback": 3, "atr_period": 5, "atr_method": "rma_tr", "offset": 1}),
]


@pytest.mark.parametrize("type_,params", READY_CASES)
def test_not_ready_is_nan_and_the_gate_opens_on_the_first_finite_bar(
    type_: str, params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = indicator_registry.get(type_)
    assert spec is not None
    merged = _merged(type_, params)
    lookback = pta_lookback(spec, merged)
    bars = ohlc_bars(lookback + 40, "1h")
    df = _frame(bars)
    below = invoke_compute_fn(spec, df.iloc[: lookback - 1], merged)
    for out in spec.output_names:
        assert below[out].isna().all(), f"{type_}.{out} finite below the lookback"
    at = invoke_compute_fn(spec, df.iloc[:lookback], merged)
    first_out = [o for o in spec.output_names if np.isfinite(at[o].iloc[-1])]
    assert first_out, f"{type_} has no finite output at the lookback"
    snap = stream(type_, params, bars, "1h", monkeypatch, trimmed=False)
    want = reference(type_, params, bars)
    firsts = {}
    for out in spec.output_names:
        firsts[out] = next((i for i, v in enumerate(snap[out]) if not math.isnan(v)), None)
        first_ref = next((i for i, v in enumerate(want[out]) if not math.isnan(v)), None)
        assert firsts[out] == first_ref, f"{type_}.{out} first stored value at {firsts[out]}"
    assert min(i for i in firsts.values() if i is not None) == lookback - 1


@pytest.mark.parametrize(
    "params",
    [
        {"conversion_period": 2, "base_period": 4, "span_period": 6},
        {"conversion_period": 2, "base_period": 3, "span_period": 9, "offset": 1},
        {"conversion_period": 2, "base_period": 7, "span_period": 3},
    ],
)
def test_ichimoku_lead_gate_opens_when_both_windows_exist(
    params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One scalar lookback gates both outputs, so it is the LATEST window (+ offset): nothing is
    stored before it, and from it on both outputs equal the full-history reference."""
    spec = indicator_registry.get("ICHIMOKU_LEAD")
    assert spec is not None
    merged = _merged("ICHIMOKU_LEAD", params)
    lookback = pta_lookback(spec, merged)
    assert lookback == (
        max(params["conversion_period"], params["base_period"], params["span_period"])
        + params.get("offset", 0)
    )
    bars = ohlc_bars(lookback + 40, "1h")
    snap = stream("ICHIMOKU_LEAD", params, bars, "1h", monkeypatch, trimmed=False)
    want = reference("ICHIMOKU_LEAD", params, bars)
    for out in spec.output_names:
        assert all(math.isnan(v) for v in snap[out][: lookback - 1]), out
        assert not math.isnan(snap[out][lookback - 1]), out
        assert_identical(
            snap[out][lookback - 1 :],
            want[out][lookback - 1 :],
            f"ICHIMOKU_LEAD.{out} after the gate",
        )


@pytest.mark.parametrize("offset", [0, 2])
def test_rsi_div_not_ready_rsi_is_nan_and_pulses_are_zero(
    offset: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = {"period": 4, "offset": offset}
    spec = indicator_registry.get("RSI_DIV")
    assert spec is not None
    merged = _merged("RSI_DIV", params)
    lookback = pta_lookback(spec, merged)
    assert lookback == 4 + 1 + offset
    bars = ohlc_bars(lookback + 30, "1h")
    df = _frame(bars)
    below = invoke_compute_fn(spec, df.iloc[: lookback - 1], merged)
    assert below["rsi"].isna().all()
    for name in PULSES:
        assert (below[name] == 0.0).all(), f"{name} must be 0.0, never NaN, while rsi is not ready"
    snap = stream("RSI_DIV", params, bars, "1h", monkeypatch, trimmed=False)
    first_rsi = next(i for i, v in enumerate(snap["rsi"]) if not math.isnan(v))
    assert first_rsi == lookback - 1
    assert all(math.isnan(v) for v in snap["rsi"][: lookback - 1])


# ---------------------------------------------------------------------------
# Literal values (hand-computed)
# ---------------------------------------------------------------------------

# bar:    0   1   2   3  4  5
HIGH = [11.0, 12.0, 13.0, 12.0, 8.0, 14.0]
LOW = [9.0, 10.0, 11.0, 6.0, 6.0, 8.0]
CLOSE = [10.0, 11.0, 12.0, 7.0, 7.0, 13.0]


def _hlc() -> pd.DataFrame:
    return pd.DataFrame(
        {"open": CLOSE, "high": HIGH, "low": LOW, "close": CLOSE, "volume": [1.0] * 6}
    )


def _run(type_: str, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, list[float]]:
    spec = indicator_registry.get(type_)
    assert spec is not None
    out = invoke_compute_fn(spec, df, _merged(type_, params))
    return {k: v.tolist() for k, v in out.items()}


def _same(got: list[float], want: list[float]) -> None:
    assert_identical(got, want, "literal")


# A fall then a rally: RSI rises 0 -> 33.3 -> 55.6 -> 70.4 ... with no pivot divergence at all, so
# the RSI leg is the ONLY way ``long_exit`` can fire here.
RAMP_CLOSE = [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]


def _bars(close: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [1.0] * len(close),
        }
    )


def test_rsi_div_long_exit_fires_on_the_rsi_cross_alone() -> None:
    """Only the crossover(osc, take_profit_level) leg: no bearish pivot exists anywhere here.

    period 3, lb_left/lb_right 1, level 60: rsi = [_,_,_,0,0,0,33.3,55.6,70.4,80.2,86.8,91.2,94.1],
    so the single crossing of 60 is on bar 8 (70.4 vs 55.6). No bear pulse fires on any bar.
    """
    got = _run(
        "RSI_DIV",
        _bars(RAMP_CLOSE),
        {"period": 3, "lb_left": 1, "lb_right": 1, "take_profit_level": 60.0},
    )
    assert got["bear"] == [0.0] * len(RAMP_CLOSE), "this ramp holds no bearish divergence"
    assert got["long_exit"] == [0.0] * 8 + [1.0] + [0.0] * 4
    assert got["rsi"][8] > 60.0 and got["rsi"][7] <= 60.0


def test_rsi_div_long_exit_fires_on_the_bear_leg_alone() -> None:
    """A confirmed bearish divergence with the RSI level out of reach: only the bear leg fires.

    Zig-zag close [10,13,16,12,9,13,17,12,9,14,18,13,10] at period 2, lb 1/1: the pivothigh
    confirmed on bar 11 is a regular bearish divergence (lower RSI high, higher price high).
    With take_profit_level 1000 the RSI can never cross, so ``long_exit`` is exactly ``bear``.
    """
    got = _run(
        "RSI_DIV",
        _bars([10.0, 13.0, 16.0, 12.0, 9.0, 13.0, 17.0, 12.0, 9.0, 14.0, 18.0, 13.0, 10.0]),
        {"period": 2, "lb_left": 1, "lb_right": 1, "take_profit_level": 1000.0, "range_lower": 1},
    )
    assert got["bear"] == [0.0] * 11 + [1.0, 0.0], "the fixture must hold that one divergence"
    assert got["long_exit"] == got["bear"]
    assert all(v < 1000.0 for v in got["rsi"] if v == v), "the level must be unreachable"


def test_supertrend_literal_values() -> None:
    """period 2, multiplier 1, src hl2 = [10,11,12,9,7,11]; tr = [2,2,2,6,2,7].

    rma ATR (seed bar 1): [-,2,2,4,3,5]. Bar 3: up 9-4=5 is ratcheted back to up1=10 (close[1]=12 > 10),
    close 7 < up1 10 -> trend -1, line = dn 13. Bar 5: close 13 > dn1 10 -> trend +1, line = up 6."""
    got = _run("SUPERTREND", _hlc(), {"period": 2, "multiplier": 1.0})
    _same(got["value"], [NAN, 9.0, 10.0, 13.0, 10.0, 6.0])
    _same(got["dir"], [NAN, 1.0, 1.0, -1.0, -1.0, 1.0])
    _same(got["up"], [NAN, 9.0, 10.0, 10.0, 4.0, 6.0])
    _same(got["dn"], [NAN, 13.0, 13.0, 13.0, 10.0, 10.0])
    # sma(tr, 2) is one bar later (bare tr is na on bar 0): [-,-,2,4,4,4.5]
    sma = _run("SUPERTREND", _hlc(), {"period": 2, "multiplier": 1.0, "atr_method": "sma"})
    _same(sma["value"], [NAN, NAN, 10.0, 13.0, 11.0, 6.5])
    _same(sma["dir"], [NAN, NAN, 1.0, -1.0, -1.0, 1.0])
    lagged = _run("SUPERTREND", _hlc(), {"period": 2, "multiplier": 1.0, "offset": 1})
    _same(lagged["dir"], [NAN, NAN, 1.0, 1.0, -1.0, -1.0])


def test_pmax_literal_values() -> None:
    """sma(hl2, 2) = [-,10.5,11.5,10.5,8,9]; atr as above, multiplier 1.

    longStop raw [-,8.5,9.5,6.5,5,4]: bar 3 ratchets to 9.5 (MAvg 10.5 > 9.5), bar 4 MAvg 8 < 9.5 -> dir -1
    and PMax = shortStop 11; bar 5 shortStop raw 14 ratchets down to 11."""
    got = _run("PMAX", _hlc(), {"period": 2, "ma_period": 2, "kind": "sma", "multiplier": 1.0})
    _same(got["value"], [NAN, 8.5, 9.5, 9.5, 11.0, 11.0])
    _same(got["mavg"], [NAN, 10.5, 11.5, 10.5, 8.0, 9.0])
    _same(got["dir"], [NAN, 1.0, 1.0, 1.0, -1.0, -1.0])


def test_signal_levels_literal_values() -> None:
    """lookback 2, atr_period 2 (rma ATR [-,2,2,4,3,5]), risk_mult 1.

    lowest(low,2) = [-,9,10,6,6,6], highest(high,2) = [-,12,13,13,12,14]. Bar 3: long stop 6-4=2,
    risk 7-2=5; rr 2 -> target 7+10=17; short stop 13+4=17, target 7-2*(17-7)=-13."""
    rr = _run("SIGNAL_LEVELS", _hlc(), {"lookback": 2, "atr_period": 2, "rr": 2.0})
    _same(rr["long_stop"], [NAN, 7.0, 8.0, 2.0, 3.0, 1.0])
    _same(rr["long_target"], [NAN, 19.0, 20.0, 17.0, 15.0, 37.0])
    _same(rr["short_stop"], [NAN, 14.0, 15.0, 17.0, 15.0, 19.0])
    _same(rr["short_target"], [NAN, 5.0, 6.0, -13.0, -9.0, 1.0])
    atr = _run(
        "SIGNAL_LEVELS",
        _hlc(),
        {"lookback": 2, "atr_period": 2, "target_mode": "atr", "target_atr": 1.5},
    )
    _same(atr["long_stop"], [NAN, 7.0, 8.0, 2.0, 3.0, 1.0])  # risk_mult 1 default, same stops
    _same(atr["long_target"], [NAN, 14.0, 15.0, 13.0, 11.5, 20.5])
    _same(atr["short_target"], [NAN, 8.0, 9.0, 1.0, 2.5, 5.5])
    # 08: stoploss = low - atr, takeprofit = close + atr, atr = rma(tr, n) (seeded one bar later)
    s08 = _run(
        "SIGNAL_LEVELS",
        _hlc(),
        {"lookback": 1, "atr_period": 2, "atr_method": "rma_tr", "target_mode": "atr"},
    )
    # tr(false) = [-,2,2,6,2,7]; rma seed = sma at bar 2 = 2, bar 3 = (6+2)/2 = 4
    _same(s08["long_stop"], [NAN, NAN, 9.0, 2.0, 3.0, 3.0])  # low 11,6,6,8 minus atr 2,4,3,5
    _same(s08["long_target"], [NAN, NAN, 14.0, 11.0, 10.0, 18.0])  # close 12,7,7,13 plus atr
    _same(s08["short_stop"], [NAN, NAN, 15.0, 16.0, 11.0, 19.0])  # high 13,12,8,14 plus atr


def test_ichimoku_lead_literal_values() -> None:
    """conversion 2, base 3, span 4. donchian(2) = [-,10.5,11.5,9.5,9,10], donchian(3) = [-,-,11,9.5,9.5,10]
    -> lead1 = [-,-,11.25,9.5,9.25,10]; lead2 = donchian(4) = [-,-,-,9.5,9.5,10]."""
    got = _run(
        "ICHIMOKU_LEAD",
        _hlc(),
        {"conversion_period": 2, "base_period": 3, "span_period": 4},
    )
    _same(got["lead1"], [NAN, NAN, 11.25, 9.5, 9.25, 10.0])
    _same(got["lead2"], [NAN, NAN, NAN, 9.5, 9.5, 10.0])
    lagged = _run(
        "ICHIMOKU_LEAD",
        _hlc(),
        {"conversion_period": 2, "base_period": 3, "span_period": 4, "offset": 1},
    )
    _same(lagged["lead1"], [NAN, NAN, NAN, 11.25, 9.5, 9.25])


def test_rsi_div_literal_values() -> None:
    """period 2 RSI of close [10,12,11,12,11,13,12,13,11,12,11.5] = [-,-,66.67,80,44.44,80,48.78,71.23,25.87,54.71,39.39]
    (bar 2: up 1, down 0.5 -> 100-100/3; bar 3: up 1, down 0.25 -> 80; bar 4: up 0.5, down 0.625 -> 44.44).
    Osc pivot lows (left 1, right 1) at bars 4, 6, 8 (confirmed at 5, 7, 9); pivot highs at 3, 5, 7, 9.
    Bar 7: osc 48.78 > 44.44 (higher low) and low 9 < 10 (lower low), bars since previous pivot found
    = 7-1-5 = 1 -> bull. Bar 9: osc 25.87 < 48.78 and low 10 > 9 -> hidden bull. Bar 8: osc 71.23 < 80
    and high 15 > 14 -> bear. Bar 6: osc 80 is not < 80 -> no bear (the tie)."""
    close = [10.0, 12.0, 11.0, 12.0, 11.0, 13.0, 12.0, 13.0, 11.0, 12.0, 11.5]
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    high[7] = 15.0
    low[6] = 9.0
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": [1.0] * 11}
    )
    params = {"period": 2, "lb_left": 1, "lb_right": 1, "range_lower": 1, "range_upper": 10}
    got = _run("RSI_DIV", df, params)
    assert np.isnan(got["rsi"][:2]).all()
    assert got["rsi"][2:9] == pytest.approx(
        [66.6667, 80.0, 44.4444, 80.0, 48.7805, 71.2329, 25.8706], abs=1e-3
    )
    zeros = [0.0] * 11
    _same(got["bull"], [*zeros[:7], 1.0, 0.0, 0.0, 0.0])
    _same(got["hidden_bull"], [*zeros[:9], 1.0, 0.0])
    _same(got["bear"], [*zeros[:8], 1.0, 0.0, 0.0])
    _same(got["hidden_bear"], zeros)
    _same(got["long"], [*zeros[:7], 1.0, 0.0, 1.0, 0.0])
    flagged = _run("RSI_DIV", df, {**params, "plot_bull": 0, "plot_hidden_bull": 0})
    _same(flagged["long"], zeros)
    _same(flagged["bear"], [*zeros[:8], 1.0, 0.0, 0.0])
    narrow = _run("RSI_DIV", df, {**params, "range_lower": 2})
    _same(narrow["long"], zeros)  # bars since previous pivot = 1 is below the lower bound


# ---------------------------------------------------------------------------
# Parameter validation, registry metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "type_,params,match",
    [
        ("SUPERTREND", {"src": "volume"}, "src"),
        ("SUPERTREND", {"offset": -1}, "offset"),
        ("SUPERTREND", {"atr_method": "ema"}, "unknown ATR method"),
        ("SUPERTREND", {"period": 0}, "length >= 1"),
        ("PMAX", {"kind": "hma"}, "kind"),
        ("PMAX", {"kind": "sma", "ma_period": 0}, "length >= 1"),
        ("PMAX", {"src": "vol"}, "src"),
        ("PMAX", {"offset": -3}, "offset"),
        ("SIGNAL_LEVELS", {"lookback": 0}, "lookback"),
        ("SIGNAL_LEVELS", {"target_mode": "x"}, "target_mode"),
        ("SIGNAL_LEVELS", {"offset": -1}, "offset"),
        ("ICHIMOKU_LEAD", {"base_period": 0}, "periods"),
        ("ICHIMOKU_LEAD", {"offset": -1}, "offset"),
        ("RSI_DIV", {"period": 0}, "period"),
        ("RSI_DIV", {"lb_right": 0}, "lb_right"),
        ("RSI_DIV", {"src": "x"}, "src"),
        ("RSI_DIV", {"offset": -1}, "offset"),
    ],
)
def test_bad_params_fail_loudly_at_lookback_time(
    type_: str, params: dict[str, Any], match: str
) -> None:
    spec = indicator_registry.get(type_)
    assert spec is not None
    with pytest.raises(ValueError, match=match):
        pta_lookback(spec, _merged(type_, params))


def test_catalogue_metadata_and_full_history_flags() -> None:
    expected = {
        # name: (placement, full_history, outputs, primary)
        "SUPERTREND": ("overlay", True, ("value", "dir", "up", "dn"), "value"),
        "PMAX": ("overlay", True, ("value", "mavg", "dir"), "value"),
        "SIGNAL_LEVELS": (
            "overlay",
            True,
            ("long_stop", "long_target", "short_stop", "short_target"),
            "long_stop",
        ),
        "ICHIMOKU_LEAD": ("overlay", False, ("lead1", "lead2"), "lead1"),
        "RSI_DIV": (
            "oscillator",
            True,
            ("rsi", "bull", "hidden_bull", "bear", "hidden_bear", "long", "long_exit"),
            "rsi",
        ),
    }
    for name, (where, full, outputs, primary) in expected.items():
        spec = indicator_registry.get(name)
        assert spec is not None, name
        assert spec.chart_placement == where
        assert spec.requires_full_history is full
        assert spec.output_names == outputs
        assert spec.primary_output == primary
        assert spec.threshold_range is None  # R53: GA pools unchanged
        assert spec.ma_kind is False
        assert spec.param_ranges == {}
        assert spec.nt_class is None
        assert "offset" in spec.default_params


def test_plugins_compile_through_the_dsl_and_take_unknown_params_loudly() -> None:
    for type_, params in [
        ("SUPERTREND", {"period": 10, "multiplier": 3}),
        ("PMAX", {"kind": "tsf", "ma_period": 5}),
        ("SIGNAL_LEVELS", {"rr": 2}),
        ("ICHIMOKU_LEAD", {"span_period": 20}),
        ("RSI_DIV", {"lb_right": 2}),
    ]:
        module, cls = compile_indicators({"x": {"type": type_, **params}})
        assert module is not None and cls is not None
    with pytest.raises(DSLValidationError, match="unknown param"):
        compile_indicators({"x": {"type": "SUPERTREND", "bogus": 1}})
