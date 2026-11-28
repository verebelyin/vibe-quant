"""Paper warm-up classification: every registered spec declares one (design.md R2.1).

``IndicatorSpec.paper_warmup`` answers "how many bars does this indicator need
before a paper start's seed series matches a full-history (validation) series
within 1e-9?"; a kind that never converges refuses instead. The first test is
the registry gate (a new indicator must be classified by its author); the R3-R6
and R14/R18 tests pin the class arithmetic at exact edges; the slow test proves
each declared envelope on the real BTC/ETH/SOL 15m catalog.

Tolerance is the pending Q-A measure (design.md R2.7): relative error divided by
``max(|validation|, median |output| over the seed)``, so an oscillator near zero
is not judged on its instantaneous magnitude. Measured horizons are what the
envelopes were fitted to (``out_bounds.txt``, ``out_capped_all.txt``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import derived
from vibe_quant.dsl.indicators import indicator_registry, pta_buffer_cap, pta_lookback
from vibe_quant.dsl.pta_replay import replay_pta_series
from vibe_quant.dsl.warmup_class import (
    CONTEXT_REASON,
    CUMULATIVE_REASON,
    D42M5_REASON,
    SEED_PARITY_REASON,
    WarmupUnsupported,
    by_kind,
    daily,
    finite,
    windowed,
)

BAR_SECONDS = 900  # the 15m bar every bound below is expressed in
SEED_8000 = 8_000
SEED_3000 = 3_000


def _spec(name: str):
    spec = indicator_registry.get(name)
    assert spec is not None, f"{name} is not registered"
    assert getattr(spec, "paper_warmup", None) is not None, f"{name} declares no paper warm-up"
    return spec


def _bound(
    name: str,
    overrides: dict[str, object] | None = None,
    bar_seconds: int = BAR_SECONDS,
) -> int:
    """Declared warm-up bars for ``name`` at default params plus ``overrides``."""
    spec = _spec(name)
    assert spec.paper_warmup is not None
    params = {**spec.default_params, **(overrides or {})}
    return spec.paper_warmup(params, bar_seconds)


def _refusal(name: str, overrides: dict[str, object] | None = None) -> str:
    """The ``WarmupUnsupported`` reason ``name`` raises for ``overrides``."""
    spec = _spec(name)
    assert spec.paper_warmup is not None
    params = {**spec.default_params, **(overrides or {})}
    with pytest.raises(WarmupUnsupported) as excinfo:
        spec.paper_warmup(params, BAR_SECONDS)
    return str(excinfo.value)


def _unclassified(specs) -> list[str]:
    """Names of specs whose ``paper_warmup`` is ``None`` (the registry gate's message)."""
    return [spec.name for spec in specs if getattr(spec, "paper_warmup", None) is None]


def test_every_indicator_has_paper_warmup() -> None:
    """A new indicator must declare its class; an unclassified one is named here."""
    missing = _unclassified(indicator_registry.all_specs())
    assert missing == [], f"unclassified paper warm-up (add one in the spec): {missing}"


def test_registry_gate_names_an_unclassified_spec() -> None:
    """R6: the gate above fails naming the spec, not with a bare assertion."""
    unclassified = indicator_registry.get("OBV")
    assert unclassified is not None
    from dataclasses import replace

    fake = replace(unclassified, name="UNCLASSIFIED_PROBE", paper_warmup=None)
    assert _unclassified([fake, _spec("RSI")]) == ["UNCLASSIFIED_PROBE"]


def test_obv_refuses_the_start_r3() -> None:
    """R3 (class side): the cumulative level never converges, so paper refuses it."""
    assert _refusal("OBV") == "cumulative level never converges"
    assert CUMULATIVE_REASON == "cumulative level never converges"


def test_ema_600_exceeds_the_seed_and_ema_200_fits_r4() -> None:
    """R4: ``ema`` is 15 x period, so EMA(600) needs 9,000 > 8,000 and EMA(200) 3,000."""
    assert _bound("EMA", {"period": 600}) == 9000
    assert _bound("EMA", {"period": 200}) == 3000
    assert 9000 > SEED_8000 >= 3000


def test_stale_phase_kinds_refuse_r5() -> None:
    """R5: VIDYA/ADAPTIVE_RSI compute from the catalog origin in backtests, which a paper
    seed cannot be shown to reach; SQUEEZE_RATIO drifts in validation itself (d42m5)."""
    for name, reason_text in (
        ("VIDYA", SEED_PARITY_REASON),
        ("ADAPTIVE_RSI", SEED_PARITY_REASON),
        ("SQUEEZE_RATIO", D42M5_REASON),
    ):
        # Two unrelated inputs: the same refusal, whatever the params.
        assert _refusal(name) == reason_text, name
        assert _refusal(name, {"period": 50}) == reason_text, name
    assert D42M5_REASON == "start-phase drift above 1e-9 in validation itself (vibe-quant-d42m5)"
    assert SEED_PARITY_REASON == (
        "paper's seed cannot be shown to reach the backtest's catalog-origin value "
        "(vibe-quant-d42m5; self-check vibe-quant-jriky)"
    )


def test_context_indicators_refuse_the_start() -> None:
    """``needs_context`` kinds never reach a live node (node.py rejects them first)."""
    for name in (
        "FUNDING",
        "FUNDING_Z",
        "TAKER_IMBALANCE",
        "BTC_TREND",
        "BTC_ROC",
        "OWN_TREND",
        "OI_CHANGE",
        "LS_RATIO_Z",
        "DVOL_Z",
    ):
        assert _refusal(name) == CONTEXT_REASON, name


def test_seed_of_three_thousand_rows_r14() -> None:
    """R14 bound part: EMA(200) fits 3,000 rows; EMA(250) needs 3,750 and is refused."""
    assert _bound("EMA", {"period": 200}) == 3000
    assert _bound("EMA", {"period": 250}) == 3750
    assert 3000 <= SEED_3000 < 3750


def test_sizing_atr_bound_is_420_r18() -> None:
    """R18 bound part: the sizing ATR(14) is Wilder, i.e. 30 x 14 = 420 bars."""
    assert _bound("ATR", {"period": 14}) == 420


def test_daily_class_is_one_day_of_bars() -> None:
    """VWAP resets at UTC midnight: 96 fifteen-minute bars plus the boundary bar."""
    assert _bound("VWAP") == 97
    assert daily()({}, 3600) == 25
    assert daily()({}, 4 * 3600) == 7
    with pytest.raises(ValueError, match="must be positive"):
        daily()({}, 0)


def test_windowed_class_is_trim_cycle_plus_lookback() -> None:
    """The capped path needs one full trim cycle (cap + cap//4) plus the lookback."""
    # MACD lookback = slow + signal = 35, cap = max(400, 10 x 35) = 400.
    assert pta_lookback(_spec("MACD"), {"slow_period": 26, "signal_period": 9}) == 35
    assert _bound("MACD") == 535
    # Unrelated input: slow 50 + signal 13 = 63, cap = 630.
    assert _bound("MACD", {"fast_period": 21, "slow_period": 50, "signal_period": 13}) == 850
    assert windowed(lambda p: 35)({}, BAR_SECONDS) == 535


def test_adx_timeframe_of_the_buffer_decides() -> None:
    """ADX at 400/16 passes, 400/17 does not: the cap stops covering Wilder x Wilder."""
    assert _bound("ADX", {"period": 14}) == 514
    assert _bound("ADX", {"period": 16}) == 516
    with pytest.raises(WarmupUnsupported) as excinfo:
        _spec("ADX").paper_warmup({**_spec("ADX").default_params, "period": 17}, BAR_SECONDS)  # type: ignore[misc]
    assert str(excinfo.value) == f"{SEED_PARITY_REASON}: cap 400 < 25 x lookback 17"
    assert _refusal("ADX", {"period": 30}) == f"{SEED_PARITY_REASON}: cap 400 < 25 x lookback 30"


def test_kind_dispatch_follows_the_smoothing_family() -> None:
    """A kind param swaps the kernel, so the bound must follow it (finite/EMA/Wilder/adaptive)."""
    assert _bound("MA", {"kind": "sma", "period": 100}) == 200
    assert _bound("MA", {"kind": "ema", "period": 100}) == 1500
    assert _bound("MA", {"kind": "wwma", "period": 100}) == 3000
    assert _bound("MA", {"kind": "var", "period": 100}) == 6000
    # HULL swaps kernel by ``mode``: ehma is an EMA recursion, the others are finite.
    assert _bound("HULL", {"mode": "hma", "period": 200}) == 400
    assert _bound("HULL", {"mode": "ehma", "period": 200}) == 3000
    # PMAX kind=var is adaptive over the longest of period/ma_period.
    assert _bound("PMAX", {"kind": "ema", "period": 50, "ma_period": 50}) == 1500
    assert _bound("PMAX", {"kind": "var", "period": 50, "ma_period": 50}) == 3000
    # MA_MACD follows the kind on max(fast, slow) + signal (the signal cascades on the legs).
    macd = {"fast_period": 50, "slow_period": 100, "signal_period": 20}
    assert _bound("MA_MACD", {**macd, "kind": "sma"}) == 240
    assert _bound("MA_MACD", {**macd, "kind": "ema"}) == 1800
    assert _bound("MA_MACD", {**macd, "kind": "var"}) == 7200
    # fast above slow and a long signal both widen it (the reviewer's under-counted cases).
    assert _bound("MA_MACD", {"kind": "sma", "fast_period": 500}) == 2 * (500 + 9)
    assert _bound("MA_MACD", {"kind": "sma", "signal_period": 100}) == 2 * (26 + 100)
    # RSI_DIV: adaptive RSI recursion plus lb_left + lb_right + range_upper of pivot history.
    assert _bound("RSI_DIV") == 60 * 9 + 1 + 3 + 60
    assert _bound("RSI_DIV", {"period": 2, "lb_right": 200}) == 60 * 2 + 1 + 200 + 60
    with pytest.raises(WarmupUnsupported, match="unknown kind"):
        by_kind({"sma": finite()})({"kind": "nope"}, BAR_SECONDS)


# ---------------------------------------------------------------------------
# Slow bound proof: the declared envelope must cover the measured 1e-9 horizon
# (Q-A scaled) on the real BTC/ETH/SOL 15m catalog, with 0 signal flips after it.
# ---------------------------------------------------------------------------

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
_STARTS = (50_000, 80_000)
_HORIZON_MARGIN = 200  # bars past the declared bound used to watch for signal flips


def _arrays() -> dict[str, dict[str, np.ndarray]]:
    from vibe_quant.screening.nt_runner import _pta_bar_arrays

    return {
        sym: _pta_bar_arrays(
            "data/catalog",
            f"{sym}-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
            "2023-01-01",
            "2026-12-31",
        )
        for sym in _SYMBOLS
    }


def _nt_kwargs(spec, params: dict[str, object]) -> dict[str, object]:
    """Constructor kwargs exactly as the compiler emits them (no bare ``period`` guess)."""
    kwargs: dict[str, object] = {}
    for nt_kwarg, dsl_field in spec.nt_codegen_kwargs:
        # STOCH's DSL fields are ``period``/``d_period`` while its spec defaults
        # use the ``period_k``/``period_d`` aliases compute_fn also accepts.
        if dsl_field in params:
            kwargs[nt_kwarg] = params[dsl_field]
        else:
            kwargs[nt_kwarg] = params[nt_kwarg]
    for nt_kwarg, _helper, dsl_field in spec.nt_codegen_helper_kwargs:
        kwargs[nt_kwarg] = getattr(derived, _helper)(params[dsl_field])
    return kwargs


def _nt_series(
    spec, frame: pd.DataFrame, params: dict[str, object], ts_ns
) -> dict[str, np.ndarray]:
    """Feed the NT indicator bar by bar and read the outputs the compiler reads."""
    ind = spec.nt_class(**_nt_kwargs(spec, params))
    n = len(frame)
    out = {name: np.full(n, np.nan) for name in spec.output_names}
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    volume = frame["volume"].to_numpy()
    open_ = frame["open"].to_numpy()
    from datetime import UTC, datetime

    cls = spec.nt_class.__name__
    for i in range(n):
        if cls == "OnBalanceVolume":
            ind.update_raw(float(open_[i]), float(close[i]), float(volume[i]))
        elif cls == "VolumeWeightedAveragePrice":
            stamp = datetime.fromtimestamp(int(ts_ns[i]) / 1e9, tz=UTC)
            ind.update_raw(float(close[i]), float(volume[i]), stamp)
        elif cls == "DonchianChannel":
            ind.update_raw(float(high[i]), float(low[i]))
        elif cls in ("CloseBollingerBands", "CloseKeltnerChannel", "SmaDStochastics") or (
            spec.requires_high_low
        ):
            ind.update_raw(float(high[i]), float(low[i]), float(close[i]))
        else:
            ind.update_raw(float(close[i]))
        if not ind.initialized:
            continue
        if spec.primary_helper:
            out[spec.output_names[0]][i] = getattr(derived, spec.primary_helper)(
                ind, float(close[i]), BAR_SECONDS // 60
            )
            continue
        for name in spec.output_names:
            if name in spec.computed_outputs:
                out[name][i] = getattr(derived, spec.computed_outputs[name])(ind, float(close[i]))
            elif name in spec.nt_output_attrs:
                value = getattr(ind, spec.nt_output_attrs[name])
                out[name][i] = float(value) * spec.nt_output_scale.get(name, 1.0)
    return out


def _series(spec, frame: pd.DataFrame, params, ts_ns) -> dict[str, np.ndarray]:
    """The runtime's own path over ``frame``: capped replay, or the NT indicator."""
    if spec.nt_class is not None:
        return _nt_series(spec, frame, params, ts_ns)
    cap = pta_buffer_cap([pta_lookback(spec, params)], full_history=spec.requires_full_history)
    return replay_pta_series(spec, frame, params, cap)


def _measure(spec, params: dict[str, object], arrays, bound: int) -> tuple[int, int]:
    """(last bar whose scaled error > 1e-9, sign-vs-close flips after it) over all starts.

    The reference runs ``bound`` bars ahead of the cold start (warm by construction);
    the window ends ``_HORIZON_MARGIN`` bars past the declared bound so a too-small
    bound shows up as ``horizon > bound`` and a late flip is counted.
    """
    horizon = bound + _HORIZON_MARGIN
    worst = 0
    flips = 0
    uncovered: list[str] = []
    for symbol, arrays_by_symbol in arrays.items():
        evaluated = 0
        finite_seen = False
        frame = pd.DataFrame(
            {k: arrays_by_symbol[k] for k in ("open", "high", "low", "close", "volume")}
        )
        close = arrays_by_symbol["close"]
        ts_ns = arrays_by_symbol["ts_init"]
        n = len(frame)
        for start in _STARTS:
            if start + horizon > n:
                continue
            lead = max(bound, 1_000)
            lo = max(0, start - lead)
            ref = _series(
                spec,
                frame.iloc[lo : start + horizon].reset_index(drop=True),
                params,
                ts_ns[lo : start + horizon],
            )
            cold = _series(
                spec,
                frame.iloc[start : start + horizon].reset_index(drop=True),
                params,
                ts_ns[start : start + horizon],
            )
            offset = start - lo
            evaluated += 1
            horizon_bad = 0
            for name, full in ref.items():
                full = full[offset:]
                cold_values = cold[name]
                both = ~np.isnan(full) & ~np.isnan(cold_values)
                finite_seen = finite_seen or bool(both.any())
                floor = float(np.median(np.abs(full[both]))) if both.any() else 0.0
                denom = np.maximum(np.abs(full), floor if floor > 0 else 1e-300)
                rel = np.zeros(len(cold_values))
                rel[both] = np.abs(full[both] - cold_values[both]) / denom[both]
                bad = (rel > 1e-9) | (np.isnan(full) != np.isnan(cold_values))
                idx = np.flatnonzero(bad)
                horizon_bad = max(horizon_bad, int(idx[-1]) + 1 if idx.size else 0)
            post = np.arange(horizon) >= horizon_bad
            for name, full in ref.items():
                full = full[offset:]
                cold_values = cold[name]
                both = ~np.isnan(full) & ~np.isnan(cold_values)
                flips += int(
                    (
                        (
                            np.sign(close[start : start + horizon] - full)
                            != np.sign(close[start : start + horizon] - cold_values)
                        )
                        & both
                        & post
                    ).sum()
                )
            worst = max(worst, horizon_bad)
        if not evaluated or not finite_seen:
            uncovered.append(
                f"{symbol} (starts evaluated={evaluated}, finite outputs={finite_seen})"
            )
    # A start past the end of the catalog is skipped; zero samples must never read as "proven".
    assert not uncovered, (
        f"data coverage: no evaluated start or no finite output for {uncovered}; "
        f"need rows >= start + {horizon} for one of {_STARTS}"
    )
    return worst, flips


def _cases() -> list[tuple[str, dict[str, object]]]:
    """Default params, top of ``param_ranges``, plus the documented large-period cases."""
    cases: list[tuple[str, dict[str, object]]] = []
    for spec in indicator_registry.all_specs():
        cases.append((spec.name, {}))
        top: dict[str, object] = {}
        for key, (_lo, hi) in spec.param_ranges.items():
            if spec.param_schema.get(key) is int:
                top[key] = int(hi)
        if top:
            cases.append((spec.name, top))
    cases += [
        ("EMA", {"period": 200}),
        ("EMA", {"period": 400}),
        ("DEMA", {"period": 100}),
        ("ADX", {"period": 16}),
        ("SUPERTREND", {"period": 30}),
        ("SIGNAL_LEVELS", {"atr_period": 50}),
        ("RSI_DIV", {"period": 30}),
        ("TEMA", {"period": 50}),
        ("KAMA", {"period": 50}),
        ("FRAMA", {"period": 50}),
        ("VOLSMA", {"period": 50}),
        ("MACD", {"fast_period": 21, "slow_period": 50, "signal_period": 13}),
        ("MA", {"kind": "hma", "period": 200}),
        ("MA", {"kind": "thma", "period": 200}),
        ("MA", {"kind": "ema", "period": 200}),
        ("MA", {"kind": "ehma", "period": 200}),
        ("MA", {"kind": "dema", "period": 200}),
        ("MA", {"kind": "tema", "period": 200}),
        ("MA", {"kind": "t3", "period": 200}),
        ("MA", {"kind": "zlema", "period": 200}),
        ("MA", {"kind": "rma", "period": 200}),
        ("MA", {"kind": "wwma", "period": 200}),
        ("MA", {"kind": "var", "period": 200}),
        ("MA_MACD", {"kind": "sma", "fast_period": 50, "slow_period": 100, "signal_period": 20}),
        ("MA_MACD", {"kind": "ema", "fast_period": 50, "slow_period": 100, "signal_period": 20}),
        ("MA_MACD", {"kind": "var", "fast_period": 50, "slow_period": 100, "signal_period": 20}),
        ("MA_MACD", {"kind": "wwma", "fast_period": 50, "slow_period": 100, "signal_period": 20}),
        # Cascade: the signal MA smooths a series still converging (signal > slow, fast > slow).
        ("MA_MACD", {"signal_period": 100}),
        ("MA_MACD", {"kind": "ema", "signal_period": 100}),
        ("MA_MACD", {"fast_period": 500}),
        ("MA_MACD", {"kind": "ema", "fast_period": 390, "slow_period": 100, "signal_period": 30}),
        ("MA_MACD", {"kind": "wwma", "signal_period": 100}),
        ("MA_MACD", {"kind": "var", "signal_period": 60}),
        ("MA_MACD", {"kind": "hma", "signal_period": 100}),
        ("MA_MACD", {"kind": "ema", "fast_period": 52, "slow_period": 252, "signal_period": 9}),
        # Pivot history: lb_right/range_upper/lb_left independent of the RSI period.
        ("RSI_DIV", {"period": 2, "lb_right": 200}),
        ("RSI_DIV", {"period": 9, "lb_right": 1000}),
        ("RSI_DIV", {"period": 2, "lb_left": 300}),
        ("RSI_DIV", {"period": 2, "range_upper": 500}),
        ("RSI_DIV", {"period": 2, "lb_right": 120, "range_upper": 300, "range_lower": 100}),
        ("PMAX", {"kind": "var", "period": 50, "ma_period": 50}),
        ("PMAX", {"kind": "wwma", "period": 50, "ma_period": 50}),
        ("HULL", {"mode": "ehma", "period": 200}),
        ("HULL", {"mode": "thma", "period": 200}),
    ]
    return cases


def test_short_catalog_fails_instead_of_measuring_nothing() -> None:
    """A catalog too short for every start must fail with a coverage message, not (0, 0)."""
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(size=1_000))
    short = {
        sym: {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(1_000),
            "ts_init": np.arange(1_000, dtype=np.int64) * 900_000_000_000,
        }
        for sym in _SYMBOLS
    }
    spec = _spec("EMA")
    with pytest.raises(AssertionError, match="data coverage"):
        _measure(spec, {**spec.default_params, "period": 600}, short, 1)


@pytest.mark.slow(
    reason="real BTC/ETH/SOL 15m catalog: 3 symbols x 2 starts per classification case"
)
def test_paper_warmup_bounds_hold() -> None:
    """Every declared bound covers the measured 1e-9 horizon, with 0 signal flips after it."""
    arrays = _arrays()
    failures: list[str] = []
    for name, overrides in _cases():
        spec = _spec(name)
        assert spec.paper_warmup is not None
        params = {**spec.default_params, **overrides}
        try:
            bound = int(spec.paper_warmup(params, BAR_SECONDS))
        except WarmupUnsupported as exc:
            # A refused kind is proven by its reason (R3/R5/ADX/cumulative/context).
            assert str(exc), f"{name} {overrides}: refused without a reason"
            continue
        horizon, flips = _measure(spec, params, arrays, bound)
        if horizon > bound or flips:
            failures.append(f"{name} {overrides}: bound={bound} horizon={horizon} flips={flips}")
    assert failures == [], "\n".join(failures)
