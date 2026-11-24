"""Catalog origin (vibe-quant-d42m5, DSL layer): ADX(period > 16), VIDYA and ADAPTIVE_RSI.

A backtest computes these compute_fn recursions from the bar type's first catalog bar,
so the run start never changes their value at a bar T. This file pins the DSL side:
the spec flag and its effective-params predicate (user decision Q1 = B: ADX(<=16) keeps
main's capped path), the generated ``_pta_origin_tfs``, the shared origin rule
(``origin_offset`` / ``install_origin``) against the generated live per-bar path, the
VIDYA zero-denominator hold, and paper admission (unchanged from main apart from the
reason text). The runners wire the origin in a later task.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import indicator_registry, pta_origin_tfs, uses_catalog_origin
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.plugins.vidya import _vidya_library, _vidya_port, compute_vidya
from vibe_quant.dsl.pta_replay import (
    PtaOriginError,
    install_origin,
    origin_offset,
    replay_pta_series,
    strategy_pta_columns,
)
from vibe_quant.dsl.warmup_class import D42M5_REASON, SEED_PARITY_REASON
from vibe_quant.paper.warmup import WarmupNeed, WarmupRefused, check_warmup, indicator_needs

REPO = Path(__file__).resolve().parents[2]
IID = "BTCUSDT-PERP.BINANCE"
OHLCV = ("open", "high", "low", "close", "volume")


def _dsl(
    indicators: dict[str, dict[str, Any]], timeframe: str = "1h", extra: list[str] | None = None
) -> Any:
    first = next(iter(indicators))
    return validate_strategy_dict(
        {
            "name": "origin_probe",
            "timeframe": timeframe,
            "additional_timeframes": extra or [],
            "indicators": indicators,
            "entry_conditions": {"long": [f"{first} > 0"]},
            "exit_conditions": {"long": [f"{first} < 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
    )


def _strategy(
    indicators: dict[str, dict[str, Any]],
    timeframe: str = "1h",
    extra: list[str] | None = None,
    **overrides: Any,
) -> Any:
    """A real generated strategy instance (its ``__init__`` reads ``overrides`` from config)."""
    module = StrategyCompiler().compile_to_module(_dsl(indicators, timeframe, extra))
    return module.OriginProbeStrategy(
        config=module.OriginProbeConfig(instrument_id=IID, **overrides)
    )


def _frame(close: np.ndarray, spread: float = 0.0, seed: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    noise = np.abs(rng.randn(len(close))) * spread
    return pd.DataFrame(
        {
            "open": np.r_[close[:1], close[:-1]],
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.ones(len(close)),
        }
    )


def _arrays(frame: pd.DataFrame, t0: int = 1_600_000_000_000_000_000) -> dict[str, np.ndarray]:
    out = {name: frame[name].to_numpy(dtype=np.float64) for name in OHLCV}
    out["ts_init"] = t0 + np.arange(len(frame), dtype=np.int64) * 3_600_000_000_000
    return out


def _slice(arrays: dict[str, np.ndarray], start: int, stop: int | None = None) -> dict[str, np.ndarray]:
    return {key: value[start:stop] for key, value in arrays.items()}


def _feed(strategy: Any, tf: str, arrays: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Push bars through the generated live per-bar path; ``_pta_values`` after each."""
    out: list[dict[str, float]] = []
    for i in range(len(arrays["close"])):
        bar = SimpleNamespace(
            **{name: float(arrays[name][i]) for name in OHLCV}, ts_init=int(arrays["ts_init"][i])
        )
        strategy._feed_pta_buffer(tf, bar)
        out.append(dict(strategy._pta_values))
    return out


def _bits(value: float) -> int:
    return int(np.float64(value).view(np.int64))


# --------------------------------------------------------------------------- the flag (R2)


def test_catalog_origin_flags() -> None:
    """VIDYA and ADAPTIVE_RSI always; ADX only past period 16 (Q1 = B); nothing else."""
    for params in ({"period": 5}, {"period": 50}):
        assert uses_catalog_origin("VIDYA", params) is True
    assert uses_catalog_origin("ADAPTIVE_RSI", {"period": 14, "alpha": 0.7}) is True
    assert uses_catalog_origin("ADAPTIVE_RSI", {"period": 30, "alpha": 0.9}) is True
    assert uses_catalog_origin("ADX", {"period": 12}) is False
    assert uses_catalog_origin("ADX", {"period": 16}) is False  # equality edge: main's path
    assert uses_catalog_origin("ADX", {"period": 17}) is True
    assert uses_catalog_origin("ADX", {"period": 30}) is True
    assert uses_catalog_origin("ADX", {}) is False  # default 14
    for name in ("KAMA", "RSI", "PRICE_POSITION", "SQUEEZE_RATIO", "OBV"):
        spec = indicator_registry.get(name)
        assert spec is not None
        assert uses_catalog_origin(spec, dict(spec.default_params)) is False, name
    assert uses_catalog_origin("NO_SUCH_KIND", {}) is False
    # requires_full_history must stay off: it would change paper's compiled caps.
    for name in ("ADX", "VIDYA", "ADAPTIVE_RSI"):
        spec = indicator_registry.get(name)
        assert spec is not None and spec.requires_full_history is False, name


def test_origin_timeframes_follow_effective_params_without_recompile() -> None:
    """A 239-like 4h strategy (ADX(12) + KAMA + PRICE_POSITION) has no origin timeframe;
    a config override of the ADX period flips it at the 16/17 edge, same compiled class."""
    module = StrategyCompiler().compile_to_module(
        _dsl(
            {
                "adx": {"type": "ADX", "period": 12},
                "kama": {"type": "KAMA", "period": 25},
                "pp": {"type": "PRICE_POSITION", "period": 133},
            },
            "4h",
        )
    )
    cls, cfg = module.OriginProbeStrategy, module.OriginProbeConfig
    assert cls(config=cfg(instrument_id=IID))._pta_origin_tfs == ()
    assert cls(config=cfg(instrument_id=IID, adx_period=16))._pta_origin_tfs == ()
    assert cls(config=cfg(instrument_id=IID, adx_period=17))._pta_origin_tfs == ("4h",)
    assert cls(config=cfg(instrument_id=IID, adx_period=20))._pta_origin_tfs == ("4h",)
    assert cls._PTA_KINDS == {
        "adx": ("ADX", "4h"),
        "kama": ("KAMA", "4h"),
        "pp": ("PRICE_POSITION", "4h"),
    }


def test_origin_timeframes_of_mixed_layouts() -> None:
    """Each origin kind names its own timeframe; a neighbour-only timeframe stays out."""
    mixed = _strategy(
        {
            "arsi": {"type": "ADAPTIVE_RSI", "period": 14, "alpha": 0.7},
            "slow": {"type": "VIDYA", "period": 20, "timeframe": "4h"},
            "kama": {"type": "KAMA", "period": 25, "timeframe": "15m"},
        },
        "1h",
        ["4h", "15m"],
    )
    assert mixed._pta_origin_tfs == ("1h", "4h")
    only_4h = _strategy(
        {"kama": {"type": "KAMA", "period": 25}, "adx": {"type": "ADX", "period": 30, "timeframe": "4h"}},
        "1h",
        ["4h"],
    )
    assert only_4h._pta_origin_tfs == ("4h",)
    assert pta_origin_tfs({"a": {"period": 30}}, {"a": ("ADX", "1d")}) == ("1d",)
    assert pta_origin_tfs({"a": {"period": 8}}, {"a": ("ADX", "1d")}) == ()


def test_strategies_without_origin_kinds_compile_without_origin_code() -> None:
    """Byte-identity of the goldens is pinned by test_compiler_golden_source; here: no trace."""
    source = StrategyCompiler().compile(_dsl({"kama": {"type": "KAMA", "period": 25}}))
    assert "_PTA_KINDS" not in source and "pta_origin_tfs" not in source
    kama_only = _strategy({"kama": {"type": "KAMA", "period": 25}})
    assert not hasattr(type(kama_only), "_PTA_KINDS") and not hasattr(kama_only, "_pta_origin_tfs")
    adx = _strategy({"adx": {"type": "ADX", "period": 12}})
    assert adx._pta_origin_tfs == ()
    assert _strategy({"adx": {"type": "ADX", "period": 12}}, adx_period=17)._pta_origin_tfs == ("1h",)


# ------------------------------------------------------------------------ origin_offset (R5)


def test_origin_offset_finds_the_run_suffix() -> None:
    hist = _arrays(_frame(100.0 + np.arange(50.0)))
    assert origin_offset(hist, _slice(hist, 17)) == 17
    assert origin_offset(hist, _slice(hist, 0)) == 0
    assert origin_offset(hist, _slice(hist, 49)) == 49


@pytest.mark.parametrize(
    "case", ["close_changed", "ts_changed", "run_longer", "run_ends_early", "start_missing", "empty"]
)
def test_origin_offset_fails_closed(case: str) -> None:
    """Any difference between ``hist[k:]`` and the run fails, never guesses."""
    hist = _arrays(_frame(100.0 + np.arange(50.0)))
    run = _slice(hist, 20)
    run = {key: value.copy() for key, value in run.items()}
    if case == "close_changed":
        run["close"][5] = np.nextafter(run["close"][5], np.inf)
    elif case == "ts_changed":
        run["ts_init"][3] += 1
    elif case == "run_longer":
        run = {key: np.r_[value, value[-1:]] for key, value in run.items()}
    elif case == "run_ends_early":
        run = _slice(run, 0, -1)
    elif case == "start_missing":
        run["ts_init"] = run["ts_init"] + 1_800_000_000_000
    else:
        run = _slice(run, 0, 0)
    with pytest.raises(PtaOriginError):
        origin_offset(hist, run)


# ------------------------------------------------------- install_origin vs the live path (R5)

_LIVE_CASES = [
    ("ADX", {"period": 30}),
    ("VIDYA", {"period": 14}),
    ("ADAPTIVE_RSI", {"period": 14, "alpha": 0.7}),
]


def _flat_to_moving(seed: int = 9) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    walk = 100.0 + np.cumsum(rng.randn(260) * 0.6)
    close = np.r_[walk, np.full(40, walk[-1]), walk[-1] + np.sin(np.arange(120))]
    frame = _frame(close, spread=0.3, seed=seed)
    flat = slice(260, 300)
    for name in ("open", "high", "low"):
        frame.loc[frame.index[flat], name] = walk[-1]
    return frame


def _expected_columns(strategy: Any, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    meta = {name: kind for name, kind in type(strategy)._PTA_KINDS.items()}
    return dict(strategy_pta_columns(strategy._pta_params, meta, {"1h": 0}, "1h", frame))


@pytest.mark.parametrize(("kind", "params"), _LIVE_CASES)
@pytest.mark.parametrize("k", [120, 280])  # run starts in the walk / inside the flat stretch
def test_origin_prefix_and_live_path(kind: str, params: dict[str, object], k: int) -> None:
    """After ``install_origin`` the generated per-bar path equals the cap-0 replay over the
    origin history at every run bar, for the origin kind AND its KAMA neighbour, from the
    first engine bar on (ready at bar 0); later bars never change an earlier value."""
    frame = _flat_to_moving()
    hist = _arrays(frame)
    indicators = {"x": {"type": kind, **params}, "kama": {"type": "KAMA", "period": 25}}
    strategy = _strategy(indicators)
    assert strategy._pta_origin_tfs == ("1h",)
    assert strategy._pta_buffer_cap["1h"] > 0  # main's cap until the origin is installed
    install_origin(strategy, "1h", hist, k)
    assert strategy._pta_buffer_cap["1h"] == 0
    assert len(strategy._pta_bufs["1h"]) == k
    assert strategy._pta_origin["1h"][1] == k
    got = _feed(strategy, "1h", _slice(hist, k))

    expected = _expected_columns(strategy, frame)
    assert set(expected) == {"x", "kama"}
    assert "x" in got[0] and "kama" in got[0]  # ready at the first engine bar
    for key, column in expected.items():
        for j, snapshot in enumerate(got):
            want = column[k + j]
            assert not np.isnan(want), (key, j)
            assert _bits(snapshot[key]) == _bits(want), (kind, key, k, j)

    # Truncated / mutated future: the value at T only reads bars through T.
    t = 25
    mutated = {key: value.copy() for key, value in _slice(hist, k).items()}
    for name in ("open", "high", "low", "close"):
        mutated[name][t + 1 :] *= 1.5
    again = _strategy(indicators)
    install_origin(again, "1h", hist, k)
    replay_t = _feed(again, "1h", mutated)
    assert [_bits(v["x"]) for v in replay_t[: t + 1]] == [_bits(v["x"]) for v in got[: t + 1]]


def test_install_origin_holds_the_prefix_value_through_a_nan_bar() -> None:
    """The prefill seeds the runtime's NaN-hold: a run whose first compute is NaN keeps the
    value the origin history held (as the precompute's held column would)."""
    frame = _flat_to_moving()
    hist = _arrays(frame)
    strategy = _strategy({"x": {"type": "ADX", "period": 30}})
    install_origin(strategy, "1h", hist, 0)
    assert strategy._pta_values == {}  # nothing before the first catalog bar
    strategy = _strategy({"x": {"type": "ADX", "period": 30}})
    install_origin(strategy, "1h", hist, 200)
    spec = indicator_registry.get("ADX")
    assert spec is not None
    held = replay_pta_series(spec, frame.iloc[:200], {"period": 30}, 0)["value"][-1]
    assert _bits(strategy._pta_values["x"]) == _bits(held)


def test_install_origin_refuses_a_buffer_that_already_holds_bars() -> None:
    strategy = _strategy({"x": {"type": "VIDYA", "period": 14}})
    hist = _arrays(_flat_to_moving())
    _feed(strategy, "1h", _slice(hist, 0, 3))
    with pytest.raises(PtaOriginError, match="after 3 bars"):
        install_origin(strategy, "1h", hist, 10)


# ------------------------------------------------------------------- stress cases (R4)


def _stress_cases() -> list[tuple[str, str, dict[str, object], pd.DataFrame]]:
    rng = np.random.RandomState(11)
    walk = 100.0 + np.cumsum(rng.randn(3000) * 0.5)
    return [
        (
            "alt8201",
            "VIDYA",
            {"period": 106},
            _frame(np.r_[50.0, np.tile([100.0, 101.0], 4100)]),
        ),
        (
            "flat_suffix",
            "ADAPTIVE_RSI",
            {"period": 14, "alpha": 0.8},
            _frame(np.r_[50.0, np.tile([100.0, 101.0], 10), np.full(8100, 101.0)]),
        ),
        (
            "walk_then_flat",
            "ADX",
            {"period": 30},
            # short enough that the Wilder x Wilder start still shows (the negative control)
            _frame(np.r_[walk[:300], np.full(60, walk[299]), walk[299] + np.sin(np.arange(60))], 0.4),
        ),
        (
            "walk",
            "ADAPTIVE_RSI",
            {"period": 14, "alpha": 0.7},
            # 300 bars: at alpha 0.7 a 3000-bar walk settles, so the shifted control would not move
            _frame(walk[:300], 0.4),
        ),
    ]


@pytest.mark.parametrize(
    ("label", "kind", "params", "frame"), _stress_cases(), ids=[c[0] for c in _stress_cases()]
)
def test_origin_stress_cases(
    label: str, kind: str, params: dict[str, object], frame: pd.DataFrame
) -> None:
    """Start shifted by k in {1, 7, 37}, origin fixed: the value at T is bit-equal to the
    cap-0 replay over the origin history. Negative control: replaying from the shifted
    start (no origin) moves the value at the same T."""
    spec = indicator_registry.get(kind)
    assert spec is not None
    hist = _arrays(frame)
    t = len(frame) - 1
    want = replay_pta_series(spec, frame, params, 0)[spec.output_names[0]][t]
    assert np.isfinite(want)
    for k in (1, 7, 37):
        start = t - k
        strategy = _strategy({"x": {"type": kind, **params}})
        install_origin(strategy, "1h", hist, start)
        got = _feed(strategy, "1h", _slice(hist, start))[-1]["x"]
        assert _bits(got) == _bits(want), (label, k)
    shifted = replay_pta_series(spec, frame.iloc[1:], params, 0)[spec.output_names[0]][t - 1]
    assert abs(shifted - want) > 1e-9 * abs(want), (label, shifted, want)


# ------------------------------------------------------------------- VIDYA 0/0 hold (R6)


def test_vidya_flat_run_holds_value() -> None:
    """A full window of equal closes (CMO 0/0) holds the previous value and resumes on the
    next move; bit-equal to the library (which masks the same denominator since 0.9)."""
    frame = _frame(np.r_[np.linspace(10.0, 20.0, 143), np.full(40, 20.0), 19.0 + np.sin(np.arange(180))])
    port = _vidya_port(frame, 14)
    assert port is not None
    fixed = port.to_numpy()
    library = _vidya_library(frame, 14).to_numpy()
    assert np.array_equal(fixed.view(np.int64), library.view(np.int64))
    assert np.array_equal(fixed[155:183], np.full(28, fixed[155]))
    assert fixed[183] != fixed[182]
    assert np.isfinite(fixed[156:]).all()


def test_vidya_flat_window_on_a_small_input() -> None:
    """Hand-computed (period 3, alpha 0.5): seed 2, 2.5, 2.75, hold 2.75, then 3.375."""
    frame = _frame(np.array([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0]))
    np.testing.assert_array_equal(
        compute_vidya(frame, {"period": 3}).to_numpy(),
        [np.nan, np.nan, 2.0, 2.5, 2.75, 2.75, 3.375],
    )


# ---------------------------------------------------------------- paper unchanged (R7-R9)


def _needs(
    indicators: dict[str, dict[str, Any]], timeframe: str = "15m", extra: list[str] | None = None
) -> tuple[list[WarmupNeed], dict[str, int]]:
    dsl = _dsl(indicators, timeframe, extra)
    compiler = StrategyCompiler()
    module = compiler.compile_to_module(dsl)
    strategy = module.OriginProbeStrategy(config=module.OriginProbeConfig(instrument_id=IID))
    infos = compiler._gather_indicator_info(dsl)
    caps = dict(getattr(strategy, "_pta_buffer_cap", {}))  # node.py reads it the same way
    return indicator_needs(infos, strategy.config, timeframe), caps


@pytest.mark.parametrize(
    ("indicator", "reason"),
    [
        ({"type": "VIDYA"}, SEED_PARITY_REASON),
        ({"type": "VIDYA", "period": 40}, SEED_PARITY_REASON),
        ({"type": "ADAPTIVE_RSI"}, SEED_PARITY_REASON),
        ({"type": "ADX", "period": 17}, f"{SEED_PARITY_REASON}: cap 400 < 25 x lookback 17"),
        ({"type": "ADX", "period": 30}, f"{SEED_PARITY_REASON}: cap 400 < 25 x lookback 30"),
        ({"type": "SQUEEZE_RATIO"}, D42M5_REASON),
    ],
)
def test_paper_refusals_unchanged(indicator: dict[str, Any], reason: str) -> None:
    """Refused at classification (rows=None, nothing fetched), with the new reason text for
    the three origin kinds and main's text for SQUEEZE_RATIO."""
    needs, _caps = _needs({"x": indicator})
    with pytest.raises(WarmupRefused) as excinfo:
        check_warmup("BTCUSDT", needs)
    assert str(excinfo.value) == (
        f"paper warm-up refused: BTCUSDT 15m x ({indicator['type']}): {reason}"
    )


@pytest.mark.parametrize(
    ("period", "rows", "admitted"),
    [(14, 450, False), (14, 513, False), (14, 514, True), (16, 515, False), (16, 516, True)],
)
def test_paper_adx_bounds_and_caps_unchanged(period: int, rows: int, admitted: bool) -> None:
    """ADX(14)/(16) bounds 514/516 and cap 400 are main's (a 450-bar seed still refuses)."""
    needs, caps = _needs({"adx": {"type": "ADX", "period": period}})
    assert caps == {"15m": 400}
    assert [n.bars for n in needs] == [500 + period]
    if admitted:
        check_warmup("BTCUSDT", needs, {"15m": rows}, caps)
        return
    with pytest.raises(WarmupRefused, match=f"needs {500 + period} bars, seed holds {rows}"):
        check_warmup("BTCUSDT", needs, {"15m": rows}, caps)


@pytest.mark.parametrize(
    ("layout", "indicators", "extra", "admitted"),
    [
        ("vidya_4h", {"r": {"type": "RSI"}, "v": {"type": "VIDYA", "timeframe": "4h"}}, ["4h"], False),
        ("arsi", {"a": {"type": "ADAPTIVE_RSI", "period": 10, "alpha": 0.7}}, [], False),
        ("adx20_kama", {"a": {"type": "ADX", "period": 20}, "k": {"type": "KAMA"}}, [], False),
        ("adx12_kama_pp", {"a": {"type": "ADX", "period": 12}, "k": {"type": "KAMA", "period": 48},
                           "p": {"type": "PRICE_POSITION", "period": 133}}, [], True),
        ("adx14_on_4h", {"e": {"type": "EMA", "period": 20},
                         "a": {"type": "ADX", "period": 14, "timeframe": "4h"}}, ["4h"], True),
        ("no_kind", {"e": {"type": "EMA", "period": 50}, "r": {"type": "RSI"}}, [], True),
    ],
)
def test_paper_admission_unchanged(
    layout: str, indicators: dict[str, dict[str, Any]], extra: list[str], admitted: bool
) -> None:
    """Classification AND the post-fetch gate with the compiled caps give main's decisions."""
    needs, caps = _needs(indicators, "15m", extra)
    rows = dict.fromkeys(caps, 8_000) | {"15m": 8_000}
    if admitted:
        check_warmup("BTCUSDT", needs)
        check_warmup("BTCUSDT", needs, rows, caps)
        return
    with pytest.raises(WarmupRefused, match="vibe-quant-d42m5"):
        check_warmup("BTCUSDT", needs)


def test_paper_never_installs_the_origin() -> None:
    """Paper buffers keep main's caps: no paper module calls the origin install."""
    for path in sorted((REPO / "vibe_quant" / "paper").rglob("*.py")):
        text = path.read_text()
        assert "install_origin" not in text and "_pta_origin" not in text, path


# ------------------------------------------------- install_origin seeds only its own tf (R5)


def test_install_origin_seeds_only_the_installed_timeframe() -> None:
    """VIDYA 1h + KAMA 4h, install 1h: the 4h KAMA gets no value from 1h bars; VIDYA holds
    the cap-0 replay of the 1h prefix."""
    frame = _flat_to_moving()
    strategy = _strategy(
        {"v": {"type": "VIDYA", "period": 14}, "kama": {"type": "KAMA", "period": 25, "timeframe": "4h"}},
        "1h",
        ["4h"],
    )
    install_origin(strategy, "1h", _arrays(frame), 200)
    spec = indicator_registry.get("VIDYA")
    assert spec is not None
    held = replay_pta_series(spec, frame.iloc[:200], {"period": 14}, 0)["value"][-1]
    assert set(strategy._pta_values) == {"v"}
    assert _bits(strategy._pta_values["v"]) == _bits(held)


@pytest.mark.parametrize("k", [1, 41])  # VIDYA(14) lookback 42: 41 is the last unready prefix
def test_install_origin_short_prefix_seeds_nothing(k: int) -> None:
    """A prefix shorter than the lookback leaves ``_pta_values`` empty (no NaN key: the
    strategy's readiness reads key presence); one more bar seeds the value."""
    hist = _arrays(_flat_to_moving())
    strategy = _strategy({"v": {"type": "VIDYA", "period": 14}})
    assert strategy._pta_lookback["v"] == 42
    install_origin(strategy, "1h", hist, k)
    assert strategy._pta_values == {}
    ready = _strategy({"v": {"type": "VIDYA", "period": 14}})
    install_origin(ready, "1h", hist, 42)
    assert set(ready._pta_values) == {"v"} and np.isfinite(ready._pta_values["v"])


def test_install_origin_prefills_a_context_buffer_like_live_appends() -> None:
    """VIDYA + FUNDING (needs_context) on 1h: the prefilled buffer equals k live appends,
    frame and ``attrs["bar_close_ns"]`` (ts_init rounded to the bar boundary) alike; the
    context indicator gets no seeded value."""
    hist = _arrays(_flat_to_moving())
    hist["ts_init"] = hist["ts_init"] - 1_000  # NT bar ts_init sits just before the boundary
    indicators = {"v": {"type": "VIDYA", "period": 14}, "f": {"type": "FUNDING"}}
    k = 60
    origin = _strategy(indicators)
    install_origin(origin, "1h", hist, k)
    live = _strategy(indicators)
    live._update_pta_indicators = lambda tf: None  # buffer only: FUNDING reads aux data
    _feed(live, "1h", _slice(hist, 0, k))
    got, want = origin._pta_bufs["1h"].frame(), live._pta_bufs["1h"].frame()
    pd.testing.assert_frame_equal(got, want)
    np.testing.assert_array_equal(got.attrs["bar_close_ns"], want.attrs["bar_close_ns"])
    assert got.attrs["bar_close_ns"][0] == 1_599_998_400_000_000_000  # t0 rounded to the hour
    assert set(origin._pta_values) == {"v"}


# ------------------------------------------- cap 0 replay: no per-bar Python (d42m5 T5 speed)


def test_cap_zero_window_starts_never_drive_a_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap 0 never trims: every start is 0 without one ``PtaBuffer.append`` per bar (a
    3.2M-bar BTC 1m origin spent seconds in that loop per replay). Capped schedules still
    come from the buffer."""
    from vibe_quant.dsl import pta_replay

    real = pta_replay.PtaBuffer

    class NoUncappedBuffer(real):  # type: ignore[misc, valid-type]
        def __init__(self, cap: int, with_close_ns: bool = False) -> None:
            assert cap, "cap 0 drove a PtaBuffer"
            super().__init__(cap, with_close_ns)

    monkeypatch.setattr(pta_replay, "PtaBuffer", NoUncappedBuffer)
    starts = pta_replay.pta_window_starts(5000, 0)
    assert starts.dtype == np.int64 and starts.shape == (5000,) and not starts.any()
    assert pta_replay.pta_window_starts(0, 0).shape == (0,)
    capped = pta_replay.pta_window_starts(60, 8)
    assert capped[:10].tolist() == [0] * 10 and capped[10] == 3  # trims at 11 > 8 + 2


def _reference_replay(
    spec: Any, ohlcv: pd.DataFrame, params: dict[str, object], cap: int
) -> dict[str, np.ndarray]:
    """b36a7011's per-bar replay loop, verbatim: the bit-exact reference of the fast one."""
    from vibe_quant.dsl.indicators import pta_lookback
    from vibe_quant.dsl.pta_buffer import PtaBuffer
    from vibe_quant.dsl.pta_replay import _segment_arrays

    names = spec.output_names
    frame = ohlcv.loc[:, list(OHLCV)].astype("float64").reset_index(drop=True)
    n_bars = len(frame)
    output = {name: np.full(n_bars, np.nan, dtype=np.float64) for name in names}
    held = dict.fromkeys(names, np.nan)
    starts = np.empty(n_bars, dtype=np.int64)
    buf = PtaBuffer(cap)
    for i in range(n_bars):
        buf.append(0.0, 0.0, 0.0, 0.0, 0.0)
        starts[i] = i + 1 - len(buf)
    lookback = pta_lookback(spec, params)
    i = 0
    while i < n_bars:
        start = int(starts[i])
        end = i
        while end + 1 < n_bars and int(starts[end + 1]) == start:
            end += 1
        window = frame.iloc[start : end + 1].reset_index(drop=True)
        segment = _segment_arrays(spec.compute_fn(window, params), names, len(window))
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


def _holey(df: pd.DataFrame, _params: dict[str, object]) -> dict[str, np.ndarray]:
    """Two causal outputs with NaN runs (incl. a long one), -0.0 and a NaN-only second output
    on short windows: every branch of the NaN-hold."""
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(close)
    a = np.cumsum(close) / np.arange(1, n + 1)
    a[np.arange(n) % 5 == 3] = np.nan
    a[(np.arange(n) >= 70) & (np.arange(n) < 140)] = np.nan
    a[np.arange(n) % 11 == 0] = -0.0
    b = np.full(n, np.nan) if n < 30 else close - close[0]
    return {"value": a, "other": b}


@pytest.mark.parametrize("cap", [0, 8, 40])
@pytest.mark.parametrize("period", [1, 6, 50])
def test_replay_equals_the_per_bar_reference(cap: int, period: int) -> None:
    """The vectorised segment/hold replay is bit-equal (incl. NaN mask and -0.0) to the
    per-bar loop it replaced, on every cap and lookback edge, single and multi output."""
    from dataclasses import replace

    base = indicator_registry.get("VIDYA")
    assert base is not None
    rng = np.random.RandomState(period + cap)
    frame = _frame(100.0 + np.cumsum(rng.randn(400)), spread=0.2, seed=cap)
    params: dict[str, object] = {"period": period}
    for spec in (
        replace(base, compute_fn=_holey, output_names=("value", "other")),
        indicator_registry.get("ADAPTIVE_RSI"),
    ):
        assert spec is not None
        p = params if spec.compute_fn is _holey else {"period": max(period, 2), "alpha": 0.7}
        got = replay_pta_series(spec, frame, p, cap)
        want = _reference_replay(spec, frame, p, cap)
        assert set(got) == set(want)
        for name in want:
            assert got[name].tobytes() == want[name].tobytes(), (spec.name, name, cap, period)
