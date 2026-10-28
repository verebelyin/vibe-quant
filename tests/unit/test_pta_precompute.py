"""Precomputed compute_fn columns: the shared rule and the generated-code seam.

``strategy_pta_columns`` (one rule, shared with the fast tier) replays a strategy's
compute_fn indicators over a frame; ``PtaPrecomputed`` feeds those columns into the
generated ``_feed_pta_buffer`` guarded per bar by (ts_init, close) with an exact
fallback to the live per-bar path. These tests drive the REAL generated
``_feed_pta_buffer`` through ``pta_stream_harness.run_stream`` and require the
per-bar ``_pta_values`` snapshots to match the live stream BITWISE (no tolerance).
Poisoned columns past an injected failure prove the fallback never reads them.
"""

from __future__ import annotations

import struct
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.unit.pta_stream_harness import (
    BarEvent,
    compile_indicators,
    compile_source,
    run_stream,
    stream_buffer_caps,
)
from vibe_quant.dsl.indicators import indicator_registry, pta_buffer_cap, pta_lookback
from vibe_quant.dsl.pta_replay import (
    PtaPrecomputed,
    PtaReplayUnsupported,
    strategy_pta_columns,
)

REPO = Path(__file__).resolve().parents[2]
ZEC_1M = REPO / "data" / "catalog" / "data" / "bar" / "ZECUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
needs_zec = pytest.mark.skipif(not ZEC_1M.is_dir(), reason="needs the ZECUSDT 1m catalog")

Specs = dict[str, tuple[str, dict[str, object], str]]
POISON = 12345.0


def _candidates() -> list[str]:
    names: list[str] = []
    for name in sorted(indicator_registry.list_indicators()):
        spec = indicator_registry.get(name)
        if (
            spec is not None
            and spec.compute_fn is not None
            and spec.nt_class is None  # an NT-class spec compiles through the NT indicator path
            and not spec.needs_context
        ):
            names.append(name)
    return names


_PTA_CANDIDATES = _candidates()


def test_registry_has_the_sixteen_buffered_compute_specs() -> None:
    """Guard against the parametrisation silently shrinking."""
    assert len(_PTA_CANDIDATES) >= 16


def _bars_for(cap: int) -> int:
    """Bars that cross buffer trims for a rolling ``cap`` (trim k lands at ``cap + k*(cap//4 + 1)``).

    Two trims for the common caps (400..800: the first grows past the cap, the second
    starts from the steady state), one for 800 < cap <= 2000. A cap above 2000 (the
    RAMS / TREND_ENSEMBLE defaults) would need 3100+ bars of whole-window recomputes per
    case: it runs 1000 bars (4x its lookback) of warm-up only (TREND_ENSEMBLE
    max_lookback 60 crosses trims at a small cap).
    """
    if cap > 2000:
        return 1000
    trims = 2 if cap <= 800 else 1
    return cap + trims * (cap // 4 + 1) + 25


def _cap(name: str, params: dict[str, object] | None = None) -> int:
    spec = indicator_registry.get(name)
    assert spec is not None
    lookback = pta_lookback(spec, {**spec.default_params, **(params or {})})
    return pta_buffer_cap([lookback], full_history=False)


def _flat_bars(n: int = 750, tf: str = "1h") -> list[BarEvent]:
    """The architect's flat-probe set: 40-bar o=h=l=c stretches, volume 0 every 97th bar."""
    rng = np.random.RandomState(5)
    c = 100 + np.cumsum(rng.randn(n) * 0.3)
    for s in range(50, n, 280):
        c[s : s + 40] = c[s]
    h = c + np.abs(rng.randn(n)) * 0.2
    lo = c - np.abs(rng.randn(n)) * 0.2
    o = np.roll(c, 1)
    o[0] = c[0]
    for s in range(50, n, 280):
        h[s : s + 40] = c[s]
        lo[s : s + 40] = c[s]
        o[s : s + 40] = c[s]
    v = np.where(np.arange(n) % 97 == 0, 0.0, 1.0 + rng.rand(n))
    return [(tf, float(o[i]), float(h[i]), float(lo[i]), float(c[i]), float(v[i])) for i in range(n)]


@cache
def _compile_cached(name: str, params: tuple[tuple[str, object], ...]) -> tuple[type, Specs]:
    """One compile per (indicator, params): classes and specs are read-only to the tests."""
    _, cls = compile_indicators({"ind": {"type": name, **dict(params)}})
    assert hasattr(cls, "_feed_pta_buffer"), f"{name}: not on the compute_fn buffer path"
    spec = indicator_registry.get(name)
    assert spec is not None
    return cls, {"ind": (name, {**spec.default_params, **dict(params)}, "1h")}


def _compile(name: str, params: dict[str, object] | None = None) -> tuple[type, Specs]:
    return _compile_cached(name, tuple(sorted((params or {}).items())))


def _precompute(
    cls: Any, specs: Specs, bars: list[BarEvent], *, n: int | None = None
) -> dict[str, PtaPrecomputed]:
    """What the runner will install: replay columns + per-tf (index, close) guards."""
    caps = stream_buffer_caps(specs)
    params = {name: p for name, (_, p, _) in specs.items()}
    pre: dict[str, PtaPrecomputed] = {}
    for tf in caps:
        tf_bars = [b for b in bars if b[0] == tf][: n if n is not None else None]
        frame = pd.DataFrame(
            {
                "open": [b[1] for b in tf_bars],
                "high": [b[2] for b in tf_bars],
                "low": [b[3] for b in tf_bars],
                "close": [b[4] for b in tf_bars],
                "volume": [b[5] for b in tf_bars],
            }
        )
        columns = strategy_pta_columns(params, cls._PTA_META, caps, tf, frame)
        pre[tf] = PtaPrecomputed(
            list(range(len(tf_bars))),
            [b[4] for b in tf_bars],
            [(key, col.tolist()) for key, col in columns],
        )
    return pre


def _poison(pre: PtaPrecomputed, start: int) -> None:
    """Overwrite every column value from ``start`` so any use of it shows up."""
    pre.columns = [(key, col[:start] + [POISON] * (len(col) - start)) for key, col in pre.columns]


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


def _differing_bars(a: list[dict[str, float]], b: list[dict[str, float]]) -> int:
    assert len(a) == len(b)
    bad = 0
    for sa, sb in zip(a, b, strict=True):
        if sa.keys() != sb.keys() or any(_bits(sa[k]) != _bits(sb[k]) for k in sa):
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# R1 / R7b: every registry compute_fn spec, flat bars, bitwise.
# ---------------------------------------------------------------------------

# Every registry spec at its defaults, KAMA / ADX with explicit and varied lookbacks, and
# TREND_ENSEMBLE 60, which brings that long-cap spec down to a cap whose trims fit in a
# test (RAMS always looks back 200 + period bars: its cap stays above 2000).
_CASES: list[tuple[str, dict[str, object]]] = [(name, {}) for name in _PTA_CANDIDATES] + [
    ("KAMA", {"period": 10}),
    ("KAMA", {"period": 25}),
    ("KAMA", {"period": 48}),
    ("ADX", {"period": 12}),
    ("ADX", {"period": 14}),
    ("TREND_ENSEMBLE", {"max_lookback": 60}),
]


@pytest.mark.parametrize(("name", "params"), _CASES, ids=[f"{n}{p or ''}" for n, p in _CASES])
def test_precomputed_feed_equals_live_stream_for_every_registry_spec(
    name: str, params: dict[str, object]
) -> None:
    """Precompute == live per-bar ``_pta_values`` on flat bars: same keys, bit-equal values.

    ``_bars_for(cap)`` bars (three buffer trims for the common caps) with 40-bar o=h=l=c
    stretches and zero-volume bars: the hostile inputs for ADX/KAMA. ``fed == n``
    proves the precompute path really served every bar (otherwise equality would be
    trivial).
    """
    cls, specs = _compile(name, params)
    bars = _flat_bars(_bars_for(_cap(name, params)))
    live = run_stream(cls, bars, memo=False, specs=specs)
    pre = _precompute(cls, specs, bars)
    fast = run_stream(cls, bars, memo=False, specs=specs, precompute=pre)

    assert pre["1h"].fed == len(bars)
    assert pre["1h"].failed_at is None
    assert _differing_bars(fast, live) == 0
    assert live[-1], f"{name}: live stream produced no value"


def test_multi_output_keys_primary_and_sub_outputs() -> None:
    """MACD: the primary under ``ind``, every sub-output under ``ind_<key>``."""
    cls, specs = _compile("MACD")
    bars = _flat_bars(600)
    pre = _precompute(cls, specs, bars)
    keys = [key for key, _ in pre["1h"].columns]
    spec = indicator_registry.get("MACD")
    assert spec is not None
    assert keys[0] == "ind"
    assert keys[1:] == [f"ind_{k}" for k in spec.output_names]
    live = run_stream(cls, bars, memo=False, specs=specs)
    assert set(live[-1]) == set(keys)


def test_two_timeframes_each_get_their_own_precompute() -> None:
    """A 4h ADX on a 1h strategy: each timeframe's index/close guard is independent."""
    _, cls = compile_indicators(
        {"a": {"type": "ADX", "timeframe": "4h"}, "k": {"type": "KAMA"}},
        timeframe="1h",
        additional_timeframes=["4h"],
    )
    adx = indicator_registry.get("ADX")
    kama = indicator_registry.get("KAMA")
    assert adx is not None and kama is not None
    specs: Specs = {
        "a": ("ADX", dict(adx.default_params), "4h"),
        "k": ("KAMA", dict(kama.default_params), "1h"),
    }
    one_h = _flat_bars(760, "1h")  # crosses the 1h buffer's trims at 501, 602 and 703
    four_h = _flat_bars(190, "4h")
    bars: list[BarEvent] = []
    for i, bar in enumerate(one_h):  # a 4h bar closes after every 4th 1h bar
        bars.append(bar)
        if i % 4 == 3:
            bars.append(four_h[i // 4])
    live = run_stream(cls, bars, memo=False, specs=specs)
    pre = _precompute(cls, specs, bars)
    fast = run_stream(cls, bars, memo=False, specs=specs, precompute=pre)

    assert set(pre) == {"1h", "4h"}
    assert all(p.fed == p.n and p.failed_at is None for p in pre.values())
    assert _differing_bars(fast, live) == 0


# ---------------------------------------------------------------------------
# R2: real 1m bars.
# ---------------------------------------------------------------------------

_REAL_CASES: list[tuple[str, dict[str, object]]] = [
    ("ADX", {}),
    ("ADX", {"period": 12}),
    ("KAMA", {"period": 10}),
    ("KAMA", {"period": 25}),
    ("KAMA", {"period": 48}),
    ("PRICE_POSITION", {}),
    ("PRICE_POSITION", {"period": 133}),
    ("RAMS", {}),
]


@cache
def _zec_bars() -> list[BarEvent]:
    """ZECUSDT 1m bars of 2026-03-01..03 (4320 bars, many zero close diffs), as events."""
    from vibe_quant.fast_tier.signals import load_bars

    raw = load_bars(
        str(REPO / "data" / "catalog"),
        "ZECUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL",
        "2026-03-01",
        "2026-03-04",
    )
    return [
        ("1h", float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume))
        for b in raw
    ]


@needs_zec
@pytest.mark.parametrize(
    ("name", "params"), _REAL_CASES, ids=[f"{n}{p or ''}" for n, p in _REAL_CASES]
)
def test_replay_equals_live_on_real_1m_bars(name: str, params: dict[str, object]) -> None:
    """Real ZECUSDT 1m bars (many zero close diffs): precompute == live, trims crossed."""
    n = _bars_for(_cap(name, params))
    bars = _zec_bars()[:n]
    assert len(bars) == n
    cls, specs = _compile(name, params)
    live = run_stream(cls, bars, memo=False, specs=specs)
    pre = _precompute(cls, specs, bars)
    fast = run_stream(cls, bars, memo=False, specs=specs, precompute=pre)

    assert pre["1h"].fed == n
    assert _differing_bars(fast, live) == 0


def test_installed_columns_are_what_the_strategy_reads_without_recompute() -> None:
    """Columns that differ from live are served verbatim: a covered bar skips the recompute."""
    cls, specs = _compile("KAMA", {"period": 10})
    bars = _flat_bars(800)
    live = run_stream(cls, bars, memo=False, specs=specs)
    pre = _precompute(cls, specs, bars)["1h"]
    pre.columns = [(key, [v + 1000.0 for v in col]) for key, col in pre.columns]
    shifted = pre.columns[0][1]

    out = run_stream(cls, bars, memo=False, specs=specs, precompute={"1h": pre})

    assert pre.fed == len(bars)
    held = float("nan")
    for i, snap in enumerate(out):
        if shifted[i] == shifted[i]:
            held = shifted[i]
        assert snap.get("ind", float("nan")) == held or held != held
    assert out[-1]["ind"] == shifted[-1] != live[-1]["ind"]


# ---------------------------------------------------------------------------
# R3 / R4 / R5 / R15: guard failures fall back exactly.
# ---------------------------------------------------------------------------


@cache
def _fallback_case() -> tuple[Any, Specs, list[BarEvent], list[dict[str, float]]]:
    """KAMA over 900 bars (trims at 501, 602, 703, 804); the live stream is read-only, so shared."""
    cls, specs = _compile("KAMA", {"period": 10})
    bars = _flat_bars(900)
    return cls, specs, bars, run_stream(cls, bars, memo=False, specs=specs)


def test_ts_mismatch_falls_back_exactly() -> None:
    cls, specs, bars, live = _fallback_case()
    pre = _precompute(cls, specs, bars)["1h"]
    pre.ts = [t if i < 700 else t + 1000 for i, t in enumerate(pre.ts)]
    _poison(pre, 700)  # any bar served from the precompute after 700 would differ

    out = run_stream(cls, bars, memo=False, specs=specs, precompute={"1h": pre})

    assert pre.failed_at == 700
    assert pre.fed == 700
    assert _differing_bars(out, live) == 0


def test_short_precompute_falls_back_at_end() -> None:
    cls, specs, bars, live = _fallback_case()
    pre = _precompute(cls, specs, bars, n=800)["1h"]

    out = run_stream(cls, bars, memo=False, specs=specs, precompute={"1h": pre})

    assert pre.failed_at == 800
    assert pre.fed == 800
    assert _differing_bars(out, live) == 0


def test_close_mismatch_falls_back() -> None:
    cls, specs, bars, live = _fallback_case()
    pre = _precompute(cls, specs, bars)["1h"]
    nxt = np.nextafter(pre.close[600], np.inf)  # one ulp: the compare is exact
    pre.close[600] = float(nxt)
    _poison(pre, 600)

    out = run_stream(cls, bars, memo=False, specs=specs, precompute={"1h": pre})

    assert pre.failed_at == 600
    assert pre.fed == 600
    assert _differing_bars(out, live) == 0


def test_shifted_start_fails_at_zero() -> None:
    """A leading extra catalog bar: no offset search, the whole run is live and exact."""
    cls, specs, bars, live = _fallback_case()
    pre = _precompute(cls, specs, bars)["1h"]
    pre.ts = [t + 1 for t in pre.ts]
    _poison(pre, 0)

    out = run_stream(cls, bars, memo=False, specs=specs, precompute={"1h": pre})

    assert pre.failed_at == 0
    assert pre.fed == 0
    assert _differing_bars(out, live) == 0


# ---------------------------------------------------------------------------
# PtaPrecomputed.feed contract.
# ---------------------------------------------------------------------------


def test_feed_never_recovers_after_failure() -> None:
    pre = PtaPrecomputed([10, 11, 12], [1.0, 2.0, 3.0], [("k", [5.0, 6.0, 7.0])])
    values: dict[str, float] = {}

    assert pre.feed(values, 10, 1.0) is True
    assert values == {"k": 5.0}
    assert pre.feed(values, 99, 2.0) is False  # ts mismatch at index 1
    assert (pre.failed_at, pre.fed) == (1, 1)
    # the next bar now matches index 1 exactly -- still never True, nothing written
    assert pre.feed(values, 11, 2.0) is False
    assert pre.feed(values, 12, 3.0) is False
    assert values == {"k": 5.0}
    assert (pre.failed_at, pre.fed) == (1, 1)  # first failure position is kept


def test_feed_nan_never_overwrites_the_held_value() -> None:
    nan = float("nan")
    pre = PtaPrecomputed([0, 1, 2, 3], [1.0] * 4, [("k", [nan, 4.0, nan, 5.0])])
    values: dict[str, float] = {}

    assert pre.feed(values, 0, 1.0) is True
    assert values == {}  # a leading NaN writes nothing (key absent, as live)
    assert pre.feed(values, 1, 1.0) is True
    assert values == {"k": 4.0}
    assert pre.feed(values, 2, 1.0) is True
    assert values == {"k": 4.0}  # NaN-hold
    assert pre.feed(values, 3, 1.0) is True
    assert values == {"k": 5.0}


def test_feed_equality_edges() -> None:
    """ts one off, close one ulp off, and the exhausted index each fail; exact passes."""
    one_ulp = float(np.nextafter(2.0, np.inf))
    for ts, close, fed_ok in [(5, 2.0, True), (6, 2.0, False), (4, 2.0, False), (5, one_ulp, False)]:
        pre = PtaPrecomputed([5], [2.0], [("k", [9.0])])
        values: dict[str, float] = {}
        assert pre.feed(values, ts, close) is fed_ok
        assert values == ({"k": 9.0} if fed_ok else {})

    pre = PtaPrecomputed([5], [2.0], [("k", [9.0])])
    assert pre.feed({}, 5, 2.0) is True
    assert pre.feed({}, 6, 2.0) is False  # index exhausted
    assert (pre.failed_at, pre.fed) == (1, 1)


# ---------------------------------------------------------------------------
# R6 / R7: the seam is inert unless installed; context strategies have none.
# ---------------------------------------------------------------------------


def test_context_strategy_has_no_precompute_seam() -> None:
    import vibe_quant.dsl.plugins.funding  # noqa: F401  (registers FUNDING_Z)

    src = compile_source({"fz": {"type": "FUNDING_Z", "period": 30}}, timeframe="4h")

    assert "_PTA_META" not in src
    assert "_pta_pre" not in src
    assert "_pre.feed" not in src
    assert "((bar.ts_init + _step // 2) // _step) * _step," in src  # the context feed is unchanged


def test_non_context_source_carries_the_seam() -> None:
    src = compile_source({"ind": {"type": "KAMA", "period": 25}, "m": {"type": "MACD"}})

    assert "    _pta_pre = None\n" in src
    assert '    _PTA_META = {"ind": ("KAMA", "1h"), "m": ("MACD", "1h")}\n' in src
    assert src.index("_PTA_META") < src.index("def __init__")
    assert "if _pre is not None and _pre.feed(self._pta_values, bar.ts_init, float(bar.close)):" in src


def test_no_install_is_todays_path() -> None:
    """``_pta_pre`` None (validation, paper, ``__new__`` harness): the live path."""
    cls, specs = _compile("ADX", {"period": 12})
    bars = _flat_bars(700)
    assert cls._pta_pre is None
    plain = run_stream(cls, bars, memo=False, specs=specs)
    empty = run_stream(cls, bars, memo=False, specs=specs, precompute={})
    other_tf = run_stream(
        cls, bars, memo=False, specs=specs, precompute=_other_tf(cls, specs, bars)
    )

    assert _differing_bars(empty, plain) == 0
    assert _differing_bars(other_tf, plain) == 0
    pre = _precompute(cls, specs, bars)
    assert _differing_bars(run_stream(cls, bars, memo=False, specs=specs, precompute=pre), plain) == 0


def _other_tf(cls: Any, specs: Specs, bars: list[BarEvent]) -> dict[str, PtaPrecomputed]:
    """A precompute installed for a timeframe the strategy does not buffer: ignored."""
    pre = _precompute(cls, specs, bars)["1h"]
    _poison(pre, 0)
    return {"4h": pre}


# ---------------------------------------------------------------------------
# strategy_pta_columns.
# ---------------------------------------------------------------------------


def test_strategy_pta_columns_rejects_unreplayable_indicator() -> None:
    frame = pd.DataFrame({k: [1.0, 2.0] for k in ("open", "high", "low", "close", "volume")})
    with pytest.raises(PtaReplayUnsupported, match="compute_fn indicator ind cannot be replayed"):
        strategy_pta_columns({"ind": {}}, {"ind": ("NOT_A_REAL_TYPE", "1m")}, {"1m": 0}, "1m", frame)
    with pytest.raises(PtaReplayUnsupported, match="compute_fn indicator ind cannot be replayed"):
        strategy_pta_columns({"ind": {}}, {"ind": ("NATR", "1m")}, {"1m": 0}, "1m", frame)  # no compute_fn
    assert issubclass(PtaReplayUnsupported, ValueError)


def test_strategy_pta_columns_selects_by_timeframe_and_uses_replay_hook() -> None:
    """Only ``tf``'s indicators; the injected ``replay`` is what computes them."""
    frame = pd.DataFrame({k: [1.0, 2.0, 3.0] for k in ("open", "high", "low", "close", "volume")})
    calls: list[tuple[str, int]] = []

    def fake(spec: Any, ohlcv: pd.DataFrame, params: dict[str, object], cap: int) -> Any:
        calls.append((spec.name, cap))
        return {name: np.array([1.0, 2.0, 3.0]) for name in spec.output_names}

    meta = {"a": ("KAMA", "1m"), "b": ("KAMA", "4h"), "m": ("MACD", "1m")}
    params = {"a": {"period": 10}, "b": {"period": 10}, "m": {}}
    out = strategy_pta_columns(params, meta, {"1m": 400, "4h": 800}, "1m", frame, replay=fake)

    assert calls == [("KAMA", 400), ("MACD", 400)]
    macd = indicator_registry.get("MACD")
    assert macd is not None
    assert [k for k, _ in out] == ["a", "m", *(f"m_{k}" for k in macd.output_names)]
