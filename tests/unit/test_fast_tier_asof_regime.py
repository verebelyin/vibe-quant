"""BTC_TREND / OWN_TREND / BTC_ROC on the fast tier: whole-frame columns equal NT's feed.

All three specs answer from a DAILY reference series for the bar's OWN close time only:
the latest daily bar that closed at or before it, NaN when that bar is more than two days
old. So the fast tier computes one column over the whole run frame instead of replaying the
rolling buffer (``IndicatorSpec.asof_context``).

Every comparison is against the REAL generated ``_feed_pta_buffer`` +
``_update_pta_indicators`` (NT's source of truth), bitwise and NaN included, on a strategy
symbol whose daily bars are its own (ZECUSDT, OWN_TREND) and on a non-BTC symbol (ETHUSDT),
where BTC_TREND / BTC_ROC read an aux frame of another symbol.

Causality has two halves: the frame-level prefix test (later BARS cannot move an earlier
value) and the synthetic-reference tests, where each daily close is a distinct power of two
so the answer before / exactly at / after a daily close is a literal, and rewriting ONLY
future daily bars must leave the past bit-identical.

Failing before the flag flip: ``asof_context_columns`` skips unflagged specs (it returns
no column for them) and the end-to-end genomes fall back to NT with one recorded fallback.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import indicator_registry, pta_lookback
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.pta_buffer import PtaBuffer
from vibe_quant.dsl.pta_replay import asof_context_columns, bar_close_ns
from vibe_quant.fast_tier import fallbacks
from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
from vibe_quant.fast_tier.signals import Signals, compute_signals, load_bars

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
MIN = 60_000_000_000
DAY = 86_400 * 1_000_000_000

REGIME_SPECS = ("BTC_TREND", "OWN_TREND", "BTC_ROC")


def _instrument_id(symbol: str) -> str:
    return f"{symbol}-PERP.BINANCE"


def _bar_type(symbol: str) -> str:
    return f"{_instrument_id(symbol)}-1-MINUTE-LAST-EXTERNAL"


def _skip_without_data(symbol: str) -> None:
    """Skip (never fail) when this symbol's 1m catalog bars or daily bars are absent."""
    if not (CATALOG / "data" / "bar" / _bar_type(symbol)).is_dir():
        pytest.skip(f"needs the {symbol} 1m catalog")
    for ref in ("BTCUSDT", symbol):
        if not (CATALOG / "data" / "bar" / f"{_instrument_id(ref)}-1-DAY-LAST-EXTERNAL").is_dir():
            pytest.skip(f"needs the {ref} daily catalog")


@pytest.fixture
def _aux_data() -> Any:
    """Point aux lookups at the repo catalog/archive and restore the globals after."""
    aux_data.configure(ARCHIVE, CATALOG)
    try:
        yield
    finally:
        aux_data.configure(None, None)


# ---------------------------------------------------------------------------
# Harness: the generated strategy, its generated feed, and the as-of column.
# ---------------------------------------------------------------------------


def _strategy(symbol: str, indicator: str, params: dict[str, object], cap: int) -> Any:
    """A generated 1m strategy whose ``_pta_*`` state mirrors what its ``__init__`` builds."""
    indicators: dict[str, dict[str, Any]] = {"ind": {"type": indicator, **params}}
    dsl = validate_strategy_dict(
        {
            "name": "regime_asof",
            "timeframe": "1m",
            "indicators": indicators,
            "entry_conditions": {"long": ["ind > 0"]},
            "exit_conditions": {"long": ["ind < 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 0.8},
            "take_profit": {"type": "fixed_pct", "percent": 1.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cls = next(v for k, v in vars(module).items() if k.endswith("Strategy") and k != "Strategy")
    config_cls = getattr(module, cls.__name__.removesuffix("Strategy") + "Config")
    harness = type(
        "Wired", (cls,), {"config": config_cls(instrument_id=_instrument_id(symbol))}
    )
    inst: Any = harness.__new__(harness)
    effective = {k: v for k, v in indicators["ind"].items() if k != "type"}
    inst._pta_values = {}
    inst._prev_values = {}
    inst._last_close = 0.0
    inst._pta_params = {"ind": effective}
    inst._pta_lookback = {"ind": pta_lookback(indicator, effective)}
    inst._pta_buffer_cap = {"1m": cap}
    inst._pta_bufs = {"1m": PtaBuffer(cap, with_close_ns=True)}
    inst._pta_step_ns = {"1m": MIN}
    return inst


def _bars(symbol: str, start: str, end: str) -> list[Bar]:
    return load_bars(str(CATALOG), _bar_type(symbol), start, end)


def _frame(bars: list[Bar]) -> pd.DataFrame:
    return pd.DataFrame(
        {c: [float(getattr(b, c)) for b in bars] for c in ("open", "high", "low", "close", "volume")}
    )


def _ts_init(bars: list[Bar]) -> np.ndarray:
    return np.fromiter((int(b.ts_init) for b in bars), dtype=np.int64, count=len(bars))


def _bits(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.int64)


def _drive_generated(strategy: Any, bars: list[Bar]) -> np.ndarray:
    """``_pta_values["ind"]`` after each bar's REAL generated ``_feed_pta_buffer``."""
    out = []
    for bar in bars:
        strategy._feed_pta_buffer("1m", bar)
        out.append(strategy._pta_values["ind"])
    return np.array(out, dtype=np.float64)


def _asof_column(
    symbol: str,
    indicator: str,
    params: dict[str, object],
    frame: pd.DataFrame,
    close: np.ndarray,
) -> tuple[np.ndarray, int]:
    """The as-of column of ``ind`` over ``frame`` and its first written row."""
    [(key, column, first)] = asof_context_columns(
        {"ind": params},
        {"ind": (indicator, "1m")},
        "1m",
        frame,
        close,
        _instrument_id(symbol),
    )
    assert key == "ind"
    assert frame.attrs == {}, "the shared frame must not be mutated"
    return column, first


def _feed_reference(strategy: Any, bars: list[Bar]) -> Signals:
    """``compute_signals``' loop with the values the generated feed writes (NT's own)."""
    n = len(bars)
    ready = np.zeros(n, dtype=bool)
    le = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    checks = [
        getattr(strategy, name, None)
        for name in (
            "_check_long_entry",
            "_check_short_entry",
            "_check_long_exit",
            "_check_short_exit",
        )
    ]
    for i, bar in enumerate(bars):
        strategy._feed_pta_buffer("1m", bar)
        strategy._last_close = float(bar.close)
        if not strategy._indicators_ready():
            continue
        ready[i] = True
        for out, check in zip((le, se, lx, sx), checks, strict=True):
            if check is not None:
                out[i] = check(bar)
        strategy._update_prev_values(bar)
    return Signals(ready, le, se, lx, sx)


CASES = [
    ("ZECUSDT", "BTC_TREND", {"period": 200}),
    ("ZECUSDT", "OWN_TREND", {"period": 200}),
    ("ZECUSDT", "BTC_ROC", {"period": 20}),
    ("ETHUSDT", "BTC_TREND", {"period": 200}),
    ("ETHUSDT", "OWN_TREND", {"period": 200}),
    ("ETHUSDT", "BTC_ROC", {"period": 20}),
]
CASE_IDS = [
    "ZEC_BTC_TREND",
    "ZEC_OWN_TREND",
    "ZEC_BTC_ROC",
    "ETH_BTC_TREND",
    "ETH_OWN_TREND",
    "ETH_BTC_ROC",
]


# ---------------------------------------------------------------------------
# The flag itself: every regime spec opts in.
# ---------------------------------------------------------------------------


def test_regime_specs_are_flagged_asof_context() -> None:
    """BTC_TREND / OWN_TREND / BTC_ROC are pointwise as-of context specs."""
    for name in REGIME_SPECS:
        spec = indicator_registry.get(name)
        assert spec is not None, name
        assert spec.asof_context is True, name
        assert spec.needs_context is True, name
        assert spec.output_names == ("value",), name
        assert spec.compute_fn is not None, name
        assert pta_lookback(spec, dict(spec.default_params)) == 1, name


# ---------------------------------------------------------------------------
# Each spec reads its own reference series (the as-of replay must not mix them up).
# ---------------------------------------------------------------------------


def test_each_spec_reads_its_own_reference(_aux_data: None) -> None:
    """The three columns of one ETHUSDT run are pairwise different and BTC_TREND != OWN_TREND.

    A name -> spec slip in the replay would make two columns identical; the parity runs
    cannot see that (NT calls the same helpers). OWN_TREND must be the strategy symbol's
    daily trend and BTC_TREND the BTCUSDT one, on data where the two differ everywhere.
    """
    _skip_without_data("ETHUSDT")
    bars = _bars("ETHUSDT", "2026-03-02", "2026-03-03")
    close = bar_close_ns(_ts_init(bars), MIN)
    frame = _frame(bars)
    columns = {}
    for name in REGIME_SPECS:
        spec = indicator_registry.get(name)
        assert spec is not None
        columns[name], _ = _asof_column("ETHUSDT", name, dict(spec.default_params), frame, close)

    for a, b in itertools.combinations(REGIME_SPECS, 2):
        assert not np.array_equal(_bits(columns[a]), _bits(columns[b])), f"{a} == {b}"

    btc = aux_data.ref_trend_asof("BTCUSDT", close, 200)
    own = aux_data.ref_trend_asof("ETHUSDT", close, 200)
    assert int((_bits(btc) != _bits(own)).sum()) == len(close), "fixture: BTC and ETH trends differ"
    assert np.array_equal(_bits(columns["BTC_TREND"]), _bits(btc))
    assert np.array_equal(_bits(columns["OWN_TREND"]), _bits(own))
    assert np.array_equal(
        _bits(columns["BTC_ROC"]), _bits(aux_data.ref_roc_asof("BTCUSDT", close, 20))
    )


# ---------------------------------------------------------------------------
# Per-bar equality: the whole-frame column == the generated per-bar feed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("symbol", "indicator", "params"), CASES, ids=CASE_IDS)
def test_asof_column_equals_generated_feed_bar_for_bar(
    symbol: str, indicator: str, params: dict[str, object], _aux_data: None
) -> None:
    """1440 bars with cap 400 (trims run): same float bits on every bar, NaN included."""
    _skip_without_data(symbol)
    bars = _bars(symbol, "2026-03-02", "2026-03-03")
    assert len(bars) == 1440

    strategy = _strategy(symbol, indicator, params, cap=400)
    fed = _drive_generated(strategy, bars)

    frame = _frame(bars)
    column, first = _asof_column(
        symbol, indicator, params, frame, bar_close_ns(_ts_init(bars), MIN)
    )

    assert first == 0
    assert np.array_equal(_bits(fed), _bits(column)), "as-of column diverged from NT's feed"
    assert int(np.isnan(column).sum()) == 0


@pytest.mark.parametrize(("symbol", "indicator", "params"), CASES, ids=CASE_IDS)
def test_signals_match_the_generated_feed_loop(
    symbol: str, indicator: str, params: dict[str, object], _aux_data: None
) -> None:
    """The fast-tier signal arrays equal the same loop driven by NT's own feed values."""
    _skip_without_data(symbol)
    bars = _bars(symbol, "2026-03-02", "2026-03-03")
    reference = _feed_reference(_strategy(symbol, indicator, params, cap=400), bars)

    fast = compute_signals(
        _strategy(symbol, indicator, params, cap=400), [], bars, {"ind": indicator}
    )
    assert np.array_equal(fast.ready, reference.ready)
    assert np.array_equal(fast.long_entry, reference.long_entry)
    assert np.array_equal(fast.short_entry, reference.short_entry)
    assert np.array_equal(fast.long_exit, reference.long_exit)
    assert np.array_equal(fast.short_exit, reference.short_exit)


# ---------------------------------------------------------------------------
# No look-ahead, frame half: the value at bar k ignores every later strategy bar.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("symbol", "indicator", "params"), CASES, ids=CASE_IDS)
def test_bar_frame_prefix_uses_only_the_bars_at_or_before_it(
    symbol: str, indicator: str, params: dict[str, object], _aux_data: None
) -> None:
    """Bar-frame prefix invariance: truncating the frame after bar k leaves bar k's value.

    Only the strategy frame is truncated, so this is invariance to LATER BARS, not to the
    daily bars behind them (the reference series is the same in both calls). The
    daily-bar-level causality is pinned by the synthetic-reference tests below.
    """
    _skip_without_data(symbol)
    bars = _bars(symbol, "2026-03-02", "2026-03-03")
    frame = _frame(bars)
    close = bar_close_ns(_ts_init(bars), MIN)
    full, _ = _asof_column(symbol, indicator, params, frame, close)
    fed = _drive_generated(_strategy(symbol, indicator, params, cap=400), bars)

    for k in (0, 1, 17, 359, 720, 1439):
        truncated, _ = _asof_column(
            symbol, indicator, params, frame.iloc[: k + 1].reset_index(drop=True), close[: k + 1]
        )
        assert int(_bits(truncated)[k]) == int(_bits(full)[k]), f"look-ahead at bar {k}"
        assert int(_bits(truncated)[k]) == int(_bits(fed)[k]), f"NT divergence at bar {k}"


# ---------------------------------------------------------------------------
# No look-ahead, daily-bar half: a synthetic reference with known closes.
#
# Every close is a distinct power of two, so the ROC answer is an exact binary literal
# (2x -> 1.0, /4 -> -0.75, 3x -> 2.0). The answer immediately before, exactly at and
# immediately after a daily close is therefore a literal, and rewriting ONLY future
# daily bars must leave the past bit-identical.
# ---------------------------------------------------------------------------

_SYN_TS0 = 1_704_067_200 * 10**9  # 2024-01-01T00:00:00Z (a daily close boundary)
_SYN_CLOSES = (128.0, 256.0, 512.0, 1024.0, 256.0, 512.0, 1536.0, 384.0)
# BTC_ROC(period 1) at daily close k; None = NaN (the first close has no previous bar).
_SYN_ROC_1: tuple[float | None, ...] = (None, 1.0, 1.0, 1.0, -0.75, 1.0, 2.0, -0.75)


class _SynBars:
    """Identity token of one synthetic daily series (the derived cache keys on it)."""

    __slots__ = ("close", "ts")

    def __init__(self, ts: np.ndarray, close: np.ndarray) -> None:
        self.ts = ts
        self.close = close


def _syn_ts(n: int) -> np.ndarray:
    return np.asarray([_SYN_TS0 + (i + 1) * DAY for i in range(n)], dtype=np.int64)


def _patched_ref(monkeypatch: pytest.MonkeyPatch, ts: np.ndarray, close: np.ndarray) -> None:
    """Serve ``(ts, close)`` as every symbol's daily reference, with fresh cache identity."""

    def fake(symbol: str, tf: str = "1d", refresh: bool = False) -> tuple[object, Any, Any]:
        return _SynBars(ts, close), ts, close

    monkeypatch.setattr(aux_data, "_ref_bars", fake)
    monkeypatch.setattr(aux_data, "_REF", {})
    monkeypatch.setattr(aux_data, "_REF_DERIVED", {})


def _syn_frame(n_rows: int) -> pd.DataFrame:
    """A frame carrying only what a pointwise as-of spec reads: its row count."""
    return pd.DataFrame({"close": np.full(n_rows, 100.0)})


def test_daily_edge_uses_only_the_bar_closed_at_or_before_the_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1ns before / exactly at / 1ns after each daily close answers from THAT close.

    Literal expectations: before close ``k`` the answer is close ``k-1``'s ROC, at and
    after it close ``k``'s. A next-close lookup answers ``k`` / ``k+1`` / ``k+1`` and
    fails every edge; a ``side="left"`` lookup answers ``k-1`` / ``k-1`` / ``k`` and
    fails the boundary it is supposed to include.
    """
    ts = _syn_ts(len(_SYN_CLOSES))
    _patched_ref(monkeypatch, ts, np.asarray(_SYN_CLOSES, dtype=np.float64))

    closes: list[int] = []
    expected: list[float] = []
    for k in (4, 5, 6, 7):
        for delta in (-1, 0, 1):
            closes.append(int(ts[k]) + delta)
            day = k - 1 if delta < 0 else k
            value = _SYN_ROC_1[day]
            expected.append(float("nan") if value is None else value)

    column, first = _asof_column(
        "BTCUSDT", "BTC_ROC", {"period": 1}, _syn_frame(len(closes)), np.asarray(closes)
    )
    assert first == 0
    assert np.array_equal(_bits(column), _bits(np.asarray(expected))), "daily edge"


def test_trend_daily_edge_picks_the_bar_at_the_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BTC_TREND obeys the same rule: before close k it is day k-1's, at/after day k's.

    The trend series is not a binary literal, so the pin is the value the SAME spec
    returns when queried exactly at each daily close: a next-close lookup, a
    ``side="left"`` lookup or a missing ``- 1`` breaks one of these edges.
    """
    ts = _syn_ts(len(_SYN_CLOSES))
    _patched_ref(monkeypatch, ts, np.asarray(_SYN_CLOSES, dtype=np.float64))
    at_close = aux_data.ref_trend_asof("BTCUSDT", ts, 2)

    closes = np.asarray([int(ts[k]) + d for k in (4, 5, 6, 7) for d in (-1, 0, 1)], dtype=np.int64)
    got = aux_data.ref_trend_asof("BTCUSDT", closes, 2)
    expected = np.asarray(
        [at_close[k - 1] if d < 0 else at_close[k] for k in (4, 5, 6, 7) for d in (-1, 0, 1)],
        dtype=np.float64,
    )
    assert len({float(v) for v in at_close[3:]}) > 1, "fixture: consecutive closes must differ"
    assert np.array_equal(_bits(got), _bits(expected)), "trend daily edge"


def test_past_values_uses_only_daily_bars_at_or_before_the_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rewriting every daily bar AFTER close 4 leaves closes 1..4 bit-identical.

    The probe must bite: rewriting a PAST bar does move the answers at and after it, and
    that moved answer is the literal of the new closes (2x -> 1.0).
    """
    ts = _syn_ts(len(_SYN_CLOSES))
    closes = np.asarray(_SYN_CLOSES, dtype=np.float64)
    close_ns = np.asarray(
        [int(ts[k]) + d for k in (1, 2, 3, 4) for d in (-1, 0, 1)], dtype=np.int64
    )

    _patched_ref(monkeypatch, ts, closes)
    before_roc, _ = _asof_column("BTCUSDT", "BTC_ROC", {"period": 1}, _syn_frame(len(close_ns)), close_ns)
    before_trend, _ = _asof_column("BTCUSDT", "BTC_TREND", {"period": 2}, _syn_frame(len(close_ns)), close_ns)

    future = closes.copy()
    future[5:] = 1e9
    _patched_ref(monkeypatch, ts, future)
    after_roc, _ = _asof_column("BTCUSDT", "BTC_ROC", {"period": 1}, _syn_frame(len(close_ns)), close_ns)
    after_trend, _ = _asof_column("BTCUSDT", "BTC_TREND", {"period": 2}, _syn_frame(len(close_ns)), close_ns)

    assert np.array_equal(_bits(after_roc), _bits(before_roc)), "a future daily bar moved BTC_ROC"
    assert np.array_equal(_bits(after_trend), _bits(before_trend)), "a future daily bar moved the trend"

    past = closes.copy()
    past[3] = 128.0
    _patched_ref(monkeypatch, ts, past)
    moved_roc, _ = _asof_column("BTCUSDT", "BTC_ROC", {"period": 1}, _syn_frame(len(close_ns)), close_ns)
    moved_trend, _ = _asof_column("BTCUSDT", "BTC_TREND", {"period": 2}, _syn_frame(len(close_ns)), close_ns)

    assert float(moved_roc[10]) == 1.0, "stale cache: a past daily bar did not reach the answer"
    assert not np.array_equal(_bits(moved_roc), _bits(before_roc)), "the probe does not bite"
    assert not np.array_equal(_bits(moved_trend), _bits(before_trend)), "the probe does not bite"


# ---------------------------------------------------------------------------
# A stale daily bar is a NaN VALUE (never held) on both paths.
# ---------------------------------------------------------------------------


def test_stale_daily_bar_is_nan_and_not_held(
    _aux_data: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 3-day reference hole makes a NaN stretch that the fast tier writes, not holds.

    Real 1m bars over 4 days, a synthetic daily series whose closes sit 20 and 10 days
    before the run, then inside it, then after a 3-day hole. Every bar before the run's
    first in-run close sees a 10-day-old bar (stale, NaN), the next two days see that
    close's value, the hole is NaN again, and the far close's value is the last answer.
    The generated feed (rolling buffer, same helper) must agree bit for bit, and no entry
    may fire on a NaN bar.
    """
    _skip_without_data("ZECUSDT")
    bars = _bars("ZECUSDT", "2026-03-02", "2026-03-06")
    close = bar_close_ns(_ts_init(bars), MIN)
    assert len(bars) == 4 * 1440

    first_close = int(close[60])  # a daily boundary inside the run
    second_close = first_close + 3 * DAY
    ts = np.asarray(
        [first_close - 20 * DAY, first_close - 10 * DAY, first_close, second_close], dtype=np.int64
    )
    _patched_ref(monkeypatch, ts, np.asarray([1024.0, 2048.0, 512.0, 256.0], dtype=np.float64))

    column, first = _asof_column(
        "ZECUSDT", "BTC_ROC", {"period": 1}, _frame(bars), close
    )
    assert first == 0

    nan_rows = np.flatnonzero(np.isnan(column))
    expected_nan = (close < first_close) | (
        (close > first_close + 2 * DAY) & (close < second_close)
    )
    assert nan_rows.tolist() == np.flatnonzero(expected_nan).tolist()
    assert nan_rows.size > 0 and nan_rows.size < len(close), "fixture: no NaN stretch"
    in_run = (close >= first_close) & (close <= first_close + 2 * DAY)
    assert set(np.unique(column[in_run]).tolist()) == {-0.75}, "fixture: literal in-run answer"
    assert float(column[close >= second_close][0]) == -0.5, "fixture: literal far answer"

    strategy = _strategy("ZECUSDT", "BTC_ROC", {"period": 1}, cap=400)
    fed = _drive_generated(strategy, bars)
    assert np.array_equal(_bits(fed), _bits(column)), "NT holds a NaN the fast tier does not"

    reference = _feed_reference(_strategy("ZECUSDT", "BTC_ROC", {"period": 1}, cap=400), bars)
    assert reference.long_entry[nan_rows].sum() == 0, "an entry fired on a NaN regime value"
    fast = compute_signals(
        _strategy("ZECUSDT", "BTC_ROC", {"period": 1}, cap=400), [], bars, {"ind": "BTC_ROC"}
    )
    assert np.array_equal(fast.long_entry, reference.long_entry)
    assert np.array_equal(fast.ready, reference.ready)


# ---------------------------------------------------------------------------
# End to end: fills, metrics and every other result key equal to NT, zero fallbacks.
# ---------------------------------------------------------------------------


def _genome(indicator: str, params: dict[str, float], threshold: float) -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    return StrategyChromosome(
        entry_genes=[StrategyGene(indicator, params, ConditionType.GT, threshold)],
        exit_genes=[StrategyGene("WILLR", {"period": 14.0}, ConditionType.GT, -20.0)],
        stop_loss_pct=0.8,
        take_profit_pct=1.0,
        direction=Direction.LONG,
    )


E2E = [
    ("ZECUSDT", _genome("BTC_TREND", {"period": 200.0}, -0.24)),
    ("ZECUSDT", _genome("OWN_TREND", {"period": 200.0}, -0.23)),
    ("ZECUSDT", _genome("BTC_ROC", {"period": 20.0}, 0.0)),
]
E2E_IDS = ["ZEC_BTC_TREND", "ZEC_OWN_TREND", "ZEC_BTC_ROC"]

METRICS = ("total_trades", "total_return", "sharpe_ratio", "profit_factor", "max_drawdown")


def _fills(positions: list[Any]) -> list[tuple[Any, ...]]:
    out = []
    for pos in sorted(positions, key=lambda p: int(p.ts_opened)):
        for f in pos.events:
            out.append(
                (
                    int(f.ts_event),
                    f.order_side.name,
                    f.order_type.name,
                    str(f.last_qty),
                    str(f.last_px),
                    str(f.commission),
                )
            )
    return out


@pytest.mark.usefixtures("_aux_data")
@pytest.mark.parametrize(("symbol", "genome"), E2E, ids=E2E_IDS)
def test_genome_matches_nt_with_no_fallback(
    symbol: str, genome: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fills, every result value and MTM bytes equal NT's; no NT fallback recorded."""
    _skip_without_data(symbol)
    from vibe_quant.screening import nt_runner
    from vibe_quant.validation.extraction import all_positions

    captured: dict[str, list[Any]] = {}
    original = nt_runner.NTScreeningRunner._extract_metrics

    def grab(self: Any, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        captured["positions"] = all_positions(engine)
        return original(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "_extract_metrics", grab)

    fallbacks.clear()
    fast_fn = FastBacktestFn([symbol], "1m", "2026-03-02", "2026-03-06", collect_daily_returns=True)
    fast = fast_fn._run_single(genome, "2026-03-02", "2026-03-06")
    fast_fills = _fills(fast_fn.last_positions)

    nt_fn = fast_fn.nt_twin()
    captured.clear()
    nt = nt_fn._run_single(genome, "2026-03-02", "2026-03-06")
    nt_fills = _fills(captured.get("positions", []))

    assert fallbacks.summary() == {"total": 0, "by_reason": {}}
    assert nt["total_trades"] > 0
    assert nt_fills == fast_fills
    assert {m: nt[m] for m in METRICS} == {m: fast[m] for m in METRICS}
    assert fast.keys() == nt.keys()
    for key in fast:
        if key == "mtm_daily_returns":
            continue
        assert repr(fast[key]) == repr(nt[key]), key
    a = fast["mtm_daily_returns"]
    b = nt["mtm_daily_returns"]
    assert a.first_day == b.first_day
    assert a.values.tobytes() == b.values.tobytes()
