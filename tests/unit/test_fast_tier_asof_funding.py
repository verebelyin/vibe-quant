"""FUNDING / FUNDING_Z on the fast tier: whole-frame columns equal the generated NT feed.

Both specs answer from the funding archive for the bar's OWN close time only, so the
fast tier computes one column over the whole run frame instead of replaying the rolling
buffer (``IndicatorSpec.asof_context``). Every comparison here is against the REAL
generated ``_feed_pta_buffer`` + ``_update_pta_indicators`` (NT's source of truth),
bitwise and NaN included, on an 8h-funding symbol (ZECUSDT) and a 4h-funding symbol
(HYPEUSDT) whose settlement timestamps carry sub-second offsets.

Failing before the flag flip: ``asof_context_columns`` skips unflagged specs (empty
column list) and the end-to-end genomes fall back to NT (one recorded fallback).
"""

from __future__ import annotations

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
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nautilus_trader.model.data import Bar

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
MIN = 60_000_000_000

FUNDING_SPECS = ("FUNDING", "FUNDING_Z")


def _instrument_id(symbol: str) -> str:
    return f"{symbol}-PERP.BINANCE"


def _bar_type(symbol: str) -> str:
    return f"{_instrument_id(symbol)}-1-MINUTE-LAST-EXTERNAL"


def _skip_without_data(symbol: str) -> None:
    """Skip (never fail) when this symbol's 1m catalog bars or the archive are absent."""
    if not (CATALOG / "data" / "bar" / _bar_type(symbol)).is_dir():
        pytest.skip(f"needs the {symbol} 1m catalog")
    if not ARCHIVE.is_file():
        pytest.skip("needs the funding/klines archive")


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
            "name": "funding_asof",
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


def _asof_column(
    symbol: str,
    indicator: str,
    params: dict[str, object],
    frame: pd.DataFrame,
    close: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """The as-of column of ``ind`` over ``frame`` (and the frame's own close times)."""
    if close is None:
        raise AssertionError("close times are required")
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


def _feed_reference(strategy: Any, bars: list[Bar]) -> tuple[Signals, np.ndarray]:
    """One pass of ``compute_signals``' loop with the values the generated feed writes.

    Returns the signals and ``_pta_values["ind"]`` after each bar's REAL generated
    ``_feed_pta_buffer`` (NT's own values, NaN included).
    """
    n = len(bars)
    ready = np.zeros(n, dtype=bool)
    le = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    fed = np.empty(n, dtype=np.float64)
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
        fed[i] = strategy._pta_values["ind"]
        strategy._last_close = float(bar.close)
        if not strategy._indicators_ready():
            continue
        ready[i] = True
        for out, check in zip((le, se, lx, sx), checks, strict=True):
            if check is not None:
                out[i] = check(bar)
        strategy._update_prev_values(bar)
    return Signals(ready, le, se, lx, sx), fed


CASES = [
    ("ZECUSDT", "FUNDING", {"period": 6}),
    ("ZECUSDT", "FUNDING_Z", {"period": 30}),
    ("HYPEUSDT", "FUNDING", {"period": 6}),
    ("HYPEUSDT", "FUNDING_Z", {"period": 30}),
]
CASE_IDS = ["ZEC_FUNDING_8h", "ZEC_FUNDING_Z_8h", "HYPE_FUNDING_4h", "HYPE_FUNDING_Z_4h"]


# ---------------------------------------------------------------------------
# The flag itself: both funding specs opt in, the regime specs stay out.
# ---------------------------------------------------------------------------


def test_funding_specs_are_flagged_asof_context() -> None:
    """FUNDING/FUNDING_Z are pointwise as-of context specs."""
    for name in FUNDING_SPECS:
        spec = indicator_registry.get(name)
        assert spec is not None
        assert spec.asof_context is True
        assert spec.needs_context is True
        assert spec.output_names == ("value",)
        assert spec.compute_fn is not None


# ---------------------------------------------------------------------------
# Per-bar equality and the bar-frame half of no look-ahead, from ONE generated-feed pass.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("symbol", "indicator", "params"), CASES, ids=CASE_IDS)
def test_row_matches_the_generated_feed(
    symbol: str, indicator: str, params: dict[str, object], _aux_data: None
) -> None:
    """1440 bars with cap 400 (trims run): column, signals and prefix values equal NT's feed.

    Same float bits on every bar, NaN included; the fast-tier signal arrays equal the same
    loop driven by NT's own feed values; and truncating the OHLCV frame after bar k leaves
    bar k's value. Only the frame is truncated, so that last check is invariance to LATER
    BARS, not to the settlements behind them (the archive is the same in both calls): a
    lookup that answered the next settlement would pass all of it. The settlement-level
    causality is pinned by the synthetic-archive tests below.
    """
    _skip_without_data(symbol)
    bars = _bars(symbol, "2026-03-02", "2026-03-03")
    assert len(bars) == 1440
    reference, fed = _feed_reference(_strategy(symbol, indicator, params, cap=400), bars)

    frame = _frame(bars)
    close = bar_close_ns(_ts_init(bars), MIN)
    column, first = _asof_column(symbol, indicator, params, frame, close)
    assert first == 0
    assert np.array_equal(_bits(fed), _bits(column)), "as-of column diverged from NT's feed"
    assert int(np.isnan(column).sum()) == 0

    fast = compute_signals(
        _strategy(symbol, indicator, params, cap=400), [], bars, {"ind": indicator}
    )
    assert np.array_equal(fast.ready, reference.ready)
    assert np.array_equal(fast.long_entry, reference.long_entry)
    assert np.array_equal(fast.short_entry, reference.short_entry)
    assert np.array_equal(fast.long_exit, reference.long_exit)
    assert np.array_equal(fast.short_exit, reference.short_exit)

    for k in (0, 1, 17, 359, 720, 1439):
        truncated, _ = _asof_column(
            symbol,
            indicator,
            params,
            frame.iloc[: k + 1].reset_index(drop=True),
            close[: k + 1],
        )
        assert int(_bits(truncated)[k]) == int(_bits(column)[k]), f"look-ahead at bar {k}"
        assert int(_bits(truncated)[k]) == int(_bits(fed)[k]), f"NT divergence at bar {k}"


# ---------------------------------------------------------------------------
# Settlement-level causality on a synthetic archive: literal answers at each edge.
#
# The real-data tests above cannot see a next-settlement lookup: the archive (and the
# generated reference, which reads the same helper) never changes between calls. Here
# every rate is distinct and known, so the answer immediately before, exactly at and
# immediately after a settlement is a literal, and rewriting ONLY future rows must
# leave the past bit-identical.
# ---------------------------------------------------------------------------

SYN = "SYNFUND"
H = 3_600 * 1_000_000_000
MS = 1_000_000
SYN_BASE = 1_704_067_200 * 10**9  # 2024-01-01T00:00:00Z
# Non-linear and all distinct. Each rate is n/8192, so ``rate * 1e4`` (the FUNDING bps
# answer) is exact and written out literally below.
SYN_N = (3, 11, 5, 17, 7, 23, 2, 13, 29, 6, 19, 4, 27, 9)
# 6 settlements 8h apart, then 8 more 4h apart: an 8h-to-4h interval switch mid-run.
SYN_HOURS = (8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4)
# FUNDING (period 1) answer, in bps, once settlement k has settled.
SYN_BPS = (
    3.662109375,
    13.427734375,
    6.103515625,
    20.751953125,
    8.544921875,
    28.076171875,
    2.44140625,
    15.869140625,
    35.400390625,
    7.32421875,
    23.193359375,
    4.8828125,
    32.958984375,
    10.986328125,
)
# FUNDING_Z (period 4) answer at settlement k; NaN (None) until 4 have settled.
SYN_Z: tuple[float | None, ...] = (
    None,
    None,
    None,
    1.2649110640673518,
    -0.5669467095138409,
    1.1785113019775793,
    -1.0789473684210527,
    0.19374804358299202,
    1.0343908107558353,
    -0.5461092327709238,
    0.23094644374201817,
    -0.8949008088202394,
    1.1900424855098393,
    -0.5595902774449283,
)


def _syn_settlements() -> list[int]:
    """Settlement times (ns) of the synthetic archive, ascending.

    Each settlement is spaced by the interval in effect at the NEXT one, so the switch
    from 8h to 4h rows happens at the first 4h settlement (see ``SYN_HOURS``).
    """
    times = [SYN_BASE]
    for hours in SYN_HOURS[1:]:
        times.append(times[-1] + hours * H)
    return times


def _syn_archive(path: Path, rates: list[float]) -> Path:
    """A funding archive holding ``rates`` at :func:`_syn_settlements` (intervals included)."""
    from vibe_quant.data.archive import RawDataArchive

    archive = RawDataArchive(path)
    try:
        for ts, rate, hours in zip(_syn_settlements(), rates, SYN_HOURS, strict=True):
            archive.conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate, source) "
                "VALUES (?, ?, ?, 'test')",
                (SYN, ts // MS, rate),
            )
            archive.conn.execute(
                "INSERT INTO raw_funding_intervals "
                "(symbol, funding_time, interval_hours, source) VALUES (?, ?, ?, 'test')",
                (SYN, ts // MS, hours),
            )
        archive.conn.commit()
    finally:
        archive.close()
    return path


@pytest.fixture
def syn_archive(tmp_path: Path) -> Iterator[Path]:
    """The synthetic archive configured for aux lookups, with the rate caches cleared."""
    clear_rate_cache()
    path = _syn_archive(tmp_path / "synthetic_funding.db", [n / 8192.0 for n in SYN_N])
    aux_data.configure(path, None)
    try:
        yield path
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def _rewrite_after(path: Path, after_ns: int, rate: float) -> None:
    """Replace every rate settled after ``after_ns`` and drop the funding caches."""
    from vibe_quant.data.archive import RawDataArchive

    archive = RawDataArchive(path)
    try:
        archive.conn.execute(
            "UPDATE raw_funding_rates SET funding_rate = ? WHERE symbol = ? AND funding_time > ?",
            (rate, SYN, after_ns // MS),
        )
        archive.conn.commit()
    finally:
        archive.close()
    clear_rate_cache()


def _syn_frame(n_rows: int) -> pd.DataFrame:
    """A frame carrying only what an as-of context spec reads: its row count."""
    return pd.DataFrame({"close": np.arange(n_rows, dtype=np.float64) + 100.0})


def test_settlement_edges_uses_only_settlements_at_or_before_the_close(
    syn_archive: Path,
) -> None:
    """1ns before / exactly at / 1ns after each settlement answers from THAT settlement.

    Literal expectations over a distinct non-linear archive: before settlement ``k`` the
    answer is settlement ``k-1``'s, at and after it settlement ``k``'s. A next-settlement
    lookup (``side="right"`` without the ``- 1``) answers ``k`` / ``k+1`` / ``k+1`` and
    fails all three edges, over an 8h-to-4h switch and both NaN warmups.
    """
    settlements = _syn_settlements()
    closes = np.asarray([t + d for t in settlements for d in (-1, 0, 1)], dtype=np.int64)

    # Fixture shape, so the run really crosses the interval switch it claims to: 8h
    # settlements then 4h, and the archive's own interval column says the same.
    gaps = np.diff(np.asarray(settlements, dtype=np.int64))
    assert gaps.tolist() == [8 * H] * 5 + [4 * H] * 8, "fixture: no 8h-to-4h switch"
    assert FundingCalculator(syn_archive)._load_intervals(SYN) == [8] * 6 + [4] * 8

    funding, first = _asof_column(SYN, "FUNDING", {"period": 1}, _syn_frame(len(closes)), closes)
    zed, _ = _asof_column(SYN, "FUNDING_Z", {"period": 4}, _syn_frame(len(closes)), closes)

    expected_funding: list[float] = []
    expected_z: list[float] = []
    for k in range(len(settlements)):
        for settled in (k - 1, k, k):
            expected_funding.append(SYN_BPS[settled] if settled >= 0 else float("nan"))
            z = SYN_Z[settled] if settled >= 0 else None
            expected_z.append(float("nan") if z is None else z)

    assert first == 0
    assert np.array_equal(_bits(funding), _bits(np.asarray(expected_funding))), "FUNDING edge"
    assert np.array_equal(_bits(zed), _bits(np.asarray(expected_z))), "FUNDING_Z edge"


EDGE_SYM = "SYNFUNDEDGE"
# 2024-03-06T08:00:00Z, on an 8h UTC boundary (ms multiples of 8h since the epoch).
EDGE_BASE = 1_709_712_000 * 10**9
EDGE_PERIOD = 8 * H
SEC = 1_000_000_000


def _edge_archive(path: Path, rows: list[tuple[int, float]]) -> Path:
    """A funding archive with exactly ``rows`` as (archived ns, rate)."""
    from vibe_quant.data.archive import RawDataArchive

    archive = RawDataArchive(path)
    try:
        for ts, rate in rows:
            archive.conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate, source) "
                "VALUES (?, ?, ?, 'test')",
                (EDGE_SYM, ts // MS, rate),
            )
        archive.conn.commit()
    finally:
        archive.close()
    return path


def test_settlement_times_are_snapped_to_the_boundary_within_the_tolerance(
    tmp_path: Path,
) -> None:
    """Archived settlement ms jitter within +-60s is snapped to the 8h boundary.

    Five 8h-spaced settlements with distinct rates, each archived off its boundary by a
    different amount: on time (the previous rate is a number), +1s, -1s, +60s (the
    inclusive tolerance edge) and +61s (outside it). A dropped snap sees the +1s / -1s
    rows late (the previous rate at the boundary); a strict ``<`` snap loses the +60s
    row at the boundary. The +61s row is never snapped: invisible at its boundary,
    visible 61s later.
    """
    b = [EDGE_BASE + k * EDGE_PERIOD for k in range(5)]
    rows = [
        (b[0], 0.0001),
        (b[1] + 1 * SEC, 0.0002),
        (b[2] - 1 * SEC, 0.0003),
        (b[3] + 60 * SEC, 0.0004),
        (b[4] + 61 * SEC, 0.0005),
    ]
    assert all(t % EDGE_PERIOD == 0 for t in b), "fixture: boundaries are 8h-aligned"

    clear_rate_cache()
    path = _edge_archive(tmp_path / "edge_funding.db", rows)
    aux_data.configure(path, None)
    try:
        closes = np.asarray(
            [
                b[1],  # +1s row snapped BACK to b[1]: visible exactly at the boundary
                b[2] - 1,  # -1s row snapped to b[2]: still 1ns in the future
                b[2],  # ... and visible at b[2]
                b[3],  # +60s row: tolerance is inclusive
                b[4],  # +61s row is not snapped and not yet visible
                b[4] + 61 * SEC,  # ... visible at its own archived time
            ],
            dtype=np.int64,
        )
        got = aux_data.funding_asof(EDGE_SYM, closes, 1)
        expected = [0.0002, 0.0002, 0.0003, 0.0004, 0.0004, 0.0005]
        assert np.array_equal(_bits(got), _bits(np.asarray(expected))), (
            f"snapped as-of answers {got.tolist()} != {expected}"
        )
    finally:
        aux_data.configure(None, None)
        clear_rate_cache()


def test_past_values_uses_only_settlements_at_or_before_the_close(syn_archive: Path) -> None:
    """Rewriting every settlement AFTER settlement 7 leaves the past bit-identical.

    The closes end exactly on settlement 7 (and 1ns after it), so only settlement 8
    onwards is rewritten: a next-settlement lookup reads that rewritten row there and
    diverges. The second half proves the probe bites — rewriting a PAST row does change
    the answer, to its literal replacement.
    """
    settlements = _syn_settlements()
    closes = np.asarray([t + d for t in settlements[:8] for d in (-1, 0, 1)], dtype=np.int64)
    frame = _syn_frame(len(closes))
    before_f, _ = _asof_column(SYN, "FUNDING", {"period": 1}, frame, closes)
    before_z, _ = _asof_column(SYN, "FUNDING_Z", {"period": 4}, frame, closes)
    assert float(before_f[-1]) == SYN_BPS[7], "fixture: last close is on settlement 7"

    _rewrite_after(syn_archive, int(settlements[7]), 0.5)
    after_f, _ = _asof_column(SYN, "FUNDING", {"period": 1}, frame, closes)
    after_z, _ = _asof_column(SYN, "FUNDING_Z", {"period": 4}, frame, closes)
    assert np.array_equal(_bits(after_f), _bits(before_f)), "a future settlement moved FUNDING"
    assert np.array_equal(_bits(after_z), _bits(before_z)), "a future settlement moved FUNDING_Z"

    _rewrite_after(syn_archive, int(closes[0]), 0.5)
    moved_f, _ = _asof_column(SYN, "FUNDING", {"period": 1}, frame, closes)
    moved_z, _ = _asof_column(SYN, "FUNDING_Z", {"period": 4}, frame, closes)
    assert float(moved_f[1]) == 5000.0, "stale rate cache: a past row did not reach the answer"
    assert not np.isnan(before_z[12]) and np.isnan(moved_z[12]), "stale derived cache"


# ---------------------------------------------------------------------------
# A NaN funding answer is a value (never held) on both paths.
# ---------------------------------------------------------------------------


def test_nan_funding_value_is_written_not_held(
    _aux_data: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hole in the funding series makes 5 NaN bars: written as NaN, no entry on them.

    The hole is a function of the bar close time, so the generated feed (which sees a
    rolling buffer slice) and the whole-frame column must agree bar for bar.
    """
    _skip_without_data("ZECUSDT")
    bars = _bars("ZECUSDT", "2026-03-02", "2026-03-03")
    close = bar_close_ns(_ts_init(bars), MIN)
    hole_from = int(close[100])
    hole_to = int(close[105])
    real = aux_data.funding_z_asof

    def holed(symbol: str, bar_close_ns: np.ndarray, period: int) -> np.ndarray:
        out = np.array(real(symbol, bar_close_ns, period), dtype=np.float64)
        times = np.asarray(bar_close_ns, dtype=np.int64)
        out[(times >= hole_from) & (times < hole_to)] = np.nan
        return out

    monkeypatch.setattr(aux_data, "funding_z_asof", holed)
    params: dict[str, object] = {"period": 30}
    frame = _frame(bars)
    column, first = _asof_column("ZECUSDT", "FUNDING_Z", params, frame, close)

    nan_rows = np.flatnonzero(np.isnan(column))
    assert nan_rows.tolist() == list(range(100, 105))
    assert first == 0
    assert np.isnan(column[96:100]).sum() == 0, "a NaN must not extend past the hole"

    reference, fed = _feed_reference(_strategy("ZECUSDT", "FUNDING_Z", params, cap=400), bars)
    assert np.array_equal(_bits(fed), _bits(column)), "NT holds a NaN the fast tier does not"
    assert reference.long_entry[nan_rows].sum() == 0

    fast = compute_signals(
        _strategy("ZECUSDT", "FUNDING_Z", params, cap=400), [], bars, {"ind": "FUNDING_Z"}
    )
    assert np.array_equal(fast.ready, reference.ready)
    assert np.array_equal(fast.long_entry, reference.long_entry)
    assert np.array_equal(fast.long_exit, reference.long_exit)


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
    ("ZECUSDT", _genome("FUNDING", {"period": 6.0}, 0.0)),
    ("ZECUSDT", _genome("FUNDING_Z", {"period": 30.0}, -1.0)),
    ("HYPEUSDT", _genome("FUNDING", {"period": 6.0}, 0.0)),
    ("HYPEUSDT", _genome("FUNDING_Z", {"period": 30.0}, -1.0)),
]
E2E_IDS = ["ZEC_FUNDING_8h", "ZEC_FUNDING_Z_8h", "HYPE_FUNDING_4h", "HYPE_FUNDING_Z_4h"]

# One day (3 settlements at 8h, 6 at 4h): every genome trades on 2026-03-06 (112 to 145 trades).
E2E_START, E2E_END = "2026-03-06", "2026-03-07"
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
    fast_fn = FastBacktestFn([symbol], "1m", E2E_START, E2E_END, collect_daily_returns=True)
    fast = fast_fn._run_single(genome, E2E_START, E2E_END)
    fast_fills = _fills(fast_fn.last_positions)

    nt_fn = fast_fn.nt_twin()
    captured.clear()
    nt = nt_fn._run_single(genome, E2E_START, E2E_END)
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
