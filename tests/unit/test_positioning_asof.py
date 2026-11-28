"""Positioning as-of lookups: OI_CHANGE, LS_RATIO_Z (Binance Vision ``metrics``) and DVOL_Z
(Deribit DVOL candles).

Fixtures (S0 = 2024-01-01 00:00 UTC, BTCUSDT, closes given as HH:MM on S0's day):
- A: rows k = -24..60 at S0 + 5m*k, ``sum_open_interest`` x_k = 1000 + k.
- B: rows k = -1..3, ``sum_toptrader_long_short_ratio`` y = 1.0, 1.0, 2.0, 3.0, 7.0.
- D: BTC DVOL hourly candles opened 00:00, 01:00, 02:00, 03:00 with close 50, 52, 54, 58.

A metrics row stamped ``s`` is usable from close ``s + 600 s`` and stays usable while the
close is less than 600 s past that; a DVOL candle opened at ``o`` is usable from close
``o + 3600 s`` (its end) while the close is less than 7200 s past that. Every value is read
four ways -- ``aux_data.*_asof`` (vector),
``aux_data.*_at`` (scalar), the plugin ``compute_fn`` and its ``asof_scalar`` -- and all
four must agree bit for bit.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import vibe_quant.dsl.plugin_loader  # noqa: F401  (registers plugin indicators)
from vibe_quant.data.archive import MetricsFileMeta, MetricsRow, RawDataArchive
from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError
from vibe_quant.dsl.indicators import indicator_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

S0 = datetime(2024, 1, 1, tzinfo=UTC)
SYMBOL = "BTCUSDT"
FIVE_MIN = timedelta(minutes=5)

# (create_time, sum_open_interest, sum_toptrader_long_short_ratio)
Row = tuple[datetime, float | None, float | None]
# (open time, close) of one hourly DVOL candle
Candle = tuple[datetime, float]
Dvol = dict[str, list[Candle]]


def _ms(t: datetime) -> int:
    return int(t.timestamp()) * 1000


def _close(hhmm: str) -> int:
    """Close time in ns of HH:MM on S0's day."""
    hours, minutes = (int(p) for p in hhmm.split(":"))
    return _ms(S0 + timedelta(hours=hours, minutes=minutes)) * 1_000_000


def _write(path: Path, rows: list[Row], symbol: str = SYMBOL, dvol: Dvol | None = None) -> Path:
    """An archive holding ``rows`` as ``ok`` metrics files (one per UTC day) and ``dvol``
    hourly candles per currency."""
    by_day: dict[date, list[MetricsRow]] = {}
    for t, oi, ratio in rows:
        by_day.setdefault(t.date(), []).append((_ms(t), oi, None, None, ratio, None, None))
    archive = RawDataArchive(path)
    try:
        for day, day_rows in sorted(by_day.items()):
            meta = MetricsFileMeta(
                filename=f"{symbol}-metrics-{day}.zip",
                md5=f"md5-{symbol}-{day}",
                sha256="0" * 64,
                size=1,
                vision_last_modified=None,
                csv_rows=len(day_rows),
                status="ok",
                detail=None,
            )
            archive.insert_metrics_file(symbol, day, meta, day_rows)
        for currency, candles in (dvol or {}).items():
            archive.insert_dvol(
                currency, 3600, [(_ms(t), c, c, c, c) for t, c in candles], "deribit"
            )
    finally:
        archive.close()
    return path


def _fixture_d(*, drop: tuple[int, ...] = ()) -> Dvol:
    closes = (50.0, 52.0, 54.0, 58.0)
    hour = timedelta(hours=1)
    return {"BTC": [(S0 + h * hour, c) for h, c in enumerate(closes) if h not in drop]}


def _fixture_a(*, drop: tuple[int, ...] = (), x0: float = 1000.0) -> list[Row]:
    rows: list[Row] = []
    for k in range(-24, 61):
        if k in drop:
            continue
        rows.append((S0 + k * FIVE_MIN, x0 if k == 0 else 1000.0 + k, None))
    return rows


def _fixture_b(
    ys: tuple[float | None, ...] = (1.0, 1.0, 2.0, 3.0, 7.0), *, drop: tuple[int, ...] = ()
) -> list[Row]:
    return [
        (S0 + k * FIVE_MIN, 1000.0 + k, y)
        for k, y in zip(range(-1, 4), ys, strict=True)
        if k not in drop
    ]


@pytest.fixture
def use_archive(tmp_path: Path) -> Iterator[Callable[..., Path]]:
    """Write rows (and DVOL candles) to a fresh tmp archive and point aux lookups at it."""
    count = 0

    def use(rows: list[Row], dvol: Dvol | None = None) -> Path:
        nonlocal count
        count += 1
        path = _write(tmp_path / f"raw_data_{count}.db", rows, dvol=dvol)
        aux_data.configure(path, None)
        return path

    try:
        yield use
    finally:
        aux_data.configure(None, None)


def _bits(value: float) -> int:
    return int(np.float64(value).view(np.int64))


_READERS: dict[str, tuple[Callable[..., np.ndarray], Callable[..., float]]] = {
    "OI_CHANGE": (aux_data.oi_change_asof, aux_data.oi_change_at),
    "LS_RATIO_Z": (aux_data.ls_ratio_z_asof, aux_data.ls_ratio_z_at),
    "DVOL_Z": (aux_data.dvol_z_asof, aux_data.dvol_z_at),
}


def _value(name: str, hhmm: str, period: int, symbol: str = SYMBOL) -> float:
    """The indicator at one close, after asserting its four read paths agree bitwise."""
    close_ns = _close(hhmm)
    asof, at = _READERS[name]
    vec = float(asof(symbol, np.array([close_ns], dtype=np.int64), period)[0])
    scalar = at(symbol, close_ns, period)
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    df = pd.DataFrame({c: [1.0] for c in ("open", "high", "low", "close", "volume")})
    df.attrs["bar_close_ns"] = np.array([close_ns], dtype=np.int64)
    instrument = f"{symbol}-PERP.BINANCE"
    df.attrs["symbol"] = instrument
    frame_value = float(spec.compute_fn(df, {"period": period}).iloc[-1])
    plugin_scalar = spec.compute_fn.asof_scalar(instrument, close_ns, {"period": period})  # type: ignore[attr-defined]
    got = {_bits(v) for v in (vec, scalar, frame_value, plugin_scalar)}
    assert len(got) == 1, (name, hhmm, period, vec, scalar, frame_value, plugin_scalar)
    return scalar


def _same(actual: float, expected: float) -> bool:
    if math.isnan(expected):
        return math.isnan(actual)
    return _bits(actual) == _bits(expected)


# ---------------------------------------------------------------------------
# R1-R4: the 600 s publication lag and the one-missing-row staleness limit.
# ---------------------------------------------------------------------------


def test_row_usable_only_after_lag(use_archive: Callable[..., Path]) -> None:
    """R1/R2: a row stamped s is first used at close s + 600 s (equality usable)."""
    use_archive(_fixture_a())
    r1 = _value("OI_CHANGE", "00:09", 1)
    assert _same(r1, 999 / 998 - 1), r1  # as-of row 23:55
    assert r1 == 0.0010020040080160886
    for hhmm, expected in (
        ("00:10", 1000 / 999 - 1),
        ("00:14", 1000 / 999 - 1),
        ("00:15", 1001 / 1000 - 1),
    ):
        got = _value("OI_CHANGE", hhmm, 1)
        assert _same(got, expected), (hhmm, got)


def test_gap_limit_one_row(use_archive: Callable[..., Path]) -> None:
    """R3/R4: one missing row is bridged; at age 600 s past usable the value is NaN."""
    use_archive(_fixture_a(drop=(2,)))  # minus row 00:10
    for hhmm in ("00:20", "00:24"):
        got = _value("OI_CHANGE", hhmm, 1)
        assert _same(got, 1001 / 1000 - 1), (hhmm, got)  # as-of row 00:05

    use_archive(_fixture_a(drop=(2, 3)))  # minus rows 00:10, 00:15
    assert _same(_value("OI_CHANGE", "00:24", 1), 1001 / 1000 - 1)
    assert _same(_value("OI_CHANGE", "00:25", 1), math.nan)


# ---------------------------------------------------------------------------
# R5-R6b: OI_CHANGE endpoints.
# ---------------------------------------------------------------------------


def test_oi_change_endpoints(use_archive: Callable[..., Path]) -> None:
    """R5/R6: x[i] / x[row stamped ts[i] - p*5m] - 1; NaN if that row is absent or <= 0."""
    use_archive(_fixture_a())
    assert _same(_value("OI_CHANGE", "01:09", 12), 1011 / 999 - 1)
    assert _same(_value("OI_CHANGE", "01:10", 12), 1012 / 1000 - 1)

    use_archive(_fixture_a(drop=(0,)))
    assert _same(_value("OI_CHANGE", "01:10", 12), math.nan)
    use_archive(_fixture_a(x0=0.0))
    assert _same(_value("OI_CHANGE", "01:10", 12), math.nan)


def test_oi_change_endpoint_rule(use_archive: Callable[..., Path]) -> None:
    """R6b: holes between the endpoints do not matter; a missing endpoint does."""
    use_archive(_fixture_a(drop=(2,)))  # minus row 00:10
    two = _value("OI_CHANGE", "00:25", 2)
    assert _same(two, 1003 / 1001 - 1)
    assert two == 0.0019980019980019303
    assert _same(_value("OI_CHANGE", "00:25", 1), math.nan)


# ---------------------------------------------------------------------------
# R7-R9: LS_RATIO_Z strict windows.
# ---------------------------------------------------------------------------


def test_ls_ratio_z_contiguous(use_archive: Callable[..., Path]) -> None:
    """R7/R8: z over p contiguous rows ending at the as-of row; NaN over a hole or flat window."""
    use_archive(_fixture_b())
    z19 = _value("LS_RATIO_Z", "00:19", 3)
    assert _same(z19, 2 / math.sqrt(3)), z19  # window 1, 1, 2
    assert z19 == 1.1547005383792517
    assert _bits(_value("LS_RATIO_Z", "00:20", 3)) == _bits(1.0)  # window 1, 2, 3

    use_archive(_fixture_b(drop=(1,)))  # minus row 00:05: the window spans 15 m
    assert _same(_value("LS_RATIO_Z", "00:20", 3), math.nan)
    use_archive(_fixture_b((1.0, 2.0, 2.0, 2.0, 7.0)))  # flat 2, 2, 2 at 00:00-00:10
    assert _same(_value("LS_RATIO_Z", "00:20", 3), math.nan)


def test_null_field_is_a_hole(use_archive: Callable[..., Path]) -> None:
    """R9: a NULL ratio leaves the ratio series only; OI_CHANGE on the same row is unaffected."""
    use_archive(_fixture_b((1.0, 1.0, None, 3.0, 7.0)))
    assert _same(_value("LS_RATIO_Z", "00:20", 3), math.nan)
    assert _same(_value("OI_CHANGE", "00:20", 1), 1002 / 1001 - 1)  # rows 00:10 / 00:05


# ---------------------------------------------------------------------------
# R13/R13b/R14: DVOL_Z candles, staleness and currency mapping.
# ---------------------------------------------------------------------------


def test_dvol_candle_usable_at_its_end(use_archive: Callable[..., Path]) -> None:
    """R13/R13b: a candle opened at o is used from close o + 1 h; it stays usable while the
    close is less than 2 h past that."""
    use_archive([], _fixture_d())
    assert _same(_value("DVOL_Z", "02:00", 3), math.nan)  # candles 00:00, 01:00 usable
    assert _same(_value("DVOL_Z", "02:59", 3), math.nan)
    assert _bits(_value("DVOL_Z", "03:00", 3)) == _bits(1.0)  # window 50, 52, 54

    use_archive([], _fixture_d(drop=(3,)))  # minus candle 03:00
    assert _bits(_value("DVOL_Z", "04:59", 3)) == _bits(1.0)  # as-of 02:00, age 1 h 59 m
    assert _same(_value("DVOL_Z", "05:00", 3), math.nan)  # age 2 h, not < 2 h


def test_dvol_currency_mapping(use_archive: Callable[..., Path]) -> None:
    """R14: ETHUSDT reads ETH DVOL; every other symbol reads BTC DVOL."""
    hour = timedelta(hours=1)
    eth = [(S0 + h * hour, c) for h, c in enumerate((10.0, 20.0, 60.0, 61.0))]
    use_archive([], {**_fixture_d(), "ETH": eth})
    assert [aux_data.dvol_currency(s) for s in ("ETHUSDT", "BTCUSDT", "ZECUSDT")] == [
        "ETH",
        "BTC",
        "BTC",
    ]
    btc = _value("DVOL_Z", "03:00", 3, symbol="BTCUSDT")
    assert _bits(btc) == _bits(1.0)
    assert _bits(_value("DVOL_Z", "03:00", 3, symbol="ZECUSDT")) == _bits(btc)
    z_eth = _value("DVOL_Z", "03:00", 3, symbol="ETHUSDT")  # window 10, 20, 60
    assert _same(z_eth, 3 / math.sqrt(7)), z_eth
    assert z_eth == 1.1338934190276817


# ---------------------------------------------------------------------------
# R10: values before T + lag do not depend on rows stamped at or after T.
# ---------------------------------------------------------------------------


def test_truncation_invariant(use_archive: Callable[..., Path]) -> None:
    day0 = datetime(2024, 3, 1, tzinfo=UTC)
    rng = np.random.default_rng(7)
    n = 3 * 288
    oi = 1000.0 + rng.random(n) * 50.0
    ratio = 1.0 + rng.random(n)
    holes = {40, 41, 300}  # day 0 and day 1 06:00, more than a day before T
    rows: list[Row] = [
        (day0 + i * FIVE_MIN, float(oi[i]), float(ratio[i])) for i in range(n) if i not in holes
    ]
    hour = timedelta(hours=1)
    dvol_first = day0 - timedelta(days=8)
    n_candles = 11 * 24 + 15  # through day 2 14:00
    dvol_close = 40.0 + rng.random(n_candles) * 20.0
    candles: list[Candle] = [
        (dvol_first + h * hour, float(dvol_close[h])) for h in range(n_candles) if h != 5
    ]
    cut = datetime(2024, 3, 3, 12, tzinfo=UTC)  # T = day 2 12:00
    t_ns = _ms(cut) * 1_000_000
    minute_ns = 60 * 10**9
    closes = np.arange(
        _ms(day0 + timedelta(days=1)) * 1_000_000,
        _ms(datetime(2024, 3, 3, 14, tzinfo=UTC)) * 1_000_000 + 1,
        minute_ns,
        dtype=np.int64,
    )
    # metrics rows are usable 600 s after their stamp, DVOL candles 3600 s after their open
    before = {"oi": closes < t_ns + 600 * 10**9, "z": closes < t_ns + 600 * 10**9}
    before["dvol"] = closes < t_ns + 3600 * 10**9
    readers = {
        "oi": (aux_data.oi_change_asof, aux_data.oi_change_at),
        "z": (aux_data.ls_ratio_z_asof, aux_data.ls_ratio_z_at),
        "dvol": (aux_data.dvol_z_asof, aux_data.dvol_z_at),
    }
    periods = {"oi": (3, 12, 288), "z": (3, 12, 288), "dvol": (3, 24, 168)}

    def run(rows_used: list[Row], candles_used: list[Candle]) -> dict[tuple[str, int], np.ndarray]:
        use_archive(rows_used, {"BTC": candles_used})
        out: dict[tuple[str, int], np.ndarray] = {}
        for kind, (asof, at) in readers.items():
            for period in periods[kind]:
                out[(kind, period)] = vec = asof(SYMBOL, closes, period)
                scalars = [at(SYMBOL, int(c), period) for c in closes[::37]]
                assert np.array_equal(
                    np.asarray(scalars).view(np.int64), vec[::37].view(np.int64)
                ), (kind, period)
        return out

    full = run(rows, candles)
    truncated = run([r for r in rows if r[0] < cut], [c for c in candles if c[0] < cut])
    differs: dict[str, bool] = {}
    for (kind, period), values in full.items():
        cut_values = truncated[(kind, period)]
        early = before[kind]
        fresh = (closes >= t_ns) & early
        assert np.array_equal(values[early].view(np.int64), cut_values[early].view(np.int64)), (
            kind,
            period,
        )
        assert np.isfinite(values[fresh]).any(), (kind, period)
        late_differs = not np.array_equal(values[~early], cut_values[~early], equal_nan=True)
        differs[kind] = differs.get(kind, False) or late_differs
    # rows at or after T must matter after T + lag (else the test would be vacuous)
    assert differs == {"oi": True, "z": True, "dvol": True}


def _cached_bytes() -> int:
    return sum(v[1].nbytes for v in aux_data._METRICS_DERIVED.values())


@pytest.mark.parametrize("budget", [1, 100])
def test_derived_cache_byte_budget(
    use_archive: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    """R-pos-10: the derived cache holds at most ``DERIVED_CACHE_BYTES`` (fixture arrays are
    32-40 bytes; 1 byte is smaller than any entry, 100 holds two). Every read, evicted or
    not, returns the bits an unbounded cache gives."""
    use_archive(_fixture_b(), _fixture_d())
    reads = [("LS_RATIO_Z", "00:20", p) for p in range(2, 6)]
    reads += [("DVOL_Z", "03:00", p) for p in range(2, 5)]
    aux_data._METRICS_DERIVED.clear()
    expected = [_bits(_value(*r)) for r in reads]
    monkeypatch.setattr(aux_data, "DERIVED_CACHE_BYTES", budget)
    aux_data._METRICS_DERIVED.clear()
    for _ in range(2):
        assert [_bits(_value(*r)) for r in reads] == expected
        assert _cached_bytes() <= budget or len(aux_data._METRICS_DERIVED) == 1
    assert _bits(_value("LS_RATIO_Z", "00:20", 3)) == _bits(1.0)
    assert _bits(_value("DVOL_Z", "03:00", 3)) == _bits(1.0)


def _count_derivations(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"n": 0}
    for kind, (column, fn, lag, stale) in list(aux_data._KINDS.items()):

        def counted(ts: np.ndarray, y: np.ndarray, p: int, _fn: Any = fn) -> np.ndarray:
            calls["n"] += 1
            return np.asarray(_fn(ts, y, p))

        monkeypatch.setitem(aux_data._KINDS, kind, (column, counted, lag, stale))
    return calls


def test_derived_cache_holds_a_multi_symbol_run(
    use_archive: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-pos-10: 17 symbols x {OI_CHANGE(12), LS_RATIO_Z(5)} read per bar the way the NT
    strategies do (interleaved) derive each of the 34 arrays once; a budget one byte short
    of the 34 arrays re-derives (LRU thrash), the exact sum does not."""
    symbols = [SYMBOL] + [f"S{i:02d}USDT" for i in range(16)]
    rows: list[Row] = [(S0 + k * FIVE_MIN, 1000.0 + k, 1.0 + k % 7) for k in range(30)]
    path = use_archive(rows)
    for i, symbol in enumerate(symbols[1:]):
        _write(path, [(t, x + i, y) for t, x, y in rows], symbol=symbol)
    calls = _count_derivations(monkeypatch)
    closes = [_close(hhmm) for hhmm in ("02:30", "02:35", "02:40")]

    def run() -> list[int]:
        aux_data._METRICS_DERIVED.clear()
        calls["n"] = 0
        out: list[int] = []
        for close_ns in closes:
            for symbol in symbols:
                out.append(_bits(aux_data.oi_change_at(symbol, close_ns, 12)))
                out.append(_bits(aux_data.ls_ratio_z_at(symbol, close_ns, 5)))
        return out

    values = run()
    assert calls["n"] == 34
    # as-of row 28 at 02:30: OI x[28]/x[16] - 1 for symbol 0; z over y = 4,5,6,7,1
    assert values[0] == _bits(1028.0 / 1016.0 - 1.0)
    assert values[1] == _bits((1.0 - 4.6) / np.std([4.0, 5.0, 6.0, 7.0, 1.0], ddof=1))
    total = _cached_bytes()
    monkeypatch.setattr(aux_data, "DERIVED_CACHE_BYTES", total)
    assert run() == values and calls["n"] == 34
    monkeypatch.setattr(aux_data, "DERIVED_CACHE_BYTES", total - 1)
    assert run() == values and calls["n"] == 34 * len(closes)


def _z_reference(ts: np.ndarray, y: np.ndarray, period: int, step_ns: int) -> np.ndarray:
    """The unchunked ``_z_rows`` (T3): one ``n x period`` window for all rows."""
    from numpy.lib.stride_tricks import sliding_window_view

    out = np.full(ts.size, np.nan)
    if period < 2 or ts.size < period:
        return out
    win = sliding_window_view(y, period)
    mean = win.mean(axis=1)
    std = win.std(axis=1, ddof=1)
    contiguous = ts[period - 1 :] - ts[: ts.size - period + 1] == (period - 1) * step_ns
    with np.errstate(invalid="ignore", divide="ignore"):
        out[period - 1 :] = np.where(
            contiguous & (std > 1e-15), (y[period - 1 :] - mean) / std, np.nan
        )
    return out


def _z_inputs() -> list[tuple[np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(7)
    step = aux_data.METRICS_STEP_NS
    # walk with holes (dropped rows) and one NaN value
    keep = np.ones(3000, dtype=bool)
    keep[[17, 400, 401, 1500, 2999]] = False
    ts_a = (np.arange(3000, dtype=np.int64) * step)[keep]
    y_a = 1.0 + np.cumsum(rng.normal(0.0, 0.01, 3000))[keep]
    y_a[900] = np.nan
    # BTC-like ratio levels with flat stretches (std <= 1e-15) and a hole
    y_b = np.round(2.0 + rng.random(2500), 2)
    y_b[600:1000] = 1.5
    ts_b = np.delete(np.arange(2501, dtype=np.int64) * step, 1234)
    return [(ts_a, y_a, 5), (ts_a, y_a, 288), (ts_b, y_b, 24), (ts_b, y_b, 288)]


@pytest.mark.parametrize("chunk_elems", [1, 7 * 24 + 3, 1000, 1 << 22])
def test_z_rows_chunked_matches_unchunked(
    monkeypatch: pytest.MonkeyPatch, chunk_elems: int
) -> None:
    """R-pos-11: row-chunked z (bounded ``win.std`` temporaries) is bit-identical to the
    unchunked window over every row, with chunk edges inside windows, holes and NaN."""
    monkeypatch.setattr(aux_data, "_Z_CHUNK_ELEMS", chunk_elems)
    for ts, y, period in _z_inputs():
        ref = _z_reference(ts, y, period, aux_data.METRICS_STEP_NS)
        got = aux_data._z_rows(ts, y, period)
        assert np.array_equal(got.view(np.int64), ref.view(np.int64)), period
    finite = np.isfinite(aux_data._z_rows(*_z_inputs()[3][:2], 288))
    assert 0 < finite.sum() < finite.size  # holes and flat stretches leave NaN rows


# ---------------------------------------------------------------------------
# R15/R16: preflight.
# ---------------------------------------------------------------------------


def test_preflight_metrics_not_funding(use_archive: Callable[..., Path]) -> None:
    """R15: OI_CHANGE / LS_RATIO_Z need metrics, never funding (0 funding rows here)."""
    path = use_archive(_fixture_b())
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_funding_rates").fetchone() == (0,)
    aux_data.preflight(["OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-02")
    aux_data.preflight(["LS_RATIO_Z", "OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-02")


def _span(first: datetime, last: datetime) -> list[Row]:
    rows: list[Row] = []
    t = first
    while t <= last:
        rows.append((t, 1000.0, 1.0))
        t += FIVE_MIN
    return rows


def test_preflight_metrics_refusals(tmp_path: Path, use_archive: Any) -> None:
    """R16: missing table / symbol, a late first row, an early last row (edges included)."""
    bare = tmp_path / "bare.db"
    with sqlite3.connect(bare) as conn:
        conn.execute("CREATE TABLE unrelated (a INTEGER)")
    aux_data.configure(bare, None)
    missing = (
        f"No Binance metrics archived for BTCUSDT in {bare} — "
        "python -m vibe_quant.data download-metrics --symbols BTCUSDT"
    )
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-06")
    assert str(err.value) == missing

    other = _write(tmp_path / "eth.db", _span(S0, S0 + timedelta(days=5)), symbol="ETHUSDT")
    aux_data.configure(other, None)
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["LS_RATIO_Z"], [SYMBOL], "2024-01-01", "2024-01-06")
    assert str(err.value) == missing.replace(str(bare), str(other))

    # window 2024-01-01..2024-01-06: first row may be 01-04 00:00, last row 01-05 00:00
    first_ok, last_ok = datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC)
    use_archive(_span(first_ok, last_ok))
    aux_data.preflight(["OI_CHANGE", "LS_RATIO_Z"], [SYMBOL], "2024-01-01", "2024-01-06")

    use_archive(_span(first_ok + FIVE_MIN, last_ok))
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-06")
    assert str(err.value) == (
        "Binance metrics for BTCUSDT start after the requested window "
        "(first row 2024-01-04 00:05, window start 2024-01-01) — download-metrics"
    )

    use_archive(_span(first_ok, last_ok - FIVE_MIN))
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["LS_RATIO_Z"], [SYMBOL], "2024-01-01", "2024-01-06")
    assert str(err.value) == (
        "Binance metrics for BTCUSDT end before the requested window "
        "(last row 2024-01-04 23:55, window end 2024-01-06) — download-metrics"
    )


def test_preflight_rereads_a_refreshed_archive(use_archive: Callable[..., Path]) -> None:
    """A lookup cached before a refresh is not served to the next launch check."""
    first_ok, last_ok = datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC)
    path = use_archive(_span(first_ok, last_ok - FIVE_MIN))
    with pytest.raises(AuxDataUnavailableError, match="end before"):
        aux_data.preflight(["OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-06")
    _write(path, [(last_ok, 1000.0, 1.0)])  # the next day's file lands
    aux_data.preflight(["OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-06")


def _hourly(first: datetime, last: datetime) -> list[Candle]:
    candles: list[Candle] = []
    t = first
    while t <= last:
        candles.append((t, 50.0))
        t += timedelta(hours=1)
    return candles


def test_preflight_dvol_refusals(tmp_path: Path, use_archive: Any) -> None:
    """R16 (DVOL): missing table / currency, a late first candle, an early last candle (edges
    included); no funding or metrics rows are needed, and ZECUSDT checks BTC DVOL."""
    bare = tmp_path / "bare.db"
    with sqlite3.connect(bare) as conn:
        conn.execute("CREATE TABLE unrelated (a INTEGER)")
    aux_data.configure(bare, None)
    missing = (
        f"No Deribit BTC DVOL archived in {bare} — "
        "python -m vibe_quant.data download-dvol --currencies BTC"
    )
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["DVOL_Z"], [SYMBOL], "2024-01-01", "2024-01-06")
    assert str(err.value) == missing

    # window 2024-01-01..2024-01-06: first candle may open 01-04 00:00, last 01-05 00:00
    first_ok, last_ok = datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC)
    path = use_archive([], {"ETH": _hourly(first_ok, last_ok)})
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["DVOL_Z"], ["ZECUSDT"], "2024-01-01", "2024-01-06")
    assert str(err.value) == (
        f"No Deribit BTC DVOL archived in {path} — "
        "python -m vibe_quant.data download-dvol --currencies BTC"
    )
    aux_data.preflight(["DVOL_Z"], ["ETHUSDT"], "2024-01-01", "2024-01-06")

    path = use_archive([], {"BTC": _hourly(first_ok, last_ok)})
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT (SELECT COUNT(*) FROM raw_funding_rates) + (SELECT COUNT(*) FROM raw_metrics)"
        ).fetchone() == (0,)
    aux_data.preflight(["DVOL_Z"], [SYMBOL, "ZECUSDT"], "2024-01-01", "2024-01-06")

    use_archive([], {"BTC": _hourly(first_ok + timedelta(hours=1), last_ok)})
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["DVOL_Z"], [SYMBOL], "2024-01-01", "2024-01-06")
    assert str(err.value) == (
        "Deribit BTC DVOL starts after the requested window "
        "(first candle 2024-01-04 01:00, window start 2024-01-01) — download-dvol"
    )

    use_archive([], {"BTC": _hourly(first_ok, last_ok - timedelta(hours=1))})
    with pytest.raises(AuxDataUnavailableError) as err:
        aux_data.preflight(["DVOL_Z"], ["ZECUSDT"], "2024-01-01", "2024-01-06")
    assert str(err.value) == (
        "Deribit BTC DVOL ends before the requested window "
        "(last candle 2024-01-04 23:00, window end 2024-01-06) — download-dvol"
    )
