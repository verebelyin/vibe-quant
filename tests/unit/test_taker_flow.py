"""TAKER_IMBALANCE as-of context indicator (vibe-quant-ns78u tasks 1-3).

The value at a bar closing at ``t`` is
``sum(taker_buy_volume) / sum(volume) - 0.5`` over archived 1m klines closed
in ``(t - period*60s, t]``: integer running sums, one float division, and NaN
unless the window holds exactly ``period`` consecutive klines ending at ``t``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

MIN = 60 * 1_000_000_000
DAY = 24 * 3_600 * 1_000_000_000
DAY0 = 1_704_067_200 * 1_000_000_000  # 2024-01-01T00:00:00Z


def _make_kline_archive(
    path: Path,
    volumes: Sequence[float],
    takers: Sequence[float | None] | None = None,
    symbol: str = "ZECUSDT",
    start_ns: int = DAY0,
    minute_indices: Sequence[int] | None = None,
) -> Path:
    """Archive 1m klines ``start_ns + minute_indices[i]*1m`` with given volumes."""
    from vibe_quant.data.archive import RawDataArchive

    archive = RawDataArchive(path)
    try:
        rows = []
        for i, volume in enumerate(volumes):
            minute = i if minute_indices is None else minute_indices[i]
            open_ms = (start_ns + minute * MIN) // 1_000_000
            taker = volume if takers is None else takers[i]
            rows.append(
                (symbol, open_ms, volume, open_ms + 59_999, taker)
            )
        archive.conn.executemany(
            "INSERT INTO raw_klines (symbol, interval, open_time, open, high, low, "
            "close, volume, close_time, taker_buy_volume, source) "
            "VALUES (?, '1m', ?, 1.0, 1.0, 1.0, 1.0, ?, ?, ?, 'test')",
            rows,
        )
        archive.conn.commit()
    finally:
        archive.close()
    return path


@pytest.fixture
def clean() -> Iterator[None]:
    yield
    aux_data.configure(None, None)


def _value(archive: Path, t: int, period: int, symbol: str = "ZECUSDT") -> float:
    aux_data.configure(archive, None)
    out = aux_data.taker_imbalance_asof(symbol, np.array([t], dtype=np.int64), period)
    return float(out[0])


# -- 1. as-of window -----------------------------------------------------------------


def test_taker_excludes_kline_closing_after_bar(
    tmp_path: Path, clean: None
) -> None:
    """A kline OPENING at ``t`` closes after ``t`` and must not enter the window."""
    volumes = [round(1.001 * (i + 1), 3) for i in range(11)]  # 3-decimal volumes
    takers = [round(v * 0.5, 3) for v in volumes]
    takers[10] = 999.0  # kline 10 opens exactly at t: answer must not see it
    archive = _make_kline_archive(tmp_path / "a.db", volumes, takers)

    t = DAY0 + 10 * MIN  # close of kline 9; window opens t-3m (klines 7,8,9)
    got = _value(archive, t, period=3)
    hand = sum(takers[7:10]) / sum(volumes[7:10]) - 0.5
    assert hand != 0.0
    assert got == hand

    # one minute later the same kline is closed and is the newest window member
    got_later = _value(archive, t + MIN, period=3)
    hand_later = sum(takers[8:11]) / sum(volumes[8:11]) - 0.5
    assert got_later == pytest.approx(hand_later)
    assert got_later > 30  # the 999 kline is visible here


def test_window_gap_or_zero_volume_is_nan(tmp_path: Path, clean: None) -> None:
    volumes = [1.0] * 10
    takers = [1.0] * 10
    # kline 5 (closing at DAY0+6m) is missing: the period-5 window ending at
    # DAY0+10m spans it, so exactly-period-consecutive fails.
    present = [i for i in range(10) if i != 5]
    gaps = _make_kline_archive(
        tmp_path / "gap.db",
        [volumes[i] for i in present],
        [takers[i] for i in present],
        minute_indices=present,
    )
    t = DAY0 + 10 * MIN
    assert np.isnan(_value(gaps, t, period=5))

    zeros = _make_kline_archive(tmp_path / "zero.db", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert np.isnan(_value(zeros, DAY0 + 3 * MIN, period=3))


def test_value_independent_of_query_set(tmp_path: Path, clean: None) -> None:
    """One close and 10k closes give bit-equal values for the same ``t``."""
    n = 10_000
    rng = np.random.default_rng(7)
    volumes = np.round(rng.uniform(0.001, 50.0, n), 3)
    takers = np.round(volumes * rng.uniform(0.2, 0.8, n), 3)
    archive = _make_kline_archive(tmp_path / "many.db", volumes, takers)

    t = DAY0 + (n - 1) * MIN
    one = _value(archive, t, period=60)

    aux_data.configure(archive, None)
    all_closes = DAY0 + (np.arange(n, dtype=np.int64) + 1) * MIN
    many = aux_data.taker_imbalance_asof("ZECUSDT", all_closes, 60)
    assert np.isfinite(one)
    assert many[-2] == one  # all_closes[-2] == t (last is one minute later)
    assert np.array_equal(many, aux_data.taker_imbalance_asof("ZECUSDT", all_closes, 60),
                          equal_nan=True)


def test_non_integral_volume_raises(tmp_path: Path, clean: None) -> None:
    # 1.00005 needs 10^5 to be integral: no p <= 4 exists.
    archive = _make_kline_archive(tmp_path / "frac.db", [1.00005] * 5, [0.5] * 5)
    with pytest.raises(AuxDataUnavailableError, match="not integral"):
        _value(archive, DAY0 + 5 * MIN, period=3)


def _sqlite_klines(path: Path, rows: Sequence[tuple[int, float, float | None]]) -> Path:
    """Bare ``raw_klines`` table (open_time ms, volume, taker_buy_volume) for ZECUSDT 1m."""
    import sqlite3

    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE raw_klines(symbol TEXT, interval TEXT, open_time INTEGER, "
        "volume REAL, taker_buy_volume REAL)"
    )
    db.executemany("INSERT INTO raw_klines VALUES ('ZECUSDT', '1m', ?, ?, ?)", rows)
    db.commit()
    db.close()
    return path


def test_scale_uses_both_columns_and_ignores_unneeded_prefix(
    tmp_path: Path, clean: None
) -> None:
    """Total volume 1.0 with taker 0.5 needs p=1; an unused earlier row must not change it."""
    tail = [(i * 60_000, 1.0, 0.5) for i in range(1, 6)]
    full = _sqlite_klines(tmp_path / "full.db", [(0, 1.001, 0.5), *tail])
    cropped = _sqlite_klines(tmp_path / "crop.db", tail)
    window_close = np.array([6 * MIN], dtype=np.int64)
    aux_data.configure(full, None)
    expected = aux_data.taker_imbalance_asof("ZECUSDT", window_close, 5)[0]
    assert expected == 0.0
    aux_data.configure(cropped, None)
    assert aux_data.taker_imbalance_asof("ZECUSDT", window_close, 5)[0] == expected


def test_unrepresentable_eight_decimal_volume_fails_closed(
    tmp_path: Path, clean: None
) -> None:
    for volume, taker in ((1.00000001, 1.0), (1.0, 0.99999999)):
        path = _sqlite_klines(tmp_path / f"bad_{taker}.db", [(0, volume, taker)])
        aux_data.configure(path, None)
        with pytest.raises(AuxDataUnavailableError, match="not integral"):
            aux_data.taker_imbalance_asof("ZECUSDT", np.array([MIN], dtype=np.int64), 1)


def test_scaling_accepts_float_representation_error_of_real_decimals(
    tmp_path: Path, clean: None
) -> None:
    """3-decimal volumes (0.1+0.2-style floats) stay exact; the ratio is the integer ratio."""
    rows = [(i * 60_000, 0.1 * (i + 1) + 0.2, 0.1 * (i + 1)) for i in range(5)]
    rows = [(t, round(v, 3), round(k, 3)) for t, v, k in rows]
    path = _sqlite_klines(tmp_path / "dec.db", rows)
    aux_data.configure(path, None)
    got = aux_data.taker_imbalance_asof("ZECUSDT", np.array([5 * MIN], dtype=np.int64), 5)[0]
    vol = sum(round(v * 1000) for _, v, _ in rows)
    tak = sum(round(k * 1000) for _, _, k in rows)
    assert got == tak / vol - 0.5


@pytest.mark.parametrize(
    ("volume", "taker"),
    [(1e20, 5e19), (1.0, float("inf")), (float("inf"), 1.0)],
)
def test_out_of_range_or_nonfinite_quantity_fails_closed(
    tmp_path: Path, clean: None, volume: float, taker: float
) -> None:
    import warnings

    path = _sqlite_klines(tmp_path / "range.db", [(0, volume, taker)])
    aux_data.configure(path, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AuxDataUnavailableError):
            aux_data.taker_imbalance_asof("ZECUSDT", np.array([MIN], dtype=np.int64), 1)


def test_running_sum_beyond_int64_fails_closed(tmp_path: Path, clean: None) -> None:
    """Each value is a legal integer, but their running sum would wrap int64."""
    import warnings

    rows = [(i * 60_000, 1e16, 1e16) for i in range(1000)]  # 1e19 > 2^63
    path = _sqlite_klines(tmp_path / "sum.db", rows)
    aux_data.configure(path, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AuxDataUnavailableError):
            aux_data.taker_imbalance_asof("ZECUSDT", np.array([1000 * MIN], dtype=np.int64), 5)


def test_archive_change_reloads_without_clearing_caches(tmp_path: Path, clean: None) -> None:
    """Appended rows, then a replaced archive, are seen after the next launch check
    (preflight revalidates the cache; per-bar lookups reuse it)."""
    path = _sqlite_klines(tmp_path / "grow.db", [(i * 60_000, 2.0, 1.0) for i in range(5)])
    aux_data.configure(path, None)
    ask = np.array([6 * MIN], dtype=np.int64)
    assert np.isnan(aux_data.taker_imbalance_asof("ZECUSDT", ask, 2)).all()

    import sqlite3

    db = sqlite3.connect(path)
    db.execute("INSERT INTO raw_klines VALUES ('ZECUSDT', '1m', 300000, 2.0, 1.5)")
    db.commit()
    db.close()
    aux_data._taker_klines("ZECUSDT", revalidate=True)
    assert aux_data.taker_imbalance_asof("ZECUSDT", ask, 1)[0] == 0.25

    path.unlink()
    with pytest.raises(AuxDataUnavailableError, match="kline archive"):
        aux_data._taker_klines("ZECUSDT", revalidate=True)
    _sqlite_klines(path, [(300_000, 4.0, 1.0)])
    aux_data._taker_klines("ZECUSDT", revalidate=True)
    assert aux_data.taker_imbalance_asof("ZECUSDT", ask, 1)[0] == -0.25


def test_wal_only_change_reloads(tmp_path: Path, clean: None) -> None:
    """A commit that lives only in the ``-wal`` file leaves the main file untouched."""
    import sqlite3

    path = _sqlite_klines(tmp_path / "wal.db", [(0, 2.0, 1.0)])
    writer = sqlite3.connect(path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO raw_klines VALUES ('ZECUSDT', '1m', 60000, 2.0, 1.0)")
        writer.commit()
        aux_data.configure(path, None)
        ask = np.array([3 * MIN], dtype=np.int64)
        assert np.isnan(aux_data.taker_imbalance_asof("ZECUSDT", ask, 1)).all()
        main_stamp = path.stat().st_mtime_ns, path.stat().st_size
        writer.execute("INSERT INTO raw_klines VALUES ('ZECUSDT', '1m', 120000, 2.0, 2.0)")
        writer.commit()
        assert (path.stat().st_mtime_ns, path.stat().st_size) == main_stamp  # main file unchanged
        aux_data._taker_klines("ZECUSDT", revalidate=True)
        assert aux_data.taker_imbalance_asof("ZECUSDT", ask, 1)[0] == 0.5
    finally:
        writer.close()


def test_missing_or_directory_archive_raises(tmp_path: Path, clean: None) -> None:
    aux_data.configure(tmp_path / "nope.db", None)
    with pytest.raises(AuxDataUnavailableError, match="kline archive"):
        aux_data.taker_imbalance_asof("ZECUSDT", np.array([DAY0], dtype=np.int64), 3)
    aux_data.configure(tmp_path, None)
    with pytest.raises(AuxDataUnavailableError, match="kline archive"):
        aux_data.taker_imbalance_asof("ZECUSDT", np.array([DAY0], dtype=np.int64), 3)


def test_archive_without_symbol_rows_raises(tmp_path: Path, clean: None) -> None:
    archive = _make_kline_archive(tmp_path / "eth.db", [1.0] * 5, [0.5] * 5, symbol="ETHUSDT")
    with pytest.raises(AuxDataUnavailableError, match="No 1m klines archived for ZECUSDT"):
        _value(archive, DAY0 + 5 * MIN, period=3)


# -- 2. preflight / discovery pool / paper ---------------------------------------------


def test_preflight_taker_needs_klines_not_funding(
    tmp_path: Path, clean: None
) -> None:
    """TAKER_IMBALANCE checks 1m klines; it must not enter the funding branch."""
    full = _make_kline_archive(tmp_path / "cover.db", [1.0] * 2882, start_ns=DAY0 - DAY - MIN)
    aux_data.configure(full, None)
    # The archive holds no funding rows at all: a funding-branch route would raise.
    aux_data.preflight(["TAKER_IMBALANCE"], ["ZECUSDT"], "2024-01-01", "2024-01-02")

    late = _make_kline_archive(tmp_path / "late.db", [1.0] * 2881, start_ns=DAY0)
    aux_data.configure(late, None)
    with pytest.raises(AuxDataUnavailableError, match="start after the requested window"):
        aux_data.preflight(["TAKER_IMBALANCE"], ["ZECUSDT"], "2024-01-01", "2024-01-02")

    short = _make_kline_archive(
        tmp_path / "short.db", [1.0] * 1441, start_ns=DAY0 - DAY - MIN
    )
    aux_data.configure(short, None)
    with pytest.raises(AuxDataUnavailableError, match="end before the requested window"):
        aux_data.preflight(["TAKER_IMBALANCE"], ["ZECUSDT"], "2024-01-01", "2024-01-02")

    # No taker indicator: no kline archive needed at all.
    aux_data.preflight(["RSI"], ["ZECUSDT"], "1999-01-01", "2099-01-01")


@pytest.fixture
def restore_ga_pool() -> Iterator[None]:
    from vibe_quant.discovery import operators as ops

    caches = [
        ops.INDICATOR_POOL,
        ops._INDICATOR_NAMES,
        ops.THRESHOLD_RANGES,
        ops._INT_PARAMS,
        ops._PARAM_DEFAULTS,
    ]
    saved = [(c, c.copy()) for c in caches]
    for c in caches:
        c.clear()
    yield
    for c, snapshot in saved:
        c.clear()
        if isinstance(c, dict):
            c.update(snapshot)  # type: ignore[arg-type]
        else:
            c.extend(snapshot)  # type: ignore[arg-type]


def _pipeline(indicator_pool: list[str] | None):  # type: ignore[no-untyped-def]
    from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    return DiscoveryPipeline(_make_config(indicator_pool=indicator_pool), _mock_backtest)


def test_default_pool_excludes_taker(restore_ga_pool: None) -> None:
    import vibe_quant.dsl.plugins.taker  # noqa: F401
    from vibe_quant.discovery.operators import INDICATOR_POOL

    _pipeline(None)._apply_indicator_pool_filter()
    assert "TAKER_IMBALANCE" not in INDICATOR_POOL
    assert "RSI" in INDICATOR_POOL


def test_explicit_pool_includes_taker(restore_ga_pool: None) -> None:
    import vibe_quant.dsl.plugins.taker  # noqa: F401
    from vibe_quant.discovery.operators import INDICATOR_POOL, THRESHOLD_RANGES

    _pipeline(["TAKER_IMBALANCE", "RSI"])._apply_indicator_pool_filter()
    assert set(INDICATOR_POOL) == {"TAKER_IMBALANCE", "RSI"}
    assert THRESHOLD_RANGES["TAKER_IMBALANCE"] == (-0.15, 0.15)
    assert INDICATOR_POOL["TAKER_IMBALANCE"] == {"period": (5.0, 240.0)}


def test_compute_fn_defaults_period_to_60(tmp_path: Path, clean: None) -> None:
    """The plugin's no-param default must be the documented 60 minutes."""
    import pandas as pd

    import vibe_quant.dsl.plugins.taker  # noqa: F401
    from vibe_quant.dsl.indicators import indicator_registry, invoke_compute_fn

    volumes = [round(1.0 + 0.001 * i, 3) for i in range(70)]
    takers = [round(v * 0.4, 3) for v in volumes]
    aux_data.configure(_make_kline_archive(tmp_path / "def.db", volumes, takers), None)
    spec = indicator_registry.get("TAKER_IMBALANCE")
    assert spec is not None
    assert spec.default_params == {"period": 60}
    df = pd.DataFrame({c: np.ones(1) for c in ("open", "high", "low", "close", "volume")})
    df.attrs["bar_close_ns"] = np.array([DAY0 + 70 * MIN], dtype=np.int64)
    df.attrs["symbol"] = "ZECUSDT-PERP.BINANCE"
    default = invoke_compute_fn(spec, df, {})
    explicit = invoke_compute_fn(spec, df, {"period": 60})
    assert float(default.iloc[0]) == float(explicit.iloc[0])


# -- 3. NT end to end ------------------------------------------------------------------


def _taker_dsl(period: int = 60) -> dict[str, object]:
    return {
        "name": "taker_smoke",
        "timeframe": "1h",
        "indicators": {"x": {"type": "TAKER_IMBALANCE", "period": period}},
        "entry_conditions": {"long": ["x > -0.5"]},
        "exit_conditions": {"long": ["x > 0.45"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def test_taker_1h_strategy_values_match_asof(clean: None) -> None:
    """The generated strategy stores, at each bar, the as-of lookup at its rounded close."""
    from pathlib import Path

    import pandas as pd

    import vibe_quant.dsl.plugins.taker  # noqa: F401
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.dsl.pta_buffer import PtaBuffer
    from vibe_quant.validation.extraction import load_catalog_bars

    bar_type = "ZECUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL"
    if not Path("data/catalog/data/bar", bar_type).is_dir():
        pytest.skip("ZEC 1h catalog data not available")
    aux_data.configure(None, None)
    bars = load_catalog_bars("data/catalog", bar_type)
    step = 3_600 * 1_000_000_000
    first = int(pd.Timestamp("2025-06-02", tz="UTC").value)
    lo = int(np.searchsorted(bars.ts, first, side="left"))
    n = 300
    sl = slice(lo, lo + n)
    raw_ts = bars.ts[sl].astype(np.int64)
    close_ns = ((raw_ts + step // 2) // step) * step
    expected = aux_data.taker_imbalance_asof("ZECUSDT", close_ns, 60)
    assert np.isfinite(expected).any()

    module = StrategyCompiler().compile_to_module(validate_strategy_dict(_taker_dsl()))
    cls = next(v for k, v in vars(module).items() if k.endswith("Strategy") and k != "Strategy")
    cfg = type("Cfg", (), {"instrument_id": "ZECUSDT-PERP.BINANCE"})()
    harness = type("Harness", (cls,), {"config": cfg})
    inst = harness.__new__(harness)
    inst._pta_values = {}
    inst._pta_params = {"x": {"period": 60}}
    inst._pta_lookback = {"x": 1}
    buf = PtaBuffer(0, with_close_ns=True)
    inst._pta_bufs = {"1h": buf}

    got = []
    for i in range(n):
        buf.append(
            float(bars.open[sl][i]), float(bars.high[sl][i]), float(bars.low[sl][i]),
            float(bars.close[sl][i]), 1.0, int(close_ns[i]),
        )
        inst._update_pta_indicators("1h")
        got.append(inst._pta_values.get("x", float("nan")))
    assert np.array_equal(np.asarray(got), expected, equal_nan=True)


def test_taker_strategy_runs_in_screening(clean: None) -> None:
    from pathlib import Path

    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import _dsl_to_dict

    if not DEFAULT_ARCHIVE_PATH.exists():
        pytest.skip("real 1m archive not available")
    if not Path("data/catalog/data/bar/ZECUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").is_dir():
        pytest.skip("ZEC 1m catalog data not available")
    aux_data.configure(None, None)
    dsl = validate_strategy_dict(_taker_dsl())
    metrics = NTScreeningRunner(
        _dsl_to_dict(dsl), ["ZECUSDT"], "2025-06-02", "2025-06-09"
    )({})
    assert metrics.total_trades > 0


