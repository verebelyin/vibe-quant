"""Positioning metrics (Binance Vision ``metrics``) as-of lookups: OI_CHANGE and LS_RATIO_Z.

Fixtures (S0 = 2024-01-01 00:00 UTC, BTCUSDT, closes given as HH:MM on S0's day):
- A: rows k = -24..60 at S0 + 5m*k, ``sum_open_interest`` x_k = 1000 + k.
- B: rows k = -1..3, ``sum_toptrader_long_short_ratio`` y = 1.0, 1.0, 2.0, 3.0, 7.0.

A row stamped ``s`` is usable from close ``s + 600 s`` and stays usable while the close is
less than 600 s past that. Every value is read four ways -- ``aux_data.*_asof`` (vector),
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
INSTRUMENT = "BTCUSDT-PERP.BINANCE"
FIVE_MIN = timedelta(minutes=5)

# (create_time, sum_open_interest, sum_toptrader_long_short_ratio)
Row = tuple[datetime, float | None, float | None]


def _ms(t: datetime) -> int:
    return int(t.timestamp()) * 1000


def _close(hhmm: str) -> int:
    """Close time in ns of HH:MM on S0's day."""
    hours, minutes = (int(p) for p in hhmm.split(":"))
    return _ms(S0 + timedelta(hours=hours, minutes=minutes)) * 1_000_000


def _write(path: Path, rows: list[Row], symbol: str = SYMBOL) -> Path:
    """An archive holding ``rows`` as ``ok`` metrics files (one per UTC day)."""
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
    finally:
        archive.close()
    return path


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
def use_archive(tmp_path: Path) -> Iterator[Callable[[list[Row]], Path]]:
    """Write rows to a fresh tmp archive and point aux lookups at it."""
    count = 0

    def use(rows: list[Row]) -> Path:
        nonlocal count
        count += 1
        path = _write(tmp_path / f"raw_data_{count}.db", rows)
        aux_data.configure(path, None)
        return path

    try:
        yield use
    finally:
        aux_data.configure(None, None)


def _bits(value: float) -> int:
    return int(np.float64(value).view(np.int64))


def _value(name: str, hhmm: str, period: int) -> float:
    """The indicator at one close, after asserting its four read paths agree bitwise."""
    close_ns = _close(hhmm)
    asof, at = {
        "OI_CHANGE": (aux_data.oi_change_asof, aux_data.oi_change_at),
        "LS_RATIO_Z": (aux_data.ls_ratio_z_asof, aux_data.ls_ratio_z_at),
    }[name]
    vec = float(asof(SYMBOL, np.array([close_ns], dtype=np.int64), period)[0])
    scalar = at(SYMBOL, close_ns, period)
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    df = pd.DataFrame({c: [1.0] for c in ("open", "high", "low", "close", "volume")})
    df.attrs["bar_close_ns"] = np.array([close_ns], dtype=np.int64)
    df.attrs["symbol"] = INSTRUMENT
    frame_value = float(spec.compute_fn(df, {"period": period}).iloc[-1])
    plugin_scalar = spec.compute_fn.asof_scalar(INSTRUMENT, close_ns, {"period": period})  # type: ignore[attr-defined]
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


def test_row_usable_only_after_lag(use_archive: Callable[[list[Row]], Path]) -> None:
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


def test_gap_limit_one_row(use_archive: Callable[[list[Row]], Path]) -> None:
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


def test_oi_change_endpoints(use_archive: Callable[[list[Row]], Path]) -> None:
    """R5/R6: x[i] / x[row stamped ts[i] - p*5m] - 1; NaN if that row is absent or <= 0."""
    use_archive(_fixture_a())
    assert _same(_value("OI_CHANGE", "01:09", 12), 1011 / 999 - 1)
    assert _same(_value("OI_CHANGE", "01:10", 12), 1012 / 1000 - 1)

    use_archive(_fixture_a(drop=(0,)))
    assert _same(_value("OI_CHANGE", "01:10", 12), math.nan)
    use_archive(_fixture_a(x0=0.0))
    assert _same(_value("OI_CHANGE", "01:10", 12), math.nan)


def test_oi_change_endpoint_rule(use_archive: Callable[[list[Row]], Path]) -> None:
    """R6b: holes between the endpoints do not matter; a missing endpoint does."""
    use_archive(_fixture_a(drop=(2,)))  # minus row 00:10
    two = _value("OI_CHANGE", "00:25", 2)
    assert _same(two, 1003 / 1001 - 1)
    assert two == 0.0019980019980019303
    assert _same(_value("OI_CHANGE", "00:25", 1), math.nan)


# ---------------------------------------------------------------------------
# R7-R9: LS_RATIO_Z strict windows.
# ---------------------------------------------------------------------------


def test_ls_ratio_z_contiguous(use_archive: Callable[[list[Row]], Path]) -> None:
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


def test_null_field_is_a_hole(use_archive: Callable[[list[Row]], Path]) -> None:
    """R9: a NULL ratio leaves the ratio series only; OI_CHANGE on the same row is unaffected."""
    use_archive(_fixture_b((1.0, 1.0, None, 3.0, 7.0)))
    assert _same(_value("LS_RATIO_Z", "00:20", 3), math.nan)
    assert _same(_value("OI_CHANGE", "00:20", 1), 1002 / 1001 - 1)  # rows 00:10 / 00:05


# ---------------------------------------------------------------------------
# R10: values before T + 600 s do not depend on rows stamped at or after T.
# ---------------------------------------------------------------------------


def test_truncation_invariant(use_archive: Callable[[list[Row]], Path]) -> None:
    day0 = datetime(2024, 3, 1, tzinfo=UTC)
    rng = np.random.default_rng(7)
    n = 3 * 288
    oi = 1000.0 + rng.random(n) * 50.0
    ratio = 1.0 + rng.random(n)
    holes = {40, 41, 300}  # day 0 and day 1 06:00, more than a day before T
    rows: list[Row] = [
        (day0 + i * FIVE_MIN, float(oi[i]), float(ratio[i])) for i in range(n) if i not in holes
    ]
    cut = datetime(2024, 3, 3, 12, tzinfo=UTC)  # T = day 2 12:00
    t_ns = _ms(cut) * 1_000_000
    minute_ns = 60 * 10**9
    closes = np.arange(
        _ms(day0 + timedelta(days=1)) * 1_000_000,
        _ms(datetime(2024, 3, 3, 13, tzinfo=UTC)) * 1_000_000 + 1,
        minute_ns,
        dtype=np.int64,
    )
    before = closes < t_ns + 600 * 10**9
    fresh = (closes >= t_ns) & before

    def run(rows_used: list[Row]) -> dict[tuple[str, int], np.ndarray]:
        use_archive(rows_used)
        out: dict[tuple[str, int], np.ndarray] = {}
        for period in (3, 12, 288):
            out[("oi", period)] = aux_data.oi_change_asof(SYMBOL, closes, period)
            out[("z", period)] = aux_data.ls_ratio_z_asof(SYMBOL, closes, period)
            scalars = [aux_data.oi_change_at(SYMBOL, int(c), period) for c in closes[::37]]
            assert np.array_equal(
                np.asarray(scalars).view(np.int64), out[("oi", period)][::37].view(np.int64)
            )
        return out

    full = run(rows)
    truncated = run([r for r in rows if r[0] < cut])
    differs = False
    for key, values in full.items():
        cut_values = truncated[key]
        assert np.array_equal(values[before].view(np.int64), cut_values[before].view(np.int64)), key
        assert np.isfinite(values[fresh]).any(), key
        differs |= not np.array_equal(values[~before], cut_values[~before], equal_nan=True)
    assert differs, "rows at or after T must matter after T + 600 s (the test would be vacuous)"


# ---------------------------------------------------------------------------
# R15/R16: preflight.
# ---------------------------------------------------------------------------


def test_preflight_metrics_not_funding(use_archive: Callable[[list[Row]], Path]) -> None:
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


def test_preflight_rereads_a_refreshed_archive(use_archive: Callable[[list[Row]], Path]) -> None:
    """A lookup cached before a refresh is not served to the next launch check."""
    first_ok, last_ok = datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC)
    path = use_archive(_span(first_ok, last_ok - FIVE_MIN))
    with pytest.raises(AuxDataUnavailableError, match="end before"):
        aux_data.preflight(["OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-06")
    _write(path, [(last_ok, 1000.0, 1.0)])  # the next day's file lands
    aux_data.preflight(["OI_CHANGE"], [SYMBOL], "2024-01-01", "2024-01-06")
