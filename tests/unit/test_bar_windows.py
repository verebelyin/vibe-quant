"""Windowed catalog bar reads: ``extraction.load_bar_windows`` (row R17c).

R17c: only the row groups whose ``ts_init`` min/max statistics overlap some
window ``[start, end]`` are read, the decoded values are bit-identical to the
matching slice of a full ``load_catalog_bars`` read, ``_BAR_CACHE`` is
untouched (same length, same series objects), a missing bar dir is ``None``,
overlapping windows are deduplicated and the result is sorted by ``ts_init``.
R17d fixes the interface (``Iterable[tuple[int, int]]`` of inclusive ns windows,
every ``BarSeries`` column decoded), R17f the row mask (row-group pruning is
only the I/O filter: the returned series holds exactly the rows inside the
windows, so it is the full read's slice) and R17e the empty case: files exist
but nothing overlaps -> empty ``BarSeries``, never ``None``.

The tiny parquet fixtures carry NT's fixed-point price encoding (128-bit
little-endian int scaled by 1e16) so the real decoder runs unchanged; the
optional last test repeats the comparison on the real BTCUSDT 1m LAST file.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vibe_quant.validation import extraction
from vibe_quant.validation.extraction import load_bar_windows, load_catalog_bars

_BAR_TYPE = "TESTUSDT-PERP.TEST-1-MINUTE-LAST-EXTERNAL"
_START_NS = 1_577_836_800_000_000_000  # 2020-01-01T00:00:00Z
_MINUTE_NS = 60_000_000_000
_PRICE_SCALE = 10**16
_REAL_CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog"
_REAL_BAR_TYPE = "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"


def _encode_price(price: float) -> bytes:
    """NT fixed-point price: 128-bit little-endian int, low 8 bytes first."""
    raw = int(round(price * _PRICE_SCALE))
    return (raw & (2**64 - 1)).to_bytes(8, "little") + ((raw >> 64) & (2**64 - 1)).to_bytes(
        8, "little"
    )


def _write_bars(
    catalog: Path,
    rows: list[tuple[int, float]],
    *,
    row_group_size: int,
    name: str = "part.parquet",
    write_statistics: bool = True,
) -> Path:
    """Write 1m bars (``ts_init``, base price) as a catalog parquet file."""
    table = pa.table(
        {
            "open": pa.array([_encode_price(p) for _, p in rows], type=pa.binary(16)),
            "high": pa.array([_encode_price(p + 2.0) for _, p in rows], type=pa.binary(16)),
            "low": pa.array([_encode_price(p - 2.0) for _, p in rows], type=pa.binary(16)),
            "close": pa.array([_encode_price(p + 0.5) for _, p in rows], type=pa.binary(16)),
            "volume": pa.array([_encode_price(1.0) for _ in rows], type=pa.binary(16)),
            "ts_event": pa.array([ts for ts, _ in rows], type=pa.uint64()),
            "ts_init": pa.array([ts for ts, _ in rows], type=pa.uint64()),
        }
    )
    bar_dir = catalog / "data" / "bar" / _BAR_TYPE
    bar_dir.mkdir(parents=True, exist_ok=True)
    path = bar_dir / name
    pq.write_table(table, path, row_group_size=row_group_size, write_statistics=write_statistics)
    return path


def _minutes(prices: list[float]) -> list[tuple[int, float]]:
    """One bar per minute from ``_START_NS``, opening at each given price."""
    return [(_START_NS + i * _MINUTE_NS, price) for i, price in enumerate(prices)]


def _assert_matches_full_slice(series: object, full: object, expected_ts: np.ndarray) -> None:
    """Every decoded column equals the full read at ``expected_ts``."""
    ts = np.asarray(series.ts)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(ts, expected_ts)
    idx = np.searchsorted(np.asarray(full.ts), expected_ts)  # type: ignore[attr-defined]
    for name in ("open", "high", "low", "close"):
        np.testing.assert_array_equal(
            np.asarray(getattr(series, name)), np.asarray(getattr(full, name))[idx]
        )


def _spy_read_row_groups(monkeypatch: pytest.MonkeyPatch, calls: list[list[int]]) -> None:
    """Record the row groups every ``read_row_groups`` call asks for."""
    original = pq.ParquetFile.read_row_groups

    def spy(self: object, row_groups: object, columns: object = None, **kwargs: object) -> object:
        calls.append([int(group) for group in row_groups])  # type: ignore[union-attr]
        return original(self, row_groups, columns=columns, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", spy)


def test_window_read_matches_full_series_slice_and_leaves_bar_cache_untouched(
    tmp_path: Path,
) -> None:
    """Windowed values are the matching full-read slice; _BAR_CACHE untouched."""
    catalog = tmp_path / "catalog"
    _write_bars(
        catalog,
        _minutes([100.0, 101.5, 103.25, 99.75, 104.5, 102.0]),
        row_group_size=2,
    )

    full = load_catalog_bars(catalog, _BAR_TYPE)
    assert full is not None
    cache_keys = list(extraction._BAR_CACHE)
    cache_series_ids = {key: id(entry[1]) for key, entry in extraction._BAR_CACHE.items()}

    windows = [
        (_START_NS, _START_NS + _MINUTE_NS),  # bars 0, 1
        (_START_NS + 2 * _MINUTE_NS, _START_NS + 3 * _MINUTE_NS),  # bars 2, 3
    ]
    series = load_bar_windows(catalog, _BAR_TYPE, windows)
    assert series is not None

    expected_ts = np.array(
        [_START_NS + i * _MINUTE_NS for i in range(4)], dtype=np.int64
    )
    _assert_matches_full_slice(series, full, expected_ts)

    assert list(extraction._BAR_CACHE) == cache_keys
    assert {key: id(entry[1]) for key, entry in extraction._BAR_CACHE.items()} == cache_series_ids


def test_only_overlapping_row_groups_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only row groups whose ts_init statistics overlap a window are read."""
    catalog = tmp_path / "catalog"
    _write_bars(catalog, _minutes([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]), row_group_size=2)
    read_groups: list[list[int]] = []
    _spy_read_row_groups(monkeypatch, read_groups)

    # row groups are [bar0, bar1] [bar2, bar3] [bar4, bar5]: only the last one overlaps
    series = load_bar_windows(
        catalog, _BAR_TYPE, [(_START_NS + 5 * _MINUTE_NS, _START_NS + 5 * _MINUTE_NS)]
    )
    assert read_groups == [[2]]
    assert series is not None
    assert series.ts.tolist() == [_START_NS + 5 * _MINUTE_NS]

    # a window ending exactly on a row group's first ts_init still overlaps it
    read_groups.clear()
    series = load_bar_windows(
        catalog, _BAR_TYPE, [(_START_NS - _MINUTE_NS, _START_NS + 2 * _MINUTE_NS)]
    )
    assert read_groups == [[0, 1]]
    assert series is not None
    assert series.ts.tolist() == [_START_NS, _START_NS + _MINUTE_NS, _START_NS + 2 * _MINUTE_NS]


def test_row_group_without_statistics_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undecidable overlap (no ts_init statistics) fails open."""
    catalog = tmp_path / "catalog"
    rows = _minutes([100.0, 101.0, 102.0, 103.0])
    _write_bars(catalog, rows, row_group_size=2, write_statistics=False)
    read_groups: list[list[int]] = []
    _spy_read_row_groups(monkeypatch, read_groups)

    series = load_bar_windows(catalog, _BAR_TYPE, [(_START_NS, _START_NS + _MINUTE_NS)])
    assert read_groups == [[0, 1]]
    assert series is not None
    full = load_catalog_bars(catalog, _BAR_TYPE)
    assert full is not None
    _assert_matches_full_slice(
        series, full, np.array([_START_NS, _START_NS + _MINUTE_NS], dtype=np.int64)
    )


def test_overlapping_windows_dedupe_and_sort_by_ts_init(tmp_path: Path) -> None:
    """Overlapping windows concat -> sort -> dedupe exactly like the full read."""
    catalog = tmp_path / "catalog"
    _write_bars(catalog, _minutes([100.0, 101.0, 102.0, 103.0]), row_group_size=2)
    # second file: bar 2 again (duplicate ts_init, different price) and bar 4, out of order
    _write_bars(
        catalog,
        [(_START_NS + 4 * _MINUTE_NS, 104.0), (_START_NS + 2 * _MINUTE_NS, 999.0)],
        row_group_size=1,
        name="b.parquet",
    )

    full = load_catalog_bars(catalog, _BAR_TYPE)
    assert full is not None
    windows = [
        (_START_NS, _START_NS + 2 * _MINUTE_NS),
        (_START_NS + 2 * _MINUTE_NS, _START_NS + 4 * _MINUTE_NS),
    ]
    series = load_bar_windows(catalog, _BAR_TYPE, windows)
    assert series is not None

    assert len(series) == 5
    assert series.ts.tolist() == sorted(series.ts.tolist())
    expected_ts = np.array(
        [_START_NS + i * _MINUTE_NS for i in range(5)], dtype=np.int64
    )
    _assert_matches_full_slice(series, full, expected_ts)


def test_missing_bar_dir_returns_none_and_no_overlap_returns_empty_series(
    tmp_path: Path,
) -> None:
    """No bar dir or files -> None; files but no overlap -> empty series."""
    assert load_bar_windows(tmp_path / "absent", _BAR_TYPE, [(_START_NS, _START_NS)]) is None
    assert load_bar_windows(tmp_path / "empty", _BAR_TYPE, [(_START_NS, _START_NS)]) is None

    catalog = tmp_path / "catalog"
    _write_bars(catalog, _minutes([100.0, 101.0, 102.0, 103.0]), row_group_size=2)

    before = _START_NS - 10 * _MINUTE_NS
    empty = load_bar_windows(catalog, _BAR_TYPE, [(before, before + _MINUTE_NS)])
    assert empty is not None and len(empty) == 0

    no_windows = load_bar_windows(catalog, _BAR_TYPE, [])
    assert no_windows is not None and len(no_windows) == 0


@pytest.mark.skipif(
    not (_REAL_CATALOG / "data" / "bar" / _REAL_BAR_TYPE).is_dir(),
    reason="needs data/catalog with BTCUSDT 1m LAST",
)
def test_real_catalog_windows_match_full_read() -> None:
    """66 windows over BTCUSDT 1m LAST match the full slices."""
    full = load_catalog_bars(_REAL_CATALOG, _REAL_BAR_TYPE)
    assert full is not None
    step = len(full) // 67
    windows = [
        (int(full.ts[i * step]), int(full.ts[i * step]) + _MINUTE_NS - 1) for i in range(1, 67)
    ]

    series = load_bar_windows(_REAL_CATALOG, _REAL_BAR_TYPE, windows)
    assert series is not None
    parts = [
        full.ts[(full.ts >= start) & (full.ts <= end)] for start, end in windows
    ]
    expected_ts = np.unique(np.concatenate(parts))
    _assert_matches_full_slice(series, full, expected_ts)


_INT64_MAX = 2**63 - 1


def test_disjoint_windows_return_only_rows_inside_each_window(tmp_path: Path) -> None:
    """Two disjoint windows return their own rows, never the gap between."""
    catalog = tmp_path / "catalog"
    _write_bars(catalog, _minutes([100.0, 101.5, 103.25, 99.75, 104.5, 102.0]), row_group_size=2)
    full = load_catalog_bars(catalog, _BAR_TYPE)
    assert full is not None

    windows = [
        (_START_NS, _START_NS),  # bar 0
        (_START_NS + 5 * _MINUTE_NS, _START_NS + 5 * _MINUTE_NS),  # bar 5
    ]
    series = load_bar_windows(catalog, _BAR_TYPE, windows)
    assert series is not None

    expected_ts = np.array([_START_NS, _START_NS + 5 * _MINUTE_NS], dtype=np.int64)
    _assert_matches_full_slice(series, full, expected_ts)
    # the gap bars carry different prices, so a bounding-span union cannot pass this
    assert series.open.tolist() == [full.open[0], full.open[5]]
    assert series.high.tolist() == [full.high[0], full.high[5]]
    assert series.low.tolist() == [full.low[0], full.low[5]]
    assert series.close.tolist() == [full.close[0], full.close[5]]


@pytest.mark.parametrize(
    ("window", "match"),
    [
        ((-1, _START_NS), r"window #0 \(-1, 1577836800000000000\).*\[0, 9223372036854775807\]"),
        ((_START_NS, 2**63), r"window #0 \(1577836800000000000, 9223372036854775808\)"),
        ((_START_NS, _INT64_MAX + 10**30), r"window #0 .*\[0, 9223372036854775807\]"),
        ((_START_NS, -(10**30)), r"window #0 .*\[0, 9223372036854775807\]"),
        ((True, _START_NS), r"window #0 \(True, 1577836800000000000\).*integer"),
        ((_START_NS, False), r"window #0 .*integer"),
        ((_START_NS, 1.5e18), r"window #0 .*integer"),
        (("1", _START_NS), r"window #0 .*integer"),
        ((None, _START_NS), r"window #0 .*integer"),
        ((_START_NS + 1, _START_NS), r"window #0 .*start > end"),
    ],
)
def test_out_of_range_or_malformed_window_raises_value_error_naming_the_window(
    tmp_path: Path, window: tuple[object, object], match: str
) -> None:
    """Bounds are int64 ns in [0, 2**63-1], start <= end; no clamping."""
    catalog = tmp_path / "catalog"
    _write_bars(catalog, _minutes([100.0, 101.0, 102.0, 103.0]), row_group_size=2)
    with pytest.raises(ValueError, match=match):
        load_bar_windows(catalog, _BAR_TYPE, [window])  # type: ignore[list-item]


def test_window_validation_happens_before_the_bar_dir_is_looked_up(tmp_path: Path) -> None:
    """A bad window raises even when there are no bar files (no silent None)."""
    with pytest.raises(ValueError, match=r"window #1 \(0, 9223372036854775808\)"):
        load_bar_windows(tmp_path / "absent", _BAR_TYPE, [(0, 5), (0, 2**63)])


def test_window_bounds_at_the_int64_limits_are_accepted(tmp_path: Path) -> None:
    """0 and 2**63-1 are valid bounds; numpy integers are accepted."""
    catalog = tmp_path / "catalog"
    _write_bars(catalog, _minutes([100.0, 101.0, 102.0, 103.0]), row_group_size=2)
    series = load_bar_windows(catalog, _BAR_TYPE, [(np.int64(0), np.int64(_INT64_MAX))])
    assert series is not None and len(series) == 4
    series = load_bar_windows(catalog, _BAR_TYPE, [(_START_NS, _START_NS)])
    assert series is not None and len(series) == 1
