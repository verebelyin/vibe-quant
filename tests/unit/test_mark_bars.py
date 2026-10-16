"""MARK-price 1m bar type + loader (vibe-quant-2gyk1.1.2).

Liquidation triggers on the mark price, so the archive's mark klines
(``raw_mark_klines``) are materialised into the catalog as a MARK bar type and
served to the liquidation scanner as five compact numpy arrays (ts/open/low/
high/close) through an ``.npy`` memmap cache. These tests run entirely against temp
archives and temp catalogs: the worktree's data directories are symlinks into
the real datasets and must never be written.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest
from nautilus_trader.model.enums import PriceType

from vibe_quant.data import ingest, mark_bars
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    CatalogManager,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.data.mark_bars import (
    MarkDataMissing,
    MarkSeries,
    build_mark_bars,
    check_mark_coverage,
    load_mark_series,
    mark_bar_type,
    require_mark_coverage,
)

if TYPE_CHECKING:
    from pathlib import Path

MINUTE_MS = 60_000
_MINUTE_NS = 60_000_000_000

# 2023-11-14 22:13:20 UTC, snapped to the minute.
T0_MS = (1_700_000_000_000 // MINUTE_MS) * MINUTE_MS
THREE_DAYS = 3 * 24 * 60  # 4320 one-minute bars

EXPECTED_MARK_BAR_TYPE = "BTCUSDT-PERP.BINANCE-1-MINUTE-MARK-EXTERNAL"


def _mark_open(i: int) -> float:
    """Mark open of bar *i*: 8-decimal prices, so tick rounding is exercised."""
    return 100.0 + (i % 500) * 0.2 + 0.00012345


def _mark_row(i: int) -> tuple[float, ...]:
    """A mark-price row in Binance order (volume columns 0, count 60)."""
    o = _mark_open(i)
    return (o, o + 1.5, o - 1.5, o + 0.7 - (i % 7) * 0.2, 0.0, 0.0, 60, 0.0, 0.0)


def _mark_klines(count: int = THREE_DAYS) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for i in range(count):
        open_ms = T0_MS + i * MINUTE_MS
        o, h, low, c, *_rest = _mark_row(i)
        rows.append((open_ms, o, h, low, c, 0.0, open_ms + MINUTE_MS - 1, 0, 60, 0.0, 0.0))
    return rows


def _round_tick(value: float, precision: int = 1) -> float:
    """What klines_to_bars stores: the value formatted to the instrument tick."""
    return float(f"{value:.{precision}f}")


def _expected_close() -> np.ndarray:
    return np.array([_round_tick(_mark_row(i)[3]) for i in range(THREE_DAYS)], dtype=np.float64)


def _expected_open() -> np.ndarray:
    return np.array([_round_tick(_mark_row(i)[0]) for i in range(THREE_DAYS)], dtype=np.float64)


def _expected_ts() -> np.ndarray:
    return np.array([T0_MS + i * MINUTE_MS for i in range(THREE_DAYS)], dtype=np.int64) * 1_000_000


@pytest.fixture
def archive(tmp_path: Path) -> RawDataArchive:
    return RawDataArchive(tmp_path / "mark_archive.db")


@pytest.fixture
def catalog(tmp_path: Path) -> CatalogManager:
    return CatalogManager(tmp_path / "catalog")


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    return tmp_path / "catalog"


def _bar_dir(catalog_path: Path, bar_type: str) -> Path:
    return catalog_path / "data" / "bar" / bar_type


def _parquet_digest(bar_dir: Path) -> dict[str, str]:
    return {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(bar_dir.glob("*.parquet"))
    }


class TestMarkBarType:
    def test_mark_bar_type_string(self) -> None:
        assert mark_bar_type("BTCUSDT") == EXPECTED_MARK_BAR_TYPE


class TestMarkSeriesRoundtrip:
    def test_roundtrip_equals_inputs_rounded(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path
    ) -> None:
        archive.insert_mark_klines("BTCUSDT", "1m", _mark_klines(), "test")
        written = build_mark_bars(catalog, archive, "BTCUSDT")
        assert written == THREE_DAYS
        assert _bar_dir(catalog_path, EXPECTED_MARK_BAR_TYPE).is_dir()

        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert series.ts.dtype == np.int64
        assert series.low.dtype == np.float64
        assert series.high.dtype == np.float64
        assert series.close.dtype == np.float64
        assert series.open.dtype == np.float64

        # ts is the bar OPEN time (ts_event), strictly increasing.
        assert np.array_equal(series.ts, _expected_ts())
        assert np.all(np.diff(series.ts) > 0)

        # low/high/close equal the inputs formatted to the instrument tick.
        assert np.array_equal(series.low, np.array(
            [_round_tick(_mark_row(i)[2]) for i in range(THREE_DAYS)], dtype=np.float64
        ))
        assert np.array_equal(series.high, np.array(
            [_round_tick(_mark_row(i)[1]) for i in range(THREE_DAYS)], dtype=np.float64
        ))
        assert np.array_equal(series.close, _expected_close())
        # the bar open (the liquidation fill rule needs it), from the archive's open column
        assert np.array_equal(series.open, _expected_open())
        assert not np.array_equal(series.open, series.close)

    def test_no_mark_bars_returns_none(self, catalog_path: Path) -> None:
        assert load_mark_series(catalog_path, "BTCUSDT") is None

    def test_second_load_is_memmap_backed(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path
    ) -> None:
        archive.insert_mark_klines("BTCUSDT", "1m", _mark_klines(), "test")
        build_mark_bars(catalog, archive, "BTCUSDT")
        first = load_mark_series(catalog_path, "BTCUSDT")
        assert first is not None

        second = load_mark_series(catalog_path, "BTCUSDT")
        assert second is not None
        assert isinstance(second.ts, np.memmap)
        assert isinstance(second.low, np.memmap)
        assert isinstance(second.high, np.memmap)
        assert isinstance(second.close, np.memmap)
        assert isinstance(second.open, np.memmap)
        assert np.array_equal(second.close, first.close)
        assert np.array_equal(second.open, _expected_open())

    def test_changed_parquet_signature_rebuilds_cache(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path
    ) -> None:
        rows = _mark_klines()
        archive.insert_mark_klines("BTCUSDT", "1m", rows, "test")
        build_mark_bars(catalog, archive, "BTCUSDT")
        before = load_mark_series(catalog_path, "BTCUSDT")
        assert before is not None
        assert len(before.ts) == THREE_DAYS

        # One more minute -> new parquet (different size/signature) -> cache rebuild.
        extra_ms = T0_MS + THREE_DAYS * MINUTE_MS
        o, h, low, c, *_ = _mark_row(THREE_DAYS)
        archive.insert_mark_klines(
            "BTCUSDT",
            "1m",
            [(extra_ms, o, h, low, c, 0.0, extra_ms + MINUTE_MS - 1, 0, 60, 0.0, 0.0)],
            "test",
        )
        build_mark_bars(catalog, archive, "BTCUSDT")
        after = load_mark_series(catalog_path, "BTCUSDT")
        assert after is not None
        assert len(after.ts) == THREE_DAYS + 1
        assert after.ts[-1] == extra_ms * 1_000_000
        assert after.close[-1] == _round_tick(c)


def _last_and_mark_klines(count: int = 480) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    last_rows: list[tuple[float, ...]] = []
    mark_rows: list[tuple[float, ...]] = []
    for i in range(count):
        open_ms = T0_MS + i * MINUTE_MS
        o = 100.0 + (i % 500) * 0.2 + 0.00012345
        c = o + 0.7 - (i % 7) * 0.2
        # (open_time, open, high, low, close, volume, close_time, quote, count, taker, taker_quote)
        last_rows.append(
            (open_ms, o, o + 1.5, o - 1.5, c, 12.5, open_ms + MINUTE_MS - 1, 0.0, 10, 0.0, 0.0)
        )
        mark_rows.append((open_ms, o, o + 1.6, o - 1.6, c, 0.0, open_ms + MINUTE_MS - 1, 0, 60, 0.0, 0.0))
    return last_rows, mark_rows


class TestLastBarsUnaffected:
    def test_last_parquet_bytes_identical_after_adding_mark(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path
    ) -> None:
        last_rows, mark_rows = _last_and_mark_klines()
        archive.insert_klines("BTCUSDT", "1m", last_rows, "test")
        archive.insert_mark_klines("BTCUSDT", "1m", mark_rows, "test")

        instrument = create_instrument("BTCUSDT")
        catalog.write_instrument(instrument)
        last_bt = "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
        bars = klines_to_bars(
            archive.get_klines("BTCUSDT", "1m"),
            instrument.id,
            get_bar_type("BTCUSDT", "1m"),
            instrument.size_precision,
            instrument.price_precision,
        )
        catalog.write_bars(bars)
        before = _parquet_digest(_bar_dir(catalog_path, last_bt))
        assert before  # LAST bars landed

        build_mark_bars(catalog, archive, "BTCUSDT")

        after = _parquet_digest(_bar_dir(catalog_path, last_bt))
        assert after == before


def _series_from_ts(ts: np.ndarray) -> MarkSeries:
    n = len(ts)
    return MarkSeries(
        ts=ts,
        low=np.full(n, 99.0, dtype=np.float64),
        high=np.full(n, 101.0, dtype=np.float64),
        close=np.full(n, 100.0, dtype=np.float64),
        open=np.full(n, 100.0, dtype=np.float64),
    )


class TestRequireMarkCoverage:
    def _full(self) -> MarkSeries:
        ts = (T0_MS * 1_000_000 + np.arange(120, dtype=np.int64) * _MINUTE_NS)
        return _series_from_ts(ts)

    def test_none_series_raises_naming_symbol(self) -> None:
        with pytest.raises(MarkDataMissing) as exc:
            require_mark_coverage(None, "BTCUSDT", T0_MS * 1_000_000, T0_MS * 1_000_000)
        assert "BTCUSDT" in str(exc.value)

    def test_start_before_first_bar_raises(self) -> None:
        s = self._full()
        with pytest.raises(MarkDataMissing) as exc:
            require_mark_coverage(s, "BTCUSDT", s.ts[0] - _MINUTE_NS, s.ts[-1])
        assert "BTCUSDT" in str(exc.value)

    def test_end_after_last_bar_raises(self) -> None:
        s = self._full()
        with pytest.raises(MarkDataMissing) as exc:
            require_mark_coverage(s, "BTCUSDT", s.ts[0], s.ts[-1] + 2 * 3_600 * 1_000_000_000)
        assert "BTCUSDT" in str(exc.value)

    def test_internal_gap_raises_naming_symbol(self) -> None:
        ts = T0_MS * 1_000_000 + np.arange(180, dtype=np.int64) * _MINUTE_NS
        # Drop 120 minutes (indices 30..149): consecutive diff becomes 121 min.
        keep = np.concatenate([np.arange(30), np.arange(150, 180)])
        s = _series_from_ts(ts[keep])
        with pytest.raises(MarkDataMissing) as exc:
            require_mark_coverage(s, "BTCUSDT", s.ts[0], s.ts[-1])
        assert "BTCUSDT" in str(exc.value)

    def test_full_coverage_passes(self) -> None:
        s = self._full()
        require_mark_coverage(s, "BTCUSDT", s.ts[0], s.ts[-1])

    def test_gap_at_limit_passes(self) -> None:
        # Bars 0..19 then 79..119: the bridge (minute 19 -> 79) is exactly 60
        # minutes, which is within max_gap_minutes=60 (only > raises).
        idx = np.concatenate([np.arange(20), np.arange(79, 120)])
        ts = T0_MS * 1_000_000 + idx.astype(np.int64) * _MINUTE_NS
        s = _series_from_ts(ts)
        assert int((s.ts[20] - s.ts[19]) // _MINUTE_NS) == 60
        require_mark_coverage(s, "BTCUSDT", s.ts[0], s.ts[-1], max_gap_minutes=60)

    # Window is half-open [start_ns, end_ns): every minute with start <= m < end
    # must be a bar or inside an allowed gap. Bars 0..9 and 100..119 leave
    # minutes 10..99 missing (bar-to-bar diff 91 min > 60).
    def _gapped(self) -> MarkSeries:
        idx = np.concatenate([np.arange(10), np.arange(100, 120)])
        return _series_from_ts(T0_MS * 1_000_000 + idx.astype(np.int64) * _MINUTE_NS)

    def _at(self, minute: int) -> int:
        return T0_MS * 1_000_000 + minute * _MINUTE_NS

    def test_window_entirely_inside_a_gap_raises(self) -> None:
        # Bars at minutes [0, 180] only: the empty window [30, 150] is not covered.
        s = _series_from_ts(
            np.array([self._at(0), self._at(180)], dtype=np.int64)
        )
        with pytest.raises(MarkDataMissing) as exc:
            require_mark_coverage(s, "BTCUSDT", self._at(30), self._at(150))
        assert "BTCUSDT" in str(exc.value)

    def test_gap_straddling_window_start_raises(self) -> None:
        s = self._gapped()
        with pytest.raises(MarkDataMissing, match="gap"):
            require_mark_coverage(s, "BTCUSDT", self._at(5), self._at(120))

    def test_gap_straddling_window_end_raises(self) -> None:
        s = self._gapped()
        with pytest.raises(MarkDataMissing, match="gap"):
            require_mark_coverage(s, "BTCUSDT", self._at(0), self._at(50))

    def test_window_boundaries_are_half_open(self) -> None:
        s = self._gapped()
        # First missing minute is 10, last is 99.
        require_mark_coverage(s, "BTCUSDT", self._at(0), self._at(10))  # ends at last bar+1
        with pytest.raises(MarkDataMissing, match="gap"):
            require_mark_coverage(s, "BTCUSDT", self._at(0), self._at(11))  # includes minute 10
        require_mark_coverage(s, "BTCUSDT", self._at(100), self._at(120))  # starts at next bar
        with pytest.raises(MarkDataMissing, match="gap"):
            require_mark_coverage(s, "BTCUSDT", self._at(99), self._at(120))  # includes minute 99
        # The bar opening exactly at end_ns is not needed: last bar >= end - 60 s suffices.
        require_mark_coverage(s, "BTCUSDT", self._at(100), self._at(120))
        with pytest.raises(MarkDataMissing, match="ends"):
            require_mark_coverage(s, "BTCUSDT", self._at(100), self._at(121))

    def test_subminute_windows_overlapping_a_gap_raise(self) -> None:
        # Bars at minutes [0, 100]: the gap interval is (0, 100) minutes. A window
        # that starts or ends sub-minute inside it still overlaps the gap.
        s = _series_from_ts(np.array([self._at(0), self._at(100)], dtype=np.int64))
        half_minute = _MINUTE_NS // 2
        with pytest.raises(MarkDataMissing, match="gap"):
            require_mark_coverage(s, "BTCUSDT", self._at(100) - half_minute, self._at(100))
        with pytest.raises(MarkDataMissing, match="gap"):
            require_mark_coverage(s, "BTCUSDT", self._at(100) - 1, self._at(101))

    def test_oversized_gap_outside_window_passes(self) -> None:
        s = self._gapped()
        require_mark_coverage(s, "BTCUSDT", self._at(100), self._at(110))
        require_mark_coverage(s, "BTCUSDT", self._at(0), self._at(5))


class TestCheckMarkCoverage:
    """``check_mark_coverage`` loops symbols in order; first failure wins (M10)."""

    def test_check_mark_coverage_first_failure_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[object, str]] = []

        def fake_load(catalog_path: object, symbol: str) -> MarkSeries | None:
            seen.append((catalog_path, symbol))
            if symbol == "MISSINGUSDT":
                return None
            # Bars at minutes [0, 10) and [100, 120): the 91-minute gap
            # (minute 9 -> 100) sits inside the window and exceeds the 60 min limit.
            idx = np.concatenate([np.arange(10), np.arange(100, 120)]).astype(np.int64)
            return _series_from_ts(T0_MS * 1_000_000 + idx * _MINUTE_NS)

        monkeypatch.setattr(mark_bars, "load_mark_series", fake_load)
        start = T0_MS * 1_000_000
        end = start + 120 * _MINUTE_NS

        with pytest.raises(MarkDataMissing, match="gap") as exc:
            check_mark_coverage("cat", ["BTCUSDT"], start, end)
        assert "BTCUSDT" in str(exc.value)
        assert seen == [("cat", "BTCUSDT")]

        # A covered window passes: the gap lies outside [start, end), so no raise.
        seen.clear()
        check_mark_coverage("cat", ["BTCUSDT"], start, start + 10 * _MINUTE_NS)

        # Every symbol is visited in order when the first is covered: the second
        # symbol's absent series still raises (a symbols[:1] loop would pass).
        seen.clear()
        with pytest.raises(MarkDataMissing, match="MISSINGUSDT"):
            check_mark_coverage("cat", ["BTCUSDT", "MISSINGUSDT"], start, start + 10 * _MINUTE_NS)
        assert seen == [("cat", "BTCUSDT"), ("cat", "MISSINGUSDT")]

        # All symbols covered: nothing raises, and both were loaded in order.
        seen.clear()
        check_mark_coverage("cat", ["BTCUSDT", "ETHUSDT"], start, start + 10 * _MINUTE_NS)
        assert seen == [("cat", "BTCUSDT"), ("cat", "ETHUSDT")]

        # First failure wins: the first symbol's absent series raises before the
        # second symbol is even loaded.
        seen.clear()
        with pytest.raises(MarkDataMissing, match="MISSINGUSDT"):
            check_mark_coverage("cat", ["MISSINGUSDT", "BTCUSDT"], start, end)
        assert seen == [("cat", "MISSINGUSDT")]


def _build(archive: RawDataArchive, catalog: CatalogManager, count: int) -> None:
    archive.insert_mark_klines("BTCUSDT", "1m", _mark_klines(count), "test")
    build_mark_bars(catalog, archive, "BTCUSDT")


def _generation_dirs(catalog_path: Path) -> list[Path]:
    root = catalog_path / "cache" / "mark" / "BTCUSDT"
    return [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".tmp-")]


def _bar_files(catalog_path: Path) -> list[Path]:
    return sorted(_bar_dir(catalog_path, EXPECTED_MARK_BAR_TYPE).glob("*.parquet"))


def _append_mark_file(catalog: CatalogManager, start: int, stop: int) -> None:
    """Write mark bars [start, stop) as an extra parquet file (no clearing)."""
    instrument = create_instrument("BTCUSDT")
    rows = [
        {
            "open_time": r[0],
            "close_time": r[6],
            "open": r[1],
            "high": r[2],
            "low": r[3],
            "close": r[4],
            "volume": 0.0,
        }
        for r in _mark_klines(stop)[start:stop]
    ]
    bars = klines_to_bars(
        rows,  # type: ignore[arg-type]
        instrument.id,
        get_bar_type("BTCUSDT", "1m", price_type=PriceType.MARK),
        instrument.size_precision,
        instrument.price_precision,
    )
    catalog.write_bars(bars)


class TestCacheGenerations:
    def test_interleaved_writers_do_not_share_files_or_fail(
        self,
        archive: RawDataArchive,
        catalog: CatalogManager,
        catalog_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Writer A is parked mid-write while writer B completes the same cache."""
        _build(archive, catalog, THREE_DAYS)
        a_saved = threading.Event()
        release_a = threading.Event()
        real_save = np.save
        parked = {"done": False}

        def save(file: object, arr: object, *args: object, **kwargs: object) -> None:
            real_save(file, arr, *args, **kwargs)  # type: ignore[arg-type]
            if threading.current_thread().name == "writer-a" and not parked["done"]:
                parked["done"] = True
                a_saved.set()
                assert release_a.wait(10)

        monkeypatch.setattr(np, "save", save)
        results: dict[str, object] = {}

        def writer_a() -> None:
            try:
                results["a"] = load_mark_series(catalog_path, "BTCUSDT")
            except BaseException as e:  # surfaced via the assertion below
                results["a"] = e

        thread = threading.Thread(target=writer_a, name="writer-a")
        thread.start()
        assert a_saved.wait(10)
        b = load_mark_series(catalog_path, "BTCUSDT")  # B finishes while A is parked
        release_a.set()
        thread.join(10)

        a = results["a"]
        assert isinstance(a, MarkSeries), f"writer A failed: {a!r}"
        assert b is not None
        for series in (a, b):
            assert {len(x) for x in series} == {THREE_DAYS}
            assert np.array_equal(series.ts, _expected_ts())
            assert np.array_equal(series.close, _expected_close())
        assert len(_generation_dirs(catalog_path)) == 1

    def test_parquet_rewritten_during_decode_never_publishes_stale_generation(
        self,
        archive: RawDataArchive,
        catalog: CatalogManager,
        catalog_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _build(archive, catalog, THREE_DAYS - 100)
        real_decode = mark_bars._decode_mark_bars
        calls = {"n": 0}

        def decode(files: list[Path]) -> MarkSeries:
            series = real_decode(files)
            calls["n"] += 1
            if calls["n"] == 1:  # a concurrent build_mark_bars lands mid-decode
                archive.insert_mark_klines(
                    "BTCUSDT", "1m", _mark_klines(THREE_DAYS)[THREE_DAYS - 100 :], "test"
                )
                build_mark_bars(catalog, archive, "BTCUSDT")
            return series

        monkeypatch.setattr(mark_bars, "_decode_mark_bars", decode)
        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert calls["n"] == 2  # stale decode discarded, parquet re-decoded
        assert len(series.ts) == THREE_DAYS
        assert np.array_equal(series.close, _expected_close())
        monkeypatch.undo()
        again = load_mark_series(catalog_path, "BTCUSDT")
        assert again is not None and len(again.ts) == THREE_DAYS

    def test_parquet_file_added_during_decode_is_not_missed(
        self,
        archive: RawDataArchive,
        catalog: CatalogManager,
        catalog_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An append that adds a NEW parquet file mid-decode forces a re-decode."""
        _build(archive, catalog, THREE_DAYS - 100)
        real_decode = mark_bars._decode_mark_bars
        calls = {"n": 0}

        def decode(files: list[Path]) -> MarkSeries:
            series = real_decode(files)
            calls["n"] += 1
            if calls["n"] == 1:
                _append_mark_file(catalog, THREE_DAYS - 100, THREE_DAYS)
            return series

        monkeypatch.setattr(mark_bars, "_decode_mark_bars", decode)
        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert calls["n"] == 2
        assert len(_bar_files(catalog_path)) == 2
        assert len(series.ts) == THREE_DAYS
        assert np.array_equal(series.close, _expected_close())

    def test_cached_read_revalidates_against_current_parquet(
        self,
        archive: RawDataArchive,
        catalog: CatalogManager,
        catalog_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cache hit for an outdated signature is not returned once parquet moved on."""
        _build(archive, catalog, THREE_DAYS - 100)
        assert load_mark_series(catalog_path, "BTCUSDT") is not None  # publish gen 1
        real_load = mark_bars._load_generation
        calls = {"n": 0}

        def load(root: Path, signature: object) -> MarkSeries | None:
            got = real_load(root, signature)  # type: ignore[arg-type]
            calls["n"] += 1
            if calls["n"] == 1:
                assert got is not None
                _append_mark_file(catalog, THREE_DAYS - 100, THREE_DAYS)
            return got

        monkeypatch.setattr(mark_bars, "_load_generation", load)
        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert len(series.ts) == THREE_DAYS
        assert np.array_equal(series.close, _expected_close())

    @pytest.mark.parametrize("damage", ["short_close", "meta_length", "missing_array", "dtype"])
    def test_inconsistent_generation_is_rebuilt_not_trusted(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path, damage: str
    ) -> None:
        _build(archive, catalog, THREE_DAYS)
        first = load_mark_series(catalog_path, "BTCUSDT")
        assert first is not None
        del first
        (gen,) = _generation_dirs(catalog_path)

        def replace_array(name: str, arr: np.ndarray) -> None:
            tmp = gen / f"{name}.swap.npy"
            np.save(tmp, arr)
            os.replace(tmp, gen / f"{name}.npy")

        if damage == "short_close":
            replace_array("close", _expected_close()[:2])  # reviewer: lengths [n, n, n, 2]
        elif damage == "meta_length":
            meta = (gen / "meta.json").read_text().replace(f'"length": {THREE_DAYS}', '"length": 2')
            (gen / "meta.json").write_text(meta)
        elif damage == "missing_array":
            (gen / "low.npy").unlink()
        else:
            replace_array("high", _expected_close().astype(np.float32))

        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert {len(x) for x in series} == {THREE_DAYS}
        assert series.low.dtype == series.high.dtype == np.float64
        assert np.array_equal(series.close, _expected_close())
        assert np.array_equal(series.ts, _expected_ts())

    def test_four_array_generation_from_before_open_is_rebuilt_not_trusted(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path
    ) -> None:
        """A cache left by the layout-1 loader (ts/low/high/close, keyed by the bare parquet
        signature, meta without ``layout``) is ignored and pruned; the loader rebuilds
        from parquet and serves the open."""
        _build(archive, catalog, THREE_DAYS)
        _files, signature = mark_bars._snapshot(_bar_dir(catalog_path, EXPECTED_MARK_BAR_TYPE))
        root = catalog_path / "cache" / "mark" / "BTCUSDT"
        sig_json = mark_bars._signature_json(signature)
        old = root / hashlib.sha256(json.dumps(sig_json).encode()).hexdigest()
        old.mkdir(parents=True)
        for name, arr in (
            ("ts", _expected_ts()),
            ("low", _expected_close()),
            ("high", _expected_close()),
            ("close", _expected_close()),
        ):
            np.save(old / f"{name}.npy", arr)
        (old / "meta.json").write_text(json.dumps({"signature": sig_json, "length": THREE_DAYS}))

        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert np.array_equal(series.open, _expected_open())
        (gen,) = _generation_dirs(catalog_path)
        assert gen != old and not old.exists()
        assert json.loads((gen / "meta.json").read_text())["layout"] == mark_bars._CACHE_LAYOUT

    def test_generation_with_a_stale_layout_in_meta_is_not_trusted(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path
    ) -> None:
        """Same key and all five arrays present, but meta says layout 1: rebuilt (a wrong
        open array planted in it is never served)."""
        _build(archive, catalog, THREE_DAYS)
        assert load_mark_series(catalog_path, "BTCUSDT") is not None
        (gen,) = _generation_dirs(catalog_path)
        tmp = gen / "open.swap.npy"
        np.save(tmp, _expected_close())
        os.replace(tmp, gen / "open.npy")
        meta = json.loads((gen / "meta.json").read_text())
        meta["layout"] = 1
        (gen / "meta.json").write_text(json.dumps(meta))

        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert np.array_equal(series.open, _expected_open())

    def test_superseded_generation_and_temp_dirs_are_pruned(
        self, archive: RawDataArchive, catalog: CatalogManager, catalog_path: Path
    ) -> None:
        _build(archive, catalog, THREE_DAYS - 10)
        assert load_mark_series(catalog_path, "BTCUSDT") is not None
        old = _generation_dirs(catalog_path)
        archive.insert_mark_klines("BTCUSDT", "1m", _mark_klines()[THREE_DAYS - 10 :], "test")
        build_mark_bars(catalog, archive, "BTCUSDT")
        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None and len(series.ts) == THREE_DAYS
        new = _generation_dirs(catalog_path)
        assert len(new) == 1 and new != old
        assert [p for p in new[0].parent.iterdir() if p.name.startswith(".tmp-")] == []


class TestRebuildFromArchive:
    def test_rebuild_writes_last_and_mark_and_skips_mark_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive = RawDataArchive(tmp_path / "rebuild_archive.db")
        last_rows, mark_rows = _last_and_mark_klines()
        archive.insert_klines("BTCUSDT", "1m", last_rows, "test")
        archive.insert_mark_klines("BTCUSDT", "1m", mark_rows, "test")
        # ETHUSDT has mark rows only: not in raw_klines, so it must be skipped.
        archive.insert_mark_klines("ETHUSDT", "1m", mark_rows, "test")

        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)
        catalog_path = tmp_path / "rebuild_catalog"
        ingest.rebuild_from_archive(catalog_path=catalog_path, verbose=False)

        assert _bar_dir(catalog_path, "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").is_dir()
        assert _bar_dir(catalog_path, EXPECTED_MARK_BAR_TYPE).is_dir()

        series = load_mark_series(catalog_path, "BTCUSDT")
        assert series is not None
        assert len(series.ts) == len(mark_rows)

        # mark-only symbol: no LAST and no MARK bars written.
        assert not _bar_dir(catalog_path, "ETHUSDT-PERP.BINANCE-1-MINUTE-MARK-EXTERNAL").exists()
