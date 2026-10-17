"""Archive + Binance Vision funding_interval_hours ingest (vibe-quant-x0axf.13 P1).

The archive stores each archived funding row's real settlement interval next to
the rate, so the hole rule can look for missing settlements on a symbol's own
2h/4h grid instead of the fixed 8h one. Contract B rows R36-R40 (T2 owns the
ingest half; the hole rule itself is a later task).

All HTTP is stubbed (``httpx.MockTransport`` or a monkeypatched downloader) and
every archive lives in ``tmp_path``.
"""

from __future__ import annotations

import io
import logging
import sqlite3
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from vibe_quant.data import downloader, ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.downloader import MonthlyDownloadError

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

FUNDING_CSV_HEADER = "calc_time,funding_interval_hours,last_funding_rate"

# The OLD archive schema: raw_funding_rates only, no raw_funding_intervals.
OLD_SCHEMA = """
CREATE TABLE raw_funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    funding_time INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    mark_price REAL,
    source TEXT NOT NULL,
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, funding_time)
);
"""

HOUR_MS = 3_600_000


def _ms(year: int, month: int, day: int = 1) -> int:
    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


def _funding_zip(symbol: str, year: int, month: int, body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{symbol}-fundingRate-{year}-{month:02d}.csv", body)
    return buf.getvalue()


def _client_for(served: dict[str, bytes]) -> httpx.Client:
    """A client serving Vision zips by filename; any other URL is a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        item = served.get(request.url.path.rsplit("/", 1)[-1])
        if item is None:
            return httpx.Response(404)
        return httpx.Response(200, content=item)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _kline(open_ms: int, price: float = 100.0) -> tuple[object, ...]:
    return (open_ms, price, price + 1, price - 1, price + 0.5, 10.0, open_ms + 59_999)


class _FakeInstrument:
    id = "BTCUSDT-PERP.BINANCE"
    size_precision = 8
    price_precision = 2


class _FakeCatalog:
    def write_instrument(self, *a: object, **k: object) -> None: ...
    def clear_bar_data(self, *a: object, **k: object) -> None: ...
    def write_bars(self, *a: object, **k: object) -> None: ...


def test_parse_vision_funding_csv(caplog: pytest.LogCaptureFixture) -> None:
    """R36: header + 3 bad rows skipped (one warning); 404 -> None."""
    body = "\n".join(
        [
            FUNDING_CSV_HEADER,
            "1668009600000,8,0.0001",
            "1668010500000,,0.0002",
            "1668011400000,8.0,0.0003",
            "1668012300000,0,0.0004",
            "1668013200000,4,0.0005",
            "",
        ]
    )
    served = {"SOLUSDT-fundingRate-2022-11.zip": _funding_zip("SOLUSDT", 2022, 11, body)}
    with caplog.at_level(logging.WARNING, logger="vibe_quant.data.downloader"):
        with _client_for(served) as client:
            rows = downloader.download_monthly_funding_intervals(
                "SOLUSDT", 2022, 11, client=client
            )
        assert rows == [(1668009600000, 8), (1668013200000, 4)]
        assert "skipped 3 funding interval rows for SOLUSDT 2022-11" in caplog.text

        with _client_for({}) as client:
            assert (
                downloader.download_monthly_funding_intervals("SOLUSDT", 2022, 12, client=client)
                is None
            )


def test_old_archive_gains_interval_table(tmp_path: Path) -> None:
    """R38: opening an old-schema archive adds the table, rate rows untouched."""
    db = tmp_path / "old.db"
    times = [_ms(2026, 7, 5), _ms(2026, 7, 5) + 8 * HOUR_MS, _ms(2026, 7, 6)]
    raw = sqlite3.connect(db)
    try:
        raw.executescript(OLD_SCHEMA)
        raw.executemany(
            "INSERT INTO raw_funding_rates "
            "(symbol, funding_time, funding_rate, mark_price, source) VALUES (?, ?, ?, ?, ?)",
            [("SOLUSDT", t, 0.0001, 50000.0, "binance_api") for t in times],
        )
        raw.commit()
        before = [tuple(r) for r in raw.execute("SELECT * FROM raw_funding_rates ORDER BY id")]
        tables_before = {
            r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        raw.close()
    assert "raw_funding_intervals" not in tables_before

    archive = RawDataArchive(db)
    try:
        after = [
            tuple(r) for r in archive.conn.execute("SELECT * FROM raw_funding_rates ORDER BY id")
        ]
        tables_after = {
            r[0]
            for r in archive.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert archive.get_funding_intervals("SOLUSDT") == {}
    finally:
        archive.close()

    assert "raw_funding_intervals" in tables_after
    assert after == before
    assert [r[2] for r in after] == times


def test_insert_funding_intervals_is_append_only(tmp_path: Path) -> None:
    """A second source can never overwrite an already-archived interval row."""
    archive = RawDataArchive(tmp_path / "append.db")
    t = _ms(2025, 12, 15)
    try:
        assert archive.insert_funding_intervals("SOLUSDT", [(t, 8)], "binance_vision") == 1
        assert archive.insert_funding_intervals("SOLUSDT", [(t, 4)], "binance_api") == 0

        assert archive.get_funding_intervals("SOLUSDT") == {t: 8}
        raw = archive.conn.execute(
            "SELECT interval_hours, source FROM raw_funding_intervals "
            "WHERE symbol = ? AND funding_time = ?",
            ("SOLUSDT", t),
        ).fetchone()
        assert (int(raw[0]), raw[1]) == (8, "binance_vision")
    finally:
        archive.close()


def test_december_anchor_rolls_into_january(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A December anchor resumes at the next January, not at its own December."""
    archive = RawDataArchive(tmp_path / "rollover.db")
    calls: list[tuple[int, int]] = []

    def fake(symbol: str, year: int, month: int, *a: Any, **k: Any) -> list[tuple[int, int]]:
        calls.append((year, month))
        return []

    monkeypatch.setattr(ingest, "download_monthly_funding_intervals", fake)
    try:
        archive.insert_funding_intervals("SOLUSDT", [(_ms(2025, 12, 15), 8)], "binance_vision")

        assert (
            ingest.ingest_funding_intervals(
                "SOLUSDT", archive, now=datetime(2026, 3, 10, tzinfo=UTC)
            )
            == 0
        )
        assert calls == [(2026, 1), (2026, 2)]
    finally:
        archive.close()


def test_ingest_months_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R37: only complete months up to ``now``; the second call is a no-op."""
    archive = RawDataArchive(tmp_path / "months.db")
    calls: list[tuple[int, int]] = []

    def fake(symbol: str, year: int, month: int, *a: Any, **k: Any) -> list[tuple[int, int]]:
        calls.append((year, month))
        return [(_ms(year, month, 5), 8)]

    monkeypatch.setattr(ingest, "download_monthly_funding_intervals", fake)
    try:
        archive.insert_funding_rates("SOLUSDT", [(_ms(2026, 7, 5), 0.0001, 1.0)], "seed")
        now = datetime(2026, 10, 17, tzinfo=UTC)

        assert ingest.ingest_funding_intervals("SOLUSDT", archive, now=now) == 3
        assert calls == [(2026, 7), (2026, 8), (2026, 9)]
        assert archive.get_funding_intervals("SOLUSDT") == {
            _ms(2026, 7, 5): 8,
            _ms(2026, 8, 5): 8,
            _ms(2026, 9, 5): 8,
        }

        assert ingest.ingest_funding_intervals("SOLUSDT", archive, now=now) == 0
        assert calls == [(2026, 7), (2026, 8), (2026, 9)]
    finally:
        archive.close()


def test_no_funding_no_interval_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A symbol with no archived rates gives no month to start from."""
    archive = RawDataArchive(tmp_path / "empty.db")
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        ingest,
        "download_monthly_funding_intervals",
        lambda *a, **k: calls.append((1, 1)) or [],
    )
    try:
        assert ingest.ingest_funding_intervals("SOLUSDT", archive) == 0
        assert calls == []
    finally:
        archive.close()


def test_update_survives_interval_fetch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R39: a ConnectError on the interval fetch warns and never aborts klines."""
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    archive = RawDataArchive(tmp_path / "update.db")
    try:
        archive.insert_klines("BTCUSDT", "1m", [_kline(now_ms - 600_000)], "seed")
        # Old enough that a complete Vision month exists to request.
        archive.insert_funding_rates("BTCUSDT", [(_ms(2025, 1, 5), 0.0001, 1.0)], "seed")

        def boom(*a: Any, **k: Any) -> list[tuple[int, int]]:
            raise httpx.ConnectError("down")

        monkeypatch.setattr(ingest, "download_funding_rates", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "download_monthly_funding_intervals", boom)
        monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: [_kline(now_ms - 180_000)])
        monkeypatch.setattr(ingest, "create_instrument", lambda _s: _FakeInstrument())
        monkeypatch.setattr(ingest, "klines_to_bars", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "aggregate_bars", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "get_bar_type", lambda *a, **k: None)

        with caplog.at_level(logging.WARNING, logger="vibe_quant.data.ingest"):
            counts = ingest.update_symbol(
                "BTCUSDT", archive=archive, catalog=_FakeCatalog(), verbose=False  # type: ignore[arg-type]
            )

        assert counts["new_funding_intervals"] == 0
        assert counts["new_klines"] == 1
        assert "funding interval fetch failed" in caplog.text
    finally:
        archive.close()


def test_ingest_survives_interval_fetch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R39b: rates are stored and returned even when the interval fetch fails."""
    archive = RawDataArchive(tmp_path / "ingest.db")
    base = _ms(2026, 7, 5)
    rates = [
        (base, 0.0001, 1.0),
        (base + 8 * HOUR_MS, 0.0002, 1.0),
        (base + 16 * HOUR_MS, 0.0003, 1.0),
    ]

    def boom(*a: Any, **k: Any) -> list[tuple[int, int]]:
        raise MonthlyDownloadError("vision down")

    monkeypatch.setattr(ingest, "download_funding_rates", lambda *a, **k: rates)
    monkeypatch.setattr(ingest, "download_monthly_funding_intervals", boom)
    try:
        with caplog.at_level(logging.WARNING, logger="vibe_quant.data.ingest"):
            assert ingest.ingest_funding_rates("SOLUSDT", archive=archive, verbose=False) == 3

        assert "funding interval fetch failed" in caplog.text
        rate_times = [tuple(r)[2] for r in archive.get_funding_rates("SOLUSDT")]
        assert rate_times == [t for t, _, _ in rates]
    finally:
        archive.close()


def test_cli_funding_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R40: the subcommand backfills and prints the per-symbol count."""
    archive = RawDataArchive(tmp_path / "cli.db")
    monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)
    base = _ms(2026, 9, 5)
    monkeypatch.setattr(
        ingest,
        "download_monthly_funding_intervals",
        lambda *a, **k: [(base, 8), (base + 8 * HOUR_MS, 8), (base + 16 * HOUR_MS, 8)],
    )
    try:
        archive.insert_funding_rates("SOLUSDT", [(base, 0.0001, 1.0)], "seed")

        assert ingest.main(["funding-intervals", "--symbols", "SOLUSDT"]) == 0
        assert "SOLUSDT: archived 3 funding intervals" in capsys.readouterr().out
    finally:
        archive.close()
