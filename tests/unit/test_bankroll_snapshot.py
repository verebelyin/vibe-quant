"""Report-local market snapshot: one stable seed catalog, a read-only archive backup and
private per-engine clones (contract C17 fs part and C26 scratch reclaim, design 2gyk1.3)."""

from __future__ import annotations

import hashlib
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.bankroll import snapshot as snap
from vibe_quant.bankroll.snapshot import (
    MarketSnapshotError,
    capture_market,
    clone_for_engine,
    reclaim_abandoned,
    report_root,
)

_ROOT = Path(__file__).resolve().parents[2]


def _tree(root: Path) -> dict[str, bytes | None]:
    """relpath -> content (``None`` for a directory)."""
    out: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        out[rel] = None if path.is_dir() else path.read_bytes()
    return out


def _make_catalog(root: Path) -> Path:
    (root / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL").mkdir(parents=True)
    (
        root / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL" / "part-0.parquet"
    ).write_bytes(b"bars-v1" * 100)
    (root / "data" / "crypto_perpetual").mkdir(parents=True)
    (root / "data" / "crypto_perpetual" / "BTCUSDT.parquet").write_bytes(b"instrument-v1")
    (root / "empty_dir").mkdir()
    return root


_ARCHIVE_DDL = (
    """CREATE TABLE raw_klines (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, interval TEXT NOT NULL,
        open_time INTEGER NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
        taker_buy_volume REAL, source TEXT NOT NULL, UNIQUE(symbol, interval, open_time))""",
    "CREATE INDEX idx_raw_klines_symbol_time ON raw_klines(symbol, interval, open_time)",
    """CREATE TABLE raw_funding_rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
        funding_rate REAL NOT NULL, mark_price REAL, source TEXT NOT NULL,
        UNIQUE(symbol, funding_time))""",
    "CREATE INDEX idx_raw_funding_symbol_time ON raw_funding_rates(symbol, funding_time)",
    """CREATE TABLE raw_funding_intervals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
        interval_hours INTEGER NOT NULL, source TEXT NOT NULL, UNIQUE(symbol, funding_time))""",
    """CREATE TABLE raw_mark_klines (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, interval TEXT NOT NULL,
        open_time INTEGER NOT NULL, close REAL NOT NULL, source TEXT NOT NULL)""",
    """CREATE TABLE download_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbols TEXT NOT NULL, source TEXT NOT NULL)""",
)
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _make_archive(path: Path, rows: int, *, klines: int = 0) -> Path:
    """An archive shaped like ``raw_data.db``: funding and intervals for three symbols
    (``rows`` each, inserted interleaved so ids are not grouped by symbol), ``klines``
    1m and 4h klines per symbol, mark klines and download sessions."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    for ddl in _ARCHIVE_DDL:
        conn.execute(ddl)
    for i in range(rows):
        for n, symbol in enumerate(_SYMBOLS):
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate, mark_price,"
                " source) VALUES (?, ?, ?, NULL, 'test')",
                (symbol, i, 0.0001 * i * (n + 1)),
            )
            conn.execute(
                "INSERT INTO raw_funding_intervals (symbol, funding_time, interval_hours, source)"
                " VALUES (?, ?, ?, 'test')",
                (symbol, i, 4 + 4 * (i % 2)),
            )
    for symbol in _SYMBOLS:
        for interval in ("1m", "4h"):
            conn.executemany(
                "INSERT INTO raw_klines (symbol, interval, open_time, close, volume,"
                " taker_buy_volume, source) VALUES (?, ?, ?, 1.5, 10.0, 4.0, 'test')",
                [(symbol, interval, t) for t in range(klines)],
            )
        conn.executemany(
            "INSERT INTO raw_mark_klines (symbol, interval, open_time, close, source)"
            " VALUES (?, '1m', ?, 1.5, 'test')",
            [(symbol, t) for t in range(klines)],
        )
    conn.execute("INSERT INTO download_sessions (symbols, source) VALUES ('[]', 'test')")
    conn.commit()
    conn.close()
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(
    path: Path, table: str = "raw_funding_rates", symbol: str | None = "BTCUSDT"
) -> list[tuple[object, ...]]:
    """Every column of ``table`` (``symbol`` None: all symbols), in id order."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if symbol is None:
            return conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
        return conn.execute(
            f"SELECT * FROM {table} WHERE symbol = ? ORDER BY id", (symbol,)
        ).fetchall()
    finally:
        conn.close()


def _schema(path: Path) -> set[tuple[str, str, str]]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return set(
            conn.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
        )
    finally:
        conn.close()


BTC = ("BTCUSDT",)


def test_market_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The source changes during the first capture: the capture retries once and the seed
    is the stable revision; later source changes never reach a clone."""
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 5)
    archive_sha = _sha(archive)
    bars = (
        source / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL" / "part-0.parquet"
    )

    real_clone = snap._clone_tree
    calls: list[Path] = []

    def racing_clone(src: Path, dst: Path) -> None:
        real_clone(src, dst)
        if not calls:
            bars.write_bytes(b"bars-v2" * 100)  # same size, new mtime and content
        calls.append(dst)

    monkeypatch.setattr(snap, "_clone_tree", racing_clone)
    root = report_root(tmp_path / "scratch", 41)
    snapshot = capture_market(source, archive, root, symbols=BTC)

    assert len(calls) == 2  # one retry
    assert snapshot.root == root
    assert snapshot.seed == root / "seed"
    assert snapshot.archive == root / "archive.db"
    assert snapshot.source_catalog == source.resolve()
    assert _tree(snapshot.seed) == _tree(source)
    assert _tree(snapshot.seed)[
        "data/bar/BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL/part-0.parquet"
    ] == (b"bars-v2" * 100)
    assert _rows(snapshot.archive) == _rows(archive)
    assert _sha(archive) == archive_sha

    monkeypatch.setattr(snap, "_clone_tree", real_clone)
    seed_tree = _tree(snapshot.seed)
    # A source change after the seed was captured reaches no clone.
    bars.write_bytes(b"bars-v3" * 100)
    (source / "data" / "new.parquet").write_bytes(b"late")
    baseline = clone_for_engine(snapshot, root / "baseline" / "catalog")
    worker = clone_for_engine(snapshot, root / "w0" / "catalog")
    assert baseline == root / "baseline" / "catalog"
    assert _tree(baseline) == seed_tree == _tree(worker)

    # A write made in a clone reaches neither the seed, another clone nor the source.
    source_tree = _tree(source)
    (worker / "data" / "crypto_perpetual" / "BTCUSDT.parquet").write_bytes(b"engine-write")
    (worker / "data" / "written.parquet").write_bytes(b"x")
    assert _tree(snapshot.seed) == seed_tree
    assert _tree(baseline) == seed_tree
    assert _tree(source) == source_tree


def test_market_snapshot_changed_twice_fails_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 3)
    real_clone = snap._clone_tree
    counter = iter(range(100))

    def always_racing(src: Path, dst: Path) -> None:
        real_clone(src, dst)
        (source / "data" / f"tick-{next(counter)}.parquet").write_bytes(b"t")

    monkeypatch.setattr(snap, "_clone_tree", always_racing)
    root = report_root(tmp_path / "scratch", 42)
    with pytest.raises(MarketSnapshotError) as err:
        capture_market(source, archive, root, symbols=BTC)
    assert err.value.code == "market_snapshot_changed"
    assert not (root / "seed").exists()
    assert not (root / "archive.db").exists()
    # The source only holds what the racing writer added: nothing of the capture.
    assert sorted(p.name for p in (source / "data").glob("tick-*")) == [
        "tick-0.parquet",
        "tick-1.parquet",
    ]


@pytest.mark.parametrize("deletes", [1, 100])
def test_market_snapshot_file_deleted_mid_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deletes: int
) -> None:
    """A catalog writer deletes a file between the walk's listing and its ``lstat`` (or the
    copy): a source change, so the capture retries once, then fails
    ``market_snapshot_changed`` -- never a raw ``FileNotFoundError``."""
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 2)
    real_walk = os.walk
    victims = iter(range(deletes))

    def deleting_walk(top: object, **kwargs: Any) -> Any:
        for dirpath, dirnames, filenames in real_walk(top, **kwargs):  # type: ignore[call-overload]
            here = Path(dirpath).resolve() == source.resolve() / "data" / "crypto_perpetual"
            if here and next(victims, None) is not None:
                name = f"gone-{len(filenames)}.parquet"
                (Path(dirpath) / name).write_bytes(b"x")
                filenames = [*filenames, name]
                (Path(dirpath) / name).unlink()
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(snap.os, "walk", deleting_walk)
    root = report_root(tmp_path / "scratch", 43)
    if deletes == 1:
        snapshot = capture_market(source, archive, root, symbols=BTC)
        assert _tree(snapshot.seed) == _tree(source)
        return
    with pytest.raises(MarketSnapshotError) as err:
        capture_market(source, archive, root, symbols=BTC)
    assert err.value.code == "market_snapshot_changed"
    assert not (root / "seed").exists()
    assert not (root / "archive.db").exists()


def test_market_snapshot_resolves_symlinked_root_and_refuses_inner_links(
    tmp_path: Path,
) -> None:
    real = _make_catalog(tmp_path / "real_catalog")
    link = tmp_path / "linked_catalog"
    link.symlink_to(real)
    archive = _make_archive(tmp_path / "raw_data.db", 2)

    snapshot = capture_market(link, archive, report_root(tmp_path / "scratch", 1), symbols=BTC)
    assert snapshot.source_catalog == real.resolve()
    assert _tree(snapshot.seed) == _tree(real)

    (real / "data" / "bar" / "alias").symlink_to(real / "data" / "crypto_perpetual")
    before = _tree(real)
    root = report_root(tmp_path / "scratch", 2)
    with pytest.raises(MarketSnapshotError) as err:
        capture_market(link, archive, root, symbols=BTC)
    assert err.value.code == "catalog_link_unsupported"
    assert not (root / "seed").exists()
    assert _tree(real) == before


def test_archive_backup_is_read_only_and_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 4)
    archive_sha = _sha(archive)

    # An uncommitted write on the source is not in the snapshot.
    writer = sqlite3.connect(archive)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate, source)"
        " VALUES ('BTCUSDT', 99, 0.5, 'test')"
    )

    opened: list[str] = []
    real_connect = sqlite3.connect

    def spy(database: str | Path, **kwargs: Any) -> sqlite3.Connection:
        opened.append(str(database))
        return real_connect(database, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    try:
        snapshot = capture_market(
            source, archive, report_root(tmp_path / "scratch", 3), symbols=BTC
        )
    finally:
        writer.rollback()
        writer.close()
    monkeypatch.setattr(sqlite3, "connect", real_connect)

    assert [r[2] for r in _rows(snapshot.archive)] == [0, 1, 2, 3]
    source_opens = [o for o in opened if "raw_data.db" in o]
    assert source_opens and all(
        o.startswith("file:") and o.endswith("?mode=ro") for o in source_opens
    )
    assert _sha(archive) == archive_sha

    # A missing archive fails the capture and leaves no seed behind.
    root = report_root(tmp_path / "scratch", 4)
    with pytest.raises(sqlite3.OperationalError):
        capture_market(source, tmp_path / "absent.db", root, symbols=BTC)
    assert not (root / "seed").exists()
    assert not (root / "archive.db").exists()


def test_archive_snapshot_copies_only_the_read_tables_of_the_run_symbols(
    tmp_path: Path,
) -> None:
    """Ruling 2026-11-19 (disk is limited): the snapshot holds the archive's whole schema
    but only the rows a validation run reads -- funding rates and funding intervals of the
    run's symbols, plus 1m klines of ``kline_symbols`` (taker indicators) -- every column
    (ids included) as in the source; never mark klines, other intervals, other symbols or
    download sessions."""
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 6, klines=3000)
    archive_sha = _sha(archive)

    btc = capture_market(source, archive, report_root(tmp_path / "s", 1), symbols=BTC)
    assert _schema(btc.archive) == _schema(archive)
    assert _rows(btc.archive, symbol=None) == _rows(archive, symbol="BTCUSDT")
    assert len(_rows(btc.archive, symbol=None)) == 6
    assert _rows(btc.archive, "raw_funding_intervals", None) == _rows(
        archive, "raw_funding_intervals", "BTCUSDT"
    )
    for table in ("raw_klines", "raw_mark_klines", "download_sessions"):
        assert _rows(btc.archive, table, None) == []
    assert btc.archive.stat().st_size * 10 < archive.stat().st_size

    two = capture_market(
        source,
        archive,
        report_root(tmp_path / "s", 2),
        symbols=["ETHUSDT", "SOLUSDT"],
        kline_symbols=["SOLUSDT"],
    )
    assert _rows(two.archive, symbol=None) == sorted(
        _rows(archive, symbol="ETHUSDT") + _rows(archive, symbol="SOLUSDT")
    )
    assert len(_rows(two.archive, "raw_funding_intervals", None)) == 12
    klines = _rows(two.archive, "raw_klines", None)
    assert klines == [r for r in _rows(archive, "raw_klines", "SOLUSDT") if r[2] == "1m"]
    assert len(klines) == 3000
    assert _rows(two.archive, "raw_mark_klines", None) == []

    # WAL already set: engines opening it through the shared connector switch nothing.
    conn = sqlite3.connect(f"file:{two.archive}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        conn.close()
    assert _sha(archive) == archive_sha


def test_archive_snapshot_reads_in_one_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source commit landing between two table copies is in neither: every table is
    read inside one read transaction (one consistent archive revision)."""
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 3)
    real_copy = snap._copy_rows
    copied: list[str] = []

    def racing_copy(src: sqlite3.Connection, dst: sqlite3.Connection, table: str,
                    where: str, params: tuple[str, ...]) -> None:  # fmt: skip
        real_copy(src, dst, table, where, params)
        if not copied:
            writer = sqlite3.connect(archive)
            for t, extra in (("raw_funding_rates", "funding_rate"),
                             ("raw_funding_intervals", "interval_hours")):  # fmt: skip
                writer.execute(
                    f"INSERT INTO {t} (symbol, funding_time, {extra}, source)"
                    " VALUES ('BTCUSDT', 500, 8, 'late')"
                )
            writer.commit()
            writer.close()
        copied.append(table)

    monkeypatch.setattr(snap, "_copy_rows", racing_copy)
    snapshot = capture_market(source, archive, report_root(tmp_path / "s", 1), symbols=BTC)
    assert copied == ["raw_funding_rates", "raw_funding_intervals"]
    for table in copied:
        assert [r[2] for r in _rows(snapshot.archive, table, None)] == [0, 1, 2]
        assert [r[2] for r in _rows(archive, table)] == [0, 1, 2, 500]


def test_clone_target_must_not_exist_and_fallback_copies_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 1)
    root = report_root(tmp_path / "scratch", 5)
    snapshot = capture_market(source, archive, root, symbols=BTC)
    seed_tree = _tree(snapshot.seed)

    existing = root / "w0" / "catalog"
    existing.mkdir(parents=True)
    (existing / "stale.parquet").write_bytes(b"stale")
    with pytest.raises(FileExistsError):
        clone_for_engine(snapshot, existing)
    assert _tree(existing) == {"stale.parquet": b"stale"}

    # The fast clone fails half way: the partial copy is removed and the fallback copies
    # the SEED (not the source, which has moved on).
    (source / "data" / "after-seed.parquet").write_bytes(b"moved on")

    def broken_clone(src: Path, dst: Path) -> None:
        dst.mkdir(parents=True)
        (dst / "partial.tmp").write_bytes(b"half")
        raise OSError("clonefile failed")

    monkeypatch.setattr(snap, "_clone_tree", broken_clone)
    target = clone_for_engine(snapshot, root / "w1" / "catalog")
    assert _tree(target) == seed_tree
    assert "partial.tmp" not in _tree(target)


_KILL_FIXTURE = """
import sys, time
from pathlib import Path
from vibe_quant.bankroll.snapshot import capture_market, clone_for_engine, report_root
base, catalog, archive = (Path(a) for a in sys.argv[1:4])
snapshot = capture_market(catalog, archive, report_root(base, 7), symbols=["BTCUSDT"])
clone_for_engine(snapshot, snapshot.root / "baseline" / "catalog")
print("ready", flush=True)
time.sleep(600)
"""


def test_reclaim_abandoned_after_sigkill(tmp_path: Path) -> None:
    source = _make_catalog(tmp_path / "catalog")
    archive = _make_archive(tmp_path / "raw_data.db", 2)
    base = tmp_path / "scratch"
    proc = subprocess.Popen(
        [sys.executable, "-c", _KILL_FIXTURE, str(base), str(source), str(archive)],
        stdout=subprocess.PIPE,
        text=True,
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL
    abandoned = report_root(base, 7)
    assert (abandoned / "baseline" / "catalog").is_dir()

    live = report_root(base, 9)
    live.mkdir(parents=True)
    foreign = base / "vq-bankroll" / "not-a-report"
    foreign.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"keep")
    (base / "vq-bankroll" / "12").symlink_to(outside)

    removed = reclaim_abandoned(base, live_report_ids={9})
    assert removed == [abandoned]
    assert not abandoned.exists()
    assert live.is_dir()
    assert foreign.is_dir()
    assert (outside / "keep.txt").read_bytes() == b"keep"
    assert _tree(source)  # the source catalog is untouched by the reclaim
    assert reclaim_abandoned(tmp_path / "no-scratch-yet", live_report_ids=set()) == []


def test_report_root_layout(tmp_path: Path) -> None:
    assert report_root(tmp_path, 3) == tmp_path / "vq-bankroll" / "3"
    assert report_root(tmp_path / "x", 15) == tmp_path / "x" / "vq-bankroll" / "15"


def test_snapshot_import_isolation() -> None:
    code = (
        "import sys\n"
        "import vibe_quant.bankroll.snapshot\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'nautilus_trader'"
        " or m.startswith(('vibe_quant.validation', 'vibe_quant.db', 'vibe_quant.data')))\n"
        "print(bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env={"PYTHONPATH": str(_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert out.stdout.strip() == "[]"
