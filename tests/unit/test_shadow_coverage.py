"""Shadow tracker data coverage and the catalog read snapshot.

Contract rows R20-R24 (coverage day C), R34-R35 (snapshot), plus the writers'
side: catalog writers serialise on ``catalog_write_lock`` and a failed funding
refresh is reported as ``funding_failed``.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.objects import Price, Quantity

from vibe_quant.data import catalog as catalog_mod
from vibe_quant.data import ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    CatalogManager,
    catalog_write_lock,
    create_instrument,
    get_bar_type,
)
from vibe_quant.data.downloader import RestDownloadError
from vibe_quant.shadow.coverage import coverage_day, remove_stale_snapshots, snapshot

if TYPE_CHECKING:
    from pathlib import Path

RUN_START = date(2026, 10, 20)
_MIN = 60_000
_PERIOD_MS = {"1m": _MIN, "4h": 4 * 60 * _MIN}


def _ms(month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(datetime(2026, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def _kline(open_ms: int) -> tuple[object, ...]:
    return (open_ms, 100.0, 101.0, 99.0, 100.5, 1.0, open_ms + _MIN - 1)


def _daily_rows(last_open: int, gaps: tuple[date, ...]) -> list[tuple[object, ...]]:
    """One 12:00 row per UTC day from RUN_START - 5 d, plus the tail row ``last_open``."""
    rows = []
    day = RUN_START - timedelta(days=5)
    while (noon := _ms(day.month, day.day, 12)) < last_open:
        if day not in gaps:
            rows.append(_kline(noon))
        day += timedelta(days=1)
    rows.append(_kline(last_open))
    return rows


def _bars(sym: str, tf: str, opens: list[int], price_type: PriceType = PriceType.LAST) -> list[Bar]:
    bt = get_bar_type(sym, tf, price_type)
    out = []
    for i, o in enumerate(opens):
        px = Price.from_str(f"{100 + i}.0")
        out.append(
            Bar(
                bar_type=bt,
                open=px,
                high=px,
                low=px,
                close=px,
                volume=Quantity.from_str("1.000"),
                ts_event=o * 1_000_000,
                ts_init=(o + _PERIOD_MS[tf] - 1) * 1_000_000,
            )
        )
    return out


def _build(
    tmp_path: Path,
    *,
    sym: str = "BTCUSDT",
    k1m_tail: int | None = None,
    mark_tail: int | None = None,
    funding_last: int | None = None,
    cat_1m_tail: int | None = None,
    cat_4h_tail: int | None = None,
    cat_mark_tail: int | None = None,
    gaps_1m: tuple[date, ...] = (),
    gaps_mark: tuple[date, ...] = (),
    drop_funding: tuple[int, ...] = (),
) -> tuple[Path, Path]:
    """Archive + catalog whose default tails are BTC's MEASURED ones (R20)."""
    k1m_tail = _ms(11, 10, 18) if k1m_tail is None else k1m_tail
    mark_tail = _ms(11, 9, 23, 59) if mark_tail is None else mark_tail
    funding_last = _ms(11, 10, 16) if funding_last is None else funding_last
    cat_path, arch_path = tmp_path / "catalog", tmp_path / "archive.db"
    arch = RawDataArchive(arch_path)
    try:
        arch.insert_klines(sym, "1m", _daily_rows(k1m_tail, gaps_1m), "t")
        arch.insert_mark_klines(sym, "1m", _daily_rows(mark_tail, gaps_mark), "t")
        rates = []
        t = _ms(10, 19)
        while t <= funding_last:
            if t not in drop_funding:
                rates.append((t + 3, 0.0001, 100.0))  # archived rows carry ms of jitter
            t += 8 * 3_600_000
        arch.insert_funding_rates(sym, rates, "t")
    finally:
        arch.close()
    cm = CatalogManager(cat_path)
    cm.write_instrument(create_instrument(sym))
    start = _ms(RUN_START.month, RUN_START.day)
    cm.write_bars(_bars(sym, "1m", [start, k1m_tail if cat_1m_tail is None else cat_1m_tail]))
    cm.write_bars(
        _bars(sym, "4h", [start, _ms(11, 10, 12) if cat_4h_tail is None else cat_4h_tail])
    )
    cm.write_bars(
        _bars(
            sym,
            "1m",
            [start, mark_tail if cat_mark_tail is None else cat_mark_tail],
            PriceType.MARK,
        )
    )
    return cat_path, arch_path


def _cov(
    cat: Path,
    arch: Path,
    *,
    tfs: tuple[str, ...] = ("4h", "1m"),
    mark: bool = True,
    syms: tuple[str, ...] = ("BTCUSDT",),
) -> tuple[date, str | None]:
    return coverage_day(
        list(syms), list(tfs), catalog=cat, archive=arch, mark=mark, run_start=RUN_START
    )


# ---------------------------------------------------------------- coverage day (R20-R24)


@pytest.mark.parametrize("mark", [True, False])
def test_tails(tmp_path: Path, mark: bool) -> None:
    """R20: archive tails 1m 11-10 18:00, MARK 11-09 23:59, funding 11-10 16:00 -> C = 11-09."""
    cat, arch = _build(tmp_path)
    assert _cov(cat, arch, mark=mark) == (date(2026, 11, 9), None)


def test_research_tier_coin_is_never_covered(tmp_path: Path) -> None:
    """A research-tier coin (1h/1d archive, no 1m) blocks the row instead of passing as full tier."""
    cat, arch = _build(tmp_path)
    a = RawDataArchive(arch)
    try:
        a.insert_klines("HYPEUSDT", "1h", [_kline(_ms(11, 10, h)) for h in range(5)], "t")
    finally:
        a.close()
    day, reason = _cov(cat, arch, syms=("BTCUSDT", "HYPEUSDT"))
    assert day == RUN_START - timedelta(days=1)
    assert reason == "no 1m klines archived for HYPEUSDT"


@pytest.mark.parametrize(
    ("k1m_tail", "expected"),
    [
        (_ms(11, 9, 23, 59), date(2026, 11, 9)),  # last minute of the day archived: full
        (_ms(11, 9, 23, 58), date(2026, 11, 8)),  # one minute short: not full
        (_ms(11, 9, 15), date(2026, 11, 8)),  # update's REST pages failed from 11-09 15:01
    ],
    ids=["tail-2359", "tail-2358", "failed-range"],
)
def test_archive_1m_tail_edges(tmp_path: Path, k1m_tail: int, expected: date) -> None:
    cat, arch = _build(tmp_path, k1m_tail=k1m_tail, cat_1m_tail=_ms(11, 10, 18))
    assert _cov(cat, arch, mark=False) == (expected, None)


@pytest.mark.parametrize(
    ("archive_tail", "catalog_tail", "mark", "expected"),
    [
        (_ms(11, 7, 23, 59), None, True, date(2026, 11, 7)),  # archive MARK 2 days short
        (_ms(11, 7, 23, 59), None, False, date(2026, 11, 9)),  # liquidation off: MARK ignored
        (_ms(11, 9, 23, 59), _ms(11, 7, 23, 59), True, date(2026, 11, 7)),  # catalog MARK short
        (_ms(11, 9, 23, 59), _ms(11, 9, 23, 58), True, date(2026, 11, 8)),  # catalog edge
    ],
    ids=["archive-on", "archive-off", "catalog-on", "catalog-2358"],
)
def test_mark_tail(
    tmp_path: Path, archive_tail: int, catalog_tail: int | None, mark: bool, expected: date
) -> None:
    """R21: a short MARK tail limits C only when liquidation is on."""
    cat, arch = _build(tmp_path, mark_tail=archive_tail, cat_mark_tail=catalog_tail)
    assert _cov(cat, arch, mark=mark) == (expected, None)


@pytest.mark.parametrize(
    ("gap", "mark_gap", "expected"),
    [
        (date(2026, 11, 8), None, (date(2026, 11, 7), "data gap at 2026-11-08 (BTCUSDT)")),
        (RUN_START, None, (RUN_START - timedelta(days=1), "data gap at 2026-10-20 (BTCUSDT)")),
        (RUN_START - timedelta(days=1), None, (date(2026, 11, 9), None)),  # before warm-up
        (None, date(2026, 11, 3), (date(2026, 11, 2), "mark data gap at 2026-11-03 (BTCUSDT)")),
    ],
    ids=["after-candidate", "at-run-start", "before-run-start", "mark"],
)
def test_gap_after_candidate(
    tmp_path: Path, gap: date | None, mark_gap: date | None, expected: tuple[date, str | None]
) -> None:
    """R22: a zero-row day G inside [run_start, tail] caps C at G - 1 and names G."""
    cat, arch = _build(
        tmp_path,
        gaps_1m=(gap,) if gap else (),
        gaps_mark=(mark_gap,) if mark_gap else (),
    )
    assert _cov(cat, arch) == expected


@pytest.mark.parametrize(
    ("cat_4h_tail", "tfs", "expected"),
    [
        (_ms(11, 8, 20), ("4h", "1m"), date(2026, 11, 8)),  # last 4h bar closes 11-09 00:00
        (_ms(11, 9, 16), ("4h", "1m"), date(2026, 11, 8)),  # closes 11-09 20:00: not full
        (_ms(11, 8, 20), ("1m",), date(2026, 11, 9)),  # strategy without 4h ignores it
    ],
    ids=["4h-limits", "4h-partial-day", "1m-only"],
)
def test_higher_tf_tail(
    tmp_path: Path, cat_4h_tail: int, tfs: tuple[str, ...], expected: date
) -> None:
    """R23: the catalog 4h file ending a day before 1m limits C when the strategy uses 4h."""
    cat, arch = _build(tmp_path, cat_4h_tail=cat_4h_tail)
    assert _cov(cat, arch, tfs=tfs) == (expected, None)


def test_gap_reason_wins_tie_with_silent_tail(tmp_path: Path) -> None:
    """A zero-row day and a silent tail on the same day: the gap's reason is reported."""
    # 1m gap 11-10 -> 11-09 (reason); funding tail 11-10 16:00 -> 11-09 (silent)
    cat, arch = _build(tmp_path, k1m_tail=_ms(11, 11, 6), gaps_1m=(date(2026, 11, 10),))
    assert _cov(cat, arch, tfs=("1m",), mark=False) == (
        date(2026, 11, 9),
        "data gap at 2026-11-10 (BTCUSDT)",
    )


def test_missing_archive_is_not_created(tmp_path: Path) -> None:
    cat, _ = _build(tmp_path)
    wrong = tmp_path / "no_such_archive.db"
    with pytest.raises(FileNotFoundError, match="no_such_archive"):
        _cov(cat, wrong)
    assert not wrong.exists()


def test_missing_timeframe_blocks(tmp_path: Path) -> None:
    cat, arch = _build(tmp_path)
    assert _cov(cat, arch, tfs=("1h",)) == (
        RUN_START - timedelta(days=1),
        "no catalog 1h LAST bars for BTCUSDT",
    )


def test_late_funding(tmp_path: Path) -> None:
    """R24: the C+1 00:00 settlement missing holds C one day back; its arrival restores C.

    No cache clear between the reads: a late settlement must count in the same process.
    """
    midnight = _ms(11, 10)
    cat, arch = _build(tmp_path, drop_funding=(midnight,))
    assert _cov(cat, arch) == (
        date(2026, 11, 8),
        "funding settlement 2026-11-10T00:00:00+00:00 missing (BTCUSDT)",
    )
    a = RawDataArchive(arch)
    try:
        a.insert_funding_rates("BTCUSDT", [(midnight + 3, 0.0001, 100.0)], "late")
    finally:
        a.close()
    assert _cov(cat, arch) == (date(2026, 11, 9), None)


@pytest.mark.parametrize(
    ("funding_last", "expected"),
    [
        (_ms(11, 9, 16), date(2026, 11, 8)),  # 11-10 00:00 not yet published
        (_ms(11, 10), date(2026, 11, 9)),  # published (archived at 00:00:00.003)
    ],
    ids=["tail-before-midnight", "tail-at-midnight"],
)
def test_funding_tail(tmp_path: Path, funding_last: int, expected: date) -> None:
    cat, arch = _build(tmp_path, funding_last=funding_last)
    assert _cov(cat, arch) == (expected, None)


# ---------------------------------------------------------------- snapshot (R34, R35)


def _closes(cat: Path) -> list[tuple[int, float]]:
    return [(b.ts_event, float(b.close)) for b in CatalogManager(cat).get_bars("BTCUSDT", "1m")]


def test_snapshot_survives_rebuild(tmp_path: Path) -> None:
    """R34: a rebuild while a snapshot is read leaves the snapshot's bars and the shared instrument."""
    cat, _ = _build(tmp_path)
    before = _closes(cat)
    inst_dir = cat / "data" / "crypto_perpetual" / "BTCUSDT-PERP.BINANCE"
    inst_mtime = inst_dir.stat().st_mtime_ns
    with snapshot(cat, ["BTCUSDT"]) as snap:
        assert snap.parent == cat.resolve().parent
        assert snap.name.startswith(".shadow-snap-")
        # the data update rebuilds the shared LAST 1m series meanwhile
        shared = CatalogManager(cat)
        shared.clear_bar_data("BTCUSDT", "1m")
        shared.write_bars(_bars("BTCUSDT", "1m", [_ms(11, 1), _ms(11, 2), _ms(11, 3)]))
        # the run's runner rewrites the instrument inside its snapshot
        changed = create_instrument("BTCUSDT")
        changed = type(changed).from_dict({**type(changed).to_dict(changed), "taker_fee": "0.0009"})
        CatalogManager(snap).write_instrument(changed)
        assert _closes(snap) == before
        assert len(CatalogManager(snap).get_bars("BTCUSDT", "4h")) == 2
        mark_dir = snap / "data" / "bar" / str(get_bar_type("BTCUSDT", "1m", PriceType.MARK))
        assert list(mark_dir.glob("*.parquet"))
    assert not snap.exists()
    assert inst_dir.stat().st_mtime_ns == inst_mtime
    assert (
        CatalogManager(cat).get_instruments()[0].taker_fee == create_instrument("BTCUSDT").taker_fee
    )
    assert [c for _, c in _closes(cat)] == [100.0, 101.0, 102.0]


def test_snapshot_waits_for_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A snapshot taken while a writer holds the catalog lock starts only after it releases."""
    monkeypatch.setattr(catalog_mod, "_LOCK_POLL_S", 0.02)
    cat, _ = _build(tmp_path)
    taken: list[list[tuple[int, float]]] = []

    def take() -> None:
        with snapshot(cat, ["BTCUSDT"]) as snap:
            taken.append(_closes(snap))

    with catalog_write_lock(cat):
        worker = threading.Thread(target=take)
        worker.start()
        time.sleep(0.3)
        assert worker.is_alive()
        shared = CatalogManager(cat)
        shared.clear_bar_data("BTCUSDT", "1m")
        shared.write_bars(_bars("BTCUSDT", "1m", [_ms(11, 1)]))
    worker.join(timeout=30)
    assert taken == [[(_ms(11, 1) * 1_000_000, 100.0)]]


def test_catalog_lock_times_out(tmp_path: Path) -> None:
    cat = tmp_path / "catalog"
    with (
        catalog_write_lock(cat),
        pytest.raises(TimeoutError, match="catalog busy"),
        catalog_write_lock(cat, timeout_s=0),
    ):
        pass
    with catalog_write_lock(cat, timeout_s=0):  # released on exit
        pass


@pytest.mark.simple_target(
    reason="contract row R35: deletes directories beside the real catalog; prefix-only is data safety"
)
def test_snapshot_cleanup_prefix_only(tmp_path: Path) -> None:
    """R35: stale .shadow-snap-* dirs beside the catalog go; anything else stays."""
    cat = tmp_path / "data" / "catalog"
    cat.mkdir(parents=True)
    stale = tmp_path / "data" / ".shadow-snap-x1"
    (stale / "data" / "bar").mkdir(parents=True)
    (stale / "data" / "bar" / "f.parquet").write_bytes(b"x")
    keep = [tmp_path / "data" / n for n in ("sentinel", ".shadow-snapshot-notes", "shadow-snap-y")]
    for k in keep:
        k.mkdir()
    keep_file = tmp_path / "data" / ".shadow-snap-file.txt"
    keep_file.write_text("x")
    assert remove_stale_snapshots(cat) == 1
    assert not stale.exists()
    assert all(k.is_dir() for k in keep)
    assert keep_file.exists()
    assert cat.is_dir()


def test_snapshot_cleanup_skips_live_snapshot(tmp_path: Path) -> None:
    """A second tracker's cleanup (other state DB, same catalog) never removes a snapshot in use."""
    cat, _ = _build(tmp_path)
    with snapshot(cat, ["BTCUSDT"]) as snap:
        assert remove_stale_snapshots(cat) == 0
        assert _closes(snap) == _closes(cat)
    # a crashed owner (lock released by the kernel) leaves a removable directory
    dead = tmp_path / ".shadow-snap-dead"
    dead.mkdir()
    (dead / ".owner.lock").write_bytes(b"")
    assert remove_stale_snapshots(cat) == 1
    assert not dead.exists()


# ---------------------------------------------------------------- writers take the lock


@pytest.mark.simple_target(
    reason="data safety: a worktree rebuild through data/catalog would empty the shared catalog"
)
def test_rebuild_refuses_symlinked_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worktree's data/catalog symlink: rebuild refuses and the shared target keeps its series."""
    monkeypatch.setattr(ingest, "RawDataArchive", lambda *_a: RawDataArchive(tmp_path / "a.db"))
    real = tmp_path / "main_catalog"
    series = real / "data" / "bar" / "OTHER-AGENT-SERIES"
    series.mkdir(parents=True)
    (series / "x.parquet").write_bytes(b"x")
    link = tmp_path / "wt_catalog"
    link.symlink_to(real)
    with pytest.raises(OSError, match="refusing to rebuild through symlink"):
        ingest.rebuild_from_archive(catalog_path=link, verbose=False)
    assert (series / "x.parquet").read_bytes() == b"x"


def test_rebuild_waits_for_lock_and_keeps_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild never deletes the catalog under a snapshot, nor the lock file itself."""
    monkeypatch.setattr(catalog_mod, "_LOCK_POLL_S", 0.02)
    arch_path = tmp_path / "archive.db"
    a = RawDataArchive(arch_path)
    try:
        a.insert_klines("BTCUSDT", "1m", [_kline(_ms(11, 1, 0, m)) for m in range(3)], "t")
    finally:
        a.close()
    monkeypatch.setattr(ingest, "RawDataArchive", lambda *_a: RawDataArchive(arch_path))
    cat = tmp_path / "catalog"
    marker = cat / "data" / "old_marker"
    marker.mkdir(parents=True)
    errors: list[BaseException] = []

    def rebuild() -> None:
        try:
            ingest.rebuild_from_archive(catalog_path=cat, verbose=False)
        except BaseException as e:  # surfaced by the assertion below
            errors.append(e)

    with catalog_write_lock(cat):
        inode = os.stat(cat / ".write.lock").st_ino
        worker = threading.Thread(target=rebuild)
        worker.start()
        time.sleep(0.3)
        assert worker.is_alive()
        assert marker.is_dir()
    worker.join(timeout=60)
    assert errors == []
    assert not marker.exists()
    assert CatalogManager(cat).get_bar_count("BTCUSDT", "1m") == 3
    assert os.stat(cat / ".write.lock").st_ino == inode


def test_update_all_writes_under_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every symbol's update runs under the catalog lock; the lock is free once update_all returns."""
    cat = tmp_path / "catalog"
    monkeypatch.setattr(ingest, "RawDataArchive", lambda *_a: RawDataArchive(tmp_path / "a.db"))
    monkeypatch.setattr(ingest, "CatalogManager", lambda *_a: CatalogManager(cat))
    held: dict[str, bool] = {}

    def fake_update(symbol: str, **_k: object) -> dict[str, object]:
        try:
            with catalog_write_lock(cat, timeout_s=0):
                held[symbol] = False
        except TimeoutError:
            held[symbol] = True
        return {"new_klines": 0, "failed_ranges": []}

    monkeypatch.setattr(ingest, "update_symbol", fake_update)
    ingest.update_all(["BTCUSDT", "ETHUSDT"], verbose=False)
    assert held == {"BTCUSDT": True, "ETHUSDT": True}
    with catalog_write_lock(cat, timeout_s=0):
        pass


@pytest.mark.parametrize("failure", ["http-status", "connect", "none"])
def test_update_flags_failed_funding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A funding refresh that fails (incl. a 429/5xx page) is reported, never passed off as complete."""
    now_ms = int(time.time() * 1000)
    archive = RawDataArchive(tmp_path / "f.db")
    try:
        archive.insert_klines("BTCUSDT", "1m", [_kline(now_ms - 30_000)], "seed")
        archive.insert_funding_rates("BTCUSDT", [(now_ms - 3 * 8 * 3_600_000, 0.0001, 1.0)], "seed")

        partial = (now_ms - 8 * 3_600_000, 0.0002, 1.0)

        def fake_funding(
            symbol: str, start: int, end: int, timeout: float = 30.0, *, strict: bool = False
        ) -> list[tuple[object, ...]]:
            if failure == "connect":
                raise httpx.ConnectError("down")
            if failure == "http-status":
                # the real downloader: non-strict logs a 503 and returns the pages so far
                if strict:  # the page before the failed one is valid and kept
                    raise RestDownloadError("503 after retries", [partial], start)
                return []
            return [partial]

        monkeypatch.setattr(ingest, "download_funding_rates", fake_funding)
        monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "ingest_funding_intervals", lambda *a, **k: 0)
        counts = ingest.update_symbol("BTCUSDT", archive=archive, catalog=None, verbose=False)
        archived = archive.conn.execute("SELECT COUNT(*) FROM raw_funding_rates").fetchone()[0]
    finally:
        archive.close()
    assert counts.get("funding_failed", False) is (failure != "none")
    assert counts["new_funding_rates"] == (1 if failure == "none" else 0)
    assert archived == (1 if failure == "connect" else 2)


def test_snapshot_cleanup_age_floor(tmp_path: Path) -> None:
    """A lockless snapshot younger than the floor may be a peer tracker's between mkdtemp and
    its owner lock (other state DB, same catalog): kept. An older one is an orphan: removed."""
    cat = tmp_path / "catalog"
    cat.mkdir()
    old, young = tmp_path / ".shadow-snap-old", tmp_path / ".shadow-snap-young"
    now = time.time()
    for path, age in ((old, 1801.0), (young, 1799.0)):
        path.mkdir()
        os.utime(path, (now - age, now - age))
    assert remove_stale_snapshots(cat, min_age_s=1800.0) == 1
    assert not old.exists()
    assert young.is_dir()
