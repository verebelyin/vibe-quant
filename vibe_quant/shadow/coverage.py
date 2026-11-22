"""Data coverage day and the private catalog snapshot for a shadow row's run.

``coverage_day`` answers "through which UTC day is every input of this row
complete?": the engine may only be run to ``C + 1`` 00:00 when the catalog bars
of every timeframe used, the archived 1m (and MARK when liquidation is on) rows
and the funding settlements through ``C + 1`` 00:00 are all there. A tail that is
simply not published yet holds C back silently; a hole inside ``[run_start, tail]``
(a zero-row day, a missing settlement) holds it back with a reason. No exchange
gap is treated as intentional: a zero-row day blocks until data arrives.

``snapshot`` gives one run a stable read view: under ``catalog_write_lock`` it
hard-links the row's bar files and copies its instrument definition into a
sibling ``.shadow-snap-*`` directory. A rebuild (``clear_bar_data`` = rmtree)
unlinks the shared names only, so the linked inodes stay intact, and a runner
rewriting the instrument touches the copy only.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from nautilus_trader.model.enums import PriceType

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import catalog_write_lock, get_bar_type
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

SNAPSHOT_PREFIX = ".shadow-snap-"
OWNER_LOCK_NAME = ".owner.lock"
_DAY_NS = 86_400 * 10**9
_MIN_MS = 60_000
_EPOCH = date(1970, 1, 1)

Limit = tuple[date, str | None]


def _day_of(ns: int) -> date:
    return _EPOCH + timedelta(days=ns // _DAY_NS)


def _midnight_ns(day: date) -> int:
    return (day - _EPOCH).days * _DAY_NS


def _last_full_bar_day(
    catalog: Path, symbol: str, timeframe: str, price_type: PriceType
) -> date | None:
    """Last UTC day whose bars of this series are complete: last bar close >= next midnight."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    bar_dir = catalog / "data" / "bar" / str(get_bar_type(symbol, timeframe, price_type))
    ends = [
        pc.max(pq.read_table(f, columns=["ts_init"])["ts_init"]).as_py()
        for f in sorted(bar_dir.glob("*.parquet"))
    ]
    ends = [e for e in ends if e is not None]
    if not ends:
        return None
    # ts_init is the close minus 1 ms (``klines_to_bars``)
    return _day_of(int(max(ends)) + 1_000_000) - timedelta(days=1)


def _archive_limits(
    archive: RawDataArchive, symbol: str, *, mark: bool, run_start: date, blocked: date
) -> list[Limit]:
    """1m (or MARK) archive tail day and the first zero-row day inside the warm-up+forward span."""
    kind = "mark" if mark else "1m"
    span = (
        archive.get_mark_date_range(symbol, "1m") if mark else archive.get_date_range(symbol, "1m")
    )
    if span is None:
        return [(blocked, f"no {kind} klines archived for {symbol}")]
    # the tail day is full when its 23:59 minute is archived
    limits: list[Limit] = [(_day_of((span[1] + _MIN_MS) * 1_000_000) - timedelta(days=1), None)]
    table = "raw_mark_klines" if mark else "raw_klines"
    gaps = archive.missing_days(symbol, "1m", start_day=run_start, table=table)
    if gaps:
        label = "mark data gap" if mark else "data gap"
        limits.append((gaps[0] - timedelta(days=1), f"{label} at {gaps[0].isoformat()} ({symbol})"))
    return limits


def _funding_limit(
    funding: FundingCalculator, symbol: str, run_start: date, blocked: date
) -> Limit:
    """Last day d with every settlement in (run_start, d+1 00:00] archived, d+1 00:00 included."""
    last = funding.last_settlement_ns(symbol)
    if last is None:
        return (blocked, f"no funding archived for {symbol}")
    day = _day_of(last) - timedelta(days=1)
    missing = funding.uncovered_settlements(
        symbol, _midnight_ns(run_start), _midnight_ns(day + timedelta(days=1))
    )
    if not missing:
        return (day, None)
    ts = missing[0]
    when = datetime.fromtimestamp(ts // 10**9, tz=UTC).isoformat()
    # the last midnight strictly before the hole is the furthest supported C + 1
    return (_day_of(ts - 1) - timedelta(days=1), f"funding settlement {when} missing ({symbol})")


def coverage_day(
    sym_list: Iterable[str],
    timeframes: Iterable[str],
    *,
    catalog: Path,
    archive: Path,
    mark: bool,
    run_start: date,
) -> tuple[date, str | None]:
    """The row's coverage day C and, when a hole (not a tail) sets it, the reason.

    C is the minimum over symbols of: the catalog LAST last full day of every
    ``timeframes`` entry (strategy timeframes + detail), the archived 1m tail day
    and first zero-row day - 1 in ``[run_start, ...]``, the same two plus the
    catalog MARK 1m day when ``mark``, and the funding day (see
    :func:`_funding_limit`). A symbol with no 1m archive (research tier), no
    catalog series or no funding gives ``run_start - 1`` with a reason. On equal
    days a reason wins over a silent tail.

    The process funding cache is dropped first, so a settlement archived since the
    last read counts (late settlement).
    """
    if not archive.exists():  # RawDataArchive would create an empty one
        raise FileNotFoundError(f"archive {archive} does not exist")
    clear_rate_cache()
    blocked = run_start - timedelta(days=1)
    funding = FundingCalculator(archive)
    arch = RawDataArchive(archive)
    limits: list[Limit] = []
    try:
        for sym in sym_list:
            # archive first: on equal days its reason (e.g. a research-tier coin) is reported
            limits += _archive_limits(arch, sym, mark=False, run_start=run_start, blocked=blocked)
            if mark:
                limits += _archive_limits(
                    arch, sym, mark=True, run_start=run_start, blocked=blocked
                )
            limits.append(_funding_limit(funding, sym, run_start, blocked))
            series = [(tf, PriceType.LAST) for tf in timeframes]
            if mark:
                series.append(("1m", PriceType.MARK))
            for tf, price_type in series:
                day = _last_full_bar_day(catalog, sym, tf, price_type)
                limits.append(
                    (day, None)
                    if day is not None
                    else (blocked, f"no catalog {tf} {price_type.name} bars for {sym}")
                )
    finally:
        arch.close()
    return min(limits, key=lambda lim: (lim[0], lim[1] is None))


@contextmanager
def snapshot(catalog: Path, symbols: Iterable[str]) -> Iterator[Path]:
    """A private catalog holding ``symbols``' bars (all series) and instruments.

    Created beside the catalog (same filesystem, so bar files are hard-linked) under
    ``catalog_write_lock``; the lock is released before the caller reads it, and the
    directory is removed on exit. Its ``.owner.lock`` stays flocked meanwhile, so
    :func:`remove_stale_snapshots` skips it.
    """
    catalog = catalog.resolve()
    snap = Path(tempfile.mkdtemp(dir=catalog.parent, prefix=SNAPSHOT_PREFIX))
    owner: int | None = None
    try:
        # a cleanup that removed the fresh, still unlocked directory makes this raise
        owner = os.open(snap / OWNER_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with catalog_write_lock(catalog):
            for sym in symbols:
                for src in sorted((catalog / "data" / "bar").glob(f"{sym}-PERP.BINANCE-*")):
                    for f in src.rglob("*"):
                        if f.is_file():
                            dst = snap / "data" / "bar" / src.name / f.relative_to(src)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            dst.hardlink_to(f)
                for src in (catalog / "data" / "crypto_perpetual").glob(f"{sym}-PERP.*"):
                    shutil.copytree(src, snap / "data" / "crypto_perpetual" / src.name)
        yield snap
    finally:
        shutil.rmtree(snap, ignore_errors=True)
        if owner is not None:
            os.close(owner)


def remove_stale_snapshots(catalog: Path) -> int:
    """Remove orphaned ``.shadow-snap-*`` directories beside the catalog.

    A directory whose ``.owner.lock`` is flocked belongs to a running snapshot (e.g.
    another tracker on another state DB sharing this catalog) and is kept; the kernel
    releases a dead owner's lock.

    Returns:
        How many were removed; files and other directories are never touched.
    """
    removed = 0
    for path in catalog.resolve().parent.glob(f"{SNAPSHOT_PREFIX}*"):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            fd = os.open(path / OWNER_LOCK_NAME, os.O_RDWR)
        except FileNotFoundError:
            pass  # owner died before creating it (or is about to: its open then raises)
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # in use
            finally:
                os.close(fd)
        shutil.rmtree(path)
        removed += 1
    return removed
