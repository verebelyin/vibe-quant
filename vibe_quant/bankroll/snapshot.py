"""The report-local market snapshot every engine of one bankroll report reads.

One report captures the market ONCE, and every engine (baseline and stress cells, serial or
parallel) then runs on the same revision (design 2gyk1.3, C17):

- ``seed/``: a clone of the source catalog taken against a stable source generation. The
  source tree's manifest (relpath, size, mtime_ns) must be equal before and after the
  clone; on a change the seed is removed and the capture retried once, then it fails
  ``market_snapshot_changed``. The source root is resolved once (a worktree's symlinked
  ``data/catalog`` is followed); any symlink INSIDE the tree fails
  ``catalog_link_unsupported``.
- ``archive.db``: a small report-local copy of the raw-data archive (ruling 2026-11-19: disk
  is limited, so a report never backs up the whole multi-GB archive). It holds the
  archive's whole schema but only the rows a validation run reads: funding rates and
  funding intervals of the run's symbols, plus 1m klines of the symbols whose taker
  indicators need them. Every table is read inside ONE read transaction of a read-only
  connection (``file:...?mode=ro``, never the shared connector), so the copy is one
  consistent archive revision.
- :func:`clone_for_engine` clones the seed (never the source) into a NON-existing private
  directory per engine, so a catalog write made by one engine reaches no other engine,
  the seed or the source. A failed fast clone is removed before the ``shutil.copytree``
  fallback, which copies the same seed.

Everything lives under the report-owned root ``<db scratch>/vq-bankroll/<report_id>/``
(:func:`report_root`). Report ids are per state DB, so the scratch base is namespaced by
the state DB first (:func:`db_scratch`): a ui-check/E2E stack on a DB copy sharing the
tmp never reclaims or removes the main backend's roots. :func:`reclaim_abandoned` removes
the roots of one DB's reports that are no longer running (after a SIGTERM/SIGKILL of
their coordinator).

Imports no engine, validation, data or DB module (import isolation).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SCRATCH_DIRNAME = "vq-bankroll"
_ATTEMPTS = 2

_CLONE_CMD: tuple[str, ...] = (
    ("cp", "-c", "-R") if sys.platform == "darwin" else ("cp", "-R", "--reflink=auto")
)
"""Copy-on-write tree clone (APFS clonefile / reflink); falls back to a plain copy."""

FUNDING_TABLES: tuple[str, ...] = ("raw_funding_rates", "raw_funding_intervals")
"""Archive tables a validation run reads for its symbols (funding accrual and intervals)."""
KLINE_TABLE = "raw_klines"
KLINE_INTERVAL = "1m"
"""Taker-flow indicators read the run symbols' 1m klines (``dsl/aux_data.py``)."""

Manifest = dict[str, tuple[int, int]]
"""relpath -> (size, mtime_ns); a directory has size -1."""


class MarketSnapshotError(RuntimeError):
    """The market could not be captured; ``code`` names why (stored as the report error)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class MarketSnapshot:
    """One report's captured market: the seed catalog and the archive backup."""

    root: Path
    source_catalog: Path
    seed: Path
    archive: Path


def db_scratch(scratch_base: Path, db_path: Path) -> Path:
    """One state DB's scratch base ``<scratch_base>/vq-bankroll-<sha256(resolved db)[:12]>``."""
    key = hashlib.sha256(str(Path(db_path).resolve()).encode()).hexdigest()[:12]
    return Path(scratch_base) / f"{SCRATCH_DIRNAME}-{key}"


def report_root(scratch_base: Path, report_id: int) -> Path:
    """The report-owned scratch root ``<scratch_base>/vq-bankroll/<report_id>``."""
    return Path(scratch_base) / SCRATCH_DIRNAME / str(report_id)


def _manifest(root: Path) -> Manifest:
    """Every entry under ``root`` (not following links); a symlink anywhere fails."""
    out: Manifest = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            path = Path(dirpath) / name
            st = path.lstat()
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise MarketSnapshotError("catalog_link_unsupported", f"symlink in catalog: {rel}")
            out[rel] = (-1, 0) if path.is_dir() else (st.st_size, st.st_mtime_ns)
    return out


def _shape(manifest: Manifest) -> dict[str, int]:
    """Path set and sizes (a copy keeps both, not necessarily the mtimes)."""
    return {rel: size for rel, (size, _mtime) in manifest.items()}


def _clone_tree(src: Path, dst: Path) -> None:
    """Copy-on-write clone of ``src`` as the new directory ``dst``."""
    subprocess.run([*_CLONE_CMD, str(src), str(dst)], check=True, capture_output=True)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _copy_tree(src: Path, dst: Path) -> None:
    """Clone ``src`` to ``dst``; on a failed clone remove the partial copy, then copy."""
    try:
        _clone_tree(src, dst)
    except (OSError, subprocess.CalledProcessError):
        _remove(dst)
        shutil.copytree(src, dst, symlinks=True)


def _copy_rows(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    where: str,
    params: tuple[str, ...],
) -> None:
    """Copy ``table``'s rows matching ``where`` (every column, ids included) in id order."""
    cursor = source.execute(f'SELECT * FROM "{table}" WHERE {where} ORDER BY rowid', params)
    marks = ", ".join("?" * len(cursor.description))
    target.executemany(f'INSERT INTO "{table}" VALUES ({marks})', cursor)


def _subset_archive(
    src: Path, dst: Path, symbols: Sequence[str], kline_symbols: Sequence[str]
) -> None:
    """Copy the archive schema plus the rows a run reads, in one read transaction."""
    source = sqlite3.connect(f"{src.as_uri()}?mode=ro", uri=True, isolation_level=None)
    try:
        target = sqlite3.connect(dst)
        try:
            target.execute("PRAGMA journal_mode=WAL")
            source.execute("BEGIN")
            schema = source.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
                "ORDER BY type = 'table' DESC, rowid"
            ).fetchall()
            for _type, _name, sql in schema:
                target.execute(sql)
            tables = {name for kind, name, _sql in schema if kind == "table"}
            syms = tuple(dict.fromkeys(symbols))
            marks = ", ".join("?" * len(syms))
            for table in FUNDING_TABLES:
                if table in tables and syms:
                    _copy_rows(source, target, table, f"symbol IN ({marks})", syms)
            kline_syms = tuple(dict.fromkeys(kline_symbols))
            if KLINE_TABLE in tables and kline_syms:
                where = f"interval = ? AND symbol IN ({', '.join('?' * len(kline_syms))})"
                _copy_rows(source, target, KLINE_TABLE, where, (KLINE_INTERVAL, *kline_syms))
            source.execute("COMMIT")
            target.commit()
        finally:
            target.close()
    finally:
        source.close()


def _sidecars(db: Path) -> tuple[Path, Path]:
    return db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")


def _capture_seed(source: Path, seed: Path) -> None:
    for _attempt in range(_ATTEMPTS):
        try:
            before = _manifest(source)
            _copy_tree(source, seed)
            after = _manifest(source)
        except (FileNotFoundError, shutil.Error):
            # A catalog writer deleted a file mid-capture: a source change, so retry.
            _remove(seed)
            continue
        if before == after:
            if _shape(_manifest(seed)) != _shape(before):
                _remove(seed)
                raise MarketSnapshotError("market_snapshot_changed", "seed differs from source")
            return
        _remove(seed)
    raise MarketSnapshotError(
        "market_snapshot_changed", f"source catalog changed during {_ATTEMPTS} captures"
    )


def capture_market(
    catalog_src: Path,
    archive_src: Path,
    root: Path,
    *,
    symbols: Sequence[str],
    kline_symbols: Sequence[str] = (),
) -> MarketSnapshot:
    """Capture ``root/seed`` (stable catalog clone) and ``root/archive.db`` (the rows of
    ``symbols`` a run reads, plus ``kline_symbols``' 1m klines; one read transaction).

    On any failure nothing captured is left behind under ``root``.
    """
    source = Path(catalog_src).resolve()
    if not source.is_dir():
        raise MarketSnapshotError("catalog_missing", f"no catalog directory at {source}")
    root = Path(root)
    seed, archive = root / "seed", root / "archive.db"
    if seed.exists() or archive.exists():
        raise FileExistsError(f"market snapshot already captured under {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        _capture_seed(source, seed)
        _subset_archive(Path(archive_src).resolve(), archive, symbols, kline_symbols)
    except BaseException:
        _remove(seed)
        for path in (archive, *_sidecars(archive)):
            _remove(path)
        raise
    return MarketSnapshot(root=root, source_catalog=source, seed=seed, archive=archive)


def clone_for_engine(snapshot: MarketSnapshot, dst: Path) -> Path:
    """Clone the seed into the NON-existing directory ``dst`` (its parents are created)."""
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"engine catalog target already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(snapshot.seed, dst)
    return dst


def reclaim_abandoned(scratch_base: Path, live_report_ids: Iterable[int]) -> list[Path]:
    """Remove every report root under ``scratch_base`` whose id is not live.

    Only real directories named by a report id are touched (never a symlink or a foreign
    name). Returns the removed roots, sorted by id.
    """
    base = Path(scratch_base) / SCRATCH_DIRNAME
    if not base.is_dir():
        return []
    live = set(live_report_ids)
    removed: list[Path] = []
    for child in sorted(base.iterdir(), key=lambda p: p.name):
        if child.is_symlink() or not child.is_dir() or not child.name.isdecimal():
            continue
        if int(child.name) in live:
            continue
        shutil.rmtree(child)
        removed.append(child)
    return sorted(removed, key=lambda p: int(p.name))
