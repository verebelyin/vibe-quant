"""Lifecycle of the data-layer catalog and archive handles.

- ``CatalogManager.catalog`` builds the ParquetDataCatalog once even when several
  threads hit it on first use (the data router's GET handlers run in the
  threadpool and can reach ``get_bar_count`` concurrently).
- ``get_download_preview`` closes the RawDataArchive it opened itself, also when
  the loop raises, and leaves a caller-supplied archive open.

regression: vibe-quant-llatj
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from vibe_quant.data import catalog as catalog_module
from vibe_quant.data import ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import CatalogManager, create_instrument

if TYPE_CHECKING:
    from pathlib import Path


class _ResetRecordingCatalogManager(CatalogManager):
    """Records, for every ``_catalog`` reset to None, whether _catalog_lock was held."""

    def __init__(self, catalog_path: Path) -> None:
        self.resets_locked: list[bool] = []
        super().__init__(catalog_path)

    def __setattr__(self, name: str, value: object) -> None:
        # __init__ assigns _catalog before the lock exists; skip that first assignment.
        if name == "_catalog" and value is None and "_catalog_lock" in self.__dict__:
            self.resets_locked.append(self._catalog_lock.locked())
        super().__setattr__(name, value)


class TestCatalogResetUnderLock:
    """Cache resets race the lazy getter unless they take _catalog_lock.

    regression: vibe-quant-gq9i6
    """

    def test_clear_bar_data_resets_cache_under_lock(self, tmp_path: Path) -> None:
        manager = _ResetRecordingCatalogManager(tmp_path / "catalog")

        manager.clear_bar_data("BTCUSDT", "1m")

        assert manager.resets_locked == [True]

    def test_write_instrument_reset_under_lock(self, tmp_path: Path) -> None:
        """A changed definition rmtrees the instrument dir and resets the cache."""
        new = create_instrument("SOLUSDT")
        old = type(new).from_dict(
            {
                **type(new).to_dict(new),
                "size_increment": "1",
                "size_precision": 0,
                "min_quantity": None,
            }
        )
        manager = _ResetRecordingCatalogManager(tmp_path / "catalog")
        manager.write_instrument(old)
        manager.write_instrument(new)

        assert manager.resets_locked == [True]


class _ResetAfterFirstRead(CatalogManager):
    """Built catalog on the first read of ``_catalog``; None on every later read.

    Forces the interleaving where a locked reset lands between the getter's
    check and its return.

    regression: vibe-quant-gq9i6
    """

    def __init__(self, catalog_path: Path) -> None:
        self._reads = 0
        self._built = object()
        super().__init__(catalog_path)

    @property  # type: ignore[misc]
    def _catalog(self) -> object:
        self._reads += 1
        return self._built if self._reads == 1 else None

    @_catalog.setter
    def _catalog(self, value: object) -> None:
        pass  # the forced sequence ignores writes; the getter never builds here


class TestCatalogGetterReturnsLocalRef:
    """The catalog getter returns the reference it checked, not a re-read of the cache."""

    def test_reset_between_check_and_return_does_not_yield_none(self, tmp_path: Path) -> None:
        manager = _ResetAfterFirstRead(tmp_path / "catalog")

        got = manager.catalog

        assert got is manager._built


class TestCatalogLazyInit:
    """CatalogManager.catalog under concurrent first use."""

    def test_concurrent_first_use_constructs_catalog_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """8 threads racing on first access share one ParquetDataCatalog."""
        built: list[object] = []

        class _SlowFakeCatalog:
            def __init__(self, path: str) -> None:
                # Widen the check-then-set window so an unlocked property races.
                time.sleep(0.05)
                built.append(self)

        monkeypatch.setattr(catalog_module, "ParquetDataCatalog", _SlowFakeCatalog)
        manager = CatalogManager(tmp_path / "catalog")

        n_threads = 8
        start_gate = threading.Barrier(n_threads)
        seen: list[object | None] = [None] * n_threads

        def _worker(idx: int) -> None:
            start_gate.wait()
            seen[idx] = manager.catalog

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(built) == 1, f"ParquetDataCatalog constructed {len(built)} times"
        assert all(s is built[0] for s in seen), "threads got different catalog objects"


class _RecordingArchive(RawDataArchive):
    """RawDataArchive that records close() and can fail on the first month lookup."""

    closes: int = 0
    fail_lookup: bool = False

    def get_month_kline_count(self, symbol: str, interval: str, year: int, month: int) -> int:
        if self.fail_lookup:
            msg = "lookup failed mid-preview"
            raise RuntimeError(msg)
        return 0

    def close(self) -> None:
        self.closes += 1
        super().close()


class TestGetDownloadPreviewClosesArchive:
    """get_download_preview owns and closes the archive it creates."""

    def test_owned_archive_is_closed_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No archive passed in: the one created internally is closed exactly once."""
        created: list[_RecordingArchive] = []

        def _factory() -> _RecordingArchive:
            arc = _RecordingArchive()
            created.append(arc)
            return arc

        monkeypatch.setattr(ingest, "RawDataArchive", _factory)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)

        preview = ingest.get_download_preview(["BTCUSDT"], start, end)

        assert len(preview) == 1
        assert len(created) == 1
        assert created[0].closes == 1

    def test_owned_archive_is_closed_when_loop_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing lookup still closes the internally created archive."""
        created: list[_RecordingArchive] = []

        def _factory() -> _RecordingArchive:
            arc = _RecordingArchive()
            arc.fail_lookup = True
            created.append(arc)
            return arc

        monkeypatch.setattr(ingest, "RawDataArchive", _factory)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)

        with pytest.raises(RuntimeError, match="lookup failed mid-preview"):
            ingest.get_download_preview(["BTCUSDT"], start, end)

        assert len(created) == 1
        assert created[0].closes == 1

    def test_caller_archive_is_left_open(self, tmp_path: Path) -> None:
        """A caller-supplied archive belongs to the caller; it is not closed here."""
        archive = _RecordingArchive(tmp_path / "archive.db")
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)

        ingest.get_download_preview(["BTCUSDT"], start, end, archive=archive)

        assert archive.closes == 0
        archive.close()
