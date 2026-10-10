"""Data-layer hardening (vibe-quant-yul7u.26 + vibe-quant-yul7u.27).

- ``ingest_all`` / ``update_all`` record a ``partial`` download session (with the
  failed months/ranges) instead of silently reporting ``completed``.
- ``write_instrument`` only treats genuinely unreadable/corrupt parquet as
  "replace me"; any other error propagates and nothing is deleted.
- ``RestDownloadError`` survives a pickle round-trip.
- Mutant-killing tests for the retry tuple, the CLI exit codes and the REST
  failed-range recording.
"""

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data import downloader, ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import CatalogManager, create_instrument

if TYPE_CHECKING:
    from pathlib import Path


class _NoopCatalog:
    """Catalog stand-in for session tests: ingest_symbol is monkeypatched."""

    def write_instrument(self, *a: object, **k: object) -> None: ...
    def clear_bar_data(self, *a: object, **k: object) -> None: ...
    def write_bars(self, *a: object, **k: object) -> None: ...


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "raw.db"


@pytest.fixture
def sessions_env(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point ingest's module-level archive/catalog at a tmp db."""
    monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: RawDataArchive(db_path))
    monkeypatch.setattr(ingest, "CatalogManager", lambda *a, **k: _NoopCatalog())
    return db_path


def _last_session(db_path: Path) -> Any:
    archive = RawDataArchive(db_path)
    try:
        return archive.get_download_sessions(1)[0]
    finally:
        archive.close()


class TestPartialSessionStatus:
    def test_ingest_all_marks_partial_and_lists_failures(
        self, sessions_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_symbol(symbol: str, **kw: Any) -> dict[str, Any]:
            return {
                "klines_fetched": 0,
                "klines_inserted": 0,
                "failed_months": ["2024-01"],
                "failed_ranges": ["2024-03-05T00:01:00+00:00..2024-03-20T00:00:00+00:00"],
                "unavailable_months": [],
            }

        monkeypatch.setattr(ingest, "ingest_symbol", fake_symbol)
        monkeypatch.setattr(ingest, "ingest_funding_rates", lambda symbol, **kw: 0)

        ingest.ingest_all(symbols=["BTCUSDT"], verbose=False)

        row = _last_session(sessions_env)
        assert row["status"] == "partial"
        payload = json.loads(row["error_message"])
        assert payload["BTCUSDT"]["failed_months"] == ["2024-01"]
        assert payload["BTCUSDT"]["failed_ranges"] == [
            "2024-03-05T00:01:00+00:00..2024-03-20T00:00:00+00:00"
        ]

    def test_ingest_all_completed_when_clean(
        self, sessions_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ingest,
            "ingest_symbol",
            lambda symbol, **kw: {
                "klines_fetched": 3,
                "klines_inserted": 3,
                "failed_months": [],
                "failed_ranges": [],
                "unavailable_months": [],
            },
        )
        monkeypatch.setattr(ingest, "ingest_funding_rates", lambda symbol, **kw: 0)

        ingest.ingest_all(symbols=["BTCUSDT"], verbose=False)

        row = _last_session(sessions_env)
        assert row["status"] == "completed"
        assert row["error_message"] is None

    def test_update_all_marks_partial_and_lists_failures(
        self, sessions_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_update(
            symbol: str, archive: Any = None, catalog: Any = None, verbose: bool = True
        ) -> dict[str, Any]:
            return {
                "new_klines": 0,
                "new_funding_rates": 0,
                "failed_ranges": ["2024-03-05T00:01:00+00:00..2024-03-20T00:00:00+00:00"],
            }

        monkeypatch.setattr(ingest, "update_symbol", fake_update)

        ingest.update_all(symbols=["BTCUSDT"], verbose=False)

        row = _last_session(sessions_env)
        assert row["status"] == "partial"
        payload = json.loads(row["error_message"])
        assert payload["BTCUSDT"]["failed_ranges"] == [
            "2024-03-05T00:01:00+00:00..2024-03-20T00:00:00+00:00"
        ]


class TestWriteInstrumentHardening:
    def test_non_corruption_error_propagates_and_dir_survives(
        self, tmp_path: Path
    ) -> None:
        mgr = CatalogManager(tmp_path)
        inst = create_instrument("BTCUSDT")
        inst_dir = tmp_path / "data" / "crypto_perpetual" / str(inst.id)
        inst_dir.mkdir(parents=True)
        keep = inst_dir / "keepme.txt"
        keep.write_text("x")

        class Boom:
            def instruments(self) -> list[Any]:
                raise RuntimeError("catalog exploded")

        mgr._catalog = Boom()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="catalog exploded"):
            mgr.write_instrument(inst)

        assert inst_dir.exists()
        assert keep.exists()

    def test_corrupt_parquet_is_replaced_once(self, tmp_path: Path) -> None:
        mgr = CatalogManager(tmp_path)
        inst = create_instrument("BTCUSDT")
        mgr.write_instrument(inst)
        pdir = tmp_path / "data" / "crypto_perpetual" / str(inst.id)

        # Replace with a corrupt, non-epoch parquet name: epoch stubs get removed
        # by cleanup_epoch_parquet before instruments() ever reads them, so this
        # is the path that actually reaches the unreadable-parquet handler.
        for f in pdir.glob("*.parquet"):
            f.unlink()
        bad = (
            pdir
            / "2024-01-01T00-00-00-000000000Z_2024-01-01T01-00-00-000000000Z.parquet"
        )
        bad.write_bytes(b"not a parquet" * 300)

        mgr.write_instrument(inst)  # corrupt -> rmtree + rewrite

        got = CatalogManager(tmp_path).get_instruments()
        assert [i.id for i in got] == [inst.id]

        files = list(pdir.glob("*.parquet"))
        assert len(files) == 1
        mtime = files[0].stat().st_mtime_ns

        mgr.write_instrument(inst)  # now readable and identical -> no churn
        assert files[0].stat().st_mtime_ns == mtime


class TestRestDownloadErrorPickle:
    def test_pickle_round_trip_keeps_fields(self) -> None:
        err = downloader.RestDownloadError("page failed", [(1_000, 1.0, 2.0)], 5_000)
        restored = pickle.loads(pickle.dumps(err))

        assert isinstance(restored, downloader.RestDownloadError)
        assert str(restored) == "page failed"
        assert restored.partial == [(1_000, 1.0, 2.0)]
        assert restored.failed_from == 5_000


class TestD2Mutants:
    def test_eof_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.sleep", lambda *_: None)
        calls: list[int] = []

        def fn() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise EOFError("truncated zip stream")
            return "ok"

        assert downloader._call_with_retry(fn, "eof") == "ok"
        assert len(calls) == 2

    def test_cli_update_exits_1_on_failed_range(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            ingest,
            "update_all",
            lambda **k: {"BTCUSDT": {"failed_ranges": ["x..y"]}},
        )
        assert ingest.main(["update", "--symbols", "BTCUSDT"]) == 1
        assert "x..y" in capsys.readouterr().err

    def test_cli_detail_exits_1_on_failed_range(
        self,
        db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            ingest, "RawDataArchive", lambda *a, **k: RawDataArchive(db_path)
        )
        monkeypatch.setattr(ingest, "CatalogManager", lambda *a, **k: _NoopCatalog())
        monkeypatch.setattr(
            ingest, "ingest_detail_data", lambda **k: {"failed_ranges": ["x..y"]}
        )
        code = ingest.main(["detail", "--symbols", "BTCUSDT", "--start", "2024-01-01"])
        assert code == 1
        assert "x..y" in capsys.readouterr().err

    def test_fetch_rest_klines_records_range_and_returns_partial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        partial: list[tuple[Any, ...]] = [(1_000, 1.0, 2.0)]

        def boom(*a: Any, **k: Any) -> list[Any]:
            raise downloader.RestDownloadError("500", partial, 2_000)

        monkeypatch.setattr(ingest, "download_recent_klines", boom)
        failed: list[str] = []
        out = ingest._fetch_rest_klines("BTCUSDT", "1m", 0, 10_000, failed)
        assert out == partial
        assert len(failed) == 1
