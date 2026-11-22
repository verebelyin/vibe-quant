"""Full tier (1m LAST via ``ingest``, 1m MARK via ``download-mark``) preflight.

Before any archive or catalog is opened for writing: a coin without an
instrument spec is refused, the size estimate and free disk are printed, free
disk below estimate + 5 GB refuses, and only y/yes (or ``--yes``) proceeds.
The archive is a tmp file; the download functions are stubs that record calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data import archive as archive_mod
from vibe_quant.data import ingest
from vibe_quant.data.archive import RawDataArchive

if TYPE_CHECKING:
    from pathlib import Path

GB = 1024**3
JAN_2024 = ["--start", "2024-01-01", "--end", "2024-01-31"]
JAN_ROWS = 31 * 1440


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[Any]]:
    """Stub the writers; route the archive to tmp; plenty of free disk."""
    seen: dict[str, list[Any]] = {"ingest_all": [], "mark": [], "archive": []}

    def fake_ingest_all(**kwargs: Any) -> dict[str, dict[str, Any]]:
        seen["ingest_all"].append(kwargs["symbols"])
        return {}

    def fake_mark(symbol: str, **_kwargs: Any) -> dict[str, Any]:
        seen["mark"].append(symbol)
        return {
            "months_downloaded": 0,
            "months_skipped": 0,
            "unavailable_months": [],
            "failed_months": [],
            "unpublished_days": [],
            "failed_days": [],
            "mark_klines_inserted": 0,
        }

    def fake_archive(*args: Any, **kwargs: Any) -> RawDataArchive:
        seen["archive"].append(args)
        return RawDataArchive(tmp_path / "raw_data.db")

    monkeypatch.setattr(ingest, "ingest_all", fake_ingest_all)
    monkeypatch.setattr(ingest, "ingest_mark_klines", fake_mark)
    monkeypatch.setattr(ingest, "RawDataArchive", fake_archive)
    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", tmp_path / "raw_data.db")
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _p: (100 * GB, 0, 50 * GB))
    return seen


@pytest.mark.parametrize("command", ["ingest", "download-mark"])
def test_coin_without_instrument_spec_is_refused_before_opening_anything(
    calls: dict[str, list[Any]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    code = ingest.main([command, "--symbols", "BTCUSDT,XRPUSDT", *JAN_2024, "--yes"])
    assert code == 1
    err = capsys.readouterr().err
    assert "XRPUSDT" in err and "INSTRUMENT_CONFIGS" in err and "download-research" in err
    assert calls == {"ingest_all": [], "mark": [], "archive": []}
    assert not (tmp_path / "raw_data.db").exists()


@pytest.mark.parametrize("command", ["ingest", "download-mark"])
@pytest.mark.parametrize("answer", ["n", "", "yes please", EOFError])
def test_answer_other_than_yes_aborts_after_printing_the_estimate(
    calls: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    answer: str | type[EOFError],
) -> None:
    def reply(_prompt: str = "") -> str:
        if isinstance(answer, str):
            return answer
        raise answer  # no stdin (a job, a pipe): abort, never a traceback

    monkeypatch.setattr("builtins.input", reply)
    assert ingest.main([command, "--symbols", "BTCUSDT", *JAN_2024]) == 0
    out = capsys.readouterr().out
    assert "aborted" in out
    assert f"{JAN_ROWS:,}" in out and "free disk" in out.lower()
    assert calls == {"ingest_all": [], "mark": [], "archive": []}


@pytest.mark.parametrize("command", ["ingest", "download-mark"])
@pytest.mark.parametrize("answer", ["y", " YES "])
def test_yes_answer_or_flag_proceeds(
    calls: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    answer: str,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    assert ingest.main([command, "--symbols", "BTCUSDT", *JAN_2024]) == 0
    assert ingest.main([command, "--symbols", "BTCUSDT", *JAN_2024, "--yes"]) == 0
    done = calls["ingest_all"] if command == "ingest" else calls["mark"]
    assert done == ([["BTCUSDT"], ["BTCUSDT"]] if command == "ingest" else ["BTCUSDT", "BTCUSDT"])


def _estimate(command: str) -> int:
    rows_bytes = JAN_ROWS * ingest._EST_ROW_BYTES["kline"]
    if command == "ingest":  # catalog Parquet bars on top of the archive rows
        return rows_bytes + int(rows_bytes * ingest._CATALOG_PER_ARCHIVE_BYTE)
    return rows_bytes


@pytest.mark.parametrize("command", ["ingest", "download-mark"])
def test_free_disk_below_estimate_plus_headroom_refuses_equality_proceeds(
    calls: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    need = _estimate(command) + 5 * GB
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _p: (100 * GB, 0, need - 1))
    assert ingest.main([command, "--symbols", "BTCUSDT", *JAN_2024, "--yes"]) == 1
    assert "refus" in capsys.readouterr().err.lower()
    assert calls == {"ingest_all": [], "mark": [], "archive": []}

    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _p: (100 * GB, 0, need))
    assert ingest.main([command, "--symbols", "BTCUSDT", *JAN_2024, "--yes"]) == 0


def test_estimate_counts_only_rows_not_archived_yet(
    calls: dict[str, list[Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arc = RawDataArchive(tmp_path / "raw_data.db")
    t0 = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
    rows = [(t0 + i * 60_000, 1, 1, 1, 1, 1, t0 + i * 60_000 + 59_999) for i in range(1000)]
    arc.insert_klines("BTCUSDT", "1m", rows, "seed")
    arc.insert_mark_klines("ETHUSDT", "1m", rows, "seed")
    arc.close()
    assert ingest.main(["ingest", "--symbols", "BTCUSDT,ETHUSDT", *JAN_2024, "--yes"]) == 0
    out = capsys.readouterr().out
    btc = next(ln for ln in out.splitlines() if ln.strip().startswith("BTCUSDT"))
    eth = next(ln for ln in out.splitlines() if ln.strip().startswith("ETHUSDT"))
    assert f"{JAN_ROWS - 1000:,}" in btc
    assert f"{JAN_ROWS:,}" in eth  # MARK rows are another table
    assert ingest.main(["download-mark", "--symbols", "ETHUSDT", *JAN_2024, "--yes"]) == 0
    eth_mark = next(
        ln for ln in capsys.readouterr().out.splitlines() if ln.strip().startswith("ETHUSDT")
    )
    assert f"{JAN_ROWS - 1000:,}" in eth_mark
