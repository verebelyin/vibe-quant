"""Positioning indicators (OI_CHANGE, LS_RATIO_Z, DVOL_Z) in the GA discovery pool.

Like the other context indicators they join a run only when the indicator pool names them;
a pool that names one preflights its archive data before the GA starts, so a run without
the downloaded metrics fails at launch with the download hint instead of breeding the
positioning chromosomes out quietly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import vibe_quant.dsl.plugin_loader  # noqa: F401  (registers plugin indicators)
from vibe_quant.data.archive import MetricsFileMeta, MetricsRow, RawDataArchive
from vibe_quant.discovery.operators import INDICATOR_POOL
from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from vibe_quant.discovery.pipeline import DiscoveryPipeline

POSITIONING = {"OI_CHANGE", "LS_RATIO_Z", "DVOL_Z"}


@pytest.fixture
def clean() -> Iterator[None]:
    yield
    aux_data.configure(None, None)


def _pipeline(indicator_pool: list[str] | None, symbols: list[str]) -> DiscoveryPipeline:
    from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    cfg = _make_config(
        indicator_pool=indicator_pool,
        symbols=symbols,
        start_date="2024-01-10",
        end_date="2024-01-20",
    )
    return DiscoveryPipeline(cfg, _mock_backtest)


def _metrics_archive(path: Path, symbol: str, first: date, last: date) -> Path:
    """``ok`` metrics files with 5-minute ``sum_open_interest`` rows for every day first..last."""
    archive = RawDataArchive(path)
    try:
        day = first
        while day <= last:
            t0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
            rows: list[MetricsRow] = [
                (int((t0 + k * timedelta(minutes=5)).timestamp()) * 1000, 1000.0 + k) + (None,) * 5
                for k in range(288)
            ]
            meta = MetricsFileMeta(
                filename=f"{symbol}-metrics-{day}.zip",
                md5=f"md5-{symbol}-{day}",
                sha256="0" * 64,
                size=1,
                vision_last_modified=None,
                csv_rows=len(rows),
                status="ok",
                detail=None,
            )
            archive.insert_metrics_file(symbol, day, meta, rows)
            day += timedelta(days=1)
    finally:
        archive.close()
    return path


def test_default_pool_excludes_positioning() -> None:
    _pipeline(None, ["BTCUSDT"])._apply_indicator_pool_filter()
    assert not POSITIONING & set(INDICATOR_POOL)
    assert "RSI" in INDICATOR_POOL


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ZECUSDT"])
def test_explicit_pool_preflights_metrics(tmp_path: Path, clean: None, symbol: str) -> None:
    """The named positioning type stays in the pool and its metrics are checked per symbol:
    an archive without them refuses with the download hint, one covering the window passes."""
    empty = tmp_path / "empty.db"
    RawDataArchive(empty).close()
    aux_data.configure(empty, None)
    pipeline = _pipeline(["RSI", "OI_CHANGE"], [symbol])
    pipeline._apply_indicator_pool_filter()
    assert set(INDICATOR_POOL) == {"RSI", "OI_CHANGE"}

    with pytest.raises(AuxDataUnavailableError) as refused:
        pipeline._preflight_aux_data()
    assert str(refused.value) == (
        f"No Binance metrics archived for {symbol} in {empty} — "
        f"python -m vibe_quant.data download-metrics --symbols {symbol}"
    )

    covered = _metrics_archive(tmp_path / "covered.db", symbol, date(2024, 1, 9), date(2024, 1, 20))
    aux_data.configure(covered, None)
    pipeline._preflight_aux_data()
