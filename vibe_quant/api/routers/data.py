"""Data management router."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from vibe_quant.api.deps import get_catalog_manager, get_job_manager
from vibe_quant.api.schemas.data import (
    BrowseDataResponse,
    DataCoverageItem,
    DataCoverageResponse,
    DataQualityResponse,
    DataStatusResponse,
    IngestPreviewResponse,
    IngestRequest,
)
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.jobs.manager import BacktestJobManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])

CatMgr = Annotated[CatalogManager, Depends(get_catalog_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]


def _get_archive():  # noqa: ANN202
    from vibe_quant.data.archive import RawDataArchive

    return RawDataArchive()


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            with contextlib.suppress(OSError):
                total += os.path.getsize(fp)
    return total


# --- Storage status ---


@router.get("/status", response_model=DataStatusResponse)
async def data_status() -> DataStatusResponse:
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

    archive_size = 0
    if DEFAULT_ARCHIVE_PATH.exists():
        archive_size = DEFAULT_ARCHIVE_PATH.stat().st_size

    catalog_size = _dir_size(str(DEFAULT_CATALOG_PATH)) if DEFAULT_CATALOG_PATH.exists() else 0

    return DataStatusResponse(
        archive_size_bytes=archive_size,
        catalog_size_bytes=catalog_size,
        total_size_bytes=archive_size + catalog_size,
    )


# --- Coverage ---


@router.get("/coverage", response_model=DataCoverageResponse)
async def data_coverage(catalog: CatMgr) -> DataCoverageResponse:
    archive = _get_archive()
    try:
        symbols = archive.get_symbols()
        items: list[DataCoverageItem] = []

        for symbol in symbols:
            kline_count = archive.get_kline_count(symbol, "1m")
            date_range = archive.get_date_range(symbol, "1m")

            start_date = ""
            end_date = ""
            if date_range:
                start_date = datetime.fromtimestamp(date_range[0] / 1000, tz=UTC).strftime("%Y-%m-%d")
                end_date = datetime.fromtimestamp(date_range[1] / 1000, tz=UTC).strftime("%Y-%m-%d")

            bar_count = catalog.get_bar_count(symbol, "1m")

            # Count funding rates via direct query
            row = archive.conn.execute(
                "SELECT COUNT(*) FROM raw_funding_rates WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            funding_rate_count = row[0] if row else 0

            items.append(DataCoverageItem(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                kline_count=kline_count,
                bar_count=bar_count,
                funding_rate_count=funding_rate_count,
            ))

        return DataCoverageResponse(coverage=items)
    finally:
        archive.close()


# --- Symbols ---


@router.get("/symbols")
async def list_symbols() -> list[str]:
    archive = _get_archive()
    try:
        return archive.get_symbols()
    finally:
        archive.close()


# --- Ingest preview ---


@router.post("/ingest/preview", response_model=IngestPreviewResponse)
async def ingest_preview(body: IngestRequest) -> IngestPreviewResponse:
    from vibe_quant.data.ingest import get_download_preview

    try:
        start_dt = datetime.strptime(body.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(body.end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {exc}") from exc

    preview = get_download_preview(body.symbols, start_dt, end_dt)
    total = len(preview)
    archived = sum(1 for p in preview if p["Status"] == "Archived")
    new = total - archived

    return IngestPreviewResponse(
        total_months=total,
        archived_months=archived,
        new_months=new,
    )


# --- Ingest (background job) ---


@router.post("/ingest", status_code=202)
async def start_ingest(body: IngestRequest, jobs: JobMgr) -> dict[str, object]:
    log_file = f"logs/ingest_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant.data",
        "ingest",
        "--symbols",
        ",".join(body.symbols),
        "--start",
        body.start_date,
        "--end",
        body.end_date,
    ]

    # Use run_id=0 as sentinel for non-backtest jobs
    try:
        pid = jobs.start_job(0, "data_ingest", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info("data ingest started pid=%d symbols=%s", pid, body.symbols)
    return {
        "status": "started",
        "pid": pid,
        "symbols": body.symbols,
        "start_date": body.start_date,
        "end_date": body.end_date,
    }


# --- Update (background job) ---


@router.post("/update", status_code=202)
async def start_update(jobs: JobMgr) -> dict[str, object]:
    log_file = f"logs/update_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant.data",
        "update",
    ]

    try:
        pid = jobs.start_job(0, "data_update", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info("data update started pid=%d", pid)
    return {"status": "started", "pid": pid}


# --- Rebuild catalog (stub) ---


@router.post("/rebuild", status_code=202)
async def rebuild_catalog(jobs: JobMgr) -> dict[str, object]:
    log_file = f"logs/rebuild_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant.data",
        "rebuild",
        "--from-archive",
    ]

    try:
        pid = jobs.start_job(0, "catalog_rebuild", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info("catalog rebuild started pid=%d", pid)
    return {"status": "started", "pid": pid}


# --- Browse OHLCV data ---


@router.get("/browse/{symbol}", response_model=BrowseDataResponse)
async def browse_data(
    symbol: str,
    interval: str = Query(default="1m"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
) -> BrowseDataResponse:
    archive = _get_archive()
    try:
        start_ts: int | None = None
        end_ts: int | None = None

        if start:
            try:
                start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid start date: {exc}") from exc
        if end:
            try:
                end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid end date: {exc}") from exc

        rows = archive.get_klines(symbol, interval, start_ts, end_ts)
        data: list[dict[str, object]] = []
        for row in rows:
            data.append({
                "open_time": row["open_time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "close_time": row["close_time"],
            })

        return BrowseDataResponse(symbol=symbol, interval=interval, data=data)
    finally:
        archive.close()


# --- Data quality (stub) ---


@router.get("/quality/{symbol}", response_model=DataQualityResponse)
async def data_quality(symbol: str) -> DataQualityResponse:
    return DataQualityResponse(
        symbol=symbol,
        gaps=[],
        quality_score=1.0,
    )


# --- Download history ---


@router.get("/history")
async def download_history(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, object]]:
    archive = _get_archive()
    try:
        sessions = archive.get_download_sessions(limit=limit)
        result: list[dict[str, object]] = []
        for s in sessions:
            result.append({
                "id": s["id"],
                "started_at": s["started_at"],
                "completed_at": s["completed_at"],
                "symbols": s["symbols"],
                "source": s["source"],
                "klines_fetched": s["klines_fetched"],
                "klines_inserted": s["klines_inserted"],
                "status": s["status"],
                "error_message": s["error_message"],
            })
        return result
    finally:
        archive.close()
